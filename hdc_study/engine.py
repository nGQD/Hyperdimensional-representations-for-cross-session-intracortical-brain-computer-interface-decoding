"""GPU training and causally masked replay for the H1 study."""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import random
import time
import numpy as np
from scipy.signal import lfilter
from sklearn.metrics import r2_score
import torch
from torch.nn import functional as F
from data import partition_day
from model import Config, Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEMA = 1
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


@contextmanager
def precision(inference=False):
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = not inference
    torch.backends.cudnn.allow_tf32 = not inference
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def moments(records):
    x = np.concatenate([r["x"] for r in records])
    y = np.concatenate([r["y"][r["mask"]] for r in records])
    return dict(mean=x.mean(0), std=np.maximum(x.std(0), .01),
                ymean=y.mean(0), ystd=np.maximum(y.std(0), 1e-6))


def normalize(x, stats, online=True, alpha=.9):
    x = x.astype(np.float64)
    mean, std = stats["mean"], stats["std"]
    if online:
        n = np.arange(1, len(x) + 1)[:, None]
        running_mean = np.cumsum(x, axis=0) / n
        variance = np.maximum(np.cumsum(x * x, axis=0) / n - running_mean ** 2, 0)
        running_std = np.maximum(np.sqrt(variance), .01)
        blend = np.where(n >= 30, np.minimum(.7, n / 2000), 0)
        mean = (1 - blend) * mean + blend * running_mean
        std = (1 - blend) * std + blend * running_std
    z = (x - mean) / std
    if alpha:
        z = lfilter([1 - alpha], [1, -alpha], z, axis=0, zi=(alpha * z[0])[None])[0]
    return z.astype(np.float32)


class Windows:
    def __init__(self, records, stats, config):
        xs, ys, masks, starts = [], [], [], []
        offset = 0
        for r in records:
            x = normalize(r["x"], stats, config.online_norm, config.smoothing)
            ends = list(range(0, len(x) - config.window + 1, config.stride))
            if len(x) >= config.window and (not ends or ends[-1] != len(x) - config.window):
                ends.append(len(x) - config.window)
            starts.extend(offset + i for i in ends if r["mask"][i:i + config.window].sum() >= 10)
            xs.append(x)
            ys.append((r["y"] - stats["ymean"]) / stats["ystd"])
            masks.append(r["mask"])
            offset += len(x)
        if not starts:
            raise ValueError("No eligible training windows")
        self.x = torch.as_tensor(np.concatenate(xs), device=DEVICE)
        self.y = torch.as_tensor(np.concatenate(ys), dtype=torch.float32, device=DEVICE)
        self.mask = torch.as_tensor(np.concatenate(masks), device=DEVICE)
        self.starts = torch.as_tensor(starts, device=DEVICE)
        self.offsets = torch.arange(config.window, device=DEVICE)

    def batches(self, batch):
        order = torch.randperm(len(self.starts), device=DEVICE)
        for part in order.split(batch):
            indices = self.starts[part, None] + self.offsets[None]
            yield self.x[indices], self.y[indices], self.mask[indices]


def fit(records, config, seed, path=None, epochs=None, progress=None):
    seed_all(seed)
    started = time.perf_counter()
    stats = moments(records)
    data = Windows(records, stats, config)
    model = Decoder(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
        weight_decay=config.weight_decay, fused=DEVICE == "cuda")
    n_epochs = config.epochs if epochs is None else epochs
    losses = []
    with precision():
        for epoch in range(n_epochs):
            model.train()
            schedule = min((epoch + 1) / 3, 1) * (.01 + .99 * .5 *
                (1 + np.cos(np.pi * max(0, epoch - 3) / max(1, n_epochs - 3))))
            for group in optimizer.param_groups:
                group["lr"] = config.lr * schedule
            batch_losses = []
            for x, y, mask in data.batches(config.batch):
                x = x + torch.randn_like(x) * .15
                keep = (torch.rand(x.shape[0], 1, x.shape[-1], device=DEVICE) > .1).float() / .9
                x = x * keep
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=config.amp):
                    prediction, _, _ = model(x)
                    error = F.huber_loss(prediction, y, reduction="none").mean(-1)
                    loss = (error * mask).sum() / mask.sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                batch_losses.append(loss.detach())
            losses.append(float(torch.stack(batch_losses).mean()))
            if progress:
                progress(epoch + 1, losses[-1])
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    record = dict(schema=SCHEMA, config=asdict(config), seed=seed,
        train_days=sorted({r["day"] for r in records}),
        train_files=[dict(path=r["path"], sha256=r["sha256"]) for r in records],
        stats=stats, state=model.state_dict(), epochs=n_epochs,
        training_losses=losses, elapsed_seconds=time.perf_counter() - started,
        parameters=model.parameter_count(), torch_version=torch.__version__)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(record, temporary)
        temporary.replace(path)
    return model, record


def load(path):
    record = torch.load(path, map_location=DEVICE, weights_only=False)
    if record["schema"] != SCHEMA:
        raise ValueError("Incompatible checkpoint")
    model = Decoder(Config(**record["config"])).to(DEVICE)
    model.load_state_dict(record["state"], strict=True)
    return model, record


@torch.no_grad()
def predict(model, x, chunk=1024):
    model.eval()
    hidden = history = None
    outputs = []
    with precision(inference=True):
        for part in torch.as_tensor(x, device=DEVICE).split(chunk):
            prediction, hidden, history = model(part[None], hidden, history)
            outputs.append(prediction[0].float().cpu().numpy())
    return np.concatenate(outputs)


def calibrate(model, records, stats, steps=20):
    model = copy.deepcopy(model).eval()
    xs = np.concatenate([r["x"][r["calibration"]] for r in records if r["calibration"].any()])
    prior = dict(mean=xs.mean(0), std=np.maximum(xs.std(0), .01))
    features, targets = [], []
    for record in records:
        n = int(record["calibration"].sum())
        if not n:
            continue
        captured = []
        hook = model.readout.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        try:
            predict(model, normalize(record["x"][:n], prior, model.config.online_norm, model.config.smoothing))
        finally:
            hook.remove()
        mask = torch.as_tensor(record["mask"][:n], device=DEVICE)
        features.append(torch.cat(captured, 1)[0][mask])
        target = (record["y"][:n] - stats["ymean"]) / stats["ystd"]
        targets.append(torch.as_tensor(target, device=DEVICE)[mask])
    x, y = torch.cat(features), torch.cat(targets)
    if len(x) == 0:
        raise ValueError("No labeled calibration observations")
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=1e-4, weight_decay=1e-4)
    with precision(inference=True):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.huber_loss(model.readout(x), y)
            loss.backward()
            optimizer.step()
    return model, prior


def evaluate(model, record, records, regime="zero_shot", prefix=None, save=None,
             perturbation=None, level=0, perturb_seed=0):
    if set(record["train_days"]) & {r["day"] for r in records}:
        raise ValueError("Evaluated session was used for training")
    if regime not in ("zero_shot", "few_shot"):
        raise ValueError(regime)
    prefix = (3000 if regime == "few_shot" else 0) if prefix is None else prefix
    if regime == "zero_shot" and prefix:
        raise ValueError("Zero-label evaluation cannot use labeled calibration")
    result, predictions, targets, masks, lengths = [], [], [], [], []
    for day in sorted({r["day"] for r in records}, key=lambda d: int(d[1:])):
        parts = partition_day([r for r in records if r["day"] == day], prefix)
        used, prior = calibrate(model, parts, record["stats"]) if prefix else (model, record["stats"])
        rng = np.random.default_rng(perturb_seed + 1009 * int(day[1:]))
        factors = np.ones(model.config.channels)
        if perturbation == "gain":
            factors = np.exp(level * rng.normal(size=len(factors)))
        elif perturbation == "dropout":
            factors[rng.permutation(len(factors))[:round(level * len(factors))]] = 0
        elif perturbation is not None:
            raise ValueError(perturbation)
        pp, yy, mm, offset = [], [], [], 0
        for part in parts:
            raw = part["x"].copy()
            start = max(0, 3256 - offset)
            raw[start:] *= factors
            x = normalize(raw, prior, model.config.online_norm, model.config.smoothing)
            p = predict(used, x) * record["stats"]["ystd"] + record["stats"]["ymean"]
            pp.append(p)
            yy.append(part["y"])
            mm.append(part["score_mask"])
            offset += len(raw)
        p, y, mask = np.concatenate(pp), np.concatenate(yy), np.concatenate(mm)
        score = float(r2_score(y[mask], p[mask], multioutput="variance_weighted"))
        result.append(dict(day=day, r2=score, bins=int(mask.sum()), regime=regime,
                           calibration_bins=prefix, score_start=3256,
                           perturbation=perturbation, perturbation_level=level))
        predictions.append(p)
        targets.append(y)
        masks.append(mask)
        lengths.append(len(p))
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, prediction=np.concatenate(predictions), target=np.concatenate(targets),
                            mask=np.concatenate(masks), lengths=lengths, days=[r["day"] for r in result])
    return result

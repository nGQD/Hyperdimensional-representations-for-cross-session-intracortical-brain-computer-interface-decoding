"""Correctness checks required before the experiment protocol can be frozen."""
from dataclasses import asdict, replace
import json
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from sklearn.metrics import r2_score
import torch
from data import ROOT, REPO, load_h1, partition_day, sessions, sha, write_json
from model import Config, Decoder
from engine import DEVICE, normalize, predict, calibrate, moments, evaluate, precision
from adapter import H1Decoder
from run import fingerprint, candidates


def close(a, b, tolerance=3e-5):
    np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def check_sources():
    records = load_h1(verify=True) + load_h1("held-out-calib", verify=True)
    from falcon_challenge.config import FalconTask, H1_NEW_TO_OLD
    from falcon_challenge.dataloaders import load_nwb
    from falcon_challenge import dataloaders
    for row in records:
        source = REPO / row["path"]
        if not source.exists():
            raise FileNotFoundError(source)
        assert sha(source) == row["sha256"]
        assert sha(dataloaders.__file__) == row["loader_sha256"]
        x, y, trial, mask = load_nwb(source, FalconTask.h1)
        close(x.astype(np.float32), row["x"], 0)
        close(y.astype(np.float32), row["y"], 0)
        np.testing.assert_array_equal(mask, row["mask"])
        np.testing.assert_array_equal(trial, row["trial"])
        assert H1_NEW_TO_OLD[source.stem.split("ses-")[-1]] == row["set"]
    for day in sessions(records):
        day_rows = [r for r in records if r["day"] == day]
        reference = None
        for prefix in (0, 750, 1500, 3000):
            parts = partition_day(day_rows, prefix)
            mask = np.concatenate([r["score_mask"] for r in parts])
            cal = np.concatenate([r["calibration"] for r in parts])
            assert not np.any(mask & cal) and cal.sum() == prefix
            assert not mask[:3256].any()
            if reference is not None:
                np.testing.assert_array_equal(mask, reference)
            reference = mask
    return dict(files=len(records), groups=len(sessions(records)),
                target_processing="Exact official-loader arrays, no target smoothing or delay")


def check_causality_and_calibration():
    rng = np.random.default_rng(20261015)
    config = Config(channels=6, outputs=3, hidden=12, dimension=16,
                    temporal_bins=8, dropout=0, batch=4)
    x = rng.normal(size=(3400, 6)).astype(np.float32)
    y = rng.normal(size=(3400, 3)).astype(np.float32)
    records = [dict(day="S100", x=x, y=y, mask=np.ones(len(x), bool))]
    stats = moments(records)
    changed_x = x.copy()
    changed_x[95:] += 100
    close(normalize(x, stats)[:95], normalize(changed_x, stats)[:95], 0)
    largest_error = 0.0
    for family in ("hdc", "random", "mlp", "plain"):
        c = replace(config, family=family)
        torch.manual_seed(42)
        model = Decoder(c).to(DEVICE).eval()
        normalized = normalize(x[:160], stats)
        reference = predict(model, normalized, chunk=160)
        for chunk in (1, 13, 39):
            alternative = predict(model, normalized, chunk=chunk)
            close(reference, alternative)
            largest_error = max(largest_error, float(abs(reference-alternative).max()))
        record = dict(schema=1, config=asdict(c), state=model.state_dict(), stats=stats, train_days=["S99"])
        adapter = H1Decoder(SimpleNamespace(n_channels=6, out_dim=3), record, device=DEVICE)
        actual = np.concatenate([adapter.predict(row[None]) for row in x[:160]])
        close(actual, reference * stats["ystd"] + stats["ymean"])
        paired = np.stack([normalized, normalized + .25])
        with precision(inference=True), torch.no_grad():
            both = model(torch.as_tensor(paired, device=DEVICE))[0].cpu().numpy()
        close(both[0], reference)
    model = Decoder(config).to(DEVICE).eval()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    parts = partition_day(records, prefix=750)
    mutated = [dict(parts[0], y=parts[0]["y"].copy(), x=parts[0]["x"].copy())]
    mutated[0]["y"][750:] += 1000
    mutated[0]["x"][750:] -= 1000
    calibrated, prior = calibrate(model, parts, stats)
    alternative, alternative_prior = calibrate(model, mutated, stats)
    for key in prior:
        close(prior[key], alternative_prior[key], 0)
    for key, value in calibrated.state_dict().items():
        close(value.cpu().numpy(), alternative.state_dict()[key].cpu().numpy(), 0)
        if not key.startswith("readout."):
            close(value.cpu().numpy(), before[key].cpu().numpy(), 0)
    assert any(not torch.equal(calibrated.state_dict()[k], before[k]) for k in before if k.startswith("readout."))
    assert all(torch.equal(model.state_dict()[k], before[k]) for k in before)
    try:
        evaluate(model, dict(stats=stats, train_days=["S100"]), records)
        raise AssertionError("Training/evaluation overlap was accepted")
    except ValueError as error:
        assert "training" in str(error)
    return dict(device=DEVICE, largest_streaming_error=largest_error,
                calibrated_parameters="Readout only; labels and neural inputs after prefix are irrelevant to adaptation")


def check_selection_and_metric():
    configs = candidates(256)
    keys = [json.dumps(asdict(c), sort_keys=True) for c in configs]
    assert len(keys) == len(set(keys)) == 16
    counts = {}
    for d in (256, 512):
        for k in (1, 8):
            models = {f:Decoder(Config(family=f, dimension=d, temporal_bins=k)) for f in ("hdc", "random", "mlp")}
            n = {f:sum(p.numel() for p in m.input_projection.parameters()) for f,m in models.items()}
            assert n["hdc"] == n["random"]
            assert abs(n["mlp"] / n["hdc"] - 1) < .01
            counts[f"D{d}_K{k}"] = {f:m.parameter_count() for f,m in models.items()}
    rng = np.random.default_rng(19)
    y = rng.normal(size=(200, 7)) * np.arange(1,8)
    p = y + rng.normal(size=y.shape)
    direct = 1 - np.sum((y-p)**2) / np.sum((y-y.mean(0))**2)
    close(np.array(r2_score(y, p, multioutput="variance_weighted")), np.array(direct), 1e-12)
    return counts


def main():
    started = time.time()
    checks = {}
    for script in ("check_model.py", "check_adapter.py"):
        result = subprocess.run([sys.executable, str(ROOT / script)], check=True, text=True, capture_output=True)
        checks[script] = result.stdout.strip()
    checks["source_and_masks"] = check_sources()
    checks["causality_calibration_streaming"] = check_causality_and_calibration()
    checks["parameter_matching_metric"] = check_selection_and_metric()
    result = dict(passed=True, fingerprint=fingerprint(), checks=checks,
                  elapsed_seconds=time.time()-started, torch_version=torch.__version__)
    write_json(ROOT / "results/validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

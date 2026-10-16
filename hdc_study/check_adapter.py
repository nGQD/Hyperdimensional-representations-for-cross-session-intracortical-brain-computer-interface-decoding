"""Offline CPU checks for the official FALCON H1 adapter, using synthetic data."""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import torch

from adapter import H1Decoder
from model import Config, Decoder
from falcon_challenge.config import FalconConfig, FalconTask


def assert_close(a, b, atol=2e-5, rtol=2e-5):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"max difference {float(np.max(np.abs(a-b)))}")


class PoisonTag:
    def __str__(self):
        raise AssertionError("adapter inspected a dataset tag")


def make_checkpoint(config, seed=17):
    torch.manual_seed(seed)
    model = Decoder(config)
    record = {
        "config": asdict(config),
        "state": model.state_dict(),
        "stats": {
            "mean": np.linspace(-0.2, 0.2, config.channels),
            "std": np.linspace(0.8, 1.2, config.channels),
            "ymean": np.linspace(-0.1, 0.1, config.outputs),
            "ystd": np.linspace(0.7, 1.3, config.outputs),
        },
    }
    return record


def offline_causal_preprocess(x, mean, std, online, alpha):
    """Independent scalar-time implementation of adapter normalization/smoothing."""
    x = np.asarray(x, dtype=np.float64)
    b, t, c = x.shape
    total = np.zeros((b, c), dtype=np.float64)
    squares = np.zeros_like(total)
    smooth = None
    result = []
    for i in range(t):
        total += x[:, i]
        squares += x[:, i] ** 2
        if online:
            n = i + 1
            running_mean = total / n
            running_std = np.maximum(np.sqrt(np.maximum(squares / n - running_mean**2, 0)), .01)
            blend = min(.7, n/2000.0) if n >= 30 else 0.0
            current_mean = (1-blend)*mean[None, :] + blend*running_mean
            current_std = (1-blend)*std[None, :] + blend*running_std
        else:
            current_mean = np.broadcast_to(mean, (b, c))
            current_std = np.broadcast_to(std, (b, c))
        z = (x[:, i] - current_mean) / current_std
        smooth = z if smooth is None else alpha*smooth + (1-alpha)*z
        result.append(smooth.astype(np.float32))
    return np.stack(result, axis=1)


def make_adapter(config, batch_size=1, calibration_state=None):
    checkpoint = make_checkpoint(config)
    task = SimpleNamespace(n_channels=config.channels, out_dim=config.outputs)
    return H1Decoder(task, checkpoint, batch_size=batch_size,
                     calibration_state=calibration_state, device="cpu"), checkpoint


def test_offline_streaming_and_poison_tags():
    torch.set_num_threads(1)
    config = Config(channels=5, outputs=3, hidden=7, dimension=6,
                    temporal_bins=4, levels=8, online_norm=True,
                    smoothing=.8, dropout=0.0)
    adapter, record = make_adapter(config, batch_size=2)
    assert adapter.__class__.__mro__[1].__name__ == "BCIDecoder"
    x = np.random.default_rng(12).normal(size=(2, 47, config.channels))
    adapter.reset(dataset_tags=[PoisonTag(), PoisonTag()])
    actual = np.stack([adapter.predict(x[:, t]) for t in range(x.shape[1])], axis=1)

    normalized = offline_causal_preprocess(
        x, record["stats"]["mean"], record["stats"]["std"],
        config.online_norm, config.smoothing)
    model = Decoder(config).eval()
    model.load_state_dict(record["state"], strict=True)
    with torch.inference_mode():
        raw, _, _ = model(torch.as_tensor(normalized, dtype=torch.float32))
    expected = raw.numpy() * record["stats"]["ystd"] + record["stats"]["ymean"]
    assert_close(actual, expected)
    assert adapter.hidden is not None and adapter.history is not None

    # A reset ignores arbitrary tags and clears both model and normalizer state.
    adapter.reset(dataset_tags=[PoisonTag(), PoisonTag()])
    assert adapter.hidden is None and adapter.history is None and adapter.count.tolist() == [0, 0]
    after_reset = np.stack([adapter.predict(x[:, t]) for t in range(x.shape[1])], axis=1)
    assert_close(actual, after_reset)


def test_batch_isolation_and_set_batch_size():
    config = Config(channels=5, outputs=3, hidden=7, dimension=6,
                    temporal_bins=3, levels=8, online_norm=True,
                    smoothing=.9, dropout=0.0, family="random")
    rng = np.random.default_rng(13)
    x = rng.normal(size=(2, 21, config.channels))
    batched, _ = make_adapter(config, batch_size=2)
    y_batch = np.stack([batched.predict(x[:, t]) for t in range(x.shape[1])], axis=1)
    separate = []
    for i in range(2):
        single, _ = make_adapter(config, batch_size=1)
        separate.append(np.stack([single.predict(x[i:i+1, t]) for t in range(x.shape[1])], axis=1)[0])
    assert_close(y_batch[0], separate[0])
    assert_close(y_batch[1], separate[1])
    batched.set_batch_size(1)
    assert batched.batch_size == 1 and batched.count.tolist() == [0]


def test_explicit_frozen_prefix_state_only():
    config = Config(channels=5, outputs=3, hidden=7, dimension=6,
                    temporal_bins=2, levels=4, online_norm=False,
                    smoothing=0.0, dropout=0.0)
    calibration = {
        "source": "allowed_prefix",
        "prefix_bins": 3000,
        "neural_mean": np.full(config.channels, 0.25),
        "neural_std": np.full(config.channels, 1.5),
    }
    adapter, _ = make_adapter(config, calibration_state=calibration)
    assert np.array_equal(adapter.mean[0], calibration["neural_mean"])
    assert np.array_equal(adapter.std[0], calibration["neural_std"])
    adapter.reset(dataset_tags=[PoisonTag()])
    assert np.array_equal(adapter.mean[0], calibration["neural_mean"])
    before = {k: v.detach().clone() for k, v in adapter.model.state_dict().items()}
    adapter.predict(np.zeros((1, config.channels)))
    after = adapter.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_official_h1_task_config():
    config = Config(channels=176, outputs=7, hidden=4, dimension=2,
                    temporal_bins=1, levels=2, online_norm=False,
                    smoothing=0.0, dropout=0.0)
    checkpoint = make_checkpoint(config)
    adapter = H1Decoder(FalconConfig(task=FalconTask.h1), checkpoint,
                        batch_size=1, device="cpu")
    prediction = adapter.predict(np.zeros((1, 176), dtype=np.float32))
    assert prediction.shape == (1, 7)


def main():
    test_offline_streaming_and_poison_tags()
    test_batch_isolation_and_set_batch_size()
    test_explicit_frozen_prefix_state_only()
    test_official_h1_task_config()
    print("All CPU FALCON adapter checks passed.")


if __name__ == "__main__":
    main()

"""FALCON H1 adapter for the causal HDC study decoder.

The official observation API supplies neural samples and opaque dataset tags,
but no behavioral labels. Tags are deliberately ignored. The adapter supports
only pretrained decoding plus optional, explicitly supplied frozen
calibration state; it performs no online weight updates.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import torch


_REPO = Path(__file__).resolve().parents[1]
_CHALLENGE = _REPO / "falcon-challenge"
if _CHALLENGE.is_dir() and str(_CHALLENGE) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE))

from falcon_challenge.interface import BCIDecoder
from model import Config, Decoder


@contextmanager
def _strict_fp32_inference():
    """Match validation inference even if training enabled CUDA TF32 globally."""
    old = (torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32) = old


class H1Decoder(BCIDecoder):
    """Causal H1 decoder with batch-local recurrent and normalization state.

    ``calibration_state`` is optional, session-agnostic, immutable state made
    outside this adapter from a documented allowed prefix. Its mapping may
    contain ``neural_mean`` and ``neural_std`` vectors, a ``readout_state``
    state dict, or both, and must include ``source='allowed_prefix'`` and a
    positive integer ``prefix_bins``. The same supplied state is used after
    every reset. No labels are received or learned from during prediction.
    """
    def __init__(self, task_config, checkpoint, batch_size=1,
                 calibration_state=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.record = self._load_record(checkpoint)
        try:
            self.config = Config(**self.record["config"])
            self.model = Decoder(self.config).to(self.device)
            self.model.load_state_dict(self.record["state"], strict=True)
        except (KeyError, TypeError) as exc:
            raise ValueError("checkpoint must contain a compatible config and state") from exc
        self.model.eval()
        self._train_stats = self.record.get("stats")
        if not isinstance(self._train_stats, Mapping):
            raise ValueError("checkpoint is missing training normalization/output statistics")
        for field in ("mean", "std", "ymean", "ystd"):
            if field not in self._train_stats:
                raise ValueError(f"checkpoint statistics missing {field!r}")
        self._train_mean = np.asarray(self._train_stats["mean"], dtype=np.float64)
        self._train_std = np.asarray(self._train_stats["std"], dtype=np.float64)
        self._ymean = np.asarray(self._train_stats["ymean"], dtype=np.float64)
        self._ystd = np.asarray(self._train_stats["ystd"], dtype=np.float64)
        if self._train_mean.shape != (self.config.channels,) or self._train_std.shape != (self.config.channels,):
            raise ValueError("checkpoint neural statistics do not match model channels")
        if self._ymean.shape != (self.config.outputs,) or self._ystd.shape != (self.config.outputs,):
            raise ValueError("checkpoint output statistics do not match model outputs")

        self._calibration = self._validate_calibration_state(calibration_state)
        self._prior_mean = (self._train_mean.copy() if self._calibration is None or
                            self._calibration.get("neural_mean") is None else
                            self._calibration["neural_mean"].copy())
        self._prior_std = (self._train_std.copy() if self._calibration is None or
                           self._calibration.get("neural_std") is None else
                           self._calibration["neural_std"].copy())

        if task_config.n_channels != self.config.channels or task_config.out_dim != self.config.outputs:
            raise ValueError("FALCON task dimensions do not match the checkpoint")
        super().__init__(task_config, batch_size)
        self.reset()

    @staticmethod
    def _load_record(checkpoint):
        if isinstance(checkpoint, Mapping):
            return checkpoint
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu", weights_only=False)

    def _validate_calibration_state(self, state):
        if state is None:
            return None
        if not isinstance(state, Mapping):
            raise TypeError("calibration_state must be an explicit mapping")
        if state.get("source") != "allowed_prefix":
            raise ValueError("calibration_state must identify source='allowed_prefix'")
        prefix_bins = state.get("prefix_bins")
        if not isinstance(prefix_bins, (int, np.integer)) or prefix_bins <= 0:
            raise ValueError("calibration_state must include a positive prefix_bins count")
        out = {"source": "allowed_prefix", "prefix_bins": int(prefix_bins)}
        has_mean, has_std = "neural_mean" in state, "neural_std" in state
        if has_mean != has_std:
            raise ValueError("neural_mean and neural_std must be supplied together")
        if has_mean:
            mean = np.asarray(state["neural_mean"], dtype=np.float64)
            std = np.asarray(state["neural_std"], dtype=np.float64)
            if mean.shape != (self.config.channels,) or std.shape != (self.config.channels,):
                raise ValueError("calibration neural statistics have the wrong shape")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("calibration neural statistics must be finite with positive std")
            out["neural_mean"], out["neural_std"] = mean, np.maximum(std, .01)
        if "readout_state" in state:
            readout_state = state["readout_state"]
            if not isinstance(readout_state, Mapping):
                raise TypeError("readout_state must be a state-dict mapping")
            self.model.readout.load_state_dict(readout_state, strict=True)
            self.model.eval()
        elif not has_mean:
            raise ValueError("calibration_state must contain neural statistics or a readout_state")
        return out

    def reset(self, dataset_tags=()):
        """Clear causal state; opaque dataset tags are intentionally unused."""
        # Do not stringify, parse, hash, index with, or perform I/O from tags.
        del dataset_tags
        b, c = self.batch_size, self.config.channels
        self.hidden = None
        self.history = None
        self.count = np.zeros(b, dtype=np.int64)
        self.total = np.zeros((b, c), dtype=np.float64)
        self.squares = np.zeros((b, c), dtype=np.float64)
        self.smooth = None
        self.mean = np.broadcast_to(self._prior_mean, (b, c)).copy()
        self.std = np.broadcast_to(self._prior_std, (b, c)).copy()

    def set_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.reset()

    def _normalise_step(self, x):
        self.count += 1
        if self.config.online_norm:
            self.total += x
            self.squares += x * x
            n = self.count[:, None].astype(np.float64)
            running_mean = self.total / n
            variance = np.maximum(self.squares / n - running_mean * running_mean, 0.0)
            running_std = np.maximum(np.sqrt(variance), .01)
            blend = np.where(self.count >= 30, np.minimum(.7, self.count / 2000.0), 0.0)[:, None]
            mean = (1.0 - blend) * self._prior_mean[None, :] + blend * running_mean
            std = (1.0 - blend) * self._prior_std[None, :] + blend * running_std
        else:
            mean, std = self.mean, self.std
        z = ((x - mean) / std).astype(np.float32)
        if self.smooth is None:
            self.smooth = z
        else:
            alpha = self.config.smoothing
            self.smooth = alpha * self.smooth + (1.0 - alpha) * z
        return self.smooth.astype(np.float32, copy=False)

    @torch.inference_mode()
    def predict(self, neural_observations):
        x = np.asarray(neural_observations, dtype=np.float64)
        expected = (self.batch_size, self.config.channels)
        if x.shape != expected:
            raise ValueError(f"expected neural batch shape {expected}, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("neural observations must be finite")
        normalized = self._normalise_step(x)
        tensor = torch.as_tensor(normalized[:, None, :], dtype=torch.float32, device=self.device)
        with _strict_fp32_inference():
            prediction, self.hidden, self.history = self.model(tensor, self.hidden, self.history)
        prediction = prediction[:, 0].float().cpu().numpy()
        return (prediction * self._ystd + self._ymean).astype(np.float32, copy=False)

    def on_done(self, dones):
        """H1 neural state continues across trial markers."""
        return None

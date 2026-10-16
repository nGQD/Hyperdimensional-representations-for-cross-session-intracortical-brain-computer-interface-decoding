#!/usr/bin/env python3
"""Reproducible batch-one efficiency audit for the frozen H1 decoder study.

Run with the repository environment from the repository root:
    .venv/bin/python hdc_study/efficiency.py

Only existing selected transfer checkpoints are loaded. No fitting or labels
are used. The optional fused path is a separately measured FP32 algebraic
rewrite for the selected K=1 HDC/random checkpoints; it is not the frozen
implementation used for the primary measurements.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
OUT_JSON = RESULTS / "efficiency.json"
OUT_TEX = REPO / "neuroinformatics_submission" / "efficiency_generated.tex"
FAMILIES = ("hdc", "random", "mlp", "plain")
SOURCE_FILES = (
    HERE / "model.py",
    HERE / "engine.py",
    HERE / "adapter.py",
    HERE / "results" / "summary.json",
    HERE / "results" / "transfer_selection.json",
    REPO / "neuroinformatics_submission" / "protocol_generated.tex",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bytes_of_tensor(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "samples": int(arr.size),
    }


def _cpu_model_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return {"query": q.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return None


def _strict_inference_context():
    # Match adapter._strict_fp32_inference; preserve process-wide settings.
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        old = (torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    return ctx()


def _load_project_modules():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    challenge = REPO / "falcon-challenge"
    if str(challenge) not in sys.path:
        sys.path.insert(0, str(challenge))
    from adapter import H1Decoder
    from falcon_challenge.config import FalconConfig, FalconTask
    return H1Decoder, FalconConfig, FalconTask


def _load_measurement_inputs() -> tuple[np.ndarray, Path, dict[str, Any]]:
    manifest_path = HERE / "provenance" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [r for r in manifest if r.get("split") == "held-out-calib"]
    if not rows:
        raise RuntimeError("No existing held-out calibration cache is recorded")
    row = sorted(rows, key=lambda r: r["cache"])[0]
    path = HERE / "cache" / row["cache"]
    actual_hash = sha256_file(path)
    if actual_hash != row["cache_sha256"]:
        raise RuntimeError(f"Held-out input cache hash mismatch: {path.name}")
    # Intentionally load only neural observations. No behavior or label arrays.
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] != 176 or not np.isfinite(x).all():
        raise RuntimeError(f"Unexpected neural input array: {x.shape}")
    return x, path, {"manifest_row": row, "sha256": actual_hash}


def _decoder(H1Decoder, FalconConfig, FalconTask, checkpoint: Path, device: str):
    return H1Decoder(
        FalconConfig(task=FalconTask.h1), checkpoint=str(checkpoint),
        batch_size=1, device=device,
    )


def _norm_only_measure(decoder, samples: np.ndarray, warmup: int, repeats: int) -> dict[str, Any]:
    decoder.reset()
    latencies = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            start = time.perf_counter_ns() if i >= warmup else 0
            decoder._normalise_step(samples[i][None, :].astype(np.float64, copy=False))
            if i >= warmup:
                latencies.append((time.perf_counter_ns() - start) / 1e6)
    return _percentiles(latencies)


def _normalized_samples(decoder, samples: np.ndarray, warmup: int, repeats: int) -> list[np.ndarray]:
    decoder.reset()
    values = []
    for i in range(warmup + repeats):
        z = decoder._normalise_step(samples[i][None, :].astype(np.float64, copy=False))
        values.append(z.astype(np.float32, copy=True))
    return values


def _model_forward_measure(decoder, normalized: list[np.ndarray], warmup: int,
                           repeats: int, device: str) -> dict[str, Any]:
    decoder.reset()
    inputs = [torch.as_tensor(z[:, None, :], dtype=torch.float32, device=device)
              for z in normalized]
    hidden = history = None
    if device.startswith("cuda"):
        starts: list[torch.cuda.Event] = []
        ends: list[torch.cuda.Event] = []
        with torch.inference_mode(), _strict_inference_context():
            for i in range(warmup):
                _, hidden, history = decoder.model(inputs[i], hidden, history)
            torch.cuda.synchronize()
            for j in range(repeats):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _, hidden, history = decoder.model(inputs[warmup + j], hidden, history)
                end_event.record()
                starts.append(start_event)
                ends.append(end_event)
                if (j + 1) % 100 == 0:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
        vals = [float(a.elapsed_time(b)) for a, b in zip(starts, ends)]
        out = _percentiles(vals)
        out["clock"] = "CUDA events around model.forward; device input already prepared"
        return out

    vals = []
    with torch.inference_mode(), _strict_inference_context():
        for i in range(warmup):
            _, hidden, history = decoder.model(inputs[i], hidden, history)
        for j in range(repeats):
            start = time.perf_counter_ns()
            _, hidden, history = decoder.model(inputs[warmup + j], hidden, history)
            vals.append((time.perf_counter_ns() - start) / 1e6)
    out = _percentiles(vals)
    out["clock"] = "perf_counter_ns around model.forward; input tensor prepared"
    return out


def _api_measure(decoder, samples: np.ndarray, warmup: int, repeats: int,
                 device: str) -> dict[str, Any]:
    decoder.reset()
    vals = []
    for i in range(warmup + repeats):
        if i < warmup:
            decoder.predict(samples[i][None, :])
            continue
        start = time.perf_counter_ns()
        decoder.predict(samples[i][None, :])
        if device.startswith("cuda"):
            # Adapter .cpu().numpy() synchronizes already; make the timed boundary explicit.
            torch.cuda.synchronize()
        vals.append((time.perf_counter_ns() - start) / 1e6)
    return _percentiles(vals)


def _state_sizes(model) -> dict[str, Any]:
    param_bytes = 0
    param_count = 0
    for p in model.parameters():
        param_bytes += _bytes_of_tensor(p)
        param_count += int(p.numel())
    buffers = {"codebook_basis": 0, "temporal_roles": 0, "anchors_other": 0}
    buffer_count = 0
    for name, b in model.named_buffers():
        nbytes = _bytes_of_tensor(b)
        buffer_count += int(b.numel())
        if name.endswith("basis_matrix"):
            buffers["codebook_basis"] += nbytes
        elif ".role_" in name:
            buffers["temporal_roles"] += nbytes
        else:
            buffers["anchors_other"] += nbytes
    buffer_bytes = sum(buffers.values())
    return {
        "parameter_count": param_count,
        "parameter_bytes_fp32": param_bytes,
        "buffer_count": buffer_count,
        "buffers_bytes": buffers,
        "total_buffer_bytes": buffer_bytes,
        "resident_parameter_plus_buffer_bytes": param_bytes + buffer_bytes,
        "resident_parameter_plus_buffer_mib": (param_bytes + buffer_bytes) / (1024 ** 2),
    }


def _operation_counts(config: dict[str, Any]) -> dict[str, Any]:
    c = int(config["channels"])
    d = int(config["dimension"])
    h = int(config["hidden"])
    o = int(config["outputs"])
    k = int(config["temporal_bins"])
    l = int(config["levels"])
    family = config["family"]
    encoder_gemm = c * l * (2 * d) if family in ("hdc", "random") else 0
    if family in ("hdc", "random"):
        projection_layers = {"hypervector_to_hidden": (2 * d, h)}
    elif family == "mlp":
        width = max(1, round((2 * d * h) / (c * k + h + 1)))
        projection_layers = {"raw_lag_to_mlp": (c * k, width),
                             "mlp_to_hidden": (width, h)}
    else:
        projection_layers = {"raw_lag_to_hidden": (c * k, h)}
    projection_macs = sum(a * b for a, b in projection_layers.values())
    gru_input = 3 * h * h  # (3H x input_size), input_size=H for selected models
    gru_recurrent = 3 * h * h
    readout = h * h + h * o
    learned_macs = projection_macs + gru_input + gru_recurrent + readout
    # Source-level elementwise work for PopulationEncoder.basis_weights.
    interpolation = None
    bundler = None
    if family in ("hdc", "random"):
        interpolation = {
            "input_tanh": c,
            "input_adds_before_grid": c,
            "input_multiplies_before_grid": 2 * c,
            "grid_subtract_abs_subtract_relu": 4 * c * l,
            "dense_candidate_weights": c * l,
            "note": "Exact source-level scalar counts for basis_weights; tensor reshape/GEMM excluded.",
        }
        # K=1 selected rows. Complex rotation: 4 mul + 2 add/sub per coordinate;
        # adding to the zero accumulator adds two more additions per coordinate.
        bundler = {
            "complex_real_multiplications": 4 * d,
            "complex_add_subtracts_including_zero_accumulator": 4 * d,
            "output_divisions_by_sqrt_k": 2 * d,
            "temporal_bins": k,
            "note": "For K=1. Neural-level nonlinearities, LayerNorm, GELU, GRU gate elementwise operations and tensor movement are additional and excluded from MAC totals.",
        }
    return {
        "projection_layer_dimensions_in_out": {
            name: {"input": a, "output": b, "macs": a * b}
            for name, (a, b) in projection_layers.items()
        },
        "fixed_encoder_dense_gemm_macs": encoder_gemm,
        "projection_macs": projection_macs,
        "gru_input_affine_macs": gru_input,
        "gru_recurrent_affine_macs": gru_recurrent,
        "readout_linear_macs": readout,
        "projection_plus_gru_plus_readout_macs": learned_macs,
        "total_dense_macs_per_bin": encoder_gemm + learned_macs,
        "interpolation_elementwise": interpolation,
        "role_binding_elementwise": bundler,
        "excluded_from_mac_total": [
            "normalization, tanh and triangular interpolation arithmetic",
            "complex temporal role binding and accumulation",
            "LayerNorm reductions/affine operations, GELU, GRU sigmoid/tanh and gates",
            "bias additions, history concatenation/slicing, memory movement, API dispatch",
        ],
        "mac_definition": "One multiply-accumulate is counted as one MAC; bias and non-GEMM elementwise operations are separate.",
    }


class StreamNormalizer:
    """Small batch-one copy of the frozen adapter's online normalization state."""
    def __init__(self, prior_mean, prior_std, online: bool, alpha: float):
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64).copy()
        self.prior_std = np.asarray(prior_std, dtype=np.float64).copy()
        self.online = bool(online)
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self.count = 0
        self.total = np.zeros_like(self.prior_mean)
        self.squares = np.zeros_like(self.prior_mean)
        self.smooth = None

    def step(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        if self.online:
            self.total += x
            self.squares += x * x
            n = float(self.count)
            running_mean = self.total / n
            variance = np.maximum(self.squares / n - running_mean * running_mean, 0.0)
            running_std = np.maximum(np.sqrt(variance), .01)
            blend = min(.7, n / 2000.0) if self.count >= 30 else 0.0
            mean = (1.0 - blend) * self.prior_mean + blend * running_mean
            std = (1.0 - blend) * self.prior_std + blend * running_std
        else:
            mean, std = self.prior_mean, self.prior_std
        z = ((x - mean) / std).astype(np.float32)
        if self.smooth is None:
            self.smooth = z
        else:
            self.smooth = self.alpha * self.smooth + (1.0 - self.alpha) * z
        return self.smooth.astype(np.float32, copy=False)


class FusedSparseRuntime:
    """Algebraic K=1 fusion for one selected HDC/random checkpoint."""
    def __init__(self, base_decoder, device: str):
        model = base_decoder.model
        c = model.config
        if c.family not in ("hdc", "random") or c.temporal_bins != 1:
            raise ValueError("Sparse fused runtime requires HDC/random with K=1")
        if model.input_projection.__class__.__name__ != "Linear":
            raise ValueError("Expected the selected single affine input projection")
        self.device = torch.device(device)
        self.config = c
        self.input_norm = model.input_norm
        self.gru = model.gru
        self.state_norm = model.state_norm
        self.readout = model.readout
        self.original_parameter_count = sum(int(p.numel()) for p in model.parameters())
        self.removed_projection_parameter_count = sum(
            int(p.numel()) for p in model.input_projection.parameters()
        )
        self.ymean = base_decoder._ymean.copy()
        self.ystd = base_decoder._ystd.copy()
        self.normalizer = StreamNormalizer(
            base_decoder._prior_mean, base_decoder._prior_std,
            c.online_norm, c.smoothing,
        )
        self.hidden = None
        self.precompute_macs = c.channels * c.levels * (2 * c.dimension) * c.hidden
        start = time.perf_counter_ns()
        with torch.inference_mode(), _strict_inference_context():
            basis = model.encoder.basis_matrix.reshape(
                c.channels, c.levels, 2 * c.dimension
            )
            d = c.dimension
            real, imag = basis[..., :d], basis[..., d:]
            role_real = model.bundler.role_real[0]
            role_imag = model.bundler.role_imag[0]
            rotated = torch.cat((
                real * role_real - imag * role_imag,
                real * role_imag + imag * role_real,
            ), dim=-1)
            table = rotated.reshape(-1, 2 * d) @ model.input_projection.weight.T
            self.table = table.reshape(c.channels, c.levels, c.hidden).to(self.device).contiguous()
            self.bias = model.input_projection.bias.detach().to(self.device).contiguous()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.precompute_seconds = (time.perf_counter_ns() - start) / 1e9
        self.reset()

    def reset(self):
        self.normalizer.reset()
        self.hidden = None

    def _project(self, x):
        c = self.config
        # Exact two-neighbor form of the frozen triangular basis: tanh bounds
        # scaled values to [0,L-1], so no other anchor has nonzero weight.
        scaled = (torch.tanh(x.float()) + 1.0) * 0.5 * (c.levels - 1)
        lower = torch.floor(scaled).long().clamp(0, c.levels - 1)
        fraction = scaled - lower.to(scaled.dtype)
        upper = (lower + 1).clamp(max=c.levels - 1)
        lo = lower[0, 0]
        hi = upper[0, 0]
        frac = fraction[0, 0]
        channels = torch.arange(c.channels, device=x.device)
        table = self.table
        lo_values = table[channels, lo]
        hi_values = table[channels, hi]
        combined = lo_values * (1.0 - frac[:, None]) + hi_values * frac[:, None]
        return (combined.sum(dim=0) + self.bias)[None, None, :]

    def _forward(self, x):
        projected = self.input_norm(self._project(x))
        state, self.hidden = self.gru(projected, self.hidden)
        prediction = self.readout(self.state_norm(state))
        return prediction

    def predict(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        normalized = self.normalizer.step(raw)
        tensor = torch.as_tensor(normalized[None, None, :], dtype=torch.float32,
                                 device=self.device)
        with torch.inference_mode(), _strict_inference_context():
            prediction = self._forward(tensor)
        result = prediction[:, 0].float().cpu().numpy()
        return (result * self.ystd + self.ymean).astype(np.float32, copy=False)


def _fused_memory(runtime: FusedSparseRuntime) -> dict[str, Any]:
    remaining_params = sum(
        int(p.numel())
        for module in (runtime.input_norm, runtime.gru, runtime.state_norm, runtime.readout)
        for p in module.parameters()
    )
    all_params = runtime.original_parameter_count
    projection_params = runtime.removed_projection_parameter_count
    table_bytes = _bytes_of_tensor(runtime.table)
    # The affine bias remains as a frozen inference tensor after its weight
    # matrix has been folded into the lookup table.
    other_buffers_bytes = _bytes_of_tensor(runtime.bias)
    # Runtime drops the original encoder, temporal rotation buffers and affine.
    return {
        "original_checkpoint_parameter_count": int(all_params),
        "compiled_retained_trained_parameter_values": remaining_params,
        "compiled_fused_table_values": int(runtime.table.numel()),
        "compiled_fused_table_bytes_fp32": table_bytes,
        "compiled_bias_buffer_bytes_fp32": _bytes_of_tensor(runtime.bias),
        "compiled_other_runtime_buffer_bytes": other_buffers_bytes,
        "compiled_static_tensor_bytes": remaining_params * 4 + table_bytes + other_buffers_bytes,
        "compiled_static_tensor_mib": (remaining_params * 4 + table_bytes + other_buffers_bytes) / (1024 ** 2),
        "removed_projection_parameter_count": int(projection_params),
        "note": "Inference-only compiled representation; the table depends on fixed checkpoint buffers and trained projection weights and is not a trainable model parameter.",
    }


def _fused_forward_measure(runtime: FusedSparseRuntime, normalized: list[np.ndarray],
                            warmup: int, repeats: int, device: str) -> dict[str, Any]:
    runtime.reset()
    inputs = [torch.as_tensor(z[:, None, :], dtype=torch.float32, device=device)
              for z in normalized]
    if device.startswith("cuda"):
        starts, ends = [], []
        with torch.inference_mode(), _strict_inference_context():
            for i in range(warmup):
                runtime._forward(inputs[i])
            torch.cuda.synchronize()
            for j in range(repeats):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                runtime._forward(inputs[warmup + j])
                b.record()
                starts.append(a)
                ends.append(b)
                if (j + 1) % 100 == 0:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
        out = _percentiles([float(a.elapsed_time(b)) for a, b in zip(starts, ends)])
        out["clock"] = "CUDA events around fused model.forward; device input already prepared"
        return out
    vals = []
    with torch.inference_mode(), _strict_inference_context():
        for i in range(warmup):
            runtime._forward(inputs[i])
        for j in range(repeats):
            start = time.perf_counter_ns()
            runtime._forward(inputs[warmup + j])
            vals.append((time.perf_counter_ns() - start) / 1e6)
    out = _percentiles(vals)
    out["clock"] = "perf_counter_ns around fused model.forward; input tensor prepared"
    return out


def _fused_api_measure(runtime: FusedSparseRuntime, samples: np.ndarray,
                       warmup: int, repeats: int, device: str) -> dict[str, Any]:
    runtime.reset()
    vals = []
    for i in range(warmup + repeats):
        if i < warmup:
            runtime.predict(samples[i])
            continue
        start = time.perf_counter_ns()
        runtime.predict(samples[i])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        vals.append((time.perf_counter_ns() - start) / 1e6)
    return _percentiles(vals)


def _fused_norm_only_measure(runtime: FusedSparseRuntime, samples: np.ndarray,
                             warmup: int, repeats: int) -> dict[str, Any]:
    runtime.reset()
    vals = []
    for i in range(warmup + repeats):
        start = time.perf_counter_ns() if i >= warmup else 0
        runtime.normalizer.step(samples[i])
        if i >= warmup:
            vals.append((time.perf_counter_ns() - start) / 1e6)
    out = _percentiles(vals)
    out["clock"] = "perf_counter_ns around fused-runtime host normalization only"
    return out


def _validate_fusion(base_decoder, runtime: FusedSparseRuntime, samples: np.ndarray,
                     device: str, atol: float = 2e-5, rtol: float = 2e-5) -> dict[str, Any]:
    base_decoder.reset()
    runtime.reset()
    pred_abs = []
    hidden_abs = []
    passed = True
    with torch.inference_mode():
        for raw in samples:
            pred_base = base_decoder.predict(raw[None, :])[0]
            pred_fused = runtime.predict(raw)[0]
            diff = np.abs(pred_base - pred_fused)
            pred_abs.extend(diff.tolist())
            if not np.allclose(pred_base, pred_fused, atol=atol, rtol=rtol):
                passed = False
            if base_decoder.hidden is not None and runtime.hidden is not None:
                diff_h = (base_decoder.hidden - runtime.hidden).abs().detach().cpu().numpy()
                hidden_abs.extend(diff_h.reshape(-1).tolist())
                if not np.allclose(base_decoder.hidden.detach().cpu().numpy(),
                                   runtime.hidden.detach().cpu().numpy(),
                                   atol=atol, rtol=rtol):
                    passed = False
    if hidden_abs:
        h = np.asarray(hidden_abs)
        hidden_max = float(h.max())
        hidden_p99 = float(np.percentile(h, 99))
    else:
        hidden_max = hidden_p99 = None
    p = np.asarray(pred_abs, dtype=np.float64)
    return {
        "passed": bool(passed),
        "device": device,
        "stream_bins": int(len(samples)),
        "comparison": "checkpoint adapter prediction and final recurrent hidden state against FP32 sparse fused runtime",
        "atol": atol,
        "rtol": rtol,
        "prediction_max_abs": float(p.max()) if p.size else 0.0,
        "prediction_p99_abs": float(np.percentile(p, 99)) if p.size else 0.0,
        "hidden_max_abs_over_stream": hidden_max,
        "hidden_p99_abs_over_stream": hidden_p99,
        "floating_point_note": "The real-valued algebra is equivalent; composing GEMMs and changing summation order changes FP32 rounding. Values are checked against the stated tolerances.",
    }


def _normalized_fused_samples(runtime: FusedSparseRuntime, samples: np.ndarray,
                              warmup: int, repeats: int) -> list[np.ndarray]:
    runtime.reset()
    values = []
    for i in range(warmup + repeats):
        z = runtime.normalizer.step(samples[i])
        values.append(z.astype(np.float32, copy=True)[None, :])
    return values


def _fmt_pair(value: dict[str, Any] | None) -> str:
    if not value or "p50_ms" not in value:
        return "--"
    return f"{value['p50_ms']:.3f}/{value['p95_ms']:.3f}"


def _compute_row_tex(label: str, d: int, k: int, params: str,
                     fixed_mib: float, enc_macs: int, tail_macs: int,
                     total_macs: int) -> str:
    return (f"{label} & $D={d},K={k}$ & {params} & {fixed_mib:.3f} & "
            f"{enc_macs:,} & {tail_macs:,} & {total_macs:,} \\\\")


def _latency_row_tex(label: str, cpu_api: dict[str, Any], cpu_norm: dict[str, Any],
                     cpu_model: dict[str, Any], gpu_api: dict[str, Any] | None,
                     gpu_model: dict[str, Any] | None) -> str:
    return (f"{label} & {_fmt_pair(cpu_api)} & {_fmt_pair(cpu_norm)} & "
            f"{_fmt_pair(cpu_model)} & {_fmt_pair(gpu_api)} & "
            f"{_fmt_pair(gpu_model)} \\\\")


def _write_tex(result: dict[str, Any]) -> None:
    fam = result["families"]
    opt = result["fused_variants"]
    compute_rows = []
    latency_rows = []
    labels = {"hdc": "HDC", "random": "Random", "mlp": "MLP", "plain": "Plain"}
    for family in FAMILIES:
        row = fam[family]
        ops = row["operations_per_bin"]
        fixed_mib = row["memory"]["total_buffer_bytes"] / (1024 ** 2)
        total_mib = row["memory"]["resident_parameter_plus_buffer_mib"]
        label = labels[family]
        compute_rows.append(
            f"{label} & {ops['total_dense_macs_per_bin']:,} & "
            f"{fixed_mib:.3f} & {total_mib:.3f} \\\\")
        cpu = row["latency"]["cpu"]["adapter_predict"]
        gpu = row["latency"]["gpu"].get("adapter_predict")
        latency_rows.append(
            f"{label} & {_fmt_pair(cpu)} & {_fmt_pair(gpu)} \\\\")
    validation_text = []
    for family in ("hdc", "random"):
        row = opt[family]
        ops = row["operations_per_bin"]
        fixed_bytes = (row["compiled_memory"]["compiled_fused_table_bytes_fp32"] +
                       row["compiled_memory"]["compiled_other_runtime_buffer_bytes"])
        fixed_mib = fixed_bytes / (1024 ** 2)
        total_mib = row["compiled_memory"]["compiled_static_tensor_mib"]
        label = f"{labels[family]} compiled"
        compute_rows.append(
            f"{label} & {ops['total_online_mac_equivalents']:,} & "
            f"{fixed_mib:.3f} & {total_mib:.3f} \\\\")
        cpu = row["latency"]["cpu"]["adapter_predict"]
        gpu = row["latency"]["gpu"].get("adapter_predict")
        latency_rows.append(
            f"{label} & {_fmt_pair(cpu)} & {_fmt_pair(gpu)} \\\\")
        v = row["validation"]
        validation_text.append(
            f"{labels[family]} compiled was compared with the original over {v['stream_bins']:,} bins; "
            f"maximum absolute prediction and hidden-state differences were "
            f"{v['prediction_max_abs']:.2g} and {v['hidden_max_abs_over_stream']:.2g}, "
            f"respectively ($|e|\\leq {v['atol']:.0e}+{v['rtol']:.0e}|y|$).")

    proxy = result["energy"]["arithmetic_only_sensitivity_proxy"]["per_bin_subtotal_microjoules"]
    parameter_text = (
        "Stored trained parameter values in selected checkpoints were " +
        ", ".join(f"{labels[f]} {fam[f]['selected_checkpoint']['parameters']:,}"
                   for f in FAMILIES) + ". Each compiled HDC/random runtime retains " +
        f"{opt['hdc']['compiled_memory']['compiled_retained_trained_parameter_values']:,} "
        "trained tail values; the compiled table and retained affine bias are counted as static storage."
    )
    energy_line = (
        f"An arithmetic-only sensitivity subtotal is {proxy['hdc']:.3g} $\\mu$J/bin for HDC and random, "
        f"{proxy['mlp']:.3g} for MLP, {proxy['plain']:.3g} for plain, and "
        f"{proxy['hdc_fused']:.3g} for either compiled model. This illustration assigns one FP32 multiply "
        r"(3.7 pJ) and one FP32 add (0.9 pJ) to each MAC, or 4.6 pJ/MAC at the cited 45-nm technology point. "
        r"It is not measured energy for either benchmark device and excludes memory movement, nonlinearities, "
        r"normalization, and system overhead; operation counts cannot establish battery use or emissions "
        r"\citep{horowitz2014energy,schwartz2020green}.\par"
    )

    methods_template = r"""\providecommand{\EfficiencyMethods}{%
We profiled existing selected transfer checkpoints at batch size one and one 20-ms neural bin per call, preserving causal state between calls. The pooled inner-validation winners came from \texttt{transfer\_selection.json}; each representative checkpoint was trained with seed 101 on the six held-in groups. The selected configurations were HDC D=512/K=1, random D=512/K=1, MLP D=512/K=8, and plain D=256/K=1; respective learning rates were $2\times10^{-4}$, $2\times10^{-4}$, $2\times10^{-4}$, and $8\times10^{-4}$. All models use 176 inputs, 128 hidden units, and seven outputs; the channel-value codebook has eight levels. Only HDC/random codebooks and temporal roles are fixed; all input projections, GRU cells, and readouts were trained. No fitting was performed for this analysis. We used @WARMUP@ warm-up and @REPEATS@ measured steps from an existing held-out calibration neural input cache and loaded only the neural array, not behavior labels. The saved adapter performs host-side cumulative normalization and smoothing, strict-FP32 model inference, and output unnormalization; all are included in end-to-end latency. CPU inference used four PyTorch threads on @CPU@. The GPU was @GPU@; software was PyTorch @TORCH@ with CUDA @CUDA@. GPU model-forward timings use synchronized CUDA events on preloaded device inputs; end-to-end timings include host normalization, transfer, and synchronized output return. Memory reports exact FP32 parameter and registered-buffer tensor bytes, excluding activations, Python objects, and allocator overhead. Affine and GRU counts are exact matrix multiply-accumulates per bin; the fixed dense encoder GEMM is reported separately. Non-GEMM elementwise work, model-only latency, and normalization-only latency are also recorded in the machine-readable JSON.\par}
"""
    methods = methods_template.replace("@WARMUP@", str(result["measurement"]["warmup_steps"]))
    methods = methods.replace("@REPEATS@", f"{result['measurement']['measured_steps']:,}")
    methods = methods.replace("@CPU@", str(result["measurement"]["cpu"].get("model") or "CPU model not reported"))
    methods = methods.replace("@GPU@", str(result["measurement"].get("gpu_name") or "unavailable"))
    methods = methods.replace("@TORCH@", str(result["measurement"]["torch_version"]))
    methods = methods.replace("@CUDA@", str(result["measurement"].get("cuda_version") or "unavailable"))

    results_template = r"""\providecommand{\EfficiencyResults}{%
\begin{table}[t]\centering\small
\caption{Per-bin arithmetic and static tensor storage. MAC counts include affine and GRU matrix operations. Fixed buffers are separated from stored trained parameter values; compiled lookup tables are derived from the selected codebook and trained input affine.}\label{tab:efficiency-cost}
\begin{tabular}{lrrr}\hline
Runtime & MAC/bin & Fixed/compiled MiB & Total MiB \\ \hline
@COMPUTE_ROWS@
\hline\end{tabular}\end{table}
\begin{table}[t]\centering\small
\caption{End-to-end adapter latency in milliseconds, reported as p50/p95. The nominal 20-ms budget corresponds to a 50-Hz input rate. Separate model-forward and host-normalization timings are recorded in the accompanying JSON.}\label{tab:efficiency-latency}
\begin{tabular}{lrr}\hline
Runtime & CPU p50/p95 & GPU p50/p95 \\ \hline
@LATENCY_ROWS@
\hline\end{tabular}\end{table}
The frozen models are the primary results; compiled HDC/random rows are separate inference-only variants. Tables~\ref{tab:efficiency-cost} and~\ref{tab:efficiency-latency} summarize arithmetic, storage, and end-to-end latency. @PARAMETER_TEXT@ For K=1, let $b_{c\ell}$ be a fixed basis row, $r_0$ the fixed complex temporal role, $w_{c\ell}(x_{tc})$ its interpolation weight, and $(W,b)$ the trained input affine. Linearity gives
\[\begin{aligned}
u_t&=\sum_{c,\ell}w_{c\ell}(x_{tc})b_{c\ell},\\
p_t&=(u_t\odot_{\mathbb C}r_0)W^\top+b=\sum_{c,\ell}w_{c\ell}(x_{tc})T_{c\ell}+b,\\
T_{c\ell}&=(b_{c\ell}\odot_{\mathbb C}r_0)W^\top.
\end{aligned}\]
Each channel has at most two nonzero adjacent-anchor weights, so runtime gathers two table rows, weights them, and sums across channels. Table construction costs $CL(2D)H=184{,}549{,}376$ MACs per checkpoint. It saves 1,527,808 online MAC-equivalents per bin; the arithmetic-only amortization point is 121 bins (2.42 s at 50 Hz), excluding setup, memory movement, and runtime overhead. Reassociation changes FP32 rounding. @VALIDATION_TEXT@ No target labels were read.\par
@ENERGY_LINE@
}
"""
    results_macro = results_template.replace("@COMPUTE_ROWS@", "\n".join(compute_rows))
    results_macro = results_macro.replace("@LATENCY_ROWS@", "\n".join(latency_rows))
    results_macro = results_macro.replace("@PARAMETER_TEXT@", parameter_text)
    results_macro = results_macro.replace("@VALIDATION_TEXT@", " ".join(validation_text))
    results_macro = results_macro.replace("@ENERGY_LINE@", energy_line)
    discussion = r"""\providecommand{\EfficiencyDiscussion}{%
The dense HDC/random encoder applies a fixed $CL\times2D$ codebook matrix once per bin, adding 1,441,792 MACs to the trained projection, GRU, and readout at the selected D=512/K=1 setting. In the compiled variants, the fixed codebook and K=1 temporal role are folded with the trained input affine into a lookup table; each channel then contributes at most two weighted entries. This lowers counted online arithmetic and static tensor storage, while the measured latency remains specific to this implementation and hardware. The nominal 50-Hz deadline is not a hard real-time guarantee. No instrumented energy, power, battery, or emissions measurements were available; the arithmetic-only subtotal is a literature-based sensitivity illustration, not a device estimate.\par}
"""
    text = "% Generated by hdc_study/efficiency.py; do not edit by hand.\n" + methods + "\n" + results_macro + "\n" + discussion
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(text)

def _build_result(warmup: int, repeats: int) -> dict[str, Any]:
    H1Decoder, FalconConfig, FalconTask = _load_project_modules()
    selection_path = HERE / "results" / "transfer_selection.json"
    selection = json.loads(selection_path.read_text())
    samples, cache_path, cache_meta = _load_measurement_inputs()
    needed = warmup + repeats
    if samples.shape[0] < max(needed, 1200):
        raise RuntimeError(f"Need at least {max(needed,1200)} input bins, found {len(samples)}")
    selected_records = {}
    decoders = {}
    configs = {}
    for family in FAMILIES:
        checkpoint = HERE / "checkpoints" / f"transfer_{family}_s101.pt"
        decoder = _decoder(H1Decoder, FalconConfig, FalconTask, checkpoint, "cpu")
        record = decoder.record
        selected = selection[family]
        expected_config = dict(selected["config"])
        expected_config["codebook_seed"] = 100101  # frozen representation seed rule, 100000 + 101
        if record.get("config") != expected_config:
            raise RuntimeError(f"Checkpoint does not match selected transfer config: {family}")
        if record.get("seed") != 101 or record.get("parameters") != selected["parameters"]:
            raise RuntimeError(f"Checkpoint seed/parameter mismatch: {family}")
        result_path = HERE / "results" / "runs" / f"transfer_{family}_s101.json"
        run_result = json.loads(result_path.read_text())
        if run_result.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError(f"Run record checkpoint hash mismatch: {family}")
        configs[family] = record["config"]
        selected_records[family] = {
            "family": family,
            "selected_key": selected["key"],
            "inner_selected_mean_score": selected["score"],
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "parameters": record["parameters"],
            "training_seed": record["seed"],
            "codebook_seed": record["config"]["codebook_seed"],
            "train_days": record["train_days"],
            "training_file_count": len(record["train_files"]),
            "config": record["config"],
            "transfer_run_record_sha256": sha256_file(result_path),
        }
        decoders[family] = decoder

    # A single GPU process performs all GPU timing sequentially. The frozen
    # implementation and fused variants are not interleaved with other work.
    cuda_usable = bool(torch.cuda.is_available())
    cuda_error = None
    if cuda_usable:
        try:
            torch.cuda.init()
            _ = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
        except Exception as exc:  # include exact failure and continue CPU-only
            cuda_usable = False
            cuda_error = f"{type(exc).__name__}: {exc}"
    device_name = torch.cuda.get_device_name(0) if cuda_usable else None
    torch.set_num_threads(4)
    gpu_snapshot_before_benchmark = _nvidia_smi_snapshot()

    family_results: dict[str, Any] = {}
    for family in FAMILIES:
        base = decoders[family]
        # The original adapter is freshly reset for each independently measured path.
        cpu_decoder = base
        cpu_norm = _norm_only_measure(cpu_decoder, samples, warmup, repeats)
        normalized = _normalized_samples(cpu_decoder, samples, warmup, repeats)
        cpu_forward = _model_forward_measure(cpu_decoder, normalized, warmup, repeats, "cpu")
        cpu_api = _api_measure(cpu_decoder, samples, warmup, repeats, "cpu")
        latency = {
            "cpu": {
                "adapter_predict": cpu_api,
                "model_forward": cpu_forward,
                "normalization_only": cpu_norm,
            }
        }
        if cuda_usable:
            gpu_decoder = _decoder(
                H1Decoder, FalconConfig, FalconTask,
                HERE / "checkpoints" / f"transfer_{family}_s101.pt", "cuda",
            )
            gpu_normalized = _normalized_samples(gpu_decoder, samples, warmup, repeats)
            gpu_forward = _model_forward_measure(gpu_decoder, gpu_normalized, warmup, repeats, "cuda")
            gpu_api = _api_measure(gpu_decoder, samples, warmup, repeats, "cuda")
            latency["gpu"] = {
                "adapter_predict": gpu_api,
                "model_forward": gpu_forward,
                "normalization_only_host_cpu": cpu_norm,
            }
            del gpu_decoder
            torch.cuda.empty_cache()
        else:
            latency["gpu"] = {"available": False, "reason": cuda_error or "CUDA unavailable"}
        family_results[family] = {
            "selected_checkpoint": selected_records[family],
            "memory": _state_sizes(base.model),
            "operations_per_bin": _operation_counts(configs[family]),
            "latency": latency,
        }

    fused_results: dict[str, Any] = {}
    for family in ("hdc", "random"):
        cpu_base = _decoder(
            H1Decoder, FalconConfig, FalconTask,
            HERE / "checkpoints" / f"transfer_{family}_s101.pt", "cpu",
        )
        cpu_runtime = FusedSparseRuntime(cpu_base, "cpu")
        validation = _validate_fusion(cpu_base, cpu_runtime, samples, "cpu")
        if not validation["passed"]:
            raise RuntimeError(f"Fused {family} numerical validation failed: {validation}")
        normalized = _normalized_fused_samples(cpu_runtime, samples, warmup, repeats)
        # The compiled runtime holds only the fused table and the unchanged
        # normalizer/GRU/readout tail; release the original encoder and affine.
        del cpu_base
        gc.collect()
        cpu_forward = _fused_forward_measure(cpu_runtime, normalized, warmup, repeats, "cpu")
        cpu_api = _fused_api_measure(cpu_runtime, samples, warmup, repeats, "cpu")
        cpu_norm = _fused_norm_only_measure(cpu_runtime, samples, warmup, repeats)
        latency = {"cpu": {
            "adapter_predict": cpu_api,
            "model_forward": cpu_forward,
            "normalization_only": cpu_norm,
        }}
        gpu_validation = None
        gpu_precompute_seconds = None
        if cuda_usable:
            gpu_base = _decoder(
                H1Decoder, FalconConfig, FalconTask,
                HERE / "checkpoints" / f"transfer_{family}_s101.pt", "cuda",
            )
            gpu_runtime = FusedSparseRuntime(gpu_base, "cuda")
            gpu_precompute_seconds = gpu_runtime.precompute_seconds
            # Validate the same full real-input stream on GPU before timing.
            gpu_validation = _validate_fusion(gpu_base, gpu_runtime, samples, "cuda")
            if not gpu_validation["passed"]:
                raise RuntimeError(f"GPU fused {family} numerical validation failed: {gpu_validation}")
            gpu_normalized = _normalized_fused_samples(gpu_runtime, samples, warmup, repeats)
            del gpu_base
            gc.collect()
            torch.cuda.empty_cache()
            gpu_forward = _fused_forward_measure(gpu_runtime, gpu_normalized, warmup, repeats, "cuda")
            gpu_api = _fused_api_measure(gpu_runtime, samples, warmup, repeats, "cuda")
            latency["gpu"] = {
                "adapter_predict": gpu_api,
                "model_forward": gpu_forward,
                "normalization_only_host_cpu": cpu_norm,
            }
            del gpu_runtime
            torch.cuda.empty_cache()
        else:
            latency["gpu"] = {"available": False, "reason": cuda_error or "CUDA unavailable"}

        cfg = configs[family]
        c, h = cfg["channels"], cfg["hidden"]
        sparse_macs = 2 * c * h
        gru_readout = (6 * h * h) + (h * h + h * cfg["outputs"])
        original_macs = family_results[family]["operations_per_bin"]["total_dense_macs_per_bin"]
        fused_macs = sparse_macs + gru_readout
        precompute_macs = cpu_runtime.precompute_macs
        macs_saved = original_macs - fused_macs
        break_even_bins = math.ceil(precompute_macs / macs_saved)
        fused_results[family] = {
            "variant": "FP32 offline fused basis/temporal rotation/input projection plus exact two-anchor sparse interpolation",
            "training_or_primary_results_changed": False,
            "operations_per_bin": {
                "fixed_encoder_dense_gemm_macs": 0,
                "sparse_lookup_mac_equivalent": sparse_macs,
                "gru_plus_readout_macs": gru_readout,
                "total_online_mac_equivalents": fused_macs,
                "original_dense_online_macs": original_macs,
                "online_mac_equivalents_saved_per_bin": macs_saved,
                "one_time_offline_fusion_gemm_macs": precompute_macs,
                "arithmetic_only_amortization_bins": break_even_bins,
                "arithmetic_only_amortization_seconds_at_50hz": break_even_bins / 50.0,
                "sparse_lookup_scalar_multiplications": 2 * c * h,
                "sparse_lookup_scalar_additions": (2 * c - 1) * h,
                "note": "Sparse lookup count treats each of two weighted anchors per channel as one MAC-equivalent; table construction is one-time and excluded from per-bin counts.",
            },
            "compiled_memory": _fused_memory(cpu_runtime),
            "offline_precompute_seconds_cpu": cpu_runtime.precompute_seconds,
            "offline_precompute_seconds_gpu": gpu_precompute_seconds,
            "validation": validation,
            "gpu_validation": gpu_validation,
            "latency": latency,
        }
        del cpu_runtime

    selected_payload = {
        "selection_source": "hdc_study/results/transfer_selection.json; pooled mean of the six inner validation group selections",
        "representative_checkpoint_rule": "Existing trained transfer checkpoint, training seed 101, fitted on all six held-in groups; codebook_seed=100000+seed=100101",
        "selection_file_sha256": sha256_file(selection_path),
        "checkpoints": selected_records,
    }
    result = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Compute, memory, and one-bin causal latency audit of existing selected HDC/random/MLP/plain models; no retraining.",
        "selection": selected_payload,
        "measurement": {
            "command": ".venv/bin/python hdc_study/efficiency.py",
            "batch_size": 1,
            "bins_per_predict_call": 1,
            "nominal_bin_rate_hz": 50,
            "nominal_deadline_ms": 20.0,
            "warmup_steps": warmup,
            "measured_steps": repeats,
            "model_forward_includes_normalization": False,
            "adapter_predict_includes": [
                "finite-value check", "host online moments and smoothing",
                "host-to-device transfer where applicable", "strict-FP32 model forward",
                "prediction transfer and target unnormalization",
            ],
            "adapter_predict_excludes": ["file I/O", "data loading", "calibration fitting", "trial scheduling"],
            "inputs": {
                "cache": str(cache_path.relative_to(REPO)),
                "sha256": cache_meta["sha256"],
                "held_out_input_bins": int(samples.shape[0]),
                "input_shape": list(samples.shape),
                "fields_loaded": ["x"],
                "labels_loaded_or_used": False,
                "first_measured_recording": cache_meta["manifest_row"].get("path"),
            },
            "cpu": {
                "model": _cpu_model_name(),
                "logical_cpus": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "python": sys.version.split()[0],
                "os": platform.platform(),
            },
            "torch_version": torch.__version__,
            "cuda_available_and_kernel_smoke_tested": cuda_usable,
            "cuda_version": torch.version.cuda,
            "gpu_name": device_name,
            "gpu_driver_snapshot_before_benchmark": gpu_snapshot_before_benchmark,
            "inference_precision": "FP32; TF32 disabled in model forward, autocast disabled",
            "gpu_model_forward_clock": "CUDA events with synchronization every 100 samples and at end",
            "gpu_adapter_clock": "perf_counter_ns around complete predict, synchronized after each returned output",
            "latency_note": "Single process, batch one, sequential causal stream; p50/p95 are empirical timing quantiles and do not establish hard real-time behavior.",
        },
        "families": family_results,
        "fused_variants": fused_results,
        "energy": {
            "measured": False,
            "reason": "No instrumented energy or power measurement was available.",
            "claim": "MAC and tensor-storage counts are compute/memory descriptors only; no device watts, joules, battery life, or CO2 are inferred.",
            "arithmetic_only_sensitivity_proxy": {
                "measured": False,
                "energy_per_fp32_multiply_pj": 3.7,
                "energy_per_fp32_add_pj": 0.9,
                "arithmetic_subtotal_per_mac_pj": 4.6,
                "technology_point": "Horowitz 45-nm FP32 rough operation-energy illustration; not either measured device",
                "per_bin_subtotal_microjoules": {
                    **{f: family_results[f]["operations_per_bin"]["total_dense_macs_per_bin"] * 4.6 / 1e6
                       for f in FAMILIES},
                    **{f"{f}_fused": fused_results[f]["operations_per_bin"]["total_online_mac_equivalents"] * 4.6 / 1e6
                       for f in ("hdc", "random")},
                },
                "limitations": [
                    "Excludes memory movement, elementwise operations, normalization, and system overhead.",
                    "This technology-point arithmetic subtotal cannot predict measured device energy or power.",
                    "No battery-life or emissions inference is made.",
                ],
                "citation_keys": ["horowitz2014energy", "schwartz2020green"],
            },
        },
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in SOURCE_FILES},
        "efficiency_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    return result


def verify_result() -> tuple[bool, list[str]]:
    """Check provenance and generated-TeX freshness without rerunning timings."""
    if not OUT_JSON.is_file() or not OUT_TEX.is_file():
        return False, ["efficiency.json or efficiency_generated.tex is missing"]
    result = json.loads(OUT_JSON.read_text())
    errors = []
    if result.get("efficiency_script_sha256") != sha256_file(Path(__file__).resolve()):
        errors.append("efficiency.py hash differs from measured result")
    if result.get("generated_tex_sha256") != sha256_file(OUT_TEX):
        errors.append("efficiency_generated.tex hash differs from measured result")
    for rel, digest in result.get("source_sha256", {}).items():
        path = REPO / rel
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"frozen source hash differs: {rel}")
    for family, rec in result.get("selection", {}).get("checkpoints", {}).items():
        path = REPO / rec["checkpoint"]
        if not path.is_file() or sha256_file(path) != rec["checkpoint_sha256"]:
            errors.append(f"selected checkpoint hash differs: {family}")
    if errors:
        return False, errors
    return True, ["efficiency JSON, generated TeX, frozen sources, and selected checkpoints match recorded hashes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--verify", action="store_true",
                        help="Check recorded hashes without rerunning benchmarks")
    args = parser.parse_args()
    if args.verify:
        ok, messages = verify_result()
        print(json.dumps({"ok": ok, "messages": messages}, indent=2))
        return 0 if ok else 1
    if args.warmup < 1 or args.repeats < 20:
        parser.error("--warmup must be positive and --repeats at least 20")
    torch.set_num_threads(4)
    result = _build_result(args.warmup, args.repeats)
    _write_tex(result)
    result["generated_tex_sha256"] = sha256_file(OUT_TEX)
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    ok, messages = verify_result()
    print(json.dumps({
        "output_json": str(OUT_JSON),
        "output_tex": str(OUT_TEX),
        "verify_ok": ok,
        "verify_messages": messages,
        "family_latency_ms": {
            f: {device: fam["latency"][device].get("adapter_predict")
                for device in ("cpu", "gpu")}
            for f, fam in result["families"].items()
        },
        "fused_latency_ms": {
            f: {device: v["latency"][device].get("adapter_predict")
                for device in ("cpu", "gpu")}
            for f, v in result["fused_variants"].items()
        },
        "fusion_validation": {
            f: {"cpu": v["validation"], "gpu": v["gpu_validation"]}
            for f, v in result["fused_variants"].items()
        },
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

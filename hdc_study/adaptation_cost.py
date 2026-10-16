#!/usr/bin/env python3
"""Exact parameter and analytic AdamW-state accounting for selected H1 models.

Run from the repository root:
    python hdc_study/adaptation_cost.py

The selected configurations are instantiated with the frozen model definition
on CPU. No fitting, optimizer step, timing, data loading, or checkpoint loading
is performed. Output is deterministic and contains source hashes for audit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
SELECTION = RESULTS / "transfer_selection.json"
OUT_JSON = RESULTS / "adaptation_cost.json"
OUT_TEX = ROOT / "neuroinformatics_submission" / "adaptation_generated.tex"
FAMILIES = ("hdc", "random", "mlp", "plain")
BYTES_PER_FP32 = 4
MIB = 1024 * 1024

try:
    from .model import Config, Decoder
except ImportError:  # Direct invocation: python hdc_study/adaptation_cost.py
    sys.path.insert(0, str(HERE))
    from model import Config, Decoder


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _fp32_bytes(tensor: torch.Tensor, label: str) -> int:
    if tensor.dtype != torch.float32:
        raise TypeError(f"{label} must be FP32 for this accounting, got {tensor.dtype}")
    return int(tensor.numel() * tensor.element_size())


def _parameter_block(name: str) -> str:
    if name.startswith("input_projection."):
        return "input_projection"
    if name.startswith("input_norm."):
        return "input_norm"
    if name.startswith("gru."):
        return "gru"
    if name.startswith("state_norm."):
        return "state_norm"
    if name.startswith("readout."):
        return "readout"
    raise ValueError(f"Unaccounted trainable parameter {name!r}")


def _buffer_block(name: str) -> str:
    if name == "encoder.basis_matrix":
        return "basis_matrix"
    if name == "encoder.anchors":
        return "value_anchors"
    if name.startswith("encoder."):
        return "encoder_other"
    if name.startswith("bundler."):
        return "temporal_roles"
    raise ValueError(f"Unaccounted registered buffer {name!r}")


def _calibration_protocol() -> dict[str, Any]:
    """Read and enforce the active calibration contract from engine.py AST."""
    path = HERE / "engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fn = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "calibrate"),
        None,
    )
    if fn is None:
        raise ValueError("engine.py does not define calibrate")

    defaults: dict[str, Any] = {}
    for arg, default in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults):
        if isinstance(default, ast.Constant):
            defaults[arg.arg] = default.value
    if defaults.get("steps") != 20:
        raise ValueError(f"Expected engine.calibrate default steps=20, got {defaults}")

    adamw_calls = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AdamW"
    ]
    if len(adamw_calls) != 1:
        raise ValueError(f"Expected one AdamW call in calibrate, found {len(adamw_calls)}")
    adamw = adamw_calls[0]
    target = ast.unparse(adamw.args[0]) if adamw.args else ""
    if target != "model.readout.parameters()":
        raise ValueError(f"Expected readout-only AdamW target, got {target!r}")
    kwargs = {
        kw.arg: kw.value.value
        for kw in adamw.keywords
        if kw.arg in ("lr", "weight_decay") and isinstance(kw.value, ast.Constant)
    }
    if kwargs != {"lr": 1e-4, "weight_decay": 1e-4}:
        raise ValueError(f"Unexpected calibration AdamW settings: {kwargs}")

    step_loops = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "steps"
    ]
    if len(step_loops) != 1:
        raise ValueError("Expected one calibration loop over range(steps)")
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "huber_loss"
        for node in ast.walk(fn)
    ):
        raise ValueError("Expected Huber loss in the calibration update")

    has_backward = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
        for node in ast.walk(fn)
    )
    has_step = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "optimizer"
        for node in ast.walk(fn)
    )
    if not (has_backward and has_step):
        raise ValueError("Calibration must perform backward() and optimizer.step()")

    return {
        "source": "hdc_study/engine.py::calibrate",
        "default_steps": defaults["steps"],
        "optimizer": "AdamW(model.readout.parameters())",
        "learning_rate": kwargs["lr"],
        "weight_decay": kwargs["weight_decay"],
        "loss": "Huber loss",
        "updates_only_readout": True,
        "backward_and_optimizer_step_present": True,
    }


def _memory_accounting(parameter_count: int) -> dict[str, int]:
    weight_bytes = parameter_count * BYTES_PER_FP32
    gradient_bytes = parameter_count * BYTES_PER_FP32
    moment_bytes = parameter_count * 2 * BYTES_PER_FP32
    return {
        "parameters": int(parameter_count),
        "fp32_weight_bytes": int(weight_bytes),
        "fp32_gradient_bytes": int(gradient_bytes),
        "fp32_adamw_first_and_second_moment_bytes": int(moment_bytes),
        "parameter_gradient_and_moment_subtotal_bytes": int(
            weight_bytes + gradient_bytes + moment_bytes
        ),
    }


def _collect_family(family: str, selected: dict[str, Any]) -> dict[str, Any]:
    config_record = selected["config"]
    if config_record.get("family") != family:
        raise ValueError(f"Selection family mismatch for {family}")
    config = Config(**config_record)
    model = Decoder(config).cpu()

    parameter_blocks = {
        "input_projection": 0,
        "input_norm": 0,
        "gru": 0,
        "state_norm": 0,
        "readout": 0,
    }
    parameter_tensors = []
    total_parameters = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        _fp32_bytes(parameter, name)
        block = _parameter_block(name)
        count = int(parameter.numel())
        parameter_blocks[block] += count
        total_parameters += count
        parameter_tensors.append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": count,
            "dtype": str(parameter.dtype).removeprefix("torch."),
        })

    if total_parameters != int(selected["parameters"]):
        raise ValueError(
            f"{family}: model.py count {total_parameters} differs from selection count "
            f"{selected['parameters']}"
        )

    buffer_blocks: dict[str, int] = {
        "basis_matrix": 0,
        "value_anchors": 0,
        "temporal_roles": 0,
        "encoder_other": 0,
    }
    buffer_tensors = []
    for name, tensor in model.named_buffers():
        byte_count = _fp32_bytes(tensor, name)
        block = _buffer_block(name)
        buffer_blocks[block] += byte_count
        buffer_tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "bytes": byte_count,
        })

    readout_parameters = parameter_blocks["readout"]
    readout_accounting = _memory_accounting(readout_parameters)
    full_accounting = _memory_accounting(total_parameters)
    static_buffer_bytes = sum(buffer_blocks.values())
    return {
        "selected_key": selected.get("key"),
        "config": config_record,
        "parameter_count_from_selection": int(selected["parameters"]),
        "trainable_parameter_count_from_model": total_parameters,
        "parameter_blocks": parameter_blocks,
        "parameter_tensors": parameter_tensors,
        "readout_only_calibration_accounting": readout_accounting,
        "all_trainable_parameters_hypothetical_accounting": full_accounting,
        "registered_static_buffer_bytes": static_buffer_bytes,
        "registered_static_buffer_mib": static_buffer_bytes / MIB,
        "static_buffer_blocks_bytes": buffer_blocks,
        "static_buffer_tensors": buffer_tensors,
        "temporal_roles_used_by_forward": family in ("hdc", "random"),
        "basis_gradient_enabled": False,
    }


def _format_bytes(value: int) -> str:
    return f"{value:,}"


def _render_tex(families: dict[str, Any], protocol: dict[str, Any]) -> str:
    labels = {"hdc": "HDC", "random": "Random", "mlp": "MLP", "plain": "Plain GRU"}
    methods = (
        "For each family, we instantiated the selected transfer configuration from "
        r"\texttt{transfer\_selection.json} on CPU with the study's model implementation "
        "and counted its trainable tensors and registered FP32 buffers; no fit or optimizer "
        "step was run for this accounting. The implemented calibration protocol uses "
        f"{protocol['default_steps']} AdamW steps over the readout alone for all four families "
        f"(learning rate and weight decay each $10^{{-4}}$, Huber loss on valid prefix labels). "
        "We report exact parameter-derived storage for FP32 weights, gradients, and two "
        "AdamW moments. These byte counts exclude activations and runtime workspaces and "
        "are not measurements of peak training memory, latency, or energy."
    )

    hdc_readout = families["hdc"]["readout_only_calibration_accounting"]
    hdc_full = families["hdc"]["all_trainable_parameters_hypothetical_accounting"]
    readout_moment_scalars = 2 * hdc_readout["parameters"]
    full_moment_scalars = 2 * hdc_full["parameters"]
    moment_ratio = full_moment_scalars / readout_moment_scalars
    results = (
        f"The shared readout-only calibration updates {hdc_readout['parameters']:,} HDC readout "
        f"parameters, corresponding to {readout_moment_scalars:,} FP32 AdamW moment scalars "
        f"({hdc_readout['fp32_adamw_first_and_second_moment_bytes']:,} bytes); this policy "
        "also applies to random features, MLP features, and plain GRU. For selected HDC, "
        f"the hypothetical all-trainable count is {hdc_full['parameters']:,} parameters, or "
        f"{full_moment_scalars:,} moment scalars, {moment_ratio:.2f} times as many. This is "
        "parameter-derived optimizer-state arithmetic, not a measured memory reduction or "
        "an evaluated full-model adaptation baseline."
    )

    discussion = (
        "The fixed HDC basis receives no gradients, while the input projection, "
        "normalization layers, GRU, and readout are gradient-trained during model fitting. "
        "The selected HDC configuration uses one input bin, so its fixed basis and trained "
        "input affine can be fused into the lookup table used by the separate compiled "
        "inference path. Readout-only calibration leaves both fusion inputs unchanged, "
        "so it is compatible with that compiled representation; updating the input affine "
        "would require rebuilding the table. The same readout-only calibration rule is used "
        "for HDC, random features, MLP, and plain GRU, and all families score without weight "
        "updates. The calibration analysis is not an on-device training demonstration: the "
        "submission adapter accepts externally prepared calibration state and performs "
        "inference only."
    )

    row_end = r"\\"
    block_headers = "Family & $D/K$ & Input map & Input norm & GRU & State norm & Readout & Total " + row_end
    block_rows = []
    for family in FAMILIES:
        record = families[family]
        config = record["config"]
        blocks = record["parameter_blocks"]
        dimension_bins = (
            f"--/{config['temporal_bins']}" if family == "plain"
            else f"{config['dimension']}/{config['temporal_bins']}"
        )
        block_rows.append(
            f"{labels[family]} & {dimension_bins} & "
            f"{blocks['input_projection']:,} & {blocks['input_norm']:,} & "
            f"{blocks['gru']:,} & {blocks['state_norm']:,} & {blocks['readout']:,} & "
            f"{record['trainable_parameter_count_from_model']:,} " + row_end
        )
    parameter_table = "\n".join([
        r"\begin{table}[htbp]\centering\scriptsize",
        r"\caption{Trainable parameter blocks from CPU-instantiated selected configurations.}",
        r"\label{tab:adaptation-parameter-blocks}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lrrrrrrr}\toprule",
        block_headers,
        r"\midrule",
        *block_rows,
        r"\bottomrule\end{tabular}\end{table}",
    ])

    memory_rows = []
    for family in FAMILIES:
        record = families[family]
        readout = record["readout_only_calibration_accounting"]
        full = record["all_trainable_parameters_hypothetical_accounting"]
        memory_rows.append(
            f"{labels[family]} & Readout & {readout['parameters']:,} & "
            f"{_format_bytes(readout['fp32_weight_bytes'])} & "
            f"{_format_bytes(readout['fp32_gradient_bytes'])} & "
            f"{_format_bytes(readout['fp32_adamw_first_and_second_moment_bytes'])} & "
            f"{_format_bytes(readout['parameter_gradient_and_moment_subtotal_bytes'])} " + row_end
        )
        memory_rows.append(
            f"{labels[family]} & All trainable & {full['parameters']:,} & "
            f"{_format_bytes(full['fp32_weight_bytes'])} & "
            f"{_format_bytes(full['fp32_gradient_bytes'])} & "
            f"{_format_bytes(full['fp32_adamw_first_and_second_moment_bytes'])} & "
            f"{_format_bytes(full['parameter_gradient_and_moment_subtotal_bytes'])} " + row_end
        )
    memory_table = "\n".join([
        r"\begin{table}[htbp]\centering\scriptsize",
        r"\caption{Parameter and AdamW-state storage arithmetic in bytes. The all-trainable rows are hypothetical accounting, not evaluated adaptation baselines.}",
        r"\label{tab:adaptation-memory}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrrrr}\toprule",
        r"Family & Update scope & $N$ & Weights & Gradients & AdamW $m,v$ & Subtotal \\",
        r"\midrule",
        *memory_rows,
        r"\bottomrule\end{tabular}\end{table}",
    ])

    buffer_rows = []
    for family in FAMILIES:
        record = families[family]
        contents = record["static_buffer_blocks_bytes"]
        if family in ("hdc", "random"):
            description = (
                f"basis {_format_bytes(contents['basis_matrix'])} B; "
                f"anchors {_format_bytes(contents['value_anchors'])} B; "
                f"temporal roles {_format_bytes(contents['temporal_roles'])} B"
            )
        else:
            description = (
                f"registered temporal-role buffers {_format_bytes(contents['temporal_roles'])} B "
                "(not used by the raw-lag forward path)"
            )
        buffer_rows.append(
            f"{labels[family]} & {description} & "
            f"{_format_bytes(record['registered_static_buffer_bytes'])} & "
            f"{record['registered_static_buffer_mib']:.6f} " + row_end
        )
    buffer_table = "\n".join([
        r"\begin{table}[htbp]\centering\scriptsize",
        r"\caption{Registered static FP32 buffers in the instantiated models.}",
        r"\label{tab:adaptation-static-buffers}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lp{0.54\linewidth}rr}\toprule",
        r"Family & Registered buffers & Bytes & MiB \\",
        r"\midrule",
        *buffer_rows,
        r"\bottomrule\end{tabular}\end{table}",
    ])

    supplement = "\n\n".join([
        r"\subsection{Parameter and optimizer-state accounting}",
        r"The readout is a two-layer map from 128 hidden units to 128 hidden units and then "
        r"seven outputs, giving $(128\times128+128)+(128\times7+7)=17{,}415$ parameters "
        "in every family. For $N$ FP32 trainable parameters, the parameter-derived storage "
        r"is $4N$ bytes for weights, $4N$ for gradients, and $8N$ for the first and second "
        "AdamW moments; the listed subtotal is $16N$ bytes. The readout-only calibration "
        "therefore has 69,660 bytes each for weights and gradients, 139,320 bytes for "
        "moments, and a 278,640-byte subtotal. The all-trainable rows show the arithmetic "
        "if every existing trainable tensor were adapted, with the fixed codebook still "
        "excluded from the optimizer; no such full-model calibration condition was run.",
        r"Tables~\ref{tab:adaptation-parameter-blocks}, "
        r"\ref{tab:adaptation-memory}, and "
        r"\ref{tab:adaptation-static-buffers} report, in order, trainable parameter blocks, "
        r"parameter and optimizer-state storage arithmetic, and registered static buffers.",
        parameter_table,
        memory_table,
        buffer_table,
        r"For selected HDC, the fixed basis buffer contains 176 channels $\times$ 8 anchors "
        r"$\times$ 1,024 real/imaginary coordinates, or 1,441,792 FP32 values "
        "(5.500 MiB). Static buffers are not optimizer state. MLP and plain GRU instantiate "
        "temporal-role buffers that their raw-lag forward path does not use. The byte totals "
        "exclude AdamW step scalars, input/target storage, cached hidden features, saved "
        "activations, autograd workspaces, allocator overhead, and all other runtime state. "
        "In particular, full-model backpropagation must retain or recompute intermediate "
        "activations through the learned projection and recurrent layers; these counts do "
        "not estimate that peak memory or its compute cost. The calibration procedure is "
        "an offline evaluation analysis, and no on-device training feasibility or energy "
        "measurement was made.",
    ])

    return "\n".join([
        r"% Generated by hdc_study/adaptation_cost.py; do not edit by hand.",
        r"\providecommand{\AdaptationMethods}{%",
        methods + r"\par}",
        "",
        r"\providecommand{\AdaptationResults}{%",
        results + r"\par}",
        "",
        r"\providecommand{\AdaptationDiscussion}{%",
        discussion + r"\par}",
        "",
        r"\providecommand{\AdaptationSupplement}{%",
        supplement + r"\par}",
        "",
    ])


def _source_hashes(tex_bytes: bytes) -> dict[str, str]:
    return {
        "adaptation_cost.py": sha256_file(Path(__file__).resolve()),
        "model.py": sha256_file(HERE / "model.py"),
        "engine.py": sha256_file(HERE / "engine.py"),
        "adapter.py": sha256_file(HERE / "adapter.py"),
        "efficiency.py": sha256_file(HERE / "efficiency.py"),
        "transfer_selection.json": sha256_file(SELECTION),
        "adaptation_generated.tex": sha256_bytes(tex_bytes),
    }


def build_result() -> tuple[dict[str, Any], str]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    families = {
        family: _collect_family(family, selection[family])
        for family in FAMILIES
    }
    calibration_protocol = _calibration_protocol()
    tex = _render_tex(families, calibration_protocol)
    tex_bytes = tex.encode("utf-8")
    result = {
        "schema": 1,
        "purpose": (
            "CPU-instantiated exact parameter and static-buffer accounting plus analytic "
            "FP32 AdamW-state arithmetic; no fitting, timing, or full-model adaptation run."
        ),
        "selection_source": "hdc_study/results/transfer_selection.json",
        "model_source": "hdc_study/model.py",
        "calibration_source": calibration_protocol,
        "accounting_assumptions": {
            "parameter_dtype": "float32",
            "gradient_dtype": "float32",
            "adamw_moment_dtypes": ["float32", "float32"],
            "bytes_per_fp32_value": BYTES_PER_FP32,
            "mib_bytes": MIB,
            "adamw_moments_per_parameter": 2,
            "step_scalars_included": False,
            "activation_or_workspace_bytes_included": False,
            "readout_only_policy_applies_to_all_families": True,
            "all_trainable_adaptation_is_hypothetical": True,
            "fixed_codebooks_are_trainable": False,
        },
        "families": families,
        "compilation_compatibility": {
            "selected_hdc_temporal_bins": int(families["hdc"]["config"]["temporal_bins"]),
            "fusion_dependencies": [
                "fixed HDC basis and K=1 temporal role",
                "trained input projection weight and bias",
            ],
            "readout_only_update_changes_fusion_dependencies": False,
            "readout_only_update_preserves_compiled_lookup": True,
            "projection_update_requires_rebuild": True,
            "compiled_calibration_runtime_evaluated": False,
        },
        "inference_and_deployment_scope": {
            "scored_inference_weight_updates": "none for all four families",
            "zero_label_condition_available_for_all_families": True,
            "submission_adapter_performs_online_weight_updates": False,
            "on_device_training_demonstrated": False,
            "latency_or_energy_measurements_in_this_artifact": False,
        },
        "source_hashes": _source_hashes(tex_bytes),
    }
    return result, tex


def verify_result() -> tuple[bool, list[str]]:
    """Verify deterministic outputs, model-derived counts, and source hashes."""
    issues: list[str] = []
    try:
        expected, expected_tex = build_result()
    except Exception as exc:  # Report stale/incompatible source rather than raising.
        return False, [f"Could not rebuild adaptation accounting: {type(exc).__name__}: {exc}"]

    if not OUT_JSON.is_file():
        issues.append(f"Missing generated result: {OUT_JSON.relative_to(ROOT)}")
    else:
        try:
            observed = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            if observed != expected:
                if observed.get("source_hashes") != expected["source_hashes"]:
                    issues.append("adaptation_cost.json source hashes are stale")
                if observed.get("families") != expected["families"]:
                    issues.append("adaptation_cost.json model-derived counts are stale")
                if observed != expected:
                    issues.append("adaptation_cost.json differs from deterministic rebuild")
        except Exception as exc:
            issues.append(f"Could not parse adaptation_cost.json: {type(exc).__name__}: {exc}")

    if not OUT_TEX.is_file():
        issues.append(f"Missing generated TeX: {OUT_TEX.relative_to(ROOT)}")
    elif OUT_TEX.read_text(encoding="utf-8") != expected_tex:
        issues.append("adaptation_generated.tex differs from deterministic rebuild")

    return not issues, issues


def _write_outputs() -> None:
    result, tex = build_result()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUT_TEX.write_text(tex, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true",
        help="verify existing JSON/TeX outputs and source hashes without writing",
    )
    args = parser.parse_args(argv)
    if args.verify:
        ok, issues = verify_result()
        if ok:
            print("PASS: adaptation-cost outputs, model counts, and source hashes match")
            return 0
        for issue in issues:
            print(f"FAIL: {issue}", file=sys.stderr)
        return 1

    _write_outputs()
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

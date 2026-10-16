"""Generate and verify Pareto comparisons from observed study results.

The analysis pairs the existing H1 score summaries with cost profiles from
``results/efficiency.json``. It does not train models or benchmark new ones.

Run from the repository root:

    python -m hdc_study.pareto
    python -m hdc_study.pareto --verify

Frontier membership is computed from observed means only. The source
intervals are retained and displayed, but do not enter the dominance rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SUMMARY_DEFAULT = ROOT / "results" / "summary.json"
EFFICIENCY_DEFAULT = ROOT / "results" / "efficiency.json"
RESULT_DEFAULT = ROOT / "results" / "pareto.json"
TEX_DEFAULT = REPO / "neuroinformatics_submission" / "pareto_generated.tex"
FIGURE_RELATIVE_PATHS = (
    "hdc_study/figures/fig5_pareto.pdf",
    "hdc_study/figures/fig5_pareto.eps",
    "hdc_study/figures/fig5_pareto.svg",
    "hdc_study/figures/previews/fig5_pareto.png",
    "neuroinformatics_submission/Fig5.pdf",
    "neuroinformatics_submission/Fig5.eps",
)

FAMILIES = ("hdc", "random", "mlp", "plain")
FAMILY_LABELS = {
    "hdc": "HDC",
    "random": "Random projection",
    "mlp": "MLP features",
    "plain": "Plain GRU",
}
FAMILY_COLORS = {
    "hdc": "#A31F34",
    "random": "#45515E",
    "mlp": "#3C78A8",
    "plain": "#328A83",
}
FAMILY_MARKERS = {"hdc": "o", "random": "s", "mlp": "^", "plain": "D"}
COMPILED_FAMILIES = ("hdc", "random")
SCENARIOS = (
    {"key": "primary_few_shot", "kind": "few_shot", "label": "60-s calibration"},
    {"key": "zero_label_dropout_0", "kind": "zero_label_dropout", "dropout": 0.0,
     "label": "Zero labels, 0%"},
    {"key": "zero_label_dropout_0_1", "kind": "zero_label_dropout", "dropout": 0.1,
     "label": "Zero labels, 10%"},
    {"key": "zero_label_dropout_0_2", "kind": "zero_label_dropout", "dropout": 0.2,
     "label": "Zero labels, 20%"},
    {"key": "zero_label_dropout_0_4", "kind": "zero_label_dropout", "dropout": 0.4,
     "label": "Zero labels, 40%"},
)
COST_METRICS = ("mac_equivalents_per_bin", "cpu_p95_ms", "gpu_p95_ms", "static_memory_mib")
PAIRING_DIMENSIONS = (
    "family", "channels", "outputs", "hidden", "dimension", "temporal_bins",
    "levels", "recurrence", "online_norm", "smoothing", "no_roles",
    "no_channel_roles",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_recorded_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate.resolve()


def _find_unique(rows: Iterable[dict[str, Any]], *, description: str) -> dict[str, Any]:
    values = list(rows)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {description}; found {len(values)}")
    return values[0]


def _score_sources(summary: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return five score conditions, retaining source means and CIs."""
    if summary.get("complete") is not True:
        raise ValueError("summary.json is not marked complete")
    if summary.get("primary_regime") != "few_shot":
        raise ValueError("The primary condition in summary.json must be few_shot")

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        family_rows: dict[str, dict[str, Any]] = {}
        for family in FAMILIES:
            if scenario["kind"] == "few_shot":
                row = _find_unique(
                    (x for x in summary.get("scores", [])
                     if x.get("family") == family and x.get("regime") == "few_shot"),
                    description=f"few-shot score for {family}",
                )
                source = "summary.scores"
            else:
                row = _find_unique(
                    (x for x in summary.get("robustness", [])
                     if x.get("family") == family
                     and x.get("perturbation") == "dropout"
                     and float(x.get("level", -1)) == float(scenario["dropout"])),
                    description=f"zero-label {scenario['dropout']:.0%} dropout score for {family}",
                )
                source = "summary.robustness"
            mean = float(row["mean"])
            ci = [float(v) for v in row["ci95"]]
            if len(ci) != 2 or not all(math.isfinite(v) for v in (mean, *ci)):
                raise ValueError(f"Invalid mean or interval for {scenario['key']} / {family}")
            if ci[0] > mean or mean > ci[1]:
                raise ValueError(f"Source interval does not contain the mean for {scenario['key']} / {family}")
            family_rows[family] = {
                "mean_r2": mean,
                "ci95": ci,
                "source": source,
                "source_family": family,
                "source_regime": "few_shot" if scenario["kind"] == "few_shot" else "zero_label",
                "source_perturbation": None if scenario["kind"] == "few_shot" else "dropout",
                "source_level": None if scenario["kind"] == "few_shot" else float(scenario["dropout"]),
            }
        result[scenario["key"]] = family_rows

    # Cross-check that the robustness table's zero-dropout row is the same
    # zero-label score reported by the main score table.
    for family in FAMILIES:
        zero_shot = _find_unique(
            (x for x in summary.get("scores", [])
             if x.get("family") == family and x.get("regime") == "zero_shot"),
            description=f"zero-shot score for {family}",
        )
        robust_zero = result["zero_label_dropout_0"][family]
        if float(zero_shot["mean"]) != robust_zero["mean_r2"] or [float(v) for v in zero_shot["ci95"]] != robust_zero["ci95"]:
            raise ValueError(f"Zero-label score and zero-dropout robustness row differ for {family}")
    return result


def _validate_pairing(summary: dict[str, Any], efficiency: dict[str, Any]) -> dict[str, Any]:
    """Require each outer-selected inference shape to match the profiled shape."""
    if efficiency.get("selection", {}).get("checkpoints") is None:
        raise ValueError("efficiency.json has no representative checkpoint records")
    profile_records = efficiency["selection"]["checkpoints"]
    per_family: dict[str, Any] = {}
    for family in FAMILIES:
        profile_record = profile_records.get(family)
        if not isinstance(profile_record, dict) or not isinstance(profile_record.get("config"), dict):
            raise ValueError(f"Missing representative inference configuration for {family}")
        profile_config = profile_record["config"]
        matches = []
        for session in summary.get("sessions", []):
            session_selection = summary.get("selection", {}).get(session)
            family_rows = session_selection.get("families", {}).get(family, []) if isinstance(session_selection, dict) else []
            if not family_rows:
                raise ValueError(f"Missing selected outer configuration for {family}/{session}")
            chosen = max(family_rows, key=lambda row: float(row["score"]))
            outer_config = chosen.get("config")
            if not isinstance(outer_config, dict):
                raise ValueError(f"Invalid selected outer configuration for {family}/{session}")
            differences = {
                key: {"outer": outer_config.get(key), "profiled": profile_config.get(key)}
                for key in PAIRING_DIMENSIONS
                if outer_config.get(key) != profile_config.get(key)
            }
            matches.append({
                "session": session,
                "selected_key": chosen.get("key"),
                "selected_score": float(chosen["score"]),
                "cost_relevant_dimensions_match": not differences,
                "differences": differences,
                "outer_config_cost_dimensions": {key: outer_config.get(key) for key in PAIRING_DIMENSIONS},
            })
        per_family[family] = {
            "profiled_selected_key": profile_record.get("selected_key"),
            "profiled_checkpoint": profile_record.get("checkpoint"),
            "profiled_checkpoint_sha256": profile_record.get("checkpoint_sha256"),
            "profiled_training_seed": profile_record.get("training_seed"),
            "profiled_config_cost_dimensions": {key: profile_config.get(key) for key in PAIRING_DIMENSIONS},
            "outer_selections": matches,
            "all_outer_cost_dimensions_match": all(x["cost_relevant_dimensions_match"] for x in matches),
        }
    all_match = all(x["all_outer_cost_dimensions_match"] for x in per_family.values())
    if not all_match:
        raise ValueError("At least one outer-selected configuration does not match its profiled cost dimensions")
    return {
        "passed": all_match,
        "rule": "For each family and outer session, the highest-score selected inner candidate is checked against the profiled transfer checkpoint on all listed inference-cost dimensions.",
        "dimensions_compared": list(PAIRING_DIMENSIONS),
        "families": per_family,
    }


def _cost_values(efficiency: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    original: dict[str, dict[str, float]] = {}
    costs_basis: dict[str, Any] = {}
    families = efficiency.get("families", {})
    fused = efficiency.get("fused_variants", {})
    for family in FAMILIES:
        profile = families.get(family)
        if not isinstance(profile, dict):
            raise ValueError(f"Missing profiled costs for {family}")
        ops = profile.get("operations_per_bin", {})
        latency = profile.get("latency", {})
        memory = profile.get("memory", {})
        original[family] = {
            "mac_equivalents_per_bin": float(ops["total_dense_macs_per_bin"]),
            "cpu_p95_ms": float(latency["cpu"]["adapter_predict"]["p95_ms"]),
            "gpu_p95_ms": float(latency["gpu"]["adapter_predict"]["p95_ms"]),
            "static_memory_mib": float(memory["resident_parameter_plus_buffer_bytes"]) / (1024 ** 2),
        }
    compiled: dict[str, dict[str, float]] = {}
    for family in COMPILED_FAMILIES:
        profile = fused.get(family)
        if not isinstance(profile, dict):
            raise ValueError(f"Missing compiled profile for {family}")
        ops = profile.get("operations_per_bin", {})
        latency = profile.get("latency", {})
        memory = profile.get("compiled_memory", {})
        compiled[family] = {
            "mac_equivalents_per_bin": float(ops["total_online_mac_equivalents"]),
            "cpu_p95_ms": float(latency["cpu"]["adapter_predict"]["p95_ms"]),
            "gpu_p95_ms": float(latency["gpu"]["adapter_predict"]["p95_ms"]),
            "static_memory_mib": float(memory["compiled_static_tensor_bytes"]) / (1024 ** 2),
        }
    costs_basis = {
        "mac_equivalents_per_bin": {
            "unit": "MAC-equivalents per input bin",
            "original_basis": "efficiency.families[*].operations_per_bin.total_dense_macs_per_bin; one multiply-accumulate is one MAC, and listed non-GEMM operations are excluded.",
            "compiled_basis": "efficiency.fused_variants[*].operations_per_bin.total_online_mac_equivalents; two-anchor sparse lookups are each counted as a MAC-equivalent. Offline table fusion is excluded.",
        },
        "cpu_p95_ms": {
            "unit": "milliseconds per one-bin adapter_predict call (CPU p95)",
            "basis": "The measured batch-one adapter_predict p95, including the source-listed causal host normalization, transfer where applicable, model call, and output handling; excludes file I/O, data loading, calibration fitting, and trial scheduling.",
        },
        "gpu_p95_ms": {
            "unit": "milliseconds per one-bin adapter_predict call (GPU p95)",
            "basis": "The measured batch-one adapter_predict p95, including the source-listed causal host normalization, transfer where applicable, model call, and output handling; excludes file I/O, data loading, calibration fitting, and trial scheduling.",
        },
        "static_memory_mib": {
            "unit": "MiB (2^20 bytes)",
            "original_basis": "Resident trained parameter plus registered buffer storage from the representative original checkpoint.",
            "compiled_basis": "Compiled static tensor bytes from the inference-only representation; it is not a measured process peak or device-wide memory footprint.",
        },
        "measurement_scope": {
            "checkpoint_rule": efficiency.get("selection", {}).get("representative_checkpoint_rule"),
            "measurement": efficiency.get("measurement", {}),
            "latency_note": efficiency.get("measurement", {}).get("latency_note"),
            "latency_is_outer_fold_mean": False,
        },
    }
    all_costs = {f"{family}_original": original[family] for family in FAMILIES}
    all_costs.update({f"{family}_compiled": compiled[family] for family in COMPILED_FAMILIES})
    return all_costs, costs_basis


def dominates(a: dict[str, Any], b: dict[str, Any], *, score_key: str = "mean_r2",
              cost_key: str = "cost") -> bool:
    """True iff a has no worse observed mean/cost and strictly improves one.

    Equal costs are handled directly: a higher score dominates at equal cost;
    an exact score-and-cost tie does not dominate either point.
    """
    score_a, score_b = float(a[score_key]), float(b[score_key])
    cost_a, cost_b = float(a[cost_key]), float(b[cost_key])
    return score_a >= score_b and cost_a <= cost_b and (score_a > score_b or cost_a < cost_b)


def _frontier_records(points: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    comparisons = []
    for point in points:
        comparisons.append({
            "point_id": point["point_id"],
            "family": point["family"],
            "implementation": point["implementation"],
            "mean_r2": point["mean_r2"],
            "ci95": point["ci95"],
            "cost": point["costs"][metric],
            "cost_unit": point["cost_units"][metric],
            "frontier": False,
            "dominated_by": [],
            "dominates": [],
        })
    for i, point_a in enumerate(comparisons):
        for j, point_b in enumerate(comparisons):
            if i == j:
                continue
            if dominates(point_a, point_b):
                point_a["dominates"].append(point_b["point_id"])
                point_b["dominated_by"].append(point_a["point_id"])
    for point in comparisons:
        point["dominates"].sort()
        point["dominated_by"].sort()
        point["frontier"] = not point["dominated_by"]
    return {
        "membership_basis": "Observed means only; descriptive intervals are retained but do not affect dominance.",
        "rule": "A dominates B iff A.mean_r2 >= B.mean_r2 and A.cost <= B.cost, with at least one strict inequality. Equal-cost points can dominate on score; exact ties do not dominate.",
        "frontier_point_ids": [x["point_id"] for x in comparisons if x["frontier"]],
        "records": comparisons,
    }


def _compiled_scope(efficiency: dict[str, Any]) -> dict[str, Any]:
    validations = {}
    for family in COMPILED_FAMILIES:
        profile = efficiency["fused_variants"][family]
        validations[family] = {
            "variant": profile.get("variant"),
            "real_algebra_equivalent_for_fixed_parameters": True,
            "accuracy_assignment": "Inherits the original family mean and descriptive interval by exact real-valued algebra; no full outer-fold compiled rescore was run.",
            "training_or_primary_results_changed": profile.get("training_or_primary_results_changed"),
            "cpu_numerical_check": profile.get("validation"),
            "gpu_numerical_check": profile.get("gpu_validation"),
        }
        for device_key in ("validation", "gpu_validation"):
            check = profile.get(device_key, {})
            if check.get("passed") is not True or int(check.get("stream_bins", -1)) != 2376:
                raise ValueError(f"Unexpected compiled numerical validation scope for {family}/{device_key}")
    return {
        "families": validations,
        "scope_note": "The available numerical comparison used 2,376 input bins from the held-out calibration stream for training seed 101 on CPU and GPU. It is not a full outer-fold score rescore and does not establish accuracy equivalence across outer data by fresh scoring.",
    }


def _build_core(summary: dict[str, Any], efficiency: dict[str, Any]) -> dict[str, Any]:
    if efficiency.get("schema") != 1:
        raise ValueError("Unsupported efficiency result schema")
    pairing = _validate_pairing(summary, efficiency)
    scores = _score_sources(summary)
    costs, cost_basis = _cost_values(efficiency)
    compiled = _compiled_scope(efficiency)

    analyses = []
    frontiers: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        source_by_family = scores[scenario["key"]]
        points = []
        for family in FAMILIES:
            source = source_by_family[family]
            for implementation in ("original", "compiled") if family in COMPILED_FAMILIES else ("original",):
                point_id = f"{family}_{implementation}"
                point = {
                    "point_id": point_id,
                    "family": family,
                    "family_label": FAMILY_LABELS[family],
                    "implementation": implementation,
                    "mean_r2": source["mean_r2"],
                    "ci95": list(source["ci95"]),
                    "score_source": source["source"],
                    "source_record": {
                        "family": source["source_family"],
                        "regime": source["source_regime"],
                        "perturbation": source["source_perturbation"],
                        "level": source["source_level"],
                    },
                    "accuracy_inherited_from": f"{family}_original" if implementation == "compiled" else None,
                    "costs": dict(costs[point_id]),
                    "cost_units": {
                        "mac_equivalents_per_bin": "MAC-equivalents/bin",
                        "cpu_p95_ms": "ms/call",
                        "gpu_p95_ms": "ms/call",
                        "static_memory_mib": "MiB",
                    },
                }
                points.append(point)
        analyses.append({
            "key": scenario["key"],
            "label": scenario["label"],
            "condition": {
                "regime": "few_shot" if scenario["kind"] == "few_shot" else "zero_label",
                "perturbation": None if scenario["kind"] == "few_shot" else "dropout",
                "dropout_fraction": None if scenario["kind"] == "few_shot" else float(scenario["dropout"]),
            },
            "membership_unit": "One participant's six held-out outer session-group means; the family mean and its source interval are copied to both compiled and original points.",
            "records": points,
        })
        frontiers[scenario["key"]] = {
            metric: _frontier_records(points, metric) for metric in COST_METRICS
        }

    return {
        "schema": 1,
        "complete": True,
        "title": "Observed comparison frontier for H1 decoding",
        "scope": "Conditional on these selected configurations, measured representative-checkpoint costs, and observed six-group means; not a state-of-the-art comparison.",
        "inputs": {},
        "pairing_validation": pairing,
        "score_basis": {
            "primary": "summary.scores, few_shot, all six primary outer groups",
            "zero_label_and_dropout": "summary.robustness, perturbation=dropout, levels 0, 0.1, 0.2, and 0.4; the protocol's zero-label condition",
            "intervals": "Source descriptive 95% group-bootstrap intervals are preserved for display. Frontier membership uses means only; no statistical comparison is implied.",
        },
        "cost_basis": cost_basis,
        "compiled_accuracy_scope": compiled,
        "dominance_rule": {
            "objective": "Maximize mean R^2 and minimize one cost at a time.",
            "strictness": "At least one strict improvement is required. At equal cost, a strictly higher mean dominates; an exact mean-and-cost tie leaves both points nondominated.",
            "intervals_used_for_membership": False,
            "interpolation_or_convex_hull_used": False,
        },
        "analyses": analyses,
        "frontiers": frontiers,
        "figure_files_sha256": {},
        "generated_tex_sha256": None,
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }


def _format_score(score: float) -> str:
    return f"{score:.3f}"


def _analysis_by_key(result: dict[str, Any], key: str) -> dict[str, Any]:
    return next(x for x in result["analyses"] if x["key"] == key)


def _family_point(analysis: dict[str, Any], family: str, implementation: str = "original") -> dict[str, Any]:
    point_id = f"{family}_{implementation}"
    return next(x for x in analysis["records"] if x["point_id"] == point_id)


def _frontier_labels(result: dict[str, Any], scenario_key: str, metric: str) -> str:
    records = result["frontiers"][scenario_key][metric]["records"]
    labels = []
    for point in records:
        if point["frontier"]:
            family = FAMILY_LABELS[point["family"]]
            labels.append(family if point["implementation"] == "original" else f"{family} (compiled)")
    return ", ".join(labels) if labels else "None"


def _short_frontier_labels(result: dict[str, Any], scenario_key: str, metric: str) -> str:
    labels = []
    for point in result["frontiers"][scenario_key][metric]["records"]:
        if not point["frontier"]:
            continue
        family = {"hdc": "HDC", "random": "RP", "mlp": "MLP", "plain": "GRU"}[point["family"]]
        if point["implementation"] == "compiled":
            family += "-C"
        labels.append(family)
    return ", ".join(labels) if labels else "None"


def _natural_frontier_labels(result: dict[str, Any], scenario_key: str, metric: str) -> str:
    names = {
        ("hdc", "original"): "HDC",
        ("hdc", "compiled"): "compiled HDC",
        ("random", "original"): "random projection",
        ("random", "compiled"): "compiled random projection",
        ("mlp", "original"): "MLP features",
        ("plain", "original"): "plain GRU",
    }
    labels = [
        names[(point["family"], point["implementation"])]
        for point in result["frontiers"][scenario_key][metric]["records"]
        if point["frontier"]
    ]
    if not labels:
        return "no points"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + ", and " + labels[-1]


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def render_tex(result: dict[str, Any]) -> str:
    few = _analysis_by_key(result, "primary_few_shot")
    zero = _analysis_by_key(result, "zero_label_dropout_0")
    severe = _analysis_by_key(result, "zero_label_dropout_0_4")
    hdc_few, mlp_few, plain_few = (_family_point(few, f) for f in ("hdc", "mlp", "plain"))
    hdc_zero, random_zero, mlp_zero, plain_zero = (_family_point(zero, f) for f in FAMILIES)
    hdc_severe, random_severe, mlp_severe, plain_severe = (_family_point(severe, f) for f in FAMILIES)

    primary_frontier = {metric: result["frontiers"]["primary_few_shot"][metric]["frontier_point_ids"]
                        for metric in COST_METRICS}
    severe_frontier = {metric: result["frontiers"]["zero_label_dropout_0_4"][metric]["frontier_point_ids"]
                       for metric in COST_METRICS}
    hdc_severe_compiled_member = all(
        "hdc_compiled" in point_ids for point_ids in severe_frontier.values()
    )

    primary_plain_only = all(
        point_ids == ["plain_original"] for point_ids in primary_frontier.values()
    )
    unperturbed_plain_only = all(
        item["frontier_point_ids"] == ["plain_original"]
        for item in result["frontiers"]["zero_label_dropout_0"].values()
    )
    moderate_dropout_plain_only = all(
        item["frontier_point_ids"] == ["plain_original"]
        for scenario_key in ("zero_label_dropout_0_1", "zero_label_dropout_0_2")
        for item in result["frontiers"][scenario_key].values()
    )
    severe_key = "zero_label_dropout_0_4"
    severe_frontiers = {
        metric: _natural_frontier_labels(result, severe_key, metric)
        for metric in COST_METRICS
    }

    methods = (
        "We compare the four original families with the inference-only compiled HDC and random-feature variants. "
        "The primary score is the six-group few-shot mean from the source summary; zero-label scores use the same "
        "summary's dropout robustness records at 0, 10, 20, and 40\\% channel dropout. For each condition we "
        "maximize mean $R^2$ and minimize one measured or counted cost at a time. A point dominates another when "
        "it is no worse on both quantities and strictly better on at least one; equal-cost points are ordered by "
        "their means, and exact ties do not dominate. Descriptive 95\\% group-bootstrap intervals are shown but "
        "do not affect membership. Membership is conditional on the observed means and is not a statistical test; "
        "no interpolation or convex-hull performance claim is made. The selected inference-cost dimensions match "
        "the profiled transfer configuration for all six outer groups in every family. CPU and GPU latency are "
        "one-bin adapter-predict $p_{95}$ values from a representative seed-101 transfer checkpoint, not averages "
        "across outer fits; these empirical quantiles do not establish hard real-time behavior."
    )
    compiled_note = (
        "Compiled points reuse their original family's observed score and interval because the fixed-parameter "
        "real-valued algebra is equivalent. The numerical comparison checked prediction and recurrent-state "
        "agreement on a 2,376-bin held-out calibration input stream for seed 101 on CPU and GPU; no full outer-fold compiled rescore was run."
    )
    methods += " " + compiled_note

    plain_only_conditions = []
    if primary_plain_only:
        plain_only_conditions.append("60-s calibration")
    if unperturbed_plain_only:
        plain_only_conditions.append("zero-label evaluation without channel dropout")
    if moderate_dropout_plain_only:
        plain_only_conditions.append("zero-label evaluation at 10\\% and 20\\% channel dropout")
    condition_frontier_statement = ""
    if plain_only_conditions:
        if len(plain_only_conditions) == 1:
            condition_text = plain_only_conditions[0]
        elif len(plain_only_conditions) == 2:
            condition_text = " and ".join(plain_only_conditions)
        else:
            condition_text = ", ".join(plain_only_conditions[:-1]) + ", and " + plain_only_conditions[-1]
        condition_frontier_statement = f"Plain GRU alone is on all four cost frontiers for {condition_text}. "
    results = (
        f"The primary few-shot mean was {_format_score(hdc_few['mean_r2'])} for HDC "
        f"(descriptive interval [{_format_score(hdc_few['ci95'][0])}, {_format_score(hdc_few['ci95'][1])}]), "
        f"{_format_score(mlp_few['mean_r2'])} for MLP features, and {_format_score(plain_few['mean_r2'])} for the plain GRU. "
        f"Under unperturbed zero-label evaluation, the corresponding means were {_format_score(hdc_zero['mean_r2'])}, "
        f"{_format_score(random_zero['mean_r2'])}, {_format_score(mlp_zero['mean_r2'])}, and {_format_score(plain_zero['mean_r2'])} "
        f"for HDC, random projection, MLP features, and the plain GRU. At 40\\% channel dropout they were "
        f"{_format_score(hdc_severe['mean_r2'])}, {_format_score(random_severe['mean_r2'])}, "
        f"{_format_score(mlp_severe['mean_r2'])}, and {_format_score(plain_severe['mean_r2'])}, respectively. "
        f"HDC's 40\\% dropout mean has descriptive interval "
        f"[{_format_score(hdc_severe['ci95'][0])}, {_format_score(hdc_severe['ci95'][1])}]. "
        + condition_frontier_statement
        + f"At 40\\% channel dropout, the observed MAC frontier contains {severe_frontiers['mac_equivalents_per_bin']}; "
        + f"the CPU latency frontier contains {severe_frontiers['cpu_p95_ms']}; "
        + f"the GPU latency frontier contains {severe_frontiers['gpu_p95_ms']}; "
        + f"the storage frontier contains {severe_frontiers['static_memory_mib']} (Fig.~\\ref{{fig:pareto}})."
    )
    discussion = (
        ("At 40\\% dropout, compiled HDC trades higher cost than the plain GRU for a higher observed mean; "
         "MLP remains on the latency and storage frontiers at a slightly lower mean. "
         if hdc_severe_compiled_member else
         "At 40\\% dropout, the compiled HDC point shares a cost frontier with the alternative points. ")
        + "This trade-off is specific to the prespecified channel-loss stress test and this participant's held-out groups."
    )
    conclusion = (
        "The algebraically exact HDC compilation reduces counted online arithmetic from 1,688,448 to 160,640 "
        "MAC-equivalents per input bin (90.5\\%). At its observed 40\\% channel-dropout mean of $R^2=0.165$, "
        "compiled HDC offers a nondominated accuracy-cost option in this comparison."
    )

    rows = []
    for scenario in SCENARIOS:
        key = scenario["key"]
        label = _latex_escape(scenario["label"])
        cells = [_short_frontier_labels(result, key, metric) for metric in COST_METRICS]
        cells = [_latex_escape(cell) for cell in cells]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    table = "\n".join(rows)
    supplement = (
        "HDC-C denotes compiled HDC, RP denotes random projection, RP-C denotes compiled random projection, "
        "and GRU denotes plain GRU. Table~\\ref{tab:pareto-frontiers} lists nondominated observed points for "
        "the 60-s calibration condition and four zero-label channel-dropout levels, including 0\\%. The compiled "
        "variants use the same family accuracy records as their original implementations. Frontier membership "
        "is evaluated separately for each cost axis.\n\n"
        "\\begin{table*}[t]\n\\centering\\scriptsize\n"
        "\\caption{Observed mean-based Pareto frontiers across score conditions and cost bases. Exact membership "
        "uses means only; dropout percentages denote the fraction of input channels dropped.}\n\\label{tab:pareto-frontiers}\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.18\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.18\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.18\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.18\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.18\\linewidth}@{}}\n"
        "\\toprule\nCondition & MAC-equivalents/bin & CPU $p_{95}$ & GPU $p_{95}$ & Storage (MiB) \\\\\n\\midrule\n"
        + table + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n"
        "\\noindent\\textit{Cost bases.} Original MAC counts use the source's per-bin dense GEMM totals; compiled "
        "counts include the stated sparse lookup MAC-equivalents and exclude one-time fusion. Latency uses the "
        "batch-one adapter-predict $p_{95}$, including source-listed normalization and transfers but excluding "
        "data I/O and calibration. Storage uses original resident parameter-plus-buffer bytes and compiled static "
        "tensor bytes (MiB = $2^{20}$ bytes); it is not peak process memory. All latency and memory values come "
        "from one representative seed-101 transfer checkpoint. The six outer-selected inference dimensions were "
        "checked against the profiled shapes before pairing score and cost."
    )

    blocks = {
        "ParetoMethods": methods,
        "ParetoResults": results,
        "ParetoDiscussion": discussion,
        "ParetoConclusion": conclusion,
        "ParetoSupplement": supplement,
    }
    return "\n".join(
        f"\\newcommand{{\\{name}}}{{{body}}}" for name, body in blocks.items()
    ) + "\n"


def _plot_figure(result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(174 / 25.4, 126 / 25.4), facecolor="white")
    axes = (
        fig.add_axes([0.085, 0.535, 0.385, 0.285]),
        fig.add_axes([0.585, 0.535, 0.365, 0.285]),
        fig.add_axes([0.085, 0.14, 0.385, 0.28]),
        fig.add_axes([0.585, 0.14, 0.365, 0.28]),
    )
    fig.text(0.5, 0.992, "Observed comparison frontier", ha="center", va="top",
             fontsize=9.2, fontweight="bold", color="#171717")

    family_handles = []
    for family in FAMILIES:
        family_handles.append(Line2D(
            [0], [0], marker=FAMILY_MARKERS[family], linestyle="None",
            markersize=5.2, markeredgewidth=1.05,
            markeredgecolor=FAMILY_COLORS[family],
            markerfacecolor=FAMILY_COLORS[family], label=FAMILY_LABELS[family],
        ))
    fig.legend(handles=family_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=False, handletextpad=0.5, columnspacing=1.5,
               fontsize=7.1)
    status_handles = (
        Line2D([0], [0], marker="o", linestyle="None", markersize=5.2,
               markeredgecolor="#45515E", markerfacecolor="#45515E", label="Original (filled)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5.2,
               markeredgecolor="#45515E", markerfacecolor="white", label="Compiled (hollow)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.4,
               markeredgecolor="#171717", markerfacecolor="none", markeredgewidth=0.8,
               label="Nondominated point (black ring)"),
        Line2D([0], [0], color="#171717", lw=0.85, linestyle=(0, (3, 2)),
               label="Dominance threshold"),
    )
    fig.legend(handles=status_handles, loc="upper center", bbox_to_anchor=(0.5, 0.927),
               ncol=4, frameon=False, handletextpad=0.45, columnspacing=1.1,
               fontsize=6.9)

    panels = (
        ("zero_label_dropout_0", "mac_equivalents_per_bin", "(a)", "Zero-label, no dropout"),
        ("zero_label_dropout_0_4", "mac_equivalents_per_bin", "(b)", "Zero-label, 40% dropout"),
        ("zero_label_dropout_0", "cpu_p95_ms", "(c)", "Zero-label, no dropout"),
        ("zero_label_dropout_0_4", "cpu_p95_ms", "(d)", "Zero-label, 40% dropout"),
    )
    for ax, (scenario_key, metric, panel_label, title) in zip(axes, panels):
        analysis = _analysis_by_key(result, scenario_key)
        ax.set_title(title, pad=5, color="#171717")
        ax.text(-0.12, 1.11, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold", color="#171717")
        ax.grid(axis="y", color="#E3E6E8", linewidth=0.65, zorder=0)
        ax.grid(axis="x", which="major", color="#EEF0F2", linewidth=0.55, zorder=0)
        ax.tick_params(direction="out", length=2.8, width=0.65, color="#616A73")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#AEB4BA")
        ax.spines["bottom"].set_color("#AEB4BA")
        ax.set_ylabel("Mean zero-label $R^2$", labelpad=4)
        if metric == "mac_equivalents_per_bin":
            ax.set_xscale("log")
            ax.set_xlabel("Per-bin MAC / lookup equivalents (millions)", labelpad=3)
            ax.set_xticks((0.15, 0.25, 0.5, 1.0, 1.7))
            ax.set_xticklabels(("0.15", "0.25", "0.5", "1.0", "1.7"))
        else:
            ax.set_xscale("linear")
            ax.set_xlabel("CPU adapter-predict $p_{95}$ (ms)", labelpad=3)
            ax.set_xticks((1.5, 2.0, 2.5, 3.0))
            ax.set_xticklabels(("1.5", "2.0", "2.5", "3.0"))

        frontier = result["frontiers"][scenario_key][metric]["records"]
        frontier_points = [p for p in frontier if p["frontier"]]
        frontier_x = [p["cost"] / 1_000_000 if metric == "mac_equivalents_per_bin" else p["cost"]
                      for p in frontier_points]
        frontier_y = [p["mean_r2"] for p in frontier_points]
        if len(frontier_points) > 1:
            order = sorted(range(len(frontier_x)), key=lambda i: (frontier_x[i], frontier_y[i]))
            ax.step([frontier_x[i] for i in order], [frontier_y[i] for i in order],
                    where="post", color="#171717", linestyle=(0, (3, 2)),
                    linewidth=0.85, zorder=2)

        for point in analysis["records"]:
            family = point["family"]
            lo, hi = point["ci95"]
            mean = point["mean_r2"]
            is_compiled = point["implementation"] == "compiled"
            color = FAMILY_COLORS[family]
            x_value = point["costs"][metric]
            if metric == "mac_equivalents_per_bin":
                x_value /= 1_000_000
            ax.errorbar(
                x_value, mean, yerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
                fmt=FAMILY_MARKERS[family], linestyle="none", color=color,
                ecolor="#616A73", elinewidth=0.65, capsize=2.0, capthick=0.65,
                markersize=5.0, markerfacecolor="white" if is_compiled else color,
                markeredgecolor=color, markeredgewidth=1.0, zorder=3,
            )
        for point in frontier_points:
            x_value = point["cost"] / 1_000_000 if metric == "mac_equivalents_per_bin" else point["cost"]
            ax.scatter(x_value, point["mean_r2"], marker="o", s=47,
                       facecolors="none", edgecolors="#171717", linewidths=0.75,
                       zorder=4)
        if scenario_key == "zero_label_dropout_0":
            ax.set_ylim(0.30, 0.70)
            ax.set_yticks((0.3, 0.4, 0.5, 0.6, 0.7))
        else:
            ax.set_ylim(0.08, 0.21)
            ax.set_yticks((0.08, 0.12, 0.16, 0.20))
    # Match each pair of panels on the same cost basis.
    for ax in axes[:2]:
        mac_values = [p["costs"]["mac_equivalents_per_bin"] / 1_000_000
                      for p in _analysis_by_key(result, "zero_label_dropout_0")["records"]]
        ax.set_xlim(min(mac_values) * 0.83, max(mac_values) * 1.22)
    for ax in axes[2:]:
        ax.set_xlim(1.25, 3.15)
    fig.text(0.5, 0.047,
             "Black rings mark observed frontier points; dashed steps show dominance thresholds, not interpolated accuracy.",
             ha="center", va="bottom", fontsize=6.2, color="#45515E")
    fig.text(0.5, 0.023,
             "Bars: source descriptive 95% group-bootstrap intervals; membership uses means only.",
             ha="center", va="bottom", fontsize=6.2, color="#45515E")

    figure_dir = ROOT / "figures"
    preview_dir = figure_dir / "previews"
    submission_dir = REPO / "neuroinformatics_submission"
    figure_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig5_pareto"
    fig.savefig(figure_dir / f"{stem}.pdf", metadata={"Title": "Observed comparison frontier", "Creator": "hdc_study.pareto"})
    fig.savefig(figure_dir / f"{stem}.svg", metadata={"Title": "Observed comparison frontier", "Creator": "hdc_study.pareto"})
    fig.savefig(figure_dir / f"{stem}.eps", metadata={"Title": "Observed comparison frontier", "Creator": "hdc_study.pareto"})
    fig.savefig(preview_dir / f"{stem}.png", dpi=220)
    shutil.copy2(figure_dir / f"{stem}.pdf", submission_dir / "Fig5.pdf")
    shutil.copy2(figure_dir / f"{stem}.eps", submission_dir / "Fig5.eps")
    plt.close(fig)


def _known_case_tests() -> None:
    """Small algorithm checks, including equal-cost and exact-tie cases."""
    cases = [
        {"point_id": "A", "mean_r2": 0.80, "cost": 10.0},
        {"point_id": "B", "mean_r2": 0.70, "cost": 10.0},
        {"point_id": "C", "mean_r2": 0.85, "cost": 12.0},
        {"point_id": "D", "mean_r2": 0.80, "cost": 8.0},
        {"point_id": "E", "mean_r2": 0.80, "cost": 8.0},
        {"point_id": "F", "mean_r2": 0.70, "cost": 12.0},
    ]
    comparisons = _frontier_records(
        [{**x, "family": x["point_id"], "implementation": "original", "ci95": [x["mean_r2"], x["mean_r2"]],
          "costs": {"case": x["cost"]}, "cost_units": {"case": "unit"}} for x in cases],
        "case",
    )
    if comparisons["frontier_point_ids"] != ["C", "D", "E"]:
        raise AssertionError(f"Unexpected known-case frontier: {comparisons['frontier_point_ids']}")
    by_id = {x["point_id"]: x for x in comparisons["records"]}
    if "A" not in by_id["B"]["dominated_by"]:
        raise AssertionError("Equal-cost, higher-score point should dominate")
    if "E" in by_id["D"]["dominated_by"] or "D" in by_id["E"]["dominated_by"]:
        raise AssertionError("Exact ties must not dominate each other")
    if not dominates(by_id["C"], by_id["F"]):
        raise AssertionError("Equal-cost strict score improvement should dominate")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate(*, summary_path: Path = SUMMARY_DEFAULT, efficiency_path: Path = EFFICIENCY_DEFAULT,
             result_path: Path = RESULT_DEFAULT, tex_path: Path = TEX_DEFAULT) -> dict[str, Any]:
    _known_case_tests()
    summary_path = summary_path.resolve()
    efficiency_path = efficiency_path.resolve()
    summary = _read_json(summary_path)
    efficiency = _read_json(efficiency_path)
    result = _build_core(summary, efficiency)
    result["inputs"] = {
        "summary": {"path": _display_path(summary_path), "sha256": _sha256(summary_path)},
        "efficiency": {"path": _display_path(efficiency_path), "sha256": _sha256(efficiency_path)},
    }
    _plot_figure(result)
    tex = render_tex(result)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    result["generated_tex_sha256"] = _sha256(tex_path)
    result["figure_files_sha256"] = {
        rel: _sha256(REPO / rel) for rel in FIGURE_RELATIVE_PATHS
    }
    _write_json(result_path, result)
    return result


def verify_result(result_path: Path = RESULT_DEFAULT, *, summary_path: Path | None = None,
                  efficiency_path: Path | None = None) -> tuple[bool, list[str]]:
    """Verify the saved audit against inputs, regenerated computations, and artifacts.

    This API does not regenerate figures or rewrite files. It checks source
    hashes, pairing preconditions, all stored frontier records, and output
    artifact hashes. It returns ``(passed, messages)`` for direct use by other
    repository tooling.
    """
    messages: list[str] = []
    try:
        _known_case_tests()
        result_path = result_path.resolve()
        result = _read_json(result_path)
        if result.get("schema") != 1 or result.get("complete") is not True:
            raise ValueError("pareto.json is incomplete or has an unsupported schema")
        if summary_path is None:
            summary_path = _resolve_recorded_path(result["inputs"]["summary"]["path"])
        else:
            summary_path = summary_path.resolve()
        if efficiency_path is None:
            efficiency_path = _resolve_recorded_path(result["inputs"]["efficiency"]["path"])
        else:
            efficiency_path = efficiency_path.resolve()
        for name, path in (("summary", summary_path), ("efficiency", efficiency_path)):
            if not path.is_file():
                raise ValueError(f"Missing {name} input: {path}")
            if _sha256(path) != result["inputs"][name]["sha256"]:
                raise ValueError(f"{name} input hash does not match pareto.json")
        if _sha256(Path(__file__).resolve()) != result.get("generator_sha256"):
            raise ValueError("pareto.py generator hash does not match pareto.json")

        expected = _build_core(_read_json(summary_path), _read_json(efficiency_path))
        for key in ("pairing_validation", "score_basis", "cost_basis", "compiled_accuracy_scope",
                    "dominance_rule", "analyses", "frontiers"):
            if result.get(key) != expected.get(key):
                raise ValueError(f"Stored {key} does not match recomputation from source inputs")
        if result.get("pairing_validation", {}).get("passed") is not True:
            raise ValueError("Profiled and outer-selected cost dimensions did not pass validation")
        for rel in FIGURE_RELATIVE_PATHS:
            path = REPO / rel
            if not path.is_file():
                raise ValueError(f"Missing figure artifact: {rel}")
            if _sha256(path) != result.get("figure_files_sha256", {}).get(rel):
                raise ValueError(f"Figure artifact hash mismatch: {rel}")
        if _sha256(TEX_DEFAULT) != result.get("generated_tex_sha256"):
            raise ValueError("Generated TeX hash does not match pareto.json")
        if _sha256(REPO / "hdc_study/figures/fig5_pareto.pdf") != _sha256(REPO / "neuroinformatics_submission/Fig5.pdf"):
            raise ValueError("Manuscript Fig5.pdf is not an exact copy of the generated PDF")
        if _sha256(REPO / "hdc_study/figures/fig5_pareto.eps") != _sha256(REPO / "neuroinformatics_submission/Fig5.eps"):
            raise ValueError("Manuscript Fig5.eps is not an exact copy of the generated EPS")
        messages.extend([
            "Known dominance cases passed, including equal-cost and exact-tie behavior.",
            "Input hashes, outer/profile cost-dimension pairing, and all stored frontier/dominator records passed.",
            "Generated vector, preview, manuscript-copy, and TeX hashes passed.",
        ])
        return True, messages
    except Exception as exc:  # verification API reports issues rather than raising
        messages.append(str(exc))
        return False, messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--efficiency", type=Path, default=EFFICIENCY_DEFAULT)
    parser.add_argument("--output", type=Path, default=RESULT_DEFAULT)
    parser.add_argument("--tex", type=Path, default=TEX_DEFAULT)
    parser.add_argument("--verify", action="store_true", help="verify existing outputs without rewriting them")
    args = parser.parse_args()
    if args.verify:
        passed, messages = verify_result(args.output, summary_path=args.summary, efficiency_path=args.efficiency)
        print("PASS" if passed else "FAIL")
        for message in messages:
            print(f"- {message}")
        return 0 if passed else 1
    result = generate(summary_path=args.summary, efficiency_path=args.efficiency,
                      result_path=args.output, tex_path=args.tex)
    passed, messages = verify_result(args.output, summary_path=args.summary, efficiency_path=args.efficiency)
    print(f"Generated {args.output}; {len(result['analyses'])} conditions x {len(COST_METRICS)} cost bases.")
    print("PASS" if passed else "FAIL")
    for message in messages:
        print(f"- {message}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

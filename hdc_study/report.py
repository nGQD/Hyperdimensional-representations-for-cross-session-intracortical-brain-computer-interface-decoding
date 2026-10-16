"""Generate tables and manuscript text from the complete audited study only."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
from data import ROOT, REPO, sha, write_json
from run import DEV_SEEDS, EXPECTED, FINAL_SEEDS, FAMILIES, fingerprint

DAYS = [f"S{i}" for i in range(6)]
REGIMES = ("few_shot", "zero_shot")
LABELS = dict(hdc="HDC", random="Random features", mlp="MLP features", plain="Contextual GRU",
              selected_control="Selected control", no_channel_roles="No channel roles",
              no_roles="No temporal roles", no_gru="No GRU", no_online_norm="Fixed normalization",
              no_smoothing="No smoothing", no_temporal="Single-bin input")
MANUSCRIPT = REPO / "neuroinformatics_submission"


def read(path):
    return json.loads(Path(path).read_text())


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20261015)
    draws = values[rng.integers(0, len(values), (20000, len(values)))].mean(1)
    return np.quantile(draws, [.025, .975]).tolist()


def aggregate(matrix, days=DAYS):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(days), len(FINAL_SEEDS)) or not np.isfinite(matrix).all():
        raise ValueError("Expected one finite paired score per group and final seed")
    means = matrix.mean(1)
    return dict(mean=float(means.mean()), session_sd=float(means.std(ddof=1)),
                seed_sd=float(matrix.mean(0).std(ddof=1)), ci95=interval(means),
                per_session=[dict(session=d, mean=float(means[i]), seed_scores=matrix[i].tolist()) for i,d in enumerate(days)])


def contrast(a, b, comparator, regime):
    delta = np.asarray(a).mean(1) - np.asarray(b).mean(1)
    signs = np.asarray(list(itertools.product((-1,1), repeat=len(delta))))
    descriptive_p = float(np.mean(np.abs((signs * delta).mean(1)) >= abs(delta.mean()) - 1e-14))
    return dict(comparator=comparator, regime=regime, mean_difference=float(delta.mean()),
                ci95=interval(delta), positive_groups=int((delta>0).sum()),
                exploratory_sign_flip_p=descriptive_p,
                per_session=[dict(session=d, delta=float(v)) for d,v in zip(DAYS, delta)])


def verified_inputs():
    protocol = read(ROOT / "provenance/protocol.json")
    digest = sha(ROOT / "provenance/protocol.json")
    if protocol["fingerprint"] != fingerprint():
        raise RuntimeError("Frozen sources changed")
    audit = read(ROOT / "results/prediction_audit.json")
    if (not audit.get("passed") or audit["protocol_sha256"] != digest
            or audit.get("audited_fit_count") != 640
            or audit.get("audited_archive_count") != 1280
            or audit.get("scored_group_count") != 1760
            or not np.isfinite(audit.get("maximum_absolute_r2_error", np.nan))
            or audit["maximum_absolute_r2_error"] > 1e-8):
        raise RuntimeError("Complete prediction audit required")
    rows = {}
    counts = {s:0 for s in EXPECTED}
    for path in (ROOT / "results/runs").glob("*.json"):
        row = read(path)
        if row["id"] in rows or row["signature"]["protocol_sha256"] != digest:
            raise RuntimeError("Duplicate or incompatible fit record")
        if row["signature"].get("fingerprint") != protocol["fingerprint"]:
            raise RuntimeError("Fit source fingerprint differs from the frozen protocol")
        rows[row["id"]] = row
        counts[row["stage"]] += 1
    if counts != EXPECTED:
        raise RuntimeError("Incomplete fit matrix")
    days = [f"S{i}" for i in range(6)]
    expected_variants = FAMILIES + [
        "no_channel_roles", "no_roles", "no_gru", "no_online_norm",
        "no_smoothing", "no_temporal"]
    if protocol.get("final_variants") != expected_variants:
        raise RuntimeError("Frozen final-variant matrix differs from the report schema")
    configs = protocol.get("candidate_configs", [])
    if len(configs) != 16:
        raise RuntimeError("Frozen protocol does not contain 16 candidate configurations")
    config_keys = [f"{c['family']}_d{c['dimension']}_k{c['temporal_bins']}_lr{c['lr']:g}"
                   for c in configs]
    expected_ids = {
        "inner": {
            f"inner_{a}_{b}_{key}_s{seed}"
            for i, a in enumerate(days) for b in days[i + 1:]
            for key in config_keys for seed in DEV_SEEDS
        },
        "outer": {
            f"outer_{day}_{variant}_s{seed}"
            for day in days for variant in expected_variants for seed in FINAL_SEEDS
        },
        "transfer": {
            f"transfer_{family}_s{seed}"
            for family in FAMILIES for seed in FINAL_SEEDS
        },
    }
    for stage, expected in expected_ids.items():
        actual = {identifier for identifier, row in rows.items() if row["stage"] == stage}
        if actual != expected:
            raise RuntimeError(f"The {stage} fit identifiers differ from the frozen matrix")
    for identifier, row in rows.items():
        expected_names = ({f"{identifier}_zero_shot.npz", f"{identifier}_few_shot.npz"}
                          if row["stage"] in ("outer", "transfer") else set())
        if set(row.get("prediction_sha256", {})) != expected_names:
            raise RuntimeError(f"{identifier} saved-prediction manifest is incomplete")
    expected_prediction_hashes = {
        filename: digest_value
        for row in rows.values() if row["stage"] in ("outer", "transfer")
        for filename, digest_value in row.get("prediction_sha256", {}).items()
    }
    audited_files = audit.get("files")
    if not isinstance(audited_files, list) or len(audited_files) != 1280:
        raise RuntimeError("Prediction audit file manifest is incomplete")
    audited_hashes = {row.get("filename"): row.get("sha256") for row in audited_files}
    if len(audited_hashes) != len(audited_files) or audited_hashes != expected_prediction_hashes:
        raise RuntimeError("Prediction audit manifest does not match saved fit predictions")
    for filename, expected_hash in expected_prediction_hashes.items():
        prediction_path = ROOT / "results/predictions" / filename
        if not prediction_path.is_file() or sha(prediction_path) != expected_hash:
            raise RuntimeError(f"Saved prediction changed after audit: {filename}")
    status = read(ROOT / "results/diagnostics/status.json")
    if status["state"] != "complete" or status["completed"] != 240 or status["protocol_sha256"] != digest:
        raise RuntimeError("Complete diagnostics required")
    diagnostics = {}
    from diagnostics import validate_diagnostic_record
    for day in DAYS:
        for family in FAMILIES:
            for seed in FINAL_SEEDS:
                identifier = f"outer_{day}_{family}_s{seed}"
                row = read(ROOT / "results/diagnostics" / (identifier + ".json"))
                validate_diagnostic_record(row, identifier, family, seed, day,
                    digest, rows[identifier]["checkpoint_sha256"], rows[identifier]["metrics"])
                diagnostics[identifier] = row
    return protocol, digest, rows, diagnostics


def verified_matched_controls(protocol, primary_digest, primary_rows, selection,
                               primary_matrices):
    """Verify the complete, separately frozen capacity-matched control analysis.

    The runner's verifier checks checkpoints and prediction arrays against the
    verified H1 caches. This report layer additionally binds each comparison
    to the inner-selected HDC configuration and to the exact primary HDC outer
    fit. Nothing is summarized until all 120 records pass both checks.
    """
    from matched_controls import verify_completed_matrix

    result = verify_completed_matrix()
    capacity = result.get("capacity")
    capacity_path = ROOT / "provenance/capacity_controls_protocol.json"
    capacity_digest = sha(capacity_path)
    if (not isinstance(capacity, dict)
            or capacity.get("schema") != 1
            or capacity.get("design_state") != "frozen_before_outer_results"
            or capacity.get("no_outer_results_at_design") is not True
            or capacity.get("primary_protocol_sha256") != primary_digest
            or capacity.get("core_fingerprint") != protocol.get("fingerprint")
            or capacity.get("comparison_count") != 120
            or capacity.get("days") != DAYS
            or capacity.get("final_seeds") != FINAL_SEEDS
            or capacity.get("primary_endpoint_unchanged") is not True
            or capacity.get("outcome_dependent_choices") is not False
            or result.get("primary_sha256") != primary_digest
            or result.get("capacity_sha256") != capacity_digest
            or result.get("core_fingerprint") != protocol.get("fingerprint")):
        raise RuntimeError("A complete capacity-control protocol bound to the frozen primary study is required")
    from control_implementation import verify_control_implementation
    current_module_sha = verify_control_implementation(ROOT)
    if result.get("module_sha256") != current_module_sha:
        raise RuntimeError("Matched-control source fingerprint differs from its frozen design")

    expected_ids = {
        f"matched_{day}_{family}_s{seed}"
        for day in DAYS for family in ("random", "mlp") for seed in FINAL_SEEDS
    }
    matched_rows = result.get("rows")
    if not isinstance(matched_rows, dict) or set(matched_rows) != expected_ids:
        raise RuntimeError("All 120 verified matched-control records are required")

    status_path = ROOT / "results/matched_controls/status.json"
    summary_path = ROOT / "results/matched_controls/summary.json"
    if not status_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Complete matched-control status and summary files are required")
    status = read(status_path)
    saved_summary = read(summary_path)
    expected_family_counts = {"random": 60, "mlp": 60}
    if (status.get("state") != "complete" or status.get("completed") != 120
            or status.get("expected") != 120
            or status.get("family_counts") != expected_family_counts
            or status.get("primary_protocol_sha256") != primary_digest
            or status.get("capacity_protocol_sha256") != capacity_digest):
        raise RuntimeError("Matched-control runner status does not attest to all 120 completed records")
    saved_comparisons = saved_summary.get("comparisons")
    if (saved_summary.get("comparison_count") != 120
            or saved_summary.get("schema") != 1
            or saved_summary.get("primary_protocol_sha256") != primary_digest
            or saved_summary.get("capacity_protocol_sha256") != capacity_digest
            or saved_summary.get("core_fingerprint") != protocol.get("fingerprint")
            or saved_summary.get("module_sha256") != current_module_sha
            or saved_summary.get("metric_rows_per_comparison") != 2
            or saved_summary.get("predictions_verified") != 240
            or saved_summary.get("family_counts") != expected_family_counts
            or not isinstance(saved_comparisons, list) or len(saved_comparisons) != 120):
        raise RuntimeError("Matched-control summary is incomplete or uses a different frozen protocol")
    summary_by_id = {row.get("id"): row for row in saved_comparisons}
    if len(summary_by_id) != 120 or set(summary_by_id) != expected_ids:
        raise RuntimeError("Matched-control summary identifiers differ from the complete expected matrix")

    # The saved summary is itself an output artifact; bind each displayed row
    # to the validated per-comparison record rather than trusting its values.
    row_fields = ("id", "day", "family", "seed", "config", "protocol_sha256",
                  "capacity_protocol_sha256", "matched_hdc_identifier",
                  "matched_hdc_config", "reused_identifier", "checkpoint",
                  "predictions", "prediction_sha256", "metrics", "parameters")
    matched_matrices = {(family, regime): []
                        for family in ("random", "mlp") for regime in REGIMES}
    reused_counts = {family: 0 for family in ("random", "mlp")}
    new_counts = {family: 0 for family in ("random", "mlp")}
    result_sources, checkpoint_sources, prediction_sources = {}, {}, {}

    for day in DAYS:
        day_configs = {}
        selected_hdc = selection[day]["families"]["hdc"][0]["config"]
        for family in ("random", "mlp"):
            for seed in FINAL_SEEDS:
                identifier = f"matched_{day}_{family}_s{seed}"
                row = matched_rows[identifier]
                if (row.get("id") != identifier or row.get("day") != day
                        or row.get("family") != family or row.get("seed") != seed
                        or row.get("protocol_sha256") != primary_digest
                        or row.get("capacity_protocol_sha256") != capacity_digest):
                    raise RuntimeError(f"Matched-control identity or protocol mismatch: {identifier}")

                expected_hdc = dict(selected_hdc)
                expected_hdc["family"] = "hdc"
                expected_hdc["codebook_seed"] = 100000 + seed
                expected_config = dict(selected_hdc)
                expected_config["family"] = family
                expected_config["codebook_seed"] = 100000 + seed
                hdc_identifier = f"outer_{day}_hdc_s{seed}"
                hdc_row = primary_rows[hdc_identifier]
                if hdc_row["signature"].get("config") != expected_hdc:
                    raise RuntimeError(f"Outer HDC fit does not match its inner-selected config: {hdc_identifier}")
                if (row.get("matched_hdc_identifier") != hdc_identifier
                        or row.get("matched_hdc_config") != expected_hdc
                        or row.get("config") != expected_config):
                    raise RuntimeError(f"Matched config differs from selected HDC capacity: {identifier}")

                reuse = row.get("reused_identifier")
                expected_reuse = f"outer_{day}_{family}_s{seed}"
                if reuse not in (None, expected_reuse):
                    raise RuntimeError(f"Unexpected fit reuse source for {identifier}: {reuse}")
                if reuse is not None:
                    reused_counts[family] += 1
                    reused_primary = primary_rows[reuse]
                    if reused_primary["signature"].get("config") != expected_config:
                        raise RuntimeError(f"Reused fit is not exactly capacity matched: {identifier}")
                else:
                    new_counts[family] += 1

                summary_row = summary_by_id[identifier]
                if any(summary_row.get(key) != row.get(key) for key in row_fields):
                    raise RuntimeError(f"Saved matched-control summary differs from validated row: {identifier}")
                result_path = ROOT / "results/matched_controls" / f"{day}_{family}_s{seed}.json"
                if not result_path.is_file():
                    raise RuntimeError(f"Missing matched-control result record: {identifier}")
                result_sources[result_path.name] = sha(result_path)

                checkpoint = row.get("checkpoint", {})
                checkpoint_rel = checkpoint.get("path")
                checkpoint_path = ROOT / str(checkpoint_rel or "")
                if (not checkpoint_rel or not checkpoint_path.is_file()
                        or checkpoint.get("sha256") != sha(checkpoint_path)):
                    raise RuntimeError(f"Matched-control checkpoint changed after validation: {identifier}")
                checkpoint_sources[str(checkpoint_rel)] = checkpoint.get("sha256")

                predictions = row.get("predictions")
                if not isinstance(predictions, dict) or set(predictions) != set(REGIMES):
                    raise RuntimeError(f"Matched-control prediction manifest is incomplete: {identifier}")
                values_by_regime = {}
                for regime in REGIMES:
                    reference = predictions[regime]
                    if not isinstance(reference, dict):
                        raise RuntimeError(f"Malformed prediction reference: {identifier}/{regime}")
                    filename = (f"{reuse}_{regime}.npz" if reuse else f"{identifier}_{regime}.npz")
                    expected_dir = "results/predictions" if reuse else "results/matched_predictions"
                    expected_path = f"{expected_dir}/{filename}"
                    if reference.get("path") != expected_path:
                        raise RuntimeError(f"Unexpected prediction source for {identifier}/{regime}")
                    prediction_path = ROOT / expected_path
                    prediction_digest = reference.get("sha256")
                    if (not prediction_path.is_file() or not isinstance(prediction_digest, str)
                            or sha(prediction_path) != prediction_digest):
                        raise RuntimeError(f"Matched-control prediction hash mismatch: {identifier}/{regime}")
                    prediction_sources[expected_path] = prediction_digest
                    if reuse:
                        primary_name = f"{reuse}_{regime}.npz"
                        if primary_rows[reuse].get("prediction_sha256", {}).get(primary_name) != prediction_digest:
                            raise RuntimeError(f"Reused prediction is not bound to its primary fit: {identifier}/{regime}")

                    metric_rows = [m for m in row.get("metrics", [])
                                   if m.get("day") == day and m.get("regime") == regime]
                    if len(metric_rows) != 1 or not np.isfinite(metric_rows[0].get("r2", np.nan)):
                        raise RuntimeError(f"Missing or invalid matched score: {identifier}/{regime}")
                    audit = row.get("prediction_audit", {}).get(regime, {})
                    if (not np.isfinite(audit.get("r2", np.nan))
                            or abs(float(audit["r2"]) - float(metric_rows[0]["r2"])) > 1e-8):
                        raise RuntimeError(f"Matched prediction audit differs from score: {identifier}/{regime}")
                    values_by_regime[regime] = float(metric_rows[0]["r2"])
                day_configs.setdefault(family, {})[seed] = values_by_regime
            # Seed order is fixed and verified above; build a paired 6 x 10 matrix.
            matched_matrices[family, "few_shot"].append(
                [day_configs[family][seed]["few_shot"] for seed in FINAL_SEEDS])
            matched_matrices[family, "zero_shot"].append(
                [day_configs[family][seed]["zero_shot"] for seed in FINAL_SEEDS])

    scores, contrasts = [], []
    for family in ("random", "mlp"):
        for regime in REGIMES:
            matrix = np.asarray(matched_matrices[family, regime], dtype=np.float64)
            if matrix.shape != (len(DAYS), len(FINAL_SEEDS)) or not np.isfinite(matrix).all():
                raise RuntimeError(f"Incomplete matched-control score matrix: {family}/{regime}")
            matched_matrices[family, regime] = matrix
            label = f"matched_{family}"
            scores.append(dict(family=label, matched_family=family,
                               capacity_match="inner-selected HDC D and K", regime=regime,
                               **aggregate(matrix)))
            paired = contrast(primary_matrices["hdc", regime], matrix,
                              label, regime)
            paired["matched_family"] = family
            paired["capacity_match"] = "inner-selected HDC D and K"
            contrasts.append(paired)

    all_records = {
        identifier: {key: row.get(key) for key in row_fields}
        for identifier, row in sorted(matched_rows.items())
    }
    result = dict(
        complete=True,
        study_id=capacity["study_id"],
        primary_protocol_sha256=primary_digest,
        capacity_protocol_sha256=capacity_digest,
        comparison_count=120,
        counts=dict(
            per_family={family: dict(reused_primary=reused_counts[family],
                                     newly_trained=new_counts[family],
                                     total=reused_counts[family] + new_counts[family])
                        for family in ("random", "mlp")},
            reused_primary=sum(reused_counts.values()),
            newly_trained=sum(new_counts.values()),
        ),
        scores=scores,
        contrasts=contrasts,
        comparisons=all_records,
        source_hashes=dict(
            capacity_protocol=capacity_digest,
            runner=current_module_sha,
            status=sha(status_path),
            summary=sha(summary_path),
            results=result_sources,
            checkpoints=checkpoint_sources,
            predictions=prediction_sources,
        ),
    )
    if (result["counts"]["reused_primary"] + result["counts"]["newly_trained"] != 120
            or result["counts"]["per_family"] != {
                family: dict(reused_primary=reused_counts[family],
                             newly_trained=new_counts[family], total=60)
                for family in ("random", "mlp")}):
        raise RuntimeError("Matched-control reused/new fit accounting is inconsistent")
    return result, matched_matrices


def make_summary():
    protocol, digest, rows, diagnostics = verified_inputs()
    selection = read(ROOT / "results/selection.json")
    variants = protocol["final_variants"] + ["selected_control"]
    matrices, scores, contrasts, ablation_contrasts = {}, [], [], []
    def effective(day, family):
        return selection[day]["selected_control"] if family == "selected_control" else family
    for regime in REGIMES:
        for family in variants:
            matrix = []
            for day in DAYS:
                values = []
                for seed in FINAL_SEEDS:
                    row = rows[f"outer_{day}_{effective(day,family)}_s{seed}"]
                    metrics = [m for m in row["metrics"] if m["day"] == day and m["regime"] == regime]
                    if len(metrics) != 1:
                        raise RuntimeError("Missing or duplicate group metric")
                    values.append(metrics[0]["r2"])
                matrix.append(values)
            matrices[family,regime] = np.asarray(matrix)
            scores.append(dict(family=family, regime=regime, **aggregate(matrix)))
        for family in variants:
            if family == "hdc":
                continue
            row = contrast(matrices["hdc",regime], matrices[family,regime], family, regime)
            (contrasts if family in FAMILIES + ["selected_control"] else ablation_contrasts).append(row)
    matched_controls, _matched_matrices = verified_matched_controls(
        protocol, digest, rows, selection, matrices)
    calibration, robustness = [], []
    for family in FAMILIES + ["selected_control"]:
        for seconds in (0,15,30,60):
            matrix = [[next(r["r2"] for r in diagnostics[f"outer_{day}_{effective(day,family)}_s{seed}"]["calibration"]
                            if r["prefix_seconds"] == seconds) for seed in FINAL_SEEDS] for day in DAYS]
            calibration.append(dict(family=family, prefix_seconds=seconds, **aggregate(matrix)))
        for kind in ("gain", "dropout"):
            for level in (0,.1,.2,.4):
                matrix = [[next(r["r2"] for r in diagnostics[f"outer_{day}_{effective(day,family)}_s{seed}"]["robustness"]
                                if r["perturbation"] == kind and r["level"] == level) for seed in FINAL_SEEDS] for day in DAYS]
                robustness.append(dict(family=family, perturbation=kind, level=level, **aggregate(matrix)))
    transfer = []
    transfer_days = [f"S{i}" for i in range(6,13)]
    for regime in REGIMES:
        for family in FAMILIES:
            matrix = [[next(r["r2"] for r in rows[f"transfer_{family}_s{seed}"]["metrics"]
                            if r["regime"] == regime and r["day"] == day)
                       for seed in FINAL_SEEDS] for day in transfer_days]
            transfer.append(dict(family=family, regime=regime, **aggregate(matrix, transfer_days)))
    summary = dict(schema=1, complete=True, study_id=protocol["study_id"], protocol_sha256=digest,
        counts={s:dict(completed=n, expected=n) for s,n in EXPECTED.items()}, sessions=DAYS,
        primary_regime="few_shot", scores=scores, contrasts=contrasts, ablation_contrasts=ablation_contrasts,
        matched_capacity_controls=matched_controls,
        calibration=calibration, robustness=robustness, transfer=transfer,
        selection=selection, final_seeds=FINAL_SEEDS,
        uncertainty="20,000 group-bootstrap draws; descriptive only because sessions share one participant and folds share training data. Sign-flip summaries assume exchangeable independent group signs and are exploratory, not confirmatory tests.",
        selected_control="Random or MLP family chosen separately for each outer group using inner validation only",
        prediction_audit_sha256=sha(ROOT / "results/prediction_audit.json"),
        sources={p.name:sha(p) for p in sorted((ROOT / "results/runs").glob("*.json"))},
        diagnostic_sources={p.name:sha(p) for p in sorted((ROOT / "results/diagnostics").glob("outer*.json"))})
    write_json(ROOT / "results/summary.json", summary)
    return summary


def table(caption, headers, rows, label, compact=False):
    columns = "l" + "r" * (len(headers)-1)
    sizing = (r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}"
              if compact else r"\centering\small")
    return "\n".join([r"\begin{table}[!htbp]", r"\caption{"+caption+"}", r"\label{"+label+"}",
        sizing, r"\begin{tabular}{"+columns+"}", r"\toprule",
        " & ".join(headers)+r" \\", r"\midrule", *[" & ".join(map(str,row))+r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def fmt(value):
    return f"{value:.3f}"


def capacity_interpretation(contrasts):
    calibrated = {row["matched_family"]: row for row in contrasts
                  if row["regime"] == "few_shot"}
    if set(calibrated) != {"random", "mlp"}:
        raise ValueError("Both calibrated matched-capacity contrasts are required")
    if calibrated["random"]["ci95"][0] > 0 and calibrated["mlp"]["ci95"][1] < 0:
        return ("At HDC's selected size and temporal window, HDC outperformed the matched random-feature control "
                "but fell below the matched MLP, with both descriptive difference intervals excluding zero. "
                "In these six groups, the correlated-anchor codebook improved on independent random mixing, "
                "while the tested learned input features had the highest mean score among the three matched models.")
    if all(row["ci95"][0] > 0 for row in calibrated.values()):
        return ("At HDC's selected size and temporal window, both matched-control contrasts also favor HDC, "
                "with descriptive group-resampling intervals above zero. This supports a representation "
                "advantage within the evaluated groups and fixed training protocol.")
    better = ["random-feature" if family == "random" else "MLP-feature"
              for family, row in calibrated.items() if row["ci95"][1] < 0]
    if better:
        names = " and ".join(better)
        return (f"At HDC's selected size and temporal window, the {names} "
                + ("control has" if len(better) == 1 else "controls have")
                + " a higher mean score with a descriptive interval for the HDC difference below zero. "
                "The matched-capacity analysis therefore does not support superiority of the HDC representation.")
    return ("At HDC's selected size and temporal window, at least one matched-control interval includes zero. "
            "The observed comparisons do not establish a consistent representation advantage over both controls.")


def recurrent_baseline_interpretation(contrasts):
    rows = [row for row in contrasts
            if row["comparator"] == "plain" and row["regime"] == "few_shot"]
    if len(rows) != 1:
        raise ValueError("One calibrated contextual-GRU contrast is required")
    row = rows[0]
    lo, hi = row["ci95"]
    values = (f"The HDC-minus-contextual-GRU difference was {row['mean_difference']:+.3f} $R^2$ "
              f"(descriptive interval [{fmt(lo)}, {fmt(hi)}]). ")
    if hi < 0:
        return values + "The conventional recurrent baseline provided the strongest mean accuracy in the primary evaluation."
    if lo > 0:
        return values + "This comparison also favors HDC under the shared evaluation protocol."
    return values + "This comparison does not establish a consistent advantage over the conventional recurrent baseline."


def temporal_selection_interpretation(selection):
    windows = [selection[day]["families"]["hdc"][0]["config"]["temporal_bins"] for day in DAYS]
    if any(window not in (1, 8) for window in windows):
        raise ValueError("HDC selections must use a prescribed one- or eight-bin context")
    single = windows.count(1)
    if single == len(DAYS):
        return ("Inner validation selected a one-bin HDC input ($K=1$) in all six outer folds. "
                "The selected architecture uses HDC for current-bin population encoding and the GRU for temporal memory, "
                "without a separate HDC lag buffer. The no-temporal comparison is identical to the selected model "
                "in every fold; removing temporal roles changes its fixed coordinate rotation. "
                "The component results therefore describe channel-value encoding with recurrent temporal integration.")
    return (f"Inner validation selected an eight-bin HDC input in {len(DAYS) - single} of six outer folds "
            f"and a one-bin input in {single}. The no-temporal ablation has the same configuration as the full model "
            "where $K=1$; only folds selecting $K=8$ test the added HDC input window.")


def build_protocol():
    protocol = read(ROOT / "provenance/protocol.json")
    MANUSCRIPT.mkdir(exist_ok=True)
    text = r"""All families use 128-dimensional hidden states. HDC and fixed random features search $D\in\{256,512\}$ complex-equivalent dimensions and $K\in\{1,8\}$ bins; the learned MLP uses the corresponding matched projection budgets. The contextual GRU searches $K\in\{1,8\}$ and learning rates $2\times10^{-4}$ or $8\times10^{-4}$. Each family receives four candidates. HDC, random-feature, and MLP candidates are paired by $D$ and $K$; selection is conducted independently for each family, so their final $D$, $K$, and trainable parameter counts can differ. The three feature families use learning rate $2\times10^{-4}$. The channel-value code uses eight anchors. Development seeds are 11, 23, 37, 51, and 71; final seeds are 101, 113, 127, 139, 151, 163, 179, 191, 211, and 223. Each codebook seed equals its training seed plus 100,000.

For each outer group, selection averages the 25 inner group-by-seed scores using a 60-second calibration prefix. Ties favor fewer trainable parameters, then the lexical configuration identifier. Excluding the same pair of groups gives an identical training set for two inner/outer assignments; each such fit is reused. The primary design contains 1,200 inner fits, 600 outer fits across four families and six conditional HDC ablations, and 40 transfer fits. The transfer analysis trains on all six held-in groups and evaluates the seven released held-out calibration groups, using configuration choices pooled only from inner results.

Training uses 256-bin windows at a 64-bin stride and resets recurrent state at each training window. Inference carries state through each recording and resets at recording boundaries. All candidates receive 50 epochs of AdamW optimization, a three-epoch warmup followed by cosine decay, weight decay $10^{-4}$, gradient-norm clipping at 5, and a Huber objective on the official valid-target mask. Training adds Gaussian noise with standard deviation 0.15 to standardized inputs and independently drops channels with probability 0.1, scaling retained channels by $1/0.9$. Training-target means and standard deviations are estimated only from valid labels in the training groups. Readout dropout is 0.2 during training and disabled during inference and calibration.

Training groups provide pooled neural means $\mu_0$ and standard deviations $\sigma_0$. Within each recording, cumulative moments through bin $t$ give $\hat\mu_t$ and $\hat\sigma_t$, with a standard-deviation floor of 0.01. The mixing weight is $\lambda_t=0$ for $t<30$ and $\min(0.7,t/2000)$ otherwise. Normalization uses
\[
 \mu_t=(1-\lambda_t)\mu_0+\lambda_t\hat\mu_t,\qquad
 \sigma_t=(1-\lambda_t)\sigma_0+\lambda_t\hat\sigma_t,\qquad
 z_t=(x_t-\mu_t)/\sigma_t.
\]
The input smoother is $s_t=0.9s_{t-1}+0.1z_t$, initialized with $s_1=z_1$. For labeled calibration, the allowed prefix supplies the prior neural moments; the prefix is then replayed with those moments. Twenty AdamW steps at learning rate $10^{-4}$ and weight decay $10^{-4}$ update only the readout from valid prefix labels. The entire allowed prefix is available before any scored prediction. Recurrent and encoder parameters remain fixed, and no behavioral labels from the scored suffix enter an update. The no-online-normalization condition keeps its supplied prior fixed, including the permitted prefix prior in the calibrated condition.

At each group, $R^2$ is computed as
\[
 R^2=1-\frac{\sum_{t\in\mathcal{M}}\sum_{d=1}^{7}(y_{td}-\hat y_{td})^2}
 {\sum_{t\in\mathcal{M}}\sum_{d=1}^{7}(y_{td}-\bar y_d)^2},
\]
where $\mathcal{M}$ is the common suffix intersected with the official mask. Targets retain the official loader's values and alignment; no additional target smoothing or delay search is performed. Ten replicate scores are averaged within each group, then the six group means receive equal weight. We report the standard deviation across group means and, separately, the standard deviation across replicate-wise six-group means. Percentile intervals use 20,000 resamples of the six paired group effects. They describe sensitivity to these groups and are not population confidence intervals.

Synthetic perturbations begin at the first scored suffix bin and leave labels unchanged. Gain perturbations multiply each channel by $\exp(\ell\epsilon_c)$, where $\epsilon_c\sim\mathcal N(0,1)$ and $\ell\in\{0,0.1,0.2,0.4\}$. Dropout sets a fixed fraction $\ell$ of channels to zero. Perturbations use seed $700000+$ training seed and are paired across families. These analyses use the zero-label condition, with causal normalization continuing throughout the perturbation.

The software uses PyTorch 2.5.1 with CUDA 12.4 on an NVIDIA GeForce RTX 4080 Laptop GPU. Training permits TensorFloat-32 kernels; inference and adaptation use strict 32-bit floating-point matrix operations. The measured throughput pilot selects a shared minibatch of 256. Source, data, configuration, checkpoint, and saved-prediction hashes accompany the results.
"""
    (MANUSCRIPT / "protocol_generated.tex").write_text(text)
    shutil.copy2(ROOT / "theory.tex", MANUSCRIPT / "theory_generated.tex")


def write_manuscript(summary):
    score = {(r["family"],r["regime"]):r for r in summary["scores"]}
    selected_families = {summary["selection"][day].get("selected_control") for day in DAYS}
    common_control = next(iter(selected_families)) if len(selected_families) == 1 else None
    common_control = common_control if common_control in ("random", "mlp") else None
    if common_control and any(score[common_control, regime]["per_session"] !=
                              score["selected_control", regime]["per_session"] for regime in REGIMES):
        raise ValueError("A common selected control must match its family's reported scores")
    table_families = FAMILIES if common_control else FAMILIES + ["selected_control"]
    display_labels = dict(LABELS)
    if common_control:
        display_labels[common_control] = ("MLP" if common_control == "mlp" else "Random") + " (selected)"
    control_label = "inner-selected " + ("MLP" if common_control == "mlp" else
                                         "random-feature control" if common_control == "random" else "control")
    hdc, control = score["hdc","few_shot"], score["selected_control","few_shot"]
    primary = next(r for r in summary["contrasts"] if r["comparator"] == "selected_control" and r["regime"] == "few_shot")
    lo, hi = primary["ci95"]
    difference = primary["mean_difference"]
    observed = "higher" if difference > 0 else "lower" if difference < 0 else "equal"
    random_primary = next(row for row in summary["contrasts"]
                          if row["comparator"] == "random" and row["regime"] == "few_shot")
    abstract = (f"Mean outer-group $R^2$ was {fmt(hdc['mean'])} for HDC, "
                f"{fmt(score['random', 'few_shot']['mean'])} for random features, "
                f"{fmt(control['mean'])} for the {control_label}, and "
                f"{fmt(score['plain', 'few_shot']['mean'])} for the plain GRU. "
                f"The primary HDC-minus-control difference was {difference:+.3f} "
                f"(descriptive group-bootstrap interval [{fmt(lo)}, {fmt(hi)}]); "
                f"the HDC-minus-random difference was {random_primary['mean_difference']:+.3f}. ")
    if all(summary["selection"][day]["families"]["hdc"][0]["config"]["temporal_bins"] == 1
           for day in DAYS):
        abstract += "All six HDC selections paired current-bin population codes with recurrent temporal memory."
    dropout_endpoint = {row["family"]: row for row in summary["robustness"]
                        if row["perturbation"] == "dropout" and row["level"] == .4}
    hdc_leads_dropout = all(dropout_endpoint["hdc"]["mean"] > dropout_endpoint[family]["mean"]
                           for family in ("random", "mlp", "plain"))
    if hdc_leads_dropout:
        abstract += (" At the largest prespecified dropout fraction (40\\%), HDC had the highest zero-label "
                     f"mean $R^2$ ({fmt(dropout_endpoint['hdc']['mean'])}) among the four families.")
    body = ("The primary design comprised 1,840 fits, and the separate capacity-matched analysis contains 120 comparison records. "
        "All primary outer and transfer archives and the matched-control prediction arrays passed their source, mask, hash, and score audits. "
        f"With a 60-second calibration prefix, mean HDC $R^2$ was {fmt(hdc['mean'])}, with a group standard deviation of {fmt(hdc['session_sd'])} "
        f"and replicate-mean standard deviation of {fmt(hdc['seed_sd'])}. The selected control scored {fmt(control['mean'])}. "
        f"The paired HDC difference was {difference:+.3f}, with a descriptive 95\\% interval [{fmt(lo)}, {fmt(hi)}]; "
        f"{primary['positive_groups']} of six group effects favored HDC. Figure~\\ref{{fig:session-results}} shows all group effects.\n\n")
    main_rows = []
    for family in table_families:
        a,b = score[family,"few_shot"],score[family,"zero_shot"]
        main_rows.append([display_labels[family],fmt(a["mean"]),fmt(a["session_sd"]),fmt(a["seed_sd"]),fmt(b["mean"])])
    if common_control:
        body += (f"Inner validation selected {LABELS[common_control]} as the control in all six outer folds; "
                 "its family and selected-control results are therefore identical and are shown once.\n\n")
    body += "Table~\\ref{tab:main} summarizes performance and variation across groups and replicates.\n"
    body += table("Outer-group performance on the common scored suffix", ["Family","Calibrated $R^2$","Group SD","Replicate SD","Zero-label $R^2$"],main_rows,"tab:main")
    zero = score["hdc","zero_shot"]
    body += (f"\n\nWithout evaluated-group labels, HDC achieved mean $R^2={fmt(zero['mean'])}$ on the identical target bins. "
             "The calibration curves and synthetic gain/dropout analyses appear in Figure~\\ref{fig:calibration-robustness}. "
             "These perturbations probe defined changes in recorded inputs; they do not reproduce all sources of biological recording drift.\n\n")
    calibration = {(r["family"], r["prefix_seconds"]): r for r in summary["calibration"]}
    calibration_delta = {family: calibration[family, 60]["mean"] - calibration[family, 0]["mean"]
                         for family in ("hdc", "selected_control")}
    body += (f"The 60-second-minus-zero-label difference on the common suffix was {calibration_delta['hdc']:+.3f} "
             f"$R^2$ for HDC and {calibration_delta['selected_control']:+.3f} for the selected control. "
             "This contrast measures the complete calibration procedure, which changes both the neural normalization prior "
             "and the readout weights. It does not isolate supervised weight adaptation. ")
    robustness = {(r["family"], r["perturbation"], r["level"]): r for r in summary["robustness"]}
    for kind, condition in (("gain", "gain log-standard deviation of 0.4"),
                            ("dropout", "channel-dropout fraction of 0.4")):
        hdc_perturbed = robustness["hdc", kind, .4]["mean"]
        control_perturbed = robustness["selected_control", kind, .4]["mean"]
        hdc_change = hdc_perturbed - robustness["hdc", kind, 0]["mean"]
        control_change = control_perturbed - robustness["selected_control", kind, 0]["mean"]
        body += (f"At the largest specified {condition}, HDC and the selected control scored "
                 f"{fmt(hdc_perturbed)} and {fmt(control_perturbed)}, respectively, with changes of "
                 f"{hdc_change:+.3f} and {control_change:+.3f} from their unperturbed zero-label scores. ")
    body += "\n\n"
    matched = summary["matched_capacity_controls"]
    matched_scores = {(r["family"], r["regime"]): r for r in matched["scores"]}
    matched_contrasts = {(r["matched_family"], r["regime"]): r for r in matched["contrasts"]}

    def matched_comparison(family, regime):
        row = matched_contrasts[family, regime]
        delta = row["mean_difference"]
        if delta > 0:
            direction = "scored higher than"
        elif delta < 0:
            direction = "scored lower than"
        else:
            direction = "scored the same as"
        ci = row["ci95"]
        hdc_mean = score["hdc", regime]["mean"]
        control_mean = matched_scores[f"matched_{family}", regime]["mean"]
        control_name = "MLP" if family == "mlp" else "random-feature"
        return (f"HDC {direction} the matched {control_name} control ({fmt(hdc_mean)} vs {fmt(control_mean)}; "
                f"difference {delta:+.3f}, descriptive 95\\% group-bootstrap interval "
                f"[{fmt(ci[0])}, {fmt(ci[1])}], {row['positive_groups']} of six group effects positive)")

    few_random = matched_comparison("random", "few_shot")
    few_mlp = matched_comparison("mlp", "few_shot")
    zero_random = matched_comparison("random", "zero_shot")
    zero_mlp = matched_comparison("mlp", "zero_shot")
    reused = matched["counts"]["reused_primary"]
    newly_trained = matched["counts"]["newly_trained"]
    body += ("\n\nA separately frozen capacity-matched analysis copied each outer group's inner-selected HDC "
             "$D$ and $K$ and changed the feature family. In the calibrated condition, " + few_random + "; " + few_mlp + ". "
             "Without evaluated-group labels, " + zero_random + "; " + zero_mlp + ". "
             f"Of the 120 comparisons, {reused} reused an exactly matching primary fit and {newly_trained} required a new fit. "
             "These paired group summaries are descriptive for the observed sessions.\n\n")
    ablation_rows = []
    for row in summary["ablation_contrasts"]:
        if row["regime"] == "few_shot":
            ci=row["ci95"]
            ablation_rows.append([LABELS[row["comparator"]],fmt(score[row["comparator"],"few_shot"]["mean"]),
                f"{row['mean_difference']:+.3f}",f"[{fmt(ci[0])}, {fmt(ci[1])}]"])
    body += "Table~\\ref{tab:ablations} reports the conditional component comparisons after calibration.\n"
    body += table("Conditional HDC component comparisons after calibration",["Condition","$R^2$","HDC difference","Descriptive interval"],ablation_rows,"tab:ablations")
    temporal_text = temporal_selection_interpretation(summary["selection"])
    body += ("\n\nAblations retain the selected HDC configuration and training settings. " + temporal_text
             + " Selected configurations and all seven additional transfer groups are reported in the supplement.")
    if lo > 0:
        interpretation = ("The observed primary contrast favors HDC, and the descriptive group-resampling interval remains above zero. "
                          "This is consistent with an HDC advantage in the evaluated sessions under the prespecified candidate grid and training protocol. "
                          "It does not establish that HDC is the best decoder across participants, tasks, or optimization budgets.")
    elif hi < 0:
        interpretation = (f"The {control_label} had higher primary accuracy. "
                          "The representation and input-loss analyses characterize complementary properties of the streaming decoder.")
    else:
        interpretation = ("The descriptive interval spans zero, so the primary comparison does not establish a consistent HDC advantage across the observed session groups. "
                          "The distributed-code rationale remains a mechanistic hypothesis whose practical value depends on the measured setting.")
    comparison = (f"HDC's mean performance was {observed} than the inner-selected comparator"
                  if observed != "equal" else "HDC's mean performance equaled the inner-selected comparator")
    capacity_text = capacity_interpretation(matched["contrasts"])
    baseline_text = recurrent_baseline_interpretation(summary["contrasts"])
    discussion = (capacity_text + " " + f"{comparison} by {abs(difference):.3f} $R^2$. "
                  + baseline_text)
    discussion += ("\n\nThe kernel derivation explains how the two fixed encoders organize channel values. "
                   "In expectation, HDC's Gaussian-frequency anchors induce graded similarity between nearby values; "
                   "independent Gaussian features preserve only interpolation-weight overlap. "
                   "Both use the same bounded input basis. Their matched comparison therefore tests the fixed codebook "
                   "under a common downstream architecture. The MLP and plain GRU instead learn from normalized rates "
                   "without the bounded anchor transform. The present comparisons cannot attribute their difference "
                   "to any single feature of that transform. Likewise, the representation perturbation bound does not "
                   "bound errors in the seven-dimensional decoder output.\n\n"
                   "The plain GRU searches context length and learning rate, whereas the three feature families search "
                   "code dimension and context at a fixed learning rate. Equal candidate counts therefore compare "
                   "specified search budgets, rather than optimization over identical hyperparameters. "
                   "The fixed-capacity controls share HDC's training settings and address the narrower representation comparison.")
    if hdc_leads_dropout:
        discussion += ("\n\nSevere input loss changed the mean ranking. At the prespecified 40\\% channel-dropout level, "
                       f"zero-label $R^2$ was {fmt(dropout_endpoint['hdc']['mean'])} for HDC, "
                       f"{fmt(dropout_endpoint['random']['mean'])} for random features, "
                       f"{fmt(dropout_endpoint['mlp']['mean'])} for MLP features, and "
                       f"{fmt(dropout_endpoint['plain']['mean'])} for the plain GRU. "
                       "All four families lost accuracy relative to their unperturbed zero-label condition. "
                       "HDC nevertheless retained the highest mean score at this specified stress level. "
                       "This crossover motivates evaluating missing-channel robustness separately from ordinary cross-session accuracy.")
    single_bin_only = all(summary["selection"][day]["families"]["hdc"][0]["config"]["temporal_bins"] == 1
                          for day in DAYS)
    representation = ("Channel-value phase codes provide an HDC input representation for causal intracortical regression. "
                      "All six HDC selections used a one-bin input with recurrent temporal state. "
                      if single_bin_only else
                      "Channel-value phase codes and lag-role binding provide an HDC input representation for causal intracortical regression. ")
    conclusion = representation + capacity_text
    if hdc_leads_dropout:
        conclusion += (" At the prespecified 40\\% channel-dropout level, HDC retained the highest observed "
                       f"zero-label mean among the four families ($R^2={fmt(dropout_endpoint['hdc']['mean'])}$).")
    supplement = []
    for regime in REGIMES:
        rows = [[display_labels[f], *[fmt(r["mean"]) for r in score[f,regime]["per_session"]]] for f in table_families]
        condition = "calibrated" if regime == "few_shot" else "zero-label"
        label = "tab:groups-" + regime
        supplement.append("Table~\\ref{" + label + "} gives each outer group's mean in the " + condition + " condition.\n" +
            table("Group means in the " + condition + " condition", ["Family", *DAYS], rows, label))
    zero_ablation_rows = []
    for row in summary["ablation_contrasts"]:
        if row["regime"] == "zero_shot":
            ci = row["ci95"]
            zero_ablation_rows.append([LABELS[row["comparator"]],
                fmt(score[row["comparator"], "zero_shot"]["mean"]),
                f"{row['mean_difference']:+.3f}", f"[{fmt(ci[0])}, {fmt(ci[1])}]"])
    supplement.append("Table~\\ref{tab:ablations-zero-label} reports component comparisons without evaluated-group labels.\n" +
        table("Conditional HDC component comparisons without evaluated-group labels",
        ["Condition", "$R^2$", "HDC difference", "Descriptive interval"],
        zero_ablation_rows, "tab:ablations-zero-label"))
    selection_rows=[]
    for day in DAYS:
        for family in FAMILIES:
            selected=summary["selection"][day]["families"][family][0]
            c=selected["config"]
            selection_rows.append([day,LABELS[family],c["dimension"] if family!="plain" else "--",c["temporal_bins"],f"{c['lr']:g}",selected["parameters"]])
    supplement.append("Table~\\ref{tab:selection} lists the configurations selected using inner validation and their trainable parameter counts.\n" +
        table("Selected configurations and trainable parameter counts",["Group","Family","$D$","$K$","Learning rate","Parameters"],selection_rows,"tab:selection"))
    transfer_rows=[[LABELS[r["family"]],"Calibrated" if r["regime"]=="few_shot" else "Zero-label",fmt(r["mean"]),fmt(r["session_sd"])] for r in summary["transfer"]]
    supplement.append("The additional seven-group analysis uses released held-out calibration recordings that had been available during development. It is distinct from private benchmark-server testing. The same prefix and suffix protocol is used. Table~\\ref{tab:transfer} summarizes performance across these groups.\n"+table("Additional released-group transfer analysis",["Family","Regime","Mean $R^2$","Group SD"],transfer_rows,"tab:transfer"))
    transfer_scores = {(r["family"], r["regime"]): r for r in summary["transfer"]}
    transfer_days = [f"S{i}" for i in range(6, 13)]
    for regime in REGIMES:
        transfer_group_rows = []
        for family in FAMILIES:
            per_group = {r["session"]: r["mean"]
                         for r in transfer_scores[family, regime]["per_session"]}
            transfer_group_rows.append([LABELS[family], *[fmt(per_group[d]) for d in transfer_days]])
        condition = "calibrated" if regime == "few_shot" else "zero-label"
        label = "tab:transfer-groups-" + regime
        supplement.append("Table~\\ref{" + label + "} gives the individual transfer-group means in the " + condition + " condition.\n" +
            table("Released transfer-group means in the " + condition + " condition",
            ["Family", *transfer_days], transfer_group_rows, label))
    matched_rows = []
    for family in ("random", "mlp"):
        for regime in REGIMES:
            contrast_row = matched_contrasts[family, regime]
            ci = contrast_row["ci95"]
            matched_rows.append([
                "Random features" if family == "random" else "MLP features",
                "Calibrated" if regime == "few_shot" else "Zero-label",
                fmt(score["hdc", regime]["mean"]),
                fmt(matched_scores[f"matched_{family}", regime]["mean"]),
                f"{contrast_row['mean_difference']:+.3f} [{fmt(ci[0])}, {fmt(ci[1])}]",
            ])
    supplement.append(
        f"The separately frozen capacity-matched matrix copied each group's inner-selected HDC dimension and temporal-bin count to both control families. "
        f"It contains 120 verified comparisons: {reused} reused completed primary fits and {newly_trained} required new fits. "
        "Intervals resample the six paired group effects and are descriptive. Table~\\ref{tab:matched-capacity} reports both labeling conditions.\n" +
        table("HDC-selected-capacity matched controls", ["Matched family", "Regime", "HDC $R^2$",
              "Matched $R^2$", "HDC minus matched (95\\% interval)"],
              matched_rows, "tab:matched-capacity", compact=True))
    supplement.append("The machine-readable evidence includes every group-by-seed score, conditional ablation, calibration dose, perturbation level, selected configuration, and source/checkpoint hash. Group resampling and exploratory sign-flip summaries are descriptive because the sessions belong to one participant and the folds share training data.")
    macros=dict(AbstractResults=abstract,ResultsBody=body,ResultDrivenDiscussion=discussion,
                ResultDrivenConclusion=conclusion,SupplementaryResults="\n\n".join(supplement))
    macro_path = MANUSCRIPT / "results_generated.tex"
    temporary_macro_path = macro_path.with_suffix(macro_path.suffix + ".tmp")
    temporary_macro_path.write_text("\n\n".join("\\newcommand{\\"+name+"}{%\n"+text+"\n}" for name,text in macros.items())+"\n")
    temporary_macro_path.replace(macro_path)
    csv_path=ROOT / "results/group_seed_scores.csv"
    with csv_path.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["family","regime","session","seed","r2"])
        for row in summary["scores"]:
            for group in row["per_session"]:
                for seed,value in zip(FINAL_SEEDS,group["seed_scores"]):
                    writer.writerow([row["family"],row["regime"],group["session"],seed,value])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only",action="store_true")
    args=parser.parse_args()
    if args.protocol_only:
        build_protocol()
        return
    # Invalidate the previous generated results block before auditing this
    # run, so a failed/incomplete matched-control analysis cannot leave stale
    # manuscript results available for compilation.
    for generated in (MANUSCRIPT / "results_generated.tex",
                      ROOT / "results/summary.json",
                      ROOT / "results/group_seed_scores.csv",
                      ROOT / "results/manuscript_generation.json"):
        if generated.exists():
            generated.unlink()
    summary=make_summary()
    build_protocol()
    write_manuscript(summary)
    summary_path = ROOT / "results/summary.json"
    generated_tex = ("results_generated.tex", "protocol_generated.tex", "theory_generated.tex")
    generation_manifest = dict(
        schema=1,
        complete=True,
        summary_sha256=sha(summary_path),
        report_sha256=sha(Path(__file__).resolve()),
        generated_tex_sha256={name:sha(MANUSCRIPT / name) for name in generated_tex},
    )
    manifest_path = ROOT / "results/manuscript_generation.json"
    write_json(manifest_path, generation_manifest)
    print(json.dumps(dict(state="complete",fit_count=sum(EXPECTED.values()),
                          summary=str(summary_path),manuscript_generation=str(manifest_path))))


if __name__=="__main__":
    main()

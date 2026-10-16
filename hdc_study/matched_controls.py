"""Prespecified, HDC-selected-capacity matched random/MLP controls.

This is a secondary analysis.  It reads the frozen primary study and writes
only to ``results/matched_controls``, ``results/matched_predictions``, and new
``checkpoints/matched_*.pt`` files.  It never changes primary fit records or
selection outputs.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

import run
from data import ROOT, load_h1, partition_day, sessions, sha, write_json
from engine import evaluate, fit, load
from model import Config, Decoder
from control_implementation import verify_control_implementation


FAMILIES = ("random", "mlp")
WORKERS = 3
EPOCHS = 50
CAPACITY_PROTOCOL = ROOT / "provenance/capacity_controls_protocol.json"
RESULT_DIR = ROOT / "results/matched_controls"
PREDICTION_DIR = ROOT / "results/matched_predictions"
_WORKER_STATE = None


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def primary_protocol_path(root=ROOT):
    return Path(root) / "provenance/protocol.json"


def capacity_protocol_path(root=ROOT):
    return Path(root) / "provenance/capacity_controls_protocol.json"


def result_identifier(day, family, seed):
    if day not in {f"S{i}" for i in range(6)} or family not in FAMILIES or seed not in run.FINAL_SEEDS:
        raise ValueError(f"Invalid matched-control key: {day}, {family}, {seed}")
    return f"matched_{day}_{family}_s{seed}"


def comparison_filename(day, family, seed):
    return f"{day}_{family}_s{seed}.json"


def matched_config(selected_hdc_config, family, seed):
    """Copy an inner-selected HDC config, changing family and replicate seed."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported matched-control family: {family}")
    config = dict(selected_hdc_config)
    if config.get("family") != "hdc":
        raise ValueError("Capacity matching requires an HDC-selected source config")
    config["family"] = family
    config["codebook_seed"] = 100000 + int(seed)
    return asdict(Config(**config))


def _outer_result_files(root):
    return sorted((Path(root) / "results/runs").glob("outer_*.json"))


def assert_no_outer_results(root=ROOT):
    """Freeze-time guard; accepts a root for temporary-directory CPU tests."""
    outer = _outer_result_files(root)
    if outer:
        raise RuntimeError(
            f"Capacity-control design must be frozen before outer outcomes; found {len(outer)} outer result files"
        )
    status_path = Path(root) / "results/status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("counts", {}).get("outer", 0) != 0:
            raise RuntimeError("Capacity-control design must be frozen before outer outcomes")
    return 0


def freeze_capacity_protocol(root=ROOT, *, primary_protocol_sha256=None,
                             core_fingerprint=None, module_sha256=None,
                             created_utc=None):
    """Write the pre-outer design once; parameters make the guard testable."""
    root = Path(root)
    assert_no_outer_results(root)
    source_protocol = primary_protocol_path(root)
    if not source_protocol.exists():
        raise FileNotFoundError(source_protocol)
    primary = json.loads(source_protocol.read_text())
    if primary_protocol_sha256 is None:
        primary_protocol_sha256 = sha(source_protocol)
    if core_fingerprint is None:
        if root.resolve() != ROOT.resolve():
            raise ValueError("A temporary freeze fixture must supply core_fingerprint")
        core_fingerprint = run.fingerprint()
    if module_sha256 is None:
        module_sha256 = sha(Path(__file__))
    if primary.get("fingerprint") != core_fingerprint:
        raise RuntimeError("Primary protocol and current core fingerprint differ")
    if primary_protocol_sha256 != sha(source_protocol):
        raise RuntimeError("Primary protocol hash changed while freezing capacity controls")
    if root.resolve() == ROOT.resolve():
        if run.fingerprint() != core_fingerprint:
            raise RuntimeError("Frozen core source or data manifest changed")
        if module_sha256 != sha(Path(__file__)):
            raise RuntimeError("Matched-control module hash changed while freezing")

    output_path = capacity_protocol_path(root)
    design = dict(
        schema=1,
        study_id="h1_channel_temporal_hdc_capacity_controls",
        created_utc=created_utc or utc_now(),
        design_state="frozen_before_outer_results",
        no_outer_results_at_design=True,
        outer_result_files_at_design=0,
        primary_status_outer_count_at_design=0,
        primary_protocol_sha256=primary_protocol_sha256,
        core_fingerprint=core_fingerprint,
        module_sha256=module_sha256,
        comparison_count=120,
        days=[f"S{i}" for i in range(6)],
        families={family: dict(comparisons_per_day=10, total=60) for family in FAMILIES},
        final_seeds=list(run.FINAL_SEEDS),
        epochs=EPOCHS,
        training_groups="The five held-in groups other than the evaluated outer group",
        configuration_rule=(
            "For each outer group, copy that group's HDC configuration selected using inner data; "
            "change only family to random or mlp and set codebook_seed to 100000 + final seed."
        ),
        selection_rule=(
            "Use the existing inner-only HDC family winner from results/selection.json. "
            "Outer metrics do not select or alter any matched configuration."
        ),
        evaluation_regimes=["zero_shot", "few_shot"],
        epochs_rule="Fixed 50 epochs, matching the frozen primary protocol.",
        primary_endpoint_unchanged=True,
        outcome_dependent_choices=False,
    )
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        immutable = {key: value for key, value in design.items()
                     if key not in ("created_utc", "module_sha256")}
        actual = {key: value for key, value in existing.items()
                  if key not in ("created_utc", "module_sha256")}
        if actual != immutable:
            raise RuntimeError("An existing capacity-control freeze differs; refusing to overwrite it")
        if existing.get("module_sha256") != design["module_sha256"]:
            # A source-only implementation refinement may refresh the bound
            # module hash while the no-outer guard still holds.  All design
            # fields remain byte-for-byte equivalent.
            write_json(output_path, design)
            return design
        return existing
    write_json(output_path, design)
    return design


def run_cpu_fixture_checks():
    """Small CPU-only checks for config copying, identifiers, and freeze timing."""
    fixture = asdict(Config(family="hdc", dimension=512, temporal_bins=8, lr=0.0002))
    random_config = matched_config(fixture, "random", 101)
    mlp_config = matched_config(fixture, "mlp", 101)
    for family, config in (("random", random_config), ("mlp", mlp_config)):
        expected = dict(fixture)
        expected["family"] = family
        expected["codebook_seed"] = 100101
        assert config == expected
    ids = {result_identifier(day, family, seed)
           for day in (f"S{i}" for i in range(6))
           for family in FAMILIES for seed in run.FINAL_SEEDS}
    assert len(ids) == 120
    reuse_spec = dict(id="matched_S0_random_s101", day="S0", family="random", seed=101,
        config={}, train_days=[f"S{i}" for i in range(1, 6)], test_days=["S0"],
        reused_identifier="outer_S0_random_s101")
    reuse_state = dict(primary_sha="fixture-primary-sha")
    reuse_metrics = [dict(day="S0", regime="zero_shot", r2=0.1, bins=10,
                          calibration_bins=0, score_start=3256, perturbation=None),
                     dict(day="S0", regime="few_shot", r2=0.2, bins=10,
                          calibration_bins=3000, score_start=3256, perturbation=None)]
    with tempfile.TemporaryDirectory(prefix="matched-controls-test-") as tmp:
        root = Path(tmp)
        (root / "provenance").mkdir()
        (root / "results/runs").mkdir(parents=True)
        (root / "results/predictions").mkdir(parents=True)
        write_json(root / "provenance/protocol.json", {"fingerprint": {"fixture": "core"}})
        prediction_hashes = {}
        for regime in ("zero_shot", "few_shot"):
            name = f"outer_S0_random_s101_{regime}.npz"
            prediction = root / "results/predictions" / name
            prediction.write_bytes(("fixture-" + regime).encode())
            prediction_hashes[name] = sha(prediction)
        reuse_row = dict(id="outer_S0_random_s101", signature=dict(
            config={}, seed=101, train_days=[f"S{i}" for i in range(1, 6)],
            test_days=["S0"], protocol_sha256="fixture-primary-sha", fingerprint={"fixture": "core"}),
            metrics=reuse_metrics, prediction_sha256=prediction_hashes)
        _preflight_primary_reuse_artifacts(reuse_spec, reuse_state, reuse_row, root=root)
        reuse_row["prediction_sha256"]["outer_S0_random_s101_zero_shot.npz"] = "bad-hash"
        try:
            _preflight_primary_reuse_artifacts(reuse_spec, reuse_state, reuse_row, root=root)
        except RuntimeError as exc:
            assert "prediction hash mismatch" in str(exc)
        else:
            raise AssertionError("Reuse preflight accepted a mismatched prediction hash")
        psha = sha(root / "provenance/protocol.json")
        frozen = freeze_capacity_protocol(
            root, primary_protocol_sha256=psha,
            core_fingerprint={"fixture": "core"}, module_sha256="fixture-module-sha",
            created_utc="2026-01-01T00:00:00Z")
        assert frozen["no_outer_results_at_design"] is True
        assert frozen["comparison_count"] == 120
        (root / "results/runs/outer_S0_hdc_s101.json").write_text("{}\n")
        try:
            freeze_capacity_protocol(
                root, primary_protocol_sha256=psha,
                core_fingerprint={"fixture": "core"}, module_sha256="fixture-module-sha")
        except RuntimeError as exc:
            assert "before outer outcomes" in str(exc)
        else:
            raise AssertionError("Freeze accepted an outer result in its temporary fixture")
    return {"config_matching": "passed", "unique_ids": 120,
            "temporary_reuse_prediction_preflight": "passed",
            "temporary_outer_freeze_guard": "passed"}


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON: {path}") from exc


def _expected_config(config, seed):
    return asdict(replace(config, codebook_seed=100000 + seed))


def _checkpoint_path(identifier):
    return ROOT / "checkpoints" / f"{identifier}.pt"


def _validate_fit_row(identifier, row, config, seed, train_days, test_days,
                      stage, primary_sha, core_fingerprint, verify_checkpoint=True):
    if row.get("id") != identifier or row.get("stage") != stage:
        raise RuntimeError(f"Primary result identity/stage mismatch: {identifier}")
    signature = row.get("signature", {})
    expected = dict(config=_expected_config(config, seed), seed=seed,
                    train_days=sorted(train_days, key=lambda d: int(d[1:])),
                    test_days=sorted(test_days, key=lambda d: int(d[1:])),
                    protocol_sha256=primary_sha, fingerprint=core_fingerprint)
    for key, value in expected.items():
        actual = signature.get(key)
        if key in ("train_days", "test_days"):
            if set(actual or ()) != set(value):
                raise RuntimeError(f"Primary {key} mismatch: {identifier}")
        elif actual != value:
            raise RuntimeError(f"Primary {key} mismatch: {identifier}")
    if not isinstance(row.get("metrics"), list) or not row["metrics"]:
        raise RuntimeError(f"Missing primary metrics: {identifier}")
    checkpoint = _checkpoint_path(identifier)
    if verify_checkpoint:
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing primary checkpoint: {identifier}")
        checkpoint_sha = sha(checkpoint)
        if checkpoint_sha != row.get("checkpoint_sha256"):
            raise RuntimeError(f"Primary checkpoint hash mismatch: {identifier}")
    return row


def _compare_selection_entry(actual, expected, where):
    if actual.get("config") != expected["config"]:
        raise RuntimeError(f"Inner-only selection config mismatch: {where}")
    if actual.get("key") != expected["key"] or actual.get("parameters") != expected["parameters"]:
        raise RuntimeError(f"Inner-only selection key/parameter mismatch: {where}")
    if not math.isclose(float(actual.get("score", float("nan"))), expected["score"],
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Inner-only selection score mismatch: {where}")


def _recompute_selection(days, configs, rows):
    """Recreate the primary selection solely from committed inner metrics."""
    result = {}
    for outer in days:
        by_family = {}
        for family in run.FAMILIES:
            ranking = []
            for config in (c for c in configs if c.family == family):
                values = []
                for inner in days:
                    if inner == outer:
                        continue
                    a, b = sorted((outer, inner), key=lambda day: int(day[1:]))
                    for seed in run.DEV_SEEDS:
                        identifier = f"inner_{a}_{b}_{run.config_key(config)}_s{seed}"
                        row = rows[identifier]
                        scores = [m["r2"] for m in row["metrics"]
                                  if m.get("day") == inner and m.get("regime") == "few_shot"]
                        if len(scores) != 1 or not np.isfinite(scores[0]):
                            raise RuntimeError(f"Invalid inner selection score: {identifier}/{inner}")
                        values.append(float(scores[0]))
                ranking.append(dict(config=asdict(config), key=run.config_key(config),
                                    score=float(np.mean(values)),
                                    parameters=Decoder(config).parameter_count()))
            by_family[family] = sorted(
                ranking, key=lambda item: (-item["score"], item["parameters"], item["key"]))
        control = sorted((by_family[family][0] for family in ("random", "mlp")),
                         key=lambda item: (-item["score"], item["parameters"], item["key"]))[0]
        result[outer] = dict(families=by_family, selected_control=control["config"]["family"])
    return result


def _validate_selection(saved, recomputed, days):
    if set(saved) != set(days):
        raise RuntimeError("Selection file does not cover all outer groups")
    for day in days:
        if saved[day].get("selected_control") != recomputed[day]["selected_control"]:
            raise RuntimeError(f"Selected control family differs from inner-only recomputation: {day}")
        for family in run.FAMILIES:
            actual = saved[day].get("families", {}).get(family, [])
            expected = recomputed[day]["families"][family]
            if len(actual) != len(expected):
                raise RuntimeError(f"Selection candidate count mismatch: {day}/{family}")
            for i, (observed, wanted) in enumerate(zip(actual, expected)):
                _compare_selection_entry(observed, wanted, f"{day}/{family}/rank{i + 1}")


def _recompute_transfer_selection(days, configs, selection):
    result = {}
    for family in run.FAMILIES:
        ranking = []
        for config in (c for c in configs if c.family == family):
            entries = [next(item for item in selection[day]["families"][family]
                            if item["key"] == run.config_key(config)) for day in days]
            ranking.append(dict(config=asdict(config),
                                score=float(np.mean([item["score"] for item in entries])),
                                key=run.config_key(config),
                                parameters=entries[0]["parameters"]))
        result[family] = sorted(
            ranking, key=lambda item: (-item["score"], item["parameters"], item["key"]))[0]
    return result


def _validate_transfer_selection(saved, recomputed):
    for family in run.FAMILIES:
        if family not in saved:
            raise RuntimeError(f"Missing transfer selection for {family}")
        _compare_selection_entry(saved[family], recomputed[family], f"transfer/{family}")


def _expected_primary_specs(days, configs, selection, transfer_selection, heldout_days):
    expected = {}
    for a, b in itertools.combinations(days, 2):
        train_days = set(days) - {a, b}
        test_days = {a, b}
        for config in configs:
            for seed in run.DEV_SEEDS:
                identifier = f"inner_{a}_{b}_{run.config_key(config)}_s{seed}"
                expected[identifier] = (config, seed, train_days, test_days, "inner")
    for day in days:
        variants = {family: Config(**selection[day]["families"][family][0]["config"])
                    for family in run.FAMILIES}
        variants.update(run.ablations(Config(**selection[day]["families"]["hdc"][0]["config"])))
        for family, config in variants.items():
            for seed in run.FINAL_SEEDS:
                identifier = f"outer_{day}_{family}_s{seed}"
                expected[identifier] = (config, seed, set(days) - {day}, {day}, "outer")
    for family, selected in transfer_selection.items():
        config = Config(**selected["config"])
        for seed in run.FINAL_SEEDS:
            identifier = f"transfer_{family}_s{seed}"
            expected[identifier] = (config, seed, set(days), set(heldout_days), "transfer")
    return expected


def verify_primary_matrix():
    """CPU parent preflight: hashes, full matrix, and inner-only selection."""
    primary_path = primary_protocol_path()
    capacity_path = capacity_protocol_path()
    if not capacity_path.exists():
        raise RuntimeError("Missing pre-outer capacity-control freeze; run --freeze first")
    capacity = _json(capacity_path)
    primary_sha = sha(primary_path)
    primary = _json(primary_path)
    core_fingerprint = run.fingerprint()
    module_sha = verify_control_implementation(ROOT)
    if capacity.get("primary_protocol_sha256") != primary_sha:
        raise RuntimeError("Primary protocol hash differs from frozen capacity design")
    if capacity.get("core_fingerprint") != core_fingerprint or primary.get("fingerprint") != core_fingerprint:
        raise RuntimeError("Core fingerprint differs from frozen capacity design")
    if capacity.get("no_outer_results_at_design") is not True:
        raise RuntimeError("Capacity design lacks a verified pre-outer freeze")

    protocol_sha = sha(capacity_path)
    days = [f"S{i}" for i in range(6)]
    if list(primary.get("final_seeds", [])) != run.FINAL_SEEDS:
        raise RuntimeError("Final seeds differ from the primary frozen protocol")
    configs = [Config(**item) for item in primary["candidate_configs"]]
    selection_path = ROOT / "results/selection.json"
    transfer_path = ROOT / "results/transfer_selection.json"
    if not selection_path.exists() or not transfer_path.exists():
        raise RuntimeError("Primary inner-only selection artifacts are incomplete")
    saved_selection = _json(selection_path)
    recomputed_selection = _recompute_selection(days, configs,
        _load_and_validate_inner(primary_sha, core_fingerprint, configs, days))
    _validate_selection(saved_selection, recomputed_selection, days)
    saved_transfer = _json(transfer_path)
    recomputed_transfer = _recompute_transfer_selection(days, configs, recomputed_selection)
    _validate_transfer_selection(saved_transfer, recomputed_transfer)

    heldout_days = sessions(load_h1("held-out-calib", verify=True))
    expected = _expected_primary_specs(days, configs, saved_selection, saved_transfer, heldout_days)
    run_dir = ROOT / "results/runs"
    paths = {path.stem: path for path in run_dir.glob("*.json")}
    if set(paths) != set(expected):
        missing = sorted(set(expected) - set(paths))[:5]
        extra = sorted(set(paths) - set(expected))[:5]
        raise RuntimeError(f"Primary result matrix is incomplete or unexpected; missing={missing}, extra={extra}")
    rows = {}
    for identifier, (config, seed, train_days, test_days, stage) in expected.items():
        row = _json(paths[identifier])
        _validate_fit_row(identifier, row, config, seed, train_days, test_days, stage,
                          primary_sha, core_fingerprint)
        rows[identifier] = row

    # Verify that each primary HDC outer fit implements the HDC config selected
    # for that group from inner data. Its outer score is never read here.
    for day in days:
        selected_hdc = saved_selection[day]["families"]["hdc"][0]["config"]
        for seed in run.FINAL_SEEDS:
            identifier = f"outer_{day}_hdc_s{seed}"
            expected_hdc = _expected_config(Config(**selected_hdc), seed)
            if rows[identifier]["signature"]["config"] != expected_hdc:
                raise RuntimeError(f"HDC outer fit does not match inner-selected source config: {identifier}")

    return dict(capacity=capacity, primary=primary, primary_sha=primary_sha,
                capacity_sha=protocol_sha, core_fingerprint=core_fingerprint,
                module_sha256=module_sha, days=days, configs=configs,
                selection=saved_selection, rows=rows, heldout_days=heldout_days)


def _load_and_validate_inner(primary_sha, core_fingerprint, configs, days):
    """Read/hash-check the inner matrix before recomputing selection."""
    rows = {}
    for a, b in itertools.combinations(days, 2):
        for config in configs:
            for seed in run.DEV_SEEDS:
                identifier = f"inner_{a}_{b}_{run.config_key(config)}_s{seed}"
                path = ROOT / "results/runs" / f"{identifier}.json"
                if not path.exists():
                    raise RuntimeError(f"Missing inner fit required for preselection validation: {identifier}")
                row = _json(path)
                _validate_fit_row(identifier, row, config, seed, set(days) - {a, b},
                                  {a, b}, "inner", primary_sha, core_fingerprint)
                rows[identifier] = row
    if len(rows) != 1200:
        raise RuntimeError(f"Expected 1200 primary inner fits, found {len(rows)}")
    return rows


def _task_specs(preflight):
    specs = []
    for day in preflight["days"]:
        selected_hdc = preflight["selection"][day]["families"]["hdc"][0]["config"]
        train_days = sorted(set(preflight["days"]) - {day}, key=lambda d: int(d[1:]))
        for family in FAMILIES:
            for seed in run.FINAL_SEEDS:
                config = matched_config(selected_hdc, family, seed)
                source_id = f"outer_{day}_hdc_s{seed}"
                source_row = preflight["rows"][source_id]
                source_checkpoint = _checkpoint_path(source_id)
                if (source_row["signature"]["config"] != _expected_config(Config(**selected_hdc), seed)
                        or source_row["checkpoint_sha256"] != sha(source_checkpoint)):
                    raise RuntimeError(f"Selected HDC source fit/checkpoint mismatch: {source_id}")
                identifier = result_identifier(day, family, seed)
                expected_signature = dict(
                    config=config, seed=seed, train_days=train_days,
                    test_days=[day], protocol_sha256=preflight["primary_sha"],
                    fingerprint=preflight["core_fingerprint"])
                # Reuse only the same outer family result, after exact config,
                # split, seed, protocol, fingerprint, and checkpoint checks.
                candidate_id = f"outer_{day}_{family}_s{seed}"
                candidate = preflight["rows"].get(candidate_id)
                reusable = None
                if candidate is not None:
                    sig = candidate["signature"]
                    if (sig.get("config") == config and sig.get("seed") == seed
                            and set(sig.get("train_days", [])) == set(train_days)
                            and set(sig.get("test_days", [])) == {day}
                            and sig.get("protocol_sha256") == preflight["primary_sha"]
                            and sig.get("fingerprint") == preflight["core_fingerprint"]
                            and candidate.get("checkpoint_sha256") == sha(_checkpoint_path(candidate_id))):
                        reusable = candidate_id
                specs.append(dict(
                    id=identifier, day=day, family=family, seed=seed, config=config,
                    train_days=train_days, test_days=[day], expected_signature=expected_signature,
                    matched_hdc_identifier=source_id, matched_hdc_checkpoint_sha256=source_row["checkpoint_sha256"],
                    matched_hdc_config=source_row["signature"]["config"],
                    reused_identifier=reusable))
    if len(specs) != 120 or len({s["id"] for s in specs}) != 120:
        raise RuntimeError("Matched-control task matrix is not exactly 120 unique comparisons")
    state = dict(primary_sha=preflight["primary_sha"])
    for spec in specs:
        if spec["reused_identifier"] is not None:
            source_row = preflight["rows"][spec["reused_identifier"]]
            _preflight_primary_reuse_artifacts(spec, state, source_row)
    return specs


def _preflight_primary_reuse_artifacts(spec, state, row, root=ROOT):
    """CPU check all artifacts for an exact primary-fit reuse before spawning."""
    identifier = spec.get("reused_identifier")
    if identifier is None:
        return
    signature = row.get("signature", {})
    if (row.get("id") != identifier
            or signature.get("config") != spec["config"]
            or signature.get("seed") != spec["seed"]
            or set(signature.get("train_days", [])) != set(spec["train_days"])
            or set(signature.get("test_days", [])) != set(spec["test_days"])
            or signature.get("protocol_sha256") != state["primary_sha"]):
        raise RuntimeError(f"Primary reuse signature mismatch before spawn: {identifier}")
    metric_rows = [metric for metric in row.get("metrics", [])
                   if metric.get("day") == spec["day"]]
    metric_map = {metric.get("regime"): metric for metric in metric_rows}
    if set(metric_map) != {"zero_shot", "few_shot"}:
        raise RuntimeError(f"Primary reuse metric rows are incomplete before spawn: {identifier}")
    for regime, calibration_bins in (("zero_shot", 0), ("few_shot", 3000)):
        metric = metric_map[regime]
        if (not isinstance(metric.get("r2"), (int, float))
                or not np.isfinite(metric["r2"])
                or not isinstance(metric.get("bins"), int) or metric["bins"] <= 0
                or metric.get("calibration_bins") != calibration_bins
                or metric.get("score_start") != 3256
                or metric.get("perturbation") is not None):
            raise RuntimeError(f"Primary reuse metric row is invalid before spawn: {identifier}/{regime}")
    hashes = row.get("prediction_sha256", {})
    root = Path(root)
    for regime in ("zero_shot", "few_shot"):
        filename = f"{identifier}_{regime}.npz"
        prediction_path = root / "results/predictions" / filename
        expected_sha = hashes.get(filename)
        if (not prediction_path.is_file() or not expected_sha
                or sha(prediction_path) != expected_sha):
            raise RuntimeError(f"Primary reuse prediction hash mismatch before spawn: {identifier}/{regime}")


def _assert_job_fingerprints(state):
    if sha(primary_protocol_path()) != state["primary_sha"]:
        raise RuntimeError("Primary protocol changed during matched-control job")
    if sha(CAPACITY_PROTOCOL) != state["capacity_sha"]:
        raise RuntimeError("Capacity-control protocol changed during matched-control job")
    if run.fingerprint() != state["core_fingerprint"]:
        raise RuntimeError("Core source/data fingerprint changed during matched-control job")
    if sha(Path(__file__)) != state["module_sha256"]:
        raise RuntimeError("Matched-control source changed during matched-control job")


def _validate_checkpoint_contents(path, config, seed, train_days, records):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected_files = sorted((r["path"], r["sha256"]) for r in records)
    actual_files = sorted((r["path"], r["sha256"]) for r in saved.get("train_files", []))
    if (saved.get("schema") != 1 or saved.get("config") != config or saved.get("seed") != seed
            or set(saved.get("train_days", [])) != set(train_days)
            or actual_files != expected_files or saved.get("epochs") != EPOCHS):
        raise RuntimeError(f"Checkpoint contents do not match the matched-control design: {path.name}")
    return saved


def _valid_existing_result(path, spec, state):
    row = _json(path)
    expected = spec["expected_signature"]
    if (row.get("schema") != 1 or row.get("id") != spec["id"]
            or row.get("day") != spec["day"] or row.get("family") != spec["family"]
            or row.get("seed") != spec["seed"] or row.get("config") != spec["config"]
            or row.get("protocol_sha256") != state["primary_sha"]
            or row.get("capacity_protocol_sha256") != state["capacity_sha"]
            or row.get("reused_identifier") != spec["reused_identifier"]
            or row.get("matched_hdc_identifier") != spec["matched_hdc_identifier"]
            or row.get("matched_hdc_config") != spec["matched_hdc_config"]):
        raise RuntimeError(f"Existing matched-control result is incompatible: {spec['id']}")
    checkpoint = row.get("checkpoint", {})
    checkpoint_path = ROOT / checkpoint.get("path", "")
    if (not checkpoint_path.is_file() or checkpoint.get("sha256") != sha(checkpoint_path)):
        raise RuntimeError(f"Existing matched-control checkpoint hash mismatch: {spec['id']}")
    if row.get("parameters") is None or not isinstance(row.get("metrics"), list):
        raise RuntimeError(f"Existing matched-control result is incomplete: {spec['id']}")
    expected_metric_keys = {(spec["day"], regime) for regime in ("zero_shot", "few_shot")}
    actual_metric_keys = {(m.get("day"), m.get("regime")) for m in row["metrics"]}
    if actual_metric_keys != expected_metric_keys:
        raise RuntimeError(f"Existing matched-control metrics are incomplete: {spec['id']}")
    predictions = row.get("predictions", {})
    if set(predictions) != {"zero_shot", "few_shot"}:
        raise RuntimeError(f"Existing matched-control prediction manifest is incomplete: {spec['id']}")
    expected_prediction_hashes = {regime: prediction.get("sha256")
                                 for regime, prediction in predictions.items()}
    if row.get("prediction_sha256") != expected_prediction_hashes:
        raise RuntimeError(f"Prediction hash map differs from prediction references: {spec['id']}")
    for regime, prediction in predictions.items():
        prediction_path = ROOT / prediction.get("path", "")
        if not prediction_path.is_file() or prediction.get("sha256") != sha(prediction_path):
            raise RuntimeError(f"Existing matched-control prediction hash mismatch: {spec['id']}/{regime}")
    return row


def _verify_prediction_against_cache(prediction_ref, day, regime, metrics, records):
    from sklearn.metrics import r2_score

    path = ROOT / prediction_ref["path"]
    day_records = [record for record in records if record["day"] == day]
    prefix = 0 if regime == "zero_shot" else 3000
    parts = partition_day(day_records, prefix=prefix)
    expected_target = np.concatenate([part["y"] for part in parts])
    expected_mask = np.concatenate([part["score_mask"] for part in parts])
    expected_lengths = [sum(len(part["x"]) for part in parts)]
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        target = archive["target"]
        mask = archive["mask"].astype(bool)
        lengths = archive["lengths"].astype(int).tolist()
        days = archive["days"].astype(str).tolist()
    if (days != [day] or lengths != expected_lengths
            or not np.array_equal(target, expected_target)
            or not np.array_equal(mask, expected_mask)
            or prediction.shape != expected_target.shape
            or not np.isfinite(prediction).all()):
        raise RuntimeError(f"Prediction arrays do not match verified H1 cache: {path.name}")
    metric = next((item for item in metrics
                   if item.get("day") == day and item.get("regime") == regime), None)
    if metric is None:
        raise RuntimeError(f"Missing metric row for {day}/{regime}")
    score = float(r2_score(target[mask], prediction[mask], multioutput="variance_weighted"))
    expected_bins = int(mask.sum())
    if (not math.isclose(float(metric.get("r2", float("nan"))), score,
                         rel_tol=0.0, abs_tol=1e-10)
            or metric.get("bins") != expected_bins
            or metric.get("calibration_bins") != prefix
            or metric.get("score_start") != 3256):
        raise RuntimeError(f"Prediction-derived metric differs from stored metric: {path.name}")
    return dict(path=str(path.relative_to(ROOT)), sha256=prediction_ref["sha256"],
                day=day, regime=regime, r2=score, bins=expected_bins)


def verify_completed_matrix(preflight=None, *, require_release_artifacts=True):
    """Validate every matched row, checkpoint, prediction, and provenance.

    The return value is suitable for report consumers.  ``preflight`` may be
    the result of ``verify_primary_matrix()`` to avoid repeating its primary
    matrix and selection checks.
    """
    preflight = verify_primary_matrix() if preflight is None else preflight
    state = dict(primary_sha=preflight["primary_sha"],
                 capacity_sha=preflight["capacity_sha"],
                 core_fingerprint=preflight["core_fingerprint"],
                 module_sha256=preflight["module_sha256"])
    specs = _task_specs(preflight)
    records = load_h1("held-in-calib", verify=True)
    rows = {}
    for spec in specs:
        path = RESULT_DIR / comparison_filename(spec["day"], spec["family"], spec["seed"])
        row = _valid_existing_result(path, spec, state)
        if row["config"] != matched_config(spec["matched_hdc_config"], spec["family"], spec["seed"]):
            raise RuntimeError(f"Control config is not an exact family-only HDC match: {spec['id']}")
        checkpoint_path = ROOT / row["checkpoint"]["path"]
        checkpoint = _validate_checkpoint_contents(checkpoint_path, spec["config"],
            spec["seed"], spec["train_days"],
            [record for record in records if record["day"] in set(spec["train_days"])])
        if int(checkpoint.get("parameters", -1)) != int(row["parameters"]):
            raise RuntimeError(f"Matched parameter count differs from checkpoint: {spec['id']}")
        source_id = spec["matched_hdc_identifier"]
        source = preflight["rows"][source_id]
        if (source["signature"]["config"] != spec["matched_hdc_config"]
                or source["checkpoint_sha256"] != spec["matched_hdc_checkpoint_sha256"]):
            raise RuntimeError(f"Selected HDC source provenance mismatch: {spec['id']}")
        audit_predictions = {}
        for regime in ("zero_shot", "few_shot"):
            audit_predictions[regime] = _verify_prediction_against_cache(
                row["predictions"][regime], spec["day"], regime, row["metrics"], records)
        row["prediction_audit"] = audit_predictions
        rows[row["id"]] = row
    if len(rows) != 120:
        raise RuntimeError(f"Expected 120 validated comparisons, found {len(rows)}")
    if require_release_artifacts:
        status_path = RESULT_DIR / "status.json"
        summary_path = RESULT_DIR / "summary.json"
        if not status_path.exists() or not summary_path.exists():
            raise RuntimeError("Matched-control status or summary artifact is missing")
        status = _json(status_path)
        summary = _json(summary_path)
        if (status.get("state") != "complete" or status.get("completed") != 120
                or status.get("expected") != 120
                or status.get("family_counts") != {"random": 60, "mlp": 60}):
            raise RuntimeError("Matched-control master status is not complete")
        expected_summary = dict(
            schema=1, primary_protocol_sha256=preflight["primary_sha"],
            capacity_protocol_sha256=preflight["capacity_sha"],
            core_fingerprint=preflight["core_fingerprint"],
            module_sha256=preflight["module_sha256"], comparison_count=120,
            family_counts={"random": 60, "mlp": 60}, metric_rows_per_comparison=2)
        for key, value in expected_summary.items():
            if summary.get(key) != value:
                raise RuntimeError(f"Matched-control summary provenance/count mismatch: {key}")
        summary_rows = summary.get("comparisons", [])
        if len(summary_rows) != 120 or {row.get("id") for row in summary_rows} != set(rows):
            raise RuntimeError("Matched-control summary does not contain the validated result matrix")
        for summary_row in summary_rows:
            result_row = _json(RESULT_DIR / comparison_filename(
                summary_row.get("day"), summary_row.get("family"), summary_row.get("seed")))
            if summary_row != result_row:
                raise RuntimeError(f"Summary comparison differs from per-record JSON: {summary_row.get('id')}")
    return dict(capacity=preflight["capacity"], primary_sha256=preflight["primary_sha"],
                capacity_sha256=preflight["capacity_sha"],
                core_fingerprint=preflight["core_fingerprint"],
                module_sha256=preflight["module_sha256"], rows=rows)


def _primary_reuse(spec, state, records):
    identifier = spec.get("reused_identifier")
    if not identifier:
        return None
    path = ROOT / "results/runs" / f"{identifier}.json"
    row = _json(path)
    signature = row["signature"]
    if (signature.get("config") != spec["config"]
            or signature.get("seed") != spec["seed"]
            or set(signature.get("train_days", [])) != set(spec["train_days"])
            or set(signature.get("test_days", [])) != set(spec["test_days"])
            or signature.get("protocol_sha256") != state["primary_sha"]
            or signature.get("fingerprint") != state["core_fingerprint"]):
        raise RuntimeError(f"Primary result is not an exact matched control: {identifier}")
    checkpoint = _checkpoint_path(identifier)
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != row.get("checkpoint_sha256"):
        raise RuntimeError(f"Primary reusable checkpoint hash mismatch: {identifier}")
    saved = _validate_checkpoint_contents(checkpoint, spec["config"], spec["seed"],
                                          spec["train_days"], records)
    metrics = [m for m in row["metrics"] if m.get("day") == spec["day"]]
    if {(m.get("day"), m.get("regime")) for m in metrics} != {
            (spec["day"], "zero_shot"), (spec["day"], "few_shot")}:
        raise RuntimeError(f"Primary reusable metrics are incomplete: {identifier}")
    primary_predictions = row.get("prediction_sha256", {})
    predictions = {}
    for regime in ("zero_shot", "few_shot"):
        filename = f"{identifier}_{regime}.npz"
        prediction_path = ROOT / "results/predictions" / filename
        expected_sha = primary_predictions.get(filename)
        if not prediction_path.is_file() or expected_sha != sha(prediction_path):
            raise RuntimeError(f"Primary reusable prediction hash mismatch: {identifier}/{regime}")
        predictions[regime] = dict(path=str(prediction_path.relative_to(ROOT)), sha256=expected_sha)
    return dict(metrics=metrics, parameters=row["parameters"],
                checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_sha,
                predictions=predictions,
                elapsed_seconds=float(saved.get("elapsed_seconds", row.get("train_seconds", 0.0))),
                saved=saved)


def _execute_task(spec):
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("Matched-control worker was not initialized")
    _assert_job_fingerprints(state)
    records = state["records"]
    day, family, seed = spec["day"], spec["family"], spec["seed"]
    train = [r for r in records if r["day"] in set(spec["train_days"])]
    test = [r for r in records if r["day"] in set(spec["test_days"])]
    if sessions(train) != spec["train_days"] or sessions(test) != spec["test_days"]:
        raise RuntimeError(f"Unexpected held-in split for {spec['id']}")
    result_path = RESULT_DIR / comparison_filename(day, family, seed)
    if result_path.exists():
        row = _valid_existing_result(result_path, spec, state)
        _assert_job_fingerprints(state)
        return row["id"]

    started = time.perf_counter()
    reusable = _primary_reuse(spec, state, train)
    if reusable is not None:
        metrics = reusable["metrics"]
        parameters = reusable["parameters"]
        checkpoint_path = reusable["checkpoint_path"]
        checkpoint_sha = reusable["checkpoint_sha256"]
        predictions = reusable["predictions"]
        elapsed_seconds = reusable["elapsed_seconds"]
        reused_identifier = spec["reused_identifier"]
    else:
        checkpoint_path = ROOT / "checkpoints" / f"{spec['id']}.pt"
        if checkpoint_path.exists():
            checkpoint_record = _validate_checkpoint_contents(
                checkpoint_path, spec["config"], seed, spec["train_days"], train)
            model, record = load(checkpoint_path)
            # The retained checkpoint is a completed fit from a previous
            # interrupted comparison job; evaluation can safely resume.
            if record["config"] != spec["config"] or record["seed"] != seed:
                raise RuntimeError(f"Retained matched checkpoint config mismatch: {spec['id']}")
        else:
            model, record = fit(train, Config(**spec["config"]), seed,
                                checkpoint_path, epochs=EPOCHS)
        if record.get("config") != spec["config"] or record.get("seed") != seed:
            raise RuntimeError(f"New matched fit returned an unexpected config: {spec['id']}")
        if set(record.get("train_days", [])) != set(spec["train_days"]):
            raise RuntimeError(f"New matched fit used unexpected groups: {spec['id']}")
        if record.get("epochs") != EPOCHS:
            raise RuntimeError(f"New matched fit did not use exactly {EPOCHS} epochs: {spec['id']}")
        metrics = []
        predictions = {}
        for regime in ("zero_shot", "few_shot"):
            prediction_path = PREDICTION_DIR / f"{spec['id']}_{regime}.npz"
            metrics.extend(evaluate(model, record, test, regime, save=prediction_path))
            predictions[regime] = dict(
                path=str(prediction_path.relative_to(ROOT)), sha256=sha(prediction_path))
        parameters = int(record["parameters"])
        checkpoint_sha = sha(checkpoint_path)
        elapsed_seconds = float(record["elapsed_seconds"])
        reused_identifier = None

    _assert_job_fingerprints(state)
    result = dict(
        schema=1,
        id=spec["id"],
        day=day,
        family=family,
        seed=seed,
        config=spec["config"],
        protocol_sha256=state["primary_sha"],
        capacity_protocol_sha256=state["capacity_sha"],
        matched_hdc_identifier=spec["matched_hdc_identifier"],
        matched_hdc_config=spec["matched_hdc_config"],
        checkpoint=dict(path=str(checkpoint_path.relative_to(ROOT)), sha256=checkpoint_sha),
        reused_identifier=reused_identifier,
        predictions=predictions,
        prediction_sha256={regime: entry["sha256"] for regime, entry in predictions.items()},
        metrics=metrics,
        parameters=parameters,
        seconds=float(elapsed_seconds),
    )
    write_json(result_path, result)
    return result["id"]


def _worker_init(primary_sha, capacity_sha, core_fingerprint, module_sha256):
    global _WORKER_STATE
    current = dict(primary_sha=primary_sha, capacity_sha=capacity_sha,
                   core_fingerprint=core_fingerprint, module_sha256=module_sha256)
    _assert_job_fingerprints(current)
    records = load_h1("held-in-calib", verify=True)
    if sessions(records) != [f"S{i}" for i in range(6)]:
        raise RuntimeError("Unexpected held-in groups in matched-control worker")
    current["records"] = records
    _WORKER_STATE = current


def _status(state, done, active, *, status="running", error=None):
    rows = []
    if RESULT_DIR.exists():
        for path in RESULT_DIR.glob("S[0-9]_*_s*.json"):
            try:
                row = _json(path)
            except (OSError, RuntimeError):
                continue
            rows.append(row)
    counts = {family: sum(row.get("family") == family for row in rows) for family in FAMILIES}
    fields = dict(
        schema=1, updated_utc=utc_now(), state=status, workers=WORKERS,
        primary_protocol_sha256=state["primary_sha"],
        capacity_protocol_sha256=state["capacity_sha"],
        completed=len(rows), expected=120, family_counts=counts,
        active_jobs=list(active),
    )
    if error is not None:
        fields["error"] = error
    write_json(RESULT_DIR / "status.json", fields)


def _write_summary(preflight):
    verified = verify_completed_matrix(preflight, require_release_artifacts=False)
    rows = [dict(row) for row in verified["rows"].values()]
    rows.sort(key=lambda row: (int(row["day"][1:]), FAMILIES.index(row["family"]), row["seed"]))
    summary = dict(
        schema=1,
        primary_protocol_sha256=preflight["primary_sha"],
        capacity_protocol_sha256=preflight["capacity_sha"],
        core_fingerprint=preflight["core_fingerprint"],
        module_sha256=preflight["module_sha256"],
        comparison_count=len(rows),
        family_counts={family: sum(row["family"] == family for row in rows) for family in FAMILIES},
        metric_rows_per_comparison=2,
        predictions_verified=240,
        comparisons=[{key: value for key, value in row.items() if key != "prediction_audit"}
                     for row in rows],
    )
    if len(rows) != 120 or summary["family_counts"] != {"random": 60, "mlp": 60}:
        raise RuntimeError("Cannot summarize an incomplete matched-control matrix")
    write_json(RESULT_DIR / "summary.json", summary)
    return summary


def run_matrix(workers=None):
    # This complete CPU verification happens before any spawn worker can use a GPU.
    workers = WORKERS if workers is None else workers
    if workers != WORKERS:
        raise ValueError(f"Matched capacity controls use exactly {WORKERS} spawn workers")
    preflight = verify_primary_matrix()
    specs = _task_specs(preflight)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    state = dict(primary_sha=preflight["primary_sha"],
                 capacity_sha=preflight["capacity_sha"],
                 core_fingerprint=preflight["core_fingerprint"],
                 module_sha256=preflight["module_sha256"])
    _assert_job_fingerprints(state)
    done = 0
    pending = {}
    context = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(
                max_workers=WORKERS, mp_context=context,
                initializer=_worker_init,
                initargs=(state["primary_sha"], state["capacity_sha"],
                          state["core_fingerprint"], state["module_sha256"])) as pool:
            iterator = iter(specs)
            while len(pending) < WORKERS:
                spec = next(iterator, None)
                if spec is None:
                    break
                pending[pool.submit(_execute_task, spec)] = spec
            _status(state, done, [s["id"] for s in pending.values()])
            last_status = time.monotonic()
            while pending:
                finished, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                if not finished:
                    if time.monotonic() - last_status >= 5:
                        _status(state, done, [s["id"] for s in pending.values()])
                        last_status = time.monotonic()
                    continue
                for future in finished:
                    future.result()
                    pending.pop(future)
                    done += 1
                    spec = next(iterator, None)
                    if spec is not None:
                        pending[pool.submit(_execute_task, spec)] = spec
                _assert_job_fingerprints(state)
                _status(state, done, [s["id"] for s in pending.values()])
                last_status = time.monotonic()
        summary = _write_summary(preflight)
        _status(state, 120, [], status="complete")
        return summary
    except BaseException:
        _status(state, done, [s["id"] for s in pending.values()],
                status="failed", error=traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true",
                        help="Freeze and record the capacity-control design, then exit without fitting")
    parser.add_argument("--self-test", action="store_true",
                        help="Run CPU-only fixture checks without touching study results")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(run_cpu_fixture_checks(), sort_keys=True))
        return
    if args.freeze:
        checks = run_cpu_fixture_checks()
        design = freeze_capacity_protocol()
        print(json.dumps(dict(state="frozen", capacity_protocol_sha256=sha(CAPACITY_PROTOCOL),
                              comparisons=design["comparison_count"], cpu_checks=checks), sort_keys=True))
        return
    run_matrix()


if __name__ == "__main__":
    main()

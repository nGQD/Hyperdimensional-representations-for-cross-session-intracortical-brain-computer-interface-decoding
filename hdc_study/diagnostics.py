"""Run matched-capacity controls, then audit predictions and evaluate diagnostics.

The diagnostic routine verifies the complete primary fit matrix and stores one
resumable calibration/robustness record per outer group, family, and final seed.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import itertools
import json
import gc
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
import torch

from data import ROOT, load_h1, partition_day, sha, write_json
from engine import evaluate, load
from run import DEV_SEEDS, FINAL_SEEDS, config_key, fingerprint
from model import Config


SCHEMA = 1
MAIN_FAMILIES = ("hdc", "random", "mlp", "plain")
PREFIXES = ((0, 0), (15, 750), (30, 1500), (60, 3000))
ROBUSTNESS_LEVELS = (0.1, 0.2, 0.4)
EXPECTED_FITS = {"inner": 1200, "outer": 600, "transfer": 40}
R2_TOLERANCE = 1e-8

_WORKER_RECORDS = None
_WORKER_PROTOCOL_SHA = None


def _json(path):
    return json.loads(Path(path).read_text())


def _assert_equal(a, b, message):
    if a != b:
        raise RuntimeError(f"{message}: {a!r} != {b!r}")


def _validate_frozen_protocol():
    protocol_path = ROOT / "provenance/protocol.json"
    validation_path = ROOT / "results/validation.json"
    pilot_path = ROOT / "results/pilot.json"
    if not protocol_path.is_file() or not validation_path.is_file() or not pilot_path.is_file():
        raise FileNotFoundError("Frozen protocol, validation, and pilot records are required")
    protocol = _json(protocol_path)
    validation = _json(validation_path)
    pilot = _json(pilot_path)
    current_fingerprint = fingerprint()
    _assert_equal(protocol.get("fingerprint"), current_fingerprint,
                  "Frozen source/data fingerprint has changed")
    if not validation.get("passed"):
        raise RuntimeError("Validation record is not passing")
    _assert_equal(validation.get("fingerprint"), current_fingerprint,
                  "Validation fingerprint does not match current sources")
    _assert_equal(protocol.get("validation_sha256"), sha(validation_path),
                  "Validation file hash does not match the frozen protocol")
    _assert_equal(protocol.get("pilot_sha256"), sha(pilot_path),
                  "Pilot file hash does not match the frozen protocol")
    if not pilot.get("complete"):
        raise RuntimeError("Throughput pilot is incomplete")
    _assert_equal(protocol.get("expected_fits"), EXPECTED_FITS,
                  "Unexpected frozen fit-matrix dimensions")
    _assert_equal(protocol.get("final_seeds"), FINAL_SEEDS,
                  "Unexpected final seed list")
    protocol_sha = sha(protocol_path)
    return protocol, protocol_sha, current_fingerprint


def _validate_result_and_checkpoint(identifier, expected_stage, protocol_sha,
                                    frozen_fingerprint):
    result_path = ROOT / "results/runs" / f"{identifier}.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing completed fit result: {result_path.name}")
    result = _json(result_path)
    _assert_equal(result.get("id"), identifier, "Result id mismatch")
    _assert_equal(result.get("stage"), expected_stage, "Result stage mismatch")
    signature = result.get("signature", {})
    _assert_equal(signature.get("protocol_sha256"), protocol_sha,
                  f"{identifier} protocol signature mismatch")
    _assert_equal(signature.get("fingerprint"), frozen_fingerprint,
                  f"{identifier} source fingerprint mismatch")
    checkpoint = ROOT / "checkpoints" / f"{identifier}.pt"
    sidecar = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Missing checkpoint or signature sidecar for {identifier}")
    checkpoint_hash = sha(checkpoint)
    _assert_equal(result.get("checkpoint_sha256"), checkpoint_hash,
                  f"{identifier} checkpoint hash mismatch")
    _assert_equal(_json(sidecar), signature,
                  f"{identifier} checkpoint sidecar signature mismatch")
    return result, checkpoint, checkpoint_hash


def _expected_identifiers():
    days = [f"S{i}" for i in range(6)]
    main = []
    for day in days:
        for family in MAIN_FAMILIES:
            for seed in FINAL_SEEDS:
                main.append((f"outer_{day}_{family}_s{seed}", "outer"))
    transfer = [(f"transfer_{family}_s{seed}", "transfer")
                for family in MAIN_FAMILIES for seed in FINAL_SEEDS]
    return days, main, transfer


def _expected_full_matrix(protocol):
    days, main, transfer = _expected_identifiers()
    variants = protocol.get("final_variants")
    expected_variants = list(MAIN_FAMILIES) + [
        "no_channel_roles", "no_roles", "no_gru", "no_online_norm",
        "no_smoothing", "no_temporal"]
    _assert_equal(variants, expected_variants,
                  "Frozen final-variant list differs from the declared 10-family matrix")
    outer = [(f"outer_{day}_{variant}_s{seed}", "outer")
             for day in days for variant in variants for seed in FINAL_SEEDS]
    transfer_ids = transfer
    candidate_configs = [Config(**row) for row in protocol.get("candidate_configs", [])]
    if len(candidate_configs) != 16:
        raise RuntimeError("Frozen protocol does not contain the full 16-candidate grid")
    inner = []
    for a, b in itertools.combinations(days, 2):
        for config in candidate_configs:
            for seed in DEV_SEEDS:
                inner.append((f"inner_{a}_{b}_{config_key(config)}_s{seed}", "inner"))
    return days, inner, outer, transfer_ids, main


def _array_exact(actual, expected, filename, name):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.array_equal(actual, expected)):
        raise RuntimeError(f"{filename} {name} differs from the verified evaluation cache")


def _expected_prediction_arrays(records, days, prefix):
    """Reconstruct the arrays written by engine.evaluate from verified caches."""
    targets, masks, lengths = [], [], []
    for day in days:
        group = [record for record in records if record["day"] == day]
        if not group:
            raise RuntimeError(f"No verified recordings for expected prediction group {day}")
        parts = partition_day(group, prefix=prefix, score_start=3256)
        targets.append(np.concatenate([part["y"] for part in parts], axis=0))
        masks.append(np.concatenate([part["score_mask"] for part in parts], axis=0))
        lengths.append(sum(len(part["x"]) for part in parts))
    return dict(target=np.concatenate(targets, axis=0),
                mask=np.concatenate(masks, axis=0),
                lengths=np.asarray(lengths, dtype=np.int64),
                days=np.asarray(days, dtype=str))


def _build_prediction_expectations(held_in_records, held_out_records):
    """Build exact saved-array expectations once from both verified cache splits."""
    held_in_days = sorted({r["day"] for r in held_in_records}, key=lambda d: int(d[1:]))
    held_out_days = sorted({r["day"] for r in held_out_records}, key=lambda d: int(d[1:]))
    if held_in_days != [f"S{i}" for i in range(6)]:
        raise RuntimeError("Verified held-in cache does not contain exactly S0-S5")
    if held_out_days != [f"S{i}" for i in range(6, 13)]:
        raise RuntimeError("Verified held-out cache does not contain exactly S6-S12")

    expectations = {}
    for regime, prefix in (("zero_shot", 0), ("few_shot", 3000)):
        for day in held_in_days:
            expectations[("outer", day, regime)] = _expected_prediction_arrays(
                held_in_records, [day], prefix)
        expectations[("transfer", None, regime)] = _expected_prediction_arrays(
            held_out_records, held_out_days, prefix)
    return expectations


def _audit_saved_predictions(result, identifier, expectations):
    """Audit saved predictions against independently reconstructed cache arrays."""
    files = result.get("prediction_sha256")
    if not isinstance(files, dict) or set(files) != {
            f"{identifier}_zero_shot.npz", f"{identifier}_few_shot.npz"}:
        raise RuntimeError(f"{identifier} has an incomplete saved-prediction manifest")
    metrics = result.get("metrics", [])
    if identifier.startswith("outer_"):
        stage = "outer"
    elif identifier.startswith("transfer_"):
        stage = "transfer"
    else:
        stage = None
    if stage == "outer":
        pieces = identifier.split("_")
        if len(pieces) < 4 or pieces[1] not in {f"S{i}" for i in range(6)}:
            raise RuntimeError(f"{identifier} has an invalid outer-group identifier")
        group = pieces[1]
    elif stage == "transfer":
        group = None
    else:
        raise RuntimeError(f"{identifier} is not an outer or transfer fit")

    expected_days = expectations[(stage, group, "zero_shot")]["days"].tolist()
    expected_keys = {(day, regime) for day in expected_days
                     for regime in ("zero_shot", "few_shot")}
    metric_keys = [(row.get("day"), row.get("regime")) for row in metrics]
    if (len(metrics) != len(expected_keys) or len(set(metric_keys)) != len(metric_keys)
            or set(metric_keys) != expected_keys):
        raise RuntimeError(f"{identifier} metric day/regime rows differ from the verified groups")
    metric_map = {(row.get("day"), row.get("regime")): row for row in metrics}

    audited = []
    for filename, expected_hash in sorted(files.items()):
        path = ROOT / "results/predictions" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction archive: {filename}")
        actual_hash = sha(path)
        _assert_equal(expected_hash, actual_hash, f"{filename} hash mismatch")
        regime = "zero_shot" if filename.endswith("_zero_shot.npz") else "few_shot"
        expected = expectations[(stage, group, regime)]
        with np.load(path, allow_pickle=False) as archive:
            required = {"prediction", "target", "mask", "lengths", "days"}
            if set(archive.files) != required:
                raise RuntimeError(f"{filename} array names differ from the saved-prediction schema")
            prediction = archive["prediction"]
            target = archive["target"]
            mask = archive["mask"]
            lengths = archive["lengths"]
            days = archive["days"]
        for name, actual in (("target", target), ("mask", mask),
                             ("lengths", lengths), ("days", days)):
            _array_exact(actual, expected[name], filename, name)
        if prediction.ndim != 2 or target.shape != prediction.shape:
            raise RuntimeError(f"{filename} prediction/target shapes disagree")
        if (mask.shape != (len(prediction),) or mask.dtype != np.bool_
                or lengths.ndim != 1 or lengths.dtype != np.int64
                or days.ndim != 1 or not np.issubdtype(days.dtype, np.str_)
                or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
            raise RuntimeError(f"{filename} contains malformed or nonfinite arrays")
        if len(lengths) != len(days) or np.any(lengths <= 0) or int(lengths.sum()) != len(prediction):
            raise RuntimeError(f"{filename} group lengths do not cover the saved arrays")
        metric_days = [row["day"] for row in metrics if row["regime"] == regime]
        _assert_equal(days.tolist(), metric_days, f"{filename} day order differs from metrics")
        offset = 0
        max_error = 0.0
        valid_bins = 0
        for day, length in zip(days, lengths.tolist()):
            row = metric_map.get((day, regime))
            if row is None:
                raise RuntimeError(f"{filename} has no metric row for {day}/{regime}")
            stop = offset + length
            group_mask = mask[offset:stop]
            if int(group_mask.sum()) != int(row["bins"]):
                raise RuntimeError(f"{filename} valid-bin count differs for {day}/{regime}")
            if not group_mask.any():
                raise RuntimeError(f"{filename} has no scored targets for {day}/{regime}")
            recomputed = float(r2_score(target[offset:stop][group_mask],
                                        prediction[offset:stop][group_mask],
                                        multioutput="variance_weighted"))
            error = abs(recomputed - float(row["r2"]))
            max_error = max(max_error, error)
            if error > R2_TOLERANCE:
                raise RuntimeError(f"{filename} R2 mismatch for {day}/{regime}: {error:g}")
            offset = stop
            valid_bins += int(group_mask.sum())
        audited.append(dict(filename=filename, sha256=actual_hash, groups=len(days),
                            valid_bins=valid_bins, max_abs_r2_error=max_error))
    return audited


def _load_selection(protocol):
    selection_path = ROOT / "results/selection.json"
    transfer_path = ROOT / "results/transfer_selection.json"
    if not selection_path.is_file() or not transfer_path.is_file():
        raise FileNotFoundError("Outer and transfer selection files are required")
    selection = _json(selection_path)
    transfer = _json(transfer_path)
    days, _, _ = _expected_identifiers()
    if set(selection) != set(days):
        raise RuntimeError("Outer selection does not contain exactly six groups")
    if set(transfer) != set(MAIN_FAMILIES):
        raise RuntimeError("Transfer selection does not contain all four main families")
    return selection, transfer


def _validate_main_fit(identifier, result, family, seed, expected_config):
    signature = result["signature"]
    config = signature.get("config", {})
    selected = dict(expected_config)
    selected["codebook_seed"] = 100000 + seed
    _assert_equal(config, selected, f"{identifier} does not use frozen selected config")
    _assert_equal(config.get("family"), family, f"{identifier} family mismatch")
    _assert_equal(signature.get("seed"), seed, f"{identifier} seed mismatch")
    _assert_equal(config.get("codebook_seed"), 100000 + seed,
                  f"{identifier} codebook seed does not follow frozen rule")


def _load_checkpoint_model(identifier, checkpoint, expected_signature):
    # engine.load enforces model schema and strict state-dict compatibility.
    model, record = load(checkpoint)
    _assert_equal(record.get("config"), expected_signature["config"],
                  f"{identifier} checkpoint config differs from its signature")
    _assert_equal(record.get("seed"), expected_signature["seed"],
                  f"{identifier} checkpoint seed differs from its signature")
    _assert_equal(record.get("train_days"), expected_signature["train_days"],
                  f"{identifier} checkpoint training groups differ from its signature")
    return model, record


def validate_diagnostic_record(row, identifier, family, seed, day,
                               protocol_sha, checkpoint_hash, metrics):
    expected = dict(schema=SCHEMA, id=identifier, family=family, seed=seed,
                    day=day, protocol_sha256=protocol_sha,
                    checkpoint_sha256=checkpoint_hash, perturb_seed=700000 + seed)
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{identifier} diagnostic provenance mismatch")
    expected_conditions = {
        "calibration": {seconds for seconds, _ in PREFIXES},
        "robustness": {(kind, level) for kind in ("gain", "dropout")
                       for level in (0.0, *ROBUSTNESS_LEVELS)},
    }
    tables = {}
    for name, conditions in expected_conditions.items():
        entries = row.get(name)
        if not isinstance(entries, list) or len(entries) != len(conditions):
            raise RuntimeError(f"{identifier} {name} condition count mismatch")
        by_condition = {}
        for entry in entries:
            if (not isinstance(entry, dict) or entry.get("day") != day
                    or entry.get("family") != family or entry.get("seed") != seed):
                raise RuntimeError(f"{identifier} {name} row identity mismatch")
            value = entry.get("r2")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError(f"{identifier} {name} score is not finite")
            if name == "calibration":
                condition = entry.get("prefix_seconds")
                if type(condition) is not int:
                    raise RuntimeError(f"{identifier} calibration prefix must be an integer")
            else:
                level = entry.get("level")
                if isinstance(level, bool) or not isinstance(level, (int, float)):
                    raise RuntimeError(f"{identifier} perturbation level must be numeric")
                condition = (entry.get("perturbation"), level)
            if condition not in conditions or condition in by_condition:
                raise RuntimeError(f"{identifier} {name} conditions are missing, duplicated, or unexpected")
            by_condition[condition] = float(value)
        tables[name] = by_condition
    baseline = {}
    for metric in metrics:
        regime = metric.get("regime")
        if metric.get("day") != day or regime not in ("zero_shot", "few_shot") or regime in baseline:
            raise RuntimeError(f"{identifier} baseline metric identity mismatch")
        prefix = 0 if regime == "zero_shot" else 3000
        value = metric.get("r2")
        if (metric.get("calibration_bins") != prefix or metric.get("score_start") != 3256
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value)):
            raise RuntimeError(f"{identifier} baseline metric protocol mismatch")
        baseline[regime] = float(value)
    if set(baseline) != {"zero_shot", "few_shot"}:
        raise RuntimeError(f"{identifier} diagnostic baseline metrics are incomplete")
    endpoints = ((tables["calibration"][0], baseline["zero_shot"]),
                 (tables["calibration"][60], baseline["few_shot"]),
                 (tables["robustness"]["gain", 0.0], baseline["zero_shot"]),
                 (tables["robustness"]["dropout", 0.0], baseline["zero_shot"]))
    if any(abs(actual - expected_value) > R2_TOLERANCE for actual, expected_value in endpoints):
        raise RuntimeError(f"{identifier} diagnostic endpoints differ from audited primary scores")
    return row


def _existing_diagnostics(path, identifier, protocol_sha, checkpoint_hash):
    if not path.is_file():
        return None
    row = _json(path)
    _assert_equal(row.get("schema"), SCHEMA, f"{path.name} schema mismatch")
    _assert_equal(row.get("id"), identifier, f"{path.name} id mismatch")
    _assert_equal(row.get("protocol_sha256"), protocol_sha,
                  f"{path.name} protocol hash mismatch")
    _assert_equal(row.get("checkpoint_sha256"), checkpoint_hash,
                  f"{path.name} checkpoint hash mismatch")
    if not isinstance(row.get("calibration"), list) or len(row["calibration"]) != 4:
        raise RuntimeError(f"{path.name} calibration rows are incomplete")
    if not isinstance(row.get("robustness"), list) or len(row["robustness"]) != 8:
        raise RuntimeError(f"{path.name} robustness rows are incomplete")
    return row


def _diagnose_one(identifier, family, seed, day, result, checkpoint,
                  checkpoint_hash, expected_signature, records, protocol_sha):
    path = ROOT / "results/diagnostics" / f"{identifier}.json"
    existing = _existing_diagnostics(path, identifier, protocol_sha, checkpoint_hash)
    if existing is not None:
        validate_diagnostic_record(existing, identifier, family, seed, day,
                                   protocol_sha, checkpoint_hash, result["metrics"])
        return existing

    model, record = _load_checkpoint_model(identifier, checkpoint, expected_signature)
    group = [r for r in records if r["day"] == day]
    if not group:
        raise RuntimeError(f"No verified input recordings for {day}")
    metrics = result["metrics"]
    zero = next((m for m in metrics if m["day"] == day and m["regime"] == "zero_shot"), None)
    few = next((m for m in metrics if m["day"] == day and m["regime"] == "few_shot"), None)
    if zero is None or few is None:
        raise RuntimeError(f"{identifier} is missing baseline zero/few-shot results")
    if int(zero.get("calibration_bins", -1)) != 0 or int(few.get("calibration_bins", -1)) != 3000:
        raise RuntimeError(f"{identifier} baseline calibration budgets are inconsistent")
    if int(zero.get("score_start", -1)) != 3256 or int(few.get("score_start", -1)) != 3256:
        raise RuntimeError(f"{identifier} baseline score suffix differs from frozen protocol")

    calibration = []
    for seconds, bins in PREFIXES:
        if bins == 0:
            score = float(zero["r2"])
        elif bins == 3000:
            score = float(few["r2"])
        else:
            rows = evaluate(model, record, group, regime="few_shot", prefix=bins)
            if len(rows) != 1 or rows[0]["day"] != day:
                raise RuntimeError(f"Unexpected calibration evaluation rows for {identifier}")
            if int(rows[0]["calibration_bins"]) != bins or int(rows[0]["score_start"]) != 3256:
                raise RuntimeError(f"Calibration protocol mismatch for {identifier}, {bins} bins")
            score = float(rows[0]["r2"])
        calibration.append(dict(day=day, family=family, seed=seed,
                                prefix_seconds=seconds, r2=score))

    robustness = [dict(day=day, family=family, seed=seed,
                       perturbation=kind, level=0.0, r2=float(zero["r2"]))
                  for kind in ("gain", "dropout")]
    perturb_seed = 700000 + seed
    for kind in ("gain", "dropout"):
        for level in ROBUSTNESS_LEVELS:
            rows = evaluate(model, record, group, regime="zero_shot", prefix=0,
                            perturbation=kind, level=level,
                            perturb_seed=perturb_seed)
            if len(rows) != 1 or rows[0]["day"] != day:
                raise RuntimeError(f"Unexpected robustness evaluation rows for {identifier}")
            if rows[0]["regime"] != "zero_shot" or int(rows[0]["score_start"]) != 3256:
                raise RuntimeError(f"Robustness protocol mismatch for {identifier}/{kind}/{level}")
            robustness.append(dict(day=day, family=family, seed=seed,
                                   perturbation=kind, level=level,
                                   r2=float(rows[0]["r2"])))
    output = dict(schema=SCHEMA, id=identifier, day=day, family=family, seed=seed,
                  protocol_sha256=protocol_sha, checkpoint_sha256=checkpoint_hash,
                  perturb_seed=perturb_seed, calibration=calibration,
                  robustness=robustness)
    validate_diagnostic_record(output, identifier, family, seed, day,
                               protocol_sha, checkpoint_hash, result["metrics"])
    write_json(path, output)
    return output


def _initialize_diagnostic_worker(protocol_sha, frozen_fingerprint):
    """Load verified held-in inputs once and verify the frozen source state."""
    global _WORKER_RECORDS, _WORKER_PROTOCOL_SHA
    if fingerprint() != frozen_fingerprint:
        raise RuntimeError("Worker scientific sources differ from the frozen fingerprint")
    if sha(ROOT / "provenance/protocol.json") != protocol_sha:
        raise RuntimeError("Worker protocol differs from the frozen protocol")
    records = load_h1("held-in-calib", verify=True)
    days = sorted({row["day"] for row in records}, key=lambda day: int(day[1:]))
    if days != [f"S{i}" for i in range(6)]:
        raise RuntimeError("Worker verified held-in cache does not contain exactly S0-S5")
    _WORKER_RECORDS = records
    _WORKER_PROTOCOL_SHA = protocol_sha


def _diagnostic_worker(payload):
    """Run one independent inference task using worker-local cached records."""
    if _WORKER_RECORDS is None or _WORKER_PROTOCOL_SHA is None:
        raise RuntimeError("Diagnostics worker was not initialized")
    protocol_sha = payload["protocol_sha256"]
    _assert_equal(protocol_sha, _WORKER_PROTOCOL_SHA,
                  "Diagnostic task protocol hash differs from worker initialization")
    checkpoint = Path(payload["checkpoint_path"])
    _assert_equal(sha(checkpoint), payload["checkpoint_sha256"],
                  f"{payload['identifier']} checkpoint changed before worker inference")
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return _diagnose_one(
            payload["identifier"], payload["family"], payload["seed"], payload["day"],
            payload["result"], checkpoint, payload["checkpoint_sha256"],
            payload["signature"], _WORKER_RECORDS, protocol_sha)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _dispatch_diagnostic_tasks(tasks, workers, on_complete=None,
                               executor_factory=ProcessPoolExecutor,
                               worker_function=_diagnostic_worker,
                               initializer=_initialize_diagnostic_worker,
                               initargs=()):
    """Bound pending inference to worker count; parent alone reports progress."""
    iterator = iter(tasks)
    pending = {}
    completed = 0
    with executor_factory(max_workers=workers, mp_context=mp.get_context("spawn"),
                          initializer=initializer, initargs=initargs) as executor:
        def fill():
            while len(pending) < workers:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
                pending[executor.submit(worker_function, task)] = task

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            finished = []
            for future in done:
                task = pending.pop(future)
                value = future.result()
                completed += 1
                finished.append((task, value, completed))
            fill()
            active = [pending_task["identifier"] for pending_task in pending.values()]
            if on_complete is not None:
                for task, value, task_count in finished:
                    on_complete(task, value, task_count, active)
    return completed


def _pilot_worker_count(requested=None):
    """Use the complete measured concurrency choice, falling back to serial."""
    pilot_path = ROOT / "results/concurrency_pilot.json"
    try:
        pilot = _json(pilot_path)
    except (OSError, json.JSONDecodeError):
        pilot = {}
    if pilot.get("complete") is True:
        chosen = pilot.get("chosen_workers")
        rows = pilot.get("rows", [])
        measured = next((row for row in rows if isinstance(row, dict)
                         and row.get("workers") == chosen), None) if isinstance(rows, list) else None
        seconds = measured.get("total_seconds") if measured is not None else None
        if (type(chosen) is not int or chosen not in (1, 2, 3, 4)
                or measured is None or measured.get("failed")
                or not isinstance(seconds, (int, float))
                or not np.isfinite(seconds) or seconds <= 0):
            raise RuntimeError("Completed concurrency pilot has an invalid worker choice")
    else:
        chosen = 1
    if requested is None:
        return chosen
    if type(requested) is not int or requested not in (1, 2, 3, 4):
        raise ValueError("--workers must be an integer from 1 through 4")
    if requested > chosen:
        raise RuntimeError(f"Requested {requested} diagnostics workers, but the validated choice is {chosen}")
    return requested


def run_diagnostics(workers=None):
    """Validate the full 1840-fit matrix, audit predictions, then diagnose 240 fits."""
    workers = _pilot_worker_count(workers)
    protocol, protocol_sha, frozen_fingerprint = _validate_frozen_protocol()
    days, inner_ids, outer_ids, transfer_ids, main_ids = _expected_full_matrix(protocol)
    selection, transfer_selection = _load_selection(protocol)

    run_files = sorted((ROOT / "results/runs").glob("*.json"))
    all_results = {}
    counts = {stage: 0 for stage in EXPECTED_FITS}
    for path in run_files:
        row = _json(path)
        stage = row.get("stage")
        if stage not in counts:
            raise RuntimeError(f"Unexpected result stage in {path.name}: {stage!r}")
        counts[stage] += 1
        identifier = row.get("id")
        if not identifier or identifier in all_results:
            raise RuntimeError(f"Missing or duplicate fit result id in {path.name}")
        all_results[identifier] = row
    _assert_equal(counts, EXPECTED_FITS, "The fit matrix is incomplete")
    _assert_equal(len(run_files), sum(EXPECTED_FITS.values()),
                  "Unexpected number of fit result files")
    expected_by_stage = {
        "inner": {identifier for identifier, _ in inner_ids},
        "outer": {identifier for identifier, _ in outer_ids},
        "transfer": {identifier for identifier, _ in transfer_ids},
    }
    for stage, expected_ids in expected_by_stage.items():
        actual_ids = {identifier for identifier, row in all_results.items()
                      if row["stage"] == stage}
        _assert_equal(actual_ids, expected_ids,
                      f"The {stage} result identifiers do not match the frozen matrix")

    # Validate result signatures, sidecars, and checkpoint hashes across every fit.
    checkpoint_info = {}
    for identifier, row in all_results.items():
        result, checkpoint, checkpoint_hash = _validate_result_and_checkpoint(
            identifier, row["stage"], protocol_sha, frozen_fingerprint)
        checkpoint_info[identifier] = (result, checkpoint, checkpoint_hash)
    for identifier, expected_stage in outer_ids + transfer_ids:
        if identifier not in all_results:
            raise FileNotFoundError(f"Full final matrix is missing {identifier}")
        _assert_equal(all_results[identifier]["stage"], expected_stage,
                      f"{identifier} is in the wrong stage")

    # Check selected configurations and selected final-run signatures.
    for day in days:
        for family in MAIN_FAMILIES:
            expected = selection[day]["families"][family][0]["config"]
            for seed in FINAL_SEEDS:
                identifier = f"outer_{day}_{family}_s{seed}"
                _validate_main_fit(identifier, all_results[identifier], family, seed, expected)
    for family in MAIN_FAMILIES:
        expected = transfer_selection[family]["config"]
        for seed in FINAL_SEEDS:
            identifier = f"transfer_{family}_s{seed}"
            _validate_main_fit(identifier, all_results[identifier], family, seed, expected)

    # Load and verify each evaluation split once. The same held-in records feed
    # the later calibration/robustness diagnostics.
    held_in_records = load_h1("held-in-calib", verify=True)
    held_out_records = load_h1("held-out-calib", verify=True)
    prediction_expectations = _build_prediction_expectations(
        held_in_records, held_out_records)

    # Auditing all held-out predictions is required before creating diagnostics.
    prediction_audits = []
    for identifier, stage in outer_ids + transfer_ids:
        result = all_results[identifier]
        prediction_audits.extend(_audit_saved_predictions(
            result, identifier, prediction_expectations))
    audit = dict(schema=SCHEMA, passed=True, protocol_sha256=protocol_sha,
                 fingerprint=frozen_fingerprint, audited_fit_count=len(outer_ids) + len(transfer_ids),
                 audited_archive_count=len(prediction_audits),
                 scored_group_count=sum(row["groups"] for row in prediction_audits),
                 total_valid_bins=sum(row["valid_bins"] for row in prediction_audits),
                 maximum_absolute_r2_error=max(row["max_abs_r2_error"] for row in prediction_audits),
                 files=prediction_audits)
    write_json(ROOT / "results/prediction_audit.json", audit)
    del held_in_records, held_out_records, prediction_expectations

    out_dir = ROOT / "results/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    total = len(main_ids)
    tasks = []
    for identifier, _ in main_ids:
        result, checkpoint, checkpoint_hash = checkpoint_info[identifier]
        _, day, family, seed_text = identifier.split("_")
        seed = int(seed_text[1:])
        expected_config = selection[day]["families"][family][0]["config"]
        _validate_main_fit(identifier, result, family, seed, expected_config)
        tasks.append(dict(
            identifier=identifier, family=family, seed=seed, day=day,
            result={"metrics": result["metrics"]},
            checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_hash,
            signature=result["signature"], protocol_sha256=protocol_sha))
    del all_results, checkpoint_info, selection, transfer_selection

    progress = {"completed": 0, "current": None}

    def write_progress(task=None, value=None, completed=0, active=()):
        progress["completed"] = completed
        progress["current"] = task["identifier"] if task is not None else None
        write_json(status_path, dict(schema=SCHEMA, state="running",
                    protocol_sha256=protocol_sha, completed=completed,
                    expected=total, current=progress["current"], workers=workers,
                    active_jobs=list(active), updated_utc=time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    write_progress(completed=0, active=[task["identifier"] for task in tasks[:workers]])
    try:
        completed = _dispatch_diagnostic_tasks(
            tasks, workers, on_complete=write_progress,
            initargs=(protocol_sha, frozen_fingerprint))
    except BaseException as exc:
        write_json(status_path, dict(schema=SCHEMA, state="failed",
                    protocol_sha256=protocol_sha, completed=progress["completed"],
                    expected=total, current=progress["current"], workers=workers,
                    error=f"{type(exc).__name__}: {exc}", updated_utc=time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        raise
    write_json(status_path, dict(schema=SCHEMA, state="complete",
                protocol_sha256=protocol_sha, completed=completed,
                expected=total, current=None, workers=workers,
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    return dict(state="complete", diagnostics=completed,
                expected=total, prediction_audit=audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="inference worker count (default: completed pilot choice, else 1)")
    args = parser.parse_args()
    from matched_controls import run_matrix
    run_matrix()
    result = run_diagnostics(args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != "prediction_audit"}, indent=2))


if __name__ == "__main__":
    main()

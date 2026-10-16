"""Run the frozen H1 fit matrix with pilot-approved GPU worker concurrency.

This scheduler delegates each unchanged ``run.job`` fit to one persistent
worker. It keeps one job in flight per worker, loads each worker's data once,
and leaves the scientific fit, evaluation, selection, and checkpoint code in
``run.py`` unchanged.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import gc
import itertools
import json
import multiprocessing as mp
import os
import re
import signal
import time
import traceback

import numpy as np

import run
from data import ROOT, load_h1, sessions, sha, write_json
from model import Config


_WORKER_RUN = None
_WORKER_RECORDS = None
_WORKER_PROTOCOL_SHA = None


def _worker_status(**fields):
    """Keep worker progress separate from the master status file."""
    path = ROOT / "results/workers" / f"{os.getpid()}.json"
    write_json(path, dict(pid=os.getpid(), updated_unix=time.time(), **fields))


def _initialize_worker(protocol_sha):
    """Initialize frozen globals and verified data once in each worker."""
    global _WORKER_RUN, _WORKER_RECORDS, _WORKER_PROTOCOL_SHA
    protocol = run.freeze()
    current_sha = sha(ROOT / "provenance/protocol.json")
    if current_sha != protocol_sha or protocol["fingerprint"] != run.fingerprint():
        raise RuntimeError("Worker sources or protocol differ from the master freeze")
    _WORKER_PROTOCOL_SHA = protocol_sha
    run.ACTIVE = protocol
    run.ACTIVE_SHA = protocol_sha
    run.ROWS = []
    # run.job reports epoch progress through this process-local hook.  The
    # master is the only process that writes results/status.json.
    run.status = _worker_status
    _WORKER_RUN = run
    _WORKER_RECORDS = {
        "held-in-calib": load_h1("held-in-calib", verify=True),
        "held-out-calib": load_h1("held-out-calib", verify=True),
    }


def _execute_job(spec):
    """Resolve compact day identifiers against the worker-local data cache."""
    if _WORKER_RUN is None or _WORKER_RECORDS is None:
        raise RuntimeError("Parallel worker was not initialized")
    if sha(ROOT / "provenance/protocol.json") != _WORKER_PROTOCOL_SHA:
        raise RuntimeError("Frozen protocol changed during the fit matrix")
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    split = "held-out-calib" if spec["stage"] == "transfer" else "held-in-calib"
    records = _WORKER_RECORDS[split]
    train_days, test_days = set(spec["train_days"]), set(spec["test_days"])
    train = [r for r in _WORKER_RECORDS["held-in-calib"] if r["day"] in train_days]
    test = [r for r in records if r["day"] in test_days]
    if not train or not test:
        raise RuntimeError(f"Empty train or test records for {spec['identifier']}")
    try:
        result = _WORKER_RUN.job(
            spec["identifier"], Config(**spec["config"]), spec["seed"],
            train, test, spec["stage"])
        _validate_fit_payload(result)
        return result
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _validate_fit_payload(row):
    identifier = row.get("id") if isinstance(row, dict) else None
    if not isinstance(identifier, str) or row.get("stage") not in run.EXPECTED:
        raise RuntimeError("Completed fit has an invalid identity or stage")
    signature = row.get("signature")
    if not isinstance(signature, dict) or not isinstance(signature.get("config"), dict):
        raise RuntimeError(f"{identifier} is missing its configuration signature")
    test_days = signature.get("test_days")
    if (not isinstance(test_days, list) or not test_days
            or not all(isinstance(day, str) for day in test_days)
            or len(set(test_days)) != len(test_days)):
        raise RuntimeError(f"{identifier} has invalid evaluation groups")
    expected = {(day, regime) for day in test_days for regime in ("zero_shot", "few_shot")}
    metrics = row.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != len(expected):
        raise RuntimeError(f"{identifier} has incomplete evaluation metrics")
    seen = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            raise RuntimeError(f"{identifier} has malformed evaluation metrics")
        key = (metric.get("day"), metric.get("regime"))
        value = metric.get("r2")
        prefix = 0 if metric.get("regime") == "zero_shot" else 3000
        if (key not in expected or key in seen or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not np.isfinite(value)
                or type(metric.get("bins")) is not int or metric["bins"] <= 0
                or metric.get("calibration_bins") != prefix or metric.get("score_start") != 3256
                or metric.get("perturbation") is not None or metric.get("perturbation_level") != 0):
            raise RuntimeError(f"{identifier} has invalid or duplicated metric conditions")
        seen.add(key)
    for name in ("train_seconds", "total_seconds"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError(f"{identifier} has missing or invalid {name}")
    if type(row.get("parameters")) is not int or row["parameters"] <= 0:
        raise RuntimeError(f"{identifier} has an invalid parameter count")
    losses = row.get("training_losses")
    if (not isinstance(losses, list) or len(losses) != signature["config"].get("epochs")
            or not losses or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                    and np.isfinite(v) for v in losses)):
        raise RuntimeError(f"{identifier} has an incomplete training record")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("checkpoint_sha256", ""))):
        raise RuntimeError(f"{identifier} has an invalid checkpoint digest")
    hashes = row.get("prediction_sha256")
    names = ({f"{identifier}_{regime}.npz" for regime in ("zero_shot", "few_shot")}
             if row["stage"] != "inner" else set())
    if (not isinstance(hashes, dict) or set(hashes) != names
            or not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hashes.values())):
        raise RuntimeError(f"{identifier} has an incomplete saved-prediction manifest")


def _fit_rows():
    """Read only committed result files; job writes use atomic replacement."""
    rows = []
    for path in (ROOT / "results/runs").glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Incomplete fit result file: {path.name}") from exc
        if row.get("id") != path.stem:
            raise RuntimeError(f"Fit result filename and identifier differ: {path.name}")
        _validate_fit_payload(row)
        rows.append(row)
    return rows


def _validate_worker_budget(workers):
    if workers not in (1, 2, 3, 4):
        raise ValueError("workers must be between 1 and 4")
    if workers <= 2:
        return
    pilot_path = ROOT / "results/concurrency_pilot.json"
    try:
        pilot = json.loads(pilot_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Workers above two require a complete concurrency pilot") from exc
    chosen = pilot.get("chosen_workers")
    measured = {row.get("workers"): row for row in pilot.get("rows", [])}
    if (pilot.get("complete") is not True or not isinstance(chosen, int)
            or chosen not in (1, 2, 3, 4) or chosen < workers
            or chosen not in measured or measured[chosen].get("failed")
            or not isinstance(measured[chosen].get("total_seconds"), (int, float))):
        raise RuntimeError(
            f"Requested {workers} workers exceeds the complete concurrency pilot choice "
            f"({chosen!r}); rerun only with a pilot-validated count")


def _result_counts():
    rows = _fit_rows()
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not identifier or identifier in seen:
            raise RuntimeError("Result directory contains a missing or duplicate fit identifier")
        seen.add(identifier)
        if row.get("stage") not in run.EXPECTED:
            raise RuntimeError(f"Unexpected result stage for {identifier}")
    counts = {name: sum(row["stage"] == name for row in rows)
              for name in run.EXPECTED}
    return rows, counts


def _require_complete(stage):
    _, counts = _result_counts()
    stages = run.EXPECTED if stage == "all" else (stage,)
    incomplete = {name: (counts[name], run.EXPECTED[name]) for name in stages
                  if counts[name] != run.EXPECTED[name]}
    if incomplete:
        raise RuntimeError(f"Cannot mark {stage} complete; fit counts differ: {incomplete}")


def _master_status(stage, started, active_jobs=(), state="running", note=None, workers=1):
    rows, counts = _result_counts()
    # Result files are committed atomically, so filesystem modification time
    # provides the completion order for the rolling throughput estimate.
    result_dir = ROOT / "results/runs"
    dated_rows = []
    for path in result_dir.glob("*.json"):
        dated_rows.append((path.stat().st_mtime_ns, path.stem))
    dated_rows.sort()
    rows_by_id = {row["id"]: row for row in rows}
    recent = [rows_by_id[identifier] for _, identifier in dated_rows[-40:]
              if identifier in rows_by_id]
    durations = [row["total_seconds"] / (len(row["signature"]["train_days"]) / 4)
                 for row in recent
                 if row.get("total_seconds") is not None
                 and row.get("signature", {}).get("train_days")]
    remaining = sum(max(0, run.EXPECTED[name] - counts[name]) * weight
                    for name, weight in (("inner", 1), ("outer", 1.25), ("transfer", 1.5)))
    # This is a throughput-scaled wall-clock estimate. The stage weights
    # approximate the relative training volume; diagnostics are excluded.
    eta = float(np.mean(durations) * remaining / workers) if durations else None
    active = [job["identifier"] if isinstance(job, dict) else str(job)
              for job in active_jobs]
    fields = dict(updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  state=state, stage=stage, pid=os.getpid(), workers=workers,
                  counts=counts,
                  expected=run.EXPECTED, elapsed_seconds=time.time() - started,
                  remaining_fit_seconds=eta,
                  eta_note=("Approximate concurrent fit/evaluation wall time from recent committed fits; "
                            "diagnostics and release checks follow"),
                  active_jobs=active)
    if note is not None:
        fields["note"] = note
    write_json(ROOT / "results/status.json", fields)


def _run_jobs(stage, specs, executor, started, workers, limit=None, total_done=0):
    """Stream at most one job per worker, updating status on a five-second cadence."""
    iterator = iter(specs)
    pending = {}
    submitted = 0

    def fill():
        nonlocal submitted
        while len(pending) < workers:
            if limit is not None and total_done + submitted >= limit:
                return
            try:
                spec = next(iterator)
            except StopIteration:
                return
            future = executor.submit(_execute_job, spec)
            pending[future] = spec
            submitted += 1

    fill()
    _master_status(stage, started, (s["identifier"] for s in pending.values()), workers=workers)
    finished_here = 0
    last_update = time.monotonic()
    while pending:
        done, _ = wait(pending, timeout=max(0.0, 5.0 - (time.monotonic() - last_update)),
                       return_when=FIRST_COMPLETED)
        if not done:
            _master_status(stage, started, (s["identifier"] for s in pending.values()), workers=workers)
            last_update = time.monotonic()
            continue
        for future in done:
            spec = pending.pop(future)
            future.result()
            finished_here += 1
        fill()
        _master_status(stage, started, (s["identifier"] for s in pending.values()), workers=workers)
        last_update = time.monotonic()
        if limit is not None and total_done + finished_here >= limit:
            break

    hit_limit = limit is not None and total_done + finished_here >= limit
    if hit_limit:
        _master_status(stage, started, state="limited",
                       note=f"Stopped after the requested limit of {limit} jobs",
                       active_jobs=(s["identifier"] for s in pending.values()), workers=workers)
    if pending:
        # This can occur only when a limit is reached while two jobs are
        # already running.  Wait for those submitted fits before returning.
        for future, spec in list(pending.items()):
            future.result()
            pending.pop(future)
            finished_here += 1
            _master_status(stage, started,
                           (s["identifier"] for s in pending.values()), workers=workers)
    return finished_here, hit_limit


def _verify_specs(stage, specs, days, configs, selection=None, transfer=None):
    """Check generated identifiers and split structure without fitting models."""
    if stage == "inner":
        expected = {
            f"inner_{a}_{b}_{run.config_key(config)}_s{seed}":
                (asdict(config), seed, set(days) - {a, b}, {a, b})
            for a, b in itertools.combinations(days, 2)
            for config in configs for seed in run.DEV_SEEDS
        }
    elif stage == "outer":
        expected = {}
        for day in days:
            variants = {family: Config(**selection[day]["families"][family][0]["config"])
                        for family in run.FAMILIES}
            variants.update(run.ablations(variants["hdc"]))
            expected.update({
                f"outer_{day}_{family}_s{seed}":
                    (asdict(config), seed, set(days) - {day}, {day})
                for family, config in variants.items() for seed in run.FINAL_SEEDS
            })
    elif stage == "transfer":
        expected = {
            f"transfer_{family}_s{seed}":
                (selected["config"], seed, {f"S{i}" for i in range(6)},
                 {f"S{i}" for i in range(6, 13)})
            for family, selected in transfer.items() for seed in run.FINAL_SEEDS
        }
    else:
        raise ValueError(f"Unknown fit stage: {stage}")

    actual_ids = [spec.get("identifier") for spec in specs]
    if len(actual_ids) != run.EXPECTED[stage] or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError(f"{stage} job matrix has an incorrect size or duplicate identifiers")
    if set(actual_ids) != set(expected):
        raise RuntimeError(f"{stage} job identifiers differ from the frozen expected matrix")
    for spec in specs:
        train_days, test_days = set(spec["train_days"]), set(spec["test_days"])
        if spec["stage"] != stage or train_days & test_days:
            raise RuntimeError(f"Invalid split in {spec['identifier']}")
        expected_config, expected_seed, expected_train, expected_test = expected[spec["identifier"]]
        if (spec["config"] != expected_config or spec["seed"] != expected_seed
                or train_days != expected_train or test_days != expected_test):
            raise RuntimeError(f"Configuration, seed, or split differs in {spec['identifier']}")
        if stage == "inner" and (len(test_days) != 2 or train_days != set(days) - test_days):
            raise RuntimeError(f"Invalid inner split in {spec['identifier']}")
        if stage == "outer" and (len(test_days) != 1 or train_days != set(days) - test_days):
            raise RuntimeError(f"Invalid outer split in {spec['identifier']}")
        if stage == "transfer" and (train_days != {f"S{i}" for i in range(6)}
                                     or test_days != {f"S{i}" for i in range(6, 13)}):
            raise RuntimeError(f"Invalid transfer split in {spec['identifier']}")
    if len(specs) != run.EXPECTED[stage]:
        raise RuntimeError(f"{stage} matrix count differs from frozen expected fits")


def _inner_specs(days, configs):
    for a, b in itertools.combinations(days, 2):
        train_days = [day for day in days if day not in (a, b)]
        test_days = [a, b]
        for config in configs:
            for seed in run.DEV_SEEDS:
                yield dict(identifier=f"inner_{a}_{b}_{run.config_key(config)}_s{seed}",
                           config=asdict(config), seed=seed, train_days=train_days,
                           test_days=test_days, stage="inner")


def _outer_specs(days, selection):
    for day in days:
        variants = {family: Config(**selection[day]["families"][family][0]["config"])
                    for family in run.FAMILIES}
        variants.update(run.ablations(variants["hdc"]))
        train_days = [other for other in days if other != day]
        for family, config in variants.items():
            for seed in run.FINAL_SEEDS:
                yield dict(identifier=f"outer_{day}_{family}_s{seed}",
                           config=asdict(config), seed=seed, train_days=train_days,
                           test_days=[day], stage="outer")


def _transfer_selection(days, configs, selection):
    transfer = {}
    for family in run.FAMILIES:
        rows = []
        for config in (c for c in configs if c.family == family):
            entries = [next(row for row in selection[day]["families"][family]
                            if row["key"] == run.config_key(config)) for day in days]
            rows.append(dict(config=asdict(config),
                             score=float(np.mean([row["score"] for row in entries])),
                             key=run.config_key(config),
                             parameters=entries[0]["parameters"]))
        transfer[family] = sorted(rows, key=lambda row: (
            -row["score"], row["parameters"], row["key"]))[0]
    write_json(ROOT / "results/transfer_selection.json", transfer)
    return transfer


def _transfer_specs(transfer):
    days = [f"S{i}" for i in range(6, 13)]
    train_days = [f"S{i}" for i in range(6)]
    for family, selected in transfer.items():
        for seed in run.FINAL_SEEDS:
            yield dict(identifier=f"transfer_{family}_s{seed}",
                       config=selected["config"], seed=seed,
                       train_days=train_days, test_days=days, stage="transfer")


def run_parallel(stage="all", limit=None, workers=2):
    global run
    _validate_worker_budget(workers)
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    protocol = run.freeze()
    protocol_sha = sha(ROOT / "provenance/protocol.json")
    active = run.ACTIVE
    active_sha = run.ACTIVE_SHA
    run.ACTIVE, run.ACTIVE_SHA = protocol, protocol_sha
    records = load_h1("held-in-calib", verify=True)
    days = sessions(records)
    del records
    if days != [f"S{i}" for i in range(6)]:
        raise RuntimeError("Unexpected outer groups")
    configs = [Config(**row) for row in protocol["candidate_configs"]]
    config_counts = {family: sum(config.family == family for config in configs)
                     for family in run.FAMILIES}
    if (len(configs) != 16 or len({run.config_key(config) for config in configs}) != 16
            or config_counts != {family: 4 for family in run.FAMILIES}):
        raise RuntimeError("Frozen protocol must contain four unique candidates per model family")
    started = time.time()
    run.assert_frozen()
    total_done = 0
    limited = False
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn"),
                                 initializer=_initialize_worker,
                                 initargs=(protocol_sha,)) as executor:
            if stage in ("all", "inner"):
                inner_specs = list(_inner_specs(days, configs))
                _verify_specs("inner", inner_specs, days, configs)
                done, limited = _run_jobs("inner", inner_specs,
                                          executor, started, workers, limit, total_done)
                total_done += done
                if limited or stage == "inner":
                    if not limited:
                        _require_complete("inner")
                    _master_status("inner", started,
                                   state="limited" if limited else "complete", workers=workers)
                    return

            run.assert_frozen()
            selection = run.select(days, configs)
            if stage in ("all", "outer"):
                outer_specs = list(_outer_specs(days, selection))
                _verify_specs("outer", outer_specs, days, configs, selection=selection)
                done, limited = _run_jobs("outer", outer_specs,
                                          executor, started, workers, limit, total_done)
                total_done += done
                if limited or stage == "outer":
                    if not limited:
                        _require_complete("outer")
                    _master_status("outer", started,
                                   state="limited" if limited else "complete", workers=workers)
                    return

            if stage in ("all", "transfer"):
                transfer = _transfer_selection(days, configs, selection)
                transfer_specs = list(_transfer_specs(transfer))
                _verify_specs("transfer", transfer_specs, days, configs, transfer=transfer)
                done, limited = _run_jobs("transfer", transfer_specs,
                                          executor, started, workers, limit, total_done)
                total_done += done
                if limited:
                    _master_status("transfer", started, state="limited", workers=workers)
                    return
        run.assert_frozen()
        _require_complete(stage)
        _master_status(stage, started, state="complete", workers=workers)
    except KeyboardInterrupt:
        _master_status(stage, started, state="interrupted", workers=workers,
                       note="Stop requested; already-running fits were allowed to finish")
        raise
    except BaseException:
        _master_status(stage, started, state="failed", note=traceback.format_exc(),
                       workers=workers)
        raise
    finally:
        run.ACTIVE, run.ACTIVE_SHA = active, active_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "inner", "outer", "transfer"), default="all")
    parser.add_argument("--limit", type=int,
                        help="maximum submitted/completed jobs for this invocation")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2,
                        help="GPU worker count; above two requires a complete validated pilot")
    args = parser.parse_args()
    previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_to_keyboardinterrupt)
    try:
        run_parallel(args.stage, args.limit, args.workers)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _sigterm_to_keyboardinterrupt(signum, frame):
    """Route SIGTERM through executor cleanup instead of orphaning workers."""
    raise KeyboardInterrupt("SIGTERM received")


if __name__ == "__main__":
    main()

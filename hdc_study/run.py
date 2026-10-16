"""Frozen, resumable nested selection with equally budgeted model families."""
import argparse
from dataclasses import asdict, replace
import itertools
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from data import ROOT, REPO, load_h1, sessions, sha, write_json
from model import Config, Decoder
from engine import fit, load, evaluate

DEV_SEEDS = [11, 23, 37, 51, 71]
FINAL_SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
FAMILIES = ["hdc", "random", "mlp", "plain"]
EXPECTED = dict(inner=1200, outer=600, transfer=40)
CORE = ["data.py", "model.py", "engine.py", "run.py", "validate.py", "adapter.py", "check_model.py", "check_adapter.py"]
ACTIVE = None
ACTIVE_SHA = None
STARTED = time.time()
ROWS = None


def fingerprint():
    sources = {name: sha(ROOT / name) for name in CORE}
    official = REPO / "falcon-challenge/falcon_challenge"
    if not official.exists():
        official = ROOT / "falcon_challenge"
    for name in ["config.py", "interface.py", "dataloaders.py", "evaluator.py"]:
        sources["falcon_challenge/" + name] = sha(official / name)
    return dict(sources=sources, data_manifest_sha256=sha(ROOT / "provenance/data_manifest.json"))


def config_key(c):
    return f"{c.family}_d{c.dimension}_k{c.temporal_bins}_lr{c.lr:g}"


def candidates(batch):
    result = [replace(Config(), family=f, dimension=d, temporal_bins=k, batch=batch)
              for f in FAMILIES[:3] for d in [256, 512] for k in [1, 8]]
    result += [replace(Config(), family="plain", temporal_bins=k, lr=lr, batch=batch)
               for k in [1, 8] for lr in [.0002, .0008]]
    return result


def freeze():
    current = fingerprint()
    path = ROOT / "provenance/protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol["fingerprint"] != current:
            raise RuntimeError("Core source or data changed after protocol freeze")
        return protocol
    validation = json.loads((ROOT / "results/validation.json").read_text())
    if not validation["passed"] or validation["fingerprint"] != current:
        raise RuntimeError("Current sources require a passing validation record")
    pilot = json.loads((ROOT / "results/pilot.json").read_text())
    if not pilot["complete"]:
        raise RuntimeError("Throughput pilot incomplete")
    protocol = dict(schema=1, study_id="h1_channel_temporal_hdc",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        primary_metric="Mean group-level variance-weighted R2 after 60 seconds labeled calibration",
        primary_contrast="HDC minus the strongest random/MLP family selected on inner data",
        secondary_metric="Zero evaluated-session labels on the identical scored suffix",
        score_start=3256, bin_seconds=.02, calibration_bins=3000, exclusion_bins=256,
        development_seeds=DEV_SEEDS, final_seeds=FINAL_SEEDS,
        representation_seed_rule="100000 + training seed; joint optimizer/codebook replicates",
        inner_design="All 15 excluded pairs; each four-group fit reused for two valid outer/inner assignments",
        selection="Mean inner-session few-shot R2 across five validation groups and five seeds; ties fewer trainable parameters then lexical configuration key",
        candidate_configs=[asdict(c) for c in candidates(pilot["chosen_batch"])],
        final_variants=FAMILIES + ["no_channel_roles", "no_roles", "no_gru", "no_online_norm", "no_smoothing", "no_temporal"],
        expected_fits=EXPECTED, epochs=50, fixed_readout_calibration_steps=20,
        adaptation="Unlabeled cumulative moments through current bin; no scored-suffix weight update",
        training_augmentation=dict(normalized_gaussian_sd=.15, channel_dropout_probability=.1),
        calibration_curve_seconds=[0, 15, 30, 60],
        robustness=dict(regime="zero_shot", start=3256, gain_log_sd=[0, .1, .2, .4], dropout_fraction=[0, .1, .2, .4]),
        stopping="Fixed epochs and complete matrix; no outcome-dependent search or extension",
        inference="Strict FP32, causal resets at recording boundaries, official target and mask, no target smoothing or lag tuning",
        scope="One participant; related recording groups. Local public data and server feedback had been encountered during earlier development. This experiment does not restore an untouched test set.",
        statistics="Average joint encoder/optimizer replicates within group; all group effects and descriptive group bootstrap intervals. Seeds do not increase biological sample size.",
        fingerprint=current, validation_sha256=sha(ROOT / "results/validation.json"),
        pilot_sha256=sha(ROOT / "results/pilot.json"))
    write_json(path, protocol)
    return protocol


def assert_frozen():
    if fingerprint() != ACTIVE["fingerprint"] or sha(ROOT / "provenance/protocol.json") != ACTIVE_SHA:
        raise RuntimeError("Frozen scientific sources or protocol changed")


def completed_rows():
    global ROWS
    if ROWS is None:
        ROWS = [json.loads(p.read_text()) for p in (ROOT / "results/runs").glob("*.json")]
    return ROWS


def status(**fields):
    rows = completed_rows()
    counts = {stage: sum(r["stage"] == stage for r in rows) for stage in EXPECTED}
    durations = [r["total_seconds"] / (len(r["signature"]["train_days"]) / 4) for r in rows]
    remaining = sum((EXPECTED[s] - counts[s]) * {"inner":1, "outer":1.25, "transfer":1.5}[s] for s in EXPECTED)
    eta = float(np.mean(durations[-40:]) * remaining) if durations else None
    write_json(ROOT / "results/status.json", dict(updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        pid=os.getpid(), counts=counts, expected=EXPECTED, elapsed_seconds=time.time() - STARTED,
        remaining_fit_seconds=eta, eta_note="Training and evaluation fits only; diagnostics and final release checks follow", **fields))


def job(identifier, config, seed, train, test, stage):
    assert_frozen()
    c = replace(config, codebook_seed=100000 + seed)
    signature = dict(config=asdict(c), seed=seed, train_days=sessions(train), test_days=sessions(test),
                     protocol_sha256=ACTIVE_SHA, fingerprint=ACTIVE["fingerprint"])
    result_path = ROOT / "results/runs" / (identifier + ".json")
    checkpoint = ROOT / "checkpoints" / (identifier + ".pt")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["signature"] != signature or result["checkpoint_sha256"] != sha(checkpoint):
            raise RuntimeError("Incompatible completed result: " + identifier)
        return result
    started = time.perf_counter()
    status(state="running", stage=stage, job=identifier, epoch=0)
    if checkpoint.exists():
        if json.loads(checkpoint.with_suffix(".json").read_text()) != signature:
            raise RuntimeError("Incompatible retained checkpoint")
        model, record = load(checkpoint)
        if record["config"] != asdict(c) or record["seed"] != seed or set(record["train_days"]) != set(sessions(train)):
            raise RuntimeError("Checkpoint contents disagree with signature")
    else:
        write_json(checkpoint.with_suffix(".json"), signature)
        model, record = fit(train, c, seed, checkpoint,
            progress=lambda epoch, loss: status(state="running", stage=stage, job=identifier, epoch=epoch, loss=loss))
    metrics = []
    prediction_hashes = {}
    for regime in ["zero_shot", "few_shot"]:
        prediction_path = ROOT / "results/predictions" / (identifier + "_" + regime + ".npz") if stage != "inner" else None
        metrics.extend(evaluate(model, record, test, regime, save=prediction_path))
        if prediction_path:
            prediction_hashes[prediction_path.name] = sha(prediction_path)
    assert_frozen()
    result = dict(id=identifier, stage=stage, signature=signature, metrics=metrics,
        parameters=record["parameters"], train_seconds=record["elapsed_seconds"],
        total_seconds=time.perf_counter() - started, training_losses=record["training_losses"],
        checkpoint_sha256=sha(checkpoint), prediction_sha256=prediction_hashes)
    write_json(result_path, result)
    completed_rows().append(result)
    print(json.dumps(dict(id=identifier, seconds=result["total_seconds"], metrics=metrics)), flush=True)
    return result


def select(days, configs):
    output = {}
    for outer in days:
        by_family = {}
        for family in FAMILIES:
            ranking = []
            for c in [c for c in configs if c.family == family]:
                values = []
                for inner in days:
                    if inner == outer:
                        continue
                    a, b = sorted([outer, inner], key=lambda d: int(d[1:]))
                    for seed in DEV_SEEDS:
                        path = ROOT / "results/runs" / f"inner_{a}_{b}_{config_key(c)}_s{seed}.json"
                        row = json.loads(path.read_text())
                        sig = row["signature"]
                        expected_config = asdict(replace(c, codebook_seed=100000 + seed))
                        if (sig["protocol_sha256"] != ACTIVE_SHA or sig["fingerprint"] != ACTIVE["fingerprint"]
                                or sig["config"] != expected_config or sig["seed"] != seed
                                or set(sig["train_days"]) != set(days) - {a, b}
                                or set(sig["test_days"]) != {a, b}):
                            raise RuntimeError("Inner provenance mismatch")
                        scores = [m["r2"] for m in row["metrics"] if m["day"] == inner and m["regime"] == "few_shot"]
                        if len(scores) != 1 or not np.isfinite(scores[0]):
                            raise RuntimeError("Missing inner score")
                        values += scores
                ranking.append(dict(config=asdict(c), key=config_key(c), score=float(np.mean(values)), parameters=Decoder(c).parameter_count()))
            by_family[family] = sorted(ranking, key=lambda r: (-r["score"], r["parameters"], r["key"]))
        control = sorted([by_family[f][0] for f in ["random", "mlp"]], key=lambda r: (-r["score"], r["parameters"], r["key"]))[0]
        output[outer] = dict(families=by_family, selected_control=control["config"]["family"])
    write_json(ROOT / "results/selection.json", output)
    return output


def ablations(c):
    return dict(no_channel_roles=replace(c, no_channel_roles=True), no_roles=replace(c, no_roles=True),
                no_gru=replace(c, recurrence=False), no_online_norm=replace(c, online_norm=False),
                no_smoothing=replace(c, smoothing=0), no_temporal=replace(c, temporal_bins=1))


def run(stage="all", limit=None):
    global ACTIVE, ACTIVE_SHA
    ACTIVE = freeze()
    ACTIVE_SHA = sha(ROOT / "provenance/protocol.json")
    records = load_h1(verify=True)
    days = sessions(records)
    if days != ["S0", "S1", "S2", "S3", "S4", "S5"]:
        raise RuntimeError("Unexpected outer groups")
    configs = [Config(**c) for c in ACTIVE["candidate_configs"]]
    count = 0
    if stage in ("all", "inner"):
        for a, b in itertools.combinations(days, 2):
            train = [r for r in records if r["day"] not in (a, b)]
            test = [r for r in records if r["day"] in (a, b)]
            for c in configs:
                for seed in DEV_SEEDS:
                    job(f"inner_{a}_{b}_{config_key(c)}_s{seed}", c, seed, train, test, "inner")
                    count += 1
                    if limit and count >= limit:
                        return
    if stage in ("all", "outer", "transfer"):
        selection = select(days, configs)
        if stage in ("all", "outer"):
            for day in days:
                variants = {f: Config(**selection[day]["families"][f][0]["config"]) for f in FAMILIES}
                variants.update(ablations(variants["hdc"]))
                for family, c in variants.items():
                    for seed in FINAL_SEEDS:
                        job(f"outer_{day}_{family}_s{seed}", c, seed,
                            [r for r in records if r["day"] != day], [r for r in records if r["day"] == day], "outer")
                        count += 1
                        if limit and count >= limit:
                            return
        if stage in ("all", "transfer"):
            transfer = {}
            for family in FAMILIES:
                rows = []
                for c in [c for c in configs if c.family == family]:
                    entries = [next(r for r in selection[day]["families"][family] if r["key"] == config_key(c)) for day in days]
                    rows.append(dict(config=asdict(c), score=float(np.mean([r["score"] for r in entries])), key=config_key(c), parameters=entries[0]["parameters"]))
                transfer[family] = sorted(rows, key=lambda r: (-r["score"], r["parameters"], r["key"]))[0]
            write_json(ROOT / "results/transfer_selection.json", transfer)
            for family, chosen in transfer.items():
                for seed in FINAL_SEEDS:
                    job(f"transfer_{family}_s{seed}", Config(**chosen["config"]), seed, records, load_h1("held-out-calib", verify=True), "transfer")
    status(state="complete", stage=stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "inner", "outer", "transfer"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        run(args.stage, args.limit)
    except BaseException:
        status(state="failed", error=traceback.format_exc())
        raise

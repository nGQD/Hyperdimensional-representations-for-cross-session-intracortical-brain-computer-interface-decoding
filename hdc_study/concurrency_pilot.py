"""Compare GPU process concurrency without computing or selecting on scores."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import argparse
import multiprocessing as mp
import os
import signal
import time
from data import ROOT, load_h1, write_json

RECORDS = None


def initialize():
    global RECORDS
    import torch
    RECORDS = [r for r in load_h1() if r["day"] not in ("S4", "S5")]
    torch.ones(1, device="cuda")


def task(family):
    from engine import fit
    from model import Config
    import torch
    torch.cuda.empty_cache()
    config = replace(Config(), family=family, dimension=512, temporal_bins=8, batch=256)
    model, record = fit(RECORDS, config, 900007, epochs=3)
    return dict(family=family, fit_seconds=record["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    training_pid = None if args.no_pause else json.loads((ROOT / "results/status.json").read_text())["pid"]
    rows = []
    if training_pid:
        os.kill(training_pid, signal.SIGSTOP)
    try:
        for workers in args.workers:
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize) as pool:
                    list(pool.map(task, ("plain",) * workers))
                    start = time.perf_counter()
                    results = list(pool.map(task, ("hdc", "random", "mlp", "plain") * 2))
                    row = dict(workers=workers, total_seconds=time.perf_counter()-start, fits=8, rows=results)
            except Exception as error:
                row = dict(workers=workers, failed=True, error=str(error))
            rows.append(row)
            write_json(ROOT / "results/concurrency_pilot.json", dict(complete=False, rows=rows))
            print(json.dumps(row), flush=True)
        feasible = [r for r in rows if not r.get("failed")]
        fastest = min(feasible, key=lambda r:r["total_seconds"])
        base = feasible[0]
        chosen = fastest["workers"] if fastest["total_seconds"] < base["total_seconds"] * .9 else base["workers"]
        write_json(ROOT / "results/concurrency_pilot.json", dict(complete=True, rows=rows, chosen_workers=chosen,
            rule="Use concurrency only when measured aggregate throughput improves by at least 10 percent",
            note="Same frozen minibatch/configuration; no evaluation score computed or used"))
    finally:
        if training_pid:
            os.kill(training_pid, signal.SIGCONT)


if __name__ == "__main__":
    main()

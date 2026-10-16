"""Select a shared minibatch size using throughput, without validation scores."""
from dataclasses import replace
import json
import time
import torch
from data import ROOT, load_h1, write_json
from model import Config
from engine import fit


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU study requires CUDA")
    records = [r for r in load_h1(verify=True) if r["day"] not in ("S4", "S5")]
    rows = []
    for batch in [128, 256, 512]:
        for family in ["hdc", "random", "mlp", "plain"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            c = replace(Config(), family=family, dimension=512, temporal_bins=8, batch=batch)
            started = time.perf_counter()
            model, record = fit(records, c, 900001, epochs=3)
            row = dict(family=family, batch=batch, seconds=time.perf_counter() - started,
                       seconds_per_epoch=record["elapsed_seconds"] / 3,
                       peak_memory_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                       parameters=record["parameters"])
            rows.append(row)
            write_json(ROOT / "results/pilot.json", dict(complete=False, rows=rows))
            print(json.dumps(row), flush=True)
            del model, record
    feasible = [b for b in [128, 256, 512]
                if max(r["peak_memory_gib"] for r in rows if r["batch"] == b) < 10]
    chosen = min(feasible, key=lambda b: sum(r["seconds_per_epoch"] for r in rows if r["batch"] == b))
    write_json(ROOT / "results/pilot.json", dict(complete=True, chosen_batch=chosen,
        amp=False, rows=rows, gpu=torch.cuda.get_device_name(),
        note="Three training epochs on four groups; no evaluation scores computed or used"))


if __name__ == "__main__":
    main()

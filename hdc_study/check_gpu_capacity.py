"""Stress the selected concurrency with three largest HDC training jobs."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import time
import torch
import concurrency_pilot
from data import ROOT, load_h1, write_json


def initialize_all():
    concurrency_pilot.RECORDS = load_h1(verify=True)
    torch.ones(1, device="cuda")


def main():
    started=time.time()
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn"), initializer=initialize_all) as pool:
        rows=list(pool.map(concurrency_pilot.task,["hdc"]*3))
    result=dict(passed=True, workers=3, train_groups=6, dimension=512, temporal_bins=8,
                batch=256, epochs=3, seconds=time.time()-started, rows=rows)
    write_json(ROOT / "results/gpu_capacity.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    main()

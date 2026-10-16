"""Run the frozen experiment and its evidence-generation steps sequentially."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from data import ROOT, REPO, write_json
from run import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stage", choices=("fits", "diagnostics", "report", "efficiency", "adaptation", "pareto", "figures", "package"), default="fits")
    args = parser.parse_args()
    lock = open(ROOT / "workflow.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A study workflow is already running")
    started = time.time()
    stage = "preflight"
    try:
        freeze()
        throughput = json.loads((ROOT / "results/concurrency_pilot.json").read_text())
        if not throughput.get("complete"):
            raise RuntimeError("GPU concurrency pilot is incomplete")
        workers = throughput["chosen_workers"]
        stages = [
            ("fits", ["parallel_run.py", "--stage", "all", "--workers", str(workers)]),
            ("diagnostics", ["diagnostics.py"]),
            ("report", ["report.py"]),
            ("efficiency", ["efficiency.py"]),
            ("adaptation", ["adaptation_cost.py"]),
            ("pareto", ["pareto.py"]),
            ("figures", ["graphics.py"]),
        ]
        if (REPO / "code_to_submit_github/README.md").is_file():
            stages.append(("package", ["package.py", "--prepare"]))
        names = [name for name, _ in stages]
        if args.start_stage not in names:
            raise RuntimeError(f"Stage is unavailable in this checkout: {args.start_stage}")
        if args.start_stage == "figures":
            # A resumed figure run skips report.py, so first ensure its
            # complete summary still describes the current fit/diagnostic,
            # selection, and prediction-audit files. The following package
            # stage performs the independent matched-control artifact check.
            from package import verify_current_summary_sources
            verify_current_summary_sources(ROOT)
        stages = stages[names.index(args.start_stage):]
        for stage, arguments in stages:
            write_json(ROOT / "results/workflow.json", dict(
                state="running", stage=stage, pid=os.getpid(), started_unix=started,
                gpu_workers=workers,
                updated_unix=time.time(), scheduled_stages=[s for s,_ in stages]))
            with open(ROOT / "results" / (stage + ".log"), "a", buffering=1) as log:
                subprocess.run([sys.executable, str(ROOT / arguments[0]), *arguments[1:]],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        completed_stages = [name for name, _ in stages]
        write_json(ROOT / "results/workflow.json", dict(
            state="stages_complete", pid=os.getpid(), started_unix=started,
            finished_unix=time.time(), start_stage=args.start_stage,
            completed_stages=completed_stages,
            note="Scheduled workflow stages completed. Check PREPARATION_STATUS.json and current result records for evidence completeness; author/public release checks still apply."))
    except BaseException:
        write_json(ROOT / "results/workflow.json", dict(state="failed", stage=stage,
            pid=os.getpid(), started_unix=started, updated_unix=time.time(), error=traceback.format_exc()))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()

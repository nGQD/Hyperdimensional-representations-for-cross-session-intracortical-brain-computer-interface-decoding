"""Reproduce the H1 analysis with ``python run.py --stage all``.

Add the packaged source directories so the study scripts can resolve sibling imports.
"""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "hdc_study"
if not STUDY.is_dir():
    raise FileNotFoundError(f"H1 study sources are missing: {STUDY}")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT / "falcon-challenge"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_cache():
    """Recreate only manifest-listed H1 caches without rewriting provenance."""
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    challenge = ROOT / "falcon-challenge"
    data_root = Path(os.environ.get("FALCON_H1_DATA", challenge / "data/000954"))
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"FALCON_H1_DATA must point to the downloaded H1/000954 folder: {data_root}")
    # Preserve manifest-relative paths while keeping raw recordings outside the archive.
    expected_data = challenge / "data/000954"
    expected_data.parent.mkdir(parents=True, exist_ok=True)
    if not expected_data.exists() and not expected_data.is_symlink():
        expected_data.symlink_to(data_root.resolve(), target_is_directory=True)

    manifest_path = STUDY / "provenance/data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir = STUDY / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        target = cache_dir / row["cache"]
        if target.is_file() and _sha256(target) == row["cache_sha256"]:
            continue
        recorded_path = ROOT / row["path"]
        candidates = [recorded_path] if recorded_path.is_file() else []
        if not candidates:
            candidates = list(data_root.rglob(Path(row["path"]).name))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one source NWB for {row['path']}; found {len(candidates)} under {data_root}")
        source = candidates[0]
        if _sha256(source) != row["sha256"]:
            raise ValueError(f"Source NWB hash differs from frozen manifest: {row['path']}")
        x, y, trial, mask = load_nwb(source, FalconTask.h1)
        np.savez(target, x=x.astype(np.float32), y=y.astype(np.float32),
                 trial=trial, mask=mask)
        if _sha256(target) != row["cache_sha256"]:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"Regenerated cache hash differs from frozen manifest: {row['cache']}; "
                "use the exact recorded FALCON loader and source data")


if "--help" not in sys.argv and "-h" not in sys.argv:
    prepare_cache()
if "--prepare-cache" in sys.argv:
    print("All frozen H1 cache hashes verified")
    raise SystemExit(0)
entry = "parallel_run.py" if "--workers" in sys.argv else "run.py"
sys.argv[0] = str(STUDY / entry)
runpy.run_path(str(STUDY / entry), run_name="__main__")

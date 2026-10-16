"""H1 recordings, content hashes, and chronological calibration masks."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sessions(records):
    return sorted({r["day"] for r in records}, key=lambda day: int(day[1:]))


def load_h1(split="held-in-calib", verify=False):
    manifest = json.loads((ROOT / "provenance/data_manifest.json").read_text())
    records = []
    for row in manifest:
        if row["split"] != split:
            continue
        path = ROOT / "cache" / row["cache"]
        if verify and sha(path) != row["cache_sha256"]:
            raise ValueError(f"Changed data cache: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            records.append(dict(row, **{k: archive[k].copy() for k in archive.files}))
    records.sort(key=lambda r: (int(r["day"][1:]), int(r["set"].split("_set_")[-1])))
    if not records:
        raise ValueError(f"No recordings for {split}")
    return records


def partition_day(records, prefix=3000, score_start=3256):
    if prefix < 0 or prefix > score_start:
        raise ValueError("Calibration must end before the scored suffix")
    offset = 0
    result = []
    for record in records:
        n = len(record["x"])
        indices = offset + np.arange(n)
        calibration = indices < prefix
        mask = record["mask"].astype(bool) & (indices >= score_start)
        result.append(dict(record, calibration=calibration, score_mask=mask))
        offset += n
    if offset <= score_start or sum(r["calibration"].sum() for r in result) != prefix:
        raise ValueError("Recording group is too short for the stated protocol")
    if not sum(r["score_mask"].sum() for r in result):
        raise ValueError("Scored suffix has no valid targets")
    return result


def cache_from_nwb():
    challenge = Path(os.environ.get("FALCON_CHALLENGE_PATH", REPO / "falcon-challenge"))
    sys.path.insert(0, str(challenge))
    from falcon_challenge.config import FalconTask, H1_NEW_TO_OLD
    from falcon_challenge.dataloaders import load_nwb
    data = Path(os.environ.get("FALCON_H1_DATA", challenge / "data/000954"))
    manifest = []
    (ROOT / "cache").mkdir(exist_ok=True)
    for split in ["held-in-calib", "held-out-calib"]:
        paths = sorted(p for p in data.rglob("*.nwb") if split in p.parent.name)
        if not paths:
            raise FileNotFoundError(f"No {split} NWB files under {data}")
        for path in paths:
            x, y, trial, mask = load_nwb(path, FalconTask.h1)
            out = ROOT / "cache" / (path.stem + ".npz")
            np.savez(out, x=x.astype(np.float32), y=y.astype(np.float32), trial=trial, mask=mask)
            group = H1_NEW_TO_OLD[path.stem.split("ses-")[-1]]
            manifest.append(dict(path=str(path.relative_to(data)), sha256=sha(path),
                cache=out.name, cache_sha256=sha(out), split=split,
                day=group.split("_set_")[0], set=group, bins=len(x), scored_bins=int(mask.sum()),
                loader_sha256=sha(challenge / "falcon_challenge/dataloaders.py")))
    write_json(ROOT / "provenance/data_manifest.json", manifest)


if __name__ == "__main__":
    cache_from_nwb()

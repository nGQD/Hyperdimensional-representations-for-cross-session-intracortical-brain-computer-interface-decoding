"""Download and hash-check the H1 NWB assets listed by the frozen manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "hdc_study/provenance/data_manifest.json"
ASSET_PREFIX = "falcon-challenge/data/000954/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="External local folder for downloaded H1/000954 files")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in rows:
        path = row["path"]
        if not path.startswith(ASSET_PREFIX):
            raise ValueError(f"Unexpected data-manifest path: {path}")
        asset = path[len(ASSET_PREFIX):]
        destination = output / asset
        if not destination.is_file() or sha256(destination) != row["sha256"]:
            subprocess.run([
                "dandi", "download", "--existing", "overwrite",
                "--output-dir", str(output), f"dandi://DANDI/000954/{asset}",
            ], check=True)
        if not destination.is_file() or sha256(destination) != row["sha256"]:
            raise ValueError(
                f"Downloaded asset does not match the frozen SHA256 for {asset}; "
                "the current DANDI version differs from the study data")
    print(f"Verified {len(rows)} manifest-listed H1 files in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

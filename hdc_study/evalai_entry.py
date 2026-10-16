"""Run the official evaluator with the prespecified HDC transfer checkpoint."""
import argparse
from pathlib import Path
from data import ROOT
from adapter import H1Decoder
from package import validate_transfer_artifacts
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator


def _submission_artifacts(root: Path = ROOT) -> tuple[Path, Path]:
    """Resolve the packaged deployment pair or the author-workspace pair."""
    identifier = "transfer_hdc_s101"
    deployment = root.parent / "deployment"
    packaged_checkpoint = deployment / (identifier + ".pt")
    packaged_result = deployment / (identifier + ".json")
    if packaged_checkpoint.is_file() or packaged_result.is_file():
        if not packaged_checkpoint.is_file() or not packaged_result.is_file():
            raise FileNotFoundError("Packaged deployment checkpoint and metadata must appear together")
        return packaged_checkpoint, packaged_result
    return (root / "checkpoints" / (identifier + ".pt"),
            root / "results/runs" / (identifier + ".json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("minival", "test"), default="test")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    checkpoint, result_path = _submission_artifacts()
    _, checkpoint_record = validate_transfer_artifacts(result_path, checkpoint, ROOT)
    decoder = H1Decoder(FalconConfig(task=FalconTask.h1), checkpoint_record,
                         batch_size=args.batch_size)
    evaluator = FalconEvaluator(eval_remote=args.remote, split="h1", dataloader_workers=0)
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()

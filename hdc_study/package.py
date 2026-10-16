"""Build a deterministic, H1-only code package with explicit release gates.

``--prepare`` builds a clearly labeled preparatory folder and ZIP from source
files that currently exist, recording every missing file and metadata blocker.
``--final`` refuses to build unless the complete current result record,
checkpoint, protocol, dependencies, license, FALCON evaluator provenance and
author/repository metadata are all present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "hdc_study"
PACKAGE = REPO / "code_to_submit_github"
DIST = STUDY / "dist"

STATIC_FILES = (
    "README.md", "DATASET.md", "CITATION.cff", "release_metadata.json",
    "data_download_manifest.json", "environment.lock.yml", "run.py",
    "download_h1_data.py",
)

# Source path -> package-relative path. Published analysis records are mapped
# under evidence/ so they cannot be mistaken for writable fit outputs.
# Runtime prerequisites remain under hdc_study/results/ for a clean rerun.
SOURCE_FILES = {
    "hdc_study/__init__.py": "hdc_study/__init__.py",
    "hdc_study/model.py": "hdc_study/model.py",
    "hdc_study/engine.py": "hdc_study/engine.py",
    "hdc_study/data.py": "hdc_study/data.py",
    "hdc_study/run.py": "hdc_study/run.py",
    "hdc_study/parallel_run.py": "hdc_study/parallel_run.py",
    "hdc_study/concurrency_pilot.py": "hdc_study/concurrency_pilot.py",
    "hdc_study/check_gpu_capacity.py": "hdc_study/check_gpu_capacity.py",
    "hdc_study/evalai_entry.py": "hdc_study/evalai_entry.py",
    "hdc_study/validate.py": "hdc_study/validate.py",
    "hdc_study/pilot.py": "hdc_study/pilot.py",
    "hdc_study/workflow.py": "hdc_study/workflow.py",
    "hdc_study/report.py": "hdc_study/report.py",
    "hdc_study/diagnostics.py": "hdc_study/diagnostics.py",
    "hdc_study/adapter.py": "hdc_study/adapter.py",
    "hdc_study/graphics.py": "hdc_study/graphics.py",
    "hdc_study/efficiency.py": "hdc_study/efficiency.py",
    "hdc_study/adaptation_cost.py": "hdc_study/adaptation_cost.py",
    "hdc_study/pareto.py": "hdc_study/pareto.py",
    "hdc_study/package.py": "hdc_study/package.py",
    "hdc_study/check_model.py": "hdc_study/check_model.py",
    "hdc_study/check_adapter.py": "hdc_study/check_adapter.py",
    "hdc_study/matched_controls.py": "hdc_study/matched_controls.py",
    "hdc_study/control_implementation.py": "hdc_study/control_implementation.py",
    "hdc_study/theory.tex": "hdc_study/theory.tex",
    "hdc_study/requirements.txt": "requirements.txt",
    "hdc_study/provenance/data_manifest.json": "hdc_study/provenance/data_manifest.json",
    "hdc_study/provenance/protocol.json": "hdc_study/provenance/protocol.json",
    "hdc_study/provenance/capacity_controls_protocol.json": "hdc_study/provenance/capacity_controls_protocol.json",
    "hdc_study/provenance/matched_controls.frozen.py": "hdc_study/provenance/matched_controls.frozen.py",
    "hdc_study/provenance/capacity_controls_implementation_correction.json": "hdc_study/provenance/capacity_controls_implementation_correction.json",
    "hdc_study/results/validation.json": "hdc_study/results/validation.json",
    "hdc_study/results/pilot.json": "hdc_study/results/pilot.json",
    "hdc_study/results/concurrency_pilot.json": "hdc_study/results/concurrency_pilot.json",
    "hdc_study/results/gpu_capacity.json": "hdc_study/results/gpu_capacity.json",
    "hdc_study/results/selection.json": "evidence/hdc_study/results/selection.json",
    "hdc_study/results/prediction_audit.json": "evidence/hdc_study/results/prediction_audit.json",
    "hdc_study/results/diagnostics/status.json": "evidence/hdc_study/results/diagnostics/status.json",
    "hdc_study/results/summary.json": "evidence/hdc_study/results/summary.json",
    "hdc_study/results/efficiency.json": "evidence/hdc_study/results/efficiency.json",
    "hdc_study/results/adaptation_cost.json": "evidence/hdc_study/results/adaptation_cost.json",
    "hdc_study/results/pareto.json": "evidence/hdc_study/results/pareto.json",
    "hdc_study/results/transfer_selection.json": "evidence/hdc_study/results/transfer_selection.json",
    "hdc_study/results/runs/transfer_hdc_s101.json": "evidence/hdc_study/results/runs/transfer_hdc_s101.json",
    "hdc_study/results/group_seed_scores.csv": "evidence/hdc_study/results/group_seed_scores.csv",
    "falcon-challenge/LICENSE": "falcon-challenge/LICENSE",
    "falcon-challenge/falcon_challenge/__init__.py": "falcon-challenge/falcon_challenge/__init__.py",
    "falcon-challenge/falcon_challenge/config.py": "falcon-challenge/falcon_challenge/config.py",
    "falcon-challenge/falcon_challenge/interface.py": "falcon-challenge/falcon_challenge/interface.py",
    "falcon-challenge/falcon_challenge/dataloaders.py": "falcon-challenge/falcon_challenge/dataloaders.py",
    "falcon-challenge/falcon_challenge/evaluator.py": "falcon-challenge/falcon_challenge/evaluator.py",
}

# These provenance files are also copied beside the archived results, where
# readers such as graphics.load_summary resolve their protocol dependency.
DUPLICATE_SOURCE_FILES = {
    "hdc_study/provenance/data_manifest.json": (
        "evidence/hdc_study/provenance/data_manifest.json",
    ),
    "hdc_study/provenance/protocol.json": (
        "evidence/hdc_study/provenance/protocol.json",
    ),
    "hdc_study/provenance/capacity_controls_protocol.json": (
        "evidence/hdc_study/provenance/capacity_controls_protocol.json",
    ),
    "hdc_study/provenance/matched_controls.frozen.py": (
        "evidence/hdc_study/provenance/matched_controls.frozen.py",
    ),
    "hdc_study/provenance/capacity_controls_implementation_correction.json": (
        "evidence/hdc_study/provenance/capacity_controls_implementation_correction.json",
    ),
    "hdc_study/results/runs/transfer_hdc_s101.json": (
        "deployment/transfer_hdc_s101.json",
    ),
}

# These directories contain small, machine-readable evidence records. Keep
# the patterns narrow: logs, worker state, prediction arrays, and matched
# control checkpoints remain outside the package.
RESULT_JSON_GLOBS = {
    "hdc_study/results/runs/*.json": "evidence/hdc_study/results/runs",
    "hdc_study/results/diagnostics/*.json": "evidence/hdc_study/results/diagnostics",
    "hdc_study/results/matched_controls/*.json": "evidence/hdc_study/results/matched_controls",
}

# Exactly one deterministic EvalAI model is distributed. Search-seed models,
# optimizer state, caches, and matched-control checkpoints stay local.
FINAL_CHECKPOINT_SOURCE = "hdc_study/checkpoints/transfer_hdc_s101.pt"
FINAL_CHECKPOINT = "deployment/transfer_hdc_s101.pt"
FALCON_UPSTREAM_REPOSITORY = "https://github.com/nGQD/Falcon-BCI"
FALCON_SOURCE_FILES = (
    "falcon-challenge/LICENSE",
    "falcon-challenge/falcon_challenge/__init__.py",
    "falcon-challenge/falcon_challenge/config.py",
    "falcon-challenge/falcon_challenge/interface.py",
    "falcon-challenge/falcon_challenge/dataloaders.py",
    "falcon-challenge/falcon_challenge/evaluator.py",
)
FALCON_RECORD_PATHS = tuple(
    source.removeprefix("falcon-challenge/") for source in FALCON_SOURCE_FILES
)
FALCON_PATCH_PATH = "falcon-challenge/falcon_challenge/LOCAL_PATCH.diff"

GENERATED_DIRS = ("hdc_study", "falcon-challenge", "falcon_challenge", "docs", "provenance",
                  "results", "figures", "checkpoints", "evidence", "deployment", "__pycache__")
GENERATED_FILES = ("requirements.txt", "LICENSE", "PREPARATION_STATUS.json",
                   "PACKAGE_CONTENTS.json", "MANIFEST.sha256", "environment.lock.yml",
                   "falcon_evaluator_provenance.json", "data_download_manifest.json")
SECRET_PATTERNS = (
    re.compile(r"(?i)(evalai[_-]?token|api[_-]?key|password|secret)\s*[:=]\s*[^\s\"']+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._-]{16,}"),
)
# Compose root-path prefixes so this builder can be included in its own
# reproducible package without the content scanner mistaking the regex for a
# leaked machine-specific path.
ABSOLUTE_PATH = re.compile(r"(?:/" + r"home/|/" + r"Users/|[A-Z]:\\Users\\)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_transfer_artifacts(result_path: Path, checkpoint_path: Path,
                               study_root: Path) -> tuple[dict, dict]:
    """Verify the one published EvalAI pair against its frozen training data.

    The run record and checkpoint must describe the same H1 transfer fit, and
    both must be tied to the exact code, protocol, and input files included in
    the package.  ``study_root`` is the path to the packaged ``hdc_study``
    directory, so this check works in the author checkout and in the release.
    """
    result_path, checkpoint_path, study_root = map(Path, (result_path, checkpoint_path, study_root))
    result = _read_json(result_path)
    identifier = "transfer_hdc_s101"
    if result.get("id") != identifier or result.get("stage") != "transfer":
        raise ValueError("deployment result must be the prespecified transfer_hdc_s101 fit")
    signature = result.get("signature")
    config = signature.get("config") if isinstance(signature, dict) else None
    if (not isinstance(signature, dict) or signature.get("seed") != 101
            or not isinstance(config, dict) or config.get("family") != "hdc"):
        raise ValueError("deployment result must identify the HDC transfer fit with seed 101")
    if result.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("deployment run-record checkpoint_sha256 does not match the checkpoint")

    manifest_path = study_root / "provenance/data_manifest.json"
    protocol_path = study_root / "provenance/protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = _read_json(protocol_path)
    selection_candidates = (
        study_root / "results/transfer_selection.json",
        study_root.parent / "evidence/hdc_study/results/transfer_selection.json",
    )
    selection_path = next((path for path in selection_candidates if path.is_file()), None)
    if selection_path is None:
        raise ValueError("deployment validation requires the frozen HDC transfer selection record")
    selection = _read_json(selection_path)
    selected_hdc = selection.get("hdc")
    selected_config = selected_hdc.get("config") if isinstance(selected_hdc, dict) else None
    if not isinstance(selected_config, dict):
        raise ValueError("frozen transfer selection is missing the selected HDC configuration")
    expected_config = dict(selected_config)
    expected_config["codebook_seed"] = 100_000 + signature["seed"]
    if config != expected_config:
        raise ValueError("deployment configuration does not match frozen HDC transfer selection")
    selected_parameters = selected_hdc.get("parameters")
    if (not isinstance(selected_parameters, int)
            or result.get("parameters") != selected_parameters):
        raise ValueError("deployment parameter count does not match frozen HDC transfer selection")

    held_in = [row for row in manifest if row.get("split") == "held-in-calib"]
    held_out = [row for row in manifest if row.get("split") == "held-out-calib"]
    expected_train_days = sorted({row["day"] for row in held_in}, key=lambda day: int(day[1:]))
    expected_test_days = sorted({row["day"] for row in held_out}, key=lambda day: int(day[1:]))
    if len(expected_train_days) != 6 or len(expected_test_days) != 7:
        raise ValueError("frozen H1 manifest must contain six held-in and seven held-out groups")
    if (signature.get("train_days") != expected_train_days
            or signature.get("test_days") != expected_test_days):
        raise ValueError("deployment run record does not use the frozen H1 transfer groups")
    if (signature.get("protocol_sha256") != _sha256(protocol_path)
            or signature.get("fingerprint") != protocol.get("fingerprint")):
        raise ValueError("deployment run record does not match the frozen protocol fingerprint")

    fingerprint = signature["fingerprint"]
    if fingerprint.get("data_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("deployment data manifest does not match the frozen protocol fingerprint")
    sources = fingerprint.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("deployment run record is missing its source fingerprint")
    for name, digest in sources.items():
        if name.startswith("falcon_challenge/"):
            source_path = (study_root.parent / "falcon-challenge" / "falcon_challenge"
                           / name.split("/", 1)[1])
        else:
            source_path = study_root / name
        if not source_path.is_file() or _sha256(source_path) != digest:
            raise ValueError(f"deployment source fingerprint mismatch: {name}")

    expected_train_files = sorted(
        ({"path": row["path"], "sha256": row["sha256"]} for row in held_in),
        key=lambda row: row["path"],
    )
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"cannot load the deployment checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("deployment checkpoint must contain a training record")
    actual_train_files = checkpoint.get("train_files")
    if not isinstance(actual_train_files, list):
        raise ValueError("deployment checkpoint is missing its training file hashes")
    actual_train_files = sorted(actual_train_files, key=lambda row: row.get("path", ""))
    if (checkpoint.get("config") != config
            or checkpoint.get("seed") != signature["seed"]
            or checkpoint.get("train_days") != expected_train_days
            or actual_train_files != expected_train_files
            or checkpoint.get("parameters") != result.get("parameters")):
        raise ValueError("deployment checkpoint training provenance does not match its run record and H1 manifest")
    return result, checkpoint


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        try:
            display_path = path.relative_to(REPO).as_posix()
        except ValueError:
            display_path = path.name
        raise ValueError(f"Cannot read required JSON metadata {display_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _project_git_root() -> Path | None:
    """Return this checkout's Git root, ignoring unrelated parent repositories."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        root = Path(result.stdout.strip()).resolve(strict=True)
        return root if root == REPO.resolve(strict=True) else None
    except (OSError, ValueError):
        return None


def _validate_evaluator_provenance(path: Path, archive_root: Path) -> dict:
    """Validate provenance schema and hashes for every vendored evaluator file."""
    record = _read_json(path)
    if record.get("upstream_repository") != FALCON_UPSTREAM_REPOSITORY:
        raise ValueError("FALCON evaluator provenance has an unexpected upstream repository")
    commit = record.get("upstream_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("FALCON evaluator provenance has an invalid upstream commit")
    if record.get("license") != "MIT" or record.get("license_file") != "falcon-challenge/LICENSE":
        raise ValueError("FALCON evaluator provenance has an invalid license declaration")
    if not isinstance(record.get("note"), str) or not record["note"].strip():
        raise ValueError("FALCON evaluator provenance is missing its explanatory note")

    rows = record.get("files")
    if not isinstance(rows, list) or len(rows) != len(FALCON_RECORD_PATHS):
        raise ValueError("FALCON evaluator provenance has an incomplete file list")
    by_path = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("FALCON evaluator provenance file entries must be objects")
        row_path = row.get("path")
        if row_path not in FALCON_RECORD_PATHS or row_path in by_path:
            raise ValueError("FALCON evaluator provenance has an unexpected or duplicate file path")
        upstream_hash, vendored_hash = row.get("upstream_sha256"), row.get("vendored_sha256")
        if (not isinstance(upstream_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", upstream_hash)
                or not isinstance(vendored_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", vendored_hash)
                or not isinstance(row.get("locally_modified"), bool)
                or row["locally_modified"] != (upstream_hash != vendored_hash)):
            raise ValueError(f"FALCON evaluator provenance has invalid hashes or status for {row_path}")
        vendor_file = archive_root / "falcon-challenge" / row_path
        if not vendor_file.is_file() or _sha256(vendor_file) != vendored_hash:
            raise ValueError(f"FALCON evaluator provenance hash mismatch: {row_path}")
        by_path[row_path] = row
    if set(by_path) != set(FALCON_RECORD_PATHS):
        raise ValueError("FALCON evaluator provenance does not cover all vendored files")

    patch_file = record.get("local_patch_file")
    has_local_changes = any(row["locally_modified"] for row in by_path.values())
    if patch_file not in (None, FALCON_PATCH_PATH):
        raise ValueError("FALCON evaluator provenance has an unexpected patch path")
    if has_local_changes:
        patch_path = archive_root / FALCON_PATCH_PATH
        if patch_file != FALCON_PATCH_PATH or not patch_path.is_file() or patch_path.stat().st_size == 0:
            raise ValueError("FALCON evaluator provenance requires its nonempty local patch")
    elif patch_file is not None:
        raise ValueError("FALCON evaluator provenance declares a patch without modified files")
    return record


def _load_summary(path: Path) -> dict:
    """Load the sibling graphics module when invoked as a script or package."""
    if __package__:
        from .graphics import load_summary
    else:
        from graphics import load_summary
    return load_summary(path)


def verify_current_summary_sources(study_root: Path = STUDY,
                                   summary: dict | None = None) -> dict:
    """Bind a complete summary to the current fit, diagnostic, audit, and selection files."""
    study_root = Path(study_root)
    if summary is None:
        summary = _load_summary(study_root / "results/summary.json")
    run_hashes = {
        path.name: _sha256(path)
        for path in sorted((study_root / "results/runs").glob("*.json"))
    }
    if summary.get("sources") != run_hashes:
        raise RuntimeError("Summary run-source hashes differ from current results/runs/*.json")
    diagnostic_hashes = {
        path.name: _sha256(path)
        for path in sorted((study_root / "results/diagnostics").glob("outer*.json"))
    }
    if summary.get("diagnostic_sources") != diagnostic_hashes:
        raise RuntimeError("Summary diagnostic hashes differ from current outer diagnostic records")
    selection = _read_json(study_root / "results/selection.json")
    if summary.get("selection") != selection:
        raise RuntimeError("Summary selection differs from current results/selection.json")
    audit_path = study_root / "results/prediction_audit.json"
    if summary.get("prediction_audit_sha256") != _sha256(audit_path):
        raise RuntimeError("Summary prediction-audit hash differs from the current audit file")
    return summary


def verify_current_completed_evidence(study_root: Path = STUDY,
                                      summary: dict | None = None) -> dict:
    """Verify current controls and bind every report source hash before release."""
    study_root = Path(study_root)
    summary = verify_current_summary_sources(study_root, summary)
    # Study modules use the same script-friendly top-level imports as the
    # documented `python hdc_study/...py` commands. Keep them portable when
    # this builder is imported with `python -m hdc_study.package` too.
    study_path = str(study_root.resolve())
    if study_path not in sys.path:
        sys.path.insert(0, study_path)
    from matched_controls import verify_completed_matrix
    verified_controls = verify_completed_matrix()
    verify_summary_freshness(summary, verified_controls, study_root)
    return summary


def verify_summary_freshness(summary: dict, verified_controls: dict,
                             study_root: Path = STUDY) -> None:
    """Bind the summary to current matched-control records and audit files.

    This mirrors the package-independent evidence check without importing the
    manuscript compiler, so the compact code release can validate results
    without bundling manuscript sources or a PDF build tree.
    """
    study_root = Path(study_root)
    matched = summary.get("matched_capacity_controls")
    rows = verified_controls.get("rows") if isinstance(verified_controls, dict) else None
    if not isinstance(matched, dict) or not isinstance(rows, dict):
        raise RuntimeError("Summary or verified matched-control records are malformed")

    row_fields = (
        "id", "day", "family", "seed", "config", "protocol_sha256",
        "capacity_protocol_sha256", "matched_hdc_identifier", "matched_hdc_config",
        "reused_identifier", "checkpoint", "predictions", "prediction_sha256",
        "metrics", "parameters",
    )
    current_comparisons = {
        identifier: {key: row.get(key) for key in row_fields}
        for identifier, row in rows.items()
    }
    if matched.get("comparisons") != current_comparisons:
        raise RuntimeError("Report matched-control records differ from current verified records")

    results_dir = study_root / "results/matched_controls"
    control_results = {}
    checkpoints = {}
    predictions = {}
    for row in rows.values():
        result_path = results_dir / f"{row['day']}_{row['family']}_s{row['seed']}.json"
        control_results[result_path.name] = _sha256(result_path)
        checkpoint_ref = row["checkpoint"]
        checkpoint_path = study_root / checkpoint_ref["path"]
        checkpoints[checkpoint_ref["path"]] = _sha256(checkpoint_path)
        for prediction_ref in row["predictions"].values():
            prediction_path = study_root / prediction_ref["path"]
            predictions[prediction_ref["path"]] = _sha256(prediction_path)

    current_source_hashes = {
        "capacity_protocol": _sha256(study_root / "provenance/capacity_controls_protocol.json"),
        "runner": _sha256(study_root / "matched_controls.py"),
        "status": _sha256(results_dir / "status.json"),
        "summary": _sha256(results_dir / "summary.json"),
        "results": control_results,
        "checkpoints": checkpoints,
        "predictions": predictions,
    }
    if matched.get("source_hashes") != current_source_hashes:
        raise RuntimeError("Report matched-control source hashes differ from current verified files")
    audit_path = study_root / "results/prediction_audit.json"
    if summary.get("prediction_audit_sha256") != _sha256(audit_path):
        raise RuntimeError("Report prediction-audit hash differs from the current audit file")


def _static_paths() -> list[Path]:
    return [PACKAGE / name for name in STATIC_FILES]


def _expanded_source_files() -> dict[str, str]:
    """Expand only the explicitly approved small result-record patterns."""
    sources = dict(SOURCE_FILES)
    for pattern, destination_dir in RESULT_JSON_GLOBS.items():
        for path in sorted(REPO.glob(pattern)):
            source = path.relative_to(REPO).as_posix()
            destination = (Path(destination_dir) / path.name).as_posix()
            sources.setdefault(source, destination)
    return sources


def _source_paths() -> list[Path]:
    return [REPO / source for source in _expanded_source_files()]


def blockers() -> list[str]:
    issues = []
    for path in _static_paths():
        if not path.is_file():
            issues.append(f"missing package documentation: {path.relative_to(REPO)}")
    for source in SOURCE_FILES:
        if not (REPO / source).is_file():
            issues.append(f"missing required source/artifact: {source}")
    if not (REPO / FINAL_CHECKPOINT_SOURCE).is_file():
        issues.append(f"missing deterministic EvalAI checkpoint: {FINAL_CHECKPOINT_SOURCE}")

    metadata_path = PACKAGE / "release_metadata.json"
    try:
        meta = _read_json(metadata_path)
    except ValueError as exc:
        issues.append(str(exc))
        meta = {}
    if meta.get("status") != "complete":
        issues.append("release_metadata.json status must be 'complete'")
    if meta.get("final_manuscript_approved") is not True:
        issues.append("release_metadata.json needs author confirmation of final manuscript approval")
    repository_url = meta.get("repository_url", "")
    if not isinstance(repository_url, str) or not re.fullmatch(
            r"https://github\.com/[^/\s]+/[^/\s]+/?", repository_url):
        issues.append("release_metadata.json needs the author-approved public GitHub repository URL")
    authors = meta.get("authors")
    if not isinstance(authors, list) or not authors:
        issues.append("release_metadata.json needs the author-approved author list")
    else:
        for i, author in enumerate(authors):
            if not isinstance(author, dict) or not all(
                    isinstance(author.get(k), str) and author[k].strip()
                    for k in ("given-names", "family-names")):
                issues.append(f"author {i + 1} needs approved given-names and family-names")
    contributions = meta.get("contributions")
    if not isinstance(contributions, list) or len(contributions) != (len(authors) if isinstance(authors, list) else -1):
        issues.append("release_metadata.json needs author-approved contribution statements for each author")
    elif any(not isinstance(item, str) or not item.strip() for item in contributions):
        issues.append("release_metadata.json needs nonempty author-approved contribution statements")
    if not isinstance(meta.get("license"), str) or not meta["license"].strip():
        issues.append("release_metadata.json needs the selected license")
    if not isinstance(meta.get("reproduction_command"), str) or meta["reproduction_command"].strip() != "python run.py --stage all":
        issues.append("release_metadata.json must record the verified one-command run: python run.py --stage all")
    evaluator_commit = meta.get("falcon_evaluator_commit", "")
    evaluator_provenance_path = PACKAGE / "falcon_evaluator_provenance.json"
    if not isinstance(evaluator_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", evaluator_commit):
        issues.append("release_metadata.json needs the exact upstream FALCON evaluator commit SHA")
    elif not evaluator_provenance_path.is_file():
        issues.append("missing generated FALCON upstream/local patch provenance record")
    else:
        try:
            provenance = _validate_evaluator_provenance(evaluator_provenance_path, PACKAGE)
            if evaluator_commit != provenance["upstream_commit"]:
                issues.append("release_metadata.json FALCON commit does not match the vendored source provenance")
        except ValueError as exc:
            issues.append(f"FALCON evaluator provenance is invalid: {exc}")
    if not (PACKAGE / "LICENSE").is_file():
        issues.append("missing selected package LICENSE text")

    citation_path = PACKAGE / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8")
        if "authors: []" in citation:
            issues.append("CITATION.cff needs the author-approved author list")
        if 'repository-code: ""' in citation or repository_url not in citation:
            issues.append("CITATION.cff repository-code must match the public repository URL")
        if 'license: ""' in citation or (isinstance(meta.get("license"), str)
                                         and meta["license"] not in citation):
            issues.append("CITATION.cff license must match release_metadata.json")
        if isinstance(authors, list):
            for i, author in enumerate(authors):
                if isinstance(author, dict) and not all(
                        author.get(key, "") in citation for key in ("given-names", "family-names")):
                    issues.append(f"CITATION.cff author {i + 1} must match release_metadata.json")

    data_manifest_path = PACKAGE / "data_download_manifest.json"
    try:
        data_manifest = _read_json(data_manifest_path)
    except ValueError as exc:
        issues.append(str(exc))
        data_manifest = {}
    if data_manifest.get("status") != "complete":
        issues.append("data_download_manifest.json status must be 'complete'")
    for key in ("local_data_root", "download_command"):
        if not isinstance(data_manifest.get(key), str) or not data_manifest[key].strip():
            issues.append(f"data_download_manifest.json needs {key}")
    if data_manifest.get("raw_data_in_package") is not False:
        issues.append("data_download_manifest.json must state raw_data_in_package=false")

    lock_path = PACKAGE / "environment.lock.yml"
    if not lock_path.is_file():
        issues.append("missing exact environment.lock.yml (including the pinned CUDA/PyTorch build)")
    else:
        lock_text = lock_path.read_text(encoding="utf-8")
        if not re.search(r"(?m)^\s*-\s*torch(?:==|=)[^\s]+", lock_text):
            issues.append("environment.lock.yml must pin the exact PyTorch version")
        if not re.search(r"(?m)^\s*-\s*python(?:==|=)[^\s]+", lock_text):
            issues.append("environment.lock.yml must pin the exact Python version")

    summary_path = REPO / "hdc_study/results/summary.json"
    current_summary = None
    if not summary_path.is_file():
        issues.append("missing current experiment result hdc_study/results/summary.json")
    else:
        try:
            current_summary = _load_summary(summary_path)
        except Exception as exc:  # release gate reports exact current schema failure
            issue = str(exc).replace(str(REPO) + "/", "")
            issues.append(f"current result summary is not complete/valid: {issue}")

    # Capacity controls are a frozen secondary analysis.  Require its exact
    # protocol-bound matrix before a final package can be produced.
    capacity_path = REPO / "hdc_study/provenance/capacity_controls_protocol.json"
    primary_protocol_path = REPO / "hdc_study/provenance/protocol.json"
    controls_dir = REPO / "hdc_study/results/matched_controls"
    status_path = controls_dir / "status.json"
    controls_summary_path = controls_dir / "summary.json"
    expected_ids = set()
    try:
        capacity = _read_json(capacity_path)
        days = capacity.get("days")
        families = capacity.get("families")
        seeds = capacity.get("final_seeds")
        if (not isinstance(days, list) or not isinstance(families, dict)
                or not isinstance(seeds, list)):
            raise ValueError("days, families, and final_seeds are required")
        expected_ids = {
            f"{day}_{family}_s{seed}.json"
            for day in days for family in families for seed in seeds
        }
        if (capacity.get("design_state") != "frozen_before_outer_results"
                or capacity.get("comparison_count") != 120
                or len(expected_ids) != 120
                or capacity.get("primary_endpoint_unchanged") is not True
                or capacity.get("outcome_dependent_choices") is not False):
            issues.append("capacity-controls protocol is not the frozen 120-comparison design")
        if primary_protocol_path.is_file() and capacity.get("primary_protocol_sha256") != _sha256(primary_protocol_path):
            issues.append("capacity-controls protocol does not match the packaged primary protocol")
    except (ValueError, KeyError, TypeError) as exc:
        issues.append(f"capacity-controls protocol is incomplete: {exc}")

    try:
        if __package__:
            from .control_implementation import verify_control_implementation
        else:
            from control_implementation import verify_control_implementation
        verify_control_implementation(STUDY)
    except Exception as exc:
        issues.append(f"matched-controls implementation provenance is invalid: {exc}")

    control_status = {}
    try:
        control_status = _read_json(status_path)
        if (control_status.get("state") != "complete"
                or control_status.get("completed") != 120
                or control_status.get("expected") != 120
                or control_status.get("family_counts") != {"random": 60, "mlp": 60}):
            issues.append("matched-controls status must report all 120 comparisons complete")
        if (capacity_path.is_file()
                and control_status.get("capacity_protocol_sha256") != _sha256(capacity_path)):
            issues.append("matched-controls status does not match the frozen capacity protocol")
        if (primary_protocol_path.is_file()
                and control_status.get("primary_protocol_sha256") != _sha256(primary_protocol_path)):
            issues.append("matched-controls status does not match the primary protocol")
    except ValueError as exc:
        issues.append(str(exc))
    try:
        control_summary = _read_json(controls_summary_path)
        if (control_summary.get("comparison_count") != 120
                or control_summary.get("family_counts") != {"random": 60, "mlp": 60}
                or not isinstance(control_summary.get("comparisons"), list)
                or len(control_summary["comparisons"]) != 120):
            issues.append("matched-controls summary must contain all 120 comparisons")
        if (capacity_path.is_file()
                and control_summary.get("capacity_protocol_sha256") != _sha256(capacity_path)):
            issues.append("matched-controls summary does not match the frozen capacity protocol")
        if (primary_protocol_path.is_file()
                and control_summary.get("primary_protocol_sha256") != _sha256(primary_protocol_path)):
            issues.append("matched-controls summary does not match the primary protocol")
    except ValueError as exc:
        issues.append(str(exc))
    actual_ids = {path.name for path in controls_dir.glob("S[0-9]_*_s*.json")}
    if expected_ids and actual_ids != expected_ids:
        issues.append(f"matched-controls result files must equal the frozen 120 IDs (found {len(actual_ids)})")

    # A complete summary is release evidence, so validate its source maps and
    # independently verify all matched-control records before preparing or
    # freezing a completed-results package. An intentionally incomplete prep
    # workspace still packages with explicit blockers and missing inputs.
    if (current_summary is not None
            and control_status.get("state") == "complete"
            and control_status.get("completed") == 120
            and control_status.get("expected") == 120):
        try:
            verify_current_completed_evidence(STUDY, current_summary)
        except Exception as exc:
            issues.append(f"current completed evidence is stale or invalid: {exc}")

    # The bundled evaluator source needs its upstream license and commit
    # declaration; local copy hashes are recorded in the generated manifest.
    for path in ("__init__.py", "config.py", "interface.py", "dataloaders.py", "evaluator.py"):
        if not (REPO / "falcon-challenge/falcon_challenge" / path).is_file():
            issues.append(f"missing vendored FALCON evaluator file: {path}")
    if not (REPO / "falcon-challenge/LICENSE").is_file():
        issues.append("missing upstream FALCON evaluator license")
    return issues


def _copy_if_present(source: Path, destination: Path, *, final: bool) -> bool:
    if not source.is_file():
        if final:
            raise FileNotFoundError(f"Required release input missing: {source}")
        return False
    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(REPO.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Release source must be a regular file inside the repository: {source}") from exc
    if source.suffix.lower() in {".nwb", ".npz", ".npy", ".env"}:
        raise ValueError(f"Raw or cached data are forbidden in the package: {source}")
    if (source.suffix.lower() == ".pt"
            and source.relative_to(REPO).as_posix() != FINAL_CHECKPOINT_SOURCE):
        raise ValueError(f"Only the prespecified transfer checkpoint may be packaged: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, destination)
    return True


def _copy_package_docs(stage: Path, *, final: bool) -> list[str]:
    missing = []
    for relative in STATIC_FILES:
        source = PACKAGE / relative
        if not _copy_if_present(source, stage / relative, final=final):
            missing.append(f"package documentation {relative}")
    # Include author-selected license only after it exists. The FALCON MIT
    # attribution is separately copied beneath vendor/falcon_challenge.
    if (PACKAGE / "LICENSE").is_file():
        _copy_if_present(PACKAGE / "LICENSE", stage / "LICENSE", final=final)
    elif final:
        raise FileNotFoundError(f"Selected code license is absent: {PACKAGE / 'LICENSE'}")
    return missing


def _copy_inputs(stage: Path, *, final: bool) -> list[str]:
    missing = []
    for source_name, destination_name in _expanded_source_files().items():
        source = REPO / source_name
        if not _copy_if_present(source, stage / destination_name, final=final):
            missing.append(source_name)
    for source_name, extra_destinations in DUPLICATE_SOURCE_FILES.items():
        source = REPO / source_name
        for destination_name in extra_destinations:
            _copy_if_present(source, stage / destination_name, final=final)
    checkpoint = REPO / FINAL_CHECKPOINT_SOURCE
    if not _copy_if_present(checkpoint, stage / FINAL_CHECKPOINT, final=final):
        missing.append(FINAL_CHECKPOINT_SOURCE)
    return missing


def _initialize_runtime_outputs(stage: Path) -> None:
    """Keep the fit writer's output directory present but free of evidence."""
    run_dir = stage / "hdc_study/results/runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".gitkeep").write_text(
        "Runtime fit records are generated here during reproduction.\n",
        encoding="utf-8",
    )


def _git_bytes(revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{revision}:{path}"],
        check=True, capture_output=True).stdout


def _write_evaluator_provenance(stage: Path) -> None:
    """Record upstream provenance without treating an unrelated Git tree as origin."""
    record_path = stage / "falcon_evaluator_provenance.json"
    patch_path = stage / FALCON_PATCH_PATH
    can_read_project_git = _project_git_root() == REPO.resolve()
    if can_read_project_git:
        try:
            metadata = _read_json(PACKAGE / "release_metadata.json")
            commit = metadata.get("falcon_evaluator_commit")
            if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("release metadata has no valid FALCON upstream commit")
            source_rows = []
            for source in FALCON_SOURCE_FILES:
                upstream = _git_bytes(commit, source)
                local = (REPO / source).read_bytes()
                source_rows.append({
                    "path": source.removeprefix("falcon-challenge/"),
                    "upstream_sha256": hashlib.sha256(upstream).hexdigest(),
                    "vendored_sha256": hashlib.sha256(local).hexdigest(),
                    "locally_modified": upstream != local,
                })
            patch = subprocess.run(
                ["git", "-C", str(REPO), "diff", commit, "--", *FALCON_SOURCE_FILES[1:]],
                check=True, capture_output=True).stdout
            if patch:
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch_path.write_bytes(patch)
            else:
                patch_path.unlink(missing_ok=True)
            record = {
                "upstream_repository": FALCON_UPSTREAM_REPOSITORY,
                "upstream_commit": commit,
                "license": "MIT",
                "license_file": "falcon-challenge/LICENSE",
                "files": source_rows,
                "local_patch_file": FALCON_PATCH_PATH if patch else None,
                "note": ("Vendored files preserve the exact local bytes used by the study. "
                         "The listed upstream commit is the FALCON base; LOCAL_PATCH.diff "
                         "records vendored source changes relative to that commit."),
            }
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _validate_evaluator_provenance(record_path, stage)
            return
        except (OSError, subprocess.CalledProcessError, ValueError):
            # A shallow or unrelated checkout may contain Git metadata without
            # retaining the actual upstream commit. Fall through to the
            # hash-validated provenance shipped with the source archive.
            pass

    included_record_path = PACKAGE / "falcon_evaluator_provenance.json"
    record = _validate_evaluator_provenance(included_record_path, PACKAGE)
    try:
        metadata = _read_json(PACKAGE / "release_metadata.json")
    except ValueError:
        metadata = {}
    metadata_commit = metadata.get("falcon_evaluator_commit")
    if (isinstance(metadata_commit, str) and re.fullmatch(r"[0-9a-f]{40}", metadata_commit)
            and metadata_commit != record["upstream_commit"]):
        raise ValueError("Included FALCON provenance commit differs from release metadata")
    if record.get("local_patch_file"):
        patch_source = PACKAGE / FALCON_PATCH_PATH
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(patch_source, patch_path)
    else:
        patch_path.unlink(missing_ok=True)
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_evaluator_provenance(record_path, stage)


def _write_download_manifest(stage: Path) -> None:
    """Bind the data acquisition instructions to exact source asset hashes."""
    rows = json.loads((stage / "hdc_study/provenance/data_manifest.json").read_text(encoding="utf-8"))
    prefix = "falcon-challenge/data/000954/"
    assets = []
    for row in rows:
        path = row.get("path", "")
        if not path.startswith(prefix):
            raise ValueError(f"Unexpected frozen H1 asset path: {path}")
        assets.append({
            "dandi_asset_path": path[len(prefix):],
            "sha256": row["sha256"],
            "split": row["split"],
            "session_group": row["day"],
        })
    record = {
        "status": "complete",
        "scope": "Only source files enumerated here from FALCON H1 Dandiset 000954",
        "dandiset": "DANDI:000954",
        "local_data_root": "$FALCON_H1_DATA",
        "download_command": "python download_h1_data.py --output-dir \"$FALCON_H1_DATA\"",
        "cache_command": "python run.py --stage all (hash-checks and creates missing local NPZ caches first)",
        "raw_data_in_package": False,
        "assets": assets,
        "notes": [
            "Asset paths and SHA256 values are sourced from hdc_study/provenance/data_manifest.json.",
            "The downloader rejects current DANDI files whose bytes differ from the frozen manifest.",
            "FALCON_H1_DATA is an external local directory; no NWB or NPZ cache is distributed.",
        ],
    }
    (stage / "data_download_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_contents(stage: Path) -> None:
    """Refuse environment-specific paths, secrets, raw data and stray files."""
    active_results = stage / "hdc_study/results"
    for directory in ("runs", "diagnostics", "matched_controls"):
        if list((active_results / directory).glob("*.json")):
            raise ValueError(
                f"Published {directory} JSONs must live under evidence/hdc_study/results"
            )
    active_checkpoints = stage / "hdc_study/checkpoints"
    if active_checkpoints.exists() and list(active_checkpoints.glob("*")):
        raise ValueError("Published checkpoints must live under deployment/")
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        if (path.suffix.lower() in {".nwb", ".npz", ".npy", ".env"}
                or path.name.lower().endswith(".pt.tmp")):
            raise ValueError(f"Forbidden data/cache/temp file in release stage: {path.relative_to(stage)}")
        if (path.suffix.lower() == ".pt"
                and path.relative_to(stage).as_posix() != FINAL_CHECKPOINT):
            raise ValueError(f"Unexpected checkpoint in release stage: {path.relative_to(stage)}")
        if path.suffix.lower() not in {".json", ".md", ".cff", ".txt", ".yml", ".yaml",
                                       ".py", ".tex", ".toml", ".sh", ".cff"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ABSOLUTE_PATH.search(text):
            raise ValueError(f"Environment-specific absolute path in {path.relative_to(stage)}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"Possible secret/token in {path.relative_to(stage)}")


def _write_status(stage: Path, missing: list[str], current_blockers: list[str], *, final: bool) -> None:
    # Blocker messages can contain an OSError's absolute filename. Keep the
    # generated status portable and make sure diagnostics cannot expose the
    # author workspace path in a preparatory archive.
    repository_roots = {str(REPO.resolve()), str(REPO)}
    portable_blockers = []
    for item in current_blockers:
        for root in repository_roots:
            item = item.replace(root, "<repository>")
        portable_blockers.append(item)
    status = {
        "status": "complete" if final else "preparatory",
        "complete_results": final,
        "included_current_inputs": not missing,
        "missing_allowlisted_inputs": sorted(set(missing)),
        "release_blockers": portable_blockers,
        "generated_utc": "SOURCE_DATE_EPOCH",
        "scope": "FALCON H1 only",
    }
    (stage / "PREPARATION_STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_content_list(stage: Path) -> None:
    entries = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"PACKAGE_CONTENTS.json", "MANIFEST.sha256"}:
            entries.append(path.relative_to(stage).as_posix())
    (stage / "PACKAGE_CONTENTS.json").write_text(
        json.dumps({"scope": "FALCON H1 only", "files": entries}, indent=2) + "\n",
        encoding="utf-8")


def _write_manifest(stage: Path) -> None:
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha256(path)}  {path.relative_to(stage).as_posix()}")
    (stage / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_zip(stage: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, strict_timestamps=True) as archive:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            arcname = PurePosixPath("code_to_submit_github") / path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(str(arcname), date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    temp.replace(destination)


def _replace_generated_folder(stage: Path) -> None:
    """Refresh generated package subtrees while retaining hand-written docs."""
    for name in GENERATED_DIRS:
        target = PACKAGE / name
        if target.exists():
            shutil.rmtree(target)
        source = stage / name
        if source.exists():
            shutil.copytree(source, target)
    for name in GENERATED_FILES:
        target = PACKAGE / name
        source = stage / name
        if target.exists():
            target.unlink()
        if source.is_file():
            shutil.copyfile(source, target)


def verify_deployment_bundle(*, required: bool = False) -> None:
    """Reject stale numerical evidence before assembling the code archive."""
    evidence_paths = (
        STUDY / "results/efficiency.json",
        STUDY / "results/adaptation_cost.json",
        STUDY / "results/pareto.json",
    )
    if required and not all(path.is_file() for path in evidence_paths):
        missing = [path.name for path in evidence_paths if not path.is_file()]
        raise FileNotFoundError("Required analysis evidence is missing: " + ", ".join(missing))
    if not required and not all(path.is_file() for path in evidence_paths):
        return
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    from efficiency import verify_result as verify_efficiency
    from adaptation_cost import verify_result as verify_adaptation
    from pareto import verify_result as verify_pareto
    for name, verifier in (("efficiency", verify_efficiency),
                           ("adaptation", verify_adaptation), ("Pareto", verify_pareto)):
        valid, messages = verifier()
        if not valid:
            raise ValueError(name + " evidence is stale: " + "; ".join(messages))


def build(*, final: bool) -> int:
    if final:
        current = blockers()
        if current:
            print("FINAL PACKAGE BLOCKED:")
            for item in current:
                print(f"- {item}")
            return 2
    else:
        current = blockers()
    verify_deployment_bundle(required=final)

    with tempfile.TemporaryDirectory(prefix="hdc-package-") as temp_dir:
        stage = Path(temp_dir) / "code_to_submit_github"
        stage.mkdir()
        doc_missing = _copy_package_docs(stage, final=final)
        missing = doc_missing + _copy_inputs(stage, final=final)
        _initialize_runtime_outputs(stage)
        if final and missing:
            raise FileNotFoundError("Missing package inputs: " + ", ".join(missing))
        if (stage / "hdc_study/provenance/data_manifest.json").is_file():
            _write_download_manifest(stage)
        if final:
            evidence_result_dir = stage / "evidence/hdc_study/results"
            _load_summary(evidence_result_dir / "summary.json")
            summary = _read_json(evidence_result_dir / "summary.json")
            protocol_path = stage / "evidence/hdc_study/provenance/protocol.json"
            if _sha256(protocol_path) != summary["protocol_sha256"]:
                raise ValueError("summary protocol_sha256 does not match archived provenance/protocol.json")
            active_protocol = stage / "hdc_study/provenance/protocol.json"
            if _sha256(active_protocol) != _sha256(protocol_path):
                raise ValueError("runtime and archived frozen protocols differ")
            for name in (
                    "data_manifest.json", "capacity_controls_protocol.json",
                    "matched_controls.frozen.py",
                    "capacity_controls_implementation_correction.json"):
                active_copy = stage / "hdc_study/provenance" / name
                evidence_copy = stage / "evidence/hdc_study/provenance" / name
                if _sha256(active_copy) != _sha256(evidence_copy):
                    raise ValueError(f"runtime and archived provenance differ: {name}")
            result_dir = evidence_result_dir
            packaged_run_hashes = {
                path.name: _sha256(path) for path in sorted((result_dir / "runs").glob("*.json"))
            }
            if summary.get("sources") != packaged_run_hashes:
                raise ValueError("summary run-source hashes do not match archived results/runs/*.json")
            packaged_diagnostic_hashes = {
                path.name: _sha256(path)
                for path in sorted((result_dir / "diagnostics").glob("outer*.json"))
            }
            if summary.get("diagnostic_sources") != packaged_diagnostic_hashes:
                raise ValueError("summary diagnostic hashes do not match archived outer diagnostics")
            if summary.get("prediction_audit_sha256") != _sha256(
                    result_dir / "prediction_audit.json"):
                raise ValueError("summary prediction-audit hash does not match the archived audit")
            if summary.get("selection") != _read_json(result_dir / "selection.json"):
                raise ValueError("summary selection does not match archived selection.json")
            matched_summary = summary.get("matched_capacity_controls")
            matched_sources = (matched_summary.get("source_hashes")
                               if isinstance(matched_summary, dict) else None)
            if not isinstance(matched_sources, dict):
                raise ValueError("summary is missing matched-control source hashes")
            packaged_control_hashes = {
                path.name: _sha256(path)
                for path in sorted((result_dir / "matched_controls").glob("S[0-9]_*_s*.json"))
            }
            if matched_sources.get("results") != packaged_control_hashes:
                raise ValueError("summary matched-control hashes do not match archived comparison records")
            packaged_control_inputs = {
                "capacity_protocol": stage / "hdc_study/provenance/capacity_controls_protocol.json",
                "runner": stage / "hdc_study/matched_controls.py",
                "status": result_dir / "matched_controls/status.json",
                "summary": result_dir / "matched_controls/summary.json",
            }
            for key, path in packaged_control_inputs.items():
                if matched_sources.get(key) != _sha256(path):
                    raise ValueError(f"summary matched-control {key} hash differs from packaged source")
            if list((stage / "hdc_study/results/runs").glob("*.json")):
                raise ValueError("published fit records must not occupy the writable runtime results/runs directory")
            validate_transfer_artifacts(
                stage / "deployment/transfer_hdc_s101.json",
                stage / FINAL_CHECKPOINT,
                stage / "hdc_study",
            )
        _write_evaluator_provenance(stage)
        if final:
            packaged_provenance = _read_json(stage / "falcon_evaluator_provenance.json")
            release_metadata = _read_json(stage / "release_metadata.json")
            if packaged_provenance.get("upstream_commit") != release_metadata.get("falcon_evaluator_commit"):
                raise ValueError("release evaluator commit does not match packaged provenance")
            for source in packaged_provenance.get("files", []):
                path = stage / "falcon-challenge" / source["path"]
                if not path.is_file() or _sha256(path) != source.get("vendored_sha256"):
                    raise ValueError(f"vendored evaluator provenance mismatch: {source.get('path')}")
        _write_status(stage, missing, current, final=final)
        # Validate generated metadata too: it can include diagnostic messages
        # that were not present when the copied release inputs were scanned.
        _validate_contents(stage)
        _write_content_list(stage)
        _write_manifest(stage)
        _replace_generated_folder(stage)
        if not final:
            # Re-evaluate now that generated provenance and the manifest-bound
            # download instructions exist; the prep ZIP should show blockers
            # that actually remain after this build.
            current = blockers()
            _write_status(stage, missing, current, final=False)
            _validate_contents(stage)
            _write_content_list(stage)
            _write_manifest(stage)
            _replace_generated_folder(stage)
        zip_name = "code_to_submit_github.zip" if final else "code_to_submit_github-preparatory.zip"
        zip_path = DIST / zip_name
        _write_zip(stage, zip_path)
    print(f"Built {'final' if final else 'preparatory'} H1-only package: {PACKAGE}")
    print(f"Deterministic ZIP: {zip_path}")
    if not final:
        print(f"Release blockers: {len(current)}; see {PACKAGE / 'PREPARATION_STATUS.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true",
                      help="build a clearly marked preparatory package from present allowlisted files")
    mode.add_argument("--final", action="store_true",
                      help="build a final archive only when every release gate passes")
    args = parser.parse_args()
    return build(final=args.final)


if __name__ == "__main__":
    raise SystemExit(main())

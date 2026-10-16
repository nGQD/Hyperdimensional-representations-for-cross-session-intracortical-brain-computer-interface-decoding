"""Verify the frozen control design and its documented implementation repair."""
import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_control_implementation(study_root):
    root = Path(study_root)
    protocol_path = root / "provenance/capacity_controls_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source = root / "matched_controls.py"
    current = _sha(source)
    if current == protocol["module_sha256"]:
        return current
    record_path = root / "provenance/capacity_controls_implementation_correction.json"
    record = json.loads(record_path.read_text())
    original = root / "provenance/matched_controls.frozen.py"
    if (record.get("schema") != 1 or record.get("design_changed") is not False
            or record.get("capacity_protocol_sha256") != _sha(protocol_path)
            or record.get("primary_protocol_sha256") != _sha(root / "provenance/protocol.json")
            or record.get("original_module_sha256") != protocol["module_sha256"]
            or _sha(original) != protocol["module_sha256"]
            or record.get("corrected_module_sha256") != current
            or record.get("verifier_sha256") != _sha(Path(__file__))):
        raise RuntimeError("Capacity-control implementation correction provenance mismatch")
    reconstructed = original.read_text()
    changes = record.get("replacements")
    if not isinstance(changes, list) or not changes:
        raise RuntimeError("Capacity-control correction has no exact source changes")
    for change in changes:
        old, new, count = change["old"], change["new"], change["count"]
        if not old or type(count) is not int or count < 1 or reconstructed.count(old) != count:
            raise RuntimeError("Capacity-control correction does not match the original source")
        reconstructed = reconstructed.replace(old, new)
    if reconstructed != source.read_text():
        raise RuntimeError("Capacity-control source differs from the documented correction")
    return current

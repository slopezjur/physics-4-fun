"""Select a completed export for Godot F5/F6 without copying or overwriting policies."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .baseline import sha256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mimic_viewer_selection_v1"


def selection_path(task, root=ROOT):
    if task not in ("stand", "ball"):
        raise ValueError(f"Unsupported viewer task: {task}")
    return root / "logs/mimickit-viewer" / f"{task}.json"


def read_selection(task, root=ROOT):
    path = selection_path(task, root)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != SCHEMA or state.get("task") != task or not state.get("bundle"):
        raise ValueError(f"Invalid viewer selection: {path}")
    return state


def _bundle_path(value, root):
    return root / value[6:] if value.startswith("res://") else Path(value)


def _validate_bundle(bundle):
    contract = json.loads((bundle / "contract.json").read_text(encoding="utf-8"))
    if contract.get("diagnostic_only"):
        raise ValueError("Diagnostic contact-contract experiments cannot replace the viewer policy")
    for name, key in (("stand.onnx", "actor_sha256"), ("stand_reference.npz", "reference_sha256"),
                      ("reference_frames.json", "reference_frames_sha256")):
        if sha256(bundle / name) != contract[key]:
            raise ValueError(f"Export hash mismatch: {name}")
    manifest = bundle / "experiment.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["task"] if manifest.exists() else "stand"


def _write(state, root):
    path = selection_path(state["task"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Godot sees either the old complete selection or the new one, never partial JSON.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(state, file, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return state


def select_run(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    error = report["onnx_max_action_error"]
    if not math.isfinite(error) or not 0 <= error <= 1e-5:
        raise ValueError("Run has no valid ONNX export check")
    bundle = run / "export"
    task = _validate_bundle(bundle)
    if task != report["experiment"]["task"]:
        raise ValueError("Run/export task mismatch")
    checkpoint = report["export_source"]
    if checkpoint != "best.pt" or sha256(run / checkpoint) != report["selected_checkpoint"]["checkpoint_sha256"]:
        raise ValueError("Run's selected checkpoint does not match its report")
    value = "res://" + bundle.relative_to(root).as_posix() if bundle.is_relative_to(root) else bundle.as_posix()
    previous = read_selection(task, root)
    if previous is not None and previous["bundle"] == value:
        return previous
    return _write({"schema": SCHEMA, "task": task, "bundle": value,
                   "previous": previous["bundle"] if previous else None}, root)


def rollback(task, root=ROOT):
    state = read_selection(task, root)
    if state is None or not state.get("previous"):
        raise ValueError(f"No previous {task} viewer selection")
    if _validate_bundle(_bundle_path(state["previous"], root)) != task:
        raise ValueError("Previous export has a different task")
    return _write({**state, "bundle": state["previous"], "previous": state["bundle"]}, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--run", type=Path)
    operation.add_argument("--rollback", choices=("stand", "ball"))
    args = parser.parse_args()
    print(json.dumps(select_run(args.run) if args.run else rollback(args.rollback), indent=2))

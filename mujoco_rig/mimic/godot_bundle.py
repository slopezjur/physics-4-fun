"""Create hash-bound JSON reference frames for the isolated Godot Stand adapter."""
import argparse
import json
from pathlib import Path
import numpy as np

from .baseline import sha256


def prepare(bundle: Path):
    contract_path = bundle / "contract.json"
    contract = json.loads(contract_path.read_text())
    reference_path = bundle / "stand_reference.npz"
    if sha256(reference_path) != contract["reference_sha256"]:
        raise ValueError("Reference does not match exported actor contract")
    with np.load(reference_path, allow_pickle=False) as reference:
        frames = {"dt": float(reference["dt"]), "reference_sha256": contract["reference_sha256"],
                  "qpos": reference["qpos"].astype(np.float32).tolist(),
                  "qvel": reference["qvel"].astype(np.float32).tolist()}
    path = bundle / "reference_frames.json"
    path.write_text(json.dumps(frames), encoding="utf-8")
    contract.update(reference_frames_sha256=sha256(path), actor_sha256=sha256(bundle / "stand.onnx"),
                    deployment_status="Experimental MimicStandReplay adapter; not promoted to production")
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    prepare(parser.parse_args().bundle)

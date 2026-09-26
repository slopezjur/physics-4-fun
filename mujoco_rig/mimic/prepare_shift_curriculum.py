"""Prepare slower prefixes of admitted weight shifts with unchanged physical gates."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .baseline import sha256
from .force_retarget import reference_velocities
from .reference import StandReference
from .retarget_steps import validate_motion
from .rig import Rig
from .transfer_calibration import interpolate_poses
from .weight_shift import SCHEMA, PHYSICAL_CHECKS, validate_reference


def prepare(source, out, prefix_seconds):
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = StandReference(source, rig, "cpu")
    metadata = json.loads(source.with_suffix(".json").read_text())
    validate_reference(metadata)
    if not np.isclose(reference.duration, 16.) or not 0 < prefix_seconds < 4:
        raise ValueError("Require the original 16-second shift cycle and a prefix below four seconds")
    frames = round(prefix_seconds / reference.dt)
    if not np.isclose(frames * reference.dt, prefix_seconds) or frames < 2:
        raise ValueError("Prefix must span at least two source intervals and end on a source frame")
    samples = np.linspace(0., 4., round(4. / reference.dt) + 1)
    pieces = []
    for start in (0, round(8. / reference.dt)):
        poses = reference.qpos[start:start + frames + 1].numpy().astype(np.float64)
        segment = interpolate_poses(poses, np.linspace(0., 4., frames + 1), samples)
        loop = np.concatenate((segment, segment[-2::-1]))
        pieces.append(loop if not start else loop[1:])
    q = np.concatenate(pieces)
    v = reference_velocities(rig.model, q, reference.dt)
    with np.load(source, allow_pickle=False) as data:
        targets = np.tile(data["foot_targets"][0], (len(q), 1, 1))
    contact, soles = np.ones((len(q), 2), bool), np.ones((len(q), 4), bool)
    checks, metrics = validate_motion(rig, q, v, targets, contact, reference.dt, soles)
    excursion = float(np.linalg.norm(np.ptp(q[:, :2], axis=0)))
    admission = dict(both_feet_planted=max(metrics["foot_lift_range_m"]) < .005,
                     horizontal_excursion=excursion >= .02)
    passed = all(checks[k] for k in PHYSICAL_CHECKS) and all(admission.values())
    out.mkdir(parents=True, exist_ok=False)
    path = out / "weight_shift_reference.npz"
    np.savez_compressed(path, qpos=q, qvel=v, dt=reference.dt, foot_targets=targets,
                        contact=contact, sole_contact=soles)
    report = dict(schema=SCHEMA, passed=passed, reference_sha256=sha256(path), model_sha256=sha256(rig.path),
                  source=metadata["source"], parent_reference_sha256=sha256(source), checks=checks, metrics=metrics,
                  weight_shift_admission=admission, curriculum=dict(prefix_seconds=prefix_seconds,
                  shift_seconds=4., horizontal_excursion_m=excursion),
                  changes="Retimed prefixes of both admitted double-support paths; clamped cubic endpoints and mirrored returns. All physical gates recomputed without relaxed limits.")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(source.parent / "ATTRIBUTION.md", out / "ATTRIBUTION.md")
    with (out / "ATTRIBUTION.md").open("a", encoding="utf-8") as file:
        file.write(f"\nFurther adaptation: first {prefix_seconds} seconds of each shift retimed to four seconds with stopped endpoints.\n")
    print(json.dumps(dict(passed=passed, checks=checks, excursion_m=excursion)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix-seconds", type=float, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if prepare(args.source, args.out, args.prefix_seconds)["passed"] else 1)

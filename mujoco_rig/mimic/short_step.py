"""Retimed, physically rechecked single-step candidates; never auto-admit for training."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from .baseline import sha256
from .force_retarget import reference_velocities
from .placed_step import single_step_checks
from .retarget_steps import validate_motion
from .rig import Rig
from .track_transfer import CalibrationReference
from .transfer_calibration import interpolate_poses


SCHEMA = "mimic_short_step_candidate_v1"
SOURCE_START_SECONDS = 3.
SOURCE_END_SECONDS = 18.
PRE_HOLD_SECONDS = .25
POST_HOLD_SECONDS = 2.
DT = .05
SPARSE_KNOT_SECONDS = (3., 7., 9., 11., 14., 15., 18.)
SOURCE_SWING_START_SECONDS = 7.
SOURCE_LANDING_SECONDS = 15.


def source_times(motion_seconds):
    if (not np.isfinite(motion_seconds) or motion_seconds < .5
            or not np.isclose(motion_seconds / DT, round(motion_seconds / DT))):
        raise ValueError("Motion duration must be at least 0.5 s and align to 50 ms frames")
    count = round((PRE_HOLD_SECONDS + motion_seconds + POST_HOLD_SECONDS) / DT) + 1
    wall = np.arange(count) * DT
    progress = np.clip((wall - PRE_HOLD_SECONDS) / motion_seconds, 0., 1.)
    return wall, SOURCE_START_SECONDS + progress * (SOURCE_END_SECONDS - SOURCE_START_SECONDS)


def phase_seconds(motion_seconds):
    scale = motion_seconds / (SOURCE_END_SECONDS - SOURCE_START_SECONDS)
    return dict(swing_start=PRE_HOLD_SECONDS +
                (SOURCE_SWING_START_SECONDS - SOURCE_START_SECONDS) * scale,
                landing=PRE_HOLD_SECONDS +
                (SOURCE_LANDING_SECONDS - SOURCE_START_SECONDS) * scale)


def retime(arrays, motion_seconds, *, sparse_knots=False):
    wall, mapped = source_times(motion_seconds)
    source_dt = float(arrays["dt"])
    first, last = (round(seconds / source_dt) for seconds in
                   (SOURCE_START_SECONDS, SOURCE_END_SECONDS))
    if sparse_knots:
        indices = np.rint(np.asarray(SPARSE_KNOT_SECONDS) / source_dt).astype(int)
    else:
        indices = np.arange(first, last + 1)
    source = indices * source_dt
    poses = np.asarray(arrays["qpos"][indices], dtype=np.float64)
    qpos = interpolate_poses(poses, source, mapped)
    if sparse_knots:
        targets = CubicSpline(source, arrays["foot_targets"][indices],
                              bc_type="clamped")(mapped)
    else:
        targets = np.empty((len(wall), 2, 3))
        for foot in range(2):
            for axis in range(3):
                targets[:, foot, axis] = np.interp(
                    mapped, source, arrays["foot_targets"][indices, foot, axis])
    index = np.rint(mapped / source_dt).astype(int).clip(first, last)
    contact = arrays["contact"][index].copy()
    soles = arrays["sole_contact"][index].copy()
    return dict(qpos=qpos, foot_targets=targets, contact=contact,
                sole_contact=soles, dt=DT), wall


def prepare(source_dir, out, motion_seconds, *, sparse_knots=False):
    if out.exists():
        raise FileExistsError(out)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    parent = CalibrationReference(source_dir, rig)
    if not parent.placed_step or parent.requested_step_m <= 0:
        raise ValueError("Require the physically checked forward placed-step diagnostic")
    with np.load(parent.path, allow_pickle=False) as arrays:
        candidate, wall = retime(arrays, motion_seconds, sparse_knots=sparse_knots)
    qpos = candidate["qpos"]
    qvel = reference_velocities(rig.model, qpos, DT)
    candidate["qvel"] = qvel
    checks, metrics = validate_motion(rig, qpos, qvel, candidate["foot_targets"],
                                      candidate["contact"], DT, candidate["sole_contact"])
    data = mujoco.MjData(rig.model)
    foot = rig.model.body("Foot_L").id
    data.qpos[:] = qpos[0]
    mujoco.mj_kinematics(rig.model, data)
    first_x = float(data.xpos[foot, 0])
    data.qpos[:] = qpos[-1]
    mujoco.mj_kinematics(rig.model, data)
    displacement = float(data.xpos[foot, 0] - first_x)
    gates = single_step_checks(checks, metrics, displacement,
                               parent.requested_step_m)
    passed = all(gates.values())
    out.mkdir(parents=True, exist_ok=False)
    path = out / "short_step_reference.npz"
    np.savez_compressed(path, **candidate)
    parent_report = json.loads((source_dir / "report.json").read_text())
    report = dict(schema=SCHEMA, passed=passed, admitted_for_training=False,
                  model_sha256=sha256(rig.path), reference_sha256=sha256(path),
                  parent_reference_sha256=parent.sha256,
                  parent_report_sha256=parent.report_sha256,
                  source=parent_report["source"],
                  motion_seconds=motion_seconds, pre_hold_seconds=PRE_HOLD_SECONDS,
                  post_hold_seconds=POST_HOLD_SECONDS, total_seconds=float(wall[-1]),
                  phase_seconds=phase_seconds(motion_seconds),
                  source_interval_seconds=[SOURCE_START_SECONDS, SOURCE_END_SECONDS],
                  interpolation="sparse_clamped_cubic" if sparse_knots else "full_source_cubic",
                  source_knot_seconds=SPARSE_KNOT_SECONDS if sparse_knots else None,
                  requested_step_m=parent.requested_step_m,
                  measured_step_m=displacement, checks=checks, metrics=metrics,
                  step_checks=gates,
                  scope="Retimed single-step diagnostic with recomputed velocities and physical gates; native tracking and ball recovery not established")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    origin = parent_report["source"]
    (out / "ATTRIBUTION.md").write_text(
        f"{origin['credit']}\n\nSource: {origin['source']}\n"
        f"License: {origin['license']} ({origin['license_url']})\n\n"
        "Adaptation: the physically checked placed-step motion was retimed and "
        "revalidated as a diagnostic candidate.\n", encoding="utf-8")
    print(json.dumps(dict(motion_seconds=motion_seconds, passed=passed,
                          checks=checks, step_checks=gates)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-seconds", type=float, required=True)
    parser.add_argument("--sparse-knots", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.out, args.motion_seconds, sparse_knots=args.sparse_knots)

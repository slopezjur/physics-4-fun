"""Build a short recovery-step candidate from feasible support keyframes."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from .baseline import sha256
from .contact_projection import StanceProjection
from .force_retarget import reference_velocities
from .foundation_audit import fit_pose, settle_passive_joints
from .recovery_step_motion import (PHYSICAL_CHECKS, fit_swing_foot,
                                   reference_placement, support_schedule)
from .retarget_steps import validate_motion
from .rig import Rig
from .static_support import solve_support
from .transfer_calibration import interpolate_poses, stopped_interpolation


SCHEMA = "mimic_static_keyframe_recovery_step_v1"
DT = .05
TIMES = np.array((0., .5, 1.35, 1.55, 1.9, 2.15, 3., 5.5))
SWING_START = 1.35
TOUCHDOWN = 2.15
STRIDE_M = .15


def build(rig, interpolation="stopped"):
    if interpolation not in ("stopped", "cubic", "pchip_root"):
        raise ValueError("Interpolation must be stopped, cubic, or pchip_root")
    model, data = rig.model, mujoco.MjData(rig.model)
    data.qpos[:] = rig.rest
    mujoco.mj_forward(model, data)
    feet = [model.body(name).id for name in ("Foot_L", "Foot_R")]
    initial_feet = data.xpos[feet].copy()
    cases = (
        ("stand", (0., 0.), (0., 0.), ("Foot_L", "Foot_R", "Toe_L", "Toe_R")),
        ("transfer", (0., 0.), (0., .12), ("Foot_L", "Foot_R", "Toe_L", "Toe_R")),
        ("raise", (0., .06), (0., .12), ("Foot_R", "Toe_R")),
        ("swing", (STRIDE_M, .06), (.03, .12), ("Foot_R", "Toe_R")),
        ("touchdown", (STRIDE_M, 0.), (.05, .08), ("Foot_L", "Foot_R", "Toe_L", "Toe_R")),
        ("settle", (STRIDE_M, 0.), (.07, .02), ("Foot_L", "Foot_R", "Toe_L", "Toe_R")),
    )
    fitted, diagnostics = {}, {}
    for name, (forward, height), com, support_names in cases:
        offsets = np.zeros((2, 3))
        offsets[0] = (forward, 0., height)
        pose, fit = fit_pose(rig, offsets, np.asarray(com))
        pose, passive = settle_passive_joints(rig, pose)
        # The left foot is free in swing. Correct right sole geometry before
        # certifying its support instead of accepting a position-only IK pose.
        if name in ("raise", "swing"):
            targets = initial_feet.copy()
            targets[0] += offsets[0]
            pose = StanceProjection(rig, pose[None], targets[None],
                                    np.array([[False, True]])).project(pose[None])[0]
            pose = fit_swing_foot(rig, pose[None], targets[None],
                                  np.array([[False, True]]))[0]
        try:
            support = solve_support(rig, pose, passive_tolerance=.05,
                                    minimum_norm=True, support_names=support_names)
        except (ValueError, RuntimeError) as error:
            support = dict(feasible=False, solver_message=str(error))
        fitted[name] = pose
        diagnostics[name] = dict(fit=fit, passive=passive,
                                 support_feasible=bool(support["feasible"]),
                                 support_torque_fraction=support.get("max_torque_budget_fraction"),
                                 support_message=support.get("solver_message"))
    key_poses = np.array([fitted[name] for name in
                          ("stand", "stand", "transfer", "raise", "swing",
                           "touchdown", "settle", "settle")])
    target_initial = initial_feet.copy()
    target_raised = initial_feet.copy()
    target_raised[0, 2] += .06
    target_swing = target_raised.copy()
    target_swing[0, 0] += STRIDE_M
    target_landing = initial_feet.copy()
    target_landing[0, 0] += STRIDE_M
    key_targets = np.array([target_initial, target_initial, target_initial,
                            target_raised, target_swing, target_landing,
                            target_landing, target_landing])
    wall = np.arange(round(TIMES[-1] / DT) + 1) * DT
    poses = (interpolate_poses(key_poses, TIMES, wall) if interpolation == "cubic"
             else stopped_interpolation(key_poses, TIMES, wall, root_rotations=True))
    if interpolation == "pchip_root":
        poses[:, :3] = PchipInterpolator(TIMES, key_poses[:, :3], axis=0)(wall)
    targets = stopped_interpolation(key_targets, TIMES, wall)
    contact = np.ones((len(wall), 2), dtype=bool)
    contact[(wall > SWING_START) & (wall < TOUCHDOWN), 0] = False
    poses = StanceProjection(rig, poses, targets, contact).project(poses)
    poses = fit_swing_foot(rig, poses, targets, contact)
    poses = np.asarray([settle_passive_joints(rig, pose)[0] for pose in poses])
    soles = np.column_stack((contact, contact))
    motion = dict(qpos=poses, qvel=reference_velocities(model, poses, DT),
                  dt=DT, foot_targets=targets, contact=contact,
                  sole_contact=soles)
    return motion, diagnostics


def run(out, interpolation="stopped"):
    if out.exists():
        raise FileExistsError(out)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    motion, keyframes = build(rig, interpolation)
    checks, metrics = validate_motion(rig, motion["qpos"], motion["qvel"],
                                      motion["foot_targets"], motion["contact"],
                                      DT, motion["sole_contact"])
    _, _, _, supports = support_schedule(rig, motion)
    placement = reference_placement(rig, motion)
    physical_passed = (all(checks[name] for name in PHYSICAL_CHECKS)
                       and metrics["foot_lift_range_m"][0] >= .03
                       and abs(placement - STRIDE_M) < .02
                       and all(item["feasible"] for item in supports))
    out.mkdir(parents=True)
    path = out / "candidate.npz"
    np.savez_compressed(path, **motion)
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  candidate_sha256=sha256(path), times_seconds=TIMES.tolist(),
                  swing_start_seconds=SWING_START, touchdown_seconds=TOUCHDOWN,
                  requested_placement_m=STRIDE_M, measured_placement_m=placement,
                  keyframes=keyframes, interpolation=interpolation,
                  physical_checks=checks,
                  physical_metrics=metrics, support_schedule=supports,
                  physical_passed=physical_passed, passed=False,
                  admitted_for_training=False,
                  scope="Static-keyframe motion candidate; native tracking and landing not yet verified")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(physical_passed=physical_passed, checks=checks,
                          root_failures=metrics["root_wrench_consistency"]["infeasible_frames"],
                          support_failures=[item["seconds"] for item in supports if not item["feasible"]],
                          penetration_m=metrics["max_penetration_m"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interpolation", choices=("stopped", "cubic", "pchip_root"), default="stopped")
    args = parser.parse_args()
    run(args.out, args.interpolation)

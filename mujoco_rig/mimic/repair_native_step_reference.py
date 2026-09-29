"""Bounded arm and joint-limit repair of a captured native step reference."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .baseline import sha256
from .force_retarget import reference_velocities
from .retarget_steps import validate_motion
from .rig import Rig


SCHEMA = "mimic_native_step_reference_repair_v1"
ARM_JOINTS = ("UpperArm_R_rx", "UpperArm_R_ry", "UpperArm_R_rz", "Forearm_R_rx")
DEFAULT_WINDOW_SECONDS = (2.25, 2.85)
CLEARANCE_M = .005


def arm_clearance(model, data, poses):
    forearm = model.geom("g_Forearm_R").id
    thigh = model.geom("g_Thigh_R").id
    gaps = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        gaps.append(float(mujoco.mj_geomDistance(model, data, forearm, thigh,
                                                 .1, None)))
    return np.asarray(gaps)


def repair(rig, motion, window_seconds=DEFAULT_WINDOW_SECONDS):
    model = rig.model
    poses = motion["qpos"].copy()
    dt = float(motion["dt"])
    times = np.arange(len(poses)) * dt
    initial = poses.copy()
    limits = model.jnt_range[1:]
    poses[:, 7:] = np.clip(poses[:, 7:], limits[:, 0] + 1e-5,
                          limits[:, 1] - 1e-5)
    joints = [model.joint(name).id for name in ARM_JOINTS]
    indices = model.jnt_qposadr[joints]
    joint_limits = model.jnt_range[joints]
    phase = np.clip((times - window_seconds[0]) /
                    (window_seconds[1] - window_seconds[0]), 0., 1.)
    envelope = np.sin(np.pi * phase) ** 2
    frames = np.flatnonzero(envelope > .001)
    data = mujoco.MjData(model)
    before_gap = arm_clearance(model, data, poses)

    def candidate(delta):
        result = poses.copy()
        result[:, indices] = np.clip(
            result[:, indices] + envelope[:, None] * delta,
            joint_limits[:, 0] + 1e-5, joint_limits[:, 1] - 1e-5)
        return result

    def residual(delta):
        gaps = arm_clearance(model, data, candidate(delta)[frames])
        return np.r_[1000. * np.maximum(0., CLEARANCE_M - gaps),
                     .2 * delta]

    fit = least_squares(residual, np.zeros(len(indices)),
                        bounds=(-.35, .35), max_nfev=60)
    poses = candidate(fit.x)
    after_gap = arm_clearance(model, data, poses)
    result = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in motion.items()}
    result["qpos"] = poses
    result["qvel"] = reference_velocities(model, poses, dt)
    details = dict(arm_joint_names=list(ARM_JOINTS), arm_delta_rad=fit.x.tolist(),
                   arm_window_seconds=list(window_seconds),
                   optimizer_converged=bool(fit.success),
                   arm_clearance_before_m=float(before_gap.min()),
                   arm_clearance_after_m=float(after_gap.min()),
                   arm_penetrating_frames=np.flatnonzero(after_gap < 0.).tolist(),
                   max_joint_pose_change_rad=float(np.abs(poses[:, 7:] -
                                                       initial[:, 7:]).max()))
    return result, details


def run(args):
    if (args.out.exists() or not 0. <= args.arm_start < 2.55 < args.arm_end <= 4.):
        raise ValueError("Require a new output and an arm window around touchdown")
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    report_path = args.audit / "report.json"
    source_path = args.audit / "native_rollout.npz"
    audit = json.loads(report_path.read_text(encoding="utf-8"))
    if (audit["schema"] != "mimic_native_step_dynamics_audit_v2"
            or audit["model_sha256"] != sha256(rig.path)
            or not audit["native_passed"]):
        raise ValueError("Require a matching successful native step audit")
    with np.load(source_path, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    if "support_contact" not in motion:
        raise ValueError("Require measured independent support contacts")

    def assess(candidate):
        return validate_motion(
            rig, candidate["qpos"], candidate["qvel"],
            candidate["foot_targets"], candidate["contact"],
            float(candidate["dt"]), support_contact=candidate["support_contact"])

    before_checks, before_metrics = assess(motion)
    candidate, repair_details = repair(rig, motion,
                                       (args.arm_start, args.arm_end))
    after_checks, after_metrics = assess(candidate)
    args.out.mkdir(parents=True)
    candidate_path = args.out / "candidate.npz"
    np.savez_compressed(candidate_path, **candidate)
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  source_audit_sha256=sha256(report_path),
                  source_motion_sha256=sha256(source_path),
                  candidate_sha256=sha256(candidate_path),
                  repair=repair_details, before_checks=before_checks,
                  after_checks=after_checks, before_metrics=before_metrics,
                  after_metrics=after_metrics,
                  passed=False, admitted_for_training=False,
                  scope="Bounded captured-motion repair; no native controller replay, policy training or scene selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(repair=repair_details, before_checks=before_checks,
                          after_checks=after_checks)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm-start", type=float, default=DEFAULT_WINDOW_SECONDS[0])
    parser.add_argument("--arm-end", type=float, default=DEFAULT_WINDOW_SECONDS[1])
    run(parser.parse_args())

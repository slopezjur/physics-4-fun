"""Diagnostic single placed step assembled from existing calibrated support poses."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .baseline import sha256
from .contact_projection import StanceProjection
from .force_retarget import reference_velocities, refine_support
from .foundation_audit import fit_pose, settle_passive_joints
from .model_contract import collision_pairs
from .retarget_steps import validate_motion
from .rig import Rig
from .transfer_calibration import stopped_interpolation


def single_step_checks(checks, metrics, measured, requested, foot=0):
    if foot not in (0, 1):
        raise ValueError('Single-step checks require left or right foot')
    return dict(physical=all(value for key, value in checks.items() if key != "foot_lift"),
                swing_foot_lift=bool(metrics["foot_lift_range_m"][foot] >= .03),
                placement=bool(abs(measured - requested) < .02))


def clear_arm_interference(model, qpos):
    """Fit a bounded temporary arm motion around the swing clearance conflict."""
    names = ("UpperArm_L_rx", "UpperArm_L_ry", "UpperArm_L_rz", "Forearm_L_rx")
    joints = [model.joint(name).id for name in names]
    indices = model.jnt_qposadr[joints]
    pairs = [(a, b) for a, b in collision_pairs(model)
             if any(name in (model.geom(a).name, model.geom(b).name)
                    for name in ("g_UpperArm_L", "g_Forearm_L", "g_Hand_L"))]
    data = mujoco.MjData(model)
    times = np.arange(len(qpos)) * .05
    phase = np.clip((times - 7.) / 4., 0., 1.)
    envelope = np.sin(np.pi * phase) ** 2
    envelope[(times <= 7.) | (times >= 11.)] = 0.
    selected = np.flatnonzero(envelope > 0)[::3]

    def residual(values):
        result = []
        for frame in selected:
            data.qpos[:] = qpos[frame]
            data.qpos[indices] += envelope[frame] * values
            mujoco.mj_forward(model, data)
            result.extend(max(0., .002 - mujoco.mj_geomDistance(model, data, a, b, .1, None))
                          for a, b in pairs)
        result.extend(.01 * values)
        return np.asarray(result)

    fit = least_squares(residual, np.zeros(len(indices)), bounds=(-.5, .5), max_nfev=30)
    result = qpos.copy()
    result[:, indices] += envelope[:, None] * fit.x
    return result, dict(converged=bool(fit.success), evaluations=int(fit.nfev),
                        arm_joint_names=list(names), joint_delta_rad=fit.x.tolist(),
                        max_sampled_clearance_deficit_m=float(np.max(residual(fit.x)[:-len(indices)])))


def generate(calibration: Path, audit: Path, out: Path, direction: str = "forward"):
    if out.exists():
        raise FileExistsError(out)
    if direction not in ("forward", "backward"):
        raise ValueError("Direction must be forward or backward")
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    model = rig.model
    calibration_report = json.loads((calibration / "report.json").read_text())
    audit_report = json.loads((audit / "report.json").read_text())
    if (calibration_report.get("schema") != "mimic_transfer_calibration_v1"
            or calibration_report.get("passed") is not True
            or calibration_report.get("model_sha256") != sha256(rig.path)
            or calibration_report.get("knots_sha256") != sha256(calibration / "knots.npz")
            or audit_report.get("schema") != "mimic_foundation_audit_v1"
            or audit_report.get("model_sha256") != sha256(rig.path)
            or calibration_report.get("source", {}).get("license") != "CC-BY-4.0"):
        raise ValueError("Source provenance or rig mismatch")

    with np.load(calibration / "knots.npz", allow_pickle=False) as source:
        original, ground = source["qpos"], source["foot_targets"][0].copy()
    with np.load(audit / "poses.npz", allow_pickle=False) as source:
        # Audit order: center, left-support cases, right-support cases.
        swing = source["qpos"][7 if direction == "forward" else 8].copy()
    stride = .15 if direction == "forward" else -.15
    lifted = ground.copy()
    lifted[0] += [stride, 0., .06]
    swing_contact = np.tile([False, True], (3, 1))
    swing_poses, _, _, swing_fit = refine_support(
        rig, np.tile(swing, (3, 1)), np.tile(lifted, (3, 1, 1)),
        swing_contact, .05, max_evaluations=100, root_seed_drop_m=.02,
        source_soles=np.column_stack((swing_contact, swing_contact)),
        stationary=True, linear_solver="exact", self_clearance_m=.015)
    swing = swing_poses[0]
    mid_target = ground + .4 * (lifted - ground)
    mid_seed = stopped_interpolation(
        np.array([original[5], swing]), [0., 1.], [.5], root_rotations=True)[0]
    middle_poses, _, _, middle_fit = refine_support(
        rig, np.tile(mid_seed, (3, 1)), np.tile(mid_target, (3, 1, 1)),
        swing_contact, .05, max_evaluations=100, root_seed_drop_m=.02,
        source_soles=np.column_stack((swing_contact, swing_contact)),
        stationary=True, linear_solver="exact", self_clearance_m=.015)
    middle = middle_poses[0]
    landed = ground.copy()
    landed[0, 0] += stride
    offsets = np.zeros((2, 3))
    offsets[0, 0] = stride
    def static_pose(foot_offsets, com_offset, target, support):
        seed, fit = fit_pose(rig, foot_offsets, np.asarray(com_offset))
        seed, passive = settle_passive_joints(rig, seed)
        contacts = np.tile(support, (3, 1))
        fitted, _, _, support_fit = refine_support(
            rig, np.tile(seed, (3, 1)), np.tile(target, (3, 1, 1)), contacts, .05,
            max_evaluations=100, root_seed_drop_m=.02,
            source_soles=np.column_stack((contacts, contacts)),
            stationary=True, linear_solver="exact")
        return fitted[0], dict(ik=fit, passive=passive, support=support_fit)

    near_floor = landed.copy()
    near_floor[0, 2] += .01
    near_offsets = offsets.copy()
    near_offsets[0, 2] = .01
    prelanding, prelanding_fit = static_pose(
        near_offsets, [.03, .155], near_floor, [False, True])
    shifted_landing, shifted_fit = static_pose(
        offsets, [.03, .155], landed, [True, True])
    landing, landing_fit = static_pose(
        offsets, [stride * .25, .04], landed, [True, True])

    # Preserve the calibrated collision-aware unloading knots.
    source_knots = np.array([0., 1., 2., 2.5, 3., 4.])
    times = np.r_[0., source_knots + 3., 9., 11., 14., 15., 18., 22.]
    poses = np.array([original[0], *original[:6], middle, swing, prelanding,
                      shifted_landing, landing, landing])
    targets = np.array([ground, *([ground] * 6), mid_target, lifted,
                        near_floor, landed, landed, landed])
    samples = np.linspace(0., times[-1], round(times[-1] / .05) + 1)
    qpos = stopped_interpolation(poses, times, samples, root_rotations=True)
    foot_targets = stopped_interpolation(targets, times, samples)
    contact = np.ones((len(samples), 2), dtype=bool)
    contact[(samples > 7.) & (samples < 15.), 0] = False
    foot_targets[contact[:, 0] & (samples <= 7.), 0] = ground[0]
    foot_targets[contact[:, 0] & (samples >= 15.), 0] = landed[0]
    foot_targets[:, 1] = ground[1]
    qpos = StanceProjection(rig, qpos, foot_targets, contact).project(qpos)
    qpos, arm_fit = clear_arm_interference(model, qpos)
    dt = float(samples[1] - samples[0])
    qvel = reference_velocities(model, qpos, dt)
    sole_contact = np.column_stack((contact, contact))
    checks, metrics = validate_motion(rig, qpos, qvel, foot_targets, contact, dt, sole_contact)
    data = mujoco.MjData(model)
    body = model.body("Foot_L").id
    data.qpos[:] = qpos[0]
    mujoco.mj_kinematics(model, data)
    start = data.xpos[body].copy()
    data.qpos[:] = qpos[-1]
    mujoco.mj_kinematics(model, data)
    finish = data.xpos[body].copy()
    measured = float(finish[0] - start[0])
    gates = single_step_checks(checks, metrics, measured, stride)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "diagnostic_poses.npz"
    np.savez_compressed(path, qpos=qpos, qvel=qvel, dt=dt,
                        foot_targets=foot_targets, contact=contact, sole_contact=sole_contact)
    report = dict(schema="mimic_placed_step_diagnostic_v1", direction=direction,
                  model_sha256=sha256(rig.path), calibration_sha256=sha256(calibration / "knots.npz"),
                  audit_sha256=sha256(audit / "poses.npz"), diagnostic_sha256=sha256(path),
                  source=calibration_report["source"],
                  source_reference_sha256=calibration_report["source_reference_sha256"],
                  requested_step_m=stride,
                  measured_step_m=measured, arm_fit=arm_fit,
                  swing_fit=swing_fit, middle_fit=middle_fit,
                  prelanding_fit=prelanding_fit,
                  shifted_landing_fit=shifted_fit, landing_fit=landing_fit,
                  checks=checks, metrics=metrics, single_step_checks=gates,
                  placement_passed=gates["placement"], passed=all(gates.values()),
                  admitted_for_training=False,
                  scope="Procedural single-step diagnostic; native tracking and impact recovery untested. No policy export or viewer selection.")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], measured_step_m=measured,
                          checks=checks)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direction", choices=("forward", "backward"), default="forward")
    args = parser.parse_args()
    raise SystemExit(0 if generate(args.calibration, args.audit, args.out, args.direction)["passed"] else 1)

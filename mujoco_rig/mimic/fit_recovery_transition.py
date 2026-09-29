"""Fit a measured recovery-step rollout to fixed contacts and dynamic support."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .baseline import sha256
from .contact_projection import StanceProjection
from .force_retarget import reference_velocities, refine_support
from .recovery_step_motion import PHYSICAL_CHECKS, fit_swing_foot
from .retarget_steps import validate_motion
from .rig import Rig
from .window_retarget import context_interval


SCHEMA = "mimic_recovery_transition_fit_v3"
SWING_START = 1.35
TOUCHDOWN = 3.85


def level_swing_sole(rig, poses, targets, contact, dt, touchdown):
    """Fit a nearly level sole while preserving the swing-foot position."""
    model, data = rig.model, mujoco.MjData(rig.model)
    names = ("Thigh_L_rx", "Thigh_L_ry", "Thigh_L_rz",
             "Shin_L_rx", "Foot_L_rx", "Foot_L_rz")
    joints = np.asarray([model.joint(name).id for name in names])
    q_indices = model.jnt_qposadr[joints]
    limits = model.jnt_range[joints]
    foot = model.body("Foot_L").id
    landing = np.flatnonzero(contact[:, 0] & (np.arange(len(poses)) * dt >= touchdown))[0]

    def yaw(frame):
        data.qpos[:] = poses[frame]
        mujoco.mj_kinematics(model, data)
        matrix = data.xmat[foot].reshape(3, 3)
        return float(np.arctan2(matrix[1, 0], matrix[0, 0]))

    start_yaw = yaw(round(SWING_START / dt))
    end_yaw = yaw(landing)
    for frame in np.flatnonzero(~contact[:, 0]):
        pose = poses[frame]
        progress = np.clip((frame * dt - SWING_START) / (touchdown - SWING_START), 0., 1.)
        desired_rotation = Rotation.from_euler(
            "z", start_yaw + progress * (end_yaw - start_yaw)).as_matrix()

        def residual(angles):
            data.qpos[:] = pose
            data.qpos[q_indices] = angles
            mujoco.mj_kinematics(model, data)
            actual_rotation = data.xmat[foot].reshape(3, 3)
            return np.r_[50 * (targets[frame, 0] - data.xpos[foot]),
                         Rotation.from_matrix(desired_rotation @ actual_rotation.T).as_rotvec()]

        fitted = least_squares(residual, pose[q_indices],
                               bounds=(limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
                               max_nfev=80)
        pose[q_indices] = fitted.x
    return poses


def prepare_seed(rig, rollout, touchdown=TOUCHDOWN, root_swing_lift_m=0.,
                 level_swing=False, root_swing_offset_xy=(0., 0.)):
    """Replace intermittent scuffs with one supported swing and one landing."""
    poses = rollout["qpos"].copy()
    targets = rollout["foot_targets"].copy()
    dt = float(rollout["dt"])
    wall = np.arange(len(poses)) * dt
    if not SWING_START < touchdown < wall[-1]:
        raise ValueError("Touchdown must lie inside the recorded swing and hold")
    if not 0. <= root_swing_lift_m <= .05:
        raise ValueError("Pelvis lift must be between zero and 5 cm")
    offset_xy = np.asarray(root_swing_offset_xy)
    if offset_xy.shape != (2,) or not np.isfinite(offset_xy).all() or np.abs(offset_xy).max() > .06:
        raise ValueError("Pelvis XY offsets must be finite and at most 6 cm")
    if root_swing_lift_m or np.any(offset_xy):
        phase = np.clip((wall - (touchdown - 1.85)) / 2., 0., 1.)
        envelope = np.sin(np.pi * phase)**2
        poses[:, :2] += envelope[:, None] * offset_xy
        poses[:, 2] += root_swing_lift_m * envelope
    contact = np.ones((len(poses), 2), dtype=bool)
    contact[(wall > SWING_START) & (wall < touchdown), 0] = False
    targets[:, 1] = targets[0, 1]
    targets[wall <= SWING_START, 0] = targets[0, 0]
    targets[wall >= touchdown, 0] = targets[-1, 0]
    swing = ~contact[:, 0]
    clearance = np.minimum((wall[swing] - SWING_START) / .3,
                           (touchdown - wall[swing]) / .3).clip(0., 1.)
    targets[swing, 0, 2] = np.maximum(
        targets[swing, 0, 2], targets[0, 0, 2] + .03 * clearance)
    poses[:, 7:] = np.clip(poses[:, 7:], rig.model.jnt_range[1:, 0] + 1e-5,
                          rig.model.jnt_range[1:, 1] - 1e-5)
    poses = StanceProjection(rig, poses, targets, contact).project(poses)
    poses = fit_swing_foot(rig, poses, targets, contact)
    if level_swing:
        poses = level_swing_sole(rig, poses, targets, contact, dt, touchdown)
    soles = np.column_stack((contact, contact))
    return dict(qpos=poses, qvel=reference_velocities(rig.model, poses, dt),
                dt=dt, foot_targets=targets, contact=contact,
                sole_contact=soles)


def run(source, out, start, stop, max_evaluations, direct_forces=False,
        touchdown=TOUCHDOWN, root_swing_lift_m=0., level_swing=False,
        root_swing_offset_xy=(0., 0.), seed_only=False, seed_kind="native"):
    if out.exists():
        raise FileExistsError(out)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    report_path = source / "report.json"
    parent = json.loads(report_path.read_text(encoding="utf-8"))
    if seed_kind not in ("native", "search", "search_raw", "authored", "candidate"):
        raise ValueError("Seed kind must be native, search, search_raw, authored, or candidate")
    if seed_kind not in ("native", "search") and (root_swing_lift_m or level_swing or any(root_swing_offset_xy)):
        raise ValueError("Pelvis/sole seed edits apply only to native rollout projection")
    source_touchdown = (parent["phase_seconds"]["touchdown"] if seed_kind == "authored"
                        else parent["touchdown_seconds"] if seed_kind == "candidate" else None)
    if (source_touchdown is not None
            and not np.isclose(touchdown, source_touchdown, atol=1e-6)):
        raise ValueError("Touchdown time must match the authored contact schedule")
    input_name, input_hash_key = {
        "native": ("native_rollout.npz", "native_rollout_sha256"),
        "search": ("best_native_rollout.npz", "native_rollout_sha256"),
        "search_raw": ("best_native_rollout.npz", "native_rollout_sha256"),
        "authored": ("step_reference.npz", "reference_sha256"),
        "candidate": ("candidate.npz", "candidate_sha256"),
    }[seed_kind]
    input_path = source / input_name
    if (parent["model_sha256"] != sha256(rig.path)
            or parent[input_hash_key] != sha256(input_path)):
        raise ValueError("Source motion or rig hash mismatch")
    with np.load(input_path, allow_pickle=False) as arrays:
        motion = (prepare_seed(rig, arrays, touchdown, root_swing_lift_m,
                               level_swing, root_swing_offset_xy)
                  if seed_kind in ("native", "search") else
                  {key: arrays[key].copy() for key in arrays.files})
    poses, targets, contact = (motion[key] for key in
                              ("qpos", "foot_targets", "contact"))
    dt = float(motion["dt"])
    soles = motion["sole_contact"]

    def assess(candidate):
        velocity = reference_velocities(rig.model, candidate, dt)
        return validate_motion(rig, candidate, velocity, targets, contact, dt, soles)

    before_checks, before_metrics = assess(poses)
    left, right, fixed = context_interval(rig, len(poses), dt, start, stop)
    out.mkdir(parents=True)
    seed_path = out / "contact_seed.npz"
    np.savez_compressed(seed_path, **motion)
    if seed_only:
        result = dict(schema=SCHEMA, source_kind=seed_kind,
                      source_motion_sha256=sha256(input_path),
                      model_sha256=sha256(rig.path), contact_seed_sha256=sha256(seed_path),
                      window=[start, stop], context=[left, right],
                      touchdown_seconds=touchdown,
                      root_swing_lift_m=root_swing_lift_m,
                      root_swing_offset_xy=list(root_swing_offset_xy),
                      level_swing=level_swing, seed_only=True,
                      checks=before_checks, metrics=before_metrics,
                      passed=False, admitted_for_training=False,
                      scope="Audited contact seed only; no fitting, training, export or viewer selection")
        (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(checks=before_checks,
                              root_failures=before_metrics["root_wrench_consistency"]["infeasible_frames"],
                              penetration_m=before_metrics["max_penetration_m"])), flush=True)
        return result
    fitted, _, _, fit = refine_support(
        rig, poses[left:right], targets[left:right], contact[left:right], dt,
        max_evaluations=max_evaluations, root_seed_drop_m=0.,
        source_soles=soles[left:right], linear_solver="exact",
        fixed_frames=fixed, constrained=True, direct_forces=direct_forces)
    if not np.array_equal(fitted[fixed], poses[left:right][fixed]):
        raise RuntimeError("The fit modified fixed context")
    candidate = poses.copy()
    candidate[start:stop] = fitted[start - left:stop - left]
    after_checks, after_metrics = assess(candidate)
    physical_passed = (all(after_checks[name] for name in PHYSICAL_CHECKS)
                       and after_metrics["foot_lift_range_m"][0] >= .03)
    np.savez_compressed(out / "candidate.npz", qpos=candidate,
                        qvel=reference_velocities(rig.model, candidate, dt),
                        dt=dt, foot_targets=targets, contact=contact,
                        sole_contact=soles)
    result = dict(schema=SCHEMA, source_kind=seed_kind,
                  source_motion_sha256=sha256(input_path),
                  model_sha256=sha256(rig.path),
                  contact_seed_sha256=sha256(out / "contact_seed.npz"),
                  candidate_sha256=sha256(out / "candidate.npz"),
                  window=[start, stop], context=[left, right],
                  touchdown_seconds=touchdown,
                  root_swing_lift_m=root_swing_lift_m,
                  root_swing_offset_xy=list(root_swing_offset_xy),
                  level_swing=level_swing,
                  direct_forces=direct_forces,
                  fitted_constraints_satisfied=fit["constraints_satisfied"],
                  physical_passed=physical_passed,
                  passed=bool(fit["constraints_satisfied"] and physical_passed),
                  fit=fit, before_checks=before_checks,
                  before_metrics=before_metrics, after_checks=after_checks,
                  after_metrics=after_metrics, admitted_for_training=False,
                  scope="Diagnostic contact-constrained native rollout fit; no training, export or viewer selection")
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(constraints_satisfied=fit["constraints_satisfied"],
                          before_root_failures=len(before_metrics["root_wrench_consistency"]["infeasible_frames"]),
                          after_root_failures=len(after_metrics["root_wrench_consistency"]["infeasible_frames"]),
                          before_checks=before_checks, after_checks=after_checks,
                          fit_seconds=fit["seconds"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed-kind", choices=("native", "search", "search_raw", "authored", "candidate"), default="native")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", required=True, help="Editable frames START:STOP")
    parser.add_argument("--max-evaluations", type=int, default=8)
    parser.add_argument("--touchdown", type=float, default=TOUCHDOWN)
    parser.add_argument("--root-swing-lift-m", type=float, default=0.)
    parser.add_argument("--root-swing-offset-x-m", type=float, default=0.)
    parser.add_argument("--root-swing-offset-y-m", type=float, default=0.)
    parser.add_argument("--level-swing", action="store_true")
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--direct-forces", action="store_true")
    args = parser.parse_args()
    try:
        start, stop = map(int, args.window.split(":"))
        if args.max_evaluations < 1:
            raise ValueError()
    except ValueError:
        parser.error("Use a START:STOP window and a positive solver budget")
    run(args.source, args.out, start, stop, args.max_evaluations,
        args.direct_forces, args.touchdown, args.root_swing_lift_m,
        args.level_swing, (args.root_swing_offset_x_m, args.root_swing_offset_y_m),
        args.seed_only, args.seed_kind)

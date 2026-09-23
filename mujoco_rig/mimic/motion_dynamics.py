"""Necessary whole-body force check for grounded, unforced motion references."""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog

from .baseline import sha256
from .rig import Rig
from .target_control import DelayedTargetControl
from mujoco_rig.rl.env_config import ACTION_LATENCY_STEPS


def frame_dynamics(model, data, qpos, frame, dt):
    """Central tangent derivatives and required generalized force at one frame."""
    backward, forward, inertia = (np.zeros(model.nv) for _ in range(3))
    mujoco.mj_differentiatePos(model, backward, dt, qpos[frame], qpos[frame - 1])
    mujoco.mj_differentiatePos(model, forward, dt, qpos[frame], qpos[frame + 1])
    data.qpos[:] = qpos[frame]
    data.qvel[:] = (forward - backward) / 2
    mujoco.mj_forward(model, data)
    mujoco.mj_mulM(model, data, inertia, (forward + backward) / dt)
    return inertia + data.qfrc_bias - data.qfrc_passive


def support_columns(model, data, active_soles=None):
    """Optimistic near-floor footprint forces, including their joint moments."""
    feet = [model.geom("g_" + name).id for name in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
    if not all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in feet):
        raise ValueError("Support diagnosis expects the existing box soles")
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    columns = []
    for sole, geom in enumerate(feet):
        if active_soles is not None and not active_soles[sole]:
            continue
        corners = signs * model.geom_size[geom]
        corners = corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        if corners[:, 2].min() > .001:
            continue
        friction = max(model.geom_friction[geom, 0], model.geom_friction[0, 0])
        points = corners[np.argsort(corners[:, 2])[:4]]
        # Explicit v6 phases cannot borrow projected support from a raised corner.
        if active_soles is not None:
            points = points[points[:, 2] <= .001]
        for point in points:
            point[2] = 0
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, point, int(model.geom_bodyid[geom]))
            for x, y in itertools.product((-1, 1), repeat=2):
                columns.append(jacobian.T @ np.array([x * friction, y * friction, 1.]))
    return np.array(columns).T if columns else np.zeros((model.nv, 0))


def force_consistency(rig, qpos, dt):
    """Check aggregate Coulomb force capacity, not joint torque or moment feasibility.

    Interior second differences measure frame-averaged COM acceleration. Allowing
    every sole's maximum friction is deliberately optimistic: failing even this
    envelope rejects a reference, while passing does not prove trackability.
    """
    if len(qpos) < 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Force consistency requires three poses and a positive timestep")
    model, data = rig.model, mujoco.MjData(rig.model)
    centers = []
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        centers.append(data.subtree_com[1].copy())
    acceleration = np.diff(centers, n=2, axis=0) / dt**2
    support = acceleration - model.opt.gravity
    feet = [model.geom("g_" + name).id for name in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
    friction = float(model.geom_friction[[0, *feet], 0].max())
    normal = support[:, 2]
    horizontal = np.linalg.norm(support[:, :2], axis=-1)
    excess = horizontal - friction * np.maximum(normal, 0.)
    # A small numerical tolerance, expressed as acceleration (force per unit mass).
    failures = (normal < -1e-5) | (excess > 1e-5)
    return dict(passed=bool(not failures.any()),
                failing_frames=(np.flatnonzero(failures) + 1).tolist(),
                friction_coefficient=friction,
                max_horizontal_com_acceleration_m_s2=float(horizontal.max()),
                min_normal_force_per_mass_m_s2=float(normal.min()),
                max_friction_excess_per_mass_m_s2=float(excess.max()),
                max_root_acceleration_m_s2=np.abs(np.diff(qpos[:, :3], n=2, axis=0) / dt**2).max(axis=0).tolist())


def root_wrench_consistency(rig, qpos, dt, sole_contact=None):
    """Diagnostic only: approximate inverse dynamics under optimistic foot support.

    Central tangent differences estimate velocity/acceleration. Unlimited joint
    torque removes motor authority as a confounder. Each sole within 1 mm of the
    floor receives its entire projected footprint, and a square OUTSIDE the
    circular friction cone supplies an optimistic force envelope. This ignores
    actuator/contact compliance and is not a policy-learning feasibility verdict.
    """
    model, data = rig.model, mujoco.MjData(rig.model)
    if len(qpos) < 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Root-wrench diagnosis requires three poses and a positive timestep")
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    failures = []
    for frame in range(1, len(qpos) - 1):
        required = frame_dynamics(model, data, qpos, frame, dt)[:6] / weight
        columns = support_columns(model, data, None if sole_contact is None else sole_contact[frame])[:6]
        feasible = columns.shape[1] > 0 and linprog(
            np.zeros(columns.shape[1]), A_eq=columns, b_eq=required,
            bounds=(0, None), method="highs").success
        if not feasible:
            failures.append(frame)
    return dict(evaluated_frames=len(qpos) - 2, infeasible_frames=failures,
                all_frames_feasible=not failures, joint_torque_limits="unbounded",
                support=("explicit active sole corners within 1 mm of floor; outer-square friction cone" if sole_contact is not None
                         else "optimistic projected near-floor sole footprints and outer-square friction cone"),
                derivatives="central tangent differences at interior frames")


def delayed_pd_bounds(rig, qpos, qvel, dt, frame, window=DelayedTargetControl.residual_radians):
    """Normalized torque interval at exact reference tracking after command delay."""
    model = rig.model
    joints = model.actuator_trnid[rig.actuators, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    times = np.arange(len(qpos)) * dt
    issued = times[frame] - ACTION_LATENCY_STEPS * rig.decimation * model.opt.timestep
    if issued < 0:
        raise ValueError("Delayed PD bounds are undefined before the first command arrives")
    target_q = np.array([np.interp(issued, times, qpos[:, i]) for i in qi])
    target_v = np.array([np.interp(issued, times, qvel[:, i]) for i in vi])
    backward, forward = np.zeros(model.nv), np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, backward, dt, qpos[frame], qpos[frame - 1])
    mujoco.mj_differentiatePos(model, forward, dt, qpos[frame], qpos[frame + 1])
    current_v = (forward - backward) / 2
    targets = np.clip(target_q[:, None] + np.array([-window, window]),
                      model.jnt_range[joints, :1], model.jnt_range[joints, 1:])
    pd = DelayedTargetControl.gain * (targets - qpos[frame, qi, None])
    pd += (DelayedTargetControl.gain * DelayedTargetControl.damping_time *
           (target_v - current_v[vi]))[:, None]
    return np.clip(pd, -1, 1)


def action_window_consistency(rig, qpos, qvel, dt, windows=(.25, .5), sole_contact=None):
    """Necessary tracking conditions, separating root support, torque and PD targets.

    The actor is allowed any action at each frame. Passive joint equilibrium and
    contact compliance are omitted, so passing cannot establish trackability.
    Widened windows are diagnostic only; they never modify the controller.
    """
    if len(qpos) < 3 or dt <= 0 or not np.isfinite(dt) or any(w <= 0 for w in windows):
        raise ValueError("Require three frames, a positive timestep and positive windows")
    model, data = rig.model, mujoco.MjData(rig.model)
    joints = model.actuator_trnid[rig.actuators, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    controlled = np.r_[np.arange(6), vi]
    motors = np.zeros((model.nv, len(vi)))
    motors[vi, np.arange(len(vi))] = rig.action_scale
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    delay = ACTION_LATENCY_STEPS * rig.decimation * model.opt.timestep
    times = np.arange(len(qpos)) * dt
    failures = {"torque": [], **{str(window): [] for window in windows}}
    supported = []
    for frame in range(1, len(qpos) - 1):
        if times[frame] < delay:
            continue  # The actual controller starts with an empty command queue.
        required = frame_dynamics(model, data, qpos, frame, dt)
        columns = support_columns(model, data, None if sole_contact is None else sole_contact[frame]) * weight
        if columns.shape[1] == 0 or not linprog(np.zeros(columns.shape[1]),
                A_eq=columns[:6] / weight, b_eq=required[:6] / weight, bounds=(0, None), method="highs").success:
            continue
        supported.append(frame)
        matrix = np.column_stack((motors, columns))[controlled] / weight
        bounds = {"torque": [(-1., 1.)] * len(vi)}
        for window in windows:
            bounds[str(window)] = delayed_pd_bounds(rig, qpos, qvel, dt, frame, window).tolist()
        for name, limits in bounds.items():
            if not linprog(np.zeros(matrix.shape[1]), A_eq=matrix, b_eq=required[controlled] / weight,
                           bounds=limits + [(0, None)] * columns.shape[1], method="highs").success:
                failures[name].append(frame)
    return dict(root_supported_frames=supported, failing_frames=failures, command_delay_seconds=delay,
                residual_windows_radians=list(windows),
                scope="Optimistic inverse dynamics conditional on root support; exact delayed PD target intervals at reference tracking. Passive joint equilibrium, initial queue fill and between-frame dynamics are excluded.")


def audit(references, out):
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    rows = []
    for path in references:
        with np.load(path, allow_pickle=False) as source:
            poses, velocities, dt = source["qpos"], source["qvel"], float(source["dt"])
            soles = source["sole_contact"] if "sole_contact" in source else None
        rows.append(dict(reference=str(path), reference_sha256=sha256(path),
                         force=force_consistency(rig, poses, dt),
                         root_wrench=root_wrench_consistency(rig, poses, dt, soles),
                         action_window=action_window_consistency(rig, poses, velocities, dt, sole_contact=soles)))
    report = dict(schema="mimic_reference_dynamics_v1", model_sha256=sha256(rig.path), results=rows,
                  scope="Reference diagnostics, not policy evaluation. Root-wrench finite differences and optimistic support cannot prove trackability or that RL cannot adapt the reference.")
    with out.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    for row in rows:
        print(json.dumps(dict(reference=row["reference"], force_passed=row["force"]["passed"],
                              root_wrench_infeasible=len(row["root_wrench"]["infeasible_frames"]),
                              evaluated_frames=row["root_wrench"]["evaluated_frames"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.reference, args.out)

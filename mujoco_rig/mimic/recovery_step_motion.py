"""Build a short step reference and measure its native landing retention."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from .baseline import sha256
from .contact_projection import StanceProjection, bounded_ik_step
from .force_retarget import reference_velocities
from .foundation_audit import fit_pose, settle_passive_joints
from .native_engine import NativeDummyEngine
from .probe_single_support_entry import (CONTROLLERS, EntryReference,
                                         ScheduledStabilizer)
from .probe_single_support import support_body_names
from .retarget_steps import validate_motion
from .rig import Rig
from .runtime import activate
from .static_support import solve_support
from .track_transfer import CalibrationReference
from .transfer_calibration import interpolate_poses, stopped_interpolation


SCHEMA = "mimic_short_loaded_step_diagnostic_v2"
DT = .05
KNOTS = ((0., 7.), (.5, 7.), (1.35, 7.85), (1.65, 11.),
         (1.95, 14.), (2.15, 15.), (3., 18.), (5.5, 18.))
SLOW_KNOTS = ((0., 7.), (.5, 7.), (1.35, 7.85), (1.8, 11.),
              (2.25, 14.), (2.55, 15.), (3.65, 18.), (6.15, 18.))
RECOVERY_KNOTS = ((0., 7.), (.5, 7.), (1.35, 7.85), (1.8, 11.),
                  (2.25, 14.), (2.55, 15.), (4.5, 18.), (7., 18.))
DELAYED_KNOTS = ((0., 7.), (.5, 7.), (1.35, 7.85), (1.9, 11.),
                 (2.8, 14.), (3.85, 15.), (4.65, 18.), (6.65, 18.))
SETTLE_KNOTS = SLOW_KNOTS[:-1] + ((8.15, 18.),)
SWING_START = 1.35
LEFT_LOAD_MIN_N = 20.
RIGHT_LOAD_MIN_N = 20.
TOUCHDOWN_LAG_MAX_SECONDS = .25
PHYSICAL_CHECKS = ("finite", "joint_limits", "penetration", "stance_slide",
                   "ground_support", "stance_contact", "foot_tracking",
                   "joint_speed", "sole_contact", "force_consistency",
                   "root_wrench_consistency", "actuation_feasibility")


def fit_swing_foot(rig, poses, targets, contact):
    """Keep the free foot on its authored path after changing pelvis motion."""
    model, data = rig.model, mujoco.MjData(rig.model)
    joints = [model.joint(name).id for name in
              ("Thigh_L_rx", "Thigh_L_ry", "Thigh_L_rz",
               "Shin_L_rx", "Foot_L_rx", "Foot_L_rz")]
    qpos = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    limits = model.jnt_range[joints]
    foot = model.body("Foot_L").id
    jacobian = np.zeros((3, model.nv))
    for frame in np.flatnonzero(~contact[:, 0]):
        pose = poses[frame]
        for _ in range(12):
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            error = targets[frame, 0] - data.xpos[foot]
            if np.linalg.norm(error) < 1e-4:
                break
            mujoco.mj_jacBody(model, data, jacobian, None, foot)
            pose[qpos] += bounded_ik_step(jacobian[:, dofs], error,
                                         pose[qpos], limits)
    return poses


def clear_arm_collision(rig, poses, wall):
    """Move the left arm clear of the pelvis during the transfer-to-swing join."""
    model, data = rig.model, mujoco.MjData(rig.model)
    names = ("UpperArm_L_rx", "UpperArm_L_ry", "UpperArm_L_rz", "Forearm_L_rx")
    joints = [model.joint(name).id for name in names]
    indices = model.jnt_qposadr[joints]
    forearm, pelvis = (model.geom(name).id for name in ("g_Forearm_L", "g_Pelvis"))
    phase = np.clip((wall - 1.2) / .5, 0., 1.)
    envelope = np.sin(np.pi * phase)**2
    frames = np.flatnonzero(envelope > .01)

    def residual(delta):
        errors = []
        for frame in frames:
            data.qpos[:] = poses[frame]
            data.qpos[indices] += envelope[frame] * delta
            mujoco.mj_forward(model, data)
            gap = mujoco.mj_geomDistance(model, data, forearm, pelvis, .1, None)
            errors.append(max(0., .003 - gap))
        return np.r_[errors, .001 * delta]

    fit = least_squares(residual, np.zeros(len(indices)), bounds=(-.4, .4), max_nfev=30)
    result = poses.copy()
    result[:, indices] += envelope[:, None] * fit.x
    return result


def build_motion(rig, arrays, root_profile="source", stride_m=.15, knots=KNOTS,
                 forward_bias_m=0., landing_com_x=None, quiet_upper_swing=False,
                 smooth_root=False, smooth_motion=False,
                 landing_target_offset_m=0., lateral_touchdown_bias_m=0.,
                 braked_swing=False):
    source_dt = float(arrays["dt"])
    if not np.isclose(source_dt, DT):
        raise ValueError("Source sample period does not match the step builder")
    touchdown, recovery_end = knots[5][0], knots[6][0]
    wall = np.arange(round(knots[-1][0] / DT) + 1) * DT
    times = np.asarray([wall_time for wall_time, _ in knots])
    source_frames = np.rint(np.asarray([source_time for _, source_time in knots]) / DT).astype(int)
    if source_frames.max() >= len(arrays["qpos"]):
        raise ValueError("Source does not contain the landing and recovery poses")
    source_poses = arrays["qpos"][source_frames].copy()
    if landing_com_x is not None:
        offsets = np.zeros((2, 3))
        offsets[0, 0] = stride_m
        terminal, fit = fit_pose(rig, offsets, np.asarray((landing_com_x, .04)))
        if not fit["optimizer_converged"] or fit["max_foot_error_m"] > .002:
            raise ValueError("Forward landing pose fit failed")
        source_poses[-2:] = terminal
    poses = (interpolate_poses(source_poses, times, wall) if smooth_motion else
             stopped_interpolation(source_poses, times, wall,
                                   root_rotations=True))
    if smooth_root:
        poses[:, :3] = PchipInterpolator(times, source_poses[:, :3], axis=0)(wall)
    if stride_m != .15:
        anchor = arrays["qpos"][round(7.85 / DT), 0]
        swing = wall > SWING_START
        poses[swing, 0] = anchor + (poses[swing, 0] - anchor) * stride_m / .15
    if root_profile == "anchored":
        root_times = np.asarray((0., SWING_START, touchdown, recovery_end, knots[-1][0]))
        entry = arrays["qpos"][round(7.85 / DT), :3]
        finish = arrays["qpos"][round(18. / DT), :3]
        landing = entry.copy()
        landing[:2] += (.02, 0.)
        settled = finish.copy()
        settled[0] -= .015
        roots = np.asarray((arrays["qpos"][source_frames[0], :3], entry,
                            landing, settled, finish))
        poses[:, :3] = stopped_interpolation(roots, root_times, wall)
    elif root_profile != "source":
        raise ValueError("Unknown root profile")
    if quiet_upper_swing:
        model = rig.model
        upper = np.asarray([model.jnt_qposadr[joint] for joint in range(1, model.njnt)
                            if not model.joint(joint).name.startswith(
                                ("Thigh_", "Shin_", "Foot_", "Toe_"))])
        entry = poses[round(SWING_START / DT)].copy()
        recovery = poses[round(recovery_end / DT)].copy()
        end = poses[-1].copy()
        mask = wall >= SWING_START
        upper_path = stopped_interpolation(
            np.asarray((entry, entry, recovery, end)),
            np.asarray((SWING_START, touchdown, recovery_end, knots[-1][0])),
            wall[mask], root_rotations=True)
        poses[np.ix_(mask, upper)] = upper_path[:, upper]
        poses[mask, 3:7] = upper_path[:, 3:7]
    if forward_bias_m:
        bias_times = (0., SWING_START, SWING_START + .45,
                      recovery_end, recovery_end + .6, knots[-1][0])
        bias = stopped_interpolation(
            np.asarray(((0.,), (0.,), (forward_bias_m,),
                        (forward_bias_m,), (0.,), (0.,))),
            bias_times, wall)
        poses[:, 0] += bias[:, 0]
    if lateral_touchdown_bias_m:
        bias_times = (0., SWING_START, touchdown - .6, touchdown,
                      touchdown + .75, touchdown + 1.25, knots[-1][0])
        if not np.all(np.diff(bias_times) > 0):
            raise ValueError("Touchdown timing cannot fit the lateral transfer")
        bias = stopped_interpolation(
            np.asarray(((0.,), (0.,), (lateral_touchdown_bias_m,),
                        (lateral_touchdown_bias_m,),
                        (lateral_touchdown_bias_m,), (0.,), (0.,))),
            bias_times, wall)
        poses[:, 1] += bias[:, 0]
    targets = stopped_interpolation(arrays["foot_targets"][source_frames], times, wall)
    if stride_m != .15:
        progress = np.clip((wall - SWING_START) / (touchdown - SWING_START), 0., 1.)
        targets[:, 0, 0] -= (.15 - stride_m) * progress**2 * (3. - 2. * progress)
    contact = np.ones((len(wall), 2), dtype=bool)
    contact[(wall > SWING_START) & (wall < touchdown), 0] = False
    targets[:, 1] = arrays["foot_targets"][source_frames[0], 1]
    targets[wall <= SWING_START, 0] = arrays["foot_targets"][source_frames[0], 0]
    targets[wall >= touchdown, 0] = arrays["foot_targets"][source_frames[-1], 0]
    targets[wall >= touchdown, 0, 0] -= .15 - stride_m
    if landing_target_offset_m:
        progress = np.clip((wall - SWING_START) / (touchdown - SWING_START), 0., 1.)
        targets[:, 0, 0] += landing_target_offset_m * progress**2 * (3. - 2. * progress)
    if braked_swing:
        if touchdown - SWING_START < .85:
            raise ValueError("Braked swing needs at least 0.85 seconds")
        initial_foot = targets[0, 0].copy()
        landing_foot = targets[-1, 0].copy()

        def ease(start, stop):
            phase = np.clip((wall - start) / (stop - start), 0., 1.)
            return phase**3 * (10. + phase * (-15. + 6. * phase))

        travel = ease(SWING_START, touchdown - .35)
        rise = ease(SWING_START, SWING_START + .45)
        descent = ease(touchdown - .35, touchdown)
        targets[:, 0] = initial_foot + travel[:, None] * (landing_foot - initial_foot)
        targets[:, 0, 2] += .06 * rise * (1. - descent)
    poses = StanceProjection(rig, poses, targets, contact).project(poses)
    if (root_profile == "anchored" or stride_m != .15 or forward_bias_m
            or quiet_upper_swing or smooth_root or smooth_motion
            or landing_target_offset_m or lateral_touchdown_bias_m or braked_swing):
        poses = fit_swing_foot(rig, poses, targets, contact)
    poses = clear_arm_collision(rig, poses, wall)
    poses = np.asarray([settle_passive_joints(rig, pose)[0] for pose in poses])
    velocity = reference_velocities(rig.model, poses, DT)
    soles = np.column_stack((contact, contact))
    return dict(qpos=poses, qvel=velocity, dt=DT, foot_targets=targets,
                contact=contact, sole_contact=soles)


def support_schedule(rig, motion):
    times, torques, centers, samples = [], [], [], []
    for frame in range(0, len(motion["qpos"]), round(.25 / DT)):
        try:
            support = solve_support(rig, motion["qpos"][frame],
                                    passive_tolerance=.05, minimum_norm=True,
                                    support_names=support_body_names(motion["contact"][frame]))
        except (ValueError, RuntimeError) as error:
            support = dict(feasible=False, solver_message=str(error))
        times.append(frame * DT)
        torques.append(support.get("normalized_motor_torque",
                                   torques[-1] if torques else np.zeros(len(rig.actuators))))
        centers.append(support.get("com_world",
                                   centers[-1] if centers else [0., 0., 0.]))
        samples.append(dict(seconds=frame * DT,
                            feasible=bool(support["feasible"]),
                            reason=support.get("solver_message"),
                            max_torque_fraction=(float(support["max_torque_budget_fraction"])
                                                 if support["feasible"] else None)))
    return np.asarray(times), np.asarray(torques), np.asarray(centers), samples


def reference_placement(rig, motion):
    data = mujoco.MjData(rig.model)
    foot = rig.model.body("Foot_L").id
    positions = []
    for pose in (motion["qpos"][0], motion["qpos"][-1]):
        data.qpos[:] = pose
        mujoco.mj_kinematics(rig.model, data)
        positions.append(data.xpos[foot].copy())
    return float(positions[1][0] - positions[0][0])


def loaded_landing_metrics(samples, touchdown, dt):
    """Require the first loaded touchdown to continue into two-foot support."""
    intervals = []
    start = last = None
    for seconds, loaded in samples:
        if loaded:
            start = seconds if start is None else start
            last = seconds
        elif start is not None:
            intervals.append((start, last - start + dt))
            start = last = None
    if start is not None:
        intervals.append((start, last - start + dt))
    first_start, first_duration = intervals[0] if intervals else (None, 0.)
    longest = max((duration for _, duration in intervals), default=0.)
    final = intervals[-1][1] if intervals and samples[-1][1] else 0.
    final_start = intervals[-1][0] if final else None
    retained = (first_start is not None
                and first_start <= touchdown + TOUCHDOWN_LAG_MAX_SECONDS
                and first_duration >= 2.)
    return dict(first_loaded_touchdown_seconds=first_start,
                first_loaded_contact_seconds=first_duration,
                touchdown_retained=retained,
                longest_loaded_landing_seconds=longest,
                final_loaded_landing_seconds=final,
                final_loaded_landing_start_seconds=final_start)


def evaluate(task, controller, stride_m, touchdown):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    foot_id = task.rig.model.body("Foot_L").id
    start_x = float(task.engine.datas[0].xpos[foot_id, 0])
    records = []
    captured = []
    observations, actions = [], []
    loaded_samples = []
    peak_lift = 0.
    start_z = float(task.engine.datas[0].xpos[foot_id, 2])
    done = DoneFlags.NULL.value
    sample_interval = round(DT / task.dt)
    native_dt = sample_interval * task.dt
    if not np.isclose(native_dt, DT, rtol=.001):
        raise ValueError("Native control period differs from the reference period")
    foot_ids = [task.rig.model.body(name).id for name in ("Foot_L", "Foot_R")]

    def capture(seconds):
        data = task.engine.datas[0]
        # mj_step's solved forces belong to its pre-integration state. The
        # engine has already called mj_forward here, so these forces match the
        # qpos and qvel stored in this reference frame.
        loads = np.abs(task.engine.current_state_ground_contact_forces(0)[task.allowed_ground, 2])
        sole_loads = loads >= 5.
        contact = np.asarray((sole_loads[0] or sole_loads[2],
                              sole_loads[1] or sole_loads[3]))
        # Keep v6-compatible labels and independent foot/toe loads separately.
        soles = np.r_[sole_loads[:2], contact]
        captured.append((seconds, data.qpos[:task.rig.model.nq].copy(),
                         data.qvel[:task.rig.model.nv].copy(),
                         data.xpos[foot_ids].copy(),
                         contact, soles, sole_loads))

    capture(0.)
    for _ in range(round(task.episode_seconds / task.dt)):
        observation = task.observations()
        with torch.no_grad():
            action = controller(observation)
        observations.append(observation[0].cpu().numpy().copy())
        actions.append(action[0].cpu().numpy().copy())
        _, _, flags, _ = task.step(action)
        done = int(flags[0])
        seconds = float(task.steps[0] * task.dt)
        if int(task.steps[0]) % sample_interval == 0:
            capture(seconds)
        pose = task.engine.native_qpos()[0]
        foot = task.engine.datas[0].xpos[foot_id]
        loads = task.engine.get_ground_contact_forces(0)[0, task.allowed_ground, 2].abs()
        left, right = float(loads[0] + loads[2]), float(loads[1] + loads[3])
        upright = bool(pose[2] >= .75 and 1 - 2 * (pose[4].square() + pose[5].square()) >= .9659)
        slow = bool(torch.linalg.vector_norm(task.engine.get_root_vel(0)[0, :2]) <= .2
                    and torch.linalg.vector_norm(task.engine.get_root_ang_vel(0)[0]) <= 1.)
        shift = float(foot[0] - start_x)
        peak_lift = max(peak_lift, float(foot[2] - start_z))
        loaded = (seconds >= touchdown and left >= LEFT_LOAD_MIN_N
                  and right >= RIGHT_LOAD_MIN_N and shift >= stride_m - .02 and upright)
        loaded_samples.append((seconds, loaded))
        if not records or seconds - records[-1]["seconds"] >= .1 or done != DoneFlags.NULL.value:
            root_velocity = task.engine.get_root_vel(0)[0]
            angular_velocity = task.engine.get_root_ang_vel(0)[0]
            data = task.engine.datas[0]
            com = data.subtree_com[1]
            com_jacobian = np.zeros((3, task.engine.rig.model.nv))
            mujoco.mj_jacSubtreeCom(task.engine.rig.model, data, com_jacobian, 1)
            com_velocity = com_jacobian @ data.qvel
            records.append(dict(seconds=seconds, left_load_n=left, right_load_n=right,
                                left_foot_shift_m=shift, left_foot_lift_m=float(foot[2] - start_z),
                                left_foot_position_m=data.xpos[foot_ids[0]].tolist(),
                                right_foot_position_m=data.xpos[foot_ids[1]].tolist(),
                                pelvis_height_m=float(pose[2]), upright=upright, slow=slow,
                                root_position_m=[float(value) for value in pose[:3]],
                                root_quaternion=[float(value) for value in pose[3:7]],
                                root_velocity_m_s=[float(value) for value in root_velocity],
                                root_angular_velocity_rad_s=[float(value) for value in angular_velocity],
                                com_position_m=[float(value) for value in com],
                                com_velocity_m_s=[float(value) for value in com_velocity]))
        if done != DoneFlags.NULL.value:
            break
    landing = loaded_landing_metrics(loaded_samples, touchdown, task.dt)
    result = dict(passed=bool(done == DoneFlags.TIME.value and landing["touchdown_retained"]
                            and peak_lift >= .03
                            and landing["final_loaded_landing_seconds"] >= 1.
                            and slow),
                survived=bool(done == DoneFlags.TIME.value), survival_seconds=seconds,
                peak_swing_lift_m=peak_lift,
                **landing,
                final_left_foot_shift_m=shift, final_slow=slow, trace=records)
    times, poses, velocities, targets, contacts, soles, measured_soles = zip(*captured)
    native_motion = dict(qpos=np.asarray(poses), qvel=np.asarray(velocities),
                         dt=native_dt, foot_targets=np.asarray(targets),
                         contact=np.asarray(contacts), sole_contact=np.asarray(soles),
                         measured_sole_contact=np.asarray(measured_soles),
                         support_contact=np.asarray(measured_soles))
    teacher_samples = dict(observation=np.asarray(observations),
                           action=np.asarray(actions), dt=task.dt)
    return result, native_motion, teacher_samples


class SwingReferenceController:
    """Keep the balance probe on the stance leg and follow the swing reference."""
    def __init__(self, task, balance, all_joints=False):
        self.task, self.balance = task, balance
        self.all_joints = all_joints
        names = [task.rig.model.actuator(int(index)).name for index in task.rig.actuators]
        self.left_leg = [index for index, name in enumerate(names)
                         if name.startswith(("Thigh_L", "Shin_L", "Foot_L"))]

    def __call__(self, observation):
        action = self.balance(observation)
        if float(self.task.steps[0] * self.task.dt) >= SWING_START:
            if self.all_joints:
                action[:] = 0.
            else:
                action[:, self.left_leg] = 0.
        return action


class TwoFootBalanceController:
    """Diagnostic use of the COM correction on the planted leg after touchdown."""
    def __init__(self, task, balance, touchdown, stride_m, include_coronal=False,
                 ankle_scale=1., com_kp=5., com_kd=1.):
        self.task, self.balance = task, balance
        self.touchdown, self.stride_m = touchdown, stride_m
        self.include_coronal = include_coronal
        self.ankle_scale, self.com_kp, self.com_kd = ankle_scale, com_kp, com_kd
        names = [task.rig.model.actuator(int(index)).name for index in task.rig.actuators]
        self.left_ankle = names.index("Foot_L_rx")
        self.right_ankle = names.index("Foot_R_rx")
        self.coronal_pairs = [(names.index(left), names.index(right)) for left, right in
                              (("Thigh_L_rz", "Thigh_R_rz"),
                               ("Foot_L_rz", "Foot_R_rz"))]
        self.left_foot = task.rig.model.body("Foot_L").id
        self.start_x = None
        self.landed = False

    def __call__(self, observation):
        action = self.balance(observation)
        data = self.task.engine.datas[0]
        if self.start_x is None:
            self.start_x = float(data.xpos[self.left_foot, 0])
        if float(self.task.steps[0] * self.task.dt) < self.touchdown:
            return action
        loads = self.task.engine.get_ground_contact_forces(0)[0, self.task.allowed_ground, 2].abs()
        left, right = float(loads[0] + loads[2]), float(loads[1] + loads[3])
        shift = float(data.xpos[self.left_foot, 0] - self.start_x)
        self.landed |= (left >= LEFT_LOAD_MIN_N and right >= RIGHT_LOAD_MIN_N
                        and shift >= self.stride_m - .02)
        if self.landed:
            self.balance.stabilizer.k_ap = self.com_kp
            self.balance.stabilizer.d_ap = self.com_kd
            correction = action[0, self.right_ankle] - self.balance.stabilizer.eq[self.right_ankle]
            action[0, self.left_ankle] = torch.clamp(
                action[0, self.left_ankle] + self.ankle_scale * correction, -1., 1.)
            if self.include_coronal:
                for left, right in self.coronal_pairs:
                    correction = action[0, right] - self.balance.stabilizer.eq[right]
                    action[0, left] = torch.clamp(action[0, left] + correction, -1., 1.)
        return action


class ResidualCorrectionController:
    """Use balance corrections without duplicating direct support feedforward."""
    def __init__(self, base, balance):
        self.base, self.balance = base, balance

    def __call__(self, observation):
        return (self.base(observation)
                - self.balance.stabilizer.eq.unsqueeze(0)).clamp(-1., 1.)


class FootAnchorController:
    """Native-only bounded IK correction for swing tracking or landing retention."""
    def __init__(self, task, base, touchdown, shift_m, blend, targets=None,
                 latch_world_at_contact=False, landing_time=None, latch_press_m=0.):
        self.task, self.base = task, base
        self.touchdown, self.shift_m = touchdown, shift_m
        self.blend = blend
        self.targets = targets
        self.latch_world_at_contact = latch_world_at_contact
        self.landing_time = touchdown if landing_time is None else landing_time
        self.latch_press_m = latch_press_m
        self.latched_world = None
        self.latched_at_seconds = None
        self.target_times = (np.arange(len(targets)) * task.reference.dt
                             if targets is not None else None)
        model = task.rig.model
        names = ("Thigh_L_rx", "Thigh_L_ry", "Thigh_L_rz",
                 "Shin_L_rx", "Foot_L_rx", "Foot_L_rz")
        joints = [model.joint(name).id for name in names]
        self.qpos = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints]
        actuator_names = [model.actuator(int(index)).name for index in task.rig.actuators]
        self.actions = [actuator_names.index(name) for name in names]
        self.foot = model.body("Foot_L").id
        self.start = None
        self.jacobian = np.zeros((3, model.nv))
        self.samples = 0
        self.saturated = 0

    def __call__(self, observation):
        action = self.base(observation)
        data = self.task.engine.datas[0]
        if self.start is None:
            self.start = data.xpos[self.foot].copy()
        seconds = float(self.task.steps[0] * self.task.dt)
        if seconds < self.touchdown:
            return action
        if self.latch_world_at_contact and self.latched_world is None and seconds >= self.landing_time:
            loads = self.task.engine.get_ground_contact_forces(0)[0, self.task.allowed_ground, 2].abs()
            left, right = float(loads[0] + loads[2]), float(loads[1] + loads[3])
            shift = float(data.xpos[self.foot, 0] - self.start[0])
            if (left >= LEFT_LOAD_MIN_N and right >= RIGHT_LOAD_MIN_N
                    and shift >= self.shift_m - .02):
                self.latched_world = data.xpos[self.foot].copy()
                self.latched_world[2] -= self.latch_press_m
                self.latched_at_seconds = seconds
        if self.latched_world is not None:
            desired = self.latched_world
        elif self.targets is None:
            desired = self.start + np.asarray((self.shift_m, 0., 0.))
        else:
            desired = np.asarray([np.interp(float(self.task.steps[0] * self.task.dt),
                                            self.target_times, self.targets[:, axis])
                                  for axis in range(3)])
        error = desired - data.xpos[self.foot]
        mujoco.mj_jacBody(self.task.rig.model, data, self.jacobian, None, self.foot)
        delta = bounded_ik_step(self.jacobian[:, self.dofs], error,
                                data.qpos[self.qpos], self.limits)
        target = data.qpos[self.qpos] + delta
        reference, _ = self.task.reference.sample(
            self.task.offset + self.task.steps * self.task.dt)
        residual = np.clip((target - reference[0, self.qpos].numpy()) / .25, -1., 1.)
        self.samples += 1
        self.saturated += int(np.any(np.abs(residual) >= .99))
        action[0, self.actions] = ((1. - self.blend) * action[0, self.actions]
                                   + self.blend * torch.as_tensor(residual, dtype=action.dtype))
        return action


def install_direct_feedforward(task, times, torques):
    """Add native support torque after PD; isolated from the exported action contract."""
    control = task.engine.policy_control
    original = control.physics_control

    def physics_control(qpos, qvel):
        result = original(qpos, qvel)
        seconds = float(task.steps[0] * task.dt)
        feedforward = torch.as_tensor(
            [np.interp(seconds, times, torques[:, joint]) for joint in range(torques.shape[1])],
            dtype=result.dtype)
        result[:, control.indices] = (
            result[:, control.indices]
            + feedforward.unsqueeze(0) * control.scale * control.active[:, 60:61]
        ).clamp(-control.scale, control.scale)
        return result

    control.physics_control = physics_control


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.smooth_root and args.smooth_motion:
        raise ValueError("Choose one smoothing mode")
    if sum(bool(mode) for mode in (args.zero_swing_residual, args.zero_all_residual,
                                  args.two_foot_balance)) > 1:
        raise ValueError("Choose one diagnostic controller mode")
    if (not np.isfinite((args.landing_ankle_scale, args.landing_com_kp,
                         args.landing_com_kd, args.forward_bias_m,
                         args.landing_com_x if args.landing_com_x is not None else 0.,
                         args.foot_anchor_m if args.foot_anchor_m is not None else 0.,
                         args.foot_anchor_blend, args.landing_target_offset_m,
                         args.lateral_touchdown_bias_m, args.latch_press_m,
                         args.hip_integral_ki)).all()
            or not 0 <= args.landing_ankle_scale <= 3
            or not 0 <= args.landing_com_kp <= 20
            or not 0 <= args.landing_com_kd <= 5
            or not -.05 <= args.forward_bias_m <= .05
            or args.landing_com_x is not None and not 0 <= args.landing_com_x <= .15
            or args.foot_anchor_m is not None and not .05 <= args.foot_anchor_m <= .25
            or not 0 <= args.foot_anchor_blend <= 1
            or not 0 <= args.hip_integral_ki <= 3
            or not 0 <= args.landing_target_offset_m <= .1
            or not -.06 <= args.lateral_touchdown_bias_m <= .06
            or not 0 <= args.latch_press_m <= .04):
        raise ValueError("Diagnostic gains or root bias are outside bounds")
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = CalibrationReference(args.reference, rig)
    with np.load(source.path, allow_pickle=False) as arrays:
        knots = {"fast": KNOTS, "slow": SLOW_KNOTS,
                 "recovery": RECOVERY_KNOTS, "delayed": DELAYED_KNOTS,
                 "settle": SETTLE_KNOTS}[args.tempo]
        motion = build_motion(rig, arrays, args.root_profile, args.stride_m,
                              knots, args.forward_bias_m, args.landing_com_x,
                              args.quiet_upper_swing, args.smooth_root,
                              args.smooth_motion, args.landing_target_offset_m,
                              args.lateral_touchdown_bias_m, args.braked_swing)
    checks, metrics = validate_motion(rig, motion["qpos"], motion["qvel"],
                                      motion["foot_targets"], motion["contact"],
                                      DT, motion["sole_contact"])
    placement = reference_placement(rig, motion)
    times, torques, centers, supports = support_schedule(rig, motion)
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = EntryReference(motion["qpos"], rig, DT)
    task.episode_seconds = task.reference.duration
    balance = ScheduledStabilizer(task, times, torques, centers,
                                  CONTROLLERS["balanced_axes"])
    balance.stabilizer.k_i_hr = args.hip_integral_ki
    requested_placement = args.stride_m + args.landing_target_offset_m
    if args.two_foot_balance:
        controller = TwoFootBalanceController(task, balance, knots[5][0], requested_placement,
                                               include_coronal=args.two_foot_balance == "full",
                                               ankle_scale=args.landing_ankle_scale,
                                               com_kp=args.landing_com_kp,
                                               com_kd=args.landing_com_kd)
    elif args.zero_swing_residual or args.zero_all_residual:
        controller = SwingReferenceController(task, balance, args.zero_all_residual)
    else:
        controller = balance
    foot_anchor = None
    if args.foot_anchor_m is not None or args.track_swing_foot or args.latch_world_at_contact:
        foot_anchor = FootAnchorController(
            task, controller, SWING_START if args.track_swing_foot else knots[5][0],
            requested_placement if args.latch_world_at_contact else args.foot_anchor_m,
            args.foot_anchor_blend,
            motion["foot_targets"][:, 0] if args.track_swing_foot else None,
            args.latch_world_at_contact, knots[5][0], args.latch_press_m)
        controller = foot_anchor
    if args.direct_feedforward:
        install_direct_feedforward(task, times, torques)
        controller = ResidualCorrectionController(controller, balance)
    result, native_motion, teacher_samples = evaluate(
        task, controller, requested_placement, knots[5][0])
    physical_passed = (all(checks[name] for name in PHYSICAL_CHECKS)
                       and abs(placement - requested_placement) < .02
                       and metrics["foot_lift_range_m"][0] >= .03)
    support_passed = all(sample["feasible"] for sample in supports)
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "step_reference.npz"
    np.savez_compressed(path, **motion)
    native_path = args.out / "native_rollout.npz"
    np.savez_compressed(native_path, **native_motion)
    teacher_path = args.out / "teacher_samples.npz"
    np.savez_compressed(teacher_path, **teacher_samples)
    native_checks, native_metrics = validate_motion(
        rig, native_motion["qpos"], native_motion["qvel"],
        native_motion["foot_targets"], native_motion["contact"], native_motion["dt"],
        support_contact=native_motion["support_contact"])
    parent = json.loads((args.reference / "report.json").read_text(encoding="utf-8"))
    origin = parent["source"]
    (args.out / "ATTRIBUTION.md").write_text(
        f"{origin['credit']}\n\nSource: {origin['source']}\n"
        f"License: {origin['license']} ({origin['license_url']})\n\n"
        "Adaptation: short transfer, swing, touchdown, and recovery diagnostic.\n",
        encoding="utf-8")
    report = dict(schema=SCHEMA, source_reference_sha256=source.sha256,
                  model_sha256=sha256(rig.path), reference_sha256=sha256(path),
                  phase_seconds=dict(swing_start=SWING_START, touchdown=knots[5][0],
                                     recovery_end=knots[6][0], end=knots[-1][0]),
                  requested_placement_m=requested_placement, measured_placement_m=placement,
                  physical_checks=checks,
                  physical_metrics=metrics, support_schedule=supports,
                  native_rollout_sha256=sha256(native_path),
                  teacher_samples_sha256=sha256(teacher_path),
                  native_rollout_checks=native_checks,
                  native_rollout_metrics=native_metrics,
                  native_evaluation=result, physical_passed=physical_passed,
                  support_passed=support_passed,
                  passed=bool(physical_passed and support_passed and result["passed"]),
                  admitted_for_training=False,
                  zero_swing_residual=args.zero_swing_residual,
                  zero_all_residual=args.zero_all_residual,
                  two_foot_balance=args.two_foot_balance,
                  landing_ankle_scale=args.landing_ankle_scale,
                  landing_com_kp=args.landing_com_kp,
                  landing_com_kd=args.landing_com_kd,
                  hip_integral_ki=args.hip_integral_ki,
                  root_profile=args.root_profile,
                  tempo=args.tempo,
                  forward_bias_m=args.forward_bias_m,
                  landing_com_x=args.landing_com_x,
                  quiet_upper_swing=args.quiet_upper_swing,
                  smooth_root=args.smooth_root,
                  smooth_motion=args.smooth_motion,
                  direct_feedforward=args.direct_feedforward,
                  foot_anchor_m=args.foot_anchor_m,
                  foot_anchor_blend=args.foot_anchor_blend,
                  track_swing_foot=args.track_swing_foot,
                  landing_target_offset_m=args.landing_target_offset_m,
                  lateral_touchdown_bias_m=args.lateral_touchdown_bias_m,
                  braked_swing=args.braked_swing,
                  foot_anchor_saturated_fraction=(foot_anchor.saturated / max(1, foot_anchor.samples)
                                                  if foot_anchor else None),
                  latch_world_at_contact=args.latch_world_at_contact,
                  latch_press_m=args.latch_press_m,
                  world_latched_seconds=foot_anchor.latched_at_seconds if foot_anchor else None,
                  world_latched_position_m=(foot_anchor.latched_world.tolist()
                                            if foot_anchor and foot_anchor.latched_world is not None else None),
                  scope="Native diagnostic motion and oracle-COM controller; no training, export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"],
                          measured_placement_m=placement,
                          physical_checks=checks,
                          infeasible_support_samples=sum(not s["feasible"] for s in supports),
                          native={k: v for k, v in result.items() if k != "trace"})), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--zero-swing-residual", action="store_true")
    parser.add_argument("--zero-all-residual", action="store_true")
    parser.add_argument("--two-foot-balance", nargs="?", const="ankle",
                        choices=("ankle", "full"))
    parser.add_argument("--landing-ankle-scale", type=float, default=1.)
    parser.add_argument("--landing-com-kp", type=float, default=5.)
    parser.add_argument("--landing-com-kd", type=float, default=1.)
    parser.add_argument("--hip-integral-ki", type=float, default=3.)
    parser.add_argument("--root-profile", choices=("source", "anchored"), default="source")
    parser.add_argument("--stride-m", type=float, choices=(.05, .10, .15), default=.15)
    parser.add_argument("--tempo", choices=("fast", "slow", "recovery", "delayed", "settle"), default="fast")
    parser.add_argument("--forward-bias-m", type=float, default=0.)
    parser.add_argument("--landing-com-x", type=float)
    parser.add_argument("--quiet-upper-swing", action="store_true")
    parser.add_argument("--smooth-root", action="store_true")
    parser.add_argument("--smooth-motion", action="store_true")
    parser.add_argument("--direct-feedforward", action="store_true")
    parser.add_argument("--foot-anchor-m", type=float)
    parser.add_argument("--foot-anchor-blend", type=float, default=1.)
    parser.add_argument("--track-swing-foot", action="store_true")
    parser.add_argument("--latch-world-at-contact", action="store_true")
    parser.add_argument("--latch-press-m", type=float, default=0.)
    parser.add_argument("--landing-target-offset-m", type=float, default=0.)
    parser.add_argument("--lateral-touchdown-bias-m", type=float, default=0.)
    parser.add_argument("--braked-swing", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())

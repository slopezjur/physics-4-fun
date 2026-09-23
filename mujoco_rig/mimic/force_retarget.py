"""Refine walking references with stance constraints and delayed motor authority."""
import argparse
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear, linprog
from scipy.sparse import lil_matrix

from .baseline import sha256
from .model_contract import collision_pairs
from .motion_dynamics import frame_dynamics, delayed_pd_bounds
from .contact_projection import StanceProjection
from .collision_clearance import CollisionClearance
from .constrained_fit import constrained_least_squares
from .direct_forces import force_balance
from .transfer_seed import rotation_offsets
from .reference_limits import FOOT_TARGET_ERROR_M, SOLE_HEIGHT_ERROR_M, STANCE_SLIDE_M_S, PENETRATION_M
from .motion_contacts import (infer_toe_off, sole_schedule, validate_sole_schedule, TOE_REFERENCE_SCHEMA,
                              stance_phase_ids, expand_landing_offsets)
from .rig import Rig
from mujoco_rig.rl.env_config import ACTION_LATENCY_STEPS


def friction_rays(friction):
    """An inner pyramid: every nonnegative combination satisfies Coulomb friction."""
    if not np.isfinite(friction) or friction < 0:
        raise ValueError("Friction must be finite and nonnegative")
    return np.array([[x * friction / np.sqrt(2), y * friction / np.sqrt(2), 1.]
                     for x, y in itertools.product((-1, 1), repeat=2)])


def contact_force_columns(model, data, soles, points, rays, active):
    """Generalized force columns for the fitter's unchanged contact envelope."""
    columns = []
    for sole in np.flatnonzero(active):
        geom = soles[sole]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        for point in points[sole] @ rotation.T + data.geom_xpos[geom]:
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, point, int(model.geom_bodyid[geom]))
            columns.extend((jacobian.T @ rays[sole].T).T)
    return np.asarray(columns).reshape(-1, model.nv).T


def rotate_root(quaternions, offsets):
    """Apply bounded world-frame rotation-vector offsets to native wxyz roots."""
    angle = np.linalg.norm(offsets, axis=1, keepdims=True)
    w = np.cos(angle / 2)
    xyz = .5 * np.sinc(angle / (2 * np.pi)) * offsets
    return np.column_stack((w[:, 0] * quaternions[:, 0] - np.sum(xyz * quaternions[:, 1:], axis=1),
                            w * quaternions[:, 1:] + quaternions[:, :1] * xyz + np.cross(xyz, quaternions[:, 1:])))


def reference_velocities(model, poses, dt):
    """Central tangent velocities, with one-sided derivatives at the endpoints."""
    if len(poses) < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Velocity reconstruction requires two frames and a positive timestep")
    result = np.zeros((len(poses), model.nv))
    for frame in range(len(poses)):
        if frame == 0:
            mujoco.mj_differentiatePos(model, result[frame], dt, poses[frame], poses[frame + 1])
        elif frame == len(poses) - 1:
            mujoco.mj_differentiatePos(model, result[frame], dt, poses[frame - 1], poses[frame])
        else:
            backward, forward = np.zeros(model.nv), np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, backward, dt, poses[frame], poses[frame - 1])
            mujoco.mj_differentiatePos(model, forward, dt, poses[frame], poses[frame + 1])
            result[frame] = (forward - backward) / 2
    return result


def complete_stance_targets(targets, original_contact, schedule):
    """Extend a stance lock from its existing source anchor into inferred gaps."""
    result = targets.copy()
    for foot in range(2):
        starts = np.flatnonzero(schedule[:, foot] & ~np.r_[False, schedule[:-1, foot]])
        stops = np.flatnonzero(schedule[:, foot] & ~np.r_[schedule[1:, foot], False]) + 1
        for start, stop in zip(starts, stops):
            existing = np.flatnonzero(original_contact[start:stop, foot])
            anchor = start + int(existing[0]) if len(existing) else start
            result[start:stop, foot] = targets[anchor, foot]
    return result


def project_actuated_force(contact_columns, required, controlled, torque_scale, weight, bounds, return_forces=False):
    """Nearest generalized force using unilateral contacts and bounded motor commands.

    Contact coefficients are in body-weight units; motor coefficients are normalized
    commands. Fixed (including saturated) motor intervals are eliminated exactly.
    Root residuals use body weight, and joint residuals use their motor authority.
    """
    scale = np.r_[np.full(6, weight), torque_scale]
    matrix = contact_columns[controlled] * weight / scale[:, None]
    target = required[controlled] / scale
    # Contact forces do not act on arms or the swing leg's generalized coordinates.
    # Those independent motor rows have an exact clipped solution; eliminate them
    # before the coupled solve rather than repeatedly factor a mostly zero matrix.
    coupled = np.any(np.abs(matrix[6:]) > 1e-12, axis=1)
    error = target.copy()
    error[6:] -= np.clip(target[6:], bounds[:, 0], bounds[:, 1])
    if matrix.shape[1] == 0:
        return (error, np.empty(0)) if return_forces else error
    rows = np.r_[np.arange(6), np.flatnonzero(coupled) + 6]
    matrix, target, bounds = matrix[rows], target[rows], bounds[coupled]
    motors = np.zeros((len(rows), int(coupled.sum())))
    motors[6:] = np.eye(int(coupled.sum()))
    fixed = bounds[:, 1] - bounds[:, 0] < 1e-10
    target -= motors[:, fixed] @ bounds[fixed, 0]
    matrix = np.column_stack((matrix, motors[:, ~fixed]))
    lower = np.r_[np.zeros(contact_columns.shape[1]), bounds[~fixed, 0]]
    upper = np.r_[np.full(contact_columns.shape[1], np.inf), bounds[~fixed, 1]]
    solution = lsq_linear(matrix, target, bounds=(lower, upper), method="bvls", tol=1e-9)
    error[rows] = target - matrix @ solution.x
    return (error, solution.x[:contact_columns.shape[1]]) if return_forces else error


def refine_support(rig, poses, foot_targets, contact, dt, max_evaluations=80, linear_iterations=1000,
                   progress_callback=None, root_seed_drop_m=.02, toe_off=False, source_soles=None,
                   self_clearance_m=0., adjust_landings=False, landing_seed=None, linear_solver="lsmr",
                   fixed_frames=None, stance_rotations=None, constrained=False, direct_forces=False, initial_poses=None,
                   stationary=False):
    """Fit poses to admissible contact forces and motor commands on the existing rig.

    By default an inner bounded least-squares solve eliminates contact and motor variables.
    Contact forces are in body-weight units. A shrunken footprint and inner friction
    pyramid leave margin for the independent full-footprint feasibility check.
    Bounded stance-leg IK is part of each residual evaluation, never a postprocess.
    Fixed-context windows can additionally require explicit placement, contact
    geometry and force-feasibility inequalities with constrained=True.
    direct_forces=True exposes contact forces and restores feasibility first.
    """
    if len(poses) < 3 or not np.isfinite(dt) or dt <= 0 or min(max_evaluations, linear_iterations) < 1:
        raise ValueError("Support refinement requires three frames and a positive timestep")
    if not np.isfinite(self_clearance_m) or not 0 <= self_clearance_m <= .05:
        raise ValueError("Self-contact clearance must be between 0 and 0.05 metres")
    if linear_solver not in ("lsmr", "exact"):
        raise ValueError("Linear solver must be lsmr or exact")
    fixed = np.zeros(len(poses), dtype=bool) if fixed_frames is None else np.asarray(fixed_frames)
    if fixed.shape != (len(poses),) or fixed.dtype != bool or fixed.all():
        raise ValueError("Fixed frames must be a boolean mask leaving at least one editable pose")
    if fixed.any() and adjust_landings:
        raise ValueError("Window fitting must preserve landing anchors across fixed context")
    if constrained and (not fixed.any() or progress_callback is not None):
        raise ValueError("Constrained fitting requires fixed context and final full-clip validation")
    if direct_forces and not constrained:
        raise ValueError("Direct forces require constrained window fitting")
    if stationary and (constrained or fixed.any() or adjust_landings or landing_seed is not None or progress_callback is not None
                       or initial_poses is not None or toe_off
                       or not np.all(poses == poses[0]) or not np.all(foot_targets == foot_targets[0])
                       or not np.all(contact == contact[0])
                       or (stance_rotations is not None and not np.all(stance_rotations == stance_rotations[0]))
                       or (source_soles is not None and not np.all(source_soles == source_soles[0]))):
        raise ValueError("Stationary fitting requires identical poses, targets and support, without window/landing edits")
    if source_soles is not None:
        toe_off = True
    model, data = rig.model, mujoco.MjData(rig.model)
    if (poses.shape != (len(poses), model.nq) or foot_targets.shape != (len(poses), 2, 3)
            or contact.shape != (len(poses), 2) or not np.isfinite(poses).all()
            or not np.isfinite(foot_targets).all()):
        raise ValueError("Invalid reference dimensions or nonfinite poses/targets")
    indices = model.jnt_qposadr[model.actuator_trnid[rig.actuators, 0]]
    joints = model.actuator_trnid[rig.actuators, 0]
    controlled = np.r_[np.arange(6), model.jnt_dofadr[joints]]
    delay = ACTION_LATENCY_STEPS * rig.decimation * model.opt.timestep
    if stationary and dt < delay:
        raise ValueError("Stationary fitting requires a post-delay interior frame")
    if stationary:
        linear_solver = "exact"
    feet = [model.body(name).id for name in ("Foot_L", "Foot_R")]
    soles = [model.geom("g_" + name).id for name in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in soles):
        raise ValueError("Support refinement requires box soles")
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    local_corners = signs[None] * model.geom_size[soles, None]
    # Inset force points by 15 mm so an approximate least-squares solution has
    # support margin. All four physical stance corners are constrained to floor.
    sole_points = np.array([[x, y, -1.] for x, y in itertools.product((-1., 1.), repeat=2)])
    force_points = sole_points[None] * model.geom_size[soles, None]
    force_points[:, :, :2] *= (model.geom_size[soles, :2] - .015)[:, None] / model.geom_size[soles, None, :2]
    rays = [friction_rays(.8 * max(model.geom_friction[g, 0], model.geom_friction[0, 0])) for g in soles]
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    pairs = sorted(collision_pairs(model))
    distance_query = CollisionClearance(model, pairs)
    # A soft penalty aimed exactly at the admission boundary can settle just below
    # it. Leave a 1 mm fitting reserve; the independent clearance gate is unchanged.
    clearance_reserve = .001 if self_clearance_m > 0 else 0.
    clearance = np.array([self_clearance_m + clearance_reserve if 0 not in pair else 0. for pair in pairs])
    schedule = np.asarray(contact, dtype=bool).copy()
    filled = []
    rotations = []
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        rotations.append(data.xmat[feet].copy().reshape(2, 3, 3))
        if not schedule[frame].any():
            corners = local_corners @ data.geom_xmat[soles].reshape(4, 3, 3).transpose(0, 2, 1) + data.geom_xpos[soles, None]
            gaps = corners[:, :, 2].min(axis=1)
            foot_gaps = np.minimum(gaps[:2], gaps[2:])
            schedule[frame] = foot_gaps <= foot_gaps.min() + .001
            filled.append(frame)
    rotations = np.asarray(rotations)
    toe_only = (validate_sole_schedule(contact, source_soles) if source_soles is not None
                else infer_toe_off(contact, schedule, rotations) if toe_off else np.zeros_like(schedule))
    soles_active = sole_schedule(schedule, toe_only)
    phase_ids, phase_count = stance_phase_ids(schedule)
    if landing_seed is not None:
        adjust_landings = not fixed.any()
    landing_initial = np.zeros((phase_count, 2)) if landing_seed is None else np.asarray(landing_seed, dtype=float)
    if landing_initial.shape != (phase_count, 2) or not np.isfinite(landing_initial).all() or np.abs(landing_initial).max() > .020001:
        raise ValueError("Landing offsets must match the stance intervals and remain within 2 cm per axis")
    landing_width = 2 * phase_count if adjust_landings else 0
    original_targets = foot_targets
    foot_targets = complete_stance_targets(foot_targets, contact, schedule)
    seed_poses = poses.copy()
    if initial_poses is not None:
        seed_poses = np.asarray(initial_poses).copy()
        if (not fixed.any() or root_seed_drop_m != 0. or seed_poses.shape != poses.shape
                or not np.isfinite(seed_poses).all() or not np.array_equal(seed_poses[fixed], poses[fixed])
                or not np.allclose(np.linalg.norm(seed_poses[:, 3:7], axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError("Initial poses require fixed context, unit roots and zero additional root drop")
        passive = np.setdiff1d(np.arange(7, model.nq), indices)
        if not np.array_equal(seed_poses[:, passive], poses[:, passive]):
            raise ValueError("Initial poses cannot change uncontrolled coordinates")
    seed_poses[:, 2] -= root_seed_drop_m
    seed_poses[fixed] = poses[fixed]
    if source_soles is None:
        for foot, side in enumerate(("L", "R")):
            toe_q = model.jnt_qposadr[model.joint(f"Toe_{side}_rx").id]
            pitch = np.arctan2(-rotations[:, foot, 2, 0], np.linalg.norm(rotations[:, foot, :2, 0], axis=1))
            seed_poses[toe_only[:, foot], toe_q] = np.clip(pitch[toe_only[:, foot]], .02, .8)
    projection = StanceProjection(rig, seed_poses, foot_targets, schedule, toe_only,
                                  expand_landing_offsets(phase_ids, landing_initial), stance_rotations)
    foot_targets = projection.targets
    pose_width = 6 + len(indices)
    frames, width = len(poses), pose_width
    root_offsets = rotation_offsets(poses[:, 3:7], seed_poses[:, 3:7]) if initial_poses is not None else np.zeros((frames, 3))
    initial = np.column_stack((seed_poses[:, :3], root_offsets, seed_poses[:, indices]))
    active = projection.independent_coordinates(indices, root_width=6)
    active[fixed] = False
    frame_width = 6 + len(indices) + 6 + 18 + 16 + len(pairs)
    dynamics_width = len(controlled)
    total_rows = frames * frame_width + (frames - 2) * (dynamics_width + pose_width) + landing_width
    pose_variables = int(active.sum())
    past_dependency = int(np.ceil(delay / dt)) + 1
    affected_dynamics = np.array([(~fixed[max(0, frame - past_dependency):frame + 2]).any()
                                  for frame in range(1, frames - 1)])
    root_dependencies = np.array([(~fixed[frame - 1:frame + 2]).any() for frame in range(1, frames - 1)])
    deferred_dynamics = []
    if constrained:
        # Later commands may depend on edited poses even though the current
        # pose, velocity and acceleration are immutable. An unsupported frozen
        # root cannot be repaired by changing delayed motor bounds.
        for frame in np.flatnonzero(affected_dynamics & ~root_dependencies) + 1:
            required = frame_dynamics(model, data, poses, frame, dt)
            matrix = contact_force_columns(model, data, soles, force_points, rays, soles_active[frame])
            certificate = linprog(np.zeros(matrix.shape[1]), A_eq=matrix[:6], b_eq=required[:6] / weight,
                                  bounds=(0, None), method="highs")
            # Only a certified infeasible LP is deferred. Numerical failure or
            # incomplete optimization is not evidence that repair is impossible.
            if certificate.status == 2:
                support = lsq_linear(matrix[:6], required[:6] / weight, bounds=(0, np.inf), method="bvls", tol=1e-10)
                violation = float(np.abs(matrix[:6] @ support.x - required[:6] / weight).max())
                affected_dynamics[frame - 1] = False
                deferred_dynamics.append(dict(frame=int(frame), root_residual=violation, lp_status=int(certificate.status),
                                               reason="Immutable root infeasible in fitting contact envelope"))
    support_bodies = model.geom_bodyid[soles] if toe_off else feet
    support_schedule = soles_active if toe_off else schedule
    affected_slides = (support_schedule[1:] & support_schedule[:-1]
                       & ((~fixed[1:]) | (~fixed[:-1]))[:, None])
    constraint_groups = {}
    force_slices, force_width = {}, 0
    if direct_forces:
        for frame in np.flatnonzero(affected_dynamics) + 1:
            count = 16 * int(soles_active[frame].sum())
            force_slices[frame] = slice(force_width, force_width + count)
            force_width += count
    force_seed = np.zeros(force_width)

    def decode(flat):
        x = initial.copy()
        x[active] = flat[:pose_variables]
        landings = flat[pose_variables:].reshape(-1, 2) if adjust_landings else landing_initial
        result = poses.copy()
        result[:, :3], result[:, indices] = x[:, :3], x[:, 6:]
        result[:, 3:7] = rotate_root(poses[:, 3:7], x[:, 3:6])
        result = projection.project(result, anchor_offsets=expand_landing_offsets(phase_ids, landings), active_frames=~fixed)
        result[fixed] = poses[fixed]
        return x, result

    def evaluate(flat, seed_forces=False, force_derivatives=False):
        x, trajectory = decode(flat)
        velocities = reference_velocities(model, trajectory, dt)
        parts, wrenches = [], []
        placement, support_positions = [], []
        force_margins, force_blocks = [], []
        for frame, pose in enumerate(trajectory):
            if 0 < frame < frames - 1:
                required = frame_dynamics(model, data, trajectory, frame, dt)
            else:
                data.qpos[:] = pose
                mujoco.mj_kinematics(model, data)
                required = np.zeros(model.nv)
            heights = []
            for sole, geom in enumerate(soles):
                rotation = data.geom_xmat[geom].reshape(3, 3)
                physical = sole_points * model.geom_size[geom]
                heights.extend(soles_active[frame, sole] * (physical @ rotation.T + data.geom_xpos[geom])[:, 2])
            distances = distance_query.distances(data, max(.01, self_clearance_m + clearance_reserve))
            if constrained:
                support_positions.append(data.xpos[support_bodies].copy())
                if not fixed[frame]:
                    foot_error = np.linalg.norm(data.xpos[feet] - foot_targets[frame], axis=1)
                    active_heights = np.asarray(heights).reshape(4, 4)[soles_active[frame]].ravel()
                    # Tiny reserves keep strict admission inequalities off their
                    # boundary. These constraints cannot be traded for lower cost.
                    placement.extend(((FOOT_TARGET_ERROR_M - 1e-6 - foot_error) / FOOT_TARGET_ERROR_M,
                                      (SOLE_HEIGHT_ERROR_M - 1e-6 - active_heights) / SOLE_HEIGHT_ERROR_M,
                                      (SOLE_HEIGHT_ERROR_M - 1e-6 + active_heights) / SOLE_HEIGHT_ERROR_M))
                    minimum_distance = np.where((clearance > 0), clearance, -PENETRATION_M + 1e-6)
                    placement.append((distances - minimum_distance) / PENETRATION_M)
            parts.extend((2 * (pose[:3] - poses[frame, :3]), x[frame, 3:6],
                          .2 * (pose[indices] - poses[frame, indices]),
                          (np.where(schedule[frame] & ~toe_only[frame] & (not adjust_landings), 500., 80.)[:, None] * (data.xpos[feet] - foot_targets[frame])).ravel(),
                          (10 * (data.xmat[feet].reshape(2, 3, 3) - rotations[frame])).ravel(),
                          3000 * np.array(heights), 1000 * np.minimum(distances - clearance, 0)))
            if 0 < frame < frames - 1:
                matrix = contact_force_columns(model, data, soles, force_points, rays, soles_active[frame])
                bounds = None
                if frame * dt >= delay:
                    bounds = delayed_pd_bounds(rig, trajectory, velocities, dt, frame)
                    reserve = .02 * (bounds[:, 1] - bounds[:, 0])
                    bounds += np.column_stack((reserve, -reserve))
                if direct_forces and frame in force_slices:
                    columns_slice = force_slices[frame]
                    if seed_forces:
                        if bounds is None:
                            coefficients = lsq_linear(matrix[:6], required[:6] / weight,
                                                      bounds=(0, np.inf), method="bvls").x
                        else:
                            _, coefficients = project_actuated_force(matrix, required, controlled,
                                                                      rig.action_scale, weight, bounds, return_forces=True)
                        # BVLS can return tiny negative roundoff at an active
                        # zero bound; least_squares requires a feasible seed.
                        force_seed[columns_slice] = np.maximum(coefficients, 0.)
                    coefficients = force_seed[columns_slice] if seed_forces else flat[pose_variables:][columns_slice]
                    error, feasible, error_jac, feasible_jac = force_balance(
                        matrix, required, controlled, rig.action_scale, weight, coefficients, bounds)
                    force_margins.append(feasible)
                    if force_derivatives:
                        force_blocks.append((frame, columns_slice, error_jac, feasible_jac))
                elif bounds is None:
                    forces = lsq_linear(matrix[:6], required[:6] / weight, bounds=(0, np.inf), method="bvls")
                    error = np.r_[required[:6] / weight - matrix[:6] @ forces.x, np.zeros(len(indices))]
                else:
                    # Keep a small reserve inside the real target-PD envelope.
                    error = project_actuated_force(matrix, required, controlled, rig.action_scale, weight, bounds)
                wrenches.append(200 * error)
        parts.append(np.array(wrenches).ravel())
        coordinates = np.column_stack((trajectory[:, :3], x[:, 3:6], trajectory[:, indices]))
        parts.append((np.diff(coordinates, n=2, axis=0) / dt**2 * np.r_[np.full(6, .02), np.full(len(indices), .002)]).ravel())
        if adjust_landings:
            parts.append(10 * flat[pose_variables:])
        margins = []
        if constrained:
            speed = np.linalg.norm(np.diff(np.asarray(support_positions)[..., :2], axis=0), axis=-1) / dt
            placement.append((STANCE_SLIDE_M_S - 1e-6 - speed[affected_slides]) / STANCE_SLIDE_M_S)
            geometry = np.concatenate(placement)
            equilibrium = np.asarray(wrenches)[affected_dynamics].ravel() / 200
            # Include every dependent frame, including fixed poses whose forces
            # or delayed command intervals change when neighboring poses move.
            margins = (np.r_[geometry, np.concatenate(force_margins)] if direct_forces
                       else np.r_[geometry, 1e-8 - equilibrium, 1e-8 + equilibrium])
            constraint_groups.update(placement=slice(0, len(geometry)),
                                     equilibrium=slice(len(geometry), len(margins)))
        result = np.concatenate(parts)
        if force_derivatives:
            jr, jc = np.zeros((len(result), force_width)), np.zeros((len(margins), force_width))
            row = len(geometry)
            for frame, columns_slice, error_jac, feasible_jac in force_blocks:
                first = frames * frame_width + (frame - 1) * dynamics_width
                jr[first:first + dynamics_width, columns_slice] = 200 * error_jac
                jc[row:row + len(feasible_jac), columns_slice] = feasible_jac
                row += len(feasible_jac)
            return jr, jc
        return result, np.asarray(margins)

    def residual(flat):
        return evaluate(flat)[0]

    sparsity = lil_matrix((total_rows, frames * width + landing_width), dtype=int)

    def landing_dependencies(rows, neighbor):
        if adjust_landings:
            for phase in phase_ids[neighbor]:
                if phase >= 0:
                    column = frames * width + 2 * phase
                    sparsity[rows, column:column + 2] = 1

    for frame in range(frames):
        sparsity[frame * frame_width:(frame + 1) * frame_width, frame * width:(frame + 1) * width] = 1
        landing_dependencies(slice(frame * frame_width, (frame + 1) * frame_width), frame)
    base = frames * frame_width
    for frame in range(frames - 2):
        for neighbor in range(max(0, frame + 1 - past_dependency), frame + 3):
            sparsity[base + frame * dynamics_width:base + (frame + 1) * dynamics_width,
                     neighbor * width:(neighbor + 1) * width] = 1
            landing_dependencies(slice(base + frame * dynamics_width, base + (frame + 1) * dynamics_width), neighbor)
        row = base + (frames - 2) * dynamics_width + frame * pose_width
        sparsity[row:row + pose_width, frame * width:(frame + 3) * width] = 1
        for neighbor in range(frame, frame + 3):
            landing_dependencies(slice(row, row + pose_width), neighbor)
    if adjust_landings:
        for i in range(landing_width):
            sparsity[total_rows - landing_width + i, frames * width + i] = 1
    lower = np.column_stack((poses[:, :3] - [.2, .2, .12], np.full((frames, 3), -.15), np.tile(model.jnt_range[joints, 0] + 1e-5, (frames, 1))))
    upper = np.column_stack((poses[:, :3] + [.2, .2, .05], np.full((frames, 3), .15), np.tile(model.jnt_range[joints, 1] - 1e-5, (frames, 1))))
    last_checked = 0

    def check_progress(intermediate_result):
        nonlocal last_checked
        if progress_callback is not None and intermediate_result.nfev - last_checked >= 5:
            last_checked = int(intermediate_result.nfev)
            _, candidate = decode(intermediate_result.x)
            if progress_callback(candidate, foot_targets, schedule, last_checked, soles_active if toe_off else None):
                raise StopIteration

    started = time.monotonic()
    variables = np.r_[active.ravel(), np.ones(landing_width, dtype=bool)]
    starting = np.r_[initial.clip(lower, upper)[active], landing_initial.ravel() if adjust_landings else []]
    if stationary:
        coordinates = pose_variables // frames
        fitted = least_squares(lambda flat: residual(np.tile(flat, frames)), starting[:coordinates],
                               bounds=(lower[active][:coordinates], upper[active][:coordinates]),
                               max_nfev=max_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                               x_scale="jac", tr_solver="exact")
        fitted.x = np.tile(fitted.x, frames)
    elif constrained:
        if direct_forces:
            evaluate(starting, seed_forces=True)
            starting = np.r_[starting, force_seed]
        fitted = constrained_least_squares(evaluate, starting,
                                           np.r_[lower[active], np.full(landing_width, -.02), np.zeros(force_width)],
                                           np.r_[upper[active], np.full(landing_width, .02), np.full(force_width, np.inf)],
                                           max_evaluations,
                                           numerical_columns=pose_variables if direct_forces else None,
                                           analytic_tail=(lambda flat: evaluate(flat, force_derivatives=True)) if direct_forces else None,
                                           restore_feasibility=direct_forces)
    else:
        fitted = least_squares(residual, starting,
                           bounds=(np.r_[lower[active], np.full(landing_width, -.02)],
                                   np.r_[upper[active], np.full(landing_width, .02)]),
                           jac_sparsity=sparsity.tocsr()[:, variables] if linear_solver == "lsmr" else None,
                           max_nfev=max_evaluations, ftol=1e-5, x_scale="jac", tr_solver=linear_solver,
                           tr_options=({"atol": 1e-6, "btol": 1e-6, "maxiter": linear_iterations}
                                       if linear_solver == "lsmr" else {}),
                           callback=check_progress, verbose=2)
    _, result = decode(fitted.x)
    equilibrium = residual(fitted.x)[base:base + (frames - 2) * dynamics_width].reshape(-1, dynamics_width) / 200
    restoration_evaluations = (fitted.restoration["evaluations"] if constrained and fitted.restoration else 0)
    report = dict(converged=bool(fitted.success), evaluations=int(fitted.nfev) + restoration_evaluations, cost=float(fitted.cost),
                  optimality=float(fitted.optimality) if fitted.optimality is not None else None,
                  seconds=time.monotonic() - started, solver_message=fitted.message,
                  max_root_force_residual_body_weight=float(np.abs(equilibrium[:, :3]).max()),
                  max_root_moment_residual_body_weight_m=float(np.abs(equilibrium[:, 3:6]).max()),
                  max_motor_residual_authority_fraction=float(np.abs(equilibrium[:, 6:]).max()),
                  inferred_support_gap_frames=filled, force_point_inset_m=.015, friction_fraction=.8,
                  linear_solver=linear_solver,
                  fixed_frame_indices=np.flatnonzero(fixed).tolist(),
                  linear_solver_iterations=linear_iterations if linear_solver == "lsmr" else None,
                  support_bodies=[model.geom(g).name for g in soles],
                  max_stance_target_adjustment_m=float(np.linalg.norm(foot_targets - original_targets, axis=-1).max()),
                  stance_method="Bounded six-hinge IK; ankle yaw independent; explicit flat-foot/toe-only anchors",
                  sole_contact=soles_active.tolist() if toe_off else None,
                  toe_only_frames=[np.flatnonzero(toe_only[:, foot]).tolist() for foot in range(2)],
                  root_rotation_correction_bound_per_axis_rad=.15,
                  root_seed_drop_m=root_seed_drop_m,
                  supplied_initial_poses=initial_poses is not None,
                  stationary=stationary,
                  clipped_initial_coordinates=int(np.count_nonzero(initial[active] != initial.clip(lower, upper)[active])),
                  self_clearance_m=self_clearance_m,
                  clearance_reserve_m=clearance_reserve,
                  clearance_distance="conservative box SAT; native other shapes",
                  stance_anchor_offsets_xy=fitted.x[pose_variables:].reshape(-1, 2).tolist() if adjust_landings else None,
                  stance_anchor_bound_per_axis_m=.02 if adjust_landings else 0.,
                  pd_interval_reserve_fraction=.02, optimized_coordinates=len(fitted.x) // frames if stationary else len(fitted.x),
                  scope="Root and controlled-joint equilibrium inside delayed target-PD intervals; passive joints, initial queue fill and closed-loop stability remain outside this necessary-condition fit.")
    if constrained:
        margins = evaluate(fitted.x)[1]
        report.update(optimizer="SLSQP", iterations=int(fitted.nit), iteration_budget=max_evaluations,
                      residual_evaluations=fitted.residual_evaluations,
                      constraints_satisfied=fitted.constraints_satisfied,
                      used_feasible_incumbent=fitted.used_feasible_incumbent,
                      final_iterate_constraint_violation=fitted.final_iterate_constraint_violation,
                      feasibility_restoration=fitted.restoration,
                      max_constraint_violation=fitted.max_constraint_violation,
                      constraint_violations={name: float(max(0., -margins[rows].min(initial=0.)))
                                             for name, rows in constraint_groups.items()},
                      constrained_dynamics_frames=(np.flatnonzero(affected_dynamics) + 1).tolist(),
                      deferred_immutable_dynamics=deferred_dynamics,
                      constraint_history=fitted.history,
                      force_formulation="explicit nonnegative friction rays" if direct_forces else "projected bounded forces",
                      force_variables=force_width,
                      force_coefficients={str(frame): fitted.x[pose_variables:][columns_slice].tolist()
                                          for frame, columns_slice in force_slices.items()},
                      linear_solver=None, linear_solver_iterations=None)
    return result, foot_targets, schedule, report


def refine_file(reference, out, max_evaluations, linear_iterations=1000, toe_off=False, self_clearance_m=None, adjust_landings=False, linear_solver="lsmr"):
    """Refine a hash-bound archived reference without modifying its original bytes."""
    from .retarget_steps import attribution, validate_motion
    from .motion_protocol import REFERENCE_SCHEMA
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    metadata = json.loads(reference.with_suffix(".json").read_text())
    if self_clearance_m is None:
        self_clearance_m = metadata.get("force_fit", {}).get("self_clearance_m", 0.)
    parent_hash = sha256(reference)
    if metadata["model_sha256"] != sha256(rig.path) or metadata["reference_sha256"] != parent_hash:
        raise ValueError("Reference or rig hash mismatch")
    if metadata["source"]["license"] != "CC-BY-4.0":
        raise ValueError("Expected the vetted CC-BY-4.0 source")
    if out.exists():
        raise FileExistsError(out)
    with np.load(reference, allow_pickle=False) as source:
        poses, targets, contact, dt = source["qpos"], source["foot_targets"], source["contact"], float(source["dt"])
        original_velocities = source["qvel"]
        source_soles = source["sole_contact"] if "sole_contact" in source else None
    if source_soles is not None:
        toe_off = True
    baseline_checks, baseline_metrics = validate_motion(rig, poses, original_velocities, targets, contact, dt, source_soles, self_clearance_m)
    progress = []

    def check_progress(candidate, candidate_targets, candidate_contact, evaluations, candidate_soles):
        velocities = reference_velocities(rig.model, candidate, dt)
        checks, metrics = validate_motion(rig, candidate, velocities, candidate_targets, candidate_contact, dt, candidate_soles, self_clearance_m)
        row = dict(evaluations=evaluations, checks=checks,
                   root_failures=len(metrics["root_wrench_consistency"]["infeasible_frames"]),
                   pd_failures=len(metrics["action_window_consistency"]["failing_frames"]["0.25"]),
                   max_stance_slide_m_s=metrics["max_stance_slide_m_s"])
        progress.append(row)
        print(json.dumps(dict(validation_progress=row)), flush=True)
        return all(checks.values())

    result, targets, contact, fit = refine_support(rig, poses, targets, contact, dt, max_evaluations, linear_iterations,
                                                 progress_callback=check_progress,
                                                 toe_off=toe_off, source_soles=source_soles,
                                                 self_clearance_m=self_clearance_m,
                                                 adjust_landings=adjust_landings,
                                                 linear_solver=linear_solver,
                                                 landing_seed=metadata.get("force_fit", {}).get("stance_anchor_offsets_xy"),
                                                 root_seed_drop_m=0. if metadata.get("force_fit", {}).get("stance_method") else .02)
    fit["validation_progress"] = progress
    velocities = reference_velocities(rig.model, result, dt)
    soles_active = np.asarray(fit["sole_contact"], dtype=bool) if toe_off else None
    checks, metrics = validate_motion(rig, result, velocities, targets, contact, dt, soles_active, self_clearance_m)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "step_reference.npz"
    phases = dict(sole_contact=soles_active) if toe_off else {}
    np.savez_compressed(path, qpos=result, qvel=velocities, dt=dt, contact=contact, foot_targets=targets, **phases)
    metadata.update(schema=TOE_REFERENCE_SCHEMA if toe_off else REFERENCE_SCHEMA, passed=all(checks.values()), checks=checks, metrics=metrics,
                    force_fit=fit, parent_reference_sha256=parent_hash, reference_sha256=sha256(path),
                    baseline_checks=baseline_checks, baseline_metrics=baseline_metrics)
    metadata["changes"] += " Joint support/actuator refinement inside delayed target-PD bounds, bounded stance-leg IK, pelvis rotation corrections, inner friction pyramid and inset support points; central tangent velocities."
    if fit["stance_anchor_offsets_xy"] is not None:
        metadata["changes"] += " Phase-constant landing corrections bounded to 2 cm per axis; validation targets remain unchanged."
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(attribution(metadata), encoding="utf-8")
    print(json.dumps(dict(passed=metadata["passed"], checks=checks, metrics=metrics, force_fit=fit)), flush=True)
    return metadata["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--linear-iterations", type=int, default=1000)
    parser.add_argument("--linear-solver", choices=("lsmr", "exact"), default="lsmr",
                        help="Sparse iterative solve by default; exact dense solve is intended for short diagnostics")
    parser.add_argument("--toe-off", action="store_true", help="Fit explicit toe-only phases at heel-up support gaps (v6)")
    parser.add_argument("--self-clearance", type=float, default=None, help="Minimum self-collision clearance in metres; floor support is exempt")
    parser.add_argument("--adjust-landings", action="store_true", help="Allow a fixed XY offset per stance, bounded to 2 cm per axis")
    args = parser.parse_args()
    if min(args.max_evaluations, args.linear_iterations) < 1:
        parser.error("Solver iteration budgets must be positive")
    raise SystemExit(0 if refine_file(args.reference, args.out, args.max_evaluations, args.linear_iterations, args.toe_off, args.self_clearance, args.adjust_landings, args.linear_solver) else 1)

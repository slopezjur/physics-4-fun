"""Contact-aware walking priors for step-tracking experiments, not impact recordings."""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .acquire_motion import STEP_FILES, verify_source
from .baseline import sha256
from .bvh import BVH
from .model_contract import collision_pairs
from .collision_clearance import CollisionClearance
from .retarget_stand import SEGMENTS, SOURCE_TO_DUMMY
from .rig import Rig
from .motion_protocol import REFERENCE_SCHEMA
from .motion_dynamics import force_consistency, root_wrench_consistency, action_window_consistency
from .force_retarget import refine_support, reference_velocities
from .reference_limits import FOOT_TARGET_ERROR_M, SOLE_HEIGHT_ERROR_M, STANCE_SLIDE_M_S, PENETRATION_M

# Fixed before tracking evaluation. Both excerpts contain real forward/backward steps.
CLIPS = {"forward": ("Neutral_FW.bvh", 480, 720), "backward": ("Neutral_BW.bvh", 520, 760)}


def attribution(report):
    source = report["source"]
    return (f"# Motion reference attribution\n\n{source['credit']}.\n\n"
            f"Source: {source['source']}\n\nLicense: {source['license']}\n{source['license_url']}\n\n"
            f"Adapted {report['source_file']}, frames [{report['source_frame_start']}, "
            f"{report['source_frame_stop_exclusive']}), stride {report['source_stride']}.\n\n"
            f"Changes: {report['changes']}\n\n{report['scope']}\n\n"
            "Retain credit, source, license and indication of changes when redistributing. No endorsement is implied.\n")


def lock_stance(positions, contact):
    """Keep each inferred stance interval at its landing position; retain swing frames."""
    result = positions.copy()
    for foot in range(2):
        anchor = None
        for frame in range(len(result)):
            if contact[frame, foot]:
                if anchor is None:
                    anchor = result[frame, foot].copy()
                result[frame, foot] = anchor
            else:
                anchor = None
    return result


def source_targets(bvh, start, stop, rig):
    positions, rotations = bvh.world_poses(start, stop)
    positions, rotations = positions[::2], rotations[::2]
    mapped = SOURCE_TO_DUMMY @ rotations @ SOURCE_TO_DUMMY.T
    forward = mapped[0, 0, :, 0]
    heading = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0])).as_matrix()
    positions = positions @ (heading @ SOURCE_TO_DUMMY).T
    mapped = heading @ mapped
    dt = bvh.dt * 2
    ankles = [bvh.names.index(n) for n in ("LeftAnkle", "RightAnkle")]
    toes = [bvh.names.index(n) for n in ("LeftToe", "RightToe")]
    feet = gaussian_filter1d(positions[:, ankles], 1., axis=0)
    height = feet[..., 2] - np.quantile(feet[..., 2], .05, axis=0)
    speed = np.linalg.norm(np.gradient(feet[..., :2], dt, axis=0), axis=-1)
    contact = (height < .035) & (speed < .35)
    data = mujoco.MjData(rig.model)
    data.qpos[:] = rig.rest
    mujoco.mj_kinematics(rig.model, data)
    body_ids = [rig.model.body(n).id for n in ("Foot_L", "Foot_R")]
    root = (positions[:, 0] - positions[0, 0]) * .9 + rig.rest[:3]
    feet[..., :2] = (feet[..., :2] - positions[0, 0, :2]) * .9 + rig.rest[:2]
    feet[..., 2] = np.maximum(height, 0) * .9 + data.xpos[body_ids, 2]
    for foot in range(2):
        feet[contact[:, foot], foot, 2] = data.xpos[body_ids[foot], 2]
    foot_rotations = []
    for i in range(2):
        vector = positions[:, toes[i]] - positions[:, ankles[i]]
        yaw = np.unwrap(np.arctan2(vector[:, 1], vector[:, 0]))
        pitch = -np.arctan2(vector[:, 2], np.linalg.norm(vector[:, :2], axis=-1))
        if not contact[:, i].any():
            raise ValueError("Source has no inferred stance interval")
        pitch -= np.median(pitch[contact[:, i]])
        pitch[contact[:, i]] = 0
        pitch = gaussian_filter1d(pitch, 1.)
        pitch[contact[:, i]] = 0
        foot_rotations.append(Rotation.from_euler("ZYX", np.stack((yaw, pitch, np.zeros_like(yaw)), -1)).as_matrix())
    return positions, mapped, root, lock_stance(feet, contact), lock_stance(np.stack(foot_rotations, 1), contact), contact, dt


def sole_gaps(model, data):
    return np.array([min(mujoco.mj_geomDistance(model, data, 0, model.geom("g_" + name).id, .3, None)
                         for name in names) for names in (("Foot_L", "Toe_L"), ("Foot_R", "Toe_R"))])


def ground_walking_frames(rig, qpos):
    """Walking has no flight phase: project the lowest sole onto the floor each frame."""
    result = qpos.copy()
    data = mujoco.MjData(rig.model)
    for pose in result:
        data.qpos[:] = pose
        mujoco.mj_kinematics(rig.model, data)
        pose[2] -= sole_gaps(rig.model, data).min()
    return result


def refine_trajectory(rig, qpos, root, feet_target, feet_rot, contact, dt):
    """Fit all frames together so contact IK cannot hide impulsive root/joint jumps."""
    model, data = rig.model, mujoco.MjData(rig.model)
    joints = model.actuator_trnid[rig.actuators, 0]
    indices = model.jnt_qposadr[joints]
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    pairs = sorted(collision_pairs(model))
    initial = np.column_stack((qpos[:, :3], qpos[:, indices]))
    frames, width = initial.shape
    frame_width = 3 + 30 + 6 + 18 + len(pairs)

    def residual(flat):
        trajectory = flat.reshape(frames, width)
        parts = []
        for frame, x in enumerate(trajectory):
            data.qpos[:] = qpos[frame]
            data.qpos[:3], data.qpos[indices] = x[:3], x[3:]
            mujoco.mj_kinematics(model, data)
            distances = np.array([mujoco.mj_geomDistance(model, data, a, b, .01, None) for a, b in pairs])
            parts.extend((8 * (x[:3] - root[frame]), 2 * (x[3:] - initial[frame, 3:]),
                          (np.where(contact[frame], 500., 120.)[:, None] *
                           (data.xpos[feet] - feet_target[frame])).ravel(),
                          (8 * (data.xmat[feet].reshape(2, 3, 3) - feet_rot[frame])).ravel(),
                          1000 * np.minimum(distances, 0)))
        # Penalize acceleration, not velocity: sustained walking remains unconstrained.
        acceleration = np.diff(trajectory, n=2, axis=0) / dt**2
        parts.append((acceleration * np.r_[np.full(3, .3), np.full(30, .02)]).ravel())
        return np.concatenate(parts)

    sparsity = lil_matrix((frames * frame_width + (frames - 2) * width, frames * width), dtype=int)
    for frame in range(frames):
        sparsity[frame * frame_width:(frame + 1) * frame_width, frame * width:(frame + 1) * width] = 1
    for frame in range(frames - 2):
        for coordinate in range(width):
            row = frames * frame_width + frame * width + coordinate
            sparsity[row, np.arange(frame, frame + 3) * width + coordinate] = 1
    lower = np.column_stack((root - [.15, .15, .15], np.tile(model.jnt_range[joints, 0] + 1e-5, (frames, 1))))
    upper = np.column_stack((root + [.15, .15, .1], np.tile(model.jnt_range[joints, 1] - 1e-5, (frames, 1))))
    fitted = least_squares(residual, initial.clip(lower, upper).ravel(), bounds=(lower.ravel(), upper.ravel()),
                           jac_sparsity=sparsity.tocsr(), max_nfev=50, ftol=1e-4, verbose=1)
    result = qpos.copy()
    solution = fitted.x.reshape(frames, width)
    result[:, :3], result[:, indices] = solution[:, :3], solution[:, 3:]
    return result, dict(converged=bool(fitted.success), evaluations=int(fitted.nfev),
                        cost=float(fitted.cost), optimality=float(fitted.optimality),
                        root_acceleration_weight=.3, joint_acceleration_weight=.02,
                        solver_message=fitted.message)


def validate_motion(rig, qpos, qvel, foot_targets, contact, dt, sole_contact=None, self_clearance_m=0.):
    from .motion_contacts import validate_sole_schedule, SOLE_NAMES
    model, data = rig.model, mujoco.MjData(rig.model)
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    support_bodies = [model.body(n).id for n in SOLE_NAMES]
    support_positions, support_corners = [], []
    self_pairs = [pair for pair in sorted(collision_pairs(model)) if 0 not in pair]
    distance_query = CollisionClearance(model, self_pairs)
    min_clearance, closest_pair, closest_frame = .05, None, None
    if sole_contact is not None:
        validate_sole_schedule(contact, sole_contact)
    positions, gaps, penetration = [], [], 0.
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        if self_clearance_m > 0:
            distances = distance_query.distances(data)
            if len(distances) and distances.min() < min_clearance:
                i = int(distances.argmin())
                min_clearance, closest_pair, closest_frame = float(distances[i]), self_pairs[i], frame
        positions.append(data.xpos[feet].copy())
        if sole_contact is not None:
            support_positions.append(data.xpos[support_bodies].copy())
            corners = []
            for name in SOLE_NAMES:
                g = model.geom("g_" + name).id
                local = np.array([[x, y, -1.] for x in (-1., 1.) for y in (-1., 1.)]) * model.geom_size[g]
                corners.append((local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2])
            support_corners.append(corners)
        gaps.append(sole_gaps(model, data))
        penetration = max(penetration, max((-float(c.dist) for c in data.contact[:data.ncon]), default=0.))
    positions = np.asarray(positions)
    gaps = np.asarray(gaps)
    stance_pairs = contact[1:] & contact[:-1]
    speed = np.linalg.norm(np.diff(positions[..., :2], axis=0), axis=-1) / dt
    slide = float(speed[stance_pairs].max()) if stance_pairs.any() else float("inf")
    if sole_contact is not None:
        supported_pairs = sole_contact[1:] & sole_contact[:-1]
        support_speed = np.linalg.norm(np.diff(np.asarray(support_positions)[..., :2], axis=0), axis=-1) / dt
        slide = float(support_speed[supported_pairs].max()) if supported_pairs.any() else float("inf")
    fit = float(np.linalg.norm(positions - foot_targets, axis=-1).max())
    lift = np.ptp(positions[..., 2], axis=0)
    checks = dict(finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
                  joint_limits=bool(((qpos[:, 7:] >= model.jnt_range[1:, 0] - 1e-6) &
                                     (qpos[:, 7:] <= model.jnt_range[1:, 1] + 1e-6)).all()),
                  penetration=penetration < PENETRATION_M, stance_slide=slide < STANCE_SLIDE_M_S,
                  ground_support=bool((gaps.min(axis=1) <= SOLE_HEIGHT_ERROR_M).all()),
                  stance_contact=bool(contact.any() and (gaps[contact] <= SOLE_HEIGHT_ERROR_M).all()),
                  foot_tracking=fit < FOOT_TARGET_ERROR_M, foot_lift=bool((lift > .04).all()),
                  joint_speed=bool(np.abs(qvel[:, 6:]).max() < 12.))
    if sole_contact is not None:
        checks["sole_contact"] = bool(sole_contact.any(axis=1).all() and
                                     (np.abs(np.asarray(support_corners)[sole_contact]) <= SOLE_HEIGHT_ERROR_M).all())
    if self_clearance_m > 0:
        checks["self_clearance"] = min_clearance >= self_clearance_m - 1e-5
    dynamics = force_consistency(rig, qpos, dt)
    checks["force_consistency"] = dynamics["passed"]
    wrench = root_wrench_consistency(rig, qpos, dt, sole_contact)
    checks["root_wrench_consistency"] = wrench["all_frames_feasible"]
    actuation = action_window_consistency(rig, qpos, qvel, dt, sole_contact=sole_contact)
    checks["actuation_feasibility"] = bool(wrench["all_frames_feasible"] and actuation["root_supported_frames"]
                                           and not actuation["failing_frames"]["0.25"])
    clearance_metrics = (dict(minimum_m=min_clearance, required_m=self_clearance_m,
                              closest_pair=[model.geom(g).name for g in closest_pair] if closest_pair else None,
                              closest_frame=closest_frame) if self_clearance_m > 0 else None)
    return checks, dict(force_consistency=dynamics, root_wrench_consistency=wrench, self_clearance=clearance_metrics,
                        action_window_consistency=actuation,
                        max_penetration_m=penetration, max_stance_slide_m_s=slide,
                        max_support_gap_m=float(gaps.min(axis=1).max()),
                        max_stance_gap_m=float(gaps[contact].max()) if contact.any() else None,
                        max_foot_target_error_m=fit, foot_lift_range_m=lift.tolist(),
                        max_joint_speed_rad_s=float(np.abs(qvel[:, 6:]).max()),
                        root_displacement_m=(qpos[-1, :3] - qpos[0, :3]).tolist())


def retarget(source, rig, out, direction, force_evaluations=80):
    if out.exists():
        raise FileExistsError(out)
    verify_source(source, STEP_FILES)
    filename, start, stop = CLIPS[direction]
    with (source / "Frame_Cuts.csv").open(newline="") as file:
        cuts = next(row for row in csv.DictReader(file) if row["STYLE_NAME"] == "Neutral")
    kind = filename.removeprefix("Neutral_").removesuffix(".bvh")
    if not int(cuts[f"{kind}_START"]) <= start < stop <= int(cuts[f"{kind}_STOP"]):
        raise ValueError("Step excerpt exceeds the published frame cuts")
    bvh = BVH.load(source / filename)
    positions, rotations, root, feet_target, feet_rot, contact, dt = source_targets(bvh, start, stop, rig)
    model, data = rig.model, mujoco.MjData(rig.model)
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    anchor = lambda name: model.body_jntadr[model.body(name).id]
    segments = [(anchor(a), anchor(b), bvh.names.index(c), bvh.names.index(d)) for a, b, c, d in SEGMENTS]
    torso = [(model.body("Spine").id, bvh.names.index("Chest2")), (model.body("Chest").id, bvh.names.index("Chest4"))]
    joints = model.actuator_trnid[rig.actuators, 0]
    indices = model.jnt_qposadr[joints]
    toe_indices = [model.jnt_qposadr[model.joint(n).id] for n in ("Toe_L_rx", "Toe_R_rx")]
    pairs = sorted(collision_pairs(model))
    q = rig.rest.copy()
    previous = np.r_[root[0], q[indices]]
    result = []
    for frame in range(len(root)):
        q[3:7] = Rotation.from_matrix(rotations[frame, 0]).as_quat()[[3, 0, 1, 2]]
        directions = [positions[frame, b] - positions[frame, a] for _, _, a, b in segments]
        directions = [v / np.linalg.norm(v) for v in directions]

        def residual(x):
            q[:3], q[indices] = x[:3], x[3:]
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            parts = [8 * (q[:3] - root[frame]), .3 * (x[3:] - previous[3:]), 4 * q[toe_indices]]
            for (a, b, _, _), desired in zip(segments, directions):
                v = data.xanchor[b] - data.xanchor[a]
                parts.append(2 * (v / np.linalg.norm(v) - desired))
            for body, source_body in torso:
                parts.append((data.xmat[body].reshape(3, 3) - rotations[frame, source_body]).ravel())
            foot_weight = np.where(contact[frame], 500., 120.)[:, None]
            parts += [(foot_weight * (data.xpos[feet] - feet_target[frame])).ravel(),
                      8 * (data.xmat[feet].reshape(2, 3, 3) - feet_rot[frame]).ravel()]
            distances = np.array([mujoco.mj_geomDistance(model, data, a, b, .01, None) for a, b in pairs])
            parts.append(1000 * np.minimum(distances, 0))
            return np.concatenate(parts)

        low = np.r_[root[frame] - [.15, .15, .15], model.jnt_range[joints, 0] + 1e-5]
        high = np.r_[root[frame] + [.15, .15, .1], model.jnt_range[joints, 1] - 1e-5]
        if frame:
            # Prevent frame-to-frame IK branch jumps without changing physical joint limits.
            low[3:] = np.maximum(low[3:], previous[3:] - 10 * dt)
            high[3:] = np.minimum(high[3:], previous[3:] + 10 * dt)
        fitted = least_squares(residual, previous.clip(low, high), bounds=(low, high), max_nfev=40)
        residual(fitted.x)
        result.append(q.copy())
        previous = fitted.x.copy()
    qpos, trajectory_fit = refine_trajectory(rig, np.array(result), root, feet_target, feet_rot, contact, dt)
    qpos = ground_walking_frames(rig, qpos)
    qpos, feet_target, contact, force_fit = refine_support(rig, qpos, feet_target, contact, dt, force_evaluations)
    for i in range(1, len(qpos)):
        if qpos[i, 3:7] @ qpos[i-1, 3:7] < 0:
            qpos[i, 3:7] *= -1
    qvel = reference_velocities(model, qpos, dt)
    checks, metrics = validate_motion(rig, qpos, qvel, feet_target, contact, dt)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "step_reference.npz"
    np.savez_compressed(path, qpos=qpos, qvel=qvel, dt=dt, contact=contact, foot_targets=feet_target)
    report = dict(schema=REFERENCE_SCHEMA, passed=all(checks.values()), checks=checks, metrics=metrics,
                  trajectory_fit=trajectory_fit, force_fit=force_fit,
                  source=json.loads((source / "provenance.json").read_text()), source_file=filename,
                  source_frame_start=start, source_frame_stop_exclusive=stop, source_stride=2,
                  model_sha256=sha256(rig.path), reference_sha256=sha256(path), frames=len(qpos), dt=dt,
                  direction=direction, changes="Cropped/resampled, semantic-axis retargeting, scale 0.9, grounded inferred stance locks, smoothed foot pitch, moving-foot collision-aware IK followed by acceleration regularization and joint contact/actuator refinement inside delayed target-PD bounds. Bounded stance-leg IK, pelvis rotation corrections, unilateral inner-pyramid friction and inset support points are included in every force-fit evaluation. Central tangent velocities. No rig or motor changes.",
                  scope="Walking stepping prior; not a recorded impact recovery. Kinematic checks do not establish dynamic tracking.")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(attribution(report), encoding="utf-8")
    print(json.dumps(dict(direction=direction, passed=report["passed"], checks=checks, metrics=metrics)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force-evaluations", type=int, default=80)
    args = parser.parse_args()
    if args.force_evaluations < 1:
        parser.error("--force-evaluations must be positive")
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reports = [retarget(args.source, rig, args.out / direction, direction, args.force_evaluations) for direction in CLIPS]
    raise SystemExit(0 if all(r["passed"] for r in reports) else 1)

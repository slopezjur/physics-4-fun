"""Retarget a bounded neutral-idle excerpt with fixed-foot inverse kinematics."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .baseline import sha256
from .acquire_motion import verify_source
from .bvh import BVH
from .rig import Rig
from .model_contract import collision_pairs

# Match semantic left/up/forward, including the source/target handedness difference.
SOURCE_TO_DUMMY = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
SEGMENTS = [
    ("UpperArm_L", "Forearm_L", "LeftShoulder", "LeftElbow"),
    ("Forearm_L", "Hand_L", "LeftElbow", "LeftWrist"),
    ("UpperArm_R", "Forearm_R", "RightShoulder", "RightElbow"),
    ("Forearm_R", "Hand_R", "RightElbow", "RightWrist"),
    ("Thigh_L", "Shin_L", "LeftHip", "LeftKnee"),
    ("Shin_L", "Foot_L", "LeftKnee", "LeftAnkle"),
    ("Thigh_R", "Shin_R", "RightHip", "RightKnee"),
    ("Shin_R", "Foot_R", "RightKnee", "RightAnkle"),
]


def retarget(source: Path, rig: Rig, out: Path, start=700, frames=360, stride=2):
    verify_source(source)
    bvh = BVH.load(source / "Neutral_ID.bvh")
    # Published neutral-idle trim is [655, 1555]; use an interior six-second excerpt.
    if start < 655 or start + frames > 1555:
        raise ValueError("Excerpt must stay within the published neutral idle interval")
    positions, rotations = bvh.world_poses(start, start + frames)
    positions, rotations = positions[::stride], rotations[::stride]
    dt = bvh.dt * stride
    model, data = rig.model, mujoco.MjData(rig.model)
    data.qpos[:] = rig.rest
    mujoco.mj_kinematics(model, data)
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    fixed_feet = data.xpos[feet].copy()
    nearby_pairs = [(a, b) for a, b in sorted(collision_pairs(model)) if a != 0 and b != 0
                    and mujoco.mj_geomDistance(model, data, a, b, 0.2, None) < 0.2]
    anchor = lambda name: model.body_jntadr[model.body(name).id]
    segments = [(anchor(a), anchor(b), bvh.names.index(c), bvh.names.index(d)) for a, b, c, d in SEGMENTS]
    torso = [(model.body("Spine").id, bvh.names.index("Chest2")),
             (model.body("Chest").id, bvh.names.index("Chest4"))]
    indices = model.jnt_qposadr[model.actuator_trnid[rig.actuators, 0]]
    joints = model.actuator_trnid[rig.actuators, 0]
    rotation_mapped = SOURCE_TO_DUMMY @ rotations @ SOURCE_TO_DUMMY.T
    forward = rotation_mapped[0, 0, :, 0]
    heading = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0])).as_matrix()
    world_map = heading @ SOURCE_TO_DUMMY
    target_rot = heading @ rotation_mapped
    target_pos = positions @ world_map.T
    root_target = (target_pos[:, 0] - target_pos[0, 0]) * 0.9 + rig.rest[:3]
    q = rig.rest.copy()
    previous = np.concatenate([q[:3], q[indices]])
    low = np.concatenate([rig.rest[:3] + [-0.12, -0.12, -0.12], model.jnt_range[joints, 0] + 1e-5])
    high = np.concatenate([rig.rest[:3] + [0.12, 0.12, 0.08], model.jnt_range[joints, 1] - 1e-5])
    result, fit_errors = [], []
    for f in range(len(positions)):
        q[3:7] = Rotation.from_matrix(target_rot[f, 0]).as_quat()[[3, 0, 1, 2]]
        desired_directions = []
        for _, _, a, b in segments:
            vec = target_pos[f, b] - target_pos[f, a]
            desired_directions.append(vec / np.linalg.norm(vec))

        def residual(x):
            q[:3], q[indices] = x[:3], x[3:]
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            parts = [2 * (q[:3] - root_target[f]), 0.15 * (x[3:] - previous[3:]),
                     0.05 * (x[3:] - rig.rest[indices])]
            for (a, b, _, _), target in zip(segments, desired_directions):
                vec = data.xanchor[b] - data.xanchor[a]
                parts.append(2 * (vec / np.linalg.norm(vec) - target))
            for body, source_body in torso:
                parts.append((data.xmat[body].reshape(3, 3) - target_rot[f, source_body]).ravel())
            parts.append(80 * (data.xpos[feet] - fixed_feet).ravel())
            parts.append(12 * (data.xmat[feet].reshape(2, 3, 3) - np.eye(3)).ravel())
            distances = np.array([mujoco.mj_geomDistance(model, data, a, b, 0.05, None)
                                  for a, b in nearby_pairs])
            parts.append(80 * np.minimum(distances - 0.002, 0))
            return np.concatenate(parts)

        fitted = least_squares(residual, previous.clip(low, high), bounds=(low, high),
                               max_nfev=35, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        residual(fitted.x)
        result.append(q.copy())
        fit_errors.append(float(np.linalg.norm(fitted.fun)))
        previous = fitted.x.copy()
    qpos = np.asarray(result)
    qpos = gaussian_filter1d(qpos, sigma=1.2, axis=0, mode="nearest")
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qvel = np.zeros((len(qpos), model.nv))
    for i in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i], qpos[i + 1])
    qvel[-1] = qvel[-2]
    max_penetration = max_foot_error = max_slide_speed = 0.0
    prev_foot = None
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        max_foot_error = max(max_foot_error, float(np.max(np.abs(data.xpos[feet] - fixed_feet))))
        if prev_foot is not None:
            max_slide_speed = max(max_slide_speed, float(np.max(np.linalg.norm(data.xpos[feet, :2] - prev_foot, axis=1))) / dt)
        prev_foot = data.xpos[feet, :2].copy()
        for contact in data.contact[:data.ncon]:
            max_penetration = max(max_penetration, float(-contact.dist))
    checks = {"finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
              "joint_limits": bool(((qpos[:, 7:] >= model.jnt_range[1:, 0] - 1e-6)
                                     & (qpos[:, 7:] <= model.jnt_range[1:, 1] + 1e-6)).all()),
              "foot_position": max_foot_error < 0.005, "foot_slide_speed": max_slide_speed < 0.03,
              "penetration": max_penetration < 0.005,
              "joint_speed": bool(np.abs(qvel[:, 6:]).max() < 2.0)}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "stand_reference.npz", qpos=qpos, qvel=qvel, dt=dt)
    provenance = json.loads((source / "provenance.json").read_text())
    report = {"passed": all(checks.values()), "checks": checks, "source": provenance,
              "source_frame_start": start, "source_frame_stop_exclusive": start + frames,
              "source_stride": stride, "frames": len(qpos), "dt": dt,
              "model_sha256": sha256(rig.path), "reference_sha256": sha256(out / "stand_reference.npz"),
              "changes": "Cropped/resampled; semantic-axis remapping; fixed-foot collision-aware IK; Gaussian temporal smoothing (sigma 1.2 frames); neck/wrists retain zero targets; non-looping excerpt.",
              "metrics": {"max_foot_position_error_m": max_foot_error, "max_slide_speed_m_s": max_slide_speed,
                          "max_penetration_m": max_penetration, "max_joint_velocity_rad_s": float(np.abs(qvel[:, 6:]).max()),
                          "mean_fit_residual": float(np.mean(fit_errors))}}
    (out / "stand_reference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = retarget(args.source, Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml"), args.out)
    raise SystemExit(0 if report["passed"] else 1)

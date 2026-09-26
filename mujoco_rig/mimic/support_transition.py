"""Versioned micro-transition from the last held pose to the first failed pose."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .force_retarget import reference_velocities
from .track_transfer import CalibrationReference


SCHEMA = "mimic_prelift_support_gate_v1"
START_SECONDS = 3.25
TARGET_SECONDS = 3.5
DURATION_SECONDS = 6.
MOTION_START_SECONDS = .5
MOTION_END_SECONDS = 1.5
TRACKING_REWARD_SCHEMA = "mimic_support_tracking_v1"
TRACKING_REWARD_WEIGHTS = (0.7, 0.2, 0.1)
TRACKING_POSITION_SCALE_M = 0.08
TRACKING_VELOCITY_SCALE_M_S = 0.15


def support_tracking_reward(base, root_xy, target_xy, velocity_xy, target_velocity_xy):
    """Reward upright imitation while damping horizontal support-pose drift."""
    position_error = (root_xy - target_xy).square().sum(dim=-1)
    velocity_error = (velocity_xy - target_velocity_xy).square().sum(dim=-1)
    position = torch.exp(-position_error / TRACKING_POSITION_SCALE_M ** 2)
    velocity = torch.exp(-velocity_error / TRACKING_VELOCITY_SCALE_M_S ** 2)
    base_weight, position_weight, velocity_weight = TRACKING_REWARD_WEIGHTS
    return base_weight * base + position_weight * position + velocity_weight * velocity


class SupportTransitionReference:
    """Static target or a slow entry into it; never a production motion clip."""
    def __init__(self, path: Path, rig, device="cpu", *, mode="static"):
        if mode not in ("static", "transition"):
            raise ValueError("Unknown support-transition mode")
        source = CalibrationReference(path, rig, "cpu")
        if not source.placed_step or source.requested_step_m <= 0:
            raise ValueError("Require a validated forward placed-step diagnostic")
        self.path = source.path
        self.source_sha256 = source.sha256
        self.source_report_sha256 = source.report_sha256
        self.mode = mode
        self.dt = source.dt
        frames = round(DURATION_SECONDS / self.dt) + 1
        times = np.arange(frames) * self.dt
        if abs(times[-1] - DURATION_SECONDS) > 1e-6:
            raise ValueError("Support gate duration must align to source frames")
        if mode == "static":
            sample_times = np.full(frames, TARGET_SECONDS)
        else:
            phase = np.clip((times - MOTION_START_SECONDS) /
                            (MOTION_END_SECONDS - MOTION_START_SECONDS), 0., 1.)
            ease = phase * phase * (3. - 2. * phase)
            sample_times = START_SECONDS + (TARGET_SECONDS - START_SECONDS) * ease
        q, _ = source.sample(torch.as_tensor(sample_times, dtype=torch.float32))
        poses = q.numpy().astype(np.float64)
        velocity = reference_velocities(rig.model, poses, self.dt)
        self.qpos = torch.tensor(poses, dtype=torch.float32, device=device)
        self.qvel = torch.tensor(velocity, dtype=torch.float32, device=device)
        self.duration = DURATION_SECONDS
        digest = hashlib.sha256()
        digest.update(SCHEMA.encode())
        digest.update(mode.encode())
        digest.update(source.sha256.encode())
        digest.update(poses.tobytes())
        digest.update(velocity.tobytes())
        self.sha256 = digest.hexdigest()

    def sample(self, time):
        frame = (time / self.dt).clamp(0, len(self.qpos) - 1)
        index = frame.long().clamp(max=len(self.qpos) - 2)
        alpha = (frame - index).unsqueeze(-1)
        q = torch.lerp(self.qpos[index], self.qpos[index + 1], alpha)
        q[..., 3:7] = torch.nn.functional.normalize(q[..., 3:7], dim=-1)
        v = torch.lerp(self.qvel[index], self.qvel[index + 1], alpha)
        return q, v

    def save(self, path):
        np.savez_compressed(path, qpos=self.qpos.cpu().numpy(),
                            qvel=self.qvel.cpu().numpy(), dt=self.dt)

    def contract(self):
        return dict(schema=SCHEMA, mode=self.mode,
                    start_pose_seconds=START_SECONDS, target_pose_seconds=TARGET_SECONDS,
                    duration_seconds=self.duration, motion_window_seconds=(
                        MOTION_START_SECONDS, MOTION_END_SECONDS) if self.mode == "transition" else None,
                    source_sha256=self.source_sha256,
                    source_report_sha256=self.source_report_sha256,
                    reference_sha256=self.sha256)

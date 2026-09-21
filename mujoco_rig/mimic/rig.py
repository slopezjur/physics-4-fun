"""The existing MuJoCo model is the authority for the experimental control contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from mujoco_rig.rl.env_config import ACTION_LATENCY_STEPS, POLICY_EXCLUDE, TORQUE_AUTHORITY
from .baseline import sha256


@dataclass
class Rig:
    path: Path
    model: mujoco.MjModel
    actuators: np.ndarray
    rest: np.ndarray
    decimation: int = 4

    @classmethod
    def load(cls, path: Path) -> "Rig":
        model = mujoco.MjModel.from_xml_path(str(path))
        actuators = np.array([i for i in range(model.nu)
                              if model.actuator(i).name.split("_")[0] not in POLICY_EXCLUDE])
        if not ((model.actuator_gaintype[actuators] == mujoco.mjtGain.mjGAIN_FIXED).all()
                and (model.actuator_biastype[actuators] == mujoco.mjtBias.mjBIAS_NONE).all()
                and np.allclose(model.actuator_gear[actuators, 0], 1.0)):
            raise ValueError("The compatibility path requires the existing torque actuators")
        rest = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest")
        if key < 0:
            raise ValueError("The source rig must provide its validated rest keyframe")
        mujoco.mj_resetDataKeyframe(model, rest, key)
        return cls(path.resolve(), model, actuators, rest.qpos.copy())

    @property
    def action_scale(self) -> np.ndarray:
        return self.model.actuator_forcerange[self.actuators, 1] * TORQUE_AUTHORITY

    def contract(self) -> dict:
        return {
            "schema": "mimickit_compat_v1", "model_sha256": sha256(self.path),
            "action_mode": "normalized_motor_torque", "action_clip": [-1.0, 1.0],
            "actuator_names": [self.model.actuator(int(i)).name for i in self.actuators],
            "actuator_indices": self.actuators.tolist(), "torque_scale_nm": self.action_scale.tolist(),
            "physics_timestep": self.model.opt.timestep, "decimation": self.decimation,
            "control_timestep": self.model.opt.timestep * self.decimation,
            "latency_steps": ACTION_LATENCY_STEPS,
            "non_policy_actuators": [self.model.actuator(i).name for i in range(self.model.nu)
                                      if i not in self.actuators],
            "non_policy_control": "zero position targets, original neck gains and force limits",
            "joint_names": [self.model.joint(i).name for i in range(1, self.model.njnt)],
            "joint_coordinates": "original ordered hinge angles in radians; not ball-joint exponential maps",
            "root_quaternion": "xyzw in MimicKit/Newton; wxyz in native MuJoCo",
            "observation_contract": "not yet defined; no policy export or training in this stage",
        }


class DelayedTorqueControl:
    """Map normalized actions to native actuator controls with the production delay."""

    def __init__(self, rig: Rig, num_envs: int, device: str):
        self.indices = torch.as_tensor(rig.actuators, device=device)
        self.scale = torch.as_tensor(rig.action_scale, dtype=torch.float32, device=device)
        self.ctrl = torch.zeros((num_envs, rig.model.nu), device=device)
        self.pending = torch.zeros((ACTION_LATENCY_STEPS, num_envs, len(rig.actuators)), device=device)
        self.cursor = 0

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.ctrl.zero_()
            self.pending.zero_()
            self.cursor = 0
        else:
            self.ctrl[env_ids] = 0
            self.pending[:, env_ids] = 0

    def apply(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != self.ctrl[:, self.indices].shape:
            raise ValueError(f"Expected {(self.ctrl.shape[0], len(self.indices))} actions, got {action.shape}")
        if not torch.isfinite(action).all():
            raise ValueError("Non-finite policy action")
        bounded = action.clamp(-1, 1)
        self.ctrl.zero_()
        if len(self.pending):
            self.ctrl[:, self.indices] = self.pending[self.cursor] * self.scale
            self.pending[self.cursor].copy_(bounded)
            self.cursor = (self.cursor + 1) % len(self.pending)
        else:
            self.ctrl[:, self.indices] = bounded * self.scale
        return self.ctrl

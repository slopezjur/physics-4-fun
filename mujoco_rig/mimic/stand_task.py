"""Motion-guided Stand using upstream MimicKit's DeepMimic reward and PPO API."""
from pathlib import Path
import numpy as np
import torch
import gymnasium.spaces as spaces

from envs.base_env import BaseEnv, DoneFlags, EnvMode
from envs.deepmimic_env import compute_reward
from util.torch_util import quat_conjugate, quat_mul, quat_rotate, quat_to_tan_norm

from .kinematics import DummyKinematics
from .newton_adapter import DummyNewtonEngine
from .reference import StandReference
from .rig import Rig
from .target_control import DelayedTargetControl


class StandTask(BaseEnv):
    def __init__(self, rig: Rig, reference: Path, num_envs=128, device="cuda:0", seed=210921,
                 engine_factory=DummyNewtonEngine, control_mode="torque"):
        super().__init__(False)
        self.rig, self.device = rig, device
        self.reference = StandReference(reference, rig, device)
        if control_mode not in ("torque", "target_pd"):
            raise ValueError(f"Unknown control mode: {control_mode}")
        self.target_pd = control_mode == "target_pd"
        if self.target_pd:
            self.engine = engine_factory(rig, num_envs, device, control_factory=DelayedTargetControl)
        else:
            self.engine = engine_factory(rig, num_envs, device)
        self.mapper = DummyKinematics(rig, device)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.dt = self.engine.get_timestep()
        self.episode_seconds = 3.0
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.offset = torch.zeros(num_envs, device=device)
        self.ids = torch.arange(num_envs, device=device)
        self.parents = torch.as_tensor(rig.model.body_parentid[2:] - 1, device=device)
        self.key_bodies = torch.tensor([rig.model.body(n).id - 1 for n in
                                       ("Head", "Hand_L", "Hand_R", "Foot_L", "Foot_R")], device=device)
        self.feet = torch.tensor([rig.model.body(n).id - 1 for n in ("Foot_L", "Foot_R")], device=device)
        self.allowed_ground = torch.tensor([rig.model.body(n).id - 1 for n in
                                            ("Foot_L", "Foot_R", "Toe_L", "Toe_R")], device=device)
        self.joint_weights = torch.ones(rig.model.nbody - 2, device=device)
        # Neck and wrists are not controlled by this actor; avoid unreachable pose targets.
        for name in ("Head", "Hand_L", "Hand_R"):
            self.joint_weights[rig.model.body(name).id - 2] = 0
        self.dof_weights = torch.zeros(rig.model.nv - 6, device=device)
        for actuator in rig.actuators:
            self.dof_weights[rig.model.jnt_dofadr[rig.model.actuator_trnid[actuator, 0]] - 6] = 1
        self._action_space = spaces.Box(-1.0, 1.0, (len(rig.actuators),), dtype=np.float32)
        self.reset()

    def get_num_envs(self):
        return len(self.steps)

    def get_env_time(self, env_ids=None):
        return self.steps * self.dt if env_ids is None else self.steps[env_ids] * self.dt

    def reset(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        if len(ids):
            self.steps[ids] = 0
            if self._mode == EnvMode.TRAIN:
                self.offset[ids] = torch.rand(len(ids), device=self.device, generator=self.rng) * (self.reference.duration - self.episode_seconds)
            else:
                # Deterministic phase coverage for repeated evaluation.
                self.offset[ids] = ids / max(1, self.get_num_envs() - 1) * (self.reference.duration - self.episode_seconds)
            q, v = self.reference.sample(self.offset[ids])
            self.engine.reset_envs(ids, q, v)
        return self.observations(), {}

    def joint_rotations(self, body_rot):
        return quat_mul(quat_conjugate(body_rot[:, self.parents]), body_rot[:, 1:])

    def observations(self):
        q = self.engine.native_qpos()
        root_rot = self.engine.get_root_rot(0)
        phase = self.offset + self.steps * self.dt
        future_q, _ = self.reference.sample(phase[:, None] + self.dt * torch.tensor([1, 2, 3], device=self.device))
        future = torch.cat([future_q[..., :3], quat_to_tan_norm(future_q[..., [4, 5, 6, 3]]), future_q[..., 7:]], dim=-1)
        control = self.engine.policy_control
        pending = torch.cat([control.pending[(control.cursor + i) % len(control.pending)] for i in range(len(control.pending))], dim=-1)
        loads = self.engine.get_ground_contact_forces(0)[:, self.feet, 2].abs() * 0.001
        return torch.cat([q[:, :3], quat_to_tan_norm(root_rot), self.engine.get_root_vel(0),
                          self.engine.get_root_ang_vel(0), q[:, 7:], self.engine.get_dof_vel(0) * 0.1,
                          loads, pending, future.flatten(1), (phase / self.reference.duration)[:, None]], dim=-1)

    def reward(self):
        q, v = self.reference.sample(self.offset + self.steps * self.dt)
        target_pos, target_rot = self.mapper.from_native_qpos(q)
        body_pos, body_rot = self.engine.get_body_pos(0), self.engine.get_body_rot(0)
        result = compute_reward(
            self.engine.get_root_pos(0), self.engine.get_root_rot(0), self.engine.get_root_vel(0),
            self.engine.get_root_ang_vel(0), self.joint_rotations(body_rot), self.engine.get_dof_vel(0),
            body_pos[:, self.key_bodies], q[:, :3], q[:, [4, 5, 6, 3]], v[:, :3],
            quat_rotate(q[:, [4, 5, 6, 3]], v[:, 3:6]), self.joint_rotations(target_rot), v[:, 6:],
            target_pos[:, self.key_bodies], self.joint_weights, self.dof_weights, True, True,
            0.5, 0.1, 0.15, 0.1, 0.15, 0.25, 0.01, 5.0, 1.0, 10.0)
        self._diagnostics["root_tracking_error_m"] = torch.linalg.vector_norm(body_pos[:, 0] - q[:, :3], dim=-1).mean()
        return result

    def step(self, action):
        if self.target_pd:
            self.engine.policy_control.set_reference(*self.reference.sample(self.offset + self.steps * self.dt))
        self.engine.set_cmd(0, action)
        self.engine.step()
        self.steps += 1
        obs = self.observations()
        reward = self.reward()
        done = torch.zeros_like(self.steps, dtype=torch.int)
        done[self.steps * self.dt >= self.episode_seconds] = DoneFlags.TIME.value
        contact = self.engine.get_ground_contact_forces(0).clone()
        contact[:, self.allowed_ground] = 0
        fallen = self.engine.get_root_pos(0)[:, 2] < 0.65
        fallen |= contact.abs().amax(dim=(1, 2)) > 1.0
        finite = torch.isfinite(obs).all(dim=1) & torch.isfinite(reward)
        done[fallen | ~finite] = DoneFlags.FAIL.value
        reward[fallen | ~finite] = 0
        # Failed observations still pass through the agent's terminal buffer.
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4), reward, done, {}

    def observation_contract(self):
        groups = [("root_position_world", 3), ("root_rotation_tangent_normal", 6),
                  ("root_linear_velocity_world", 3), ("root_angular_velocity_world", 3),
                  ("all_hinge_positions", 39), ("all_hinge_velocities_scaled_0.1", 39),
                  ("foot_normal_load_kN_L_R", 2), ("pending_actions_oldest_first", 60),
                  ("future_reference_poses_1_2_3", 144), ("reference_phase", 1)]
        offset, layout = 0, []
        if self.target_pd:
            groups[7] = ("pending_targets_q_v_valid_oldest_first", 122)
        for name, width in groups:
            layout.append({"name": name, "offset": offset, "width": width})
            offset += width
        control_contract = self.engine.policy_control.contract() if self.target_pd else {}
        return {**self.rig.contract(), **control_contract,
                "observation_contract": "mimic_stand_target_pd_v1" if self.target_pd else "mimic_stand_v1", "num_obs": offset,
                "observation_layout": layout, "reference_sha256": self.reference.sha256,
                "reference_duration": self.reference.duration,
                "deployment_status": "Experimental contract; requires a matching Godot observation/reference adapter"}

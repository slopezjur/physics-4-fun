"""Motion-guided recovery under real projectile collisions, with unchanged actor inputs."""
import math
import numpy as np
import torch

from envs.base_env import EnvMode
from .ball import launch_state
from .ball_native import BallNativeEngine
from .ball_newton import BallNewtonEngine
from .stand_task import StandTask
from .ball_protocol import TARGET_BODIES, validation_cases
from .recovery_reward import RecoveryReward


class BallTask(StandTask):
    def __init__(self, rig, reference, num_envs=128, device="cuda:0", seed=210921,
                 engine_factory=None, control_mode="target_pd", speed_min=1., speed_max=2., quiet_fraction=.25,
                 reward_mode="reference", contact_mode="legacy"):
        if reward_mode not in ("reference", "recovery_v1"):
            raise ValueError(f"Unknown ball reward: {reward_mode}")
        self.reward_mode = reward_mode
        self.recovery_reward = RecoveryReward()
        if control_mode != "target_pd" or not 1 <= speed_min <= speed_max <= 8 or not 0 <= quiet_fraction <= 1:
            raise ValueError("Ball task requires target PD, speeds 1–8 m/s and a valid quiet fraction")
        # Keep the sampled schedule in the same dtype as the float shot-speed buffer.  The
        # CLI passes floats, but callers and deterministic probes commonly pass integer
        # literals; torch.where preserves that integer dtype and then rejects the assignment.
        self.speed_min, self.speed_max, self.quiet_fraction = (
            float(speed_min), float(speed_max), float(quiet_fraction))
        self.target_bodies = TARGET_BODIES
        self.validation_cases = validation_cases(self.speed_min, self.speed_max)
        self.target_indices = torch.tensor(
            [rig.model.body(name).id - 1 for name in self.target_bodies],
            device=device, dtype=torch.long
        )
        super().__init__(rig, reference, num_envs, device, seed,
                         engine_factory or (BallNativeEngine if device == "cpu" else BallNewtonEngine), control_mode, contact_mode)
        self.episode_seconds = 5.
        self.shot_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.shot_speeds = torch.zeros(num_envs, device=device)
        self.directions = torch.zeros((num_envs, 3), device=device)
        self.shot_target_body = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.shot_enabled = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.reset()

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        if not hasattr(self, "shot_steps"):
            return result
        ids = self.ids if env_ids is None else env_ids
        if self._mode == EnvMode.TRAIN:
            self.shot_steps[ids] = torch.randint(60, 121, (len(ids),), generator=self.rng, device=self.device)
            self.shot_speeds[ids] = self.speed_min + (self.speed_max - self.speed_min) * torch.rand(len(ids), generator=self.rng, device=self.device)
            angle = 2 * math.pi * torch.rand(len(ids), generator=self.rng, device=self.device)
            target_pick = torch.randint(0, len(self.target_bodies), (len(ids),), generator=self.rng, device=self.device)
            self.shot_target_body[ids] = self.target_indices[target_pick]
            self.shot_enabled[ids] = torch.rand(len(ids), generator=self.rng, device=self.device) >= self.quiet_fraction
        else:
            self.shot_steps[ids] = 60
            # Full Cartesian coverage: bodies vary fastest, then direction, then speed.
            case_ids = ids % len(self.validation_cases)
            level = case_ids // (len(self.target_bodies) * 4)
            self.shot_speeds[ids] = torch.where(level == 0, self.speed_min, self.speed_max)
            angle = ((case_ids // len(self.target_bodies)) % 4) * math.pi / 2
            self.shot_target_body[ids] = self.target_indices[ids % len(self.target_bodies)]
            self.shot_enabled[ids] = True
            self.offset[ids] = level.float() * .8
            self.engine.reset_envs(ids, *self.reference.sample(self.offset[ids]))
        self.directions[ids, 0] = torch.cos(angle)
        self.directions[ids, 1] = torch.sin(angle)
        self.directions[ids, 2] = 0
        # Only TEST overrides the state sampled by the parent reset. Training shot
        # parameters are not actor inputs, so its returned observation is still valid.
        return (self.observations(), {}) if self._mode != EnvMode.TRAIN and len(ids) else result

    def step(self, action):
        ids = torch.nonzero((self.steps == self.shot_steps) & self.shot_enabled).flatten()
        if len(ids):
            target = self.engine.get_body_pos(0)[ids, self.shot_target_body[ids]].cpu().numpy()
            position, velocity = launch_state(target, self.directions[ids].cpu().numpy(),
                                              self.shot_speeds[ids].cpu().numpy(), self.rig.model.opt.gravity)
            self.engine.launch_ball(ids, position, velocity)
        return super().step(action)

    def experiment_contract(self):
        from .baseline import sha256
        world = self.engine.rig.model
        geom = world.geom("g_ball").id
        return {"task": "ball", "world_model_sha256": sha256(self.engine.rig.path),
                "ball_mass_kg": float(world.body_mass[world.body("ball").id]),
                "ball_radius_m": float(world.geom_size[geom, 0]), "ball_friction": world.geom_friction[geom].tolist(),
                "ball_solref": world.geom_solref[geom].tolist(), "gravity": world.opt.gravity.tolist(),
                "horizontal_speed_min": self.speed_min, "horizontal_speed_max": self.speed_max,
                "quiet_fraction": self.quiet_fraction, "launch_distance_m": .65,
                "launch_steps_inclusive": [60, 120], "target_body": "multi_body",
                "target_bodies": list(self.target_bodies),
                "flight": "ballistic; aim at launch-time target; no gravity cancellation",
                "reward": (self.recovery_reward.contract() if self.reward_mode == "recovery_v1" else
                           {"schema": "standing_reference_v1", "description": "unchanged DeepMimic standing reference"})}

    def reward(self):
        standing = super().reward()
        if self.reward_mode == "reference":
            return standing
        q = self.engine.native_qpos()
        target, _ = self.reference.sample(self.offset + self.steps * self.dt)
        loads = self.engine.get_ground_contact_forces(0)[:, self.allowed_ground, 2].abs()
        # allowed_ground is Foot_L, Foot_R, Toe_L, Toe_R. A stepping foot may unload.
        feet = torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), dim=-1)
        recovery = self.recovery_reward(
            self.reference_reward(track_root=False), q[:, 2], target[:, 2],
            1 - 2 * (q[:, 4].square() + q[:, 5].square()),
            self.engine.get_root_vel(0), self.engine.get_root_ang_vel(0), feet)
        # Launch time is not impact evidence; misses and quiet worlds keep imitation.
        return torch.where(self.engine.hit & self.shot_enabled, recovery, standing)

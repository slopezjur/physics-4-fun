"""Motion-guided recovery under real projectile collisions, with unchanged actor inputs."""
import math
import numpy as np
import torch

from envs.base_env import EnvMode
from .ball import launch_state
from .ball_native import BallNativeEngine
from .ball_newton import BallNewtonEngine
from .stand_task import StandTask


class BallTask(StandTask):
    def __init__(self, rig, reference, num_envs=128, device="cuda:0", seed=210921,
                 engine_factory=None, control_mode="target_pd", speed_min=1., speed_max=2., quiet_fraction=.25):
        if control_mode != "target_pd" or not 1 <= speed_min <= speed_max <= 8 or not 0 <= quiet_fraction <= 1:
            raise ValueError("Ball task requires target PD, speeds 1–8 m/s and a valid quiet fraction")
        self.speed_min, self.speed_max, self.quiet_fraction = speed_min, speed_max, quiet_fraction
        super().__init__(rig, reference, num_envs, device, seed,
                         engine_factory or (BallNativeEngine if device == "cpu" else BallNewtonEngine), control_mode)
        self.episode_seconds = 5.
        self.shot_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.shot_speeds = torch.zeros(num_envs, device=device)
        self.directions = torch.zeros((num_envs, 3), device=device)
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
            self.shot_enabled[ids] = torch.rand(len(ids), generator=self.rng, device=self.device) >= self.quiet_fraction
        else:
            self.shot_steps[ids] = 60
            self.shot_speeds[ids] = torch.where(ids % 8 < 4, self.speed_min, self.speed_max)
            angle = (ids % 4) * math.pi / 2
            self.shot_enabled[ids] = True
        self.directions[ids, 0] = torch.cos(angle)
        self.directions[ids, 1] = torch.sin(angle)
        self.directions[ids, 2] = 0
        return result

    def step(self, action):
        ids = torch.nonzero((self.steps == self.shot_steps) & self.shot_enabled).flatten()
        if len(ids):
            target = self.engine.get_body_pos(0)[ids, self.rig.model.body("Chest").id - 1].cpu().numpy()
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
                "launch_steps_inclusive": [60, 120], "target_body": "Chest",
                "flight": "ballistic; aim at launch-time target; no gravity cancellation",
                "reward": "unchanged DeepMimic standing reference; experimental baseline objective"}

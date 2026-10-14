"""Shared GPU observation contract and fixed single-hit AMP curriculum."""
from functools import partial

import numpy as np
import torch

from .representation_task import RepresentationMixin
from .stand_ball_gpu import StandBallGPU
from .torque_stand_task import TorqueStandTask, AMPTorqueStandTask


class StandBallGPUTask(RepresentationMixin, TorqueStandTask):
    def __init__(self, rig, reference, *, ball_path, seconds):
        self.representation = 'target_pd'
        TorqueStandTask.__init__(self, rig, reference, 8, 'cuda:0', reward_mode='reference_v9',
            episode_seconds=seconds, engine_factory=partial(StandBallGPU, ball_path=ball_path, trace=True))


class StandBallAMPTask(RepresentationMixin, AMPTorqueStandTask):
    def __init__(self, rig, reference, *, ball_path, seed, arm, demonstrations):
        if arm not in ('quiet', 'single_ball'):
            raise ValueError('Unknown matched arm')
        self.arm, self.representation = arm, 'target_pd'
        self.launch_records = []
        AMPTorqueStandTask.__init__(self, rig, reference, 64, 'cuda:0', seed=seed,
            reward_mode='reference_v9', episode_seconds=12., demonstrations=demonstrations,
            demonstration_rig=rig, engine_factory=partial(StandBallGPU, ball_path=ball_path))
        self.mirror_demos = False
        self.launch_step = int(np.ceil(5./self.dt))
        self.direction = torch.zeros((64, 3), device=self.device)
        self.direction[:16, 0], self.direction[16:32, 0] = 1., -1.

    def step(self, action):
        enabled = (self.steps == self.launch_step) & (self.ids < 32)
        if self.arm == 'quiet':
            enabled[:] = False
        ids = self.ids[enabled]
        if len(ids):
            target = self.engine.get_body_pos(0)[ids, self.rig.model.body('Chest').id-1]
            direction = self.direction[ids]
            position = target - .65*direction
            gravity = torch.as_tensor(self.rig.model.opt.gravity, device=self.device, dtype=torch.float32)
            velocity = direction - .5*gravity*.65  # Fixed 1 m/s horizontal speed.
            self.engine.launch_ball(ids, position, velocity)
        self.launch_records.append(enabled.detach().clone())
        return super().step(action)

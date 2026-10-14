"""Isolated launch-geometry successors to the sealed matched ball curriculum."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from . import run_stand_ball_pilot as pilot
from .stand_ball_training_evidence import TrainingEvidence


@contextmanager
def curriculum(settings):
    from . import stand_ball_task
    base = stand_ball_task.StandBallAMPTask
    distance = float(settings['launch_distance_m'])
    speed = float(settings['training_speed_m_s'])
    onset = float(settings['launch_seconds'])
    seconds = float(settings['seconds'])
    if not (.2 <= distance <= .65 and .5 <= speed <= 3 and 1 <= onset < seconds <= 12):
        raise ValueError('Unbounded curriculum')

    class ChestTask(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            local = settings.get('per_seed', {}).get(str(kwargs['seed']), {})
            self.launch_distance_m = float(local.get('launch_distance_m',distance))
            self.launch_speed_m_s = float(local.get('training_speed_m_s',speed))
            if not (.2 <= self.launch_distance_m <= .65 and .5 <= self.launch_speed_m_s <= 3):
                raise ValueError('Unbounded seed curriculum')
            self.launch_step = int(np.ceil(onset/self.dt))
            self.episode_seconds = seconds

        def step(self, action):
            enabled = (self.steps == self.launch_step) & (self.ids < 32)
            if self.arm == 'quiet':
                enabled[:] = False
            ids = self.ids[enabled]
            if len(ids):
                target = self.engine.get_body_pos(0)[ids, self.rig.model.body('Chest').id-1]
                direction = self.direction[ids]
                gravity = torch.as_tensor(self.rig.model.opt.gravity, device=self.device, dtype=torch.float32)
                d, v = self.launch_distance_m, self.launch_speed_m_s
                self.engine.launch_ball(ids, target-d*direction, v*direction-.5*gravity*(d/v))
            self.launch_records.append(enabled.detach().clone())
            # Preserve RepresentationMixin's PD reference and AMP history update.
            return super(base, self).step(action)

    class ChestEvidence(TrainingEvidence):
        def launch(self, ids, position, velocity):
            if bool(self.launched[ids].any()):
                raise RuntimeError('Duplicate projectile launch')
            self.launched[ids] = True
            target = self.task.engine.get_body_pos(0)[ids, self.task.rig.model.body('Chest').id-1]
            direction = self.task.direction[ids]
            gravity = torch.as_tensor(self.task.rig.model.opt.gravity, device=self.task.device, dtype=torch.float32)
            d, v = self.task.launch_distance_m, self.task.launch_speed_m_s
            self.position_error[ids] = (position-(target-d*direction)).abs().amax(-1)
            self.velocity_error[ids] = (velocity-(v*direction-.5*gravity*(d/v))).abs().amax(-1)
            self.original_launch(ids, position, velocity)
            q, v = self.task.engine.ball_state()
            self.position_error[ids] = torch.maximum(self.position_error[ids], (q[ids,:3]-position).abs().amax(-1))
            self.velocity_error[ids] = torch.maximum(self.velocity_error[ids], (v[ids,:3]-velocity).abs().amax(-1))

    with patch.object(stand_ball_task, 'StandBallAMPTask', ChestTask), patch.object(pilot, 'TrainingEvidence', ChestEvidence):
        yield

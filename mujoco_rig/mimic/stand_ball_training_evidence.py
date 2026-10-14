"""Observe the admitted task without changing its reward, reset or throw schedule."""
import numpy as np
import torch

from .replicate_stand_representations import cpu


class TrainingEvidence:
    def __init__(self, task):
        self.task = task
        self.original_step = task.step
        self.original_launch = task.engine.launch_ball
        self.rows = []

    def __enter__(self):
        self.task.step = self.step
        self.task.engine.launch_ball = self.launch
        return self

    def __exit__(self, *_):
        self.task.step = self.original_step
        self.task.engine.launch_ball = self.original_launch

    def launch(self, ids, position, velocity):
        if bool(self.launched[ids].any()):
            raise RuntimeError('Duplicate projectile launch in one control interval')
        self.launched[ids] = True
        target = self.task.engine.get_body_pos(0)[ids, self.task.rig.model.body('Chest').id-1]
        direction = self.task.direction[ids]
        gravity = torch.as_tensor(self.task.rig.model.opt.gravity, device=self.task.device, dtype=torch.float32)
        expected_position = target - .65*direction
        expected_velocity = direction - .5*gravity*.65
        self.position_error[ids] = (position-expected_position).abs().amax(-1)
        self.velocity_error[ids] = (velocity-expected_velocity).abs().amax(-1)
        self.original_launch(ids, position, velocity)
        q, v = self.task.engine.ball_state()
        self.position_error[ids] = torch.maximum(self.position_error[ids], (q[ids, :3]-position).abs().amax(-1))
        self.velocity_error[ids] = torch.maximum(self.velocity_error[ids], (v[ids, :3]-velocity).abs().amax(-1))

    def step(self, action):
        task = self.task
        steps = task.steps.detach().clone()
        self.launched = torch.zeros(64, device=task.device, dtype=torch.bool)
        self.position_error = torch.zeros(64, device=task.device)
        self.velocity_error = torch.zeros(64, device=task.device)
        result = self.original_step(action)
        self.rows.append(dict(steps=steps, launched=self.launched.clone(),
            position_error=self.position_error.clone(), velocity_error=self.velocity_error.clone(),
            normal_impulse=task.engine.normal_impulse.detach().clone(),
            contact_impulse=task.engine.contact_impulse.detach().clone(),
            task_reward=result[1].detach().clone(), done=result[2].detach().clone()))
        return result

    def flush(self, path):
        if len(self.rows) != 64:
            raise RuntimeError('Expected one 64-step rollout per PPO update')
        values = {key: cpu(torch.stack([row[key] for row in self.rows])) for key in self.rows[0]}
        self.rows.clear()
        np.savez_compressed(path, **values)
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        assert all(np.isfinite(value).all() for value in values.values())
        expected = (values['steps'] == self.task.launch_step) & (np.arange(64)[None] < 32)
        if self.task.arm == 'quiet':
            expected[:] = False
        assert np.array_equal(values['launched'], expected)
        assert (values['position_error'] <= 2e-6).all() and (values['velocity_error'] <= 2e-6).all()
        assert (values['normal_impulse'] >= 0.).all()
        quiet = np.arange(64) >= (0 if self.task.arm == 'quiet' else 32)
        assert not values['normal_impulse'][:, quiet].any()
        assert not values['contact_impulse'][:, quiet].any()
        # A no-reset interval must advance one step; reset episodes restart at zero.
        assert ((values['steps'][1:] == values['steps'][:-1]+1) | (values['steps'][1:] == 0)).all()
        return dict(launches=int(values['launched'].sum()),
            contact_world_intervals=int(np.count_nonzero(values['normal_impulse'] > 0)),
            task_reward_mean=float(values['task_reward'].mean()))

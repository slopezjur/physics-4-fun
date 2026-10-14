"""Diagnostic reference reward with direct root-angular-speed feedback."""
import numpy as np
import torch

from .diagnose_stand_foundations import write_json


THRESHOLD_RAD_S = 1.
SCALE_RAD_S = 1.


def contract(weight):
    return dict(schema='diagnostic_reference_v9_angular_speed_v1', weight=float(weight),
        threshold_rad_s=THRESHOLD_RAD_S, scale_rad_s=SCALE_RAD_S,
        composition='reference_v9 - weight*(1-exp(-(max(root_angular_speed-1,0)/1)**2))',
        sampling='Ending control-interval root angular speed; physical FAIL task reward remains zero',
        outward_limit_penalty=0., negative_task_rewards_allowed=True)


def cost(angular_velocity):
    speed = torch.linalg.vector_norm(angular_velocity, dim=-1)
    excess = ((speed-THRESHOLD_RAD_S)/SCALE_RAD_S).clamp_min(0)
    return speed, -torch.expm1(-excess.square())


def task_factory(weight, *, record_components=False):
    from .torque_stand_task import AMPTorqueStandTask
    from envs.base_env import DoneFlags

    class AngularSpeedTask(AMPTorqueStandTask):
        def __init__(self, *args, **kwargs):
            self.reward_records = []
            super().__init__(*args, **kwargs)
            assert self.reward_mode == 'reference_v9'

        def reward(self):
            raw = super().reward()
            speed, angular_cost = cost(self.engine.get_root_ang_vel(0))
            self.raw_reference_reward, self.angular_cost, self.angular_speed = raw, angular_cost, speed
            self._diagnostics.update(raw_reference_reward=raw.mean(), angular_speed_rad_s=speed.mean(),
                angular_speed_cost=angular_cost.mean(), angular_speed_penalty=weight*angular_cost.mean())
            return raw if weight == 0 else raw-weight*angular_cost

        def step(self, action):
            result = super().step(action)
            if record_components:
                valid = result[2] != DoneFlags.FAIL.value
                self.reward_records.append(dict(reference_reward=(self.raw_reference_reward*valid).cpu().numpy().copy(),
                    angular_speed_rad_s=self.angular_speed.cpu().numpy().copy(),
                    angular_speed_cost=self.angular_cost.cpu().numpy().copy(),
                    reward_penalty=(weight*self.angular_cost*valid).cpu().numpy().copy()))
            return result

        def reset(self, env_ids=None):
            if record_components:
                self.reward_records.clear()
            return super().reset(env_ids)

        def observation_contract(self):
            return dict(super().observation_contract(), diagnostic_reward=contract(weight))

    return AngularSpeedTask


def augment_trace(task, report, out):
    with np.load(out/'trace.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update({key: np.stack([row[key] for row in task.reward_records]) for key in task.reward_records[0]})
    assert np.allclose(arrays['task_reward'], arrays['reference_reward']-arrays['reward_penalty'], rtol=0, atol=1e-7)
    independent_speed = np.linalg.norm(arrays['qvel'][..., 3:6], axis=-1)
    independent_cost = -np.expm1(-np.maximum(independent_speed-1, 0)**2)
    assert np.allclose(arrays['angular_speed_rad_s'], independent_speed, rtol=1e-6, atol=1e-6)
    assert np.allclose(arrays['angular_speed_cost'], independent_cost, rtol=0, atol=1e-6)
    for case in report['cases']:
        index = case['case']
        alive = arrays['alive_before'][:, index]
        discounts = .99**np.arange(int(alive.sum()))
        case['diagnostic_discounted_components'] = {key: float(discounts @ arrays[key][alive, index]) for key in
            ('reference_reward', 'angular_speed_cost', 'reward_penalty')}
        case['negative_task_reward_fraction'] = float((arrays['task_reward'][alive, index] < 0).mean())
    np.savez_compressed(out/'trace.npz', **arrays)
    report['diagnostic_discounted_components'] = {key: float(np.mean([row['diagnostic_discounted_components'][key] for row in report['cases']])) for key in
        ('reference_reward', 'angular_speed_cost', 'reward_penalty')}
    report['independent_angular_cost_verified'] = True
    write_json(out/'report.json', report)

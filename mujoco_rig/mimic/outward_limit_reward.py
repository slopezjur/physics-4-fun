"""Diagnostic-only reward cost; motor controls and the physical plant are untouched."""
import numpy as np
import torch

from .diagnose_stand_foundations import write_json

MARGIN_DEGREES = 5.


def contract(weight):
    return dict(schema='diagnostic_reference_v9_outward_limit_v1', weight=float(weight),
        margin_degrees=MARGIN_DEGREES, actuated_hinges=33,
        composition='reference_v9 - weight*mean(max(lower_proximity*relu(-held_normalized_torque), upper_proximity*relu(held_normalized_torque)))',
        sampling='Actual held delayed torque and ending hinge angles; physical FAIL task reward remains zero')


def cost(position, held_normalized_torque, lower, upper):
    margin = np.deg2rad(MARGIN_DEGREES)
    lower_proximity = ((lower+margin-position)/margin).clamp(0, 1)
    upper_proximity = ((position-upper+margin)/margin).clamp(0, 1)
    return torch.maximum(lower_proximity*(-held_normalized_torque).clamp_min(0),
                         upper_proximity*held_normalized_torque.clamp_min(0)).mean(-1)


def task_factory(weight, *, record_components=False):
    from .torque_stand_task import AMPTorqueStandTask
    from envs.base_env import DoneFlags

    class OutwardLimitTask(AMPTorqueStandTask):
        def __init__(self, rig, reference, num_envs=8, device='cpu', **kwargs):
            self.reward_records = []
            super().__init__(rig, reference, num_envs, device, **kwargs)
            assert self.reward_mode == 'reference_v9' and len(rig.actuators) == 33
            joints = rig.model.actuator_trnid[rig.actuators, 0]
            self.limit_qpos = torch.tensor(rig.model.jnt_qposadr[joints], device=device, dtype=torch.long)
            self.limit_lower = torch.tensor(rig.model.jnt_range[joints, 0], device=device, dtype=torch.float32)
            self.limit_upper = torch.tensor(rig.model.jnt_range[joints, 1], device=device, dtype=torch.float32)
            self.limit_actuators = torch.tensor(rig.actuators, device=device, dtype=torch.long)
            self.limit_scale = torch.tensor(rig.action_scale, device=device, dtype=torch.float32)

        def reward(self):
            raw = super().reward()
            q = self.engine.native_qpos()[:, self.limit_qpos]
            held = (self.engine.policy_control.ctrl[:, self.limit_actuators]/self.limit_scale).clamp(-1, 1)
            outward = cost(q, held, self.limit_lower, self.limit_upper)
            self.raw_reference_reward, self.outward_cost = raw, outward
            self._diagnostics['raw_reference_reward'] = raw.mean()
            self._diagnostics['outward_limit_cost'] = outward.mean()
            self._diagnostics['outward_limit_penalty'] = weight*outward.mean()
            return raw if weight == 0 else raw-weight*outward

        def step(self, action):
            result = super().step(action)
            if record_components:
                valid = result[2] != DoneFlags.FAIL.value
                self.reward_records.append(dict(reference_reward=(self.raw_reference_reward*valid).cpu().numpy().copy(),
                    outward_limit_cost=self.outward_cost.cpu().numpy().copy(),
                    reward_penalty=(weight*self.outward_cost*valid).cpu().numpy().copy()))
            return result

        def reset(self, env_ids=None):
            if record_components:
                self.reward_records.clear()
            return super().reset(env_ids)

        def observation_contract(self):
            return dict(super().observation_contract(), diagnostic_reward=contract(weight))

    return OutwardLimitTask


def augment_trace(task, report, out):
    with np.load(out/'trace.npz', allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in archive.files}
    arrays.update({key:np.stack([row[key] for row in task.reward_records]) for key in task.reward_records[0]})
    assert np.allclose(arrays['task_reward'], arrays['reference_reward']-arrays['reward_penalty'], rtol=0, atol=1e-7)
    for case in report['cases']:
        index = case['case']
        alive = arrays['alive_before'][:, index]
        discounts = .99**np.arange(int(alive.sum()))
        case['diagnostic_discounted_components'] = {key:float(discounts @ arrays[key][alive, index]) for key in
            ('reference_reward', 'outward_limit_cost', 'reward_penalty')}
    np.savez_compressed(out/'trace.npz', **arrays)
    report['diagnostic_discounted_components'] = {key:float(np.mean([row['diagnostic_discounted_components'][key] for row in report['cases']])) for key in
        ('reference_reward', 'outward_limit_cost', 'reward_penalty')}
    write_json(out/'report.json', report)

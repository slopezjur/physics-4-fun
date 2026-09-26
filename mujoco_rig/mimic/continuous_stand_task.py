"""Opt-in stationary contract. Finite-clip tasks and exported policies are unchanged."""
import torch

from .continuous_stand import CONTRACT, SEMANTICS, GATES, StationaryReference
from .stand_task import StandTask


class ContinuousStandTask(StandTask):
    def __init__(self, rig, reference, num_envs=8, device='cpu', **kwargs):
        self.reward_scale = 1.
        super().__init__(rig, reference, num_envs, device, control_mode='target_pd',
                         contact_mode='support_v2', reference_factory=StationaryReference, **kwargs)
        self.episode_seconds = GATES['seconds']

    def reward(self):
        return super().reward() * self.reward_scale

    def sample_reference(self, time, env_ids=None):
        return self.reference.sample(time, self.ids if env_ids is None else env_ids)

    def reference_phase(self, time):
        return torch.zeros_like(time)

    def reset(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        if len(ids):
            self.steps[ids] = 0
            self.offset[ids] = 0
            self.engine.reset_envs(ids, *self.sample_reference(self.offset[ids], ids))
        return self.observations(), {}

    def observation_contract(self):
        result = super().observation_contract()
        result.update(observation_contract=CONTRACT, reference_semantics=SEMANTICS,
                      deployment_status='New stationary contract; legacy actors are explicit evaluation-only substitutions.')
        result['observation_layout'][-1]['name'] = 'stationary_reference_phase_zero'
        return result

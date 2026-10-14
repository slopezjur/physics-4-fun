"""Matched 374-input tasks; physical reward and AMP features are inherited."""
from functools import partial
import torch

from .representation_engine import RepresentationGPU, RepresentationNative
from .rig import DelayedTorqueControl
from .target_control33 import DelayedTargetControl33
from .torque_stand_task import AMPTorqueStandTask, TorqueStandTask


class RepresentationMixin:
    def __init__(self, rig, reference, num_envs=8, device='cpu', *, representation, trace=False, **kwargs):
        if representation not in ('torque', 'target_pd'):
            raise ValueError('Unknown experimental representation')
        self.representation = representation
        factory = DelayedTargetControl33 if representation == 'target_pd' else DelayedTorqueControl
        engine = RepresentationNative if device == 'cpu' else RepresentationGPU
        super().__init__(rig, reference, num_envs, device,
            engine_factory=partial(engine, control_factory=factory, trace=trace), **kwargs)

    def observations(self):
        original = super().observations()
        if self.representation == 'torque':
            original = torch.cat((original[:, :161], torch.zeros_like(original[:, :68]), original[:, 161:]), dim=-1)
        if original.shape[-1] != 374:
            raise RuntimeError('Representation observation contract mismatch')
        return original

    def step(self, action):
        if self.representation == 'target_pd':
            self.engine.policy_control.set_reference(*self.sample_reference(self.offset+self.steps*self.dt))
        return super().step(action)

    def observation_contract(self):
        contract = super().observation_contract()
        layout = contract['observation_layout']
        layout[7].update(width=134, name='pending_targets_q_v_valid_oldest_first' if self.representation == 'target_pd'
            else 'pending_torque_66_then_zero_padding_68')
        layout[8]['offset'], layout[9]['offset'] = 229, 373
        contract.update(num_obs=374, observation_contract=f'mimic_stand_representation_{self.representation}_v1',
            standing_pose_servo=self.representation == 'target_pd', representation=self.representation,
            reference_role='Reset states, observations and soft reward; sampled motor targets in target_pd arm only')
        if self.representation == 'target_pd':
            contract.update(self.engine.policy_control.contract())
        return contract


class RepresentationTask(RepresentationMixin, TorqueStandTask):
    pass


class AMPRepresentationTask(RepresentationMixin, AMPTorqueStandTask):
    pass

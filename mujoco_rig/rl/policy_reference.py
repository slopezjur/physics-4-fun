"""Fixed behavior targets, independent of PPO rollouts and reward definitions."""
from __future__ import annotations

import math

import torch


REFERENCE_VERSION = 'perturb_reference_v1'


class PolicyReference:
    """Equal standing/impact retention loss with a frozen normalization scale.

    The loss is the Gaussian mean-KL term at the reference's fixed variance.
    Changing the student's exploration cannot make deviations cheaper. This is
    a soft constraint on sampled states, not a guarantee of rollout survival.
    """

    def __init__(self, data, device='cpu', coefficient=1.0, seed=2718):
        if data.get('version') != REFERENCE_VERSION:
            raise ValueError('Unsupported policy reference version')
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Reference coefficient must be finite and non-negative')
        self.coefficient = coefficient
        self.source = dict(data['source'])
        self.scale = data['scale'].detach().to(device).clone()
        if self.scale.ndim != 1 or not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError('Reference action scales must be a positive finite vector')
        self.groups = {}
        width = None
        for name in ('quiet', 'impact'):
            obs, mean = (data['groups'][name][key].detach().to(device).clone() for key in ('obs', 'mean'))
            if (obs.ndim != 2 or mean.ndim != 2 or not len(obs) or len(obs) != len(mean)
                    or mean.shape[1] != len(self.scale) or not torch.isfinite(obs).all()
                    or not torch.isfinite(mean).all() or (width is not None and width != obs.shape[1])):
                raise ValueError(f'Invalid {name} reference observations/targets')
            width = obs.shape[1]
            self.groups[name] = (obs, mean)
        self.num_obs, self.num_actions = width, len(self.scale)
        # Sampling must not change the rollout/PPO random stream in an ablation.
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def _drift(self, actor, obs, mean):
        return 0.5 * ((actor(obs) - mean) / self.scale).square().sum(-1).mean()

    def same_targets(self, other):
        return (self.source == other.source and torch.equal(self.scale, other.scale)
                and all(torch.equal(left, right) for name in self.groups
                        for left, right in zip(self.groups[name], other.groups[name])))

    def loss(self, actor, samples_per_group=256):
        observations, targets = [], []
        for obs, mean in self.groups.values():
            idx = torch.randint(len(obs), (samples_per_group,), generator=self.generator).to(obs.device)
            observations.append(obs[idx])
            targets.append(mean[idx])
        return self._drift(actor, torch.cat(observations), torch.cat(targets))

    @torch.no_grad()
    def measure(self, actor):
        return {f'reference_{name}': self._drift(actor, obs, mean).item()
                for name, (obs, mean) in self.groups.items()}

    def state_dict(self):
        return {'version': REFERENCE_VERSION, 'source': self.source,
                'scale': self.scale.detach().cpu().clone(), 'coefficient': self.coefficient,
                'generator_state': self.generator.get_state(),
                'groups': {name: {'obs': obs.detach().cpu().clone(), 'mean': mean.detach().cpu().clone()}
                           for name, (obs, mean) in self.groups.items()}}

    @classmethod
    def from_state_dict(cls, state, device='cpu'):
        reference = cls(state, device, state['coefficient'])
        reference.generator.set_state(state['generator_state'].cpu())
        return reference

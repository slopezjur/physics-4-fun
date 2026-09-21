"""Portable deterministic MimicKit actor, including its learned normalization."""
import copy

import torch


class StandPolicy(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.obs_norm = copy.deepcopy(agent._obs_norm)
        self.action_norm = copy.deepcopy(agent._a_norm)
        self.actor = copy.deepcopy(agent._model._actor_layers)
        self.mean = copy.deepcopy(agent._model._action_dist._mean_net)

    def forward(self, observation):
        mean = self.mean(self.actor(self.obs_norm.normalize(observation)))
        return self.action_norm.unnormalize(mean).clamp(-1, 1)

"""MimicKit actor network with a frozen standing base and observable recovery branch."""
import sys
from types import SimpleNamespace

import torch

NAME = "stationary_gated_recovery_v1"
CAPTURE_NAME = "stationary_capture_recovery_v2"
SUPPORT_NAME = "stationary_support_recovery_v3"
MOTION_NAME = "stationary_motion_recovery_v4"
DAMPING_NAME = "stationary_capture_damping_v5"


def motion_recovery_gate(observation, thresholds=(.15, .30)):
    """Permit damping while the capture point is supported but the root is moving."""
    capture = ((observation[..., 362] - .001) / .019).clamp(0, 1)
    speed = torch.linalg.vector_norm(observation[..., 9:11], dim=-1)
    motion = ((speed - thresholds[0]) / (thresholds[1] - thresholds[0])).clamp(0, 1)
    return torch.maximum(capture, motion)


def damping_recovery_gate(observation, specification=(.15, .30, 30.)):
    capture = ((observation[..., 362] - .001) / .019).clamp(0, 1)
    speed = torch.linalg.vector_norm(observation[..., 9:11], dim=-1)
    moving = ((speed - specification[0]) / (specification[1] - specification[0])).clamp(0, 1)
    supported = (observation[..., 93:95].amin(-1) * 1000 / specification[2]).clamp(0, 1)
    return (1 - capture) * moving * supported


def support_recovery_gate(observation):
    """Keep feedback available during unloading, using only current physical state."""
    capture = ((observation[..., 362] - .001) / .019).clamp(0, 1)
    loads = observation[..., 93:95] * 1000
    transfer = ((30 - loads.amin(-1)) / 25).clamp(0, 1)
    support = (loads.amax(-1) / 50).clamp(0, 1)
    return torch.maximum(capture, transfer * support)


def recovery_gate(observation):
    speed = torch.linalg.vector_norm(observation[..., 9:11], dim=-1)
    rotation = torch.linalg.vector_norm(observation[..., 12:14], dim=-1)
    return torch.maximum((speed - .2) / .2, (rotation - .5) / .5).clamp(0, 1)


def quiet_rehearsal_indices(observation, limit, *, support_activation=False, motion_activation=False,
                            damping_activation=False):
    """Retain brief branch-active quiet states that uniform time sampling can miss."""
    count = min(int(limit), len(observation))
    if count < 1:
        raise ValueError("Quiet rehearsal requires observations and a positive limit")
    gate = (damping_recovery_gate(observation) if damping_activation else
            motion_recovery_gate(observation) if motion_activation else
            support_recovery_gate(observation) if support_activation else
            ((observation[..., 362] - .001) / .019).clamp(0, 1)
            if observation.shape[-1] == 363 else recovery_gate(observation))
    critical = torch.nonzero(gate > 0).flatten()
    critical_count = min(len(critical), max(1, count // 2))
    critical = critical[torch.argsort(gate[critical], descending=True)[:critical_count]]
    available = torch.ones(len(observation), dtype=torch.bool, device=observation.device)
    available[critical] = False
    remaining = torch.nonzero(available).flatten()
    picks = torch.linspace(0, len(remaining)-1, count-critical_count, device=observation.device).long()
    return torch.cat((critical, remaining[picks])).sort().values


class GatedRecoveryNet(torch.nn.Module):
    def __init__(self, base, width, *, support_activation=False, motion_activation=False,
                 damping_activation=False, correction_limit=.8):
        super().__init__()
        if width not in (362, 363):
            raise ValueError("Recovery branch requires a supported stationary contract")
        self.capture_observation = width == 363
        if support_activation and width != 363:
            raise ValueError("Support recovery requires the capture observation contract")
        if motion_activation and (width != 363 or support_activation):
            raise ValueError("Motion recovery requires capture observations without support activation")
        if damping_activation and (width != 363 or support_activation or motion_activation):
            raise ValueError("Damping requires an unchanged capture-only recovery branch")
        self.support_activation = support_activation
        self.motion_activation = motion_activation
        self.damping_activation = damping_activation
        if motion_activation:
            self.register_buffer("motion_gate_thresholds", torch.tensor([.15, .30]))
        self.correction_limit = correction_limit
        self.base = base.requires_grad_(False)
        self.base_mean = torch.nn.Linear(512, 30).requires_grad_(False)
        self.velocity = torch.nn.Linear(4, 30, bias=False)
        self.recovery = torch.nn.Sequential(torch.nn.Linear(width, 128), torch.nn.ReLU(),
                                            torch.nn.Linear(128, 30))
        if damping_activation:
            self.register_buffer("damping_gate_specification", torch.tensor([.15, .30, 30.]))
            self.damping = torch.nn.Linear(4, 30, bias=False)
            torch.nn.init.zeros_(self.damping.weight)
            self.velocity.requires_grad_(False)
            self.recovery.requires_grad_(False)
        torch.nn.init.zeros_(self.velocity.weight)
        torch.nn.init.zeros_(self.recovery[-1].weight)
        torch.nn.init.zeros_(self.recovery[-1].bias)
        self.register_buffer("observation_mean", torch.zeros(width))
        self.register_buffer("observation_std", torch.ones(width))
        self.register_buffer("action_std", torch.ones(30))
        self.register_buffer("leg_mask", torch.zeros(30))

    def forward(self, normalized):
        raw = normalized * self.observation_std + self.observation_mean
        features = torch.cat((raw[..., 9:11] / .5, raw[..., 12:14] / 2.), -1).clamp(-2, 2)
        correction = (self.velocity(features) + self.recovery(normalized)).clamp(
            -self.correction_limit, self.correction_limit)
        gate = (motion_recovery_gate(raw, self.motion_gate_thresholds) if self.motion_activation else
                support_recovery_gate(raw) if self.support_activation else
                ((raw[..., 362] - .001) / .019).clamp(0, 1) if self.capture_observation else recovery_gate(raw))
        correction = correction * self.leg_mask * gate[..., None] / self.action_std
        if self.damping_activation:
            damping = self.damping(features).clamp(-self.correction_limit, self.correction_limit)
            damping_gate = damping_recovery_gate(raw, self.damping_gate_specification)
            correction = correction + damping * self.leg_mask * damping_gate[..., None] / self.action_std
        return self.base_mean(self.base(normalized[..., :362])) + correction


def build_net(input_dict, activation):
    from learning.nets.fc_2layers_1024units import build_net as build_base
    width = sum(int(torch.tensor(value.shape).prod()) for value in input_dict.values())
    base, _ = build_base({"obs": torch.empty(362)}, activation)
    return GatedRecoveryNet(base, width), {}


def build_support_net(input_dict, activation):
    from learning.nets.fc_2layers_1024units import build_net as build_base
    width = sum(int(torch.tensor(value.shape).prod()) for value in input_dict.values())
    base, _ = build_base({"obs": torch.empty(362)}, activation)
    return GatedRecoveryNet(base, width, support_activation=True, correction_limit=2.), {}


def build_motion_net(input_dict, activation):
    from learning.nets.fc_2layers_1024units import build_net as build_base
    width = sum(int(torch.tensor(value.shape).prod()) for value in input_dict.values())
    base, _ = build_base({"obs": torch.empty(362)}, activation)
    return GatedRecoveryNet(base, width, motion_activation=True), {}


def build_damping_net(input_dict, activation):
    from learning.nets.fc_2layers_1024units import build_net as build_base
    width = sum(int(torch.tensor(value.shape).prod()) for value in input_dict.values())
    base, _ = build_base({"obs": torch.empty(362)}, activation)
    return GatedRecoveryNet(base, width, damping_activation=True), {}


def register():
    """Register locally; the pinned upstream checkout remains unchanged."""
    from learning.nets import net_builder
    from learning.ppo_model import PPOModel
    setattr(net_builder, NAME, sys.modules[__name__])
    setattr(net_builder, CAPTURE_NAME, sys.modules[__name__])
    setattr(net_builder, SUPPORT_NAME, SimpleNamespace(build_net=build_support_net))
    setattr(net_builder, MOTION_NAME, SimpleNamespace(build_net=build_motion_net))
    setattr(net_builder, DAMPING_NAME, SimpleNamespace(build_net=build_damping_net))
    if getattr(PPOModel, "_mimic_gated_registered", False):
        return
    original = PPOModel._build_actor

    def build_actor(self, config, env):
        original(self, config, env)
        if config["actor_net"] in (NAME, CAPTURE_NAME, SUPPORT_NAME, MOTION_NAME, DAMPING_NAME):
            readout = self._action_dist._mean_net
            with torch.no_grad():
                readout.weight.copy_(torch.eye(30, device=readout.weight.device))
                readout.bias.zero_()
            readout.requires_grad_(False)

    PPOModel._build_actor = build_actor
    PPOModel._mimic_gated_registered = True

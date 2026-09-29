"""Observable-state leg exploration with matching PPO sampling and likelihoods."""
import math

import torch


def conditioned_logstd(raw_observation, base_logstd, leg_mask, maximum_std, gate_mode="speed"):
    # Existing stationary contract: root linear velocity occupies channels 9:12.
    speed = torch.linalg.vector_norm(raw_observation[..., 9:11], dim=-1)
    gate = ((speed - .1) / .3).clamp(0, 1)[..., None]
    if gate_mode == "support":
        if raw_observation.shape[-1] != 363:
            raise ValueError("Support exploration requires the 363-value capture observation")
        from .gated_recovery_net import support_recovery_gate
        gate = torch.maximum(gate, support_recovery_gate(raw_observation)[..., None])
    elif gate_mode != "speed":
        raise ValueError("Unknown exploration gate")
    base_std = base_logstd.exp()
    target = torch.maximum(base_std, torch.as_tensor(maximum_std, device=base_std.device))
    leg_std = base_std + gate * (target - base_std)
    return torch.where(leg_mask, leg_std.log(), base_logstd)


def install(agent, maximum_std, gate_mode="speed"):
    if not math.isfinite(maximum_std) or not .05 <= maximum_std <= .4:
        raise ValueError("Risk exploration standard deviation must be within 0.05..0.4")
    if gate_mode not in ("speed", "support"):
        raise ValueError("Unknown exploration gate")
    if gate_mode == "support" and agent._obs_norm._mean.numel() != 363:
        raise ValueError("Support exploration requires the capture observation contract")
    from learning.distribution_gaussian_diag import DistributionGaussianDiag
    normalizer = agent._obs_norm
    contract = agent._env.observation_contract()
    velocity = next(g for g in contract["observation_layout"] if g["name"] == "root_linear_velocity_world")
    if velocity["offset"] != 9 or velocity["width"] != 3:
        raise ValueError("Unexpected stationary root velocity layout")
    legs = torch.tensor([name.startswith(("Thigh_", "Shin_", "Foot_", "Toe_"))
                         for name in contract["actuator_names"]], device=agent._device)
    original = agent._model.eval_actor

    def evaluate(normalized_obs):
        distribution = original(normalized_obs)
        raw = normalized_obs * normalizer._std + normalizer._mean
        logstd = conditioned_logstd(raw, distribution.logstd, legs, maximum_std, gate_mode)
        return DistributionGaussianDiag(distribution.mean, logstd)

    agent._model.eval_actor = evaluate

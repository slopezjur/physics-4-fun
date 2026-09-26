"""Explicit centimetre-scale tracking objective for double-support experiments."""
import torch

SCHEMA = "mimic_weight_shift_precision_v1"
POSITION_SCALE_M = .005
VELOCITY_SCALE_M_S = .05


def contract(mode):
    if mode == "reference":
        return {"schema": "mimic_reference_reward_v1"}
    if mode != "precision_v1":
        raise ValueError(f"Unknown weight-shift reward: {mode}")
    return dict(schema=SCHEMA, base_weight=.5, position_weight=.4, velocity_weight=.1,
                horizontal_position_scale_m=POSITION_SCALE_M,
                horizontal_velocity_scale_m_s=VELOCITY_SCALE_M_S,
                quiet_reward="unchanged reference reward")


def precision_reward(base, actual_position, target_position, actual_velocity, target_velocity):
    position_error = ((actual_position[..., :2] - target_position[..., :2]) / POSITION_SCALE_M).square().sum(-1)
    velocity_error = ((actual_velocity[..., :2] - target_velocity[..., :2]) / VELOCITY_SCALE_M_S).square().sum(-1)
    return .5 * base + .4 * torch.exp(-position_error) + .1 * torch.exp(-velocity_error)

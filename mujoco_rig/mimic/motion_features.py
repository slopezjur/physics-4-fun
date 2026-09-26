"""Versioned extra motion features; retain the pretrained input path exactly."""
import torch

SCHEMA = "mimic_dual_motion_features_v1"


def validate_lineage(lineage, contract):
    expected = [i for group in contract["observation_layout"]
                if group["name"] in ("pending_targets_q_v_valid_oldest_first", "future_reference_poses_1_2_3")
                for i in range(group["offset"], group["offset"] + group["width"])]
    if (lineage.get("feature_schema") != SCHEMA or lineage.get("migration_mode") != "additive_features"
            or lineage.get("motion_feature_indices") != expected or not expected):
        raise ValueError("Inherited motion-feature architecture disagrees with the observation contract")


class MotionFeatureNormalizer(torch.nn.Module):
    def __init__(self, original, indices):
        super().__init__()
        for name in ("_mean", "_std", "_count"):
            self.register_parameter(name, torch.nn.Parameter(getattr(original, name).detach().clone(), requires_grad=False))
        self.register_buffer("_motion_indices", torch.as_tensor(indices, dtype=torch.long, device=self._mean.device))
        self.register_buffer("_motion_std", self._std.detach().clone())

    def normalize(self, observation):
        difference = observation - self._mean
        original = (difference / self._std).clamp(-10, 10)
        motion = (difference / self._motion_std)[..., self._motion_indices].clamp(-10, 10)
        return torch.cat((original, motion), dim=-1)

    def get_count(self):
        return self._count

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std


def install(agent, indices):
    """Call before optimizer construction and before loading an expanded checkpoint."""
    agent._obs_norm = MotionFeatureNormalizer(agent._obs_norm, indices)
    for layers in (agent._model._actor_layers, agent._model._critic_layers):
        original = layers[0]
        expanded = torch.nn.Linear(original.in_features + len(indices), original.out_features,
                                   device=original.weight.device, dtype=original.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :original.in_features].copy_(original.weight)
            expanded.bias.copy_(original.bias)
        layers[0] = expanded

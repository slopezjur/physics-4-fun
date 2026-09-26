"""Experimental learned ankle residual around a frozen Stand actor."""
import math

import torch

from .baseline import sha256


PILOT_SCHEMA = "mimic_com_residual_distillation_pilot_v1"
MIXED_SCHEMA = "mimic_com_residual_mixed_distillation_v1"


def build_residual_network(num_obs):
    return torch.nn.Sequential(torch.nn.Linear(num_obs, 128), torch.nn.SiLU(),
                               torch.nn.Linear(128, 128), torch.nn.SiLU(),
                               torch.nn.Linear(128, 1), torch.nn.Tanh())


class ComResidualPolicy:
    def __init__(self, actor, model, mean, scale, ankles, residual_max):
        if (mean.ndim != 1 or scale.shape != mean.shape or len(ankles) != 2
                or not math.isfinite(residual_max) or not 0 < residual_max <= 1
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or not (scale > 0).all()):
            raise ValueError("Invalid experimental residual contract")
        self.actor = actor.eval()
        self.model = model.eval()
        self.mean, self.scale = mean, scale
        self.ankles, self.residual_max = tuple(ankles), residual_max

    @torch.no_grad()
    def __call__(self, observation):
        if observation.ndim != 2 or observation.shape[1] != len(self.mean):
            raise ValueError("Residual observation shape mismatch")
        correction = self.residual_max * self.model(
            ((observation - self.mean) / self.scale).clamp(-5., 5.))
        action = self.actor(observation).clone()
        action[:, self.ankles] += correction
        return action.clamp_(-1., 1.)


def load_experimental_residual(path, actor, rig, source_contract, source_checkpoint):
    """Reconstruct only a hash-matched diagnostic policy, never select an export."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("schema") not in (PILOT_SCHEMA, MIXED_SCHEMA)
            or saved.get("source_contract_sha256") != sha256(source_contract)
            or saved.get("source_checkpoint_sha256") != sha256(source_checkpoint)):
        raise ValueError("Experimental residual provenance mismatch")
    names = [rig.model.actuator(int(a)).name for a in rig.actuators]
    ankles = [names.index(name) for name in ("Foot_L_rx", "Foot_R_rx")]
    if saved.get("ankle_indices") != ankles:
        raise ValueError("Experimental residual actuator mapping mismatch")
    model = build_residual_network(len(saved["mean"]))
    model.load_state_dict(saved["model"])
    return ComResidualPolicy(actor, model, saved["mean"], saved["scale"],
                             ankles, saved["residual_max"])

"""Versioned post-contact objective, independent of the simulator and actor inputs."""
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RecoveryReward:
    # Retain a small motion prior while allowing displaced, differently posed recovery.
    imitation_weight: float = .25
    height_scale_m: float = .15
    upright_cosine_scale: float = .2
    linear_speed_scale_m_s: float = .5
    angular_speed_scale_rad_s: float = 2.
    support_load_n: float = 5.
    posture_weight: float = .6
    settling_weight: float = .3
    supported_settling_weight: float = .1
    sustained_settling_weight: float = 0.
    sustained_settling_seconds: float = 2.

    def contract(self):
        values = asdict(self)
        if not self.sustained_settling_weight:
            values.pop("sustained_settling_weight")
            values.pop("sustained_settling_seconds")
        return {"schema": "ball_recovery_v2" if self.sustained_settling_weight else "ball_recovery_v1",
                **values,
                "activation": "sticky ball/character contact until episode reset",
                "quiet_and_pre_contact": "unchanged standing imitation",
                "post_contact_imitation": "root-relative; horizontal translation and heading invariant",
                "post_contact_task": "reference height, upright pelvis, low root velocity, supported settling",
                "support": ("continuous two-foot settling earns an additional bonus; either foot may step"
                            if self.sustained_settling_weight else
                            "small settling bonus only; single-support recovery is allowed")}

    def __call__(self, imitation, height, target_height, up_z, velocity, angular_velocity,
                 foot_loads, settled_seconds=None):
        posture = torch.exp(-((height - target_height) / self.height_scale_m).square()
                            - ((1 - up_z.clamp(-1, 1)) / self.upright_cosine_scale).square())
        settling = torch.exp(-(velocity / self.linear_speed_scale_m_s).square().sum(-1)
                             - (angular_velocity / self.angular_speed_scale_rad_s).square().sum(-1))
        support = (foot_loads.clamp_min(0) / self.support_load_n).clamp_max(1).amin(-1)
        recovery = posture * (self.posture_weight + self.settling_weight * settling
                              + self.supported_settling_weight * settling * support)
        sustained = (settled_seconds / self.sustained_settling_seconds).clamp(0, 1) if settled_seconds is not None else 0.
        return (self.imitation_weight * imitation + (1 - self.imitation_weight) * recovery
                + self.sustained_settling_weight * sustained)

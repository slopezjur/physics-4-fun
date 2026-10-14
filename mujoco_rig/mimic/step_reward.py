"""State-based swing guidance; landing and settled recovery are measured separately."""
from dataclasses import dataclass

import torch

from .capture_reward import CaptureRecoveryReward


def swing_quality(com, velocity, sole_points, reference_height, loads, gravity=9.81, preparation_weight=0.):
    ground = sole_points[..., 2].amin(dim=(1, 2))
    capture = com[:, :2] + velocity[:, :2] * ((com[:, 2] - ground).clamp_min(.05) / gravity).sqrt()[:, None]
    location = sole_points[:, :, :, :2].mean(2)
    error = torch.linalg.vector_norm(location - capture[:, None], dim=-1)
    clearance = (sole_points[..., 2] - reference_height).amin(-1)
    lift = (clearance / .02).clamp(0, 1) * torch.exp(-((clearance - .04).clamp_min(0) / .06).square())
    unloaded = (1 - loads / 5.).clamp(0, 1)
    opposite_support = (loads.flip(-1) / 100.).clamp(0, 1)
    share = loads / loads.sum(-1, keepdim=True).clamp_min(1.)
    transfer = (1 - 2 * share).clamp(0, 1)
    progress = preparation_weight * transfer + (1-preparation_weight) * lift * unloaded
    return (torch.exp(-(error / .15).square()) * progress * opposite_support).amax(-1)


@dataclass(frozen=True)
class StepRecoveryReward(CaptureRecoveryReward):
    def contract(self):
        result = super().contract()
        result.update(schema="ball_step_recovery_v4", objective="upright survival, loaded capture, useful swing, settled stance",
            swing="state-based reach toward momentum capture point; either unloaded foot, opposite foot loaded",
            swing_weight=.15, risk_scale_m=.06,
            settling="reduced while capture point lies outside loaded support; no step-count or timing bonus")
        return result

    def __call__(self, imitation, height, target_height, up_z, velocity, angular_velocity,
                 capture_error, swing):
        posture = torch.exp(-((height - target_height) / .15).square()
                            - ((1 - up_z.clamp(-1, 1)) / .2).square())
        capture = torch.exp(-(capture_error / self.capture_scale_m).square())
        risk = (capture_error / .06).clamp(0, 1)
        settling = torch.exp(-(velocity / .5).square().sum(-1)
                             - (angular_velocity / 2.).square().sum(-1))
        recovery = posture * (.35 + .45 * capture + .2 * (1-risk) * settling + .15 * risk * swing)
        return self.imitation_weight * imitation + (1-self.imitation_weight) * recovery


@dataclass(frozen=True)
class TransferRecoveryReward(StepRecoveryReward):
    """Credit useful load transfer before lift, then require two-foot settling."""
    def contract(self):
        result = super().contract()
        result.update(schema="ball_step_recovery_v5", preparation_weight=.3,
            preparation="continuous unloading relative to total load; either foot, opposite support required",
            settling="two feet each carrying at least 30 N; velocity and capture criteria unchanged")
        return result

    def __call__(self, imitation, height, target_height, up_z, velocity, angular_velocity,
                 capture_error, swing, loads):
        value = super().__call__(imitation, height, target_height, up_z, velocity, angular_velocity,
                                 capture_error, swing)
        posture = torch.exp(-((height - target_height) / .15).square()
                            - ((1 - up_z.clamp(-1, 1)) / .2).square())
        risk = (capture_error / .06).clamp(0, 1)
        settling = torch.exp(-(velocity / .5).square().sum(-1)
                             - (angular_velocity / 2.).square().sum(-1))
        bilateral = (loads.amin(-1) / 30.).clamp(0, 1)
        return value - (1-self.imitation_weight) * posture * .2 * (1-risk) * settling * (1-bilateral)


@dataclass(frozen=True)
class PlacementRecoveryReward(TransferRecoveryReward):
    def contract(self):
        from .support_motion import UsefulPlacement
        result = super().contract()
        result.update(schema='ball_step_recovery_v6', preparation_weight=0.,
            preparation='No sustained grounded-foot transfer credit',
            placement=UsefulPlacement.contract(), placement_event_weight=.25,
            loaded_slip_proxy_weight=.04, slip_deadband_m_s=.02,
            slip_scale_m_s=.2, maximum_slip_cost=.04)
        return result

    def __call__(self, *args, placement, slip):
        value = super().__call__(*args)
        height,target_height,up_z = args[1:4]
        posture = torch.exp(-((height-target_height)/.15).square()
                            -((1-up_z.clamp(-1,1))/.2).square())
        return value + .25*posture*placement - .04*slip


@dataclass(frozen=True)
class PeakPlacementRecoveryReward(PlacementRecoveryReward):
    def contract(self):
        from .support_motion import UsefulPlacement
        result = super().contract()
        result.update(schema='ball_step_recovery_v7',placement=UsefulPlacement.contract(peak_articulation=True))
        return result

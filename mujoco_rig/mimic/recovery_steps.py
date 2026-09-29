"""Measured lift/landing events; no prescribed foot, trajectory or policy clock."""
import torch


class RecoveryStepTracker:
    def __init__(self, sole_points, dt):
        self.start = sole_points.clone()  # [world, left/right, foot/toe, xyz]
        self.dt = dt
        shape = sole_points.shape[:2]
        self.air = torch.zeros(shape, dtype=torch.long, device=sole_points.device)
        self.landing = torch.zeros_like(self.air)
        self.in_swing = torch.zeros(shape, dtype=torch.bool, device=sole_points.device)
        self.qualified = torch.zeros_like(self.in_swing)
        self.lifted = torch.zeros_like(self.in_swing)
        self.steps = torch.zeros_like(self.air)
        self.takeoff = sole_points[:, :, :, :2].mean(2).clone()
        self.max_clearance = torch.zeros(shape, device=sole_points.device)
        self.max_placement = torch.zeros_like(self.max_clearance)

    def update(self, sole_points, foot_loads, active, hit):
        valid = (active & hit)[:, None]
        airborne = foot_loads < 5.
        xy = sole_points[:, :, :, :2].mean(2)
        clear = (sole_points[:, :, :, 2] - self.start[:, :, :, 2]).amin(-1)
        taking_off = valid & airborne & ~self.in_swing
        self.takeoff = torch.where(taking_off[:, :, None], xy, self.takeoff)
        self.in_swing |= taking_off
        self.air = torch.where(valid & airborne, self.air + 1, 0)
        self.qualified |= valid & (self.air * self.dt >= .05) & (clear >= .015)
        self.lifted |= self.qualified
        self.max_clearance = torch.maximum(self.max_clearance, torch.where(valid & airborne, clear, 0))
        self.landing = torch.where(valid & ~airborne & self.in_swing, self.landing + 1, 0)
        landed = valid & self.in_swing & (self.landing * self.dt >= .05)
        placement = torch.linalg.vector_norm(xy - self.takeoff, dim=-1)
        completed = landed & self.qualified & (placement >= .03)
        self.steps += completed.long()
        self.max_placement = torch.maximum(self.max_placement, torch.where(completed, placement, 0))
        self.in_swing &= ~landed
        self.qualified &= ~landed

    @staticmethod
    def contract():
        return dict(airborne_foot_load_below_n=5., minimum_airborne_seconds=.05,
                    minimum_foot_and_toe_clearance_m=.015, minimum_placement_m=.03,
                    minimum_loaded_landing_seconds=.05, either_foot=True,
                    recovered_step="measured step plus existing final upright/two-foot settling gate")

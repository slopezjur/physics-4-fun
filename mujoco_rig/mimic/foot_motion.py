"""Bounded temporal costs for foot chatter, with no preferred foot or step clock."""
import math

import torch


class FootMotionCost:
    def __init__(self, positions, loads, dt):
        self.dt = dt
        self.position = positions.clone()
        self.velocity = torch.zeros_like(positions)
        self.ready = torch.zeros(len(positions), dtype=torch.long, device=positions.device)
        self.supported = loads >= 30.
        self.age = torch.full_like(loads, math.ceil(.2 / dt), dtype=torch.long)

    def reset(self, ids, positions, loads):
        self.position[ids] = positions[ids]
        self.velocity[ids] = 0
        self.ready[ids] = 0
        self.supported[ids] = loads[ids] >= 30.
        self.age[ids] = math.ceil(.2 / self.dt)

    def update(self, positions, loads):
        velocity = (positions - self.position) / self.dt
        acceleration = torch.linalg.vector_norm((velocity - self.velocity) / self.dt, dim=-1)
        supported = torch.where(self.supported, loads >= 15., loads >= 30.)
        changed = supported != self.supported
        reversal = changed & (self.age * self.dt < .2)
        # Do not charge the first lift or landing. Hysteresis rejects threshold noise.
        # Acceleration has a deadband and a hard cap, leaving useful steps possible.
        acceleration_cost = ((acceleration - 10.).clamp_min(0) / 30.).clamp_max(1)
        acceleration_cost = torch.where((self.ready >= 2)[:, None], acceleration_cost, 0.)
        cost = .75 * acceleration_cost.mean(-1) + .25 * reversal.float().mean(-1)
        self.position = positions.clone()
        self.velocity = velocity
        self.ready += 1
        self.age = torch.where(changed, 0, self.age + 1)
        self.supported = supported
        return cost, acceleration, reversal

    @staticmethod
    def contract():
        return dict(schema="foot_motion_cost_v1", acceleration_deadband_m_s2=10.,
                    acceleration_scale_m_s2=30., acceleration_weight=.75,
                    contact_reversal_weight=.25, contact_on_n=30., contact_off_n=15.,
                    rapid_reversal_seconds=.2, maximum_cost=1.,
                    activation="confirmed hit only; first lift/landing free; reset clears history",
                    sampling="control-rate finite differences of foot body positions")

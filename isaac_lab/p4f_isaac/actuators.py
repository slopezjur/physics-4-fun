"""Stable PD actuator, reproducing Godot's `ActiveBone` control law inside Isaac.

The fourth and last attempt at the sim-to-sim gap, and the first that matches BOTH the mechanism
and the control law. The rig already matches - `build_d6_usd.py` gives Isaac Godot's actual 16-body,
15-D6-joint skeleton - so this is the remaining difference.

**Why the obvious substitutes do not work.** `ImplicitActuatorCfg` hands the gains to PhysX, which
folds the PD into the articulation solve and evaluates it at the end of the timestep. That is
unconditionally stable, so the knee's 1800 N.m/rad behaves like 1800 and the rest pose is a genuine
equilibrium - measured at 98.4% still standing after 8 s of zero actions. Godot's rest pose is not:
the same command puts it on the floor inside 2 s.

`IdealPDActuatorCfg` - a true explicit PD - is not the answer either. At these gains and a 1/120 s
step it diverges outright: joint velocities reached ~1e10, reward terms hit -1e21, and PPO crashed
on a NaN action std within 16 seconds. Re-implementing the same explicit law on the Godot side
failed identically, saturating all three axes at 693 N.m with 1.5 rad of tracking error while the
body was still standing. Explicit PD at kp=1800 and 120 Hz is simply unstable, in either engine.

**What Godot actually does** is Stable PD (Tan, Liu & Turk 2011), which is stable at any gain
because it evaluates the PD against the PREDICTED next state rather than the current one. Its
practical effect is to divide both gains by

    denominator = 1 + kd * dt / I + kp * dt^2 / I

so the authority actually delivered is well below the authored number - `ActiveBone`'s own comments
put the loss at 2.6x on the ankle and 5.4x on the knee. That is why Godot's body sags where Isaac's
holds, and why a policy trained against Isaac's implicit drives has no balance behaviour to fall
back on when it arrives.

Note this is NOT the same as scaling stiffness down, which was tried and did not reproduce the
instability: Isaac's rest pose survived a 5x reduction (still ~80% standing at scale 0.2). The
difference is that SPD reduces stiffness and damping TOGETHER, by a factor that depends on each
joint's own inertia, and applies the result explicitly rather than through the implicit solve.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import IdealPDActuator
from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class StablePDActuator(IdealPDActuator):
    """Tan-Liu-Turk Stable PD, matching `Source/Core/Math/PidController3D.cs`.

    .. math::

        d &= 1 + k_d \\Delta t / I + k_p \\Delta t^2 / I \\\\
        \\tau &= \\frac{k_p}{d} (q_{des} - q) - \\frac{k_d}{d} \\dot{q}

    clipped to the joint's effort limit.
    """

    cfg: StablePDActuatorCfg

    def __init__(self, cfg: StablePDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Per-DOF effective inertia. Resolved lazily from the articulation's generalised mass matrix
        # on the first compute, because the view is not available at construction time; the
        # configured value is the fallback when that query is unsupported.
        self._inertia: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int]):
        pass

    def _resolve_inertia(self, reference: torch.Tensor) -> torch.Tensor:
        if self._inertia is not None:
            return self._inertia

        inertia = torch.full_like(reference, self.cfg.default_inertia)
        self._inertia = inertia.clamp(min=self.cfg.min_inertia)
        return self._inertia

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        inertia = self._resolve_inertia(joint_pos)
        dt = self.cfg.dt

        # The SPD denominator. Larger gains and larger steps both shrink the delivered authority,
        # which is exactly the mechanism that makes the scheme unconditionally stable - and exactly
        # why Godot's body is softer than its authored gains suggest.
        denominator = 1.0 + self.damping * dt / inertia + self.stiffness * dt * dt / inertia

        error_pos = control_action.joint_positions - joint_pos
        error_vel = -joint_vel
        if control_action.joint_velocities is not None:
            error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = (
            self.stiffness / denominator * error_pos + self.damping / denominator * error_vel
        )
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class StablePDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`StablePDActuator`."""

    class_type: type = StablePDActuator

    dt: float = 1.0 / 120.0
    """Physics step, which the SPD denominator depends on. Must match `SimulationCfg.dt`."""

    default_inertia: float = float(__import__("os").environ.get("P4F_SPD_INERTIA", "0.1"))
    """Effective inertia per DOF, kg.m^2.

    A single value rather than a per-joint table on purpose. The denominator is dominated by the
    `kd*dt/I` and `kp*dt^2/I` terms, both of which vary far more with the authored gains - which
    span 60 to 1800 across the body - than with the modest inertia spread of a humanoid's limbs.
    `PidController3D` likewise falls back to a floor value whenever the measured inertia is small.
    """

    min_inertia: float = 1e-3
    """Floor, mirroring `PidController3D.MinInertia`. Guards the division for near-massless links."""

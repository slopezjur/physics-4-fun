"""Walk under Newton/XPBD: follow a velocity command while staying upright.

Inherits `StandEnv` rather than restating it. The observation layout, action mapping, contact
handling and reset are shared verbatim, which is the precondition for bootstrapping this policy
from a Stand checkpoint (`train.py --init_from`).

Three things change, and each is forced by something measured rather than chosen:

1. **The reward is a product, not a sum.** Posture gates the task terms instead of being paid for
   separately - see `WalkEnvCfg` for the measurement that made this a project rule.
2. **Air time is derived, not sensed.** Newton has no working `ContactSensor` here, so
   `compute_first_contact` and `last_air_time` do not exist. Both are reconstructed from the same
   height proxy the observation's contact flags already use.
3. **Body-frame velocity is derived, not read.** `root_lin_vel_b` and `root_ang_vel_b` are frozen
   under XPBD and read a clean zero, so a reward built on them would pay a perfect tracking score
   to a body lying motionless on the floor. Everything comes through `NewtonRigState`.
"""

from __future__ import annotations

import torch

from p4f_newton.assets import CONTACT_BONES
from p4f_newton.state import quat_rotate_inverse, yaw_only
from p4f_newton.tasks.stand.stand_env import StandEnv

from .walk_env_cfg import WalkEnvCfg


class WalkEnv(StandEnv):
    cfg: WalkEnvCfg

    def __init__(self, cfg: WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Positions of the two feet within the contact-flag vector, resolved by name rather than
        # assumed to be the trailing pair. The order comes from the rig JSON, which is the
        # authoritative contract - prose about it has been wrong before.
        self._feet_slots = torch.tensor(
            [CONTACT_BONES.index("Foot_L"), CONTACT_BONES.index("Foot_R")],
            device=self.device,
            dtype=torch.long,
        )

        # Air-time bookkeeping, standing in for the ContactSensor that does not report here.
        self._feet_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._feet_grounded = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ commands

    def _resample_commands(self, env_ids) -> None:
        count = len(env_ids)
        if count == 0:
            return

        def uniform(bounds: tuple[float, float]) -> torch.Tensor:
            return torch.empty(count, device=self.device).uniform_(*bounds)

        command = torch.stack(
            [
                uniform(self.cfg.cmd_lin_vel_x),
                uniform(self.cfg.cmd_lin_vel_y),
                uniform(self.cfg.cmd_ang_vel_z),
            ],
            dim=-1,
        )
        # A slice of episodes commanded to hold still, so standing stays in the policy rather than
        # being trained out of it by a reward that only ever pays for moving.
        stand_still = torch.rand(count, device=self.device) < self.cfg.cmd_zero_probability
        command[stand_still] = 0.0
        self._command[env_ids] = command

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        # A fresh body is standing on both feet. Leaving the clock running would pay a spurious
        # first "step" on the first touchdown after the reset.
        self._feet_air_time[env_ids] = 0.0
        self._feet_grounded[env_ids] = True

    # ------------------------------------------------------------------ reward

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Posture-gated task terms plus additive costs.

        Extends `StandEnv._reward_terms` by key rather than rebuilding the dictionary. The 2.3.2
        Walk rebuilt it and silently lost `action_clip` - the barrier that stops the policy mean
        drifting outside the clip range - which every quality metric reported as fine.
        """
        terms = super()._reward_terms()
        dt = self.step_dt
        rig = self.rig

        # Stand's objective was to be motionless and upright. Both are now either the wrong
        # objective or the wrong shape, so they are deleted by key, visibly, rather than left to be
        # outweighed. `action_clip`, `joint_vel`, `effort` and `action_rate` survive untouched.
        del terms["alive"]  # paying for existence is what made the additive form stand still
        del terms["upright"]  # becomes a gate below, not a payment
        del terms["head_height"]  # likewise
        del terms["lin_vel"]  # planar motion is the objective now, not a cost
        del terms["ang_vel"]  # replaced below with a planar-only version; yaw is commanded

        quat = rig.root_quat_w
        # Heading-relative, matching observation slice [3:6]: the policy reads and is scored on the
        # same "forward", so a command of +x means the direction the dummy faces.
        lin_vel_b = quat_rotate_inverse(yaw_only(quat), rig.root_lin_vel_w)
        ang_vel_b = quat_rotate_inverse(quat, rig.root_ang_vel_w)

        posture = rig.upright().clamp(min=0.0) * self._height_reward()

        # One product, not a sum of two. See `WalkEnvCfg.rew_track`.
        obedience = self._drive(lin_vel_b) * self._yaw_factor(ang_vel_b)
        terms["track"] = self.cfg.rew_track * dt * posture * obedience
        terms["feet_air_time"] = self.cfg.rew_feet_air_time * posture * self._air_time_reward()

        terms["lin_vel_z"] = self.cfg.rew_lin_vel_z * dt * lin_vel_b[:, 2] ** 2
        terms["ang_vel_xy"] = self.cfg.rew_ang_vel_xy * dt * torch.sum(ang_vel_b[:, :2] ** 2, dim=1)
        terms["termination"] = self.cfg.rew_termination * self._fell.float()
        return terms

    def _drive(self, lin_vel_b: torch.Tensor) -> torch.Tensor:
        """How much of the commanded motion is actually being produced, in [0, 1].

        Speed along the commanded direction over the commanded speed, saturating at 1 and clamped
        at 0. Saturating matters because the first thing a policy finds is that diving forward
        produces speed; clamping at 0 matters because a negative factor would flip the sign of the
        posture gate it multiplies, making a backwards topple outscore a backwards step.

        A commanded stand-still is scored by a stillness kernel instead of by division: zero is a
        real objective for 15% of episodes, not a degenerate case.
        """
        cmd = self._command[:, :2]
        cmd_speed = torch.linalg.vector_norm(cmd, dim=1)
        moving = cmd_speed > self.cfg.cmd_deadband

        direction = cmd / cmd_speed.clamp(min=1e-6).unsqueeze(-1)
        along = torch.sum(lin_vel_b[:, :2] * direction, dim=1)
        tracking = (along / cmd_speed.clamp(min=1e-6)).clamp(0.0, 1.0)

        speed = torch.sum(lin_vel_b[:, :2] ** 2, dim=1)
        stillness = torch.exp(-speed / self.cfg.still_velocity_scale)
        return torch.where(moving, tracking, stillness)

    def _yaw_factor(self, ang_vel_b: torch.Tensor) -> torch.Tensor:
        """Yaw-rate tracking, in [0, 1].

        An exponential kernel rather than a negative squared error: error terms are unbounded below,
        so a policy that has fallen and can track nothing keeps accruing large negatives and the
        gradient ends up dominated by the worst episodes rather than the informative ones. Bounded
        in [0, 1] is also what lets this be multiplied by the posture gate.
        """
        error = (self._command[:, 2] - ang_vel_b[:, 2]) ** 2
        return torch.exp(-error / 0.25)

    # ------------------------------------------------------------------ air time

    def _air_time_reward(self) -> torch.Tensor:
        """Paid once per touchdown, for how much longer than the threshold that foot was airborne.

        **The sign is easy to get backwards, and it was, in 2.3.2.** Written as
        `(min(air_time, target) - target)` the term is <= 0 everywhere, so a foot that never leaves
        the ground registers no touchdown and scores 0 while a foot that actually steps scores
        negative - it penalises walking. Measured at -3.08 per episode against a +18.1 tracking
        term, quietly pushing the policy toward sliding.

        As written here it is positive for any step longer than the threshold and capped, so a long
        step is rewarded and hanging in the air is not worth progressively more.
        """
        grounded = self._contacts()[:, self._feet_slots] > 0.5
        touchdown = grounded & ~self._feet_grounded

        # Read before the update: at a touchdown `_feet_air_time` still holds the flight just ended.
        # **`min=0.0` is not decoration.** `clamp(max=cap)` alone caps only the top, so any flight
        # shorter than the threshold scores NEGATIVE and a foot that never leaves the ground scores
        # zero - the term then pays for sliding, which is the precise inversion described above.
        # Measured in the smoke run at -0.0197 per episode before this clamp was corrected.
        flight = (self._feet_air_time - self.cfg.feet_air_time_threshold).clamp(
            min=0.0, max=self.cfg.feet_air_time_cap
        )
        reward = torch.sum(flight * touchdown.float(), dim=1)

        self._feet_air_time = torch.where(
            grounded, torch.zeros_like(self._feet_air_time), self._feet_air_time + self.step_dt
        )
        self._feet_grounded = grounded

        # Only while a motion is commanded - otherwise the cheapest way to collect this is to march
        # on the spot when asked to stand still.
        commanded = torch.linalg.vector_norm(self._command[:, :2], dim=1) > self.cfg.cmd_deadband
        return reward * commanded.float()

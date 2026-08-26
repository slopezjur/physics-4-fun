"""Walk: follow a velocity command while staying upright.

Inherits `StandEnv` rather than restating it. The observation layout, action mapping, contact
handling and reset are shared verbatim, which is the precondition for bootstrapping this policy
from a Stand checkpoint (`train.py --init_from`). Only three things change: the command becomes
non-zero, the reward pays for tracking it, and a feet-air-time term pushes the result toward
stepping rather than sliding.

The Godot counterpart is `WalkForwardReward`, whose `HeadingWeight` is pinned to 0.0 because the
113-dim observation carries no heading. This layout reserves the command slots, so that
restriction does not apply here.
"""

from __future__ import annotations

import torch

from p4f_isaac.tasks.stand.stand_env import StandEnv

from .walk_env_cfg import WalkEnvCfg


class WalkEnv(StandEnv):
    cfg: WalkEnvCfg

    def __init__(self, cfg: WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._feet_ids, _ = self.contact_sensor.find_bodies(["Foot_L", "Foot_R"], preserve_order=True)

    def _resample_commands(self, env_ids) -> None:
        count = len(env_ids)
        if count == 0:
            return

        def uniform(bounds: tuple[float, float]) -> torch.Tensor:
            return torch.empty(count, device=self.device).uniform_(*bounds)

        command = torch.stack(
            [uniform(self.cfg.cmd_lin_vel_x), uniform(self.cfg.cmd_lin_vel_y), uniform(self.cfg.cmd_ang_vel_z)],
            dim=-1,
        )
        # A slice of episodes commanded to hold still, so standing stays in the policy rather than
        # being trained out of it by a reward that only ever pays for moving.
        stand_still = torch.rand(count, device=self.device) < self.cfg.cmd_zero_probability
        command[stand_still] = 0.0
        self._command[env_ids] = command

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Stand's terms, with the standing-still objective swapped for command tracking.

        Extends rather than replaces. The previous version rebuilt the whole dictionary and lost
        `action_clip` in the process - the barrier that stops the policy mean drifting outside the
        clip range - which every quality metric would have reported as fine. Anything Stand adds
        later arrives here automatically; anything Walk genuinely does not want is deleted by key,
        visibly.
        """
        terms = super()._reward_terms()
        data = self.robot.data
        dt = self.step_dt

        # Planar motion is now the objective, so Stand's penalty on it is wrong rather than merely
        # unhelpful. Vertical motion is still penalised, just far more weakly - see rew_lin_vel_z.
        del terms["lin_vel"]

        # Exponential kernels rather than negative squared error: error terms are unbounded below,
        # so a policy that has fallen and cannot track anything keeps accruing large negatives and
        # the gradient ends up dominated by the worst episodes instead of the informative ones.
        lin_vel_error = torch.sum((self._command[:, :2] - data.root_lin_vel_b[:, :2]) ** 2, dim=1)
        ang_vel_error = (self._command[:, 2] - data.root_ang_vel_b[:, 2]) ** 2

        terms["track_lin_vel"] = self.cfg.rew_track_lin_vel * dt * torch.exp(-lin_vel_error / 0.25)
        terms["track_ang_vel"] = self.cfg.rew_track_ang_vel * dt * torch.exp(-ang_vel_error / 0.25)
        terms["feet_air_time"] = self.cfg.rew_feet_air_time * self._air_time_reward()
        terms["lin_vel_z"] = self.cfg.rew_lin_vel_z * dt * data.root_lin_vel_b[:, 2] ** 2
        # Only the planar components: yaw is commanded, so penalising it fights track_ang_vel.
        terms["ang_vel"] = self.cfg.rew_ang_vel * dt * torch.sum(data.root_ang_vel_b[:, :2] ** 2, dim=1)
        return terms

    def _air_time_reward(self) -> torch.Tensor:
        """Paid once per touchdown, for how much longer than the threshold that foot was airborne.

        The sign matters and is easy to get backwards. Written as `(min(air_time, target) - target)`
        this term is <= 0 everywhere, so a foot that never leaves the ground registers no touchdown
        and scores 0 while a foot that actually steps scores negative - i.e. it penalises walking.
        Measured that way at -3.08 per episode against a +18.1 tracking term, quietly pushing the
        policy toward sliding.

        As written it is positive for any step longer than the threshold and capped, so long steps
        are rewarded but hanging in the air is not worth more and more.
        """
        first_contact = self.contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        air_time = self.contact_sensor.data.last_air_time[:, self._feet_ids]
        reward = torch.sum(
            (air_time - self.cfg.feet_air_time_threshold).clamp(max=self.cfg.feet_air_time_cap)
            * first_contact,
            dim=1,
        )
        # Only while a motion is commanded - otherwise the cheapest way to earn it is to march on
        # the spot when asked to stand still.
        return reward * (torch.norm(self._command[:, :2], dim=1) > 0.1).float()

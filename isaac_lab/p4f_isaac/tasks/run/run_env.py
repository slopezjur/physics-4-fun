"""Run: track a fast velocity command, with a flight phase.

Inherits `WalkEnv`. The only behavioural addition is a double-support penalty: the term that makes
running distinct from a fast walk. Everything about the observation, action mapping and reset is
shared, so this bootstraps from a Walk checkpoint.
"""

from __future__ import annotations

import torch

from p4f_isaac.tasks.walk.walk_env import WalkEnv

from .run_env_cfg import RunEnvCfg


class RunEnv(WalkEnv):
    cfg: RunEnvCfg

    def __init__(self, cfg: RunEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Walk's terms plus a double-support penalty - the term that separates a run from a
        fast walk.

        Both feet loaded at once is high by definition in a walk and near zero in a run. Applied
        only while a motion is commanded, so the stand-still episodes that keep deceleration in the
        policy are not punished for standing on two feet.
        """
        terms = super()._reward_terms()
        contacts = self.contact_sensor.data.net_forces_w[:, self._feet_ids, :].norm(dim=-1) > 1.0
        double = contacts.all(dim=1).float()
        double *= (torch.norm(self._command[:, :2], dim=1) > 0.1).float()
        terms["double_support"] = self.cfg.rew_double_support * self.step_dt * double
        return terms

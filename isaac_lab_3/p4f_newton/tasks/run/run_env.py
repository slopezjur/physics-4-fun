"""Run under Newton/XPBD: track a fast velocity command, with a flight phase.

Inherits `WalkEnv`. The only behavioural addition is a double-support penalty - the term that makes
running distinct from a fast walk. Observation, action mapping and reset are shared, so this
bootstraps from a Walk checkpoint.

Double support is measured from the same height proxy the contact flags and the air-time reward
use, because Newton's contact reporting does not surface through Isaac Lab's `ContactSensor` on this
backend. The 2.3.2 version read `net_forces_w` from that sensor, which is simply unavailable here.
"""

from __future__ import annotations

import torch

from p4f_newton.tasks.walk.walk_env import WalkEnv

from .run_env_cfg import RunEnvCfg


class RunEnv(WalkEnv):
    cfg: RunEnvCfg

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Walk's terms plus a double-support penalty.

        Both feet loaded at once is high by definition in a walk and near zero in a run. Applied
        only while a motion is commanded, so the stand-still episodes that keep deceleration in the
        policy are not punished for standing on two feet.
        """
        terms = super()._reward_terms()

        grounded = self._contacts()[:, self._feet_slots] > 0.5
        double = grounded.all(dim=1).float()
        double = double * (
            torch.linalg.vector_norm(self._command[:, :2], dim=1) > self.cfg.cmd_deadband
        ).float()
        terms["double_support"] = self.cfg.rew_double_support * self.step_dt * double
        return terms

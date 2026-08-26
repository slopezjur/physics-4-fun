"""PPO configuration for the Run task. Same network shape as Walk so the bootstrap loads."""

from __future__ import annotations

from isaaclab.utils import configclass

from p4f_isaac.tasks.walk.agents.rsl_rl_ppo_cfg import WalkPPORunnerCfg


@configclass
class RunPPORunnerCfg(WalkPPORunnerCfg):
    experiment_name = "p4f_run"
    max_iterations = 4000

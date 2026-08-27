"""PPO configuration for the Run task under Newton. Inherits Walk's network shape so a Walk
checkpoint can seed it."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from p4f_newton.tasks.walk.agents.rsl_rl_ppo_cfg import WalkPPORunnerCfg


@configclass
class RunPPORunnerCfg(WalkPPORunnerCfg):
    experiment_name = "p4f_newton_run"
    max_iterations = 8000

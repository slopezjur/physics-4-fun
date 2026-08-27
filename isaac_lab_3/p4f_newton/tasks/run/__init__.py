"""Run task registration."""

import gymnasium as gym

from . import agents
from .run_env import RunEnv
from .run_env_cfg import RunEnvCfg

gym.register(
    id="P4F-Dummy-Run-Newton-v0",
    entry_point="p4f_newton.tasks.run.run_env:RunEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RunEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RunPPORunnerCfg",
    },
)

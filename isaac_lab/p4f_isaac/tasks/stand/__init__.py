"""Stand task registration."""

import gymnasium as gym

from . import agents
from .stand_env import StandEnv
from .stand_env_cfg import StandEnvCfg

gym.register(
    id="P4F-Dummy-Stand-Direct-v0",
    entry_point="p4f_isaac.tasks.stand.stand_env:StandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandPPORunnerCfg",
    },
)

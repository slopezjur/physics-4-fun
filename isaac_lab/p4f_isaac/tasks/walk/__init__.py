"""Walk task registration."""

import gymnasium as gym

from . import agents
from .walk_env import WalkEnv
from .walk_env_cfg import WalkEnvCfg

gym.register(
    id="P4F-Dummy-Walk-Direct-v0",
    entry_point="p4f_isaac.tasks.walk.walk_env:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WalkEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkPPORunnerCfg",
    },
)

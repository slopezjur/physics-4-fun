"""Walk task registration."""

import gymnasium as gym

from . import agents
from .walk_env import WalkEnv
from .walk_env_cfg import WalkEnvCfg

# Distinct id from the 2.3.2 `P4F-Dummy-Walk-Direct-v0` on purpose. The observation layout is the
# same 143 floats, but it is produced against a different solver and in a different DOF order
# (42 of 45 slots differ). A checkpoint from one loaded into the other would run, and be nonsense.
gym.register(
    id="P4F-Dummy-Walk-Newton-v0",
    entry_point="p4f_newton.tasks.walk.walk_env:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WalkEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkPPORunnerCfg",
    },
)

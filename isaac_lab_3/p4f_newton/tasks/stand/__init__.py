"""Stand task registration."""

import gymnasium as gym

from . import agents
from .stand_env import StandEnv
from .stand_env_cfg import StandEnvCfg

# Distinct id from the 2.3.2 task on purpose. The two are NOT interchangeable: the observation
# layout is the same 143 floats, but they are produced against different solvers and, more
# importantly, in a different DOF order (42 of 45 slots differ). A checkpoint from one loaded into
# the other would run, and be nonsense.
gym.register(
    id="P4F-Dummy-Stand-Newton-v0",
    entry_point="p4f_newton.tasks.stand.stand_env:StandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandPPORunnerCfg",
    },
)

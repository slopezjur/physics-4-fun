"""Perturbation task registration."""

import gymnasium as gym

from . import agents
from .perturb_env import PerturbEnv
from .perturb_env_cfg import PerturbEnvCfg

gym.register(
    id="P4F-Dummy-Perturb-Direct-v0",
    entry_point="p4f_isaac.tasks.perturb.perturb_env:PerturbEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PerturbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PerturbPPORunnerCfg",
    },
)

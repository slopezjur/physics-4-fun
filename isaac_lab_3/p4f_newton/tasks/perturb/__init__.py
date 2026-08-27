"""Perturbation task registration."""

import gymnasium as gym

from . import agents
from .perturb_env import PerturbEnv
from .perturb_env_cfg import PerturbEnvCfg

# Distinct id from the 2.3.2 `P4F-Dummy-Perturb-Direct-v0`. Same 143-float layout, but produced
# against a different solver and in a different DOF order - a checkpoint from one loaded into the
# other would run, and be nonsense.
gym.register(
    id="P4F-Dummy-Perturb-Newton-v0",
    entry_point="p4f_newton.tasks.perturb.perturb_env:PerturbEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PerturbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PerturbPPORunnerCfg",
    },
)

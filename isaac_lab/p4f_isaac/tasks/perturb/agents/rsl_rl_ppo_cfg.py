"""PPO configuration for the Perturbation task.

Same network shape as Stand - a requirement, not a convenience, since this task is initialised
from a Stand checkpoint and the state dict must load.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from p4f_isaac.tasks.stand.agents.rsl_rl_ppo_cfg import StandPPORunnerCfg


@configclass
class PerturbPPORunnerCfg(StandPPORunnerCfg):
    experiment_name = "p4f_perturb"
    max_iterations = 6000

    def __post_init__(self):
        super().__post_init__()
        # The policy arrives already able to stand. A full-size first step would undo that before
        # it has seen a single disturbance.
        self.algorithm.learning_rate = 5.0e-4

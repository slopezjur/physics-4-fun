"""PPO configuration for the Walk task.

Identical network shape to Stand, and that is a requirement rather than a convenience: Walk is
initialised from a Stand checkpoint, so the layer dimensions have to match exactly or the state
dict will not load.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from p4f_isaac.tasks.stand.agents.rsl_rl_ppo_cfg import StandPPORunnerCfg


@configclass
class WalkPPORunnerCfg(StandPPORunnerCfg):
    experiment_name = "p4f_walk"
    max_iterations = 6000

    def __post_init__(self):
        super().__post_init__()
        # Lower than Stand's 1e-3. The policy arrives already competent at not falling over;
        # a large first step would undo that before the tracking terms have taught it anything.
        self.algorithm.learning_rate = 5.0e-4

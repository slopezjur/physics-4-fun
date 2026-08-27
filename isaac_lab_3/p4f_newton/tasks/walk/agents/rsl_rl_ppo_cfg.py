"""PPO configuration for the Walk task under Newton.

Inherits Stand's configuration so the network shape is identical. That is not tidiness: Walk is
bootstrapped from a Stand checkpoint, and `runner.load` will only accept those weights while every
layer width matches. Changing `hidden_dims` here would not raise a useful error - it would raise a
state-dict size mismatch several minutes into a run.

Only the experiment name and two exploration knobs differ.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from p4f_newton.tasks.stand.agents.rsl_rl_ppo_cfg import StandPPORunnerCfg


@configclass
class WalkPPORunnerCfg(StandPPORunnerCfg):
    experiment_name = "p4f_newton_walk"
    max_iterations = 6000

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # A little wider than the 0.244 a converged Stand decays to, and no more. The reasoning for
        # going wider was sound - a bootstrapped policy sits at a sharp local optimum and the product
        # reward pays nothing for holding it - but 0.4 was too much: sampling that noisily across 36
        # joint targets destroys the balance the bootstrap was supposed to provide, and mean episode
        # length fell from 50 steps back to 27, worse than doing nothing at all.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.28),
    )

"""PPO configuration for the Perturbation task under Newton.

Inherits Stand's so the network shape is identical - the precondition for bootstrapping from a Stand
checkpoint, which `runner.load` enforces as a state-dict size match rather than a useful error.

**One deliberate difference: the standard deviation is parameterised in LOG space.** rsl_rl's default
`std_type="scalar"` learns std directly, and a large enough policy update drives it negative, at
which point sampling dies with `RuntimeError: normal expects all elements of std >= 0.0`. Measured
here: training is stable with an impulse ceiling of 12 N.s and crashes at 20, because a bigger
disturbance means bigger advantages and bigger updates. Capping the disturbance to suit the
parameterisation would be backwards - the real Godot shot is 18 N.s and has to be inside the
training distribution. `exp(log_std)` cannot go negative, so the ceiling is free again.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from p4f_newton.tasks.stand.agents.rsl_rl_ppo_cfg import StandPPORunnerCfg


@configclass
class PerturbPPORunnerCfg(StandPPORunnerCfg):
    experiment_name = "p4f_newton_perturb"
    max_iterations = 6000

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.25, std_type="log"),
    )

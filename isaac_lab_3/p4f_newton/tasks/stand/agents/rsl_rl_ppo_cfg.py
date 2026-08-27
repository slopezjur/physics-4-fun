"""PPO configuration for the Stand task under Newton.

Ported from the 2.3.2 task with the same hyperparameters, restated in the schema rsl-rl 5.x uses:
`policy = RslRlPpoActorCriticCfg(actor_hidden_dims=..., actor_obs_normalization=...)` became
separate `actor` / `critic` `RslRlMLPModelCfg` entries, and `init_noise_std` moved into the actor's
distribution config. The old form still imports and fails at runtime with `KeyError: 'class_name'`,
which names nothing.

`obs_normalization` is on deliberately, and it is not a tuning choice. rsl_rl's ONNX exporter folds
the empirical normalizer *into the exported graph*, so Godot receives a network that takes raw
observations and needs no normalisation statistics shipped alongside it. Turning it off would
export a policy that silently expects pre-normalised input — a mismatch that produces a dummy which
twitches rather than an error message.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class StandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "p4f_newton_stand"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        # 143 observations into 36 actions. Wider than the 128-unit flat-terrain locomotion nets
        # because the action space is nearly triple theirs, and narrow enough that the forward pass
        # stays cheap next to the physics step.
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # 0.25, not the usual 1.0. Actions span each joint's entire range, so std=1.0 commands
        # near-limit poses at random from the first step — measured at 0.5 s episodes against the
        # 2.8 s a zero-action policy survives from the same pose. Starting tighter keeps early
        # exploration near the rest pose, which is the only region where the "did not fall over"
        # signal exists at all; PPO widens it again if that pays.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.25),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Non-zero, unlike the stock humanoid config. Standing has an obvious local optimum — lock
        # every joint rigid and hope — and a little entropy pressure keeps the policy exploring
        # postures long enough to find one that actually rejects disturbances.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

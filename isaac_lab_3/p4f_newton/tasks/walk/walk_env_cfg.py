"""Configuration for the Walk task under Newton/XPBD.

Inherits `StandEnvCfg` wholesale and changes only what walking needs. That is deliberate and it is
load-bearing: Walk is bootstrapped from a Stand checkpoint, and the bootstrap is valid only while
the observation and action layouts are byte-identical. Anything that would change a width belongs
in `StandEnvCfg`, where both tasks see it, not here.

Two things the 2.3.2 Walk config had are deliberately absent:

* **`ContactSensorCfg`.** Newton's contact reporting does not surface through Isaac Lab's
  `ContactSensor` on this backend, so `track_air_time` has nothing to track. `WalkEnv` derives air
  time from the same height proxy the contact observation flags already use.
* **The additive reward weights.** See below — the shape of the reward changed, not just its
  numbers, so carrying the old weights across would have been a false comparison.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from p4f_newton.tasks.stand.stand_env_cfg import StandEnvCfg


@configclass
class WalkEnvCfg(StandEnvCfg):
    # Longer than Stand's 8 s. A gait is a cycle, and an episode has to be long enough to contain
    # several of them for the return to distinguish "walked" from "got lucky falling forwards".
    episode_length_s = 12.0

    # Much gentler than Stand's 0.6. Stand needs a shove because without one, standing still IS the
    # optimal policy and a statue scores full marks. Walk does not have that problem - its product
    # reward already pays exactly zero for standing still under a movement command - so the shove
    # buys little and costs a lot. Measured with Stand's value inherited: mean episode length fell
    # from 50 back to 27 steps, i.e. the bootstrapped policy was going down faster than a passive
    # body, which is not a gait problem at all.
    push_velocity = 0.25
    push_ang_velocity = 0.4

    # --- command sampling ---
    # Modest ranges. The dummy is 80.6 kg with 0.38 m thighs; asking for 2 m/s before it can hold a
    # gait at 0.5 m/s just trains a policy that dives forward and calls the fall progress.
    cmd_lin_vel_x = (-0.3, 1.0)
    cmd_lin_vel_y = (-0.3, 0.3)
    cmd_ang_vel_z = (-0.5, 0.5)
    # Share of episodes commanded to hold perfectly still. Without them the policy forgets how to
    # stand, which is the one behaviour already paid for in full.
    cmd_zero_probability = 0.15
    # Below this commanded speed an episode counts as a stand-still command.
    cmd_deadband = 0.1

    # --- reward shape: PRODUCT, not additive ---
    #
    # `docs/RL-SESSION-INVARIANTS.md` records this as a rule for Walk specifically, and
    # `docs/RL-TRAINING.md` records the measurement behind it. An additive reward pays an `alive`
    # and `upright` term for merely existing, so standing still holds a positive score worth
    # protecting, and the path to walking descends before it climbs: a real step risks a fall while
    # creeping forward pays almost nothing. Measured in Godot, forward speed went 0.0186 -> 0.0069
    # under the additive form. It did not fail to learn; it learned to stand still, quickly.
    #
    # So posture is a GATE, not a payment:
    #
    #     reward = w_track x posture x drive x yaw + w_air x posture x steps + penalties
    #     posture = clamp(upright, 0, 1) x height_factor       every factor in [0, 1]
    #
    # No factor is collectable without the others. Yaw tracking is a FACTOR rather than its own
    # term for that reason: measured in the smoke run, a separate additive `turn` term paid a
    # frozen upright body about +7 per episode for not spinning, which is exactly the comfortable
    # score worth protecting that the product form exists to remove. As a factor it still rewards
    # turning in place - a zero linear command scores `drive` through the stillness kernel, so the
    # product reduces to yaw tracking alone.
    rew_track = 5.0
    rew_feet_air_time = 1.0

    # `drive` saturates at the commanded speed rather than growing with it. The first thing a policy
    # discovers is that diving forward produces speed; capping means exceeding the command buys
    # nothing, so the only way to score higher is to SUSTAIN. It also clamps at 0 rather than going
    # negative, because a negative factor would flip the sign of the posture gate and make walking
    # backwards while toppling score better than walking backwards while upright.
    #
    # Stand-still commands are scored by a stillness kernel instead, so a zero command is a real
    # objective rather than a division by zero.
    still_velocity_scale = 0.05

    # --- penalties, additive and small ---
    # These stay additive on purpose: they are costs, and a cost that multiplies a gate can be
    # dodged by driving the gate to zero.
    #
    # `rew_lin_vel` is gone entirely: Stand penalises planar motion, which is now the objective.
    # Vertical motion is still penalised, and yaw is NOT, because yaw is commanded.
    rew_lin_vel_z = -1.0
    rew_ang_vel_xy = -0.05
    # A third of Stand's -0.01, and the smoke run is why. Under the product form the positive terms
    # are worth nothing until the policy can actually walk, while `action_rate` accrues every step
    # regardless: measured at -1.71 against a total of -1.72, i.e. the entire reward. A gait also
    # has genuinely larger action deltas than a stand, so Stand's weight prices normal walking as a
    # fault. Left alone, the cheapest policy is to freeze - the failure this reward shape is
    # supposed to make impossible, reintroduced through a penalty.
    rew_action_rate = -0.003
    rew_joint_vel = -1.0e-4
    rew_effort = -0.02

    # **Zero, deliberately.** Under a product reward a fall penalty is actively harmful: early in
    # training the policy cannot walk, so everything scores near zero, and a negative termination
    # makes standing still (0) strictly better than attempting anything (risking -10) — the exact
    # risk aversion the product form exists to remove. The real cost of falling is the forfeited
    # remainder of the episode, which already scales with how much there is to lose.
    rew_termination = 0.0

    # A step must clear this much air before it earns anything, and stops earning beyond the cap.
    # The threshold separates a real step from a foot skimming the floor; the cap stops the term
    # paying for hanging in the air as long as possible.
    feet_air_time_threshold = 0.2
    feet_air_time_cap = 0.3

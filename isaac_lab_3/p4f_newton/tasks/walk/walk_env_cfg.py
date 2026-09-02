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

    # Blend factor for the forward-progress average used by `_drive`.
    #
    # **0.008, a ~2 s time constant at 60 Hz, widened from 0.02 (~0.83 s).** At 0.83 s the loophole
    # was narrowed but not closed: measured over two legs, `mean drive` kept climbing to 0.83 while
    # net displacement FELL from 0.052 to 0.036 m/s - an oscillation slower than the window still
    # gets paid for its forward half. The averaging window has to be longer than the slowest rocking
    # the policy can find, not merely longer than a stride.
    drive_smoothing = 0.008

    # --- command curriculum ---
    # **A fixed command range lets the policy harvest its easy end.** `_drive` scores achieved over
    # commanded, so a 0.15 m/s command is satisfied trivially while 0.8 is not. Measured over five
    # chained legs: upright at a 0.4 m/s command rose 10% -> 85% and episode length 143 -> 393,
    # while steps FELL 79 -> 33 and distance fell 0.47 -> 0.36 m. The policy was not learning to
    # walk; it was learning to stand still through the movement commands it could not satisfy and
    # collect on the small ones it could.
    #
    # This is the same mistake the Perturb impulse ceiling made - sizing the demand against the
    # target scenario rather than against what the policy can currently do. Same remedy: sample
    # inside a ceiling that only widens once the policy is actually tracking near the top of it.
    cmd_curriculum = True

    # Where the ramp starts, and the top it may reach (the configured `cmd_lin_vel_x` upper bound).
    # Negative start means "use cmd_speed_min".
    cmd_speed_start = -1.0
    cmd_speed_min = 0.25
    cmd_speed_max = 1.0

    # Episodes with a moving command, near the ceiling, before one decision. Same sizing logic as
    # Perturb's window: matched to how fast the policy learns, not how fast episodes finish.
    cmd_curriculum_window = 16384

    # An episode counts as "near the ceiling" when its commanded speed reached this fraction of it,
    # and counts as a success when it did not fall AND averaged at least `cmd_success_drive` of the
    # commanded speed. Tracking is what has to improve; staying upright is necessary, not sufficient
    # - a policy that stands still through the command satisfies uprightness and learns nothing.
    cmd_curriculum_band = 0.8
    cmd_success_drive = 0.5

    cmd_curriculum_raise_above = 0.60
    cmd_curriculum_lower_below = 0.30
    cmd_curriculum_raise_factor = 1.08
    cmd_curriculum_lower_factor = 0.90

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
    # **10.0, raised from 1.0.** At the measured flight time the term was contributing about 0.05
    # per episode against a `track` term of 13 - arithmetically incapable of changing behaviour
    # whatever its threshold. Weight and threshold had to move together; either alone does nothing.
    rew_feet_air_time = 10.0

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
    #
    # **0.10, lowered from 0.20, because at 0.20 the term paid exactly zero.** Measured over two
    # chained 15-minute legs (1,400 iterations): `Episode_Reward/feet_air_time` was 0.000 in every
    # single window while `track` grew 7.0 -> 8.5. The evaluator meanwhile counted ~40 touchdowns
    # per 12 s episode, about 1.7 Hz per foot - so the feet ARE lifting, in fast shuffles whose
    # flight never reaches 0.2 s.
    #
    # A threshold that gates all reward until the behaviour already exists supplies no gradient
    # toward it: the term was dead weight, not a shaping signal. 0.10 s is still a real lift rather
    # than a skim, and it starts paying for the longer end of the shuffles the policy already
    # produces, which is what gives it somewhere to climb. The cap moves with it so the payable
    # band keeps its width.
    # **Set from the measured distribution, not guessed.** Mean flight at touchdown is 0.023 s -
    # 1.4 policy steps at 60 Hz, a foot leaving the floor for barely one tick. The threshold was
    # 0.20 and then 0.10; both were far above anything the gait produces, so the term paid exactly
    # zero for every leg of this session and supplied no gradient at all. I changed it twice by
    # guessing before measuring it, which is the mistake this comment exists to prevent.
    #
    # 0.02 sits just under the current distribution so a longer-than-typical step earns something,
    # and the cap at 0.15 leaves a long way to climb - the reward grows all the way from a skim to
    # a real swing phase instead of saturating immediately.
    feet_air_time_threshold = 0.02
    feet_air_time_cap = 0.15

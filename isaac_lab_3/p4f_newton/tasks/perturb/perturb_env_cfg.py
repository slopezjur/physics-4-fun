"""Configuration for the Perturbation task under Newton/XPBD.

Ported from `isaac_lab/p4f_isaac/tasks/perturb/perturb_env_cfg.py`, which in turn ported
`Source/RL/Perturbation/BallGun.cs`. The physics homework carries across unchanged, because it is
about the rig rather than the solver:

    small ball  3.0 kg @ 6 m/s = 18.0 N.s  -> a uniformly chosen bone   <- what actually fires
    heavy ball  0.75 kg @ 6 m/s = 4.5 N.s  -> always the chest

`BallGun` sets `SmallBallProbability = 1.0`, so every Godot shot is the 18 N.s one. Its own
arithmetic against this rig (80.6 kg, CoM 0.840 m, I about the toe line 69.5 kg.m^2, impact near
1.25 m) puts the energy to tip a PASSIVE body over its toe edge at 6.74 J.

Inherits `StandEnvCfg`, so every fidelity fix comes with it - the effort limit XPBD ignores, the
per-bone vector torque clamp, the Hill force-velocity ceiling, the joint-velocity clip, and the
balance assist. That inheritance is also what makes a bootstrap from a Stand checkpoint valid: the
observation and action layouts are identical.

Two departures from the Godot implementation, both kept from the 2.3.2 port:

**No PHYSICAL ball is spawned.** At thousands of environments a projectile means thousands of extra
bodies and colliders to deliver what is physically an impulse. The equivalent impulse goes straight
onto the target body - same momentum, same lever arm about the ankles, none of the cost. The
vertical aim jitter survives as the torque an off-centre hit produces.

A cosmetic marker ball IS drawn whenever something is rendering - see `show_impact_ball`. The
earlier wording here said only "no ball is spawned", which read as a decision not to draw one at
all, and the marker 2.3.2 had was lost in this port on the strength of that sentence.

**The impulse is sampled per episode rather than fixed.** Uniform over `[0, max]` is a curriculum
that needs no state and cannot stall: easy episodes always exist so there is always a gradient,
while the hard tail keeps arriving.
"""

from __future__ import annotations

import os

from isaaclab.utils.configclass import configclass

from p4f_newton.tasks.stand.stand_env_cfg import StandEnvCfg


@configclass
class PerturbEnvCfg(StandEnvCfg):
    episode_length_s = 8.0

    # --- impulse ---
    # **Deliberately lower than the 2.3.2 ceiling of 50 N.s.** That number was set for a policy
    # trained against drives with no effort limit, which had margin this one does not: XPBD ignores
    # `joint_effort_limit`, so the 2.3.2 Newton-era Stand held itself up on a fraction of its
    # nominal torque and could absorb shoves no real actuator would. With the limit enforced and the
    # torque clamped per bone as a vector, the honest ceiling is far lower.
    #
    # **10 N.s, and 25 was measurably wrong.** A sampled curriculum only works while a useful share
    # of episodes are survivable; past that the policy correctly learns that nothing it does
    # matters. At a ceiling of 25 the mean shot was 12.3 N.s against a policy measured to survive 8
    # and fall at 10 - so most episodes were unwinnable, and a seeded Stand collapsed monotonically
    # over one 15-minute run: mean episode length 274 -> 150 -> 77 -> 41 -> 32.
    #
    # This is the same overshoot as `effort_scale_range`, made again. The rule that keeps being
    # relearned: **size a disturbance against what the policy can currently survive, not against
    # what the target scenario throws.** The real Godot shot is 18 N.s and is NOT yet inside this
    # range - it should be raised toward it as the measured survival rate improves, which is a
    # curriculum, not a fixed setting.
    push_impulse_range = (
        (lambda m: (0.0, m))(float(os.environ.get("P4F_PUSH_MAX", "10.0")))
    )

    # Bones a shot may target, uniform per episode - `BallGun.SmallBallTargetBones` verbatim.
    #
    # A limb hit is a genuinely different disturbance from a torso shove: less effective mass behind
    # it, a longer moment arm, and it can catch a leg mid-swing while the policy is already
    # committed to a foot placement. A policy trained only on chest shots has never seen that.
    push_target_bones = (
        "Head", "Chest", "Spine", "Pelvis",
        "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
        "Thigh_L", "Shin_L", "Thigh_R", "Shin_R",
    )

    # `BallGun.AimHeightJitter`. An off-centre hit is not a pure force - it adds r x F, which is
    # most of what makes a shove hard to reject.
    push_height_jitter = 0.25

    # First shot, then one every `push_interval_s` after it. This is the ARENA cadence, and BallGun
    # is explicit that training and arena differ: it ships `IntervalSeconds = 10` so training gets
    # exactly one hit - "one perturbation, one recovery, one outcome". That argument is about credit
    # assignment for a reward that scores the recovery; this task's reward is Stand's per-step
    # upright term, scored continuously, so repeated hits are not ambiguous in the same way.
    # Set `push_interval_s = 0` for one shot per episode.
    push_first_s = 2.0
    push_interval_s = 3.0

    # A fixed direction is learnable as a single scripted counter-lean, which is not balance.
    push_randomize_direction = True

    # --- disturbance already in the base task ---
    # Stand's own spawn shove is redundant here and would confound the measurement: the point is to
    # attribute a fall to a KNOWN impulse at a known time.
    push_velocity = 0.0

    # Same reward as Stand - the objective is unchanged, something is just pushing back. Success
    # deliberately does not end the episode: if it did, the episode would end before the hit landed
    # and the perturbation would never be experienced.
    rew_termination = -10.0

    # --- arena visualisation ---
    # Draw the incoming ball. Purely cosmetic: a marker with no collider and no mass, absent from
    # the observation and from every reward term, so a policy cannot react to it and training is
    # bit-for-bit unaffected.
    #
    # Without it the arena shows a dummy being knocked around by nothing - unreadable now the target
    # bone varies per episode, since there is no way to tell WHEN a shot fired or WHERE it came
    # from. Automatically inactive when nothing is rendering, so headless training never builds it.
    show_impact_ball = True

    # `BallGun.SmallBallRadius` / `SmallBallSpeed` / `SmallBallSpawnDistance`. The marker flies in at
    # the real launch speed from the real spawn distance, so what you see is when and where the hit
    # actually lands rather than a decorative approximation of it.
    ball_radius = 0.06
    ball_speed = 6.0
    ball_spawn_distance = 2.0

    # How long the rebounding ball stays visible after the hit, so an impact is not a single frame
    # you can blink through.
    ball_trail_s = 0.35

    # Rebound speed as a fraction of the launch speed. A ball bouncing off 80 kg of dummy does not
    # come away as fast as it arrived; this is cosmetic, NOT a restitution coefficient - nothing
    # here is simulated.
    ball_rebound_speed_factor = 0.45

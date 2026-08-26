"""Configuration for the Perturbation task.

Ported from `Source/RL/Perturbation/BallGun.cs`, which did the physics homework already:

    ball 0.75 kg at 6 m/s  ->  4.5 N.s of linear impulse, delivered at the chest

That file's own arithmetic, against this rig's real numbers (80.6 kg, CoM 0.840 m, I about the toe
line 69.5 kg.m^2, chest impact at 1.250 m), puts the energy needed to tip a *passive* body over its
toe edge at 6.74 J, and the 0.75 kg shot at roughly 0.05x of it. It also records why the original
3 kg at 9 m/s was abandoned: at 1.75x tipping energy no balance policy can absorb the hit, so
training on it yields a flat zero success rate with no gradient - the same failure as a curriculum
rung set past the cliff.

Two deliberate departures from the Godot implementation:

**No ball is spawned.** At 4096 environments a projectile means 4096 extra rigid bodies and
colliders to deliver what is physically an impulse. The equivalent impulse is applied directly to
the `Chest` body: same momentum, same lever arm about the ankles, none of the cost. The vertical
aim jitter is preserved as the torque an off-centre hit produces.

**The impulse is sampled per episode instead of fixed.** Uniform over `[0, max]` is a curriculum
that needs no state and cannot get stuck: easy episodes always exist, so there is always a
gradient, while the hard tail keeps arriving. `BallGun` names impulse as "the natural axis for a
difficulty curriculum" and this is that axis, sampled rather than scheduled.

The ceiling is set well above the Godot shot on purpose. The Stand policy trained here holds
itself up on ~6% of its available torque, so it has margin the Godot policy did not; 15 N.s is
about 0.4x the passive tipping energy - a real disturbance that must be actively nulled, still
short of a free topple.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from p4f_isaac.tasks.stand.stand_env_cfg import StandEnvCfg


@configclass
class PerturbEnvCfg(StandEnvCfg):
    episode_length_s = 8.0

    # --- impulse ---
    # BallGun fires two shot types and `SmallBallProbability = 1.0`, so in the current Godot config
    # EVERY shot is the "small" ball - small in radius (0.06 m), not in mass:
    #
    #     heavy  0.75 kg @ 6 m/s =  4.5 N.s  -> always the chest
    #     small  3.0  kg @ 6 m/s = 18.0 N.s  -> a uniformly chosen bone   <- what actually fires
    #
    # DOUBLED from the Godot ball: 6.0 kg @ 6 m/s = 36 N.s. The previous range topped out at 25 and
    # the policy was flat across all of it - 97.5% survival at 18 N.s with 3.9 degrees of tilt, and
    # it only started losing at 35. Training against a disturbance it already absorbs teaches
    # nothing, so the ceiling moves to 50: that puts the doubled shot comfortably inside the
    # distribution and leaves a hard tail above it.
    #
    # For scale, BallGun's own arithmetic puts 6.74 J as the energy needed to tip a PASSIVE body
    # over its toe edge. 36 N.s at the 1.25 m chest line is about 2.1x that, and 50 N.s is ~4x - so
    # the top of this range is a shove no passive body could survive, which is the point.
    push_impulse_range = (0.0, 50.0)

    # Bones a shot may target, uniform per episode - BallGun.SmallBallTargetBones verbatim.
    #
    # The first port hit the chest and nothing else, which is faithful only to the *heavy* ball,
    # the one that never fires. A limb hit is a genuinely different disturbance: less effective
    # mass behind it, a longer moment arm, and it can catch a leg mid-swing while the policy is
    # already committed to a foot placement. A policy trained only on torso shoves has never seen
    # that case.
    push_target_bones = (
        "Head", "Chest", "Spine", "Pelvis",
        "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
        "Thigh_L", "Shin_L", "Thigh_R", "Shin_R",
    )
    # BallGun.AimHeightJitter. An off-centre hit is not a pure force - it adds the torque r x F,
    # which is most of what makes a shove hard to reject.
    push_height_jitter = 0.25
    # First shot, then one every push_interval_s after it, for the rest of the episode.
    #
    # This is the ARENA cadence, and BallGun is explicit that the two differ: it ships
    # IntervalSeconds = 10 against a shorter episode specifically so training gets exactly one hit -
    # "one perturbation, one recovery, one outcome. With several hits per episode there is no way
    # to tell which one caused the fall" - while the arena overrides the interval because there the
    # point is to watch repeated recoveries.
    #
    # That argument is about credit assignment and it is a good one, but it applies to a reward
    # that scores the recovery. This task's reward is Stand's: a per-step upright term, scored
    # continuously rather than attributed to an event. Repeated hits are therefore not ambiguous
    # here in the way they would be for a get-up reward - they just make the disturbance ongoing.
    # Set push_interval_s = 0 to go back to one shot per episode.
    push_first_s = 2.0
    push_interval_s = 3.0

    # --- arena visualisation ---
    # Draw the incoming ball. Purely cosmetic: a marker prim with no collider and no mass, so it
    # cannot touch the dummy and costs nothing in physics. It exists because the impulse is applied
    # directly to a body, which makes a correct simulation look like the dummy being shoved by
    # nothing - unwatchable in an arena, especially now the target varies per episode.
    #
    # Automatically inactive when there is no GUI, so headless training is untouched.
    show_impact_ball = True
    # BallGun.SmallBallRadius and SmallBallSpeed. The marker flies in at the real launch speed and
    # from the real spawn distance, so what you see is when and where the hit lands.
    ball_radius = 0.06
    ball_speed = 6.0
    ball_spawn_distance = 2.0
    # How long the rebounding ball stays visible after the hit, so an impact is not a single frame
    # you can blink through.
    ball_trail_s = 0.35
    # Rebound speed as a fraction of the launch speed. A ball bouncing off 80 kg of dummy does not
    # come away as fast as it arrived; this is cosmetic, not a restitution coefficient.
    ball_rebound_speed_factor = 0.45

    # Random compass direction per episode, matching BallGun.RandomizeDirection, and for the reason
    # it gives: a fixed direction is learnable as a single scripted counter-lean, which is not
    # balance.
    push_randomize_direction = True

    # --- reward ---
    # Same shape as Stand: the objective is unchanged, something is just pushing back. Success is
    # deliberately NOT absorbing here - the episode always runs its full window and the per-step
    # upright term does the scoring - which is the same choice UprightTermination makes for this
    # task, and for the same reason: if success ended the episode it would end before the hit
    # landed, and the perturbation would never be experienced at all.
    rew_termination = -10.0

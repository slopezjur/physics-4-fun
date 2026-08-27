"""Configuration for the Run task under Newton/XPBD.

Run is Walk with a faster command and a gait term retuned for flight, not a new environment. It
inherits `WalkEnvCfg` wholesale so the observation and action layouts stay identical - the same
condition that lets it bootstrap from a Walk checkpoint.

**The interesting difference between walking and running is not speed, it is the flight phase.**
Walking always has at least one foot down; running has an interval with neither. Two settings
express that: a longer air-time threshold, and a penalty for both feet being grounded at once, which
Walk's reward has no opinion about.

The 2.3.2 attempt reached a fast walk - 1.86 m/s with only **0.5% flight** - and never crossed the
gait transition, because nothing in its reward paid for leaving the ground. `rew_double_support` is
that missing term.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from p4f_newton.tasks.walk.walk_env_cfg import WalkEnvCfg


@configclass
class RunEnvCfg(WalkEnvCfg):
    episode_length_s = 12.0

    # Walk tops out at 1.0 m/s. This starts above a comfortable walk and runs to a speed the rig
    # cannot reach by walking - 0.38 m thighs at 80.6 kg - so the gait has to change to earn it.
    cmd_lin_vel_x = (0.8, 2.5)
    cmd_lin_vel_y = (-0.2, 0.2)
    cmd_ang_vel_z = (-0.4, 0.4)
    # Lower than Walk's 0.15: standing still is not what this task is for, and Stand and Walk
    # already cover it. Not zero, because a policy that cannot decelerate is not useful.
    cmd_zero_probability = 0.05

    # --- reward ---
    # Tracking dominates harder than in Walk. At these speeds a posture-heavy reward is satisfiable
    # by refusing to accelerate, and the policy will take that deal. The product form already makes
    # standing still score zero under a movement command, so this raises the ceiling rather than
    # rebalancing against a competing term.
    rew_track = 6.0
    rew_feet_air_time = 1.5

    # Longer strides, and stop paying beyond a plausible flight time.
    feet_air_time_threshold = 0.25
    feet_air_time_cap = 0.45

    # **New in Run, and the term that separates a run from a fast walk.** Both feet loaded at once
    # is high by definition in a walk and near zero in a run; without a cost on it the policy
    # converges on the fastest possible walking gait, which is a local optimum it has no reason to
    # leave. Additive rather than a factor: it is a cost, and a cost folded into the product gate
    # could be dodged by driving the gate to zero.
    rew_double_support = -0.5

    # Vertical motion is part of running rather than a defect, so Walk's penalty is relaxed rather
    # than kept - at -1.0 it directly opposes the flight phase the task is trying to find.
    rew_lin_vel_z = -0.25

"""Configuration for the Run task.

Run is Walk with a faster command and a gait term retuned for flight, not a new environment. It
inherits `WalkEnv` wholesale so the observation and action layouts stay identical - the same
condition that lets it bootstrap from the Walk checkpoint.

The interesting difference between walking and running is not speed, it is the **flight phase**:
walking always has at least one foot down, running has an interval with neither. Two settings
express that here - a longer air-time threshold, and a much lower tolerance for both feet being
grounded at once, which Walk's reward has no opinion about.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from p4f_isaac.tasks.walk.walk_env_cfg import WalkEnvCfg


@configclass
class RunEnvCfg(WalkEnvCfg):
    episode_length_s = 12.0

    # Walk tops out at 1.0 m/s. This starts above a comfortable walk and runs to a speed the rig
    # cannot reach by walking - 0.38 m thighs at 80.6 kg - so the gait has to change to earn it.
    cmd_lin_vel_x = (0.8, 2.5)
    cmd_lin_vel_y = (-0.2, 0.2)
    cmd_ang_vel_z = (-0.4, 0.4)
    # Lower than Walk's 0.15: standing still is not what this task is for, and the Stand and Walk
    # policies already cover it. Not zero, because a policy that cannot decelerate is not useful.
    cmd_zero_probability = 0.05

    # --- reward weights ---
    # Tracking dominates harder than in Walk. At these speeds an upright-heavy reward is satisfiable
    # by refusing to accelerate, and the policy will take that deal.
    rew_track_lin_vel = 4.0
    rew_track_ang_vel = 1.0
    rew_upright = 1.0
    rew_head_height = 0.5
    rew_feet_air_time = 1.5

    # Longer strides, and stop paying beyond a plausible flight time.
    feet_air_time_threshold = 0.25
    feet_air_time_cap = 0.45

    # New in Run: a penalty for standing on both feet at once. This is the term that distinguishes
    # a run from a fast walk - without it the policy converges on the fastest possible walking gait,
    # which is a local optimum it has no reason to leave.
    rew_double_support = -0.5

    # Vertical motion is part of running rather than a defect, so Walk's penalty on it is relaxed
    # rather than kept - at -1.0 it directly opposes the flight phase the task is trying to find.
    rew_lin_vel_z = -0.25

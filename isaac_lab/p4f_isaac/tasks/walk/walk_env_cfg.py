"""Configuration for the Walk task.

Inherits Stand's config wholesale and changes only what walking needs. That is deliberate: Walk is
bootstrapped from a Stand checkpoint, and the bootstrap is valid only while the observation and
action layouts are identical. Anything that would change a width belongs in `StandEnvCfg` where
both tasks see it, not here.
"""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from p4f_isaac.tasks.stand.stand_env_cfg import StandEnvCfg


@configclass
class WalkEnvCfg(StandEnvCfg):
    episode_length_s = 12.0

    # Air time needs contact history; Stand reads only the instantaneous force and so runs without.
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(Hand_L|Hand_R|Foot_L|Foot_R)",
        history_length=3,
        track_air_time=True,
    )

    # --- command sampling ---
    # Modest ranges. The dummy is 80.6 kg with 0.38 m thighs; asking for 2 m/s before it can hold a
    # gait at 0.5 m/s just trains a policy that dives forward and calls the fall progress.
    cmd_lin_vel_x = (-0.3, 1.0)
    cmd_lin_vel_y = (-0.3, 0.3)
    cmd_ang_vel_z = (-0.5, 0.5)
    # Share of episodes commanded to stand perfectly still. Without these the policy forgets how to
    # stand, which is the one behaviour we already paid for.
    cmd_zero_probability = 0.15

    # --- reward weights ---
    rew_track_lin_vel = 3.0
    rew_track_ang_vel = 1.5
    # Halved from Stand: the upright and height terms still matter, but if they dominate, standing
    # still scores better than walking and the policy simply refuses to move.
    rew_upright = 1.0
    rew_head_height = 1.0
    rew_feet_air_time = 1.0
    # Vertical velocity, not planar - planar motion is now the objective, so Stand's penalty on it
    # is replaced by the command-tracking terms above.
    rew_lin_vel_z = -1.0
    rew_ang_vel = -0.05
    rew_action_rate = -0.01
    rew_joint_vel = -1.0e-4
    rew_effort = -0.02
    rew_termination = -10.0

    # A step must clear this much air time before it earns anything, and stops earning beyond the
    # cap. The threshold is what separates a real step from a foot skimming the floor; the cap is
    # what stops the term paying for hopping with both feet off the ground as long as possible.
    feet_air_time_threshold = 0.2
    feet_air_time_cap = 0.3

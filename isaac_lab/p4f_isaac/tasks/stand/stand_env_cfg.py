"""Configuration for the Stand task.

Control rate is the one number here that is a deliberate departure from Godot rather than a port
of it. The Godot RL scenes run `action_repeat = 8` against 120 Hz physics, so the policy decides
at **15 Hz** - a 66 ms control period. That is very slow for balance: a perturbation can become
unrecoverable before the network is permitted to respond at all, and no reward shaping fixes an
actuation-bandwidth problem. This task runs the same 120 Hz physics with `decimation = 2`, giving
a **60 Hz** policy. Transferring the result requires setting `action_repeat = 2` in the Godot RL
scenes; the contract file records that as a hard requirement, not a preference.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from p4f_isaac.assets import ACTUATED_JOINTS, DUMMY_CFG

# Success criteria, ported verbatim from Source/RL/Termination/UprightTermination.cs so the two
# tracks score the same behaviour as "standing".
STANDING_HEAD_HEIGHT = 1.35
STANDING_TILT_DEG = 30.0
STANDING_MAX_SPEED = 0.6

# Rest-pose head height (Head body at Godot y=1.54). Used to normalise the height reward.
REST_HEAD_HEIGHT = 1.54

# Fall thresholds. Deliberately far below the success criteria: an episode that ends the instant
# the body dips below "standing" would give the policy no room to recover, and recovery is the
# behaviour worth learning.
FALL_HEAD_HEIGHT = 0.9
FALL_TILT_DEG = 70.0


@configclass
class StandEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0

    action_space = len(ACTUATED_JOINTS)  # 36
    # 3 gravity + 3 lin vel + 3 ang vel + 1 pelvis height + 45 joint pos + 45 joint vel
    # + 4 contacts + 36 previous action + 3 velocity command. Kept explicit rather than computed so
    # a change to the observation forces a deliberate edit here - see obs_action_contract.md.
    observation_space = 143
    state_space = 0

    sim = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            # Sized for thousands of 46-body articulations. The defaults are tuned for far smaller
            # scenes and PhysX responds to overflow by dropping contacts, which shows up as bodies
            # sinking through the floor rather than as an error.
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # 16384 measured at 136k steps/s (34x the Godot rl/ track) using 14.6 GB of 16.3 GB VRAM.
    # That is ~0.5 GB of headroom once the desktop is accounted for, so anything added to the scene
    # can OOM mid-run rather than at startup; drop to 8192 (126k) if that happens. Note also that
    # each step up multiplies samples-per-gradient-update, so more envs is not automatically faster
    # to converge - 4096 was the previous default for that reason.
    scene = InteractiveSceneCfg(num_envs=16384, env_spacing=3.0, replicate_physics=True)

    robot = DUMMY_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Only the four bodies the observation actually reports. Matching `Robot/.*` instead would
    # instrument all 46 links - including the 30 massless dummies, which can never touch anything -
    # and contact reporting is per-body work on every physics step at every env.
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(Hand_L|Hand_R|Foot_L|Foot_R)",
        history_length=0,
        track_air_time=False,
    )

    # --- action scaling ---
    # Actions are absolute joint targets in [-1, 1], mapped per-axis onto that joint's own range
    # about the rest pose. The per-axis part is JointLimitedActionSpace's rationale kept intact: a
    # single global angle scale put 71.8% of the Godot action space beyond a hard stop, flattening
    # the gradient across most of the output range.
    #
    # Fraction of each joint's reachable span a unit action commands. See _apply_action for the
    # measurement that set it: at 1.0 the action space is too coarse for balance and the policy
    # scored worse than a zero-action baseline. 0.4 keeps fine control while still allowing a
    # substantial pose change - the far end of the range is reachable by sustained commands, just
    # not in a single step.
    action_scale = 0.4

    # --- reward weights ---
    # Balanced so the standing terms dominate and the regularisers shape *how* it stands without
    # ever making standing itself unprofitable.
    rew_alive = 1.0
    rew_upright = 2.0
    rew_head_height = 2.0
    rew_lin_vel = -1.0
    rew_ang_vel = -0.05
    rew_action_rate = -0.01
    # Barrier against the policy mean drifting outside the clip range - see _get_rewards. Zero
    # while every component stays inside [-1, 1], so on a healthy policy this term is exactly 0
    # and costs nothing; watch Episode_Reward/action_clip to confirm it stays there.
    rew_action_clip = -0.02
    rew_joint_vel = -1.0e-4
    # 0.02 during discovery, matching UprightProgressReward's default. HumanUP's finding, which
    # that file records as measured here too, is that deployment-strength control regularisation
    # applied from the start suppresses exactly the vigorous motion that finding the behaviour
    # requires. Raise toward 0.25 only once the dummy reliably stands.
    rew_effort = -0.02
    rew_termination = -10.0

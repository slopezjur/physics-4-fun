"""Constants that DEFINE the task, shared by every environment, scorer and exporter.

Every task has two implementations - on the CPU (`body_env.py` and the task envs built on it) and
on the GPU (`body_env_warp.py` and theirs) - kept honest by `test_env_parity.py` and
`test_walk_parity.py`. Those tests can only catch a divergence they can see. A constant that lives in
one backend and is imported by the other cannot diverge, but it does invert the dependency: the CPU
environment, which is the one Godot actually ships against, once imported its fall threshold and its
action latency from the CUDA backend it never uses. The scorer imported from there too.

So the numbers that define the task live here, and both backends depend on this module rather than
on each other.
"""
from __future__ import annotations
import math

# **Neuromuscular delay.** A human's command reaches the muscle roughly 40 ms after the decision;
# the policy runs at 60 Hz, so two steps is 33 ms. Without it the dummy has zero-latency reflexes,
# which is a large part of why a trained body looks robotic rather than alive - it can cancel a
# disturbance in the same instant it appears, something no person does.
ACTION_LATENCY_STEPS = 2

# Actuators the model has but the POLICY does not drive, by bone stem.
#
# The neck is actuated so the head is held up by muscle rather than by a stiff ligament (see
# build_mjcf.NECK_ACTUATED), but there is nothing for a policy to learn there at the stand/balance
# stage: a position actuator's zero command already means "head straight", and the head is 6% of
# body mass sitting on its own rotation axis. Leaving it in would cost 3 action dimensions and 9
# observations - both are derived from the actuator count - for the policy to explore its way back
# to the value it starts at.
#
# Excluded actuators are simply never written: `ctrl` is zero-initialised and stays there, which IS
# the held-upright target. Promoting the neck into the action space later (gaze, head protection)
# means deleting a name from this set and retraining.
POLICY_EXCLUDE = {"Head"}

# Pelvis height, as a fraction of the body's own standing height, below which the episode is a fall.
# 0.67 is the ratio the old hardcoded pair (0.55 m of 0.82 m) encoded. Both numbers were absolute,
# from a body that no longer exists, so "upright" quietly changed meaning every time the rig's
# height did - this one follows the model.
FALL_FRACTION = 0.67

# How much of the actuator the policy may use.
#
# In POSITION mode this is a fraction of each joint's range away from the rest pose, kept low
# because an untrained Gaussian policy commanding every joint to a random extreme at kp 533 drove
# MuJoCo to "Nan, Inf or huge value in QACC" before any learning happened.
#
# In TORQUE mode it is a fraction of the joint's peak human torque, and it has to be far higher:
# the policy now supplies ALL of the muscle, and 0.25 of a knee is 62 N.m to both hold and correct
# with. Measured stable at 0.6 with JOINT_ARMATURE in place - 10 rollouts of 20 s under full random
# torque, zero divergences.
POSITION_AUTHORITY = 0.25
TORQUE_AUTHORITY = 0.6

# How far from the target bone the projectile is launched, in metres.
#
# It is also why the ball's gravity is cancelled for the approach: a ballistic ball cannot reach
# 2 m below 4.43 m/s (a projectile's range is v^2/g), so the old curriculum's 2.00-2.42 m/s shots
# hit the floor 60 cm after launch and the dummy was never struck. Gravity comes back once the ball
# has had time to cover this distance, so a MISS lands and rolls instead of flying on for ever.
BALL_SPAWN_DISTANCE = 2.0

# Foot-origin height above its rest value at which a foot counts as AIRBORNE - for a PROTECTIVE step
# (perturb), and for the scorers that count one.
FOOT_CLEAR = 0.06

# The same for a WALKING step. A protective step is a big one; an amble is not. Measured on three
# walk brains at 0.25 m/s, the higher foot rises a median 2.0-2.4 cm and above 6 cm 0.0-0.3% of the
# time, so at FOOT_CLEAR a walking step had to be discovered from nothing and never was. 3 cm is
# still about 1.4x the shuffle those brains do. Both are measured from where the foot RESTS: the
# origin sits 4.4 cm up, and an absolute threshold read a 1.6 cm shuffle as a step.
#
# **1.6 cm, which is what the walk env always used** (an absolute 6 cm, with the foot resting at
# 4.4 cm) - and which pays a shuffle as a step. Every higher value tried on 2026-09-11 stopped the walk
# within one session instead of making it lift higher: 6 cm (w0911g), 3 cm (w0911h), 2.5 cm (w0911j),
# and 2 cm once training continued (w0911k) - each aborted as a statue, or 0.14 m in 40 s on a
# straight command. The shuffle's payment is what keeps walking ahead of standing still: preflight's
# +3.6/step margin is for a frictionless glide, and a real gait also pays effort, jerk, rocking and
# the risk of a fall. Strengthen the reason to walk before raising this.
WALK_FOOT_CLEAR = 0.016

# Legacy pose/step diagnostics, retained for scorer comparisons and preflight fixtures.
# These are not recovery reward weights. Grounded support and settling are defined
# in recovery_reward.py; neither swing speed nor airborne placement earns a bonus.
OFF_BALANCE = 0.06
CAPTURE_V = 1.0           # m/s, reference speed for the preflight kinematic fixture

# Strict rest-pose similarity is reported as a diagnostic. The recovery objective
# accepts a comfortable staggered stance instead of forcing this exact pose.
STANCE_SIGMA = 0.15        # m - Gaussian width on the feet's offset from the rest stance
LEG_POSE_SCALE = 0.10      # rad^2 - on the mean squared leg-joint error from the rest pose
LEG_BONES = ("Thigh", "Shin", "Foot")

# **The trunk joins the legs, and nothing rests on its stops** (2026-09-11). The shipped balance brain
# stood - quiet or under fire - with six joints on their limits: spine and chest twisted opposite ways
# at +-29 deg (limits +-28.6), both hips rotated -29 deg, the spine bent to its -23 deg stop, and the
# body leaning back 15 deg. A stop holds a pose for no torque, and `pose`, averaged over 30 joints,
# charged about 0.2 per step for all of it.
TRUNK_BONES = ("Spine", "Chest")
TRUNK_POSE_SCALE = 0.10    # rad^2, as LEG_POSE_SCALE
# A joint past this fraction of its half-range, measured from the middle of its range, is charged
# per radian beyond it - legged_gym's soft joint limit (0.9 there). The band is stretched to include
# the rest pose: a straight knee or elbow rests on its own stop, which is anatomy, not a free ride -
# unstretched, the rest pose itself paid 0.41 rad of penalty at the knees and elbows.
JOINT_SOFT_LIMIT = 0.85

# The walk's velocity-tracking kernel, RELATIVE to the commanded speed. A fixed width (sigma 0.15)
# was calibrated at 0.6 m/s, where standing still collects 9% of the tracking reward; at the slow
# stage's 0.15-0.35 m/s the same width paid a statue 44-86% of it, and the one session trained there
# stood still. Scaled by the command, a statue earns exp(-1 / TRACK_REL) = 10% at every speed.
# Commands slower than CMD_FLOOR (a stand) fall back to the floor: tight tracking around zero.
TRACK_REL = 0.434
CMD_FLOOR = 0.15

# The fastest a BODY degree of freedom may move before its world counts as numerically diverged.
# Measured over 1.2M world-steps each, falls included: the stepping walk gait peaked at 94 rad/s and
# the perturb brain under fire at 97, so 300 flags only a blow-up - the one injected on 2026-09-10
# went from 50 rad/s to a reward of -3.6e15 inside a step.
QVEL_CEILING = 300.0

# What a fall costs, and what a diverged world is paid instead of whatever its reward became.
FALL_PENALTY = 10.0

# Ceiling on the walk reward's velocity-squared penalties. A real gait measured -0.01 / -0.06
# weighted, so this only ever touches a blow-up: without it one world went -5.6e3 -> -3.6e15 -> NaN
# and took the whole PPO update with it, twice.
PENALTY_CAP = 10.0

# The name each task's shipped brain carries: `<family>_policy.onnx`, with its contract beside it.
# One name per BEHAVIOUR FAMILY, not per training run - stand and perturb are the same environment
# with the gun off or on, so they share one brain and one file; the names are the ones the Isaac3
# track already used (isaac_lab_3/exported/). Read by export_onnx.py and scripts/scoring.py.
POLICY_FAMILY = {"perturb": "balance", "walk": "locomotion"}

# The bones a shot is aimed at; each shot picks one. The scorers always pick uniformly.
TARGET_BONES = ("Head", "Chest", "Spine", "Pelvis",
                "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
                "Thigh_L", "Shin_L", "Thigh_R", "Shin_R")


def target_probabilities(weights):
    """Per-shot probability of each TARGET_BONES entry, from {bone: weight}; unnamed bones weigh 1.

    For TRAINING toward the hits a policy fails. Measured 2026-09-11 on perturb_z_s4/model_253, one
    30 N.s hit: head hits survived 10.5%, chest 26%, spine 38%, upper arms ~50%, pelvis and legs
    86-100% - the high hits are over half of all falls, and a quarter of the shots.

    `None` (no weights) is kept distinct from explicit equal weights, because a weighted draw
    consumes the random stream differently: the uniform path must stay the exact call it always
    was, or every scorer's shots would change under the checkpoints it has already scored.
    """
    if not weights:
        return None
    unknown = set(weights) - set(TARGET_BONES)
    if unknown:
        raise ValueError(f"not a target bone: {sorted(unknown)}")
    w = [float(weights.get(b, 1.0)) for b in TARGET_BONES]
    if not all(math.isfinite(x) for x in w) or not math.isfinite(sum(w)) or min(w) < 0.0 or sum(w) <= 0.0:
        raise ValueError(f"target weights must be non-negative with a positive sum: {weights}")
    return [x / sum(w) for x in w]

"""Configuration for the Stand task under Newton/XPBD.

Ported from `isaac_lab/p4f_isaac/tasks/stand/stand_env_cfg.py`. The task definition — success
criteria, action scale, reward weights — is carried across UNCHANGED so a result here is comparable
to the 2.3.2 numbers. What changed is only the physics backend and the state access it forces.

Deliberately dropped from the 2.3.2 config:

* **`PhysxCfg` and its GPU capacity knobs.** Those size PhysX's contact buffers; Newton has its own
  allocation and none of them apply. XPBD's analogous knob is `iterations`, below.
* **`rand_*` physics randomisation.** It went through `robot.root_physx_view`, which does not exist
  on a Newton backend, and it was a measured dead end besides — see HANDOFF.md attempt #7. It
  defaulted to disabled, so nothing is lost by not porting it yet.
"""

from __future__ import annotations

import os

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

from p4f_newton.assets import ACTUATED_JOINTS, DUMMY_CFG

# Success criteria, ported verbatim from Source/RL/Termination/UprightTermination.cs so all three
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

# XPBD solver iterations. **This is part of the dynamics the policy is trained against**, exactly
# as Jolt's position/velocity steps are on the Godot side — see docs/RL-SESSION-INVARIANTS.md
# invalidator #3, which is about precisely this parameter in the other engine.
#
# Measured on the zero-action hold: 2 / 8 / 16 hold the joints roughly 25x tighter as they rise
# (mean |q| 0.051 -> 0.005 -> 0.002 at t=1.0) and the body degrades far more gracefully (head at
# t=1.5: 0.857 -> 1.459 -> 1.486) — and the rest pose is unstable at every one of them, 0% standing
# at 8 s. That is the same shape as the Jolt 2/10 -> 16/30 measurement, and the same conclusion.
#
# 2 is Newton's default and the setting the gate was characterised at. Change it only at a retrain
# boundary, never between training and playback.
XPBD_ITERATIONS = int(os.environ.get("P4F_XPBD_ITERATIONS", "2"))


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
        physics=NewtonCfg(
            solver_cfg=XPBDSolverCfg(iterations=XPBD_ITERATIONS),
            num_substeps=1,
        ),
    )

    # Conservative next to the 2.3.2 default of 16384. XPBD's throughput on this rig is not yet
    # characterised, and the 2.3.2 table is explicit that an env count carried over from a
    # different task — let alone a different solver — is how that value got set wrong once already.
    # Raise it once `train.ps1` has reported real steps/s.
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    robot = DUMMY_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- action scaling ---
    # Fraction of each joint's reachable span a unit action commands, applied piecewise about the
    # REST pose. Part of the frozen contract, not a tuning knob: at 1.0 the action space is too
    # coarse to balance with and the policy scored worse than a zero-action baseline (48-step
    # episodes against 168). See obs_action_contract.md §2.
    # Fraction of each joint's range one action unit commands. `P4F_ACTION_SCALE` overrides it.
    #
    # **Small values matter for the Godot composition.** Isaac's action is already an OFFSET from the
    # rest pose, which is the same semantics Godot's `AssistMode` composes onto its balance layer's
    # base pose. But at 0.4 the policy learned to command a deep crouch - it holds head 1.18 against
    # a 1.54 rest height - because that is what survives in Isaac. Godot's balance layer holds an
    # upright 1.53, so the two maintain incompatible poses and any meaningful fraction of the
    # policy's offset destabilises the stand: measured as a fall even at 15% authority.
    #
    # A small scale forces the policy to make CORRECTIONS around the rest pose rather than commanding
    # a different one, which is what composes rather than fights.
    action_scale = float(os.environ.get("P4F_ACTION_SCALE", "0.4"))

    # --- reward weights, unchanged from the 2.3.2 task ---
    rew_alive = 1.0
    rew_upright = 2.0
    rew_head_height = 2.0
    rew_lin_vel = -1.0
    rew_ang_vel = -0.05
    rew_action_rate = -0.01
    # Barrier against the policy mean drifting outside the clip range. Zero while every component
    # stays inside [-1, 1], so on a healthy policy this term is exactly 0 and costs nothing; watch
    # Episode_Reward/action_clip to confirm it stays there. Without it, a 4000-iteration run ended
    # with 97% of components saturated and 34 of 36 joints permanently past the boundary.
    rew_action_clip = -0.02
    rew_joint_vel = -1.0e-4
    rew_effort = -0.02
    rew_termination = -10.0

    # --- actuator realism -------------------------------------------------------------
    # **`SolverXPBD` ignores `joint_effort_limit`** (see assets.py), so without this the policy
    # trains against drives that can deliver unbounded torque and learns a solution no real
    # actuator can execute. Measured on a 100%-scoring checkpoint: a single braced pose, held with
    # `max |action|` pinned at 1.00 and `joint_vel` decaying to 0.18 rad/s. A statue.
    #
    # Off reproduces the original behaviour exactly, for comparison runs.
    # --- joint-velocity channel ---------------------------------------------------------
    # **A contract addendum, and it must be applied in BOTH engines or it is worse than useless.**
    #
    # Godot's tightly-limited twist axes chatter. `Shin_L.y` is limited to +/-0.10 rad and driven at
    # kp=1800; XPBD enforces that limit as a rigid position constraint, while Jolt has an explicit
    # torque fighting the constraint, so the joint slams into its stop and bounces. Measured on a
    # body still standing at 0.81 m: `Forearm_R.y` at 33.9 rad/s, `Shin_R.y` at 68.1, against a
    # peak of 7.5 across all 45 DOF for the same policy in Isaac.
    #
    # That is 45 of the 143 observation floats arriving an order of magnitude outside anything
    # training ever produced. The normaliser is baked into the exported graph, so out-of-range input
    # is amplified straight into a saturated action - Godot's raw output reaches 3.21 against a
    # clamp of 1.
    #
    # Clipping bounds the damage; the noise is what hardens the policy against it. **But the noise
    # has to be small against the SIGNAL, and 1.5 was not.** Isaac's joint velocities peak near 2.95
    # rad/s early and settle to 0.18; noise at 1.5 std does not harden that channel, it erases it,
    # and the policy loses the only fast feedback it has about how the body is moving. Measured on a
    # fixed yardstick as a monotonic slide across four segments - 31.6, 26.6, 24.2, 22.7 - each
    # resuming from the last, so it compounded. 0.3 is a fifth of the early signal and a sensible
    # fraction of what the channel actually carries.
    # **False zeroes observation slice [55:100] entirely, in Isaac AND in Godot.**
    #
    # The joint-velocity channel is the one measured irreconcilable difference between the engines:
    # Godot's tightly-limited twist axes chatter against their stops at 20-68 rad/s where Isaac's
    # peak across all 45 DOF is 7.5. Both remedies are closed. Filtering it in Godot adds a ~0.11 s
    # lag to a balance-critical signal and measurably makes transfer WORSE (the promoted balance
    # brain stands at authority 0.10 unfiltered and falls at 0.50, 0.30 and 0.15). Hardening with
    # noise was tried at 1.5 std and degraded a fixed yardstick monotonically - 31.6, 26.6, 24.2,
    # 22.7 - because Isaac's signal only spans 0.2-3 rad/s, so noise that matches Godot's scale
    # erases the channel rather than toughening it.
    #
    # So: remove it from BOTH engines instead. A channel that carries nothing cannot disagree. The
    # objection to noise - that the policy loses its fast feedback - describes a policy that had
    # the channel and lost it; one trained from the start without it learns to balance on gravity,
    # height, joint ANGLE and contacts, all four of which the two engines already agree on.
    #
    # Godot must mask the same slice: see `IsaacObservation.JointVelocityEnabled`.
    obs_joint_vel_enabled = True

    # Zero the joint-velocity observation for any joint whose limit range is NARROWER than this,
    # in radians. 0 disables it. **This is the targeted version of `obs_joint_vel_enabled`.**
    #
    # Masking the whole channel works but costs too much: measured over six chained legs the policy
    # plateaus at 73-81% standing against 99% unmasked, because 33 of the 45 DOF carry feedback it
    # genuinely needs. The chatter is not spread across the channel - it is specifically the tightly
    # limited axes bouncing off their own stops, `Shin_R.y` at 68 rad/s on a 0.2 rad range. Twelve
    # joints are narrower than 0.5 rad (both shins' y/z at 0.2, both feet's at 0.4, and the hands),
    # and they are exactly the ones named in the chatter measurements.
    #
    # Godot derives the same mask from the same contract limits, so the two agree by construction
    # rather than by a copied list. See `IsaacObservation.JointVelocityMinRange`.
    # Selects which joints count as "narrow", in radians of limit range. **Selection only** - what
    # HAPPENS to them is decided by `obs_joint_vel_mask_narrow` and `obs_joint_vel_narrow_noise`.
    # These were one knob at first, which silently made the two treatments mutually exclusive: the
    # chatter was injected and then multiplied by the mask's zero, so the first chatter run was
    # really just another mask run wearing a different experiment name.
    obs_joint_vel_min_range = 0.0

    # Zero the narrow joints' velocity observation. Refuted for transfer - see the ladder results -
    # kept because the negative result is worth being able to reproduce.
    obs_joint_vel_mask_narrow = False

    # Noise (rad/s, std) added to the joint-velocity observation of the NARROW joints only - the
    # ones `obs_joint_vel_min_range` identifies. 0 disables it.
    #
    # **This reproduces Godot rather than sanitising it, which is the only reading left.** Three
    # ways of removing the discrepancy all made transfer worse: filtering in Godot (brain drops from
    # authority 0.10 to 0.05), noise across all 45 channels at std 1.5 (monotonic degradation), and
    # masking - full or narrow - which fails every rung of the ladder despite scoring 82.6% in
    # Isaac where a 68% unmasked policy stood at 0.05. A policy that transfers worse the more the
    # channel is cleaned is a policy that USES the channel.
    #
    # So give it Godot's channel in training. The failed noise test dosed all 45 DOF equally and
    # erased a 0.2-3 rad/s signal; the measured chatter is confined to the tightly limited axes -
    # `Shin_R.y` at 68 rad/s on a 0.2 rad range - while the wide joints stay clean in both engines.
    # 20.0 is the middle of the 8-68 rad/s Godot actually delivers there.
    obs_joint_vel_narrow_noise = 0.0

    obs_joint_vel_clip = 15.0
    obs_joint_vel_noise = 0.3

    # Set by every entry point that MEASURES rather than trains - evaluate, export, play,
    # slice_stats. Observation noise is a training device; scoring a policy through it would be
    # measuring the sampler, and Godot adds no noise of its own.
    playback = False

    # Maximum change in a commanded ACTION component per policy step, in the action's own [-1,1]
    # units. 0 disables it. Applied in action space rather than target space because that is the one
    # representation both engines share exactly; a joint's actual target moves by
    # `action_scale * limit * span`, so wide joints still slew faster than tight ones.
    #
    # **Opt-in via `train.py --action_rate_limit`, because it is a contract change** - a policy
    # trained with it must be driven with it, and Godot reads the value from the exported contract.
    # Default 0 keeps every existing checkpoint valid.
    #
    # The measurement behind it: Godot's joint velocities saturate within 33 ms - two policy steps -
    # of the first action, with the body still perfectly upright at 0.82 m, and the worst DOF is a
    # main hinge axis on a light arm rather than a chattering twist axis. So it is a genuine SLEW to
    # the first commanded target. Isaac shows the same transient at 7.54 rad/s and it decays; Godot's
    # climbs past 15. The policy commands roughly 0.785 rad of deflection as a STEP, and the two
    # engines disagree about how to execute a step: Isaac resolves it inside a position-based solve
    # that is inherently rate-limited by the timestep, Godot slews a light limb at whatever the
    # torque allows.
    #
    # Rate-limiting the target removes the transient from both, which should leave the two engines
    # agreeing in the quasi-static regime where they already agree. 0.15 per 60 Hz step moves the
    # knee about 0.16 rad per step - 9 rad/s - which is fast enough for a balance correction and far
    # below the 15+ rad/s that saturates the observation.
    action_rate_limit = 0.0

    # --- muscle model ------------------------------------------------------------------
    # Godot does not drive its joints with a bare PD. `ActiveBone` is a biomechanical actuator -
    # Hill force-velocity derating, a gravity feed-forward, a muscle-strength scale - and that is
    # the POINT of the project, not an obstacle to route around. The transfer work drifted for a
    # while into switching those off in Godot until the body matched Isaac's plain PD, which is
    # backwards: it turns the dummy into a robot. This models the muscle on the Isaac side instead.
    #
    # Hill-type force-velocity ceiling on the torque, ported from
    # `ActiveBone.ComputeForceVelocityScale`. Only the component of joint velocity ALONG the
    # commanded torque counts as shortening; a joint being braked is an eccentric contraction and
    # keeps full authority, because real muscle is stronger than isometric there. That asymmetry is
    # what makes it safe for balance - arresting a limb IS most of standing.
    #
    # 0 disables it. 15.0 rad/s matches `ActiveBone.MaxShorteningVelocity`.
    hill_max_shortening_velocity = 15.0

    # Spawn randomisation, radians on each joint and metres on the root height. Set to 0 for an
    # open-loop plant comparison against Godot: with noise on, Isaac's mean trajectory is an average
    # over starting poses Godot never has, and the two cannot be compared step by step.
    reset_joint_noise = 0.1
    reset_height_noise = 0.02

    # --- balance assist ----------------------------------------------------------------
    # Pelvis attitude stabilisation, ported from Godot's `PelvisStabilizationModule`. Strength 0-1;
    # 0 disables it. `P4F_BALANCE_ASSIST` overrides it for a sweep.
    #
    # **This exists to make "do nothing" mean the same thing in both engines.** Godot's dummy stands
    # at essentially the rest pose, but only because its balance layer supplies an external
    # stabilising wrench - the pelvis is the unactuated skeletal root and has no attitude control
    # without it. Isaac needs no such thing, because its body is passively far more stable, so a
    # policy trained here learns that zero action FALLS and must always command something. It
    # learned a crouch. In Godot with the balance layer running, zero action STANDS, so that crouch
    # is a destabilising command - measured as a fall in 4 s where doing nothing holds indefinitely.
    #
    # With the assist on both sides, zero action means "stand" in both, and the policy learns to
    # TRIM a standing body rather than to produce standing.
    #
    # Gains are Godot's verbatim: 600 / 20, capped at 300 N.m, with a 0.25 EMA on the angular
    # velocity because joint reaction chatter would otherwise dominate the damping term.
    balance_assist = float(os.environ.get("P4F_BALANCE_ASSIST", "0.0"))
    balance_gain = 600.0
    balance_damping = 20.0
    balance_max_torque = 300.0
    balance_filter_alpha = 0.25
    # Distribute the counter-torque into the feet, rather than letting the ground absorb it.
    #
    # **Off by default, and that is the physically right model here.** Godot sends the reaction to
    # feet it has confirmed GROUNDED, which then transmit it to the world through contact - so the
    # net effect on the system is an external torque from the ground. Isaac's ground is static, so
    # "the ground takes it" is exactly the same as applying nothing. Applying it to the foot BODIES
    # instead just spins them, because Isaac's contacts are soft at 2 solver iterations and the feet
    # penetrate the floor by ~8 mm rather than gripping it.
    #
    # Measured on a policy scoring 89.1% without any assist: 1.0% with the reaction applied to the
    # feet, 62.5% without it. It does not merely fail to help - it destroys the stand.
    balance_reaction = bool(int(os.environ.get("P4F_BALANCE_REACTION", "0")))

    enforce_effort_limit = True

    # Per-episode scale on every joint's torque budget, sampled uniformly in this range.
    #
    # **Domain randomisation aimed at a specific measured gap, not a general precaution.** Matching
    # Godot's `MaxTorque` numbers is not the same as matching its authority. Isaac applies its drives
    # inside the solve, so the reaction distributes through the articulation in one step; Godot
    # applies an external torque to a body pair and it has to propagate along the chain, on top of a
    # Hill force-velocity derating and a D-term that may claim half the budget. The result is a body
    # that sags where Isaac's holds: at t=0.5 s under the same policy, Godot's joints sit 0.93 rad
    # from rest against Isaac's 0.38, and once they are dragged past their targets the observation
    # leaves distribution and the actions saturate.
    #
    # A policy trained across a range of budgets cannot depend on having the full one. The upper
    # bound stays at 1.0 so the nominal actuator is always in the training distribution.
    # Lower bound set from a MEASUREMENT, not a guess. Sweeping this scale in Isaac and comparing
    # against Godot's actual trace locates where the two engines behave alike:
    #
    #   scale 1.0 -> 92.2% standing, head 1.338     scale 0.5  ->  4.7%, head 0.324
    #   scale 0.7 -> 57.0% standing, head 0.807     scale 0.35 ->  0.0%, head 0.277
    #
    # Godot collapses to a head height of 0.14-0.42 in about 1.5 s, which is Isaac at **0.4-0.5**.
    # So Godot delivers roughly HALF the authority its `MaxTorque` numbers promise. A range that
    # stopped at 0.55 never showed the policy the regime it actually has to survive in.
    # **Narrowed back from (0.4, 1.0) after it measurably backfired.** The sweep below says Godot
    # behaves like Isaac at 0.4-0.5, so that looked like the honest range to train in. But it is also
    # a range where standing is close to impossible - 4.7% at 0.5, 0.0% at 0.35 - so a large share of
    # episodes became unwinnable and the policy correctly learned that nothing it did mattered.
    # Measured on a FIXED yardstick (push 0.6, nominal effort): 40.6% before the widening, 21.9%
    # after. `docs/RL-TRAINING.md` records exactly this overshoot for a perturbation curriculum.
    #
    # 0.65 is a compromise between two measurements. Godot's mean head height after the Hill
    # force-velocity fix is about 0.45 m, which sits between Isaac's 0.324 at scale 0.5 and 0.807 at
    # 0.7 - so Godot behaves like **0.55-0.6**, not the 0.4-0.5 measured before that fix. Training
    # at 0.55 would put the task back in the unwinnable regime; 0.65 reaches toward Godot while
    # staying somewhere a policy can actually learn. Widen it further only with a curriculum that
    # ramps it rather than sampling it flat.
    # `P4F_EFFORT_SCALE` pins both ends for a plant sweep. **Effort, not stiffness, is the lever
    # that makes an XPBD joint sag.** XPBD drives are position-based, so the solver projects a joint
    # toward its target and the effective stiffness is dominated by ITERATION COUNT, not by kp -
    # measured, gain scales of 0.176 and 0.08 give near-identical joint deviation, and raising
    # iterations makes joints TIGHTER (0.07 -> 0.04 rad) rather than looser. What does limit the
    # correction is how far the target may be pulled, which is what `enforce_effort_limit` clamps.
    effort_scale_range = (
        (lambda e: (e, e) if e else (0.65, 1.0))(float(os.environ.get("P4F_EFFORT_SCALE", "0")))
    )

    # --- perturbation -----------------------------------------------------------------
    # A random shove at spawn. The other half of the statue fix: with a deterministic start and no
    # disturbance, standing still IS the optimal policy, and every standing criterion agrees.
    #
    # Deliberately modest. `docs/RL-TRAINING.md` records a perturbation curriculum overshooting
    # into a regime where falling was unavoidable, at which point the policy correctly concluded
    # that nothing it did mattered. 0.6 m/s on an 80.6 kg body is a firm push, not a launch.
    push_probability = 0.8
    push_velocity = 0.6
    push_ang_velocity = 1.0

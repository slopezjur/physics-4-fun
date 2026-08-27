# Overnight session state — 2026-08-27

Written so this survives a context compaction. **Read this first if you are resuming.**

User is asleep, back **~10:30**. They handed over the night with instructions to keep training and
adjusting, and to work on Stand → Walk → Perturbation → Run, plus Godot↔Isaac mapping.

---

## 0. The short version

**The Stand policy that scored 100% was a statue, and the reason it would not transfer is that
Isaac's actuator is not Godot's actuator.** Five separate causes were found; four are fixed and the
fifth is measured.

1. **`SolverXPBD` silently ignores `joint_effort_limit`** - the policy trained against drives that
   deliver unbounded torque, and the optimal solution under those drives is to freeze in a braced
   pose. Fixed: `enforce_effort_limit` inverts the PD law to clamp the commanded target. (§2b)
2. **The evaluation criterion could not tell a statue from a balance controller**, because nothing in
   it ever pushed. Fixed: `push_velocity` in the task, `--push` in the evaluator. (§2b)
3. **Torque was clamped per axis, where Godot clamps per bone as a vector** - Isaac could spend
   `effort x sqrt(3)`. Fixed. (§2c)
4. **Godot's Hill force-velocity derating was switching the actuators OFF** - `fvScale` reached 0.00
   by t=1 s, at which point `ActiveBone` applies exactly zero torque. Fixed:
   `IsaacPolicyDriver.DisableHillLimit`. (§2c)
5. **Godot delivers roughly HALF the authority its `MaxTorque` numbers promise** - measured by
   sweeping the budget in Isaac until it fails the way Godot does. Not fixed; this is the open
   problem. (§2c)

**The dummy still falls in Godot.** But the failure is now localised to a specific, measured
quantity rather than being "the policy did not transfer", and the passive dynamics are confirmed to
agree between the engines.

**Two mistakes of my own, both caught by measurement and both worth knowing about:**

* I widened the actuator randomisation to a regime where standing is impossible, and the policy
  correctly learned that nothing it did mattered - 40.6% down to 21.9% on a fixed yardstick. Backed
  out. (§2c)
* The NaN guard I wrote had an operator-precedence bug that meant it never fired, so
  `Diagnostics/divergences` read a reassuring 0.0000 for hours until rsl_rl killed a 12-segment run.
  Fixed. (§2c)

**Best checkpoint right now is still `perturb01/model_1078.pt`**, which is what `Models/` holds.
Take the best by evaluation, not the last one - and always with an explicit
`--push 0.6 --effort_scale 1.0`, because the config moved under the per-segment numbers.

**Top recommendation for next session:** rate-limit the change in commanded joint target per policy
step, symmetrically in both engines, and retrain. Godot's joint velocities saturate within 33 ms of
the first action while the body is still perfectly upright; Isaac's transient is half the size and
decays. (§2c)

---

## 0b. THE DUMMY STANDS - 2026-08-27 afternoon

**Scene: `Scenes/RL/Isaac3/Stand/IsaacStandCheckNewton.tscn`** (F6). 20 s, head steady at 1.52.

```
BalanceAssist = 1.0          # Godot's balance layer owns root stabilisation
ActionScaleOverride = 0.10   # the Isaac policy owns all 36 joint targets
HillVelocityFilter = 0.15
```

### What unlocked it

`HumanoidRagdoll.SuppressProceduralPose`. The ragdoll stays in the **Balanced** state so the balance
strategies actually run - **they self-gate to Balanced/Stumbling, so raising
`StateBalanceStrengthMap` for the RL state does nothing at all**, and the traces were byte-identical
at 1.0 and 0.6 before I found that. Suppressing only the procedural POSE writer leaves the policy
owning joint targets while Godot owns root stabilisation.

The diagnosis that led there: **Godot's dummy cannot hold the rest pose passively** - not at authored
gains, not at a compensated 227 effective stiffness. Its pelvis is the unactuated skeletal root and
has no attitude control without the balance layer. Isaac's body nearly holds it for free: under zero
action Isaac's joints deviate **0.12 rad** and its centre of mass drifts **0.05 m**, against Godot's
**1.01 rad** and **0.30 m**. So the policy was trained to be the sole controller of a body that
barely needs one, then asked to solo a body that genuinely does.

### Honest limits of this result

* **Zero action also stands in this configuration.** The policy is tolerated, not load-bearing.
* **A push test says the same thing directly.** Both policy and zero action survive 4 N.s; both fall
  at 8. The policy does not extend the disturbance threshold at all. That measurement is the one
  worth repeating as this improves - `IsaacArena.PushImpulse` / `PushAtSeconds` exist for it, and
  18 N.s is the reference (the real Godot ball, 3.0 kg at 6 m/s).

### The one clear trend

Training in the configuration that MIRRORS the deployment - balance assist on, matched action scale
- took Isaac Stand to **92.2%** (head 1.364, ever-fell 9.4%) and **doubled the authority Godot
tolerates**, 0.05 -> 0.10. The export also showed `0/36 outside [-1,1]` at rest for the first time:
the policy stopped saturating its own clamp. Matching the training plant to the deployment plant is
what moves this, and it moved it in one 12-minute segment.

### Godot-side knobs added today, all off by default

| export | what it does |
|---|---|
| `BalanceAssist` | balance layer in the RL role; needs `SuppressProceduralPose` |
| `ActionScaleOverride` | drive the policy at a different authority than it trained at (diagnostic) |
| `HillVelocityFilter` | feed the Hill law a filtered rate so solver chatter stops zeroing a real muscle |
| `JointMotorDrive` | drive via Jolt joint motors instead of `ActiveBone` - tracks superbly (0.03 rad) but discards the biomechanics; **not** the default, see §0c |
| `GainCompensation` | raise kp against the SPD denominator - capped by `I/dt^2`, see §2c |
| `PushImpulse` / `PushAtSeconds` | disturbance test |
| `DumpPoseAfterSeconds` | one-shot 45-DOF stance dump |

### Two bugs worth remembering

* **`IsaacActionSpace.ActionScale` was a hardcoded 0.4** while `export.py` had always written
  `action_scale` into the contract and nothing read it. Any policy trained at another scale was
  driven at 0.4 - commanding several times its intended deflection, silently. Same defect class as
  the DOF order. It now travels with the policy.
* **Godot's balance reaction torque must NOT be ported into Isaac.** Godot sends it into planted
  feet which transmit it to the world; Isaac's ground is static, so "the ground takes it" means
  applying nothing. Applying it to the foot bodies instead spins them - measured as **89.1% -> 1.0%**
  on a policy that scored 89.1% with no assist at all.

## 0c. Direction: model Godot's muscle in Isaac, never the reverse

I spent an hour going the wrong way - `DisableHillLimit`, then `DisableLoadCompensation`, then
bypassing `ActiveBone` entirely for Jolt's joint motors. Each step was justified by a measurement and
the cumulative direction was to strip the dummy into a robot, which is the thing this project exists
to avoid. The user caught it.

**Correct the premise too:** XPBD settled the SOLVER CLASS question and that stands. XPBD never
supplied a muscle - Isaac's actuator has always been `ImplicitActuatorCfg`, a plain PD. The question
was never muscles vs motors; it was always a PD, and whether it matches the muscle. Godot's Hill
curve is now ported into Isaac (`hill_max_shortening_velocity`), including the asymmetry that a
braked joint keeps full authority, because arresting a limb IS most of standing.

---

## 1. What is RUNNING right now

**Two chains, deliberately in separate experiment trees so they cannot adopt each other's weights.**
`night.py` resumes from the newest checkpoint in its experiment directory, so a shared name would
silently cross-contaminate them - hence the new `--experiment` flag on both scripts.

| | statue chain (obsolete) | **robust chain (the real one)** |
|---|---|---|
| experiment | `p4f_newton_stand` | `p4f_newton_stand_robust` |
| driver log | `logs/night_driver.log` | `logs/robust_driver.log` |
| session dir | `logs/night/2026-08-27_02-11-27/` | `logs/night/2026-08-27_05-07-37/` |
| envs | 16,384 | 8,192 |
| runs until | 10:00 | 09:00 |
| actuator | **unlimited torque** | effort-limited, vector-clamped |
| disturbance | none | 0.6 m/s spawn push |

**The first chain is obsolete - see section 2b.** It reached 100% by learning a statue and is still
running only because I could not stop it: the process-kill was refused by the sandbox
(`Stop-Process` and `Get-CimInstance` both blocked), and `night.py` has no stop file to touch. It
costs GPU and nothing else, and it does leave a complete before/after record. **Ignore its numbers.**

The robust chain is the one to read. Segment 1 scored **31.6% standing under a 0.6 m/s push**
(head 0.646, ever-fell 72.3%) - that segment trained under the old actuator and was scored under the
new one, so it is a floor, not a result. Segment 2 onward trains under everything in 2b and 2c.

**Check `logs/night/2026-08-27_05-07-37/SUMMARY.md` first when resuming.** If the chain has stopped,
restart it - it picks up from the newest checkpoint in its own tree automatically:

```
python scripts/night.py --until <HH:MM> --minutes 15 --num_envs 8192   --experiment p4f_newton_stand_robust --push 0.6
```

**Scores are now reported under a push, and that changes what they mean.** A number here is not
comparable with a pre-05:00 number: the actuator is weaker and there is a disturbance. That is the
point - see 2b.

## 2. The state of play

**The XPBD gate passed and that result still stands.** Newton/XPBD reproduces Godot's zero-action
collapse (0% standing, floor in under 2 s) where PhysX holds the same rig at ~84% for 8 seconds.
That is the finding the whole track existed to test, and it is confirmed twice over: the passive
dynamics agree between the engines, and Godot's zero-action trace is clean and quiet (`angVel` 0.52,
`jointVel` 5.80) rather than merely broken.

**The training result on top of it did not stand up.** A policy reached 100% in fifteen minutes and
held it for five straight segments, and it was a statue exploiting an actuator Godot does not have.
See §2b - read that before trusting any Stand number written before 05:00.

**So: the solver question is settled and the actuator question is the live one.** Those are
different problems, and conflating them cost this project seven attempts before Isaac Lab 3 - the
gap was never only the solver class.

---

## 2b. THE FINDING OF THE NIGHT - Stand was a statue

**Read this before trusting any Stand number in section 2.**

The Stand policy reached **100.0% standing / 0.0% fell / head 1.459** and held it from 03:01 to
04:40, five straight segments at ceiling. It still fell over immediately in Godot, with the
first-step observation verified exact.

`scripts/slice_stats.py` (new - prints the same per-slice diagnostic Godot's `IsaacPolicyDriver`
prints, so the two can be read side by side) shows why:

| slice | Isaac t=0.5s | Godot t=0.5s | Isaac t>=1.5s |
|---|---|---|---|
| `angVel` | 0.19 | **5.89** | **0.03** |
| `jointVel` | 2.95 | **10.87** | **0.18** |
| `jointPos` | 0.38 | 0.79 | 0.45 steady |
| `maxAction` | 1.00 | 1.10 | **1.00 pinned** |

The policy does not balance. It moves once into a braced pose, pins `max |action|` at exactly 1.00,
and freezes - `joint_vel` decays to 0.18 rad/s and stays there forever. **A statue.**

It works in Isaac because **`SolverXPBD` silently ignores `joint_effort_limit`** - `assets.py`
already recorded this as a known gap, but nothing connected it to the transfer failure. The policy
trained against position drives that deliver unbounded torque. Godot's actuators have a real torque
ceiling plus a Hill velocity derating, so the same commanded pose sags, the body starts to move, and
a policy that only knows how to HOLD is out of distribution within half a second. Godot's raw output
passes its clamp of 1 and reaches 3.9.

The same checkpoint, re-scored with the effort limit enforced:

| | standing | ever fell | head |
|---|---|---|---|
| unlimited torque (as trained) | **100.0** | 0.0 | 1.459 |
| effort-limited | 60.9 | 74.6 | 0.728 |
| effort-limited + 0.6 m/s spawn push | **16.8** | 94.5 | 0.378 |

**The evaluation criterion was complicit.** Head >= 1.35, tilt <= 30 deg, speed <= 0.6 held 1.5 s is
satisfied perfectly by a statue, because nothing in it ever pushes. `evaluate_stand.py --push` now
exists; a balance policy scored without a disturbance is not being asked to balance.

### The two fixes, both in `StandEnvCfg` and inherited by Walk

* **`enforce_effort_limit = True`** - clamps the commanded target to the furthest one the joint's
  torque budget can actually ask for, by inverting the PD law. The same law and the same clamp
  Godot applies. `_effort_limited()` in `stand_env.py`.
* **`push_velocity = 0.6`, `push_probability = 0.8`, `push_ang_velocity = 1.0`** - a random shove at
  spawn, written as a root twist through `NewtonRigState.reset_to(root_vel=...)`. The body has to
  reach equilibrium from somewhere different every episode, which is the property a statue does not
  have. Magnitudes deliberately modest: `docs/RL-TRAINING.md` records a perturbation curriculum
  overshooting into a regime where falling was unavoidable, at which point the policy correctly
  learned that nothing it did mattered.

Newton's root twist is **linear in [0:3], angular in [3:6]** - measured with the new
`scripts/probe_twist.py`, not assumed. warp's spatial vectors are conventionally the other way
round, and getting it backwards spins the body where a push was intended without raising anything.

### Godot side, also settled tonight

`JointSpacePd` drives the ragdoll with Isaac's own per-joint gains instead of Godot's muscle model.
It was **off** in the check scene, and turning it on made things worse, for a reason worth keeping:
the naive law `tau = kp*err - kd*rate` is unconditionally unstable at this rig's gains under
explicit integration. Isaac applies its drives INSIDE the XPBD solve (position-based, implicit,
stable at any stiffness); Godot can only add an external torque that Jolt then integrates. At the
hip's kp=1800 against a 120 Hz tick the oscillator advances more than a radian of phase per step.

Measured with a **zero** action - every joint commanded to the rest pose it is already in, which
should cost no torque at all: all 36 actuators pinned at their effort limit (693 Nm = 400 x sqrt 3),
91 rad/s of joint rate, 1.35 rad of tracking error, floor in under two seconds.
`ApplyJointSpaceTorque` now uses the Tan-Liu-Turk SPD denominator, the same formulation
`PidController3D` already ran for the Godot-native track. That fixed the pinned-torque symptom but
the path is still far more violent than the native one (`angVel` 37 vs 0.52), so **the check scene
uses the native `ActiveBone` path** and `JointSpacePd` remains an experiment, not the default.

The native path with zero action is clean and quiet (`angVel` 0.52, `jointVel` 5.80) and still
collapses in under 2 s - which is exactly what Isaac/XPBD does with zero action. **The passive
dynamics agree between the engines.** The gap was never the body; it was the actuator.

---

## 2c. Actuator fidelity - what else was aligned, and what is still open

After the effort limit, three more differences were found and closed. Each is now measured, not
assumed.

**Torque is clamped per BONE as a vector, not per axis.** Godot's `ActiveBone` bounds
`totalTorque.Length()` against one `MaxTorque`, so a bone's three axes share a single budget. A
per-axis clamp let Isaac spend `effort * sqrt(3)` on the same joint - 693 N.m where Godot delivers
400. The same policy scores 95.3% per-axis and **87.5% vector-clamped**; 87.5 is the honest number.
Godot's per-bone `MaxTorque` values (350, 400, 60, 150, 250, 100, 25) are exactly the `effort` set in
the rig JSON, so the ceiling being enforced is Godot's real one.

**The joint-velocity channel is clipped in both engines and noisy in training.**
`obs_joint_vel_clip = 15.0`, `obs_joint_vel_noise = 1.5`, written into the exported contract as
`joint_velocity_clip` and read back by `IsaacObservation.JointVelocityClip` so the two cannot drift
apart. Why: Godot's tightly-limited twist axes chatter against their stops - `Shin_L.y` is limited to
+/-0.10 rad and driven at kp=1800, and Jolt has an explicit torque fighting a constraint that XPBD
solves rigidly. Measured at 33.9 rad/s on `Forearm_R.y` and 68.1 on `Shin_R.y` with the body still
standing at 0.81 m, against 7.5 as the peak across all 45 DOF in Isaac. That is 45 of 143 floats an
order of magnitude out of distribution, amplified by the baked-in normaliser into a saturated
action. The clip bounds it; the noise is what teaches the policy not to depend on it.

`scripts/probe_twist.py` establishes that Newton's root twist is **linear in [0:3], angular in
[3:6]** - measured, because warp's spatial vectors are conventionally the other way round and
getting it backwards spins the body where a push was intended without raising anything.

**The torque budget is randomised per episode**, `effort_scale_range = (0.4, 1.0)`. Matching
Godot's `MaxTorque` numbers is not the same as matching its AUTHORITY, and the difference is
measured: at t=0.5 s under the same policy, Godot's joints sit **0.93 rad** from rest against
Isaac's **0.38**. The joints are being dragged more than twice as far for an identical commanded
target, because Isaac applies its drives inside the solve - the reaction distributes through the
articulation in one step - while Godot applies an external torque to a body pair that has to
propagate along the chain, on top of a Hill force-velocity derating and a D-term that can claim half
the budget. Once the joints are dragged past their targets the observation leaves distribution and
the actions saturate. A policy trained across a range of budgets cannot depend on having the full
one. `playback` pins the scale to 1.0 so evaluation and export report the nominal actuator.

### How weak is Godot's actuator, in a number

`evaluate_stand.py --effort_scale` pins the torque budget to a fixed fraction of nominal. Sweeping it
in Isaac and comparing against Godot's actual trace locates where the two engines behave alike:

| Isaac `effort_scale` | standing | mean head |
|---:|---:|---:|
| 1.0 | 92.2% | 1.338 |
| 0.7 | 57.0% | 0.807 |
| 0.5 | 4.7% | 0.324 |
| 0.35 | 0.0% | 0.277 |

Godot collapses to a head height of 0.14-0.42 within about 1.5 s. **That is Isaac at effort_scale
0.4-0.5** - and after the Hill force-velocity fix below, Godot's mean head rises to about 0.45 m,
which sits between Isaac's 0.324 at scale 0.5 and 0.807 at 0.7, i.e. **0.55-0.6**. The Hill fix
bought back real authority. Training range set to (0.65, 1.0), reaching toward that without
re-entering the regime where the task is unwinnable - so Godot delivers roughly HALF the authority its `MaxTorque` numbers promise, even
though those numbers are identical to the rig's `effort` values.

This is the single most useful number of the night for whoever picks this up. It converts "the
policy does not transfer" into "find the missing 50-60% of torque, or train inside it". The
randomisation range was widened from (0.55, 1.0) to **(0.4, 1.0)** on the strength of it - the
original bound never showed the policy the regime it actually has to survive.

Where the missing authority plausibly goes, in rough order of suspicion: the Hill force-velocity
derating in `ActiveBone.ComputeForceVelocityScale`; the D-term claiming up to half the budget in
`PidController3D` (`maxDerivativeTorque = MaxTorque * 0.5`); `LoadCompensationTorqueFraction`
reserving another slice; and the fact that an external torque on a body pair has to propagate along
the chain over several ticks where a solver-internal drive distributes in one. **None of these have
been measured yet** - that is the first thing to do next.

### The Hill force-velocity limit was switching the actuators OFF

`ActiveBone` scales its torque ceiling down as a joint turns in the direction it is being driven,
reaching zero at `MaxShorteningVelocity` (15 rad/s) - and when that scale reaches zero, `ActiveBone`
sets the applied torque to exactly `Vector3.Zero`. The muscle switches off.

Measured in the stand check with the new `demand`/`deliver`/`fvScale` diagnostic: **`fvScale` falls
to 0.00 by t=1.0 s** while `demand` climbs to **2.23x** the ceiling and `deliver` sits at 0.53. The
actuators were being asked for more than twice what they can give and were handing back nothing. It
is a death spiral - chatter raises joint velocity, the derating cuts the torque, the body falls,
which raises joint velocity further.

Isaac has no such model; its drives are a PD against a flat ceiling, which is what the policy was
trained against. **`IsaacPolicyDriver.DisableHillLimit` lifts it, and the check scene sets it.** The
derating is a real biomechanical property and belongs in the Godot-native track - it just is not
part of the contract an Isaac policy was trained against. With it lifted, `fvScale` holds at 1.00,
`demand` and `deliver` agree (0.18/0.18, 0.42/0.42), and `min-since-engaged` improved from 0.129 to
0.269. **The body still falls** - this was necessary, not sufficient.

### What diverges first, and the top recommendation for next session

With `DiagnosticInterval` now forwarded from the arena (it was a driver export the arena never
passed on, so setting it in a scene silently did nothing), the trace can be read every two policy
steps instead of every thirty:

```
t=0.033  angVel 0.86  jointPos 0.22  jointVel 15.00  worstDof UpperArm_L.x
t=0.067  angVel 2.44  jointPos 0.47  jointVel 15.00  worstDof Forearm_L.x
t=0.100  angVel 4.16  jointPos 0.27  jointVel 15.00  worstDof UpperArm_L.x
```

**Joint velocity saturates within 33 ms - two policy steps - with the body still perfectly upright
at 0.82 m.** The worst DOF is initially a main hinge axis on a light arm, not a twist axis, so this
is a genuine SLEW to the first commanded target and not the chatter. Isaac shows the same transient
at t=0 but at 7.54 rad/s, and it decays to 2.95; Godot's climbs.

The policy's first action commands roughly 0.785 rad of knee deflection as a step. Isaac executes
that inside a position-based solve, which is inherently rate-limited by the timestep; Godot's
explicit actuator slews a light limb at whatever the torque allows.

**This became the last experiment of the night.** `action_rate_limit` caps the per-policy-step
change in each action component, symmetrically in both engines, and is **opt-in** so no existing
checkpoint is invalidated: `train.py --action_rate_limit 0.15`, carried in the exported contract as
`action_rate_limit`, read back by `IsaacActionSpace.ActionRateLimit`. With no limit set the Godot
path is a verified no-op - the trace is identical to the previous build, line for line.

Applied in ACTION space rather than target space because that is the one representation both engines
share exactly, and **once per policy step, not per physics tick**: `_apply_action` runs `decimation`
times per policy step, so limiting there would silently double the real rate against what the policy
trained with. Godot's `Decode` is called once per policy step, which is the matching seam. The raw
network output is deliberately left untouched for the saturation and clipping counters - overwriting
it would make `clipped` read zero for exactly the policy whose barrier term has failed.

Run: `logs/rsl_rl/p4f_newton_stand_ratelimit/*_rl015/`, seeded from the same `perturb01` checkpoint
as the clean chain, so the two differ only in the limit. **It starts badly** - mean episode length 63
against the clean chain's several hundred - because the seeded policy was trained to command steps
and now takes about seven steps to reach an intended target. Whether it re-learns inside 95 minutes
is the open question; if the numbers below are still poor, that is not evidence the idea is wrong,
only that it was not given long enough from a hostile initialisation.

### I overshot the randomisation, measured it, and backed it out

Widening `effort_scale_range` to (0.4, 1.0) because that is where Godot behaves was a mistake, and
the fixed-yardstick numbers caught it:

| checkpoint | standing @ push 0.6, nominal effort |
|---|---:|
| `perturb01` (effort 1.0, per-axis clamp) | **40.6%** |
| robust chain segment 3 (effort 0.4-1.0) | 21.9% |

Standing at scale 0.4-0.5 is close to impossible (4.7% at 0.5, 0.0% at 0.35), so a large share of
episodes were unwinnable and the policy correctly learned that nothing it did mattered - the exact
overshoot `docs/RL-TRAINING.md` records for a perturbation curriculum. Narrowed to **(0.75, 1.0)**.

**The per-segment numbers in `SUMMARY.md` hide this**, because the config changed under them: seg 1
scored 31.6, seg 2 26.6, seg 3 25.8, but each was scored against a harder actuator than the last.
**A moving yardstick measures nothing.** Compare checkpoints with an explicit
`--push 0.6 --effort_scale 1.0`, which is stable across every config change tonight.

**So take the BEST checkpoint by evaluation, not the last one** - `perturb01/model_1078.pt` is still
the strongest as of 05:45, and it is what `Models/stand_policy.onnx` currently holds.

### The observation noise was larger than the signal

`obs_joint_vel_noise = 1.5` was set without checking it against what the channel actually carries.
Isaac's joint velocities peak near **2.95 rad/s** early in an episode and settle to **0.18**. Noise
at 1.5 standard deviations does not harden that channel against Godot's chatter - it erases it, and
takes with it the only fast feedback the policy has about how the body is moving.

It showed up as a monotonic slide on the fixed yardstick, compounding because every segment resumes
from the last:

| segment | standing @ push 0.6 |
|---:|---:|
| 1 | 31.6% |
| 2 | 26.6% |
| 3 | 24.2% |
| 4 | 22.7% |

Reduced to **0.3**. The general lesson is worth more than the number: *domain randomisation has to be
sized against the signal it is perturbing, and this project has no habit of measuring that.* The same
mistake in a different costume as the effort-range overshoot two subsections up - both were
"randomise harder" applied without asking what the policy still has left to work with.

### Another silent guard bug: the divergence check never fired

`~torch.isfinite(x).any(dim=1)` binds as `~(isfinite(x).any(dim=1))` - "NOT (any element is finite)"
- so it only fired when EVERY joint had gone non-finite. `Diagnostics/divergences` therefore read a
reassuring **0.0000** for hours while NaNs propagated, until rsl_rl refused a rollout with *"The
observation group 'policy' ... contains NaN values"* and **killed the 12-segment statue chain
mid-run**. The evaluator was hiding the same thing from the other side: `mean |action|` came out as
`nan` because one diverged environment in an arena that never resets poisons the mean. Both fixed;
the negation is parenthesised and the action statistic is masked to finite rows.

### Two things tried that did NOT work - do not repeat them

* **`JointSpacePd` with the naive law.** Driving Godot with Isaac's own gains via
  `tau = kp*err - kd*rate` is unconditionally unstable at this rig's stiffness under explicit
  integration. With a ZERO action - every joint commanded to the rest pose it is already in - all 36
  actuators pinned at their effort limit, 91 rad/s of joint rate, floor in under two seconds. Adding
  the Tan-Liu-Turk SPD denominator fixed the pinned torque but the path is still far more violent
  than the native one (`angVel` 37 vs 0.52). **The check scene uses the native `ActiveBone` path.**
* **`LockAxesBelow`** - commanding rest on every axis whose half-range is under 0.2 rad, on the
  theory that the policy's +/-0.04 rad of authority there was not worth the torque it cost.
  `Shin_R.y` still chattered at 15 rad/s with its target at zero. **The chatter is a symptom of the
  body moving violently under the policy's commands, not a cause.** The flag is kept, defaulting to
  0, only so nobody spends the time again.

**Inconclusive, not tested: Jolt solver iterations.** Raising
`physics/jolt_physics_3d/simulation/{velocity,position}_steps` to 30/16 produced a trace identical
to the default to the last decimal - and so did a deliberately destructive control,
`limits/max_linear_velocity = 0.1`, which should have made the fall take five times as long. **I
could not verify that project.godot physics settings take effect in a `--headless --path` run at
all.** `project.godot` was restored; the experiment is unresolved and worth redoing from the editor,
because if Godot's joints sag under load where XPBD's do not, solver iterations are exactly the knob
that would show it.

### Where the Godot transfer actually stands

Still falling, at about 1.5-2 s. Best variants measured, all with the 87.5% policy:

| Godot variant | pelvis at t=2 s | verdict |
|---|---|---|
| native path | 0.199 | FELL |
| native + joint-velocity clip | 0.199 | FELL |
| `LockAxesBelow = 0.2` | 0.196 | FELL |
| **`AssistMode`** (`IsaacStandCheckNewtonAssist.tscn`) | **0.545** | FELL at t=4 |

`AssistMode` composing on Godot's own balance layer survives roughly twice as long, which is
consistent with the remaining gap being actuator authority rather than the policy being wrong.

**The passive dynamics agree.** Zero action collapses in under 2 s in BOTH engines, and Godot's
zero-action trace is clean and quiet (`angVel` 0.52, `jointVel` 5.80). The body is not the problem.

---

## 3. Environment count — measured, do not re-derive

RTX 4080 SUPER 16 GB, Stand under XPBD, fresh process per row:

| envs | steps/s | sys RAM | VRAM |
|---:|---:|---:|---:|
| 4,096 | 252,120 | 11.8 GB | 2.7 GB |
| 8,192 | 428,552 | 12.4 GB | 3.8 GB |
| **16,384** | **572,219** | 13.3 GB | 5.9 GB |
| 32,768 | 645,849 | 15.4 GB | 10.2 GB |
| 65,536 | **86,312** | 22.7 GB | 15.5 GB |

**65,536 falls off a cliff** (7.5× slower, VRAM at 15.5 of 16.0). **16,384 is the default** — 89% of
peak at a third of the VRAM. 32,768 buys 13% for 2× samples per gradient update.

---

## 4. Traps already paid for — do not rediscover these

Every one is silent. None raise. Each produced a confident wrong answer for a while.

1. **`joint_pos`, `joint_vel`, `applied_torque`, `root_quat_w`, all `root_*` velocities and
   `projected_gravity_b` are FROZEN under XPBD** and read a clean, plausible zero. Everything
   per-BODY is live. This is not a bug — it is what a maximal-coordinate solver is, and it is the
   finding. Read state only through `p4f_newton/state.py`, which uses `newton.eval_ik` and the
   pelvis body.
2. **`eval_fk` must be called with `indices`.** Unmasked it rewrites `body_q` for every environment
   from a stale `joint_q`, so one env resetting teleports all of them upright. Symptom: `upright`
   exactly 1.0000, `ep_len` pinned at 479/480, `termination` exactly 0.0000 — a training curve that
   looks like a flawless stand. A policy trained on it for 334M steps fell **twice as fast as doing
   nothing**.
3. **`write_root_*_pose_to_sim_index` does not round-trip.** First reset lands correctly; every one
   after pins the pelvis at 1.640 (0.82 authored + 0.82 written) and then ignores the written value
   entirely. `NewtonRigState.reset_to` writes Newton's `joint_q` directly instead.
4. **Isaac Lab 3 changed the root quaternion from wxyz to xyzw.** `InitialStateCfg().rot` defaults
   to `(0,0,0,1)`. The 2.3.2 value `(1,0,0,0)` reads as x=1 — a 180° rotation about X — and the
   whole body spawns mirrored while `root_quat_w` still reports identity.
5. **`joint_q` / `joint_qd` are flat over ALL envs**, in blocks of `7 + 45` and `6 + 45`
   respectively (the counts differ). Slicing only the leading root entries leaks other envs' world
   positions into the joint array: `max|q|` reads 10.5 rad against ±0.5 limits.
6. **`joint_target_ke[:6]` is the root FREE joint** and is legitimately zero. Real joints start at
   index 6. Reading the first six looks exactly like dead actuators.
7. **`AppLauncher` is Kit-only** — dies with a bare `KeyError: 'EXP_PATH'`. Use `SimulationContext`
   on a fresh stage. This track is kit-less.
8. **Newton DOF order differs from `physx_dof_order` in 42 of 45 slots.** Verified causally
   (gravity off, one joint at a time): `eval_ik` order == `robot.joint_names`. It travels with the
   policy in the exported contract JSON.
9. **`torch.inference_mode()` breaks the env's resets** — persistent buffers refuse in-place writes
   with "Inplace update to inference tensor". Use `no_grad`, as rsl_rl does.
10. **torch's dynamo ONNX exporter writes weights to a `.onnx.data` sidecar.** The `.onnx` alone is
    13.8 KB and works locally (onnx.load follows the reference) but is broken the moment it is
    copied elsewhere. `export.py` folds it back in; a correct file is ~985 KB.
11. **PowerShell params are clobbered by dot-sourcing `config.ps1`** — it defines `$Envs` and
    `$MaxMinutes` itself, so `-Envs 4` silently became 4096. Capture params before the dot-source.
12. **`WarmupSeconds` must be 0 for a Stand policy in Godot.** The rig collapses unprovoked in under
    2 s, so a 1 s warmup hands the policy a body already at 32° tilt — far outside training. First
    actions went 2.286 (saturated) → 0.505 (healthy) with it at 0.

---

## 5. Godot side — what was changed and how to test

**Scene: `res://Scenes/RL/Isaac3/Stand/IsaacStandCheckNewton.tscn`** (F6). Runs 20 s, prints head/pelvis
every 2 s, ends with a `STOOD`/`FELL` verdict. `ZeroAction = true` in the inspector shows the
baseline - that comparison is the only thing that makes a result meaningful. **R restarts the
scene.** `IsaacStandCheckNewtonAssist.tscn` is the same thing with `AssistMode`, which composes the
policy on Godot's own balance layer and survives about twice as long.

| file | change |
|---|---|
| `IsaacRigContract.cs` | `LoadDofOrderOverride()`, `LoadJointVelocityClip()`, optional `dofOrderOverride` on `Load()` |
| `IsaacObservation.cs` | `UseHeightContacts`; `JointVelocityClip` bounding slice [55:100] |
| `IsaacActionSpace.cs` | `LockAxesBelow` (an experiment that did not work - see §2c) |
| `IsaacPolicyDriver.cs` | `PolicyContractPath`, `HeightContacts`, **`DisableHillLimit`**, **`DisableLoadCompensation`**, `LockAxesBelow`; SPD in `ApplyJointSpaceTorque`; `worstDof` and `demand`/`deliver`/`fvScale` diagnostics |
| `IsaacArena.cs` | forwards all of the above plus `WarmupSeconds` and **`DiagnosticInterval`**; R restarts |

**Four settings a Newton policy needs, and each fails silently without it:**

1. `PolicyContractPath` - carries `newton_dof_order` (differs from `physx_dof_order` in 42 of 45
   slots) and `joint_velocity_clip`.
2. `HeightContacts = true` - matches the task's height proxy for the four contact flags.
3. `WarmupSeconds = 0` - the rig collapses unprovoked, so any warmup hands the policy a fallen body.
4. **`DisableHillLimit = true`** - otherwise Godot's actuators switch off entirely under load. §2c.

`DisableLoadCompensation` is also set. It is principled - Isaac's drives are a plain PD and have no
gravity feed-forward - but unlike the others it did **not** show a clear outcome improvement, and a
single 20-second run of a chaotic ragdoll cannot resolve a difference that size. Treat it as
unproven.

Build with `dotnet build Physics4Fun.csproj`. Godot exe:
`D:\Programas\Godot_v4.7.1-stable_mono_win64\Godot_v4.7.1-stable_mono_win64_console.exe`.
Headless: `godot --headless --path . res://Scenes/RL/Isaac3/Stand/IsaacStandCheckNewton.tscn`

**Contact flags remain a documented approximation** - body height below 0.06 m, because Newton's
contact reporting does not surface through Isaac Lab's `ContactSensor`. Revisit for get-up, where
hands genuinely bear load.

**`project.godot` physics settings could not be shown to take effect in a headless run** - a
deliberately destructive control (`max_linear_velocity = 0.1`) changed nothing. Any project-setting
experiment needs redoing from the editor. The file was restored unchanged.

---

## 6. Where each phase actually got to

**Phase 0-1 (done):** NaN guard, pre-fix checkpoints quarantined, R restarts Isaac scenes.

**Phase 2 (superseded):** the parity harness was aimed at the observation, and the observation was
never the problem - the first step matches to three decimals. `scripts/slice_stats.py` replaced it
and is the more useful tool: it prints the same per-slice diagnostic Godot's driver prints, so the
two engines can be read side by side over a whole trajectory instead of at one instant.
`dump_reference.py` is still unrun and no longer a priority.

**Phase 3 (Stand): the actuator was wrong, and that was the whole story.** See 2b and 2c. Stand
itself trains fine - it hit 100% in fifteen minutes. What it was training against was not Godot.

**Phase 4 (Walk): ported and running, not yet trained.** `P4F-Dummy-Walk-Newton-v0`, inheriting
every fidelity fix through `StandEnvCfg`. Two things had to be rebuilt because Newton has no working
`ContactSensor` and no live `root_lin_vel_b`: air time is reconstructed from the same height proxy
the contact flags use, and body-frame velocity comes through `NewtonRigState`.

The reward is **product-form**, per `docs/RL-SESSION-INVARIANTS.md`:
`reward = w_track x posture x drive x yaw + w_air x posture x steps + penalties`. Posture GATES the
task terms rather than being paid for separately, so standing still under a movement command scores
exactly zero. Two defects were caught in the smoke run and fixed:

* `feet_air_time` read **-0.0197** per episode - `clamp(max=cap)` bounds only the top, so any flight
  shorter than the threshold scored negative and a foot that never left the ground scored zero. The
  term was paying for sliding. It needed `clamp(min=0.0, max=cap)`. This is the same sign inversion
  2.3.2 shipped, rediscovered by writing the warning and then committing a variant of it anyway.
* `action_rate` read **-1.71** against a total reward of -1.72 - the entire reward. Under a product
  form the positive terms are worth nothing until the policy can walk, while `action_rate` accrues
  every step regardless, so the cheapest policy was to freeze. Stand's weight also prices a normal
  gait as a fault. Cut from -0.01 to -0.003.

**It was trained for 45 minutes bootstrapped from the best Stand, and it does not walk.** Measured
at a commanded 0.8 m/s:

| | walk01 (3500 iters) | zero-action baseline |
|---|---:|---:|
| tracked speed | **-0.077 m/s** | 0.010 |
| upright fraction | **6.8%** | 13.1% |
| steps taken | 0.5 | 1.8 |
| distance | 0.651 m | 0.747 m |

**The reward said it was working and the behaviour was not.** `Episode_Reward/track` climbed 0.10 to
1.76 over the run - a seventeen-fold improvement - while the dummy learned to lean into the command
and topple. The reward is not wrong: `posture` gates it correctly, so the whole 1.76 is earned in the
first second and the body then lies on the floor for the remaining eleven. It is simply that a
rising task reward means nothing on its own when episode length is 55 steps of 720.

That is exactly why `evaluate_walk.py` reports `upright fraction` and `steps taken` beside the
speed - they caught it immediately. **`steps taken` 0.5 with any speed at all means sliding or
falling, never walking.**

**The conclusion is about Stand, not Walk.** Walk bootstraps from Stand and cannot stay upright long
enough to discover a gait while Stand itself only holds ~40% under a shove. Walk is ported,
registered, scored and ready; it should not be trained again until Stand is solid.

`scripts/evaluate_walk.py` scores it. There is deliberately no success condition - walking is
sustained behaviour with no goal state - so it reports a profile: tracked speed, distance, upright
fraction, and **steps taken**, because near-zero steps with a healthy speed means sliding. Baseline
(zero action, 0.8 m/s commanded): tracked 0.010 m/s, distance 0.747 m, upright 13.1%, 1.8 steps.

**Phases 5-6 (Perturbation, Run): not started, and Perturbation is partly moot.** A spawn push is
now part of Stand itself, because it had to be - see 2b. A separate task would still add mid-episode
impulses and the 18 N.s ball, but the reason it was queued (robustness) is now in the base task.

**Priority unchanged and now better justified: one Stand with a verified Godot transfer beats four
half-ported tasks.** Tonight found five separate reasons the transfer was failing, fixed four of
them, and measured the fifth.

## 7. Housekeeping

* **Nothing is committed.** Many untracked and modified files across `isaac_lab_3/`,
  `Source/RL/Isaac/`, `Source/Ragdoll/`, `Scenes/RL/Isaac3/`, `Models/`. Do not commit without asking.
* **`project.godot` was modified during an experiment and restored.** Verify it is clean.
* `rl/scripts/config.ps1` remains reverted to `$Task = "perturbation"` / `$MaxMinutes = 120`.
* The 2.3.2 tree (`D:\Proyectos\Juegos\Tools\IsaacLab`) is clean and untouched.
* Conda env is **`env_isaaclab3`** (Python 3.12), separate from the 2.3.2 `env_isaaclab` (3.11).
* Run selection skips anything under 50 iterations, so benchmark runs cannot be picked.
* **The statue chain stopped itself at 05:34** on a NaN the broken guard had been hiding. Its
  `SUMMARY.md` is preserved as the before-picture. Ignore its numbers.
* `logs/rsl_rl/p4f_smoke/` holds two throwaway 30-second config validation runs. Safe to delete.

### New scripts this session

| script | what it answers |
|---|---|
| `slice_stats.py` | prints Godot's per-slice observation diagnostic from the Isaac side, so the two engines can be read side by side over a trajectory |
| `probe_twist.py` | which half of Newton's root twist is linear (measured: [0:3] linear, [3:6] angular) |
| `evaluate_walk.py` | Walk profile - tracked speed, distance, upright fraction, steps taken |
| `evaluate_stand.py --push / --effort_scale` | scoring under a disturbance, and sweeping the torque budget to locate Godot's effective authority |
| `check_reward_terms.py` | static guard: which reward terms each task actually ends up with, failing if a regulariser vanished. 2.3.2's Walk lost `action_clip` this way and every metric looked healthier for it |
| `train.py --experiment / --init_from / --push` | isolating a variant's checkpoint tree, seeding from another task, choosing the training disturbance |

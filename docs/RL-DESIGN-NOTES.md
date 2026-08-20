# RL track: design notes and next steps

Status: infrastructure (M1-M6) is **complete and proven**. Nothing below is about plumbing; it is
about the fact that the current agent *cannot learn to get up*, why, and what to change.

---

## 1. Measured reality

| | value | meaning |
|---|---|---|
| `ep_rew_mean` over 500k steps | 10.2 -> 10.7 | **flat** - no learning occurred |
| Throughput (40 proc, speedup 16) | ~1800 steps/sec | 1M steps ~ 9 min |
| Actuated bones | 4 of 16 | thighs + upper arms only |
| Observation width | 32 floats | pelvis + the 4 actuated bones |
| Effective physics rate | 60 Hz | **project is configured for 120 Hz** |

The flat reward curve is the headline: 500k steps of correct, fast training produced no
improvement. That is not a tuning problem, and no amount of extra steps fixes it.

---

## 2. Why the dummy "shakes on the ground"

This was asked directly, and the answer is not "the policy is bad" - the policy is barely involved.

`ActiveBone` is a PD/SPD actuator: it drives each joint toward a **target rotation**. In the RL
state the base target is the standing rest pose, and the RL action is applied as a bounded
*offset* on top of it (`FeedForwardTargetOffset`, +/-0.6 rad).

So while the body is lying prone, every joint is being commanded toward a **standing** pose it
cannot possibly reach from that configuration. The actuators saturate, fight the ground and each
other, and the visible result is jitter. Two consequences:

1. **Before any policy exists**, the shaking is 100% the PD controller chasing an unreachable
   target. It is not a policy decision - there is no policy. (`ReinforcementLearningIdleStiffness`
   softens this but does not remove the cause.)
2. **With a policy**, +/-0.6 rad of residual is far too small to cancel a fundamentally wrong
   baseline. The policy's entire action budget is spent fighting its own starting assumption
   rather than producing useful movement.

**This is an action-parameterisation bug, not a training problem.**

---

## 3. Why the reward teaches nothing

`UprightAliveReward` = `delta * (1 + clamp(cos(tilt), 0, 1))`.

- The **alive term is constant.** Episodes end on a fixed 8s timer, not on failure, so every
  episode banks the same alive bonus regardless of behaviour. A constant has zero gradient.
- The **upright term is unreachable.** Getting `cos(tilt)` to move requires a coordinated
  whole-body movement that 4 bones cannot produce, so the term stays pinned near its floor and
  contributes almost no variance.

Net: the reward is nearly constant across all behaviours, so PPO has nothing to climb. This
matches the observed 10.2 -> 10.7.

---

> **SUPERSEDED (sections 4–5).** Residual RL was the recommendation at the time; pure RL was
> chosen instead, and the Euphoria and RL tracks are now fully decoupled. These two sections are
> kept for the reasoning, not as instructions. Current state is in "Findings after 70M steps"
> below and in `RL-TRAINING.md`.

## 4. Recommended direction: residual RL over the Euphoria controller

This matches the "keep the Euphoria idea, let RL do its magic" instinct, and it is the standard
approach for exactly this problem (learning on top of a hand-authored controller rather than
replacing it).

**Change the baseline the policy corrects.** Instead of `standing rest pose + RL offset`, use:

```
target = ProceduralGetUpTrajectory(phase)  +  RL residual
```

The project already has the trajectory machinery (`Source/Ragdoll/Trajectories/`,
`GetUpPhaseController`). The policy then starts from a *competent* baseline and only has to learn
the corrections the hand-tuned controller gets wrong - which is a far easier problem than
discovering get-up from noise, and it preserves all the biomechanical work already done.

**Second action channel - learned impedance ("measure the weights").** Let the policy also output
a per-bone stiffness scale, feeding the existing `MuscleStrength` / `LoadBearingGainScale` seams.
This is genuinely Euphoria-like: real muscle control modulates *tension*, not just target angle,
and stiffening/relaxing at the right moment is most of what makes a get-up look human. It is also
exactly the "measure the weights" idea - the policy learns how hard to push, per joint, per phase.

---

## 5. Concrete changes needed (in priority order)

1. **Action space** - expand from 4 to the joints that actually matter for getting up: hips,
   knees, ankles, shoulders, elbows, spine (~12 bones). Add the stiffness channel above.
   Re-parameterise as a residual over the procedural trajectory.
2. **Reward** - replace with progress-based shaping. Use **potential-based** shaping
   (`r = gamma * PHI(s') - PHI(s)`, with `PHI` = head/pelvis height) so the optimal policy is
   provably unchanged by the shaping term. Add: uprightness, torque/energy penalty, foot-slip
   penalty, and a terminal bonus for standing stably. The key property to preserve is a **smooth
   gradient from prone** - height gives that; `cos(tilt)` alone does not.
3. **Observation** - use the quantities `RagdollTelemetryRecorder` already computes: all 16 bones,
   CoM position/velocity relative to base of support, foot/hand contact flags, ground clearance,
   and (if residual) the trajectory phase. Replace Euler angles with quaternions or 6D rotation -
   Euler wraps discontinuously, and two near-identical poses can produce wildly different vectors
   near a boundary.
4. **Episode termination** - end on *failure*, not only on a timer, so the alive bonus regains
   meaning. Keep the timer strictly as a backstop.

The interfaces added during the SOLID pass (`Source/RL/Interfaces/IRlComponents.cs`) make 2 and 3
drop-in replacements: implement `IRlRewardFunction` / `IRlObservationBuilder` and swap the
assignment in `RagdollRLBridge._Ready()`. Nothing else changes.

---

## 6. Open issue: physics rate mismatch

`Sync` forces `Engine.physics_ticks_per_second = speedup * 60` and `time_scale = speedup`, so
effective simulation `dt` is **1/60**. The project's `project.godot` sets **120 Hz**, and the
PD/SPD gains were tuned at 120 Hz.

So a policy trained here experiences *different actuator dynamics* than the procedural controller
does in `TestChamber.tscn`. Not a bug, and `speedup` is not the cause (dt stays 1/60 at any
speedup), but it must be decided deliberately - either retune for 60 Hz or override the tick rate
in the RL arena - or behaviours will not transfer cleanly between the two tracks.

---

## 7. Throughput reference (measured, this machine)

Plateau is ~1800 steps/sec at **40 processes, speedup 16**. Going to 50 processes gained only 2%
while per-process throughput fell 44 -> 36, so 40 is the sweet spot.

The bottleneck is **single-threaded Python marshalling**, not the physics:
`StableBaselinesGodotEnv.step()` loops over every env sequentially to send, then again to receive,
JSON-encoding each on one thread. That is why one CPU core sits pegged while the rest are
underused, and why adding processes past ~40 stops helping.

---

# Phase 2 implemented: pure RL (full separation from Euphoria)

Decision: the RL dummy is **pure RL** — no procedural pose reaches it. Euphoria continues
untouched in `TestChamber.tscn` from the same scene file.

## The coupling was thinner than it looked

`ActiveRagdoll.tscn` splits cleanly into a **body** layer (ActiveBone muscles, Generic6DofJoint3D
limits, collision/mesh) and a **brain** layer (`HumanoidRagdoll`, `BalanceController`,
`RagdollDebugInput`). Three of four procedural systems were *already* disabled in the RL state
(balance strength 0.0, state-machine passthrough, debug input gated). Exactly one thing still
injected procedural pose: `UpdateBoneTargetRotations` writing `TargetLocalRotation = restPose`.

That single write was the "shaking": a standing target commanded onto a prone body every tick,
which the actuators saturate against and which the policy could not overcome, since its action
only composed on top of it.

## What changed

- `HumanoidRagdoll.UpdateBoneTargetRotations` returns early for `ReinforcementLearning`.
- `SeedTargetsToCurrentPose()` on episode start → zero PD error at reset, so all subsequent motion
  is attributable to the policy.
- `RagdollRLBridge` writes `TargetLocalRotation` directly (rest * action). Action range widened
  0.6 → 2.6 rad to span the real joint envelope; 4 → 12 controlled bones (36 actions).
- `GetUpObservation` (106 floats): root-relative **quaternions** (not wrap-prone Euler), pelvis
  up-vector, CoM offset/velocity, head height, and hand/foot **contact flags**.
- `GetUpProgressReward`, `GetUpTermination` (success on *held* standing, not momentary).
- Physics restored to **120 Hz** (`EnsurePhysicsTickRate`, derived from `Engine.TimeScale`);
  `action_repeat = 8` → 15 Hz control, chosen against the 40 ms actuator smoothing constant.

## Three bugs found and fixed by measuring, not reasoning

1. **Shaping drift (the big one).** `gamma*PHI' - PHI` is only valid applied once per agent
   decision with the trainer's gamma. It ran once per *physics tick* (8 per decision), injecting a
   constant `(gamma-1)*PHI` every tick: ~`-0.1*0.24*968 = -23` per episode. Measured exactly that —
   `ep_rew_mean` pinned at -23, flat, a floor no policy could escape. Now uses the undiscounted
   difference, which telescopes exactly to `ProgressWeight * (PHI_end - PHI_start)` regardless of
   tick rate.
2. **Effort penalty mis-scaled.** A single `ReferenceTorque = 200` against a rig whose `MaxTorque`
   spans 50–1800 N·m: a saturated hip scored 9.0, a wrist 0.25. Now a per-bone capacity fraction.
3. **Observation size mismatch.** `Size` reported 103 while `Build()` emitted 106 (root block
   miscounted as 15; it is 18). Harmless only by luck — the plugin derives the space from the
   actual array — but the not-ready fallback returned a wrong-width vector that would desync the
   transport.

## Findings after ~70M steps (measured)

Three runs, all pure RL, all with the pipeline verified working end to end.

| run | start pose | steps | outcome |
|---|---|---|---|
| `getup_v1` | prone | 0.5M | `ep_rew_mean` flat 10.2 → 10.7 |
| `getup_v2` | prone | 63.4M | `ep_rew_mean` −0.06 → +0.86, **zero successes in ~524,000 episodes** |
| `getup_v3` | standing (RSI) | 6.9M | `standing/all` drifted 0.015 → 0.012, zero successes |

`getup_v2` is the informative one. Decomposing where its +0.9 of reward gain came from:

| term | Δ over 63M steps | share |
|---|---|---|
| `upright` | +0.61 | 65% |
| `shaping` | +0.27 | 29% |
| `effort` | +0.06 | 6% |
| `terminal` | 0 | 0% |

In physical units that is a torso tilt improvement from **77.1° to 72.6°** — 4.5 degrees for 63
million steps, against a 30° success threshold. The agent converged (entropy −51 → −37, std
1.00 → 0.71) on a local optimum best described as "lie propped up, resist the post-reset settle,
spend no torque".

### The defect that plausibly explains all of it

A single `MaxActionAngle = 2.6` rad was applied uniformly to all three axes of all twelve
controlled bones. The rig's actual limits are nothing like uniform — the knee and elbow roll axes
are `±0.10` and `±0.15` rad. Measured across all 36 axes: **71.8% of the commanded action range
lay beyond a hard stop.**

Commands past a stop do not produce a smaller motion. They pin the actuator at full torque with a
permanent tracking error, so *every* out-of-range action produces the same physical result. That
flattens the policy gradient over most of the output space — an exploration problem that no reward
shaping or curriculum can repair, because the information simply is not there.

Fixed by `JointLimitedActionSpace`, which scales each component into that axis's own range. Effort
penalty fell 15× (−0.58 → −0.038) and `standing/grounded` roughly doubled in a short probe.
Whether it unblocks learning is not yet established.

### A measurement bug found while verifying that fix

`ActiveBone.IsTargetWithinJointLimits()` is unreliable for axes whose X limit exceeds ±π/2. Godot
decomposes with Euler order YXZ, whose principal branch cannot represent |x| > 1.571 rad; four of
this rig's axes do (`Thigh x +2.10`, `Shin x −2.60`, `Forearm x +2.60`, `UpperArm x +3.00`). A
commanded `(2.6, 0.05, 0.05)` returns as `(0.54, −3.09, −3.09)` and reports a violation that never
happened. This affects the procedural track's telemetry too. Documented in place; a correct check
needs swing-twist decomposition.

### Method note

Every conclusion above came from instrumenting and measuring. Every wrong conclusion in this
project came from reasoning about the scalar reward without decomposing it — including a confident
claim that the upright term contributed "exactly zero" while prone, when it was in fact the
largest positive term. Add the diagnostic before forming the theory.

---

# Findings at 12M steps (`getup_v4_2`, measured)

Regression over all 1,280 rollouts in the run's four event files. `t` is the slope's t-statistic;
anything past ~3 is a real trend, so these are not close calls.

| metric | 0M | 12M | slope /1M | t |
|---|---|---|---|---|
| `standing/all` | 0.0182 | 0.0320 | +0.00122 | **116.9** |
| `standing/grounded` | 0.0535 | 0.0948 | +0.00358 | **135.7** |
| `standing/icp` | 0.0195 | 0.0358 | +0.00142 | **118.1** |
| `standing/tilt` | 0.0720 | 0.1033 | +0.00276 | **106.1** |
| `standing/head` | 0.0721 | 0.1012 | +0.00248 | **104.1** |
| `rollout/ep_rew_mean` | −6.66 | −6.29 | +0.0332 | 11.3 |
| `train/std` | 1.0005 | 0.6671 | −0.02928 | **−141.5** |

**The agent is learning and `ep_rew_mean` hides it.** Every `standing/*` rate rises with
overwhelming significance, and the rate is *accelerating* — this session's `standing/all` slope was
+0.00205/1M against the full-run +0.00122. Judging this run on total reward would have concluded
the opposite. `icp` sits just above `all`, so balance is the single binding sub-condition and the
others follow it.

**Zero successes in 12M steps.** `episode_end/Standing` and `episode_end/Inverted` do not exist as
tags — never fired once. `episode_end/TimeLimit` is 1.0000 throughout, `reward/terminal` is 0.0000.
Both terminal conditions in `GetUpTermination` are dead in practice.

**~90% of every episode was wasted.** `standing/all` = 0.032 is 31 of 968 ticks, and
`StandingHoldSeconds` needs 90 *consecutive*; under RSI those 31 are at the front. `reward/shaping`
telescopes to `10 × (Φ_end − Φ_start)`, so −8.08 with Φ_start clamped to 1.0 puts the head at
**29 cm** at the end of every episode. Only ~12 of 121 decision steps carried anything useful. Fixed
by `MaxEpisodeSeconds` 8 → 3, which raises the informative share from ~10% to ~27% at unchanged
throughput. It does *not* make success closer — the failure happens in the first second regardless.

## The truncation bug (framework-level)

`godot_rl` computes a truncation flag and discards it. `StableBaselinesGodotEnv.step` collects
`all_trunc` and returns only `all_term`; `godot_env.py:221` fills both slots from the same `done`
boolean under a standing `# TODO update API to term, trunc`. Neither `TimeLimit.truncated` nor
`terminal_observation` ever reaches SB3, and PPO bootstraps only when **both** are present
(`on_policy_algorithm.py:237-245`). So **100% of episode ends were trained as absorbing with
V(s_T) = 0.**

This was near-harmless purely by accident: at 8 s the terminal state was always prone, whose true
continuation value really is ≈ 0 — which is why `explained_variance` still read 0.92. It stops being
harmless as soon as episodes are short enough to truncate in states that still have value, so
`TruncationBootstrapWrapper` (`rl/train.py`) had to land *with* the episode-length change, not after
it. It sits inside `VecMonitor` so `ep_rew_mean` keeps measuring real environment reward.

Gated on `episode_end_reason == "TimeLimit"`, so genuine terminals keep V = 0 correctly.

## Known residual: an off-by-one in the framework's obs ordering

`godot_rl` returns the *terminal* observation on a done step, where SB3 expects the *reset* one. So
`_last_obs` after a done is the terminal obs: one action per episode is computed from the previous
episode's final pose, and one rollout-buffer sample pairs a prone observation with the fresh body's
reward. GAE is unaffected (`_last_episode_starts` is still correct).

Costs 1/45 of samples at 3 s, 1/30 at 2 s — one of the reasons not to shorten further. Fixing it
means restructuring the bridge's reset ordering, which is the most delicate part of the pipeline;
not worth it for ~2% of samples.

## Exploration was closing before the goal was found

`train/std` fell 1.0005 → 0.6671 and was still *steepening* (−0.0148/1M, t = −294 in the final
session) with `ent_coef` at 0.0001 holding nothing open. `standing/act_saturation` rose 0.342 →
0.387 with its rate tripling — a narrowing distribution pushed toward the rails, i.e. converging on
bang-bang control having never once reached the goal. Extrapolated to the ~42M steps `standing/all`
needs, std would reach 0.23.

Raised `ent_coef` to 0.001. Not 0.01: entropy for this 36-dim policy at std 0.667 is 36.5, so 0.01
would contribute 0.365 against a total `loss` of 0.14 and swamp it.

## PPO trust region

`approx_kl` climbed 0.0128 → 0.0299 (t = 32.0) against `target_kl` = 0.02, so SB3 aborts at 1.5× =
0.03 on **every** iteration, cutting epochs to 3–5 of 10. The `batch_size` = 2048 fix did work
(`clip_fraction` is 0.16, down from the 0.43–0.45 that prompted it) but did not finish the job. Left
alone for now — it is a safety valve doing its job, and changing it at the same time as three other
things would make the result unattributable.

## Throughput at 32 procs / speedup 8

**2,064.5 steps/s sustained over 1,800 s**, against 2,162 measured in a 180 s sweep — the gap is
checkpoint overhead plus sustained thermals. `time/fps` slope over the final session was −1.6/1M at
t = −1.9, i.e. statistically flat: it does not degrade under load. 40 procs measured 2,258 (+4.5%),
which is the ceiling for process-vectorized Godot on this CPU. The GPU is nowhere near the
bottleneck. Sample *composition* was worth far more than any remaining throughput gain.

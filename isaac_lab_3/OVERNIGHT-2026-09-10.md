# Overnight plan, 2026-09-10 00:10 to ~09:00

Nine hours, autonomous, 15-minute training sessions so progress is validated rather than assumed.
Decisions are made from the measurements and recorded here as they happen; nothing waits for a
reply. Results are appended to the log at the bottom as each session lands.

## Where the night starts

- **Balance is solved.** 100% upright, 16/16 over 40 s in a quiet room, actively - 20 deg of joint
  motion, under 10% of human strength. `balance_policy.onnx` is this brain.
- **Perturbation has never been trained.** Its whole curriculum ran against a ball that could not
  reach the dummy (a projectile's range is `v^2/g`: 0.60 m at 2.42 m/s, from a 2 m standoff). The
  gun was fixed yesterday; under real fire the policy falls about a second after the first ball, at
  any impulse from 22 N.s upward.
- **Walk has never been trained on this plant at all.** Every walk result predates the torque
  actuators, the rest keyframe and the height-target fix.
- Two trainer defects fixed before starting: the optimiser state now travels with a checkpoint
  (continuations opened at KL 0.059 and lost ~8 iterations to a clamped learning rate; now 0.0086),
  and the curriculum can no longer double-promote or blow the policy apart at a promotion.

## Phase 1 - Perturb: get hit, recover, stand back up (target ~3 h)

Seeded from the balance brain, chained 15-minute sessions, curriculum from 1.5 m/s (22 N.s).

**Pass:** survives 40 s under fire at >= 3.0 m/s (45 N.s) every 4-7 s, 12+ of 16 envs, and the
recovery is visible in the numbers - feet move, stance width changes, and the body returns to
within a few degrees of upright between hits rather than settling into a crouch.

**What "recovery" has to mean in the reward.** The existing `3.0 * support` pays for the COM sitting
over the feet, which a widened stance satisfies without ever lifting a foot - measured, single
support is 0.0%. If sessions plateau below the pass mark, the reward gains an explicit
return-to-pose term rather than more minutes.

## Phase 2 - Walk: a normal gait, 10 s and ~1 m/s (target ~3 h)

Seeded from the best balance brain - the 105-observation layout exists so a balance brain can seed a
walk one without surgery.

**Pass:** 10 s upright while moving, >= 0.8 m/s achieved against a 1.0 m/s command, and a gait that
reads as walking.

**Reward changes first, before any GPU time**, because "not weird" has to be paid for:

- **Penalise flight.** Nothing currently punishes both feet leaving the ground, which is what a hop
  is. This is the single most important term for "normal".
- **Air time 0.35 s -> 0.25 s.** 0.35 s above a 6 cm clearance is a run, not a walk.
- **Raise `track_yaw` 2.0 -> 3.0**, closer to `track_lin`'s 4.0, because turning quality matters as
  much as speed for a controller.
- **Command sampling shaped like a joystick**: explicit slices for stand, straight line, turn in
  place and arc, instead of three independent uniforms that mostly produce combinations no player
  will ever send. Turn-in-place - stand still and rotate - is almost never sampled today.

## Phase 3 - only if Phase 2 passes (target ~1.5 h)

In order, and only while the previous one holds: **turn in place**, then **strafe**, then **run**
(raise `CMD_FORWARD` above 1.0 m/s and let air time grow). Each gets its own sessions and its own
score; none of it is worth starting on a gait that cannot hold a straight line.

## Rules for the night

1. One trainer at a time. A shared GPU invalidated an A/B on this project once already.
2. Every session is scored on **CPU MuJoCo** with `eval.py`, reading the `POLICY, under fire`
   section - the quiet-room row is what produced yesterday's wrong conclusion.
3. A session that ends worse than it started is not chained from; the previous checkpoint is.
4. Process hygiene after every phase.
5. Nothing is promoted into `balance_policy.onnx` or `locomotion_policy.onnx` unless it scores
   better than what is already there.

---

## Log

### 00:35 — Phase 1 under way, and the walk baseline for the record

Trainer fixes verified before anything else ran. A continuation used to open at KL 0.059 with the
learning rate clamped to 5.9e-05 for ~8 iterations; carrying Adam's state it now opens at
**KL 0.0086, lr held at 4.50e-04**, and episode length continues instead of collapsing.

**Perturb session 1** (15 min, from `perturb_real/model_29`), against a ball that connects:

```
it 10  ep_len 117.0   speed 1.50 (22 N.s)
it 60  ep_len 662.5   speed 1.50
it 70  ep_len 775.4   speed 1.50
it 75  promoted to 1.65 (25 N.s)   kl +0.0185     <- a clean promotion
```

Episode length **117 -> 775 of 1199**. The promotion at it 75 passed with KL 0.0185; the one that
destroyed a policy yesterday was +11.88, so the cooldown and the iteration dwell are doing their
job. Scored at 1.5 m/s: 48.9% upright, 1 of 16 survived 40 s, and **steps taken 0.50 (max 2)** -
where every previous perturb policy measured exactly 0.00. It has started using its feet.

**Walk baseline** - the unaided plant, so tonight's walk numbers have a reference:

| | value |
|---|---|
| upright | 3.9 % |
| never fell | 0 of 8 |
| time to fall | 0.76 s |
| travel along the command | **-0.58 m** (backwards, i.e. it topples) |

Every manoeuvre reads identically because the body falls before the command means anything.

**The flight penalty is live and correctly signed**, checked directly rather than assumed: with the
same pose and command, feet down scores **+6.363** and both feet airborne **+4.445**. Hopping now
costs 1.9 per step, so a gait that leaves the ground is strictly worse than one that does not.

### 01:15 — two flaws in my own session driver, both found by running it

**1. Sessions were scored at their own curriculum stage, so they were not comparable.** A session
that earns a promotion is then judged on a harder task and reads as worse than the one before it -
and a chain-only-if-improved rule rejects precisely the sessions that made progress. Every session
is now judged at a **fixed 3.0 m/s (45 N.s)** and its own stage is reported alongside.

**2. The chain rule rejected noise as regression.** Session 2 took episode length from 252 to 735
and was discarded because its reference score moved 19.9 -> 19.0 - well inside the noise of a
16-env scoring run. Scoring now uses 32 envs, and only a regression of more than 3 points is
rejected. `best` still tracks the high-water mark, so a slow drift cannot walk the chain away from
a good policy one tolerance at a time.

Both were caught in the first hour, which is the argument for scored 15-minute sessions over one
long unattended run.

### Perturb, what the sessions show so far

| session | end ep_len | stage | at 3.0 m/s | at its own stage |
|---|---|---|---|---|
| p1 | 775 | 1.50 -> promoted 1.65 | - | 48.9% |
| s1 | 252 | 1.65 | 19.9% | 34.8% |
| s2 | **735** | 1.65 | 19.0% | 37.8% |

Episode length is climbing at 25 N.s, and the stage score with it (34.8 -> 37.8). The reference at
3.0 m/s is flat at ~19%, which is the honest reading: the policy is learning to take a 25 N.s hit
and has not yet been asked to take a 45 N.s one.

**A pattern worth watching**: within a session the learning rate swings over a 40x range - 4.4e-05
to 1.71e-03 - and episode length oscillates with it (358 -> 181 -> 252 inside one session). The
adaptive controller is chasing KL spikes on a near-deterministic policy. If the next two sessions
oscillate the same way, `--lr_max` comes down from its 1e-2 default, which bounds how far any one
update can move the policy rather than reacting after it has moved.

### 03:00 — the biggest defect of the night: every session's FIRST update was destroying the policy

Chained sessions kept losing ground for no visible reason. The cause was in the opening line of each
session's log all along - the KL of the very first update, and it got worse every time the policy
sharpened:

```
p1  kl +0.0640      s2  kl +0.7757
s1  kl +0.0437      s3  kl +5.9676   (worst seen in-session: +23.0)
```

Against a 0.01 target. Session s3 opened at +5.97 and never recovered - episode length 764 -> 116,
6.0% at the reference, and the driver correctly threw it away. That is a whole 15-minute session
lost to one bad step, and it was going to keep happening more often as the policy improved.

**A seeded start is a promotion in every way that matters**: the critic is about to see data from a
policy it never fit, so the first advantages are wrong in one direction, and on a near-deterministic
policy the resulting gradient is enormous. The learning-rate cooldown already written for curriculum
promotions applies unchanged, so `promoted_at = 0` when `--init_from` is given.

Measured on the next session:

```
before   it 1  kl +4.8680   lr 7.59e-04
after    it 1  kl +0.0040   lr 1.50e-04      <- 1,200x smaller
```

The trainer re-invokes per session, so the fix landed mid-run without restarting anything.

### Perturb progress, judged at a fixed 3.0 m/s

| session | end ep_len | stage reached | at 3.0 m/s | at its own stage |
|---|---|---|---|---|
| s1 | 764 | 1.82 (27 N.s) | 19.9% | 38.0% |
| s2 | 528 | 2.00 (30 N.s) | 19.4% | 29.9% |
| s3 | 116 | 2.00 | 6.0% | 6.0% | **rejected** - the first-update blow-up above |
| s4 | 528 | 2.00 | 18.8% | 31.8% |

The curriculum is climbing (1.50 -> 1.65 -> 1.82 -> 2.00) and the stage score holds around 30% as
the difficulty rises. The 3.0 m/s reference sits at ~19% because the policy has not been asked to
take a 45 N.s hit yet - it is five promotions away.

### 04:50 — Phase 1 closed, Phase 2 (Walk) started

Eight scored sessions, every one judged at a fixed 3.0 m/s:

| session | ep_len | stage reached | at 3.0 m/s | at its own stage |
|---|---|---|---|---|
| 1 | 764 | 1.82 (27 N.s) | 19.9% | 38.0% |
| 2 | 528 | 2.00 (30 N.s) | 19.4% | 29.9% |
| 3 | 116 | 2.00 | 6.0% | 6.0% | **rejected** |
| 4 | 528 | 2.00 | 18.8% | 31.8% |
| 5 | **1081** | 2.00 | 19.6% | 36.8% |
| 6 | 626 | 2.20 (33 N.s) | 20.8% | 33.7% |
| 7 | 599 | 2.20 | 20.9% | 33.3% |
| 8 | 601 | 2.20 | **21.1%** | 30.9% |

**What moved:** the curriculum climbed 1.50 -> 2.20 m/s (22 -> 33 N.s), episode length peaked at
1081 of 1199 - 18 of a 20-second episode while being hit - and the fixed reference rose from 19.9%
to 21.1%. Session 5, the first run after the first-update fix, produced the best episode length of
the night.

**What did not:** nothing survives 40 s at 3.0 m/s yet, so Phase 1's pass mark is not met. The
reference moves about a point per session, and the curriculum needs four more promotions to reach
the difficulty it is judged at. This is progress at a measurable rate, not a plateau - but it needs
hours, not one more session, and Walk has had none at all.

So Phase 1 stops here on schedule and Phase 2 takes the remaining time. The best perturb brain -
`perturb_n_s8/model_80` - seeds the walk, which is what the shared 105-observation layout is for.

### 08:50 — Phase 2 results, and the flight penalty bracketed by measurement

Walk had never been trained on this plant. Baseline - the unaided body - travels **-0.58 m** along
the command, i.e. it topples backwards, 3.9% upright, falling at 0.76 s.

| session | ep_len | upright | vx | **along the command** | airborne | note |
|---|---|---|---|---|---|---|
| baseline | - | 3.9% | -0.17 | **-0.58 m** | 5.7% | the unaided plant |
| s1 (old metric) | 140 | 6.1% | +0.35 | +0.9 m | 21% | lunging |
| s2 (old metric) | 763 | 6.4% | **-0.32** | - | - | walking BACKWARDS, wrongly chained |
| m_s1 | 233 | **59.0%** | +0.05 | **+1.08 m** | **3%** | chained |
| m_s2 | 184 | 51.5% | +0.03 | **+1.34 m** | 4% | chained - best of the night |
| m_s3 | 134 | 6.6% | +0.38 | +2.34 m | 21% | rejected - bought distance by lunging |
| m_s4 | 781 | **100.0%** | +0.00 | -0.01 m | 0% | rejected - a statue |

**The third driver flaw, caught by the table.** Session 2 walked backwards at -0.32 m/s against a
+0.60 command and was chained anyway, because I had reused a 3-point tolerance meant for
percentages on a metric measured in m/s. The walk metric is now **upright-weighted metres along the
commanded heading** - signed, so going the wrong way is negative, and scaled by uprightness so
distance cannot be bought by diving.

**The flight penalty is now bracketed rather than guessed.** Two sessions, same seed, one knob:

```
1.0  ->  +2.34 m along, 21% airborne,   6.6% upright     lunging
2.5  ->  -0.01 m,        0% airborne, 100.0% upright     a statue
```

At 1.0 the `4.0 * track_lin` term outweighs a hop, so hopping pays. At 2.5 standing still is safer
than any step. It now sits at **1.5**, inside the bracket, and the bracket itself is the useful
result: anything outside it produces one of those two failures.

**Best of the night, both promoted:**

- `balance_policy.onnx` <- `perturb_n_s8/model_80` - curriculum 2.20 m/s (33 N.s), 21.1% at the
  fixed 3.0 m/s reference, against the previous brain's 19.1% at 2.0 m/s and 0/16 survival.
- `locomotion_policy.onnx` <- `walk_n_s2/model_77` - **+1.34 m along the command at 51.5% upright
  and 4% airborne**, against a baseline of -0.58 m. `MujocoWalk.tscn` now loads it.

Neither meets its pass mark. Perturb needs four more curriculum promotions to reach the difficulty
it is judged at; walk moves in the right direction, upright, without hopping, but at 0.05 m/s rather
than the 0.8 the target asks for. Both are measurable progress from a standing start, on a task
that had never been trained.

### 09:10 — correction: the wrong walk checkpoint was promoted, and the right one is far better

`walk_n_s2` and `walk_m_s2` differ by one letter: the first is from the driver run that was scoring
on raw `vx` and chained a policy walking BACKWARDS, the second from the corrected run. I promoted
the first. Scored side by side over 20 s:

| | upright | never fell | time to fall | along the command |
|---|---|---|---|---|
| `walk_n_s2/model_77` (promoted in error) | 12.9% | 0% | 2.57 s | **-1.07 m** |
| `walk_m_s2/model_81` (the driver's actual best) | **91.2%** | **62.5%** | **18.22 s** | **+0.58 m** |

`locomotion_policy.onnx` now holds the correct one.

**It stays up for 18 of 20 seconds and 62.5% of runs never fall** - so the "walk for more than 10
seconds" half of the target is met. What it does not do is travel: 0.017 m/s against a 0.60 command,
and `steps 0.0` in every manoeuvre. It has learned to stay upright under a movement command by
barely moving, which is the honest reading of `single support 0.7%` and `both feet airborne 2.1%`.

That is the statue failure again, in a milder form - and the flight-penalty bracket explains why.
The remaining nine sessions confirm it: every one that moved (s3 +2.34 m, s6 +2.16 m) did it at
21-33% airborne and under 7% upright, and every one that stayed upright (s4, s9 at 100%) travelled
nothing. **The reward has no term that pays for a STEP** - `air_reward` pays for a foot landing and
`single_support` for one foot down, but neither is reachable from a stationary stance, so the
gradient from "stand" to "walk" is flat. That is the first thing to fix, not more minutes.

### Final state of the night

| | before | after |
|---|---|---|
| balance under fire | falls to any hit, 0/16 | curriculum 2.20 m/s (33 N.s), ep_len 1081/1199 |
| walk, upright | 3.9%, falls at 0.76 s | **91.2%, 18.22 s, 62.5% never fall** |
| walk, travel | -0.58 m (topples) | +0.58 m, but 0.017 m/s |
| trainer | 5 defects, all silent | all 5 measured and fixed |

Both scenes load the corrected brains. No process left running.

### Epilogue — the evening of 2026-09-10, and 2026-09-11

What the closing note asked for, and what became of it:

* **Perturb.** The reward did pay for the wrong thing: `recover_step` paid single support while off
  balance, and a stomp satisfies that. It now pays swing-foot velocity toward the escaping centre of
  mass - after a `cvel` frame error was found in the foot velocity it reads. The gate measured the
  wrong thing too: its "one hit" fired two. With both fixed, four sessions took survival of a single
  30 N·s hit from 41.0% to 68.8% (paired z +4.20 on the last), and `perturb_z_s4/model_253` ships.
* **Walk.** The step reward fixed the flat gradient. Then the tracking kernel paid a statue 44-86% at
  stage-1 speeds (made relative to the command), the first gait walked in circles (heading hold), and
  two sessions died to NaN from one exploding world (bounded and sanitised). `walk_g_s1/model_254`
  walks 8 m straight in 40 s at 0.25 m/s, 100% upright, and ships.
* **Godot** applies the heading hold, through the contract's new `command` block.
* **2026-09-11.** The uncommitted track was reviewed and restructured with no change in behaviour -
  a ball-free base environment, scorers that write JSON for one `scoring.py`, per-task tables instead
  of task branches, and the trainer, exporter, generator and C# driver split into named parts - each
  step gated on bit-identical traces, identical verdicts, identical contracts or byte-identical
  models. `mujoco_rig/STATUS.md` has the current state.

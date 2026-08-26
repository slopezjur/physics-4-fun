# Results — first Isaac Lab session

Everything here was produced in one overnight session, from `Scenes/ActiveRagdoll.tscn` to trained
policies. Numbers are measured, not estimated; where something did not work it is written down as
such.

## Summary

| task | wall clock | result | policy |
|---|---:|---|---|
| **Stand** | 39 min | 100% strict standing success, 3.9% ever-fell, 5.8% of torque used | `exported/stand_policy.onnx` |
| **Walk** | 71 min | real gait at 0.59 m/s, 1.38 steps/s, 3.5% ever-fell | `exported/walk_policy.onnx` |
| **Perturbation** | 59 + 26 min | 97.5% survival at the real 18 N·s limb shot; 86.5% at 2.0x passive tipping energy | `exported/perturb_policy.onnx` |
| **Run** | 55 min | **did not run** — a fast walk at 1.86 m/s, 0.5% flight phase | `exported/run_policy.onnx` (weakest) |

All four export as `obs[1,143] -> actions[1,36]` with the observation normalizer folded into the
graph, validated by `onnx.checker`.

Two training runs were discarded rather than shipped: a Stand run that had degenerated to bang-bang
control while passing every quality gate, and a Walk run with an inverted gait reward. Both are
written up below, because the way each was caught is more reusable than the policies themselves.

## The pipeline

```
Scenes/ActiveRagdoll.tscn
  -> tools/tscn_to_urdf.py        16 links + 30 dummy links, 45 revolute joints
  -> assets/dummy.urdf
  -> scripts/convert_asset.py     -> assets/dummy.usd
  -> PhysX articulation           46 bodies, 45 joints, 80.63 kg  [verified against dummy_rig.json]
  -> training                     -> exported/*.onnx   obs[1,143] -> actions[1,36]
```

The rest pose was verified against the Godot scene before any training: pelvis 0.820 m, head
1.538 m, feet 0.040 m, upright 1.000, and ground contact 790 N against a body weight of 791 N.

## Throughput

RTX 4080 SUPER, this rig, 12 iterations per row:

| `--num_envs` | steps/s | vs the Godot `rl/` track | VRAM |
|---:|---:|---:|---:|
| 2,048 | 77,000 | 19× | ~3 GB |
| 4,096 | 110,000 | 27× | ~5 GB |
| 8,192 | 126,000 | 31× | ~8 GB |
| 16,384 | 136,000 | 34× | 14.6 GB |

4,096 is the default. Beyond it, samples per gradient update grow 4× to buy 24% more throughput,
which usually costs more in PPO convergence than it returns.

## Stand

`P4F-Dummy-Stand-Direct-v0`, 4096 envs, 2500 iterations, **39 minutes**, 246M timesteps.

Scored with `scripts/evaluate_stand.py` against the criterion ported from `UprightTermination`:
head ≥ 1.35 m, tilt ≤ 30°, CoM speed ≤ 0.6 m/s, held continuously for 1.5 s.

```
standing success   :  100.0 %
ever fell          :    3.9 %
mean head height   :  1.533 m   (rest 1.540)
mean torso tilt    :   2.78 deg (limit 30.0)
mean CoM speed     :  0.045 m/s (limit 0.6)
mean |action rate| : 0.0052
mean effort used   :    5.8 %   of per-joint torque limit
```

Head height at 1.533 against a 1.540 rest height is the number that rules out the failure worth
worrying about — a policy surviving in a crouch, which satisfies tilt and speed while never
actually standing. 5.8% of available torque means it is balancing rather than bracing.

For context, the Godot `rl/` track reports 36% success on the same criterion from a standing start
after ~24M steps.

### The run that had to be thrown away

A first Stand run went 4000 iterations and passed every gate: 100% standing success, falls halved,
3× smoother. It was degenerate. Its raw network output had drifted to ±18, with 97% of action
components and 34 of 36 joints permanently outside the `[-1, 1]` clip range — a bang-bang policy
that stood by rigidity, with no graded response left for a disturbance to act on.

The evaluation could not see it, and that is the point: the environment clamps, so the *clamped*
behaviour genuinely was excellent. It was caught by running the exported ONNX on a plausible
observation and looking at the raw output range.

Cause: PPO's Gaussian mean is unbounded and the environment clamps, so once a component is outside
the range, pushing it further changes nothing that executes and therefore costs nothing. Fix: the
`rew_action_clip` barrier on `relu(|a| - 1)^2`, exactly zero for a healthy policy.

| | max \|a\| | saturated | always-saturated joints | ever fell | tilt |
|---|---:|---:|---:|---:|---:|
| no barrier, 4000 it | 18.26 | 97.0% | 34 / 36 | 9.8% | 9.19° |
| barrier, 2500 it | **1.08** | **0.0%** | **0 / 36** | **3.9%** | **2.78°** |

Better on every real metric, in 40% of the wall-clock. The degenerate checkpoint was discarded
rather than used as a parent for Walk or Perturbation.

## Walk

`P4F-Dummy-Walk-Direct-v0`, bootstrapped from the Stand checkpoint via `train.py --init_from`,
4096 envs, 4000 iterations, **71 minutes**. Velocity command lives in the reserved observation
slots; the reward tracks it, with a feet-air-time term to push toward stepping rather than sliding.

Scored with `scripts/evaluate_walk.py` at a fixed 0.6 m/s forward command:

```
ever fell           :    3.1 %
mean forward speed  :  0.586 m/s  (target 0.6)
mean |speed error|  :  0.032 m/s
steps per second    :   1.39
mean air time/step  :  0.376 s
both feet grounded  :   43.1 % of the time
mean uprightness    :  0.992
mean head height    :  1.499 m   (rest 1.540)
raw |action| max    :   2.83
actions saturated   :    5.7 %
```

**It is genuinely walking, not sliding.** That distinction is the whole reason the evaluator exists:
velocity tracking alone is satisfiable by shuffling with both feet permanently planted, and the
reward would pay out just the same. 1.39 touchdowns per second with 0.376 s of air per step and a
43% double-support phase is an alternating gait.

Training-term trajectory, which shows the feet-air-time fix taking effect:

| | iteration 1 | 605 | final |
|---|---:|---:|---:|
| `track_lin_vel` | 2.82 | 19.59 | 34.20 |
| `feet_air_time` | −1.14 | −0.47 | **+2.31** |
| `action_clip` | −5.05 | −1.05 | −1.15 |
| mean reward | — | 48.82 | 66.71 |
| episode length (of 720) | 137 | 705 | 707 |

### One caveat

Action saturation is 5.7% with a peak of 2.83, against Stand's 0.0% and 1.08. The barrier is
holding — this is nowhere near the 97% degenerate case — but Walk sits closer to the boundary than
Stand does. If a future Walk run drifts further, raise `rew_action_clip` for that task rather than
assuming the Stand value transfers.

### A discarded first attempt

The first Walk run was killed at 56 iterations with two bugs, both found by reading the per-term
reward breakdown rather than the total:

* **`feet_air_time` had the wrong sign.** Written as `(min(air_time, target) - target)` it is ≤ 0
  everywhere, so a foot that never leaves the ground scores 0 while a foot that steps scores
  negative. It read −3.08 per episode against a +18.1 tracking term — actively paying the policy to
  slide, in a walking task.
* **`WalkEnv` overrode `_get_rewards` and dropped the `action_clip` barrier.** Walk would have
  saturated exactly the way the first Stand run did.

Neither is visible in the scalar reward, which was rising the whole time.

## Perturbation

`P4F-Dummy-Perturb-Direct-v0`, bootstrapped from Stand, 4096 envs. Reward and termination are
Stand's, inherited unchanged — this is the same objective with something pushing back, which is
exactly how `UprightProgressReward` is shared across stand / get-up / perturbation in Godot.

Ported from `BallGun`, and the port had to be corrected once. `BallGun` fires two shot types and
`SmallBallProbability = 1.0`, so **every** shot is the "small" ball — small in *radius* (0.06 m),
not in mass:

| shot | mass | speed | impulse | target |
|---|---:|---:|---:|---|
| heavy | 0.75 kg | 6 m/s | 4.5 N·s | chest always |
| small | 3.0 kg | 6 m/s | **18 N·s** | **uniform over 12 bones** |

The first version implemented the heavy ball — the one that never fires — hitting only the chest at
0–15 N·s. It was corrected to sample the target per episode over `BallGun.SmallBallTargetBones`
(head, chest, spine, pelvis, both upper arms, forearms, thighs, shins) with impulse uniform over
`[0, 25]` N·s. The range overshoots deliberately: the field declares `SmallBallMass = 3.0f` while
the comment above it says "1.0 kg", and `[0, 25]` contains both readings.

No ball is spawned — at 4096 envs that is 4096 extra colliders to deliver what is physically an
impulse. The wrench is applied to the chosen body for exactly one step, with the ±0.25 m aim jitter
carried through as the `r × F` torque. The disturbance is deliberately **absent from the
observation**, per `BallGun`: the agent reacts through proprioception, rather than learning to
anticipate a scripted event.

### Impulse sweep

```
  impulse  x BallGun   stayed up  still standing   max tilt   recovery
    (N.s)                         at episode end      (deg)        (s)
      0.0       0.0x       97.7%          100.0%        3.5       0.00
      4.5       1.0x       96.9%          100.0%        3.4       0.00
      9.0       2.0x       97.3%          100.0%        3.5       0.00
     18.0       4.0x       97.5%          100.0%        3.9       0.01   <- the real Godot shot
     25.0       5.6x       97.5%          100.0%        4.7       0.01
     35.0       7.8x       86.5%          100.0%       11.2       0.02
```

Flat from 0 to 25 N·s, degrading only at 35 — which is 40% beyond anything it trained on.

### Per-target, probed at 35 N·s

At 18 N·s every target passes and the table says nothing; pushed past the training range, the
structure appears:

| target | fall rate | | target | fall rate |
|---|---:|---|---|---:|
| UpperArm_L | 12.6% | | Chest | 4.7% |
| Shin_R | 10.3% | | Pelvis | 3.6% |
| UpperArm_R | 9.9% | | Forearm_L | 3.4% |
| Shin_L | 5.6% | | Forearm_R | 3.1% |
| Thigh_R | 5.0% | | Thigh_L | 2.9% |
| | | | Head | 2.4% |
| | | | Spine | 2.3% |

**Upper arms and shins are roughly 4–5x harder than spine or head**, which is what the mechanics
predict: a shove to the spine passes near the CoM, while one to an upper arm has a long moment arm
about the pelvis, and a shin can be caught mid-swing with the policy already committed to a foot
placement. This is why the evaluator reports per-target rather than an aggregate — ten of the twelve
targets are easy, so a single number would have read as fine.

Do not over-read the left/right differences. At 70–100 shots per body the binomial standard error
is about 3 points, so `Shin_L 5.6%` against `Shin_R 10.3%` is inside noise. The limb-vs-torso split
is real; the left-vs-right split is not, on this sample.

### On `BallGun`'s tipping arithmetic

That file computes 6.74 J to tip a passive body over its toe edge, and concludes that at 1.75x that
energy "no balance policy can absorb the hit without stepping, so training on it would have produced
a flat zero success rate with no gradient".

Running the same numbers for the 35 N·s row — 35 x 1.25 m = 43.8 kg.m^2/s about the toe line,
I = 69.5 kg.m^2, omega = 0.63 rad/s, KE = 13.8 J — that is **2.0x the passive tipping energy**, and
it stays up 86.5% of the time.

This does not contradict the measurement in `BallGun.cs`; the policy measured there genuinely did
collapse on every hit. It contradicts the generalisation drawn from it. The ceiling was a property
of that policy, not of the rig, and the impulse range can be raised much further before it stops
being trainable.

## Run — attempted, not achieved

`P4F-Dummy-Run-Direct-v0`, bootstrapped from Walk, 4096 envs, 3000 iterations, **55 minutes**.
Commands 0.8-2.5 m/s, with a `double_support` penalty on both feet being loaded at once - the term
meant to separate a run from a fast walk - and Walk's vertical-velocity penalty relaxed from -1.0
to -0.25 because at -1.0 it directly opposes a flight phase.

**It did not learn to run.** It learned a genuinely fast walk:

| | Walk @ 0.6 m/s | Run @ 2.0 m/s |
|---|---:|---:|
| mean forward speed | 0.587 m/s | **1.863 m/s** |
| steps per second | 1.38 | **1.90** |
| both feet grounded | 43.1% | **25.4%** |
| **flight (no foot down)** | 0.5% | **0.5%** |
| ever fell | 3.5% | 3.5% |
| raw \|a\| max | 2.47 | 5.97 |
| actions saturated | 5.7% | 9.3% |

Flight phase is the number that decides it, and it is identical to Walk's - 0.5%, which at this
sampling rate is contact-force noise rather than airtime. Every other metric moved a long way: 3.2x
the speed, 38% more cadence, and double support cut nearly in half. The penalty pushed the gait
toward the walk/run boundary and never across it. For reference the human walk-run transition is
around 2.0 m/s, so this is a dummy power-walking right at its transition speed.

### Why, most likely

Three candidates, in the order worth testing:

1. **No reward for flight, only a penalty for double support.** Removing one foot from the ground
   is rewarded; removing both is not rewarded any further. A term paying directly for the
   no-contact fraction would make flight worth something on its own.
2. **`action_scale = 0.4` may cap the achievable leg swing.** Running needs larger, faster joint
   excursions than balancing does, and 0.4 was chosen to make *balance* learnable. That the policy
   is now pushing to \|a\| = 5.97 - by far the worst of the four tasks - reads as it asking for
   authority the mapping will not give it.
3. **3000 iterations from a walking parent may simply be too few** to cross a gait transition,
   which is a discrete change of contact pattern rather than a smooth optimisation.

### Caveat on this checkpoint

Run is the weakest of the four on output range: 9.3% of components saturated with a peak of 5.97,
against Stand's 0.0%/1.08. `scripts/evaluate_walk.py` flags it. That is nowhere near the 97%/18.26
degenerate case and the policy is genuinely tracking a hard command, but the trend across the four
tasks is consistent - **the harder the task, the harder the policy pushes at the clip boundary** -
and `rew_action_clip = -0.02`, inherited from Stand, is too weak here. Raise it for Run before
trusting this checkpoint the way the other three can be trusted.

## What Godot needs before any of this runs in-engine

All three are recorded in [`obs_action_contract.md`](obs_action_contract.md) with the measurements
behind them.

1. **`action_repeat` 8 → 2** in the four RL scenes. Godot currently decides at 15 Hz (a 66 ms
   control period); these policies decide at 60 Hz. A 60 Hz policy replayed at 15 Hz sees every
   observation four times too late and holds every action four times too long.
2. **Clamp the ONNX output to `[-1, 1]`** before the joint mapping. Not defensive politeness — the
   network output is an unbounded Gaussian mean, and the discarded run above shows what unclamped
   values look like.
3. **`ACTION_SCALE = 0.4`**, and move the action zero point from the midpoint of each joint's
   limits to the rest pose. `JointLimitedActionSpace` currently does neither.

Points 1 and 3 are worth reading independently of this port. A 66 ms control period and an action
space where a neutral output commands a deep crouch are both plausible contributors to the get-up
never working in the Godot track, and neither is visible as a bug.

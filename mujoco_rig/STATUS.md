# MuJoCo track — status, 2026-09-11

Written as a handover. `README.md` explains how the pieces work; this says where the project
actually is, what is known, and what the next person (or session) should do first.

## The one-paragraph version

The dummy's physics lives in MuJoCo; Godot renders it and drives it through P/Invoke. The body is
1.750 m, 69.6 kg, 18 bodies, 33 actuators of which a policy drives 30, built by `build_mjcf.py`
from Godot's own rig dump. It is a **ragdoll with muscles**: `motor` actuators over passive
ligaments, so zero policy output is zero torque and an unpowered body crumples. **Balance is
solved** — 100% upright indefinitely in a quiet room, actively, on under 10% of human strength.
**Perturbation survives 92.4% of a single 30 N·s hit** (68.8% at dusk on 2026-09-10), after four
measured jumps, each from a change aimed at a measured failure: more training shots at the
high hits it fell to, a step reward that keeps growing to 1 m/s of swing, and a term that pays where
the foot lands against the capture point - then a rest stance it is pulled back to once balanced,
which it ends in. **Walk follows most of a joystick at an amble** — straight,
arcs both ways, and it stops when told to — but it cannot turn on the spot, and measured from rest
its gait is a shuffle. **Neither brain has been seen in the scenes since it changed.**

## What is shipping

| file | from | what it does |
|---|---|---|
| `balance_policy.onnx` | `p0911i_s1/model_503` | stands, and takes hits: survives **92.4%** of a single 30 N·s hit (512 envs); the brain it replaced survives 87.5% (paired z +2.93). Ends back in its stance: feet at the rest width (0.31 m), 0% crossed, leg error 0.24 rad. Under a hit every 3 s, 53% last 20 s |
| `locomotion_policy.onnx` | `w0911f_s1/model_253` | **46.3% of the joystick** (32 envs, five manoeuvres): +6.0 m on a 0.25 m/s straight line, arcs −807° / +367°, stands still to 0.05 m; falls turning on the spot, and in some runs of the left arc. Its contract carries the heading hold; `MujocoWalk.tscn` commands 0.25 m/s |

Promotion is automatic and measured: `scripts/promote.py` scores challenger against the incumbent
read from the shipped contract, and exports only on a win. `overnight.py` calls it at the end of a
run. Do not export by hand — doing so once shipped a policy that walked **backwards**. (Re-exporting
the checkpoint a contract already names, to regenerate the contract, is not choosing a checkpoint.)

**Perturb is promoted on one real hit at 6 m/s (30 N·s), 512 envs, both checkpoints scored at once
on the same shots, and shipped only if the paired McNemar z reaches 2.** The gates before it failed
twice: upright over 40 s at 2.2 m/s read ~100% for everything, and a "one-hit" test that actually
fired two shots shipped on noise. **Walk is still promoted on 32 envs with no paired test**, on
the joystick score (all five of `eval_walk`'s manoeuvres, `scoring.WalkScorer`) — and on 2026-09-11
a +0.9-point win shipped a brain that falls more on one arc than the one it replaced.

## Run this before any long session

```
python mujoco_rig/scripts/preflight.py --task perturb --speed_end 6.0 --seed <checkpoint>
python mujoco_rig/scripts/preflight.py --task walk --seed <checkpoint>
```

Three minutes, no GPU. It checks the start pose does not self-intersect; that the reward ranks
standing / stepping / airborne / fallen in the intended order; for perturb, that **a capture step
earns more than a stomp and a step away from the falling COM earns nothing**; for walk, that
**walking out-earns a statue at every commanded speed**; that **one exploding world stays
harmless**; that the projectile connects and the curriculum's end is physically survivable; and that
the seed's exploration suits the task. **Six multi-hour failures on 2026-09-09/10 were task-level
errors these checks now catch before the first session.** A task that fails preflight cannot be
fixed by training.

## Numbers worth not re-deriving

**The plant**
- Torque actuators over ligaments (5% of peak torque per rad). Zero output ≈ `dummy_limp.xml`.
- `JOINT_ARMATURE = 0.02` is load-bearing. Without it, 10 of 10 random-torque rollouts diverge with
  `Nan, Inf or huge value in QACC`, and **MuJoCo silently resets the state** when that happens.
- `qpos0` is NOT a valid pose — the arms hang inside the legs there. Every model carries a `rest`
  keyframe; the envs and `MjBridge.ResetData` use it.
- The ball is **8 kg** since 2026-09-11 (48 N·s at 6 m/s); every perturb score before that is at
  **5 kg** (30 N·s), including the shipped brain's 92.4%. At 15 kg a single hit was survived 0 times in 12. A stepping recovery
  arrests about 1.0 m/s of COM velocity; ankles alone 0.3–0.5.
- **The body is left/right symmetric** — bodies, joints, actuators and geoms all checked. A gait
  that veers is a learned handedness, not the plant.
- Fastest body DOF over 1.2M world-steps, falls included: 94 rad/s walking, 97 under fire. A world
  past `QVEL_CEILING` (300) is diverged, not falling.

**Measurement traps, each of which produced a wrong conclusion**
- **MuJoCo `cvel` is referenced to the subtree centre of mass**, not to the body. Raw
  `cvel[foot, 3:6]` read −0.14 m/s for a foot moving at +0.70 — the wrong sign. `com_velocity`
  now shifts it: summed raw, the whole-body COM velocity was off by a median 0.32 m/s whenever the
  body moved (fixed 2026-09-11; preflight's "137% delivered" was the same error - it is 93%). The
  pelvis velocity the walk reward and the observation read is still raw.
- **Worlds that reset together time out together.** The curriculum promoted on that clock, and for
  the first 75 iterations the logged episode length counts only worlds that FELL.
  `randomize_episode_phase` now staggers every task.
- **The "one hit" test fired two** for most of 2026-09-10. `eval.py` now prints `hits per env`.
- **"Survived when it stepped" is confounded by how hard the hit was.** Never decide on it.
- **Survival near 30–40% is noisy.** Compare checkpoints only paired, on the same shots.
- **A calibration is only valid for the range it was made on.** The ball's mass silently re-based
  perturb's reference; the walk's command range silently re-based its tracking kernel.
- **The training log's walk vx is pelvis-frame.** It read +0.16–0.22 for a policy walking in
  circles. Judge walk on eval_walk's `along` and `heading swept`.
- **One exploding world poisons the whole batch** unless its reward is bounded AND sanitised. The
  `nefc overflow` printed before each walk crash was that world, not the gait (72 rows of 256).
- **A behaviour the env computes must exist in Godot too.** The heading hold lives in the walk env's
  `step()`; `MjPolicyDriver` wrote the yaw command raw. It now applies the same correction, driven
  by the contract's `command` block, with a heading formula identical to the env's over 10,000
  random rotations.
- **Measure a foot's lift from where it rests.** The walk env read an absolute 6 cm while the foot
  origin rests at 4.4 cm, so a 1.6 cm shuffle counted as a step and every gait term paid it.
- **A score clipped at 1.0 cannot see overshoot.** The joystick score read 43% too fast as a perfect
  straight line; each part is now 1 - |achieved / commanded - 1|.

**The trainer**
- Batch: **4,096 envs × 16 steps**; the split matters more than the batch.
- `ms per policy step = 35.8 + 0.04157 × envs`. Envs are nearly free; steps are sequential.
- A seeded start needs the promotion cooldown: without it the first update ran KL to +5.97.
- **Exploration does not travel with the weights.** `--reset_std` for any cross-task seed, even one
  whose std passes preflight — the perturb brain seeded into walk at 0.173 settled into a statue;
  reset to 0.4 with entropy 0.005, it stepped within two sessions.
- Scoring 512 envs takes ~70 s per checkpoint; the gates run their scorings concurrently.

**Reward brackets, measured**
- Perturb stance term must be **gated on being balanced**; `recover_step` must pay for the swing
  foot moving **toward** the escaping COM, not for single support (a stomp satisfied that).
- Walk tracking must be **relative to the command** (a fixed width paid a statue 44–86% at 0.15–0.35
  m/s); a **zero yaw command must hold the heading**, not just penalise yaw rate (a circle cost a
  flat 0.95/step); the velocity-squared penalties are clamped at `PENALTY_CAP`.
- Perturb: **aim the training shots at the failures** (head, chest and spine were half the falls on a
  quarter of the shots); pay the step up to **1 m/s** of swing (`CAPTURE_V`); pay **where the foot
  lands** against the capture point (`CAPTURE_PLACE_SIGMA` 0.15); pull a balanced body back to its
  **rest stance** measured in the pelvis frame and signed, legs included (`STANCE_SIGMA`,
  `LEG_POSE_SCALE`) - world axes with abs() read crossed feet as a normal width.
- Walk: **pay both feet down on a stand command** (`STAND_PLANTED` 1.5) — without it every brain
  walked away from a stand. The shuffle's payment is what keeps it walking: every `WALK_FOOT_CLEAR`
  above 1.6 cm tried (2, 2.5, 3, 6 cm) stopped the walk within one session.

## Where the tasks stand

**Perturb** — one real 30 N·s hit, 512 envs, the same shots for every row. The night is written up
in `isaac_lab_3/OVERNIGHT-2026-09-11.md`.

| checkpoint | survived | paired z vs row above | what changed |
|---|---|---|---|
| `perturb_z_s4/model_253` | 68.8% | — | shipped at dusk on 2026-09-10 |
| `p0911b_s1/model_249` | 74.2% | +2.60 | training shots weighted toward head, chest and spine |
| `p0911c_s1/model_255` | 80.9% | +3.08 | `CAPTURE_V` 0.5 -> 1.0 |
| `p0911e_s1/model_256` | 87.5% | +3.18 | a correct COM velocity + a capture-point placement term |
| `p0911i_s1/model_503` **(shipped)** | **92.4%** | +2.93 | a signed, pelvis-frame rest stance with the legs, paid while balanced (30 min) |

A continuation after each change did not improve (80.3%, 85.7%): the next gain needs a change, not
minutes. What is left: head 63% and chest 65%; pushed-right 79% against ~90% for every other
direction (a learned handedness - the body is symmetric); the first step still lands ~0.17 m short
of the capture point, so the last jump came from something the step probe does not see.

**The rest stance** (the morning after): watched in the scenes, the dummy stood oddly after hits, and
the stance term could not see it - world axes with `abs()` read crossed feet as a normal width, and
`pose` averaged all 30 joints. It is now the feet in the pelvis's own heading frame, signed, times
the legs against the rest pose, still only while balanced. It gained survival as well as posture:

| | before (`model_256`) | after (`model_503`) |
|---|---|---|
| one hit: survived / end stance | 87.5% / 0.29 | **92.4% / 0.37** |
| one hit: width (rest 0.31 m) / split / leg error | 0.34 m / 0.06 m / 0.27 rad | 0.31 m / 0.04 m / 0.24 rad |
| a hit every 3 s, 20 s: survived / end stance | 41.4% / 0.24 | **53.1% / 0.29** |

Two other changes that morning each cost ~4 points and did not ship: training at 6.6 m/s (83.0%)
and weights refreshed from the brain's own profile (83.4%).

**Walk** — stage 4 (the amble with turning and stopping), scored on the whole joystick:

| | `walk_g_s1/model_254` (dusk) | `w0911f_s1/model_253` **(shipped)** |
|---|---|---|
| joystick, 32 envs | 17.6% | **46.3%** |
| straight line, 40 s at 0.25 m/s | +8.1 m | +6.0 m |
| turn left / right | −7° / −158° | +367° / −807° |
| turn on the spot | +26° | falls |
| stand still | drifts 3.7 m | drifts 0.05 m |

Measured from rest, it takes **no real steps**: its higher foot rises a median 2.4 cm and never
6 cm, and the walk env had counted 1.6 cm as a step all along. Every clearance that stopped paying that shuffle - 6, 3 and 2.5 cm, and 2 cm once training continued - turned the walk into a statue within one session instead of making it lift higher, so `WALK_FOOT_CLEAR` is left at 1.6 cm above rest: what the env always used, now written relative to rest.

## What I would do next

1. **Look at the three scenes.** Both brains changed on 2026-09-11. Walk should print `heading hold
   gain 0.50` and amble straight; Perturb should take most hits with a step.
2. **Give walk the paired gate perturb has.** `w0911f` shipped on +0.9 points.
3. **A stronger reason to walk, before real steps.** Every `WALK_FOOT_CLEAR` above 1.6 cm tried
   turned the walk into a statue within a session. Turning on the spot needs real steps; without
   them it falls.
4. **The pelvis velocity** the walk reward and the observation read is still raw `cvel`. Change it in
   the env and in Godot's observation together, or not at all.
5. **Perturb:** probe every step rather than the first; the pushed-right handedness.

## Tooling

```
scripts/preflight.py     is the task winnable, does the reward pay for the right thing, is one bad world harmless
scripts/promote.py       ship only on a paired, significant win against the incumbent (perturb: one real 30 N.s hit)
scripts/overnight.py     chained scored sessions, difficulty from measurement, never chains a crash, auto-promote
scripts/scoring.py       how each task is scored; the one reader of the scorers' --json output
scripts/batch_table.py   where the useful batch is, on THIS plant
scripts/watch.ps1        a window on the scorer's engine, with live telemetry
rl/watch.py              the viewer itself
rl/env_config.py         the constants that define the task, shared by both backends
rl/body_env*.py          the base environment each task extends, CPU and GPU
```

Restructured on 2026-09-11 with no change in behaviour, every step gated: a ball-free base
environment (bit-identical perturb traces, walk in lockstep), scorers that write JSON read by one
`scoring.py` (same verdicts), per-task tables instead of task branches, and the trainer, exporter,
model generator and C# driver split into named parts (identical old-vs-new training runs,
identical contracts, byte-identical models, 0 build errors).

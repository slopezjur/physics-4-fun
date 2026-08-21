# How to train the RL dummy

Two steps every time: **export**, then **train**. Training runs the *exported build*, never the
open editor — so if you skip the export after changing C# or a scene, you will silently train the
old code.

Replace `<godot>` and `<project>` with your own paths.

---

## Windows (PowerShell)

The leading `&` is required: PowerShell parses a quoted string at the start of a line as a string
expression, not a command, and without it the `--` flags fail with "Token 'headless' inesperado".

```powershell
# 1. Export
& "<godot>/Godot_v4.7.1-stable_mono_win64/Godot_v4.7.1-stable_mono_win64_console.exe" `
  --headless --path "<project>/physics-4-fun" `
  --export-release "Windows Stand" `
  "<project>/physics-4-fun/build/RagdollStandTraining.exe"

# 2. Train (run from the project root)
rl/.venv/Scripts/python.exe rl/train.py `
  --env_path=build/RagdollStandTraining.exe `
  --n_parallel=40 --speedup=16 --timesteps=10000000 `
  --experiment_name=getup_v1 `
  --save_model_path=rl/getup_v1.zip `
  --onnx_export_path=rl/getup_v1.onnx
```

`--path` matters: without it Godot looks for `project.godot` in the current directory, so the
command only works from inside the project folder.

## Linux

```bash
# 1. Export
"<godot>/Godot_v4.7.1-stable_mono_linux_x86_64/Godot_v4.7.1-stable_mono_linux.x86_64" \
  --headless --path "<project>/physics-4-fun" \
  --export-release "Linux" \
  "<project>/physics-4-fun/build/RagdollStandTraining.x86_64"

# 2. Train
rl/.venv/bin/python rl/train.py \
  --env_path=build/RagdollStandTraining.x86_64 \
  --n_parallel=40 --speedup=16 --timesteps=10000000 \
  --experiment_name=getup_v1 \
  --save_model_path=rl/getup_v1.zip \
  --onnx_export_path=rl/getup_v1.onnx
```

## macOS

```bash
# 1. Export
"<godot>/Godot_mono.app/Contents/MacOS/Godot" \
  --headless --path "<project>/physics-4-fun" \
  --export-release "macOS" \
  "<project>/physics-4-fun/build/RagdollStandTraining.zip"

# 2. Train
rl/.venv/bin/python rl/train.py \
  --env_path=build/RagdollStandTraining.zip \
  --n_parallel=40 --speedup=16 --timesteps=10000000 \
  --experiment_name=getup_v1 \
  --save_model_path=rl/getup_v1.zip \
  --onnx_export_path=rl/getup_v1.onnx
```

**Note:** `export_presets.cfg` currently defines only Windows presets (`Windows Stand`, `Windows Perturbation`, `Windows Walk`). Linux and
macOS need their preset added once via the editor (Project -> Export -> Add), and their export
templates installed, before the commands above will work. The venv path also differs:
`rl/.venv/Scripts/python.exe` on Windows, `rl/.venv/bin/python` elsewhere.

---

## Useful flags

| flag | what it does |
|---|---|
| `--n_parallel=32 --speedup=8` | measured knee on a 7800X3D: 2,162 steps/sec in a sweep, 2,065 sustained over 30 min. 40 procs reach 2,258 (+4.5%) for 25% more CPU. |
| `--speedup` | a *request*, not an achieved rate. Only matters when `speedup × 15 × n_parallel` falls below the CPU ceiling, at which point it throttles. Above that it is irrelevant, so pick the smallest safe value - the `--viz` window renders at it. |
| `--viz` | renders ONE instance in a window (others headless) so you can watch training live. Measured free on this machine: 2,235 steps/s with, 2,144 without. |
| `--restore=rl/getup_v1.zip` | continue a checkpoint instead of starting over (keeps optimizer state and step count). |
| `--timesteps` | 1M ~ 8 min at the knee. Pure-RL get-up realistically needs 10-100M. |
| `--ent_coef` | entropy bonus, default 0.001. Raise to 0.003 if `train/std` keeps falling before the task's reward has ever been reached. |
| `--batch_size` / `--target_kl` | 2048 / 0.02. Re-applied explicitly after `--restore`, since `PPO.load` would otherwise restore whatever the checkpoint was saved with. |
| Ctrl+C | stops early and still saves, via an `except KeyboardInterrupt` that writes an `interrupted_*` checkpoint. SB3's `learn()` has no handler of its own, so without it `on_training_end` never runs and everything since the last periodic save is lost. |

## Watching and testing

Run this in its **own terminal**, alongside training - it blocks until Ctrl+C. Paths must be
absolute (or run from the project root): a relative `rl/runs` resolves against the caller's
working directory, so launching from `rl/scripts` would point at `rl/scripts/rl/runs`.

```bash
rl/.venv/Scripts/tensorboard.exe --logdir <project>/rl/runs --port 6008   # -> http://localhost:6008
```

The `tensorboard.ps1` wrapper takes the path from `$ExperimentDir` in `config.ps1`, which is the
same variable the training scripts pass as `--experiment_dir` - so the two can never disagree.

- `start_prone/success` - **the get-up.** This is the headline number now.
- `rollout/ep_rew_mean` - **not readable at all** under the reverse curriculum. It is an average over
  a spread of start poses of very different difficulty, and it *falls* as the curriculum advances
  because harder levels succeed less often. Falling reward here is the curriculum working, not a
  regression. Judge on `pose_<level>/success` at the current floor instead.
- `rollout/ep_len_mean` - the episode window is interpolated with the start pose,
  `Lerp(8 s, 4 s, poseT)`, so this moves with the curriculum too. While the floor is near 1.0 nearly
  every episode gets ~4 s (60 steps); it lengthens toward 120 as the floor descends. A drop means
  episodes are ending early, i.e. reaching `Standing` (good) or `Inverted` (bad).

To watch a trained policy in-engine with Python closed: set the `Sync` node's `control_mode = 2`
(Onnx Inference) and point `onnx_model_path` at your `.onnx`, then press Play.

### Per-episode console lines

Interleaved with SB3's tables you get one line per episode from the bridge:

```
[RagdollRLBridge] Episode 387 ended (TimeLimit, reward=-7,63, duration=3,1s) -> starting episode 388
```

Exactly **one** of the 32 processes prints these — the one holding godot_rl's base port, 11008.
`rl/train.py` gives process `p` the port `DEFAULT_PORT + p`, so the base-port process is process 0.
32 processes printing near-identical lines every few seconds is unreadable, and 32 synchronous
stdout writers is not free either.

This used to be gated on *having a display* instead of on instance identity, which made the lines a
`--viz`-only feature: `resume.ps1` and `train.ps1` launch everything headless, so the log vanished
from the normal training path entirely. Now both paths show it, and `--viz` is unchanged — with
`visible_count = 1`, process 0 is both the visible instance and the base-port one, so it is the same
single process logging as before.

## Where things are stored

| path | contents |
|---|---|
| `rl/*.zip` | SB3 checkpoint: policy **+ optimizer state**. This is what `--restore` needs. |
| `rl/*.onnx` | policy only, for in-engine inference. Cannot resume training from this. |
| `rl/runs/<name>_N/` | TensorBoard logs, one directory per run. |

All are gitignored. The `.onnx` must match the current observation/action widths (today **106 -> 36**);
an `.onnx` from an older layout fails at load with a shape mismatch.

---

## Launching

Use the scripts in `rl/scripts/` (below). There is deliberately no in-editor launcher scene: one
existed briefly and was removed. Training runs N copies of whatever `run/main_scene` points at, so
a launcher scene that spawns training would, if it ever became the exported main scene, have each
of 40 processes spawn its own export and its own 40 processes. That nearly happened once. Driving
this from a script instead of a scene makes the failure mode structurally impossible.

### Watching speed

The visible instance runs at the same `--speedup` as the rest (16x by default), so motion looks
very fast. Lower `Speedup` to ~2-4 for watchable motion.

This is a genuine trade-off, not just a preference: `step()` waits for **every** env each tick, so
the rendering instance gates the entire batch's throughput. Rendering one window costs some speed,
and lowering the speedup to watch it costs a lot more. For long runs, train headless
(`ShowOneInstance = false`) and inspect the result afterwards through ONNX inference.

---

## Run provenance (`manifest.json`)

Every run writes `rl/runs/<name>_N/manifest.json`, refreshed each rollout so a crashed or Ctrl+C'd
run still leaves current numbers. It exists because neither artefact SB3 produces answers "what was
this trained on": the `.zip` stores step count and PPO hyperparameters, TensorBoard stores curves,
and **neither records the reward, termination thresholds, action range, bone set, or code
revision** — exactly the things that differ between experiments.

It records:

- **timing** — start/end UTC, elapsed seconds, steps/sec, timesteps done vs requested
- **cli_args** — every flag the run was started with
- **trainer** — algo, gamma, n_steps, batch size, device, observation/action spaces
- **environment** — pulled live from the Godot bridge, not copied: controlled bones, action size,
  observation size, max action angle, episode length, physics Hz, and each strategy's own
  description with its constants (e.g.
  `GetUpProgressReward(progress=10, upright=1, effort=0.25, standingBonus=20)`)
- **code** — git commit, branch, and a `git_dirty` flag
- **latest_metrics** — `ep_rew_mean`, `ep_len_mean`, episodes seen

The environment block comes from `RagdollRLBridge.GetRunConfig()`, which builds it from each
strategy's `Describe()`. Nothing is duplicated on the Python side, so the manifest cannot drift
from the code that actually ran.

**`git_dirty: true` means the commit alone does not identify what ran.** Commit before a run you
intend to compare against later.

### Do not reuse `--save_model_path` between runs

It silently overwrites. This already cost one checkpoint in development: a 330k-step model was
replaced by an 87k-step smoke test that reused the same path, with nothing to warn and nothing in
the file identifying which run it came from. Name checkpoints per experiment
(`--save_model_path=rl/getup_v1.zip`).

### Which scene the export builds

A Godot export boots whatever `run/main_scene` points at, and the two roles want different scenes:
pressing Play in the editor should open the inspection **Arena**, while the exported build must
start the **Training** scene. This is resolved with a feature-tagged project setting, not by
rewriting anything:

```ini
# project.godot
run/main_scene="res://Scenes/RL/Upright/RagdollPerturbationArena.tscn"
run/main_scene.stand="res://Scenes/RL/Upright/RagdollStandTraining.tscn"
run/main_scene.perturbation="res://Scenes/RL/Upright/RagdollPerturbationTraining.tscn"
run/main_scene.walk="res://Scenes/RL/Locomotion/RagdollWalkTraining.tscn"

# export_presets.cfg - one preset per task, each with its own feature tag
custom_features="stand"           # "Windows Stand"
custom_features="perturbation"    # "Windows Perturbation"
custom_features="walk"            # "Windows Walk"
```

Godot resolves `<setting>.<feature>` against the build's active feature tags, so the editor gets
the Arena and the export gets Training, automatically.

**Why not just swap the setting around each export?** That is what `export.ps1` used to do, and it
corrupted `project.godot`. Reading and rewriting the file round-tripped it through PowerShell text
encoding twice per export, and `Set-Content -Encoding utf8` on Windows PowerShell 5.1 writes UTF-8
*with* a BOM. The following run read those BOM bytes back under the ANSI codepage, turning them
into the literal text `Ã¯Â»Â¿` prefixed to `config_version`; Godot then preserved that as a quoted
key on its next save. The Project Manager began reporting *"The project uses an unknown version of
Godot."* If you ever need to machine-edit `project.godot`, write bytes - never `Set-Content`.

---

## Tasks

Three tasks share one rig, one observation vector and one action space. Set `$Task` in
`rl/scripts/config.ps1`; it picks the export preset, which picks the scene via the feature tag.

| `$Task` | Scene | Reward / termination | Window | Starts |
|---|---|---|---|---|
| `stand` | `RagdollStandTraining` | `GetUpProgressReward` / `GetUpTermination` | 4 s | always upright (t = 1.0) |
| `getup` | `RagdollGetUpTraining` | same pair | 4 s standing, 8 s prone | reverse curriculum, floor starts 0.99 |
| `perturbation` | `RagdollPerturbationTraining` | same pair, `EndEpisodeOnStandingSuccess = false` | 5 s | always standing |
| `walk` | `RagdollWalkTraining` | `WalkForwardReward` / `WalkTermination` | 6 s | standing, moving at 0.48–0.8 m/s |

`stand` and `getup` are the same components with different start distributions, and that is the
point: get-up is not a separate problem, it is *standing from progressively worse starting
positions*. They connect two ways - a get-up run **resumes from a stand checkpoint**, and its
curriculum begins at t = 0.99 (almost upright, where a stand policy already succeeds) and walks
the floor back toward flat. Train stand first; it is the reference get-up is bootstrapped from.

**The observation and action widths never change between them.** That is load-bearing, not
incidental: the widths are published to Python at handshake and baked into every checkpoint, so
changing them turns a resume into a from-scratch run. Because they are fixed at 106 and 36, a
policy trained on one task can be restored onto another — which is the only reason walking is
trainable in a couple of hours at all. It starts from a standing policy rather than from noise.

Only the reward and termination vary, selected by the `TaskKind` export on the bridge
(`RlTaskKind.GetUp` / `.Walk`). Perturbation is not a separate `TaskKind`: it is the get-up pair
with success-absorption switched off, so it needs no components of its own.

### Perturbation (balance under impact)

A `BallGun` fires at the dummy on a fixed interval — 10 s in training (longer than the 5 s window,
which is how "exactly one ball per episode" is expressed without a shot counter) and 3 s in the
arena, where the point is to watch repeated recoveries. Two profiles, chosen per shot:

| | mass | radius | speed | aimed at |
|---|---|---|---|---|
| heavy | 1.5 kg | 0.22 m | 6 m/s | chest, falling back to pelvis |
| small | 0.2 kg | 0.06 m | 6 m/s | a uniformly random bone of twelve |

`EndEpisodeOnStandingSuccess` **must** be false here. With it true the episode ends at roughly
0.6 s of settling plus `StandingHoldSeconds`, i.e. *before the first ball lands at 1.0 s*, so the
perturbation would never be experienced. Worse, a ball arriving every N seconds resets the hold
counter, so "hold 1.5 s continuously" and "get hit repeatedly" fight by construction.

Consequence: `StandingBonus` is never paid, and the return is `upright + shaping - effort`. It is
**negative** — around -3.6 — and that is correct, not a bug. Shaping telescopes to
`10 × (head_end − head_start)`, and from a standing start head height can only go down.

#### Ball strength is not the lever

Measured 2026-08-21 on `perturbation_v1_0`, resumed from the 46.7M-step get-up policy:
`reward/shaping` -7.07 with `ball/hits` 0.922, and `0.922 × -7.7 + 0.078 × 0 = -7.1` closes exactly.
Every hit put the body on the floor, every miss left it standing, recovery rate zero.

The tempting read is "the ball is too strong". The numbers say otherwise. Half those shots were the
*small* ball, which delivers about **0.023 J** of transferred energy against the **6.74 J** needed
to tip this rig — 0.3% of it — and flattened the body just as reliably. A disturbance three orders
of magnitude under the passive tipping threshold is not what is knocking it over. The policy is: it
had never experienced a disturbance in 46.7M steps, so it holds a knife-edge balance that any
contact ends. This is the same brittleness that walled the get-up curriculum at 1.8–2.7°.

> **Units trap.** Comparing the ball's own kinetic energy (27 J at these settings) against the
> body's tipping energy is meaningless — almost none of a 1.5 kg ball's energy transfers to an
> 80.6 kg body. The table in `BallGun.LaunchSpeed` is *transferred* energy. Mixing the two makes an
> already-gentle shot look like a 4× overshoot, and did.

Ball strength remains the natural axis for a difficulty curriculum later — the frontier machinery in
`RagdollRLBridge` would drive impulse instead of start pose unchanged — but raising it before the
policy can survive a 0.023 J poke would be sequencing it backwards.

#### The 2-hour run: a clean negative result

Run to 61.3M steps (14.1M added). Every task metric degraded monotonically across all six segments
of the run, while the policy's own action noise *rose*:

| segment | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `standing/all` | 0.480 | 0.479 | 0.472 | 0.465 | 0.457 | 0.459 |
| `reward/shaping` | −6.873 | −6.825 | −6.867 | −6.871 | −6.990 | −7.010 |
| `train/std` | 0.567 | 0.574 | 0.581 | 0.588 | 0.593 | **0.599** |

`train/std` is the diagnostic. A rising Gaussian policy std means PPO is finding no direction that
improves reward, so the entropy bonus is the only force acting and the policy diffuses — slowly
degrading the standing behaviour it inherited. This is not "needs more time"; it is 14.1M steps of
monotone evidence that the task has no usable gradient **as configured**.

The cause is the missing primitive. Ankle and hip strategy cannot recover a capture point that has
left the support polygon — only a step can — and every ball contact pushes it out. The same wall
stopped the get-up curriculum at `pose_0.91`. Balance-under-impact is therefore blocked *behind*
walking, not parallel to it, and the next perturbation run should resume from a policy that can
step rather than from this one.

### Walking

`WalkForwardReward` is structured differently from the get-up reward on purpose. Getting up is a
one-off transition to a goal state, so shaping on a potential is right. Walking is a *sustained
periodic* behaviour with no goal state, so the dominant term is a **rate** — forward speed. A
potential on distance travelled would telescope to total displacement and pay identically for
walking 6 m and for falling forward 6 m.

The reward is a **product**, not a weighted sum:

```
progress = 6.0 × clamp(v_forward / 1.0, 0, 1) × clamp(cos(tilt), 0, 1) × dt
reward   = progress − 0.5·lateral_drift·dt − 0.02·effort·dt
```

| term | weight | notes |
|---|---|---|
| progress | 6.0 | product of speed factor and upright factor, both `[0,1]` |
| heading | −0.5 | per metre of lateral drift per second |
| effort | −0.02 | mean actuator capacity fraction |
| fall | 0.0 | see below |

#### How that form was arrived at (three runs)

The first version was additive — `velocity + alive + upright − heading − effort` — weighted so
walking at target speed scored 48 over a 6 s episode against 12 for standing still. Resumed from a
policy that could already stand, it **converged to standing perfectly still inside five minutes**:

| | `walk/forward_speed` | `walk/fell` |
|---|---|---|
| additive reward | 0.0186 → **0.0069** | 0.164 → 0.011 |

It did not fail to learn. It learned the wrong thing, quickly.

The 4× ratio was not the flaw. An additive alive term pays for *existing*, so standing still had a
positive score worth protecting, and the path to walking descends before it climbs: a real step
risks a fall while creeping forward pays almost nothing, since the velocity term scales with speed.

Seeding forward momentum at episode start (`RagdollRLBridge.InitialForwardSpeed`, RSI aimed at the
exploration barrier rather than at a start pose) was tried next. It fired — falls quadrupled on the
first rollouts — and was then **absorbed just as cleanly**: the agent learned to plant and kill
0.65 m/s in about 0.18 s, roughly 289 N, well inside foot friction. Any fix gets absorbed while
standing still still pays.

In product form standing still scores exactly **zero**, because the speed factor is zero. There is
no comfortable state left to protect. Measured on the same 5-minute budget:

| | start | end |
|---|---|---|
| `walk/forward_speed` | 0.0165 | **0.0507** |
| `walk/distance` | 0.131 | **0.303** |
| `reward/progress` | 0.737 | **1.548** |
| `walk/fell` | 0.162 | 0.476 |

Falls rose because the policy is now attempting motion instead of protecting a score. The two
changes are synergistic: RSI supplies initial speed, and under the product form killing that speed
drops the reward to zero, so the momentum becomes something to preserve rather than to damp.

**The fall penalty is 0, deliberately.** It was −5. Under a product reward that is actively harmful:
early in training the agent cannot walk, so everything scores about zero, and a negative fall
penalty makes standing still (0) strictly better than attempting anything (risking −5) — the exact
risk aversion the rewrite exists to remove. The real cost of falling is the forfeited remainder of
the episode, which correctly scales with how much the agent has to lose and is near zero while it
has nothing to lose yet.

**The speed factor clamps to `[0,1]`, not `[-1,1]`.** A negative speed factor would flip the sign of
the uprightness factor, so walking backwards while upright would score *worse* than walking
backwards while toppling. Backward motion earns nothing; it does not earn negative.

**Saturating at target speed.** The first thing a policy discovers is that diving forward produces
speed. Capping means exceeding target buys nothing, so the only way to score higher is to *sustain*
it — which requires not falling.

#### The 2-hour run: locomotion, then a second local optimum

Run to 62.1M steps. The product reward worked — the dummy genuinely moves:

| | start | end |
|---|---|---|
| `walk/forward_speed` | 0.0165 | **0.4305 m/s** |
| `walk/distance` | 0.131 | **0.978 m** |
| `walk/lateral_drift` | 0.154 | **0.122** ↓ |
| `standing/grounded` | 0.999 | 0.834 |

Drift *fell* while distance rose sevenfold, so it is travelling roughly straight, and `grounded`
dropping to 0.834 means the feet leave the ground on 17% of ticks — flight phases, i.e. real
stepping rather than sliding.

But it falls in **every** episode, surviving about 2.0 s of 6. And the segment trace shows why that
is not simply "not finished yet":

| segment | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `forward_speed` | 0.155 | 0.361 | 0.391 | 0.407 | 0.414 | 0.420 | 0.424 | 0.428 |
| `alive_fraction` | 0.644 | 0.343 | 0.332 | 0.329 | 0.331 | 0.333 | 0.336 | 0.338 |
| `grounded` | 0.983 | 0.931 | 0.902 | 0.885 | 0.870 | 0.856 | 0.845 | 0.834 |

`alive_fraction` is flat across six segments while speed keeps creeping up and `grounded` keeps
falling. The policy is optimising the wrong axis: it became a **2-second sprint ending in a
guaranteed fall**, and got progressively more ballistic about it.

It is a genuine barrier, not slow progress. Escaping requires going *slower* for a while — trading
peak speed for balance — and every step in that direction loses reward immediately.

The fix was `TargetSpeed` 1.0 → 0.4 m/s. Saturation was always the intended guard, but at 1.0 m/s it
never engaged: at 0.43 m/s the speed factor was only 0.43, so faster still paid linearly. Setting
the target *below the speed already achieved* pins the factor at 1.0, makes extra speed worth
exactly nothing, and leaves surviving the window as the only remaining gradient. This is a
curriculum value — raise it once a full window is sustained.

**A heading penalty, because the observation has no heading.** Bone rotations in `GetUpObservation`
are root-relative, so the policy cannot see which way it points in world terms, and nothing else in
the reward would object to walking in circles. The forward axis is captured once at episode start
(`-pelvisBasis.Z` projected flat, the rig convention shared with `BiomechanicalKinematics` and
`OrientationClassifier`) and held constant — recomputing it per tick would let the agent turn to
face wherever it is drifting and collect the velocity term for a circle.

`WalkTermination` has **no success condition**, deliberately. The only meaningful success is "still
walking when the window ended", which is what reaching `TimeLimit` already means; a threshold like
"5 m travelled" would end the episode at the moment the agent is doing the thing being trained, and
pay it to stop. Falling *does* end the episode, and that asymmetry carries the signal — going down
at t = 1 s of 6 s forfeits roughly 40 reward, far more than any sane explicit penalty.

Fall is `head < 1.00 m` **or** `tilt > 50°`. Height alone cannot catch a fall in progress: a body
pitched 60° forward with its head still at 1.05 m is unrecoverable on this rig but passes the height
check, and every tick it survives there pays alive plus a large forward-velocity term as it
accelerates downward. A 0.25 s settle grace stops a reset transient ending the episode on tick one.

Metrics land in their own `walk/` namespace (`forward_speed`, `distance`, `lateral_drift`, `fell`,
`alive_fraction`) because they carry **units**. They are neither reward terms — which must sum to
the episode reward, an invariant that makes the decomposition checkable — nor `[0,1]` rates. "Walked
0.4 m" versus "walked 4 m" is the whole question, and no fraction expresses it.

#### It is bounding, not walking — the next thing to fix

`standing/grounded` is the fraction of ticks with at least one foot in contact. A walk has a foot
down essentially always, so this should sit near 1.0. It does not, and it keeps getting worse:

| run | `grounded` start → end |
|---|---|
| `walk_v2_0` (2 h) | 0.983 → 0.834 |
| `walk_v3_0` (2.5 h) | 0.822 → 0.785 |

At 21% airborne the dummy is **bounding**: it launches, covers about 0.9 m in roughly two ballistic
strides, and crashes. `walk/fell` has been pinned at 1.000 for both runs, and `alive_fraction`
asymptotes near 0.37 — about 2.2 s of a 6 s window — no matter what the speed terms do.

The cause is a gap in the reward: **nothing in it asks for ground contact.** `progress` is
`speed × upright`, and a ballistic leap satisfies both factors beautifully right up until landing.
Flight is not merely tolerated, it is the most efficient way to score, because airborne travel is
fast and the torso stays vertical.

The obvious next change is a third factor on the product — a ground-contact term, so that leaving
the floor stops paying:

```
progress = VelocityWeight × speedFactor × uprightFactor × groundedFactor × dt
```

Kept as a factor rather than an additive penalty for the same reason the rest is multiplicative: an
additive contact bonus becomes another thing to farm while standing still, and the whole point of
the product form is that no term can be collected without the others. It also wants to be lenient
about brief double-flight rather than binary, or it will punish the natural moment of transfer
between steps.

This was diagnosed at 07:45 with 1 h 20 m of run left and deliberately **not** applied: a new
variable introduced that late would be under-trained and its effect unattributable, and it would
have forfeited the run's steady gains for an untested hypothesis. It is the first thing to try next.

#### What to expect

Standing took 46.7M steps to reach ~95%. Two hours is roughly 15M steps at ~2,000 steps/s, for a
harder problem; published humanoid walking with PPO typically needs 50–200M. **A 5 s straight-line
walk is not a two-hour result.** Grade it in tiers instead: forward displacement above baseline
while upright (expected) → visible weight shift and one recognisable step (plausible) → two
consecutive steps without falling (optimistic) → 5 s walk (not expected).

This is not a detour from the get-up either. The get-up curriculum's hard wall at `pose_0.91` was
diagnosed as *requires stepping*, so a stepping prior feeds directly back into it.

---

## Environment architecture (`Source/RL/`)

Four things define the RL task, each behind its own interface in `Interfaces/IRlComponents.cs` and
composed in `RagdollRLBridge._Ready()`. The bridge owns transport and episode bookkeeping only; it
does not decide what the task *is*.

| concern | interface | current implementation |
|---|---|---|
| what the policy can do | `IRlActionSpace` | `JointLimitedActionSpace` |
| what the policy sees | `IRlObservationBuilder` | `GetUpObservation` (106 floats) |
| what it is paid for | `IRlRewardFunction` | `GetUpProgressReward` |
| when an episode ends | `IRlTerminationCondition` | `GetUpTermination` |

Three optional companion interfaces (`IRlRewardDiagnostics`, `IRlTerminationDiagnostics`,
`IRlActionDiagnostics`) let an implementation report *why* it produced what it did. They are kept
separate so a component works without them, and every one of them exists because a scalar alone
was ambiguous in a way that cost real training time:

- a flat total reward cannot distinguish "no term has signal" from "two terms cancel";
- a success condition that never fires cannot distinguish "never got there" from "unreachable";
- a saturated actuator cannot be seen at all from the action vector.

Each implementation also has `Describe()`, whose output is written into every run's
`manifest.json`. The values are read from the live rig rather than restated on the Python side, so
the record cannot drift from the code that produced it.

### Metrics this produces in TensorBoard

| tag | meaning |
|---|---|
| `reward/{shaping,upright,effort,terminal}` | per-term episode totals; signed to sum to `reward/total` |
| `standing/{grounded,head,tilt,speed,icp,all}` | fraction of ticks each success sub-condition held |
| `standing/act_saturation` | fraction of action components pinned at a joint bound |
| `standing/started_standing` | fraction of episodes begun standing (see RSI below) |
| `standing/start_pose_t` | mean start pose drawn, on the 0 = prone → 1 = standing scale |
| `standing/curriculum_t` | the curriculum floor: hardest level currently being sampled |
| `episode_end/<reason>` | fraction of episodes ending for each reason |
| `start_standing/*` | began at exactly upright — the refresher |
| `start_prone/*` | began exactly flat. **The get-up.** Definition unchanged since the first run, so the whole history stays comparable |
| `curriculum/*` | everything strictly between the two endpoints |
| `pose_0.97/success` … | per-level, two decimals. Says *which rung* the agent is stuck on |

`reward/*` summing to `reward/total` is a real cross-check, not an identity: the total is
accumulated independently in the bridge.

Useful calibration for `act_saturation`: an untrained Gaussian policy sits at **0.342**
(= P(|x| ≥ 0.95) for N(0,1)). A value near that means the policy has not learned to modulate
effort yet, regardless of what the reward curve is doing.

### Reverse curriculum (start-pose continuum)

The start pose is no longer one of two options. `HumanoidRagdoll.TeleportToGetUpPose(t)` places the
body anywhere on a continuum where **t = 0 is prone and t = 1 is standing**, and both endpoints
reproduce the old poses exactly — `t=1` is the identity transform, `t=0` is `DropToProne`'s −90°
pitch plus its 0.5 m lift. It generalises the two rather than adding a third convention.

**Why a continuum at all.** Prone-start training scored 0% over 63M steps in one run and 32M in
another. The reward's shaping term telescopes to `10 × (height gained)`, so an episode that never
rises earns nothing regardless of what it attempted — there was no gradient anywhere between the two
poses the project had.

**Why it runs backwards.** The curriculum starts at the *goal* and walks toward prone, because the
get-up's terminal state is the one thing already learned. Standing reached 87% on a 1.5 s hold at
32M steps; at 36% it would not have been solid enough to build on.

**Why the interpolation is a rigid whole-body transform.** Driving it from `ProneRecoveryTrajectory`
would give anatomically real intermediates — an actual half-kneel instead of a tilted body — but that
trajectory returns **joint angles only**. It says nothing about where the pelvis ends up in the
world, which is the half of the pose that separates lying down from kneeling. Recovering the root
needs forward kinematics over the bone hierarchy, and an FK error there produces a self-intersecting
pose the solver resolves explosively. A rigid transform moves every bone by the same matrix, so all
relative joint geometry is preserved exactly — the property that already makes `DropToProne` safe.

#### The wall is brittleness, not geometry

Measured at 32.6M steps, one rung apart:

| level | tilt | success |
|---|---|---|
| `pose_1.00` | 0.0° | 0.857 |
| `pose_0.99` | 0.9° | 0.799 |
| `pose_0.98` | 1.8° | **0.385** |
| `pose_0.97` | 2.7° | **0.081** |
| `pose_0.96` | 3.6° | 0.005 |

A support-polygon argument predicts the tipping point at `asin(0.15 / 0.90) = 9.6°`, i.e. t ≈ 0.893.
**The real cliff is at 1.8–2.7°**, where the CoM has moved 0.03–0.04 m against a 0.15 m allowance —
nowhere near the geometric limit.

So the limit is the policy, not the body. After 32M steps starting from one identical pose at zero
velocity it learned something close to an open-loop trajectory rather than a feedback controller, and
2° of tilt is already out of distribution. This is also why the arena shows it toppling after 1–2 s,
and why "87% success" overstated the skill. Brittleness is learnable; a physical limit would not be.

#### Frontier sampling — the part that is easy to get wrong

`CurriculumStep` is **0.01**, not 0.1: at 0.1 the very first rung lands past the cliff, and since a
level only advances *on success*, the curriculum freezes there for the whole run while looking
healthy.

More subtly, **only frontier episodes may vote.** Sampling is uniform over `[floor, 1]`, so with the
floor at 0.955 the mean draw is 0.978 — the advance window fills with easy rehearsal episodes and
clears the 70% gate while the frontier itself is at zero. That is not hypothetical; it is what the
first curriculum run did:

```
floor descended 0.990 -> 0.955     while   pose_0.96 = 0.005,  pose_0.95 = 0.000
```

The floor had walked past levels the agent could not do at all. Two changes fix it:

* `CurriculumFrontierShare` (**0.5**) draws half of curriculum episodes from `[floor, floor+step]`.
  This also keeps the decision rate constant as the range widens — under uniform sampling the
  frontier's share shrinks from 22% at floor 0.955 to 2% at floor 0.5, so voting would have crawled
  exactly when the levels got interesting.
* Only poses inside that band count toward the advance. Rehearsal and refresher episodes measure the
  past.

Verified against the same checkpoint and wall-clock: the floor now settles at 0.970 and **holds**
instead of drifting, and `pose_0.97` climbed 0.081 → 0.270 within two minutes purely from the extra
frontier samples.

The advance window is **cleared** on promotion, not slid — a slid window still holds the successes
that triggered it and would re-fire immediately, running several levels down before the policy has
seen the new distribution.

#### What this does not do

It will not produce a get-up. Expect the floor to stall somewhere in the 0.90s, because a rigid tilt
cannot express a **change of support** — feet, to hands and knees, back to feet — which is what an
actual get-up is. Getting past that needs start poses with real support configurations, which means
snapshotting the procedural get-up (run it, record bone transforms at N points, teleport to a stored
snapshot) rather than interpolating a transform. Those are guaranteed valid because simulation
produced them.

What it *does* buy is robustness, and the table above shows how badly that is needed.

### Reference State Initialization

`RagdollRLBridge.StandingStartProbability` (now **0.2**) is the fraction of episodes that begin at
exactly upright. Since the reverse curriculum above took over start-pose selection, this is no longer
the standing/prone split — it is a **refresher** that keeps the goal state alive no matter how far
the curriculum has walked, and its episodes are excluded from the advance vote. Set it to 0 to hand
every episode to the curriculum.

Watch it: when the curriculum introduces harder poses, `start_standing/success` drops (0.87 → 0.68
over two minutes in one measurement) as the policy trades a memorised pose for a general one. That is
expected. A slide past ~0.5 that does not recover is catastrophic forgetting, and this is the dial.

The original rationale still holds for why standing starts exist at all. A policy that only ever
starts prone never observes a standing
state at all, so it cannot learn "stay up" — DeepMimic's ablation reports the agent "never
discovers such high reward states" without this.

It ran at 1.0 for the first 22.6M steps and reached a 94% success rate — but on a 1.05 s episode of
which 0.75 s is the mandatory hold, i.e. it learned *"don't fall over in the first third of a
second"*, never a get-up. 0.5 reintroduces the prone start.

Share by **episode** is not share by **sample**, which is what trains: episode length scales with the
start pose, so easy poses contribute far fewer samples than their episode count suggests. Under the
old 50/50 standing/prone split a standing episode was ~30 decision steps against a prone one's 120,
putting ~80% of samples on the prone half despite an even episode split. The same asymmetry now
applies across the continuum, since the window is `Lerp(8 s, 4 s, poseT)` — see **Episode length**.

That interpolation is why one shared window constant cannot serve: a start halfway up needs more time
than a standing one (there is a rise to perform first) and less than a prone one (most of the rise is
already done), and giving every non-standing pose the full 8 s would spend most of the curriculum's
samples on episodes that ended long before the timer.

**Once this is below 1.0, `ep_rew_mean` is a mixture of two tasks** with very different difficulty
and stops being readable on its own. Judge on `start_prone/success` — that is the get-up.

`EffortWeight` (default 0.02) is deliberately near zero during discovery: HumanUP reports that
deployment-strength control regularization applied from the start "fails entirely", because the
penalty suppresses exactly the vigorous motion that finding a get-up requires. Raise it toward
0.25 once a get-up exists and the goal becomes making it smooth.

### Episode length

`RagdollRLBridge.MaxEpisodeSeconds` is **4.0** (8.0 for the first 12M steps, then 3.0). At 15 Hz
that is ~60 decision steps per episode. No scene overrides it, so the C# default governs all four
RL scenes.

The 8 → 3 cut was sample composition, not difficulty. Under RSI the body starts standing, holds the
full success criterion for ~0.26 s, falls, and then lies on the floor for the rest of the episode —
only ~12 of 121 decision steps carried any signal at 8 s. Shortening raised that share from ~10% to
~27% at unchanged throughput. It did **not** make success closer: the failure happens in the first
second either way.

The 3 → 4 raise landed with `StandingHoldSeconds` 0.75 → 1.5, and it buys margin rather than time.
The cap is asymmetric — a standing episode that succeeds ends near 2.1 s and never reaches it, so a
longer window is only paid for on failures. What 3 s cost was **false negatives**: with a 1.5 s hold
the latest a hold can begin and still complete is t = 1.5, against a policy that enters the standing
region at ~0.6 s. An attempt entering at t = 1.6 is truncated mid-hold and scored a failure while on
track — and `Inverted` has never once fired in this project, so falling and timing out report the
same terminal reason and cannot be separated after the fact. 4 s moves the false-negative band from
[1.5 s, ∞) to [2.5 s, ∞).

Not shorter than 3 s, for two reasons — see RL-DESIGN-NOTES for both: truncated episodes are only
as good as the value bootstrap handling them, and the framework has an obs off-by-one costing 1/45
of samples at 3 s, 1/60 at 4 s, versus 1/30 at 2 s.

### Standing hold

`GetUpTermination.StandingHoldSeconds` is **1.5** (was 0.75). This is the success *specification*,
and at 0.75 s it was under-specified: a policy that falls over immediately afterwards satisfies it.
One did. At 22.7M steps `start_standing/success` read 0.96 while the same policy in
`RagdollStandArena` — which sets `PlaybackMode`, so nothing ever terminates or resets — stayed up 1–2 s
and then went down. The 0.96 was accurate and measured nothing past t = 0.75 s, because success ends
the episode there.

`reward/terminal` was 9.38 of an `ep_rew_mean` of 9.02, so the standing bonus is effectively the
entire return and the policy optimised exactly the quantity that stops being observed.

**The arena is the honest test.** `PlaybackMode` disables every termination, so what you watch there
is unbounded, unlike anything training ever sees. Check a policy there before trusting a success
rate.

Expect `start_standing/success` to fall from 0.96 when this lands. **0.3–0.7 is the right
difficulty**; a collapse to ~0 means 1.5 s is past what the policy can reach, the +20 bonus has
vanished from the return, and the fallback is 1.0–1.25 rather than pushing on.

### Learning rate and the KL trust region

`--learning_rate` is **1.5e-4**, half of SB3's 3e-4 default.

At 3e-4 the trust region was the binding constraint on *every* iteration: `approx_kl` sat at
0.028–0.032 against `target_kl = 0.02`, so SB3's `1.5 × target_kl` abort fired as
`Early stopping at step 3` on every rollout and **7 of 10 epochs were discarded**. Raising
`target_kl` would have silenced the message by removing the safety valve; halving the step size
instead makes the updates fit inside the region PPO was configured with, so all 10 epochs run.

This does **not** promise faster learning — it makes the optimization well-posed. Ten small epochs
versus three large ones is an empirical question; judge it on the `standing/icp` slope over ~2M
steps, not on `approx_kl` alone.

> **Changing it on a resume takes two lines, not one.** `learning_rate` is *not* like
> `batch_size` / `target_kl` / `ent_coef`, which PPO reads fresh at loss time. PPO applies
> `self.lr_schedule(self._current_progress_remaining)`; `self.learning_rate` is only the value
> `_setup_lr_schedule()` converts into that callable. Verified on SB3 2.4.0 against a real
> checkpoint: assigning the attribute left `lr_schedule(1.0)` at the restored 3e-4, and only
> `model._setup_lr_schedule()` moved it. Both lines are in the restore path in `rl/train.py`.
> Getting this wrong looks exactly like the change working.

`manifest.json` records it under `trainer.learning_rate`, read through the schedule rather than the
attribute so it reports what is actually in effect.

### Truncation bootstrapping

`TruncationBootstrapWrapper` in `rl/train.py` fixes a framework-level bug: `godot_rl` computes a
truncation flag and then discards it, so SB3 never received `TimeLimit.truncated` or
`terminal_observation` and **never bootstrapped a timed-out episode** — every episode end was
trained as absorbing with V(s_T) = 0.

It prints `truncation bootstrapping active` once, on the first episode it handles. **If that line
does not appear, the wrapper is inert** and something upstream stopped emitting `episode_end_reason`
(the bridge only emits it when the reward implements `IRlRewardDiagnostics`).

It sits *inside* `VecMonitor` deliberately, so `rollout/ep_rew_mean` keeps measuring real
environment reward rather than absorbing the bootstrap term.

### Comparing runs across an episode-length change

`ep_rew_mean` is **not comparable** across a change to `MaxEpisodeSeconds`, in either direction:
`reward/shaping` telescopes to `10 × (Φ_end − Φ_start)` and so gets less negative purely because
there is less time to reach the floor, while `reward/upright` is capped at `MaxEpisodeSeconds`.
Expect a large jump that means nothing. Judge on `standing/*`, which are per-tick fractions and
therefore length-invariant.

A change to `StandingHoldSeconds` breaks it further and in the opposite direction: the +20 terminal
bonus gets strictly harder to earn, so `ep_rew_mean` drops for a reason that has nothing to do with
the policy getting worse. The 3.0 → 4.0 and 0.75 → 1.5 changes landed together, so **do not read
anything into `ep_rew_mean` across that boundary at all.** `start_standing/success` and
`start_prone/success` stay meaningful, because each is a rate against its own criterion.

## Quick-launch scripts (`rl/scripts/`)

Edit **`config.ps1`** — every other script reads it. That is the one place to change instance
count, speedup, step budget, experiment name, or checkpoint interval.

| script | what it does |
|---|---|
| `export.ps1` | rebuilds the standalone binary training runs against (never edits `project.godot`) |
| `train.ps1` | export, then train headless |
| `train-viz.ps1` | same, but one instance renders in a window |
| `resume.ps1 <checkpoint.zip>` | continues from a checkpoint (policy + optimizer + step count) |
| `tensorboard.ps1` | curves at http://localhost:6008 (own terminal; blocks) |
| `test-scene.ps1` | opens the inspection scene (no training, no server) |
| `promote.ps1 <file.onnx>` | copies a trained policy to `Models/policy.onnx` for in-engine playback |

## Checkpointing

Saved every `$SaveEverySeconds` (default 60) as a **matched `.zip` + `.onnx` pair**, step-stamped,
inside `rl/runs/<name>_N/`:

```
rl/runs/getup_v1_1/
  checkpoint_000123456.zip / .onnx
  final_001000000.zip / .onnx
  manifest.json
  events.out.tfevents...
```

Nothing is ever overwritten. Filenames carry the step count and the run directory auto-increments
per session, so **every session keeps its full history** and you can always go back to an earlier,
more stable policy. A crash costs at most one interval; Ctrl+C stops cleanly and writes a final
pair.

Both formats are written together deliberately. They diverged once in development (an `.onnx`
exported from a checkpoint that was later overwritten), leaving two files in a folder that came
from different models.

### Why not "train in short segments and restart each time"

It sounds safer but measures much worse. Startup at 40 processes costs **~100 seconds** —
`GodotEnv` construction is sequential and `sync.gd` waits 1s per process before connecting. In a
measured run, 104s of wall time contained ~5s of actual training.

| approach | overhead |
|---|---|
| in-process save every 60s | **~2.7%** |
| restart loop, 60s segments | ~63% |
| restart loop, 10min segments | ~14% |

In-process checkpointing already bounds crash loss to one interval, which is the property the
restart loop was meant to buy. Restarting would only add value against memory leaks or hangs over
very long runs; `--restore` makes that easy to add later if it ever becomes a real problem.

## Why `rl/` and `build/` are hidden from Godot

Both contain a `.gdignore`. Without it the export packs every checkpoint `.onnx` and every
`manifest.json` into the game build, which then grows with each training run (it also packed its
own previous output). `Models/` is scanned instead — use `promote.ps1` to copy a chosen policy
there for in-engine ONNX playback.

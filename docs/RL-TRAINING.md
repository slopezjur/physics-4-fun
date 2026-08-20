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
  --export-release "Windows Desktop" `
  "<project>/physics-4-fun/build/RagdollRLArena.exe"

# 2. Train (run from the project root)
rl/.venv/Scripts/python.exe rl/train.py `
  --env_path=build/RagdollRLArena.exe `
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
  "<project>/physics-4-fun/build/RagdollRLArena.x86_64"

# 2. Train
rl/.venv/bin/python rl/train.py \
  --env_path=build/RagdollRLArena.x86_64 \
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
  "<project>/physics-4-fun/build/RagdollRLArena.zip"

# 2. Train
rl/.venv/bin/python rl/train.py \
  --env_path=build/RagdollRLArena.zip \
  --n_parallel=40 --speedup=16 --timesteps=10000000 \
  --experiment_name=getup_v1 \
  --save_model_path=rl/getup_v1.zip \
  --onnx_export_path=rl/getup_v1.onnx
```

**Note:** `export_presets.cfg` currently defines **only** the `Windows Desktop` preset. Linux and
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
- `rollout/ep_rew_mean` - only readable while `StandingStartProbability` is 0 or 1. In between it
  is a mixture of two tasks of very different difficulty and will mislead you.
- `rollout/ep_len_mean` - with a 50/50 start split and windows of 4 s / 8 s, expect roughly
  `0.5*60 + 0.5*120 = 90` when nothing is succeeding. A standing start that succeeds ends near
  2.1 s (~32 steps) instead of the 60-step cap, so the range to expect in practice is **77-90**,
  falling as `start_standing/success` rises. Dropping below that means episodes are ending early,
  i.e. reaching `Standing` (good) or `Inverted` (bad).

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
run/main_scene="res://Scenes/RL/RagdollRLArena.tscn"
run/main_scene.training="res://Scenes/RL/RagdollRLTraining.tscn"

# export_presets.cfg
custom_features="training"
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
| `episode_end/<reason>` | fraction of episodes ending for each reason |
| `start_prone/success`, `start_standing/success` | success rate split by start pose |
| `start_prone/episodes`, `start_standing/episodes` | episode counts behind those rates |

`reward/*` summing to `reward/total` is a real cross-check, not an identity: the total is
accumulated independently in the bridge.

Useful calibration for `act_saturation`: an untrained Gaussian policy sits at **0.342**
(= P(|x| ≥ 0.95) for N(0,1)). A value near that means the policy has not learned to modulate
effort yet, regardless of what the reward curve is doing.

### Reference State Initialization

`RagdollRLBridge.StandingStartProbability` (now **0.5**) is the fraction of episodes that begin
already standing rather than prone. A policy that only ever starts prone never observes a standing
state at all, so it cannot learn "stay up" — DeepMimic's ablation reports the agent "never
discovers such high reward states" without this.

It ran at 1.0 for the first 22.6M steps and reached a 94% success rate — but on a 1.05 s episode of
which 0.75 s is the mandatory hold, i.e. it learned *"don't fall over in the first third of a
second"*, never a get-up. 0.5 reintroduces the prone start.

Half by **episode** is far more than half by **sample**, which is what trains: a standing episode is
~16 decision steps and a prone one ~120, so a 50/50 episode split puts ~88% of samples on the prone
task. That is also why the episode window is per-start-pose — `MaxEpisodeSeconds` (4.0 s) for
standing starts, `ProneMaxEpisodeSeconds` (8.0 s) for prone. A get-up cannot complete in the
standing window, so one shared constant would make the prone task unsolvable by construction.

Raising `StandingHoldSeconds` to 1.5 s shifts that sample split, and the shift is the non-obvious
cost of the change. Standing episodes grow from ~20 to ~32 decision steps on success and 60 on
failure, so the standing task's share of **samples** roughly doubles, ~14% to ~28%, taken directly
from prone. That is intended — a policy that can actually stand is a prerequisite for a get-up that
terminates in standing — but `StandingStartProbability` was left at 0.5 rather than lowered to
compensate, so the reallocation is real and deliberate.

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
`RagdollRLArena` — which sets `PlaybackMode`, so nothing ever terminates or resets — stayed up 1–2 s
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

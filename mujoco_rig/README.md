# MuJoCo rig

The dummy's physics lives in MuJoCo and Godot renders it. Phase 0 measured ~100 authored
configurations of the Jolt ragdoll and none could hold single-leg support; MuJoCo holds it for 17.5 s
on the same rig. Godot drives MuJoCo's C library through P/Invoke, so there is no GDExtension and no
second physics authoring.

**Where the track stands, what is shipping and what to do next: [STATUS.md](STATUS.md).** This file
says how the pieces work.

The isolated motion-imitation experiment is documented in [mimic/README.md](mimic/README.md).
It includes a licensed standing reference, raw-torque and reference-relative target/PD
Stand experiments, and an isolated Godot replay adapter. The new 362-channel target/PD
actor learned to pass eight of eight three- and five-second Stand trials during a
30-minute run. Late updates regressed, so the selected minute-24 checkpoint is
preserved separately. It passes Godot transfer checks and remains experimental.
See its measured limitations and reproduction commands before training further.
The trainer now supports guarded fine-tuning with frozen normalization and bounded
actor updates. New runs retain and export their best evaluated checkpoint separately
from the final training state; see the experiment README for the explicit warm-start command.

Open `Scenes/RL/Isaac3/MuJoCo/MimicStand.tscn` with F6, or use F5 (the configured
main scene), to view the selected checkpoint. It needs no environment variables.
R restarts the five-second trial; P pauses/resumes. The result stays visible at the
end rather than automatically resetting. Policy bundle and native library paths
are configured on the root node in the Inspector. `MimicStandReplay.tscn` is for
automated parity checks only.

For controlled pushes, open `Scenes/RL/Isaac3/MuJoCo/MimicPerturb.tscn` with F6.
It probes the retained Stand policy with timed chest forces; no Perturb actor has
been trained yet. Direction buttons restart, R repeats and P pauses. Force and timing
are Inspector settings. The 52-case baseline and its simulator comparison are in
[mimic/README.md](mimic/README.md) and [STATUS.md](STATUS.md).

## The body

```
build_mjcf.py      generates dummy*.xml from GODOT's own rig dump - Godot stays the single source
                   of truth, the rig is never hand-copied
plant_dump.txt     captured [PLANT] output; regenerate with godot_run.py if the rig changes
validate.py        gate: height, mass, a rest pose that does not self-intersect, an unpowered fall
                   that crumples
gen_offsets.py     regenerates Source/RL/MuJoCo/MjLayout.cs after a MuJoCo version bump
```

Regenerate after any rig change:

```
python isaac_lab_3/scripts/godot_run.py --scene Walk/IsaacWalkCheckNewton.tscn --seconds 2 \
  | grep "^\[PLANT\]" > mujoco_rig/plant_dump.txt
python mujoco_rig/build_mjcf.py                                  # dummy.xml + dummy_ball.xml
python mujoco_rig/build_mjcf.py --no_actuators --out dummy_limp   # the unpowered ragdoll
python mujoco_rig/validate.py
```

`dummy*.xml` are generated. Do not hand-edit them. `build_mjcf.py` runs in named stages - scale to
height, reshape torso and head, place the arms, normalise the height, emit - and the dimensions those
stages derive live in a `Proportions` object rather than in the module's own constants.

**1.750 m, 69.6 kg (BMI 22.7), 18 bodies, 33 actuators of which the policy drives 30.**

### Muscle and ligament

Every joint carries two layers, and this is the single most important thing about the plant:

| layer | what it is | value |
|---|---|---|
| ligament | passive joint spring + damping, always present | 5% of the joint's peak human torque per rad |
| muscle | a `motor` actuator - the policy's output IS newton-metres | `forcerange = ±HUMAN_TORQUE` |

**Zero policy output is zero torque, so an unpowered body is a ragdoll.** The earlier plant used
`position` actuators, whose zero command means *hold the rest pose* - the body flexed 1.8° on average
while it toppled, and there was no setting that meant "no muscle". "Dead", "stunned" and "alive" are
now a scale on the policy's output rather than different models.

The neck is the deliberate exception: it keeps a `position` actuator, because it is outside the
action space (`env_config.POLICY_EXCLUDE`) and its whole job is muscle tone. Drop its gain to zero
and the head flops to the floor.

`JOINT_ARMATURE = 0.02` is not optional - without it 10 of 10 random-torque rollouts diverged with
`Nan, Inf or huge value in QACC`, and MuJoCo *silently resets the state* when that happens.

### The rest keyframe

`qpos0` is **not** a valid pose: the arms hang inside the legs there, 5.5 cm of forearm inside the
thigh, 11 contacts, 331.8 N·m per shoulder just to hold. Every model therefore carries a `rest`
keyframe with the shoulders abducted 8°, and everything that starts an episode resets to it -
`body_env.py`, `body_env_warp.py` and `MjBridge.ResetData`. `build_mjcf.add_rest_keyframe` refuses to
write a file whose rest pose self-intersects.

## Training

```
scripts/overnight.py    chained sessions (Perturb: at most 5 min): train, score on CPU, reject regressions,
                        offer the best to promote.py at the end
scripts/promote.py      ships a checkpoint to the scenes only on a measured win over the incumbent
scripts/scoring.py      how each task is scored - the one reader of the scorers' JSON
scripts/preflight.py    is the task winnable, does the reward pay for the right thing, is one
                        exploding world harmless - run before any long session
scripts/batch_table.py  where the useful batch is, on this plant
scripts/bench.py        throughput and hardware headroom
scripts/benchmark_perturb.py  balanced CPU impact bank, selection and locked held-out comparison
scripts/config.ps1      settings for the PowerShell wrappers, with the measurement behind each one
scripts/train.ps1, eval.ps1, parity.ps1, watch.ps1   the wrappers
```

```powershell
python mujoco_rig\scripts\preflight.py --task perturb --seed <checkpoint>
python mujoco_rig\scripts\overnight.py --task perturb --sessions 8 --minutes 5 `
  --envs 4096 --steps 16 --ball_speed 2 --seed <checkpoint> --no_promote
python mujoco_rig\scripts\promote.py --task perturb --checkpoint <checkpoint> --dry_run
cd mujoco_rig\scripts; .\parity.ps1      # after ANY rig, MJCF, env or mujoco version change
```

**Two engines, one rule.** Training runs on `mujoco_warp` (GPU, float32; 4,096 worlds × 16 rollout
steps - the split matters more than the batch). Godot runs MuJoCo's C library, which is float64.
Those two do not produce identical trajectories - `parity.ps1` measures how far apart - so:

> Train on GPU. Score and ship on CPU MuJoCo. A checkpoint is never judged by the engine that
> trained it.

`eval.py` and `eval_walk.py` are CPU-only by design. That is the whole reason this track works where
Isaac did not.

### Stratified Perturb benchmark

`scripts/benchmark_perturb.py` supplements the unchanged `eval.py` promotion gates.
It runs 768 trials per split by default: 12 intended target bones x four world-space
approach sectors x 16 independent pose/heading/launch-time samples. Approach labels
identify the projectile's spawn side (+X, +Y, -X, -Y), not its travel direction or
the dummy's local forward axis. An intended target may be intercepted by another limb.

The accepted CPU policy generates each projectile launch. Candidates replay the same
float32-representable initial pose and world-space projectile position/velocity, with
deterministic CPU inference, one 6 m/s shot, six-second episodes and no automatic resets.
Native rewards stop accumulating at the first training termination. Recovery uses
the existing historical gate and also reports measured-contact recovery. Contact
presence is sampled at control boundaries; an unobserved contact is not proof of a miss.

The tool records the protocol and checkpoint/plant hashes before running any trial.
Selection ranks candidates by recovery count, then survival count, then label; it writes
`selection-lock.json` before generating held-out launches. Only that candidate and the
accepted policy run on the held-out split. Diagnostic confirmation requires no survival
regression and a positive recovery gain with paired z >= 2 on **both** splits. This does
not authorize promotion: quiet standing and the shipped-policy comparison still use
the established gates. The tool never trains, exports, commits or changes those gates.

```powershell
python mujoco_rig/scripts/benchmark_perturb.py `
  --accepted <accepted-checkpoint> `
  --candidate candidate_a=<checkpoint-a> --candidate candidate_b=<checkpoint-b> `
  --out logs/perturb-benchmark-<unique-name>
```

An output directory must be new. `--workers` defaults to four CPU processes. The default
eight shards each contain two samples of every target/sector pair; `--repeats` must be
divisible by `--shards`. Keep shard size fixed when comparing saved banks: inference
batch shape can change floating-point rounding and later contact trajectories. Each
split saves scenario NPZs, trial-ID-aligned per-policy JSON, aggregate/per-body/per-sector
comparisons, all 48 cell comparisons and an artifact hash inventory. Keep a held-out set
sealed until selection; after inspecting it for tuning, treat it as development data
and define a new held-out protocol before a future confirmation run.
`--selection_seed` and `--heldout_seed` choose new independent splits and must differ;
their defaults reproduce the original bank, which is no longer unseen after this audit.

### Offline recovery feasibility probe

`scripts/probe_recovery_feasibility.py` tests whether selected policy failures admit
a recovery with the existing CPU plant, torque authority and two-command action delay.
It selects the first accepted-policy fall in each Head/Chest x four approach strata
from a saved benchmark selection split. It replays the original inference batch,
checks every benchmark output against the saved record, and captures the first
control boundary after a positive-force projectile/body contact. Pending actions
remain queued; no intervention occurs before contact.

```powershell
python mujoco_rig/scripts/probe_recovery_feasibility.py `
  --bank logs/perturb-benchmark-20260921 --out logs/recovery-feasibility-<unique-name>
```

The fixed CEM/MPC search uses full simulator state, a one-second prediction horizon,
96 candidates x four iterations, and replans every six control steps. Intervention
lasts at most two simulated seconds, subject to an 80-second wall budget checked
before each planning call, then returns control to the accepted actor. Evaluation
continues to six seconds after contact or the first fall. The search surrogate is
independent of the unchanged training reward; recovery is scored with both existing
metrics. Original batch-action replay and singleton actor inference are separate
baselines. Saved optimized commands must reproduce all states and applied controls
exactly on an independent CPU replay.

A new feasible recovery requires both baselines to fall and the optimized trajectory
to survive and finish settled under both recovery definitions. This establishes
physical feasibility for that sampled state only. Failure of this bounded optimizer
is inconclusive; neither outcome establishes realistic motion or population-level
policy quality. The oracle has privileged state and substantial offline compute.
It does not train, export, alter the rig or change PPO/reward settings.

### Bounded recovery demonstration pilot

`scripts/recovery_demonstrations.py` extends the feasibility probe to the first three
accepted-policy falls in each Head/Chest approach stratum. The first two trajectories
per stratum train; the third validates. Assignment precedes optimization, and only
surviving trajectories that finish settled under both metrics supply labels. Existing
oracle traces are reused only after a hash check and exact physical replay. New traces
use the same fixed optimizer. Native snapshots retain derived state and are stored
locally with hashes; do not load snapshot pickle files from untrusted sources.

The dataset contains **pre-command `foundation_v2` observations and issued actions**.
Full physics state, projectile state, optimizer information and future observations
never enter actor inputs. The first 120 actions are demonstration targets; successful
policy handback states augment retention targets. Quiet and successful-impact retention
rollouts use separate training seeds. Whole validation trajectories are excluded from
all training groups, including handback retention.

```powershell
python mujoco_rig/scripts/recovery_demonstrations.py `
  --bank logs/perturb-benchmark-20260921 --probe logs/recovery-feasibility-20260921 `
  --out logs/recovery-demonstrations-<unique-name>
python mujoco_rig/scripts/learn_recovery_demonstrations.py train `
  --experiment logs/recovery-demonstrations-<unique-name>
python mujoco_rig/scripts/learn_recovery_demonstrations.py diagnose `
  --experiment logs/recovery-demonstrations-<unique-name>
python mujoco_rig/scripts/learn_recovery_demonstrations.py benchmark `
  --experiment logs/recovery-demonstrations-<unique-name>
```

The collection protocol freezes training settings, candidate selection and new benchmark
seeds before collecting additional teacher outcomes. This is actor-only supervised
learning: critic and exploration parameters remain fixed, and no PPO or reward setting
changes. Candidate selection uses whole-trajectory validation error subject to retention
error limits. If none qualifies, the best-fitting candidate is explicitly rejected by
that filter but can still be evaluated diagnostically. Its hash is locked before any
closed-loop evaluation. Low imitation error alone cannot authorize continuation or
promotion: quiet standing, CPU recovery, a fresh balanced benchmark and motion quality
still determine whether the candidate is useful. These checkpoints are tagged with an
offline training version so a future PPO initialization resets stale value/optimizer state.
The `benchmark` stage evaluates the locked candidate on the newly seeded development
split. Keep the final held-out seed unused when retention or CPU gates already reject
the candidate; only a candidate passing the prerequisites warrants opening that split.

### Environments

```
rl/env_config.py          the constants that DEFINE the tasks; every env, scorer and exporter reads them
rl/body_env.py            CPU base: plant, observation, reset, step, divergence guard, clock stagger
rl/body_env_warp.py       the same on the GPU
rl/perturb_env*.py        + projectile, contact history and recovery feature extraction
rl/recovery_reward.py    shared NumPy/Torch grounded-support and settling objective
rl/recovery_metrics.py   sustained recovery, foot sliding and rapid replant measurements
rl/walk_env*.py           + velocity commands, the heading hold and the walking reward
rl/ppo.py                 the learner
rl/train.py               the loop, --task perturb|walk, --backend cpu|warp
rl/eval.py, eval_walk.py  scoring on CPU; --json writes the result as data
rl/export_onnx.py         actor -> ONNX plus the generated contract
rl/test_env_parity.py     CPU and GPU perturb define the same task - numbers AND interface
rl/test_walk_parity.py    the same, for walk
rl/parity_gpu.py          the two ENGINES still agree, with a CPU-vs-CPU control run
rl/watch.py               a window on the scorer's engine, with live telemetry
rl/probe_*.py             one-off measurements: noise tolerance, the single-impact envelope, what a
                          ball delivers
```

A task environment adds only what differs, through the base's hooks (`_reset_world`,
`_before_physics`, `_after_physics`, `_step_extras`, `reward`): walk no longer inherits a disabled
gun from perturb. History advances once per control step; reading rewards does not mutate it.

### Scoring is data, and tasks are tables

`eval.py` and `eval_walk.py` print for people and write `--json` for tools. `scoring.py` holds one
`Scorer` per task - its command, which result it reads, its metric, what makes a result unusable as
a seed - and both `promote.py` and `overnight.py` decide through it. The printed text used to be
parsed with regular expressions, which is how the quiet-room block was once read as the under-fire
one.

None of the tools branches on the task name. A task is an entry in `train.TASKS` (env classes,
constructor arguments, its field in the log line), `scoring.TASKS`, `overnight.POLICIES` (extra
training arguments, when a running session is hopeless, how a finished one is judged) and
`preflight.CHECKS`.

## Two tasks

```
perturb   stay standing through ball impacts, using the legs
walk      follow a commanded forward speed, lateral speed and turn rate
```

Both default to the **105-observation `legacy_v1` layout**, ending in three command slots that
Perturb leaves at zero and Walk fills. Perturb also supports the opt-in `foundation_v2` contract
described below; its 140 inputs cannot seed the current Walk policy by dropping channels.

### Versioned physical observations

`rl/observation_contract.py` defines the layout and checkpoint validation. Missing version
metadata means `legacy_v1`; unknown versions or incompatible widths fail before inference.
The legacy inputs retain their original meanings, including the foot-height contact proxy
and subtree-reference spatial linear velocity. This preserves shipped checkpoints.

`foundation_v2` appends pelvis-origin linear velocity in the pelvis frame (3 values), actual
floor normal loads summed across each foot/toe in kN (2), and the oldest pending action (30).
Together with the existing previous-action channel, this exposes the complete two-step
actuation queue. CPU, GPU and Godot sample these sensors at the same final physics substep.
No joint geometry, torque limits, delay, reward weights or acceptance metrics change.

Select it explicitly with `train.py --observation_version foundation_v2`. Omitted flags resume
the seed's contract. Migration copies the original actor columns and zeroes the new ones,
preserves exploration, resets the critic/optimizers and requires at least 50 actor-frozen
critic updates plus the existing EV gate. Critic EV is still a bootstrapped-target diagnostic.
A reference-constrained seed also requires explicit `--reference_data` collected with
`build_policy_reference.py --observation_version foundation_v2` from the original teacher.
New sensors are measured during those trajectories, never filled with fabricated zeros.
The evaluator, viewer and ONNX exporter read the checkpoint version automatically.

Foundation validation:

```powershell
python -m unittest discover -s mujoco_rig/rl -p test_observation_contract.py
python mujoco_rig/rl/foundation_audit.py --json logs/foundation-audit.json --native_fixture_dir logs/foundation-native
$env:P4F_FOUNDATION_FIXTURES = (Resolve-Path logs/foundation-native).Path
dotnet test Tests/Physics4Fun.Tests/Physics4Fun.Tests.csproj --filter FullyQualifiedName~Mj
```

The audit checks mass/inertia, rest self-contact, each actuator's signed response and passive
fall behavior. Native integration compares full observations across physics steps plus loaded
heel/toe fixtures against Python. Ordinary managed tests skip native integration unless its
fixture directory is supplied. These checks do not prove all projectile impacts are recoverable.

`test_env_parity.py` and `test_walk_parity.py` assert each task's CPU and GPU implementations agree,
on the numbers **and on the interface**: a missing `env.model` attribute once killed a 55-minute run
at its first curriculum promotion after passing every numeric check.

Walk command ranges and mixture weights live in `rl/walk_config.py`. Pass a frozen
`WalkCommandConfig` to either backend; `train.py --walk_stage` selects one per environment. There is
no process-wide `set_stage`. Scorers set `env.auto_reset = False` and, for one hit,
`env.max_shots_per_episode = 1`; `reset_all()` still performs a real reset.

Fast CPU regression checks (no CUDA required):

```powershell
python -m unittest discover -s mujoco_rig/rl -p test_architecture.py -v
python -m unittest discover -s mujoco_rig/rl -p test_recovery.py -v
python -m unittest discover -s mujoco_rig/rl -p test_training_safety.py -v
python -m unittest discover -s mujoco_rig/rl -p test_policy_reference.py -v
python -m unittest discover -s mujoco_rig/rl -p test_foot_contacts.py -v
```

These cover configuration isolation, landing rewards, complete resets, single-hit behavior,
contract timing, rejected promotions, best-checkpoint selection, and export rollback. Run both
parity scripts after environment changes; the walk test also asserts that a 0.2 s landing earns
its intended reward, so a shared CPU/GPU bug cannot pass merely because both sides agree.
The contact suite includes an additional physical CPU/GPU test when CUDA is available;
it is explicitly skipped on CPU-only hosts.

### Perturb recovery objective, September 2026

`loaded_contact_recovery_v3` retains the grounded-recovery objective introduced by
`grounded_recovery_v2`, which replaced the swing-speed and airborne-placement bonuses. Those terms
paid for fast repeated foot motion without requiring useful support or a stable landing.
The objective now rewards capture-point support from grounded feet and low COM speed, with
costs for grounded sliding, hard landings, rapid repeated replants, crossed feet, joint-limit
excess and summed action changes. A comfortable staggered stance is accepted after a recovery;
the old strict rest-pose score remains a diagnostic.

Ground contact now comes from floor-contact normal load summed across each foot and toe.
The shared hysteresis enters contact above 5 N and exits at or below 1 N. Both adapters read
the final physics substep, using the same reference frame as the body's kinematics. Ball
and self contacts do not count as support. `foot_contacts.py` and `foot_contacts_warp.py` own
measurement; the shared reward kernel and contact history consume it.

Sliding uses the load-weighted mean **squared tangential contact-point speed** per foot,
then the existing grounded mask, cap and weight. Squaring before averaging prevents opposite
velocities during a twist from cancelling. A stationary heel/toe pivot has no slip even if
the foot origin moves. The support region remains a 6 cm capsule around grounded foot centres,
not a measured force polygon. Landing velocity and settled-foot speed still use foot-origin
velocity. The v3 reward migration retained the legacy observations; `foundation_v2` above adds
physical policy sensors separately. Action latency, plant, reward weights and smoothing are unchanged.
The reward version change resets/warm-ups the critic while preserving the incumbent actor.
See [the diagnosis](PERTURB_DIAGNOSIS.md) for the measured defect this fixes.

PPO supports temporal L1 regularisation of deterministic policy means, excluding reset
transitions. This follows the mean-action smoothing approach in
[Reactive Stepping for Humanoid Robots](https://arxiv.org/abs/2203.01148); exploration samples
are not regularised directly. Perturb defaults to a smoothing weight of `0.1` and an actor
learning-rate ceiling of `1e-6` from the first update. The adaptive floor stays below the ceiling,
so a KL stop can still reduce a very small learning rate.
`--temporal_smoothness` and `--lr_max` override these values. Walk keeps its previous defaults.

Checkpoints record reward/trainer versions, both optimisers, remaining critic warm-up and the
full training arguments. A changed reward or value-target version retains the actor/exploration
and resets the critic/optimisers. Critic warm-up freezes the actor and exploration for at least
50 iterations and until explained variance reaches 0.5. Warm-up survives interrupted sessions.
Perturb uses unclipped value regression; actor and critic have separate gradient clipping.
Logs retain post-fit `ev` and also report `ev_before`, signed `bias_before` (prediction
minus target), `rmse_before` and raw `adv_std` from the collection values. These measure
agreement with the batch's bootstrapped GAE targets, not held-out recovery accuracy or
Monte Carlo returns. A high EV can coexist with substantial value bias; neither replaces
CPU survival/recovery gates. The diagnostics do not change the warm-up gate or optimizer.
Exact Gaussian KL can stop actor updates within an iteration (target 0.001, cutoff 1.5x).
A timeout bootstraps from the final pre-reset observation, while a fall never bootstraps; neither
allows GAE to propagate through a reset. Recorded episode returns contain only actual rewards.
Existing ONNX files do not learn these changes; retraining and measured promotion are required.

The CPU scorer's promotion gate remains explicitly versioned `height_proxy_v2`. Its independent
height-contact history preserves historical per-world recovery, sliding and replant outcomes,
even though training now uses measured loads. It requires survival and a final 0.5 s hold: pelvis upright cosine above 0.9,
height above 90% of rest, COM speed below 0.15 m/s, pelvis angular speed below 0.5 rad/s, each
foot below 0.10 m/s, both feet grounded and capture-point support error below 2 cm. A fired
shot must have had time to arrive. Shot counts describe launches, not confirmed contacts.
It also reports sliding distance, rapid replants and mean action change while alive.
Additional `contact_*` results, versioned `loaded_contact_v3`, report measured-contact recovery
and integrated per-foot RMS contact slip alongside the frozen gate. These diagnostics do not
authorize promotion. If the gate is migrated later, both policies must be rebaselined together.
Promotion first requires 100% survival and settling in 16 quiet-room worlds over 40 seconds,
then compares paired recovered outcomes and rejects any survival regression on the same batch. Missing recovery metrics fail closed. Long repeated-shot evaluation is still necessary: the impact portion of the gate is a six-second
single-shot test.

Overnight sessions keep their training speed fixed, advancing only at 85% settled recovery
on the CPU stage test. The trainer's internal curriculum also requires settled episode endings.
The September 20 trials have not produced an accepted replacement. Keep the incumbent.
This is a guarded diagnostic entry point, starting below the game's 48 N.s shots:

```powershell
python mujoco_rig/scripts/overnight.py --task perturb --sessions 1 --minutes 5 `
  --envs 4096 --steps 16 --ball_speed 2 --speed_end 6 --advance_at 85 --no_promote `
  --seed logs/mujoco/2026-09-11_10-08-42_p0911i_s1/model_503.pt `
  --extra "--lr_max 0.000001 --desired_kl 0.001 --temporal_smoothness 0.1"
```

The chain applies the starting checkpoint's CPU stage recommendation before its first
training chunk, just as it does after an accepted chunk. A mastered seed therefore advances
without repeating its old difficulty. The PowerShell resume wrapper inherits the saved
difficulty; `$SpeedStart` controls direct/fresh runs. To deliberately override a resumed stage,
call `overnight.py --ball_speed <speed>` explicitly. Repeated direct-training experiments
at one fixed difficulty remain deliberate ablations; they do not exercise progression
toward the game's impacts.

`train.ps1 -InitFrom <checkpoint> -Minutes 15` runs three evaluated five-minute chunks. It
requires an explicit seed and does not export automatically. `train.py` also validates Perturb
seeds on CPU before GPU setup; deliberate fresh-policy experiments require `--from_scratch`.
A failed quiet-room, reference or stage test cannot seed another chunk. Recovery/survival cannot
drift down through a chain. Perturb stops at the first rejected or aborted chunk (exit code 2),
retaining the accepted seed instead of repeating the same failed experiment. Smoothing coefficients
remain experimental. The controlled comparison of 0, 0.1 and 0.5 at a `3e-5` learning-rate
ceiling rejected all three; lowering smoothing alone did not prevent collapse. A follow-up at
`1e-6` preserved quiet standing through 50 updates. See `STATUS.md` for the impact results before
choosing a checkpoint for further training.

Use the best checkpoint printed by the chain with `promote.py --dry_run` before shipping it.
The separately printed accepted seed carries continuation state, including unfinished critic
warm-up. After a rejection, adjust the experiment before attempting another chunk.
For a measured exploration ablation, direct `train.py --init_from <checkpoint>
--exploration_scale 0.25` scales the saved per-joint Gaussian standard deviations while
preserving the actor's mean function and its optimizer state. Only the standard-deviation
Adam state is cleared. Perturb refits its critic for at least 50 updates, retaining the
explained-variance gate; the actor stays frozen during this adaptation. The existing minimum
standard deviation still applies. Supply the factor only on the initial direct invocation:
later chunks inherit the saved scale. Do not repeat it through a chain's `--extra` arguments.
It is mutually exclusive with `--reset_std`, requires a seed, and leaves the fixed teacher's
reference variance unchanged. Default training behavior is unchanged.
The 2026-09-11 92.4% result used a 5 kg ball at 6 m/s (30 N.s). The current model uses 8 kg
at 6 m/s (48 N.s); comparisons across those masses do not measure policy improvement.

To train a pure STAND (no projectiles) rather than perturb, disable the gun rather than changing the
task - it is the same environment:

```powershell
python mujoco_rig\rl\train.py --task perturb --backend warp --num_envs 4096 --steps 16 `
  --init_from <validated-checkpoint> --ball_every 1000000 1000000 `
  --promote_at 2.0 --max_minutes 5 --run_name stand
```

### Fixed-reference retention experiment

`build_policy_reference.py` collects deterministic incumbent observations and mean actions on
CPU. Seeds 101 and 103 are separate from evaluation seed 17. Only complete trajectories that
survive and settle are kept: standing for 20 seconds and recovery from one 6 m/s shot over six
seconds. The fixed corpus protects demonstrated successes without imitating failed recoveries.
Uniform subsampling over all control steps preserves alternating phases that fixed-stride
sampling can miss.

```powershell
python mujoco_rig/rl/build_policy_reference.py `
  --checkpoint logs/mujoco/2026-09-11_10-08-42_p0911i_s1/model_503.pt `
  --out logs/recovery-reference-data.pt
```

Pass `--reference_data logs/recovery-reference-data.pt --reference_coef 1` to a short Perturb
training run from a validated seed. The zero-coefficient control uses the same corpus and seed.
Reward weights and the temporal smoothing term remain unchanged. Reference sampling has its own
random generator, so enabling it does not perturb the rollout/PPO random stream.

Each actor update adds an equally weighted standing/impact penalty:
`0.5 * sum(((mean_action - fixed_target) / fixed_teacher_std) ** 2)`.
The scale stays fixed; increasing exploration cannot weaken the penalty. Critic-only warm-up
does not sample the reference or update the actor. This is a soft constraint on sampled states,
not a guarantee that the policy retains its physical recovery behavior.

Checkpoints embed the original targets, coefficient, sampling state, source checkpoint hash and
plant hashes. A resumed run reuses that reference, not its own new actor. Current plant hashes
must match, and task/dimension changes fail explicitly. An explicit new artifact replaces the
reference; otherwise omission preserves it. Logs report standing and impact drift separately.
Repeating the same artifact flag preserves the inherited coefficient and sampling position unless
the coefficient is explicitly overridden.
This remains opt-in: CPU survival, settled recovery and movement measurements decide acceptance.

## Running a trained policy in Godot

`promote.py` exports a checkpoint to the scenes when it wins. By hand, for a checkpoint the contract
already names:

```powershell
python mujoco_rig\rl\export_onnx.py --checkpoint logs\mujoco\<run>\model_N.pt
```

writes `balance_policy.onnx` (stand and perturb - one behaviour family, one brain) or
`locomotion_policy.onnx` (walk), plus a generated `.contract.json` beside it. The names are the ones
the Isaac3 track used, so a policy keeps its identity across engines. Point a scene at it:

```
PolicyPath  = "res://mujoco_rig/locomotion_policy.onnx"
WalkCommand = Vector3(0.25, 0, 0)     # forward m/s, lateral m/s, turn rad/s - the speed it trained at
```

`MjPolicyDriver` reads the observation layout, joint order and action mapping **from the contract**,
never from constants in C# - the Isaac track's hand-written contract was wrong four ways and every
one was silent. The contract's `action_to_control.mode` says whether the action is a torque or a
joint target, and its `command` block says how the command channel is filled: a walk brain trained
with the heading hold must see, while its commanded yaw is zero, a correction toward the heading the
command began on (`MjHeadingHold`). Writing the raw zero runs a different policy from the one that
was scored.

The driver validates channel offsets and widths, joint mappings, network dimensions and simulation
timestep before use. The generated `action_latency_steps` reproduces the training action queue;
`command.source` selects zero commands for perturb or velocity commands for walk. Legacy contracts
without latency keep immediate actuation; regenerate their metadata to deploy a delayed policy.
The shipped contracts have been regenerated without changing their ONNX weights.

Native library discovery uses `MujocoLibraryDirectory` when explicitly set, then
`P4F_MUJOCO_LIBRARY`, the active conda environment, the registered `env_isaaclab3`, and finally the
platform loader. No personal installation path is embedded in the scene node. The generated native
layout requires 64-bit MuJoCo at exactly `MjLayout.Version`; mismatches fail before pointer reads.

Setting `PolicyPath` switches OFF both the scripted gait and the pelvis balance assist, which is the
external torque the policy exists to replace. Against a torque-actuated model the scripted
controller is disabled outright, with a log line: its outputs are joint targets in radians, which a
`motor` would read as newton-metres.

The whole path is one engine after training: GPU `mujoco_warp` -> CPU MuJoCo for scoring -> the same
CPU MuJoCo inside Godot. That is what "sim-to-sim" means here.

## Scenes

```
Scenes/RL/Isaac3/MuJoCo/MujocoStand.tscn     stand / balance, no projectiles
Scenes/RL/Isaac3/MuJoCo/MujocoPerturb.tscn   the ball gun, every 3 s
Scenes/RL/Isaac3/MuJoCo/MujocoWalk.tscn      commanded locomotion, 0.25 m/s
```

Stand and Perturb load the SAME `balance_policy.onnx` - they are one environment with the gun off or
on. Walk needs its own `locomotion_policy.onnx`: all three share the 105-observation layout, which is
what lets a balance brain seed a walk one, but a brain that never learned the three command slots
will not walk.

Set `Passive = true` on any of them to drive NOTHING - no gait, no assist, no policy. On the torque
plant that is a true ragdoll, and it is the baseline every result is read against.

## Scene keys

Matching `RagdollDebugInput`, so the MuJoCo scenes behave like the Jolt ones:

```
R        reset the dummy to its rest pose and clear the metrics
B        fire a ball now, instead of waiting out BallInterval
Space    shove the pelvis by PushForce
WASD/QE  fly the camera (CameraController), Shift to move faster
```

`R` matters most: without it a fallen dummy is a dead scene that has to be relaunched, and a
relaunch costs a MuJoCo model load.

## C# layout

```
MjInterop.cs             P/Invoke declarations and the mjtObj constants
MjLayout.cs              GENERATED struct offsets - regenerate with gen_offsets.py
MjBridge.cs              the model/data handle: step, read bodies, write controls and forces; owns
                         the one Godot <-> MuJoCo frame map
MujocoDummy.cs           the scene node: load, drive (nothing / policy / scripted), step, render, input
MjProxyBuilder.cs        builds the Godot meshes by reading the model's own MJCF
MjGaitMetrics.cs         strikes, uprightness, single support, travel, perturbation response
MjScriptedController.cs  the pre-RL gait oscillator and balance assist, for a position-actuated model
MjPolicyDriver.cs        policy clock, delayed actuation and complete reset
MjPolicyContract.cs      validated deployment metadata and action scaling
MjPolicyObservation.cs   observation channels through a read-only simulation interface
MjOnnxPolicy.cs          inference session ownership and tensor validation
IMjPolicyState.cs        observation interface; IMjPolicyPlant adds actuator writes
MjModelDefinition.cs     timestep and actuator mode from the generated MJCF, read before rendering
MjHeadingHold.cs         the contract's command-filling rule for walk
MjBallGun.cs             the projectile, its aim and its delivered impulse
```

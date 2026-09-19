# MuJoCo rig

The dummy's physics lives in MuJoCo and Godot renders it. Phase 0 measured ~100 authored
configurations of the Jolt ragdoll and none could hold single-leg support; MuJoCo holds it for 17.5 s
on the same rig. Godot drives MuJoCo's C library through P/Invoke, so there is no GDExtension and no
second physics authoring.

**Where the track stands, what is shipping and what to do next: [STATUS.md](STATUS.md).** This file
says how the pieces work.

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
scripts/overnight.py    chained 15-minute sessions: train, score on CPU, chain only a non-regression,
                        offer the best to promote.py at the end
scripts/promote.py      ships a checkpoint to the scenes only on a measured win over the incumbent
scripts/scoring.py      how each task is scored - the one reader of the scorers' JSON
scripts/preflight.py    is the task winnable, does the reward pay for the right thing, is one
                        exploding world harmless - run before any long session
scripts/batch_table.py  where the useful batch is, on this plant
scripts/bench.py        throughput and hardware headroom
scripts/config.ps1      settings for the PowerShell wrappers, with the measurement behind each one
scripts/train.ps1, eval.ps1, parity.ps1, watch.ps1   the wrappers
```

```powershell
python mujoco_rig\scripts\preflight.py --task perturb --seed <checkpoint>
python mujoco_rig\scripts\overnight.py --task perturb --sessions 8 --minutes 15 `
  --envs 4096 --steps 16 --seed <checkpoint>
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

### Environments

```
rl/env_config.py          the constants that DEFINE the tasks; every env, scorer and exporter reads them
rl/body_env.py            CPU base: plant, observation, reset, step, divergence guard, clock stagger
rl/body_env_warp.py       the same on the GPU
rl/perturb_env*.py        + the projectile and the balance / capture-step reward
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
`_before_physics`, `reward`): walk no longer inherits a disabled gun from perturb.

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

Both use **one 105-observation layout**, ending in three command slots that perturb leaves at zero
and walk fills. That is what lets a perturb brain seed a walk one without surgery on its first
layer.

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
```

These cover configuration isolation, landing rewards, complete resets, single-hit behavior,
contract timing, rejected promotions, best-checkpoint selection, and export rollback. Run both
parity scripts after environment changes; the walk test also asserts that a 0.2 s landing earns
its intended reward, so a shared CPU/GPU bug cannot pass merely because both sides agree.

To train a pure STAND (no projectiles) rather than perturb, disable the gun rather than changing the
task - it is the same environment:

```powershell
python mujoco_rig\rl\train.py --task perturb --backend warp --num_envs 4096 --steps 16 `
  --ball_every 1000000 1000000 --promote_at 2.0 --max_minutes 15 --run_name stand
```

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

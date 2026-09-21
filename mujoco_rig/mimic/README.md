# MimicKit motion-guided Stand experiment

This directory preserves the existing Godot/native MuJoCo bridge and trains an
experimental motion-guided Stand task through MimicKit's Newton/MuJoCo-Warp engine
with the same dummy. The raw-torque experiment uses 300 observations; the new
reference-relative target/PD experiment uses 362. Both have a separate, versioned
Godot replay path and are **not promoted to the production scenes**.

## Reproduce

Run from the project root, using Windows and Python 3.12:

```powershell
./mujoco_rig/mimic/setup.ps1 -BasePython D:/Programas/anaconda3/envs/env_isaaclab3/python.exe
$env:MIMICKIT_PATH = 'D:/Proyectos/Juegos/Tools/MimicKit'
./mujoco_rig/mimic/.venv/Scripts/python.exe -m unittest mujoco_rig.mimic.test_compat -v
./mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.validate `
  --mimickit $env:MIMICKIT_PATH --out logs/mimickit-compat/validation
```

The setup uses a real virtual environment (`include-system-site-packages = false`),
with pinned dependencies in `requirements.lock.txt`. The production Conda environment
is not modified. MimicKit is an external, clean checkout at revision `2ed1e6c`;
`runtime.activate` rejects a different revision or tracked local modifications.
No source file in that checkout is patched. CUDA Torch is downloaded from the
official PyTorch CUDA 12.8 wheel index. Setup itself downloads no motion/model archive.

## Preserved baseline

`baseline.py` captured 530 files, including the previous uncommitted work, shipped
ONNX policies/contracts, generated MJCF and accepted checkpoint
`logs/mujoco/2026-09-20_19-14-28_contact_v3_full5m/model_78.pt`.
The snapshot is `logs/mimickit-compat/baseline/`; its manifest contains file hashes,
Git revisions and the original status. It does not commit, stash or reset anything.
Re-running the snapshot requires a new destination so the original cannot be overwritten.

## Adapter boundary

- `rig.py` derives a contract from `dummy.xml` and shared production constants:
  30 normalized motor-torque actions, 60% authority, two control steps of delay,
  four physics steps per action, and the original **0.004167 s** timestep.
  The three excluded neck position actuators retain their original gains and zero
  targets. Passive wrists remain passive. This remains the default `--control torque`
  path; the opt-in target/PD path below deliberately changes the action contract.
- `newton_adapter.py` subclasses MimicKit's Newton engine. Its MJCF importer keeps
  ordered hinge joints (`convert_3d_hinge_to_ball_joints=False`) and native actuators
  (`ctrl_direct=True`). It retains passive springs/damping and the original floor,
  rather than adding a second floor. Source joint-limit parameters, timestep and
  multi-contact collision settings are preserved explicitly.
- `model_contract.py` rejects changed topology, physical properties, actuator
  mappings, collision eligibility or solver settings before accepting the adapter.
  Newton's renamed objects, collision bitmasks, explicit parent exclusions, and
  unused primitive-size components are representation differences, not necessarily
  different physics. Compare their physical effect rather than raw names/bitmasks.
- `kinematics.py` maps native hinge angles to batched body poses with MimicKit's
  xyzw quaternion convention. It rotates around the original offset pivots in
  the original hinge order. It is **not** a general motion retargeter or a claim that
  the stock MimicKit MJCF reader now accepts this rig.
- Reset clears the delayed commands, both Newton state buffers, external forces
  and MuJoCo solver state, then refreshes derived data and contact sensors.

`DummyNewtonEngine` owns one dummy per world and supports independent pose/velocity
resets, including command queues and solver state. Its `set_cmd(0, action)` expects normalized **30-channel**
actions, while its kinematic state contains all **39 hinge coordinates**. Do not
attach stock MimicKit environments blindly: their default action construction
assumes one action per kinematic degree of freedom. `StandTask` defines the matching
observation/action spaces and asynchronous episode resets.

## Validation on 2026-09-21

Environment: RTX 4080 SUPER; Python 3.12.13; Torch 2.11.0+cu128; Newton 1.2.1;
MuJoCo and MuJoCo-Warp 3.8.1; Warp 1.13.0. MimicKit documents testing Newton 1.0.0;
this local compatibility run establishes the narrower checks below on 1.2.1.

- Five regression tests pass: clipping/delay/neck control, queue reset, invalid
  commands, rejection of changed mechanics, and offset/hinge pose mapping.
- All integration gates pass over the rest pose plus 64 seeded varied poses.
  Maximum mapper position error: 1.054e-6 m; Newton pose error: 1.171e-6 m.
  Effective collision eligibility matches, with zero sampled contact-pair mismatches.
- Two GPU worlds execute different commands against independent native MuJoCo
  references. Over 0.50004 s, maximum root error is 1.714e-6 m, joint error is
  1.749e-5 rad and applied-control error is 2.385e-7 Nm in the recorded run.
- Reset replay starts within 1.449e-8 of the source pose and its first step agrees
  within 1.812e-9. The recorded full replay differs by at most 1.371e-5 in hinge
  coordinates. These float32 GPU trajectories are not promised bitwise deterministic.

`validate.py` writes `report.json` and `control_contract.json` and exits nonzero on
failure. The predeclared short-rollout gates are 0.02 m root / 0.1 rad joint error;
pose mapping is gated at 3e-6. Reset additionally requires initial-state and first-step
agreement within 1e-6. An initial whole-trajectory 1e-6 reset check was replaced by
these separated checks after diagnostics showed roundoff-scale first-step differences
and floating-point factorization variability. Both reset buffers are now cleared.
These are compatibility checks, not proof of stable standing, long-horizon parity,
robust recovery, or successful Godot inference of a new actor.

The final rerun is retained separately in `logs/mimickit-compat/final-validation/`:
all 77 gates pass, root/joint errors are 7.664e-6 m / 3.963e-5 rad, and reset replay
differs by up to 0.001007 rad after 0.5 seconds despite first-step agreement within
2.382e-9. This variability remains a measured limitation; longer rollouts and learned
policies require further evaluation rather than assuming deterministic transfer.

The raw importer audit can also be reproduced with:

```powershell
./mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.probe_newton `
  --out logs/mimickit-compat/newton
```

## Licensed reference and task

The source is Ian Mason's [100STYLE](https://www.ianxmason.com/100style/), whose author
permits creative/commercial use with attribution under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The adapted neutral idle
and its source hashes, transformations and validation are in `assets/`.
Keep [ATTRIBUTION.md](assets/ATTRIBUTION.md) with redistributed adaptations.
No noncommercial motion source or pretrained model was used.

`acquire_motion.py` fetches only the neutral-idle BVH and frame-cut metadata, checking
pinned hashes. `retarget_stand.py` maps a six-second interior excerpt into the original
hinges using collision-aware fixed-foot IK and temporal smoothing. It does not change
the rig. The 180-frame, 30 Hz reference is non-looping. Validation found maximum
foot error 0.355 mm, foot slide 0.122 mm/s, penetration 0.378 mm and joint speed
0.674 rad/s. Rendered poses were inspected for foot/limb alignment.

`StandTask` uses MimicKit's actual DeepMimic reward function and PPO implementation.
The upstream network, optimizer, loss, reward coefficients and exploration settings
are retained; output/test frequency is shortened for the probe. This is a project
torque-control adaptation, not an unchanged stock humanoid benchmark: 30 delayed
torque actions, 39 observed hinges, uncontrolled neck/wrist tracking weights masked,
native root/body reference velocities, and randomized reference start phase.
Three-second episodes terminate on low pelvis or non-foot ground contact. There
are no perturbations. The reference is reset state/target data, not a PD controller.

The 300 observations contain root state, all hinge positions/velocities, foot loads,
pending commands, three future target poses and reference phase. `contract.json`
records their order, units, torque/delay contract and model/reference hashes.
ONNX includes learned observation normalization and action clipping. The experimental
bundle runs through `MjMimicStandDriver`; existing 105/140-channel actors and loaders
use their original contracts. Production policies and scenes remain unchanged.

## Reproduce the Stand probe

```powershell
$python = './mujoco_rig/mimic/.venv/Scripts/python.exe'
$env:MIMICKIT_PATH = 'D:/Proyectos/Juegos/Tools/MimicKit'
& $python -m unittest mujoco_rig.mimic.test_compat mujoco_rig.mimic.test_motion -v
# The validated adapted clip is included; these two commands reproduce it from source.
& $python -m mujoco_rig.mimic.acquire_motion --out logs/mimickit-stand/source
& $python -m mujoco_rig.mimic.retarget_stand --source logs/mimickit-stand/source --out mujoco_rig/mimic/assets
& $python -m mujoco_rig.mimic.preflight_stand --mimickit $env:MIMICKIT_PATH --out logs/mimickit-stand/preflight.json
# Use a new output directory for every run. Stop if preflight fails.
& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH --out logs/mimickit-stand/ppo-smoke-new --envs 128 --iterations 32 --seconds 120
```

The time cap is checked between PPO iterations; setup, evaluation/export and a
currently running iteration are outside that cap. Runs start fresh and preserve
initial/final checkpoints, resolved agent YAML, native rollout, ONNX bundle and report.
The source/reference/plant are hash checked; the external MimicKit checkout stays clean.
Native evaluation every 64 iterations writes `learning_curve.json` and retains the
evaluated checkpoint. `gates.json`, written before training, requires eight of eight
three-second episodes and mean root tracking error at most 0.05 m. The report evaluates
the final actor; checkpoint retention does not silently promote a selected candidate.

## Godot replay

`Scenes/RL/Isaac3/MuJoCo/MimicStandReplay.tscn` uses the existing `MjBridge` and native
MuJoCo library. New bridge methods provide full pose/velocity resets and world-frame
ground forces. Generated contact-frame offsets preserve the ABI check; shipped policy
observations and control timing retain their existing paths. The experimental driver
validates the model, actor and reference hashes, hinge/action order and complete
observation layout. It matches float32 reference interpolation, two pending commands,
torque scaling, four physics steps and post-step contact refresh.

```powershell
dotnet build Physics4Fun.csproj --no-restore
& $python -m mujoco_rig.mimic.godot_replay --mimickit $env:MIMICKIT_PATH --bundle logs/mimickit-stand/ppo-15min-01/export
```

`godot_replay.py` launches the actual headless Godot scene and independently evaluates
the same ONNX actor in Python/native MuJoCo. It compares the first 16 control steps
per phase (all observation channels, actions and poses), plus all eight full episode
outcomes and durations. The gates are 0.003 observation error, 1e-4 action/pose error,
identical outcomes and survival differences below 0.1 seconds. This is a bounded
transfer check, not proof of long-horizon equivalence or recovery quality.

For live viewing, open `Scenes/RL/Isaac3/MuJoCo/MimicStand.tscn` with F6, or run
F5 (it is the configured main scene). The root node's Inspector selects the bundle,
MuJoCo library directory, trial duration and start phase. The scene defaults to the
verified `target-pd-30min-01/selected-000833/export` bundle and the isolated Mimic
Python environment's native library. It requires no environment variables and
ignores the batch runner's bundle/output variables.

The default five-second trial runs live policy-controlled physics, then holds the
result for inspection. A fall also ends the trial; neither case silently resets.
Use R or Restart to start again, and P or Pause to pause/resume. The existing camera
supports right-drag look, WASD/Q/E movement, Shift boost and wheel speed while looking.
Duration must be at least three seconds and fit within the non-looping reference
from the selected start phase. The viewer does not extrapolate indefinite standing
or add a perturbation controller. It never replaces the actors in `MujocoStand` or
`MujocoPerturb`. The referenced experimental bundle and virtual environment are
local ignored artifacts; another checkout must provide them or configure its own.

`MimicStandReplay.tscn` is batch-only and requires both `P4F_MIMIC_BUNDLE` and
`P4F_MIMIC_REPLAY_OUTPUT`, set automatically by `godot_replay.py`.

The project excludes `logs/**/*.cs` from compilation: preserved experiment snapshots
contain source copies and otherwise create duplicate-type build errors.

## First Stand result (2026-09-21)

Nine regression tests and all 12 task preflight checks pass. Preflight tests reference
reward, observation parity, delayed short trajectories and isolation of partial resets.
GPU/native short-rollout pose error was 4.36e-6 and reward error 1.67e-6.

`logs/mimickit-stand/ppo-smoke-01/report.json` records 32 PPO iterations, 131,072
samples, 128 worlds and 67.8 seconds of training/output evaluation. Eight deterministic
reference phases evaluated in native MuJoCo survive 0.448 seconds initially and
1.006 seconds after training; Newton gives 1.010 seconds. **Zero of eight survive
the full three seconds.** Native return rises from 18.26 to 36.96, while mean root
tracking error over the longer episodes rises from 0.079 m to 0.107 m. This verifies
the learning/export pipeline, not realistic standing or perturbation recovery.
ONNX versus Torch maximum action error is 5.78e-8 over recorded observations.

The smoke actor also passes actual Godot replay: observation/action/pose errors over
the first 16 steps are 9.05e-6 / 2.38e-6 / 2.04e-6, with identical eight-episode
outcomes and survival durations within 1.01e-7 seconds. The longer experiment below
uses the same plant, reference, PPO settings and reward.

## Fixed-setting 15-minute result (2026-09-21)

`logs/mimickit-stand/ppo-15min-01/` contains the final checkpoint, eight periodic
evaluation checkpoints, learning curve, export, native rollout and actual Godot
replay. This fresh run used the same seed and PPO YAML as the smoke run, with a
900-second cap and a high sample ceiling. It completed 486 iterations / 1,990,656
samples in 900.515 seconds, including intermediate evaluation/output overhead.

Native mean survival is 1.663 seconds (initial 0.448 s), versus 1.667 s in Newton.
The best final phase survives 2.134 s; zero of eight reach the three-second target.
Mean root tracking error is 0.117 m, above the predeclared 0.05 m gate. Periodic
evaluations show gradual improvement rather than convergence within this budget.
The pose strip `trained-poses.png` still shows forward collapse.

The final ONNX actor passes all five actual Godot replay gates: prefix maximum
observation/action/pose errors 1.713e-5 / 3.540e-6 / 1.611e-6, identical outcomes
and duration differences below 1.01e-7 s. Torch/ONNX action error is 1.79e-7.
The adapter passes 127 managed tests plus the opt-in native foundation regression;
nine Python reference/control regressions pass. These are transfer and software
checks; the standing behavior gate still fails.

Next, verify reference-controller feasibility with the current torque limits and
delay to distinguish a control/target problem from slow direct-torque learning.
Do not promote this actor or begin Perturb based on its higher return alone.

## Reference-controller diagnostic (2026-09-21)

Completed that feasibility investigation with bounded controllers and clearly
separated counterfactuals. No controller passed both tracking and ten-second hold
gates. This does not prove the rig/reference impossible to control.

`ReferencePD` uses the same normalized 30 motor torques, 60% authority and two-step
delay as training. One family scales proportional gains by motor torque capacity
(multipliers 1/2/4/8; derivative time constants 0.02/0.05/0.1 s). The other uses the
reference mass-matrix diagonal with frequencies 1/2/3 Hz, damping ratios 0.7/1.0
and optional static joint-bias/passive-force cancellation. The latter is not support
compensation. Calibration uses three phases; five further phases are held out from
gain selection. All eight are included in final reports.

The primary normalized controller survives 1.002 s on average, with 0/8 reaching
three seconds. Inertia scaling also fails and some candidates cause numerical
instability. Zero-delay counterfactuals preserve torque limits but explicitly change
the control contract; normalized PD then reaches 1.502 s, still 0/8. Zero-delay
observation queue channels are zero padding and are not a trainable policy interface.

`probe_target_pd.py` tests a different action structure: reference position/velocity
targets wait two control intervals, while torque feedback runs every native physics
step. The first two control intervals still apply zero policy torque. The same model,
passive mechanics and force limits are retained; no root forces are injected.
This alternative reaches 1.875 s without feedforward. With a quasi-static support
allocation it reaches 7/8 three-second trials and 0.0147 rad joint RMSE, but misses
the root/foot gates and all ten-second holds (mean survival 3.586 s).

`static_support.py` solves a bounded contact-force/motor-torque equilibrium problem
with conservative friction pyramids. The initial one-foot contact sets fail; the
second foot establishes native contact in 16.7 ms of unpowered stepping. Allowing
prospective sole contacts within 1 mm and passive joint relaxation, all 180 frames
admit controlled-joint/base equilibrium with at most 14.4% of the available motor
budget. The largest uncontrolled-joint residual is 0.795 Nm; exact passive angles
are not proved supportable. This analysis cannot establish dynamic balance.

```powershell
& $python -m unittest mujoco_rig.mimic.test_compat mujoco_rig.mimic.test_motion mujoco_rig.mimic.test_reference_control -v
& $python -m mujoco_rig.mimic.probe_reference_control --mimickit $env:MIMICKIT_PATH --out logs/mimickit-stand/reference-control-new
& $python -m mujoco_rig.mimic.probe_reference_control --mimickit $env:MIMICKIT_PATH --family inertia --out logs/mimickit-stand/reference-inertia-new
# Counterfactuals only: these are not deployable through the current actor contract.
& $python -m mujoco_rig.mimic.probe_reference_control --mimickit $env:MIMICKIT_PATH --zero-delay --out logs/mimickit-stand/reference-no-delay-new
& $python -m mujoco_rig.mimic.probe_target_pd --support-feedforward --out logs/mimickit-stand/reference-target-new
& $python -m mujoco_rig.mimic.static_support --out logs/mimickit-stand/static-support-new.json
```

Protocols are written before each candidate sweep. Canonical primary reports are
`reference-control-normalized-02` and `reference-control-inertia-02`; the original
normalized pilot used an error-first tie-break that favored early failed episodes.
Version 2 ranks successes, survival and then error, and reports numerical failures.
The alternative's results are in `reference-target-support-01` and explicitly say
`under_production_control_contract: false`. Its feedforward uses privileged model
information and is a diagnostic, not a learned or shipped controller.

**Next:** preserve the skeleton and native bridge, and test MimicKit's joint-target/PD
action structure with matching Godot actuation. The pinned Newton preset uses
`control_mode: pos`; our raw-torque adaptation made the learning problem different.
RL must still learn balance corrections around the reference. Do not infer that
position tracking alone now solves standing, or continue the raw-torque run merely
because nominal torque capacity appears sufficient.

## Reference-relative target/PD contract

`--control target_pd` selects `mimic_stand_target_pd_v1`. A fresh actor outputs 30
normalized residuals, each scaled by 0.25 rad and added to the reference hinge
angle sampled when the action is issued. Targets are clamped to the existing
joint limits. Reference velocities accompany the position targets. Both wait two
control intervals and then remain fixed for four physics steps. Initial invalid
queue slots apply zero motor torque.

At each physics step, feedback uses the latest hinge position and velocity:
`torque = limit * clip(4 * position_error + 0.08 * velocity_error, -1, 1)`.
`limit` is the original 60%-authority torque budget. These settings were fixed
before the five-minute run. There is no support-force feedforward, root assistance,
body-strength increase or reference modification. Balance must be learned.

The observation layout replaces 60 pending-action channels with 122 queued-target
channels: for each of two slots, 30 positions, 30 velocities and one validity flag,
oldest first. All other channels remain unchanged, giving 362 observations. The
contract records gains, residual scale, joint indices/limits and timing semantics.
Old torque checkpoints are incompatible with this action/observation contract.

`target_control.py` owns target transport and native feedback. A Warp kernel in
the Newton adapter runs inside each captured physics substep, including swapped
states. `MjMimicTargetControl` implements the matching Godot feedback through the
existing native bridge. It does not modify the generated rig or shipped actors.

```powershell
$python = './mujoco_rig/mimic/.venv/Scripts/python.exe'
$env:MIMICKIT_PATH = 'D:/Proyectos/Juegos/Tools/MimicKit'
& $python -m unittest mujoco_rig.mimic.test_compat mujoco_rig.mimic.test_motion mujoco_rig.mimic.test_reference_control mujoco_rig.mimic.test_target_control -v
& $python -m mujoco_rig.mimic.preflight_stand --mimickit $env:MIMICKIT_PATH --control target_pd --out logs/mimickit-stand/target-pd-preflight-new.json
& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH --control target_pd --out logs/mimickit-stand/target-pd-new --envs 128 --iterations 10000 --seconds 300
& $python -m mujoco_rig.mimic.godot_replay --mimickit $env:MIMICKIT_PATH --bundle logs/mimickit-stand/target-pd-new/export
```

The target preflight passes 14 checks, including partial-reset isolation and
native/Newton torque agreement (maximum 0.00034 Nm over the short replay). A
zero-residual ONNX bundle passes all five actual Godot replay checks before PPO
training. Its native mean survival is 1.786 s, 0/8 three-second successes; PD alone
does not solve balance. Evidence: `target-pd-preflight.json`,
`target-pd-zero-01/godot-parity.json`, under `logs/mimickit-stand/`.

The completed five-minute run is `target-pd-5min-02`: 177 iterations, 724,992 samples,
302.344 s. Mean native survival improves from 1.808 to 2.013 s, best trial 2.400 s;
Newton agrees on all eight durations. There are still 0/8 three-second successes,
and mean root tracking error is 0.162 m (gate 0.05 m). The trained export passes all
five Godot replay gates, with maximum prefix action error 3.35e-6 and maximum
episode-duration error 2e-7 s. The old raw-torque replay also continues to pass.

The first attempt, `target-pd-5min-01`, terminated with a native Windows access
violation after roughly 25 s. The identical-setting retry used `python -u -X
faulthandler` and completed; the crash has no established cause. Keep its failure
record and use fault tracing for the next bounded run. This is modest learning
progress, not a passed Stand gate. Next: a fixed 15-minute benchmark with these
settings before deciding whether further training is justified. Do not resume an
old raw-torque checkpoint into this contract or start Perturb yet.

## Thirty-minute run: passing checkpoint retained, final actor regressed

`logs/mimickit-stand/target-pd-30min-01` completed 1,800.063 s, 1,071 iterations
and 4,386,816 samples without a native crash. This was a fresh run with the same
seed, 128 worlds, reference, reward, PPO configuration and target/PD settings.
The command used `python -u -X faulthandler -m mujoco_rig.mimic.train_stand`,
`--control target_pd --envs 128 --iterations 10000 --seconds 1800`.

Seven consecutive saved checkpoints pass the original standing gates from about
minute 19 to minute 29. Checkpoint **`evaluation_000833.pt`**, at 24.15 minutes,
has the smallest mean root error among those passing checkpoints: **8/8 three-second
successes, 0.02968 m native root error**. Re-evaluation reproduces its saved metrics.
Its ONNX export is separate: **`target-pd-30min-01/selected-000833/export/`**.
Actual Godot replay passes all five parity gates and all eight episodes.

An additional five-second test, spanning eight starts in
`[0, reference_duration - 5]` on the same clip, gives **8/8 successes** with mean
root error **0.03407 m in native MuJoCo** and **0.03378 m in Newton**. The selection
uses the original evaluation phases; neither it nor the extended test demonstrates
generalization to new motions, perturbations or long-duration standing.

Late training regresses sharply around iteration 1032. The final actor reaches
0/8 three-second successes and 1.652 s mean survival, reproduced in both engines
and Godot. Its top-level `export/` is retained for diagnosis, **not selected for
replay**. The final five-second test also fails (0/8). The abrupt change coincides
with increased PPO clipping; its exact cause still needs investigation.

Validate the selected actor through the batch harness:

```powershell
& $python -m mujoco_rig.mimic.godot_replay --mimickit $env:MIMICKIT_PATH --bundle logs/mimickit-stand/target-pd-30min-01/selected-000833/export
```

For interactive viewing, run `MimicStand.tscn`; its Inspector already selects this bundle.

The selected checkpoint's provenance, native evaluations, and ONNX parity are in
`selected-000833/report.json`; GPU evaluations are in `newton-evaluation.json`,
and actual Godot results in `godot-parity.json`. Keep the selected checkpoint and
investigate late PPO update stability and automatic best-checkpoint retention
before extending training or beginning Perturb. No production actor was replaced.

## Guarded fine-tuning and checkpoint selection

`probe_policy_regression.py` separates actor changes from observation normalization
changes using the saved checkpoints. Comparing iteration 1025 with the failed final
checkpoint gives 8/8 with both earlier components, 6/8 with earlier weights and final
normalization, and 0/8 with final weights and either normalization. Both components
contribute; this does not isolate the first damaging minibatch.
Evidence: `logs/mimickit-stand/ppo-regression-ablation-01.json`.

`--stability guarded` is an explicit fine-tuning mode requiring a trained checkpoint
and its matching control/observation/reference/model contract. It freezes loaded
observation statistics, clips actor gradient norm to 1, and measures exact diagonal
Gaussian KL against the rollout's initial actor over every collected observation.
Each minibatch exceeding the cumulative mean KL bound (default 0.02) is rolled back,
including optimizer momentum, and ends actor optimization for that iteration.
This bounds updates on sampled states, not cumulative drift over the entire run
or performance on unseen states. Critic updates retain upstream PPO behavior.
The local extension supports one training process; the external checkout is unchanged.

```powershell
& $python -u -X faulthandler -m mujoco_rig.mimic.train_stand `
  --mimickit $env:MIMICKIT_PATH --control target_pd --stability guarded `
  --initialize-from logs/mimickit-stand/target-pd-30min-01/evaluation_000833.pt `
  --source-contract logs/mimickit-stand/target-pd-30min-01/selected-000833/export/contract.json `
  --out logs/mimickit-stand/guarded-finetune-new `
  --envs 128 --iterations 10000 --seconds 180 --evaluation-interval 16
```

Initialization loads weights and normalization only: optimizer state, rollout state,
random sequence and local counters start afresh. It is not an exact training resume.
Fresh training retains `--stability upstream`; guarded mode rejects an untrained
normalizer. Source hashes and settings are recorded in `protocol.json` before training.

Both modes now retain `best.pt` and its hashed contract/metrics in `best.json`, starting
with the initial policy. Selection ranks the standing gate, successes, survival,
then lower root error on the fixed eight native evaluation phases. `model.pt` remains
the final training state; **new runs export `best.pt` to `export/`**, even when the
last update regresses. Historical exports are unchanged. `report.json` distinguishes
final and selected evaluations and adds a five-second selected-policy check.
`updates.jsonl` records guarded accepted/rejected updates and KL measurements.
The interactive scene's accepted bundle is never changed automatically.

The first guarded validation (`guarded-finetune-01`) completed 181 s / 53 iterations /
217,088 samples from the accepted iteration 833 checkpoint. All periodic and final
three-second evaluations passed 8/8. Selected local iteration 33 improved native
root error from 0.02968 to 0.02833 m, passed 8/8 five-second trials (0.03324 m), and
passed all five actual Godot parity checks. Saved normalization tensors are exactly
unchanged; maximum accepted rollout KL was 0.01275 with no rejection at the normal
0.02 bound. A separate CPU integration check at a deliberately tiny 1e-12 bound
rejected the first actor step and verified exact restoration of weights and optimizer
state. This validates the mechanism, not long-run learning stability. The scene's
selected iteration 833 bundle remains unchanged pending longer validation.

### Fifteen-minute guarded validation

`logs/mimickit-stand/guarded-15min-01` initialized from `guarded-finetune-01/best.pt`
and its export contract, retaining the same settings with `--seconds 900`,
`--iterations 10000`, `--envs 128`, and `--evaluation-interval 16`. It completed
901.156 s, 273 iterations and 1,118,208 samples. All 18 periodic three-second
evaluations passed 8/8; the final policy also passed 8/8 in native MuJoCo and Newton,
and 8/8 on the additional five-second native evaluation.

Additional training did not improve the candidate: final three-second native root
error was 0.03406 m versus 0.02833 m initially. Automatic retention/export selected
the initial weights exactly. That export passes all five Godot parity gates and
8/8 five-second native trials (0.03324 m root error). Normalization remained exactly
fixed; maximum accepted KL was 0.01280, with no rejected updates. The final policy's
five-second error was 0.03620 m. See `extended-validation.json` for the additional
check and preservation invariants, and `report.json` for final versus selected results.

The scene's accepted policy remains unchanged. Prefer controlled push evaluation of
the retained candidate before a Perturb curriculum, rather than another unchanged
Stand training run. These trials use the same finite reference and evaluation phases.

## Controlled Perturb baseline (no new training)

Open `Scenes/RL/Isaac3/MuJoCo/MimicPerturb.tscn` and press F6. The scene uses the
retained `guarded-finetune-01/export` Stand actor. Four direction buttons restart
the same five-second trial with a different horizontal push. R repeats, P pauses;
the finished/fallen pose is held. Inspector properties control force (default 20 N),
onset (60 control steps), pulse length (6 steps), trial duration and reference phase.
The HUD shows pulse timing/impulse, displacement, foot travel and contact switching.
It is a live physics probe, not recorded animation. F5 still runs MimicStand.

Pushes act at the Chest center of mass, in MuJoCo world coordinates: +X forward,
-X backward, -Y left, +Y right. They are physical external forces, independent of
the motor-command delay. Each engine holds the force through all four physics
substeps, then clears it. Reset clears a world's force without clearing its peers.
No force schedule or future disturbance is exposed to the policy.

```powershell
& $python -m unittest mujoco_rig.mimic.test_perturb -v
& $python -u -X faulthandler -m mujoco_rig.mimic.benchmark_perturb `
  --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-stand/guarded-finetune-01/export `
  --out logs/mimickit-perturb/baseline-new
& $python -m mujoco_rig.mimic.godot_perturb --run logs/mimickit-perturb/baseline-new
```

`protocol.json` binds model/reference/actor hashes, units, coordinate frame, COM
application point, integer force windows and recovery thresholds. Each five-second
trial uses one pulse: 10/20/40 N for 0.100008 s, at step 60 or 120, in one of four
directions, from reference phase 0 or 0.8 s. Four matched no-push trials give 52
cases in total. `force-parity.json` gates short Newton/native transport, expiry and
reset isolation before the baseline runs. Native/Newton episodes include diagnostic
metrics and traces around force onset/end; `godot_perturb` runs the same protocol
through the actual Godot batch scene and validates timing, trajectories and metrics.

The final `baseline-02` native/Godot survival counts are 4/4 without pushes, 16/16
at 10 N, 16/16 at 20 N, and 13/16 at 40 N. Newton gave 14/16 at 40 N (13/16 in the
earlier run); the report names the near-failure-boundary disagreement. All nine
Godot/native checks pass. This baseline changes no rewards or policy weights.

Recovery requires surviving the trial and ending with a continuous post-pulse 0.5 s
window satisfying the thresholds in `perturb.RECOVERY`. Its latency includes that
confirmation window; losing stability clears an earlier recovery. The metric is
diagnostic: only 2/4 no-push trials pass, so it is not yet a useful promotion gate.
Likewise foot travel includes the reference motion; compare matched unforced trials
before interpreting it as excessive stepping. Establish reference-relative recovery
criteria before guarded Perturb fine-tuning; retain independent no-push checks.

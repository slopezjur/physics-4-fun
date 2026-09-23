# MimicKit motion-guided Stand and Perturb experiments

This directory preserves the existing Godot/native MuJoCo bridge for two
experimental tasks through MimicKit's Newton/MuJoCo-Warp engine: motion-guided
**Stand** and physical-ball **Perturb**. Stand uses the reference-relative
target/PD contract with 362 observations. Perturb reuses that Stand actor and
adds a simulated ball in `dummy_ball.xml`; its training and cross-engine impact
parity are still experimental. Both tasks have separate, versioned Godot replay
paths and are **not promoted to the production scenes**.

## Contact contract v2 (2026-09-23)

`-ContactMode support_v2` selects `mimic_stand_target_pd_v2`. It retains the
362-input layout but replaces the two foot-only loads with each foot plus its toe
normal load, in kN. GPU, native evaluation and Godot sample the last physics
solve in the control interval, before any post-integration forward refresh.
Reset supplies freshly solved contacts for the reset worlds. Ball launch preserves
the preceding contact snapshot; partial GPU resets preserve other worlds' snapshots.
Contract metadata fixes both sampling and the ordered support bodies.

Legacy contracts keep their original inputs and backend timing. Equal input width
does not make the weights or normalization compatible. Warm starts reject a
different contact contract. The configured default remains `legacy` because the
retained checkpoints use v1. A fresh v2 Stand experiment can be launched explicitly:

```powershell
./mujoco_rig/mimic/scripts/train.ps1 -Experiment stand -Stability upstream `
  -Control target_pd -ContactMode support_v2 -Minutes 15
```

The contact correction was validated before training. The subsequent v2 Stand pilot
is recorded below. Perturb requires a trained v2 Stand checkpoint and its matching
contract via `-InitFrom` and `-SourceContract`, with guarded target-PD.

The frozen-actor comparison uses identical weights and normalization on 96 fixed
impact cases plus eight quiet phases at three and five seconds. Legacy survives
72/96 impacts, timing-only 73/96, and complete v2 67/96; recovered counts are
17, 18 and 17. All retain 8/8 quiet trials at both durations. This is an explicit
input intervention on a v1 actor, not a trained-v2 comparison or evidence of improved
learning. Diagnostic exports are marked and rejected by training and viewer selection.
The existing Godot selection is unchanged.

Evidence is under `logs/mimickit-contact-v2`: `comparison-01`, `sensor-parity-03.json`,
`gpu-ball-01`, `godot-v2-01`, `godot-legacy-01` and `preflight-01.json`.
Both contracts pass the 96-case Godot/native replay and eight-phase standing parity.
The v2 GPU/native probe passes all eight projectile cases, including hit timing,
launch snapshot preservation and partial-reset contact isolation. Pitched heel,
flat and toe-support fixtures, including physical unloading to toe-off, differ
by at most 0.0061 N across backends (0.1 N gate).

```powershell
& $python -m mujoco_rig.mimic.compare_contact_contracts --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-training/<legacy-run>/export --out logs/mimickit-contact-v2/<new-comparison>
& $python -m mujoco_rig.mimic.foundation_sensor_parity --mimickit $env:MIMICKIT_PATH `
  --contact-mode support_v2 --out logs/mimickit-contact-v2/<new-sensor-report>.json
& $python -m mujoco_rig.mimic.preflight_stand --mimickit $env:MIMICKIT_PATH `
  --control target_pd --contact-mode support_v2 --out logs/mimickit-contact-v2/<new-preflight>.json
```

### First trained v2 Stand pilot

`logs/mimickit-training/stand-support-v2-15min-01` is a fresh upstream PPO run,
seed 210921, 2048 CUDA environments, target PD, contact v2, with native validation
every 16 updates. It completed 310 updates / 20,316,160 transitions in 901.797 seconds.
Checkpoint 240 was selected and exported. No legacy weights or normalizer were loaded.

The initial random actor survived 0/8 three-second phases (mean 1.81 seconds).
The selected actor survives 8/8 at both three and five seconds, with mean root
errors of 0.01846 / 0.02363 m. Native and Newton both pass 8/8; actual Godot replay
passes every parity check, with maximum action error 2.69e-6 and pose component
error 3.91e-7. Open `MimicStand.tscn` and press F6 to inspect the automatically
selected export. F5 still starts the existing Perturb scene and its separate policy.

`stand_diagnostics` compares eight matched quiet phases at three and five seconds,
excluding reset frames after termination. Against `guarded-finetune-01` at five
seconds, mean foot tracking RMSE falls from 8.91 to 1.96 mm and reference-relative
foot velocity RMS error from 0.03689 to 0.00364 m/s. Mean absolute action change
per control step falls from 0.01250 to 0.00422. Both actors survive all eight cases.
These are short standing measurements and movement proxies, not a visual realism
rating, long-horizon guarantee, controlled learning-efficiency comparison or impact
recovery result. Baseline reports are under `logs/mimickit-contact-v2/stand-baselines-01`;
the new actor's report is `stand-trained-v2-01` in the same directory.

```powershell
& $python -m mujoco_rig.mimic.stand_diagnostics --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-stand/guarded-finetune-01/export `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --out logs/mimickit-contact-v2/<new-quiet-comparison>

# Next experiment, not launched by the Stand pilot:
./mujoco_rig/mimic/scripts/train.ps1 -Experiment perturb -Minutes 15 `
  -ContactMode support_v2 -Control target_pd -Stability guarded `
  -InitFrom logs/mimickit-training/stand-support-v2-15min-01/best.pt `
  -SourceContract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json
```

Keep the existing quiet-standing retention and directional impact checks. Walk
remains blocked by reference support-moment failures; no walking or ball PPO run
was started during this Stand pilot.

### Perturb v2 pilot outcome

The subsequent `logs/mimickit-training/perturb-support-v2-15min-01` fine-tune ran
133 updates / 8,716,288 transitions in 900.188 seconds, with guarded PPO, reference
reward, 2048 CUDA worlds and evaluation interval 64. It used the trained v2 Stand
checkpoint above, with no physics, action-window or reward changes.

On the fixed 96-case suite, survival changed from 44 to 40 and settled recovery
from 25 to 36. The final candidate regressed in two directions and remained at
0/6 forward torso and 0/6 backward torso survivals. No trained checkpoint passed
directional retention; `best.pt` and the normal `export` therefore contain the
unchanged starting actor. Quiet standing remains 8/8 at three and five seconds,
although foot movement is greater than the Stand v2 parent's.

On 120 separately frozen random impacts (seed 23092317), the final candidate
improves from 47 to 57 survivals and 31 to 35 settled recoveries. The existing
legacy Perturb export scores 100 survivals and 21 recoveries on those same cases.
This is a limited single-seed improvement in some outcomes, not a general
replacement or a resolution of the forward/back torso weakness. Actual Godot
replay agrees with the final candidate's native survival and passes every parity gate.

The trainer auto-selected its retained export, then the review rolled the Perturb
viewer back to `perturb-recovery-v1-30min-02/export`. The trained Stand v2 selection
is unchanged. The final trained actor is saved in a separate, diagnostic-only
bundle for inspection, with no automatic promotion or training reuse:

```powershell
./mujoco_rig/mimic/scripts/watch.ps1 `
  -Run logs/mimickit-training/perturb-support-v2-15min-01/final-diagnostic -BallSpeed 2.5
```

Reports are in the run directory (`report.json`, `final-random-test.json`) and
`logs/mimickit-contact-v2/perturb-final-godot-01`, `perturb-final-quiet-01`, and
`perturb-pilot-random-baselines-01`. The random suite has now informed development;
reserve a new suite for the next independent test. Prioritize force-aware stepping
references and measurable dynamic tracking before a longer Perturb run.

## Reproduce

Run from the project root, using Windows and Python 3.12:

```powershell
./mujoco_rig/mimic/setup.ps1 -BasePython python
$env:MIMICKIT_PATH = (Resolve-Path 'tools/MimicKit').Path
./mujoco_rig/mimic/.venv/Scripts/python.exe -m unittest mujoco_rig.mimic.test_compat -v
./mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.validate `
  --mimickit $env:MIMICKIT_PATH --out logs/mimickit-compat/validation
```

The isolated wrapper scripts keep these commands reproducible without touching the
legacy MuJoCo training scripts:

`scripts/config.ps1` resolves relative paths from the project root, regardless of
the shell's working directory. Its portable defaults are `tools/MimicKit` and
`tools/Godot/Godot.exe` (use the .NET Godot distribution). Those installation
directories are ignored by Git. For existing installations elsewhere, create
`scripts/config.local.ps1`, also ignored by Git:

```powershell
$Mimic.MimicKit = 'path/to/MimicKit'
$Mimic.Godot = 'path/to/Godot.exe'
```

Use actual relative or absolute paths in that local file. `MIMICKIT_PATH` and
`P4F_GODOT_EXE` environment variables override the local settings. Direct Python
replay commands accept `--godot` or `P4F_GODOT_EXE`, falling back to
`tools/Godot/Godot.exe`; they do not load the PowerShell local configuration.
Use an absolute environment path when invoking Python outside the project root.
`setup.ps1` requires the selected `-BasePython` interpreter to be Python 3.12.

```powershell
# Show the resolved default Perturb command without starting training.
./mujoco_rig/mimic/scripts/train.ps1 -DryRun

# Start the default physical-ball experiment: guarded target-PD, warm-started from
# the configured retained checkpoint.
./mujoco_rig/mimic/scripts/train.ps1 -Experiment perturb -Minutes 15

# Compare the experimental post-contact recovery reward from the configured checkpoint.
./mujoco_rig/mimic/scripts/train.ps1 -Experiment perturb -Minutes 15 -BallReward recovery_v1

# Select a fresh upstream Stand run or another supported control contract.
./mujoco_rig/mimic/scripts/train.ps1 -Experiment stand -Stability upstream -Control torque
./mujoco_rig/mimic/scripts/train.ps1 -Experiment stand -Stability guarded -Control target_pd `
  -InitFrom logs/mimickit-stand/guarded-finetune-01/best.pt `
  -SourceContract logs/mimickit-stand/guarded-finetune-01/export/contract.json

# Watch the configured bundle, or infer the task from a completed run.
./mujoco_rig/mimic/scripts/watch.ps1 -Experiment perturb
./mujoco_rig/mimic/scripts/watch.ps1 -Run logs/mimickit-training/<run-name>

# Restore the previous F5/F6 selection, or select any completed run explicitly.
./mujoco_rig/mimic/scripts/select.ps1 -Previous -Experiment perturb
./mujoco_rig/mimic/scripts/select.ps1 -Run logs/mimickit-training/<run-name>
```

`train.ps1` exposes `-Control torque|target_pd` and `-Stability upstream|guarded`.
Guarded mode requires a checkpoint and matching source contract; upstream mode may
start fresh. The `perturb` preset is intentionally restricted by the Python runner
to guarded target-PD because its ball task depends on that observation/action
contract. `watch.ps1 -Run` validates `contract.json` and treats exports without the
new `experiment.json` manifest as Stand exports, which keeps historical bundles
usable. New ball exports contain the manifest and select `MimicPerturb` automatically.

Completed training automatically selects its `best.pt` export for the next F5/F6
launch of the matching scene. The selection and previous selection are kept in
Git-ignored `logs/mimickit-viewer/ball.json` or `stand.json`; exports are never copied
over older runs. Internal paths use `res://`, keeping machine paths out of tracked
scene files. Selection requires a completed report and valid export/checkpoint hashes,
but does not require the experimental recovery gate to pass: this default is for
visual review. Initializing the next training run still uses the configured checkpoint.

`train.ps1 -NoSelectViewer` (Python `--no-select-viewer`) leaves the selection alone.
Explicit `watch.ps1 -Run` / `--mimic-bundle` overrides it for one launch. Otherwise,
`watch.ps1` and the interactive scenes use the same selection. With no selection,
the scene uses its Inspector `BundleDirectory` and the wrapper uses `WatchBundle`.
Uncheck `UseLatestExport` in the Inspector to pin that scene to `BundleDirectory`.
Restart the scene after selecting or reverting an export; running trials do not hot-reload.

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

## Foundation audit

The [foundation audit](FOUNDATION_AUDIT.md) records independent mechanical,
actuator, foot-contact and fixed-pose support probes. It found omitted toe loads
in actor observations and different force-sampling times between GPU training
and native/Godot evaluation. These remain implementation follow-ups; the audit
does not change existing policy contracts or exports. The compiled rig passes
the tested structural and transmission checks, but dynamic stepping and general
protective behavior are not established.

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

## Forward/back stepping experiments

`assets/steps` contains separate, non-looping four-second forward/back walking
references adapted from the commercially usable CC BY 4.0 100STYLE neutral motions.
They are stepping priors, **not recordings of impact recovery**. Attribution and
changes are in `assets/ATTRIBUTION.md`; each reference has a hash-bound manifest.
The standing clip and the rig/torque/observation contracts remain unchanged.

`retarget_steps.py` uses moving foot targets, inferred stance intervals, stance locks,
collision-aware IK and trajectory-wide acceleration regularization. It does not reuse the standing
retargeter's fixed-foot assumption. It rejects excessive penetration, stance sliding,
foot-target error and joint speed. Inferred stance targets are grounded, and walking
root height is projected onto the lowest sole. Validation requires floor support on
every frame and each inferred stance sole within 1 mm of the floor. Avoiding floor
penetration alone is insufficient: an earlier candidate floated above the floor.
Independent per-frame fits previously introduced large pelvis acceleration spikes
despite passing these geometry checks. A sparse trajectory fit now penalizes root
and joint acceleration while retaining contact, collision and joint-limit constraints.
The geometric fit is followed by `force_retarget.py`: the trajectory optimizer
minimizes required root and controlled-joint force errors together. The inner
bounded least-squares solve combines unilateral foot/toe forces with the existing
delayed target-PD command intervals, retaining a 2% interval reserve. Contact rays
use inset sole corners and a friction pyramid inside the circular Coulomb cone.
Motor rows unaffected by contact have an exact clipped solution and are eliminated
from the coupled solve; this changes computation cost, not the feasible envelope.

`contact_projection.py` uses six leg hinges to solve each stance foot's fixed world
pose, while ankle yaw remains independently adjustable and the stance toe is flat.
Pelvis position and orientation are optimized; orientation corrections are bounded
to +/-0.15 radians per world rotation-vector axis. A fresh geometric reference starts
2 cm lower to seed the bent-knee IK branch; continuing an existing stance-projected
fit preserves its starting height. Independent validation runs every five objective
evaluations and stops fitting early only when every admission check passes.
Native joint limits remain enforced. Unreachable anchors remain visible as fitting
and validation errors. This replaces the earlier soft stance-only penalty and
posture-dependent lowest-sole height projection in the dynamic fit. Projection is
part of every objective evaluation; there is no final grounding edit. Source frames
with no inferred stance use the nearest soles (within 1 mm of the lowest one) as
a walking support hypothesis. Extended stance intervals retain the existing source
landing anchor, avoiding moving planted-foot targets. The fit report records these
frames and target adjustments; the resulting schedule is independently validated.
New references use central tangent velocities matching the dynamics diagnosis.
When an IK correction reaches a hinge limit, a bounded least-squares step
redistributes the correction among the other hinges instead of simply clipping it.
Optional `--adjust-landings` adds one XY offset per uninterrupted stance interval,
bounded to +/-2 cm per axis. The same offset applies through flat-foot and toe-only
support; it cannot move a planted anchor over time. Saved offsets are inherited
as absolute corrections on continuation. Foot-target validation still uses the
original targets, so placement adaptation cannot hide tracking error by rebasing
its target. This is an offline fitting option, not additional policy authority.
The fitter and independent actuator check share the runtime-derived PD bounds;
tests compare those bounds with the actual controller at both action extremes.
The sparse derivative pattern includes the delayed reference and its velocity
neighbors. Passive-joint equilibrium, initial queue fill and between-frame control
are still excluded. This offline CPU fit does not establish closed-loop trackability.
`--linear-solver exact` selects SciPy's dense trust-region solve using its public
finite-difference interface. It is intended for short transition diagnostics;
the sparse `lsmr` default is retained for whole clips. The solver and applicable
iteration budget are recorded in the fit report. A small fitting cost or a local
dense solution does not admit a full trajectory.
The independent gates decide whether
its output can be used; reaching the optimizer's iteration budget is not a pass.
New motion training requires `mimic_step_reference_v5` or explicit-phase v6, including aggregate COM
force/friction, root force/moment consistency and actuator feasibility within the
unchanged delayed +/-0.25-radian PD window. These are evaluated on the final
trajectory; failure rejects the retargeted candidate. Root support alone can pass
while the required joint torques cannot be delivered. Older manifests cannot
silently reenter training. Packaged v3 clips remain readable for archived replay
and diagnostics but are no longer accepted for new motion training. v4 diagnostic
fits are also excluded. Passing these optimistic necessary conditions still does
not establish closed-loop tracking or passive-joint equilibrium.
Failed candidates remain under `logs/`.

For forward heel-up transitions, `force_retarget --toe-off` writes an explicit
`mimic_step_reference_v6` candidate with `sole_contact` in Foot_L, Foot_R, Toe_L,
Toe_R order. Missing source support with heel-up pitch gets a candidate toe-only
phase plus two lead-in frames. Toe-only IK anchors the toe body and leaves its
hinge adjustable so the heel can rise. Forces and stance sliding are checked on
the declared supporting soles, and raised corners cannot supply projected support.
The inference is a hypothesis requiring validation, not a measured contact label.
V5 references keep their original interpretation; their assets and weights are
not relabeled. V6 requires both an explicit phase array and the extra sole-contact
gate. Continuations retain the saved phases.

`--self-clearance 0.005` adds a 5 mm constraint between collision-enabled body
pairs, without lifting the soles away from the floor. The configured clearance is
recorded, inherited on continuation and gated before admission. The fitting cost
aims 1 mm beyond the requested clearance; admission still uses the requested value.
Box pairs use a
conservative separating-axis distance bound: a native distance query was observed
to jump from 6.23 mm to zero under a 1e-8-radian perturbation, stalling numerical
derivatives. Other shapes use native distance; MuJoCo's actual penetration check
remains an independent gate. The stable box query is also used when no positive
self-clearance is requested. No collision masks or motor limits change in training.

`motion_tracking_audit` captures substep controls, solved contact forces and
accelerations alongside post-integration poses/velocities. It repeats the eight
native phases with zero residual, the existing Stand actor, and separate diagnostic
interventions for command-queue prefill and self-collision removal. Those
interventions use private native models only; they do not alter the controller
contract or any exported policy. Traces stop at each world's first failure.

`transition_probe` isolates a source frame interval (exclusive stop) and compares
its original and fitted geometry, support and delayed-PD checks. It verifies source
and rig hashes, preserves absolute landing offsets when cropping a stance, and
reports failures in source frame indices. Endpoints are free and command history
is local to the window: a successful interval still needs surrounding-trajectory
continuity and full-reference validation. Its output is explicitly diagnostic and
contains no training manifest or policy export. The standalone `motion_dynamics`
audit honors saved sole-contact phases; files without them retain legacy support.

`window_retarget` fits editable intervals inside the full reference. Each interval
gets a fixed context margin of `ceil(command_delay / reference_dt) + 2` frames on
both sides, clipped only at the real clip endpoints. The margin includes central
derivatives of delayed interpolated targets and the later commands affected by an
edit. Context poses are preserved exactly, including joints that stance IK would
otherwise reconstruct. Full-stance world headings and landing positions remain
fixed across cropped intervals; optional offsets saved by an earlier fit are
inherited without becoming editable again.

The default window solver is now constrained SLSQP. Foot-target error, declared
sole heights, collision clearance, stance sliding and delayed root/motor force
feasibility are explicit inequalities, not interchangeable objective penalties.
Geometry constraints apply to editable poses and affected stance transitions;
force constraints include dependent neighboring frames, even when their pose is
fixed. Root dynamics depend on the current pose and its two central-difference
neighbors; delayed motor bounds have a longer dependency interval. A later frame
with immutable, infeasible root dynamics is reported as deferred rather than
imposing an impossible local constraint. Supported later frames still constrain
delayed motors. All deferred failures remain subject to full-clip validation.
Constraints use the existing admission
limits with small numerical reserves; rig, action bounds and acceptance limits
are unchanged. The objective and constraint Jacobians share finite differences.
`--max-evaluations` bounds SLSQP iterations (each requires many residual calls).
Use `--solver least-squares` to reproduce the previous dense diagnostic solver.

`--solver direct-forces` is an experimental alternative. It exposes nonnegative
contact-ray forces as optimization variables and computes exact force Jacobians.
Root balance and delayed motor intervals are explicit algebraic constraints;
friction, force-point locations and actuator authority are unchanged. An initial
feasibility-restoration phase minimizes violated inequalities before SLSQP can
polish the pose-fitting objective. Its residual evaluations and subsequent SLSQP
iterations share `--max-evaluations`; reports separate the stages and count actual
combined residual calls. An incomplete restoration remains rejected. Both
constrained paths retain the best feasible incumbent if a later iterate fails;
returning that incumbent is not reported as optimizer convergence.

Each proposed interval is checked in the assembled full clip. It is rejected if
the optimizer leaves an explicit constraint unsatisfied, breaks a previously
passing gate, introduces a root-support or combined
root/PD failure at a previously feasible frame, or increases maximum penetration,
stance sliding or foot-target error beyond numerical tolerance. Rejected candidates
and their metrics remain available, but do not enter the working reference. The
saved full reference receives a fresh content hash and complete admission checks;
accepted local edits alone never imply a training-ready reference.

`--seed-probe DIRECTORY` initializes the first window from an isolated
`transition_probe` result. Source/rig hashes, timestep, dimensions and the source
frame interval must match; the probe must lie inside that editable window.
Original targets, bounds, fixed poses and full-stance anchors remain authoritative.
Probe landing offsets are recorded but not imported. New probes carry a diagnostic
content hash; older probes are accepted only as initialization with historical
hash verification explicitly marked false. Raw seed metrics and regression reasons
are recorded separately from the fitted candidate. This does not admit a probe as
a motion reference.

`transfer_calibration` creates a separate slow stepping-in-place diagnostic from
the vetted standing pose. Static solves share one set of coordinates across all
frames, preventing acceleration from substituting for support. The contact switch
uses a single-support-feasible unloaded pose before lifting. Quintic segments stop
at each knot; stance targets remain fixed. A single bounded pass can insert up to
four double-support clearance poses. The complete trajectory is independently
validated afterward; unresolved violations still fail.

```powershell
& $python -m mujoco_rig.mimic.transfer_calibration `
  --out logs/mimickit-steps/<new-calibration> --max-evaluations 100 --phase-seconds 4
```

The output contains `report.json`, hashed `diagnostic_poses.npz` and `knots.npz`,
static-pose evidence and attribution. It intentionally has no training manifest
and is always marked `admitted_for_training: false`. Static checks omit only the
inapplicable temporal foot-lift test; the final motion checks include it. The
32-second `transfer-calibration-02` passes every full-motion check with zero root
or delayed-PD failures. It is a procedural calibration with pauses, not recorded
human walking or evidence of closed-loop balance. Native tracking is the next gate.

The first force-aware fits (`force-retarget-04`) remove the backward clip's 64/118
root-support failures, but 22/117 post-delay frames still fail the current PD
envelope and neither diagnostic controller tracks it successfully. Forward still
fails support, stance sliding and foot tracking. Neither is approved for training.
The final v5 comparison and rejected manifests are under
`logs/mimickit-steps/force-retarget-review-01`. Joint contact/actuator fitting is now
implemented. `actuated-retarget-04/backward` passes every unchanged v5 admission
check: 0/118 root-support failures, 0/117 motor/PD failures and 39.949 mm maximum
foot-target error. This admits an isolated motion experiment; it does not establish
that the reference is trackable by a learned policy.
The three-minute fresh CUDA pilot (`actuated-backward-pilot-01`, 46 updates) remains
0/8 on native tracking and retains its initial checkpoint; its export is diagnostic,
not an improved controller. See `mujoco_rig/STATUS.md` for the measured outcomes.
`actuated-retarget-02/forward` still fails support and geometry. Its largest stance
slips occur where missing contact labels were filled with flat-foot support near
toe-off, reaching the hip-extension limit. That transition needs heel-lift/toe-only
support modeling; a lower optimization cost alone is not acceptance.
The later `constrained-retarget-01` backward candidate resolves the PD failure
but still misses its 5 mm self-clearance gate. Both the constrained forward fit
and `landing-retarget-01/forward` remain rejected. The landing trial changes the
original validation targets by exactly zero and shows no admission benefit at
the matched ten-evaluation checkpoint. See `STATUS.md` for measured failures;
these candidates must not be substituted into a training command.

```powershell
& $python -m mujoco_rig.mimic.acquire_motion --steps --out logs/mimickit-steps/source
& $python -m mujoco_rig.mimic.retarget_steps --source logs/mimickit-steps/source `
  --out logs/mimickit-steps/<new-retarget-directory>
# Reuse a hash-verified geometric fit without repeating source IK:
& $python -m mujoco_rig.mimic.force_retarget `
  --reference mujoco_rig/mimic/assets/steps/forward_reference.npz `
  --out logs/mimickit-steps/<new-force-fit-directory> --max-evaluations 80

# Explicit forward toe-off candidate; start from geometric source contact labels.
& $python -m mujoco_rig.mimic.force_retarget `
  --reference mujoco_rig/mimic/assets/steps/forward_reference.npz `
  --toe-off --out logs/mimickit-steps/<new-toe-fit-directory> --max-evaluations 10

# Diagnose a validated reference with the existing matching Stand bundle.
& $python -m mujoco_rig.mimic.motion_tracking_audit --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --reference logs/mimickit-steps/actuated-retarget-04/backward/step_reference.npz `
  --out logs/mimickit-steps/<new-tracking-audit>
# Isolate the first forward transfer without producing an admissible motion asset.
& $python -m mujoco_rig.mimic.transition_probe `
  --reference logs/mimickit-steps/constrained-retarget-01/forward/step_reference.npz `
  --start 20 --stop 35 --linear-solver exact --adjust-landings --max-evaluations 20 `
  --out logs/mimickit-steps/<new-transition-probe>
# Fit overlapping intervals with preserved surroundings and full-clip admission.
& $python -m mujoco_rig.mimic.window_retarget `
  --reference logs/mimickit-steps/constrained-retarget-01/forward/step_reference.npz `
  --window 22:31 --window 27:36 --max-evaluations 20 `
  --out logs/mimickit-steps/<new-window-fit>
& $python -m mujoco_rig.mimic.motion_dynamics `
  --reference mujoco_rig/mimic/assets/steps/forward_reference.npz `
  --reference mujoco_rig/mimic/assets/steps/backward_reference.npz `
  --out logs/mimickit-steps/<new-dynamics-report>.json
& $python -m mujoco_rig.mimic.probe_steps --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-training/perturb-recovery-v1-15min-01/export `
  --reference mujoco_rig/mimic/assets/steps/forward_reference.npz `
  --reference mujoco_rig/mimic/assets/steps/backward_reference.npz `
  --out logs/mimickit-steps/<new-probe-directory>
```

The probe explicitly substitutes the motion reference while requiring every other
source actor contract field to match. It compares zero-residual target PD with the
loaded standing actor, measures root/joint/foot tracking and reports clipping under
the old observation normalizer. It never selects an export or changes the controller.
The standing normalizer clips over half of future-motion input components on these
new clips, so the initial tracking pilots use fresh policies and upstream PPO with
adaptive normalization. Existing warm-start hash checks remain strict.

`train_stand --task motion` reuses the existing GPU trainer with separate native
motion-tracking gates: eight three-second phases, root error < 0.15 m, per-phase
mean foot error < 0.10 m, joint RMSE < 0.25 rad and at least 3 cm of lift range
for each foot in each phase. It requires target PD and
`--no-select-viewer`. Completed pilots export to their own run directory and cannot
replace the interactive Stand/Perturb selection. A four-second clip is not padded
into the standing task's five-second check; motion reports leave that field null.

```powershell
# Only after a new retargeted reference passes all v5 gates.
& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH `
  --task motion --reference logs/mimickit-steps/<validated-v5>/forward/step_reference.npz `
  --control target_pd --contact-mode support_v2 --stability upstream --envs 2048 --iterations 1000000 `
  --seconds 180 --evaluation-interval 16 --seed 230923 --no-select-viewer `
  --out logs/mimickit-steps/<new-forward-pilot>
```

Exports keep the Godot adapter's canonical `stand_reference.npz`/JSON filenames
regardless of the input motion name. Required reference/attribution files are checked
before training. If an older run completed training but failed during export, recover
its selected motion checkpoint into a new bundle without repeating PPO:

```powershell
& $python -m mujoco_rig.mimic.export_motion_checkpoint --mimickit $env:MIMICKIT_PATH `
  --run logs/mimickit-steps/<interrupted-run> --reference <matching-reference.npz> `
  --out logs/mimickit-steps/<interrupted-run>/<new-export-directory>
```

The recovery tool validates the original best-checkpoint hash and complete reference/
control contract, reevaluates it, and writes an export report. It does not invent a
completed training report or change viewer selection. References revised after a
pilot are incompatible with that checkpoint's contract; use the archived reference
matching the run, not a newer asset with the same filename.

Forward/back pilots are independent motion trackers. They do not yet form one
impact-conditioned recovery policy. Require measured dynamic tracking before
integrating them into Perturb; ordinary walking examples alone do not teach when
to step after a hit. The current Godot Perturb export remains separately selected.
The superseded contact-only references (`forward-pilot-02`, `backward-pilot-02`)
both scored 0/8 full tracking trials, with mean survival only 0.84/1.43 seconds.
The acceleration-regularized forward pilot (`forward-pilot-03`) survives 1/8,
with mean survival 1.22 seconds, but still fails root/foot tracking. The independent
root-wrench diagnostic flags 95/118 forward and 64/118 backward interior frames,
even with unlimited joint torque and optimistic foot support. These finite-difference
diagnostics concern exact reference following; they do not prove that RL cannot
learn a physically adapted trajectory. The next retargeting work must address
support forces and moments, before committing more time to Perturb training.

`motion_dynamics` now separates root support, motor authority and delayed PD target
reachability. On the 23 root-supported forward frames, 1 fails the torque envelope
and 6 fail either +/-0.25 or +/-0.5 rad targets. Of 54 backward frames, 2 fail the
torque envelope, 10 fail +/-0.25 and 7 fail +/-0.5. The standing clip has no root
support failures and all 177 evaluated frames after queue fill fit the original
window. Report: `logs/mimickit-contact-v2/dynamics-01.json`. This optimistic inverse
dynamics diagnostic omits passive-joint equilibrium and between-frame behavior.
It does not justify widening the action contract: the much larger root-support
defect remains. Force-aware trajectory optimization is still pending; the new gate
detects its absence rather than claiming to have generated working steps.
Falling, bracing and get-up tasks remain gated on measured balance and stepping.

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
$env:MIMICKIT_PATH = (Resolve-Path 'tools/MimicKit').Path
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

For live Stand viewing, open `Scenes/RL/Isaac3/MuJoCo/MimicStand.tscn` and press F6.
F6 runs the scene currently open in the editor. F5 runs the project's configured
main scene; in the current checkout that is `MimicPerturb.tscn`, not Stand. The root
node's Inspector selects the bundle,
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
$env:MIMICKIT_PATH = (Resolve-Path 'tools/MimicKit').Path
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

For interactive viewing, run `MimicStand.tscn` with F6; its Inspector already selects
this bundle.

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
Completed training now selects that export for interactive review; use `-NoSelectViewer`
to preserve the current selection. Selecting a viewer candidate is not recovery acceptance.

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

## Physical-ball Perturb scene

Open `Scenes/RL/Isaac3/MuJoCo/MimicPerturb.tscn` and press F6 (or press F5, as it is the
configured project main scene). The scene loads the latest selected ball export,
falling back to `logs/mimickit-perturb/guarded-perturb-04/export`, with `dummy_ball.xml`. Four
direction buttons restart the trial with ballistic shots from the cardinal directions,
and the `Random` button selects a randomized 360° azimuth aiming dynamically at a
randomly chosen limb from the 12-bone targeting pool (`Head`, `Chest`, `Spine`, `Pelvis`,
`UpperArm_L/R`, `Forearm_L/R`, `Thigh_L/R`, `Shin_L/R`). `B` or the launch button fires
immediately; `R` repeats, `P` pauses, and `AutoFire` enables continuous projectile
disturbance at configurable `FireInterval`. Inspector properties control ball speed,
target body, direction mode, and launch timing. The HUD reports active target body, launch
status, contact detection, and an approximate capture-point support margin. This
diagnostic does not alter policy targets; airborne feet do not count as support.

The scene also retains a force-pulse diagnostic mode for bridge checks. That mode is
separate from physical-ball training and must not be treated as ball-impact parity.

The projectile uses an 8 kg mass and 9 cm radius aligned with gameplay projectile dynamics.
The current viewer selection is stored in `logs/mimickit-viewer/ball.json`.
At 2.5 m/s the projectile has 20 kg m/s horizontal momentum; this is not a measurement
of impulse transferred to the dummy.

### Validation and checkpoint selection

The versioned `mimic_ball_validation_v2` selection suite contains 96 cases: all 12
bodies, four cardinal directions, and the configured minimum/maximum speeds. The two
speed groups start at reference phases 0 and 0.8 seconds; launch occurs at control
step 60. Each trial lasts 300 control steps (about five seconds). These cases are
validation data because they select checkpoints, not an independent held-out test.

Reports separate confirmed impacts, survived impacts, and settled recoveries. Settled
recovery requires a final uninterrupted 0.5-second interval after contact with pelvis
height >= 0.75 m, tilt <= 15 degrees, horizontal speed <= 0.2 m/s, angular speed <= 1
rad/s, and > 5 N support on each foot including toe contacts. A missed shot never
counts as recovered. No five-second result establishes long-term fall resistance.
Selection policy `directional_retention_v1` first rejects a candidate if confirmed
hits, survived hits or settled recoveries decrease in any cardinal direction or its
Chest/Spine/Pelvis subgroup, compared with the retained checkpoint. Eligible candidates
then rank by quiet standing, survived impacts, settled impacts and other survival/tracking
diagnostics. Case definitions and recovery criteria must match exactly; aggregate-only
evidence cannot select a ball checkpoint. These are subgroup count floors, not a guarantee
that every individual case is retained. A run can legitimately retain its initial actor.
`best.json` records the policy and subgroup floors; the learning curve and final report
record candidate decisions and regression reasons. Previous eight-case scores are not comparable.

### Diagnosing forward/back torso failures

`ball_diagnostics` replays the fixed suite with native MuJoCo and records torso cases
in the +X/-X push directions, followed by no-ball trials with identical phases. It
records target errors and normalized motor commands at every physics substep, plus
foot positions/loads, pelvis motion and termination causes. Summaries exclude frames
after termination/reset. Foot-origin lift or travel alone does not establish a recovery
step; compare it with the matched quiet trace and pelvis motion. The runner checks
actor/rig/reference contracts and never changes training or viewer selection.

```powershell
& $python -m mujoco_rig.mimic.ball_diagnostics --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-training/<starting-run>/export `
  --bundle logs/mimickit-training/<candidate-run>/export `
  --out logs/mimickit-training/<new-diagnostic-directory>
```

Defaults match the 1.0/2.5 m/s selection suite; `--speed-min` and `--speed-max` change
the two diagnostic speeds (phases remain 0 and 0.8 seconds respectively). Direction
labels describe projectile travel, not the side from which the ball approaches:
0 = forward (+X), 2 = backward (-X), 1 = right (+Y), 3 = left (-Y).

### Evaluation cadence

`EvaluationInterval = 64` schedules full native validation after every 64 completed
PPO updates. Initial validation and final-candidate validation remain mandatory;
there is no extra validation after update 1. Override the cadence per run with
`train.ps1 -EvaluationInterval 32` (positive multiples of eight).
The duplicate upstream GPU evaluation is disabled; its nominal eight episodes used
to require at least one completed episode per training world. Reports record separate
rollout, optimization and native-evaluation timings. `Envs = 2048` remains the working
default; ball runs require at least 96 worlds to score the complete suite. Initial
and final validation/export run outside the timed training loop.

Within one run, identical actor weights and normalization state reuse cached native
metrics and traces for the same physics contract, case suite and episode duration.
The cache holds at most three evaluations and pins the currently retained best.
This avoids reevaluating an unchanged final or selected checkpoint; if selected and
final actors match, their final Newton evaluation is also reused. Reports expose
`evaluation_cache` hit/miss counts and `selected_newton_reused_final` alongside timing.
The cache is not shared across runs and does not replace Godot validation. Longer
intervals reduce CPU interruptions but can miss transient improvements between checks.
Before promoting an export, require complete native validation and the actual Godot
replay checks below. Automatic viewer selection is for reviewing the export, not a
claim that those checks or the behavioral recovery gate have passed.

Build the Godot C# project, then validate the selected actor against the actual
headless scene (use a new output directory for each run):

```powershell
dotnet build Physics4Fun.csproj
. ./mujoco_rig/mimic/scripts/common.ps1
$settings = Read-MimicConfig 'mujoco_rig/mimic/scripts/config.ps1'
& (Require-MimicPath $settings.Python) -m mujoco_rig.mimic.godot_ball `
  --mimickit (Require-MimicPath $settings.MimicKit) --godot (Require-MimicPath $settings.Godot) `
  --bundle (Require-MimicPath $settings.WatchBundle) --speed-min 1 --speed-max 2.5 `
  --out logs/mimickit-perturb/ball-validation-new
```

The report checks observation/action/pose traces before and around impact, projectile
positions, hit timing, and complete-trial survival. Both implementations predict an
action before applying the scheduled ball launch, including immediate launch overlaps.
`ball-validation-v2-02` passes all parity checks for the retained checkpoint: 96 hits,
58 survivals, one settled recovery. That retained actor used standing imitation; this does
not demonstrate that the actor has learned realistic recovery steps.

### Experimental post-contact recovery objective

`-BallReward reference|recovery_v1` (Python `--ball-reward`) selects the objective.
The configured default remains `reference` until a recovery candidate is validated.
`recovery_v1` keeps the exact standing reward before a reported ball/character contact,
for missed shots, and throughout quiet episodes. Contact activation is sticky until
reset; launching a ball alone does not activate it. No contact flag or shot schedule
is added to actor inputs. The existing engine contact detectors are used: native
MuJoCo requires positive contact force; Newton records a ball/character contact pair.

After contact, 25% of the reward retains DeepMimic imitation in the root-relative
frame, allowing horizontal displacement and changes in heading. The remaining 75%
rewards reference pelvis height and upright orientation, with additional rewards
for low root linear/angular velocity and supported settling. Both feet being loaded
adds only a small settling bonus; lifting a foot to step is permitted. Parameters
and the `ball_recovery_v1` schema are recorded in each run's experiment manifest.
Failure termination and zero reward on failure remain unchanged.

This tests an identified objective conflict: the baseline assigns 65% of its reward
to standing joint poses and root-relative hand/foot positions. Recovery steps depart
from those targets. It does not establish that rewards alone are sufficient. The
actor still commands at most +/-0.25 radians of joint-target residual around the
standing reference; changing that limit would require a separate control contract
and another bridge validation.

Compare starting and candidate policies using the unchanged 96-case validation,
settled recoveries, survival, and independent quiet-standing checks. Reward returns
across different objectives are not comparable. The five-second evaluation remains
a short-window check, and its cases are used for checkpoint selection. Completed
training selects the export for the viewer, keeping the previous selection for rollback.

The first 15-minute experiment, `perturb-recovery-v1-15min-01`, improved native
impact survival from 58/96 to 70/96 and settling from 1/96 to 6/96, with quiet
standing preserved and actual Godot parity passing. Update 64 achieved 66 survivals
and 10 recoveries, so additional training did not improve both measures. Both
checkpoints remain available; the survival-first selector used at the time exported the
final policy. New runs apply directional retention before ranking. These are
single-seed validation results, not a promotion or a visual realism assessment.
See [STATUS.md](../STATUS.md) for timings and checks.

The older `benchmark_ball` module measures fixed native/Newton transport and reset
parity. It has no `--envs` scaling option and is not a PPO throughput benchmark.

### Independent randomized impact test

`ball_generalization` freezes a separate native-MuJoCo test protocol before inspecting
candidate results. The default is 120 cases, ten per target body, with continuous
random azimuths, speeds in the source experiment's range, start phases within the
finite reference, and launch steps from 60 through 120. It uses the existing
survival and final-settling measurements, reports confirmed hits separately from
misses, and never changes checkpoints or the viewer selection. Cases are drawn
from the same disturbance family as training; this does not test harder impacts,
multiple simultaneous balls, or long-horizon recovery.

```powershell
# Freeze once, before scoring the policies. Existing protocol files cannot be overwritten.
& $python -m mujoco_rig.mimic.ball_generalization --create-protocol `
  --bundle logs/mimickit-training/<starting-run>/export `
  --protocol logs/mimickit-training/<continuation-run>/random-test-protocol.json

# Compare the starting and selected exports on exactly the same new cases.
& $python -m mujoco_rig.mimic.ball_generalization --mimickit $env:MIMICKIT_PATH `
  --protocol logs/mimickit-training/<continuation-run>/random-test-protocol.json `
  --bundle logs/mimickit-training/<starting-run>/export `
  --bundle logs/mimickit-training/<continuation-run>/export `
  --out logs/mimickit-training/<continuation-run>/random-test
```

The report binds protocol, actor, model and reference hashes. Once results inform
another development decision, treat the suite as a regression suite and reserve a
new seed for the next independent test. Keep the original 96-case selection suite
unchanged when comparing training continuations.

## Controlled-force baseline (legacy diagnostic)

The controlled-force baseline remains useful for checking the bridge independently
of projectile flight. It applies a horizontal pulse at the Chest center of mass and
does not validate the physical-ball path. Four direction buttons restart the same
five-second trial with a different force direction. Inspector properties control
force (default 20 N), onset (60 control steps), pulse length (6 steps), trial
duration and reference phase. The HUD shows pulse timing/impulse, displacement,
foot travel and contact switching.

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

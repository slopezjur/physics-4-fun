# MimicKit Stand and reactive Perturb experiments

This directory preserves the existing Godot/native MuJoCo bridge for two
experimental tasks through MimicKit's Newton/MuJoCo-Warp engine: stationary
**Stand** and physical-ball **Perturb**. Historical motion-guided experiments
are retained below. Stand uses the reference-relative
target/PD contract with 362 observations. Perturb reuses that Stand actor and
adds a simulated ball in `dummy_ball.xml`; its training and cross-engine impact
parity are still experimental. `MimicPerturbArena.tscn` now previews the final
actor from `reactive-ball-arena-continue-5min-03/model.pt`, explicitly selected
for manual inspection. It recovers 26/32 fixed 1.5 m/s hits, and 15/32 at the
Arena's current 2 m/s speed, but fails directional and
quiet-settling retention; it is not an admitted replacement for the validated
2 kg seed. Physical 5 kg balls recur every configurable 2.5 seconds from random
angles and torso targets, with no time limit.

## Stationary reactive Perturb seed (2026-09-27)

Current recovery development is described in [RECOVERY_LEARNING_AUDIT.md](RECOVERY_LEARNING_AUDIT.md).
The current 2 m/s stepping experiment and its measured lift/landing gate are
documented in [REACTIVE_STEPPING.md](REACTIVE_STEPPING.md). `ReactiveGatedTraining`
freezes the standing network and trains a velocity-gated recovery branch.
`ReactiveCaptureTraining` adds a versioned 363rd input measuring instantaneous
balance relative to loaded feet, with verified native/GPU/Godot parity. Both
remain experimental; the extra input does not establish successful stepping.
`ReactiveTransferTraining` tests load-transfer guidance before liftoff; its first
ten-minute run regressed fixed recovery and was rejected. The strongest current
experimental seed passes eight 600-second quiet holds and 51/64 unseen 2 m/s
hits, with no learned recovery steps. It has not replaced the Arena actor.
Use `scripts/train-reactive.ps1` for the stationary, physical 5 kg recovery
curriculum. It reads the selected `-Preset` (default `ReactiveTraining`), `Envs`, and `EvaluationInterval` from
`scripts/config.ps1`; `-DryRun` prints the complete resolved command. The older
`scripts/train.ps1` remains the finite-reference experiment launcher. Training
does not automatically select an Arena policy.

Explicit previews use `export_reactive_ball_scene --experimental`; ordinary
exports still require complete admission evidence. To continue an already
exported final preview, use `train-reactive.ps1 -ResumeExperimentalFinal
-InitializeFrom <run>/model.pt`. Its provenance must match the preview bundle,
and the source must pass short quiet standing. The current run 03 final actor
fails that standing gate, so it needs standing retention repair before further
continuation through this mode.

```powershell
pwsh -File mujoco_rig/mimic/scripts/train-reactive.ps1 -Minutes 5
```

`reactive-ball-training-2kg-01/best.pt` starts from the validated stationary
Stand actor. Update 32 retains 8/8 quiet standing and improves 2 kg cardinal
physical-hit recovery from 30/32 to 32/32. A new 33-second suite passes 64/64
randomized torso hits, Newton confirms and recovers 8/8 forward hits, and an
uninterrupted ten-minute quiet suite passes 8/8.
The 5 kg stress score is 18/32, so the next stage trains heavier hits. No
fixed stepping leg or motion clock is used. See
[REACTIVE_PERTURB_SEED.md](REACTIVE_PERTURB_SEED.md) for the checkpoint, protocol,
validation, remaining limits and continuation command.

The earlier 15-minute wall-clock 5 kg continuation did not improve the 18/32
selection score and failed the long-standing gate. Its weights remain excluded
from the scene. At that stage, the admitted 2 kg ONNX actor drove the 5 kg stress
scene; this did not establish a validated 5 kg recovery policy.

A 3 kg curriculum used 902 seconds of synchronized training-update time across
two branches. The selected actor reached 29/32 cardinal hits and 64/64 unseen
12-second hits, but only 3/8 ten-minute quiet cases passed; it is not selected
in Godot. The update-time budget excludes native CPU evaluations but includes
CPU dispatch inside each training update.

## Stationary Stand foundation ready for reactive Perturb training (2026-09-27)

`stand-foundation-anchor-tight-01/best.pt` (selected update 16) passes **8/8
uninterrupted ten-minute native holds**, **64/64 fresh one-minute velocity
starts**, and both native/GPU 30-second gates. Final-minute foot motion is at
most 0.014 mm. Its stationary ONNX bundle is in the run's `export/` directory;
maximum PyTorch/ONNX action difference is 8.94e-8. Use `best.pt`, not the rejected
final `model.pt`.

Training fixes unstable normalization and late foot creep using stationary
references, a symmetric quiet-standing foot objective, and real late-state
starts. The physical rig, motor limits and delayed PD contract are unchanged.
The existing finite-reference Godot selection is unchanged. This checkpoint is
the seed for closed-loop, randomized physical-hit recovery with either foot;
the fixed left-step controller below is historical work, not the next recovery
objective. Preserve quiet standing and require unseen-hit validation before
Godot promotion. See [STANDING_FOUNDATION.md](STANDING_FOUNDATION.md) for exact
hashes, gates, training history and evidence.

## Native recovery training and scene adapter (2026-09-27)

The stateful-controller-plus-PPO-residual path has completed a 15-minute native
continuation. Its selected checkpoint passes 5/5 physical-ball and 13/13
velocity/force recovery cases at the 0.25 kg curriculum stage. The local
`MimicPerturbArena.tscn` now loads the same controller with a 5 kg projectile as a
stress test. Its cached support schedule reduces measured cold scene loading
from 10.68 s to 2.81 s without changing the 586-step +X trial. The scene
starts its step on confirmed ball contact; both feet
stay within 3 mm of their starting positions before the default hit. The
contact-triggered scene passes 2/4 cardinal recovery checks at 5 kg and 4/4
at the original 0.25 kg mass. The older time-driven 5 kg run passed 0/4.
The passing 0.25 kg scene bundle remains available. See
[RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) for the measured results, scene
bundles, and curriculum limits. The older diagnostic sections below preserve
the failed standalone-cloning history.

## Historical loaded step controller diagnostic (2026-09-27)

Latest: `step-conditional-arm-025-full-01` passes 13/13 fixed native recovery
outcomes with an observation-only scripted controller and bounded arm
clearance. Ten retain uninterrupted first touchdown; three pushed cases
briefly unload and then settle on both feet for over five seconds. A held-out
12 N negative-Y push settles too late. The 6,357-sample teacher dataset and
subsequent imitation/DAgger pilots do not produce a learned actor that passes
nominal loaded touchdown. Native capture contact labels are now synchronized
with stored poses, but the unmodified compliant rollout still fails authored
reference checks. See the latest section of `../STATUS.md` for measurements.
Perturb training, ball evaluation, and scene selection remain blocked.

The v7 diagnostic contact path represents heel-only support with independent
foot-box/toe-box bits (`support_contact`). The loader requires all physical
checks and matching support labels. `native-step-dynamics-audit-08` saves the
native rollout and checks it using both the old v6 projection and raw v7
support. It finds 40 left-heel-only frames, but the raw clip still misses the
1 mm contact-corner limit in 13 frames (maximum 2.77 mm) and fails joint
limits, penetration, sliding, and actuation. A 0.001 body-weight numerical
tolerance clears the four old coarse root-wrench flags; the authored motion
still has 16 failures up to 0.154 body weight. The 13.13 mm maximum
penetration is one right forearm–thigh overlap; floor penetration peaks near
3.39 mm. V7 fixes the format gap and the root diagnostic is now calibrated,
but the clip is not admitted.

`repair_native_step_reference.py` applies a bounded right-arm clearance
correction and clamps brief joint-limit overruns in the saved native motion.
`native-step-repair-arm-joints-03` then passes joint limits and penetration
(3.39 mm maximum), but still fails support-corner geometry, body-center
sliding, and delayed actuation. The largest constrained-force shortfall at
touchdown is 0.054 body weight. This repaired file has not been replayed by a
native controller or admitted as a step reference.

The corresponding live right-arm correction retains nominal touchdown and
cuts maximum native penetration to 3.52 mm
(`native-step-arm-clearance-live-01`), but the full disturbance gate falls to
4/13 (`step-observation-adaptive-arm-full-01`). It is diagnostic only; the
unchanged observation controller remains the 11/13 baseline.

The isolated early negative-X push passes with capture-point foot placement
and stronger landing COM feedback. Those gains break nominal survival, and
the early negative-Y push still unloads the left foot. A load-weighted
two-leg roll correction restores nominal survival under those gains but does
not fix negative-Y. The complete 13-case gate remains 11/13. Training and
`MimicPerturbArena.tscn` selection remain blocked.

The bounded action search in `optimize_landing_actions.py` produces a strict
native first-touchdown pass on `short-loaded-step-36`: first loaded contact at
2.550 s continues for 5.617 s, with an 8.151 s survival and a 22.4 cm final
left-foot displacement. `step-teacher-robustness-03` passes 8/13 fixed velocity
and Chest-push cases. This supersedes the earlier conclusion that every native
controller loses the first loaded contact.

`observation_step_teacher.py` reconstructs its required state from the existing
362-value support-v2 observation. It matches all 489 saved native teacher
actions to at most 1.46e-6 and independently reproduces the nominal native
landing and 8/13 disturbed cases (`step-observation-robustness-01`). To replay
the hash-bound observation controller:

```powershell
& $python -m mujoco_rig.mimic.observation_step_teacher `
  --mimickit $env:MIMICKIT_PATH `
  --source logs/mimickit-steps/short-loaded-step-36 `
  --search logs/mimickit-steps/landing-action-forward-com-01 `
  --contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --out logs/mimickit-steps/<new-observation-probe>
```

The successful native rollout still fails the unchanged physical reference
audit: joint limits, 13.1 mm penetration, 0.203 m/s stance sliding, sole
contact, four root-wrench frames, and actuation. A touchdown projection and
local constrained fit also fail. `step-distill-native-02` and
`step-dagger-native-01` train against the passing teacher but yield no native
step policy; exact action playback passes while a 2% student blend falls.
The joint-corrected observation playback passes only 1/13 disturbances.
These artifacts remain diagnostic; the short motion and independently stable
expert gates still precede Perturb training, ball evaluation, and Godot scene
selection. See `mujoco_rig/STATUS.md` for the full measured sequence.

The physics-step audit `native-step-dynamics-audit-02` shows that the four
coarse root-wrench failures are not missing native support: both feet carry
load there and MuJoCo's solved root residual is below 5.2e-16 body weight.
Instead, the left foot lands on its heel for 40 captured frames. The v6 motion
contract cannot represent heel-only support. Native captures now preserve raw
`measured_sole_contact` beside the projected v6 contact schedule. The other
physical checks above still fail, so this audit does not admit the clip.

An observation-state ankle correction passes 11/13 native cases
(`step-observation-adaptive-ankle-full-01`), including nominal, every initial
velocity offset, and every 2.75 s push. The two early 1.75 s negative-X/Y
pushes fail first-touchdown retention and later survival. A fixed ankle offset,
world-space latch, sole leveling, stronger COM gains, larger COM bias, and
Stand-actor blend do not fix them. A capture-point swing-target shift extends
the negative-X first loaded contact to 2.684 s but still ends in a 6.00 s fall;
negative-Y remains below 0.5 s (`step-observation-capture-3`).

The outcome-residual PPO pilot `step-outcome-ppo-01` collected 90,112 samples
under randomized early/late Chest pushes. All 11 trained evaluations fail the
native step. Its best checkpoint is the unchanged initial actor. The corrected
report includes `best_iteration: 0`, `learned_checkpoint_passed: false`, and
`passed: false`; future pilots require a trained best checkpoint for `passed`.
The remaining work is a contact-aware heel-strike motion representation and
state-conditioned touchdown/settling control. No step actor was admitted for
Perturb training or loaded into `MimicPerturbArena.tscn`.

## Native swing and landing probe (2026-09-26)

`recovery_step_motion.py` now builds a separate 5.5 s diagnostic motion from
the calibrated forward-step poses. It schedules weight transfer, a 0.8 s swing,
touchdown at 2.15 s, two-foot recovery through 3.0 s, and a final hold. The
reference places the left foot 15 cm forward and has 6.0 cm of swing lift. Its
native MuJoCo probe does **not** pass: the baseline controller retains a loaded
landing for only 0.13 s and falls at 3.90 s. Zeroing the swing-leg residual or
all residuals after swing also fails. Offline validation finds 8.9 mm maximum
penetration and 20 frames without a feasible root support wrench. The motion is
marked diagnostic only; no step expert was trained and the Godot viewer selection
was not changed. See `logs/mimickit-steps/short-loaded-step-{01,02,03,04,05}`;
`05` is the final baseline rerun after the report and provenance updates.

Follow-up diagnostics `06`–`12` revised the root path, stride, arm clearance,
timing, and post-contact balance without training. The arm adjustment removes
the isolated forearm/pelvis penetration. The 10 cm stride and anchored-root
variants do not improve landing retention. A 1.2 s swing reduces offline
root-wrench failures from 20 to 13 frames but still fails delayed-target
feasibility and falls at 4.47 s. Extending two-foot recovery also fails. A
native-only correction through the planted ankle extends continuous loaded
contact to 0.30 s; adding coronal corrections is worse. The constrained local
support fit rejects its candidate because it adds root-support failures and
increases foot-target error; see `short-loaded-step-window-01`.

No candidate passes the required two seconds of upright, loaded two-foot
support. Contact labels in the reference are only a schedule; the native
measured-load gate decides whether touchdown actually succeeded. These
references remain diagnostic and must not seed a step expert or replace the
selected Godot Perturb bundle.

The bounded landing-gain checks `13`–`15` use the slow timing and planted-ankle
diagnostic. The best (`13`, COM gains 10/2) holds loaded contact for 0.37 s and
survives 5.13 s, then retracts the foot and falls. More gain does not pass.
`short-loaded-step-window-01/window-00.json` records the rejected constrained
fit; its final wrapper stopped during attribution because this diagnostic
reference lacks the source-frame metadata expected by the retargeting tool.
The next reference attempt needs a new dynamically supported root/contact
trajectory through touchdown, followed by a native two-foot settle gate;
another gain sweep around this clip does not establish that trajectory.

The follow-up runs `16`–`28` test that trajectory question directly. A fitted
static landing pose with the left foot 15 cm forward can support its own weight,
but joining it to the swing does not produce a supported transition. Adding
support feedforward, quieting the upper body, and shifting the pelvis forward
also fail. Smoothing only the root path (`23`) keeps the native dummy upright
through the 6.15 s probe. At 2.52 s the swinging foot is 23 cm forward and
briefly loaded; it then unloads and returns to 7 cm forward by the final hold.
A bounded landing-foot correction (`25`) also survives the full probe but ends
9.4 cm forward, with only 0.28 s of loaded contact at the required placement.
Increasing that correction (`27`) ends at 8.5 cm. A fully smoothed joint/root
interpolation (`28`) breaks sole contact and falls at 4.08 s. Even the surviving
root-smoothed references have 17 frames without a feasible root support wrench
and fail delayed-actuation feasibility. These runs do not establish a retained
landing or an admissible reference. The remaining work is to synthesize a
contact-constrained dynamic transition, then recheck its measured landing and
settle; step-expert training remains gated on that result.

Run `30` preserves the native rollout separately from the authored motion. It
shows a later, stable two-foot plant around 3.8 s with the left foot 9.4 cm
forward through the final hold. This is not a valid shortcut to a reference:
the captured trajectory has 0.33 m/s stance sliding, 8.5 mm penetration,
joint-limit violations, and 17 infeasible root-wrench frames in the transition.
The saved `native_rollout.npz` is diagnostic evidence for a future constrained
fit, not an admitted training clip. Run `29` stopped before reporting because
its first capture used an incompatible sole-contact layout; `30` is the
corrected capture and validation.

`fit_recovery_transition.py` turns that rollout into a single swing ending at
3.85 s, fixes both stance anchors, clips the few joint-limit excursions, and
fits a local window with contact geometry and delayed-PD equilibrium
constraints. The projected seed still has 16 infeasible root-wrench frames.
The narrow `recovery-transition-fit-{01,02}` windows do not satisfy the fit
constraints; explicit-force restoration leaves the root failures unchanged.
The expanded `recovery-transition-fit-03` window includes the previously
immutable unsupported frame and reduces optimizer constraint violation, but
still fails the constraints after 12 iterations and has 19 root-wrench failures
when placed back in the full motion. Its candidate is rejected. A 10 cm motion
with swing delayed to 3.85 s (`short-loaded-step-31`) also falls before a loaded
landing. The local fit and timing change therefore do not yield a reference
for a step expert. A full contact-aware trajectory still needs the one-leg
support dynamics and landing to pass the reference audit.

Runs `32`–`36` initially appeared to pass a native landing gate under a
privileged COM/foot-position teacher. The gate measured the longest later
loaded interval, but did not require the **first touchdown to remain loaded**.
The corrected v2 gate in `short-loaded-step-38` exposes the error: the left
foot first loads at 2.55 s, unloads almost immediately, and plants again around
3.7 s. That later plant lasts 4.52 s and the dummy survives 8.15 s, but the
planned touchdown is not retained. The authored reference also fails offline
root-wrench, actuation, and foot-lift checks. No motion is admitted for a step
expert. The earlier v1 native `passed` flags are superseded by the v2 result.

Before this gate correction, `distill_recovery_step.py` and
`dagger_recovery_step.py` ran bounded diagnostic policy pilots on the teacher's
action trace. Distillation regressed quiet standing; DAgger retained quiet
standing but its best independent step survived only 5.50 s. Increasing fit
epochs (`step-dagger-pilot-02`), removing hidden hip-roll integral state from
the teacher (`short-loaded-step-36`, `step-dagger-pilot-03`), and emphasizing
landing actions (`step-dagger-pilot-04`) did not produce an independent step.
Both pilot scripts now reject a teacher without the v2 first-touchdown gate.
No actor was selected or exported and no ball hit was tested.

`probe_recovery_playback.py` shows the old teacher's exact recorded actions
reproduce the late plant (`step-playback-sensitivity-01`). A 2% blend of the
DAgger student action makes it fall at 5.88 s. A joint-position correction
rescues that one blend, while 5% and 100% blends still fail. This is a nominal
replay sensitivity result, not a completed recovery step.

`probe_recovery_robustness.py` uses four +/-0.05 m/s initial root velocities
and eight 10 N, 0.1 s Chest COM pushes during swing or after the scheduled
touchdown. The old late-plant gate passes 8/13 cases (`step-teacher-robustness-01`);
the corrected first-touchdown gate passes 0/13 (`step-teacher-robustness-02`).
The nominal motor demand stays below 0.73 of the limit, and three cases that
remain upright but lose placement have no landing motor saturation. Faster COM
targets, faster recovery timing, stronger foot-anchor blending, and a measured
touchdown latch also fail to retain the first contact. The next motion must
coordinate COM transfer and foot descent into one supported touchdown before
another step-policy or ball experiment.

The 2026-09-27 contact-aware continuation did not produce that motion.
`fit_recovery_transition.py --seed-only` projects the saved native rollout
onto one contact schedule; `evaluate_recovery_candidate.py` checks any saved
candidate against the physical audit and the strict native touchdown gate.
`recovery-transition-seed-00` retains 17.1 mm penetration and 8 infeasible
root-wrench frames, then falls at 3.87 s before loaded touchdown. Leveling the
swing sole and shifting the pelvis in `seed-01` reduces penetration to 6.7 mm
but creates 18 root-wrench failures and a fall at 3.58 s. Both native-check
reports are in their seed directories. The broad `fit-04` constrained attempt
diverged and was stopped before emitting a candidate. The 10 cm and 5 cm short
motions (`short-loaded-step-{39,40}`) retain the first contact for only 0.017 s
and 0.20 s, respectively, before falling. The long projected seed swing lasts
2.35 s, beyond the measured 0.65–0.78 s single-support window.
`short-loaded-step-41` advances 4 cm of lateral pelvis transfer to touchdown;
first contact still unloads after 0.033 s, the reference has 15 root-wrench
failures and 19.2 mm penetration, and the dummy falls at 6.03 s. A new
motion must solve the support and touchdown phases together within a short
swing; these candidates remain excluded from step-expert training and scene
selection.

An authored-motion fit over the actual 1.25–2.65 s swing and landing
(`authored-transition-fit-01`) leaves 16 root-wrench failures, violated
equilibrium constraints, and a 0.033 s first loaded contact in native replay.
World-space foot latching at contact (`short-loaded-step-42`) holds for only
0.017 s. The explicit braked swing target in `short-loaded-step-43` raises
that interval to 0.367 s, then falls at 5.20 s; actual forward foot travel
overshoots the 15 cm target to roughly 26.5 cm. Its foot correction saturates
on 57% of samples while the pelvis advances at about 0.47 m/s. A 4 cm rearward
pelvis bias (`short-loaded-step-44`) breaks stance support and cuts first
contact to 0.050 s. The required next motion must jointly control pelvis
velocity, stance support, swing-foot braking, and load transfer. The native
gate still rejects all candidates.

`synthesize_recovery_step.py` independently fits six static support poses and
connects them into a 5.5 s step. All six key poses have feasible static support,
including the loaded 15 cm landing. The stopped candidate
(`static-keyframe-step-01`) has 12 root-wrench failures, 7.4 mm penetration,
and falls natively at 3.08 s before touchdown. Its bounded contact fit
(`static-keyframe-fit-01`) remains infeasible and loses ground-contact geometry.
Cubic pose interpolation (`step-02`) creates 67 root failures; pelvis-only
shape-preserving interpolation (`step-03`) leaves 10 and falls at 3.27 s.
The `step-03/native-anchor-1` full-strength foot tracker saturates on 91% of
samples and still never lands. Static support therefore does not certify the
transition or the existing target-PD action window. Training remains gated.

```powershell
& mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.recovery_step_motion `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --out logs/mimickit-steps/<new-short-step-probe>
```

`probe_support_step.py` uses the measured right-foot support window to attempt
a 0.55 s, 10 cm left-foot step. The diagnostic controller lifts and moves the
foot and briefly makes loaded contact beyond 5 cm, but cannot keep it there:
contact lasts at most 0.28 s from source 7 s and 0.017 s from source 3 s.
The foot retracts and the dummy falls. The existing .25 rad action range
saturates on 88–92% of IK samples. A separate .5 rad diagnostic cuts
saturation to 65% from the later start but still loses contact within 0.17 s.
An adaptive touchdown reference does not fix retention either. See
`logs/mimickit-steps/support-window-step-{7s,3s}-02/report.json`,
`support-window-step-wide-7s-02/report.json`, and
`support-window-step-latched-7s-01/report.json`.
These contacts are not completed steps. The next motion needs an explicit
loaded landing and a post-landing balance phase before policy training.

Run the entry probe first to identify a supported interval, then run the
landing probe from the same source start. Each `--out` path must be new:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
$reference = "logs/mimickit-steps/placed-step-forward-09"
& $python -m mujoco_rig.mimic.probe_single_support_entry `
  --mimickit $env:MIMICKIT_PATH --reference $reference `
  --start-seconds 7 --controller balanced_axes `
  --out logs/mimickit-steps/single-support-entry-new
& $python -m mujoco_rig.mimic.probe_support_step `
  --mimickit $env:MIMICKIT_PATH --reference $reference `
  --start-seconds 7 --out logs/mimickit-steps/support-window-step-new
```

The optional `--action-radians 0.5` and `--latch-landing-reference` flags
change only this native diagnostic; neither changes the policy contract.

## Native dynamic support window (2026-09-26)

`probe_single_support_entry.py --controller balanced_axes` uses fixed, native
COM feedback and scheduled equilibrium torques to test weight transfer. From
source 7 s it reaches the right-leg target, then falls at 5.23 s; from source
3 s it falls at 8.25 s after reaching the target. Neither produces a retained
one-leg hold. The version 2 probe checks foot-plus-toe loads and uprightness at
every control step. It finds a 0.78 s clean right-leg window from the later
start and a 0.65 s window from the earlier start. Reports:
`logs/mimickit-steps/single-support-entry-{7s,3s}-balanced-02/report.json`.
These windows can seed a new short swing/landing experiment. They do not pass
the transfer or prove that the dummy can place a foot. The old motion's left
foot remains unloaded for almost eight seconds, so a new short motion is
needed for a reactive step. This controller reads native COM and remains
diagnostic only.

## Single-leg support PPO pilot (2026-09-25)

`train_single_support.py` initializes from the retained Stand actor and
trains against the physically valid, static right-leg-only pose at source
8 s. It keeps the existing observation contract and frozen normalization,
and selects only checkpoints that pass native quiet standing and a three-second
one-leg hold. The three-minute GPU pilot reached 72 iterations and 2,359,296
transitions. Quiet standing passed in the selected checkpoint, but the hold
never improved beyond 0.93 s. A one-minute update audit recorded 335 accepted
and 17 rejected actor steps, with the same hold outcome. See
`logs/mimickit-steps/single-support-train-3min-01/report.json` and
`logs/mimickit-steps/single-support-update-audit-1min-01/report.json`.
These are diagnostic checkpoints only; no export or Godot selection changed.
The present result does not justify another identical PPO run. A separate
closed-loop weight-transfer entry and measured one-leg retention are needed
before adding swing/landing or further Perturb training.

## Longer single-leg support and transfer probe (2026-09-26)

The original 3 s hold result was too weak: the 9 s pose stabilizer falls at
3.80 s in a 10 s continuation and briefly puts 27.4 N on the left foot after
startup. `probe_single_support.py` now records the maximum swing-foot load and
requires it to stay below 5 N after the first 0.5 s. Its extended result fails.
See `logs/mimickit-steps/single-support-holds-07/report.json`.

`probe_single_support_entry.py` tests the same controller while following the
source weight-transfer motion, with passive joint equilibrium and scheduled
static support torques. A 7-to-9 s source transition falls 2.72 s after
starting; a 3-to-9 s transition falls at 4.25 s, before reaching the target.
See `logs/mimickit-steps/single-support-entry-{7s,3s}-01/report.json`.
The teacher reads native whole-body COM, so it remains a diagnostic. No actor
was trained or selected from these runs. The support motion still needs a
longer continuous hold and transfer before step training. The diagnostic
`train_single_support.py` now uses the settled pose consistently and applies
the version 2 ten-second native selection gate; no new training was run.
That gate requires the swing foot to remain below 5 N after the first 0.5 s
throughout the hold, as well as a final continuous 0.5 s settled interval.

## Original short single-leg hold result (2026-09-26)

`probe_single_support.py` settles passive joints (head/wrists) to $< 10^{-9}$ Nm
residuals and evaluates five controllers across static poses at 7, 8, and 9 s.
At 9 s (COM centered over the right foot polygon), `CoronalSagittalStabilizer`
combines static equilibrium feedforward with stance ankle sagittal pitch PD,
stance hip roll PID ($k_p=5.0, k_i=3.0, d_v=0.5, d_\omega=0.05$), and opposing
ankle roll coupling. It survives the full 3.00 s and passes the original
final-state single-support gate
with a 0.52 s continuous settled final window ($\ge 0.50$ s required). See
`logs/mimickit-steps/single-support-holds-04/report.json`.

## Single-leg support gate (2026-09-25)

`probe_single_support.py` rechecks three static poses from the physically
validated forward placed step, then asks reference PD, the retained Stand
actor, and observation-COM feedback to hold each for three seconds. The
7/8/9-second poses are statically feasible; 8 and 9 seconds require the left
foot unloaded and the right foot supported. All controllers fall before three
seconds. Reference PD survives 1.42–2.05 s, the actor 0.90–1.07 s, and COM
feedback 0.95–1.23 s. See
`logs/mimickit-steps/single-support-holds-03/report.json`. Passing also
requires the correct foot-load pattern and a continuous final 0.5 s upright,
slow window.

The next capability is controlled weight transfer into right-leg support.
It needs a versioned one-leg observation/reward/retention contract; the
existing Stand and Perturb checkpoints must keep their original meaning.
Only after support passes should a swing-and-landing expert be trained.

## Short-step prerequisite check (2026-09-25)

`short_step.py` retimes the physically checked 15 cm forward placed step,
preserves its contact schedule, recomputes velocities, and reruns the full
physical gate. In a fixed 2/4/6/8/10/12/14-second bracket, only 14 seconds
passed the single-step physical checks. A sparse-pose cubic 2/4-second variant
failed additional stance and floor checks. These references are diagnostic
and are never marked as admitted training examples. See the
`logs/mimickit-steps/short-step-forward-*s-01/report.json` reports.

`evaluate_short_step.py` tested the 14-second passing candidate in native
MuJoCo. Reference PD fell at 1.87 s, the frozen Stand actor at 2.25 s, and
observation-COM feedback at 2.62 s. The swing phase begins at 3.98 s, so the
controller fails during weight transfer, before the foot can step. See
`logs/mimickit-steps/short-step-forward-14s-native-02/report.json`.

The next task is an independent, closed-loop single-leg weight-transfer gate.
Only a controller that survives that gate should supply starting states for a
short swing-and-landing task. Retiming the old clip is not a recovery-motion
shortcut; do not feed these candidates to Perturb training.

## Ball-hit stepping diagnostic (2026-09-25)

`evaluate_com_ball.py --reflex-sweep` compares the observation-COM ankle
teacher with bounded ankle/hip variants on the same eight original-reference
ball hits. Each recovered 4/8 and passed the short quiet-standing gate; none
showed a recovery step. See
`logs/mimickit-steps/com-ball-original-reflex-sweep-01/report.json`.

`capture_step_reflex.py` is a native-only capture-point trigger and one-foot
Jacobian placement probe. On the four shots the balance teacher failed, it
requested a 10 cm step. It lifted the foot on three shots but never completed
a supported landing at least 5 cm from its start, so recovery stayed 4/8.
The original 0.25 rad target envelope saturated on 88–94% of swing samples.
An isolated 0.5 rad diagnostic preserved the source actor's original joint
targets but still failed every placement and left 78–83% saturation. These
are diagnostic action-contract interventions, not compatible exports or
trained policies. See `com-ball-original-capture-step-03` and
`com-ball-original-capture-step-wide-01` under `logs/mimickit-steps/`.

The next training prerequisite is a short, native-trackable weight-transfer,
swing and loaded-landing task. The existing 22-second placed-step diagnostic
does not supply that reflex. Keep this probe and the mixed residual out of
Godot policy selection.

## Mixed-reference COM residual pilot (2026-09-25)

The observation-derived COM feedback teacher recovered 4/8 confirmed
original-reference ball hits over a 33-second test; native COM feedback
recovered the same four. The frozen source actor and earlier pre-lift residual
recovered 0/8. See
`logs/mimickit-steps/com-ball-original-teacher-33s-01/report.json`.

`distill_mixed_com.py` trained one unselected residual from support pushes,
original-reference physical-ball shots and quiet-standing rehearsal. Its eight
training shots use different angles, speeds, launch times, targets and phases
from the fixed eight-shot evaluation. The mixed artifact recovered 3/8
pre-lift hits but 0/8 original-reference hits, despite retaining 8/8 short
quiet-standing phases and the 33-second pre-lift no-ball hold. This is a
failed transfer pilot, not a Perturb checkpoint. See
`logs/mimickit-steps/com-residual-mixed-01/report.json` and the paired
`com-residual-mixed-ball-{support,original}-01` reports. No export or selected
Godot viewer policy was changed. Longer PPO or export from this
residual is blocked pending a teacher/controller that produces stable recovery
and a genuine step on the original reference.

Reproduce the mixed pilot from the project root with a new output directory:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.distill_mixed_com `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-steps/support-entry-5min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --agent-config logs/mimickit-steps/support-entry-5min-01/agent.yaml `
  --out logs/mimickit-steps/com-residual-mixed-new
```

## Physical-ball stress for the COM residual (2026-09-25)

`evaluate_com_ball.py` compares the unchanged support-entry actor and the
experimental COM residual under eight real ballistic hits: front, back and
two diagonal headings at 1.5/2.0 m/s with the 8 kg, 9 cm ball. All eight
shots hit in both runs. The 33-second window leaves over 30 seconds after
impact and requires a final uninterrupted half-second upright, slow,
two-foot-supported state. On the pre-lift reference, the source actor
recovered 0/8; the residual recovered 3/8. Those three contacts imparted
26.7–41.2 N·s normal impulse, but the feet moved at most 3.6–6.9 mm, so this
is balance without a demonstrated step. The residual survived a 33-second
no-ball hold, versus 7.067 seconds for the source actor. See
`logs/mimickit-steps/com-residual-ball-33s-02/report.json`.

The same paired test on the original standing reference recovered 0/8 for
both controllers (`com-residual-ball-original-33s-01/report.json`). These
stress tests clamp reference phase after the clip endpoint as in residual
training; current Godot long-run inference does not. The result blocks
export/promotion and longer Perturb training from this residual. It does not
test the selected Godot Perturb checkpoint.

Run from the project root with a new output directory; add
`--ball-reference original_stand` for the transfer check:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.evaluate_com_ball `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-steps/support-entry-5min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --agent-config logs/mimickit-steps/support-entry-5min-01/agent.yaml `
  --residual logs/mimickit-steps/com-residual-distill-02/experimental-residual.pt `
  --out logs/mimickit-steps/com-residual-ball-new
```

## Cross-engine COM parity and residual distillation (2026-09-25)

`godot_com_parity.py` compares the existing support-v2 observation decoder,
native MuJoCo, and Godot's `MjBridge` on 24 matching pose/velocity fixtures.
Godot COM X and velocity differ by at most 5.08e-9 m and 2.32e-10 m/s in
these reset-state character-only cases. See
`logs/mimickit-steps/godot-com-parity-01/report.json`.

`distill_com_feedback.py` trains an experimental 362-input ankle residual
around the frozen Stand actor, supervised by the observation-driven COM
teacher. The training set has ±80/160 N pushes at 2 s; the held-out set has
±120 N pushes at 1.75/2.25 s. The first pilot failed quiet retention (7/8)
despite passing 8/8 pushes. A single revision with fourfold quiet rehearsal
weight and a 0.2 action cap passed 8/8 held-out 12-second settled-root gates
and 8/8 original quiet-standing phases. Its hash-bound diagnostic artifact is
`logs/mimickit-steps/com-residual-distill-02/experimental-residual.pt`.
`com_residual_policy.py` reloads that artifact without selecting it for Godot.
These cases do not validate ball impacts, stepping, longer quiet retention,
multiple training seeds, or live Godot control.

Run either diagnostic from the project root with new output directories:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.godot_com_parity `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --godot $env:P4F_GODOT_EXE --out logs/mimickit-steps/godot-com-parity-new
& $python -m mujoco_rig.mimic.distill_com_feedback `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-steps/support-entry-5min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --agent-config logs/mimickit-steps/support-entry-5min-01/agent.yaml `
  --out logs/mimickit-steps/com-residual-new
```

## Observation-derived COM feedback (2026-09-25)

`observation_com.py` defines `mimic_com_from_stand_observation_v1`, a derived
state computed from the existing support-v2 observation and the hash-checked
rig. It does not change the actor's 362 inputs or saved normalization. On
four native ±120 N pelvis-pulse cases covering static and entry holds, the
retained actor passed 0/4 12-second settled-root gates; both the native-oracle
and observation-driven feedback passed 4/4. The maximum observed COM X and
velocity reconstruction errors were below 3e-8 m and 6.5e-8 m/s. See
`logs/mimickit-steps/com-observation-validation-01/report.json`. These cases
validate native signal parity and feedback only. Godot parity, ball impacts,
stepping, and a learned actor remain untested.

Run the native pulse comparison from the project root with a new output path:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.evaluate_com_observation `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-steps/support-entry-5min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --agent-config logs/mimickit-steps/support-entry-5min-01/agent.yaml `
  --out logs/mimickit-steps/com-observation-new
```

## Native COM feedback teacher (2026-09-25)

`com_feedback_teacher.py` adds bounded paired ankle-pitch feedback from
reference-relative whole-body COM X and velocity to the retained Stand actor.
This is a native MuJoCo oracle diagnostic; the actor and Godot observation
contract do not yet contain equivalent COM signals. On the isolated pre-lift
entry and 12-second terminal-pose hold, the actor alone fell at 7.067 s;
four feedback variants completed the hold and passed the settled-root gate.
The 1.5/1.5 gains used at most 0.113 normalized ankle action, also passed a
static 12-second hold, and retained 8/8 original quiet-standing cases. See
`logs/mimickit-steps/com-teacher-feasibility-01/report.json` and
`com-teacher-gates-01/report.json`. No training, export, or Godot selection
resulted from this test. Independent support shifts and an observation-parity
check are required before treating this controller as a distillation target.

Run the native feasibility check from the project root with a new output
directory:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.evaluate_com_teacher `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-steps/support-entry-5min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --agent-config logs/mimickit-steps/support-entry-5min-01/agent.yaml `
  --out logs/mimickit-steps/com-teacher-new
```

## Pre-lift support-transition gate (2026-09-24)

`support_transition.py` builds two six-second local tasks from the validated
forward placed-step reference: hold its 3.5-second pose, or enter that pose
from the stable 3.25-second pose over one second and then hold. The training
actor keeps the support-v2 observation/action contract. Native evaluation
requires six-second survival and settled root motion on both tasks, plus the
original quiet-standing check. These references diagnose the first failing
support transition; they are not full steps or impact-recovery demonstrations.

The five-minute, 2,048-world static stage improved native pose-hold survival
from 1.933 to 6.000 seconds and entry survival from 2.500 to 4.267 seconds.
Quiet standing stayed intact. The pose hold missed the final-root-error gate
by 3.5 mm, and the entry still fell. See
`logs/mimickit-steps/support-transition-5min-01/report.json`. No policy was
exported or selected for Godot.

An entry-targeted continuation reached six-second survival in both cases at
update 32 and passed the static settling gate, but its entry endpoint was
13.6 cm from target (limit 5 cm). Later weights regressed; the final actor
failed both survival checks. The validated update-32 `best.pt` is retained as
an isolated partial result, not a deployable step controller. See
`logs/mimickit-steps/support-entry-5min-01/{report.json,best.json}`. Diagnose
root/support trajectory and action behavior before further PPO on this task.

The paired native audit in `logs/mimickit-steps/support-entry-audit-02` found
a fore-aft oscillation with planted, loaded feet and no saturated motors. The
retained actor is still moving forward at six seconds and falls at 7.067 s in
the 12-second terminal-pose hold
(`support-entry-retention-12s-01`). The original reward's horizontal-root
position and velocity terms are very weak at the measured error. A versioned
support-only tracking reward can be selected with
`--reward-mode mimic_support_tracking_v1`; the default remains `reference`.
Its 128-world, 16-update smoke test lost one of eight quiet-standing cases and
worsened entry endpoint error. The unchanged starting checkpoint was retained;
`support-tracking-smoke-01` was not exported or selected. Do not extend this
entry-only reward run; the rehearsal and longer gate below test that change.

`support_rehearsal_task.py` now provides that isolated design: fixed quiet
worlds train on the original reference reward, while entry worlds practice
the terminal pose for 12 seconds. Native selection checks original quiet
standing, static hold, entry and a 12-second hold with the reference phase
clamped after six seconds. Two short integration controls kept quiet 8/8 but
did not improve support. A 2,048-world five-minute pilot likewise kept quiet
8/8 while entry and longer-hold survival regressed to 2.567 s from 6.000 and
7.067 s respectively. Its `best.pt` is unchanged iteration zero. See
`logs/mimickit-steps/support-rehearsal-5min-01/{report.json,best.json}`.
Do not extend this PPO configuration or promote it to Godot; diagnose the
support policy update/critic signal or introduce explicit COM/support control.

The read-only `audit_support_critic.py` now checks the value head against
discounted native return to failure. On the retained actor's delayed-fall
trajectory, the critic overpredicts the return at six seconds by about 31.
Critic-only fitting with `probe_support_calibration.py` improves aggregate
quiet/support validation RMSE, but worsens error on the critical 2–4-second
native interval. The actor is unchanged. See
`logs/mimickit-steps/support-critic-audit-02.json`,
`support-critic-calibration-01/critic_calibration.json` and
`support-critic-audit-calibrated-01.json`. The calibrated weights are
diagnostic only; they do not justify another actor-training run.

The bounded rehearsal command uses new output directories:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.train_support_transition `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-steps/support-entry-5min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --out logs/mimickit-steps/support-rehearsal-new `
  --training-mode transition --reward-mode mimic_support_tracking_v1 `
  --quiet-fraction 0.25 --support-seconds 12 --retention-seconds 12 `
  --envs 2048 --seconds 300
```

For repeatable read-only root, contact, action and motor traces, run:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.audit_support_entry `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --agent-config logs/mimickit-steps/support-entry-5min-01/agent.yaml `
  --checkpoint entry logs/mimickit-steps/support-entry-5min-01/best.pt `
  --episode-seconds 12 --out logs/mimickit-steps/support-entry-audit-new
```

Run an isolated pilot from the project root with a new output directory:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.train_support_transition `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --initialize-from logs/mimickit-training/stand-support-v2-15min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --out logs/mimickit-steps/support-transition-new `
  --training-mode static --envs 2048 --seconds 300
```

`--training-mode transition` targets the entry motion directly. A continuation
can use a prior pilot's `best.pt`; the `best.json` file binds that checkpoint
to its contract and hash. Outputs remain isolated and do not update the
viewer selection.

## AMP placed-step pilot (2026-09-24)

`amp_step.py` implements MimicKit's motion-discriminator interface on the
existing Newton/MuJoCo dummy. It uses consecutive native pose/velocity states
and validated CC-BY-4.0 standing and forward/back stepping demonstrations.
The actor still uses the 362-input support-v2 contract; no Godot adapter or
viewer selection changes are needed for this isolated experiment. The task
reward targets an upright body, support COM transfer and placed left foot while
the discriminator provides a whole-body motion prior.

The initial 2,048-world pilot completed 19 GPU updates and 1,245,184
transitions, but inherited three-second training episodes. The corrected
eight-second connected-episode pilot completed 21 updates and 1,376,256
transitions. Quiet standing remained 8/8; complete native step survival only
changed from 4.667 to 4.717 seconds, before the planned 7-second lift. Its
final discriminator gave every sampled demonstration a negative logit, although
demonstrations still scored above generated motion on average. This pilot does
not support promoting the actor or merely extending training. Results and
source hashes are in `logs/mimickit-steps/amp-step-connected-01`.

The corrected task reward includes pre-lift COM transfer and waits until after
landing to reward low root speed. A matched three-minute pilot retained quiet
standing but reached only 4.700 seconds of native survival, again without a
step (`amp-step-transfer-reward-01`). The repeatable native `audit_prelift.py`
localizes the control boundary between reference poses at 3.25 seconds
(sustained standing) and 3.5 seconds (falling). Removing self-collisions and
rescaling the actor's phase do not fix it; stronger PD is worse. See
`prelift-audit-01/report.json`. The separate support-transition experiment
above now measures that boundary directly.

Run a new isolated pilot from the project root with new output directories:

```powershell
$python = "mujoco_rig/mimic/.venv/Scripts/python.exe"
& $python -m mujoco_rig.mimic.train_amp_step `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --demonstration mujoco_rig/mimic/assets/stand_reference.npz `
  --demonstration mujoco_rig/mimic/assets/steps/forward_reference.npz `
  --demonstration mujoco_rig/mimic/assets/steps/backward_reference.npz `
  --demonstration logs/mimickit-steps/placed-step-forward-09 `
  --demonstration logs/mimickit-steps/placed-step-backward-09 `
  --initialize-from logs/mimickit-training/stand-support-v2-15min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --out logs/mimickit-steps/amp-step-new --envs 2048 --seconds 180
```

The native complete-step and original quiet-standing tests are recorded before,
during and after training. No export is performed. The discriminator's motion
examples do not include recorded ball-impact recovery; this pilot does not
establish Perturb behavior.

Run the native pre-lift audit with a new output directory:

```powershell
& $python -m mujoco_rig.mimic.audit_prelift `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/placed-step-forward-09 `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --out logs/mimickit-steps/prelift-audit-new
```

## Single placed-step diagnostic (2026-09-24)

`placed_step.py` builds a forward or backward 15 cm left-foot placement from the
existing licensed standing pose, static step audit and calibrated support-transfer
poses. Both 22-second references pass the one-foot placement, contact, penetration,
root-support and delayed motor feasibility gates. This is a procedural reference,
not a motion-captured human recovery or a trained skill. The generic walking
validator expects *both* feet to lift, so its `foot_lift` flag is intentionally
false; `single_step_checks` instead requires the swing foot to lift at least 3 cm
and finish within 2 cm of the requested 15 cm placement.

The native Stand v2 actor falls at 4.667 seconds on either complete reference,
before the planned lift at 7 seconds. Two isolated five-minute guarded PPO pilots
on the forward reference completed 27 / 39 updates and 1,769,472 / 2,555,904
transitions. The second used connected eight-second episodes beginning at rest,
weight shift or toe-off. Both kept quiet standing at 8/8 but only extended native
full-step survival to 4.700 / 4.784 seconds; neither placed the foot. Further
runs with the same PPO imitation setup are not supported by these results. No
policy was exported to Godot. Reports are under
`logs/mimickit-steps/placed-step-{forward,backward}-09`,
`placed-step-{forward,backward}-native-01`, `placed-step-forward-pilot-01` and
`placed-step-forward-staged-01`.
The Godot Perturb selection is independent; inspect `logs/mimickit-viewer/ball.json`
for the current user-selected bundle.

Reproduce with the configured Mimic virtual environment and pinned MimicKit
checkout, choosing new output directories:

```powershell
& $python -m mujoco_rig.mimic.placed_step `
  --calibration logs/mimickit-steps/transfer-calibration-02 `
  --audit logs/mimickit-foundation/audit-04 `
  --direction forward --out logs/mimickit-steps/placed-step-forward-new
& $python -m mujoco_rig.mimic.track_transfer `
  --mimickit $env:MIMICKIT_PATH `
  --calibration logs/mimickit-steps/placed-step-forward-new `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --out logs/mimickit-steps/placed-step-native-new
```

## Current milestone: front/back impact recovery

The mixed follow-up (`perturb-mixed-30min-01`) is complete and **rejected as a
replacement**. It ran 223 updates in 1805.453 seconds. Relative to the reviewed
candidate, focused survival fell 6/12 -> 3/12 (forward 2/6 -> 1/6, backward
4/6 -> 2/6), and full-suite settled recovery fell 25/96 -> 5/96. Quiet survival
remained 8/8 at five and 30 seconds. Godot/native parity passes. The final export
was created and selected automatically, then rolled back to the reviewed
`perturb-front-back-15min-01/final-export` at run completion. The user later
selected the mixed final export for visual review; see `ball.json` above for the
current viewer choice. No further run is queued.
`perturb_mixed` remains available to reproduce the experiment, not as a recommended
automatic continuation. See `logs/mimickit-training/perturb-mixed-30min-01/comparison.json`.

After visual review of the first candidate, we tested mixed recovery training.
To reproduce that bounded experiment:

```powershell
./mujoco_rig/mimic/scripts/train.ps1 -Experiment perturb_mixed
```

This 30-minute preset starts from `perturb-front-back-15min-01/model.pt` and its
`final-export/contract.json`, not the retained original actor in `best.pt`.
`mixed_recovery_v1` samples mutually exclusive episode categories: 50% focused
front/back torso shots, 25% full-angle/all-body shots, and 25% quiet episodes.
These are probabilities at reset, not guaranteed proportions of transitions:
episodes can have different lengths. The broad component can naturally include
front/back torso shots too. This version requires `QuietFraction = 0.25`.
Ball speeds remain 1–2 m/s, and physics, reward, observation contract and optimizer
settings are unchanged. All 96 validation cases remain active.

The fixed comparison targets are at least 4/6 backward survivals (reviewed
candidate), 3/6 forward survivals (original baseline), one focused settled
recovery, and 8/8 quiet survivals. Report the full suite against both policies
as well; the original baseline had 76/96 survivals and 26 settled recoveries.
Repeat the 30-second runtime stress with the same finite-reference limitation
described below. Export the final candidate for Godot review, preserve rollback,
and stop after the paired comparison rather than chaining further runs.

The first run (`perturb-front-back-15min-01`) completed 110 updates in 900.407
seconds. Focused five-second survival improved 3/12 -> 6/12 and settled recovery
0/12 -> 1/12; backward survival improved 0/6 -> 4/6 but forward survival fell
3/6 -> 2/6. Full-suite survival fell 76/96 -> 72/96. Quiet survival remained 8/8
at five and 30 seconds; all six surviving candidate impact cases also settled by
the end of the long runtime stress. Godot/native parity passes. The candidate is
selected for review, with the baseline available for rollback. This is partial
progress with trade-offs, not a successful overall replacement or authorization
to automatically extend training. See `logs/mimickit-training/perturb-front-back-15min-01/comparison.json`.

The continuous-standing/optimizer investigation below is paused. It is not a
prerequisite for this gameplay experiment. Preserve the current Perturb export,
then run one bounded curriculum before deciding whether to train longer:

```powershell
./mujoco_rig/mimic/scripts/train.ps1 -Experiment perturb_front_back
```

This preset runs 15 minutes with the configured 2,048 GPU worlds, 25% quiet
episodes, and 8 kg physical balls at 1–2 m/s. Training targets Chest, Spine and
Pelvis from world +/-X with +/-10 degrees of angular variation. The existing
`recovery_v1` reward permits displaced, single-support recovery; no new stepping
reference, observation migration, rig change or optimizer adjustment is implied.
The pinned starting checkpoint is the previously selected
`perturb-resume-15min-01/best.pt`, using its original legacy contact contract.
Override both `-InitFrom` and `-SourceContract` together to use another matching
checkpoint. Do not substitute support_v2 semantics into these weights.

Full 96-case native checkpoint selection remains unchanged. Review the 12
front/back torso cases separately: all confirmed hits, survival, final settled
recovery, and quiet-standing retention. `review_ball_recovery` runs those cases
plus eight quiet phases for five seconds and for 30-second runtime stress. The
long test preserves Godot's existing endpoint sampling and out-of-range phase;
it is not proof of an in-distribution continuous-standing capability.

`-ViewerPolicy final` (enabled by this preset) exports the actual final trained
candidate to `final-export` and selects it for F5/F6 review. `best.pt` and `export`
still contain the policy retained by validation, possibly the unchanged starting
actor. The final export is explicitly for review, not a claim that it passed
retention. Stand's viewer selection is independent. Existing presets continue to
select `best` by default. After training, run `MimicPerturbArena.tscn`; choose Chest,
Spine or Pelvis in the Inspector, 1–2 m/s, and use the Forward/Backward buttons
for the focused comparison. Random targets include untrained curriculum cases.

```powershell
# Review the current selection, or a particular run's final candidate.
./mujoco_rig/mimic/scripts/watch.ps1 -Experiment perturb_front_back
./mujoco_rig/mimic/scripts/watch.ps1 -Run logs/mimickit-training/YOUR_RUN -Policy final
# Select an already exported final candidate or restore the previous viewer.
./mujoco_rig/mimic/scripts/select.ps1 -Run logs/mimickit-training/YOUR_RUN -Policy final
./mujoco_rig/mimic/scripts/select.ps1 -Experiment perturb -Previous
```

Stopping rule: complete one 15-minute run and the paired review. More settled
front/back recoveries without directional survival or quiet-standing regression
justify considering continuation. Otherwise, record that the focused curriculum
did not meet the milestone and decide explicitly whether to supply a usable
recovery-step motion prior or change the control approach. Do not automatically
chain reward/optimizer adjustments or increase projectile difficulty.

## Continuous-standing reference and retention gate (paused investigation)

`continuous_stand` prepares eight independent stationary targets from fixed postures
of the physically admitted small weight-shift reference. Each target holds its pose
for unlimited time, has exactly zero reference velocity/acceleration, and exposes
zero in the phase channel. Cases never interpolate into one another. Root height
is lowered by an explicit 0.1 mm contact seating depth before repeating the existing
geometry, foot/toe support, root-wrench and actuator-window checks. This avoids
CPU/GPU contact-set disagreement at exactly tangent float32 reset poses; it does
not change the rig, solver, contact parameters, action authority, or delayed PD.

The versioned reference is `mimic_stationary_stand_reference_v1`; the policy contract
is `mimic_stationary_stand_target_pd_v1`. Its 362 observations have the existing
physical/control groups, but its last channel means stationary phase zero and its
reference duration is null. `ContinuousStandTask` explicitly selects this reference
and resets every world to its assigned posture with zero velocity. The ordinary
finite-clip loader rejects these files. Legacy exports, normalization statistics,
training commands and Godot adapters keep their existing semantics.

`probe_continuous_stand` evaluates saved policies without training, exporting or
selecting them. Substituting this new contract into an existing actor is an explicit
**transfer diagnostic**, not a claim that the actor was trained for these targets.
The first supplied checkpoint is the teacher; each checkpoint's parent must contain
its matching `agent.yaml`. The fixed anchor bank measures teacher-to-candidate KL on
the same states used by the preceding safeguard experiment.

```powershell
python -m mujoco_rig.mimic.continuous_stand `
  --source logs/mimickit-steps/weight-shift-small-01/weight_shift_reference.npz `
  --out logs/mimickit-steps/continuous-stand-new

python -m mujoco_rig.mimic.probe_continuous_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/continuous-stand-new/stationary_reference.npz `
  --anchor-bank logs/mimickit-training/shift-precision-5min-01/quiet_anchor.npz `
  --checkpoint logs/mimickit-training/small-shift-15min-01/best.pt `
  --checkpoint logs/mimickit-training/stand-support-v2-15min-01/best.pt `
  --out logs/mimickit-training/continuous-stand-comparison-new
```

Use the experiment virtual environment and previously configured `MIMICKIT_PATH`.
Output directories must be new. The 30-second gate counts only each world's first
episode and requires all eight to survive, each mean root error to stay at or below
5 cm, and the final two seconds to remain settled. Settled means pelvis height at
least 0.75 m, tilt at most 15 degrees, horizontal speed at most 0.2 m/s, angular speed
at most 1 rad/s, at least 5 N support per foot including toes, and foot-position
error at most 3 cm. Timeouts alone do not pass; maintenance resets cannot replace
failed episodes. Thresholds are versioned in `continuous_stand.GATES`.

The report includes per-case measurements, trace/checkpoint/source hashes and a
safeguard calibration outcome. A teacher failing this gate yields
`no_passing_teacher` and no proposed KL limit: even zero deviation cannot certify
the missing behavior. A sampled separation between passing/failing policies is
also not a universal safe radius. This gate is not silently added to selection for
legacy tasks. Train/admit the new contract explicitly before using it for promotion.

### Bounded continuous-Stand training

`train_continuous_stand` is a separate entry point for this contract. It reuses the
existing GuardedPPO updates, target PD, contact observations and DeepMimic reward;
the finite-clip Stand/Perturb commands are unchanged. Initialization must match
the checkpoint hash and contract in its source `best.json`. A finite-clip
`support_v2` source additionally requires `--allow-reference-migration`. All
physical/control fields and observation groups except the explicitly changed
phase/reference semantics must match. Weights and normalization statistics load
unchanged; the destination receives its own contract and provenance.

```powershell
python -m mujoco_rig.mimic.train_continuous_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --initialize-from logs/mimickit-training/small-shift-15min-01/best.pt `
  --source-contract logs/mimickit-training/small-shift-15min-01/export/contract.json `
  --allow-reference-migration --envs 2048 --seconds 300 --iterations 64 `
  --evaluation-interval 16 --calibration-steps 200 --actor-max-kl .02 `
  --out logs/mimickit-training/continuous-stand-pilot-new
```

Each world keeps one of the eight postures for a 30-second episode. The local
outer loop uses upstream rollout/update methods but does **not** inherit upstream's
all-world reset at logging checkpoints: only initial setup, actual per-world
failure and the 30-second timeout reset training episodes. Separate native
validation never resets the GPU training worlds. `updates.jsonl` records each
iteration; `learning_curve.json` records native evaluations and selection decisions.

Before PPO, the frozen actor generates critic-calibration episodes. Every posture
gets equal loss weight and whole-world training/validation separation. Validation
uses `(world_index // 8) % 4 == 3`, so all eight postures appear in both partitions;
plain `world_index % 4` would omit whole postures from one partition. Calibration
must reduce balanced validation MSE by at least 10%, without increasing any
posture's RMSE, while preserving actor/normalization fingerprints.

The existing rollout KL limit remains active. There is no fixed-teacher distance
anchor or new optimizer search. Selection refuses to lose **any** incumbent
passing 30-second case, even if total successes increase. Otherwise it ranks the
complete gate, passed cases, survival count, worst/mean duration and root error.
The pilot stops at the first measured native retention regression, the iteration
limit or the time budget. `--seconds` includes periodic native checks; startup,
initial validation, calibration and final native/GPU validation are additional.

Artifacts include `initial.pt`, `calibrated.pt`, `model.pt`, `best.pt`, `best.json`,
the resolved `agent.yaml`, `contract.json`, `protocol.json`, calibration results,
and a final report. The config comes from the source run's `agent.yaml`, with the
documented guard, clipping and evaluation overrides recorded in the new run.
The current runner saves checkpoints only: the stationary contract still needs a
matching Godot viewer/export adapter. It does not select a legacy viewer bundle.

The first pilot (`continuous-stand-5min-01`) stopped at iteration 16 after losing
all three baseline 30-second successes (3/8 to 0/8). The selector retained the
initial checkpoint; native and GPU rechecks both retain 3/8. The five-minute PPO
ceiling was cut short after 71.86 seconds. Critic calibration passed, but neither
that result nor rollout KL below 0.02 established behavioral improvement. PPO
rollouts had reached only 8.534 seconds of episode age before this check; delayed
failures of previously stable cases occurred at 17.55–21.54 seconds. Investigate
long-horizon collection and value/advantage coverage before extending training.

#### Temporal readiness and the sustained-standing objective

The opt-in `--objective retention_30s_v1 --temporal-ready` experiment addresses
the credit audit findings without changing the rig, actor inputs or native gate.
It defines a continuing tracking objective in physical time:
`discount = exp(-control_timestep / 30)` and
`reward = (1 - discount) * existing_tracking_reward`. Failure still returns zero
and terminates with zero value; the 30-second collection timeout bootstraps the
critic. Thirty seconds is the discount's e-folding time, not a finite terminal
horizon. Normalization keeps the ideal continuing value near one without changing
the ordering of policies under that discount. This objective still requires the
separate per-case settled-retention gate; it does not guarantee safe behavior.

```powershell
python -m mujoco_rig.mimic.train_continuous_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --initialize-from logs/mimickit-training/continuous-stand-5min-01/best.pt `
  --source-contract logs/mimickit-training/continuous-stand-5min-01/contract.json `
  --objective retention_30s_v1 --temporal-ready --calibration-steps 1000 `
  --envs 2048 --seconds 300 --iterations 64 --evaluation-interval 8 `
  --out logs/mimickit-training/continuous-stand-temporal-new
```

`agent.yaml` and `protocol.json` record the objective. `value_initialization.json`
records the critic-head unit scaling; actor and normalization are untouched.
Scaling supplies only an initialization prior, not a valid conversion between
discount horizons. Mandatory frozen-actor calibration must fit the new targets.
Its v2 gate retains every posture check and adds age bands 0–5, 5–15 and 15+ seconds
and 0.5/2/5-second windows preceding failures. Each slice needs at least 32 training
and 32 validation samples. No slice RMSE may increase, and balanced posture MSE
must improve at least 10%. All measured groups receive equal calibration loss
weight. Rejection saves a report and starts no PPO; no gate is relaxed automatically.

After successful calibration, a frozen-policy preparation pass simulates 30
seconds with staggered per-world resets. It generates real live states at different
episode ages; it does not set arbitrary clocks on fresh poses or replay old PPO
samples. Every incumbent passing posture must have at least four early (<5 s) and
four late (>=15 s) live worlds. Training preserves these states, discards preparation
samples and collects a fresh on-policy batch. The first actual batch is checked
again before any update. `preparation.json`, `first_rollout_coverage.json` and
`samples_age_15_plus` in each update record make coverage inspectable. Preparation,
calibration and final checks are outside the bounded PPO-stage budget.

The first combined experiment (`continuous-stand-temporal-01`) passed calibration
and preparation, including 8,891 late-state samples in its first 65,536-sample PPO
batch. It still regressed from 3/8 to 0/8 at iteration eight, on both native and GPU
retention checks, and stopped after 57.42 seconds of its five-minute PPO ceiling.
The baseline actor remains selected. Treat these options as a verified experimental
pipeline, not an established improved controller; isolate actor-update direction
before extending training or changing additional parameters.

### Frozen standing credit audit

`probe_standing_credit` compares the initial, calibrated and final checkpoints of
a stationary pilot without updating parameters. It uses stochastic GPU first
episodes through failure or 30 seconds, the same actor trajectories for the two
baseline critics, and the candidate's own trajectories for its critic. It saves
scalar traces and reports value errors/advantages by episode age and time to fall.

```powershell
python -m mujoco_rig.mimic.probe_standing_credit --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --run logs/mimickit-training/continuous-stand-5min-01 `
  --envs 256 --seed 210922 --out logs/mimickit-training/standing-credit-audit-new
```

The source run must use the stationary contract and contain `initial.pt`,
`calibrated.pt`, `model.pt`, `agent.yaml` and `contract.json`. Reports include
checkpoint, source and trace hashes. Raw 32-step TD-lambda targets use upstream
MimicKit code; failure values are zero and timeouts bootstrap the evaluated critic.
Full sampled returns also bootstrap timeouts, and are noisy estimates rather than
ground-truth action advantages. Only first-episode masks enter reported metrics.
These stochastic timeouts do not substitute for native settled-retention passes.

The first audit (`standing-credit-audit-01`) found that calibration improved overall
value RMSE while worsening the five-second pre-failure slice. The rejected actor
also achieved a slightly higher short-discount return despite fewer 30-second
survivors. Its independent `verification.json` records a finite-horizon comparison:
merely correcting discount to the upstream 30-Hz timebase still favors that actor,
whereas the undiscounted 30-second reward sum does not. This is diagnostic evidence,
not a trained replacement objective. Keep existing weights and Godot selections
until temporal coverage and behavioral validation support a replacement.

### Isolated actor-update direction

`probe_actor_update` starts from a temporally calibrated stationary checkpoint,
prepares live episode ages, and collects one fresh 32-step batch. It compares an
unchanged actor, one optimizer step and one normal actor phase (40 optimizer steps
with the current config). Both updated branches independently start from the same
weights, fresh optimizer state, sampler permutation and random-generator states.
The critic is never updated. Whole-batch actor loss is measured before/after;
critic, normalization and fixed-distribution tensors must remain bitwise unchanged.

```powershell
python -m mujoco_rig.mimic.probe_actor_update --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --run logs/mimickit-training/continuous-stand-temporal-01 `
  --envs 2048 --trials 256 --seed 210923 `
  --out logs/mimickit-training/actor-direction-new
```

GPU validation advances all branches together, including an unchanged-policy
repeat, with identical Gaussian noise for each paired trial. Only the first
episode counts. Measured discounted 30-second reward prefixes have zero tails;
a separately labeled estimate bootstraps surviving timeouts with the same frozen
critic. Prefix return is not the full continuing objective. Reported uncertainty
resamples paired replicas within each posture and is conditional on the one
training batch; it does not measure variation across training seeds. Native
deterministic 30-second settled-retention checks accompany these stochastic trials.

Artifacts include diagnostic branch weights, the exact rollout/advantage batch and
sampler permutation (`batch.pt`), preparation coverage, shared evaluation noise,
scalar first-episode traces, native retention traces, provenance and comparisons.
The source run and Godot selections are never modified. This is a direction probe,
not a replacement training loop or an automatic policy-selection path.

The first probe (`actor-direction-01`) preserved the same three native standing
cases for both updated branches. Whole-batch PPO loss improved, but matched
stochastic return/survival comparisons remained inconclusive: the normal actor
phase survived 19/256 trials versus 16/256 for baseline, with paired return
uncertainty including zero. Independent CPU arithmetic reproduced the first SGD
step. This does not identify a wrong-sign gradient or establish a better controller;
the remaining question concerns successive updates and critic feedback. Source
weights and Godot selections remain unchanged.

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
error 3.91e-7. Open `MimicStandArena.tscn` and press F6 to inspect the automatically
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

Use `track_transfer` for that native diagnostic without admitting the calibration
to training. It verifies the saved source/rig/content hashes and reruns physical
checks, then compares reference-only PD, the existing Stand actor, and a diagnostic
prefilled command queue. It includes the actor's original standing baseline,
eight three-second transfer phases, and a full-duration attempt from frame zero.
Tracking errors use post-integration reference times and exclude subsequent resets.
Full motion requires both foot lifts; short phase probes do not. All results
remain diagnostic and cannot select a Godot policy.

```powershell
& $python -m mujoco_rig.mimic.track_transfer --mimickit $env:MIMICKIT_PATH `
  --calibration logs/mimickit-steps/transfer-calibration-02 `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --out logs/mimickit-steps/<new-tracking>
```

`--interventions-only` runs three separate three-second initial-phase diagnostics:
hold the first pose, use the old actor's original phase scaling, or disable
self-collisions in a private model. These interventions cannot qualify a reference
or controller for promotion. `transfer-tracking-01` fails native transfer tracking:
the actor falls at 1.650 s, reference PD at 1.883 s, before the first planned lift.
The original standing actor still passes 8/8 baseline trials. Holding the initial
calibration pose passes; phase scaling and collision interventions do not fix the
moving transfer. See `STATUS.md` for the interpretation and next experiment.

### Isolated weight-shift learning pilot

`weight_shift` extracts the calibration's two double-support weight shifts and
reverses them back to standing, recomputing velocities and physical checks. Its
separate `mimic_weight_shift_reference_v1` schema requires both feet/toes planted
and measurable horizontal movement. It does not require foot lift; the stepping
schemas and their admission requirements are unchanged. The parent calibration
remains diagnostic. No generated references are silently substituted into Stand
or Perturb training.

`train_stand --task weight_shift` uses the same target-PD controller and guarded
PPO, with an explicit reference-only warm start from the compatible Stand actor
or a continuation from a checkpoint using the exact same motion reference.
All other contract fields, including contact semantics, must match. Reference
hashes/durations and unchanged frozen normalization are recorded in provenance.
Native selection evaluates eight motion phases and both three- and five-second
quiet standing trials. Quiet retention takes priority over motion progress. Motion
gates include horizontal displacement error, so survival without following the
weight shift is insufficient. `--no-select-viewer` is mandatory.

```powershell
& $python -m mujoco_rig.mimic.weight_shift `
  --calibration logs/mimickit-steps/transfer-calibration-02 `
  --out logs/mimickit-steps/<new-weight-shift-reference>

& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/<new-weight-shift-reference>/weight_shift_reference.npz `
  --task weight_shift --control target_pd --contact-mode support_v2 --stability guarded `
  --initialize-from logs/mimickit-training/stand-support-v2-15min-01/best.pt `
  --source-contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --envs 2048 --seconds 300 --iterations 1000000 --evaluation-interval 16 `
  --actor-max-kl .02 --critic-grad-clip 1 --seed 210921 --no-select-viewer `
  --out logs/mimickit-training/<new-weight-shift-pilot>
```

The critic clip is an explicit optimizer option, defaulting to zero (unchanged
upstream behavior) for existing experiments. The first weight-shift attempt failed
numerically: finite rollout data entered critic SGD, whose loss/weights diverged
within seven minibatches. The bounded reproduction is saved in
`logs/mimickit-training/weight-shift-critic-probe.json`. The retry uses the existing
MimicKit gradient clipping implementation at norm 1.0 without changing rewards,
physics, actor KL, or normalization. A generic nonfinite-parameter stop now records
parameter names and metrics in `numerical_failure.json`; it does not export a
failed policy. Pilot outcomes are recorded in [`../STATUS.md`](../STATUS.md).

To diagnose the early fall, `audit_weight_shift` accepts repeated `--bundle`
arguments and compares each actor on quiet standing, a held start pose, quarter,
half and full motion amplitude. It records the first episode's actions, physics
substeps, contact loads and normalization coverage. Reduced amplitudes exist only
in memory and are not admitted training references. `--normalizer-unclipped`
instead runs quiet/full-shift cases with only the ONNX observation clip bypassed
in memory; learned statistics, weights and action clipping remain unchanged.

```powershell
& $python -m mujoco_rig.mimic.audit_weight_shift --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --bundle logs/mimickit-training/weight-shift-5min-02/export `
  --reference logs/mimickit-steps/weight-shift-reference-01/weight_shift_reference.npz `
  --out logs/mimickit-steps/<new-weight-shift-audit>
```

Reports distinguish original and intervened cases. Neither mode trains, writes
policy exports or selects a Godot viewer bundle.

### Smaller-shift curriculum and motion features

`prepare_shift_curriculum` creates stopped, retimed prefixes of the two admitted
weight shifts, recomputes velocities and reruns all physical gates. The one-second
and 1.5-second prefixes each become a 16-second two-sided shift/return cycle. Foot
lift remains inapplicable to this separate task; neither curriculum stage weakens
the existing physical limits.

```powershell
& $python -m mujoco_rig.mimic.prepare_shift_curriculum `
  --source logs/mimickit-steps/weight-shift-reference-01/weight_shift_reference.npz `
  --prefix-seconds 1 --out logs/mimickit-steps/<new-small-reference>
```

Repeat with `--prefix-seconds 1.5` and a different output directory for the next
stage. Weight-shift evaluation v2 adds a relative displacement gate so small motions
cannot pass by remaining stationary. V1 selection reports are rejected by the v2
ranker rather than reinterpreted.

`prepare_motion_normalization` defaults to `additive_features`: it keeps the entire
pretrained normalization path and appends separately normalized pending-target and
future-reference features. New actor/critic input weights start at zero, preserving
the old policy while allowing broader motion features to learn. External input
width stays 362; internal normalized width becomes 628. Both paths remain frozen
during guarded PPO. Direct scale replacement and compensated migration are retained
as explicit diagnostic modes; the tested candidates failed retention and cannot be
used as approved initializations.

```powershell
& $python -m mujoco_rig.mimic.prepare_motion_normalization --mimickit $env:MIMICKIT_PATH `
  --bundle logs/mimickit-training/stand-support-v2-15min-01/export `
  --checkpoint logs/mimickit-training/stand-support-v2-15min-01/best.pt `
  --reference logs/mimickit-steps/weight-shift-small-01/weight_shift_reference.npz `
  --reference logs/mimickit-steps/weight-shift-medium-01/weight_shift_reference.npz `
  --migration-mode additive_features --out logs/mimickit-steps/<new-normalization-preparation>
```

The preparation must pass quiet retention, retain motion baseline survival, cover
the new inputs without clipping in its validation cases, and preserve calibration
actions within export tolerance. Training checks the manifest, checkpoint and
reference hashes and records the feature architecture in export lineage. A rejected
candidate has `diagnostic_only: true`. Preparation artifacts are not viewer bundles.

The admitted `shift-normalization-03` package can initialize this first learning
pilot (the command starts training; preparation alone does not):

```powershell
& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/weight-shift-small-01/weight_shift_reference.npz `
  --task weight_shift --control target_pd --contact-mode support_v2 --stability guarded `
  --initialize-from logs/mimickit-steps/shift-normalization-03/candidate.pt `
  --source-contract logs/mimickit-steps/shift-normalization-03/contract.json `
  --envs 2048 --seconds 900 --iterations 1000000 --evaluation-interval 16 `
  --actor-max-kl .02 --critic-grad-clip 1 --seed 210921 --no-select-viewer `
  --out logs/mimickit-training/<new-small-shift-pilot>
```

### Quiet-standing rehearsal and longer motion episodes

The rehearsal configuration uses `--weight-shift-seconds 8` and
`--weight-shift-quiet-fraction .5`. Half the training worlds practice the original
standing reference for five seconds, while half practice the motion for eight
seconds. World groups are fixed, ensuring the requested transition mixture even
when motion episodes fail more frequently. Each reset samples a new phase within
its own reference's valid interval. Reset poses, reference observations, phase,
reward targets, delayed PD targets and timeouts all use that world's reference.

The flags are opt-in; existing Stand/Ball and three-second motion runs retain
their defaults. Rehearsal TEST mode uses motion only, preserving the existing
native/GPU comparison. Native selection keeps the eight three-second motion cases
and three-/five-second standing checks, and adds eight motion cases spanning the
configured longer horizon. The longer suite has its own phase grid. Selection
uses `mimic_weight_shift_rehearsal_v1`, prioritizes standing retention and complete
tracking, then longer-horizon survival before short-horizon survival. Different
evaluation horizons cannot be compared in the same selection history.

Continuation restores the additive feature architecture from the source export's
validated normalization lineage before optimizer construction and checkpoint
loading. It does not replace statistics or expand the observation interface.
Use the retained checkpoint from the previous 15-minute pilot, rather than its
regressed final weights or the two-update integration smoke checkpoint:

```powershell
& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/weight-shift-small-01/weight_shift_reference.npz `
  --task weight_shift --control target_pd --contact-mode support_v2 --stability guarded `
  --initialize-from logs/mimickit-training/small-shift-15min-01/best.pt `
  --source-contract logs/mimickit-training/small-shift-15min-01/export/contract.json `
  --weight-shift-seconds 8 --weight-shift-quiet-fraction .5 `
  --envs 2048 --seconds 900 --iterations 1000000 --evaluation-interval 16 `
  --actor-max-kl .02 --critic-grad-clip 1 --seed 210921 --no-select-viewer `
  --out logs/mimickit-training/<new-rehearsal-pilot>
```

The completed rehearsal run regressed; do not extend that configuration unchanged.
See `STATUS.md` for its measured result and subsequent diagnostics.

### Precision reward, fixed standing anchor and critic calibration

Three opt-in controls address the observed weak movement incentive and cumulative
standing-policy drift. Existing Stand, Ball and reference-reward defaults are unchanged.

- `--weight-shift-reward precision_v1` selects `mimic_weight_shift_precision_v1`.
  Motion reward is 0.5 times the original reward, plus 0.4 times a horizontal
  position Gaussian (5 mm scale), plus 0.1 times a horizontal velocity Gaussian
  (0.05 m/s scale). Quiet worlds retain the original reward. Native and CUDA
  motion evaluation use the same objective. Tracking/survival gates are unchanged.
- `--quiet-anchor-max-kl .02` protects fixed states from the starting policy's
  eight five-second standing trajectories. The teacher must pass standing retention.
  Every actor optimizer step must satisfy both the usual rollout KL bound and
  the maximum per-state KL against this frozen teacher. Rejection restores weights
  and optimizer momentum. The state bank, source checkpoint hash and limit are
  recorded; each new run anchors its explicitly loaded starting policy. This is
  not a guarantee for states outside the bank, so native retention gates remain.
- `--critic-calibration-steps 200` samples complete first episodes with the actor
  and normalization frozen, then fits only the critic. Failures use zero terminal
  bootstrap; timeouts bootstrap from the initial critic. Every fourth world is
  reserved for validation, with both tasks represented. Admission requires at
  least 10% reduction in balanced validation MSE and no task RMSE regression.
  Failure restores the original critic and stops before actor updates. The best
  admitted critic is saved; calibration momentum is discarded before PPO.

Precision reward requires mixed rehearsal and explicit critic calibration. Use
`--calibrate-critic-only` to stop after validation and save `calibrated.pt` without
PPO or export. Calibration is preparation outside the `--seconds` PPO budget.
The reward version appears in experiment and evaluation metadata; selection
rejects comparisons across reward identities. Observation/actuator contracts and
legacy normalization retain their original meaning.

```powershell
& $python -m mujoco_rig.mimic.train_stand --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/weight-shift-small-01/weight_shift_reference.npz `
  --task weight_shift --control target_pd --contact-mode support_v2 --stability guarded `
  --initialize-from logs/mimickit-training/small-shift-15min-01/best.pt `
  --source-contract logs/mimickit-training/small-shift-15min-01/export/contract.json `
  --weight-shift-seconds 8 --weight-shift-quiet-fraction .5 `
  --weight-shift-reward precision_v1 --quiet-anchor-max-kl .02 `
  --critic-calibration-steps 200 --calibrate-critic-only `
  --envs 2048 --critic-grad-clip 1 --no-select-viewer `
  --out logs/mimickit-training/<new-precision-calibration>
```

For an integration smoke run, omit `--calibrate-critic-only` and add
`--iterations 2 --seconds 120 --evaluation-interval 16`. A longer actor-training
budget is a separate experiment; neither calibration nor a smoke test establishes
learned full-motion tracking. Godot selection stays unchanged.

### Backtracking within the standing constraint

The first five-minute precision run stalled: only three of 75 iterations changed
the actor. `--actor-backtracks 8` enables an opt-in bounded search in guarded PPO.
Each proposal can try its original learning rate and up to eight halvings. Each
trial starts from the same pre-proposal weights, gradients and optimizer state;
momentum/Adam counters never accumulate across rejected trials. The configured
base learning rate is restored after acceptance or exhaustion.

Acceptance requires the existing rollout KL and fixed standing KL limits, plus
a decrease in the **same minibatch** PPO actor loss greater than
`1e-7 * max(1, abs(previous_loss))`. A feasible but uphill or numerically unchanged
proposal is rejected. This is local surrogate descent, not a behavioral guarantee.
No bounds are relaxed. The default `--actor-backtracks 0` keeps the original
single-proposal behavior.

`updates.jsonl` and the final `actor_search_summary` record trial counts, accepted
backtracked steps, sum of accepted learning-rate fractions, local loss decreases,
constraint rejections and loss rejections. Acceptance counts alone can conceal
negligibly small steps, so inspect the fractions and actual native behavior too.
Add `--actor-backtracks 8 --iterations 16 --seconds 300 --evaluation-interval 16`
to the precision command above and omit `--calibrate-critic-only` for a bounded
integration check. Exhausting all trials still stops that actor minibatch loop;
backtracking cannot guarantee progress for a direction outside the feasible region.

`--actor-project-anchor` adds a second, bounded direction search after ordinary
backtracking is exhausted. It projects the full optimizer displacement onto the
linearized constraint at the currently worst protected state, aiming 0.1% inside
the unchanged KL limit. Every projected trial still checks all protected states,
rollout KL and actual same-batch loss decrease. A local linear approximation never
overrides those checks. The optimizer records one gradient update on acceptance;
its displacement is projected, while rejected trials restore all state.

This option requires a positive fixed anchor and enabled backtracking. It records
`mimic_actor_anchor_projection_v1`, projected trial counts and accepted projected
steps. Both direction searches share the configured halving budget separately
(at most 18 trials with eight backtracks). The default remains disabled. Passing
these optimizer checks does not establish motion tracking or long-term standing.

The paired `anchor-retention-audit-01` diagnostic confirms that this fixed KL is
not a calibrated standing-stability metric. On the same 600 protected states,
unanchored update 16 passes all eight five-second standing cases at maximum KL
0.404095, while update 32 fails three at 1.583010. The original Stand actor passes
all eight at 1.787167 relative to the motion teacher. These comparisons do not
justify a new numeric limit; they show that distance alone does not rank stability.

The quiet reference is finite (5.966786 seconds). Do not obtain a longer standing
benchmark simply by increasing the standard evaluator's `episode_seconds` beyond
the reference duration: its phase-grid formula then produces negative starts.
The audit preserves valid five-second starts and separately labels two 30-second
stress tests: existing endpoint clamping with increasing phase, and a diagnostic
final-pose hold with zero reference velocity and phase clamped to one. Neither is
a trained long-standing contract. Original Stand survives 1/8 and 2/8 respectively;
the motion teacher and guarded finals survive 0/8. All delayed failures from
short-suite survivors occur after the clip ends. Establish physically validated,
explicit continuous-standing reference semantics before using such a long-horizon
gate to calibrate the safeguard. Current training bounds and viewer bundles remain
unchanged. See `../STATUS.md` and the audit's `report.json` / `verification.json`.

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

For live Stand viewing, open `Scenes/RL/Isaac3/MuJoCo/MimicStandArena.tscn` and press F6.
F6 runs the scene currently open in the editor. F5 runs the project's configured
main scene; in the current checkout that is `MimicPerturbArena.tscn`, not Stand. The root
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

For interactive viewing, run `MimicStandArena.tscn` with F6; its Inspector already selects
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

Open `Scenes/RL/Isaac3/MuJoCo/MimicPerturbArena.tscn` and press F6 (or press F5, as it is the
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

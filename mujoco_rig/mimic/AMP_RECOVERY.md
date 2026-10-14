# State-based AMP recovery pilot

## Prepared whole-body Stand objective (2026-10-01)

Implemented `whole_pose_v6` after the user authorized coherent full-pose imitation.
Its global pose score combines weighted local quaternion errors for all actuated
body chains, with angular tolerances for natural small adjustments. Arms and legs
now contribute to quiet-Stand pose matching. Unactuated wrists are excluded;
no world-space foot anchors, root-position targets or motor pose servos are added.
The reward multiplies balance, whole-body pose similarity and worst torso/head
alignment, then subtracts torque/action-change and bounded joint-speed costs.
Per-body reference-angle errors are available in training logs and native reports.

The prepared `TorqueStandWholePoseTraining` preset preserves 2,048 environments
and evaluation interval 64, with a 15-minute budget and initialization from final
`1c9238f8` plus its matching full AMP sidecar. The measured short-hold curriculum,
major-collapse training termination and independent quiet-standing gate remain.
No training, tests, compilation, export or Arena selection occurred in this change.
The preset is for quiet Stand; future impact training must explicitly relax stance
matching according to instability and recovery state. New tolerances/weights have
not been physically validated.

## Hypothesis

The existing recovery reward permits rapid foot motion. A learned motion prior
from licensed quiet-standing and natural walking clips may reduce oscillation
without losing recovery. This is a new pilot, not a continuation of the historical
timed left-foot AMP experiment.

## Training contract

- MimicKit AMP implementation, pinned by `runtime.py`; existing Newton/MuJoCo-Warp physics.
- Same 363-input actor, 30 motor actions, delayed PD, frozen standing base, and `step_v5` task reward.
- Stationary reference for every actor input. Either foot may move; no step phase or chosen foot.
- Four consecutive pose/velocity states (364 values) feed a training-only discriminator.
- Demonstrations: existing 100STYLE idle, forward and backward adaptations; sampling weights 50/25/25 percent.
- The data license is CC BY 4.0; attribution and modifications are retained in each source manifest.
- These are kinematic style examples. They are not recorded ball-recovery motions or dynamically tracked expert trajectories.
- Task/style reward weights: 0.8/0.2. Actor and discriminator normalization are initialized and frozen independently.
- Physical curriculum unchanged: 8 kg, 0.09 m radius, 2 m/s, all twelve body targets, one hit per 30-second episode, 25 percent quiet episodes.
- 2,048 worlds, evaluation every 64 updates, five minutes of synchronized training-update time. Initial/final native evaluation adds wall time.

## Checkpoints and validation

`agent.yaml` and ordinary `.pt` files preserve the PPO-compatible inference format.
`amp_agent.yaml` and `.amp.pt` sidecars preserve the discriminator and complete AMP
model. Exporting the actor does not require the discriminator. The pilot starts
from the selected `recovery-loop-candidate-all-target-30min-01/best.pt` with measured
8 x 600-second quiet retention, rather than the weaker final Arena preview.

Native initial/final assessments add contact-point slip, normalized-action
variation, action second differences, and 30 N foot-load transitions. Contact
velocity is measured at the actual ground contact with its point Jacobian, so
rotation about a planted contact is not automatically counted as sliding.
Samples are at the control rate; substep peaks can be missed. These measurements
are diagnostics, not an established naturalness score. Shorter survival must not
be reported as smoother recovery.

Independent validation must compare the frozen candidate and source on identical
fresh all-target impacts. Quiet standing remains a separate gate. Natural recovery
requires physical stepping evidence and visual assessment; an aggregate recovery
increase alone does not pass that gate. This experiment does not automatically
export or select an Arena policy.

## Command

```powershell
pwsh -File mujoco_rig/mimic/scripts/train-reactive.ps1 `
  -Preset ReactiveAMPRecoveryPilot -ResumeSelectedSameMass `
  -Out logs/mimickit-training/amp-recovery-prior-5min-01
```

The launcher reads `Envs` and `EvaluationInterval` from `scripts/config.ps1`.
Append `-DryRun` to inspect the exact resolved command.

## Results

Run: `logs/mimickit-training/amp-recovery-prior-5min-01`.
The complete paired assessment is in `pilot-validation.json`; `report.json`
contains the training and internal selection results.

- 24 updates, 3,145,728 transitions, 309.234 seconds of synchronized training updates.
- Total trainer time: 831.781 seconds, including its native validation.
- Sampled training GPU utilization: mean 38.62 percent, peak 99 percent, p95 73 percent.
  The 261 one-second samples cover part of the training window. CPU dispatch and
  native validation explain why wall time and GPU utilization differ.
- Fixed native suite: recovered 27/32 to 28/32, survived 28/32 to 29/32,
  recovered with a physical step 1 to 2. Contact slip decreased 20.64 percent,
  action variation 25.70 percent, and action second differences 28.10 percent.
- Final actor quiet standing: 8/8 uninterrupted 600-second native trials passed.
- Paired fresh all-body suite, seed 292125, 8 kg at 2 m/s for 33 seconds:
  recovered 58/64 to 57/64, survived 60/64 to 58/64, recovered with steps 1 to 0.
  There were no new recoveries. Case 47 lost recovery and survival; case 5 lost
  survival. Both physically contacted Chest with identical source/candidate
  impulses. All 64 trials contacted the dummy; 41 contacted their intended target.
- Across the 57 cases recovered by both actors, slip decreased 4.57 percent,
  action variation 3.81 percent, action second differences 4.94 percent, and
  foot-load switching 6.41 percent. Across all cases, quality metrics worsened;
  the failure trajectories must not be omitted from the overall outcome.
- Final discriminator logits averaged -0.180 for demonstrations and -2.203 for
  generated motion. Demonstration positive-class accuracy remained zero. Relative
  separation exists, but adequate calibration remains unresolved; this does not
  establish that calibration alone caused the unseen regressions.
- Nine contract tests passed, including history reset, native feature parity,
  licensed demonstration loading, and portable/AMP checkpoint identity. Final
  portable and training sidecar tensors were checked to match exactly.

Final portable checkpoint SHA-256:
`a9e278b69251b62a71d7d3674eedd01f9712bc37c1a54e05f1b6c822e03aada6`.
The selected iteration is 24. The checkpoint is retained as an experimental
candidate; it is not admitted. At the user's explicit request, its ONNX export
is selected in `MimicPerturbArena.tscn` as an experimental preview: 8 kg at 2 m/s,
random angles and body targets, every 30 seconds. Export maximum action error
against PyTorch is 3.5762786865234375e-07. Natural stepping has not been
established. The tested portable actor keeps the standing foundation.

The next experiment should first calibrate the discriminator against held-out
motion snippets and improve coverage with licensed balance/recovery examples.
Walking clips alone have not established natural impact recovery. Preserve
Chest cases 5 and 47 as retention tests. No second training run was started.

## Calibrated 15-minute follow-up

Run: `logs/mimickit-training/amp-recovery-smooth-15min-02`, initialized from the
explicitly previewed `a9e278b6` portable actor and its matching AMP sidecar.
Preset: `ReactiveAMPRecoverySmooth15min`, launched with `-ResumeExperimentalFinal`.

The task, plant, stationary actor inputs and 30-second single-hit curriculum
remain unchanged. The discriminator now also samples validated two-foot weight
transfer and forward/backward placed-step references, retaining their source
manifests. These are adapted kinematic support examples, not human ball-recovery
recordings. Sagittal mirroring includes either foot without an actor step clock.
The mixture is 35% idle, 15% each walking direction, 15% weight transfer and
10% each placed-step direction.

Discriminator calibration uses frozen native traces, holds out cardinal direction
3, and separates complete 0.5-second demonstration blocks with no overlapping
histories. Training normalization combines balanced demonstration/native samples,
with a 0.05 standard deviation floor. The actor/critic and actor normalization are
checked to remain identical during calibration. SGD learning rate is 0.001;
calibration is bounded to 256 updates and must improve held-out binary loss and
reach at least 60% accuracy for each class before policy training starts.

The maximum foot-motion cost is 0.04 before task/style mixing. It activates only
after physical contact and penalizes acceleration above 10 m/s² plus contact
reversals within 0.2 seconds. Load hysteresis is 30 N on / 15 N off. The first
support transition receives no reversal cost; both feet are treated identically. These
are control-rate diagnostics and soft costs, not a prohibition on fast recovery
steps. Quiet standing still uses the original task reward and frozen stand base.

Policy training is budgeted for 900 synchronized update seconds, 2,048 worlds and
evaluation every 64 updates. Native evaluations and discriminator calibration add
wall time. Final comparisons must include all trajectories, recovery/survival,
foot acceleration/contact reversals, common-success diagnostics, and independent
8 x 600-second quiet holds. Natural stepping is not established by these metrics
alone. Arena selection is a separate action after validation.

Setup run `amp-recovery-smooth-15min-01` completed its initial native assessment
but stopped before policy updates: the loaded discriminator normalizer's
nonserialized mean-square buffer needed initialization before setting moments.
Run 02 fixes that setup defect and vectorizes the identical sagittal reflection.

### Completed follow-up results

- Final actor: `ad7f73dc4b4b4c7dea142728331e01d4f043f70a91fa40b5d23066d3190536cc`.
  Completed 79 updates, 10,354,688 transitions and 909.515 synchronized training
  seconds. Total trainer time was 1,489.610 seconds, including calibration and
  native assessments. No additional training is running.
- Held-out calibration reached 99.58% demonstration accuracy and 97.42% native
  generated accuracy. Balanced BCE improved from 0.71875 with the new frozen
  mapping to 0.560778 after fitting. Clipped features were 0.0177%. Calibration
  preserved actor, critic and actor normalization tensors exactly.
- Final actor passed all eight uninterrupted 600-second native quiet holds,
  including foot-load, drift and final-settling gates, with no delayed fall.
- Fresh paired native suite, seed 292145: recovery 56/64 to 61/64; survival 60/64
  to 64/64. No previously successful case lost recovery or survival. All 64 balls
  physically hit; 36 contacted their intended target. One candidate recovery
  included a measured physical step, versus zero for the source.
- Across the 56 cases recovered by both actors, mean foot acceleration decreased
  11.62%, rapid contact reversals 28.24%, contact slip 10.15%, foot-load switching
  22.41%, action variation 4.51% and action second differences 6.54%. Overall
  trajectory metrics also improved, but include the benefit of avoiding failures.
- Fixed suite: recovery 28/32 to 29/32; survival 29/32 to 31/32. A directional
  regression prevented automatic selection; `best.pt` retains the source actor.
  At the user's explicit request, the final `model.pt` is loaded in
  `MimicPerturbArena.tscn` as an experimental preview. ONNX maximum action error
  is 3.2782554626464844e-07. Restart the scene to use it.
- Exact prior cases 5 and 47, seed 292125, were replayed in complete eight-posture
  blocks. Case 5 now survives but does not settle; case 47 now survives and
  recovers. Their intended targets are Chest and Spine respectively; the prior
  report recorded physical Chest contacts for both. The other 14 replayed cases
  retained their previous recovery/survival outcomes, including one existing
  Head failure. Results are in `chest-retention.json`.
- During policy update intervals, 899 GPU samples averaged 42.47% utilization,
  with p95 75.1% and peak 99%. Calibration and native evaluation are excluded.

Full paired evidence and provenance are in `pilot-validation.json`. The Arena
retains 8 kg balls at 2 m/s, random targets/angles and a 30-second interval. This
candidate remains unadmitted: unseen recovery is 61/64, and natural stepping has
not been established. The next action is visual review of the loaded Arena motion.

## Useful-placement comparison (2026-09-30)

The source `ad7f73dc` was audited on fixed native torso impacts before changes.
At 2 m/s it recovered 29/32, survived 31/32 and recovered with one measured step.
80.16% of its risk-weighted v5 swing signal came while both feet remained below
the measured lift threshold. Across successful cases, first-five-second mean
joint ranges were 5.22 degrees at the hips, 6.42 at the knees and 7.08 at the
ankles/toes. Leg movement exists; coordinated stepping remains rare. At 3 m/s,
recovery drops to 12/32, survival to 14/32, with five stepping recoveries. At
2.5 m/s the source recovers 23/32 and survives 24/32 with two stepping recoveries.
These are development probes, not untouched generalization evidence.

`ReactiveAMPPlacement15min` continues the source for 900 synchronized policy-update
seconds with 2,048 environments and EvaluationInterval=64. It preserves the AMP
mixture/calibration, motor contract, standing base, 25% quiet episodes, all body
targets, random angles, 8 kg/0.09 m balls and one shot per 30-second episode.
Training speed expands to 2-2.5 m/s; the primary paired unseen test stays at 2 m/s.
This is a combined reward/curriculum pilot, not an isolated causal ablation.

The opt-in `step_v6` objective removes sustained grounded preparation credit.
Dense swing credit requires unloading and clearance. One bounded event bonus
requires a measured lift/landing, at least 0.03 m placement, hip/knee displacement
of at least 0.08 rad, supported opposite foot, and at least 0.02 m progress toward
the capture point recorded at takeoff. No foot, trajectory or step time is fixed.
A bounded 0.04 loaded-slip cost uses finite-difference velocities of the same
material corners near the lowest sole surface, allowing rolling contact. This
is a GPU-compatible slip proxy; native contact-Jacobian slip remains the independent
metric. Quiet/pre-impact rewards are unchanged.

Twelve focused tests cover shuffling, wrong-direction/non-articulated placements,
either-foot events, reset, repeated credit, sliding versus rolling, and existing
step/settling behavior. Final 8x600-second quiet holds, paired fresh hits, contact
slip, hip/knee motion and measured recovery steps determine the result.
Run: `logs/mimickit-training/amp-useful-placement-15min-01`.

### Completed useful-placement results

- Final checkpoint: `3fa62c5d2396d8366aefe49f01e372a1afdfa4df4bd4883e5e5bac636dcb9310`.
  80 updates, 10,485,760 transitions, 903.143 synchronized policy-training seconds;
  1,438.047 total trainer seconds. All trainers and this run's GPU sampler exited.
- Native quiet standing passed 8/8 uninterrupted 600-second trials, including
  strict loads, foot motion, drift and settling. The finite reference is not the
  current blocker; the standing foundation remains intact.
- Fixed 2.5 m/s suite: recovery 23/32 to 19/32, survival 24/32 to 19/32, stepping
  recoveries 2 to 1. Intermediate update 64 also regressed. Automatic selection
  retained the initial actor; its portable tensors exactly match the source.
- Fresh paired 2 m/s all-body suite, seed 300949: recovery 63/64 to 62/64, survival
  64/64 to 63/64, stepping recoveries 2 to 2. All 64 balls contacted the character;
  41 contacted their intended target. No recovery gains. Case 15 lost recovery
  and survival: intended UpperArm_R, actual Chest contact, fall at 5.30 seconds.
- Across 62 common recoveries, native contact slip increased 5.32%, foot
  acceleration 7.77%, load switching 12.24%, rapid contact reversals 22.73%,
  action variation 1.17% and second differences 3.09%. Hip/knee ranges changed
  only +3.42%/+1.46%; these do not establish better coordinated leg recovery.
  All-case metrics also worsened and are included in `pilot-validation.json`.
- Training-window GPU utilization: 893 samples, mean 40.40%, p95 74%, peak 100%.
  Calibration and native tests are excluded. Calibration passed; final portable
  and full AMP sidecar actor tensors match exactly. Twelve focused tests passed.

The reward/curriculum combination failed this comparison. No Arena selection or
export was made; `ad7f73dc` remains the reviewed Arena brain. This does not isolate
whether loss came from the new reward terms or harder curriculum. The material
sole-speed penalty is a proxy, and it did not reduce independent contact slip.
Placement-event frequency was not logged during stochastic training, so the
deterministic step counts cannot establish how often that bonus activated.

The source already performs some physical steps under stronger hits, supporting
that stepping is possible with the current rig and motor contract. These tests
do not exclude limitations in the physical setup. Before further PPO, validate
reactive full-body demonstrations against the actual plant and add event-credit
observability. Do not repeat this setup unchanged or claim duration alone will
produce natural balance.

## Physical examples and reward activation audit (2026-09-30)

This phase uses frozen actors only. No optimizer updates, new checkpoint, export
or Arena selection were made. The Arena still uses `ad7f73dc`. Envs=2048 and
EvaluationInterval=64 remain unchanged.

### Placement observability

`RecoverySignals` counts every rollout transition on the GPU, including placement
events and overlapping rejection reasons. It drains once per optimizer iteration
to `recovery_signals.jsonl`; numeric totals also enter `updates.jsonl`. Reset retains
window totals. Final-frame logging cannot measure an infrequent touchdown bonus.

The opt-in `step_v7` checks maximum hip/knee displacement during swing instead of
only takeoff versus landing. A knee can bend then extend before landing; `step_v6`
misses that excursion. Other credit conditions and weights remain unchanged:
either foot, one event per episode, no clock. This corrects measurement; it does
not establish naturalness or a trained gain. The deployed motor contract is unchanged.

### Frozen native teachers

Run: `logs/mimickit-training/reactive-physical-examples-01`.
Two existing actors each received 64 fixed development torso impacts: 32 at
2.5 m/s and 32 at 3 m/s, 8 kg, radius 0.09 m, 33-second episodes.

- Current Arena actor: 34/64 recovered, six stepped recoveries. Five of those six
  lacked the selected hip/knee excursion; all six exceeded the selected contact
  slip and rapid-reversal bounds. Fifteen placements had articulation >=0.08 rad
  during swing but endpoint displacement below that threshold.
- Earlier heavy-40-minute selected actor (`81b0ce03`): 34/64 recovered, four stepped
  recoveries. Three lacked the selected leg excursion, all four exceeded contact
  slip bounds, and three exceeded reversal bounds. Sixteen placements had
  articulation missed by the endpoint check.
- Zero accepted clips; bilateral demonstration coverage is false. No rollout was
  admitted as an AMP example. Recorded actions/full plant poses remain in trace
  files; replay export was not exercised because no candidate passed the filter.

Filter: a recovered measured step, stepped-leg hip range vector >=0.20 rad and
knee range >=0.25 rad in the first five post-hit seconds, <=0.05 m summed native
contact slip over the post-hit episode, <=2 rapid contact reversals, <=2 steps.
These are mechanical selection criteria, not a validated naturalness score or
visual assessment. Existing placed-step examples are kinematically checked;
their recorded native transfer tracking trials failed.

### Frozen GPU rollout assay

Run: `logs/mimickit-training/recovery-reward-rollout-audit-01`, seed 300953.
Two cohorts used identical randomized all-twelve-target shot parameters,
2-2.5 m/s, one shot per 30-second episode, 2048 worlds: 513 quiet, 1535 hit worlds.
Failed worlds were reset and parked; outcomes concern first episodes only.
This is a development distribution assay, not unseen validation or PPO training.

- Deterministic mean: 1535 confirmed hits, 141 hit failures, zero quiet failures;
  80 event worlds (5.21% of hit worlds), 321 placements, 132 endpoint articulation
  misses. Peak checks still rejected 38 placements for insufficient articulation.
- Support-conditioned exploration, leg sigma up to 0.20 and base sigma 0.05:
  1535 confirmed hits, 644 hit failures, 23 quiet failures; 274 event worlds
  (17.85%), 1855 placements, 428 endpoint misses. Opposite support/initial risk
  rejected 1284 placements; capture progress rejected 684. Reasons overlap.
- Mean loaded sole-speed proxy: 0.0040 versus 0.0197 m/s, distinct from native
  contact-Jacobian slip. More credit events alone do not establish improvement.
- Wall time after initialization: 122.95 and 135.16 seconds; 7,372,800 total world
  transitions. Zero optimizer updates; the frozen checkpoint hash is unchanged.

### Motion prior versus motor authority

`audit_motion_authority.py` compares each demo against all eight stationary
hip/knee commanded target intervals. Commands are bounded at +/-0.25 rad
(14.3 degrees) from standing, further clipped to joint limits.
Results: `reactive-physical-examples-01/command-envelope.json`.

- Both walking examples have at least one hip/knee outside the commanded interval
  in every frame, for every standing posture. Maximum excess: 56.0 degrees forward
  and 37.4 degrees backward.
- Placed forward/backward examples exceed the interval in 46-74% / 34-47% of
  frames, depending on posture. Weight shift: 9-35%. Idle: zero.

This is commanded-pose incompatibility, not a hard bound on realized motion:
impacts and inertia can move joints beyond commanded bounds. It supports testing
motor authority and compatible motions before more training; it does not isolate
one exclusive cause of shuffling.

Eighteen focused tests passed: peak-versus-endpoint articulation, transition
counters, either-foot filters, false bilateral coverage and target clipping.
Both assay processes exited.

Next experiment: version an isolated recovery motor contract with more hip/knee
target range gated by observable support loss; preserve the original quiet path.
First validate command/torque behavior and quiet holds in native MuJoCo, then
collect physically tracked full-body reactions compatible with that range.
Keep the standing reward gate, tune exploration using measured failures, and
compare unseen hits before Arena promotion. Do not reuse incompatible walking
examples unchanged or present another duration-only PPO run as a fix.

## Isolated recovery motor experiment (2026-09-30)

`RecoveryTargetControl` is opt-in through `ReactiveBallTask(recovery_motor_radians=...)`.
Default tasks and the Arena retain the original motor contract. The experimental
action mode is `support_gated_target_pd_v2`; actor width remains 363 and the action
width remains 30. Old checkpoint contracts are not silently relabeled or admitted.
The native audits explicitly substitute motor semantics under a frozen source actor.

### Command contract

- Only Thigh rx/ry/rz and Shin rx may receive extra target range; either leg can
  use it. Other joints, actuator limits, PD gains, latency and physics are unchanged.
- Gate: capture error ramps from 0.03 to 0.08 m, or foot unloading from 15 to 5 N
  with opposite support and horizontal motion from 0.15 to 0.35 m/s. Inputs are
  current observations; no hit flag, selected foot, future shot or step clock.
- Gate zero preserves original queued commands and torques exactly. The newest
  issued absolute target, already visible in actor observations, is the slew origin.
  No additional unobserved controller history is introduced.
- Maximum tested radius: 1.25 rad (71.6 degrees), clipped to original joint limits.
  All existing hip/knee demo poses fit the fully open commanded interval at that
  radius. This does not establish dynamic tracking or natural recovery.
- Initial revision capped every gate-active hip/knee target at 2 rad/s. That
  damaged the trained corrective response even after target-unit conversion.
  Revision `observable_hip_knee_authority_v2` caps only requested targets beyond
  the original +/-0.25-rad interval. Existing corrective targets remain available.
- `bridge_source_action` converts old physical target displacements to the new
  normalized action units. It is an experimental initialization adapter, not an
  exported checkpoint or a new trained brain.

### Native measurements

Each impact comparison uses the same 64 development torso impacts: 32 at 2.5 m/s
and 32 at 3 m/s, 8 kg, radius 0.09 m, one hit in 33-second episodes.

- `recovery-motor-authority-01`: legacy 34 recovered / 38 survived / 6 stepped
  recoveries. Directly rescaling the unmodified actor to radii 0.5, 0.85 and 1.25
  produced 25, 22 and 20 recoveries, respectively, and no stepped recoveries.
  All 30-second quiet traces exactly matched the legacy trace, with zero active
  gate frames. These results reject direct command amplification as a migration.
- `recovery-motor-quiet600-01`: initial contract revision passed all eight strict,
  uninterrupted 600-second quiet holds. No gate activation or delayed fall;
  minimum loads by posture were 215-315 N left and 306-341 N right. Tail foot
  motion was at most 0.604 mm. The later limiter revision has the same gate and
  exactly the same closed-gate path; its direct native hold was 30 seconds, not a
  second 600-second run.
- `recovery-motor-bridge-01`: initial all-active-target limiter plus the target-unit
  adapter recovered/survived 26/64, with no stepped recoveries. This motivated
  limiting only new extension requests.
- `recovery-motor-bridge-02`: revised limiter plus adapter recovered 34/64,
  survived 38/64, with 6 stepped recoveries. Every case's survival, hit and recovery
  outcome matches the legacy source. One failed case (62) changed its measured
  step count. Recovery outcomes are retained; naturalness is not improved.

### Physical stepping teacher candidates

`CaptureStepTeacher` is a native development controller, not a trained actor.
It preserves the source target units, chooses the less loaded foot from current
loads, estimates the capture point from current COM/velocity, and solves foot IK
within the experimental command bounds. Clearance, proximity to the placement
goal and landing load drive its phases. Neither foot nor a step time is prescribed.
Only physical motor commands act on the body; no root corrections or external
stabilizing forces are applied. Source upper-body control remains unchanged.

- `recovery-capture-step-teacher-01`: trigger below 20 N, placement capped at
  0.18 m. Triggered both feet (25 left, 34 right), produced measured steps in
  22 episodes, but recovered only 9/64; no stepped episode survived. Zero clips
  passed the unchanged mechanical selection criteria.
- `recovery-capture-step-teacher-02`: revised motor limiter, trigger below 5 N
  and smaller placement capped at 0.08 m. Triggered both feet (44 left, 30 right),
  produced measured steps in 27 episodes, but recovered only 7/64; no stepped
  episode survived. Zero qualifying clips. Realized knee excursions reached
  approximately 1.44 rad, demonstrating physical leg articulation without
  demonstrating stable full-body recovery.

The second teacher changes several conditions and is not an isolated causal
ablation. Both failures reject these swing-only teacher candidates. Command range
was a mismatch; removing it does not supply coordinated support/COM management.
No expert motion was admitted, no optimizer ran, and no Arena export was made.
No independent unseen-hit suite was consumed for promotion.

Ten controller tests plus eighteen reward/example tests passed. They cover exact
legacy behavior at a closed gate, bilateral state gates, target-unit conversion,
extension speed, original corrective commands, joint/torque clipping, transport
delay and partial reset. All native experiment processes exited.

Next: a full-body motion-tracking teacher or motion-conditioned controller that
coordinates support transfer, pelvis/trunk response, swing and landing. First
require surviving physical demonstrations on both legs; keep quiet-standing
retention and independent unseen-hit gates. Another PPO run using these rejected
teacher motions would not be supported by the current evidence.

## Whole-body native teacher investigation (2026-09-30)

`FullBodyTeacher` extends the development investigation to coordinated stance,
COM braking, virtual pelvis translation/height, upright trunk, swing and landing.
Floating-root damped least squares solves desired poses on a separate MuJoCo
data object. Only bounded hinge motor targets are sent to the physical plant;
the teacher never writes the live root or applies external stabilizing forces.
The less loaded foot is selected from current state. Support loads, measured
clearance, placement error and landing loads drive transitions; there is no
fixed foot or step clock. This is an offline controller candidate, not a newly
trained brain or an exportable checkpoint migration.

The frozen source is `amp-recovery-smooth-15min-02/model.pt` (`ad7f73dc`). All
comparisons use the existing 64 development torso impacts, 8 kg / 0.09 m, half
at 2.5 m/s and half at 3 m/s, one hit in 33 seconds. Source recovery/survival is
34/38, with six stepped recoveries. Mechanical example thresholds are unchanged.
The audit now evaluates the actual teacher during the eight 30-second quiet
holds, rather than evaluating only its source actor.

- `recovery-full-body-teacher-01`: gain 0.5, blending source targets toward
  geometric IK angles. Recovered/survived 10/64, no stepped recovery or accepted
  examples. Triggered 33 left and 22 right attempts. Quiet standing matched the
  original trajectories exactly, with no activation. Adding kinematic whole-body
  coordination did not establish dynamic support.
- `recovery-full-body-teacher-02`: same gain and cases; add solved-minus-measured
  joint displacement to source targets, preserving their existing motor offsets.
  Recovered/survived 19/64, one stepped recovery, no accepted examples. Case 46
  had qualifying left hip/knee excursion but 0.142 m total contact slip and seven
  rapid reversals. The teacher remained in its lift phase for 1,871 frames despite
  returning to two-foot standing. The source-offset composition improves this
  teacher relative to geometric blending but still regresses source recovery.

- `recovery-full-body-teacher-03`: same offset composition/gain; release any
  teacher phase once observable authority is zero, both loads exceed 30 N,
  root height exceeds 0.75 m, tilt is within 15 degrees and horizontal COM speed
  is below 0.15 m/s. Recovered 26/64, survived 27/64, five stepped recoveries.
  There are surviving articulated steps on both legs, but all stepped recoveries
  fail slip/reversal criteria. Releasing without stricter rearming permits
  repeated interventions: 111 triggers and 71 releases, including 17 releases
  in case 60. No elapsed time or projectile flag drives the release.
- `recovery-full-body-teacher-04`: rearm only after both loads exceed 200 N,
  COM speed is below 0.05 m/s, tilt is within five degrees, capture error is
  below 0.01 m, leg/trunk joint speeds are below 0.1 rad/s and root angular
  speed is below 0.25 rad/s. Recovered/survived 27/64, six stepped recoveries,
  54 triggers and 20 releases. Three source failures become recoveries, while
  ten source recoveries fail. Every stepped recovery still fails slip and
  reversal criteria. Paired surviving case 58 uses both legs with knee ranges
  0.384/0.267 rad; slip improves from 0.442 to 0.259 m and reversals from 20 to
  ten relative to experiment 03, still above the 0.05 m / two-reversal limits.

All four teachers passed the actual eight 30-second quiet trials with exactly
the original qpos trajectories and no teacher activation. This phase did not
repeat the 600-second hold. Twenty-nine focused controller/reward/example tests
passed, including solver isolation, orientation feedback sign, bounded motor
targets, state release and strict rearming. All native jobs exited.

Zero examples passed the unchanged mechanical filter, so none was replayed,
exported or added to AMP. No optimizer ran, and Arena remains `ad7f73dc`.
No independent unseen-hit gate was consumed. A mechanically filtered example
would still require native action replay and visual review before motion-prior
admission; passing bilateral examples remain absent.

These ablations establish that articulated physical stepping is possible with
the experimental motor envelope, but this kinematic teacher still loses support
and produces loaded-foot sliding. The next hypothesis is contact-constrained
dynamic tracking: explicitly plan feasible support wrenches, COM/trunk and swing
accelerations within existing physical torque/friction limits. First demonstrate
low-slip bilateral recovery and retain quiet standing on native development
tests, then admit replayed motion examples. Training on the rejected teacher
traces would reinforce the behavior the user wants removed.

## Contact and torque teacher investigation (2026-09-30)

`DynamicSupportTeacher` reuses the bilateral, measured-state stepping phases but
plans COM, pelvis/trunk and foot accelerations with physical motor bounds. It
includes the full character mass matrix, bias/passive forces, unactuated joints
and actual foot/toe floor contacts. Planned forces are never applied externally;
ordinary runs convert torques to the existing delayed PD target contract. Only
private solver data is modified. These controllers remain opt-in diagnostics.

Before attempting ball impacts, `probe_dynamic_support.py` forces the controller
active during eight 30-second gravity-only holds. The regular quiet gate also
evaluates the actual state-gated adapter: it must retain the source trajectories.
Direct-torque probes refresh bounded torques each physics step and bypass target
transport only for diagnosis. Their traces cannot be admitted to AMP or exported.

- `recovery-dynamic-support-01` stopped on a runtime error: saturated torque
  intervals need fixed-variable elimination in MuJoCo's box QP. This was fixed
  without widening limits.
- `recovery-dynamic-support-02`: independent friction-ray forces, 20 motors;
  all gravity cases fail within 2.44 seconds despite feasible algebraic plans.
- `recovery-dynamic-support-03`: add actual foot/toe point tracking (weight 20);
  all cases fail within 2.27 seconds.
- `recovery-dynamic-support-04`: include all 30 policy motors; all fail within
  2.59 seconds. The independent-force direct-torque diagnostic
  `recovery-dynamic-support-oracle-01` also fails all cases within 3.74 seconds.
  Target transport is therefore not the sole cause of this controller's failure.
- `recovery-dynamic-support-inverse-01`: replace freely assigned support forces
  with MuJoCo's local inverse-contact derivative. The base forward/inverse forces
  and a small torque perturbation agree within 1e-6 in tests, but larger optimizer
  steps cross contact modes. Nonlinear unactuated-force errors reach 12.22 body
  weights; all cases fail within 2.14 seconds. The corresponding direct-torque
  diagnostic also fails, within 2.02 seconds.
- `recovery-dynamic-support-forward-01`: up to eight bounded refinements, torque
  trust radius 0.1, with backtracking on the actual native forward objective.
  This supplies realizable accelerations, but all cases fail within 1.86 seconds.
  A static diagnostic reveals the objective accepts pelvis acceleration of
  -42.1 m/s² while keeping both foot-body accelerations below 0.3 m/s².
- `recovery-dynamic-support-forward-02`: raise COM/pelvis/trunk tracking weight
  from 1 to 30 while keeping contact weight 20 and motor limits. The same static
  pelvis acceleration improves to +0.25 m/s²; all native holds still fail within
  2.09 seconds. `recovery-dynamic-support-forward-oracle-01` also fails all holds
  within 2.02 seconds with direct substep torque actuation. Improving a static
  objective does not establish a stable closed-loop support controller.

Every completed experiment retains the eight normal 30-second quiet trajectories
exactly, with no teacher activation. This confirms source retention when the
teacher is inactive; it does not validate teacher support or replace the existing
eight 600-second source-policy results. The source remains `ad7f73dc`; the rig,
finite-time-independent reference and Arena export were not changed by this phase.

Thirty focused controller/reward/example tests pass, including native forward
prediction, bounded torque/target conversion, solver isolation, saturated motor
intervals and native reward integration. Each run stores code snapshots, identity
hashes and planned/realized force and acceleration traces. The final decision is
`logs/mimickit-training/recovery-dynamic-support-forward-oracle-01/decision.json`.

No new impact suite ran because gravity support failed. No demonstrations were
replayed, admitted or trained; optimizer updates remain zero. No unseen-hit gate
was consumed. The teacher's failure is not evidence of a finite-reference blocker
or a new failure of the retained standing brain. Contact realizability and target
transport are concrete limitations, but removing them still leaves an unstable
controller objective. Another PPO run on these teacher traces is unsupported.
The next distinct approach is a motion-conditioned tracking policy, first gated
on quiet standing and slow support transfers, then physically admitted bilateral
step demonstrations, before randomized impact fine-tuning. Natural stepping and
recovery remain unachieved.

## Motion-conditioned tracking foundation

The dedicated tracker (`train_motion_tracker.py`) freezes the current source actor
`ad7f73dc` and learns a separate correction branch. Its 643 observations include
the existing physical state and controller queue plus future pose deltas and
velocities at 0.1/0.3/0.6 seconds. No phase or selected-foot command is supplied.
Quiet worlds use the exact source action with no exploration; native checks must
retain the source trajectory exactly. This is an offline motion tracking pilot,
not a ball recovery controller or an Arena-compatible export.

The launcher `scripts/train-motion-tracker.ps1` uses the existing `config.ps1`,
including Envs=2048 and EvaluationInterval=64. The five-minute budget counts
synchronized PPO update time; native evaluation and setup are additional time.
The final checkpoint retains source tensors and stores code, motion, rig and
configuration provenance. Native tracking admission requires 30-second survival,
small root/joint/foot errors, actual requested displacement and a stable terminal
stance; surviving without executing the motion is insufficient.

`motion-tracker-foundation-5min-01` completed 22 updates and 2,883,584 transitions
in 305.66 PPO seconds. Quiet retention passed 8/8 at 30 seconds with identical
source trajectories, but all eight motion trials fell. The initial implementation
fed stationary goals to the source actor while the controller tracked moving
goals. Corrected task observations now match the physical motor reference.

Motion resets also exposed tangent soles with zero measured load and false
capture risk. The task now applies the standing foundation's existing 0.1 mm
contact seating at reset only. `motion-tracker-engine-parity-02` reports a maximum
native/GPU qpos difference of 1.60e-5 over 30 control steps, initial foot-load
difference below 0.015 N and identical initial capture risk. Both feet are loaded.
This smoke check uses eight worlds; PPO retains 2048 worlds.

The zero-correction source still fails all eight trials on the physically checked
2.2 cm bilateral weight-shift curriculum, both before and after contact seating.
Seating fixes the initialization defect, not the motion controller. The corrected
`motion-tracker-foundation-small-5min-02` starts a fresh correction branch on that
smaller curriculum. Motion data are kinematic references with physical checks;
they are not yet dynamically tracked expert demonstrations. Bilateral stepping,
natural stepping and unseen impacts remain subsequent gates.

The corrected small pilot completed 23 updates and 3,014,656 transitions in
312.86 PPO seconds. Its eight native motion trials lasted 2.17–6.50 seconds;
none passed the foundation gate. Quiet retention remained exact. Partial GPU
sampling during updates averaged 39.8%, peaked at 83% and had a 95th percentile
of 78.3%; native validation is CPU work.

`motion-tracker-source-isolation-02` holds a fixed reference for 30 seconds with
the original source actor in all eight worlds, both with tangent reference
coordinates and with the 0.1 mm seated coordinates. These are survival ablations,
not motion tracking admission: the displacement gate still rejects stationary
drift. Replay of the trained motion traces finds 1.68% saturated source actions,
a maximum realized correction of 0.093 and a maximum normalized future feature
of 2.63. The probe does not establish a source-input or correction-authority
blocker. ONNX CPU inference uses one thread to avoid oversubscription.

Four-second training episodes never rehearse a hold beyond the finite motion.
The task now defaults to 30-second training episodes while retaining varied
motion start phases. `motion-tracker-foundation-hold30-5min-03` continues the
corrected small checkpoint with that change. The launcher records episode
duration explicitly. Continuations validate source and serialized policy
contracts; MimicKit restores model/normalizer weights, not optimizer state or
iteration counters. Per-run sample counts therefore represent additional samples.

That continuation improved the best native trial from 6.50 to 11.50 seconds,
with exact quiet retention. `motion-tracker-foundation-hold30-10min-04` adds
45 updates and 5,898,240 transitions in 611.13 PPO seconds. All eight small-motion
trials now survive 30 seconds with stable terminal stances. Root, foot and joint
errors meet their existing gates, but displacement error remains 15.3–21.9 mm,
above the 2.0–6.3 mm limits. The checkpoint is `f70983d3`; foundation admission
remains false. Survival improvement is real, but it does not establish correct
support-transfer execution. A native probe on the existing 4.5 cm medium-motion
reference tests whether this policy generalizes before changing training rewards.

`mirror_step_reference.py` reflects the complete rig pose and contact schedule,
recomputes velocities and repeats physical checks. The calibration reader and
placement gates now support either swing foot while preserving the existing left
default. The right forward/backward diagnostics in
`logs/mimickit-steps/placed-step-right-forward-01` and
`logs/mimickit-steps/placed-step-right-backward-01` pass physical checks, measure
approximately +0.15/-0.15 m placement and mirror body positions within 1 micrometre.
These are mirrored procedural diagnostic references derived from the existing
licensed neutral-pose calibration. They are not recorded natural recovery
demonstrations, and remain explicitly unadmitted for training pending dynamic
tracking. Attribution and source/generator hashes are saved with each reference.

The medium-reference transfer probe survives 5/8 native trials; all fail the full
tracking gate. `motion-tracker-foundation-precision-5min-05` therefore continues
the retained small-motion checkpoint on the physically revalidated small and
medium references, with root reward sigma reduced from 30 to 15 mm and root
weight raised from 0.25 to 0.45. DeepMimic weight becomes 0.35; foot weight stays
0.20. This explicitly penalizes the measured root bias, without changing the rig,
motor limits, delayed controller, quiet path or 30-second hold rehearsal.

Continuation validation allows changes only to reference hashes/duration after
the motion bank repeats physical admission. It rejects changes to stationary
identity, observations, conditioning and control semantics. Changed curriculum
fields and exact reward parameters are recorded in the new run's protocol.
Seven focused tests pass, including curriculum compatibility, rejection of
control/quiet identity changes, quiet action isolation, native reset loads,
controller/future-goal alignment, partial resets and terminal-reference holds.

The precision run completed 26 updates and 3,407,872 additional transitions in
300.23 PPO seconds. Mixed-curriculum survival remains 7/8; the failing medium
phase falls at 3.05 seconds. Displacement errors show no consistent improvement.
Quiet trajectories remain exact. The retained offline hold candidate is
`motion-tracker-foundation-hold30-10min-04/model.pt` (`f70983d3`), with verified
8/8 small-motion survival. The newer `e11cd166` checkpoint is preserved as a
precision experiment, not promoted. Both still fail motion tracking admission.

Five bounded pilots total 18,087,936 transitions and 1,839.41 synchronized PPO
seconds. This includes the discarded first branch; the subsequent four runs form
one compatible continuation chain. No dynamic stepping examples were admitted,
no new impact curriculum ran and no Arena policy changed during this phase.
The new tracker requires a 643-input adapter; the current Arena expects 363.

The correction branch currently omits pending motor targets and has no explicit
goal-relative pose-error features. These are concrete architectural omissions,
not proven causes of the residual tracking bias. The next justified experiment
is to add and isolate that feedback while preserving the retained checkpoint's
behavior, before another reward-only continuation. The decision, run hashes,
reference provenance and next action are saved in
`logs/mimickit-training/motion-tracker-foundation-precision-5min-05/decision.json`
and `logs/mimickit-training/recovery-loop/state.json`. Natural leg stepping and
unseen-hit recovery remain unvalidated.

## Explicit pose and command-queue feedback

`mimic_motion_feedback_tracker_v2` adds a separate 170-input feedback branch:
122 pending motor-target position/velocity/validity values and 48 pose errors.
Errors compare the nearest saved future pose (one control step ahead) with the
current root position, tangent/normal orientation and hinge positions. Scales
are 30 mm, 1 and 0.25 rad respectively. Command-queue values use the frozen
source statistics with a 0.05 standard-deviation floor; invalid slots mask
their targets to zero and retain explicit validity bits.

The original source and trained v1 correction are frozen. The feedback branch's
last layer starts at zero and contributes before the existing correction tanh.
The external observation contract remains 643 inputs; the actor version changes
separately. Migration accepts only the seven new feedback tensors and checks
all retained checkpoint shapes and dtypes. Same-curriculum continuation must
reproduce the retained native trace exactly before PPO begins.

`motion-feedback-small-5min-06` starts from `f70983d3`, using the same small
reference and original reward (30 mm root sigma, weight 0.25) to isolate the
feedback change. Envs=2048, EvaluationInterval=64 and 30-second training episodes
are retained. Initial native poses, velocities, target poses, actions, active
masks and foot loads match the retained trace with zero difference. Quiet
standing passes. Ten focused tests pass, including zero-output action parity,
queue masking, scaled pose-error visibility and strict checkpoint migration.
Only the feedback branch is trainable; this pilot does not train ball recovery
or admit stepping demonstrations.

### Completed feedback pilot

Checkpoint `bf05563e` completed 29 updates, 3,801,088 transitions and 301.63
synchronized PPO seconds. Native assessment took another 175.66 seconds.
GPU sampling during the optimization window measured 42.8% mean utilization,
86% peak and 84% p95; checkpoint saves and bookkeeping are included.
The source and prior correction tensors remain frozen. Quiet standing passes
8/8 trials with exactly zero pose difference from the source. All eight motion
trials survive 30 seconds and finish stable, but none passes the complete
tracking gate. Absolute displacement error worsens in seven trials, reaching
21.3–27.9 mm against case-specific limits of 2.0–6.3 mm.

The saved trajectory audit separates mean bias from remaining variation:
mean forward bias increases from 15.4 to 21.4 mm. Subtracting each trajectory's
mean error reduces centered motion error in every trial, but leaves 4.3–12.1 mm
of error; lateral correlation also declines in seven trials. These measurements
do not establish accurate motion tracking or natural stepping.

A separate eight-world, four-second identical-action native/GPU replay finds
maximum body-position disagreement of 0.34 mm and reward disagreement of
0.000609. Root body origin equals root qpos in both engines. This audit does
not establish a reward or root-coordinate mismatch. Its auxiliary feedback
feature comparison uses unit queue-normalization statistics rather than the
checkpoint's statistics; it is not full checkpoint feature-parity evidence.

Retain the previous offline hold checkpoint `f70983d3`; preserve this candidate
for analysis. Arena remains on `ad7f73dc`. No dynamic demonstrations are
admitted, and stepping and unseen-hit recovery remain unvalidated. Next,
diagnose the forward equilibrium bias and remaining motion-shape error using
isolated target holds and reward-component measurements before further
optimization. The decision and provenance are saved in
`logs/mimickit-training/motion-feedback-small-5min-06/decision.json` and loop
state. Pending-target and pose-error feedback alone did not resolve admission.

## Frozen-pose equilibrium and reward audit

`motion-equilibrium-audit-01` separates reference motion from learned equilibrium
using eight fixed phases of the small weight-shift reference, including both
endpoints. Target velocities are zero throughout; the same reference-relative
delayed PD, gravity, contact seating and actor inputs are retained. Every pose
is tested for 30 seconds. These are diagnostic poses, not stepping examples.

The source, retained tracker and feedback tracker each survive all eight holds
with stable endings. Their mean final signed forward errors are -3.7, +14.5
and +20.4 mm respectively. Mean absolute forward errors are 5.2, 14.5 and
20.4 mm; the source's worst pose still has an 18.2 mm backward error, so it
does not establish accurate tracking of every target. The source's foot errors
are 2.1–3.7 mm; the learned trackers' errors are 4.1–9.8 mm. Zero residual
control falls in every pose within 1.85–2.03 seconds. Active learned balance
is required, and the positive tracker bias persists without a moving reference.
This isolates a learned-equilibrium problem; it does not establish that all
kinematic references are dynamically realizable or naturally executed.

Reward attribution uses the upstream DeepMimic implementation with one-hot
component weights. Its reconstructed sum passes at 1e-7 tolerance, and an
independent native-backend check at the first and last frame of all six traces
differs by at most 1.20e-7. On the saved moving trials, mean total reward falls
from 0.9001 to 0.8668. Root tracking falls from 0.7213 to 0.5760 while foot
tracking improves from 0.8995 to 0.9152. The root loss outweighs that foot gain;
the existing reward does not rank this regression as an improvement. Joint
pose and velocity terms remain near 0.99–1.00, making them weak measures of
millimeter-scale root accuracy. Fixed-hold mean rewards likewise decline:
0.9619 for source, 0.9283 for retained and 0.8824 for feedback.

The 5-minute feedback run supplied only 30.94 simulated seconds per world
(3,801,088 transitions / 2,048 environments at 16.668 ms), with stochastic
motion actions. Its final training log reports 8.90-second mean completed
episodes; this mixes different worlds and is not a quiet-standing gate. The
1.067-second rollout and discount 0.99 place an approximately 1.66-second
e-folding horizon on reward. These measurements motivate better stationary
coverage and checkpoint selection; they do not prove a critic or horizon bug.

Next experiment: add varied fixed-pose worlds alongside slow moving references,
so inaccurate learned equilibria receive direct rehearsal from the first
rollout. Keep an immutable quiet subset, use a dense position-accuracy term,
and select checkpoints by native fixed and moving tracking performance.
Start from the retained tracker, keep both feet eligible, and retain the current
physical target and contact contract. Do not offset reference positions to
hide drift. Validate the curriculum and reward change before a bounded optimizer
run; accurate tracking remains necessary before stepping examples or new hits.

This audit performs zero optimizer updates and makes no Arena change. Reports,
source snapshots, traces, checkpoint hashes and the reward reconstruction check
are saved under `logs/mimickit-training/motion-equilibrium-audit-01`.

## Fixed-pose correction curriculum

`motion-equilibrium-fixed-5min-07` rehearses 25% immutable quiet worlds, 25%
fixed reference poses and 50% moving references. Fixed phases are sampled at
reset, held indefinitely and have zero target velocity. Neither a chosen foot
nor a reference phase is added to actor inputs. Native fixed evaluation covers
eight phases including endpoints, with the same contact seating and delayed PD.

Root tracking uses `1 / (1 + sum(Huber(error / 0.005 m)))`, with weight 0.45.
The positive reward has a useful error gradient outside the accuracy band,
without an unbounded negative living cost that might incentivize termination.
DeepMimic weight is 0.35 and foot tracking remains 0.20 at 40 mm sigma. The
quiet subset keeps its original reward and frozen source actions. The source
and prior correction remain frozen; only the feedback branch is trainable.

The pilot initializes from `f70983d3`, with a zero-output feedback branch.
Training is bounded to 300 synchronized PPO seconds, at Envs=2048 and
EvaluationInterval=64. Initial and final native checks score both eight moving
and eight fixed 30-second trials. `best.pt` selects the lower mean displacement
error divided by each case's existing limit, provided quiet actions remain
exact and all sixteen trials survive with two seconds of stable final support.
Initial weights remain selected when the final policy is worse or loses support.
Full admission additionally requires every fixed and moving tracking gate.
Intermediate checkpoints are preserved but are not automatically validated.

Fourteen tests pass, including held-target invariance, partial reset isolation,
real double support at fixed evaluation resets, dense-error ranking/gradient,
and selection rejection after a failed hold. This curriculum does not admit
dynamic stepping examples or alter Arena. Five minutes refers to optimizer
time; native checks add wall time.

### Completed fixed-pose pilot

The pilot completed 33 updates, 4,325,376 transitions and 300.33 synchronized
PPO seconds. Initial and final native assessment took 285.00 seconds combined.
GPU utilization during optimization averaged 44.5%, with 89% peak and 84% p95;
no native assessments interrupted this optimization window.

Quiet standing remains exactly equal to the frozen source. Source and prior
correction tensors are unchanged. All sixteen final trials survive 30 seconds
with stable endings, but all sixteen fail full tracking admission. Mean moving
displacement error increases from 18.23 to 20.60 mm, worsening seven cases.
Mean fixed-pose displacement error increases from 13.25 to 18.92 mm, worsening
every case. The combined normalized error score rises from 6.0864 to 7.8607.

`model.pt` (`b25a8d45`) preserves the trained candidate. `best.pt` (`27f0e051`)
selects initial, behavior-preserving migrated weights from `f70983d3`; it is
not an improved actor. The final candidate's new reward is also lower on native
trajectories: moving 0.6455 to 0.6289, fixed 0.6818 to 0.6285. A post-run audit
uses the same upstream reward terms and the selected dense root mode. Changing
stationary coverage and position shaping did not resolve the discrepancy
between surrogate optimization and deterministic native tracking performance.

Do not repeat this unchanged curriculum for longer based on survival alone.
Next, audit PPO advantages, critic predictions and stochastic versus
deterministic returns on the same saved policy and physical conditions before
another update. Intermediate checkpoints remain available for that audit but
are not admitted. The retained offline actor and Arena selection remain unchanged;
natural stepping and unseen-hit recovery are unvalidated. The run's decision,
reward comparison, source snapshots, hashes and next action are saved with its
report and in loop state.

## Explicit latest-weight Arena preview

At the user's request, Arena now previews final `b25a8d45` weights from
`motion-equilibrium-fixed-5min-07/model.pt`, rather than selected initial
`best.pt`. `export_motion_tracker_preview.py` embeds the complete tracker in
ONNX behind the existing 363-input stationary interface. It reconstructs all
three fixed-pose command deltas relative to stationary case zero, supplies zero
target velocities and sets motion request to one. The latest correction and
feedback therefore remain active. The physical goal is the existing admitted
stationary case-zero pose; this is a standing preview, not playback of the
weight-shift reference. There is no selected stepping foot or recovery clock.

Native and Godot both survive a 30-second adapted quiet trial. ONNX maximum
action error is 1.35e-7; replay observation, action and pose differences are
4.10e-7, 1.17e-7 and 2.96e-8 respectively. These are adapter/inference checks,
not evidence of trained ball recovery or full tracking admission. Initial preview settings
were 8 kg, 2 m/s, randomized body targets and angles, every 30 seconds without
a time limit. The prior `ad7f73dc` bundle is preserved for comparison.

The scene explicitly identifies the new candidate and incomplete impact
validation. Export provenance and parity evidence are saved in the run's
`arena-preview/export_report.json`; the full 643-input training contract remains
unchanged in the parent run directory.

The actual Arena scene also completes a 40-second headless smoke run with two
confirmed randomized arm impacts and no fall. This checks bundle loading and
continued physical operation; one randomized run is not unseen-hit admission.
The scene and ball viewer selection point to the new bundle, with hashes and
previous-bundle provenance in `arena-preview/arena-selection.json`.

## Arena independent projectiles and increased speed (2026-09-30)

At the user's request, Arena now launches at 3 m/s horizontal speed (+50%).
B uses the next unused projectile in a 64-ball pool. Previously launched balls
remain physical, including ball-to-ball collisions; none are teleported or recycled.
R resets the character and clears all projectiles. The HUD reports occupied slots
and prompts a restart when full. Automatic firing uses the same pool.

`arena_ball_pool.py` derives `arena-preview/dummy_ball_arena.xml` from the unchanged
single-ball model. Training, checkpoint weights, mass (8 kg), radius (9 cm), and
the automatic 30-second firing interval are unchanged. All 64 projectiles are
excluded from character mass and COM sensors; impacts are counted per projectile.

Headless native regression confirms three simultaneous impacts, exact preservation
of earlier projectile state on later launches, 3 m/s launch speed, zero character
mass/COM differences, pool exhaustion without mutation, and complete reset.
The original 30-second quiet replay is exactly preserved and 19 managed Mimic
tests pass. The actual 40-second Arena smoke run loads successfully and confirms
two hits; the dummy falls at 32.403 seconds after the second hit. This is a scene
integration check, not a recovery gate at the increased speed. Evidence is saved
in `arena-preview/ball-pool-replay.json` and `arena-ball-pool-smoke.log`.

## User-requested heavy Perturb continuation (2026-09-30)

Run: `logs/mimickit-training/amp-heavy-10kg-4mps-30min-01`. Continues the latest
actual Perturb actor, `amp-useful-placement-15min-01/model.pt` (`3fa62c5d`), with
its matching complete AMP sidecar and existing exact 8x600-second quiet evidence.
The isolated projectile model increases mass to 10 kg; radius remains 0.09 m.
Launch speed is fixed at 4 m/s. Training retains random angles/all body targets,
one hit per 30-second episode, 25% quiet episodes, AMP demonstrations/mirroring,
and the latest `step_v7` peak-articulation placement measurement. Envs=2048 and
EvaluationInterval=64. The trainer permits 32 stalled updates, enough to cover
a full 30-second episode before diagnosing stalled optimization.

Completed 121 updates and 15,859,712 transitions in 1,806.36 synchronized
training-update seconds (30.106 minutes). Trainer wall time was 2,323.156 seconds
(38.719 minutes), including its built-in calibration and assessments. No separate
tests, validators, Godot runs or compilation were performed for this continuation.

Fixed 10 kg / 4 m/s recovery decreases from 3/32 to 2/32; survival remains 3/32.
Recovered-with-step counts remain 2. Final quiet standing passes all eight
30-second cases; no new 600-second final-candidate validation was run. Automatic
selection retains update 64 (`6cb67cb9`): 3/32 recovered, 3/32 survived, three
recovered-with-step cases. Its separate built-in unseen suite gives 8/64 recoveries
and 10/64 survivors; those results describe selected `best.pt`, not final weights.
Natural stepping and final-candidate unseen-hit admission remain unestablished.

At the user's explicit request, Arena selects final `model.pt` (`4c26be5d`) even
though recovery regressed. ONNX export maximum action difference is 3.28e-7.
The exported world uses 64 independent 10 kg balls at 4 m/s, keeping the editable
repeating automatic interval (30 seconds default), B countdown reset, hidden
unused balls, and R clearing the pool. Restart the scene to load the selection.
The prior `b25a8d45` preview remains available. Hashes, selection and GPU sampling
scope are recorded in this run's `completion-state.json` and preview provenance.

### Second requested 30-minute heavy window (completed)

The user requested another 30 training minutes from final `4c26be5d` toward
one hour cumulative policy-training time, retaining the same physical task,
reward and AMP demonstrations. The first startup, `amp-heavy-10kg-4mps-30min-02`,
stopped before PPO: recalibration could not improve its balanced mapping.
The loaded source already classified 100% of held-out demonstrations and 95.99%
of generated histories correctly, with balanced BCE 0.3634. Recalibration's
normalization change reduced generated accuracy to 91.59% and raised BCE to
0.3726. Actor/critic tensors were unchanged; this startup counts zero policy
training seconds and its artifacts are retained.

The replacement `amp-heavy-10kg-4mps-30min-02-continue` loads the original complete
source AMP sidecar and preserves its discriminator normalization by setting
startup calibration steps to zero. Ordinary discriminator/PPO updates remain
enabled. This isolates duration continuation on the unchanged 10 kg / 4 m/s
task, with a new seed (300981), 2048 environments and evaluation every 64 updates.
Final preview selection is recorded below; no third run is authorized.

### Second heavy-window completion and Arena selection (2026-09-30)

`amp-heavy-10kg-4mps-30min-02-continue` completed 127 updates and 16,646,144
transitions in 1,807.406 synchronized training-update seconds (30.123 minutes).
Trainer wall time was 2,357.000 seconds (39.283 minutes), including built-in
assessments. Together with the first run, actual policy-training time is
3,613.766 seconds (60.229 minutes); combined trainer wall time is 4,680.156 seconds
(78.003 minutes). The failed recalibration startup adds zero policy-training time.
This is a warm start with the full AMP sidecar, not continuous optimizer/RNG state.

Paired fixed heavy-hit recovery improves from 2/32 to 3/32 and survival from 3/32
to 5/32. Recovered-with-step counts remain 2. Final quiet standing passes all eight
30-second native cases; no new 600-second final validation was run. The trainer
selects update 127 `best.pt` (`11830846`); its 12-second unseen suite gives 23/64
recoveries/survivors, including 21 recovered-with-step cases. These measurements
are specific to best.pt and do not establish admission for final model.pt.
Natural stepping remains unestablished.

Arena explicitly selects final model.pt (`9d109137`), preserving the
previous `4c26be5d` bundle. Export maximum action difference is
3.58e-07. The 64 independent balls remain 10 kg at 4 m/s,
with hidden unused balls, B countdown reset, an editable repeating auto-launch
interval defaulting to 30 seconds, unlimited duration, and R clearing the pool.
Restart the scene to load the candidate. No standalone tests, validators, Godot
runs or compilation were performed. GPU sampling covers 1038 whole-device
samples (mean 35.39%, p95 76%, peak 99%),
including native assessment idle time; this is not exclusive PPO utilization.
Provenance and outcomes are saved in completion-state.json and arena-selection.json.
The recurring completion automation remains paused at the user's request.

## Spring-free direct-torque Stand foundation (2026-09-30)

Run `stand-torque-spring-free-30min-01` starts a fresh MimicKit AMP actor, critic, discriminator and normalization. All passive rest-pose springs are removed. Neck position actuators become zero-command torque motors; the 30 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 184 updates / 24,117,248 transitions in 1103.014 synchronized training-update seconds, excluding built-in assessments. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandTorqueArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

### User-shortened Stand window and primary Arena selection

The user reduced this window to 15 minutes and requested MimicStandArena. The active runner had no live budget input; the agent missed the cutoff while implementing completion. It stopped at update 184: 1,103.014 training-update seconds (18.384 minutes), 24,117,248 transitions. This is not a completed 15-minute run. Final e1d4a99f is selected in MimicStandArena with unlimited duration. Initial survival was 0.317–0.400 seconds; final survival is 1.767–2.950 seconds, with 0/8 settled 30-second passes. No long-hold gate was attempted. The earlier scene is preserved in arena-preview/prior-stand-arena.tscn. Future runs read budget-change.json between updates to allow a cooperative shorter budget. No compilation was performed.

## Spring-free direct-torque Stand foundation (2026-09-30)

Run `stand-torque-spring-free-45min-02` continues final `e1d4a99f` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck position actuators become zero-command torque motors; the 30 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 396 updates / 51,904,512 transitions in 2705.225 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 63.471 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

## Spring-free direct-torque Stand foundation (2026-09-30)

Run `stand-torque-spring-free-15min-03` continues final `eb3011ea` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck position actuators become zero-command torque motors; the 30 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 90 updates / 11,796,480 transitions in 598.989 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 73.454 minutes. Final settled standing: 4/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

The user stopped this 15-minute continuation early at update 90: 598.989 synchronized training-update seconds (9.983 minutes), bringing cumulative training to 73.454 minutes. Final short standing improved from 0/8 to 4/8; the full standing and natural-movement gates remain unpassed. The final actor, not best.pt, is selected in MimicStandArena.

### Stand actor in the Perturb Arena (2026-09-30)

At the user's request, `MimicPerturbArena.tscn` selects final Stand actor `708a8ac0`
in `stand-torque-spring-free-15min-03/perturb-preview`. The existing ONNX actor,
reference and spring-free character are reused unchanged. The projectile world
appends 64 independent 10 kg balls of radius 0.09 m, retaining 4 m/s launches,
hidden-until-launched visuals, repeating editable auto-launches and manual B
countdown resets. Direct policy motors are enabled; there are no passive springs
or standing-pose servo. The source remains a Stand-trained policy with 4/8 short
quiet passes, without impact-recovery training or unseen-hit admission. The prior
passive Arena selection and scene are saved in the new preview. No standalone
tests, compilation, Godot execution or additional training were run.

## Whole-body torque Stand objective correction (2026-09-30)

The folded-body screenshots exposed two gaps: reward and standing admission
measured pelvis orientation/height, and converted neck motors remained outside
policy control. Historical `708a8ac0` 4/8 results describe the pelvis/support gate.

The new `mimic_stationary_stand_torque_v2` contract has 33 torque actions and 306
observations. Neck torques are learned. No passive springs, position servos or
external support are added. A soft posture multiplier rewards spine/chest/head
vertical-axis orientation, relative clearance above the pelvis and upper-body
joint preferences. Hips, knees and either foot remain free. The new gate checks
20/20/30-degree upper-body tilt and 0.05/0.06/0.10 m clearance deficits alongside
the existing pelvis, velocity and bilateral-contact conditions.

The next trainer can warm-start `708a8ac0` and its AMP sidecar with an explicit
30-to-33 action and 300-to-306 observation mapping. Existing network outputs and
normalization are preserved; new neck means and input columns begin at zero.
`TorqueStandWholeBodyTraining` keeps Envs=2048 and EvaluationInterval=64.
No training, tests, compilation or Godot execution were run. Both Arenas still
select the existing actor; behavioral changes require a newly trained checkpoint.


## Whole-body direct-torque Stand window (2026-10-01)

Run `stand-torque-whole-body-15min-04` continues final `708a8ac0` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 159 updates / 20,840,448 transitions in 901.692 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 88.482 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

The corrected whole-body gate scores the starting and final actors 0/8. Initial
30-second survival is 4/8; final survival is 3/8. At update 64 every case falls
within 2–3 seconds; update 128 recovers three 30-second survivals but no upright
passes. The final actor `ab234b41` is selected as requested, not the retained
best checkpoint. The previous Arena selection is saved in
`arena-preview/prior-stand-arena.tscn`. ONNX conversion maximum action error is
1.79e-07. GPU samples average 57.43% whole-device activity (p95 75%, peak 99%),
including built-in assessment periods. No standalone benchmark rerun was made.

Higher sampled returns and improved upper-body diagnostics do not establish
upright stability. The final sampled chest tilt remains about 35 degrees and
head-clearance deficit about 0.13 m. A further unchanged training window is not
justified by this result. The next proposed investigation is a staged posture
objective that retains survival while improving extension, with per-body native
failure diagnostics before another training decision. No further run is started.

## Staged posture implementation (2026-10-01)

`staged_v3` preserves a balance floor instead of multiplying all balance credit
by upper-body posture. Pressure is 10/25/50/70%, advancing only after two
consecutive native assessments pass all eight current-stage cases. A bounded
inverse-square posture score keeps severe folding distinguishable. The final
upright gate is fixed; relaxed stage readiness is never standing admission.

Quiet retention compares each initially surviving and balance-settled case.
Regressions reduce posture pressure; repeated loss stops the run and preserves
the initial, retention, intermediate and final checkpoints with full AMP sidecars.
No checkpoint is silently substituted for final model.pt. Native assessment
records per-body pose, foot loads, contact-based fall reasons and failed criteria.

The new `TorqueStandStagedPostureTraining` preset prepares 15 minutes from final
`ab234b41`, explicitly records the reward transition, and retains Envs=2048 and
EvaluationInterval=64. Direct motor control, neck actions and body physics remain
unchanged. No training, standalone tests, compilation, Godot execution or Arena
selection was performed during this implementation.

### Staged window outcome and latest Stand Arena selection

`stand-torque-staged-posture-15min-05` completes 903.667 synchronized training-update
seconds, 178 updates and 23,330,816 transitions. Logged trainer elapsed time is
945.766 s. Final `f4aeaa6e` is selected in StandArena, with the prior `ab234b41`
bundle preserved. Initial-to-final survival and balanced settling improve from
3/8 to 7/8; fixed upright standing remains 0/8. Initial quiet-retention cases
remain intact. Update 128 achieves 8/8 survival/balance, but the final actor loses
case 7 after 2.884 s due to low pelvis height. The better intermediate checkpoint
and AMP sidecar are preserved separately; final selection remains explicit.

The run stays at the balance stage (10% posture pressure). Update 64 has 7/8
balanced cases and update 128 has 8/8, providing only one complete scheduled
readiness pass. Higher-pressure stages are not trained. The native per-body
diagnostics show improved chest tilt but worse head tilt for the three initially
balanced cases; all final surviving cases fail upper-body tilt and head-clearance
criteria. The balance floor helps endurance without establishing natural upright
posture. No 600-second hold is attempted because the final fixed gate fails.

GPU sampling averages 56.87% whole-device activity (p95 73%, peak 99%) including
built-in assessments. No standalone tests, compilation or Godot execution were
run. PerturbArena retains its earlier Stand actor `708a8ac0`; no new Perturb
training is started.

## Spring-free direct-torque Stand window (2026-10-01)

Run `stand-torque-staged-posture-15min-05` continues final `ab234b41` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 178 updates / 23,330,816 transitions in 903.667 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 103.543 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

Staged posture ended at `balance`. Stop reason: `training_update_time_budget`. Final quiet-retention result: `{'passed': True, 'lost_cases': []}`. The fixed upright gate remains unchanged. Per-body tilt/clearance, foot loads, contacts and failed criteria are recorded in report.json. The retention checkpoint is preserved separately from final model.pt.

## Spring-free direct-torque Stand window (2026-10-01)

Run `stand-torque-worst-posture-30min-06` continues checkpoint `295ac2b4` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 192 updates / 25,165,824 transitions in 810.906 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 112.986 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

Staged posture ended at `balance`. Stop reason: `quiet_retention_regression`. Final quiet-retention result: `{'passed': False, 'lost_cases': [7]}`. The fixed upright gate remains unchanged. Per-body tilt/clearance, foot loads, contacts and failed criteria are recorded in report.json. The retention checkpoint is preserved separately from final model.pt.
## Worst-body posture attempt outcome (2026-10-01)

`stand-torque-worst-posture-30min-06` resumed measured update 128 (`295ac2b4`)
and matching full AMP state (`6a4be764`), with tensor identity to the source run's
recorded best state. Initial native assessment reproduced 8/8 balance, 0/8 upright.
`worst_v4` replaces averaged physical posture with the largest normalized absolute
torso/head tilt or reference-height deficit. The licensed style prior, spring-free
33-motor torque plant, 2,048 environments, evaluation interval 64 and retention
guard remain. Intermediate checkpoint lineage excludes the source's later updates.

The guard stopped this requested 30-minute attempt after 810.906 training seconds
(13.515 minutes), 192 updates and 25,165,824 transitions. Update 64 retained 8/8
balance; update 128 lost cases 3/6, and update 192 lost case 7. Final balance is 7/8,
upright remains 0/8, and mean native head tilt worsened from 57.999 to 74.542 degrees.
No higher posture-pressure stage or conditional 600-second gate was reached.
Final actor `7369bb90` was exported and selected as an experimental StandArena
preview; neither natural movement nor quiet retention passed. Source/best/retention
states and the previous Arena bundle are preserved. No standalone tests or build
were run. This outcome does not justify repeating an unchanged long window.

## Spring-free direct-torque Stand window (2026-10-01)

Run `stand-torque-short-upright-15min-07` continues checkpoint `295ac2b4` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 154 updates / 20,185,088 transitions in 901.393 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 114.494 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

Staged posture ended at `upright_2s`. Stop reason: `training_update_time_budget`. Final quiet-retention result: `{'passed': True, 'lost_cases': []}`. The fixed upright gate remains unchanged. Per-body tilt/clearance, foot loads, contacts and failed criteria are recorded in report.json. The retention checkpoint is preserved separately from final model.pt.

## Short-horizon upright pilot outcome (2026-10-01)

Run `stand-torque-short-upright-15min-07` resumes preserved actor `295ac2b4` and its full AMP state. The `short_upright_v5` reward removes the posture-independent balance floor. Folded training trajectories terminate; physical actuation remains 33 learned torques with no springs, motor pose servo, fixed stepping foot or leg-position reward. Training starts with two-second holds, while independent native 30-second quiet retention remains mandatory.

Completed 901.393 synchronized training seconds (15.023 minutes), 154 updates and 20,185,088 transitions; logged trainer elapsed 947.094 seconds. Native balance/survival temporarily drops to 4/8 at update 64, returns to 8/8 at update 128, and remains 8/8 at completion. Every source quiet case is retained. Final mean head tilt improves 57.999 to 24.873 degrees and chest 30.064 to 24.004; spine remains 22.195 degrees. All eight final cases survive, so the comparison is not shortened by falls.

Fixed upright admission remains 0/8 because spine/chest exceed 20 degrees. Two-second readiness also remains 0/8 (required final stable interval 0.5 seconds). No horizon extension or conditional 600-second assessment occurs. This is partial posture improvement, not natural upright or impact-recovery admission. Final `1c9238f8` is selected as experimental in StandArena; source, retention and previous bundles remain preserved. GPU samples average 55.98% (p95 70%, peak 100%), including built-in assessments. No standalone tests, compilation or Godot execution were run; no further window was started.

## Spring-free direct-torque Stand window (2026-10-01)

Run `stand-torque-whole-pose-15min-08` continues checkpoint `1c9238f8` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 153 updates / 20,054,016 transitions in 904.185 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 129.563 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

Staged posture ended at `upright_2s`. Stop reason: `training_update_time_budget`. Final quiet-retention result: `{'passed': False, 'lost_cases': [1]}`. The fixed upright gate remains unchanged. Per-body tilt/clearance, foot loads, contacts and failed criteria are recorded in report.json. The retention checkpoint is preserved separately from final model.pt.

## Whole-body pose pilot outcome (2026-10-01)

`stand-torque-whole-pose-15min-08` continued `1c9238f8` and its full AMP sidecar (`bed78191`) with `whole_pose_v6`, 2,048 environments and evaluation interval 64. Completed 904.185 synchronized training seconds (15.070 minutes), 153 updates and 20,054,016 transitions; logged trainer elapsed 955.594 seconds. Final `7f270019` is selected as an experimental StandArena preview, with the previous bundle and checkpoints preserved.

Native balanced survival regresses 8/8 to 7/8; final case 1 falls at 4.000 seconds. Upright gate and two-second readiness remain 0/8, so no horizon extension or conditional 600-second assessment occurs. For the seven matched survivors, terminal left-shoulder reference error improves 85.236 to 48.755 degrees, while right shoulder worsens 81.324 to 82.699. Spine/chest/head reference errors remain 39.801/43.662/68.914 degrees finally (initial 39.788/44.293/68.912). Regional improvement does not establish coherent posture or natural movement.

Read-only controller and normalization inspection confirms 33 motor channels, including neck, with action mean zero and scale one. No actuator exclusion or zero action-normalization scale explains the persistent neck error. Actual torque/command saturation and joint-limit occupancy were not measured; these are the next proposed diagnostic. GPU samples average 54.51% (p95 72%, peak 99%), including intrinsic assessments. No standalone tests, compilation, Godot execution or additional training were run.

## Motor diagnostic and prepared corrective reward (2026-10-01)

Explicitly authorized diagnostic `stand-torque-motor-diagnostic-01` compares the final exported actors `1c9238f8` and `7f270019` on identical spring-free plants and eight fixed quiet poses for up to 30 seconds. Source survival is 8/8; latest is 7/8, with case 1 failing at 3.984 seconds. Among the seven paired survivors, every spine and neck axis stays near its joint limit for 100% of samples after five seconds and receives torque directed farther into that limit throughout. All 33 applied actuator forces exactly match their controls. Latest neck mean torques are +6.366/+2.110/-2.157 Nm, within the 18 Nm policy authority. This establishes the end-stop bracing behavior; constraint forces include contacts as well as limits. Raw traces and hash-bound reports remain preserved. No training or Arena change occurred during the diagnostic.

User-approved code now provides `limit_aware_v7`: an additive dense pose score for each of spine/chest/head, alongside balanced whole-body imitation, and a reward penalty for delayed torque pushing the nine upper-body hinges into nearby limits. The five-degree proximity ramp penalizes outward torque only; physical limits, zero springs, damping, observations and bounded direct torque control remain unchanged. Native assessment logs mean per-joint proximity and outward drive without changing admission gates. The new initial coefficients are unvalidated.

Prepared `TorqueStandLimitAwarePilot` uses the latest `7f270019` and matching AMP sidecar, five synchronized training-update minutes, seed 300999, Envs 2048 and EvaluationInterval 64. The curriculum starts with two-second holds and retains the independent quiet guard. The v6-to-v7 reward change is recorded explicitly; the optimizer/RNG stream starts fresh as in earlier warm starts. No corrective run, test, compilation or Godot execution was started. StandArena remains on `7f270019`.

## Limit-aware five-minute pilot outcome (2026-10-01)

Authorized run `stand-torque-limit-aware-5min-09` completed 300.374 synchronized training-update seconds (5.006 minutes), 50 updates and 6,553,600 transitions. Trainer elapsed time was 326.422 seconds. Source `7f270019` and its full AMP sidecar were warm-started with a new optimizer/RNG stream, Envs 2048, EvaluationInterval 64 and `limit_aware_v7`. Final actor `efaf60dc` and AMP state are preserved; experimental export is selected in `MimicStandTorqueArena.tscn`. Primary StandArena remains on `7f270019`. Required export conversion parity is 1.788e-7 maximum action error; no standalone tests, new diagnostic, compilation or Godot execution ran.

Initial-to-final balanced survival decreases 7/8 to 5/8; lost cases are 2 and 6, failing at 8.551 and 2.584 seconds. Upright gate and two-second readiness remain 0/8. Fifty updates do not reach the scheduled evaluation interval, so only mandatory initial/final checks ran; no scheduled retention stop, horizon extension or conditional 600-second assessment occurred.

For paired survivors 0/3/4/5/7, mean upper-body outward normalized drive decreases 0.202108 to 0.140765 (30.35%), while mean five-degree proximity ramp remains high (0.982602 to 0.935784). This ramp is not an occupancy fraction. Terminal local spine/chest/head errors are initially 39.726/44.353/68.914 degrees and finally 39.610/42.397/73.392; mean head world tilt worsens 24.756 to 33.741 degrees. Lower outward torque did not produce natural pose or quiet retention. Whole-device GPU samples average 53.09% (p95 69%, peak 99%), including intrinsic assessments. No additional training was started. Proposed next hypothesis: penalize sustained limit occupancy in addition to outward drive, so reducing torque without leaving the stop region cannot satisfy the corrective objective. This change has not been implemented or authorized.

## Prepared sustained limit-occupancy objective (2026-10-01)

The user subsequently authorized implementation. New immutable mode `limit_occupancy_v8` preserves the v7 reward terms and adds `0.25 * mean(proximity squared)` as a cost for the nine spine/chest/neck hinges. Proximity uses the existing five-degree margin. The cost is independent of motor command and is charged at every occupied control step; it supplies state feedback even when the actor reduces outward torque. It adds no dwell clock, forced pose, observation, actuator intervention or changed physical limit. Historical v7 reward metadata is preserved.

Training logs add occupancy cost and near-limit hinge fraction. Native assessments add per-joint mean near-limit fraction and mean/final occupancy cost through each case's finite lifetime. These diagnostics do not change admission gates or establish naturalness. New coefficient 0.25 remains unmeasured.

Prepared `TorqueStandLimitOccupancyPilot` uses final `efaf60dc` with its matching full AMP sidecar, seed 301000, five synchronized training-update minutes, Envs 2048 and EvaluationInterval 64. Source measured balance/upright is 5/8 and 0/8. The recognized v7-to-v8 transition records explicit reward-change permission; optimizer/RNG starts a new window. Initial/final native checks remain mandatory even if the short budget ends before update 64. No new training, tests, compilation, Godot execution or Arena change ran during this implementation. The preset is ready for separately authorized execution.

## Sustained occupancy pilot failure (2026-10-01)

Authorized `stand-torque-limit-occupancy-5min-10` continued `efaf60dc` and AMP sidecar `a83fed47`, with the unchanged 33-motor spring-free plant, `limit_occupancy_v8`, seed 301000, Envs 2048 and EvaluationInterval 64. Completed 304.938 synchronized training-update seconds (5.082 minutes), 51 updates and 6,684,672 transitions; trainer elapsed was 320.516 seconds. Final `f7f089d8`, full AMP state and source/best checkpoints are preserved. Required final export conversion error is 2.980e-7. The experimental torque Arena points to this export; primary StandArena remains on `7f270019`. No standalone tests, added physical diagnostic, compilation or Godot execution ran.

Native balanced survival regresses 5/8 to 0/8. All eight final cases fall after 1.400 to 2.417 seconds with pelvis below the failure threshold; cases 0 and 5 also have non-foot ground contact. Upright admission remains 0/8. Best native rank is iteration zero, retaining the source's 5/8 result. Only two cases survive the two-second final check, with zero upright cases. The short budget ends before update 64, so there is no scheduled retention stop or horizon advancement. No 600-second hold ran. No pair survives both full assessments, preventing a valid paired-survivor posture/occupancy improvement claim.

The final rollout snapshot appears improved (15.70% near-limit hinge samples; spine/chest/head local reference errors 21.072/20.284/20.493 degrees), but average completed training episodes last 43.613 control steps, or 0.727 seconds. Upstream ReturnTracker increments episode length by one per control step. These brief trajectories and early posture resets do not demonstrate sustained balance. GPU samples average 54.92% (p95 71%, peak 99%), including intrinsic checks. No further training started. Proposed next hypothesis is to revise short episodes/early posture resets and learn longer physical balance with soft pose preferences and explicit fall cost. No new implementation or run is authorized by this outcome record.

## Authorized one-hour reference imitation (2026-10-01)

The user approved replacing corrective penalties with MimicKit reference imitation and extending both training episodes and built-in native checks to 60 seconds. Preset `TorqueStandReferenceHour` starts run `stand-torque-reference-60s-60min-11` from the stable primary StandArena actor `7f270019` and full AMP sidecar `0986997c`, with a fresh optimizer/RNG stream. The new immutable `reference_v9` reward calls pinned `envs.deepmimic_env.compute_reward` using the upstream humanoid weights/scales. Joint matching covers every actuated body, including the head; unactuated wrist rotations are excluded. Keypoints are root-relative and heading-local, so there are no world-space foot anchors. The existing learner retains 80% task reward and 20% AMP style reward.

Training lasts 3600 synchronized update seconds, excluding mandatory initial/scheduled/final native checks and normal export parity. Envs=2048 and EvaluationInterval=64 remain unchanged. The 60-second horizon supplies a longer trajectory, while exploration noise, discount and PPO rollout size are unchanged. Physical falls/contact and nonfinite state terminate episodes; soft pose mismatch does not. No passive springs, position servos, bracing/occupancy penalties or short-horizon curriculum apply to this mode. Eight starting reference poses and exact spring-free physics are preserved. Checkpoints, hashes, GPU samples and per-body pose errors are recorded. Final export targets the separate TorqueArena; primary StandArena remains on the stable source. No extra hold suite, standalone tests, compilation, Godot run or chat automation is authorized or included. A dated measured outcome is appended after completion; this entry makes no improvement claim.

## Additive reference imitation outcome (2026-10-01)

Run `stand-torque-reference-60s-60min-11` started from checkpoint `7f270019` and its full AMP state with a fresh optimizer/RNG stream. The task calls upstream DeepMimic additive tracking, including neck pose, with a 20% AMP style prior. No passive springs, pose servo, limit penalties or early posture resets are used. Training episodes last 60 seconds and built-in checks last 60 seconds; physical falls still terminate episodes.

Completed 3600.196 synchronized training-update seconds (60.003 minutes), 1051 updates and 137,756,672 transitions. Trainer elapsed: 4038.484 seconds. Paired initial/final one-minute survival: 6/8 to 6/8; balanced cases: 6/8 to 5/8; upright cases: 0/8 to 0/8. Final `801c1a18` is exported in `stand-torque-reference-60s-60min-11/arena-preview` for MimicStandTorqueArena. Primary StandArena selection is preserved. Per-body reference errors and physical failures are recorded in report.json; natural movement is not admitted. No extra hold suite, standalone tests, compilation or Godot run occurred.

## Spring-free direct-torque Stand window (2026-10-01)

Run `stand-torque-reference-60s-60min-11` continues checkpoint `7f270019` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 1051 updates / 137,756,672 transitions in 3600.196 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 189.567 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

## Final reference actor selected in StandArena (2026-10-01)

After the user requested checking and loading, the existing completed export for `stand-torque-reference-60s-60min-11/model.pt` (`801c1a18`) was selected in primary StandArena. Checkpoint, ONNX, plant and reference-frame hashes were checked against the recorded export provenance; no new physical tests or export were run. The final actor was selected explicitly, not the higher-ranked iteration-64 best actor. The prior primary bundle `7f270019` and prior scene are preserved in the selection and preview backup.

The completed hour contains 3600.196 synchronized training-update seconds, 1051 updates and 137,756,672 transitions. Logged trainer elapsed is 4038.484 seconds, excluding the subsequent export. Initial/final one-minute survival is 6/8 to 6/8; balance decreases 6/8 to 5/8; upright remains 0/8. Final cases 3 and 4 fall at 17.301 and 4.600 seconds. Head local reference error remains about 68.9 degrees, so longer training has not resolved the distorted upper-body pose. The actor remains experimental and is loaded for the user's visual inspection. Restart the scene to apply the new bundle. No further training, standalone tests, compilation or Godot launch occurred.

## Fresh untrained brain selected (2026-10-01)

The user requested removing the learned brain while retaining the training settings and loading the fresh brain into Arena. `stand-torque-reference-fresh-12/model.pt` (`aab3797d`) initializes actor, critic, discriminator and normalization from scratch with seed 301001. No trained checkpoint was read. The portable actor and matching full AMP sidecar are saved; policy updates, transitions and training time are zero. `report.json` explicitly has no physical assessment results. This is an untrained torque actor, not disabled motor actuation.

The exact existing spring-free plant and licensed reference poses are retained. Preset `TorqueStandReferenceFreshHour` clones the prior reference-hour settings and changes only the initialization checkpoint to this fresh state: one-hour update budget, 60-second episodes and checks, Envs 2048, EvaluationInterval 64, `reference_v9`, and 80% imitation / 20% AMP style reward. The AMP sidecar also contains a fresh discriminator, so no learned bracing or learned motion prior is inherited.

The normal actor export has maximum conversion difference 6.519e-9. StandArena and TorqueArena now select the fresh bundle, with HUD explicitly showing untrained status and zero updates. Prior selections and both prior scene files are saved alongside it; the completed `801c1a18` checkpoint remains intact. No training, physical rollout, standing assessment, standalone tests, compilation or Godot launch ran. Restart the scene to apply the fresh selection. The prepared training preset loads this exact initial state when a training session is authorized.

## Fresh reference-hour training launched (2026-10-01)

The user authorized execution after confirming the one-minute episode/check semantics. Run `stand-torque-reference-fresh-60s-60min-13` loads only the untrained `aab3797d` actor and its matching fresh critic, discriminator and normalization state from `stand-torque-reference-fresh-12`. The source contains zero policy updates and zero training seconds. Preset `TorqueStandReferenceFreshHour` retains seed 301001, Envs 2048, EvaluationInterval 64, 60-second episodes and native checks, `reference_v9`, 80% task / 20% AMP reward, and the unchanged spring-free 33-motor plant.

The sole trainer is PID 34580, launched by wrapper PID 21184. The requested budget is 3600 synchronized training-update seconds, excluding intrinsic checks and export. Startup completed and actual PPO updates were observed. The mandatory initial native assessment has 0/8 surviving, balanced or upright cases; all initial cases fall after 0.317-0.383 seconds. This records the untrained baseline, not a regression from an inherited trained actor. Source model, reference, checkpoint, AMP and training-code provenance are saved in protocol/request records. GPU samples are logged during the run. The current StandArena remains the untrained initial-state preview for manual comparison; final export will be prepared separately. No extra tests, compilation, Godot launch, chat automation or second trainer was started.

## Additive reference imitation outcome (2026-10-01)

Run `stand-torque-reference-fresh-60s-60min-13` started from checkpoint `aab3797d` and its full AMP state with a fresh optimizer/RNG stream. The task calls upstream DeepMimic additive tracking, including neck pose, with a 20% AMP style prior. No passive springs, pose servo, limit penalties or early posture resets are used. Training episodes last 60 seconds and built-in checks last 60 seconds; physical falls still terminate episodes.

Completed 3603.608 synchronized training-update seconds (60.060 minutes), 606 updates and 79,429,632 transitions. Trainer elapsed: 3618.938 seconds. Paired initial/final one-minute survival: 0/8 to 0/8; balanced cases: 0/8 to 0/8; upright cases: 0/8 to 0/8. Final `35bae24d` is exported in `stand-torque-reference-fresh-60s-60min-13/arena-preview` for MimicStandTorqueArena. Primary StandArena selection is preserved. Per-body reference errors and physical failures are recorded in report.json; natural movement is not admitted. No extra hold suite, standalone tests, compilation or Godot run occurred.

## Spring-free direct-torque Stand window (2026-10-01)

Run `stand-torque-reference-fresh-60s-60min-13` continues checkpoint `aab3797d` with its actor, critic, discriminator and normalization; optimizer/RNG state starts a new window. All passive rest-pose springs are removed. Neck torque is learned by the policy. The 33 policy motors use direct bounded torques, with no standing-pose PD servo. Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift clips supply a style prior; their original dynamic admission does not transfer to this plant.

Completed 606 updates / 79,429,632 transitions in 3603.608 synchronized training-update seconds, excluding built-in assessments. Cumulative training: 60.060 minutes. Final settled standing: 0/8. Conditional 600-second gate passed: False. Natural movement is not admitted. The final checkpoint is an experimental preview in `MimicStandArena.tscn`, with unlimited duration and no balls. C# contract support was edited; no Godot/C# compilation or standalone tests were run.

## Fresh-hour final actor selected (2026-10-01)

The user requested checking the completed fresh run and loading its final brain into Arena. `stand-torque-reference-fresh-60s-60min-13` completed 3603.608 synchronized training-update seconds (60.060 minutes), 606 updates and 79,429,632 transitions. Trainer elapsed was 3618.938 seconds, excluding export. The final `35bae24d` actor and full AMP state are preserved. Exact checkpoint, ONNX, plant and reference-frame provenance was checked against the existing export, then final model.pt was selected in primary StandArena. Best iteration 320 was not substituted. The prior untrained selection `aab3797d` and scene backup remain available.

Initial native falls occur at 0.317-0.383 seconds; final falls occur at 2.033-3.150 seconds. All eight final cases terminate with pelvis below the failure height. One-minute survival, balanced standing and upright gates remain 0/8. Increased pre-fall duration demonstrates limited progress, not sustained standing or natural pose. Terminal pose errors during falls and short training-rollout averages must not be treated as successful standing posture. Whole-device GPU samples average 56.61% (p95 71%, peak 100%), including setup and intrinsic checks.

StandArena selects the experimental final candidate with unlimited scene duration, spring-free learned torques and UseLatestExport=false. Restart the scene to apply the selection. No additional training, standalone tests, physical rollout, compilation or Godot launch ran during this loading step.

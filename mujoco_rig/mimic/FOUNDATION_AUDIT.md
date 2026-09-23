# Dummy foundation audit — 2026-09-23

## Decision

Keep the current skeleton and simulator bridge. This audit found no gross mass,
inertia, motor-transmission or sampled mapping defect that justifies rebuilding
them. It did find two observation defects worth correcting before further PPO or
motion-retargeting experiments. The body is not yet validated for general recovery,
protective falling or long-duration locomotion.

This audit adds diagnostics only. It does not change the rig, actor observations,
reward, controller, trained weights or selected Godot export.

## Confirmed findings

### P1: Foot-load observations omit the toes

`StandTask.observations()` selects `Foot_L` and `Foot_R` loads. The native backend
reports contacts separately on `Toe_L` and `Toe_R`; these loads are not inherited
by the parent foot. `MjMimicStandDriver` uses the same foot-only observation.
Recovery scoring already includes toes, so policy input and recovery measurement
use different definitions of foot support.

Three deterministic floor-contact fixtures use root pitch -0.15, 0 and +0.15 rad,
with the lowest sole penetrating 0.2 mm to establish measurable contact:

- Heel contact: the foot-only representation retains the measured support load.
- Flat contact: about 101 of 593 N passes through toes and is omitted (17%).
- Toe contact: about 41 N passes through toes while both actor load inputs are zero.

The final parity probe reads the actual `StandTask` observation channels on both
native and GPU backends, rather than only reconstructing the observation formula.
These are diagnostic fixtures, not measured percentages during ordinary gameplay.
The omission's effect on learned recovery remains unmeasured.

Evidence: `logs/mimickit-foundation/audit-04/report.json` and
`logs/mimickit-foundation/sensor-parity-03.json`.

### P1: Contact-force sampling times differ across training and deployment

Newton exposes the forces from the last solved physics substep. Native evaluation
calls `mj_forward` after integration, recomputing forces at the updated state.
Godot's `MjMimicStandDriver` also calls `Forward()` after its four substeps.

Across the three fixtures and four control intervals:

- GPU/native body-force differences reach 50.004 N with the current sampling.
- Comparing GPU forces with native forces captured from the same final substep
  reduces the maximum difference below 0.004 N.
- Maximum native qpos component difference stays below 7e-7 (coordinates include
  position, quaternion and joint angles; this is not a single-unit position metric).

This isolates a measurement-time mismatch rather than a body-index/sign mapping
error in these fixtures. It can affect policy input, contact-based rewards and
termination. The audit does not establish how much recovery performance it costs.

Evidence: `foundation_sensor_parity.py`, `sensor-parity-03.json`,
`NativeDummyEngine.step()`, upstream `NewtonEngine.step()`, and
`MjMimicStandDriver.Step()`.

## Mechanical and control capacity

- Total compiled mass: 69.626 kg. All moving bodies have finite positive mass and
  principal inertia; inertia triangle inequalities pass. The rest pose lies inside
  all joint limits. Bilateral mass/inertia/position comparisons are recorded in JSON.
  These checks establish mathematical consistency, not biomechanical calibration.
- All 30 policy motor channels apply exactly the intended generalized force in an
  airborne fixture, with positive response on the driven coordinate.
- Fresh Newton/native compatibility validation passes all its gates over 65 poses
  and a 0.5-second rollout. Maximum mapper error is about 1.05 micrometres; sampled
  collision-pair mismatches are zero. Rollout root/joint errors are 5.22e-5 m and
  0.00413 rad. This does not validate every pose or long-horizon transfer.
- Nine independent IK probes cover rest, 10 cm lateral weight shifts, single-foot
  lifts, and 15 cm forward/backward steps with 6 cm swing-foot lift on both sides.
  Each uses the original joint limits and collisions, with two starting seeds to
  reduce straight-leg IK singularity effects.
- Rest, both lateral shifts, and four single-support step poses admit the tested
  quasi-static support solution. Forward step poses require about 59% of the
  currently available motor budget; backward poses about 49%; lateral shifts about
  21%. The available budget already includes the production 60% authority multiplier.
- Two in-place foot-lift fits retain approximately 2 mm foot-position error and do
  not satisfy the support solver's flat-sole tolerance. They remain inconclusive;
  failure of these local fits does not establish a missing degree of freedom.
- Neck/wrist equilibrium is solved within existing limits and zero actuator controls
  before support is evaluated. All nine fits have passive residual below 5e-10 Nm;
  the support solve then enforces a 0.05 Nm residual bound. Earlier audit-01 through
  audit-03 runs relaxed passive equilibrium and are superseded for capacity/hold
  conclusions. This remains a static capacity check with prospective sole contacts
  within 1 mm of the floor, not a dynamically feasible stepping trajectory or proof
  of sufficient motor speed/power.

Evidence: `audit-04/report.json`, `audit-04/poses.npz`, and
`compatibility-01/report.json` under `logs/mimickit-foundation`.

## Limits that remain unvalidated

**Action freedom.** The tested step poses depart up to approximately 0.65 rad from
rest. Refitting with a +/-0.25 rad rest-centered target window increases step-foot
errors from below 0.02 mm to roughly 7–9 mm, with COM errors of roughly 6–10 mm.
This demonstrates a target-fitting tradeoff, not that all recovery steps are
impossible. Joint targets are not hard physical angle bounds; the reference can
also provide the larger movement. Changing the action window needs a separately
versioned and measured experiment.

**Active balance.** None of the nine fixed-pose probes survives the full ten-second
hold with zero-residual target PD, or with diagnostic static torque feedforward.
Rest lasts about 4.85 seconds; lateral shifts last about 2.05 seconds with PD and
4.65 seconds with feedforward. These controllers have local joint feedback but no
learned COM/support correction. This is not a test of the trained actor and does
not prove that the rig cannot balance. Holds start at each pose; transitions and
successful dynamic stepping are not established by this audit.

**Protective behavior.** The current task fails on non-foot ground contact and low
pelvis height. It cannot reward successful hand bracing, kneeling, controlled falls
or getting up under that definition. Wrists are passive and the neck uses fixed
position targets. Ball trajectory is not an actor input, so this task also does not
teach visually anticipated protection before contact. These are explicit task and
control limits, not evidence of corrupted sensors or a broken skeleton.

## Recommended implementation order

Implementation update (2026-09-23): the contact timing and foot-plus-toe corrections
below are now implemented as opt-in `mimic_stand_target_pd_v2`. Old exports retain
their original semantics. GPU partial reset was additionally found to refresh peer
contact readings; v2 now preserves those snapshots. GPU/native fixtures and ball
transport pass, and Godot/native replay passes 96 impacts and eight standing phases
for both versions. See `README.md` and `logs/mimickit-contact-v2`.

The frozen-actor ablation yields 72/96 legacy survivals, 73/96 timing-only and 67/96
full-v2, with 8/8 standing retention at three and five seconds in every condition.
These unchanged v1 weights/normalization do not establish v2 learning improvement.
Diagnostic copies cannot replace the viewer or seed training.

New motion training now requires v5 root-support and actuator-feasibility gates;
v4 support-only candidates remain diagnostic. Force-aware trajectory refinement
is implemented with unilateral foot-plus-toe forces, an inner friction pyramid,
stance targets and grounding inside the optimization. The backward candidate
`force-retarget-04/backward` removes all 64 previously unsupported interior frames,
but still fails the full motor envelope on 9/117 post-delay frames and the current
delayed PD window on 22/117. A doubled window still fails 20. Both zero-residual
PD and the unchanged Stand v2 actor fail all eight tracking trials. These results
separate a corrected reference-support defect from the remaining controller-demand
mismatch; they do not show that a trained motion policy could never adapt.
The subsequent joint contact/actuator optimizer includes the existing delayed PD
limits, bounded stance-leg IK and pelvis orientation corrections. On
`actuated-retarget-03/backward`, all 118 root-support and 117 post-delay actuator
checks pass. Planted-foot sliding is below 1e-12 m/s; swing-foot tracking still
misses its 40 mm limit by 0.534 mm, so this candidate remains rejected. These are
reference-feasibility improvements, not a behavioral result.
The final ten-evaluation continuation (`actuated-retarget-04/backward`) reduces
maximum foot error to 39.949 mm and passes every unchanged v5 admission check,
including all root-support and motor/PD frames. It is admitted for isolated motion
experiments, without claiming closed-loop tracking or optimizer convergence.
The subsequent fresh three-minute CUDA motion pilot (`actuated-backward-pilot-01`)
completes 46 updates but remains 0/8 on both native and Newton evaluations. Native
mean survival falls from 1.673 to 1.629 seconds; checkpoint selection retains the
initial policy. Admission alone has not produced a behavioral improvement. The
saved native traces (`actuated-probe-01`) support a targeted contact/torque diagnosis
before attributing the remaining failure to PPO hyperparameters or anatomy.
The forward fit still fails support on 92/118 frames and violates stance-slide
and foot-target limits; its contact transitions also require correction. The final
v5 admission review rejects both candidates (`force-retarget-review-01`).
The packaged walking clips and selected Godot policies have not been replaced.
The later forward joint fit (`actuated-retarget-02/forward`) still fails on 83/118
root-support frames. Its largest slips occur at inferred contact-gap extensions
(frames 28, 47 and 68), with the hip at its extension limit in frames 28 and 68.
Extending flat-foot support through toe-off is too restrictive there. Correct
heel-lift/toe-only support semantics before another forward motion training run;
these results do not justify widening anatomical limits.
The standing reference fits current motor and delayed target windows in all 177
evaluated post-queue-fill frames. Wider moving-reference targets do not resolve
the root-support failures, so controller bounds and physical rig remain unchanged.

1. Define one contact sampling convention across GPU training, native evaluation
   and Godot; test reset, quiet support, toe-off and impact transitions.
2. Add complete foot-plus-toe support to a versioned observation contract. Keep
   legacy exports on their original semantics; do not silently reinterpret old
   weights or normalization statistics.
3. Compare the corrected contract with the baseline using the same physical cases
   and standing-retention checks before attributing a behavioral improvement to it.
4. Then address dynamic support during motion retargeting and measure action-window
   adequacy. Add falling/bracing/get-up tasks as separate capabilities when balance
   and stepping have measurable success.

No anatomical replacement, stronger motors or simulator migration is justified by
the evidence collected here.

Verification: 76 targeted Python tests pass, including seven audit-fixture tests;
the fresh GPU compatibility run passes all its existing gates. A rendered static
pose preview is retained at `logs/mimickit-foundation/audit-04/pose-preview.png`.

## Subsequent native tracking and toe-off diagnosis (2026-09-23)

`tracking-audit-04` records the admitted backward reference at physics-substep
resolution. Zero-residual PD completes 0/8 trials (mean survival 1.661 s).
Prefilling its command queue gives 0/8 and 1.552 s. Removing self-collision only
in a private diagnostic model gives 0/8 and 2.163 s, with floor contacts preserved.
The first baseline phase contacts forearm/thigh before its 5 cm root deviation;
several other phases contact opposite feet/toes. The reference has sub-mm limb
clearances and small overlaps. This identifies a contributing defect, not the
sole cause: root lag and failed tracking remain without self-collision.

New v6 candidates can declare individual foot/toe support phases. Toe-only IK
permits heel lift without moving the planted toe, and raised corners cannot
provide projected support. Older v5 interpretation is unchanged. The first forward
candidate (`toe-retarget-01`) passes stance sliding at 0.08291 m/s but fails
penetration, sole-contact, foot-tracking and force/actuator gates; it is rejected.

Self-clearance fitting additionally exposed a discontinuous native box-distance
query (6.23 mm to zero under a 1e-8 radian perturbation). Conservative box SAT
separation fixes the numerical derivative; actual native penetration remains an
independent test. `clearance-retarget-02/backward` reaches 1.907 mm minimum clearance
against a required 5 mm and fails delayed PD at frame 64, despite retaining all
root-support checks. It is rejected. No new PPO run or policy promotion follows.
See the current `mujoco_rig/STATUS.md` entry for exact budgets, files and remaining
localized failures. Thirty-seven focused tests cover the new constraints and legacy
reference/export behavior.

## Subsequent constrained fitting (2026-09-23)

Bounded IK now redistributes corrections at active hinge limits rather than
clipping an unconstrained solution. Stable box distances are used in every fit,
and positive self-clearance fitting retains a 1 mm reserve above the unchanged
admission threshold. Optional phase-constant landing offsets are bounded to 2 cm
per axis; validation remains relative to the original foot targets.

`constrained-retarget-01/backward` resolves the frame-64 right-ankle PD failure
without changing motor limits or residual action authority. It passes every
root-support and delayed-PD check, but minimum self-clearance remains 2.856 mm
against a required 5 mm. Native distance agrees with the SAT value at the limiting
Shin_L/Toe_L pair, so this remaining failure is not merely a conservative-bound
artifact. The reference is rejected. The forward continuation also remains
rejected, including worse stance sliding despite lower fitting cost.
The ten-evaluation landing trial remains rejected with 102/118 root-support
failures and only sub-mm placement changes. An isolated transition diagnostic
is the next step; no additional PPO or policy promotion follows these results.

Forty-one focused tests pass. These fixes improve fitting correctness and local
feasibility; they do not demonstrate better learned recovery. See the latest
`STATUS.md` entry for candidate results and the next diagnostic boundary.

## Isolated forward transition (2026-09-23)

The subsequent isolated forward probe ([20, 35) source frames) reaches 0/13
root-support failures and sub-mm penetration with an opt-in dense solve, while
default sparse fitting still has 9/13 failures at the same twenty-evaluation
budget. One delayed-PD failure remains at source frame 22. This identifies a
local optimization limitation, not a need to widen physical joints or motor
authority. The public dense implementation costs more wall time; sparse remains
the full-clip default. Raw splicing fails whole-reference continuity/dynamics,
so constrained overlapping intervals are the next diagnostic step before PPO.

The standalone dynamics file audit now honors explicit sole phases; admission
validation already did so. Forty-four focused tests pass. Exact outputs, runtime
budgets and splice failures are recorded in the latest `STATUS.md` entry.

## Context-preserving intervals (2026-09-23)

Subsequent context-preserving interval fits are implemented in `window_retarget`.
They preserve boundary poses, delayed-command history and full-stance anchors,
then validate the assembled clip and reject regressions. Two overlapping short
windows and a wider forty-evaluation transfer fit all remain rejected: the latter
improves three root-support frames but breaks frame 28 and increases foot error
to 51.106 mm. Saved poses remain identical to the input; no policy is promoted.
A projected heel-support diagnostic does not resolve the remaining transfer
failures. The next issue is enforcing coupled placement/support constraints in
the optimizer. Forty-nine focused tests pass; full results are in `STATUS.md`.

## Explicit placement and support constraints (2026-09-23)

The window optimizer now uses SLSQP with hard placement/contact-geometry and
delayed root/motor equilibrium inequalities. Geometry applies to editable poses
and affected stance transitions; force feasibility also covers dependent fixed
neighbors. Independent full-clip validation and regression checks remain required.
The legacy least-squares solver remains available for reproducible comparisons.

`window-constrained-01/forward` runs the same [20, 35) transfer for twenty
iterations (285.484 seconds). Local placement constraints pass with 38.011 mm
maximum foot error. Equilibrium constraints remain unsatisfied. Full-clip root
failures remain 100/118, exchanging a repaired frame 22 for a new failure at 28;
aggregate force consistency fails at 29. The proposal is rejected and saved poses
equal the input exactly. This is a fitting/validation improvement, not a behavioral
improvement or proof that the dummy cannot step. Investigate the force-feasibility
formulation before more fitting budget or PPO. Fifty-three focused tests pass.

## Contact-force dependency correction (2026-09-23)

The constrained window solver incorrectly combined root-dynamics dependencies
with the longer delayed-command interval. At source frames 36 and 37, all poses
determining generalized force demand are frozen, but delayed motor bounds depend
on editable frame 34. Perturbing that editable pose changes the motor bounds and
leaves both generalized force demands exactly unchanged. Those roots are already
infeasible, with irreducible normalized fitting residuals 0.092082 and 0.076411.
Requiring their local repair made the prior constrained problem impossible.

Root dependencies are now separated. An immutable root failure is reported as
deferred and remains a full-clip admission failure; a supported fixed frame still
constrains any delayed command affected by the edit. Regression checks still
protect previously feasible frames. A second correction retains the best feasible
incumbent when a bounded solver terminates on an infeasible trial. A static
standing-reference fixture verifies retained native root/PD feasibility; this
test does not establish closed-loop survival.

The experimental direct-force path exposes nonnegative friction-ray coefficients,
with exact force Jacobians and a bounded feasibility-restoration phase. Native
finite-difference tests check the assembled Jacobians. The corrected twenty-
evaluation transfer (`window-direct-forces-04`) remains rejected: 100/118 root
failures, 14 conditional PD failures and 40.951 mm maximum foot error. These are
offline solver fixes, not demonstrated recovery improvement. Rig, policy contract,
admission limits and viewer selections are unchanged. See `STATUS.md` for the
comparison using the original constrained solver. Sixty focused tests pass.

That comparison (`window-constrained-02`, twenty iterations) also remains
rejected: it repairs root support at frame 25 but regresses frame 28, leaving
100/118 failures. No accepted reference or policy update follows either run.

## Static support and connected transfer calibration (2026-09-23)

The recorded-walk warm start (`window-seeded-transfer-01/forward`) remains
rejected after twenty restoration evaluations. Verified probe initialization
preserves original targets, stance anchors and fixed context; the saved reference
is unchanged. This result does not justify increasing PPO duration.

A separate controlled diagnostic establishes feasible static support endpoints
without rig or motor changes. `transfer_calibration` optimizes identical poses
with shared coordinates, so acceleration cannot create apparent static support.
All 21 solves in `transfer-calibration-02` pass geometry and static root/motor
checks. The final 32-second motion passes all existing reference checks over
639 interior frames: zero root and delayed-PD failures, 0.046 mm maximum
penetration and 0.296 mm maximum foot-target error. The initial connected attempt
failed; unloading before the contact switch, stopped quintic interpolation and
two collision-aware intermediate poses produced the passing trajectory.

This narrows the problem: the existing rig admits these necessary support and
motor conditions for a slow, paused lift-and-return sequence. It does not prove
passive-joint equilibrium, startup queue behavior, contact-compliance stability,
human-like walking, or a learned response to impacts. The artifact is diagnostic,
not training-admitted or deployed. Native closed-loop tracking is the next step;
there is no evidence here supporting another reward or torque change. Sixty-six
focused tests cover the fitting, initialization and interpolation paths.

## Reproduction

Use the project's Mimic virtual environment for `$python`, and the configured
MimicKit checkout for `$env:MIMICKIT_PATH`. Output destinations must be new.

```powershell
& $python -m mujoco_rig.mimic.foundation_audit --out logs/mimickit-foundation/<new-audit>
& $python -m mujoco_rig.mimic.foundation_sensor_parity --mimickit $env:MIMICKIT_PATH `
  --out logs/mimickit-foundation/<new-sensor-report>.json
& $python -m mujoco_rig.mimic.validate --mimickit $env:MIMICKIT_PATH `
  --out logs/mimickit-foundation/<new-compatibility-run>
& $python -m unittest mujoco_rig.mimic.test_foundation_audit -v
```

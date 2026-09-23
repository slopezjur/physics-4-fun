# MuJoCo track — status

## 2026-09-23: Feasible slow transfer calibration; no policy promotion

`transfer-calibration-02` provides a 32-second stepping-in-place diagnostic on
the unchanged rig: shift weight, unload one foot, lift it about 6 cm, return,
then repeat on the other leg. All existing full-motion geometry, contact,
root-support and actuation checks pass over 641 poses / 639 interior frames.
There are zero root or delayed-PD failures, maximum penetration is 0.046 mm,
and maximum foot-target error is 0.296 mm. All 21 static calibration solves
also pass their necessary-condition checks (temporal foot lift is inapplicable
to a stationary pose). These are offline feasibility results, not learned
balance, forward walking, or closed-loop survival.

The new `transfer_calibration` command fits shared stationary coordinates so
acceleration cannot manufacture static support. It uses the vetted CC-BY-4.0
standing source, loads one leg before releasing the opposite contact, and joins
knots with bounded quintic segments that stop at each knot. Stance anchors stay
fixed. One bounded collision-repair pass inserts clearance poses; in this run,
two poses at 2.5 and 13.5 seconds avoid a forearm/pelvis intersection. The final
full-motion validator remains authoritative. Hashes bind the diagnostic and
saved knots to the source and rig; attribution accompanies the result.

This follows a failed direct reuse of the recorded-walk transfer. The new
`window_retarget --seed-probe` verifies source, rig, timestep, pose shape and
available content hashes, then uses probe poses only as initialization. Original
targets, bounds, full-stance anchors and fixed context remain authoritative.
Historical probes without a content hash are explicitly marked unverified.
`window-seeded-transfer-01/forward` used editable [18, 37), context [14, 41),
20 restoration evaluations / 332.828 seconds. The candidate had 96/118 root
failures and 15 conditional PD failures and did not satisfy its optimizer
constraints. It was rejected; saved source poses are byte-for-byte equal as
arrays. The probe's independently shifted landing anchors were not imported.

The initial static-to-dynamic calibration also failed at support switches and
interpolated collisions. Explicit unloading and stopped interpolation removed
root/PD failures; collision-aware intermediate poses resolved the remaining
geometry failure. The maintained command reproduces the passing result.

Next: native closed-loop tracking of this slow calibration, including reset
and command-delay behavior, before a short transfer-learning experiment. Do not
resume the failed recorded-walk reference or infer Perturb improvement from this
result. Diagnostic artifacts remain explicitly unadmitted for training, with no
training manifest. No PPO run, export, viewer selection, rig or control-limit
change was made. Verification: 66 focused tests pass.

## 2026-09-23: Frozen-root dependency bug fixed; transfer still unadmitted

Found a concrete impossibility in the previous constrained window formulation.
Editing source frames [20, 35) changes delayed motor bounds through frame 37,
but root dynamics at frames 36–37 depend only on frozen poses. Both roots are
already unsupported. The optimizer nevertheless had to repair them locally.
`immutable-root-dependency-01.json` records a frame-34 perturbation: generalized
force changes are exactly zero at 36–37 while motor bounds change. Independent
HiGHS LPs certify both root problems infeasible in the unchanged fitting contact
envelope. Their normalized residuals are 0.092082 and 0.076411.

The fitter now separates central root-dynamics dependencies from delayed-command
dependencies. Only LP-certified, immutable root failures are deferred; numerical
solver failure is not a certificate. Supported fixed frames retain their delayed
motor constraints. Deferred failures are explicit in the report and still fail
full-reference admission. This fixes local problem construction without relaxing
physical limits or silently admitting the source clip.

A second correction preserves the best feasible incumbent when optimization ends
on an infeasible trial. A static standing-reference test retains native root and
PD feasibility after two SLSQP iterations; this is a necessary-condition fixture,
not a closed-loop survival test. Returning an incumbent is not called convergence.

Experimental `--solver direct-forces` exposes nonnegative friction-ray forces,
uses exact force Jacobians, and starts with bounded feasibility restoration.
It preserves the existing contact points, friction reserve and motor envelope.
Direct-force and projected-force residuals agree on coupled/saturated-motor tests;
independent differences verify both block and assembled force Jacobians.

Final matched-source tests, same editable [20, 35) and fixed context [16, 39):
- `window-direct-forces-04`: 20 restoration evaluations / 221.859 seconds,
  8,661 combined residual calls, no polishing because restoration remains
  infeasible. Maximum foot error 40.951 mm, 100/118 root failures, 14 conditional
  PD failures; no root case changes. Rejected. Normalized placement/force
  violations remain 0.023810 / 0.130212.
- `window-constrained-02`: original projected-force SLSQP with the dependency
  correction, 20 iterations / 306.016 seconds, 88 objective evaluations and
  8,277 combined residual calls. Frame 25 becomes root-feasible but frame 28
  regresses; total remains 100/118, with 14 conditional PD failures. Foot error
  remains 40.116 mm; local placement/contact and force constraints still fail.
  Aggregate force consistency passes. Rejected, not converged.

Both saved references preserve the source poses exactly. No policy is trained,
exported or selected; the rig, packaged references and Godot viewers are unchanged.
The dependency bug was real, but removing it does not establish feasibility of
the editable transfer or better recovery. Further PPO remains premature. The
remaining investigation is a feasible transfer initialization/formulation, not
additional reward, torque or admission-threshold tuning. All jobs have finished.

Earlier diagnostic evidence is retained: `window-direct-forces-01` stopped in its
first SLSQP subproblem; `-02` exposed tiny negative BVLS seed roundoff (now clamped
to the nonnegative bound); `-03` tested restoration before the dependency fix.
None was accepted. The direct-force solver remains opt-in rather than replacing
the default based on these unsuccessful tests.

Verification: 60 focused tests pass; the seven window tests also pass with final
LP certification. Tests cover incumbent retention, mixed Jacobians, restoration,
immutable root failures remaining full-clip failures, and boundary protection.
`git diff --check` passes.

## 2026-09-23: Explicit window constraints; placement passes, dynamics still rejected

`window_retarget` now defaults to constrained SLSQP. The existing foot-placement,
sole-height, penetration/clearance and stance-slide limits are explicit
inequalities. Joint root/motor force residuals must also satisfy an explicit
near-zero bound at every affected interior frame, including fixed neighboring
poses whose accelerations or delayed commands depend on an edit. The inner force
solve retains the existing inset footprint, friction reserve and delayed PD
authority. No physical or admission limit changes. Immutable geometry outside
the editable region is excluded from local constraints, but remains subject to
full-reference admission. A proposal with unsatisfied optimizer constraints is
rejected even if its aggregate regression checks otherwise pass.

Shared reference-limit constants keep offline fitting and admission aligned.
Objective/constraint finite differences share evaluations through public SciPy
APIs. `--max-evaluations` means SLSQP iterations for this solver; residual-call
counts are reported separately. `--solver least-squares` preserves the previous
dense diagnostic path. Other force-fitting callers keep their original defaults.

Bounded experiment: `logs/mimickit-steps/window-constrained-01/forward`, same source
as the previous window tests, editable [20, 35), unchanged context [16, 39):
- 20 SLSQP iterations, 54 objective evaluations, 7,381 combined residual calls,
  285.484 seconds. Iteration limit reached; not converged.
- All local placement/contact-geometry inequalities pass. Maximum foot error
  inside the editable interval is 38.011 mm, below the unchanged 40 mm limit.
- Force inequalities do not pass: maximum normalized affected-frame residual
  violation is 0.102062. Full-clip root-support failures remain 100/118: frame 22
  becomes feasible, but frame 28 regresses. Conditional PD failures remain 14.
- Aggregate force consistency also fails at frame 29. Global foot error remains
  40.116 mm, penetration 18.659 mm and stance slide 0.209463 m/s; remaining
  immutable geometry still prevents full admission.
- Rejected. Candidate poses outside [20, 35) are exactly unchanged. Saved
  `step_reference.npz` poses equal the original source exactly; no promotion.

This demonstrates enforced placement and reliable rejection, not improved learned
behavior or physical infeasibility of the rig. The local force-feasibility solve
is still unresolved. Before more optimizer budget or PPO, investigate feasibility
restoration/direct contact-force variables on this same transfer; another reward,
torque or acceptance-threshold adjustment is not supported by this experiment.
Godot viewer selections, policy weights, the rig and packaged references remain
unchanged. No training was run; all experiment jobs have finished.

Verification: 53 focused tests pass, including conflicting objective/constraint
cases, infeasible support, bounded finite differences, excluded immutable geometry,
affected fixed-frame dynamics, full-clip regression protection and hash-bound
output. `git diff --check` passes.

## 2026-09-23: Overlapping intervals with preserved context

`window_retarget.py` adds dense local fitting inside an unchanged full reference.
It preserves every surrounding pose exactly and carries a context margin of
`ceil(command_delay / reference_dt) + 2` frames on both sides (four for this clip).
Full-stance world headings and landing offsets are inherited without adjustment;
cropping cannot create a new anchor. Central derivatives and delayed-PD intervals
in the affected region match the full clip. Subsequent overlapping windows start
from the last accepted full reference.

Acceptance uses full-clip validation. It protects passing gates and the identities
of root-supported/combined root-PD-feasible frames, and rejects increased global
penetration, stance-slide or foot-target-error maxima. Rejected proposals are
archived separately; saved output receives a new hash and full admission results.
No locally successful window can bypass reference admission.

`window-retarget-01/forward` tests [22, 31) and [27, 36), twenty evaluations each:
- First window: 114.968 seconds, 99/118 root-support failures versus the original
  100/118. Frames 23 and 25 become feasible but frame 28 regresses. Foot error
  increases to 48.806 mm. Rejected.
- Second window: 95.640 seconds, 101/118 root-support failures and 46.986 mm foot
  error. Rejected. Both proposals' largest foot error is the incoming right swing
  foot at frame 27; this is not a reinterpretation of planted toe support.
- No proposal is accepted; saved poses are exactly equal to the input reference.

`window-retarget-02/forward` broadens the editable interval to [20, 35) and runs
forty evaluations / 452.484 seconds. Root failures fall to 98/118: frames 22,
24 and 25 improve, but the previously feasible transfer at frame 28 fails again.
Foot error increases to 51.106 mm. Aggregate force consistency remains passing,
unlike the earlier raw-splice experiment, but the candidate is rejected. Its
saved output also preserves the input poses exactly. Neither bounded run is
converged; failure does not establish that the rig is physically incapable.

An additional support-only counterfactual (`heel-support-hypothesis-01.json`)
allows the incoming right foot's near-floor corners to supply projected force.
This removes the root failure at frame 23 but not 24–27. Those poses still
penetrate the floor, so the intervention is diagnostic and cannot be admitted.
The result does not support contact relabeling alone as a solution.

The remaining issue is a coupled foot-placement/support tradeoff: fitting cost
falls while required admission constraints are violated. Before more PPO, the
next fitting experiment should enforce the existing placement/support limits
explicitly within the solve rather than relying only on weighted residuals.
No limit is relaxed, no reference is promoted and no training, export or viewer
selection changes. All fitting jobs have finished.

Verification: 49 focused tests pass, covering fixed poses, full-versus-window
delayed dynamics, original stance headings, regression frame identities and
hash-bound output. These changes affect offline fitting only.

## 2026-09-23: Isolated forward transition and solver comparison

The first forward transfer is isolated at source frames [20, 35) from
`constrained-retarget-01/forward`. Its toe-only phase is at frames 26–28. The
original interval has 12/13 root-support failures and 19.870 mm penetration;
at frame 25 the swing foot penetrates while the planted left foot remains fixed.
Support-moment failures also occur before and after toe-off. The production
sparse Jacobian matches independent directional finite differences to approximately
0.015% relative error on this interval; missing sparsity dependencies are not
supported as the main explanation for poor progress in this test.

Matched twenty-evaluation diagnostic fits use the same poses, contact schedule,
targets and permitted landing offsets:
- `transition-probe-01`: default sparse LSMR, 55.844 seconds, cost 139.845,
  9/13 root-support failures, 5.927 mm penetration and 42.850 mm foot error.
- `transition-tight-probe-01`: tighter iterative tolerances (1e-10, 5000 inner
  iterations), 59.375 seconds, cost 80.241, 3/13 root-support failures. Penetration
  and foot error still fail. This diagnostic does not change production defaults.
- `transition-exact-public-01`: dense exact solve through SciPy's public interface,
  142.219 seconds, cost 29.091, 0/13 root-support failures, 0.945 mm penetration
  and 36.981 mm foot error. Every geometry check passes; delayed PD still fails
  at source frame 22 (one of twelve evaluated post-delay frames). Full motor
  authority has no failing frames. The exploratory colored-Jacobian dense probe
  (`transition-dense-probe-01`) reproduces the same result in 56.422 seconds,
  but that private-API experiment is not used in the maintained implementation.

This establishes better local convergence with the dense solve, at higher runtime
cost through the public interface. It does not establish a complete trackable
motion or identify physical joint limits as the remaining cause. `transition_probe`
records baseline/candidate metrics and source-frame failures, verifies source/rig
hashes, preserves cropped landing identities and explicitly produces no training
manifest. The sparse solver remains the full-clip default; exact mode is opt-in.

The raw splice back into the original clip is also tested in memory
(`transition-splice-probe-01.json`). It still fails full-reference admission:
88/118 root-support failures remain, aggregate force consistency now fails, and
the joins change root acceleration (frame 34 lateral acceleration changes from
-3.985 to -8.083 m/s²). A free-endpoint local solution cannot simply replace the
original frames. Next: optimize overlapping intervals with preserved surrounding
poses and delayed-command context, then rerun full-reference gates. This is a
reference-continuity task before any further PPO, not another reward adjustment.

A separate audit defect is fixed: the `motion_dynamics` file entry point ignored
`sole_contact`. On the full forward candidate this incorrectly reported 82 rather
than 100 root-support failures by borrowing inactive/projected support. It now
honors explicit soles in both root and actuator diagnostics; legacy files retain
their original semantics. Admission validation already used explicit soles, so
previous rejected manifests remain valid. Reproduction output is saved in
`transition-contact-audit-01.json`.

Verification: 44 focused tests pass, including exact-solver probing, cropped
landing offsets, diagnostic-only output and explicit-versus-legacy file audits.
No physical model, controller contract, reference admission threshold or selected
Godot policy changed. No PPO run, export or background fitting job remains active.

## 2026-09-23: Bound-aware IK and controlled landing adaptation

`contact_projection.py` now redistributes an IK correction with bounded least
squares when a hinge reaches its limit. Previously, clipping the unconstrained
correction discarded motion that other hinges could supply. Regression fixtures
cover redistribution, unchanged joint limits and fixed toe pivots.

The stable box SAT query now applies to every force fit, including candidates
without a positive clearance request. Positive clearance fitting aims 1 mm beyond
the requested value, while independent admission retains the original threshold.
Optional `--adjust-landings` optimizes an absolute XY correction per uninterrupted
stance, bounded to 2 cm per axis and shared across flat/toe-only phases. Original
foot targets remain the validation baseline; continuation preserves the saved
offsets without accumulating them. These are reference-fitting changes only.

The frame-64 backward diagnostic (`ankle-constraint-diagnosis-01.json`) identifies
right ankle roll: its delayed-PD lower command bound was -0.417952; a feasible
solution uses -0.429152 when that motor alone is diagnostically given full authority.
No motor limit or action window was changed. The subsequent pose fit resolves the
failure inside the existing command envelope.

`constrained-retarget-01/backward` completes 25 evaluations / 887.484 seconds:
all 118 root-support and 117 delayed-PD checks pass. Foot error is 37.850 mm,
penetration 1.410 mm and stance sliding is numerical zero. Minimum self-clearance
improves from 1.907 to 2.856 mm, still below 5 mm; the closest pair is now
Shin_L/Toe_L at frame 116. It remains rejected. The fit is nonconverged, and these
necessary-condition improvements do not demonstrate closed-loop recovery.

`constrained-retarget-01/forward` completes 25 evaluations / 1103.281 seconds.
It still fails 100/118 root-support checks and has 14 delayed-PD failures among
the root-supported frames. Penetration is 19.870 mm, foot error 40.116 mm and
stance sliding 0.20946 m/s. Sliding is worse than the source candidate's 0.08291
m/s, despite a lower fitting cost. This is rejected evidence, not a forward-motion
improvement. The remaining stance error cannot be explained solely by clipping
the old unconstrained IK step.

`landing-retarget-01/forward` tests the optional landing variables from the same
parent: ten evaluations / 446.906 seconds. It fails 102/118 root-support checks
and 12 conditional delayed-PD checks, matching the fixed-landing continuation at
ten evaluations. Stance slide is 0.22730 m/s, penetration 21.394 mm and foot error
40.506 mm. The largest landing correction is only 0.334 mm per axis; the optimizer
has not approached its 20 mm bound. Source foot targets and explicit support
phases are exactly preserved. The local frame-94 benefit from a 2 cm shift has
not become a useful whole-trajectory solution within this budget. No claim of
better tracking or inadequate landing range follows from this nonconverged fit.

Next diagnostic boundary: isolate the forward stance-to-step failure and solve
its contact/pose/dynamics compatibility before another full-clip optimization.
Backward still needs clearance at the supporting left shin/toe pair. Neither
candidate is admitted to PPO, and changing rewards cannot substitute for passing
reference validation. No PPO was launched, no export/viewer was replaced and no
fitting or training process remains running after these bounded experiments.

Verification: 41 focused tests pass, including phase-constant landing offsets,
toe-off anchoring, original-target preservation, continuation and invalid-seed
rejection. Native model, motor authority and Godot viewer selections are unchanged.

## 2026-09-23: Native failure diagnosis, explicit toe-off and stable clearance fitting

Substep audit: `logs/mimickit-steps/tracking-audit-04/report.json`, using the admitted
backward reference and the same eight native phases. Baseline zero-residual PD
survives 1.661 seconds on average, with 0/8 completions. Prefilling the command
queue gives 1.552 seconds and 0/8; startup delay alone does not explain the failure.
The existing Stand actor gives 1.327 seconds and 0/8. Removing self-collision in a
private diagnostic model gives 2.163 seconds, still 0/8, with greater mean root
error. No-self-contact forces are exactly zero while floor forces remain active.
Self-collision contributes to failure, but removing it does not establish tracking.

In baseline phase 0, forearm/thigh contact exceeds 5 N at 0.0625 seconds, root error
first exceeds 5 cm at 0.2834 seconds, and both feet first unload at 0.6459 seconds.
No motor saturates before that first 5 cm error in this phase. Other phases first
contact opposite feet/toes or shin/toe pairs. The reference itself leaves sub-mm
clearances and small foot/toe overlaps. These observations localize a clearance
problem and early root lag; they do not prove that actuator authority is adequate
in every phase or that self-contact is the sole cause.

Implementation:
- `motion_tracking_audit.py` records actual physics-substep controls/contact forces,
  acceleration and state, truncating each world at its first failure. Diagnostic
  interventions never change production resets, model files or exports.
- Optional `force_retarget --toe-off` produces a v6 reference with explicit
  Foot_L/Foot_R/Toe_L/Toe_R contact phases. Toe-only IK fixes the toe anchor while
  allowing heel lift. Support forces and sliding use the declared soles; raised
  corners cannot contribute projected support. Existing v5 semantics are preserved.
- `--self-clearance 0.005` fits and validates a declared 5 mm clearance between
  collision-enabled body pairs, exempting the floor. Continuations inherit it.
- Native `mj_geomDistance` was observed to jump from 6.2319 mm to zero for a 1e-8
  radian perturbation of pose coordinate 41 at backward frame 86. This stalled
  the first clearance fit with an artificial gradient near 1e8. Box clearance now
  uses a conservative, stable separating-axis bound; other shapes retain the
  native query. Actual MuJoCo penetration remains an independent gate.

Forward candidate: `toe-retarget-01/forward`, ten evaluations / 382.297 seconds.
Stance slide is 0.08291 m/s, passing the unchanged 0.15 m/s limit; the prior flat
candidate measured 1.081 m/s. It remains rejected: penetration 21.05 mm, foot error
40.52 mm, an active sole corner 2.395 mm from the floor, and 104/118 unsupported
frames under the new explicit-phase test. The root count is not directly comparable
to v5's optimistic support test. This is a contact-model improvement, not a usable
walking controller. No PPO or viewer replacement follows a rejected reference.

Backward clearance candidate: `clearance-retarget-02/backward`, ten evaluations /
317.031 seconds after fixing the distance derivative. Minimum separation improves
from a small overlap to 1.907 mm, below the requested 5 mm. All root-support and
geometry checks pass, but delayed-PD feasibility fails at frame 64, also the
closest Foot_L/Toe_R pair. The candidate is rejected and cannot replace the older
admitted reference. Solver cost reduction does not establish tracking improvement.
The earlier native-distance run (`clearance-retarget-01`) is retained as failure
evidence. No optimizer or training job remains running after this investigation.

Remaining work is localized: reconcile swing-foot clearance and actuator demand
around backward frame 64, and complete forward stance geometry (worst active
corner at frame 94) before another motion pilot. These bounded, nonconverged fits
do not establish that the existing rig is incapable of the motion. No rig, runtime
contact semantics, selected Godot policies or packaged references were changed.

Verification: 37 focused tests pass, including toe-pivot anchoring, near-floor
corner filtering, legacy admission/export compatibility, clearance gating and
stable conservative box distances.

## 2026-09-23: Joint contact/actuator fitting and bounded stance IK

The offline reference fitter now solves root support and controlled-joint demand
together inside the existing delayed target-PD envelope. It reconstructs stance
legs with bounded six-hinge IK, keeps ankle yaw adjustable, and permits bounded
pelvis orientation corrections. The runtime-derived PD bounds are shared with
the independent validator; contact-independent motor rows are eliminated exactly
from the inner bounded solve. Rig geometry, motor strength and action limits are
unchanged. Fresh references start 2 cm lower for bent-knee IK initialization;
continuations preserve the fitted height.

`actuated-retarget-02/backward` removes the previous 9 full-motor failures and
reduces delayed-PD failures from 22 to one across 117 evaluated frames. The next
20-evaluation continuation (`actuated-retarget-03/backward`, 526.672 seconds)
removes that last PD failure. Root support passes all 118 frames, planted-foot
sliding is below 1e-12 m/s, and penetration is 0.734 mm. It still fails
the swing-foot tracking threshold: 40.534 mm versus 40 mm. Necessary-condition
improvements are not evidence of learned walking or better impact recovery.

The final ten-evaluation continuation (`actuated-retarget-04/backward`, 293.985
seconds) passes every unchanged v5 admission check: 0/118 root-support failures,
0/117 motor/PD failures, 39.949 mm maximum foot error and 0.703 mm penetration.
The validation callback stops at acceptance; optimizer convergence is not claimed.
This is the first admitted moving reference under the full v5 criteria.

Native tracking (`logs/mimickit-steps/actuated-probe-01/report.json`) still fails
all eight trials for both diagnostic controllers. Mean survival is 1.661 seconds
with zero-residual PD and 1.327 seconds with the unchanged Stand v2 actor, versus
1.629/1.265 seconds on the prior root-only fit. These small diagnostic changes
do not establish useful tracking; neither controller has learned this reference.

A fresh, isolated upstream PPO motion pilot then ran on the RTX 4080 SUPER with
2048 environments, `support_v2`, native evaluation every 16 updates and seed
230923 (`logs/mimickit-steps/actuated-backward-pilot-01`). It completed 46 updates /
3,014,656 transitions in 180.843 seconds. Native success stays 0/8; mean survival
changes from 1.673 to 1.629 seconds and root error from 0.2281 to 0.2337 m.
Newton also scores 0/8 (1.652 seconds). Checkpoint selection rejects every trained
candidate and retains iteration 0, so the run's export uses the initial weights
from `best.pt`. The final candidate is preserved as `model.pt`. ONNX action parity
error is 1.86e-8. This short pilot does not establish that longer learning cannot
help, but supplies no evidence for a longer Perturb run. Selected Godot Stand and
Perturb bundles and packaged motion assets remain unchanged.

Next: use the saved native traces to localize the first closed-loop contact/torque
failure on the admitted backward reference, and replace forward's flat-foot
contact-gap assumption with heel-lift/toe-only support. Do not claim a policy
improvement from reference admission alone.

Forward (`actuated-retarget-02/forward`, 30 evaluations, 1117.907 seconds) remains
rejected: 83/118 root-support failures, 1.081 m/s stance sliding, 47.949 mm maximum
foot-target error and 16.097 mm penetration. The largest slips are at frames
28 (left), 47 (right) and 68 (left), where missing source contact labels were
filled by extending flat-foot stance. At frames 28 and 68 the stance hip reaches
its extension limit. This identifies the flat-foot transition assumption as a
limitation of this fit; it does not justify changing the skeleton's joint limits.
Forward needs a heel-lift/toe-support transition model rather than another PPO run.

Verification: 27 targeted tests pass, including stance locking under pelvis
translation/rotation, equivalence of reduced/full bounded force solves, and
delayed-PD bounds against runtime output with moving targets and target velocities.

## 2026-09-23: Force-aware retargeting implemented; actuator/contact constraints still block motion training

`force_retarget.py` now refines the geometric references against unilateral
foot-plus-toe contact forces, an inner friction pyramid and inset support points.
Grounding is included in every optimization evaluation. It completes missing
walking contact intervals, extends the existing landing anchors and reconstructs
central tangent velocities. `retarget_steps.py` runs this phase after geometric
fitting; the separate CLI can reuse hash-verified archived fits. No rig, torque,
controller, policy weights or selected Godot exports were changed.

Final candidates: `logs/mimickit-steps/force-retarget-04`. The forward fit used
60 function evaluations (532.922 seconds). Backward used the preceding 60-evaluation
root-only fit plus 20 further evaluations (153.594 seconds) with complete sole
support. These bounded fits reached their evaluation budgets, not convergence.

- Backward: root-support failures fall from 64/118 to 0/118; all geometry checks
  pass. Maximum stance speed is 0.01874 m/s and foot-target error 0.01426 m.
  On the 117 post-delay frames, 9 fail the full motor envelope and 22 fail the
  existing +/-0.25-radian delayed PD window. Doubling it still leaves 20 failures.
- Forward: 92/118 root-support failures remain (baseline 95). Maximum stance speed
  is 1.094 m/s against the 0.15 limit; foot-target error is 0.0581 m against 0.04.
  The fit reduces its objective but does not produce an acceptable trajectory.
  Filling the source's 11 missing-support frames and re-anchoring the extended
  stance targets is insufficient to make these transitions dynamically consistent.
- Matched backward tracking: both zero-residual PD and the unchanged Stand v2 actor
  still fail all eight trials. Zero-residual mean survival rises from 1.433 to
  1.629 seconds; the Stand actor falls from 1.315 to 1.265 seconds. This is a
  reference-support improvement, not learned walking or improved policy behavior.
- Simply slowing the backward clip by factors 1.25/1.5/2 reintroduces 17/31/44
  unsupported frames. Timing cannot be changed independently of support dynamics.

New motion training now requires `mimic_step_reference_v5`: the final validation
also gates on actuator feasibility within the existing delayed PD window. The
v4 optimization outputs remain diagnostic archives. Byte-identical references
with v5 manifests and the complete comparison are saved under
`logs/mimickit-steps/force-retarget-review-01`; neither candidate is admitted.
Other evidence: `force-probe-02/report.json`, `force-backward-dynamics-02.json`
and `force-backward-timing-01.json` under `logs/mimickit-steps`.

Verification: 21 targeted tests pass, covering friction, velocity reconstruction,
stance extension, static support, reference admission and export compatibility.
Next: jointly enforce contact-transition consistency and the existing actuator/PD
limits during trajectory fitting, then repeat tracking. Do not compensate with
stronger motors, wider action windows or a longer Perturb PPO run. No new training
was started and the packaged references remain unchanged.

## 2026-09-23: First Perturb v2 pilot learns settling but fails directional retention

Run: `logs/mimickit-training/perturb-support-v2-15min-01`, initialized from
`stand-support-v2-15min-01/best.pt` and its matching v2 contract. Guarded PPO,
reference reward, 2048 CUDA worlds, 25% quiet episodes, speeds 1.0–2.5 m/s,
seed 210921, native evaluation every 64 updates. The unchanged 15-minute budget
completed 133 updates / 8,716,288 transitions in 900.188 seconds.

- Fixed 96-case validation: starting Stand v2 survives 44 and settles after 25;
  final candidate survives 40 and settles after 36. Update 128 scored 43/37.
- All evaluated trained candidates failed directional retention. Final survival
  fell from 16 to 14 in direction +Y (including torso 3 to 2), and from 4 to 2
  in direction -X. Forward/back torso survival remains 0/6 in both directions.
- The independent 120-case protocol was frozen before final scoring (seed
  23092317). Start: 47 survived / 31 recovered. Final: 57 / 35. Existing legacy
  Perturb: 100 / 21. These distinguish survival from final settling; the final
  candidate has mixed strengths and is not a replacement for either baseline.
- Final quiet standing retains 8/8 at three and five seconds. Five-second root
  error is 2.44 cm; mean foot RMSE rises from 1.96 to 5.19 mm and reference-relative
  foot velocity RMS from 0.00364 to 0.00646 m/s relative to its Stand v2 parent.
- GPU/native final survival agrees at 40/96; recovery counts differ by one
  (37 GPU, 36 native). Actual Godot/native replay passes all 96-case parity gates.
  PPO accepted all 5320 actor steps; no KL update was rejected. The unsuccessful
  promotion was a behavioral-retention decision, not a rejected optimizer update.

`best.pt` / `export` retain the unchanged starting Stand v2 actor (iteration 0).
The trainer's automatic viewer selection was rolled back to the previous legacy
Perturb export. Stand v2 remains the selected Stand policy. The final trained
candidate is retained separately as `model.pt` and `final-diagnostic/export`;
the latter is explicitly diagnostic-only and cannot seed training or auto-selection.
Use `watch.ps1 -Run logs/mimickit-training/perturb-support-v2-15min-01/final-diagnostic
-BallSpeed 2.5` for explicit inspection without changing the default selection.

Evidence: run `report.json` / `final-random-test.json`, and
`logs/mimickit-contact-v2/perturb-pilot-random-baselines-01`,
`perturb-final-godot-01`, `perturb-final-quiet-01`. The random suite is now observed
regression evidence, not an untouched test for future decisions. No longer PPO
run or reward/plant change was made. Next priority: force-aware stepping-reference
retargeting and isolated forward/back tracking before another long Perturb run.

## 2026-09-23: Fresh Stand v2 pilot passes standing and Godot parity

Run: `logs/mimickit-training/stand-support-v2-15min-01`. Fresh upstream PPO with
2048 CUDA worlds, seed 210921, corrected contact v2 and unchanged target-PD plant.
The 15-minute budget completed in 901.797 seconds: 310 updates / 20,316,160 samples.
Native evaluation every 16 updates selected checkpoint 240 (15,728,640 samples).

- Initial actor: 0/8 three-second successes, mean survival 1.81 seconds.
- Selected actor: 8/8 at three and five seconds, root error 1.85 / 2.36 cm.
- Native/Newton three-second outcomes agree. Actual Godot standing replay passes
  all gates; action error < 2.69e-6 and pose component error < 3.91e-7.
- Matched five-second legacy Stand (`guarded-finetune-01`) also passes 8/8,
  but has 3.32 cm root error. Mean foot tracking RMSE is 8.91 mm legacy versus
  1.96 mm v2; reference-relative foot velocity RMS error is 0.03689 versus
  0.00364 m/s. Action changes are also smaller. These are numerical movement
  proxies, not perceptual realism or recovery scores.

The new `stand_diagnostics.py` uses the existing native evaluator and each actor's
own versioned contract on identical phases; reset frames after failure are excluded.
Reports: `logs/mimickit-contact-v2/stand-baselines-01` and `stand-trained-v2-01`.
This is one fresh seed compared with established legacy actors, not a controlled
comparison proving the contact change alone caused the improvement.

The export is automatically selected for `MimicStand.tscn` (F6). F5 remains
`MimicPerturb.tscn` with its prior ball policy. Next: a bounded Perturb fine-tune
from the new v2 checkpoint with quiet-standing retention. No ball or Walk training
started here; walking still requires force-aware reference retargeting.

## 2026-09-23: Versioned contact correction and validation

Implemented `mimic_stand_target_pd_v2` / `ContactMode=support_v2`: last-solved-substep
contacts, per-side foot-plus-toe normal loads, reset-forward snapshots and preserved
snapshots on ball launch. Partial GPU resets now preserve non-reset worlds' contact
readings in v2. Native and Godot cache before the post-step forward refresh.
Legacy actor inputs/timing and weights/normalization are preserved; warm starts
reject incompatible contracts. Current config remains legacy for existing checkpoints.

Frozen-actor validation on identical 96-case impacts: legacy 72 survived/17 recovered,
timing-only 73/18, full v2 67/17. All pass 8/8 quiet phases at both 3 and 5 seconds.
The v2 diagnostic is not a trained v2 policy and cannot seed training or replace
the viewer selection. No training or viewer promotion occurred.

Native/Godot ball parity passes all 96 cases for both versions; standing parity
passes both. GPU/native v2 contact fixtures, eight-shot launch/contact/reset parity
and task preflight pass. Evidence: `logs/mimickit-contact-v2`.
Final verification: all 87 Mimic Python tests and 145 managed tests pass (one
opt-in native test skipped); the Godot C# project builds with its six existing warnings.

New motion training requires a v4 retarget manifest with root-wrench support in
addition to geometry and aggregate force. Archived v3 clips remain replayable but
cannot start new motion training. The dynamic action-window diagnostic confirms
95/118 forward and 64/118 backward root-support failures. Widening targets cannot
repair those. All 177 tested standing frames after queue fill fit existing motor
and +/-0.25 rad limits. See `dynamics-01.json` and the Mimic README for conditional
moving-reference results and limitations.

Remaining: generate dynamically supported walking references using force-aware
trajectory optimization, then establish tracking success. A fresh v2 Stand policy
is required before v2 Perturb training. Falling/bracing/get-up remain separate,
deferred capabilities; no stronger motors or replacement skeleton was introduced.

## 2026-09-23: Foundation audit identifies two contact-observation defects

See [the full audit](mimic/FOUNDATION_AUDIT.md) for evidence, limits and reproduction.
Diagnostics and reports are under `mimic/foundation_audit.py`,
`mimic/foundation_sensor_parity.py`, and `logs/mimickit-foundation`.

- Actual actor inputs omit toe loads on both GPU and native backends. In a toe-only
  fixture, the inputs are zero while the toes carry approximately 41 N.
- GPU contact forces come from the last solved substep; native and Godot recompute
  them after integration. The fixtures show a maximum 50.004 N difference, falling
  below 0.004 N when the same substep is compared. Poses stay closely matched.
- All structural checks and all 30 motor transmission/sign probes pass. Fresh
  65-pose Newton compatibility validation passes its existing gates.
- Seven of nine fitted elementary poses admit conditional static support within
  the available torque budget. Two in-place lift fits remain inconclusive. None
  of the zero-residual fixed-target ten-second holds succeeds; this does not test
  the trained actor or establish that the skeleton cannot learn balance.

Prioritize a consistent contact sampling convention and a versioned foot-plus-toe
observation before further PPO/retargeting tuning. Existing rig, reward, actor
contract, policy weights and selected Godot export are unchanged by this audit.

## 2026-09-23: Step-reference acceleration fix and remaining support-moment defect

The prior `retarget-04` references passed geometry checks but contained large
frame-to-frame acceleration spikes. The source pelvis motion is smooth; independent
IK frames introduced the jumps. This is a reference-generation defect, not evidence
that the rig requires stronger motors or a different simulator.

`retarget_steps.py` now refines the full trajectory with sparse least squares,
penalizing root/joint acceleration while retaining moving-foot fits, collisions and
joint limits. Grounding and all validation run after refinement. The packaged
references now come from `logs/mimickit-steps/retarget-05` and use
`mimic_step_reference_v3`. New training rejects older contact-only manifests.

- Both directions pass the nine geometry checks plus the new aggregate force check.
- Frame-second-difference peak horizontal COM acceleration fell from 102.72 to
  8.34 m/s² forward and 68.25 to 4.26 m/s² backward.
- Old references exceeded the optimistic aggregate friction/normal-force envelope
  on 69/118 and 43/118 interior frames; both new references pass every frame.
- The new references retain foot lifts, with maximum stance sliding below 0.060 m/s
  and inferred stance gaps below 0.63 mm. Physical rig/control contracts are unchanged.

Native zero-residual mean survival improves from 0.821 to 0.988 s forward and
1.402 to 1.433 s backward, still 0/8 full three-second trials. A zero-delay
counterfactual yields 0.996/1.433 s, also 0/8: the two-step motor delay is not the
main cause in this probe. Reports: `tracking-probe-03-forward`,
`tracking-probe-03-backward`, and `control-delay-probe-01.json` under
`logs/mimickit-steps`. The delay probe does not modify production control.

A matched three-minute forward pilot (`forward-pilot-03`, 2048 CUDA environments,
seed 230923, upstream PPO, evaluation interval 16) completed 44 updates and
2,883,584 transitions. Its starting policy survived 0/8 (mean 0.990 s); the
selected policy survived 1/8 (mean 1.219 s), compared with the previous reference's
0/8 and 0.840 s. Root/foot tracking still fails. This is a limited improvement in
one short, single-seed validation run, not a working step or Perturb recovery.

`motion_dynamics.py` also provides a reproducible root-force/moment diagnostic.
It uses central tangent differences and optimistically allows unlimited joint
torque, projected sole footprints and a square outside the Coulomb friction cone.
The regularized references still fail this approximate root-wrench test on 95/118
forward and 64/118 backward frames (old references: 114/118 and 113/118).
Hashes and frame indices are in `reference-dynamics-01.json`. The aggregate force
gate alone cannot detect this support-moment defect. Root-wrench results are
diagnostic, not a claim that RL cannot adapt an imperfect motion reference.

Next work: incorporate support forces/moments into retargeting and validate dynamic
tracking before another Perturb run. No additional backward PPO pilot was launched
after this finding. The current Godot Perturb selection remains unchanged.

Verification: 64 targeted Python tests cover force spikes, missing contact support,
static wrench feasibility, legacy-manifest rejection, existing tracking/export
contracts and Perturb regression tests. No C# or rig change in this investigation.

## 2026-09-23: Licensed moving-foot references and isolated tracking pilots

Added pinned 100STYLE `Neutral_FW.bvh` and `Neutral_BW.bvh` acquisition, using the
author's CC BY 4.0 license and commercial-use attribution statement. These are
walking priors, not recorded impact recoveries. `assets/steps` contains four-second,
non-looping references from frames [480, 720) and [520, 760), stride 2, with complete
source/derived hashes and attribution. The standing asset and physical rig are unchanged.

The standing retargeter intentionally pins both feet and cannot produce stepping
examples. New `retarget_steps.py` instead fits moving feet and segment directions,
locks inferred stance intervals, limits frame-to-frame joint discontinuities, and
checks collisions, joint limits/speeds, foot lift and tracking. A subsequent contact
audit caught a missing condition: avoiding penetration did not prevent floating
support. The superseded `retarget-03` backward clip had both soles over 1 mm above
the floor on 89/120 frames. Stance targets are now grounded, walking root height is
projected onto the lowest sole, and explicit support/stance-contact checks reject
floating references. This changes reference data, not motor strength or physics.

References at this stage came from `logs/mimickit-steps/retarget-04`
(superseded by the acceleration-regularized assets above):

- Both pass all nine kinematic checks and maintain a supporting sole on every frame.
- Maximum inferred stance gaps: 0.21 mm forward, 0.13 mm backward.
- Maximum penetration: 0.24 mm forward, 0.08 mm backward.
- Foot lift ranges: 13.1/13.7 cm forward, 8.1/10.1 cm backward.
- Maximum stance sliding: 0.032 m/s forward, 0.0066 m/s backward.

`probe_steps.py` measures three-second native tracking over eight start phases and
compares zero-residual PD with the old standing actor under an explicit, contract-
checked reference substitution. With the corrected references, neither survives
any full trial. Mean survival is 0.821/1.402 s for zero residual forward/backward and
0.827/1.175 s for the standing actor (`tracking-probe-02`). This establishes a failed
tracking gate, not that learning the motion is physically impossible.

The old standing normalizer clips over half of future-reference components on these
new motions. Initial learning therefore uses separate fresh policies with adaptive
upstream PPO normalization, rather than silently bypassing warm-start contracts or
reusing frozen standing statistics. `train_stand --task motion` keeps the existing
Newton/MuJoCo-Warp GPU physics, 362 observations and target-PD action contract. It
requires `--no-select-viewer`, retains checkpoints by native tracking metrics, and
does not run a five-second test beyond these four-second clips. Shared motion gates
require survival, root/foot/joint tracking and at least 3 cm lift range on each foot.
The motion trackers are independent; an impact-conditioned policy is not implemented.

The first three-minute-per-direction pilots (`*-pilot-01`) used superseded retarget-03
and failed all eight tracking trials. They are retained as diagnostics, not evidence
for the corrected references. The forward run completed 44 updates before hitting
an export filename/attribution assumption. Export now validates inputs before training
and writes canonical Godot bundle names for any reference filename. Its saved best
checkpoint was recovered without retraining into `forward-pilot-01/recovered-export`;
the interrupted run's partial export and missing final training report are preserved.
`export_motion_checkpoint.py` supports this recovery with strict checkpoint/reference
hash checks and its own reevaluation report.

Corrected-reference runs `forward-pilot-02` and `backward-pilot-02` each completed
42 PPO updates and 2,752,512 transitions, with budgets of 183.81 and 180.86 seconds.
Both used 2048 CUDA environments, upstream PPO, seed 230923, fresh policies/adaptive
normalizers and evaluation interval 16. Their final checkpoints were retained:

- Forward mean native survival: 0.831 -> 0.840 s; 0/8 full three-second trials.
- Backward mean native survival: 1.411 -> 1.431 s; 0/8 full three-second trials.
- Both fail root/foot tracking gates. Neither is a learned recovery step or a
  candidate for Perturb integration. These short runs do not establish whether
  longer training will solve tracking; survival is essentially flat so far.
- ONNX action errors: 5.22e-8 and 4.66e-8. A separate headless Godot motion scene
  loads the forward bundle with a three-second trial and reproduces native's
  phase-zero fall at 0.383 s. This is a runtime smoke test, not a full parity sweep.

Grounded references use `mimic_step_reference_v2`; new motion training rejects
older validation or missing support-contact gates before GPU setup. Pilot-02
exports' reference manifests carry this corrected validation version; their actor
and reference-array hashes are unchanged. Older pilot assets remain archived and
must not be substituted for the current references in future training.

Verification: 58 targeted Python tests pass, covering source/reference integrity,
ground support, stance-lock isolation, tracking selection, contract substitution and
ONNX/canonical-name export. No Godot scene, Perturb reward, rig or viewer selection
was changed. Reproduction commands are in `mimic/README.md`.

## 2026-09-23: Directional retention and matched torso diagnosis

The user's forward/back visual concern was confirmed. The previous aggregate
summary hid a backward regression between `perturb-recovery-v1-15min-01` and the
currently selected `perturb-recovery-v1-30min-02` export:

- Forward (+X): 12 -> 17/24 survived, 0 -> 5 recovered. All five new recoveries
  were arm/leg shots; none were Chest/Spine/Pelvis impacts.
- Backward (-X): 16 -> 12/24 survived, 1 -> 0 recovered.
- Backward torso: 2 -> 0/6 survived. Neither export settles after any forward/back
  torso case in the fixed selection suite.

`checkpoints.py` now applies `directional_retention_v1`: compared with the incumbent,
confirmed-hit, survival and final-settling counts cannot decrease in any cardinal
direction or its torso subgroup. Only eligible candidates enter the existing ranking.
It requires consistent per-case evidence and identical case definitions/recovery
criteria; it does not claim individual-case retention or statistical generalization.
The retained metadata stores subgroup floors. Learning-curve rows and the final
report expose rejection reasons. Replaying the new rule against the previous run's
saved metrics rejects updates 64, 128, 192 and 247; it would retain the starting actor.
Historical checkpoints, reports and current viewer selection were not rewritten.

Added `ball_diagnostics.py`, recording original delayed PD commands at every physics
substep and post-step feet, loads, pelvis state and termination evidence. It compares
the two exported ONNX policies under identical impacts and matched quiet trials,
excluding reset frames after a scored episode ends. No dynamics or actor changes
are applied. Outputs:

- `logs/mimickit-training/torso-diagnostics-01`: original 1/2.5 m/s suite; both actors
  reproduce every original case's hit, survival, recovery and first-contact result.
  All matched quiet torso trials survive. Every failed torso trial terminates at
  pelvis height < 0.65 m, before a non-foot ground-contact termination.
- In the selected actor's six backward torso trials, no hip/knee/ankle command
  reaches 95% of its configured motor limit and no leg residual reaches its action
  bound. Maximum change in front/back foot separation is only 0.9-1.7 cm. At
  termination the pelvis is roughly 0.67-0.70 m behind the foot origins, tilted
  52-55 degrees. The roughly 8 cm foot-origin rise is not evidence of a recovery step.
  Forward failures sometimes saturate toe commands; hip/knee/ankle saturation is
  negligible (at most 0.16% of joint/substep samples in one selected-policy case).
- `logs/mimickit-training/torso-diagnostics-2mps-01`: at 2 m/s and phase 0.8, both
  policies fail all six forward/back torso shots.
- `logs/mimickit-training/torso-diagnostics-f5-01`: at F5's default 2 m/s, phase 0
  and launch step 60, with each torso target fixed in turn, the old actor survives
  0/6 and the current actor 1/6 (forward pelvis). Both settle 0/6. F5 normally
  randomizes the target body, so these are controlled comparisons of its settings.

This supports a missing learned recovery step as the next hypothesis to test.
Increasing torque or the residual range is not justified by the backward traces.
The next experiment should establish forward/back recovery-step motion tracking
under the existing rig/bridge, then test impact fine-tuning with directional retention.
Standing-only imitation and another unchanged training extension have not established
torso recovery. The probe does not prove that all control/observation choices are optimal.

Verification: 46 targeted Python tests pass. New tests cover directional/torso/settling
regressions, inconsistent or changed case evidence, unchanged delayed substep controls,
and exclusion of pre-impact/post-termination frames. No training was launched; reward,
rig, action/observation contracts, config defaults and F5 selection remain unchanged.

## 2026-09-23: Thirty-minute recovery continuation and independent impact test

`logs/mimickit-training/perturb-recovery-v1-30min-02` warm-started weights and
normalization from `perturb-recovery-v1-15min-01/best.pt`, with a fresh optimizer.
Reward `recovery_v1`, physics, 2048 environments, seed 210921, interval 64 and the
1.0–2.5 m/s curriculum were held fixed. Training completed 1803.09 seconds,
247 updates and 16,187,392 transitions. No reward or control-contract adjustment
was made during the experiment.

The unchanged 96-case native selection suite reports:

- Starting policy: 70 survivals / 6 final settled recoveries.
- Update 64: 70 / 12; update 128: 72 / 17; update 192: 64 / 18.
- Final update 247: 65 / 16. The existing survival-first selector retains update 128.
- All these checks pass quiet standing. The selected policy additionally passes
  8/8 five-second standing cases, with 0.0413 m mean root error.
- Selected Newton results are 73 survivals / 19 recoveries versus native's 72 / 17;
  the report preserves these backend differences.

Added `ball_generalization.py`, a standalone evaluation runner with no training,
checkpoint-selection or viewer-selection writes. Before the first periodic candidate
score, it froze `random-test-protocol.json`: seed 230923, 120 cases (ten per body),
continuous random angles/speeds/phases and randomized launch steps from 60 through 120.
The cases stay within the training disturbance family and finite five-second reference
window. The protocol binds model/reference hashes and was not supplied to training.

After training, the two exported actors were evaluated on that same independent suite:

- Starting export: 120 confirmed hits, 99 survivals, 8 settled recoveries.
- Selected continuation: 120 confirmed hits, 100 survivals, 14 settled recoveries.
- Mean survival: 4.7175 -> 4.7247 seconds. Root tracking error: 0.0620 -> 0.0632 m.

These results support modest settling improvement on unfamiliar cases; survival is
nearly unchanged there. They are one training seed and one finite test suite, not
broad or long-horizon robustness. Further training was not monotonically beneficial:
the final policy lost seven survivals relative to the retained checkpoint. Preserve
the current objective/candidate rather than assuming a longer run is automatically
better. No plateau-triggered motion-reference change was made in this experiment.
After using this test to guide another change, reserve a new seed for independent
testing and retain these cases for regression checks.

Verification: 40 targeted Python tests pass, including randomized schedule coverage,
reference bounds and partial-reset isolation. All eight actual Godot/native ball
parity checks pass for the selected export, with maximum action error 9.75e-6 and
pose-coordinate error 1.37e-6. ONNX maximum action error is 3.58e-7. All 9880 actor
steps were accepted. Rollout/optimization consumed 1488.43/115.31 seconds; periodic
native validation consumed 194.67 seconds within the budget. Initial/final native
work added 67.05/67.77 seconds outside it.

F5 now selects this run's export automatically. The previous viewer selection remains
the 15-minute export and is available through `select.ps1 -Previous`. Detailed results
are in the run's `report.json`, `random-test/report.json`, and `godot-validation/report.json`.

## 2026-09-23: Automatic F5/F6 export selection

At the user's request, completed Mimic training now selects its exported best actor
for the matching interactive Godot scene. `logs/mimickit-viewer/{ball,stand}.json`
stores an atomic current/previous selection; all original run exports are preserved.
These files are already Git-ignored by `logs/`, and project-local bundle paths use
`res://`. Selection occurs after the completed report and ONNX export checks, even
when the experimental recovery gate fails, so candidates can be reviewed directly.
Training initialization remains separately configured.

The current ball selection is `perturb-recovery-v1-15min-01/export`; the previous
selection is `guarded-perturb-04/export`. F5 launches MimicPerturb with the new policy.
`scripts/select.ps1 -Previous -Experiment perturb` swaps them for rollback; `-Run`
selects another completed run. `train.ps1 -NoSelectViewer` opts out per run.
Explicit launch bundle arguments take precedence, and scenes expose `UseLatestExport`
to retain a manually pinned Inspector bundle. Stand and ball selections are separate.

Verification: four selector tests cover rollback/idempotence, task isolation,
incomplete/corrupt outputs, and review selection despite a failed behavioral gate.
Godot builds without errors or warnings. A headless project-main launch with no bundle
argument loads the new export; an explicit override still loads guarded-perturb-04.
No policy weights, rewards or training initialization were changed for this workflow.

## 2026-09-23: Post-contact recovery reward experiment

Added opt-in `train.ps1 -BallReward recovery_v1` / Python `--ball-reward recovery_v1`.
Quiet, pre-contact and missed-shot worlds retain the exact standing reward. After a
reported ball/character contact, 25% retains root-relative motion imitation and 75%
rewards upright posture at reference height, low root velocities and supported
settling. Support is a small bonus, allowing a foot to lift. The reward parameters
and schema are recorded in the experiment manifest. The physical plant, 362 actor
inputs, +/-0.25-radian target residual, terminal conditions, 96-case validation and
checkpoint selection order are unchanged. The external MimicKit checkout is untouched.

The baseline puts 65% of reward weight on standing joint poses and root-relative
hand/foot positions, which can conflict with stepping. Existing baseline traces also
show that 57 of 58 survivors exceed the horizontal-speed threshold during the final
window; none exceed tilt/angular-speed limits there. The reference's maximum root
horizontal speed is only 0.009 m/s, so the 0.2 m/s settling requirement is not asking
the actor to suppress a fast reference motion. These observations motivate the
experiment; they do not prove that reward shaping alone can produce realistic steps.

`logs/mimickit-training/perturb-recovery-v1-15min-01` starts from the configured
`guarded-perturb-04/best.pt`, with seed 210921, 2048 environments, interval 64 and the
same 1.0–2.5 m/s ball curriculum. It completed 904.14 seconds, 123 updates and
8,060,928 transitions. Initial/final native scores on the same validation cases:

- Confirmed impact survival: 58/96 -> 70/96.
- Final uninterrupted settling: 1/96 -> 6/96.
- Mean survival: 4.270 -> 4.479 seconds; mean root tracking error: 0.0912 -> 0.0796 m.
- Quiet standing passes 8/8 at three seconds, and the selected candidate also passes
  8/8 at five seconds (0.0381 m mean root error).
- Update 64 has 66 survivals and 10 recoveries. It remains saved as
  `evaluation_000064.pt`. The unchanged survival-first selector exports update 123.
  Longer training improved survival but reduced settling relative to update 64.
- Final Newton scoring gives 69 survivals and eight recoveries, versus native's
  70/six. These near-threshold backend differences remain visible in the report.
- 33 targeted Python tests pass, including contact gating, reset isolation,
  translation invariance, quiet-reward equivalence and recovery incentives.
  ONNX maximum action error is 3.58e-7; all 4920 actor steps were accepted.
- The selected export passes all eight actual Godot/native ball parity checks in
  `godot-validation/report.json`: 96 matched hits and 70 matched survivals, with
  maximum action error 1.07e-5 and pose-coordinate error 1.14e-6. Its native ONNX
  replay also reproduces six settled recoveries.

Rollout/optimization used 771.80/61.78 seconds; periodic native validation used
68.00 seconds within the training budget. Initial/final native work used another
69.69/62.56 seconds outside it. The configured reward remains `reference`, with the
new objective available explicitly for controlled comparisons; the viewer bundle
is unchanged. This is one seed on checkpoint-selection cases, not evidence of broad
generalization, long-horizon recovery, or improved visual realism. The strict recovery
gate still fails. Do not read the intermediate/final tradeoff as monotonic progress.

## 2026-09-22: Reduce repeated validation work

The working default is now `Envs = 2048`, `EvaluationInterval = 64`. Full native
validation runs after 64 completed updates, with mandatory initial and final-candidate
checks. The previous update-1 evaluation was an upstream zero-based scheduling artifact.
The PowerShell wrapper accepts `-EvaluationInterval` for a per-run override.

A bounded, run-local evaluation cache keys actor weights, normalization buffers,
physics/reference contracts, backend, case settings and duration. It pins the selected
best checkpoint and reuses its native metrics and traces when exporting; unchanged
final candidates also reuse their last validation. Matching selected/final actors
share their final Newton result. Full quiet-standing checks and the independent Godot
replay workflow remain available; no promotion is performed by the training wrapper.
No cases or recovery thresholds were removed. Less frequent validation trades fewer
CPU interruptions for fewer opportunities to retain a short-lived policy improvement.

Verification: 29 targeted Python tests pass. The eight-update, 128-world integration
run `logs/mimickit-training/perturb-evaluation-cache-smoke-01` used interval 8 and
recorded exactly one periodic validation at update 8. Cache statistics show two
misses (initial and update 8) and two hits (final and selected). Initial/periodic
native validation took 59.83/59.89 seconds; cached final native work took 0.031 seconds.
The matching final/selected Newton result was reused. Quiet standing passed 8/8 and
ONNX maximum action error was 2.98e-7. The candidate's recovery gate failed and the
viewer checkpoint was not changed. This verifies reuse and scheduling, not learning
quality or sustained throughput at the default 2048 environments.

## 2026-09-22: Perturb control parity and validation corrections

The active experimental viewer still uses `guarded-perturb-04`. It is not a validated
recovery policy. No checkpoint was promoted during these corrections.

- Removed the Godot-only scripted stepping controller. The shared Stand/Perturb driver
  again applies the exported reference-relative PD policy without extra target offsets.
  Capture-point information is diagnostic only, counts loaded feet (including toes),
  and reports no support when airborne.
- Fixed scheduled launch order: both Python and Godot observe and predict before
  launching the ball. Launch-time `mj_forward` can change contact-load observations
  when the projectile initially intersects a limb. The previous order produced
  action differences up to 0.0026 despite matching survival outcomes.
- `ball-validation-v2-02/report.json` under `logs/mimickit-perturb/` passes actual
  native/Godot ball parity across 96 cases: 12 bodies x 4 directions x 2 speeds.
  All hit times and survival outcomes match; maximum action error is 1.09e-5,
  pose-coordinate error 1.48e-6, and ball-position error 2.87e-7 m.
- On this five-second validation suite, the retained actor confirms 96/96 impacts,
  survives 58/96, and meets the final uninterrupted 0.5-second settling criterion in
  1/96. Settling requires sufficient pelvis height, low tilt and velocity, and support
  on both feet. This is a diagnostic of short-window recovery, not delayed-fall proof.
  Stand replay also passes all five parity checks and survives all eight phases.
- Previous eight-world multi-body evaluations never aimed at the four leg bodies.
  Historical 4/8 Chest, 7/8 multi-body, and 5/8 higher-speed results used different
  scenarios and cannot be read as a continuous learning curve. `guarded-perturb-04`
  started and finished selection at 5/8; its mean root error was 8.8 cm, not under 4 cm.
- `Envs = 2048` and `EvaluationInterval = 16` remain provisional defaults. The old
  15-minute comparison measured about 7,473 versus 4,507 samples/s for 2048 versus
  1024 worlds, with one additional successful case. The one-minute 4096 run spent
  59.36 of 62.47 seconds reaching its first evaluation; it did not establish a
  sustained optimization-throughput cliff or a CUDA cache-thrashing diagnosis.
- Removed duplicate periodic GPU evaluation, which ran at least one episode in every
  training world. Checkpoint output now follows the configured native-validation
  cadence. Timing separates rollout, optimization, and initial/training/final native
  evaluations. Old throughput numbers are not measurements of this revised runner.

Verification: 139 managed tests passed (one optional native fixture skipped), and
22 targeted Python tests passed. `logs/mimickit-training/perturb-validation-v2-smoke-01`
completed one 128-world update (4,096 transitions), checkpoint selection and export;
ONNX maximum action error was 3.58e-7 and five-second quiet standing was 8/8. The
smoke candidate survived 61/96 versus the starting 58/96 but settled in 0/96 versus
1/96; its recovery gate failed and it was not promoted. This is pipeline verification,
not evidence of improved recovery learning. The profiled periodic CPU validation
took 66.2 seconds; the one-update run is not a steady-state throughput benchmark.

`ball-parity-02` remains a native/Newton transport report, not Godot evidence. The
new native/Godot report is separate. The rig, motion, observation/action contract,
reward, and selected viewer checkpoint are preserved. The reward still imitates
standing joint poses and velocities, root pose, and root-relative key-body positions;
recovery steps depart from those targets. A controlled recovery-objective or stepping
reference experiment is the next learning question, not another unmeasured increase
in environment count. Longer evaluation requires an explicitly validated reference
extension: the current clip is finite and less than six seconds long.

## 2026-09-21: Physical-ball Perturb scaffold and experiment wrappers

`MimicPerturb.tscn` now defaults to the physical projectile in
`mujoco_rig/dummy_ball.xml`, targeting the Chest with the retained
`guarded-finetune-01/export` Stand actor. Press F6 with that scene open, or press B
to launch from the live viewer; F5 currently runs this Perturb scene because it is
the configured project main scene. Open `MimicStand.tscn` and press F6 to view Stand.

The ball bridge, native/Newton construction probe and C# build are complete, but no
ball collision parity run or ball training result has been accepted yet. The next
gate is a deterministic launch/contact/reset comparison across native MuJoCo,
Newton and Godot. The generated projectile is currently 8 kg with a 9 cm radius;
align it with the gameplay projectile before training.

The isolated wrappers in `mujoco_rig/mimic/scripts/` now expose control and
stability choices. `train.ps1 -Control torque|target_pd -Stability upstream|guarded`
supports fresh upstream runs and guarded warm starts. `watch.ps1 -Run` validates the
bundle contract and treats historical exports without `experiment.json` as Stand;
new ball exports select Perturb from their manifest. These wrappers do not modify the
legacy MuJoCo training scripts.

## 2026-09-21: Controlled-force baseline (legacy diagnostic)

This section records the historical force-mode configuration of
`Scenes/RL/Isaac3/MuJoCo/MimicPerturb.tscn`. It used live physics with the retained
`guarded-finetune-01/export` Stand policy, not a trained Perturb actor. At that time
the main F5 scene was MimicStand; the current project main scene is MimicPerturb and
defaults to the physical ball. The force diagnostic used 20 N at the Chest COM for
six control intervals (0.100008 s, 2.00016 N·s), starting at interval 60 (1.00008 s).
Direction buttons restart the trial; R repeats, P pauses. Force, timing, duration
and reference phase are Inspector settings. Completion or falling holds the result.

`MimicTrial` shares scene lifecycle, controls and rendering between both scenes.
`MjMimicPerturbTrial` shares force application and measurements between the viewer
and batch replay. Pulses use world-frame force at body COM, expire after each
control interval, and are cleared on reset without affecting peer environments.
The push schedule is not included in policy observations. No rig, reward, actor,
normalization or motor-control change was made; no training was run in this step.

The fixed benchmark contains **52 five-second trials**: 10/20/40 N pulses in four
horizontal directions, at two onset times and two reference phases, plus four
matching unforced trials. Evidence: `logs/mimickit-perturb/baseline-02/`.

- Native MuJoCo and Godot: **4/4 unforced, 16/16 at 10 N, 16/16 at 20 N, 13/16 at
  40 N survived**. All three falls followed forward (+X) pushes.
- Newton: identical unforced/10/20 N survival; **14/16 at 40 N** in the final run.
  The earlier run gave 13/16. `40N-+x-p0.8-t60` is near the five-second survival
  boundary; do not claim exact full-horizon GPU/native outcome parity there.
- Short forced-rollout Newton/native pose discrepancy is 8.22e-6, root velocity
  discrepancy 4.34e-6 m/s. Force expiry, partial-reset isolation and reset replay pass.
- All **nine Godot/native parity checks** pass over the 52 trials, including pulse
  timing, survival, recovery and movement metrics. Pose discrepancy in the push
  windows is 4.75e-7; survival-time difference is at most 4e-7 s.
- Native aggregate foot travel is 8.38 cm unforced, 8.45 cm at 10 N, 8.74 cm at
  20 N and 11.78 cm at 40 N. This includes normal reference motion and is not a
  standalone realism score or a penalty against necessary recovery steps.

Settled recovery is a separate diagnostic: the final uninterrupted 0.5 s must have
root height >=0.75 m, tilt <=15 degrees, horizontal speed <=0.2 m/s, angular speed
<=1 rad/s and both foot loads >5 N. Only 2/4 unforced trials satisfy it; native
counts are 7/16, 7/16 and 8/16 under 10/20/40 N. Newton also differs on four settled
classifications. **Do not use this diagnostic as a training/promotion gate yet.**
First define recovery relative to the matching unforced/reference behavior, then
use mild-push fine-tuning with separate no-push retention and held-out push checks.
The measurements identify a recovery task, not a demonstrated need to redesign the rig.

Validation: 28 Python tests, 133 managed tests (one optional native fixture test
skipped), build, both interactive scene checks, and actual Godot batch replay pass.
Viewer checks cover pause, identical-repeat determinism, four directions, stopping
at completion, and visible rejection of invalid pulse timing. All 14 protected
baseline hashes and the retained Stand checkpoint hash remain unchanged.

## 2026-09-21: fifteen-minute guarded Stand run holds balance but does not improve

The requested run completed **901.156 seconds, 273 iterations, 1,118,208 samples**
with 128 worlds, seed 210921, frozen normalization and actor KL bound 0.02.
Output: `logs/mimickit-stand/guarded-15min-01/`. It initialized weights and
normalization from `guarded-finetune-01/best.pt` (local iteration 33), with fresh
optimizer/rollout state. The source hash was verified against its saved metadata.
Reward, rig, controller and training settings were held fixed throughout the run.

- All **18 periodic evaluations passed 8/8** three-second trials; no crash or
  abrupt standing collapse occurred. Their root errors ranged from 0.02855 to 0.03590 m.
- Final native and Newton policies both passed 8/8, with root errors **0.03406 m**
  and **0.03405 m**. Initial native error was **0.02833 m**, so no later evaluated
  checkpoint improved on the starting policy.
- The final policy also passed 8/8 five-second native trials, root error **0.03620 m**.
- Automatic selection retained the initial policy: `best.pt` tensors exactly match
  `initial.pt`. Its five-second native test passes 8/8 at **0.03324 m**. Its export
  passes all five actual Godot parity gates; ONNX max action error is **2.38e-7**.
- All saved observation-normalizer tensors remained exactly unchanged. Maximum
  accepted rollout KL was **0.01280**; 10,920 actor steps were accepted, none rejected.
  This run therefore does not measure a benefit from activating rollback.

Evidence: `report.json`, `learning_curve.json`, `updates.jsonl`,
`extended-validation.json`, and `godot-parity.json` in the run directory.
All 14 protected baseline hashes match. The interactive scene continues to use its
previously accepted iteration 833 bundle; training does not promote policies automatically.

**Decision: keep the retained candidate; another unchanged Stand run is not justified
by these results.** Next, evaluate that candidate under small controlled pushes to
measure recovery before defining the Perturb curriculum. This run establishes
standing retention on the same finite clip and phases, not general reactive balance
or proof that future optimization cannot regress.

## 2026-09-21: guarded fine-tuning and automatic best-policy export validated

The late-regression ablation implicates both actor updates and normalization drift.
Iteration 1025 weights with their original normalization pass 8/8; using the final
normalization reduces this to 6/8. Final weights fail 0/8 with either normalization.
Evidence: `logs/mimickit-stand/ppo-regression-ablation-01.json`. This separates the
components but does not identify the first damaging update or prove a unique cause.

The local `GuardedPPO` extension freezes loaded observation statistics, clips actor
gradient norm to 1, and rejects minibatches exceeding cumulative mean rollout KL
0.02, restoring both weights and optimizer momentum. Fine-tuning requires explicit
checkpoint/contract paths; it loads weights and normalization with fresh optimizer,
rollout state and local counters. The external MimicKit checkout is unchanged.

The trainer now retains the initial and best evaluated candidate as `best.pt` with
hashed metadata and exports that selection. `model.pt` still records the final
training state. Selection uses the fixed eight native phases and the existing
standing gate, not held-out data. The viewer bundle is never replaced automatically.

Validation: `logs/mimickit-stand/guarded-finetune-01/`, initialized from the accepted
iteration 833 checkpoint, completed **181 seconds, 53 iterations, 217,088 samples**:

- Every periodic evaluation and the final state passed 8/8 three-second trials.
- Best local iteration 33: native root error **0.02833 m**, Newton **0.02827 m**,
  compared with initial native **0.02968 m**. Both engines passed 8/8.
- Selected five-second native test: **8/8**, root error **0.03324 m**.
- ONNX max action discrepancy **2.38e-7**; all five actual Godot parity gates pass.
- All three saved normalizer tensors remained exactly unchanged. Maximum accepted
  rollout KL **0.01275**; 2,120 accepted actor steps, no rejection at the normal bound.
- A separate one-iteration CPU integration check intentionally used KL limit 1e-12.
  Its first update was rejected (attempted KL 7.68e-6), leaving actor weights,
  normalization and optimizer state exactly unchanged. Seven safety unit tests pass;
  all 23 Python regression tests passed during implementation.

Reports: `report.json`, `invariants.json`, `godot-parity.json`, and
`rollback-check/report.json` inside that run directory. Generated rig, shipped
policies and the protected production checkpoint match all 14 baseline hashes.
The interactive scene still uses the previously accepted iteration 833 bundle.

**Next: a longer guarded Stand fine-tune to test stability over a comparable budget.**
The short run establishes working safeguards and modest tracking improvement; it
does not establish sustained stability, unseen-motion balance, or Perturb readiness.
The KL bound applies per rollout on sampled states, not cumulative change across
training. Do not treat best-checkpoint retention as a solution to generalization.

## 2026-09-21: interactive MimicStand scene ready for F5

`Scenes/RL/Isaac3/MuJoCo/MimicStand.tscn` is now the main scene. It runs live
policy-controlled native physics with the selected `evaluation_000833` bundle,
configured through Inspector properties. Its MuJoCo library directory points to
the isolated Mimic environment; no environment variables are required.

The default trial lasts five seconds, then holds the resulting pose for inspection.
It also stops on a fall, without automatically resetting. R/button restarts;
P/button pauses or resumes. The existing CameraController provides right-drag look,
WASD/Q/E movement and wheel speed adjustment. Trial duration and reference start
phase are configurable within the finite reference clip. This scene does not claim
continuous indefinite standing or add a Perturb controller.

`MimicStandReplay.tscn` is now batch-only, retaining the existing Python parity
command and shared `MjMimicStandDriver`. Build, default main-scene startup without
environment settings, five-second completion, button/keyboard pause and restart,
and all five batch parity gates pass. No policy weights or controller settings changed.

## 2026-09-21: 30-minute target/PD run learns Stand, then regresses late

**Decision: retain checkpoint `evaluation_000833.pt` as the experimental Stand
candidate. Investigate late PPO update stability and add best-checkpoint protection
before another long run or Perturb training.** The selected actor passes the
three-second gate and an additional five-second test; the final actor fails.
This establishes learned standing on this reference, not general reactive balance.

The requested fresh, unchanged-setting run completed **1,800.063 seconds**, 1,071
iterations and 4,386,816 samples with 128 worlds, seed 210921 and fault tracing.
No native crash occurred. Output: `logs/mimickit-stand/target-pd-30min-01/`.
Controller, rig, reference, reward and PPO configuration were unchanged.

Learning progressed from 0/8 successes to 8/8 at about 17.4 minutes. Both original
gates first passed at 19.2 minutes, then passed at seven consecutive saved evaluation
checkpoints through minute 29.0. The lowest root error among those checkpoints was
at iteration 833 (24.15 minutes, 3,411,968 samples):

- Re-evaluated native: **8/8 three-second successes**, mean root error **0.02968 m**.
- Newton: **8/8 three-second successes**, mean root error **0.02972 m**.
- Extended native ONNX test: **8/8 five-second successes**, mean root error **0.03407 m**.
- Extended Newton test: **8/8 five-second successes**, mean root error **0.03378 m**.
- Actual Godot replay: all five parity gates pass, **8/8 three-second successes**;
  maximum prefix action error 7.49e-6 and episode-duration difference 2e-7 s.

Selection uses the existing eight evaluation phases: among checkpoints passing
8/8 survival and <=0.05 m mean root error, choose the smallest root error. This is
checkpoint selection, not independent generalization evidence. The five-second
test uses eight evenly spaced starts in `[0, reference_duration - 5]` within the
same non-looping motion, with unchanged failure conditions and no further training.
No longer hold, new motion, perturbation, or visual realism gate is established.

**Late regression:** logged test return falls from 169.57 at iteration 1024 to
50.00 at 1032, while PPO clip fraction rises from 0.506 to 0.933. The final actor
has 0/8 successes, 1.652 s mean survival and 0.182 m root error in both native and
Newton evaluation. It also passes Godot transfer checks, confirming this loss is
present before deployment. The exact optimization/normalization cause is not yet
isolated. The final model and failed export remain intact for diagnosis.

Use **`target-pd-30min-01/selected-000833/export/`** for experimental replay, not
the run's top-level `export/`. The selected `.pt` is the original saved checkpoint;
its separate export and report are in `selected-000833/`, alongside
`godot-parity.json` and `newton-evaluation.json`. The final actor's results remain
in the run's `report.json`, `godot-parity.json`, and `five-second-final.json`.

No training/controller source was changed for this run, no production policy was
replaced, and no commit or push was made. The generated model and shipped policy
hashes remain unchanged.

## 2026-09-21: target/PD learning path implemented; five-minute comparison complete

**Decision: retain the rig and bridge, keep this actor experimental, and use a
fixed-setting 15-minute Stand benchmark as the next bounded learning test. Do not
start Perturb or an overnight run yet.** The new control path transfers correctly;
the short PPO run shows modest improvement but does not solve standing.

The opt-in `--control target_pd` contract (`mimic_stand_target_pd_v1`) uses 30
reference-relative joint targets and physics-rate bounded PD feedback. Targets and
reference velocities wait two control intervals; the first two intervals apply
zero policy torque. Residual scale is 0.25 rad, with joint-limit clipping. Feedback
is `limit * clip(4 * angle_error + 0.08 * velocity_error, -1, 1)` on every native or
Newton physics substep. Original torque authority, passive mechanics, head control,
reference, reward and PPO YAML are retained. No support-force oracle is used.
The queue's two position/velocity/validity tuples replace pending torque actions,
giving 362 observations. Raw-torque 300-channel actors remain supported separately.

Fresh run `logs/mimickit-stand/target-pd-5min-02` completed 177 iterations and 724,992
samples in 302.344 seconds, with 128 worlds and seed 210921. Native mean survival
improves from **1.808 to 2.013 s**; best final trial is 2.400 s. Native and Newton
agree on all eight survival durations. **0/8 reaches three seconds**, and mean root
tracking error is **0.162 m**, above the unchanged 0.05 m gate. Periodic native means
are 1.786, 1.671, 1.781 s at 4k/266k/528k samples, followed by 2.013 s at the end.
The initial policy has small random residuals; the separate exactly-zero residual
baseline survives 1.786 s. More training is a hypothesis to test, not a demonstrated
solution. The old raw-torque run used a different budget and is not a matched trial.

Verification:

- 16 Python regressions, 130 managed tests and the opt-in native foundation test pass.
- Target native/Newton preflight passes 14 gates: short pose error 1.85e-6,
  motor torque error 0.00034 Nm, including reset isolation.
- Zero-residual and trained ONNX actors both pass all five actual Godot replay
  gates. Trained prefix observation/action/pose errors are 3.61e-6 / 3.35e-6 /
  2.76e-7; maximum episode-duration difference is 2e-7 s.
- The existing raw-torque Godot replay still passes. Generated MJCF, shipped ONNX
  files and the accepted production checkpoint match their baseline hashes.

The first attempt (`target-pd-5min-01`) crashed in native code after roughly 25 s
with Windows access violation 0xc0000005 and no Python traceback; it is not a
completed training result. The identical-setting retry enabled unbuffered output
and Python fault tracing and completed. The native crash's cause is unresolved;
retain the failed run's `failure.json` and monitor any subsequent bounded run.

Reports: `target-pd-preflight.json`, `target-pd-zero-01/godot-parity.json`, and
`target-pd-5min-02/{report,godot-parity,learning_curve}.json` under
`logs/mimickit-stand/`. See `mimic/README.md` for commands. No production actor was
replaced, and no commit or push was made.

## 2026-09-21: reference-controller diagnostics narrow the problem to balance/control design

**Decision: preserve the rig and bridge; do not extend the current raw-torque PPO
run. Prototype reference-relative joint targets with a physics-rate PD loop and
learned balance corrections.** MimicKit's pinned `data/engines/newton_engine.yaml`
uses `control_mode: pos`; our compatibility adaptation retained delayed raw torque.
Keeping the skeleton/sim bridge does not require keeping that action representation.
This is the next experiment, not a validated production change.

Bounded reference-controller tests are complete. Each controller family had 12
predeclared gain settings, calibrated on phases 0/3/7; selected parameters were then
evaluated across all eight phases, including five unused for selection. Physical
torque limits, passive joints, model and reference stayed unchanged. The primary
PD tests retain the exact 60 Hz / two-step delayed-torque contract.

- Primary normalized PD: 0/8 three-second successes; mean survival 1.002 s.
  Inertia-scaled PD with optional joint-bias compensation also fails (0/8); some
  cases become numerically unstable. Canonical corrected reports are
  `reference-control-normalized-02` and `reference-control-inertia-02`.
- Diagnostic zero-delay controls improve normalized PD to 1.502 s, still 0/8.
  Thus delay contributes, but removing it alone does not produce balance.
- A **different control contract**, with targets delayed two control intervals but
  PD feedback evaluated every physics step, reaches 1.875 s, still 0/8.
- Adding quasi-static support feedforward to that alternative reaches **7/8 at
  three seconds**, 0.0147 rad mean joint RMSE and 0.29% clipped torques. It still
  fails the root/foot tracking gates (0.0581 m / 0.0323 m) and **0/8 survive ten-second
  fixed-pose holds**; mean hold survival is 3.586 s. Joint tracking alone is not balance.

The support calculation uses prospective sole contacts within 1 mm of the floor,
friction pyramids and the same motor torque limits. Over all 180 reference frames,
the refined solution needs at most 14.4% of the available 60%-authority motor budget.
It balances the free base and controlled joints, allowing passive joints to settle
(up to 0.795 Nm residual at their exact reference angles). This is conditional
quasi-static evidence, **not proof of dynamic feasibility or a fault-free rig**.
Strict checks using only the reference's initial contacts fail because one foot is
slightly lifted. Native stepping establishes both-foot support within 16.7 ms,
before the first delayed motor command, so this transient does not explain the
later multi-second collapse by itself.

Diagnostic code: `mimic/reference_controller.py`, `probe_reference_control.py`,
`probe_target_pd.py`, `static_support.py`. Reports live under
`logs/mimickit-stand/reference-*` and `static-support.json`; the support-feedforward
result is `reference-target-support-01/report.json`. Twelve Python regressions pass.
The early `reference-control-01` selection minimized error among failed episodes,
which favored early termination; version 2 ranks survival before error and detects
numerical failures explicitly. Use the `*-02` primary reports above.

No training, production controller replacement, body-strength increase, motion
rewrite, commit or push occurred during this diagnostic. The alternative controller
has only native-MuJoCo evidence so far; it needs a matching Godot control contract
and transfer tests before an RL experiment or promotion.

## 2026-09-21: fixed-setting 15-minute Stand run and actual Godot replay complete

**Decision: keep the actor experimental.** The predeclared gate was eight of eight
three-second successes and mean root tracking error at most 0.05 m. The final
actor fails both gates; it is not promoted to Stand/Perturb production scenes.

The fresh run used the same reference, plant, torque limits/delay, PPO YAML and
reward as the smoke test: 128 worlds, 486 iterations, 1,990,656 samples in 900.515
seconds (time budget includes periodic output/evaluation). Mean native standing
time rises from 0.448 to 1.663 seconds; Newton gives 1.667 seconds. Best final trial
is 2.134 seconds, **zero of eight reach three seconds**, and mean root tracking
error is 0.117 m. Periodic checkpoint means progress from 1.20 s at 528k samples to
1.46 s at 1.58M and 1.53 s at 1.84M. This is learning progress, not stable standing.
Rendered native poses still show forward collapse.

Added an isolated `MimicStandReplay.tscn` and hash-validated reference/control
adapter using the existing native bridge. Actual headless Godot replay passes all
five parity gates for the final ONNX actor. Across the first 16 steps of each phase,
maximum observation/action/pose differences are 1.713e-5 / 3.540e-6 / 1.611e-6;
all eight full episode outcomes agree and duration differences are below 1.01e-7 s.
ONNX versus Torch action error is 1.79e-7. These bounded checks do not identify a
Godot transfer mismatch as the cause of the falls.

Validation: 127 managed C# tests, the opt-in native foundation sensor regression,
and nine Python motion/control regressions pass. The generated ABI adds the contact
frame offset; existing bridge APIs retain their semantics. Build input now excludes
saved C# source snapshots under `logs/`. The MJCF, shipped actor and accepted old
checkpoint match their preserved hashes. No commit or push was made.

Artifacts: `logs/mimickit-stand/ppo-15min-01/{report,learning_curve,godot-parity}.json`,
`model.pt`, periodic checkpoints, `export/` and `trained-poses.png`.

**Next:** check whether reference tracking is achievable under the exact torque
limits and action delay with a reference controller, before committing to another
long PPO run or changing rewards. The current result does not distinguish slow
direct-torque learning from an unsuitable tracking/control setup. Perturb remains gated.

## 2026-09-21: licensed motion-guided Stand reaches its first PPO smoke test

The isolated [MimicKit experiment](mimic/README.md) now uses a six-second neutral-idle
excerpt from Ian Mason's 100STYLE, under CC BY 4.0 with attribution. Source hashes,
license and changes accompany the adapted asset. Fixed-foot collision-aware IK
preserves the original dummy; reference validation passes (maximum foot error
0.355 mm, penetration 0.378 mm). No noncommercial source was used.

Implemented native-hinge reference observations, partial resets, upstream DeepMimic
reward and PPO, deterministic native evaluation and ONNX export. Nine regression
tests and all 12 task preflight checks pass. Production rig, bridge, reward code
and accepted/shipped policies are unchanged.

**First run:** 128 worlds, 32 PPO iterations, 131,072 samples, 67.8 seconds. Across
eight fixed phases, native MuJoCo standing time rises from 0.448 to 1.006 seconds;
Newton gives 1.010 seconds. **All eight still fail the three-second episode.**
Tracking error over the longer trajectories increases, so this is pipeline and
early learning evidence, not a claim of realistic movement. ONNX action error is
5.78e-8. Results/checkpoints/export are under `logs/mimickit-stand/ppo-smoke-01/`.

**Next:** bounded Stand learning-curve evaluation with fixed settings; matching
Godot observation/reference adapter and replay before policy promotion. No Perturb
or long unattended training yet. The exported 300-channel actor is experimental
and cannot replace the existing 105/140-channel Godot policies.

## 2026-09-21: MimicKit compatibility setup passes; Stand task is next

Implemented setup stages 1–5 in [mimic/README.md](mimic/README.md). A separate
Python environment and project-local MimicKit Newton adapter preserve the native
MuJoCo rig, offset hinge joints, passive mechanics, 33 actuators / 30 policy actions,
60% torque authority and two-step action delay. The existing Godot bridge and
accepted/shipped policies are unchanged. The original working tree and accepted
checkpoint are preserved under `logs/mimickit-compat/baseline/`.

Five regression tests and the GPU/native compatibility gates pass: 65 poses,
matching collision eligibility and sampled contacts, delayed controls, a two-world
0.5-second open-loop comparison and reset checks. Pose mapping error is below
1.2e-6 m; the recorded rollout root/joint differences are 1.714e-6 m / 1.749e-5 rad.
This establishes a bounded compatibility result, not learned Stand behavior or
long-horizon sim-to-sim equivalence. GPU reset replay is tolerance-based, not bitwise.

**Next:** select a motion with verified training/commercial-use permissions,
retarget standing/weight shifting into native hinge coordinates, then implement
the Stand reference observations, action space, per-world resets and Godot policy
contract. No external motion/pretrained archive was downloaded and no training or
policy export occurred. Do not resume the rejected demonstration candidate below.

## 2026-09-21: demonstration cloning calms standing but regresses recovery

**Decision: reject the demonstration candidate.** Keep the accepted training actor
`contact_v3_full5m/model_78.pt` and the shipped balance policy. Do not extend this
candidate with a long training session. Rewards, PPO settings, plant, torque limits
and action delay were unchanged throughout this pilot. Nothing was exported.

Added `scripts/recovery_demonstrations.py`, `scripts/learn_recovery_demonstrations.py`
and five data/initialization contract tests. The protocol was written before new
teacher outcomes: first three accepted-policy falls in each Head/Chest approach
stratum, first two for training and third for whole-trajectory validation. Original
benchmark prefixes replayed exactly. Previously saved oracle traces were hash-checked
and reused; new cases used the same bounded CEM/MPC configuration as the feasibility
probe. All commands replayed exactly before sensor/command pairs were retained.

**Teacher coverage:** 16 of 24 cases recover and settle under both metrics, including
the earlier feasibility cases. Ten successful training trajectories provide 1,200
oracle-command pairs from their first 120 control steps. Six successful validation
trajectories provide 720 independent pairs. Failed/unsettled attempts are retained in
the report but excluded from imitation labels. The actor sees only the existing
140-channel `foundation_v2` observation before each command; no projectile state,
future state or optimizer state enters its input. Successful handback states augment
retention targets. Independent references contain 4,096 quiet states (16/16 successful
20-second trajectories, seed 28101) and 4,096 impact states (39/64 successful six-second
trajectories, seed 28103). No validation trajectory enters any training group.

**Fixed fit:** zero-extend the accepted actor/critic inputs, retain exploration, and
optimize only actor weights for 2,000 Adam updates at 3e-5, seed 29021. Each update
samples 256 examples per group; demonstration/quiet/impact MSE weights are 1/2/1.
Critic and exploration parameters were verified unchanged. Scheduled candidates at
250/500/1000/2000 updates are ranked by whole-trajectory validation MSE subject to
quiet MSE <= 1e-4 and impact retention MSE <= 4e-4. None passes these filters.
The predeclared fallback locks `model_2000.pt` for diagnostic evaluation and rejection;
its hash was saved before any closed-loop result or fresh benchmark.

Training MSE drops from 0.131908 to 0.028276 and validation MSE from 0.131243 to
0.072097. Since commands are clipped when executed, the corresponding executed-action
MSE is also measured: training 0.093943 -> 0.026949, validation 0.096965 -> 0.068001.
The improvement therefore is not solely removal of harmless out-of-range output.
However, quiet retention MSE is 0.000145 and impact retention MSE 0.002294. From the
same 24 captured contact states, the learned actor produces **0 survivors/recoveries**,
including all ten successful training demonstrations. The optimizer recovers 16;
the learned policy does not reproduce those feedback trajectories.

**Standard CPU gates, survival / historical recovery / measured-contact recovery:**

- Accepted actor: **307 / 281 / 296** out of 512.
- Shipped actor: **302 / 277 / 288** out of 512.
- Demonstration candidate: **236 / 193 / 194** out of 512.

All three pass quiet standing: 16/16 survive and settle over 40 seconds. The candidate
loses 71 survivors and 88 recoveries versus accepted; recovery gains 39 worlds and
loses 127, paired z=-6.830. Against shipped, recovery z=-6.481. Both comparisons reject
continuation/promotion through the existing survival/recovery gates.

**Fresh balanced development bank:** seed 155921, 768 trials, all 48 target/approach
strata, eight 96-world shards, fixed accepted-policy projectile launches. Accepted
versus candidate: survival **465 vs 358**, historical recovery **427 vs 289**, and
measured-contact recovery **449 vs 294**. Historical recovery gains 65 worlds and
loses 203, z=-8.430. Contact is observed at control boundaries in 768 accepted and
766 candidate trials; no trials are excluded. The final held-out seed **196613 remains
unused**, because prerequisite gates already reject this candidate. The development
seed is now inspected and must not be represented as unseen in future experiments.

**Movement:** quiet mean action change decreases from 0.03910 to 0.00300 and contact
sliding from 0.33245 to 0.27505 m/world. Among the 293 fresh-bank worlds both policies
survive, action change decreases from 0.05691 to 0.03824 and contact sliding from
0.68848 to 0.63229 m/world. These are narrower improvements in motion metrics, not a
successful recovery policy or proof of perceptually realistic movement. All-world
sliding totals alone would be misleading because the candidate falls more often.

**Interpretation and next direction:** the expanded oracle evidence supports physical
recoverability in more sampled states. Direct offline cloning of these trajectories
does not transfer the recovery skill to the current reactive actor. Reduced action
error on teacher states is insufficient evidence of stable closed-loop behavior.
This does not isolate observation insufficiency, residual fitting error, limited
coverage or divergence onto states absent from demonstrations as the sole cause.
A useful next experiment would collect expert corrections on the student's own
rollouts, protect the accepted standing/recovery behavior, and require successful
closed-loop reproduction before spending time on a broad benchmark. Further reward
tweaks or more epochs of this rejected fit are not supported by this result.

Seven targeted tests pass, covering snapshot/physics parity, persistent native state,
pre-action labels, disjoint trajectory splits, input migration and retention selection.
`git diff --check` passes. Artifacts are in `logs/recovery-demonstrations-20260921/`:
protocol, immutable dataset inventory, teacher traces, dataset, fit history, locked
checkpoint, contact-state rollouts, all three standard scorings, fresh benchmark,
matched-survivor motion comparison and final decision. No commit or push.

## 2026-09-21: bounded trajectory search establishes five feasible recoveries

The frozen accepted actor is still `contact_v3_full5m/model_78.pt`. Added the offline
`scripts/probe_recovery_feasibility.py` diagnostic and two contract tests. No rig,
actuator authority, action delay, observation contract, reward, PPO setting or policy
weight changed in this experiment.

Eight cases were selected before optimization: the first recorded accepted-policy
fall in each Head/Chest x four approach sectors of the previous benchmark's selection
split. Original 96-world inference batches were replayed. A read-only observer
captured the first control boundary after positive-force ball/body contact, including
both queued commands and the native derived state. Every benchmark output array
matched the saved uninstrumented result exactly. All four head shots contacted Head;
chest +X/-X contacted Chest, +Y contacted UpperArm_R, and -Y contacted UpperArm_L.

One fixed full-state CEM/MPC configuration used a 60-control-step horizon, seven
residual knots, 96 candidates, four iterations and replanning every six steps.
Existing normalized action bounds, 0.6 torque authority, two-command latency and
projectile gravity schedule were retained. Only joint actuator controls changed.
After 120 control steps (about two seconds), control returned to the accepted actor.
Trials ended at first fall or approximately six seconds after contact. The physical
search surrogate was separate from the unchanged RL reward and recovery gates.

**Results:** both original batch-action replay and singleton actor baselines fall
in all eight cases. Optimized control yields **6/8 survivors and 5/8 settled
recoveries**, agreeing under both historical and measured-contact recovery metrics.

- Head +X, trial 0: recovered; final measured-contact settled hold 2.15 seconds.
- Head +Y, trial 17: recovered; settled hold 1.32 seconds.
- Head -X, trial 32: falls 2.28 seconds after contact, after policy handback.
- Head -Y, trial 48: recovered; settled hold 3.87 seconds.
- Chest +X, trial 66: falls 3.68 seconds after contact, after policy handback.
- Chest +Y, trial 80, actual right-arm contact: survives but does not finish settled.
- Chest -X, trial 96: recovered; settled hold 3.67 seconds.
- Chest -Y, trial 112, actual left-arm contact: recovered; settled hold 4.57 seconds.

All eight optimized trajectories independently replayed every saved state and applied
control exactly. All traces were finite, with peak absolute body generalized velocity
75.85 or less, below the existing 300 divergence ceiling. Every case completed all
20 plans within its fixed 80-second budget: 61,440 candidate sequences and 158.4
seconds of optimization/evaluation wall time in total, excluding prefix capture.
Two targeted tests pass: exact snapshot/observation/continuation preservation, and
threaded native-rollout parity across delayed/clipped commands, body contact and
projectile gravity transition. No long training run or policy export was launched.

**Interpretation:** the current body can recover from five sampled impact states
that defeat the accepted policy, without stronger motors or external assistance.
Control/learning is therefore part of the limitation. This does not clear every
structural or sensor issue: the oracle uses full simulator state and offline compute,
and the three cases without settled recovery remain inconclusive. The failure-only
sample is diagnostic, not an estimate of population success. Realistic movement and
learned-policy improvement are not established by this result. Motion metrics for
early-falling baselines and six-second survivors have different exposure lengths.

**Next recommendation:** a bounded demonstration-learning experiment using successful
optimized recoveries, retaining quiet-standing behavior and evaluating through the
existing CPU gates plus a newly sealed balanced bank. Demonstration quantity,
observation sufficiency and motion quality must be addressed before that experiment;
five successful traces alone are not a production training set. Keep rewards and PPO
settings frozen rather than launch another long continuation on the rejected seeds.

Artifacts: `logs/recovery-feasibility-20260921/` contains the frozen protocol, case
snapshots, baseline/optimized command and state traces, per-case results, summary and
SHA-256 inventory. `case-0-comparison.mp4` shows accepted versus optimized head +X
recovery; the baseline frame is held after its first fall. README documents reproduction.
No commit or push.

## 2026-09-21: balanced CPU benchmark rejects a reliable restart improvement

Added `scripts/benchmark_perturb.py` and `rl/test_impact_benchmark.py`. The reusable
benchmark saves a protocol before execution, checkpoint/plant hashes, fixed projectile
launches, per-trial outcomes and all 48 target/approach breakdowns. Default splits each
contain **768 trials: 12 intended targets x four approach sectors x 16 samples**.
Selection seed is 104729, held-out seed 130363. Each of eight 96-world shards includes
two samples of every stratum; four CPU workers run independently. The accepted actor
generates launches, which candidates replay exactly. Physics/rewards, observation
contracts and existing continuation/promotion gates are unchanged.

The registered selection rule chooses most historical recoveries, then most survivors,
then label. Only the selected candidate and accepted actor see the held-out split;
`selection-lock.json` is written first. Supplementary confirmation requires no survival
regression and more recoveries with paired z >= 2 on both splits. It cannot promote a
policy. The tool now also accepts distinct custom split seeds for future untouched
confirmation sets; the seeds used here are no longer unseen after inspecting results.

**Selection, survival / recovery out of 768:**

- Accepted `contact_v3_full5m/model_78.pt`: **466 / 420**.
- Protected training seed 443: **468 / 424**.
- Protected training seed 457: **455 / 414**.

Seed 443 is selected, but its recovery gain is only 4 worlds: 82 gained, 78 lost,
z=0.316. Measured-contact recovery ties accepted at 445/768. Mean native episode reward
changes by +0.614. Seed 457 has recovery z=-0.487 and 11 fewer survivors; it is not
evaluated on the held-out set.

**Held-out, accepted versus selected seed 443:** survival **476 versus 480**, historical
recovery **438 versus 430**, measured-contact recovery **457 versus 457**. Historical
recovery gains 85 worlds and loses 93 (z=-0.600). Mean native reward rises by 6.399,
but neither recovery definition improves. Mean action change is 0.06562 versus 0.06565
and contact sliding is 0.73643 versus 0.73851 m/world. No realism improvement is established.
Both selection and held-out confirmation flags are false.

**Failure distribution:** upper-body targets are consistently difficult. For the
accepted policy, selection/held-out recovery is Head 6/64 and 9/64, Chest 11/64 and
14/64, Spine 13/64 and 17/64. Left-thigh recovery is 55/64 in both splits; right-thigh
recovery is 54/64 and 49/64. These labels describe intended targets, not necessarily
the first contacted body. Seed 443's per-body changes are mixed: head recovery drops
in both splits, chest falls in selection but rises on held-out, and spine does the
reverse. No individual 16-trial target/sector cell should drive a reward-weight change.

Contact presence at control boundaries is observed in 767/768 accepted selection trials
and all other trials. An unobserved control-boundary contact is not proof of a miss;
no trials were removed after observing outcomes. All 3,840 policy trials completed
with finite outputs. Artifact hashes, complete unique trial IDs, aggregate counts,
and the frozen selection lock were verified. Four targeted tests pass, covering
stratification, split independence, ranking, trial aggregation and physical launch replay.
`git diff --check` passes. See README's stratified-benchmark section for reproduction.

**Decision:** keep the accepted training actor and shipped policy. The strong gains on
the earlier 128-scenario diagnostic did not generalize to this larger, balanced bank.
Do not extend either protected candidate on the strength of that diagnostic, or make
restart warm-up a production default. This benchmark localizes weak impact classes;
it does not establish whether their cause is physical recoverability, missing policy
information or the learning objective. No further training was launched.

Run artifacts: `logs/perturb-benchmark-20260921/{protocol,selection-lock,decision}.json`,
both split directories' `summary.json`, scenario NPZs, per-policy shard JSONs and hash
inventories. All processes finished. No policy export, commit or push.

## 2026-09-21: matched CPU/Warp impacts do not show GPU-only learning gains

Completed a frozen-policy transfer audit on **128 new scenarios, seed 977**, for the
accepted `contact_v3_full5m/model_78.pt` and protected restart policies from training
seeds 443 and 457. No training or production configuration changes were made.

**Isolation:** all runs receive identical initial qpos, zero qvel, launch steps,
projectile positions and velocities. Shared inputs are stored at float32 precision;
both backends report zero initial-qpos error against that manifest. The accepted CPU
rollout defines world-space launches at 1.5–1.8 seconds; every other rollout replays
them, with one 6 m/s projectile per world, six-second episodes and no automatic
resets. All policy inference runs on CPU using deterministic means. The accepted
actor consumes its original 105 channels; learned actors consume foundation_v2's
140 channels. Both simulators expose the same 140-channel environment contract.
All 128 projectiles register body-contact presence at physics-substep resolution in
each of the six closed-loop runs. All recorded arrays are finite and Warp constraint
buffer checks pass at every control step.

**Results, survival / legacy recovery out of 128:**

- Accepted: CPU **81 / 66**, Warp **87 / 77**.
- Protected seed 443: CPU **88 / 80**, Warp **93 / 78**.
- Protected seed 457: CPU **90 / 85**, Warp **92 / 84**.

Loaded-contact recovery counts are CPU 73/85/88 and Warp 82/85/87, respectively.
On CPU, learned recovery gains/losses versus accepted are 24/10 (z=2.401) and 27/8
(z=3.212). Warp gives 16/15 (z=0.180) and 17/10 (z=1.347). These diagnostic comparisons
are not promotion gates: they use a new scenario bank with fixed world-space launches,
and omit the required quiet-room gate. Prior continuation failures remain unchanged.

**Reward:** mean native episode reward is CPU 1432.06/1527.64/1542.35 and Warp
1510.39/1531.69/1542.12 (accepted/443/457). Reward is counted through the first native
training termination, including its terminal penalty, with subsequent rewards masked
out. Post-launch reward changes have the same direction. With gamma=0.99 discounting,
CPU means are 453.58/459.42/460.39 and Warp 460.10/459.77/461.16: the Warp discounted
changes are small relative to paired standard errors. These finite deterministic
returns are not estimates of the full stochastic PPO objective or bootstrapped value
targets. This sample does not demonstrate that the reward systematically prefers
worse recovery, nor that learned gains exist only on the training engine.

**Numerical controls:** an eight-world CPU repeat reproduces all recorded trajectory
arrays exactly; replaying its fixed action sequence on CPU also reproduces them exactly.
A full 128-world Warp repeat returns 84 survivors / 79 recoveries versus its original
87/77. Fourteen individual recovery outcomes flip (8 gains, 6 losses). CPU versus
original Warp flips 23 accepted-policy recovery outcomes (17 gains, 6 losses).
Thus GPU repeat variability is material and a cross-backend trajectory difference
alone is not evidence of a sensor or physics implementation bug.

Before launches, the accepted closed-loop CPU/Warp pelvis gap at step 89 is median
0.0000061 m, p95 0.00162 m; by the last step it is median 0.0987 m. Warp-versus-Warp
final median gap is 0.0662 m. Replaying the exact CPU action sequence on Warp without
feedback produces a final median gap of 1.217 m and no survivors. The replayed actions
match exactly; this is an open-loop sensitivity diagnostic, not a deployable-policy
score or proof that a controller cannot transfer.

**Decision:** the simple explanation that training improves Warp recovery but loses
those gains on CPU is not supported here. Both learned policies improve recovery and
undiscounted reward on the fresh CPU scenarios, despite their earlier gate regressions.
Behavior varies substantially by scenario and numerical trajectory; no new foundation
defect or reward-weight correction is established. Retain the accepted training actor
and shipped policy. Before more PPO tuning or a longer run, establish a larger fixed
CPU validation bank stratified by target body and heading, with held-out scenarios,
to measure which impacts improve and regress. Do not select a new policy from this
single favorable diagnostic batch. No further experiment has been launched.

Reproduction: `logs/perturb_transfer_audit.py --worlds 128 --out logs/transfer-audit`,
then `logs/analyze_transfer_audit.py` and `logs/transfer_repeat_controls.py`.
Evidence is in `logs/transfer-audit/`: hashed manifest and scenarios, six policy/backend
JSON results and NPZ trajectories, `comparison.json`, `repeat-controls.json`, the Warp
repeat and fixed-action replay. CPU integrity evidence is under
`logs/transfer-audit-smoke/cpu-repeat/`. All runs completed; `git diff --check` passes.
No production edits, policy export, commit or push.

## 2026-09-20: second restart seed does not reproduce the continuation win

Repeated the immediate/protected restart comparison with training seed **457** from
the same `foundation_v2_pilot/model_50.pt` source. Control:
`2026-09-20_23-18-15_foundation_restart457_control/model_25.pt`; protected:
`2026-09-20_23-21-29_foundation_restart457_protected/model_50.pt`.
Settings remain 4096 worlds x 16 steps, 6 m/s and reference coefficient 1. The control
runs 25 actor-learning rounds; protected runs 25 critic-only plus 25 actor-learning
rounds in one process. Both finish below their five-minute caps (1.7/3.5 minutes).

Initial networks, teacher and reference targets match. Control settings differ from
the seed-443 control only in seed and run name. Within the new pair, only warm-up,
total rounds and run name differ. The protected round-25 actor and exploration weights
are byte-identical to the source, actor optimizer state remains empty, and readiness
passes with pre-fit/post-fit EV 0.904/0.905. It then performs 468 actor updates versus
the control's 467. Critic optimizer counts are 2000/1500 including the source's 1000.
Final protected pre-fit/post-fit EV is 0.910/0.912. Thus the intended intervention ran
correctly; this is not a failed or incomplete warm-up experiment.

**Both new checkpoints fail continuation against the accepted actor.** On the unchanged
512-world seed-17 gate, control returns **304 survivors / 282 recoveries**, protected
**303 / 283**, versus accepted **307 / 281**. Protected recovery pairs versus control
are 52 gained and 51 lost (z=0.099); versus accepted, 54/52 (z=0.194). Promotion also
fails: protected recovery z versus shipped is 0.567, below the required 2.
Measured-contact recovery is control 296/512, protected 294/512, accepted 296/512.

On evaluation seed 71 (128 worlds), control returns **76 / 71**, protected **70 / 65**,
versus accepted **73 / 71**. Protected recovery versus control gains 9 and loses 15
(z=-1.225). Both retain perfect 16-world quiet survival/recovery over 40 seconds.
Quiet action change is control 0.04726, protected 0.05379, accepted 0.03910; protected
main impact action change/contact slip is 0.06712/0.71559 m per world versus accepted
0.06617/0.70827. This repeat provides no improved realism signal either.

**Conclusion and next decision:** seed 443's protected 311/287 main result did not
repeat; seed 457 gives 303/283. Both protected runs lose independent-seed recovery
(66 and 65 versus accepted 71). Two training seeds do not establish population harm,
and the reused evaluation worlds must not be pooled as independent samples. They
also do not support making restart warm-up a production default or extending these
runs. Keep `contact_v3_full5m/model_78.pt` as the accepted training actor and the
shipped policy unchanged. The earlier seed-443 checkpoint retains its recorded main
gate pass, but is not validated as a reliably better starting point.

Before another parameter change, the next recommended investigation is matched,
deterministic impact rollouts on CPU and Warp for the accepted and learned policies:
compare recovery and accumulated reward to separate simulator transfer from an
objective that fails to improve recovery. Existing state-level observation/reward
parity checks do not establish whole-trajectory equivalence. This is an investigation
proposal, not evidence that either backend is defective; no further run was launched.

Evidence: `logs/compare_protected_restart.py` now accepts explicit control and artifact
prefixes; `logs/foundation-restart457-{control,protected}-{train,512,seed71}.log`,
the corresponding evaluation JSON files, both initial fingerprint files, and
`logs/foundation-restart457-protected-{integrity,comparison}.json`. All six training
and evaluation processes finished successfully. No production code/default changes,
policy export, commit or push. `git diff --check` passes.

## 2026-09-20: protected restart passes continuation, evidence remains mixed

Completed the approved restart experiment from
`2026-09-20_22-19-35_foundation_v2_pilot/model_50.pt`: 25 critic-only rounds followed
by 25 actor-learning rounds, in one uninterrupted process. Run:
`2026-09-20_22-58-46_foundation_protected_restart/model_50.pt`.
Seed 443, 4096 worlds x 16 steps, fixed 6 m/s, reference coefficient 1 and all other
training settings match the immediate-restart retention-on control. Only run name,
total rounds (25 to 50) and explicit warm-up (inherited zero to 25) differ.
Training finished in 3.2 minutes, below the five-minute cap.

**Integrity and critic diagnostics:** initial model, reference targets and teacher match
the control. At round 25, all actor and exploration tensors remain byte-identical to
the source, actor Adam state is empty, and critic Adam advances from 1000 to 1500 steps.
Warm-up ends on schedule with pre-fit/post-fit EV 0.895/0.897, above the existing 0.5
readiness gate. The first fresh rollout again has pre-fit EV -4.945; the actor receives
no updates during this mismatch. Rounds 26–50 perform 461 actor optimizer updates and
500 further critic updates. Final pre-fit/post-fit EV is 0.910/0.912. These EV values
describe bootstrapped targets, not ground-truth value accuracy. Warm-up also advances
environment age and random streams before actor learning; this is not an isolated
proof that critic fitting alone causes any behavioral difference.

**Unchanged CPU gates:** quiet survival and recovery remain 16/16 over 40 seconds.
Main seed 17 returns **311 survivors and 287 recoveries out of 512**, versus accepted
307/281 and immediate-restart control 294/276. Measured-contact recovery is 301/512
versus accepted 296/512. Thus the existing main continuation gate passes. Promotion
does not: paired recovery z versus shipped is 0.990, below 2 (56 gained, 46 lost).
Versus accepted, recovery gains/losses are 60/54 (z=0.562); versus immediate restart,
59/48 (z=1.063). These single-run differences do not establish a reliable improvement.

**Independent seed 71 is worse:** 69 survivors and 66 recoveries out of 128, versus
accepted 73/71 and immediate restart 71/69. Recovery versus accepted gains 10 worlds
and loses 15 (z=-1.0). Impact action change is 0.06535 versus accepted 0.06617, and
contact slip is 0.69853 versus 0.70827 m/world on the main benchmark. However, quiet
action change rises to 0.06513 from accepted 0.03910 (control 0.04620). These mixed
results do not establish improved visual realism or generalization.

**Decision:** retain this checkpoint as a continuation-eligible experimental candidate,
but keep the previously accepted actor as the stable reference and leave the shipped
policy unchanged. Do not infer a skeleton defect, change reward weights, or start an
hours-long run from this result. The next recommended experiment is a second training
seed for the same protected/immediate restart comparison, with the same CPU gates,
to check whether the main improvement repeats without the independent-seed loss.
No additional experiment has been launched. No production default was changed.

Evidence: `logs/compare_protected_restart.py`, `logs/foundation-restart-train.log`,
`logs/foundation-restart-{initial,integrity,comparison}.json`, and
`logs/foundation-restart-{512,seed71}.{json,log}`. All training and evaluation processes
completed successfully. No policy export, commit or push.

## 2026-09-20: reference-retention ablation and restart audit

Two bounded runs start from `2026-09-20_22-19-35_foundation_v2_pilot/model_50.pt`,
after its native 16-world, 40-second quiet gate passes. Each performs 25 training rounds,
4096 worlds x 16 steps, seed 443, 6 m/s, the existing reward and 0.1 temporal smoothing.
Only `reference_coef` differs: 1 or 0. The original teacher, data, actor, exploration,
critic and optimizer state are retained. This is a settings-matched comparison; disabling
retention does not disable entropy or temporal smoothing. Both finish in under two training
minutes, with 1,638,400 transitions each. Normal KL stops produce 467 actor updates with
retention and 455 without; both perform 500 additional critic updates.

Runs: `2026-09-20_22-35-39_foundation_ref1_ablation/model_25.pt` and
`2026-09-20_22-38-13_foundation_ref0_ablation/model_25.pt`.

**Isolation audit:** fingerprints confirm identical initial networks, teacher and reference
targets. Saved settings differ only in run name and reference coefficient. First-rollout
reset masks match, but numerical trajectories are not bit-identical. An independent frozen
GPU repeat confirms identical initial legacy inputs and small initial load differences
(maximum 0.000102 kN). Over 16 steps, observation RMS difference is 0.000816 and action RMS
difference 0.000552, with occasional larger excursions (maximum action difference 0.148).
Consequently a small score difference between these single runs is not conclusive evidence
about the retention coefficient.

**Identical-batch objective check:** both objectives are also applied to one shared frozen
rollout with the same minibatch random seed. Initial actor-mean gradient norms are PPO 3.118,
full-corpus retention 1.176 and weighted smoothing 0.0226. PPO/retention cosine is -0.073;
PPO/combined cosine is 0.933. This first, pre-impact batch does not show retention dominating
the actor direction; it cannot establish the relationship during recovery. Critic updates
are identical between the two common-batch arms.

**Restart finding:** the source ends with pre-fit critic EV 0.912, but its first fresh rollout
has EV approximately -4.95, signed value bias -37.3 and RMSE 61.7. Both arms update the actor
immediately because the saved warm-up counter is zero. Pre-fit EV remains negative at the
logged rounds 5, 10 and 15, then reaches about 0.87 at round 20. These are bootstrapped-target
diagnostics, not ground-truth value accuracy. Each new process recreates environments at rest;
clock staggering also offsets the shot schedule, so initial shots still wait 4–7 physical
seconds. Twenty-five rounds cover only 6.667 seconds per world across resets. This motivates
testing actor-frozen settling/critic adaptation on resume rather than inferring that an
already-adapted saved critic is ready for newly reset states.

**CPU result: both rejected.** Quiet-room survival and recovery remain 16/16 over 40 s.
On the unchanged 512-world seed-17 impact benchmark, retention-on returns 294 survivors
and 276 recoveries; retention-off returns 298/278. The accepted actor remains 307/281.
Both fail the no-regression continuation gate and neither qualifies for promotion.
Retention-off versus on has 52 newly recovered worlds and 50 lost recoveries (z=0.198):
no persuasive recovery benefit from disabling the teacher. Against the accepted actor,
recovery z is -0.488 with retention and -0.302 without; these are gate failures, not proof
of statistically conclusive harm. Independent seed 71 returns 71/69 with retention and
72/66 without, versus the accepted 73/71 (survival/recovery counts, 128 worlds).

Quiet action change is 0.04620 with retention and 0.02597 without, versus accepted 0.03910.
The smoother quiet actions without retention do not establish improved impact recovery or
overall visual realism. Impact action change is 0.06751/0.06564 (on/off), and measured contact
slip is 0.71847/0.72138 m/world versus accepted 0.70827.

**Next decision:** keep the accepted actor and do not extend either ablation checkpoint.
Keep retention enabled pending better evidence. Test a protected restart from the same
foundation model 50: use the existing `--critic_warmup_iters 25` to freeze the actor while
the new environments settle and the critic adapts, then allow 25 actor-learning rounds.
Use `--iterations 50`, the same seed 443, and the existing EV readiness gate (extend the
critic-only phase if it does not pass). Compare against this retention-on control with
the unchanged CPU gates. This isolates a concrete lifecycle hypothesis before changing
rewards, geometry or committing to a longer run; it is not yet a production-default change.

Artifacts: `logs/perturb_reference_ablation.py`, `logs/compare_reference_ablation.py`,
`logs/perturb_reference_signal.py`, `logs/foundation-ref{0,1}-{initial,train,512,seed71}.*`,
`logs/foundation-reference-signal.{json,log}` and `logs/foundation-reference-comparison.json`.
Production training defaults, reward weights, model geometry and shipped policy are unchanged.

## 2026-09-20: foundation observations and protected migration

The foundation audit found an observation gap, not evidence that the skeleton needs rebuilding.
The policy still saw foot-origin height as contact after rewards had moved to measured floor
load. It also lacked the oldest action in the two-step actuation queue. The existing linear
velocity channel uses MuJoCo's subtree-COM spatial reference, rather than the pelvis origin.
These are now addressed by the opt-in `foundation_v2` contract (140 observations); the original
105 channels remain intact for legacy weights. See README for the migration and validation commands.

**Physical checks:** body mass 69.6259 kg; positive, physically valid principal inertias;
no rest-pose self-contact; all 30 policy actuators have the expected force and acceleration
direction. The unpowered body crosses the shared fall-height threshold after 0.996 s without
non-finite state. Maximum rest-pose joint gravity demand is 3.4% of available policy torque;
this scale check is not a proof of balance feasibility under impact. Pelvis-origin velocity
matches native `mj_objectVelocity` with `mjOBJ_XBODY`.

**Implementation checks:** all 80 Python tests and all 126 managed/native C# tests pass.
CPU/GPU checks cover measured loads, body-origin velocity, action ordering and partial reset.
C# native fixtures compare all 140 observations across 21 consecutive states in three initial
poses, plus independent flat/heel/toe load fixtures. Scratch ONNX export agrees with PyTorch
within 1.192e-7. ABI offsets were regenerated against MuJoCo 3.8.1, not hand-entered.

**Game/training audit:** the scene's 3 s interval is applied after a 2 s shot-resolution window,
so successful shots launch roughly 5 s apart, within training's 4–7 s range. Game aim lead
(capped at 0.5 m), speed jitter (6 +/- 0.25 m/s) and minimum launch height (0.1 vs 0.15 m)
remain distribution differences. Neither the projectile profile nor the fixed CPU benchmark
was changed during this sensor experiment.

**Pilot:** `2026-09-20_22-19-35_foundation_v2_pilot`, 4096 worlds x 16 steps, seed 431,
6 m/s, five minutes. The source is the accepted actor's quarter-noise frozen checkpoint
`2026-09-20_20-37-48_contact_v3_hard6_quarter/model_50.pt`. New actor columns start at zero;
the critic and both optimizers reset. The original incumbent remains the reference teacher,
with new physical observations recollected on CPU seeds 101/103. Model 50 retains every old
actor weight and exploration value bit-for-bit and all new actor columns are zero; it completes
critic warm-up at EV 0.913. Model 75 completes 25 actor-learning rounds (427 actor optimizer
steps; 1500 total critic steps), with nonzero weights on the added inputs.

**CPU result: rejected.** The migrated frozen actor (model 25, identical actor to model 50)
exactly reproduces the accepted actor's 512-case outcomes: 307 survived, 281 recovered,
296 measured-contact recoveries, and identical action-change/slip diagnostics. No paired
survival or recovery outcome changes, so migration itself preserves this baseline.
The learned model 75 keeps quiet-room survival/recovery at 16/16 over 40 s, but returns
297/512 survival, 268/512 recovery and 278/512 measured-contact recovery. Paired recovery
gains/losses versus the accepted actor are 53/66 (z=-1.192); this is not statistically
conclusive harm, but it fails the established no-regression continuation gate. Independent
seed 71 is also lower: 68/128 survived and 64/128 recovered, versus 73/71. Quiet action change
rises from 0.03910 to 0.05310; impact action change rises from 0.06617 to 0.06848 and measured
slip from 0.70827 to 0.72620 m/world. No shipped policy or model has been replaced.

**Next decision:** do not extend model 75 or launch an overnight chain. The physical checks
support retaining the current rig while investigating learning. The new sensor contract is
implemented and tested, but one short pilot does not establish its learning benefit. The next
bounded experiment should isolate the PPO objective from the fixed reference-retention loss
using identical collection settings and the actor-frozen, critic-adapted v2 model 50 as the
common start. Its normal quiet-room seed gate must still pass before training. Keep the original
teacher fixed and apply the existing CPU gates; do not change reward weights or geometry at
the same time. The accepted legacy training checkpoint remains `contact_v3_full5m/model_78.pt`.

Artifacts: `logs/foundation-audit.json`, `foundation-*-tests.log`, `foundation-dotnet-all.log`,
`foundation-v2-reference.pt`, `foundation-v2-{train,migration-integrity,pilot-integrity}.*`,
`foundation-v2-frozen-512.*`, `foundation-v2-pilot-{512,seed71}.*`,
`foundation-v2-comparison.json` and `foundation-export/`.

## 2026-09-20: PPO horizon and gradient investigation

The accepted training actor is still `2026-09-20_19-14-28_contact_v3_full5m/model_78.pt`.
No replacement has been exported. The PPO surrogate, Gaussian log probabilities/KL and
timeout-aware GAE review did not identify an arithmetic defect. This investigation measures
the quality of the learning signal before changing more reward weights.

**Observability correction.** PPO now reports pre-fit `value_ev_before`, signed
`value_bias_before` (prediction minus target), `value_rmse_before` and raw `advantage_std`.
The trainer prints these beside the existing post-fit `ev`. These are diagnostics against
bootstrapped batch targets, not held-out critic accuracy; neither is proof of useful recovery
gradients. Warm-up/optimizer behavior is unchanged. A regression test demonstrates that EV
can equal one while values have a constant error of five. All 22 training-safety and ten
policy-reference tests pass.

**Frozen trajectory probe.** Source: the quarter-noise, critic-adapted update-50 checkpoint
`2026-09-20_20-37-48_contact_v3_hard6_quarter/model_50.pt`. Its mean actor is exactly the accepted
actor; no rejected learned actor is reused. Seed 419, 512 GPU worlds, 768 control steps
(12.80 seconds/world), 6 m/s impacts every 4–7 seconds, no optimizer updates. Actual ball/body
contact is measured each physics substep by the existing exposure probe. The same trajectory
is evaluated with GAE cuts every 16 or 128 steps and with full-trajectory GAE; gamma/lambda
remain 0.99/0.95. The final 128 steps are excluded from the comparison of advantages.

Within two seconds of contact, 16-step advantage signs disagree with full-trajectory GAE
on 18.30% of samples; 128 steps reduce that to 3.61% (correlations 0.844 and 0.976).
This establishes sensitivity to truncation, not that full-trajectory estimates are ground
truth. Longer lambda-one return diagnostics also bootstrap at timeouts and the final boundary.
The [GAE formulation](https://arxiv.org/abs/1506.02438) trades bias against variance; a longer
trace alone need not improve a learned controller.

At three 65,536-transition windows, initial PPO gradient norms were 3.49–3.66 for 16-step
targets, versus 0.011–0.014 for weighted temporal smoothing and 1.08–1.12 for the fixed
reference. Smoothing does not dominate these instantaneous gradients. Norms do not measure
the eventual Adam update or establish whether an auxiliary objective helps behavior.

**Matched horizon experiment.** Both arms use 512 worlds x 128 collected steps, 12 PPO rounds,
seed 419, the same source/optimizer/teacher, quarter noise, fixed 6 m/s, 20-second episodes,
LR ceiling `1e-6`, KL target 0.001 and smoothing 0.1. Only GAE truncation differs: the control
cuts at 16-step boundaries; treatment uses all 128. Both finish below the five-minute cap.
The KL guard permits 211 and 224 actor optimizer steps respectively; both take 240 critic
steps. Teacher targets, variance and provenance remain exact. GPU trajectories are not
bitwise identical across processes; this is one training seed, not a replicated result.

- Control: `2026-09-20_21-09-13_contact_v3_gae16_control/model_12.pt`.
  Main survival/recovery 293/512 and 273/512, versus accepted 307/512 and 281/512.
  Rejected. Independent seed 71: 75/128 and 72/128, versus accepted 73/128 and 71/128.
- Treatment: `2026-09-20_21-12-16_contact_v3_gae128/model_12.pt`.
  Main survival/recovery 306/512 and 287/512. Rejected by the unchanged survival gate,
  despite six additional recoveries. Recovery pairs versus accepted: 52 gained, 46 lost,
  z = 0.606; versus the control: 56 gained, 42 lost, z = 1.414. Neither establishes a
  population improvement. Measured-contact recovery is 294/512 versus accepted 296/512.
  Independent seed 71: 78/128 survived, 71/128 recovered. Quiet survival/recovery stays
  16/16 in both arms. Treatment quiet action change is 0.0436 versus accepted 0.0391;
  hard-hit sliding is 0.702 versus 0.708 m/world.

**Gradient agreement.** A separate frozen repeat splits each window by disjoint worlds,
using common whole-batch advantage normalization. For 512 worlds x 128 steps, the three
half-gradient cosines are 0.249/-0.068/0.061 with 16-step cuts and 0.191/-0.220/-0.035 with
128-step cuts. A further frozen probe at the normal 4,096-world x 16-step batch shape gives
0.011/-0.082/0.119 at control steps 384/576/768, even though pre-fit EV is 0.910/0.898/0.906.
These few snapshots show weak agreement in sampled gradient directions at both shapes;
they do not estimate a population signal-to-noise ratio or prove a specific critic defect.
No learning occurs in these probes, and all constraint-buffer checks pass.

**Larger-batch validation also rejected.** The final trial uses 4,096 worlds x 128 steps
(524,288 transitions/update), retaining the same frozen source and other training settings.
`2026-09-20_21-25-04_contact_v3_gae128_large/model_11.pt` finishes 11 rounds at the end of
the iteration crossing its five-minute budget: 190 actor optimizer steps and 220 critic
steps. Original teacher targets/variance/provenance remain exact; final LR is `6.67e-7`.
Main survival/recovery is 295/512 and 274/512, versus accepted 307/512 and 281/512.
Recovery pairs: 54 gained, 61 lost, z = -0.653. Measured-contact recovery is 287/512.
Quiet survival/recovery remains 16/16, but quiet action change rises to 0.0565; hard-hit
contact sliding falls to 0.690 m/world. Seed 71 gives 75/128 survivors and 70/128 recoveries.
Neither earlier rejected candidate was used as its seed. This is a bounded configuration
validation, not a matched replication of the 512-world experiment: it finishes 11 rather
than 12 rounds and its KL guard permits a different number of optimizer steps.

**Decision.** All three learned candidates fail the unchanged main continuation gate.
Keep the accepted `contact_v3_full5m/model_78.pt`; do not resume or export these trials.
Longer traces and a larger batch have not established a recovery improvement, and these
results do not justify an hours-long run or another reward-weight change. Weak sampled
PPO gradient agreement is measured, but its cause is not yet isolated. The next controlled
test should separate the PPO reward gradient from the fixed-reference retention objective,
using the same validated source and CPU gates, before changing either objective. Reference
gradient magnitude alone is not evidence that removing the reference is safe.

Production defaults, reward weights, physics, actor observations and all acceptance/promotion
thresholds remain unchanged. All processes have finished. No export, commit or push.
`git diff --check` passes; the 32 targeted tests cover the only production change, additional
diagnostics. Frozen probes are separate from the three bounded learning trials.

Evidence: `logs/perturb_ppo_probe.py`, `logs/perturb_ppo_worlds_probe.py`,
`logs/perturb_gae_ablation.py`, `logs/compare_gae_ablation.py`,
`logs/contact-v3-ppo-{probe,signal,worlds}.{json,log}`,
`logs/contact-v3-gae{16,128}-{train,512,seed71}.log`,
`logs/contact-v3-gae{16,128}-{512,seed71}.json`,
`logs/contact-v3-gae-{comparison,training-integrity}.json`,
`logs/contact-v3-gae128-large-{train,512,seed71}.log`,
`logs/contact-v3-gae128-large-{512,seed71,integrity}.json`.

## 2026-09-20: impact exposure measured; hard-stage and reduced-noise trials rejected

The accepted actor remains `2026-09-20_19-14-28_contact_v3_full5m/model_78.pt`.
No replacement is accepted or exported. This investigation used separate diagnostic seeds
and two bounded five-minute training trials; neither passed the existing CPU gate.

**Difficulty and exposure.** A matched CPU sweep on diagnostic seed 223, 128 worlds,
six seconds and one shot measured recovery 125/128 at 2 m/s, 120/128 at 4 m/s, and 75/128
at 6 m/s (97.66%, 93.75%, 58.59%). Survival was 127/128, 124/128 and 81/128. The policy
already handles substantially more than the earlier 2–2.2 m/s training stages. This sweep
is diagnostic; promotion continues to use the unchanged seed-17 benchmark.

`logs/perturb_exposure_probe.py` observes the frozen accepted policy under the GPU training
reset, random episode phase, 4–7 second launch interval and Gaussian action sampling.
Seed 307, 256 worlds, 1,200 control steps (20.0016 seconds per world), no learning.
Ball–body contacts require more than 1 N normal force and are checked at every physics
substep, excluding floor/ball self-pairs and stale contact slots. A kernel fixture verified
those exclusions, and all probes passed the constraint-buffer check. A hit is counted once
per launched ball; some launches can still be pending or miss at the measurement boundary.

At 400 steps, only 5.79%/9.05%/9.39% of transitions were within two seconds of a detected
impact at 2/4/6 m/s. At 1,200 steps these shares were 21.74%/21.91%/19.89%; all 256 worlds
had experienced contact. Full runs therefore do obtain impact experience. Pre-launch
transitions still occupied 46.9%/50.7%/61.7%, including repeated waits after resets.
Detected hit counts were 629/640/651 from 690/671/678 launches. Fall counts were 6/68/249,
with 3/66/246 following contact on the most recent shot. These are repeated-episode counts,
not survival percentages or the deterministic CPU acceptance metric.

**Exploration control.** No completed sampled episode met the strict settled-recovery hold
at the original per-joint noise, even at 2 m/s. Additional frozen-actor 6 m/s controls used
the same seed and probe with one-quarter noise and zero noise. Settled episode endings:
0/414 at full noise, 37/380 at quarter noise, 38/381 at zero noise; fall counts 249, 207, 219.
All controls retain the original mean actor and consume the same initial sampling stream;
subsequent resets and shots diverge, so these aggregate counts are not paired trials.
They support investigating reduced exploration, not a claim of improved learned recovery.

**Training support added.** `train.py --exploration_scale <factor>` explicitly scales a
seed's learned per-joint standard deviations, preserving actor means and mean-parameter
Adam state. Only std Adam state is cleared. Perturb forces at least 50 critic-only updates
and retains the explained-variance gate; ordinary resumes inherit the saved noise without
reapplying the factor. The original teacher variance/targets remain unchanged. Invalid
factors, missing seeds and mixing with `--reset_std` are rejected. Startup logging now shows
actual policy std rather than the unused fresh-policy `init_std`. Defaults are unchanged.
All 21 training-safety and 10 policy-reference tests pass; three new tests cover scaling,
optimizer isolation, warm-up, identity scaling, normal resume and invalid requests.

**Trial A: game-strength impacts with original exploration.**
`2026-09-20_20-21-15_contact_v3_hard6/model_78.pt` used fixed 6 m/s, all other settings
unchanged, 50 critic-only updates then 28 actor updates (560 actor optimizer steps) within
five minutes. The update-50 actor/std were verified byte-identical to the accepted seed;
critic explained variance was 0.904 at that boundary. Main survival/recovery declined
307/512 -> 290/512 and 281/512 -> 272/512 (recovery pairs 42 gained, 51 lost, z = -0.93).
Seed 71 declined 73/128 -> 69/128 and 71/128 -> 63/128. Quiet remained 16/16, but action
change increased 0.0391 -> 0.0989. Rejected.

**Trial B: game-strength impacts with quarter exploration.**
`2026-09-20_20-37-48_contact_v3_hard6_quarter/model_79.pt` started from Trial A's frozen
update-50 checkpoint, not its rejected updated actor. Noise mean changed 0.1771 -> 0.0443,
preserving its per-joint pattern; no joint reached the existing 0.01 floor. Fifty further
critic-only updates adapted to the new sampling distribution, reaching explained variance
0.910. The remaining 29 updates took 550 actor optimizer steps; KL protection reduced the
learning rate during adaptation (final `6.67e-7`). The original teacher was verified intact.
Main survival/recovery declined 307/512 -> 297/512 and 281/512 -> 272/512 (50 gained,
59 lost, z = -0.86). Seed 71 declined 73/128 -> 68/128 and 71/128 -> 65/128. Quiet remained
16/16; action change was 0.0556, better than Trial A but worse than the accepted seed.
Measured main contact sliding rose 0.708 -> 0.723 m/world. Rejected.

Both trials used matched CPU settings/metric versions against the saved accepted policy
and incumbent. They also differ in starting critic and warm-up history, so their learning
results are not a strict noise-only ablation. Neither establishes a population improvement;
the point-estimate regressions trigger the unchanged stop rule. Increased difficulty and
reduced exploration alone have not resolved learning. The quarter-noise update-50 checkpoint
retains the exact accepted mean actor with an adapted critic, for diagnostics only. Further
work should examine how PPO updates affect impact recovery and standing, not assume that
more hours or another reward-weight guess will solve the problem. All processes finished;
no export, commit or push. The two training trials consumed ten GPU minutes, separate from
the frozen-policy probes and CPU evaluations. `git diff --check` passes.

Evidence: `logs/contact-v3-exposure-speed{2,4,6}.{json,log}`,
`logs/contact-v3-exposure-{gpu,quarter,mean}.{json,log}`,
`logs/contact-v3-hard6{-quarter}-train.log`,
`logs/contact-v3-hard6{-quarter}-{512,seed71,comparison}.json` and matching evaluation logs.

## 2026-09-20: curriculum resume defects fixed; harder-stage trial rejected

Two concrete resume defects are fixed. `overnight.py` discarded the starting seed's CPU
stage recommendation; it now applies that recommendation before the first chunk, just as
it does after an accepted chunk. `train.ps1` always supplied the configured 2 m/s start,
overriding the checkpoint's saved difficulty; resumed Perturb chains now inherit that
difficulty. Deliberate overrides remain available through `overnight.py --ball_speed`.
The configuration comment and README distinguish resumed stages from direct/fresh starts.
The previous manual fixed-2 m/s ablations bypassed these entry points: these defects do not
establish the cause of their score regressions.

The initial-stage regression test failed before the fix (launched at 2.0 despite a 2.2
recommendation) and passes afterward. A PowerShell launch test captures actual wrapper
arguments and verifies that the seed is forwarded without overriding its stage. All
18 training-safety and 18 architecture tests pass. No policy, reward, physics or observation
implementation changed, and `git diff --check` passes.

The accepted `contact_v3_full5m/model_78.pt` was evaluated at its current 2 m/s stage:
512/512 survived and 508/512 recovered (99.22%). That exceeds the existing 85% advancement
threshold. The next trial used the existing 10% increment, 2.2 m/s, rather than another
fixed-2 m/s continuation. The launch applied the measured decision directly, reusing the
saved quiet/hard-hit baselines; the corrected runner and wrapper paths are covered by tests.

`2026-09-20_19-55-27_contact_v3_curriculum22/model_78.pt` completed five minutes / 78
additional updates from the accepted checkpoint, with seed 0 and every other training setting
unchanged. Its saved difficulty is 2.2 m/s, actor optimizer count 3,120, and original teacher
targets/scales/provenance remain exact. There were no reported KL stops or numerical failures.
The last logged explained variance was 0.770. Quiet survival and recovery remained 16/16.

**The learning regression is not resolved.** At the fixed 6 m/s, seed-17 benchmark,
survival declined 307/512 -> 300/512 (59.96% -> 58.59%) and recovery 281/512 -> 273/512
(54.88% -> 53.32%). Recovery pairs: 54 gained, 62 lost, z = -0.74. Measured-contact recovery
also declined, 57.81% -> 55.66%. The seed-71 batch declined 73/128 -> 70/128 in survival
and 71/128 -> 61/128 in recovery (8 gained, 18 lost, z = -1.96). All hard-hit comparisons
used matching evaluation settings and metric versions; the unchanged gate rejects this trial.

At its new 2.2 m/s stage the candidate still survived 511/512 and recovered 507/512 (99.02%).
The stage scores use different speeds and must not be read as a paired learning comparison.
They show that the tested easy stages remain far easier than the game's impacts. Under-fire
action change was nearly unchanged from the accepted seed, while measured contact sliding
rose 0.708 -> 0.719 m/world; quiet action change rose 0.0391 -> 0.0587. Raising the easy-stage
speed alone did not produce robust hard-impact recovery.

Keep the resume fixes, retain `contact_v3_full5m/model_78.pt` as the accepted training seed,
and do not resume the rejected `contact_v3_curriculum22` candidate. Strong-impact training
coverage remains the next hypothesis to test, not an established reward defect or a reason
to weaken acceptance gates. No further chunk or export was launched. All processes finished;
no commit or push was performed.

Evidence: `logs/contact-v3-stage2-512.{json,log}`,
`logs/contact-v3-curriculum22-{train,512,seed71,stage}.log`,
`logs/contact-v3-curriculum22-{512,seed71,stage,comparison}.json`.

## 2026-09-20: guarded continuation stopped after the first five-minute chunk

The approved budget was up to three five-minute chunks, with CPU evaluation before each
continuation and a stop on regression. Only chunk 1 ran. It started from the accepted
`2026-09-20_19-14-28_contact_v3_full5m/model_78.pt` and produced
`2026-09-20_19-28-30_contact_v3_chain1/model_79.pt` after 79 additional updates.
Training used the same seed 0, 4096 worlds, 16 steps, 20-second episodes, fixed 2 m/s shots
every 4–7 seconds, actor ceiling `1e-6`, KL target 0.001, smoothing 0.1 and reference
coefficient 1. The embedded original incumbent teacher was restored from the checkpoint;
targets, scales, provenance and coefficient were verified unchanged. The optimizer reached
3,140 cumulative actor steps, exactly 1,580 above the accepted seed. Reward/training versions
matched, warm-up remained complete, and no KL stops or numerical failures were reported.

**Rejected by the unchanged main continuation gate.** On 512 seed-17 hard-hit worlds,
survival fell from 307/512 to 297/512 (59.96% -> 58.01%), and settled recovery from
281/512 to 277/512 (54.88% -> 54.10%). Recovery pairs: 51 gained, 55 lost, z = -0.39;
survival pairs: 40 gained, 50 lost, z = -1.05. These point-estimate regressions trigger
the stop rule without establishing statistically significant population regressions.
Against the shipped incumbent, recovery tied 277/512 and survival fell 302/512 -> 297/512.
Measured-contact recovery also declined against the accepted seed, 57.81% -> 56.25%.

Quiet survival/recovery stayed 16/16 over 40 seconds. Under-fire mean action change fell
0.0662 -> 0.0630 and measured contact sliding 0.708 -> 0.696 m/world, but quiet action
change rose 0.0391 -> 0.0482. The smaller seed-71 diagnostic held survival at 73/128 and
lost two recoveries, 71/128 -> 69/128 (paired recovery z = -0.38). Neither calmer impact
motion nor intact quiet standing compensated for the main gate failure.

All comparisons used matching evaluation settings and metric versions. Chunks 2 and 3
were not launched. Retain `contact_v3_full5m/model_78.pt` as the accepted training seed;
do not resume `contact_v3_chain1/model_79.pt`. This trial provides no basis for an
unattended longer run under the current setup. No reward/configuration or production code
changed, no policy was exported, and no commit or push was performed. All processes finished.

Evidence: `logs/contact-v3-chain1-train.log`,
`logs/contact-v3-chain1-{512,seed71}.{json,log}` and
`logs/contact-v3-chain-comparison.json`.

## 2026-09-20: full five-minute v3 trial passes continuation, not promotion

`2026-09-20_19-14-28_contact_v3_full5m/model_78.pt` completed the full five-minute
training budget from `2026-09-20_18-39-12_contact_v3_critic_extend/model_25.pt`, whose
actor is the unchanged incumbent. The initial CPU quiet gate passed. This trial retained
seed 0 and all previous settings: 4096 worlds, 16 rollout steps, 20-second episodes,
2 m/s shots every 4–7 seconds, actor ceiling `1e-6`, KL target 0.001, smoothing 0.1,
and fixed-reference coefficient 1. Only the iteration ceiling was raised to let the
five-minute limit control duration. No reward, policy architecture or production code changed.

The final checkpoint, selected by elapsed training time before evaluation, contains 78
updates / 1,560 actor optimizer steps. There were no reported KL stops or numerical failures.
The saved teacher targets, scales and provenance match the original corpus exactly.
Training covered 20.80 simulated seconds per world, including resets, versus 6.67 in the
short trials. Actual impact exposure was not counted. The last logged critic explained
variance was 0.618 at update 75; the final saved checkpoint is update 78.

**Main CPU continuation gate passes.** On the same 512 seed-17 hard-hit worlds, survival
rose from 302/512 to 307/512 (58.98% -> 59.96%), and historical settled recovery from
277/512 to 281/512 (54.10% -> 54.88%). All 16 quiet worlds survived and settled over
40 seconds. Recovery pairs: 59 gained, 55 lost, z = +0.37. This is not a statistically
established recovery improvement and fails the unchanged promotion threshold of z >= 2.
The incumbent remains shipped; this candidate is eligible for guarded continuation only.

Quiet mean action change fell 52%, from 0.0814 to 0.0391. Under fire, it fell 17%, from
0.0795 to 0.0662, and measured contact sliding fell 6.2%, from 0.755 to 0.708 m/world.
Measured-contact recovery rose from 56.25% to 57.81%. Historical rapid replants increased
slightly, 0.174 -> 0.186; measured load replants declined 9.65 -> 8.84. These improvements
in aggregate motion do not establish visually realistic recovery.

The independent seed-71, 128-world diagnostic recovered 71/128 versus 66/128 (55.47%
versus 51.56%), paired z = +1.00, but survival declined by one world, 74/128 -> 73/128.
This mixed result supports keeping continuation bounded and evaluated; it is not a second
full gate pass. Evaluation settings and both metric versions matched the saved incumbent
results before comparison. No new gate thresholds were introduced.

Next recommendation: continue from `contact_v3_full5m/model_78.pt` in five-minute chunks,
retaining the original fixed teacher and current settings, evaluating each chunk against
the accepted checkpoint and incumbent, and stopping on regression. No further chunk was
launched in this trial. All training/evaluation processes finished. No export, commit or push.

Evidence: `logs/contact-v3-full5m-train.log`,
`logs/contact-v3-full5m-{512,seed71}.{json,log}` and
`logs/contact-v3-full5m-comparison.json`.

## 2026-09-20: two additional training seeds; neither qualifies for continuation

Repeated the v3 fixed-reference trial with training seeds 1 and 2, each starting from
`2026-09-20_18-39-12_contact_v3_critic_extend/model_25.pt`. Its actor and exploration
parameters were verified byte-identical to the shipped incumbent; warm-up was complete.
Both trials used the same settings as seed 0: 25 updates, 4096 worlds, 16 rollout steps,
20-second episodes, 2 m/s shots every 4–7 seconds, actor ceiling `1e-6`, KL target 0.001,
smoothing 0.1 and reference coefficient 1. The saved arguments differ only in seed and run
name. Both took 500 actor optimizer steps with no KL stops, about 1.7 and 1.8 GPU minutes.
The saved reference targets, scales and provenance match the original corpus exactly.

- Seed 1: `2026-09-20_18-56-58_contact_v3_seed1/model_25.pt`.
  Main seed-17 benchmark: 303/512 survived (59.18%), 273/512 recovered (53.32%), versus
  incumbent 302/512 and 277/512. Recovery pairs: 55 gained, 59 lost, z = -0.37.
  Independent seed-71 batch: 70/128 survived, 65/128 recovered, versus 74/128 and 66/128;
  recovery z = -0.19. Rejected for lower main-benchmark recovery.
- Seed 2: `2026-09-20_19-00-12_contact_v3_seed2/model_25.pt`.
  Main benchmark: 298/512 survived (58.20%), 275/512 recovered (53.71%);
  recovery pairs: 48 gained, 50 lost, z = -0.20. Independent batch: 75/128 survived,
  69/128 recovered; recovery z = +0.54. Rejected for lower main survival and recovery.

Both retained 16/16 quiet survival and recovery over 40 seconds. Quiet action change fell
from 0.0814 to 0.0429/0.0436 (47%/46%). Main-benchmark action change fell 13%/13%, and
measured contact sliding fell 5.2%/3.9%. Historical height-based replants were 0.170/0.217
versus 0.174; calmer action does not imply every foot-motion metric improved. Measured
contact recovery was 56.25%/55.66% versus 56.25%. All evaluation settings and metric
versions matched the saved incumbent evaluations before paired comparisons.

Neither new candidate satisfies the unchanged continuation gate. No five-minute extension
was launched; keep the incumbent. The differences do not establish statistically significant
population regressions, and no seed establishes a recovery improvement. All training and
evaluation processes finished. No policy export, commit or push was performed.

**Trial-design limitation:** 25 updates × 16 steps × 4 physics substeps × 0.004167 seconds
provide only 6.67 simulated seconds per world, including any resets. Fresh shots wait 4–7
seconds; nominal flight from 2 m at 2 m/s takes another second. Randomizing episode clocks
also offsets the shot schedule, so it does not supply immediate impact experience. These
trials predominantly test retention and early adaptation, not sustained recovery learning;
actual impact exposure was not counted in their logs. Their results do not justify declaring
the reward wrong or ruling out longer training. Before more reward changes or hours of
training, the next proposed experiment is a full five-minute trial from the validated incumbent
with the warmed v3 critic, followed by the same CPU gates. Do not resume a rejected candidate.

Evidence: `logs/contact-v3-seed{1,2}-train.log`,
`logs/contact-v3-seed{1,2}-{512,seed71}.{json,log}` and
`logs/contact-v3-multiseed-comparison.json`.

## 2026-09-20: measured floor contact implemented; short trial rejected

`loaded_contact_recovery_v3` replaces reward grounding by foot-origin height with
floor-contact normal load from the foot and toe geometries. The shared hysteresis enters
above 5 N and exits at or below 1 N. Sliding is the per-foot load-weighted mean squared
tangential speed at contact points: a stationary heel/toe pivot is no longer charged as
sliding merely because the foot origin moves, and opposite velocities during twisting
cannot cancel. CPU and GPU adapters read the final physics substep. Reward coefficients,
policy observations, action mapping/latency, physics and PPO defaults are unchanged.

The scorer owns a separate `height_proxy_v2` contact history. Its gate and historical
movement metrics remain unchanged; `contact_*` fields are additional `loaded_contact_v3`
diagnostics. Re-evaluating the incumbent reproduced every historical survival/recovery
outcome and all three movement aggregates on the 512-world seed-17 benchmark and its
16-world, 40-second quiet test. New contact scoring alone raises the unchanged incumbent's
reported recovery from 54.1% to 56.3%; it must not be mistaken for learning progress.

Validation: **68 unit tests pass**, including eight new contact tests. Those exercise flat
loading, raised feet, heel/toe pivots, true sliding, twisting, ball-contact exclusion,
NumPy/Torch hysteresis, partial resets, frozen scoring, and physical CPU/GPU contact parity
with empty-buffer/world isolation. Both environment parity scripts pass, as does Perturb
preflight with zero failures/warnings. Standing reward parity error was below `6.2e-5`.
The original severely penetrating fixtures produce loads up to 22 kN and independent
CPU/GPU force distributions: raw reward differences reach 0.023 there. These differences
are reported; reward arithmetic is separately checked with identical slip measurements,
while physical flat/pivot/sliding fixtures compare native load/slip without substitution.
No simulation solver settings were changed to make the tests pass. `git diff --check` passes.

The incumbent actor was re-warmed under the new reward for 50 + 25 critic-only iterations:
`2026-09-20_18-34-01_contact_v3_critic/model_50` and
`2026-09-20_18-39-12_contact_v3_critic_extend/model_25`. Explained variance rose from 0.345
after the first chunk to 0.560 after the extension. Actor and exploration parameters were
verified byte-identical to the incumbent at both boundaries; the fit threshold stayed 0.5.

The only policy-training trial, `2026-09-20_18-43-16_contact_v3_trial/model_25`, then ran
25 iterations (500 actor Adam steps, about 1.6 GPU minutes), using seed 0, 4096 worlds,
16 rollout steps, 20-second episodes, fixed 2 m/s shots every 4–7 seconds, actor ceiling
`1e-6`, KL target 0.001, smoothing 0.1 and the unchanged all-phase reference corpus with
coefficient 1. There were no KL-stop events. Saved reference observations, means and scales
remain byte-identical to the original corpus. Critic warm-up consumed about 4.9 additional
GPU minutes; CPU validation time is separate.

**Rejected by the unchanged main gate.** On 512 seed-17 hard-hit worlds, survival declined
from 302/512 (59.0%) to 291/512 (56.8%), and settled recovery from 277/512 (54.1%) to
272/512 (53.1%). Paired recovery: 51 gained, 56 lost, z = -0.48. This fails the point-estimate
safety gate without establishing a statistically significant population regression.
Measured-contact recovery also declined, 56.3% to 55.3%, so the negative result is not
merely an artifact of preserving the old gate.

Quiet survival/recovery remained 16/16 over 40 seconds. Quiet mean action change fell
0.0814 -> 0.0488 (40%), while measured contact slip fell 0.357 -> 0.338 m/world. Under fire,
action change fell 0.0795 -> 0.0704 and measured contact slip 0.755 -> 0.728 m/world (3.6%).
Measured load replants fell 9.65 -> 9.06 per world; historical height replants fell
0.174 -> 0.150. Different measurement definitions must not be compared as one series.

The independent seed-71, 128-world batch improved survival 74/128 -> 76/128 (57.8% -> 59.4%)
and recovery 66/128 -> 71/128 (51.6% -> 55.5%), paired z = +0.93. This smaller, inconclusive
improvement does not override the main gate. The contact defect is fixed and movement is
calmer, but better recovery and visually realistic behavior have not been established.
Keep the incumbent; do not seed a longer chain from this rejected candidate. No policy
was exported and no commit or push was performed.

Evidence: `logs/contact-v3-{critic,critic-extend,trial}.log`,
`logs/contact-v3-{incumbent,candidate}-{512,seed71}.json`,
`logs/contact-v3-comparison.json`, and the contact/parity/preflight tests.

## 2026-09-20: contact measurement defect confirmed in matched replays

The [recovery diagnosis](PERTURB_DIAGNOSIS.md) reproduces all 1,024 incumbent/candidate
survival and recovery outcomes from the existing 512-world evaluations and records actual
floor loads at every physics substep. The reward's foot-origin-height proxy calls loaded,
tilted feet airborne: 47.9% of airborne-labelled incumbent foot samples still carry floor
load during the first half-second after impact, rising to 61.1% over the next second.
Both proxy feet are airborne despite actual support in 40.3%/46.5% of those response samples,
forcing support reward to zero. The candidate exhibits the same defect.

Correct contact measurement before tuning movement penalties or running longer training.
The report includes matched replays, reward-term attribution, a contact-only reward
counterfactual, and a scoped CPU/GPU correction plan. This identifies a real measurement
defect; it does not establish that correcting it alone will solve the recovery problem.
No production configuration or policy changed during this diagnosis.

## 2026-09-20: fixed-reference retention experiment

The optional `--reference_data` / `--reference_coef` path adds an actor-only penalty against
frozen incumbent mean actions. `policy_reference.py` owns balanced standing/impact sampling,
the fixed teacher variance scale, drift reporting and serialization. The critic and reward
kernel are unchanged. Checkpoints embed the original reference and its sampling state; a new
training chunk does not silently replace the teacher with the latest student. Current plant
hashes must match. Reference sampling does not consume rollout/PPO randomness.

The CPU corpus uses seeds 101 and 103, separate from evaluation seed 17. All 16 quiet trajectories
and 39/64 hard-hit trajectories survived and settled. Failed recoveries are not imitation targets.
The collector now samples uniformly across all control steps, capped at 4096 states per group,
to avoid aliasing alternating actions with a fixed temporal stride. A regression test explicitly
checks both phases of an alternating controller.

The experiment holds `grounded_recovery_v2`, temporal smoothing 0.1, actor ceiling `1e-6`, KL target
0.001, seed 0, 4096 worlds, 16 steps, 20-second episodes and 2 m/s training shots fixed. Each run
starts from the same warmed `2026-09-20_16-29-23_recovery_safe_critic/model_50.pt` and takes 25
updates (about 1.6 minutes of GPU training). The fixed teacher is the shipped `p0911i_s1/model_503`.

The initial, strided-corpus comparison used coefficient 1 versus 0. The zero-penalty control
survived 56.8% and recovered 52.5% of 512 hard-hit worlds. Anchoring reached 58.8% and 53.9%,
respectively, against the incumbent's 59.0% and 54.1%. That is one fewer surviving and recovered
world than the incumbent, so the unchanged strict gate rejects it. This point difference does
not establish a population-level regression. Quiet survival/settling stayed 16/16; anchored mean
action change fell from 0.0814 to 0.0443 and quiet sliding from 0.295 to 0.284 m/world. Under fire,
action change fell from 0.0795 to 0.0721, while sliding rose from 0.367 to 0.371 m/world.

The corrected all-phase corpus is evaluated separately. The zero-coefficient control is independent
of corpus contents (covered by an exact update-equivalence test). These are short, single-training-
seed experiments, not evidence that long training or visually realistic recovery is solved.
The reused incumbent benchmark has identical CPU evaluation settings; no plant, observation,
reward or scorer changes were made for this experiment.

**Final corrected-corpus result: rejected.** `2026-09-20_17-38-23_recovery_reference_dense/model_25`
survived 295/512 (57.6%) and recovered 267/512 (52.1%), versus 302/512 and 277/512 for the incumbent.
Paired recovery z = -0.94; the point estimates fail the unchanged acceptance gates, without
establishing a statistically significant population-level regression. Quiet survival and settling
remained 16/16. Quiet action change was 0.0739 (9.2% below the incumbent), and under-fire action
change was 0.0734 (7.6% lower). Under-fire sliding was slightly higher, 0.369 versus 0.367 m/world,
and rapid replants increased from 0.174 to 0.191. Less action oscillation is therefore not evidence
of better foot placement or recovery. No tested candidate qualifies to seed a longer chain or
replace the incumbent. Reference retention remains opt-in; no reward weights or acceptance gates
were relaxed to obtain a passing result.

Validation: 10 reference tests plus the existing 50 regression tests pass. The tests cover balanced
weighting, target immutability, actor gradients, independent random streams, warm-up isolation,
zero-penalty equivalence, checkpoint continuation, repeated-artifact continuation, wrong-plant
rejection and phase coverage.
The actual trained checkpoint also retained byte-identical reference observations, targets and
scales. No policy has been exported.

Evidence: `logs/recovery-reference-{anchor,control,dense}.log`, matching `*-512.json` evaluations,
`logs/recovery-reference-dense.pt`, and the `2026-09-20_*_recovery_reference_*` checkpoints.

## 2026-09-20: reject collapsed seeds and protect fine-tuning

The subsequent 15-, 60- and 90-minute checkpoints all failed quiet standing (0/64 survivors
over six seconds). More training from those checkpoints is not justified. The seeded 15-minute
run also collapsed, so starting from scratch was not the only problem. Its earlier three-minute
smoke result did not establish that longer training was safe.

The trainer had discarded time-limit value targets as if every timeout were a fall. Both
environments now preserve final observations before reset, and GAE bootstraps genuine timeouts
without propagating advantages into the reset episode. Actor and critic now have separate Adam
optimizers and gradient clipping. Perturb uses unclipped value regression, actor-only KL stopping,
a `1e-6` actor learning-rate ceiling, and a frozen-actor critic warm-up after target-version changes.
Warm-up lasts at least 50 iterations and until explained variance reaches 0.5. All continuation
state is checkpointed; incompatible target versions retain only the behavior seed.

Perturb requires an explicit checkpoint that survives and settles in all 16 CPU quiet-room
worlds over 40 seconds. The 90-minute checkpoint was rejected by this gate before GPU setup.
`train.ps1` splits budgets into chunks of at most five training minutes. Each chunk must pass
quiet standing and CPU recovery/survival comparisons before it can seed the next one. A rejection
stops the chain with exit code 2 and retains the accepted checkpoint; repeating an identical failed
experiment is no longer automatic. Evaluation time is additional to the training budget.

Controlled trials start from `2026-09-20_16-29-23_recovery_safe_critic/model_50.pt`: 50 critic-only
iterations from the shipped `p0911i_s1/model_503`, explained variance 0.556, actor and exploration
verified byte-identical to the incumbent. Each smoothing trial uses the same warmed critic,
optimizer state and seed, 4096 worlds, 16 rollout steps, 20-second episodes, fixed 2 m/s shots
every 4–7 seconds, and 50 PPO iterations. The initial sweep used a `3e-5` ceiling, KL target
0.005 and the earlier `1e-5` adaptive floor; the final implementation also lets the adaptive
floor fall below a small ceiling. CPU comparisons use seed 17, 16 quiet worlds for
40 seconds and 64 worlds for a single 8 kg, 6 m/s shot over six seconds.

Validation: 16 training-safety tests, 16 recovery tests and 18 architecture tests pass. Both
task parity scripts pass; Perturb also exercises actual terminal observations and reset masks
on CPU and GPU. PowerShell parsing and task override defaults pass. The shipped ONNX and contract
are unchanged.

At a `3e-5` actor ceiling and KL target 0.005, smoothing weights 0, 0.1 and 0.5 all produced
0/16 quiet-room survivors after 50 updates. Under one hard shot they survived 1/64, 1/64 and 3/64
respectively, with 0/64 settled recoveries. The incumbent survived 43/64 and recovered 40/64.
Iterations 25 of the 0 and 0.5 trials also failed quiet standing. All are rejected. This isolates
smoothing strength as insufficient to prevent degradation, rather than showing that one of these
weights is ready for a longer run.

A follow-up kept smoothing at 0.1, reduced the actor ceiling to `1e-6` and KL target to 0.001.
Both iterations 25 and 50 passed 16/16 quiet survival and settling for 40 seconds. Mean action
change fell from 0.0814 to 0.0050 and 0.0070; no quiet-room steps or rapid replants were recorded.
These smaller-step settings are now the defaults. A 64-world impact pilot at iteration 25 was
42/64 survival and 39/64 recovery, one world below the incumbent on both. The standard 512-world
comparison determines acceptance; improved quiet behavior alone cannot authorize promotion.

**Final 512-world comparison: neither candidate is accepted.** The incumbent survived 59.0%
and recovered 54.1%. The smaller-step iteration 25 survived 56.8% and recovered 52.3%
(paired recovery z = -0.83); iteration 50 survived 52.1% and recovered 37.3% (z = -6.60).
Both lose survival and recovery, so the gate retains the incumbent. At iteration 25,
under-fire action change fell from 0.0795 to 0.0730, but foot sliding rose from 0.367 to
0.389 m/world. Calmer quiet standing has not yet translated into more reliable, realistic
impact recovery. The remaining problem is behavior degradation during objective adaptation;
these experiments do not establish its exact cause. Do not resume the rejected checkpoints or
launch an unattended long run. The smaller defaults limit the damage while future experiments
remain subject to the same acceptance gates.

Evidence: `logs/recovery-safe-{critic,smooth0,smooth01,smooth05,slow}.log`, the matching evaluation
JSON files, and checkpoints in the `2026-09-20_*_recovery_safe_*` run directories.
The final matched evaluations are `logs/recovery-safe-{incumbent,slow-early,slow}-512.json`.

## 2026-09-20: perturb recovery learning corrections

The shipped balance checkpoint remains `p0911i_s1/model_503`. Its historical 92.4% result below
used a **5 kg ball at 6 m/s (30 N.s)**. Today's generated model uses **8 kg at 6 m/s (48 N.s)**.
The historical success rate is not a measurement of the current scene.

The old reward paid for fast swing-foot motion and proximity to the capture point while still
airborne. It did not require a useful landing. Its action-change penalty averaged 30 joints,
making isolated ankle reversals cheap. `grounded_recovery_v2` now rewards grounded support and
settling, penalises sliding and rapid replants, and allows a comfortable staggered stance.
CPU and GPU use one reward kernel. PPO regularises changes in deterministic mean actions and
uses a lower perturb learning-rate ceiling. Changed reward versions reset critic/optimizer
state while retaining the actor seed. See [README.md](README.md#perturb-recovery-objective-september-2026).

Promotion now measures a stable final half-second hold as well as survival; improved recovery
cannot trade away survival on the comparison batch. The curriculum checks settled episode
endings, and overnight sessions advance on a CPU recovery score of at least 85%.

Validation: 16 recovery tests and 18 architecture tests passed, both task parity scripts passed,
and perturb preflight reported no failures or warnings. Perturb reward parity error was below
0.000002. A three-minute GPU run completed 49 iterations without nonfinite-reward reports; a
separate two-iteration run verified the final training integration and same-version resume.

CPU smoke comparison: 32 worlds, seed 17, six seconds, one uniformly targeted shot launched
at 1.5–1.8 s, current 8 kg ball at 6 m/s. Both policies survived all quiet-room trials.
The incumbent survived 16/32 under fire and settled in 15/32. The smoke candidate survived and
settled in 14/32. Mean clipped action change per joint per step fell from 0.0617 to 0.0140 in
the quiet room, and from 0.0769 to 0.0530 under fire. These small samples establish a working
training path and movement smoothing, not improved robustness or visually validated realism.
The candidate fails the survival gate and was **not exported**. New rewards do not alter an
existing ONNX policy; a longer curriculum run and CPU/scene validation are still required.

Local evidence: `logs/recovery-v2-{incumbent,candidate}.json`,
`logs/recovery-v2-training.log`, `logs/recovery-v2-integration.log` and
`logs/mujoco/2026-09-20_11-19-31_recovery_v2_smoke/model_49.pt`.

## Historical handover — 2026-09-11

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

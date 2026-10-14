# Twenty experiments: weight transfer to corrective stepping

Authorized 2026-10-07: proceed autonomously through twenty substantive next steps.
This continues the user's explicit authorization for checks, bounded training and
frozen evaluations. No Arena replacement, C# build, publishing or cleanup.
Current progress: 20/20 complete. The preceding corrective-step pilot remains a negative
result and is not counted again.

Current delivery (2026-10-08): a new learned Perturb recovery preview, checkpoint
f832bab8, replaces the older Stage12 inspection selection. See
PERTURB_PREVIEW_PLAN.md and the latest PERTURB_AUTONOMOUS_PLAN.md result. The scripted
support teacher remains a separate diagnostic branch: 13/16 complete sequences,
with right swing in poses 2-4 failing. Its 16/16 demonstration-export gate is
not a prerequisite for all recovery learning. No successful teacher dataset was
exported. A 100-update repeated-hit PPO continuation from Stage33 passes 8/8
stronger single-hit and 7/8 corrected two-hit cases, versus Stage12's 5/8 and
3/8, with both native/GPU agreement and 16/16 120-second quiet cases. This remains
a one-optimization-seed experimental preview; heavy random Arena hits and learned
corrective stepping are not established. Historical results remain unchanged.

Latest follow-up: the response block completed 20 substantive steps, 400 PPO
updates and 512 new frozen cases (496 admitted, 16 diagnostic exclusions).
No action/target clipping or command-queue defect explains the failed pose.
TD-lambda .995 produces one 8/8 primary candidate, but shifted 4/8-second throws
reduce it to 6/8 versus the parent's 7/8. Reject it and keep f832bab8 in Arena.
Successful recoveries move feet only 2-3 mm; corrective stepping is still absent.
See PERTURB_RESPONSE_PROGRESS.md for the complete ledger and final audit.

Previous follow-up: matched fixed/mixed launch-distance continuations completed
200 PPO updates and 192 frozen cases. Fixed-distance practice regresses; mixed
distance retains the current 8/8 single-hit and 7/8 two-hit results without a
gain. Keep f832bab8 in Arena. The remaining pose fails after the first hit,
before the second throw. See the distance result in PERTURB_AUTONOMOUS_PLAN.md.

## Scientific contract

Keep the exact 33-motor, zero-passive-spring plant, joint/torque limits, active PD
and two-control-frame delay. Start from both Stage33 source policies. A frozen
Stand actor supplies its original targets; a separately identified goal-driven
controller or learned residual may add bounded joint-target offsets. Zero offset
must reproduce the old controller. Do not call a scripted teacher a learned brain,
a commanded step a reactive recovery, or an unloaded falling foot a successful
step. The frozen actor remains part of any composite policy.

Goals describe desired load transfer, lift and placement. They never write the
root state or apply support forces. Reset states are the eight original poses
unless a stage explicitly investigates a different reset curriculum. Quiet with
the residual disabled is an architectural preservation check; quiet with the
residual enabled is a separate behavioral test.

Each numbered stage answers one question with a declared physical or learning
comparison. Code, unit checks, file generation, admission repair and summaries
do not add stages. Negative admitted comparisons count; failed infrastructure
does not. Record both backend cases rather than calling them independent seeds.
Freeze each stage's settings and source snapshots before execution. Do not tune
thresholds retrospectively. Keep actual transitions, optimization examples and
physics-free analyses separate.

## Adaptive sequence

1. Does zero added control preserve quiet Stand and the existing mild-impact
   behavior on both source policies/backends?
2. Can joint-only target changes transfer load to either side at three COM goals?
3. Does changing transfer duration improve the best admitted transfer?
4. Does measured load feedback improve transfer over the same open-loop target?
5. Does equilibrium feedforward improve the target's dynamic support?
6. Can the best transfer lift either complete foot by 5/15/30 mm?
7. Does additional residual authority help lift without disturbing initialization?
8. Can either single-foot support be held for 0.25/1/3 seconds?
9. Can an airborne foot translate forward/backward by 3/6 cm without losing support?
10. Can the foot replant and return to stable bilateral Stand after the commanded
    sequence, compared with a matched no-placement sequence?
11. Can a new residual policy learn weight transfer from original Stand resets,
    compared with zero residual and the diagnostic teacher?
12. Does a coordinated low-dimensional action basis outperform independent joint
    offsets under a matched learning budget?
13. Does the transfer learning result reproduce with independent optimization RNG?
14. Does transfer initialization improve learning lift over a fresh residual?
    If transfer is inadequate, compare original resets with physical teacher-state
    starts and retain the distinction between hold and transfer capability.
15. Does successful-example supervision improve the weakest learned component?
    Use only demonstrated physical examples; if no successful lift exists, use
    measured load-transfer examples and report the narrower question.
16. Can the learned component execute a short step and replant, compared with its
    no-placement control and the diagnostic controller?
17. Does the selected learned component generalize to held-out onset times and
    both feet across all original starting poses?
18. Does it retain 120-second quiet Stand with added control disabled, and stable
    quiet/return behavior with that control enabled and after two commanded skills?
19. Does an observable-state-triggered skill improve recovery after controlled
    5 kg / 1.6 and 2 m/s hits over the original frozen Stand actor?
20. Does the selected reactive composition reproduce across the second parent and
    held-out hit timing/repeated-hit cases? If no reactive composition qualifies,
    quantify its failure against the original actor rather than escalating force.

The next exact protocol depends on the preceding evidence; a failed prerequisite
changes the question to the stated diagnostic branch, not a fabricated success.
No 10 kg / 5 m/s claim or escalation follows without stepping and retained Stand.

## Budgets and acceptance

Initial controlled screens: at most 192 cases per stage, normally 8 poses x both
feet x configurations; 6-20 seconds per case. Learning: at most 200 PPO updates per
arm, 64 worlds x 64 steps, at most three arms per comparison and two optimization
seeds, with a 600-second cap per training arm. Do not launch every maximum budget
unconditionally. Retain compressed physical evidence and stop before disk space
would threaten existing results; never delete previous runs for this batch.

Transfer requires surviving, upright/posture-valid support with sustained target
load sharing; lift requires whole-foot clearance, opposite support and no fall;
placement requires lift, displacement, loaded replant and stable final Stand.
Keep the previous `corrective_step_audit.py` recovery definition unchanged for
impact claims. Goal skills are measured separately and never relabeled as hits.
Both parents, native/GPU agreement, retained quiet Stand and genuine recovered
steps are required for a promotion recommendation. No automatic Arena changes.

## Ledger

1. Zero addition, 64 frozen physical cases: both parents retain 8/8 quiet Stand
   on native and GPU, and the mild +X impact scores remain 6/8 and 7/8. Native
   actions, targets, poses and termination flags exactly match the archived
   baseline. GPU outcomes agree but traces are not bitwise equal; exact GPU
   reproducibility is not claimed. Output: `support-skill-stage01-zero-20261007-01`.
2. Joint-only COM goals 4/8/12 cm, 96 native cases: stable final Stand totals
   31/32, 17/32 and 0/32 respectively. No settled mechanical steps. Four-centimeter
   transfer reaches the load-sharing tolerance in all cases, including one that
   later falls. Thus transfer attainment is distinct from retained Stand.
   Output: `support-skill-stage02-transfer-20261007-02`. The earlier `-01` attempt
   stopped on a metric-call interface error after one rollout; it is not another
   completed stage. The adapter preserves the original metric thresholds.
3. Transfer ramp 0.75/1.5/3 seconds, 96 native cases: final Stand 29/32, 31/32,
   31/32. Slowing the ramp alone does not eliminate the remaining failure. Retain
   1.5 seconds for the next feedback comparison. Output:
   `support-skill-stage03-duration-20261007-01`.

Checkpoint/storage cleanup is separate maintenance and counts as zero stages.
See `docs/STORAGE_RETENTION.md`. No new policy has been promoted.

4. Load-feedback gains 0/1/3 at the 4 cm target: final Stand 31/32, 30/32,
   30/32. All achieve the transfer tolerance transiently; feedback does not fix
   retention. Output: `support-skill-stage04-feedback-20261007-01`.
5. Static equilibrium delta feedforward weights 0/0.5/1 on the 8 cm, 3-second
   transfer: final Stand 21/32, 22/32, 17/32. Half feedforward adds one survivor;
   full feedforward regresses. The smaller 4 cm transfer remains the safer
   precursor. Static support torque feasibility does not establish dynamic
   transfer; candidate IK poses also develop thigh/hand overlap, which the live
   collision solver must resolve. Output: `support-skill-stage05-equilibrium-20261007-01`.

Infrastructure checks (not scientific stages): zero initial residual output,
CPU/GPU zero-addition target equivalence, one finite PPO update (4,096 transitions),
and an unchanged frozen Stand actor passed. No new springs or root support.

6. Corrected exact-height screen (96 native cases): 5/15/30 mm goals retain final
   Stand in 31/32, 20/32 and 2/32 cases. None produces 15 mm of whole-foot
   clearance for 50 ms with valid support/posture. Initial screen `-01` mistakenly
   used a 20 mm template for both smaller goals; it is invalid for the height
   comparison and not counted. Corrected output: `support-skill-stage06-lift-20261007-02`.

Prospective adaptation before Stage 8: no qualifying lift exists, so measure the
physical prerequisite directly. Compare original resets with explicitly declared
IK single-foot reset poses under a joint-only equilibrium teacher, for 0.25/1/3
seconds. This tests hold feasibility, not learned transfer. If that also fails,
Stage 9 investigates the observed thigh/hand overlap using collision-clear arm
poses before asking for airborne placement. Stage 10 then tests entry/return
transitions under the best physically admitted teacher. A change to reset pose or
teacher is an experimental factor, never a claim about the existing Arena brain.

7. Added target bounds 0.25/0.5/0.8 rad during a 30 mm lift attempt retain final
   Stand in 22/32, 2/32 and 1/32 cases. No qualifying whole-foot lifts. Larger
   authority alone is not the missing ingredient. Output:
   `support-skill-stage07-authority-20261007-01`.

8. The initial 13 cm target failed sole geometry admission before any physics.
   An 11 cm COM goal plus 10 cm outward hand clearance yields 16 statically feasible
   target poses, with penetration below 0.1 mm. Using those admitted poses, the
   192-case native comparison separates original and single-foot resets. Starting
   in single support yields qualifying 15 mm / 50 ms support in 13/16 poses per
   parent; longest valid intervals are only 0.017-0.467 seconds, not a sustained
   one-second hold. An immediate command from original Stand yields no qualifying
   lift. This tests a scripted equilibrium controller, not a learned skill or a
   proof that the plant cannot balance. Output: `support-skill-stage08-hold-20261007-02`.
   Arms are cleared prospectively to avoid the diagnosed hand/thigh overlap;
   no collision geometry or spring is changed.

The independent Stage11 transfer-learning branch began while Stage8/9 physical
single-foot diagnostics continue. Its goal remains the demonstrated 4 cm transfer;
no unproven lift has been used as a successful training demonstration.

9. Airborne translation from the admitted lifted reset: forward/backward 3/6 cm
   produces no settled steps. Backward 3 cm retains 11/16 survivors per parent;
   forward 3 cm retains 3/16 and either 6 cm move 2/16. Brief mechanical events
   occur, but these are scripted resets/commands and do not establish learned
   entry or recovery. Output: `support-skill-stage09-air-translation-20261007-01`.
10. Slow entry from original Stand, short single-support hold, and return, with
    no placement versus +/-3 cm: all 96 cases fail final Stand. There is no usable
    full teacher transition yet. Output: `support-skill-stage10-entry-return-20261007-01`.
11. Fresh three-action residual PPO, 100 updates / 409,600 transitions: transfer
    target reached in 4/16 versus 2/16 with zero residual and 16/16 with the scripted
    target. Learned policy retains 16/16 final Stand and 8/8 enabled-quiet Stand;
    teacher retains 15/16. This small transfer gain is not a reliable skill or
    any foot lift. Outputs: `support-skill-stage11-transfer-learning-20261007-01`
    and `support-skill-stage11-controls-20261007-01`.

Prospective Stage14 diagnostic: transfer is still inadequate, so compare fresh
full-joint PPO from original resets against explicitly prepared single-support
resets. Evaluate both reset types for both learners. The lift-phase load goal
must be 100 percent on the support foot, not the two-foot COM-derived fraction;
keeping a nonzero load target on an airborne foot would make the objective
inconsistent. Declare this goal correction before training, shared across both
arms. This is a hold-curriculum diagnostic, not a claim that reset success teaches
entry or proves natural recovery.

12. Independent 33-joint residual, matched common network/critic initialization,
    100-update budget and initial motor-space exploration RMS: 2/16 transfer
    passes, versus 4/16 coordinated residual and 2/16 zero control. Both learned
    representations preserve 16/16 final Stand and 8/8 enabled quiet. No lift.
    Output: `support-skill-stage12-action-comparison-20261007-01`.
13. Fresh independent optimization seed reproduces 4/16 transfer, 16/16 final
    Stand and 8/8 enabled quiet. This replicates a small improvement, not a robust
    transfer skill. Output: `support-skill-stage13-seed-replication-20261007-01`.
14. Two 100-update full-joint runs compare original versus prepared-pose resets
    under the corrected single-support load objective. Both yield zero qualifying
    lifts from either evaluation reset. Original-reset training retains 6/8 final
    enabled quiet (5/8 whole interval); prepared-reset training retains 5/8. The
    prepared-reset learner also falls in 6/16 ordinary-start cases. Both are
    rejected. Prepared states were geometrically admitted IK poses with zero
    velocity, not dynamically settled or saved moving teacher states. Outputs:
    `support-skill-stage14-original-20261007-01` and
    `support-skill-stage14-prepared-20261007-01`.

Stage15 therefore uses only successful weight-transfer demonstrations. No lift
example is mislabeled successful, and the failed reset branch is not promoted.

15. Matched continuation versus successful-example supervision (100 updates each)
    reaches 6/16 versus 7/16 transfer passes. Both preserve 16/16 final Stand and
    8/8 enabled quiet. Supervision uses 9,000 physical examples from 15 successful
    teacher cases, presented 1,228,800 times; those are not new physics transitions.
    The one-case difference is not a demonstrated robust advantage. Both initial
    learner states are tensor-identical. Output: `support-skill-stage15-supervision-20261007-01`.
    The original finalizer rejected an edit to an unused future reactive evaluator;
    independent review verified every source snapshot, all used sources/inputs,
    both complete arms and saved trace hashes. The unused-source change is retained
    explicitly in the final verification record rather than hidden or rewritten.
16. Both learned transfer components preserve Stand through all 96 frozen
    lift/placement-command cases, but produce zero qualifying foot lifts or steps.
    The matched scripted teacher retains only 1/48 final Stand cases and also has
    zero qualifying steps. This probes new goals outside transfer training; it
    does not demonstrate that stepping was learned or is impossible. Outputs:
    `support-skill-stage16-replant-20261007-01` and `support-skill-stage16-teacher-20261007-01`.
17. Held-out transfer onset at 3/9 seconds yields 6/16 and 5/16 transfer passes,
    with 16/16 final and full-interval Stand in each condition. Native/GPU counts
    agree. The eight poses and both sides are known; onset time is held out.
    Output: `support-skill-stage17-heldout-onsets-20261007-01`.

18. All 64 quiet cases retain full-interval Stand for 120 seconds, across both
    parents, both backends, and enabled/disabled added control. All 64 two-command
    cases retain final and full-interval Stand through 30 seconds. Parent 301027
    reaches the transfer criterion in 7/16 first commands and 4/16 second commands;
    parent 301028 reaches 4/16 for each. Both commands plus final Stand succeed in
    only 4/16 per parent on each backend. No qualifying lift or settled step.
    Disabled control is architectural preservation; enabled zero-shift quiet uses
    a goal-dependent action basis and is not proof of a new independent standing
    controller. Output: `support-skill-stage18-retention-20261007-02`. Independent
    per-command review: `support-skill-batch-review-20261007-01/retention-review.json`.

19. State-triggered transfer after a 5 kg chest impact at 5 seconds reduces mild
    1.6 m/s recoveries from 6/8 to 4/8 on parent 301027. At 2 m/s both controllers
    recover in 0/8 cases. No measured or recovered steps. Native/GPU outcome,
    contact-time and impulse admission passes; native first contacts are Chest.
    This composition is rejected. It invokes a transfer component, not a trained
    stepping policy. Output: `support-skill-stage19-reactive-20261007-01`.
20. Reusing that transfer component with parent 301028 gives source versus
    triggered recoveries of 6/8 versus 6/8 at onset 1 second, 7/8 versus 8/8 at
    3 seconds, and 8/8 versus 8/8 at 9 seconds. Repeated hits at 1 and 7 seconds
    retain 6/8 for both. Native/GPU admission passes in all eight comparisons.
    All surviving repeated trials receive both Chest-first encounters; the two
    failed trials fall before the second hit and are not successful two-hit tests.
    No measured or recovered steps in any condition. The isolated one-case gain
    does not resolve the first parent's regression. This is cross-anchor reuse,
    not a second independently trained residual. No Arena promotion. Output:
    `support-skill-stage20-replication-20261007-01`.

Stage18 infrastructure incident: attempt `support-skill-stage18-retention-20261007-01`
completed its first eight native 120-second quiet cases, then Windows recorded an
access violation in `nvcuda64.dll` during GPU evaluation. This incomplete attempt
does not count as a stage. The unchanged protocol completed successfully as `-02`
after the other GPU evaluations finished; no policy retraining was needed.

## Batch accounting

All twenty substantive comparisons are complete: 1,848 accepted frozen physical
cases, 700 PPO updates / 2,867,200 new training transitions, and 1,228,800 offline
supervision presentations. Case counts include both backends and reuse of eight
poses; they are not 1,848 independently trained humanoids. Seven learning arms
ran 100 updates each. The separate infrastructure smoke update (4,096 transitions)
and incomplete/invalid attempts are excluded from these scientific totals.

The independent reconciliation verifies recorded input/output hashes, preserved
source snapshots, training accounting, finite states, target limits and sampled
whole-foot geometry. Stage15's unused-source finalization exception remains
explicitly recorded. Final record:
`logs/mimickit-training/support-skill-batch-review-20261007-01/report.json`.
No C# build, Arena replacement, policy promotion or further training was performed
after these comparisons. The separately authorized follow-up below records the
subsequent foundation work and its own accounting.

## Decision and next foundation work

This batch has not produced a learned corrective step. A small learned change
in load sharing is reproducible, but reliable single-foot support, entry,
placement and return remain missing. The impact composition commands lateral
transfer only; it was never trained to choose a recovery direction or foot
placement from an impact. Its failure cannot establish that the physical dummy
is incapable of stepping, nor that additional steps of this same training would
solve the missing task definition.

The next development gate is a physically demonstrated, closed-loop transition
from ordinary Stand into at least one second of valid single-foot support and
back, with collision-clear arms, consistent load/foot goals, and the same motor
limits and delay. Private IK/static equilibrium alone failed to demonstrate that
transition. Assess the existing `dynamic_support_teacher.py` and
`full_body_teacher.py` implementations for this goal and current control contract;
do not create another teacher without first accounting for their prior results.
Compare that feedback-controlled approach with the present open-loop equilibrium
targets before generating learning examples. Only admit
examples that actually hold support and return; an IK reset is not a successful
entry example. Then compare learning against that teacher with matched budgets,
including quiet retention. Foot placement and reactive impact timing follow that
prerequisite. No longer impact run or spring change is justified by this batch.

Audit the goal/action mapping before that comparison: this prototype regenerates
its action basis for each requested goal. A successful zero-shift quiet case
therefore does not alone prove harmless behavior under every enabled goal, and
untrained lift/placement commands are explicitly out-of-distribution probes.

Checkpoint retention is separate from these capability gates. Retain the Arena
brain, the frozen comparison anchors, and selected/final candidate policies with
their exact restoration inputs. Short new training arms in this batch save
initial/final states rather than periodic training histories. Most intermediate
brains from inexpensive, rejected runs add little value; compact settings,
metrics and targeted physical evidence explain the result. Future evidence
recording should be sized to the question, not collect all observations and
controller substeps by default. The separately authorized storage cleanup is
documented in `docs/STORAGE_RETENTION.md`.

The seven accepted learning arms total 1,190.5 seconds of reported training time
(individual arms 144.0-251.7 seconds); evaluation and preparation time are separate.
For subsequent work, gate expensive long-retention and broad impact suites behind
an actual improvement in the target prerequisite skill. A rejected transfer/lift
candidate does not need repeated promotion-scale validation. Keep a minimal
matched baseline and enough diagnostic cases to explain its failure.

## Single-support feedback follow-up (authorized 2026-10-07)

The user authorized proceeding with the next foundation priority. Earlier
authorization for checks and bounded experiments remains in scope. This follow-up
has separate accounting from the completed twenty comparisons.

Read-only review found that `DynamicSupportTeacher` and `FullBodyTeacher` retain
the old 30-motor/recovery-gate queue contract (including validity at index 60).
The recorded dynamic teacher also failed ordinary gravity support, including
direct-actuation diagnostics; see AMP_RECOVERY.md, Contact and torque teacher
investigation. Replaying that controller unchanged is not justified.

Reuse `PredictiveStandController`'s one-Hz generalized acceleration objective,
root/motor weights, bounded inverse-contact solve and two forward refinements.
Its original prediction replays delayed torques. The isolated new adapter instead
replays both committed 67-value position/velocity/valid entries with the current
33-motor PD calculation at each private physics substep. Convert bounded planned
torques to joint targets at the predicted application state. Keep the physical
gain, damping, delay, limits and zero passive springs unchanged. No live root
writes, direct-torque bypass or applied support forces.

Prospective native pilot: parent 301027, original poses 0/1, both swing feet.
First check private queue prediction against live evolution and verify isolation.
Compare four-second source/predictive quiet holds (four physical cases total).
Only if both predictive cases retain whole-interval Stand, compare equilibrium
feedforward and predictive feedback on four entry/hold/return cases each, capped
at 24 seconds per case. Maximum twelve scientific physical cases, zero PPO.

Both entry controllers use the same admitted target geometry: arms cleared
outward 10 cm, 11 cm lateral COM goal, 2 cm requested lift. Use one second of
source warmup and one second of controller blending; clear arms over 1.5 seconds,
then transfer over four seconds. Lift only after support share >=90 percent,
root horizontal speed <=0.08 m/s and upright valid support persist 0.1 seconds.
Transfer times out after six seconds. Lift takes one second. Hold requires at
least 15 mm complete-foot clearance, swing load <5 N, support load >=30 N,
ordinary upright/posture gates and support-foot drift <=2 cm for one consecutive
second; timeout after three seconds. Replant, return over four seconds, and
finish with two seconds of ordinary Stand. Timeout/fall is not a successful skill.
The teacher remains a scripted controller, never a learned recovery brain.

Snapshot loaded project dependencies and record hashes/settings plus compact
physical traces. No observations, optimizer histories, substep trace archive,
long retention sweep, impact suite, training or Arena change unless this gate
produces usable transitions. If quiet takeover fails, diagnose that mismatch
before claiming or running the single-support comparison.

Pilot `single-support-feedback-20261007-02` completed twelve cases. Both quiet
controllers retain 2/2 full-interval Stand. Equilibrium entry falls in all four
cases; feedback retains upright Stand but reaches the unloading timeout in all
four. Its measured COM is within approximately 2 mm of the requested transfer
target, yet support share stays about 74-76 percent. It is not a failed COM
transfer. Private queue prediction is exact in qpos/qvel, and inverse target
conversion error is below 2e-16. The `-01` setup attempt stopped before scientific
cases on an incorrectly bilateral admission check for a lifted-foot pose; repaired
to use the designated support foot. It is not counted as a comparison.

Prospective follow-up isolates the release rule. Saved end states place the
capture point 17.6-27.6 mm inside actual stance-foot floor-contact polygons.
Requiring 90 percent unloading while commanding both feet flat may prevent the
release needed to change load distribution. Compare the same two controllers
with a release condition of >=5 mm capture-point contact-polygon margin and
>=2/3 load on the support foot, retaining the same low-speed/upright gates and
0.1-second dwell. Actual success still requires <5 N swing load, >=15 mm full-foot
clearance, <=2 cm stance drift, one continuous second of supported hold and
two seconds of returned Stand. No success threshold is relaxed. Eight new native
entry cases only; reference the completed quiet evidence rather than repeat it.
Record the requested pose and capture margin in compact traces. This release
rule is a local support heuristic, not a guarantee of dynamic balance.

Release comparison `single-support-feedback-20261007-03` completed eight entry
cases. Feedback enters the lift/hold phase in all four and briefly transfers all
load to the support foot, but none meets the full-foot one-second hold criterion.
Three hold attempts show only millimetres of clearance; one brief larger lift
occurs with loss of support/posture. The equilibrium control again falls before
release. Do not label a commanded hold phase as a physically successful hold.

Next bounded objective comparison: preserve the capture release rule and add
explicit COM/foot acceleration feedback to the original root/joint objective.
Reuse `DynamicSupportTeacher.task_jacobians` and the existing rotation error
helper without its obsolete transport/phase code. COM uses one Hz/weight 10;
both foot positions use two Hz/weight 8 and orientations weight 1; include the
Jacobian velocity-bias term. Retain motor bounds, two refinements, target geometry
and actual success gates. Run four quiet source/feedback cases, then at most four
entry cases only if quiet passes. Reuse the completed equilibrium comparison.
This tests explicit whole-body coordination, not a change to the physical plant.

Task-space comparison `single-support-feedback-20261007-04`: both quiet feedback
cases pass, but all four entry cases physically fail and none achieves a valid
hold. Retain this negative result; the extra task tracking is not an improvement.

Final bounded diagnostic for this follow-up: compare the pose and task objectives
from the same admitted lifted reset, four cases per objective, four seconds each.
All 33 delayed target controls and physical limits remain unchanged. Only the
initial qpos/qvel differs; the teacher is active immediately, still through the
normal initially invalid two-command queue. Require one full second of actual
single-foot support under the same clearance/load/posture/drift gates and no
physical failure. This isolates hold control from release/entry and cannot count
as successful entry, return, learning or Arena readiness. No training follows
unless an ordinary-start entry/hold/return transition is actually demonstrated.

### Follow-up result and decision

Prepared-reset diagnostics `single-support-feedback-20261007-05` / `-06` fail all
four cases per objective. Pose feedback achieves only 0.250-0.350 seconds of valid
single support; task feedback achieves 0.017 seconds. All eight physically fail.
Thus release timing alone does not explain the missing skill: neither candidate
controller demonstrates a one-second hold even from the admitted lifted pose.
Static geometric/equilibrium feasibility is not closed-loop dynamic feasibility.

The follow-up completed four focused comparisons / 36 native physical cases,
59,056 private prediction substeps, zero PPO updates and no admitted teaching
examples. Pose-feedback transfer stays upright in 4/4 pilot cases and reaches
the requested COM position within approximately 2 mm; that is a scripted
two-foot transfer result, not learned single support. No full entry/hold/return
passes. Quiet feedback passes both small cases for each objective. No GPU sweep,
long retention, impact training, spring change, C# build or Arena change.

Independent reconciliation verifies 42 recorded output hashes, input identities,
source snapshots, trace timing/finite states, target limits and sampled whole-foot
geometry. The reused quiet controller class is AST-identical. The first completed
pilot omitted its late-imported static support helper from its own source list;
the helper's exact hash is recovered from matching prior-batch and follow-up
snapshots, and that coverage note is retained. Original per-run `final_stand`
also excluded skill timeouts: the independent review separates physical Stand
from sequence completion and correctly counts all four stable transfer cases.
The runner now makes that distinction prospectively; historical reports remain
unchanged. Review: `logs/mimickit-training/single-support-feedback-review-20261007-01/report.json`.

The next engineering question is whether optimizing the actual delayed PD-target
trajectory over a short horizon can stabilize single support. Current feedback
optimizes instantaneous bounded torque at the predicted application state, then
converts it to a target; it does not optimize the later substeps during which PD
torque changes. This is an untested hypothesis, not an established cause. First
compare predicted versus realized support evolution at release using the saved
states and unchanged actuator dynamics. Any controller redesign should pass the
prepared one-second hold and then ordinary-start entry/return before collecting
training examples. Do not restart reward or PPO parameter sweeps on these failures.

## Delayed-target response audit (authorized 2026-10-07)

Proceed with four small native ordinary-start pose-feedback cases, reusing the
prior quiet/equilibrium evidence. At issue times 2/6.5/7/7.5/8 seconds, fork private
copies after replaying the two committed target commands. Compare one subsequent
control interval under the true substep PD feedback against privately holding
its initial torque constant. The constant-torque branch is counterfactual and
never controls the live dummy. Match the true-PD forecast to the live state after
three control intervals, with tolerances 3e-6 qpos / 3e-5 qvel. Verify live-state
and queue isolation. Record torque change and weighted coordinate-velocity-change
error; neither a discrepancy nor a better private cost proves physical recovery.
No training, new checkpoints, impacts or long retention. Select any next controller
correction from this audit rather than another reward/gain sweep.

Audit `support-target-response-20261007-01` completes all twenty sampled forecasts
over four live cases. Maximum live-state error is 3.1e-8 qpos / 1.1e-6 qvel;
the queue and local simulation transport pass. Across one held-target interval,
motor torque changes by at most 1.17 percent of its limit (median RMS 0.117 percent),
yet the resulting weighted coordinate-velocity-change error is consistently
larger than the private constant-torque approximation. At sampled release/hold
times it is about 4.6-10.3 for actual PD versus 1.4-8.6 for constant torque.
These are controller-objective units, not recovery scores or evidence of a sole
failure cause. No command or body state was changed by the audit.

Prospective correction: retain the pose controller's initial proposal, then use
finite target derivatives (1e-4 rad) of an actual four-substep PD rollout to make
one bounded target-space correction (at most 0.05 rad per motor). Accept only an
actual-rollout cost decrease after backtracking. Preserve root/joint tracking,
physical gains, torque/joint bounds, two-command prediction and all support gates.
First run four prepared-pose cases for at most four simulated seconds. Only a
successful prepared hold justifies subsequent quiet/ordinary-entry comparisons.
This optimizes one control interval; it is not a long-horizon balance planner.

### Actual-interval correction result and decision

`single-support-interval-20261007-01` completes four prepared-pose cases. The
controller accepts 839 cost-reducing corrections over 892 queries, but none
achieves the required one-second single-foot hold. Valid support lasts 0.150,
0.250, 0.383 and 0.367 seconds. Both left-swing cases physically fail within four
seconds. Both right-swing cases survive the four-second cap, but finish with
7.9/16.6 cm stance-foot drift and no final Stand. This is not a clearance-threshold
near miss or a successful recovery. Reducing a local acceleration objective does
not establish sustained balance.

The delayed-response audit and bounded correction total eight native physical
cases, twenty sampled forecasts and 159,452 private physics substeps, excluding
the two small transport checks. Zero training transitions, PPO updates or new
brains. The correction fails the prepared-pose gate, so ordinary-start trials,
quiet-retention claims for this correction, teaching examples and impact training
do not follow. Arena selections retain their prior verified bundles.

Independent saved-evidence reconciliation verifies fifteen unique output hashes,
153 source snapshot entries, input identities, trace timing/finite states, motor
target bounds, sampled whole-foot geometry and reported support durations. The
audit trajectory is not bitwise identical to the earlier `-03` pose run: its first
command difference is frame 85, before the first audit probe at frame 120. The
intervening task-objective extraction changed floating-point expression grouping.
Thus prior quiet evidence is reused evidence, not a fresh exact reproduction.
The twenty within-run forecast comparisons and live-state/queue isolation checks
remain valid. Each run retains its own source snapshot; the runner subsequently
gained interval-controller selection and reporting.

Review: `logs/mimickit-training/support-target-response-review-20261007-01/report.json`.

Next priority: identify the first loss of support in the saved prepared traces
(foot recontact, stance slip, posture or COM/velocity divergence), then design
one intervention targeting that event. A longer-horizon contact/balance objective
is a candidate, not a proven fix. Retain the one-second prepared hold followed by
ordinary-start entry/hold/return gates before generating training examples. These
results do not justify another reward sweep, spring change or longer PPO run.

## Saved support-loss diagnosis (authorized 2026-10-07)

Inspect the four actual-interval prepared traces and four prior pose-feedback
prepared traces without advancing physics. Reconstruct body orientation and
height from saved qpos, root speed from qvel, and COM velocity from the kinematic
Jacobian. Preserve recorded foot loads and support flags; recreated contact forces
cannot replace the original solver output. Separate startup loss during the two
empty queue intervals from later support loss. Report first threshold crossings,
all valid support segments, and torso/joint/command state at the posture crossing.
Use delayed issued targets only for explicitly labelled endpoint PD estimates,
not claims about recorded substep torques. Check input hashes and reproduce the
recorded support durations before choosing a narrowly targeted intervention.

Saved-state diagnosis finds an earlier dynamic warning than torso tilt: the
COM/velocity capture-point heuristic first leaves the actual stance contact hull
at 0.050 s in all eight saved cases. The cold prepared reset begins with two
invalid command slots (33.336 ms of zero motor control). Clearance first fails at
0.033 s, temporarily recovers, then chest tilt fails at 0.50-0.57 s before swing
foot recontact and stance drift. A first crossing is not a causal proof.

Prospective reset-isolation comparison: four four-second prepared cases with the
same interval controller and poses, but initialize both committed queue slots to
the already admitted static-support targets q + normalized_torque/4. Preserve
the two-command delay for every newly issued command, physical PD gains, motor
bounds and success gates. Require initialized targets within physical limits and
the same source-relative one-radian envelope. No settling steps or state edits
after reset. Recompute initial contact measurements under the initialized motor
torque without advancing time. This is a diagnostic reset, not ordinary entry or
a production configuration. Compare with the completed cold-queue cases; no
training or additional controller objective is introduced in this comparison.

The queue-initialization attempt `single-support-initial-queue-20261007-01`
stopped at its preflight bound check, before scientific physical cases. The
admitted static torque solution requires stance-ankle target angles outside
[-0.2, 0.2] rad. A necessary admission check is missing: static support must be
feasible using torque achievable through the actual bounded position target,
not just the actuator's full torque range. At zero velocity that interval is
clip(4*(target_limit-current_angle), -1, 1). Initial constrained LP checks find
all four lifted poses infeasible under these bounds.

Next bounded, non-stepping check: compare unrestricted motor torque, all actual
target bounds, stance-ankle-only bounds and all bounds except the stance ankle.
Keep the same geometry, friction rays, gravity and passive-joint tolerance.
This separates an unlucky original torque solution from infeasibility under the
command contract. Add the missing target-bound admission to future teacher goals
if this check confirms it; retain existing pose/velocity/controller limits.

### Result: missing actuator-command feasibility check

The sixteen-way check (four poses x four constraint variants), reproduced using
the shared admission helper, finds the same pattern in all four poses:
unrestricted torque support is feasible; actual target-PD bounds are infeasible;
the stance-ankle-roll bound alone makes support infeasible; releasing only that
bound restores feasibility. This rules out merely choosing an unlucky original
LP solution under this existing quasi-static model. It does not prove that no
dynamic recovery trajectory exists, or that every earlier failure has one cause.

At the requested hold pose, stance ankle roll is approximately +/-0.178-0.180 rad
with a +/-0.2 rad target limit. The original static solution needs 30.5-37.5 percent
normalized ankle torque in the restricted direction, while legal position targets
provide only 8.0-9.0 percent at zero velocity. Its requested target would be about
-0.257 rad on the right ankle or +0.271 rad on the left. Torque availability near
a joint's target limit is therefore much smaller than full actuator authority.
Earlier geometric/full-torque admission of these four goals did not establish
feasibility through the actual target controller; that admission is superseded.

`static_support.py` now supports explicit normalized motor bounds and exposes
`target_motor_bounds` / `solve_target_support` for the zero-velocity target-PD
envelope. `run_single_support_feedback.py` applies that check before accepting
teacher targets. Mechanical limits, gains, command delay and success thresholds
are unchanged. The source-relative action envelope may narrow the available
torque further; this check is necessary, not sufficient for dynamic admission.

Saved-state timing for the interval controller: initial clearance loss at 0.033 s;
after temporary recovery, left-swing clearance fails again at 0.317/0.417 s;
chest tilt fails at 0.500 s left / 0.567 s right; actual swing-foot recontact is
0.650-0.783 s; excessive stance drift is 0.817-1.083 s. The right-swing cases
lose chest posture before clearance. This distinguishes insufficient clearance,
posture loss, actual recontact and later slipping rather than calling all four
events the first failure. The capture-point estimate is an earlier warning, not
a success gate or causal proof.

Evidence: `logs/mimickit-training/support-loss-diagnosis-20261007-02/report.json`
reconstructs eight archived cases without stepping; all recorded support durations
match. `logs/mimickit-training/support-target-bound-admission-20261007-02/report.json`
records the constraint-isolation checks and helper verification with input/source
hashes. The initial-queue experiment produced zero scientific physical cases:
its bound guard rejected setup, with only the small transport check completed.
No new learning, admitted demonstrations, Arena replacement or long runs.

Next: redesign the single-support goal geometry under the actual target-PD torque
bounds, preserving ankle margin and natural posture. Require both left/right
goals to pass this static screen, then test the normal delayed prepared hold and
ordinary-start entry/hold/return. Do not widen joint limits, increase stiffness,
or run another controller/PPO sweep merely to compensate for rejected goals.

## Feasible support geometry redesign (authorized 2026-10-07)

First screen a bounded geometry grid without advancing physics: lateral COM
offsets 8/10/11/12 cm and whole-body leans 0/4/8/12 degrees toward the stance foot,
for original poses 0/1 and both feet. Preserve original foot locations and sole
orientation targets, 2 cm lift, current cleared-arm pose and all physical limits.
Leaning shifts upper-body mass toward support while potentially reducing the
pelvis translation and ankle roll needed for the same COM position. The lean is
an IK goal only, never an applied root actuator or a live state correction.

Admit candidates only with original upright/height/posture gates, no penetration
above 0.2 mm, foot fit error <=1 mm, COM fit error <=1 mm, full swing-foot clearance
>=18 mm, stance-ankle angle margin >=0.04 rad and feasible target-PD static support.
Require all four lifted cases to pass for a common geometry. Rank qualifying
geometries by largest worst-case stance-ankle angle margin, then lower lean.
Check the matching bilateral transfer goal and source/cleared goals separately.
If none qualify, diagnose geometry constraints before expanding the search.

For the selected admitted geometry, first run four four-second prepared holds
with the current interval controller and the normal cold two-command queue. If
the unchanged one-second hold gate passes, verify quiet takeover before running
ordinary-start entry/hold/return. Save compact goals/provenance and physical traces;
no PPO, Arena replacement, spring/gain change or extra checkpoints in this screen.

Geometry screen `support-geometry-screen-20261007-01`: 64 private IK/static
cases, three common settings pass all four lifted cases (10/11/12 cm shifts with
12-degree lean). The declared ranking selects 10 cm / 12 degrees, with minimum
stance-ankle angle margin 0.11875 rad, 19.9 mm whole-foot clearance and static
peak torque 54-59 percent. Pelvis tilt is 13.26-13.62 degrees, within the original
15-degree gate but with limited angular margin. Matching transfer/reference/
cleared static targets pass before the physical pilot. Root-orientation goal
interpolation is now normalized quaternion interpolation; live root control is
still absent. Prepared targets match saved admitted geometry within 2e-6.

Cold-queue pilot `single-support-feasible-geometry-20261007-01`: all four cases
physically fail; valid support lasts only one control interval (0.016668 s).
The static correction removes one demonstrated infeasibility, but does not
establish closed-loop single support. Do not progress to ordinary entry or PPO.

Next bounded isolation reuses the previously specified initial-queue diagnostic,
now that support targets are legal: four matching four-second prepared cases,
initialize both committed slots from admitted equilibrium targets, retain the
same delayed interval controller and strict gates. No parameter or geometry
changes. Compare early support loss with the cold queue to determine whether
the initial unpowered interval explains this failure. A warm-reset result alone
cannot count as learned entry, normal-reset success or Arena readiness.

### Geometry redesign result and decision

Warm-queue comparison `single-support-feasible-geometry-20261007-02` produces
2.134/2.300 s valid left-swing holds and 1.583/1.567 s right-swing holds. All four
exceed the unchanged one-second support criterion, using real ground contacts,
full-foot clearance, posture, load and stance-drift gates. However, three cases
subsequently physically fail within the four-second trial. Only pose 1 / left
swing passes the complete prepared-hold gate (one-second hold plus no physical
failure). Do not report this as 4/4 admitted holds or learned stepping.

This is a material improvement in diagnostic physical support: the original
0.15-0.38 s holds became 1.57-2.30 s with feasible geometry AND loaded initial
commands. The matched new-geometry comparison isolates initial motor history:
cold queue 0/4 complete passes, 0.016668 s holds, falls at 1.28-1.33 s; warm queue
1/4 complete passes. Neither result demonstrates entry from ordinary standing.
The preloaded commands obey the same joint limits and source-relative envelope
(maximum source offset 0.488-0.523 rad). All subsequent commands retain the normal
two-frame delay. No external support, root control, torque override or gain change.

Independent reconciliation checks fourteen output hashes, 211 source snapshot
entries, input identities, identical requested hold poses between queue variants,
finite/timed traces, motor bounds, sampled complete-foot geometry and every
reported support duration. Quaternion interpolation also passes finite/unit,
endpoint and sign-equivalence checks for all four selected poses. Evidence:
`logs/mimickit-training/support-geometry-review-20261007-01/report.json`.

This follow-up contains 64 private geometry cases and eight scientific physical
cases, with 190,452 private controller prediction substeps (excluding two small
transport checks). Zero learning transitions/PPO updates and no Arena change.
Physical tests are frozen scripted-controller diagnostics, not learned behavior.

Next priority: diagnose and stabilize the later warm-hold failure with the new
feasible geometry, requiring all four cases to pass the complete prepared trial.
Then verify quiet takeover and real entry/one-second hold/replant/return. The cold
prepared reset is a separate startup stressor; entering from an already standing
controller has an existing motor history and must be evaluated explicitly.
No training examples or perturbation escalation are admitted by the current result.

## Warm-hold COM feedback comparison (authorized 2026-10-07)

Saved-state diagnosis `support-warm-loss-20261007-01` finds outward COM drift
before posture/recontact. At one second pelvis lateral errors are approximately
4/4/6.7/7.0 mm, while COM errors are 11.6/11.6/20.3/21.4 mm. Capture-margin loss
first occurs around 0.90/0.95 s for left swing and 0.42/0.38 s for right swing.
Chest posture then fails at 2.32 s left and 1.58-1.60 s right. This indicates
that root-position tracking is not maintaining whole-body balance; it does not
prove a unique failure cause.

Bounded correction: add explicit COM acceleration feedback (one Hz, weight 10,
finite-difference Jacobian velocity-bias term) to the pose objective in both
the initial target proposal and actual-interval refinement. Reuse the existing
task-space COM rows without its foot tracking terms. Keep 10 cm shift / 12-degree
lean, loaded initial support commands, original motor limits/gains and two-frame
delay. First compare four prepared cases for four seconds against the saved
matched baseline. Require all four complete prepared-hold gates before quiet
takeover and ordinary-start entry/hold/return; no training on a partial result.

COM-only comparison `single-support-com-feedback-20261007-01` holds both
left-swing cases continuously for all four seconds; neither loses the capture
margin, clearance or posture. Both right-swing cases regress to 0.717-0.767 s
valid holds. Saved-state diagnosis `support-com-loss-20261007-01` shows their
clearance fails first at 0.733/0.783 s, capture margin later at 0.883/1.033 s,
and chest posture at 1.633/1.733 s. Thus the added COM task alone does not preserve
the lifted-foot geometry. It passes only 2/4 complete prepared trials.

Prospective targeted extension: use the existing task-space objective (COM
one Hz/weight 10, both feet positions two Hz/weight 8, orientations two Hz/weight 1,
all Jacobian velocity-bias terms) inside the same actual-interval refinement.
Compare exactly four matching warm prepared cases, four seconds each. Keep the
geometry, initial support targets, physical limits, gains, delay and success gates.
This revisits foot coordination on now-feasible geometry; the older task-space
failure used infeasible goals and no actual-interval target refinement. No
ordinary-start or learning comparison unless all four complete holds pass.

Full-task comparison `single-support-task-feedback-20261007-01` regresses:
0/4 complete holds, 0.917-1.200 s valid support and four physical failures.
Do not promote that extension or sweep its weights.

Additional saved-geometry inspection finds an asymmetry in available balance
margin: the selected 10 cm goal starts only 6.9-7.9 mm inside the stance contact
hull for right swing, versus 15.2-16.2 mm for left swing. COM-only control keeps
left-swing COM errors near 1.2 mm, but right-swing drift consumes that smaller
margin before posture failure. The already statically admitted 12 cm / 12-degree
goal provides 26.9-36.2 mm initial contact-hull margin, while retaining >=0.084 rad
stance-ankle angle margin and passing the same target-torque bounds.

Prospective geometry refinement: re-rank only the three previously admitted
common geometries by maximum worst initial COM-to-contact-hull margin. Preserve
the original geometry screen and its selection history. Record the derived
selection with input hashes, then run four matching warm prepared cases with
COM-only interval feedback (not the regressing foot-task extension), four seconds
each. No new geometry search, controller gain or success-threshold change.

Margin comparison `single-support-com-margin-20261007-01` passes all four
prepared cases: each sustains valid single support for the full 4.00032 s with
no physical failure. The selected geometry is now 12 cm / 12-degree lean and
minimum initial contact-hull margin 26.88 mm, using COM-only interval feedback.
This completes the small prepared-hold gate, not entry or learned recovery.

Proceed to four fresh quiet cases (two source, two feedback) with this controller;
only if both feedback cases retain full-interval Stand, run four ordinary-start
entry/hold/replant/return cases with the capture release rule. No prepared reset
or preloaded queue is used in these tests. Omit an unrelated equilibrium-only
comparison; this is a gate on the selected feedback controller. Preserve the
24-second entry cap and original one-second hold/two-second returned-Stand gates.

### Result: four complete ordinary-start support sequences

`single-support-com-entry-20261007-01` passes both source quiet cases and both
fresh COM-feedback quiet cases for their full four-second intervals. The four
ordinary-start cases then all complete transfer, lift, one-second hold, loaded
replant, return and two seconds of original Stand. Left-swing valid support lasts
1.50012 s in both poses; right-swing support lasts 1.466784 s in both poses.
All four finish with `sequence_complete`, `entry_hold_return=true` and
`final_stand=true`. No prepared reset, initialized support queue, external force
or live root-state write is used during this ordinary-start comparison.

The working scripted configuration is: 12 cm COM shift, 12-degree stance-side
lean, 2 cm requested lift, cleared arms by 10 cm, capture-based release, and
`interval_com` feedback. The COM objective is one Hz / weight 10 with velocity
bias; it augments original root/joint tracking and actual four-substep target
refinement. Physical PD gains, two-command delay, joint/torque limits, source
policy, action envelope and success gates remain unchanged. The full foot-task
extension remains a recorded negative result and is not the selected controller.

Independent reconciliation verifies twenty physical cases across this follow-up,
including twelve prepared cases and four quiet plus four ordinary-entry cases.
It checks 28 output hashes, 308 source snapshot entries, input identities, trace
timing/finite states, motor target limits, sampled full-foot geometry and upper
posture, all support durations, and the two-second final Stand/sequence gates.
Private controller simulation totals 1,009,892 substeps, excluding four transport
checks (each eight private/eight live substeps). Zero training transitions or PPO
updates. Arena selections remain unchanged. Review:
`logs/mimickit-training/support-com-review-20261007-01/report.json`.

This establishes a physical scripted teacher skill in the four small pilot cases,
not a learned recovery brain, forward corrective step or impact robustness. The
next priority is frozen replication across all eight original starting poses and
both feet, with per-goal target-PD admission and the same success criteria. Once
coverage passes, collect compact demonstrated entry/hold/replant/return examples
and validate the learner's observation/goal/action contract before a bounded
training comparison. Do not skip directly to stronger perturbations.

## Full-pose support replication and dataset contract (authorized 2026-10-07)

Freeze the successful 12 cm / 12-degree, capture-release, `interval_com` teacher
with source actor 301027. Screen all eight original poses and both feet using
the same target-PD static, posture, fit, penetration, clearance and ankle-margin
criteria. Run eight four-second source quiet cases and eight feedback quiet
cases. Only if every quiet case passes, run sixteen ordinary-start sequences,
capped at 24 seconds each. Keep all motor limits, gains, delay, reference poses,
one-second support and two-second returned-Stand requirements. This includes
twelve pose/side combinations not used in the four-case pilot; it is not a new
optimization seed or second source-policy replication.

Buffer compact teacher inputs/actions during the entry comparison, but export
teaching data only if all sixteen cases complete successfully. The old learner's
380-value clock-driven goal input does not describe this event-driven teacher.
Use an explicitly versioned candidate contract: the original 374-value Stand
observation, the exact 46-value desired pose at command application time, and
the controller blending fraction (421 total); output 33 normalized joint-target
additions with radius 1 rad around frozen-source targets. Include issued/source
targets and case/frame indices for transport auditing. Do not silently feed
these examples to the old 380-input trainer. Record whole-case train/validation
splits (poses 0-5 versus 6-7); no random adjacent-frame split. No training in this
replication. Retain compact physical traces; no periodic checkpoints or substep
observation archive.

### Result: 13/16 complete; teaching-data gate remains closed

Run `single-support-all-poses-20261007-01` admits all sixteen pose/side goal sets
(32 transfer/lift targets) under the frozen static target-PD and geometry gates.
All eight source and eight feedback quiet cases retain Stand for the full four
seconds. Ordinary-start sequences complete for every left-swing case and for
right swing in poses 0, 1, 5, 6 and 7: 13/16 overall. Successful cases maintain
1.466784-1.516788 seconds of valid single support, then replant and finish with
two seconds of original Stand. These are scripted-controller results.

Right swing in poses 2, 3 and 4 fails physically at 10.084140, 10.150812 and
9.850788 seconds respectively, with zero valid single-support duration. At the
start of the hold phase, whole-foot clearance is only 8.646, 7.415 and 7.400 mm,
against the unchanged 15 mm gate. While posture/support still passes, maximum
clearance is only 9.385, 7.645 and 7.647 mm. Chest tilt first exceeds its gate at
8.534016, 8.283996 and 8.283996 seconds, before stance loading/drift fails. Static
goal feasibility therefore does not establish dynamic attainability across all
starting poses. These crossings locate symptoms; they do not prove the cause.

The prior four pilot cases reproduce exactly in every active qpos, qvel and
issued motor target. This rules out a trajectory change in those cases from
enabling full-pose coverage and sample buffering. The twelve additional cases
expose three failures that the small pilot could not detect.

The all-sixteen export gate correctly rejects this run: no teacher-samples NPZ
or teacher-contract JSON is exported, and no partial-success subset is silently
admitted. Candidate 421-input collection code and its verifier are implemented,
but end-to-end successful dataset verification remains pending. No learning or
new checkpoint is produced, and Arena selections remain unchanged.

Independent reconciliation verifies 32 physical cases, ten output hashes,
80 source snapshot entries, input identities, finite/timed traces, target limits,
support durations and final Stand/sequence gates. Private controller simulation
totals 2,315,692 substeps, excluding one transport check with eight private and
eight live substeps. Saved failure diagnosis adds no physical cases or physics
steps. Run evidence including compact traces and sources occupies 13,425,719
bytes (12.80 MiB). Evidence:
`logs/mimickit-training/support-all-poses-review-20261007-01/report.json` and
`logs/mimickit-training/support-all-poses-review-20261007-01/diagnosis.json`.

Next priority: isolate the right-swing height/posture tracking failure using the
three failed cases and matched passing cases. Compare measured versus requested
swing-foot and torso motion before changing one controller term at a time; the
earlier full COM-plus-foot task objective regressed and must not be reinstated
without a targeted comparison. Preserve the 15 mm clearance, one-second hold
and two-second returned-Stand gates. Repeat the same sixteen-case gate after a
successful correction, then export and validate teaching examples. Training and
stronger perturbations remain downstream of that physical teacher gate.


## 2026-10-08: learned support block completed (20/20)

See LEARNED_SUPPORT_PLAN.md and LEARNED_SUPPORT_PROGRESS.md for the frozen
protocol, complete ledger, and limitations. Four fresh supervised learners and
16 frozen batches (256 cases) finished. Goal-relative labels halved held-out
prediction error, but every learned entry and enabled-adapter quiet case failed.
No learned foot placement or hit-timing robustness was achieved. Current Arena
parent f832bab8 is unchanged; it passed a separate 16/16 30-second quiet control.

The historical physical teacher's all-sixteen export gate remains failed. This
successor explicitly used only its 13 complete successes as experimental data,
held out whole poses, and evaluated all sixteen combinations. Full old-command
replay failed later despite close early observation reconstruction; see the
progress document. No script or playback was promoted as a learned brain.

Next priority supersedes the earlier next-action paragraph: establish stable
learned handoff and zero-residual quiet retention, then collect corrective labels
on learner-visited states. Actual displacement and hit recovery remain gated on
successful learned support. Avoid another long run or timing sweep of the failed
adapters. Evidence: learned-support-20261008-01, learned-support-audit-20261008-03,
and learned-support-review-20261008-01 under logs/mimickit-training.

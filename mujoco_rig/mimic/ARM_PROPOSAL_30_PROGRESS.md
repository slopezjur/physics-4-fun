# Proposal foundations: second thirty-step investigation - 2026-10-08

Status: all thirty research steps completed; final independent audit passed.
Evidence: `logs/mimickit-training/arm-proposal-30-20261008-01`.
Audit: `final-audit.json` in that directory.
Plans: `ARM_PROPOSAL_30_PLAN.md`, `ARM_PROPOSAL_30_ADAPTIVE_PLAN.md`,
`ARM_PROPOSAL_30_FINAL_BLOCK.md`. Arena remains unchanged.

## Decision and progress

Retain the fresh inherited-feature, complete-data recipe for further research.
Its two independently initialized learned heads completed 5/8 and 3/8 nominal
arms-clear hold/return skills, and 5/8 and 2/8 at changed timing. The old proposal
completed zero. Both new heads retained quiet Stand exactly for sixty seconds.
This is learned task-tracking progress, but not admission to weight transfer,
single-foot support, corrective stepping or 10 kg impact recovery.

Do not promote either head into Arena. Neither passes all nominal and changed-
timing cases. Some remaining failures concern precise motor/COM tracking; others
include support failures. Do not equate staying upright with completing the skill.

Reject the task-relative feature representation and active-only teacher-sample
filter tested here: both failed in both seeds. Keep quiet samples in the dataset.
Do not extend PPO or supervised training time without first addressing the
remaining learner-state feedback error.

## What the command-bias investigation established

The archived labels are correctly defined as raw target minus desired motor
angle, before authority, displacement fade, slew and final joint clipping.
Reconstruction of the Stand source and issued teacher commands was bitwise exact.
The physical teacher again reproduced 8/8 complete nominal and 8/8 changed-timing
skills, with bitwise-equal qpos/issued traces to its archived runs.

On active teacher states the inherited proposal had 0.0050-rad training RMSE
and 0.0218-rad held-out RMSE. Quiet samples were 33.16% of the dataset. Good
regression on teacher trajectories did not ensure stable autonomous feedback.

Forty read-only teacher queries on identical inherited-policy states localized
disagreement during the ramp. Left-arm raw-target disagreement was 0.015 rad
RMS at frame 90 and 0.264 rad at frame 150. By frame 300, the learner's mean
left-arm residual was -0.506 rad and the teacher's was -0.509 rad on those same
already-drifted states. A large late offset therefore cannot be interpreted as
a simple constant-label error: a physical feedback controller also issues targets
near a displaced joint while attempting to correct it.

In the inherited trajectory, hold-phase raw left-arm target error was 0.578 rad
RMS and issued error 0.525 rad. Only about 0.08% of hold motor coordinates hit
the source envelope and 0.61% hit final joint clipping; 37.0% hit the addition
slew limit. Input clipping was sparse, not a general explanation of failure.
Local raw-target sensitivities were recorded without claiming they prove full
closed-loop stability. Teacher queries did not modify live state or queues.

## Structural ablations: distinguish scripted references from learning

Direct goals for all motors and static feedforward for all motors both failed
the full skill. Direct or static arm targets with the remaining targets supplied
by the frozen Stand anchor yielded 6/8 holds and returns but zero complete skills.
The body/COM response still mattered.

The initial name filter selected six UpperArm motors and missed the two Forearm
motors because of case-sensitive matching. Those results are preserved as
upper-arm-only ablations; step 14 explicitly included all eight arm motors.
Removing learned residuals from all eight arm motors while retaining the old
non-arm proposal produced 8/8 holds and returns, with 6/8 complete nominal and
7/8 complete changed-timing skills. Removing upper-arm residuals alone produced
0/8 nominal and 1/8 changed-timing full skills.

Elbow-only removal, or either elbow individually, still produced zero complete
skills. Thus the result is an interaction involving arm outputs, not evidence
that one elbow alone is the sole fault. Static feedforward on all eight arm
motors with the inherited non-arm controller produced 8/8 complete nominal and
6/8 changed-timing skills. These are scripted output interventions over a learned
body controller; they are not newly learned recovery policies and were not
installed in Arena.

## Fresh learned-head comparison

Both representations used the same 421-128ELU-128ELU-33 architecture, per-motor
output scaling, zero output layer, teacher dataset, Adam rate 1e-4, 4,000 updates
and batches of 512. Within each seed, initial weights were identical. The new
proposal predicts all 33 raw motor-target residuals; no output is scripted or
masked by arm/elbow group during ordinary learned evaluations.

The inherited representation uses the existing normalization and masks delayed
command columns 95:229. The alternative uses 116 fixed-scale task-relative state,
goal, velocity and load features, padded to the same network input size. Neither
fits normalization or weights on poses 6/7.

All six heads were freshly initialized. The frozen Stand anchor was shared.
Comparison with the old 256-unit continued-training proposal changes both width
and initialization; this sequence does not isolate their individual effects.

- Complete-data inherited features, seed 711028: nominal 8/8 holds, 8/8 returns,
  5/8 complete; changed timing 7/8 holds, 7/8 returns, 5/8 complete.
- Complete-data inherited features, seed 721028: nominal 6/8 holds, 8/8 returns,
  3/8 complete; changed timing 6/8 holds, 8/8 returns, 2/8 complete.
- Task-relative features, seeds 711028 and 721028: zero complete skills at both
  timings in both seeds. Lower offline held-out loss did not predict physical success.
- Active-only inherited-feature fits, seeds 711028 and 721028: zero complete
  skills at both timings in both seeds. Excluding bypassed quiet targets harmed
  physical performance, despite their commands being ignored during quiet execution.
  Their specific stabilizing role was not isolated by this experiment.

Both held-out poses completed the nominal skill under seed 711028. Its nominal
failures were pose 0's motor tracking and poses 4/5's COM tracking. The replica
also had support and tracking failures. The selected heads' largest left-arm
hold error was around 0.08 rad RMS, materially smaller than the old trajectory's
roughly 0.5-rad error, while still leaving meaningful tracking errors.

Shared evaluation-set regression metrics are in step 27. Do not compare the
active-only fits' reported aggregate loss directly with all-sample fit loss:
their validation subsets differ unless explicitly evaluated on the same set.

## Frozen robustness and retention

For complete-data seeds 711028 and 721028 respectively:

- Half-amplitude movement: 6/8 and 4/8 complete skills. These are easier goals,
  not full-amplitude admission.
- Independent 0.005-rad noise on raw proposal outputs, two seeds each: complete
  counts 5/8 and 6/8 for 711028, and 3/8 and 2/8 for 721028. This is a robustness
  challenge; these supervised heads were not trained with PPO exploration noise.
- Unused slower timing, onset/ramp/hold 1.5/2.0/4.0 seconds: 5/8 and 3/8 complete.
  The original inherited proposal completed 0/8 under this timing.
- Unused faster timing, onset/ramp/hold 0.8/1.2/2.5 seconds: 4/8 and 3/8 complete.
- Removing their learned elbow residuals: 6/8 nominal and 5/8 varied for 711028;
  4/8 nominal and 4/8 varied for 721028. Partial improvement does not justify
  calling these patched outputs fully learned or admitted.

The initial three fresh heads passed 48/48 thirty-second quiet cases. The selected
pair and an anchor baseline passed another 48/48 sixty-second cases. All candidate
qpos/issued traces matched the anchor exactly, and the sixty-second baseline's
first thirty seconds matched the archived thirty-second trace. Quiet preservation
comes through the existing bypass and does not certify active balance recovery.

## Thirty completed research steps

Each item has `step-NN/report.json` under the evidence root. Analytical steps
reuse identified data rather than claiming extra physical trials.

1. Teacher-label, normalization and regression audit: exact source/issued reconstruction.
2. Inherited proposal replay: exact nominal trace; zero full skills at both timings.
3. Direct whole-body goals: zero full skills at both timings.
4. Whole-body static feedforward: zero full skills at both timings.
5. Physical teacher replay: 8/8 full skills at both timings, exact archived traces.
6. Upper-arm goals with anchor other joints: partial stability, no full skill.
7. Upper-arm static feedforward with anchor other joints: no full skill.
8. Largest-error left-arm coordinate removal: no full skill.
9. Six upper-arm residuals removed from inherited proposal: 0/8 nominal, 1/8 varied.
10. Global residual caps 0.05/0.1 rad: no full skills; blanket clipping is not a fix.
11. Forty teacher queries on matched learner states: ramp disagreement localized.
12. Proposal/slew/joint-clip decomposition: late bias predates final clipping.
13. State/goal sensitivity and normalization audit: no widespread input clipping.
14. Complete eight-arm ablation: 6/8 nominal and 7/8 varied full skills with learned non-arm control.
15. Matched fresh inherited-feature and task-relative fits, seed 711028.
16. Nominal physical comparison: inherited 5/8 full, task-relative 0/8.
17. Changed-timing comparison: inherited 5/8 full, task-relative 0/8.
18. Independent task-relative replica: zero full skills; reject representation.
19. Thirty-second quiet retention: 48/48 exact cases.
20. Elbow-only and static-arm counterfactuals: interaction localized; static-arm diagnostic 8/8 nominal, 6/8 varied.
21. Fresh inherited-feature replica: 3/8 nominal, 2/8 varied full skills.
22. Active-only supervision seed 711028: zero full skills.
23. Active-only supervision replica 721028: zero full skills; retain complete dataset.
24. Half-amplitude comparison: complete-data heads 6/8 and 4/8 full skills.
25. Raw-proposal noise comparison: retained partial robustness; no universal pass.
26. Unused slower timing: complete-data heads 5/8 and 3/8 full skills.
27. Common-data loss, physical gates and foot/support analysis: select complete-data recipe using the predeclared two-seed criterion.
28. Elbow-output counterfactual on new heads: some improvement, still incomplete and scripted.
29. Sixty-second quiet retention: 48/48 exact cases, including the baseline.
30. Faster-timing final challenge: learned heads 4/8 and 3/8 complete; no Arena admission.

## Resources and audit

Six bounded supervised fits, 24,000 optimizer updates total and 12,288,000 sampled
example presentations. These repeatedly sample the same teacher dataset and are
not environment transitions. No PPO training ran. Measured time inside the six
fit functions was 43.53 seconds, excluding physical evaluations, setup and audits.
Initial/final weights only; no periodic checkpoint series.

The complete-data fits used 6,804 training and 2,268 held-out examples. Active-only
fits used 4,548 and 1,516. Whole poses 6/7 were excluded from every fit. There were
680 physical evaluation cases, including 96 quiet cases and 16 teacher-reference
cases, plus an additional eight-case inherited trace replay for decomposition
and forty private teacher queries. These repeat the same eight reference poses;
they are not 680 independent training seeds or characters.

Evidence occupied 407,575,921 bytes, approximately 0.408 GB, at final audit. The
audit passed sealed inputs/outputs, independently recomputed full physical gates,
projection and joint bounds, paired initial weights, dataset splits, update
accounting, seeded noise streams and exact quiet retention. Teacher replay was
bitwise identical. Rig, springs, delay, torque limits and task admission gates
were unchanged. No C# compilation or Arena file change occurred.

The first auditor used an unnecessarily strict 1e-8 m tolerance for comparing
reported foot-drift metrics with a reconstruction from double-precision initial
geometry. The maximum observed difference was 3.801e-8 m. Its source and failure
log are preserved. The successor auditor allows 1e-6 m for metric-number
reconciliation while independently retaining the exact 0.02 m physical gate.
No pass/fail threshold or physical outcome was changed.

Final auditor: `verify_arm_proposal_30_rounding.py`.
Arena ONNX remains
`c87c272a0cf60d857de45f186eb5c64b7c63652ca5acf7a5ba050afebe6ad172`.

## Retained research artifacts and next priority

The strongest ordinary learned task head is `step-15/inherited/final.pt`, seed
711028; its independent complete-data replica is `step-21/model/final.pt`, seed
721028. Load with `ProposalHead` and the adjacent architecture metadata. These
are goal-conditioned 421-input task heads over the existing Stand anchor, not
drop-in replacements for the current Arena's Stand policy.

Next, collect verified corrective teacher trajectories from these fresh heads'
remaining failure states. Begin before the first motor/COM/support violation,
keep the actual delayed queue and physical bounds, and demonstrate that teacher
takeover can complete the entire hold and return before admitting those samples.
Use training poses only for corrective-data fitting, preserve the original quiet
data, and retain whole-pose holdouts and both initialization seeds.

Compare corrective-data fitting with a matched complete-data replay control.
Do not repeat the rejected task-relative inputs, discard quiet samples, or buy
a longer PPO run as the next action. Resume weight-transfer and single-foot
learning only after reproducible full arms-clear admission. No learned foot
placement or human-like impact recovery is claimed by this result.

## 2026-10-08 - Corrective takeover and matched fitting completed

See ARM_CORRECTIVE_LEARNING_PROGRESS.md. The physical teacher recovered all 16 attempted failing training cases from exact learned prefixes, admitting 5,827 samples with no heldout labels. Four matched fresh fits then showed uniform correction augmentation regressed full learned skills from 15/32 to 5/32; original replay weights and trajectories reproduced exactly. Both augmented heads retained exact sixty-second quiet Stand, 32/32 cases. Both independent audits passed. Retain the previous complete-data heads; Arena unchanged. Next: isolate sampling balance (quiet share fell from 33.16% to 17.86%) with controlled correction weighting and matched physical evaluations, before more data or longer training. This is a confound to test, not an established cause. Evidence: logs/mimickit-training/arm-corrective-takeover-20261008-01/audit.json and logs/mimickit-training/arm-corrective-fit-20261008-01/audit.json.

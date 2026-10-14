# Autonomous recovery investigation - 2026-10-08

Status: all thirty research steps completed and audited.
Evidence root: `logs/mimickit-training/arm-recovery-30-20261008-01`.
Final audit: `final-audit.json` in that directory.

## Decision

Reject all four new actors for promotion. Quiet Stand remains intact, but the
complete arms-clear hold-and-return prerequisite still fails. This sequence
does not demonstrate learned foot placement, single-foot support or human-like
recovery from a 10 kg impact. Arena remains on the existing Stand/perturb anchor.

The investigation found a real execution-distribution mismatch and a useful
critic-initialization improvement. Neither solved the full physical skill.
Do not escalate to longer runs based on partial hold counts.

## What changed our understanding

Independent Gaussian exploration is not physically centered at the deterministic
command when slew/envelope projection is active. On the same visited states,
the RMS difference between the projected mean and mean projected command was
0.001111, 0.000922 and 0.000962 rad for the zero, 521028 and 531028 heads.
These differences are comparable to the trained raw correction means. Analytic
expectations agreed with 4,096-point normal-quantile quadrature within 1.42e-7
rad. This calculation precedes final joint clipping.

Executing the expected projected command with the untrained head produced 6/8
supported holds and 5/8 returns at nominal timing, versus 4/8 and 4/8 for zero
correction. However, changed timing gave only 1/8 and 1/8. Both conditions had
0/8 complete skills. The improvement was caused by an execution change with
untrained weights; it must not be called learning.

Short physical interventions sometimes changed from helpful to harmful when
the reward horizon grew: 54/175 nontrivial paired effects reversed sign between
one and two seconds. Frozen one-step critic differences agreed with two-second
physical reward differences in 51.25% of included comparisons and 31.25% of
held-out comparisons. These are correlated, off-policy diagnostic branches,
including twelve-frame interventions, not independent trials or a certification
of on-policy critic accuracy.

Under their own noisy policies, the older final critics showed realized-return
MSE of roughly 706-1,108 on training poses and 1,760-4,215 on held-out poses.
This is Monte Carlo error on the sampled episodes, not exact value-function
error. We therefore tested value warm-up rather than assuming the action
distribution alone explained the failures.

Two fresh critics were fitted to 11,837 alive training-pose samples from four
zero-head noisy rollouts, with four passes and 188 value updates each. Two
unused rollout seeds supplied validation. Relative to fresh zero-output critics,
validation MSE fell 83.5%/85.9% on training poses and 71.45%/71.48% on held-out
poses. Both passed the predeclared 20% improvement gate. Held-out value bias
remained approximately -28.5/-28.7 reward units; warm-up did not certify good
action ranking.

## Matched PPO outcomes

All four fits used the same zero-output actor initialization, 24,576 transitions,
32 collections of 64 frames across 12 training worlds, std 0.005 rad, actor
rate 3e-6, critic rate 3e-4, three PPO epochs, and unchanged broad rewards,
physical limits, delay, quiet bypass and complete-skill gates. Each seed pair
had identical first collection statistics. Only initial critic weights changed.
Each fit used fresh actor and critic optimizers.

- Seed 611028, warm critic: nominal 4/8 holds, 3/8 returns; changed timing 4/8
  holds, 4/8 returns. Actor updates: 231. Complete skills: zero.
- Seed 611028, fresh critic: nominal 2/8 holds, 3/8 returns; changed timing 1/8
  holds, 0/8 returns. Actor updates: 224. Complete skills: zero.
- Seed 621028, warm critic: nominal 5/8 holds, 3/8 returns; changed timing 4/8
  holds, 3/8 returns. Actor updates: 232. Complete skills: zero.
- Seed 621028, fresh critic: nominal 2/8 holds, 4/8 returns; changed timing 2/8
  holds, 1/8 returns. Actor updates: 226. Complete skills: zero.

Warm-up improved supported-hold counts against both fresh-critic controls and
improved changed-timing results. Nominal returns were mixed, and neither warm
actor retained the initial reference's 4/8 returns. These two seeds establish
a bounded comparison, not broad statistical superiority. Warm-up used extra
data and value optimization, so this is not an equal-total-compute comparison.

Fresh noisy evaluations used seeds 612001/612002, paired across all candidates:

- Untrained head: 7/16 holds, 6/16 returns, 0/16 complete.
- Warm 611028: 8/16 holds, 9/16 returns, 0/16 complete.
- Warm 621028: 8/16 holds, 5/16 returns, 0/16 complete.
- Fresh-critic 611028: 5/16 holds, 6/16 returns, 0/16 complete.
- Fresh-critic 621028: 8/16 holds, 6/16 returns, 0/16 complete.

All four final actors passed sixteen thirty-second quiet cases each. All 64
qpos and issued-command traces exactly match the archived Stand anchor through
the quiet bypass. This protects Stand; it does not validate active recovery.

## Remaining physical error

During alive frames of the final two-second nominal hold, UpperArm_L_ry remained
the largest joint error. Warm actors had actual-to-goal RMS errors of 0.571 and
0.527 rad, while issued-command-to-goal RMS errors were 0.569 and 0.514 rad.
The dominant arm error is already present in the commanded targets. Similar
magnitudes do not prove the only cause, especially across different trajectories,
but they justify auditing the inherited proposal before more PPO tuning.

This task requires fixed feet while moving the arms. Foot drift is a failure
metric here, not evidence of learned corrective stepping. No impact trials or
learned stepping admissions were added by this sequence.

## Thirty completed research steps

Each numbered item has its own `step-NN/report.json` under the evidence root.
Analytical steps reuse explicitly identified physical trials; they are not
claimed as additional simulation or training runs.

1. Mean versus training noise: 120 cases; exact deterministic replay; zero full skills.
2. Independent noise-seed replication: 96 cases; zero full skills.
3. Changed-timing mean/noise comparison: 72 cases; zero full skills.
4. Half-sized noise: 48 cases; zero full skills.
5. Quarter-sized noise: 48 cases; zero full skills.
6. Same-state projection bias: quantified analytically and checked by quadrature.
7. Expected projected commands: 24 cases; improved untrained partial counts only.
8. Expected commands under changed timing: 24 cases; nominal improvement did not transfer.
9. Three-frame correlated noise: 48 cases; zero full skills.
10. Ramp/hold/return noise isolation: 144 cases; zero full skills.
11. Matched ramp interventions: 72 two-second physical continuations.
12. Matched hold interventions: 72 two-second physical continuations.
13. Matched return interventions: 72 two-second physical continuations.
14. Ramp critic ranking versus physical outcomes: completed on matched branches.
15. Hold critic ranking versus physical outcomes: completed on matched branches.
16. Return critic ranking versus physical outcomes: completed on matched branches.
17. Reward-horizon comparison: 54/175 nontrivial effects reversed sign.
18. Reward/bootstrap decomposition: measured at 3, 12, 60 and 120 frames.
19. Learned/opposite correction directions: partial local gains, no consistent full-skill result.
20. Held-out direction checks: weak ranking persisted at the second horizon.
21. On-policy value diagnostic: 80 full episodes with frozen actors and retained features.
22. Two bounded value-only fits: 188 updates each; actor weights remained frozen.
23. Fresh-episode and held-out value validation: both passed the predeclared gate.
24. Warm-critic PPO seed 611028: bounded fit plus nominal/changed-timing evaluation.
25. Warm-critic PPO seed 621028: independent bounded replication and timing evaluation.
26. Warm-actor quiet retention: 32/32 exact thirty-second Stand cases.
27. Phase/joint/foot analysis: dominant arm tracking error also present in issued commands.
28. Fresh-critic control 611028: matched fit plus nominal/changed-timing evaluation.
29. Fresh-critic control 621028: replicated fit plus nominal/changed-timing evaluation.
30. Final admission comparison: 80 fresh-noise active cases and 32 exact control quiet cases; reject all candidates.

## Accounting, audit and implementation failure

- 98,304 completed PPO transitions; 913 actor gradient updates; 168.25 seconds
  measured inside the four learning loops. This excludes evaluation/setup time.
- 376 additional value-only updates on 11,837 unique training samples.
- 912 end-to-end physical evaluation cases: 848 active arm cases and 64 quiet
  cases. Active complete skills: 0/848. These repeat eight poses across conditions;
  they are not 848 independent humanoids or training seeds.
- 216 matched two-second continuations, counted separately from complete episodes.
- Evidence at final audit: 521,522,088 bytes, approximately 0.522 GB, including
  compressed traces, source snapshots and the preserved failed attempt.
- Initial/final actor and critic artifacts only; no periodic training checkpoints.

The first step-24 attempt inherited requires_grad=False from the evaluation
loader. It failed at the first actor backward pass after one 768-transition
collection, before any actor update. The driver now explicitly makes the
training initializer trainable. That failed attempt and its exact sources are
preserved in `step-24-failed-frozen-actor`; the successful run restarted from
identical initial weights and seed. Its 768 transitions are extra to the
98,304 completed comparison transitions. This was a harness error, not a
negative learning result, and was not counted as a research step.

The final audit passed all thirty reports, sealed input/output hashes,
independent physical gates and projection bounds, identical initial actor
weights, correct warm/fresh critic initialization, identical paired first
collections, transition/update accounting, reward-component reconciliation,
recomputed warm-up validation, exact quiet traces and archived failed sources.
No admission threshold was relaxed. No springs, physical limits or Arena files
were changed, and no C# compilation was performed.

Arena ONNX SHA256 remains
`c87c272a0cf60d857de45f186eb5c64b7c63652ca5acf7a5ba050afebe6ad172`.

## Next priority

Run a structural proposal ablation before another fit: compare the inherited
queue-masked proposal with direct goal-following and the already verified
physical teacher through the same target, delay and limit contract. Localize
the large UpperArm_L_ry issued-target error and determine whether the proposal
should be replaced or its target representation corrected. Any scripted
controller in that comparison is a diagnostic reference, not a learned brain.

Require a well-behaved command baseline and reproducible full arm hold/return
before resuming weight-transfer and single-foot learning. Keep the current
Arena brain until a learned candidate passes full physical admission.


## 2026-10-08 - Second thirty-step proposal investigation completed

See ARM_PROPOSAL_30_PROGRESS.md. All 30 steps completed and independently audited. Teacher labels and replay were exact; large late target offsets also appeared in matched teacher queries, so they were not a simple constant-label bug. Fresh inherited-feature task heads now complete 5/8 and 3/8 nominal skills, and 5/8 and 2/8 changed-timing skills. Task-relative inputs and removal of quiet samples failed both seeds. The selected pair retained exact sixty-second Stand. Six fresh supervised fits, 24,000 optimizer updates; no PPO. Scripted eight-arm ablations improved outcomes but are clearly distinguished from learned results. No candidate passes universal admission; Arena unchanged. Next: verified corrective teacher takeovers from the fresh heads' remaining motor/COM/support failures, then a matched corrective-data comparison retaining quiet data and whole-pose holdouts. Audit: logs/mimickit-training/arm-proposal-30-20261008-01/final-audit.json.

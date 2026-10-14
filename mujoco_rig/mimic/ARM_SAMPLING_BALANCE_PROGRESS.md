# Sampling balance: completed comparison and feedback diagnostics

2026-10-08. Protocol: ARM_SAMPLING_BALANCE_PLAN.md.
Evidence: `logs/mimickit-training/arm-sampling-balance-20261008-01`.

## Decision

Preserving the quiet sampling stream is insufficient to produce reliable
learned arms-clear control. The protocol selects 5% corrections for further
research because its worst-condition score improves, but its small aggregate
gain includes heldout regressions. It is not admitted for Arena, weight
transfer or stepping. Preserve the previous heads and these research heads;
do not replace the current Arena policy.

The next priority is a frozen matched-state feedback investigation during the
arm-movement ramp, before another training run. More sampling-ratio tuning or
longer fitting is not justified by this block.

## What changed

Eight fresh heads: seeds 711028 and 721028, with replay, 5%, 15% and 37.5416%
expected correction shares. All used the same 421-128-128-33 inherited-feature
architecture, normalization, zero output-layer initialization, Adam 1e-4,
4,000 updates and batches of 512. Physical bounds, delay and skill gates were
unchanged. Corrections came from the already verified 5,827 examples; there
was no new teacher collection.

Each batch began with the historical replay's original training draw. Quiet
rows were kept at precisely the same positions. Only active rows could be
replaced. Replacement masks were nested across shares, using identical random
draws and correction indices within each seed. The high share preserved the
original-active/correction ratio from concatenation while restoring quiet
sampling. It is not an exact one-variable recreation of the earlier uniform
mixture: original-active and correction frequencies both change.

Every recipe at seed 711028 received exactly 679,923 quiet presentations;
every recipe at seed 721028 received exactly 680,812. The expected quiet share
was 33.15697%; finite sampled counts were identical within each seed. Initial
weights were bitwise identical within each seed. Both replay final weights and
all four replay physical traces reproduced the historical results exactly.
The auditor independently regenerated all sampler hashes, counts and nested
masks. Validation poses 6/7 were never sampled for fitting.

## Frozen physical results

Counts below are full skill passes out of eight, nominal / varied timing.

- Replay: seed 711028 **5 / 5**; seed 721028 **3 / 2**.
  Total 15/32; worst condition 2/8.
- 5% corrections: seed 711028 **5 / 4**; seed 721028 **3 / 4**.
  Total 16/32; worst condition 3/8.
- 15% corrections: seed 711028 **6 / 5**; seed 721028 **2 / 2**.
  Total 15/32; worst condition 2/8.
- Restored-quiet high share: seed 711028 **2 / 1**; seed 721028 **1 / 1**.
  Total 5/32; worst condition 1/8.

The 5% pair was selected by the predeclared worst-condition, then total-pass
score. Both models retained exact sixty-second quiet Stand: 16/16 each,
32/32 total, including bitwise-identical commands, states and survival against
the archived anchor. Quiet gating suppresses the task addition; this does not
establish that the task controller is stable during movement.

The split matters. Replay passed 8/24 training-pose cases and 7/8 heldout
cases. The 5% mixture passed 11/24 training-pose cases and 5/8 heldouts. The
first seed lost heldout pose 6 at both timings, while gaining training pose 4.
Its varied run also lost training pose 1. The second seed gained training
poses 1 and 3 at varied timing. These are trade-offs, not uniform improvement.

For 15%, training/heldout counts were 9/24 and 6/8. For high share they were
5/24 and 0/8. No recipe passes the required complete skill across both seeds
and timings. The same correction collection was used for both fitting seeds;
this is initialization replication, not independent data collection.

## First command differences and subsequent physical response

`diagnostics.json` compares each new trajectory with its seed/timing replay.
It also applies each frozen head to the exact saved replay observations and
projects its output through the replay's source targets, previous addition,
authority, fade, slew and joint bounds. This measures direct policy differences
on common inputs separately from differences after the trajectories evolve.
Replay reconstructs itself within the 2e-6 rad inference-batch tolerance.

For 5% corrections, all 32 cases first exceeded a 0.001-rad issued-command
difference at 1.3501-1.9168 seconds, during the movement ramp. Joint differences
above 0.001 rad followed at 1.3834-2.0502 seconds. COM differences above 1 mm
appeared at 1.8001-3.5836 seconds.

Across cases, the median maximum command difference during the ramp was
0.01146 rad on identical replay inputs/history, versus 0.07148 rad along the
actual changed trajectories. Maximum values were 0.02964 and 0.21384 rad.
These are maxima within each ramp, not local stability gains or a causal
decomposition. They show that state and command-history evolution materially
change the result; they do not identify which feedback pathway is responsible.

Example: first seed, nominal heldout pose 6, formerly a full pass. Its first
1-mrad command difference occurred at 1.3501 s, joint difference at 1.5001 s,
and 1-mm COM difference at 1.8001 s. COM tracking exceeded 0.02 m at 4.5337 s,
inside the scored hold interval. The largest ramp command changes involved
UpperArm_R_ry, UpperArm_L_ry and UpperArm_R_rz. The case still maintained
support, but failed full tracking admission. This is not simply a fall detector
or episode-duration problem.

Moving-goal violation times in diagnostics are observational thresholds, not
substitutes for the original final-two-second hold and return gates. Some
transient ramp/return violations also occur in replay cases that ultimately
pass the full skill.

## Common-data fitting evidence

All mixtures slightly improved RMSE on the original quiet training examples
for both seeds. At high share, quiet RMSE improved from 0.00406 to 0.00385 rad
and from 0.00411 to 0.00393 rad, despite only 5/32 complete physical skills.
Thus neither reduced quiet exposure nor worse quiet-example fitting is needed
for the observed high-share failure.

At 5%, correction-example RMSE changed only from 0.005834 to 0.005813 rad and
0.005884 to 0.005864 rad. Active heldout RMSE slightly worsened for both seeds.
Offline prediction error remains inadequate for physical admission. This
block does not prove the cause of the earlier uniform-mixture regression.

## Next bounded investigation

Use the retained replay and 5% heads for frozen comparisons around the first
command divergence, focusing on training failures and the lost heldout pose 6.
Branch from identical integration state, delayed queue and previous addition;
compare replay, changed head and verified teacher continuations without state
assistance or relaxed bounds. Measure local response to small arm-position and
velocity errors and whether those responses reduce or amplify the error over
the following control intervals.

Audit available inputs as part of that diagnosis: the inherited task head
deliberately masks columns 95:229, the two queued commands, while the physical
teacher predicts through those commands. Earlier raw-history ablations had
their own feedback problems. Neither restoring raw history nor changing the
representation is justified without a matched test of the information needed
at the failure states. Do not repeat the rejected broad task-relative input
representation on speculation.

Require reproducible full arms-clear admission before weight transfer and
single-foot support. No learned foot repositioning or human-like hit recovery
was established here. The current Perturb Arena still uses the earlier
374-input recovery brain, not these 421-input task heads.

## Audit and accounting

- `audit.json`: passed; independent sampling reconstruction, unchanged input
  hashes, paired initializations, exact historical replay, physical gates and
  exact quiet retention.
- `diagnostics.json`: 16 physical comparisons and 40 common-data regression
  groups with hashes of models, datasets, traces and analysis source.
- 8 fresh fits; 32,000 optimizer updates; 16,384,000 sampled presentations.
  These are supervised examples, not PPO transitions.
- 128 active physical evaluations and 32 sixty-second quiet evaluations.
- Only initial/final checkpoints; no periodic checkpoint stream.
- No PPO, C# build, spring, rig, latency, bound or Arena policy change.
- All fitting, evaluation, audit and diagnostic processes completed.

## 2026-10-08 - Matched feedback and queue-input diagnosis completed

See ARM_MATCHED_FEEDBACK_PROGRESS.md. Across 24 targeted matched-prefix cases, learned continuation passed 6/24, swapping heads 7/24, and physical teacher continuation 24/24. All 24 small bounded pulses transmitted; 9 improved complete-skill outcomes, 2 regressed, and 2 introduced native failures in an already-incomplete pose. Position differences grew from 0.2 to 1 second in all pulse probes. Private queue diagnosis reproduced teacher targets exactly in 24 reconstructed states; 96 queue-only variants changed every teacher target while both masked task heads' raw outputs stayed identical. The Stand source still responds to the queue. This is an exact-imitation input mismatch, not proof of the sole failure cause. Both audits passed. No training or Arena update. Next: validate a bounded representation of pending-command effects before matched fitting; do not blindly restore raw history or continue sampling-ratio search. Evidence: logs/mimickit-training/arm-matched-feedback-20261008-01 and logs/mimickit-training/arm-private-queue-20261008-01.

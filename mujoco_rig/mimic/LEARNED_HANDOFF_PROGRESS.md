# Learned handoff progress

2026-10-08. Completed the guarded-handoff pilot and one bounded late-state correction follow-up. Arena parent f832bab8 remains selected. No successful lift, foot placement or impact recovery was demonstrated.

## Implemented and verified

The successor adapter returns zero added correction for the original quiet goal. For active goals it fades the correction with goal displacement and authority, and limits changes in the addition to 0.02 rad per control frame. This leaves joint limits, torques, zero passive springs and the two-command delay unchanged.

All three 30-second quiet batches passed 16/16. Their entire saved issued-target and qpos traces exactly match the previous zero-addition Stand control. This is an explicit controller bypass for quiet goals, not a newly learned standing skill.

Live-only addition slew was verified at 0.020000011 rad/frame (float32 rounding). The original all-world counters show 1 rad because they include inactive failed worlds resetting their addition to zero; that counter is not a live rate violation. The separate review records the corrected interpretation. Corrections can still reach the existing 1-radian envelope; this is not a global stability guarantee.

## Matched learning comparisons

The first collector queried 192 states from training poses 0, 1, 4 and 5, both sides, from 1.00008 to 3.300264 seconds. The second queried 85 additional states from 3.50028 to 6.300504 seconds; some worlds had already failed before this window. Whole-pose holdouts 6 and 7 were never labelled or trained.

Four matched 1,500-update supervised continuations ran: two controls and two correction treatments, 6,000 updates total. These are not PPO cycles. Normalization remained fixed. The physical predictor supplied local corrective labels only; it did not drive live actions. Every query checked that live position, velocity and the pending command queue were unchanged. Frozen rollouts with and without those queries match exactly.

First round correction-target RMSE: replay 0.05955 rad, correction treatment 0.00733 rad. Held-out original-trajectory RMSE worsened from 0.00865 to 0.01096 rad. Both completed 0/16 support sequences.

Late-state round correction-target RMSE: replay 0.16781 rad, correction treatment 0.00675 rad. Held-out original-trajectory RMSE was 0.01078 versus 0.01133 rad. Both completed 0/16. Better fitted labels did not improve physical control; the late treatment is rejected.

## Physical outcomes

The initial guarded learner and all four updated learners completed 0/16 entry/hold/replant/return sequences. None achieved the valid 50 ms lift diagnostic. All physically failed in the transfer phase; first support violations sometimes occurred earlier during arm movement. The later-state collection re-evaluated the first correction model and reproduced its failure exactly.

Across the first correction learner, first support violations occurred at 2.0168-4.7670 seconds and physical failures at 3.5003-6.3505 seconds. The late correction learner violated support at 1.8835-3.3836 seconds and failed physically at 3.1669-4.7670 seconds. These ranges describe different cases; they are not an average or a paired improvement claim.

There were 144 planned physical cases: six 16-case support/collection batches and three 16-case quiet batches. Two additional frozen audit batches added 32 cases. No command-timing expansion was run because the all-sixteen support gate failed. Source/input and output hashes were independently checked. No Arena export or C# compilation ran.

## Next priority

Isolate the first loss of support with fixed intermediate goals: first require the learned controller to hold the arms-clear pose, then test a small weight transfer while retaining the same quiet guard. Compare actual control stability with the physical label source before expanding the trajectory. Further offline fitting or collecting later falling states alone is not justified by these two failed comparisons. Whole-foot lift, placement and hit-triggered recovery remain downstream gates.

## Evidence

- Frozen initial plan: `LEARNED_HANDOFF_PLAN.md`.
- First pilot: `logs/mimickit-training/learned-handoff-20261008-01`.
- Prospective late-state protocol and results: `logs/mimickit-training/learned-handoff-late-20261008-01`.
- Independent audit: `logs/mimickit-training/learned-handoff-review-20261008-01/report.json`.


## 2026-10-08: fixed intermediate-goal diagnosis completed

See FIXED_SUPPORT_GOALS_PROGRESS.md. Twelve native cases compared identical
fixed goals and handoff: physical label-source feedback passed arms-clear 2/2
and small transfer 4/4; the frozen neural learner passed 0/2 and 0/4. Physical
success included supported tracked holding and return to Stand. This is a small
pose-0/pose-6 diagnostic panel, not learned recovery or all-pose admission.

The old training pool has only 72 near-arms-clear goal frames, at most 0.133344
continuous seconds, and no near-matching 4 cm transfer goals. Sustained holds
are a missing curriculum component; these results do not uniquely establish
an architecture defect. Next: exact-state sustained arms-clear training with
closed-loop validation and quiet retention, before transfer or lift. Arena
f832bab8 unchanged; no training updates or C# compilation in this diagnosis.
Evidence: fixed-support-goals-20261008-01 and its review under logs/mimickit-training.

# Sustained arms-clear learning results

2026-10-08. Completed exact-data collection, the planned matched learning pilot, and a two-seed command-history ablation. There is a reproducible partial balance improvement, but no learner passes the full arms-clear skill gate. Arena f832bab8 remains unchanged.

## Exact sustained-hold data

Both physical teacher runs completed 8/8 hold-and-return cases: nominal timing (1 s onset, 1.5 s ramp, 3 s hold) and varied timing (1.2 s onset, 1.7 s ramp, 3.5 s hold). Timing changed the visited states; no live pose or velocity writes were used. This remains physical-controller demonstration data, not learned success.

Collected 9,072 exact issue-time examples: 6,804 training examples from poses 0-5 and 2,268 held-out examples from entire poses 6-7. Every retained teacher trajectory completed the full physical gate. Each sample contains the actual 421-input observation, raw unblended target, source target, previous addition, goal and issued command.

The independent audit checked pre-step positions and pending commands against the saved physical trace, pose holdout separation, identical initial learner weights and the complete command-composition equation. Maximum reconstruction error was 1.49e-8 rad. This removes the older approximate-observation and blended-label ambiguity for these new examples.

## Planned comparison

Unchanged learner: 0/8 complete, 0/8 stable holds, 0/8 returned Stand.

Old-example replay control, seed 451028: 0/8 complete, 0/8 stable holds, 0/8 returned Stand. It retained the historical labels and is a compute control, not a raw-target-label ablation.

Exact-hold learner, seed 451028: 0/8 complete, 0/8 stable holds, 0/8 returned Stand. Its training target RMSE was 0.003825 rad and held-out RMSE 0.017019 rad. Accurate teacher-state fitting did not establish closed-loop stability. The exact-hold intervention changes both trajectory coverage and label semantics; neither is uniquely identified as a cause by this comparison.

The new full-input learner retained 16/16 30-second quiet cases, with all issued-command and qpos trace values exactly matching the existing Stand anchor through the quiet-goal bypass.

## Matched command-history ablation

A successor experiment replaced learner input columns [95,229), the two queued commands, with the frozen normalization means during both training and inference. All remaining inputs, initial weights, exact examples, normalizers, optimizer settings and update count stayed matched. The real two-frame delay remains; the frozen Stand anchor still receives its complete observations.

Seed 451028: full input 0/8 stable holds and 0/8 returned Stand; masked queue 4/8 stable holds and 4/8 returned Stand. The masked learner held support in poses 0, 1, 5 and 6 (pose 6 was held out).

Seed 461028: full input 0/8 stable holds and 0/8 returned Stand; masked queue 4/8 stable holds and 3/8 returned Stand.

Every arm still completed 0/8 full skills. The supported masked cases exceeded the unchanged motor-pose tracking tolerance of 0.05 rad during the hold. No gate was relaxed. Queue masking also worsened teacher-state prediction error (held-out RMSE approximately 0.0240/0.0241 rad), while improving physical support. This supports testing feedback architecture using closed-loop behavior, not selecting by one-step fitting error alone. It does not establish that command history is universally harmful or identify every remaining failure.

The first masked learner also passed 16/16 30-second quiet cases with exact anchor trace identity. Quiet retention of the second masked seed was not separately measured; it was not promoted. The masked checkpoints require QueueBlindLearner and include architecture metadata. Loading them as ordinary Learner changes their behavior.

## Accounting and validation

Five supervised fits, 4,000 updates each: 20,000 optimizer updates total, about 25 seconds of measured fitting. These are not PPO cycles or newly collected policy-training transitions. All learners started from the same earlier goal-relative network weights with fresh optimizer state; the Stand anchor stayed frozen.

96 physical cases: 16 teacher demonstrations, 48 frozen learned hold cases and 32 quiet cases. No varied-timing learner gate, third training seed, foot transfer, hit tests, GPU parity, export or C# build was run because the full nominal skill gate failed. All physical gates were recomputed from traces, source/output hashes checked, and queue-mask invariance tested. Exact quiet identity was checked for both evaluated quiet batches.

## Next priority

Keep the queue-masked learner as an experimental direction, not an Arena replacement. Collect small recoverable state deviations around the arms-clear hold and teach the required corrections, then assess both supported holding and accurate pose tracking on all eight poses and the existing holdouts. The current teacher data covers successful trajectories with two timings, but does not certify recovery from nearby disturbed states. Preserve the quiet-goal bypass and all physical gates. Advance to small weight transfer only after the learned arms-clear hold-and-return gate passes.

## Evidence

- Frozen plan: `ARM_HOLD_LEARNING_PLAN.md`.
- Exact data and planned comparison: `logs/mimickit-training/arm-hold-learning-20261008-01`.
- Data/physical audit: `logs/mimickit-training/arm-hold-review-20261008-01`.
- Masked-input comparison: `logs/mimickit-training/arm-hold-queue-ablation-20261008-01`.
- Matched second seed: `logs/mimickit-training/arm-hold-queue-replication-20261008-01`.
- Independent comparison audit: `logs/mimickit-training/arm-hold-comparison-review-20261008-01`.


## 2026-10-08: deviation-supervision comparison rejected

See ARM_HOLD_DEVIATIONS_PROGRESS.md. The first disturbed teacher collection
passed only 1/8 and lacked a successful held-out pose, so it ran zero training
updates. A four-times-smaller innovation bound passed 8/8 and supplied 1,440
exact clean-target correction examples with whole-pose holdouts preserved.

Matched 4,000-update continuations both regressed from the earlier 4/8-supported
experimental learner: replay retained 1/8 supported holds; correction training
retained 0/8. Both were 0/8 complete and are rejected. The treatment still
retained 16/16 exact quiet-Stand cases. Completed 48 physical cases and 8,000
supervised updates; Arena f832bab8 unchanged.

Next: bounded on-policy supported pose-tracking/return learning from the prior
partial checkpoint, after zero-update adapter parity. Do not extend these
rejected supervised fits or advance to transfer/hits. Keep the quiet safeguard,
physical limits, full tracking gate, and whole-pose holdouts.

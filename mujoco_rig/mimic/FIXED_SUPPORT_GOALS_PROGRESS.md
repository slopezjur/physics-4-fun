# Fixed intermediate-goal diagnosis

2026-10-08. Completed twelve native physical cases, with no training or Arena change. The goals and handoff are feasible in the tested cases; the frozen learned controller does not reproduce the physical controller's stable feedback.

## Matched comparisons

Tested pose 0 (a training pose) and pose 6 (a whole-pose holdout). Arm clearing used two cases; 4 cm COM transfer with 4 degree lean used both directions, four cases. Both controllers used the current f832bab8 anchor, identical requested goals, one-second authority transition, displacement fade, 0.02 rad/frame correction slew, original motor limits, zero passive springs and two-command delay.

Arms-clear: the frozen neural controller completed 0/2; the physical label-source controller completed 2/2. Small transfer: the neural controller completed 0/4; the physical controller completed 4/4. Every physical-controller success included the three-second target hold and return, with the last two hold seconds continuously supported and tracked, followed by two seconds of original Stand. No alternative teacher handoff was needed because the guarded physical controller passed.

The explicit tracking gates were motor pose RMSE <=0.05 rad, horizontal COM error <=0.02 m and foot drift <=0.02 m, in addition to existing supported-posture criteria. The verifier independently recomputed these gates from the compact traces and checked protected sources and output hashes. This is a small diagnostic panel, not an all-pose admission or impact-recovery result.

## Training coverage

The original supervised training pool contains only 72 frames whose requested pose is within 1e-4 in every goal coordinate of these arms-clear targets. The longest continuous stretch is 0.133344 seconds. The fixed hold test asks for three seconds. There are no training frames within that tolerance of the new small-transfer goals; their nearest maximum-coordinate discrepancy is about 0.01759. Goal-coordinate discrepancy mixes position and angle coordinates and is not a physical distance or a naturalness score.

This exposes a concrete curriculum gap: the old demonstrations move through arm clearing into a larger transfer, rather than teaching sustained intermediate holds. The new fixed-goal failure therefore does not uniquely identify a neural architecture defect. The physical comparison establishes feasibility for these poses; sparse hold coverage supports teaching the missing stationary subskill next. Earlier full-sequence learner failures remain unresolved.

## Next priority

Collect exact issue-time observations and unblended physical target labels for sustained arms-clear holds, including small state variations, using the current anchor. Train and assess arms-clear holding alone with closed-loop success and quiet retention as the decision criteria. Preserve pose holdouts and compare with an unchanged learner. Only after that skill passes should the learned controller advance to the small transfer and then lift/replant. Do not count physical-controller success as a learned brain or load it into Arena.

## Evidence

- Frozen prospective protocol and results: `logs/mimickit-training/fixed-support-goals-20261008-01`.
- Recomputed gates and provenance: `logs/mimickit-training/fixed-support-goals-review-20261008-01/report.json`.
- Goal-coverage audit: `logs/mimickit-training/fixed-support-goals-review-20261008-01/training-coverage.json`.
- Implementation: `run_fixed_support_goals.py`; verifier: `verify_fixed_support_goals.py`.

The label-source controller ran 2,748 private target queries for these physical diagnostic rollouts. Those rollouts are explicitly scripted feedback tests, not learned recovery. No model weights, Arena assets, physics constants or C# build changed.


## 2026-10-08: exact sustained arm-hold learning and queue ablation

See ARM_HOLD_LEARNING_PLAN.md and ARM_HOLD_LEARNING_PROGRESS.md. Both physical
teacher timing variants passed 8/8 and supplied 9,072 exact issue-time/raw-target
examples (6,804 training, 2,268 whole-pose held out). Full-input supervised
learners still achieved 0/8 supported holds. A matched learner with only its
queued-command inputs masked achieved 4/8 supported holds across two seeds;
returned Stand was 4/8 and 3/8. All remained 0/8 on the full pose-tracking gate.

This is a replicated partial support improvement, not an admitted arms-clear
skill or foot placement. Physical delay and the frozen Stand anchor's full
observations remain unchanged. The evaluated full and masked learners retained
16/16 exact quiet-Stand traces each. Completed 20,000 supervised updates and
96 physical cases. Arena f832bab8 unchanged; no export or C# compilation.

Next: teach the masked learner corrections for small recoverable deviations
around the held pose, requiring accurate tracking as well as support. Advance
to transfer only after the full learned hold-and-return gate passes. Evidence
is under logs/mimickit-training/arm-hold-*-20261008-01.

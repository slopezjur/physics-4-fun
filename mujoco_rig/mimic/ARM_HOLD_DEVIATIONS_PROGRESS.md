# Arms-clear deviation pilot

2026-10-08. Completed disturbance calibration and the bounded paired training comparison. Both continued learners regressed and are rejected. The earlier queue-masked 4/8-supported-hold checkpoint remains the experimental reference; it is not an admitted full skill or the Arena brain. Arena f832bab8 remains unchanged.

## Collection and admission

The first physical teacher run used bounded smoothed target disturbances only from 2.5 to 4.5 seconds during the arms-clear hold. Uniform innovations were bounded by +/-0.012 rad and smoothed with 0.85 previous / 0.15 innovation. Actual disturbances were inserted before the unchanged authority/fade/slew pipeline and clamped to existing target limits. No live positions or velocities were written.

That run completed only 1/8 full hold-and-return cases. Four cases held support and two returned to Stand, but only pose 0 met every gate. No held-out pose completed the sequence, so the training prerequisite assertion stopped the run. No optimizer update ran on that rejected collection. The failed run and its exclusion are retained.

A successor protocol allowed at most two smaller bounds, 0.003 and 0.00075 rad, requiring all eight teacher cases to complete. The 0.003-radian innovation bound passed 8/8, so the smaller alternative was not run. Its maximum actual raw-target disturbance was 0.00174242 rad (about 0.10 degrees); the innovation bound is not the measured disturbance amplitude.

During this admitted run, the maximum COM displacement relative to the undisturbed teacher trajectory was 0.00068876 m and the maximum controlled-joint displacement was 0.03906643 rad. States before the disturbance window exactly matched the original teacher trace. These are small motor-command deviations, not ball-impact recovery or evidence of a larger recovery basin.

Retained 1,440 exact correction examples from 2.5 to 5.5 seconds, including disturbance-free recovery: 1,080 training examples from poses 0-5 and 360 held-out examples from entire poses 6-7. Clean corrective targets were saved separately from the noisy applied targets. The independent audit verified this distinction and recomputed the target-command composition, with maximum error 0 rad. The original unsuccessful run was not added to training.

## Matched learning result

Both arms started from the same earlier queue-masked checkpoint, with fresh Adam state, seed 481028, fixed normalization, 4,000 updates, batch 512 and learning rate 0.0001. Control replayed the existing exact hold examples. Treatment used half existing examples and half newly admitted corrections. Initial weights and common sampling draws were matched; pose holdouts stayed excluded from fitting.

Replay continuation: 1/8 supported holds, 0/8 returned Stand, 0/8 full skills. Its held-out correction-target RMSE was 0.025685 rad.

Correction treatment: 0/8 supported holds, 0/8 returned Stand, 0/8 full skills. Its held-out correction-target RMSE improved to 0.024146 rad while physical behavior regressed. Existing-example held-out RMSE was 0.027309 rad versus 0.024355 for replay.

The treatment passed 16/16 30-second quiet cases with issued commands and qpos exactly matching the original Stand anchor via the quiet-goal bypass. That safeguard does not make active arm control successful.

No replication or changed-timing learner evaluation was warranted because no full-skill success was gained. No limits or tracking gates were relaxed. Both new candidates are rejected; retain the preceding experimental checkpoint for subsequent comparisons.

## What this changes

The physical label source has a limited demonstrated tolerance to command disturbances: the two tested amplitudes gave very different outcomes. We found admissible nearby-state data, but adding it did not improve this supervised learner. Replay-only continuation also damaged the earlier partial balance result. These outcomes do not prove a unique cause, but they do not support another longer run of the same pointwise fitting objective.

Next priority: a bounded on-policy learning comparison using the existing partial learner, with reward and evaluation tied to actual supported pose tracking and successful return over a rollout. First verify that the training adapter reproduces the unchanged checkpoint at zero updates. Preserve quiet bypass, original physical limits and whole-pose holdouts. Use the current 4/8 supported-hold reference as the non-regression comparator. No transfer, lifting or stronger hits until the complete arms-clear gate passes.

## Accounting and evidence

48 physical cases: two eight-case teacher batches, two eight-case learner batches and sixteen quiet cases. Two supervised fits, 8,000 updates total. The rejected initial collection ran zero updates. No PPO or impact training, GPU parity, Arena export or C# compilation ran. Physical gates, dataset provenance, clean/applied targets, initial weights and quiet trace identity were independently verified.

- Frozen initial plan: `ARM_HOLD_DEVIATIONS_PLAN.md`.
- Rejected initial collection: `logs/mimickit-training/arm-hold-deviations-20261008-01`.
- Frozen calibration protocol and admitted paired comparison: `logs/mimickit-training/arm-hold-deviations-calibration-20261008-01`.
- Independent audit: `logs/mimickit-training/arm-hold-deviations-review-20261008-01/report.json`.


## 2026-10-08 - Bounded arms-clear PPO pilot

See ARM_HOLD_PPO_PROGRESS.md for the latest completed step. Two independent 24,576-transition runs improved supported holds from 4/8 to 5/8 and 6/8; both returned Stand in 5/8, but complete skills remain 0/8. Quiet retention passed 16/16 per run with exact anchor traces. Every PPO cycle hit the actor KL stop; correct update size and inspect reward contributions before extending training. Arena remains unchanged. Audit: logs/mimickit-training/arm-hold-ppo-review-20261008-01/report.json.

# PPO update-size comparison - 2026-10-08

## Decision

Reducing actor learning rate from 1e-5 to 3e-6 reduced measured policy-change spikes and allowed more optimizer steps per collected batch, but did not improve physical skill success in this bounded comparison. Both lower-rate candidates are rejected for promotion. Complete arms-clear hold-and-return success remains 0/8 for every final actor.

The selected Arena Stand/Perturb anchor remains f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d. No Arena export, stronger-hit training, foot lifting or C# build occurred.

## Experiment integrity

Four runs used the same retained queue-masked seed-451028 actor, fresh optimizer state, fixed normalization and exploration standard deviation 0.005 rad. Two seeds, 501028 and 511028, were paired across the two rates. Each run collected exactly 24,576 transitions in 32 batches of 768. Poses 0-5 supplied 12 active worlds; whole poses 6 and 7 remained excluded from both losses. Physical limits, zero passive springs, two-command delay, command fade/slew and quiet bypass were unchanged.

The training success gate now uses the frozen evaluator's fixed double-precision hold target, time arithmetic and final return-frame count. Before learning, deterministic physical parity passed all 540 frames with zero qpos/command difference. Replaying saved failed-learner and successful-teacher traces through the corrected gate reproduced 0/8 and 8/8 outcomes. No additional teacher simulation or fitting occurred.

Full-batch analytic Gaussian KL was measured after every actor update without altering the existing minibatch stopping rule. The high-rate controls reproduced the previous pilot's final actor weights and every saved nominal trace array bit for bit for both seeds. Thus the gate correction and diagnostics did not change these control trajectories or optimization results.

## Frozen physical results

- Seed 501028, control 1e-5: 5/8 supported holds, 5/8 returned Stand, 0/8 complete.
- Seed 501028, lower 3e-6: 5/8 supported holds, 3/8 returned Stand, 0/8 complete.
- Seed 511028, control 1e-5: 6/8 supported holds, 5/8 returned Stand, 0/8 complete.
- Seed 511028, lower 3e-6: 4/8 supported holds, 4/8 returned Stand, 0/8 complete.

The starting reference remains 4/8 supported holds and 4/8 returns, with 0/8 complete. No final actor qualifies as a complete learned arms-clear skill. Both lower-rate actors still exceeded the 0.05-rad motor-RMSE and 0.02-m COM-error hold gates in every pose.

For seed 501028, the lower rate lost return success in poses 5 and 6 compared with its control. Its supported poses were 0, 1, 2, 6 and 7: it gained support in pose 0 and lost it in pose 5. For seed 511028, it lost returns in pose 5; supported poses were 0, 2, 4 and 7, gaining pose 2 but losing poses 1, 3 and 6 relative to control. These are mixed per-pose changes, not uniform progress.

All four actors passed all sixteen 30-second quiet cases. Quiet qpos and issued commands exactly matched the original anchor trace through the explicit bypass. This verifies retention of quiet Stand, not successful active-arm control. No actor qualified for the prospectively gated varied-timing evaluation.

## Update-size evidence

At seed 501028, actor gradient updates increased from 51 to 172; maximum post-update full-batch KL fell from 0.06117 to 0.01830. At seed 511028, updates increased from 68 to 177 and maximum KL fell from 0.04709 to 0.01492. Corresponding 95th-percentile KL values fell from 0.02895 to 0.01160 and from 0.02336 to 0.01171.

Early stopping still occurred in 32/32 and 29/32 lower-rate cycles, versus 32/32 in both controls. The lower rate makes individual changes smaller, allowing more updates before the stop. Neither method imposes a strict full-batch KL cap: the diagnostic is observational, the stop uses a sampled minibatch, and preceding updates are not undone. This comparison does not establish lower learning rate as a better behavioral configuration.

## Reward evidence

Hold-phase statistics use the entire commanded hold from 2.5 to 5.5 seconds, across active training worlds only. The frozen admission gate still uses its established final two-second hold window. Training trajectories differ between arms, so these are descriptive statistics, not measurements on matched states.

Tracking's weighted mean contribution was 0.0211 and 0.0218 in controls, versus 0.0266 and 0.0231 at the lower rate; its maximum is 0.45. The contribution was below 1% of its maximum in 56.9% and 56.4% of control hold samples, and 54.6% and 58.5% of lower-rate samples. COM reward was below 1% of its maximum in 39.0-45.5% of control samples and 41.3-42.5% of lower-rate samples.

Supported-posture reward averaged 0.133-0.151, substantially larger than mean tracking reward. Foot retention averaged 0.055-0.061. No run earned a complete-skill bonus. Every logged component sum, including the replacement penalty on physical failure, reconciled with the reward used for training.

These observations support testing whether the narrow exponential tracking reward provides too little distinction between the currently visited inaccurate poses. They do not prove that reward shape is the only cause. Smaller updates alone did not resolve the skill, and longer training is not yet justified by this comparison.

## Next bounded priority

Compare the present distance-reward kernels with a broader training-only kernel, starting again from the same retained reference and keeping optimizer settings fixed between arms. Use a single common width multiplier for motor and COM kernels (1 versus 2), leaving reward weights, foot/support terms, exploration and physical execution unchanged. The wider scales would be 0.10 rad and 0.04 m for training reward only; final admission remains 0.05 rad and 0.02 m without relaxation.

Use 3e-6 consistently in both reward arms to study reward shape under the smaller measured updates, not because it won this behavioral comparison. Retain both original high-rate outcomes as context. Pair the same two seeds and 24,576-transition budgets, continue full-batch KL and reward diagnostics, and admit only actual complete-skill improvement. Do not add a longer training duration or other hyperparameter changes to the same experiment. If the broader reward also fails, inspect action/state responsiveness and credit assignment before another sweep.

## Accounting and evidence

98,304 physical training transitions, 468 actor gradient updates and approximately 129.8 seconds of learning across four runs. New physical evaluations: eight zero-update parity cases, 32 nominal cases and 64 quiet cases, totaling 104. Sixteen old trace cases were replayed through the corrected gate without new simulation. The run directory occupies 34,905,583 bytes, about 34.91 MB; only initial/final actor weights, final critics and compressed evaluation traces were retained.

- Frozen plan: `ARM_HOLD_PPO_RATE_PLAN.md`.
- Collector: `train_arm_hold_ppo_rate.py`.
- Sealed sources, protocol, checks, diagnostics and final results: `logs/mimickit-training/arm-hold-ppo-rate-20261008-01`.
- Independent review: `logs/mimickit-training/arm-hold-ppo-rate-review-20261008-01/report.json`.
- Verifier: `verify_arm_hold_ppo_rate.py`.

Review passed sealed input/output hashes, identical starting weights, finite final weights, unchanged normalization, exact high-rate replication, transition/update accounting, reward reconciliation, quiet identity and independently recomputed physical gates. Holdout exclusion is established by the sealed collector's explicit training indices. Raw training trajectories were not retained for independent replay.


## 2026-10-08 - PPO reward-width comparison completed

See ARM_HOLD_PPO_REWARD_PROGRESS.md. Four matched runs compared motor/COM reward width multipliers 1 and 2 at actor rate 3e-6, 24,576 transitions each. Broader reward improved returns from 3/8 to 4/8 and 4/8 to 6/8; holds changed 5/8 to 4/8 and 4/8 to 5/8. Complete skills remain 0/8. Near-zero tracking rewards fell from 55-58% to below 1.5%. All 64 quiet cases passed exact anchor traces. Left upper-arm error closely matches an issued-command offset; next priority is a matched action-to-joint-response diagnostic, not another sweep. Arena unchanged. Audit: logs/mimickit-training/arm-hold-ppo-reward-review-20261008-01/report.json.

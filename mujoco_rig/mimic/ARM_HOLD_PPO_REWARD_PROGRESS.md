# PPO reward-width comparison - 2026-10-08

## Decision

Broader distance rewards improved return-to-Stand counts in both seeds and largely removed near-zero tracking rewards. Neither seed completed a full arms-clear hold-and-return skill. Keep this as partial learning evidence, not a promoted Arena brain or permission to advance to foot lifting and stronger hits.

The next priority is a controlled action-to-joint-response diagnostic, focusing first on the left upper arm. No additional reward sweep or long training run was started.

## Frozen comparison

Four runs started from identical retained queue-masked seed-451028 actor weights and fresh optimizers. Each used 24,576 transitions, actor rate 3e-6, fixed exploration standard deviation 0.005 rad, the same critic/PPO settings and training poses 0-5. Entire poses 6 and 7 were excluded from both losses.

Only motor/COM training reward widths changed: multiplier 1 used 0.05 rad / 0.02 m; multiplier 2 used 0.10 rad / 0.04 m. Reward weights, foot/support rewards, physical adapter, frozen Stand anchor, zero passive springs, actual two-command delay and quiet bypass stayed fixed. Final admission remained at 0.05-rad motor RMSE, 0.02-m COM error and 0.02-m foot displacement with supported hold and return to Stand.

Both reward adapters reproduced the unchanged reference's full 540-frame qpos and issued-command traces exactly before learning. The corrected success gate reproduced saved failed-learner and successful-teacher outcomes of 0/8 and 8/8. Both width-1 controls reproduced the preceding lower-rate study's final actor weights and every nominal trace array bit for bit.

## Physical results

- Seed 501028, narrow control: 5/8 supported holds, 3/8 returned Stand, 0/8 complete.
- Seed 501028, broader reward: 4/8 supported holds, 4/8 returned Stand, 0/8 complete.
- Seed 511028, narrow control: 4/8 supported holds, 4/8 returned Stand, 0/8 complete.
- Seed 511028, broader reward: 5/8 supported holds, 6/8 returned Stand, 0/8 complete.

The original retained reference remains 4/8 holds and 4/8 returns, with 0/8 complete. The earlier higher-rate PPO reference achieved 5/8 and 6/8 holds with 5/8 returns in both seeds. Thus the broader reward is not a uniform winner across all partial metrics or poses.

For seed 501028, broadening gained return success in pose 6 but lost supported hold in pose 7. For seed 511028, it gained returns in poses 2 and 5; supported holds gained poses 1, 5 and 6 while losing poses 0 and 4. Broader seed 511028 held and returned in both held-out poses; broader seed 501028 returned in both but did not hold pose 7 throughout the required window.

All sixteen poses across the two broader-reward evaluations still failed the motor tracking gate. Best peak motor RMSE per broader run was 0.06776 and 0.07545 rad, above the 0.05-rad requirement. COM/support/foot failures remain in other cases. None qualified for the prospectively gated changed-timing evaluation.

All four actors passed sixteen 30-second quiet cases each, with qpos and issued commands exactly equal to the original anchor trace through the quiet bypass. These 64 quiet successes do not demonstrate active-arm goal tracking. No new candidate was exported or loaded into Arena.

## What the reward diagnostic resolved

During the commanded hold phase, the fraction of active training samples receiving less than 1% of maximum tracking reward fell from 54.6% to 1.43% in seed 501028, and from 58.5% to 0.39% in seed 511028. The corresponding COM fractions fell from 41.3% to 11.3% and from 42.5% to 6.0%.

Mean weighted tracking reward increased from 0.0266 to 0.1321 and from 0.0231 to 0.1366. This is partly expected from changing the reward formula; it is not itself a physical-performance improvement. These summaries also sample different visited states in each run. Return counts provide the actual partial behavioral improvement, while complete skill success remains zero.

Broader runs performed 172 actor updates each, versus 172 and 177 in controls. Maximum observed full-batch post-update KL was 0.01820 and 0.01464 for the broader runs. All their 32 cycles triggered the existing sampled minibatch early-stop rule. No complete-skill bonus was awarded. Training physical-failure counts were 21 and 19, versus control 28 and 21, within the same transition budget.

## Joint-level follow-up from saved evidence

In each learner's supported-hold cohort, the largest joint tracking error was UpperArm_L_ry (qpos index 17). For the two broader actors, its RMS actual-to-goal error was 0.4712 and 0.4115 rad; RMS issued-command-to-goal error was nearly the same, 0.4705 and 0.4117 rad. Mean actual offsets were -0.3820 and -0.3542 rad versus command offsets -0.3835 and -0.3552 rad. None of these issued commands was within 0.0001 rad of the joint's configured positional limits.

This points to an offset already present in the issued command and prioritizes investigation of the learner and command path. It does not prove a unique cause, rule out other limits, or substitute for an intervention. Cohorts contain different supported poses, and compensating command offsets can be legitimate in a dynamic system. The physical teacher's saved successful trace tracked these goals much more closely; it is still a physical controller, not a learned policy.

A separate read-only comparison used 960 identical archived teacher hold inputs for the starting actor and every final actor. Broader actors changed outputs by only 0.000577 and 0.000434 rad RMS across all motors; maximum absolute changes were 0.002612 and 0.001730 rad. Left-upper-arm output changes were 0.001572 and 0.000743 rad RMS. These are measurements on fixed teacher states, not the learner's live trajectories, but show that the bounded updates moved outputs only slightly on this reference set. Small output changes can still produce noticeably different closed-loop outcomes.

This experiment does not establish that longer PPO training cannot work. It establishes that broadening the reward alone did not achieve the full skill within the matched budget, despite making tracking feedback less sparse.

## Next bounded diagnostic

Use the retained reference to reach repeatable states during the ramp, hold and return. Compare an unchanged continuation with small positive/negative interventions on UpperArm_L_ry, at the current exploration scale first. Preserve the complete state, pending command queue, command-adapter memory and timing for each matched branch. Measure raw network output, authority/fade/slew, issued and delayed applied targets, torque, actual joint displacement and subsequent reward components. Keep every physical limit and final gate unchanged.

Establish whether intended action differences survive the adapter, when the joint responds, and whether the training signal distinguishes helpful from harmful directions. Then inspect the GAE/critic credit horizon against that observed response. Choose any change to exploration, update budget or command representation from those measurements; do not infer it merely from a higher reward or another hold count. No new training is required for the first response check.

## Accounting and evidence

98,304 training transitions and 693 actor gradient updates across four runs, taking approximately 134.0 seconds of learning. Sixteen zero-update parity cases, 32 final nominal cases and 64 quiet cases: 112 new physical evaluation cases. Old teacher/learner gate traces and joint tracking were analyzed without new simulation. The run directory is 34,824,912 bytes, about 34.82 MB, plus small review files.

- Frozen plan: `ARM_HOLD_PPO_REWARD_PLAN.md`.
- Collector: `train_arm_hold_ppo_reward.py`, reusing the sealed rate-study PPO implementation.
- Run: `logs/mimickit-training/arm-hold-ppo-reward-20261008-01`.
- Independent gate/provenance audit: `logs/mimickit-training/arm-hold-ppo-reward-review-20261008-01/report.json`.
- Saved-trace joint and fixed-input diagnostics: `logs/mimickit-training/arm-hold-ppo-reward-review-20261008-01/joint-tracking.json`.
- Review tools: `verify_arm_hold_ppo_reward.py` and `inspect_arm_hold_tracking.py`.

Audit passed sealed input/output hashes, identical initial weights, finite final weights, fixed normalization, exact control replication, transition/update counts, reward reconciliation, quiet identity and independent physical-gate recomputation. Holdout-loss exclusion follows the sealed explicit training indices; raw training trajectories were not retained for independent replay.

Arena remains on anchor f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d. No Arena modifications, export or C# compilation occurred.


## 2026-10-08 - Upper-arm command-response diagnostic completed

See ARM_ACTION_RESPONSE_PROGRESS.md. Matched +/-0.005-rad one-frame probes at hold and return were completely erased by the existing slew limiter in all eight poses. Baseline left-arm slew saturation was 85.6% in hold and 80.3% in return, with no torque saturation. Twelve-frame biases reached six hold poses and all return poses; the two-frame queue and PD torque reconstruction passed exactly. No training or Arena change. Next: test corrections around the already rate-limited command while reapplying the same feasible bounds; require zero-correction parity before any learner change. Audit: logs/mimickit-training/arm-action-response-review-20261008-01/report.json.

# Arms-clear PPO pilot - 2026-10-08

## Decision

Actual-rollout PPO improved aggregate supported-hold and return counts in both sampling seeds. Neither final learner completed a single full arms-clear hold-and-return skill. Both remain experimental; no Arena promotion, foot lifting, transfer or stronger-hit training is justified by this result.

The frozen Stand/Perturb anchor remains f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d. The existing Arena bundle was not changed. No export or C# compilation ran.

## Matched start and physical equivalence

Both runs began with exactly the retained queue-masked seed-451028 actor weights and fresh optimizer state. This is continuation of a partially trained auxiliary skill, not two fresh Stand brains. The zero-update adapter reproduced all 540 frames of the retained nominal trace: qpos and issued commands matched bit for bit, including active-world masks.

Keep the frozen anchor, 33 motor limits, zero passive springs, two-command delay, quiet bypass, authority/displacement fade and 0.02-radian per-frame addition slew. The auxiliary actor is QueueBlindLearner; loading its checkpoint with the ordinary Learner class would change behavior. No physical teacher or supervised target loss was used during this pilot.

Each seed collected 24,576 physical transitions in 32 batches of 768, with 12 active worlds representing two instances of poses 0-5. Entire poses 6 and 7 were excluded from actor and critic losses. Training used nominal timing only. Final evaluations used deterministic mean actions on all eight poses.

## Frozen outcomes

- Starting reference: 4/8 supported holds, 4/8 returned Stand, 0/8 complete skills.
- Seed 501028: 5/8 supported holds, 5/8 returned Stand, 0/8 complete skills.
- Seed 511028: 6/8 supported holds, 5/8 returned Stand, 0/8 complete skills.
- Both final policies: 16/16 quiet cases maintained Stand for 30 seconds. Qpos and issued commands exactly matched the original quiet-anchor trace through the explicit quiet bypass.

Both seeds held and returned in held-out poses 6 and 7; the starting reference did so only in pose 6. This is encouraging generalization of partial balance, not successful goal tracking. Each seed also lost an earlier supported hold: seed 501028 lost pose 0, and seed 511028 lost pose 5. Higher aggregate counts therefore do not establish per-pose non-regression.

Every nominal pose in both final policies exceeded both the 0.05-radian peak motor-RMSE gate and the 0.02-metre peak COM-error gate during the hold window. Among supported holds, seed 501028 had peak motor RMSE of 0.0862-0.1705 radians and peak COM error of 0.0457-0.0617 metres; seed 511028 had 0.0839-0.1802 radians and 0.0342-0.0708 metres. Seed 511028 also exceeded the foot-displacement gate in pose 6 (0.02052 metres versus 0.02). No gate was relaxed.

Varied-timing evaluation was prospectively conditional on improved complete-skill success. Neither seed qualified, so it was not run. Retain the previous actor as the reference and both new actors as bounded experimental evidence; do not declare a production winner from partial counts.

## Optimization findings and limitation

The runs took approximately 33.68 and 33.23 seconds of learning each. They performed 51 and 68 actor gradient updates, respectively, rather than all 288 possible minibatch updates per seed. The actor's approximate-KL stop triggered in every collection cycle. Peak checked values were 0.0651 and 0.0372 against the 0.01 stop threshold.

This check prevents subsequent updates when the sampled minibatch exceeds the threshold; it does not undo a preceding update or impose a hard bound on the entire batch. Thus the nominal small learning rate still produced changes large relative to the fixed 0.005-radian exploration distribution. Lowering the rate and measuring whole-batch divergence after updates is the next optimization priority.

No training episode received the complete-skill bonus. The narrow exponential tracking rewards can be very small at the observed errors, so increased supported-hold counts alone cannot establish that goal tracking is learning effectively. Inspect reward contributions before increasing the training duration or changing several reward weights at once.

The audit also found a minor mismatch with the plan's claim of identical training/evaluation success tests: training uses the look-ahead goal at the hold/return boundary, whereas frozen evaluation uses the fixed hold pose. One boundary frame differed by at most 1.61e-10 radians before float32 conversion. Training also uses float32 targets versus the evaluator's saved goal precision. No complete-skill bonus was awarded, and the independently recomputed frozen outcomes are unaffected. Make the success-test target and window definitions identical in the next collector revision; preserve this run's sealed source.

## Next bounded step

Create a successor protocol that first aligns the training success test with frozen evaluation and verifies unchanged-policy parity. Compare a lower actor learning rate against this pilot's rate with the same starting actor, transition budget, seeds, reward and exploration scale. Log whole-batch post-update KL and separate tracking, COM, support and return reward contributions. Keep the physical admission gates and whole-pose holdouts unchanged. Advance to varied timing only after complete-skill success improves, and inspect per-pose regressions before promotion. Do not extend to a long compute run on the basis of supported-hold counts alone.

## Accounting and evidence

49,152 training transitions and 119 actor gradient updates across two independent runs. Eight unchanged-policy parity cases plus 16 final nominal cases and 32 quiet cases: 56 physical evaluation cases in this pilot. The verification script recomputed the existing baseline's eight saved cases without new simulation. New run evidence occupies 19,359,885 bytes (about 19.36 MB), including source snapshots, initial/final actors, final critics and compressed evaluation traces.

- Frozen protocol: `ARM_HOLD_PPO_PLAN.md` and `logs/mimickit-training/arm-hold-ppo-20261008-01/protocol.json`.
- Run results: `logs/mimickit-training/arm-hold-ppo-20261008-01/report.json`.
- Independent trace and provenance audit: `logs/mimickit-training/arm-hold-ppo-review-20261008-01/report.json`.
- Audit implementation: `verify_arm_hold_ppo.py`.

The audit verified sealed input/output hashes, identical initial actors, finite final weights, unchanged normalization, transition/update accounting, quiet trace identity and every nominal hold/return/tracking gate. Held-out-loss exclusion follows the sealed collector's explicit training indices; raw training trajectories were not retained for independent replay.


## 2026-10-08 - PPO learning-rate comparison completed

See ARM_HOLD_PPO_RATE_PROGRESS.md. Four matched runs compared actor rates 1e-5 and 3e-6, 24,576 transitions each. Controls exactly reproduced prior weights/traces. Smaller updates reduced KL spikes but returned Stand in only 3/8 and 4/8 versus control 5/8 in both seeds; complete skills remain 0/8. All 64 quiet cases passed with exact anchor traces. Tracking reward was below 1% of its maximum in 55-58% of hold samples. Next bounded priority: broader training-only distance rewards with unchanged final gates. Arena unchanged. Audit: logs/mimickit-training/arm-hold-ppo-rate-review-20261008-01/report.json.

# Early-recovery actor weighting

Approved successor to RECOVERY_UPDATE_TRACE_PROGRESS.md, 2026-10-09.

Train fresh seeds 921009 and 931009 from the identical initial actor (920009),
24,576 continuation transitions per seed. Change only the actor objective:
raw weight 5 for continuation age below 30 intervals, otherwise 1. Normalize
over the entire 768-sample collection to mean one, then multiply the existing
clipped PPO surrogate. Do not renormalize individual minibatches. Age and force
labels are optimizer evidence only; neither is added to the policy observation.

Keep half-noise innovations, full episodes, starts and reset distribution,
reward, GAE, advantage normalization, unweighted critic loss, learning rates,
three epochs and the existing KL guard. Keep the frozen source controller.

Save compact age/force/weight and actual minibatch objective evidence for all
32 collections per seed. Record effective early/late and quiet/signed-push
exposure. Independently verify actor weighting and unweighted critic losses,
probability scale, GAE, reset accounting and initial collection parity against
the corresponding unweighted half-noise run. Keep only initial/final actor,
final critic and first/last full collection batches, without intermediate models.

Use the established 45-trace / 720-case evaluation: continuous execution,
matched recovery starts, heldout timings and poses, signed 150/300 N X/Y forces,
sampled core pushes and 60-second quiet standing. Audit control delay, physics,
foot geometry and actual steps. Compare paired gains/losses against both the
corresponding unweighted half-noise learner and zero correction. Measure the
first half-second ankle addition and difference from matched quiet execution.

This also changes effective branch exposure and is not a pure isolation of
timing from branch occupancy. Better reward, larger response or a single seed
does not justify promotion. No Arena replacement or automatic longer run.

Evidence destination: logs/mimickit-training/recovery-age-weighted-learning-20261009-01.

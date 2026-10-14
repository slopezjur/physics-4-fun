# Saved first-collection PPO gradient audit

Approved successor to RECOVERY_HALF_NOISE_LEARNING_PROGRESS.md, 2026-10-09.

Use the first saved 64-by-12 collection from each half-noise seed. No new
physical rollouts, extended training, production edits or Arena changes.

Reconstruct initial and asynchronous reset assignments from the separate reset
RNG, ordered reset records and terminal masks. Records are sorted by source
onset within a reset call, not simply by world. Verify exact records, cleared
latent memory on reset and preserved sampled memory otherwise. Force labels
are for analysis only and must not be supplied to the actor.

Reconstruct the global RNG by creating the original initial critic and replaying
all 64 full 16-world Gaussian draws. Require exact sampled actions. Replay the
original three epochs / nine minibatch updates, including gradient clipping,
Adam and the KL guard. Require exact saved post-update actor means, critic
values and recorded KL values before interpreting the result.

At the initial actor, measure ascent score-function gradients separately for
quiet, +X and -X samples, weighted by their actual share of the 768-sample batch.
Verify their sum and compare with an independent analytic last-layer gradient.
Measure alignment/cancellation and the predicted response across each branch.
Separate explicit output bias from hidden features; also report the constant
and varying parts of the actual change in conditional mean. Parameter-gradient
norm fractions are not fractions of physical behavior.

Audit both seeds, paired with their original recorded batches and reset history.
Save compact gradients, hidden features, branch labels, permutations and replay
outputs; do not save additional policy checkpoints. Interpret only this first
collection. Its critic was zero and these results cannot establish the cause of
later learning failure or identify a universally correct ankle direction.

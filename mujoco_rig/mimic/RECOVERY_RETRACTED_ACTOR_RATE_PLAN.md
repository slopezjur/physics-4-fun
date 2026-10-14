# Fixed actor learning-rate comparison

Date: 2026-10-09. Approved successor to the next priority in
RECOVERY_RETRACTED_LEARNING_PROGRESS.md. Compare actor Adam learning rate
1e-3 against the sealed 1e-4 finite-correction controls. This is one fixed
contrast, not a rate search or a claim of proportional learning speed.

Train fresh seeds 921009 and 931009 from the same zero head initialized with
seed 920009, for 24,576 continuation transitions each. Keep critic Adam at
3e-4, all 32 collection sizes, three epochs, 256-sample minibatches, 288
nominal actor slots, actor weighting, advantage normalization, reward,
noise, action memory, reset distribution and frozen Stand parent unchanged.

Reuse the sealed collection and critic implementation. Set only the fresh
actor optimizer's learning rate before its first recorded hash or update;
its state must still be empty. Record the effective rate in every attempt
and fit report. Do not patch the global Adam constructor.

Retain the pre-step Jacobian projection of actual clipped-Adam displacement
and exactly one candidate-Jacobian correction toward pre-step quiet outputs.
Keep relative SVD cutoff 1e-6, float64 algebra with float32 displacement
addition, original zero-anchor deviation limit 1e-5 and collection-old KL
limit 0.01. Keep independent Gram gates, final-only acceptance, actor/Adam
rollback and stopping the remaining actor slots in a rejected collection.
The critic still executes every slot. No retries, extra updates, relaxed
guards, line search or surrogate acceptance gate.

Verify exact first collection, advantages, weights and critic parity with
the sealed low-rate retracted controls, and all 32 minibatch permutations.
Independently replay first-collection actor, Adam and critic math using the
declared new rate. Verify subsequent compact accounting and the complete
physical suite. Compare individual cases against low-rate retracted,
projected-only, weighted and zero controls.

Run the established 45-trace / 720-case evaluation: continuous and matched
recovery starts, both hit timings, reserved poses, signed 150/300 N X/Y
pushes, sampled core cases and sixty-second quiet Stand. Record footsteps
and first-500 ms pushed-minus-quiet ankle commands. Recompute update-scale
diagnostics for both rates, including KL, ratio tails and actual surrogate
clipping, gradient clipping, correction size and saved first/last batch
surrogate changes. Rejected/skipped updates remain outcomes.

Snapshot current sources and verify every historical control output before
running. Preserve the exact documented old PILOT diagnostic-source exception
and its archived original; all modern shared sources must match. Keep only
initial/final actor and final critic per seed, compact update records and
first/last full batches. Preserve all prior evidence and sealed sources.

Require reproducible physical recovery gains over low-rate and zero controls
with quiet retention before promotion. Bigger updates or ankle commands
alone do not qualify. No automatic Arena replacement or longer continuation.

Evidence: logs/mimickit-training/recovery-retracted-actor-rate-20261009-01.

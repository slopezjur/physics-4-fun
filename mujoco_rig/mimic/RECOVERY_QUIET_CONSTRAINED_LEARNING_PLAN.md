# Bounded quiet-constrained recovery learning

Approved successor to RECOVERY_QUIET_ACTOR_PROJECTION_PROGRESS.md. Train two
fresh seeds, 921009/931009, from the same original zero-correction head (seed
920009), 24,576 continuation transitions per seed. Keep the age-weighted
recipe and compare against its sealed corresponding controls.

## Fixed training and intervention

Retain half-noise standard deviation, latent correlation, reset distribution,
full episodes, reward, 5:1 early actor weighting normalized per collection,
ordinary advantage normalization, unweighted critic, actor/critic learning
rates, three epochs, 256-sample minibatches and collection KL limit 0.01.
Training onsets remain 40-90 excluding 55-65; poses remain 0-5. No runtime
force label, hit schedule or new actor input is introduced.

The only intervention is the constrained actor update. Compute 12 quiet-output
Jacobians at the pre-step actor. Clip its PPO gradient as before and let Adam
propose a parameter displacement. Project the ACTUAL displacement through the
quiet Jacobian nullspace in float64 using relative singular cutoff 1e-6, then
apply it from pre-step float32 parameters without renormalization. Projecting
only the raw gradient would not constrain Adam's coordinate-scaled update.

Quiet features come from the already-audited source-derived poses 0-5 at
onsets 44/74. Verify exact feature equality between both source datasets and
zero previous-action memory. Targets are recomputed from the fresh initial
head and must be zero. Do not use the rejected final actors' saved means.
These are fitted anchors; they do not stand in for full quiet evaluation.

Accept only if the whole-collection KL remains at most 0.01, all relevant
values are finite, and cumulative maximum absolute anchor deviation from the
initial target stays at most 1e-5 latent units. Otherwise restore both actor
and optimizer and stop actor updates for that collection. Critic updates
continue unchanged. Retain Adam's tentative moments only for accepted steps.
Do not compensate with extra epochs, amplified projection or changed rates.

## Checks and retained evidence

Before training, verify source/initial actor parity, exact recovery-start
restoration, probability scale, actor weighting and anchor provenance. During
each attempted update, independently verify the projection through the quiet
Gram matrix. Save compact geometry, anchor, acceptance and actor/optimizer hash
records. No per-step actor, optimizer or full-Jacobian files are retained.

Keep the existing objective/critic evidence for all collections and complete
first/last batches. Require exact first-collection and post-critic parity with
the matching weighted controls, then independently replay the first constrained
collection. Verify later objective accounting, reset labels, projection gates
and cumulative anchor bounds. Retain only initial/final actor and final critic
checkpoints, plus compact diagnostics and the final physical traces.

## Physical evaluation and decision

Run the established 45-trace / 720-case suite: ordinary continuous execution,
matched recovery starts, reserved poses/timings, signed 150/300 N X/Y pushes,
sampled core pushes and 60-second quiet Stand. Hit timings 1.0/1.7 seconds remain
reserved. Audit delayed controls, motor limits, memory, foot geometry, recovery
and actual footsteps. Compare individual gains/losses to the matching weighted
controls and zero correction, and measure early pushed-minus-quiet response.

A successful anchor bound or larger ankle response is not a recovery result.
Require reproducible physical gains and quiet retention before promotion.
No Arena replacement or longer run follows automatically. Preserve all prior
sealed implementations and evidence.

Evidence: logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01.

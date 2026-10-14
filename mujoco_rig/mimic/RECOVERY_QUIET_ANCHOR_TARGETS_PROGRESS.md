# Local quiet-target correction: both fixed candidates pass

Date: 2026-10-09. Implements RECOVERY_QUIET_ANCHOR_TARGETS_PLAN.md.
Both local-target corrections pass the original quiet-anchor and KL limits
while retaining positive full-collection and minibatch PPO surrogate gains.
The independent audit passes on the first attempt. This supports one bounded
fresh-learning comparison; it does not establish improved physical recovery.
No policy is promoted and Arena remains unchanged.

## Fixed comparison

Reuse the two sealed ninth-update candidates from seeds 921009/931009, their
pre-update and zero-corrected parameter vectors, and their candidate quiet
Jacobians. The only new intervention changes the correction target from zero
to each actor's pre-update quiet outputs. Apply one minimum-norm correction
with relative SVD cutoff 1e-6 and one float32 displacement addition.

No optimizer is constructed or modified, and no historical optimizer replay
is repeated. All three prior actors reproduce their saved hashes,
predictions, probability ratios and scalar metrics exactly. The new arm uses
the identical saved collection, ninth minibatch, advantages, actor weights,
old likelihoods and collection-old means.

The local correction target is not the acceptance target. Acceptance remains
maximum deviation from the ORIGINAL ZERO quiet targets <=1e-5 and
whole-collection KL <=0.01. No guard, retry count or parameter changed.

## Feasibility and objective retention

Seed 921009 preserves its pre-update quiet outputs within 7.9126039e-11.
Its final maximum deviation from zero is 9.7063548e-6, with KL 3.2267629e-8.
The local correction retains a full-collection surrogate gain of 1.9073486e-6
and a ninth-minibatch gain of 1.6689301e-6 over pre-update. These are about
71.9% and 59.6% of the rejected candidate's gains. The zero-target correction
instead lost 3.6358833e-6 and 7.0333481e-6 relative to pre-update.

Seed 931009 preserves pre-update quiet outputs within 1.0459189e-10.
Its final zero-target deviation is 9.0908898e-6, with KL 2.4275440e-7.
Full-collection and minibatch gains remain 5.8114529e-6 and 8.7916851e-6,
about 85.9% and 93.1% of the rejected candidate's gains. The zero-target arm
retained smaller positive gains, 2.2053719e-6 and 6.9141388e-6.

Local corrections still have an objective cost versus the rejected candidates:
7.4505806e-7 and 1.1324883e-6 for seed 921009; 9.5367432e-7 and
6.5565109e-7 for seed 931009, for full/minibatch respectively. Both remain
positive versus pre-update. Higher surrogate is better; it is a local PPO
objective, not measured recovery reward or a new physical success.

The actual correction norms are 6.7616743e-6 and 8.3353327e-6, approximately
0.04194% and 0.06074% of the rejected parameter displacements. Both candidate
Jacobians retain rank 12. The small norm is secondary evidence; actual
objective and output measurements establish the result.

## Response and interpretation

Quiet probe RMS movement relative to pre-update is 5.3792548e-11 and
5.8820747e-11 latent units. Conditional pushed RMS remains 3.0688617e-7 and
1.0740919e-6, versus rejected-candidate values 2.9490898e-7 and 1.0731499e-6.
The correction preserves the measured conditional response on these states.
These are fixed latent predictions, not ankle angles or physical trajectories.

Changing only the target avoids the first seed's objective reversal. Resetting
all accumulated quiet drift therefore mattered in that saved local comparison.
This does not establish that the earlier quiet drift was useful, nor that
preserving it is generally desirable. If correction is applied from the first
fresh training update, both zero and pre-update targets begin near zero.
The present candidates inherited eight accepted updates without finite
correction; a fresh learner will follow a different path.

## Verification and cost

The independent audit verifies 617 input hashes and eight output hashes,
exact parity of all three prior states, independently reconstructed candidate
Jacobians, the new correction and float32 parameter application, all four
states' predictions/objectives/ratios and all five change comparisons.
The Gram eigensystem agrees with the SVD corrections within 2.25e-18 maximum
absolute error and 1.30e-12 relative L2 error. Independent float64 probability
and objective discrepancies remain below 9.03e-8. No source repair was needed.

The runner took 0.93 seconds excluding process startup and separate audit.
Evidence occupies approximately 6.04 MiB. No new physical rollout, training
update, critic fit or deployable checkpoint was produced. Prior sources and
evidence remain unchanged.

Evidence root:
`logs/mimickit-training/recovery-quiet-anchor-targets-20261009-01/`.
See audit.json, results.json, contract.json, completed.json and each seed's
report.json/evidence.npz. The Jacobians and earlier parameter vectors remain
in the sealed predecessor evidence rather than being copied again.

## Next priority: bounded fresh learning with finite correction

Train fresh seeds 921009 and 931009 from the identical original zero-correction
head initialized with seed 920009. Use 24,576 continuation transitions each.
Retain the age-weighted recipe, half exploration noise, correlated action
memory, reset distribution, full episodes, reward, advantage normalization,
critic objective, learning rates, three epochs, 256-sample minibatches and
288 nominal actor slots per seed. Do not continue rejected final actors.

Keep the pre-step quiet-Jacobian projection of the actual clipped-gradient
Adam displacement. For EVERY finite projected candidate, form its current
quiet Jacobian and apply ONE minimum-norm correction toward that update's
pre-step quiet outputs. Use the same 1e-6 cutoff and explicit float32 addition.
Do not retry, line-search, rescale, add epochs or add a surrogate-improvement
acceptance criterion.

Evaluate the final corrected candidate against the original zero-reference
anchor bound 1e-5 and collection-old KL limit 0.01. An intermediate projected
candidate exceeding the anchor bound is not rejected before its correction;
that is the drift being repaired. Reject nonfinite candidates. On final guard
failure restore both actor and Adam and stop actor updates for the rest of
that collection; continue the unchanged critic updates. Retain tentative
Adam moments only on acceptance.

Record projection and finite correction separately: before/intermediate/final
outputs, correction size, local target residual, cumulative zero-target drift,
KL, acceptance/rejection reasons and skipped slots. Independently gate both
linear operations. Keep compact diagnostics, first/last complete batches and
initial/final actor plus final critic only, with no intermediate checkpoint
series.

Require exact first pre-learning collection, advantages, weights and initial/
post-collection critic parity with the corresponding historical controls.
Verify all 32 minibatch permutations. Updated actor outputs and later
trajectories are expected to differ. Independently replay the first updated
collection and audit later compact accounting and preserved inputs.

Then run the established 45-trace / 720-case physical suite: continuous and
matched execution, reserved poses/timings, signed 150/300 N X/Y pushes,
sampled core pushes, sixty-second quiet standing and measured footsteps.
Compare individual cases against projected-only constrained controls,
weighted controls and zero correction. More accepted updates and changed
directions remain coupled; an acceptance gain alone is not a recovery result.

Require reproducible physical gains with quiet retention before promotion.
No automatic Arena replacement or longer continuation follows this local
diagnostic. The current candidates remain rejected.

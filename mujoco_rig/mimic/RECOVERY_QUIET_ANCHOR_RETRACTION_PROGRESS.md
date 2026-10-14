# Finite quiet-anchor correction: feasible, objective preservation mixed

Date: 2026-10-09. Implements RECOVERY_QUIET_ANCHOR_RETRACTION_PLAN.md.
Both fixed rejected candidates were reconstructed exactly and corrected once.
Both corrections restore the quiet-anchor and KL limits. The independent
audit passes, but seed 921009 loses its proposed PPO objective improvement;
seed 931009 retains a positive improvement. This does not support another
training run or policy promotion yet. Arena remains unchanged.

## Exact experiment

For each seed, 921009 and 931009, replay the first collection's eight accepted
actor updates and ninth rejected candidate. Use the saved batch, minibatch
indices, normalized advantages, actor weights, old likelihoods and original
Adam initialization. Historical actor and optimizer hashes match exactly.

At the rejected candidate, compute the twelve current quiet-output Jacobians
and apply one minimum-norm Newton correction toward the original zero targets.
Use float64 SVD with relative cutoff 1e-6, cast the correction once to float32
and add it to the rejected float32 parameters. Keep tentative Adam moments
unchanged. No retries, line search, scale sweep or threshold change occurred.

Compare pre-ninth, rejected and corrected actors against identical saved data.
KL uses collection-old means. Higher weighted surrogate is better. Full and
minibatch objectives use their own actual float32 predictions. Fixed probe
movement is relative to the pre-ninth actor, not older dataset actor means.

## Constraint result

Seed 921009's maximum anchor deviation falls from 1.1095411e-5 to
4.6566129e-11. Corrected whole-collection KL is 3.0061269e-8.
Seed 931009's deviation falls from 1.1526415e-5 to 5.2386897e-11, with
corrected KL 2.3298922e-7. Both satisfy the unchanged 1e-5 zero-referenced
anchor bound and 0.01 KL limit.

Both candidate Jacobians retain rank 12. The actual correction norms are
5.8337862e-5 and 3.9366830e-5, only 0.36185% and 0.28689% of the actual
rejected-minus-pre displacement. Relative to the original tentative Adam
displacements, they are 0.36131% and 0.28661%. Corrected-versus-rejected
update cosines exceed 0.99999. These small parameter changes do not guarantee
small objective cost.

## Objective result

For seed 921009, the rejected candidate improves the full-collection weighted
surrogate by 2.6524067e-6 and the exact ninth-minibatch surrogate by
2.8014183e-6 over pre-update. After correction, the changes versus pre-update
are -3.6358833e-6 and -7.0333481e-6. The correction therefore removes more
than the candidate's proposed improvement on both comparisons.

For seed 931009, rejected-candidate gains are 6.7651272e-6 on the full
collection and 9.4473362e-6 on the minibatch. Corrected gains remain positive:
2.2053719e-6 and 6.9141388e-6. The repair costs 4.5597553e-6 and
2.5331974e-6 relative to the rejected candidate.

Independent float64 probability/objective reconstruction differs from the
reported float32 values by at most 9.03e-8, well below these changes. The
mixed objective outcome is not explained by scalar reduction rounding.

## Fixed-state response result

The correction changes cumulative quiet outputs back toward zero, including
drift from the eight earlier accepted updates. It does not merely remove the
ninth update's additional drift. Pre-ninth quiet maxima were 9.7062757e-6
and 9.0907852e-6, both already within the original bound.

Repair-only raw pushed RMS is 8.1149214e-6 and 6.4981231e-6 latent units,
nearly equal to repair-only quiet RMS, 8.1189173e-6 and 6.5003160e-6.
Repair-only pushed-minus-quiet RMS is much smaller: 4.7526583e-8 and
3.9449786e-8. On these fitted probes, most correction movement is shared
with quiet outputs.

Conditional pushed movement relative to pre-update remains after repair.
Its RMS changes from 2.9490898e-7 to 3.4017748e-7 for seed 921009, and from
1.0731499e-6 to 1.1121050e-6 for seed 931009. Thus the first seed's objective
loss does not mean that its fixed-probe conditional response disappeared.

These are latent-mean changes on 36 fixed diagnostic states: twelve quiet
and twenty-four pushed. They are not ankle angles, physical rollouts or
measured recovery. Historical preferred push signs came from older policies;
they do not establish optimal actions for the reconstructed actors. The
eight earlier updates' quiet drift is not assumed to be useful learning.

## Audit and cost

Independent verification passed on the first attempt: 614 input hashes,
10 output hashes, all 18 historical actor/Adam attempts, exact candidate
Jacobians, float32 application, predictions and probability ratios. The
independent Gram eigensystem agrees with the SVD corrections within
2.62e-17 maximum absolute error, with relative L2 errors below 1.87e-12.
Tentative optimizer states remain unchanged. No code repair was required.

The runner took 3.56 seconds excluding process startup and separate audit.
Evidence occupies approximately 12.46 MiB. It retains compact diagnostic
parameter vectors and Jacobians, with no deployable checkpoints. No new
trajectories, critic fitting or continued policy training occurred.

Evidence root:
`logs/mimickit-training/recovery-quiet-anchor-retraction-20261009-01/`.
See audit.json, results.json, contract.json, completed.json, and each seed's
report.json, replay.json and evidence.npz. Previous implementations and
evidence remain unchanged.

## Next priority: isolate new drift from accumulated drift

Use the same two fixed ninth candidates and the same candidate Jacobians.
Change only the correction target: use each actor's pre-ninth quiet outputs
instead of zero. Apply exactly one minimum-norm correction with the same
1e-6 cutoff and float32 addition. Preserve tentative Adam moments. Do not
retry, rescale or select a different candidate.

The global acceptance guard remains maximum deviation from the ORIGINAL
ZERO targets at most 1e-5, with whole-collection KL at most 0.01. A local
correction target does not relax or replace that global guard.

Compare this arm against pre-update, rejected and the sealed zero-target
correction, using the same full/minibatch objectives and fixed-state probes.
Independently verify hashes, correction geometry, actual residuals, parameter
application and predictions. Report both candidates even if either fails.

This isolates the objective cost of resetting accumulated quiet drift from
the cost of removing only the ninth update's new drift. It does not establish
a better fresh-training algorithm: corrections applied from the first update
would start with pre-update quiet outputs near zero. Decide from this bounded
comparison before designing another learner. No automatic physical run,
training continuation or Arena replacement follows.

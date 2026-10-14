# Quiet-constrained recovery learning: completed, candidates rejected

Date: 2026-10-09. Implements RECOVERY_QUIET_CONSTRAINED_LEARNING_PLAN.md.
Both fresh seeds completed 24,576 continuation transitions and the established
45-trace / 720-case physical suite. Independent verification passed on the
first attempt. Neither candidate improves recovery over zero correction;
neither is promoted. Arena remains unchanged.

## Physical result

Seed 921009 recovered 7/32 core signed 150 N X cases and 70/128 cases across
all four direction/strength groups. Every individual settled outcome matches
zero correction in both continuous execution and matched recovery starts.
This is outcome parity, not identical trajectories.

Seed 931009 recovered 7/32 core cases and 69/128 overall, in both execution
modes. It adds no success over zero correction and loses one case: 300 N -Y,
hit at 1.0 seconds, reserved pose 7. Both seeds retain the same core success
identities as zero correction, including 2/8 reserved-pose successes.

Each candidate retains 16/16 sixty-second quiet Stand cases and 32/32 matched
quiet cases. The sixteen quiet cases include repeated poses, not sixteen
independent poses. Both sampled core evaluations recover 10/32. Seed 921009
matches its weighted control's sampled success identities; seed 931009
exchanges one success for one failure. No corrective steps were measured
anywhere in the 720 cases.

By strength/direction, the continuous and matched settled totals are identical:
seed 921009 scores 7/32 at 150 N X, 32/32 at 150 N Y, 0/32 at 300 N X and
31/32 at 300 N Y. Seed 931009 differs only at 300 N Y, with 30/32.
Evaluation timings 1.0/1.7 seconds are reserved; poses 0-5 were seen during
training, while poses 6-7 were reserved. Seen poses do not make these hit
timings training cases.

Against the rejected weighted controls, continuous totals change from 67 to
70 and 70 to 69 out of 128; matched totals change from 68 to 70 and 68 to 69.
These changes mainly restore zero-control performance. They establish no new
recovery capability and do not justify longer training or Arena replacement.

## Fixed recipe and intervention

Both seeds start from the identical original zero-correction head initialized
with seed 920009. The actor remains the scalar ankle correction head over the
live source Stand policy. Keep half exploration noise, correlated action
memory, full episodes, reset distribution, reward, ordinary advantage
normalization, unweighted critic, learning rates, three epochs, 256-sample
minibatches and raw 5:1 early actor weighting normalized per collection.
Each seed collects 32 batches of 768 continuation transitions.

The intervention projects the actual clipped-gradient Adam parameter
displacement through the nullspace of twelve current quiet-output Jacobians.
The twelve source-derived anchors cover poses 0-5 at onsets 44/74. Their
targets are the fresh head's zero outputs, not the rejected weighted actors'
predictions. Relative SVD cutoff is 1e-6. Apply the projected displacement
without renormalization. Accept only finite candidates with whole-collection
KL at most 0.01 and cumulative anchor deviation at most 1e-5 latent units.
On rejection restore actor and Adam and skip remaining actor minibatches in
that collection; critic updates continue.

## Actor update accounting

Seed 921009 accepts 83, rejects 24 and skips 181 of 288 planned actor slots.
Its final accepted update is in collection 10. Collections 11-32 all reject
their first proposal from the same actor and Adam state. Final and maximum
accepted anchor deviation is 9.9873168e-6.

Seed 931009 accepts 82, rejects 25 and skips 181 slots. Nineteen collections
have no accepted actor update, but collections 31 and 32 each accept nine;
this seed is not permanently frozen. Final anchor deviation is 8.6786695e-6,
and maximum accepted deviation is 9.9808449e-6.

All 49 rejections are anchor-bound violations. None is caused by KL or
nonfinite values. All attempted quiet Jacobians retain rank 12. Median
projected displacement norm retention across all attempts is 99.9051% and
99.8870%. This finite Adam experiment differs from the earlier local
full-gradient projection diagnostic; its norm retention need not match it.

## Why the guard rejects updates

The tangent constraint is numerically satisfied. Independent SVD/Gram
projection disagreement is below 9e-16. Maximum quiet linear movement after
float32 parameter application is below 5.7e-11 across accepted updates.
Actual nonlinear quiet-output movement is much larger and accumulates against
the fixed 1e-5 bound.

Both first collections accept eight updates and reject the ninth. For that
ninth candidate, maximum anchor deviation changes from approximately
9.706e-6 to 1.1095e-5 in seed 921009, and from 9.091e-6 to 1.1526e-5 in
seed 931009. The corresponding rounded-displacement linear changes are below
4.1e-12, while KL is only 3.30e-8 and 2.46e-7. The guard is reacting to
finite nonlinear drift, not a failed projection or an active KL limit.

This explains rejection and reduced update counts. It does not establish
that rejected updates caused the lack of physical improvement: changing the
update direction and reducing the accepted count are coupled interventions.

The first-500 ms matched quiet-command RMS falls from 6.74e-4 to 3.95e-5 rad
for seed 921009 and from 5.57e-3 to 2.76e-4 rad for seed 931009. Corresponding
pushed-minus-quiet RMS changes from 5.94e-4 to 8.03e-5 rad and from 4.61e-4
to 4.83e-4 rad. The constraint reduces quiet drift, but does not uniformly
suppress conditional response. These are closed-loop command differences,
not isolated actor derivatives or evidence of useful recovery direction.

## Verification and retained evidence

The audit verifies 898 input hashes, 273 output hashes and 607 shared
historical inputs. It independently replays each first collection, including
all nine attempted actor updates, eight acceptances, the ninth rejection,
Adam rollback and all nine critic updates. Candidate/optimizer hashes and
post-update outputs match exactly. Initial batches and post-critic outputs
match the corresponding weighted controls; all 32 minibatch permutations per
seed also match.

The audit additionally verifies anchor provenance, projection geometry,
objectives, acceptance accounting, delayed controls, motor limits, memory,
foot geometry and physical outcomes. Maximum independent balance-feature
error is 1.85e-6. No repair or physical rerun was required.

Evidence root:
`logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01/`.
See audit.json, comparison-results.json, comparison-pairs.json,
response-diagnostics.json, contract.json and completed.json. Per-seed fit
folders retain complete first/last batches, compact objective/projection
records, initial/final actor and final critic. There is no intermediate
checkpoint series. Total evidence is approximately 202.38 MiB, primarily
physical traces. Runner time was 542.05 seconds, excluding process startup
and separate audit/analysis; training fits took 105.56 and 103.15 seconds.

## Next priority: one finite anchor correction on each saved rejected update

Reconstruct each seed's first-collection ninth candidate from the saved
initial actor, full batch and fresh Adam. Require exact parity with the saved
candidate and optimizer hashes before changing it. Use these two fixed
candidates; do not search for a favorable update.

At each rejected candidate, form its current quiet Jacobian and apply one
minimum-norm Newton correction toward the original zero targets:
`delta = -pinv(J_candidate) * (anchor(candidate) - target)`.
Retain the same relative SVD cutoff 1e-6 and explicit float32 parameter
application. Do not change tentative Adam moments. Do not retry, line-search,
rescale, widen the anchor bound or tune a parameter.

Measure actual anchor residual, whole-collection KL, correction/proposal norm
ratio, and weighted PPO surrogate on both the rejected minibatch and full
collection. Compare pre-update, rejected and corrected actors, including
pushed-versus-quiet probe movement. Independently verify the linear solve,
parameter application and predictions.

This bounded offline check asks whether a finite correction can restore the
constraint without erasing the proposed learning movement. It produces no
new physical recovery evidence. Decide from both candidates before planning
another learner; do not launch new trajectories, continue training or promote
a policy automatically.

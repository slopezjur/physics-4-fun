# Quiet-output projection feasibility - completed 2026-10-09

Decision: the existing actors can make useful local changes for both push
directions while keeping the fitted quiet outputs nearly unchanged. This
justifies a bounded learning comparison that constrains quiet-command drift.
It does not establish improved physical recovery or a new Arena policy.

Protocol: [RECOVERY_QUIET_ACTOR_PROJECTION_PLAN.md](RECOVERY_QUIET_ACTOR_PROJECTION_PLAN.md).

## Execution and verification

Reconstructed the original full ordinary ascent gradient and 36 output
Jacobians for each weighted actor, using the sealed 288-sample first-decision
dataset. Twelve quiet observations defined the constraint; 24 observations
represented signed pushes. No new trajectories were collected.

The SVD and independent Gram-eigen projections agreed before finite changes.
All three relative singular cutoffs, 1e-5/1e-6/1e-7, retained rank 12 and
produced the same direction. Maximum projection disagreement was 2.05e-13;
quiet linear residual norms were below 4.84e-18. Orthogonality, norm decomposition
and the ascent identity passed. Projected gradient norms retained 36.96% and
68.17% of the original norms for seeds 921009 and 931009 respectively.

Applied one original and one primary projected displacement per actor from
unchanged source parameters. The shared step scales were 0.1045702181 and
0.1151456268, chosen from original predicted output RMS, not from projected
results. No direction was renormalized and no halving was required.

The independent audit reconstructed all four finite float32 displacements,
parameter hashes, exact means and predictions, KL values and summary metrics.
All 605 input hashes, 11 output hashes and 589 shared historical inputs passed.
No source repair or repeated experiment was needed. The 0.01 KL guard passed
on the first attempt for both actor comparisons; its halving path was not used.

## Finite effects on the saved observations

These are latent-command changes on fitted observations, not ankle angles,
success rates or physical footsteps. The numerical sign band is 1e-7.

For seed 921009:

- Original direction: raw pushed changes favored 12/24 preferred directions.
  Quiet-change RMS was 8.72e-4, with maximum absolute change 0.0013183.
- Projected direction: raw pushed changes favored 24/24 preferred directions,
  including all 12 of each force sign. Push-change RMS was 2.20e-5, and the
  smallest signed improvement was 1.91e-5.
- Projected quiet RMS was 3.07e-9; maximum absolute quiet change was 4.54e-9.
  Actual mean KL was 2.12e-8, versus 5.00e-5 for the original direction.

For seed 931009:

- Original direction: raw pushed changes favored 12/24 preferred directions.
  Quiet-change RMS was 8.71e-4, with maximum absolute change 0.0014879.
- Projected direction: raw pushed changes favored 24/24 preferred directions,
  including all 12 of each force sign. Push-change RMS was 3.29e-5, and the
  smallest signed improvement was 2.84e-5.
- Projected quiet RMS was 2.71e-8; maximum absolute quiet change was 3.45e-8.
  Actual mean KL was 4.75e-8, versus 5.00e-5 for the original direction.

Thus all 48 projected pushed observations moved in the preferred raw direction,
while all 24 quiet changes stayed below the numerical movement band. Unlike
the original direction, this result does not rely on subtracting a larger
matched quiet shift to obtain the correct sign.

Float32 parameter addition was measured separately from nonlinear response.
Projected actual-versus-rounded-linear RMS errors were 3.13e-9 and 2.77e-8;
maximum errors were 5.00e-9 and 4.06e-8. The finite clipped-surrogate values
increased for both projected candidates. Neither surrogate improvement nor
correct local action signs establish full-episode benefit.

## Limits and cost

The same observations supplied the gradient and quiet constraints. This tests
local fitted-state feasibility at the first decision, not robustness across
later actions, new poses, hit timings or closed-loop quiet standing. It also
does not prove that unconstrained Adam or a projected Adam update will discover
the same effect during fresh training.

No optimizer or critic update, physical rollout, footstep evaluation or policy
checkpoint was produced. Both earlier weighted candidates remain rejected;
Arena is unchanged. The runner took 1.23 seconds excluding process startup and
the separate audit. New evidence occupies approximately 7.28 MiB.

Evidence: `logs/mimickit-training/recovery-quiet-actor-projection-20261009-01/`.
See audit.json, results.json, contract.json, completed.json, and per-actor
linear/finite arrays and reports.

## Next priority: bounded quiet-constrained learning

Train two fresh seeds, 921009 and 931009, from the identical original
zero-correction head initialized with seed 920009. Use 24,576 continuation
transitions per seed and retain the existing age-weighted recipe: half noise,
full episodes, reset distribution, reward, 5:1 early actor weighting, ordinary
advantage normalization, unweighted critic, learning rates, three epochs,
minibatches and KL limit. Compare against the sealed corresponding weighted
controls; do not continue training the rejected final actors.

Change the actor-update rule only. After gradient clipping and a tentative Adam
step, extract the actual proposed parameter displacement. At the pre-step actor,
form the quiet-anchor Jacobians and project that displacement into their retained
nullspace. Apply it from the pre-step parameters without renormalization. Merely
projecting the raw gradient is insufficient: Adam's coordinate scaling can
reintroduce quiet-output movement. Retain the tentative Adam moments only when
the constrained step passes; otherwise restore both actor and optimizer.

Use the 12 source-derived quiet observations already examined: poses 0-5 at
onsets 44/74. Verify identical features across the two saved datasets and zero
action memory. Recompute target outputs with the fresh initial head, which
must give zero; do not reuse the rejected actors' saved mean commands. Rebuild
the Jacobians at every attempted update. Use the fixed 1e-6 relative SVD cutoff
and log actual rank, retained displacement, update acceptance and anchor drift.

Require both the existing whole-collection KL limit 0.01 and maximum absolute
anchor deviation from the initial target no greater than 1e-5 latent units.
This is a numerical constraint on fitted anchors, not a physical standing
guarantee. Reject, restore actor/Adam and stop actor updates for that collection
on either violation, as with the existing KL guard. Do not compensate with
extra epochs, amplified displacement or a larger learning rate.

Check source immutability, initial collection parity with each matching weighted
control, projection of the actual displacement, nonlinear anchor bounds,
optimizer acceptance/rollback accounting and unchanged critic objective. Keep
compact update metrics and only initial/final actor plus final critic evidence.

Then run the established 720-case physical evaluation: continuous and matched
starts, reserved poses/timings, signed 150/300 N X/Y pushes, sampled core pushes,
60-second quiet Stand and audited footsteps. Compare individual gains/losses
against the matching weighted controls and zero correction. Promote only from
reproducible physical gains with quiet retention; this local result alone does
not justify an Arena replacement or a longer run.


## Successor completed: quiet-constrained learning

The fresh two-seed comparison and 720-case evaluation are complete. See
RECOVERY_QUIET_CONSTRAINED_LEARNING_PROGRESS.md. Both candidates retain quiet
standing but add no recovery over zero correction; no corrective steps and no
Arena change. The projection passes independent checks, while nonlinear drift
triggers the cumulative anchor guard and limits accepted actor updates to
83/82 of 288. Next is a two-candidate offline finite anchor correction check,
not longer training.

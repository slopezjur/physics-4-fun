# Training-only quiet coverage feasibility

Date: 2026-10-09. Approved successor to
RECOVERY_COMMAND_GENERALIZATION_PROGRESS.md. Compare 12 and 48 quiet
constraints on the same first historical Adam proposal for each of seeds
921009 and 931009. This is a finite-update diagnostic, not a new PPO run.

## Frozen state capture

Run one 91-interval quiet prefix with the frozen Stand parent and fresh
zero-output ankle head, using the existing 16-world protocol. Require exact
equality of every recorded field against the archived baseline quiet prefix.
Capture observations, normalization, history, previous commands and physical
state before and after intervals so the independent audit can verify phase.

Select only worlds/poses 0-5 at training onsets 40, 44, 48, 54, 66, 74, 82
and 90. Each feature is from state index onset+1, before the continuation
action. All 48 states have zero action memory. The original 12 features at
44/74 must match the sealed bank exactly, in their original order. Reserved
poses 6/7 and evaluation onsets 60/102 never enter the constraint bank.
The other worlds exist only to reproduce the historical vectorized prefix.

## Fixed proposal comparison

Use each low-rate run's sealed initial parameters, first collection, saved
normalized advantages, weights and minibatch indices. Reconstruct its first
actor gradient, clipping at 0.5, and fresh Adam proposal at learning rate
1e-4. Require historical actor/optimizer hashes, loss and gradient agreement.
Compute one tentative displacement per seed; both arms use that same vector.

For each bank, project the actual float32 Adam displacement using the
pre-update quiet Jacobian and float64 direct SVD, retaining singular values
strictly greater than 1e-6 times the largest. Apply the projected displacement
in float32. Compute the finite quiet residual and candidate Jacobian, then
apply exactly one minimum-norm correction using the same SVD cutoff and
float32 parameter application. No retry, rescaling, line search or tuning.
The original12 arm must reproduce the sealed first accepted candidate.

Keep the zero-target quiet guard at 1e-5 and collection KL guard at 0.01.
Report each arm's own-bank acceptance separately from its all48 acceptance.
Score both arms on all48 quiet states, the same 24 pushed probes and paired
quiet probes, the complete 768-sample surrogate and the original 256-sample
minibatch surrogate. Preserve the prior 1e-7 sign band and historical direction
reference (+X negative; -X positive). These directions describe older local
continuation evidence, not proof of optimal actions under every new policy.

Record pre/candidate ranks and singular spectra, tentative/projected/corrected
norms, retained norm fractions, linear and finite quiet residuals, KL,
surrogate changes, raw/paired signs and response amplitude. Do not call a
smaller quiet residual a success if the useful learning response disappears.

## Independent verification and provenance

Snapshot this plan and all local Python sources before computation; include
the frozen source model, sidecar, physics/reference files and the external
normalizer implementation in the input manifest. Validate each consumed
historical artifact against its completed output manifest and passed audit.
Require modern shared-source identity, without imposing current-source
identity on unrelated, older archived diagnostic scripts.

Independently reconstruct feature normalization and chronology, original-bank
identity, historical gradient/Adam proposal, candidate predictions, objectives,
signs and summaries. Verify the primary direct-SVD operations through thin
QR of the Jacobian transpose and SVD of the small triangular transpose. This
avoids squaring the condition number. Spectrum tolerance is absolute 1e-10,
relative 2e-7; retained rank must agree with cutoff separation greater than
64*float64 epsilon*max(1, largest singular value). Vector maximum error is
2e-8 and relative error 2e-5. Check row-space, residual and norm identities.
These tolerances are fixed before execution. Do not edit sealed sources;
preserve any failed evidence and use a successor if repair is required.

Account separately for 91 vector physics intervals (1,456 world intervals),
two reconstructions of historical Adam steps and four candidate comparisons.
No new PPO collection, critic update, deployable checkpoint or Arena change.
Save compact arrays and reports without duplicating checkpoint files.

Decision scope: whether broader training-time quiet coverage warrants a
later fresh learning comparison. A first-step result cannot establish drift
after 288 updates, reserved-pose generalization or physical recovery gains.
Document the result and next priority; do not automatically launch training.

Evidence: logs/mimickit-training/recovery-quiet-coverage-20261009-01.

# One finite quiet-anchor correction on two saved rejected updates

Approved successor to RECOVERY_QUIET_CONSTRAINED_LEARNING_PROGRESS.md.
Use only the first-collection ninth rejected candidate from each fresh seed,
921009 and 931009. Reconstruct the historical first eight accepted actor
updates and ninth tentative candidate from the original saved batch and Adam
initialization. Require exact historical actor and optimizer hash parity.
No new trajectories, critic fitting, policy continuation or physical evaluation.

At the rejected candidate, compute the current twelve quiet-output Jacobians.
Use one float64 SVD minimum-norm correction toward the original zero targets:
`delta = -pinv(J_candidate) * (quiet(candidate) - target)`.
Use the fixed relative singular cutoff 1e-6. Cast the displacement to float32
once and add it to the candidate's float32 parameters. Keep the tentative
Adam moments unchanged. No iteration, retry, line search, scale sweep,
threshold adjustment or candidate selection.

For pre-update, rejected and corrected actors, retain exact parameter vectors,
quiet outputs, full saved collection means, and fixed pushed/quiet probe
predictions. Report maximum anchor error, whole-collection KL, weighted PPO
surrogate on the exact ninth minibatch and the whole collection, correction
norm relative to the rejected proposal and proposed Adam displacement, and
raw/conditional probe movement. Higher surrogate is better. Separate
corrected-minus-pre learning movement from corrected-minus-rejected repair
cost; a feasible constraint alone does not prove useful learning is retained.

Use the existing twelve quiet anchors and the existing 36 diagnostic probe
states (twelve quiet, twenty-four pushed). These are fitted states. Previously
measured preferred push directions are descriptive diagnostics for these
states, not a recovery guarantee for the newly reconstructed actor.

Independently verify source/output provenance, exact replay, candidate
Jacobian and predictions, the minimum-norm correction through a quiet Gram
eigendecomposition, float32 application, probability/KL and objective
arithmetic. Record unexpected checks as failures; do not alter sealed source
after execution. Preserve prior evidence.

Decision: report both fixed candidates, even if either fails. A successful
anchor correction with retained objective improvement establishes only local
numerical feasibility. Physical gains remain unproven. Choose a next bounded
experiment from the evidence, with no automatic training continuation or
Arena replacement.

Evidence: logs/mimickit-training/recovery-quiet-anchor-retraction-20261009-01.
Retain compact arrays and reports; no new deployable checkpoints.

# Compare quiet-anchor correction targets on two fixed candidates

Approved successor to RECOVERY_QUIET_ANCHOR_RETRACTION_PROGRESS.md. Reuse the
sealed pre-ninth, rejected and zero-corrected parameter vectors, candidate
Jacobians and saved collection/probe data. No historical optimizer replay is
needed. No optimizer state is created or changed.

For seeds 921009 and 931009, change only the Newton correction target from
zero to the pre-ninth quiet outputs. Keep the rejected candidate and its
Jacobian fixed. Apply one minimum-norm SVD correction with relative cutoff
1e-6 and one float32 displacement addition. No retry, line search, rescaling,
new candidate or parameter sweep.

The acceptance guard remains maximum absolute quiet deviation from the
ORIGINAL ZERO target <=1e-5 and collection-old mean KL <=0.01. Also report
residual relative to the local correction target; that residual is not the
global acceptance test.

Compare four states: pre, rejected, zero-target correction, local-target
correction. Require exact prior-state predictions and hashes. Report full
768-sample and exact ninth 256-sample minibatch weighted PPO surrogates,
fixed-state raw/conditional probe movement, correction norms and residuals.
Use fixed saved advantages, weights, old likelihoods and collection-old means.
Higher surrogate is better. Assess feasibility and objective retention
separately; parameter direction or response size alone is not learning proof.

Independently reconstruct predictions and candidate Jacobians, verify the
minimum-norm correction through the quiet Gram eigensystem, float32 parameter
application, probability/objective arithmetic and source/output hashes. Keep
all prior sources and evidence unchanged. Only new diagnostic arrays/reports
are retained, with no deployable checkpoints or physical trajectories.

This comparison isolates resetting accumulated quiet drift from removing new
ninth-step drift. Both target choices start near zero if applied from the
first fresh training update; this historical comparison alone cannot decide
which trains better. Choose the next bounded learner only after results,
with no automatic training continuation, physical run or Arena replacement.

Evidence: logs/mimickit-training/recovery-quiet-anchor-targets-20261009-01.

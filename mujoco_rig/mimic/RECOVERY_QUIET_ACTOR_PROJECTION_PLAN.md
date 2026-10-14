# Quiet-output projection feasibility

Approved successor to RECOVERY_LOCAL_ACTOR_UPDATES_PROGRESS.md. Both weighted
actors remain frozen references. Use their existing 288 saved first-decision
samples, 36 unique observations and original normalized GAE. Twelve observations
are quiet; 24 are pushed. No physics, optimizer, critic fitting or checkpoints.

## Linear check before finite displacement

Reconstruct each actor's full ordinary ascent gradient g and the 36 output
Jacobians J in the original float32 computation. Cast their results to float64
for projection algebra. Let Q contain the 12 quiet rows of J. A thin SVD gives
g_perp = g - Q^+ Q g. The primary relative singular threshold is 1e-6; report
1e-5 and 1e-7 sensitivity, without selecting a threshold from outcomes or
renormalizing the projected direction.

Independently verify with eigendecomposition of Q Q^T, using the SQUARED
relative threshold for eigenvalues. Check retained rank, singular spectrum,
SVD-versus-Gram projection, orthogonality, Pythagoras, ascent identity and quiet
residual including any discarded singular directions. Projection agreement
uses maximum absolute 2e-8 and relative norm 2e-5 tolerances. Spectrum agreement
uses absolute 1e-9 plus relative 2e-5. Identity checks use absolute 2e-10 plus
relative 2e-7; quiet residual is bounded by max(1e-10, 2*smax*threshold*||g||).
Ambiguous threshold boundaries are reported as failures, not tuned away.

Require exact historical gradient reconstruction and agreement with saved
Jacobian Gram and output-response matrices before finite changes. The
independent linear gate runs and is saved before any finite candidate.

## Common finite step

Compare original and primary projected directions using one shared scalar
alpha per actor. Set alpha so RMS(alpha*Jg) across all 36 observations equals
0.01*HALF_SIGMA, giving original predicted mean KL 5e-5. Preserve the projected
norm. Each candidate starts from the unchanged original actor. Cast alpha times
its direction once to float32, then add the original float32 parameters.

Check actual mean Gaussian KL on the 288 saved samples against 0.01. If either
candidate is nonfinite or exceeds the limit, halve the shared alpha for BOTH
directions, with at most eight reductions. Never halve for better quiet
residuals, action signs or surrogate outcomes. Retain the failed comparison if
the guard cannot pass. No Adam, learning-rate search or repeated gradient step.

## Evidence and interpretation

Record proposed and rounded-displacement linear predictions separately from
actual nonlinear output changes. Preserve parameter hashes and rounding norms,
quiet RMS/max, signed push movement, quiet-subtracted movement, conditional
surrogate change, KL, rank and retained norm. The fixed 1e-7 latent sign band is
numerical, not a threshold for physical benefit. Values are latent commands,
not ankle angles, footsteps or recovery rates.

The same observations determine the gradient and quiet constraints. This is
local fitted-state feasibility, not heldout performance or a trained recovery
policy. A successful projection supports a later bounded quiet-constrained
learning comparison; it does not admit a model to Arena. Preserve compact
evidence and source hashes, without a model series or duplicate physics traces.

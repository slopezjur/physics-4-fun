# Offline paired-credit actor updates - completed 2026-10-09

Decision: pair centering reduces the minibatch imbalance in this diagnostic,
but does not solve shared quiet-command movement. Useful opposing response is
learned locally, with larger accompanying changes on quiet observations. The
next intervention must address that coupling rather than merely increase
training duration or adopt pair centering as a demonstrated fix.

Protocol: [RECOVERY_LOCAL_ACTOR_UPDATES_PLAN.md](RECOVERY_LOCAL_ACTOR_UPDATES_PLAN.md).

## Execution and independent verification

Two frozen weighted actors supplied 288 saved first-decision samples each:
144 action pairs, 36 distinct starting observations and four future-noise
repeats per start. Each actor was copied into ordinary-normalized and
pair-centered advantage arms. Both objectives retained the same original
standard deviation, samples, actions, old likelihoods and minibatch order.

All four arms accepted all six scratch Adam updates, for 24 diagnostic updates.
The independent verifier exactly reproduced all 24 updates, saved arrays and
final parameter hashes. It also reconstructed source rows, labels, initial
actor/critic predictions, normalization, gradients, Jacobians and permutations.
All 725 input hashes, 19 output hashes and 589 shared historical inputs passed.
No repairs or repeated experiment were needed.

Initial full-gradient relative differences between objectives were 4.25e-7
and 9.21e-7. Maximum elementwise difference across full/branch gradients was
1.49e-8; an independent score-times-Jacobian calculation agreed within 1.03e-8.
The intended gradient invariant passed. No critic was updated.

All candidate KL values stayed below 0.000176 against the 0.01 guard. No PPO
likelihood-ratio clipping occurred. Gradient-norm clipping activated on two
ordinary-arm updates per actor, and no centered-arm updates. No rollback was
required; its path was not exercised by these runs.

## What minibatch centering changed

At unchanged initial parameters, centering reduced the relative deviation
from the corresponding full gradient in all 12 fixed minibatches across the
two actors. Four ordinary minibatch gradients opposed the full gradient;
one centered minibatch gradient did. Ordinary 32-row remainder gradients had
norms 5.75-45.68 times their full-gradient norm, versus 1.34-4.88 after centering.

The first epoch's 32-row remainder contained no complete action pair. Later
remainders contained four and three complete pairs. The ordinary-minus-centered
gradient difference was independently reconstructed from unmatched rows. The
256/32 sample-weighted gradients reconstructed each initial full gradient.

This demonstrates a concrete effect of separating paired observations in this
fixed intervention dataset. It is not a measurement of historical PPO gradient
variance: historical data were neither this balanced antithetic sample nor a
288-row collection with a 32-row remainder. After parameters change, the two
objectives and optimizer paths can differ for additional reasons.

## Learned response versus quiet movement

All quantities below are conditional latent-mean changes on saved observations,
not ankle radians or closed-loop physical outcomes. Each arm has 12 quiet and
24 pushed observations; repeated noise draws are not extra distinct probes.

For actor 921009:

- Ordinary: quiet-change RMS 0.00044090, matched pushed-minus-quiet change RMS
  0.00002015. Raw pushed movement favored 12/24 preferred directions.
- Centered: quiet-change RMS 0.00096787, response-change RMS 0.00004650.
  Raw pushed movement still favored 12/24 directions.
- Centering increased response magnitude by 2.31 times and quiet movement by
  2.20 times. Constant movement accounted for 98.07% and 93.12% of squared
  all-probe changes in the ordinary and centered arms respectively.

For actor 931009:

- Ordinary: quiet-change RMS 0.00023262, response-change RMS 0.00003670.
  Raw pushed movement favored 12/24 directions.
- Centered: quiet-change RMS 0.00163322, response-change RMS 0.00009079.
  Raw pushed movement favored 14/24 directions.
- Centering increased response magnitude by 2.47 times and quiet movement by
  7.02 times. Constant movement accounted for 86.65% and 34.61% of squared
  all-probe changes; remaining variation is not necessarily useful recovery.

In all four arms, the change relative to the matched quiet observation favored
the correct push direction at 24/24 starts. Thus useful conditional response
exists, but quiet movement is substantially larger. Correct relative signs
do not imply that the raw commands improve both push directions.

## Coupling exists before minibatching or Adam

At the initial actors, +X and -X branch gradients had cosine similarities
-0.9874 and -0.9926. Their opposing contributions largely cancelled: the full
gradient norm retained 18.65% and 8.06% of the sum of all three branch norms.
Quiet branch norms were smaller than either pushed branch in both actors.

Matched pushed/quiet output Jacobian cosines ranged from 0.99437 to 0.99976.
The initial full-gradient direction itself favored only 12/24 raw pushed
directions per actor, despite favoring all 24/24 after subtracting matched
quiet changes. This shared movement therefore cannot be explained solely by
minibatch imbalance or Adam. High local similarity does not prove that the
network is incapable of representing a useful response.

## Scope, storage and evidence

No physical rollout, recovery/footstep evaluation or new policy checkpoint was
produced. Both weighted candidates remain rejected and Arena remains unchanged.
The scratch optimizers started fresh; they did not reproduce historical Adam
state or establish why every previous training run failed.

The runner took 5.00 seconds excluding process startup and the separate audit.
Evidence occupies approximately 24.62 MiB, including compressed gradient arrays,
sample data and source snapshots. It contains no .pt checkpoints or copied full
physics traces.

Evidence: `logs/mimickit-training/recovery-local-actor-updates-20261009-01/`.
See audit.json, results.json, contract.json, completed.json and each actor's
dataset, initial diagnostic and ordinary/centered update records.

## Next priority: quiet-preserving update feasibility

Test whether the existing actor can express useful push changes while holding
its quiet outputs fixed locally. Reconstruct the 12 quiet-output Jacobians Q
and ordinary initial full ascent gradient g. Compare g with its projection
g_perp = g - Q^+ Q g. Use float64 projection arithmetic, primary relative
singular-value threshold 1e-6, and report 1e-5/1e-7 sensitivity without choosing
a threshold from recovery outcomes. Keep the projected norm unchanged.

Verify the projection independently before any finite displacement. Report
retained gradient norm, quiet residual and signed predicted movement at all
24 pushed observations. These linear calculations are not physical outcomes.

Then compare one small finite scratch displacement along original and primary
projected directions per actor, using the SAME scalar step. Set that step from
the original direction's predicted RMS movement across all 36 probes to
0.01 * HALF_SIGMA. This sets the original linear mean KL to 5e-5, below the
existing 0.01 guard, without inflating the projected direction. If either
candidate exceeds the guard or is nonfinite, halve the shared step for both,
with at most eight reductions; retain failures rather than tuning separately.

Measure actual versus predicted movement, quiet residual and raw pushed signs.
Use scratch actor copies without Adam, critic updates, physical simulation or
checkpoints. A successful local projection would justify a subsequent bounded
quiet-constrained learning comparison; it would not establish generalization
or promote a recovery policy. Do not spend another stage only rearranging
minibatches when shared movement is already present in the full gradient.

## Successor completed

The quiet-output projection and four finite scratch displacements passed
independent verification. Both projected actors moved all 24 pushed outputs in
the useful raw direction while all 12 quiet changes stayed below 1e-7 latent
units. See [RECOVERY_QUIET_ACTOR_PROJECTION_PROGRESS.md](RECOVERY_QUIET_ACTOR_PROJECTION_PROGRESS.md)
for finite magnitudes, fitted-state limits and the next two-seed bounded
comparison projecting actual Adam displacements during fresh learning.

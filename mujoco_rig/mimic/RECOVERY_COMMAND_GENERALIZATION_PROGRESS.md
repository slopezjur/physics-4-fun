# Frozen commands: quiet bias precedes memory, and one head reverses response

Date: 2026-10-09. Completed
[RECOVERY_COMMAND_GENERALIZATION_PLAN.md](RECOVERY_COMMAND_GENERALIZATION_PLAN.md).
The independent audit passes. This diagnostic identifies two separate issues:
quiet bias outside the fitted observations already exists before action memory
accumulates, and higher-rate seed 921009 reverses half the fixed-input reference
responses. No policy was trained, simulated, promoted or loaded into Arena.

## Same-input response direction and size

All four frozen heads were evaluated on exactly the same 36 input states:
12 quiet and 24 pushed states, training poses 0-5 and onsets 44/74. Both action
memory columns are zero. The historical pushed preference is negative for +X
and positive for -X, measured by older weighted-policy continuations on these
states. Alignment uses the fixed 1e-7 latent band; it is not a fresh return
comparison under the current heads.

Both low-rate heads and higher-rate seed 931009 agree with the reference on
24/24 pushed states, for raw means and pushed-minus-quiet means. Higher-rate
seed 921009 agrees on only 12/24 and opposes the reference on 12/24. The
reversals occur on poses 2, 3 and 4, at both onsets and for both push signs.
Subtracting the paired quiet mean does not fix them. No pushed classification
is within the numerical sign band.

For seed 921009, increasing actor rate changes conditional latent RMS from
3.127e-5 to 5.022e-5, only 1.61 times. These are 0.0359% and 0.0576% of
HALF_SIGMA. Requested conditional ankle-addition RMS changes from 3.127e-6
to 5.022e-6 rad, while half the reference directions reverse.

For seed 931009, conditional latent RMS grows from 1.790e-5 to 1.237e-4,
6.91 times, or 0.0205% to 0.1419% of HALF_SIGMA. Requested conditional ankle
RMS grows from 1.790e-6 to 1.237e-5 rad. These small immediate responses are
not measurements of the force or angle needed for recovery. They do not
justify amplifying a head with incorrect reference directions or quiet bias.

Quiet latent RMS at the fixed anchor observations is 2.652e-9/5.945e-9 for
the two low-rate heads and 7.241e-8/4.297e-6 for the higher-rate heads. All
four remain within the original maximum anchor bound. The observations tested
here are the fitted quiet states; they do not establish quiet behavior elsewhere.

## Quiet bias is present on the first decision

Every matched-start first decision has exactly zero previous latent and zero
carried memory. Its issued latent is therefore its new policy contribution.
The two evaluation timings are still reserved; their quiet states are outside
the fitted anchor bank.

Pooling the first quiet decision over both timings, reserved-pose latent RMS
changes from 3.470e-5 to 1.822e-3 for seed 921009, about 52.5 times. For seed
931009 it changes from 2.110e-4 to 7.603e-3, about 36.0 times. Seen-pose values
also increase, from 6.499e-6 to 6.847e-5 and from 8.672e-6 to 1.348e-4.

For example, higher-rate seed 931009 issues a quiet latent of +0.01027 at
pose 7 at the first 1.0-second matched decision. Its push-minus-quiet change
there is approximately -0.0000728 for +X. The conditional response has the
historical reference sign, but it is much smaller than the quiet bias, so
the raw command stays positive. The corresponding first quiet value at
1.7 seconds is +0.01045. Higher-rate seed 921009 instead has a negative
quiet bias of about -0.002288 on pose 7 at both timings.

Across all 32 first matched pushed decisions, raw/reference agreement is
29/32 and 27/32 at the low rate, versus 18/32 and 27/32 at the high rate.
After subtracting matched quiet outputs, it is 32/32 and 32/32 at the low
rate, versus 20/32 and 32/32 at the high rate. These counts distinguish
shared quiet movement from conditional direction; they are not recovery scores.

## What the memory decomposition establishes

Over the fixed first 30 decisions, reserved-pose quiet memory/innovation RMS
is 0.01200/0.001821 for higher-rate seed 921009 and 0.05029/0.007627 for seed
931009. Memory and innovation have positive cross terms. The stored reports
verify both second-moment and centered-variance identities including these
terms; their RMS values must not be presented as independent causal shares.

The first-decision result establishes that memory is not the origin of the
observed quiet bias. Later traces show it being carried alongside continued
policy output, but they do not establish what would happen if memory were
removed. Changing memory would change subsequent actions and body states.

Conditional 500 ms mean commands are less consistently aligned with the
first-action historical reference. Agreement is 8/32 and 32/32 at the low
rate, versus 4/32 and 25/32 at the high rate, with the same counts in both
execution modes. The body state changes during that window, so these counts
do not prove which later commands are optimal or harmful. In contrast, the
12/24 common-input reversals above occur directly on the tested first-action
probe states.

All selected quiet and pushed intervals are live; no failed case was removed.
Quiet world slots include duplicate copies of poses. The scalar head still
adds the same correction to both ankle-pitch joints. This diagnostic neither
tests a new leg-placement controller nor changes any physical recovery score.

## Audit, cost and evidence

Independent verification passes 684 input hashes and 44 output hashes, all
four frozen heads, 144 same-input predictions, 16 paired windows, 256 paired
case windows and 7,680 paired sample rows from 28 unique archived traces.
The separate NumPy network forward differs from recorded Torch predictions
by at most 4.09e-9. Probe pairing, reference labels, tanh/slew mapping, exact
archived memory chronology, live masks, all grouped/per-case summaries and
source snapshots pass. No repair or repeated execution was needed.

The analysis runner took 3.26 seconds excluding process startup and separate
verification. Evidence occupies approximately 11.85 MiB, including source
snapshots. There are no new checkpoints, optimizer updates or physical rollouts.
Arena remains unchanged and the previously rejected policies remain rejected.

Evidence: `logs/mimickit-training/recovery-command-generalization-20261009-01/`.
See audit.json, contract.json, completed.json, results.json and each policy's
probes.npz/probes.json plus continuous/matched window arrays and summaries.
The first-decision and pooled figures above are direct reductions of the
independently reconstructed window arrays.

## Next priority: training-only quiet coverage feasibility

Materialize one frozen zero-head/Stand-parent quiet prefix, using the existing
observation and control protocol, and capture quiet features after the same
prefix phase used by the current anchors. Use poses 0-5 and onsets
40, 44, 48, 54, 66, 74, 82 and 90: 48 quiet states. All are in the existing
training distribution. The original 12 states at 44/74 must match exactly;
the 36 added states retain zero action memory. Keep poses 6/7 and evaluation
onsets 60/102 out of the constraint bank.

For each seed, exactly reconstruct the first saved fresh low-rate Adam
proposal at actor rate 1e-4, with its original samples, clipped gradient and
optimizer state. Compare the original 12-anchor projection/correction with
the proposed 48-anchor projection/correction on that SAME tentative proposal.
Keep the SVD cutoff, one finite correction, float32 application, zero target,
KL guard and all other arithmetic unchanged. Do not tune the bank or rate.

Evaluate BOTH corrected candidates against all 48 quiet states and the same
24 pushed probes. Report constraint rank, retained proposal norm, local and
global quiet residual, full/minibatch surrogate change, and raw/conditional
reference signs. Independently verify the prefix phase, original-anchor
identity, original Adam proposal and both finite operations. Preserve complete
bounded evidence without creating a deployable checkpoint or running PPO.

Broader temporal coverage must retain useful response before it warrants a
later fresh learning comparison. This first-proposal feasibility check cannot
establish accumulated behavior after 288 updates, generalization to reserved
poses or recovery gains. No automatic memory change, longer training or Arena
promotion follows.

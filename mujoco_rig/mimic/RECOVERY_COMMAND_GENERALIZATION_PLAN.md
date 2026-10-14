# Frozen command direction and quiet generalization

Date: 2026-10-09. Approved successor to
RECOVERY_RETRACTED_ACTOR_RATE_PROGRESS.md. Analyze both final actor rates
(1e-4 and 1e-3) and seeds 921009/931009 using existing evidence. No training,
optimizer or critic updates, physics integration, new checkpoint or Arena
change. Preserve all previous sources and evidence.

## Same-input policy comparison

Use the existing 36 probe observations per dataset from
recovery-local-actor-updates-20261009-01: onsets 44/74, poses 0-5, twelve
quiet observations and twenty-four signed 150 N X observations. Verify the
features and pairing labels are identical across datasets. Their final two
action-memory columns are zero. Historical means and quiet return preferences
can differ; they are not interchangeable with common input identity.

Evaluate all four frozen final heads on these common observations. Record
raw latent mean, learned innovation, paired pushed-minus-quiet response,
amplitude relative to HALF_SIGMA, and requested ankle addition after the
existing 0.1*tanh mapping and 0.02-rad slew limit. Recover prior addition
from feature column -2 times 0.1 and prior latent from column -1. Pair quiet
and push by onset and pose. Use the fixed 1e-7 latent sign band to classify
agreement with the historical pushed preference: negative for +X, positive
for -X. Do not assign that directional preference to quiet observations or
treat the older frozen-continuation preference as current-policy optimality.

## Archived trajectory decomposition

Use both continuous and matched-start deterministic evaluations, both core
timings (onset frames 60/102), and all sixteen world slots covering eight
poses and two force signs. Pair each core trace with its same-policy quiet
trace. Select the first thirty decisions after the first force interval:
frames 61:91 or 103:133. Keep every case and report explicit live masks;
do not filter failures or count repeated quiet copies as independent poses.

Extract the recorded scalar latent and previous latent. Decompose in float64:
memory = 0.9*previous_latent; innovation = latent-memory. The innovation is
the learned contribution to that issued command, not a causal estimate of
the effect of removing memory. Verify chronological memory updates and the
existing tanh/slew mapping against both ankle additions. Distinguish requested
addition from the final joint target, which also includes the frozen parent's
nominal target and joint limits.

Record quiet, push and pushed-minus-quiet moments by pose, seen/reserved group,
sign and timing. Include mean-square terms and the memory/innovation cross
term so correlated components are not described as independent energy shares.
Keep same-input inference separate from own-trajectory differences. Compare
rates without changing the archived trajectories or extrapolating new recovery
scores from the decomposition.

## Evidence and verification

Snapshot current diagnostic sources and this plan. Verify every consumed
historical artifact against its completed manifest and passed audit. Check
shared current sources against the two modern rate-control input manifests.
Retain compact probe/window arrays and summaries, not copies of full traces
or model checkpoints. Record hashes before and after computation.

Independently verify all four frozen heads with a separate network-forward
calculation, reference pairing/polarity, requested action mapping, selected
trace windows, memory arithmetic, live masks and aggregate summaries. Record
floating-point tolerances explicitly; preserve a failed audit if any repair
is necessary. Do not change sources after sealing.

Decision scope: identify raw direction errors, weak conditional response and
quiet bias outside the fitted observations. Keep reserved poses diagnostic;
do not silently add them to training or the quiet-anchor bank. This analysis
cannot by itself prove which component caused a fall or establish a better
controller. Choose the next bounded intervention from its findings.

Evidence: logs/mimickit-training/recovery-command-generalization-20261009-01.

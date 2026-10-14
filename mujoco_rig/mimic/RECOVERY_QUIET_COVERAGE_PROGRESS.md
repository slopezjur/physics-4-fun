# Training-only quiet coverage feasibility: completed

Date: 2026-10-09. Implements RECOVERY_QUIET_COVERAGE_PLAN.md and the next
priority in RECOVERY_COMMAND_GENERALIZATION_PROGRESS.md.

The expanded bank is technically feasible on the two fixed first proposals:
it nearly eliminates quiet output across all 48 states without eliminating
the proposed update, positive sampled-objective change or pushed response
amplitude. This supports a bounded fresh learning comparison. It does not
establish improved balance, recovery, reserved-pose behavior or foot placement.

## Controlled comparison

One frozen zero-head/Stand-parent quiet prefix ran for 91 control intervals
in 16 native worlds: 1,456 world intervals, or 1.516788 seconds per world.
All 24 archived trace fields matched exactly. The bank contains only poses
0-5 at onsets 40, 44, 48, 54, 66, 74, 82 and 90, with each observation
taken at state index onset+1. All action-memory entries are zero. The original
12 features and targets at 44/74 match exactly. Reserved poses 6/7 and
evaluation onsets 60/102 remain outside the bank.

Each seed's first fresh Adam proposal at actor rate 1e-4 was reconstructed
exactly from sealed initial parameters, samples, advantages, weights and
minibatch indices. The two arms used the same tentative displacement. The
original12 arm reproduces the historical actor and optimizer hashes, projected
and corrected outputs, loss, clipped gradient and KL exactly. No other setting
changed: direct SVD cutoff 1e-6, float64 algebra, float32 application, one finite
correction, zero-target bound 1e-5 and KL limit 0.01.

## Quiet coverage and retained update

Both arms pass their own quiet/KL guards and the common all48 guards. Thus
this first step does not expose a guard failure in the original bank; it
measures additional quiet drift below the existing limit.

- Seed 921009: maximum all48 quiet output falls from 9.693e-7 to 4.002e-12;
  RMS falls from 3.465e-7 to 1.473e-12. The original bank retains 60.81% of
  the Adam proposal norm; the expanded bank retains 48.92%. Projected norms
  are 6.907e-4 and 5.556e-4, respectively.
- Seed 931009: maximum all48 quiet output falls from 2.227e-6 to 4.366e-12;
  RMS falls from 5.675e-7 to 1.719e-12. Retained proposal norm falls from
  68.87% to 53.92%, with projected norms 7.822e-4 and 6.124e-4.

Pre-update and correction Jacobians have rank 12 for the original arm and
48 for the expanded arm. All 48 expanded singular directions survive the
fixed cutoff; the smallest initial singular value is 3.210e-5, compared
with largest 1.2684. There is no cutoff relaxation or rank ambiguity.

Expanded correction norms are 1.769e-7 and 2.022e-7, approximately 0.0318%
and 0.0330% of projected norm. Finite correction therefore does not restore
the removed proposal by adding a comparably large displacement. Expanded
collection KL is 1.021e-10 and 8.031e-10, below original values 3.275e-9
and 7.482e-9. These are first-step measurements, not accumulated learning.

## Objective and pushed response

All four full-collection and minibatch surrogate changes remain positive,
although expansion reduces the gains. Recorded float32 changes are:

- Seed 921009: full gain 1.103e-6 to 2.086e-7; minibatch gain 5.960e-6
  to 1.222e-6.
- Seed 931009: full gain 6.676e-6 to 9.537e-7; minibatch gain 1.156e-5
  to 3.427e-6.

A separate float64 readback of the saved predictions and probability formula
also preserves every positive gain. These tiny sampled-surrogate changes
are not a physical improvement or a statistical learning-speed estimate.

On the same 24 pushed observations, paired pushed-minus-quiet latent RMS
changes from 1.202e-6 to 1.262e-6 for seed 921009 and from 7.853e-7 to
1.120e-6 for seed 931009: approximately 1.05 and 1.43 times the original.
Expanded paired requested ankle-addition RMS is 1.262e-7 and 1.120e-7 rad.
The corresponding latent responses are only 0.001448% and 0.001284% of the
fixed exploration standard deviation. A nonzero response is retained, but
this first step cannot produce a visibly demonstrated recovery improvement.

Every first-step candidate, with either bank and either seed, opposes the
historical direction reference on all 24 probes, both raw and after quiet
subtraction; none lies within the 1e-7 classification band. This is inherited
first-step disagreement, not a reversal introduced by expansion. The previous
audit found that both fully trained lower-rate heads later align on 24/24
of these probes. The older weighted-policy continuation reference is not
proof of current zero-head optimality, and matching it is not a training
objective or sufficient promotion criterion. No extra direction-tuning branch
is justified by this comparison alone.

## Verification, cost and evidence

Independent verification passes 683 input hashes, 11 output hashes, the exact
archived prefix, raw feature normalization/history/phase, both historical
Adam proposals and all four candidates. Thin QR plus reduced SVD verifies
the direct-SVD projections and corrections without squaring the condition
number. All48 predictions, probe pairing, signs, action mapping, full/minibatch
objectives, guard classifications and summaries are verified.

The runner took 14.64 seconds excluding process startup and the separate
audit. Evidence occupies approximately 40.99 MiB, including source snapshots.
The runner reconstructs two historical Adam steps; the independent audit
reconstructs them again, adding no physics. There are zero new PPO transitions,
critic updates or deployable checkpoints. No repair or repeated physical run
was needed. Arena remains unchanged.

Evidence: `logs/mimickit-training/recovery-quiet-coverage-20261009-01/`.
See audit.json, bank-report.json, bank.npz, prefix.npz, contract.json,
completed.json, results.json and each seed's evidence.npz/report.json.

## Next priority: bounded fresh learning with 48 quiet constraints

Run fresh seeds 921009 and 931009 from the same original zero head, with
24,576 transitions per seed, actor rate 1e-4 and critic rate 3e-4. Change
only quiet-bank coverage from the original12 to the sealed expanded48. Keep
the source policy, observations, memory, action limits, reward, exploration,
weighting, PPO schedule, rollout budget, cutoff, single correction, acceptance
guards and rollback behavior fixed. Preserve all sealed sources; implement
successor helpers where the original learner assumes 12 rows.

Use the audited low-rate 12-anchor runs and zero-head evaluations as controls.
Verify exact initial collection/critic/permutation parity before learning
diverges. Record acceptance, rank, retained displacement, KL, quiet drift and
fixed-probe direction/amplitude at collection boundaries as compact numeric
summaries, without intermediate policy checkpoint series or label tuning.

Apply the established complete physical suite: both continuous and matched
starts, quiet Stand, core timings, force groups, sampled core and reserved
poses/timings. Compare paired gains and losses against the sealed controls;
measure whether quiet generalization improves beyond the 48 fitted states.
Keep reserved states diagnostic. Require reproduced physical gains with quiet
retention before promotion. Better constraint fitting or PPO surrogate alone
does not qualify. No rate sweep, longer run or Arena promotion follows
automatically. This next learning trial was not run in the feasibility step.

# Per-update recovery learning trace - completed 2026-10-09

Decision: test greater actor-loss emphasis on early recovery. Both exact
historical replays show that the first half-second supplies only about 17% of
samples and that later-frame gradients are larger in 30/32 updates. This is a
testable explanation for weak recovery learning, not proof of its cause.

Protocol: [RECOVERY_UPDATE_TRACE_PLAN.md](RECOVERY_UPDATE_TRACE_PLAN.md).

## Exact historical parity

Both original half-noise streams were reproduced for 24,576 transitions each.
Final actor and critic tensors matched bit for bit, as did all 32 numerical
history rows per seed, every reset record, reset accounting, and every array in
the previously saved first and last collections. Source controller weights
remained unchanged. Diagnostics consumed no RNG and preserved live gradients.

The independent audit passed 64 collections / 49,152 replayed transitions,
763 input hashes and 79 output hashes. It verified episode labels and ages,
latent-memory continuity, likelihoods, GAE decomposition and normalization,
gradient identities, exact fixed-probe sources and boundary continuity, and
final probe outputs from the historical checkpoints.

No new configuration or policy checkpoint was produced. The existing audited
evaluation results therefore remain applicable: the two half-noise policies
are still rejected, and Arena remains unchanged. The full evaluation suite
was not repeated.

## The consistent imbalance is early versus late

Early means the first 30 continuation intervals (0.50004 seconds), starting
after the first force interval. Late means all subsequent intervals within
the same episode, including ongoing recovery and later standing.

- Seed 921009: early 4,175/24,576 samples (16.99%); late 20,401 (83.01%).
  Late whole-batch gradient norm exceeded early in 30/32 updates, with median
  late/early ratio 4.07. Their directions opposed in 17 updates.
- Seed 931009: early 4,206/24,576 samples (17.11%); late 20,370 (82.89%).
  Late gradient norm exceeded early in 30/32 updates, with median ratio 3.83.
  Their directions opposed in 15 updates.

These are contributions under the current whole-batch objective. Larger norm
does not prove later gradients are wrong, and opposing directions do not by
themselves prove catastrophic interference. Early recovery is nevertheless
consistently a small share of the objective that trains this recovery head.

Quiet episodes supplied 39.39% and 38.09% of transitions, despite the 25% quiet
reset probability. Longer survival changes exposure. However, quiet gradients
were the largest branch contribution in only 7/32 and 8/32 updates. The data
do not support a simple diagnosis of quiet samples always dominating pushes.

## Fixed-state response across training

The probe set contains the same saved zero-correction observations at every
update: two training timings, paired quiet/pushed branches, first 30 intervals,
and 12 training worlds. Previous correction and latent memory are zero, so the
actor output isolates its learned conditional drive from inherited random
action memory. These are offline probes, not physical policy evaluations.

Seed 921009's pushed-minus-quiet RMS peaked at update 22, at 0.002943 latent
units, then ended at 0.000204. Final quiet-probe RMS was 0.002786; the contrast
was 7.31% of that magnitude.

Seed 931009's contrast peaked at update 15, at 0.002014, then ended at 0.000514.
Final quiet-probe RMS was 0.010923; the contrast was 4.70% of that magnitude.

Both response magnitude and sign changed during training. A larger intermediate
contrast is not evidence of better recovery: it could also be a harmful reaction.
These measurements therefore do not identify a previously successful brain
that should be restored. They also do not establish uniform collapse into a
constant output; pose/state variation remains substantial.

## What remains unproven

Critic fitting and reward/value GAE components were recorded at every update,
but the value targets include bootstrapping. Their fit does not independently
establish calibration or justify a critic/reward rewrite.

Original middle batches and intermediate model weights were not retained.
Historical equality is established against every available record, first/last
full batches and final tensors. Middle gradients have analytic final-layer and
full-network partition checks; they cannot be independently reconstructed from
unavailable historical intermediate weights. This limitation is recorded in
the audit rather than presented as exact equality to nonexistent arrays.

## Verification repair and storage

The first verifier combined float64 GAE recurrence differences with a tighter
normalization check. That produced a 5.37e-6 normalized discrepancy. A successor
verifier checks the recurrence independently, then normalizes the recorded
float32 advantage values. Maximum GAE error was 6.24e-5, within the established
tolerance; normalization error was 4.82e-7. No physical replay was repeated for
this verifier repair. Original sealed code was preserved; audit-harness.json
records both verifier sources and the reason for the correction.

The two replays took 208.14 seconds after setup. All new evidence is about
5.5 MiB, including compact per-transition metrics, gradients and probes. There
are no new .pt checkpoints or duplicated full physics traces.

Evidence: `logs/mimickit-training/recovery-update-trace-20261009-01/`.
See audit.json, interpretation.json, age-weighting-proposal.json, audit-harness.json
and per-update diagnostic arrays. Later analyses carry separate input/source
hashes from the original replay manifest.

## Next priority: bounded early-recovery actor weighting

Use two fresh seeds, the identical initial actor and the same 24,576-transition
budget per seed. Retain half noise, full episodes, reset distribution, reward,
GAE, critic training, learning rates, epochs and the existing KL guard.

Change only the actor objective: assign raw weight 5 to continuation age below
30 intervals and weight 1 to later samples. Normalize these weights to mean 1
over the whole collection and multiply the existing clipped PPO surrogate by
them. Keep the existing advantage normalization and critic loss unchanged.
Episode age is training metadata, not a new actor observation or deployed timer.

On the saved streams, this fixed 5:1 ratio produces 48.56% and 47.85% effective
early-recovery mass. Maximum normalized sample weights would be 4.16 and 4.71.
Exact 50/50 rebalancing per collection would instead reach 9.85 and 32.0 when
only a few early samples are present. The fixed ratio avoids that amplification;
the actual new-run weights and sample counts must still be recorded.

This objective also changes effective branch exposure: saved-stream quiet mass
would fall from 38-39% to about 34%. It is an age-weighted objective experiment,
not perfect isolation of timing from quiet/push occupancy.

Check the weighting formula, mean-one normalization, unchanged probability/KL
scale and critic objective. Evaluate normal continuous execution and matched
starts, heldout timing/poses, signed 150/300 N X/Y pushes, sampled core pushes,
sixty-second quiet Stand and audited footsteps. Compare paired gains/losses
against the corresponding unweighted half-noise learner and zero correction.
Do not promote from training reward or probe magnitude alone.


## Successor completed

The two bounded actor-weighting fits and all 720 evaluation cases are complete.
See [RECOVERY_AGE_WEIGHTED_LEARNING_PROGRESS.md](RECOVERY_AGE_WEIGHTED_LEARNING_PROGRESS.md)
for actual exposure, mixed recovery results, promotion rejection and the next
frozen quiet-command comparison.

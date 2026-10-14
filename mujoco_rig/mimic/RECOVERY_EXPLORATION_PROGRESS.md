# Frozen exploration and credit check - completed 2026-10-09

Decision: exploration can produce useful recovery, and the paired 64-interval
credit favors the observed rescues. However, persistent noise also destroys
baseline recoveries and quiet Stand. The final critics introduce additional
quiet-state ranking errors. Do not increase noise, rewrite the reward, extend
training or promote a model on the current evidence.

Protocol: [RECOVERY_EXPLORATION_PLAN.md](RECOVERY_EXPLORATION_PLAN.md).

## Scope

Seventy-two frozen continuations covered 864 training-pose cases at onset frames
44 and 74. Poses 6/7 were excluded. The initial actor and both final actors used
their existing means and Gaussian innovation scale, with four common noise seeds.
Zero correction and both established ankle pulses supplied matched controls.
Starts retained the native state, command queue and history after the first force
interval. Five force intervals remained; episodes ended at six seconds.

No optimizer ran and no model weights changed. These training timings are
diagnostic cases, not new heldout admission results. All actions continued through
the existing slew, amplitude, joint and torque limits and two-interval delay.

## What exploration actually did

The zero-correction control recovered 5/24 pushed starts and 24/24 quiet starts.
Repeated across the four noise seeds, that is a comparison baseline of 20/96
pushes and 96/96 quiet cases for each actor.

- Initial actor: sampled recovery 16/96; eight rescues and 12 losses relative to
  the corresponding zero-correction cases. Quiet recovery was 59/96.
- Actor 921009: sampled recovery 18/96; ten rescues and 12 losses. Quiet 58/96.
- Actor 931009: sampled recovery 18/96; ten rescues and 12 losses. Quiet 60/96.
- Mean-policy recovery: 5/24, 7/24 and 5/24 respectively. All three mean policies
  retained the quiet Stand gate in 24/24 cases.
- Fixed negative/positive ankle pulses: 8/24 and 12/24 pushed recoveries, with
  24/24 quiet recoveries each. These are diagnostic commands, not learned skills.
- Audited corrective steps: zero across all 864 cases.

Thus useful behavior is present in sampled trajectories, but stochastic execution
has a net recovery loss relative to zero correction. Quiet failures also show that
the same noise is disruptive outside a push. Exploration is not absent, and
increasing its scale is not justified by the small deterministic outputs alone.

## Early command coverage

In each actor's 96 sampled pushed cases, 24 maintained the diagnostic-helpful
ankle sign above 0.025 rad for at least six consecutive intervals during the first
30 intervals. Only two maintained it above 0.060 rad for that duration. Six
intervals are about 100 ms; the observation window is about 500 ms.

Strong sustained commands resembling the successful 0.075 rad pulse are rare,
but most sampled rescues do not satisfy that strong-command threshold. Therefore
the threshold is neither a necessary condition for recovery nor proof that
insufficient amplitude alone explains the learning result. The learned means
have not materially changed these aggregate coverage counts.

## What the credit estimates say

For every sampled rescue (8, 10 and 10), both the realized full-episode reward
and the first-state GAE difference against the matched zero continuation were
positive under that actor's own critic. Eleven of each actor's 12 losses had a
negative paired GAE difference; the remaining loss had positive early credit.
Later sampled actions can change the six-second outcome, so disagreement with
that outcome alone is not evidence of an incorrect first-window estimate.

We also decomposed the same 64-interval GAE difference into the lambda-weighted
reward difference and the critic's contribution. On pushes, the learned critics
reversed the reward-component sign in 3/96 and 0/96 comparisons. None of those
reversals applied to a sampled rescue. The zero initial critic cannot introduce
such a reversal.

On quiet starts, the same-window sign reversed in 39/96 and 24/96 comparisons.
Of these, 38 and 23 changed a negative reward difference into positive paired
credit; one per critic reversed the other way. The first critic's positive quiet
credit margins reached 6.32 units despite a negative reward component. This is
evidence of critic-driven quiet ranking errors in this diagnostic, not proof
that they were the sole cause of the failed training.

All 288 sampled quiet continuations had lower full-episode reward than zero
correction. Some short-window differences were positive, reinforcing the need
to distinguish a local credit window from the complete recovery objective.

Absolute or normalized advantage signs are not paired outcome rankings. For
example, the initial zero critic assigned positive normalized first-state credit
to every pushed case, including losses: early rewards exceeded its within-segment
average. The comparison above uses differences at the identical initial state.

## Verification and limitations

The independent audit passed 72 trajectories / 864 cases, 568 input hashes and
150 output hashes. It verified restored inputs, immutable policies, Gaussian
draws, actor means, critic values, action/history memory, force windows, target
composition and physical delay, native foot geometry, outcomes and step events.
The initial mean policy reproduced the zero-correction control exactly.

The verifier initially passed a time/world/feature tensor to an actor designed
for world/feature input. A successor verifier flattens those leading dimensions.
Its independent float64 TD sums also exposed ordinary accumulated rounding in
the recorded float32 GAE recurrence: maximum difference 9.55e-5. The comparison
uses absolute tolerance 1e-4 and relative tolerance 2e-6. These verifier repairs
did not alter or repeat physical rollouts; the original sealed source remains
intact. The repair sources and numerical limits are recorded in audit-harness.json.

GAE uses the same gamma .995, lambda .98 and 64-interval recurrence as PPO.
After a failure, these diagnostic worlds freeze rather than immediately reset;
credit is checked only for live transitions. Normalization covers live samples
within each diagnostic segment, not the historical mixed-episode PPO batch.
Final critics and fresh samples cannot reconstruct past optimizer updates or
prove a causal gradient failure during training.

The experiment took 284.48 seconds after setup. Evidence occupies about 212.7 MB,
including physical traces, first-window policy features, values and source copies.
No checkpoints were created. Arena selection remains unchanged.

Evidence: `logs/mimickit-training/recovery-exploration-20261009-01/`.
See audit.json, paired-credit.json, credit-decomposition.json, audit-harness.json
and individual traces/reports. The later credit decomposition and verifier repair
records have their own input/source hashes, separate from the original manifest.

## Next priority

Isolate damage from continued random commands before changing PPO. Reproduce
these same sampled first 30 recovery intervals, then stop adding Gaussian
innovations and continue with each actor's conditional mean. Retain its actual
latent memory, physical state, pending commands and history; do not zero targets
or replace the learned controller. Compare to the already saved continuously
noisy trajectories using identical early commands and case identities.

This bounded frozen comparison can determine whether later noise destroys an
otherwise viable recovery or quiet stance, and how that relates to the critic's
quiet rankings. Treat the noise cutoff as a diagnostic intervention, not a
deployable hit-clock trigger. Keep reward, noise amplitude, physical limits and
models fixed, with no optimizer updates. Then decide between changing exploration
duration/structure and investigating critic calibration; do not assume either
diagnosis is complete from the present correlations.


## Successor completed

The proposed noise-cutoff comparison is complete. See
[RECOVERY_NOISE_CUTOFF_PROGRESS.md](RECOVERY_NOISE_CUTOFF_PROGRESS.md)
for 576 audited cases, paired outcomes and the superseding next priority.

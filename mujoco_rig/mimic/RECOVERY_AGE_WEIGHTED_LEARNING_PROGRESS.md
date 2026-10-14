# Early-recovery actor weighting - completed 2026-10-09

Decision: reject both candidates for promotion. The intended early-recovery
emphasis was achieved, but recovery gains did not reproduce across seeds or
matched starts. Keep Arena unchanged. Do not extend either training run.

Protocol: [RECOVERY_AGE_WEIGHTED_LEARNING_PLAN.md](RECOVERY_AGE_WEIGHTED_LEARNING_PLAN.md).

## What changed and what was verified

Fresh seeds 921009 and 931009 each trained for 24,576 continuation transitions
from the same initial actor. The sole objective change was raw actor weight 5
for age below 30 intervals and 1 later, normalized over each 768-sample
collection to mean one. The existing clipped PPO surrogate was multiplied by
those weights. Age remained optimizer metadata, absent from observations.

Half-noise sampling, full episodes, resets, reward, GAE, advantage normalization,
unweighted critic objective, learning rates, epochs and KL guard were retained.
Both seeds accepted all 288 actor updates with zero rollbacks. Maximum KL was
0.001321 and 0.001037, below the 0.01 guard.

The independent audit passed all 64 collection objectives and 45 physical traces
/ 720 cases. It reconstructed weights, actual actor/critic losses, minibatch
permutations, full per-world age/force chronology, reset accounting, likelihoods,
GAE and normalization. Both first collections match their unweighted counterparts
exactly, including actions, advantages and post-update critic outputs. Only the
expected actor outputs differ after the weighted update.

Physics checks include source/input immutability, native foot geometry, command
limits and delay, temporal memory, sampled innovations and case-level outcomes.
The audit checked 753 input and 205 output hashes; maximum independently
reconstructed balance-feature error was 1.92e-6.

## Actual exposure

- Seed 921009: early 4,137/24,576 samples (16.83%), with 48.51% of actor weight.
  Quiet samples were 36.56% of transitions and 31.89% of actor weight. Effective
  +X/-X weights were 30.45%/37.65%; maximum normalized sample weight was 3.9344.
- Seed 931009: early 3,976/24,576 samples (16.18%), with 46.53% of actor weight.
  Quiet samples were 37.68% of transitions and 33.75% of actor weight. Effective
  +X/-X weights were 29.04%/37.21%; maximum normalized sample weight was 4.7525.

The raw 5:1 ratio does not enforce exactly equal early/late totals. Full-episode
occupancy changed after the policies diverged, and branch exposure changed with
age weighting. This does not isolate timing from branch occupancy perfectly.

## Physical results

Ordinary deterministic execution:

- Seed 921009: core 150 N X recovery 7/32, reserved poses 2/8, all push groups
  67/128. Its unweighted counterpart scored 7/32, 1/8 and 68/128.
- Seed 931009: core recovery 9/32, reserved poses 2/8, all groups 70/128.
  Its unweighted counterpart scored 5/32, 0/8 and 65/128.
- Zero correction scored 7/32 core, 2/8 reserved core, and 70/128 overall.
- Both weighted candidates retained 60-second quiet Stand in 16/16 cases.
- Both scored 32/32 at 150 N Y and 0/32 at 300 N X. Stronger 300 N Y recovery
  was 28/32 and 29/32, versus zero correction's 31/32.
- Sampled core recovery was 10/32 for both, versus unweighted 10/32 and 9/32.
- No corrective footsteps across all 720 cases.

From matched post-impact states, both candidates scored 7/32 core and 68/128
overall, with quiet standing 32/32 each. Unweighted matched totals were 68/128
and 67/128. The second candidate's extra continuous core successes disappeared.

Across the 128 ordinary deterministic push cases, gains/losses against the
corresponding unweighted learner were 3/4 and 12/7. Against zero correction,
they were 0/3 and 3/3. For matched starts, the same comparisons were 1/1 and
6/5 against unweighted, and 0/2 and 1/3 against zero.

Seed 931009's two additional core successes over zero were +X poses 3 and 4
at the 1.7-second timing. Both are training poses and both fail in matched
execution. Its remaining gain was a stronger sideways training case, offset
by three stronger-sideways reserved-case losses. Its equal 70/128 aggregate
therefore does not mean equal generalization.

## Learned response

During the first 0.5 seconds of matched core recovery, ankle-addition RMS was
0.000886 and 0.005580 rad. Pushed-minus-quiet RMS was 0.000594 and 0.000461 rad,
versus unweighted 0.000137 and 0.000329 rad. Increased contrast did not improve
matched core recovery beyond zero correction (7/32), or produce a replicated
gain over the unweighted learners. These are response measurements, not proof that
impact-dependent feedback causes any particular recovery.

First/last collections improve the weighted local surrogate. That does not
establish heldout recovery, critic calibration or a successful stepping skill.
The result also does not rule out all early/late interference; it rejects this
specific bounded weighting setting as a promotion candidate.

## Audit precision and storage

Pre-audit review found a float32 reporting mean of 1.000000238418579, while
independently averaging stored weights gave maximum error 6.18e-8 for seed
921009. A successor verifier uses four float32 epsilons only for that reported
scalar and also compares it with the independently averaged weights. Actual
weight-formula and array-mean checks remain unchanged. The original verifier
was preserved and was not run; no training or physics was repeated.
audit-harness.json records both verifier hashes and the reason.

Training took 99.14 and 97.74 seconds. Training plus evaluations took 527.39
seconds after setup. Evidence is about 197.5 MiB, mostly physical traces;
there are no intermediate model checkpoint series.

Evidence: `logs/mimickit-training/recovery-age-weighted-learning-20261009-01/`.
See audit.json, paired-outcomes.json, response_diagnostics.json, completed.json,
audit-harness.json and per-collection objective arrays.

## Next priority: test whether the two extra recoveries require impact response

Use frozen seed 931009 at the two existing core timings (32 cases), without
optimizer updates. Preserve its ordinary live prefix through the first force
interval. Then replay its own corresponding quiet-trial latent sequence through
the unchanged ankle limits, memory and delayed controls. Keep the source Stand
controller reacting to the actual pushed body throughout.

Require exact prefix and quiet-control parity, then compare the two added
successes and all remaining case identities against saved ordinary execution,
matched execution and zero correction. This is a small causal diagnostic, not
a new deployable scripted controller or a new training target.

If the gains persist, the extra successes do not require the ankle head's
impact-dependent response. Preparation and the continuing quiet-command
trajectory would remain sufficient candidates; this replay does not separate
those two contributions. Then investigate local action credit at identical
states with paired innovations before changing PPO again.

If the gains disappear, impact response contributes in combination with the
prepared state. Prioritize the mismatch between source-derived training starts
and the policy's own pre-impact states. Neither outcome authorizes promotion
from these aggregate scores alone.


## Successor completed

The frozen quiet-command comparison is complete. All 32 pushed outcomes,
including both extra recoveries, persisted without the ankle head's
impact-dependent continuation. See
[RECOVERY_WEIGHTED_QUIET_REPLAY_PROGRESS.md](RECOVERY_WEIGHTED_QUIET_REPLAY_PROGRESS.md)
for 48 audited cases, the causal limits and the next paired action-credit diagnostic.

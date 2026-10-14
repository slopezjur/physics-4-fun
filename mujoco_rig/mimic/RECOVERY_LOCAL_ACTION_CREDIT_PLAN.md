# Frozen paired first-action credit

Approved successor to RECOVERY_WEIGHTED_QUIET_REPLAY_PROGRESS.md, 2026-10-09.

Use weighted actors 921009/931009 and their own final critics without updates.
Restore exact source-derived starts after the first force interval at training
onsets 44/74, preserving native state, pending controls, history and latent memory.
Require parity with the archived training-start evidence. Use poses 0-5 only:
six quiet worlds once each and twelve pushed worlds covering +X/-X at 150 N.

For each start, compare first conditional action minus HALF_SIGMA with the same
action plus HALF_SIGMA. Consume the first full 16-world Gaussian draw before
replacing it; from the second action onward use shared standardized draws from
noise seeds 951011-951014. The policy subsequently reacts to its own state and
memory. Keep the remaining episode through the existing absolute six-second
endpoint, half noise, reward, source controller and physical limits unchanged.

The design is 2 actors x 2 timings x 6 poses x 3 branches x 2 first actions x
4 future-noise seeds = 576 continuations / 288 pairs across 64 batched rollouts.
Quiet signed-world duplicates are excluded. No reserved poses or evaluation
timings enter action selection. No training, model checkpoints or Arena update.

Retain rewards, failure/termination masks, values, means, innovations, corrections,
first-window features and the physical evidence needed to audit paired starts,
delayed effects, outcomes and footsteps. Compute first-state 64-interval GAE
with gamma 0.995 and lambda 0.98, separate reward/value contributions, observed
64-interval return, its endpoint bootstrap and full measured episode return.
Include the failure reward and exclude frozen post-failure intervals.

Independently verify hashes and historical lineage, exact restoration and initial
observations, pairwise equal means/values, shared future draws, initial delayed
physical equality, own actor/critic outputs, GAE/return arithmetic and physical
outcomes. Store compact arrays; do not save intermediate model series.

Analyze positive-minus-negative deltas by actor, branch and state. Report all
288 individual pairs separately from 72 four-noise averages. Use predeclared
1e-3 return/credit units as a descriptive sign-tie tolerance, not a significance
test. Compare reward/value components and horizons before interpreting ranking
disagreement. Four noise sequences do not establish calibrated expected values;
GAE/full-return mismatch alone does not prove critic error.

Evidence: logs/mimickit-training/recovery-local-action-credit-20261009-01/.

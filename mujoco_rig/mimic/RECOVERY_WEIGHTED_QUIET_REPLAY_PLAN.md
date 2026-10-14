# Frozen quiet-command replay of the weighted recovery actor

Approved successor to RECOVERY_AGE_WEIGHTED_LEARNING_PROGRESS.md, 2026-10-09.

Test frozen seed 931009, two existing core timings (1.0/1.7 seconds), 16 signed
150 N X cases each. Preserve its live policy prefix through the first force
interval. Starting at frame 61/103, replace only the ankle head's latent command
with its saved same-pose, same-time quiet-trial command. Preserve live Stand
source feedback, motor limits, slew, two-interval delay and physical state.
Continue for the existing six-second horizon. Do not reset at intervention.

Add one six-second unhit replay control (16 cases) to require exact equality to
the saved quiet trace over its first six seconds. This gives 32 diagnostic push
cases and 16 parity-control cases, 48 total. No training, optimizer updates,
new policy checkpoints, parameter search or Arena change.

Audit input/output hashes, immutable policy weights, exact live prefixes and
two queued physical intervals, quiet command suffixes and unhit control parity,
live source targets, actual ankle additions, history, force timing, foot geometry
and case-level recovery/step criteria. Compare against saved ordinary execution,
matched recovery and zero correction, emphasizing the two extra ordinary
successes: +X poses 3 and 4 at the later timing. Retain every case comparison.

If both gains persist, impact-dependent ankle-head feedback is unnecessary for
those successes under this intervention. Preparation and continuing quiet
commands remain combined explanations, alongside live source feedback. Then
prioritize local action-credit diagnosis. If either gain disappears, investigate
the contribution of response together with the prepared state and the mismatch
between source-derived training starts and ordinary execution. Neither result
alone supports promotion, natural stepping or conclusions about arbitrary hits.

Evidence: logs/mimickit-training/recovery-weighted-quiet-replay-20261009-01/.

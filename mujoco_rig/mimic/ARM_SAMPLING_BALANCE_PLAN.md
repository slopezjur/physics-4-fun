# Paired correction sampling with exact quiet retention

2026-10-08. Continues ARM_CORRECTIVE_LEARNING_PROGRESS.md.

Run eight fresh fits: seeds 711028 and 721028, each with replay, 5% correction,
15% correction, and restored-quiet high-share sampling. Keep the original
architecture, initialization, normalization, Adam 1e-4, 4,000 updates and
512-example batches. Preserve existing experiment files and the Arena.

For every batch, draw the same 512 original training indices as the historical
replay. Keep every quiet row at its original batch position. Use a separate
seeded generator to replace eligible active rows with correction examples.
The low shares refer to the expected fraction of the whole batch. The high
share preserves the original-active/correction ratio of uniform concatenation
while restoring quiet share: (4548/6804) * (5827/(4548+5827)), about 37.5% of
all draws. Thus all recipes retain the exact replay quiet stream (expected
2256/6804 = 33.16%), not just its row count.

Use the same replacement random values and correction indices within each
seed, so 5% replacements are a subset of 15%, and 15% a subset of high-share.
Store and independently reproduce sampler stream hashes and counts. Replay
must reproduce historical final weights and physical trajectories exactly.
Whole-pose validation rows 6/7 remain excluded from all sampling.

Evaluate all eight frozen heads at nominal and varied timing, without teacher,
output overrides or action noise. Rank recipes by worst-condition full-skill
count, then total full skills; prefer lower correction share on exact ties.
Test both seeds of the highest-ranked nonzero recipe for exact 60-second quiet
retention. Keep replay unless the new recipe strictly improves its score and
passes quiet retention. Universal learned admission still requires all eight
cases at both timings and both seeds; selection is not Arena promotion.

Compare first issued-command divergence, moving-goal tracking and COM/support
violations against replay. Also evaluate new policies on identical saved replay
observations through the original projection, distinguishing direct policy
changes from effects of subsequently different state and command histories.
Use the common original/corrective examples for regression diagnostics.

No more correction collection, longer fitting, PPO, C# build or Arena change.
If sampling balance does not produce reproducible admission, record the result
and inspect input/feedback behavior before another training experiment.

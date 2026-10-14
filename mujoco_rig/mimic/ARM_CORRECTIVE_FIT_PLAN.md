# Matched corrective-data learning comparison

Date: 2026-10-08. Conditional continuation of ARM_CORRECTIVE_TAKEOVER_PLAN.md.

Run only after the independent takeover audit passes and accepts at least one
correction sample. Freeze the pooled accepted training-pose corrections from
both source heads and both timings. Holdouts 6/7 contribute zero new labels.
This common collection set couples the two source heads; replication below
tests fitting initialization, not independent data collection.

Fit four fresh inherited-feature 421-128-128-33 proposal heads: replay and
correction-augmented datasets, each at seeds 711028 and 721028. All use the same
normalization, output scaling, zero final-layer initialization, Adam 1e-4,
4,000 updates and batches of 512. Use the existing fit function unchanged.
The replay dataset is the entire original teacher set, including quiet rows.
The augmented dataset concatenates that entire set with admitted corrections;
uniform sampling changes their relative frequency, not the optimizer budget.
Original validation rows remain identical and outside fitting.

Check identical initial weights within each pair and reproduce the previous
replay final weights exactly. Evaluate every frozen head at nominal and varied
timing through unchanged physical gates and delayed commands. Quiet retention:
both augmented heads run 60 seconds against the archived exact anchor trace.
No teacher, output overrides, new bounds or noise during learned evaluation.

Report both seeds and all cases. Retain the prior recipe unless the augmented
pair improves its worst full-skill count, then total full skills as tie-break,
with exact quiet retention. Even improvement is not Arena admission: all cases
at nominal and changed timing must pass before resuming weight transfer. No
PPO, additional hyperparameter search, C# build or Arena promotion in this block.

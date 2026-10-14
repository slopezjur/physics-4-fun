# Stable learned handoff pilot

2026-10-08. Continue the authorized bounded learning experiments after the failed twenty-step distillation block. Preserve Arena parent f832bab8 and all physical limits, springs and delays. No C# compilation or export.

Start from the goal-relative 421028 learner, chosen prospectively as the first goal-relative seed, not a new winner. Add an explicit runtime transition: zero addition for an original quiet goal; fade correction by the requested motor displacement (full at 0.05 rad) and the supplied one-second authority ramp; limit changes in the added target to 0.02 rad per control frame. This changes command composition, not physical joint limits or latency. The original labels already contain the old blending; this additional fade is an intentional successor handoff rule.

1. Verify 30-second quiet retention and exact trace equivalence with the previous zero-addition control.
2. Evaluate ordinary-start support for all sixteen cases with the guarded learner. During this rollout, query the existing private physical predictor on at most 192 learner-visited states from training poses 0, 1, 4, 5, both sides, at six-frame intervals after activation. The predictor never controls the live plant. Assert each query leaves live position, velocity and queued commands unchanged.
3. Compare matched 1,500-update continuations: old training examples only versus 75% old examples / 25% new corrections, learning rate 0.0001, batch 512, same initialization and sampling seed. Keep normalization fixed; preserve whole-pose holdouts 6 and 7. New labels are local corrective targets, including states from failed trials; they are not certified successful demonstrations.
4. Evaluate both frozen learners on all sixteen ordinary-start cases, using the same handoff. Verify quiet retention for the correction arm. Only if all sixteen entry/hold/return cases pass, evaluate command onset 3 seconds. Otherwise stop at the identified prerequisite failure; do not repeat a broad timing sweep.

Retain the existing 15 mm clearance / one-second support / final two-second Stand gate. Gradual commanded motion is not hit recovery or learned foot placement. Record partial progress and regressions explicitly. Keep compact traces, exact correction samples, source hashes, and initial/final neural states only.

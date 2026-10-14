# Corrective takeover and matched fitting results

Date: 2026-10-08. Both bounded blocks completed and independently audited.

## Decision

Retain the previous complete-data learned heads. Physical teacher takeover can
recover every tested failing training case, but uniformly adding those verified
corrections to fresh supervised fitting regressed learned task performance.
Neither augmented head qualifies for promotion. Weight transfer, single-foot
support and learned impact stepping remain gated on reproducible arms-clear
admission. The Arena policy is unchanged.

## Physical correction admission

Protocol: ARM_CORRECTIVE_TAKEOVER_PLAN.md.
Evidence: `logs/mimickit-training/arm-corrective-takeover-20261008-01`.

Replayed both original learned heads at nominal and varied timing. All four
controls exactly reproduced archived states, issued commands and survival.
For failing training poses only, the teacher took over 0.5 seconds before the
first moving-goal motor/COM/support/foot/native failure diagnostic, clipped to
task onset. The switch frames were frozen before running any continuation.

All 16 attempted training-case recoveries passed the original entire hold and
return gates. No onset fallback was needed. Combined learned-prefix/teacher
results, including untouched learned cases:

- Seed 711028 nominal: 5/8 to 8/8 full skills; 3 recovered training cases;
  1,041 accepted correction samples.
- Seed 711028 varied: 5/8 to 8/8; 3 recovered cases; 1,146 samples.
- Seed 721028 nominal: 3/8 to 8/8; 5 recovered cases; 1,732 samples.
- Seed 721028 varied: 2/8 to 7/8; 5 recovered cases; 1,908 samples.
  The remaining failure was an untouched learned holdout.

The 5,827 accepted samples contain only training poses 0-5. Poses 6/7 received
zero teacher queries and zero new fitting labels. Every takeover preserved the
control prefix, issue-time integration state, pending commands, previous
addition and observation. Live-state fingerprints were unchanged by all teacher
queries. Saved raw labels reconstruct the actually issued bounded commands
exactly, including authority, fade, slew, quiet gating and final joint clipping.
The teacher's raw-target/residual round trip stayed below 2e-7 rad.

This establishes recovery by the physical teacher from these early learned
states. It does not establish learned recovery or recovery from later, larger
failures. No simulation state was restored at takeover and no clock was reset.

## Matched learned comparison

Protocol: ARM_CORRECTIVE_FIT_PLAN.md.
Evidence: `logs/mimickit-training/arm-corrective-fit-20261008-01`.

Four fresh heads: replay and augmentation, each with initialization seeds
711028 and 721028. Every fit used 4,000 Adam updates, learning rate 1e-4 and
batches of 512, with identical architecture, normalization and output scaling.
Each matched pair had bitwise-identical initial weights. Replay final weights
and physical trajectories exactly reproduced the previously retained heads.

Replay retained 6,804 original training rows, including 2,256 quiet rows.
Augmentation retained all of those and added 5,827 accepted correction rows,
for 12,631 training rows. The same 2,268 original validation rows stayed out of
all fitting. Both source heads contributed to one common correction dataset;
these are independent fitting initializations, not independent collections.

Frozen learned results, with no teacher or output overrides:

- Seed 711028 nominal: replay 8/8 holds, 8/8 returns, 5/8 full;
  augmented 2/8 holds, 8/8 returns, 1/8 full.
- Seed 711028 varied: replay 7/8 holds, 7/8 returns, 5/8 full;
  augmented 3/8 holds, 4/8 returns, 0/8 full.
- Seed 721028 nominal: replay 6/8 holds, 8/8 returns, 3/8 full;
  augmented 3/8 holds, 5/8 returns, 2/8 full.
- Seed 721028 varied: replay 6/8 holds, 8/8 returns, 2/8 full;
  augmented 3/8 holds, 6/8 returns, 2/8 full.

Full skills fell from 15/32 to 5/32. Worst-condition success fell from 2/8 to
0/8. Both augmented heads passed all 16 sixty-second quiet cases with exact
anchor commands, states and survival, for 32/32 total. Quiet retention alone
does not admit a task controller: the existing quiet gate suppresses its task
addition while stationary.

## What this isolates and what remains uncertain

The correction labels are physically achievable, and the matched replay is
reproducible. Additional verified demonstrations alone were insufficient under
this uniform-mixture fitting recipe. Common-data diagnostics show lower errors
on active original training rows, active holdouts and correction examples in
both augmented heads, despite worse closed-loop behavior.

Original quiet-training RMSE increased from 0.00406 to 0.00527 rad for seed
711028, and from 0.00411 to 0.00538 rad for seed 721028. Uniform concatenation
reduced quiet examples' expected sampling share from 33.16% to 17.86%, even
though no quiet row was removed. This is a concrete sampling confound to test;
it does not prove quiet dilution caused the physical regressions. Low offline
prediction error continues to be an inadequate admission metric.

Common-data measurements and input hashes: `common-regression.json` in the
fitting evidence directory.

## Next priority

Keep the previous best heads and the admitted correction dataset. Isolate
sampling balance before collecting more data or extending training: preserve
the original quiet sampling share and explicitly control the weight of original
active versus corrective examples. Use matched replay controls, both fitting
seeds, identical update budgets, whole-pose holdouts and full physical gates.
Inspect the first changed issued commands and resulting tracking/COM response
when comparing recipes. Do not infer improvement from validation loss alone.

No longer PPO run or Arena replacement follows from this result. Resume the
weight-transfer curriculum only after full learned admission reproduces across
the two seeds and changed timing.

## Verification and accounting

- Takeover audit: `arm-corrective-takeover-20261008-01/audit.json`, passed.
- Fitting audit: `arm-corrective-fit-20261008-01/audit.json`, passed.
- 64 physical takeover/control cases plus 96 learned/quiet evaluation cases.
- 4 fresh supervised fits, 16,000 optimizer updates and 8,192,000 sampled
  example presentations. These are not PPO transitions.
- Only initial/final weights were retained; no periodic training checkpoints.
- No PPO, rig, spring, latency, bound, admission-gate or C# changes.
- Arena SHA-256 remains
  `c87c272a0cf60d857de45f186eb5c64b7c63652ca5acf7a5ba050afebe6ad172`.

Both experiment processes finished. No background training remains from these
blocks. Existing experiment sources and evidence were preserved.

## 2026-10-08 - Exact-quiet sampling comparison completed

See ARM_SAMPLING_BALANCE_PROGRESS.md. Eight matched fresh fits preserved every original quiet draw at its original batch position. Replay reproduced historical weights and traces exactly. Full skills: replay 15/32, 5% corrections 16/32, 15% corrections 15/32, restored-quiet high share 5/32. The selected 5% pair improved worst-condition success from 2/8 to 3/8 but lost heldout passes (7/8 to 5/8); it is not admitted. Both selected heads retained exact sixty-second Stand, 32/32. Sampling and physical audits passed. Command differences emerge during the ramp and grow as state/history diverge. Next: frozen matched-state feedback/input diagnosis before more fitting; retain previous references and Arena. No learned stepping. Evidence: logs/mimickit-training/arm-sampling-balance-20261008-01/audit.json and diagnostics.json.

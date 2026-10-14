# Reachable response supervision completed

Completed 2026-10-09; protocols were sealed on 2026-10-08.
Plans: ARM_REACHABLE_RESPONSE_PLAN.md and ARM_RESPONSE_SUPERVISION_PLAN.md.
Evidence roots retain their start-date names:

- `logs/mimickit-training/arm-reachable-response-20261008-01`
- `logs/mimickit-training/arm-response-supervision-20261008-01`

## Decision

Reject the new response-trained heads. Retain the earlier masked research
references and the existing Arena brain. We now have audited, physically
reachable feedback examples and evidence that the response objective changes
what the networks learn. That improvement did not produce successful control.
There is still no learned recovery foot placement.

## Reachable-state admission

The retained masked heads supplied prefixes for both seeds, nominal and varied
timings. Training poses 0 and 4 received one-frame left/right arm pulses of
either sign, plus an unpulsed control. Requested correction was 0.005 rad before
the existing authority, fade and slew projection.

- All 40 targeted teacher continuations passed the complete hold/return gate:
  8 unpulsed controls and 32 disturbed cases.
- All 160 paired examples were admitted at takeover offsets 0, 1, 3, 6 and 12
  control frames. Both sides of every pair passed the complete physical gate.
- Actual transmitted pulse magnitudes were 0.001734 to 0.004186 rad.
- Native positions and velocities were identical for the first two intervals.
  Physical differences appeared in the third interval; the teacher started
  querying after that interval, three frames after pulse issue.
- Pre-pulse integration/control states and learned prefixes reproduced exactly.
  Both queue slots, native velocity scaling, observations, goal timing and
  command projection were independently verified. No live state restoration,
  direct queue rewrite or body override was used.
- Teacher queries preserved live state/control. Total queries: 13,645. Poses
  6 and 7 received zero teacher queries and zero admitted labels.
- Twenty episodes covered 160 physical cases. The 120 untargeted cases remained
  exact learned-control replays; the teacher's 40/40 score applies only to the
  two targeted poses, not all eight poses in each episode.

The teacher's paired target-change norm had median 0.000716 rad, p95 0.046316,
and maximum 0.242548. All 160 examples exceeded 1e-6 rad signal. These are finite
responses across different reachable states and command histories, not isolated
queue derivatives. No ball impact or leg-repositioning task was tested.

Before fitting, both frozen recipes predicted paired changes worse in squared
error than predicting zero change. For pending-effect inputs, response error
energy relative to zero change was 1.715 and 1.439 across the two seeds. This
motivated a response objective; it did not justify promoting that input recipe.

## Matched learning comparison

Keep the pending-effect input representation and 421-128-128-33 network fixed.
Four fresh fits compared absolute-target supervision with added response
supervision, seeds 711028 and 721028. Every fit used 4,000 Adam updates at 1e-4.

Every update had the historical 512-example original batch plus 32 paired
examples, or 64 extra input views. Both recipes had identical initialization,
original draws, quiet draws and paired draws within each seed. The original
6,804 training rows retained all 2,256 quiet rows; the 2,268 heldout rows never
entered fitting. Paired examples came only from training poses 0 and 4.

Both recipes used original normalized target MSE plus weight 0.05 on paired
absolute-target MSE. The response recipe additionally used SmoothL1 error on
the paired output change, normalized by per-motor output scale and the actual
transmitted pulse. The robust loss was chosen because the admitted response
magnitudes vary widely.

A common response coefficient, 0.051005735, was calibrated from the two initial
networks' training-only gradient norms to a nominal 10% initial gradient share.
It was fixed before fitting and not selected using physical or heldout results.
The independent audit reconstructed that calibration. The 10% value describes
initial gradient scaling; it is not a bound on gradient influence later in fitting.

The fits share pooled collection origins. They are independent fitting seeds,
not independent teacher-data collections. Historical heads had fewer sample
presentations per update; the causal same-budget comparison is between the two
new recipes with the same paired data.

## Results

Full skill requires supported final two-second hold, motor RMS <=0.05 rad,
COM error <=0.02 m, both feet within 0.02 m drift, survival and a final two-second
Stand return. All gates and command limits were reconstructed without widening.

- Absolute-target control, seed 711028: nominal 1/8, varied 1/8. Changed timing
  1/8; the two noise streams 1/8 and 0/8.
- Absolute-target control, seed 721028: nominal 1/8, varied 0/8. Changed timing
  and both noise streams 0/8.
- Response supervision, both seeds: 0/8 in every nominal, varied, changed-timing
  and noise condition.
- Core totals: absolute-target control 3/32; response supervision 0/32. Both
  had worst-condition success 0/8. All three control passes were training poses;
  neither recipe passed a heldout core case.
- Timing/noise totals: absolute-target control 2/48; response supervision 0/48.
  These reused challenge settings are not fresh holdouts.
- Both response heads retained exact 60-second quiet Stand, 32/32 cases,
  matching archived qpos, issued targets and alive flags. Quiet mode disables
  the task addition; this preserves the Stand anchor rather than demonstrating
  a newly learned active skill.

The retained masked references remain stronger: core 15/32, worst condition 2/8;
timing/noise 21/48, worst condition 2/8. Neither new recipe earns promotion.

The added objective did learn its local training signal:

- Full admitted-pair response loss decreased 57.3% and 57.1% versus the matched
  absolute-target controls.
- On each seed's collection pairs, median response cosine improved from 0.108
  to 0.457, and from -0.100 to 0.590. Opposite-direction responses decreased
  from 29/80 to 9/80, and from 43/80 to 9/80.
- Response error energy relative to zero change improved from 1.481 to 0.949,
  and from 1.116 to 0.560.
- Original training-target normalized MSE worsened by 45.9% and 43.0%.
  Heldout raw residual RMSE worsened from 0.02436 to 0.02924 rad, and from
  0.02157 to 0.02305 rad.

These response improvements are fitting-set measurements. They do not establish
generalization or stability. The matched intervention demonstrates a tradeoff
between the added objective, target accuracy and physical success in this
configuration; it does not prove target-error increase is the sole failure cause.

## Next priority

Preserve the retained masked controller exactly and investigate a bounded learned
correction around it. Freeze the retained controller weights and initialize the
new correction at zero. First prove exact replay of both retained seeds across
the existing core conditions and 60-second quiet Stand before any fitting.

Use the admitted reachable pairs to inspect needed correction directions and
magnitudes, then predeclare one matched correction fit. Keep the same frozen
base, examples and budget in its controls. Require preservation of existing
passes as well as improvement on failures; inspect any regression before more
training. Do not assume a frozen base guarantees the combined controller remains
successful once a nonzero correction is applied.

This is a new bounded hypothesis, not a demonstrated solution. Avoid another
fresh whole-head response-loss sweep or a longer PPO run. Weight transfer,
single-foot support and learned recovery stepping remain gated.

## Verification and resources

- Collection `audit.json`: passed; 160 physical cases, exact prefixes/states,
  real two-interval delay, original projection/gates and 160 admitted pairs.
- Collection `response-diagnostic.json`: passed; frozen pre-fit responses.
- Fitting `audit.json`: passed; calibrated gradients, identical initialization,
  original/quiet/paired streams, heldout exclusion, noise streams, all 192
  evaluation cases and exact quiet Stand.
- Fitting `objective-summary.json`: reconciles complete fitting-set losses
  with the physical results. No further fitting or physical runs were added.
- Combined total: 352 evaluated cases; 16,000 optimizer updates; 8,192,000
  original presentations plus 1,024,000 paired-view presentations. Fitting
  took 78.482 seconds, excluding collection, physical evaluation and auditing.
- Approximately 267.4 MB of new evidence across the two roots; only initial/final
  model weights were saved. No interval checkpoints, PPO, C# compilation,
  spring/rig edits or Arena replacement. All experiment processes completed.

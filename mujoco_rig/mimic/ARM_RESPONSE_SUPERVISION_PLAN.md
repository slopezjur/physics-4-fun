# Matched response-supervision comparison

Predeclared after reachable-data admission, before fitting, 2026-10-08.
Evidence: arm-reachable-response-20261008-01/audit.json and response-diagnostic.json.

## Admitted measurements and choice

All 40 targeted teacher continuations passed; all 160 paired examples were
admitted, with zero heldout teacher queries. Both frozen input recipes had
paired-response error energy above a zero-change predictor, in both seeds.
For pending-effect inputs the ratios were 1.715 and 1.439; median response
cosines were 0.074 and -0.174. Teacher response norm ranged from 0.0000303 to
0.24255 rad, median 0.000716, p95 0.04632. Every pair exceeded 1e-6 rad signal.

Keep the already validated pending-effect representation fixed and compare
absolute-target teaching with added finite-response teaching. Use SmoothL1 for
the response term because measured responses span several orders of magnitude.
No new input features or ratio sweep. These are reachable finite trajectory
responses, not isolated queue derivatives or ball-impact recovery labels.

## Four fresh matched fits

Two recipes, seeds 711028 and 721028, identical 421-128-128-33 initialization,
4,000 Adam updates at 1e-4 each. Every update uses the historical original draw
stream of 512 examples plus 32 paired examples (64 extra input presentations).
The extra-pair generator is separate, seeded with fit seed +2,000,000, and is
identical within each seed's comparison. All original quiet draws remain at
their original positions. Pooled pairs couple collection origins across fitting
seeds; these are fresh fitting replicas, not independent collection replicas.

Original normalized residual-target MSE has weight 1. Both recipes also use
weight 0.05 on mean absolute-target MSE across the two views of each pair.
The response recipe alone adds:

`SmoothL1((prediction_delta - teacher_delta) / output_scale / abs(issued_pulse), beta=1)`.

Prediction delta uses the two task-head residual outputs. Teacher delta uses
the effective raw target difference; paired goals are identical. Pulse magnitude
is the measured transmitted target change, bounded below by the admission gate
of 1e-5 rad. No label clipping or command bypass.

Calibrate one common response coefficient before fitting using only admitted
training data and both zero-output initializations: take the geometric mean of
`0.1 * norm(absolute_loss_gradient) / norm(response_loss_gradient)` across seeds.
This sets an initial gradient scale; it is not tuned on physical or heldout
performance and is not adjusted during training. Save and independently
reconstruct the calibration. No optimizer steps occur during calibration.

Both recipes see exactly the same original and paired examples. This isolates
the added response objective. Comparisons to the older four-fit experiment are
historical references: this block adds paired examples and therefore has more
sample presentations per optimizer update than that older experiment.

## Frozen evaluations and decision

Use the same nominal, varied, changed-timing (0.9/1.4/3.2 seconds), and two
0.005-rad proposal-noise streams (710083, 710084) for all four heads. These
challenge settings were used previously; do not call them new holdouts.
Test both response heads for exact 60-second quiet Stand against the archived
anchor baseline. Reconstruct all unchanged support, motor, COM, feet, survival,
return and command-limit gates independently.

Rank core results by worst seed/schedule and then total passes; report challenge
rank separately. Research improvement requires better core rank and no worse
challenge rank than the matched absolute-target control. Also compare with the
retained masked reference (core 15/32, worst 2/8; challenges 21/48, worst 2/8).
Complete admission requires all 32 core and 48 challenge cases across both seeds,
plus exact quiet Stand. No Arena promotion based on regression or training-pair
response error. No fit uses poses 6 or 7 as labels.

Budget: 16,000 optimizer updates; 8,192,000 original presentations and 1,024,000
extra paired-view presentations; 192 physical cases. Save initial/final weights
only. No PPO transitions, C# compilation, spring/rig change or Arena replacement.

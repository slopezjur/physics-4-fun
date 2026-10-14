# Expanded quiet coverage learning: completed, no promotion

Date: 2026-10-09. Implements RECOVERY_COVERAGE_LEARNING_PLAN.md.

Both fresh seeds completed 24,576 transitions and the full evaluation suite.
Expanding the quiet bank from 12 to 48 states produces no reproducible recovery
gain. Both candidates remain at 7/32 core recoveries, retain standing, and
produce no measured footsteps. They are not promoted; Arena stays unchanged.

## Physical outcomes

Seed 921009 settles 69/128 deterministic cases in both continuous and matched
evaluations, versus 70/128 for its 12-anchor control. It loses reserved pose 7
under 300 N negative Y at 1.0 seconds in both modes. That case also passes with
the zero-head baseline. There are no gains elsewhere.

Seed 931009 settles 70/128 continuous cases versus 69/128 for its control,
regaining the same reserved pose 7/negative Y case and matching the zero head.
With matched starts it remains 69/128, identical to its control and still
missing that zero-head success. The apparent continuous gain does not survive
the matched-start comparison and is not a new success beyond the baseline.

For both candidates, core 150 N X results remain 7/32 with identical pass/fail
membership to both the 12-anchor control and zero head. Sampled core results
remain 10/32 with identical membership to their same-seed controls. Each
candidate passes 16/16 sixty-second quiet cases and 32/32 matched quiet cases.
All 45 traces contain zero measured footsteps. This scalar head adds the same
correction to the two ankle-pitch joints; independent leg placement remains
outside its direct action space.

## Training and numerical behavior

The only training change is the frozen 48-state bank at onsets 40/44/48/54/
66/74/82/90, poses 0-5. Actor rate 1e-4, critic rate 3e-4, initialization,
sampling, PPO schedule, reward, weighting, memory, control and guards remain
fixed. The original 12 rows remain exact; reserved poses 6/7 and evaluation
onsets 60/102 never enter the bank.

Both seeds accept 288/288 actor attempts. There are no rejections, skipped
actor slots, nonfinite attempts or intermediate guard violations. Every
projection and correction has rank 48. Maximum accepted quiet deviations
are 2.421e-9 and 1.781e-9, below the 1e-5 bound.

Median retained projected parameter norm is 99.865% and 99.880% of the
tentative Adam displacement, close to the original controls'99.880% and
99.862%. The 49%/54% retention from the first-proposal feasibility study does
not describe the later training steps. Most parameter displacement survives;
that does not establish sensitivity of the outputs to a disturbance.

Median collection KL is 1.221e-8 and 7.999e-9, versus 8.555e-8 and 1.278e-7
for the controls. Maxima are 1.504e-6 and 3.749e-7, far below 0.01. No ratio
crosses the PPO clip bounds and no clipped surrogate branch binds across
73,728 reused rows per seed. Median finite-correction norms are only 0.154%
and 0.150% of projected norm. Rejection or active clipping does not explain
the lack of recovery gains in this trial.

First-collection own-batch surrogate gains shrink from 2.137e-5 to 4.649e-6
and from 4.852e-5 to 2.891e-6. Last-collection gains are positive,3.305e-5
and 2.086e-6. Later collections differ between policies, so these values are
local objective changes, not a common physical-performance comparison.

## Response strength and quiet generalization

On the same 24 pushed inputs, final paired latent RMS is 6.198e-6 and
1.893e-6, only 19.8% and 10.6% of the 12-anchor controls. Corresponding
requested ankle-addition RMS is 6.198e-7 and 1.893e-7 rad. The first-update
feasibility result did not predict retention of response amplitude after
the complete learning budget.

Seed 921009 reaches 24/24 historical-reference alignment at collection
boundaries 7-9 and 13-27, then changes to 24/24 opposed at 28-32. Seed 931009
is 24/24 aligned at 8-30, mixes 14 aligned/9 opposed/1 within-band at 31,
and ends 24/24 aligned at 32. Raw and paired final classifications agree.
These are the older local-continuation reference signs, not current-policy
optimality labels. Their agreement alone is not a promotion criterion.

At the first matched decision, previous command memory is exactly zero.
Quiet latent RMS on seen poses at the reserved evaluation times improves
from 6.499e-6 to 3.540e-7 and from 8.672e-6 to 1.153e-7. Broader time coverage
therefore improves this particular seen-pose quiet measurement.

Reserved-pose behavior does not improve consistently. For seed 921009,
first quiet latent RMS grows from 3.470e-5 to 1.121e-4,3.23 times worse.
For seed 931009 it falls from 2.110e-4 to 6.207e-5,29.4% of the control.
The first 500 ms quiet ankle RMS shows the same split:2.523e-5 to 8.462e-5
rad for 921009, and 1.584e-4 to 4.659e-5 rad for 931009. Bias is still present
before memory accumulation; fitting 48 training states is not a general quiet
policy guarantee.

Matched first 500 ms conditional ankle RMS also shrinks. On seen/reserved
poses it is 65.7%/48.5% of the control for 921009 and 7.98%/15.3% for 931009.
All selected cases and live/failure masks remain in the analysis. These
differences use each policy's own pushed/quiet trajectories; they are not
causal memory ablations. Memory/innovation summaries retain their cross term.

## Verification, cost and retained evidence

Independent audit passes 1,036 input hashes,277 output hashes,45 traces and
720 cases. Initial observations/actions/advantages, seeded critic outputs,
first post-update critic outputs and every minibatch permutation match the
sealed 12-anchor control. Each first 48 candidate exactly matches the prior
feasibility result. First-collection actor/Adam/critic replay, all compact
projection/correction/rollback records and the complete physical/objective
audit pass. No repair or repeated physical run was required.

Both sets of 33 probe boundaries contain 1,188 predictions each. All numeric
summaries and actor/optimizer hash links are verified. Initial and final
forward predictions are independently reproduced. Intermediate model states
were not saved or replayed; their recorded observations are linked to the
training hash history. Analysis produces 1,280 paired comparison records.

The runner took 695.05 seconds, about 11 minutes 35 seconds, excluding process
startup and separate audit/analysis. Evidence occupies about 352.23 MiB,
including source snapshots, training records and physical traces. There are
six model files total: initial actor, final actor and final critic per seed.
There is no intermediate policy checkpoint series.

Evidence: `logs/mimickit-training/recovery-coverage-learning-20261009-01/`.
See audit.json, comparison-results.json, comparison-pairs.json,
command-generalization.json, response-diagnostics.json,
update-scale-diagnostics.json and each fit folder's fixed-probes files.

## Next priority: frozen output sensitivity under quiet constraints

Measure whether broader quiet constraints suppress disturbance sensitivity
independently of a particular optimizer proposal. Use the four final actors:
both seeds from the 12-anchor and 48-anchor runs. On EACH SAME frozen actor,
compare the original 12 bank with the expanded 48 bank, using the same 48 quiet
and 24 pushed training inputs. Do not use reserved states as constraints.

Reconstruct the quiet-output Jacobian Q and the paired conditional Jacobian
D = J_push - J_matching_quiet. With the existing 1e-6 cutoff, measure the
conditional sensitivity remaining after quiet-null projection,
D(I - Q_pseudoinverse Q). Avoid constructing a full parameter-space projector.
Report per-row retained sensitivity, conditional Gram/singular spectra and
achievable conditional-output change per unit parameter displacement. Compare
banks within each actor before comparing trained actors, separating constraint
geometry from changes in the learned parameters.

With exact nested constraints, reduced tangent sensitivity is expected;
the useful result is its magnitude and consistency, not merely a decrease.
Check retained-subspace nesting and cutoff effects before making a monotonicity
claim under truncated SVD.

Verify derivatives and projection algebra independently, preserve compact
evidence and report numerical rank limits. Do not impose historical preferred
signs as optimal targets. This is local output sensitivity, not proof of a
global learning impossibility or physical return. No optimizer update,
trajectory, new checkpoint, additional PPO-gradient branch, training-rate
sweep or Arena change is part of this next diagnostic. Its result should
determine whether to change the constraint representation or investigate the
learning signal, rather than launch another longer training run by default.

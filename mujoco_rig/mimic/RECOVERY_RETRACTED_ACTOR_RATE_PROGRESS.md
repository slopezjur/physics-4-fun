# Higher actor rate: larger updates, weaker recovery

Date: 2026-10-09. Completed
[RECOVERY_RETRACTED_ACTOR_RATE_PLAN.md](RECOVERY_RETRACTED_ACTOR_RATE_PLAN.md).
Both fresh seeds completed 24,576 continuation transitions and all 720
evaluation cases. The independent audit passed. Reject both candidates for
promotion: neither adds a deterministic recovery success over its low-rate
control, and both introduce reserved-pose losses. Arena remains unchanged.

## Controlled comparison

Actor Adam learning rate changed from 1e-4 to 1e-3. Critic rate stayed 3e-4.
Fresh initialization, seeds 921009/931009, samples, epochs, reward, weighting,
noise, memory, reset distribution, projection, finite correction and guards
remained fixed. The original collection/critic loop was reused; only the
fresh actor optimizer's rate changed before its first hash and step.

Both candidates accepted all 288 actor updates, with no rejected or skipped
slots. Both controls also accepted 288. This comparison changes update size,
not the number of accepted updates. It provides no proportional speedup.

## Recovery and standing

For seed 921009, continuous core recovery falls from 7/32 to 5/32 and overall
recovery from 70/128 to 68/128. Matched-start core falls from 7/32 to 6/32 and
overall from 70/128 to 69/128. There are no new successes. Continuous losses
are 150 N -X at 1.0 seconds on reserved poses 6 and 7; the pose-7 loss remains
with identical source-derived pre-hit starts. Reserved core recovery falls
from 2/8 to 0/8 continuous, or 1/8 matched. Sampled core falls from 10/32 to
9/32, losing reserved pose 7 under -X at 1.7 seconds.

For seed 931009, core stays 7/32, including reserved 2/8, in both execution
modes. Overall recovery falls from 69/128 to 67/128 in both modes. The two
new losses are 300 N +Y at 1.0 seconds and -Y at 1.7 seconds, both on reserved
pose 7. The previous 300 N -Y loss at 1.0 seconds also remains relative to
zero correction, which scores 70/128. Sampled core stays 10/32 with exact
settled-case membership against the low-rate control.

Both candidates retain 16/16 sixty-second quiet Stand cases and 32/32 matched
quiet cases. Quiet cases include repeated copies of the same poses. All 720
cases contain zero measured footsteps. This experiment still controls one
scalar copied to both ankle-pitch joints over the live frozen Stand parent;
it adds no independent leg-placement action channel.

## Larger updates were actually achieved

Median accepted KL rises from 8.56e-8 to 2.35e-5 for seed 921009 and from
1.28e-7 to 3.65e-5 for seed 931009, approximately 275 and 286 times. Maxima
are 0.001167 and 0.0008304, below the unchanged 0.01 guard. These are
collection-relative policy changes, not measures of physical learning speed.

Out-of-range PPO ratios occur in 27/73,728 and 35/73,728 reused minibatch
rows. Actual surrogate clipping binds in 18 and 21 rows; the low-rate runs
have none. Gradient clipping is distinct and rises from 10/288 and 14/288
attempts to 218/288 and 236/288. Neither KL rejection nor widespread ratio
clipping explains the lack of recovery gain in these runs.

Projection retains median 99.74%/99.70% of actual Adam displacement norm.
Median correction size is 0.1759%/0.2839% of the projected step; maxima are
1.866%/5.590%. Unlike the low-rate runs, 247/234 intermediate candidates
violate a guard before correction, yet all final corrected candidates pass.
Maximum accepted zero-anchor drift is 1.856e-7/7.892e-6 under the 1e-5 bound.
Maximum local correction residual is 2.022e-7/6.249e-6. Both Jacobians retain
rank 12 throughout. The correction keeps updates admissible; admissibility
does not establish improvement of the PPO objective or recovery.

First/last saved collection surrogate changes are -7.27e-6/-8.65e-5 for seed
921009, versus positive 2.14e-5/2.05e-5 at the low rate. Seed 931009 remains
positive at +3.334e-4/+4.626e-4, versus +4.85e-5/+1.42e-5. The first batches
are identical across rates; later batches differ, so last-batch changes are
within each run's own data. Acceptance deliberately has no surrogate-gain
gate. Objective movement is mixed despite larger policy movement.

## Quiet drift grows outside the anchor observations

In matched-start first-500 ms core windows, pushed-minus-quiet ankle-command
RMS rises from 4.36e-5 to 3.40e-4 rad for seed 921009 and from 8.27e-5 to
7.97e-4 rad for seed 931009: 7.79 and 9.65 times. Quiet-command RMS rises
much more, from 1.32e-5 to 6.86e-4 and from 7.97e-5 to 2.87e-3 rad:
51.94 and 35.99 times.

Split by poses, high-rate quiet RMS is 4.16e-5 rad on seen poses versus
1.371e-3 on reserved poses for seed 921009. Seed 931009 gives 9.38e-5 versus
5.731e-3 rad. The reserved-pose values are about 33 and 61 times the seen-pose
values. These summaries pool the existing per-case moments with sample
weights; every selected window has all 30 live intervals. No failed cases
were filtered out.

The 12 quiet anchors constrain poses 0-5 at onsets 44/74 with zero action
memory. They do not constrain every quiet state or reserved pose. Passing
those anchors and staying upright therefore coexist with larger quiet ankle
commands elsewhere. The observed reserved-pose drift is a generalization
problem to investigate, not proof that it caused each failure. These are
closed-loop command comparisons, not identical-input actor derivatives.

## Verification and storage

The audit passes 1,465 input hashes, 273 output hashes, 45 traces and 720
cases. First-collection actor/Adam/critic replay is exact at the declared
rates, including intermediate and final states. Initial samples, advantages,
weights and post-collection critic match the sealed controls; all 32
minibatch permutations match. Both numerical operations are independently
gated. Maximum independent balance-feature error is 2.151e-6.

No source repair, physical rerun or guard relaxation was needed. The exact
previously documented PILOT diagnostic-source exception and archived original
were retained; modern shared sources matched exactly. Post-run analysis
produces 2,624 paired comparison rows and repeatable update-scale diagnostics
for both rates, with hashes of all consumed inputs.

Runner time: 578.70 seconds excluding startup/preflight and separate audit
and analysis. Fits: 119.45/117.74 seconds. Evidence occupies about 215.06 MiB.
Only six initial/final actor/final critic files are retained; no intermediate
checkpoint series was created.

Evidence: `logs/mimickit-training/recovery-retracted-actor-rate-20261009-01/`.
See audit.json, completed.json, comparison-results.json, comparison-pairs.json,
response-diagnostics.json and update-scale-diagnostics.json.

## Next priority: frozen command direction and quiet generalization

Run one bounded offline diagnostic using existing evidence, with no new
training, physics integration or deployable checkpoint. Compare both final
rates and seeds on the same 36 saved probe inputs: 12 quiet and 24 pushed
observations from training poses/onsets. Report raw latent means and paired
pushed-minus-quiet means against the existing reference polarity (+X prefers
negative, -X positive), using the established 1e-7 latent sign band. Report
amplitudes relative to HALF_SIGMA and requested ankle additions after the
existing tanh/slew mapping. The reference polarity was measured under older
weighted-policy continuations; agreement is not proof of current optimality.

Separately use archived deterministic quiet and pushed traces over all eight
poses and both existing core timings. Decompose issued latent commands into
learned innovation and inherited memory: innovation = latent - 0.9*previous
latent. Use the prespecified first 30 decisions after the first force interval,
with same-index quiet windows, retaining every case and explicitly reporting
live masks. Split seen/reserved poses and force directions. Measure quiet
innovation, memory contribution, requested additions and conditional response.
Do not treat own-trajectory differences as same-input derivatives or infer
causal failure mechanisms from correlation alone.

Independently verify frozen weights, probe pairing, memory arithmetic, action
mapping and source hashes. Keep reserved cases diagnostic; do not silently
add them to training or the anchor bank. This check should distinguish wrong
or shared command direction, small useful response and quiet bias outside
the constrained observations before choosing another learner change. No rate
sweep, longer continuation, relaxed guard or automatic Arena promotion.

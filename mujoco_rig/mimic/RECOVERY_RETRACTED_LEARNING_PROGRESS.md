# Finite-correction learning: update blocking resolved, no recovery gain

Date: 2026-10-09. Implements RECOVERY_RETRACTED_LEARNING_PLAN.md with the
documented provenance repair. Both fresh seeds complete 24,576 continuation
transitions and the established 45-trace / 720-case physical suite. The
independent numerical and physical audit passes. Both candidates remain
rejected for promotion; Arena remains unchanged.

## Training result

Both seeds accept all 288 actor updates, with no rejections or skipped slots.
This is 576 accepted updates versus 165 in the two projected-only controls
(83/82). Critic updates, samples, epochs and minibatch sizes remain unchanged.
The raw 5:1 actor weighting gives early samples 48.92%/47.92% of effective
weight, while they occupy 17.10%/16.76% of collected samples.

The correction is applied after every finite projected Adam candidate,
targeting that update's pre-step quiet outputs. Acceptance still uses the
original zero-reference bound 1e-5 and collection-old KL limit 0.01.
Maximum cumulative quiet drift is 5.3784e-9 and 1.1269e-8, and maximum local
correction residual is 4.1910e-10 and 3.9581e-10. Both Jacobians retain rank
12 throughout. No intermediate candidate violates the anchor or KL limit:
correction from the first update prevents drift accumulation, rather than
rescuing already-rejected candidates on these fresh paths.

Median correction size is 0.01946% and 0.02898% of the projected step, with
maxima 0.09038% and 0.09985%. The numerical rejection mechanism is resolved
at this budget. This alone does not establish useful learning.

## Physical result

Seed 921009 scores 7/32 core signed 150 N X cases and 70/128 across all four
direction/strength groups. Seed 931009 scores 7/32 core and 69/128 overall.
Both continuous execution and matched recovery starts give these totals.
Core reserved-pose recovery is 2/8 for each seed.

Every individual settled outcome matches the corresponding projected-only
control: 512 deterministic cases across seeds/modes and 64 sampled core
cases. Sampled recovery remains 10/32 each. This is outcome parity, not
identical trajectories. More accepted actor updates produced no new settled
recovery in this comparison.

Seed 921009 also matches every zero-control deterministic settled outcome.
Seed 931009 adds no zero-control gain and retains one loss in each mode:
300 N -Y at 1.0 seconds, reserved pose 7. By direction/strength, totals are
7/32 at 150 N X, 32/32 at 150 N Y, 0/32 at 300 N X, and 31/32 or 30/32
at 300 N Y. Gains against older weighted controls do not exceed zero-control
performance or establish a new recovery capability.

Both candidates pass 16/16 sixty-second quiet Stand cases and 32/32 matched
quiet cases. The quiet cases include repeated poses. All 720 cases contain
zero measured corrective footsteps. This remains the scalar coordinated-ankle
head over the live frozen Stand policy; this experiment adds no leg-placement
action channel.

## Response and policy movement

Matched first-500 ms pushed-minus-quiet ankle-command RMS falls from
8.03e-5 to 4.36e-5 rad in seed 921009 and from 4.83e-4 to 8.27e-5 rad in
seed 931009 compared with projected-only controls. Quiet-command RMS also
falls, from 3.95e-5 to 1.32e-5 rad and from 2.76e-4 to 7.97e-5 rad.
These are closed-loop command differences on the core evaluation window,
not isolated actor derivatives or proof of useful direction.

Accepted-update KL is small: medians 8.5552e-8 and 1.2776e-7, maxima
8.7628e-6 and 4.8185e-6, versus the unchanged 0.01 limit. No recorded PPO
probability ratio reaches the clipping interval boundaries in either seed:
0/73,728 reused minibatch sample rows per seed. Ratio ranges are
0.94356-1.04013 and 0.97253-1.01852. Gradient clipping occurs in 10/288 and
14/288 attempts; it is distinct from PPO ratio clipping.

The first and last saved collections have positive local weighted surrogate
changes. Seed 921009 gains approximately 2.14e-5 and 2.05e-5; seed 931009
gains 4.85e-5 and 1.42e-5. Their mean-output RMS changes range from 0.0246%
to 0.0962% of the fixed exploration standard deviation on those collections.
Projection retains a median 99.88%/99.86% of Adam displacement norm, and
the finite correction is small. Accepted parameter updates therefore do not
imply large conditional-policy changes or better physical recovery.

See update-scale-diagnostics.json for sealed inputs, aggregate statistics and
the first/last collection calculations. Small KL and absent clipping motivate
a bounded update-size comparison; they do not prove that larger updates help.

## Verification, preflight repair and cost

The independent audit verifies 1,186 input hashes, 273 output hashes, all
compact update records and the full physical suite. First-collection actor,
Adam and critic replay is exact, including both finite operations. First
rollout, advantages, weights and post-collection critic match both controls;
all 32 minibatch permutations match as well. Updated actor predictions match
the new saved evidence. Maximum independent balance-feature error is 1.70e-6.
No numerical or physical audit repair or rerun was needed.

The first launch stopped before training because an added broad source check
against the oldest pilot found an already-replaced standalone exploration
diagnostic. Current sources match both recent controls. Successor wrappers
record the one exact old/current hash pair, verify the archived original and
retain every baseline-output/current-source check. See
RECOVERY_RETRACTED_PROVENANCE_REPAIR.md. Failed -01 source snapshots and
failure.json are preserved; no physical experiment was repeated.

Evidence root:
`logs/mimickit-training/recovery-retracted-learning-20261009-02/`.
See audit.json, comparison-results.json, comparison-pairs.json (1,952 paired
rows), response-diagnostics.json, update-scale-diagnostics.json and completed.json.
Only initial/final actor and final critic are retained for each seed, with
compact optimizer diagnostics and first/last complete batches. No intermediate
checkpoint series was created. Runner time was 602.42 seconds excluding
process startup/preflight and separate audit/analysis. Fits took 124.01 and
122.82 seconds. Successful-run evidence occupies approximately 214.77 MiB;
the stopped preflight snapshot is preserved separately.

## Next priority: one actor-learning-rate contrast

Train fresh seeds 921009 and 931009 from the identical original zero head
initialized with seed 920009, for 24,576 continuation transitions each.
Change ONLY actor Adam learning rate from 1e-4 to 1e-3. Keep critic learning
rate 3e-4, three epochs, minibatches, 288 nominal actor slots, weighting,
reward, noise, memory, reset distribution and all other settings unchanged.
Do not continue either rejected final actor and do not run a rate grid.

Retain both projection and exactly one local-target finite correction per
candidate, with the original zero-anchor bound and KL guard applied after
correction. Preserve rollback, collection stopping and critic behavior. If
the larger rate causes rejected/skipped updates, record that as an outcome;
do not relax guards, add correction retries or compensate with more epochs.

Verify identical first samples/advantages/weights and critic updates against
the sealed 1e-4 retracted controls, all minibatch permutations and independently
reconstructed optimizer math with the new declared actor rate. Track actual
KL, probability ratios/clipping, gradient clipping, quiet drift, correction
size, accepted/rejected/skipped slots and policy response.

Run the same 720-case physical suite and compare individual outcomes against
the sealed 1e-4 retracted and zero controls, retaining historical comparisons
where useful. Require reproducible recovery gains and quiet retention for
promotion. A larger learning rate is not a claim of proportional learning
speed; higher KL, bigger commands or more accepted updates alone do not meet
the physical criterion. No automatic Arena replacement or longer continuation.

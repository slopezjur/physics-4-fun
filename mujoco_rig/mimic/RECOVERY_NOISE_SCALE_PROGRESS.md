# Continuous exploration amplitude - completed 2026-10-09

Decision: nominate half-amplitude continuous noise for a bounded fresh-seed
training comparison. Both reduced amplitudes preserve quiet standing in this
screen. Half amplitude retains more rescuing corrections than quarter amplitude,
but neither establishes a better trained controller or earns Arena promotion.

Protocol: [RECOVERY_NOISE_SCALE_PLAN.md](RECOVERY_NOISE_SCALE_PLAN.md).

## Outcomes

The independent audit passed 32 trajectories / 384 cases. Each amplitude has
96 pushed and 96 quiet cases, using the frozen initial ankle actor. Existing
full-amplitude, cutoff and zero-correction comparators were reused.

- Half noise: push recovery 23/96, quiet whole-episode Stand 96/96.
- Quarter noise: push recovery 21/96, quiet whole-episode Stand 96/96.
- Saved full noise: push recovery 16/96, quiet recovery 59/96.
- Saved full noise with 0.5-second cutoff: pushes 23/96, quiet recovery 89/96.
- Repeated zero correction: pushes 20/96, quiet Stand 96/96.
- Corrective footsteps: zero across all 384 new cases.

Against full noise, half amplitude gained 11 pushed recoveries and lost 4;
quarter amplitude gained 10 and lost 5. Both rescued all 37 full-noise quiet
failures without losing a quiet success. Against cutoff, half amplitude gained
5 pushes and lost 5; quarter gained 8 and lost 10. Both rescued its 7 quiet failures.

Against zero correction, half amplitude rescued 9 pushes but lost 6 successful
cases. Quarter rescued 3 and lost 2. Directly comparing half with quarter,
half gained 6 cases and lost 4. These mixed identities matter more than calling
the aggregate difference a clear winner.

## Timing and sign remain important

Each onset/sign cell contains 24 trials: six training poses and four noise seeds.

- At onset frame 44, half recovered +X 4/24 and -X 12/24; quarter recovered
  +X 2/24 and -X 15/24. Zero correction was +X 0/24 and -X 16/24.
- At onset frame 74, half recovered +X 5/24 and -X 2/24; quarter recovered
  +X 4/24 and -X 0/24. Zero correction was +X 4/24 and -X 0/24.

Half-amplitude outcomes therefore show useful corrections at both timings and
some harm to the easier early negative-X cases. Quarter amplitude is closer to
the zero-control behavior and does not rescue the later negative-X cases.

These are repeated training starts, not 96 independent initial states per
branch. Reserved poses and heldout timings have not been tested at these scales.
The quiet gate here ends at the original six-second episode endpoint; it is not
a new sixty-second stochastic-standing result.

## Reward and exploration coverage

Average full discounted reward improved versus full noise by 3.585/3.726 units
on pushes and 18.658/19.492 on quiet cases for half/quarter amplitude.
Against zero correction, average reward remained lower: -1.547/-1.406 on pushes
and -1.007/-0.174 on quiet cases. Quarter amplitude thus has a smaller average
reward penalty despite fewer rescuing corrections. Neither amplitude dominates
every objective; half is a learning candidate, not a statistically proven optimum.

During the first 30 continuation intervals, half noise maintained the diagnostic
helpful sign above 0.025 rad for at least six intervals in 8/96 pushes; quarter
did so in 0/96. Full noise had 24/96. Neither reduced scale met the 0.060 rad
six-interval criterion. Nonetheless both rescued cases, so these thresholds
remain descriptive rather than necessary conditions for recovery.

Mean early ankle RMS was 0.01746 rad for half and 0.00896 rad for quarter.
Maximum early magnitude was 0.06057 and 0.03373 rad. Actual offsets are shaped
by tanh and the slew limit; halving noise does not exactly halve every action.

The unchanged initial actor has zero learned output and a 0.9 latent-memory
coefficient. The baseline innovation standard deviation is about 0.174356;
half and quarter use 0.087178 and 0.043589. Their unconstrained stationary latent
standard deviations would be 0.2 and 0.1, compared with 0.4 at full noise.
The source Stand controller remains live; no policy learned during this screen.

## Verification

The audit checked 831 input hashes and 70 output hashes; identical start
observations, native state, history and pending commands; the two initial
delayed physical intervals; exact scaled random draws throughout each rollout;
actor means, critic values, latent/history continuity, target limits and delay;
force windows; independent foot geometry, reward totals, survival and step events.
All weights remained frozen. Independent float64 GAE differed from stored
float32 recurrence by at most 8.7642e-5, within the existing numerical tolerance.

Physical rollouts took 130.24 seconds after setup. Evidence occupies about
105.5 MiB including source snapshots, traces and reports. No run or verifier
repair was needed. No checkpoints, production configuration or Arena changes.

Evidence: `logs/mimickit-training/recovery-noise-scale-20261009-01/`.
See `audit.json`, `paired-outcomes.json`, `completed.json` and individual reports.

## Next priority

Run two fresh recovery-start PPO learners with half innovation amplitude,
reusing seeds 921009 and 931009 and the identical initial actor. Keep the prior
24,576-transition budget per seed, reward, state/history inputs, temporal
correlation, start distribution, actor/critic learning rates and three epochs.
Compare against the saved full-amplitude recovery-start learners, not the older
pre-impact-training candidates. Do not resume either previous trained head.

Before training, verify that the same reduced standard deviation is used for
sampling, stored log probabilities, PPO probability ratios and the KL limit.
The sampled evaluation path must also use that scale; the existing evaluation
helper explicitly sets its own SIGMA and cannot be assumed to inherit a patch
to the learner's STD. Keep deterministic evaluation free of sampled noise.

Evaluate ordinary continuous execution and matched recovery starts at both
heldout timings, signed 150/300 N X/Y pushes, reserved poses and sixty-second
quiet Stand, with independent footstep audits. Track paired case regressions,
not only totals. Promote nothing on training reward alone. If both learners
again collapse to near-zero mean corrections, use their collected credit and
critic evidence to choose a targeted follow-up rather than extending duration.


## Successor completed

The two fresh half-noise training runs and their evaluations are complete. See
[RECOVERY_HALF_NOISE_LEARNING_PROGRESS.md](RECOVERY_HALF_NOISE_LEARNING_PROGRESS.md)
for rejection decisions, 720 audited cases and the next saved-batch diagnostic.

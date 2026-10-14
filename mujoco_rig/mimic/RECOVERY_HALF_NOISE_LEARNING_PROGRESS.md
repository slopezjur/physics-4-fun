# Half-noise recovery-start learning - completed 2026-10-09

Decision: reject both candidates. Lower exploration amplitude preserved quiet
standing but did not improve learned deterministic recovery. Keep Arena and
production configuration unchanged; do not extend either run.

Protocol: [RECOVERY_HALF_NOISE_LEARNING_PLAN.md](RECOVERY_HALF_NOISE_LEARNING_PLAN.md).

## Training and evaluation

Fresh seeds 921009 and 931009 each completed 24,576 continuation transitions
from the same initial actor used in the full-noise comparison. Innovation
standard deviation was 0.08717798, half the previous value; temporal correlation,
reward, starts, physical bounds and other PPO settings were retained.
Each performed 288 accepted actor updates with zero KL rollbacks.

Independent audit passed 45 traces / 720 evaluation cases, including ordinary
continuous execution, matched recovery starts, heldout timings, reserved poses,
signed 150/300 N X/Y forces, sampled core pushes and sixty-second quiet standing.

For ordinary deterministic execution:

- Seed 921009: core 150 N X recovery 7/32, heldout 1/8; all four push groups
  combined 68/128. Corresponding full-noise learner: 7/32, heldout 2/8, 70/128.
- Seed 931009: core recovery 5/32, heldout 0/8; combined 65/128. Corresponding
  full-noise learner: 7/32, heldout 2/8, 69/128.
- Zero correction: core 7/32, heldout 2/8, combined 70/128.
- Both candidates retained sixty-second quiet Stand in 16/16 cases.
- Both recovered 32/32 at 150 N Y and 0/32 at 300 N X. At 300 N Y they
  recovered 29/32 and 28/32, versus zero correction's 31/32.
- No corrective footsteps across all 720 evaluation cases.

Starting from matched post-impact states did not resolve the issue. Core
recovery was 7/32 and 6/32, with combined totals 68/128 and 67/128. Matched
quiet standing was 32/32 per candidate over the six-second episodes.

Sampled core recovery was 10/32 and 9/32, versus 7/32 and 6/32 in the preceding
full-noise learner evaluations. Both the training noise and evaluation noise
changed in that comparison, so it cannot isolate a learned-policy improvement.
The deterministic evaluations provide the deployment comparison.

## Paired regressions

Against the corresponding full-noise learner, ordinary execution gained/lost
2/4 cases for seed 921009 and 5/9 for seed 931009. Against zero correction the
same counts were 2/4 and 4/9. These totals include all 128 deterministic push
cases per candidate, not sampled evaluations or duplicated matched trials.

The first seed's unchanged 7/32 core total hides two gained training cases at
the later timing and two lost early cases, including one reserved pose. The
second seed lost both previously successful reserved core cases.

For matched execution, gains/losses against the full-noise learner were 0/2
and 3/5; against zero correction they were 0/2 and 2/5. Case identities and
sign/timing/group information are saved in paired-outcomes.json.

## Actual learned response and PPO evidence

During the first 0.5 seconds of matched core recovery, ankle-addition RMS was
0.001819 rad and 0.007782 rad. The pushed-minus-quiet RMS difference was only
0.000137 and 0.000329 rad; maximum differences were 0.000373 and 0.000756 rad.
Thus the second seed produced larger offsets without a proportionate increase
in impact-specific response. This is descriptive evidence, not a new causal
quiet-command replay experiment.

The KL constraint was not limiting updates. Mean accepted KL was 0.0000638 and
0.0001574; maximum was 0.001136 and 0.003073, below the 0.01 rollback threshold.
No actor update rolled back. Small KL alone does not justify increasing the
learning rate: amplifying a biased correction could worsen recovery.

First and last PPO collection batches were retained. Their clipped surrogate
improved after each recorded update: 0.001702/0.000301 for the first seed and
0.000502/0.000325 for the second (collections 1/32). This confirms local progress
on the recorded PPO objective, not progress on heldout recovery.

Last-batch critic target MSE decreased from 163.78 to 97.66 and from 556.89 to
385.55. Those targets include the critic's own bootstrap estimates; fitting
them better is not an independent calibration result. The saved data now allow
a direct examination of actual training credit instead of only fresh frozen
trajectories.

## Verification and cost

Analytical probability checks passed before training. Sampling, stored log
probabilities, PPO ratios and KL all used the reduced standard deviation.
The evaluation helper's separate SIGMA setting was explicitly overridden;
saved actor means and exact RNG draws verified the sampled evaluation scale.

The audit checked 731 input hashes and 141 output hashes, unchanged source
weights, identical initial actors, reset/transition accounting, heldout
exclusions, first/last batch likelihoods and GAE, native states, memory,
command delay and limits, force windows, independent foot geometry and outcomes.
Maximum independently reconstructed balance-feature error was 2.05e-6.
Original sealed source files and comparison evidence were preserved.

Training took 101.79 and 102.23 seconds; the full run including evaluations
took 533.76 seconds. Evidence is about 193.9 MiB. Only initial/final actors,
final critics and first/last batches were retained, without intermediate
checkpoint series. No run or verifier repair was necessary.

Evidence: `logs/mimickit-training/recovery-half-noise-learning-20261009-01/`.
See audit.json, paired-outcomes.json, response_diagnostics.json and collection
files under each fit folder. The later response analysis records its own source
and input hashes, separately from the physical-run completion manifest.

## Next priority

Audit the actual first PPO collection of each seed by quiet, positive-X and
negative-X branches. Reconstruct branch labels from ordered reset records and
terminal masks, verifying all assignments rather than inferring force from the
observations. Use the saved initial actor, collected actions, conditional means,
old log probabilities and normalized advantages.

Compute branch-specific score-function gradients, their agreement or
cancellation, and the contribution of a shared output bias versus state-dependent
features. Reconstruct the original first update and verify its saved post-update
means before interpreting optimizer behavior. This can distinguish small useful
updates from credit that mostly favors common offsets. It is a bounded saved-data
diagnostic; no new physics, full training run or Arena change is needed.

Then choose a targeted change to credit or optimization based on that evidence.
Do not increase duration or learning rate solely because deterministic outputs
are small. Reliable reactive recovery, natural foot placement and full Perturb
admission remain open.


## Successor completed

The first-collection PPO replay and branch-gradient audit are complete. See
[RECOVERY_PPO_GRADIENT_PROGRESS.md](RECOVERY_PPO_GRADIENT_PROGRESS.md)
for the corrected diagnosis and the next per-update instrumentation step.

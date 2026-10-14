# Half-noise recovery-start learning

Approved successor to RECOVERY_NOISE_SCALE_PROGRESS.md, 2026-10-09.

Run fresh seeds 921009 and 931009 from the identical initial ankle actor (seed
920009), with 24,576 continuation transitions each. Change only continuous
innovation standard deviation from SIGMA to SIGMA/2. Keep the 0.9 latent
correlation, starts, quiet fraction, reward, bounds, actor/critic learning rates,
64-step collections, three epochs, minibatches, gamma, lambda and KL limit.

Sampling, stored likelihoods, PPO ratios, analytical KL and sampled evaluations
must use the same reduced scale. Check analytical log probabilities, ratios and
KL before training and runtime scale consistency. Save only initial/final actor,
final critic and first/last collection batches for credit diagnostics. Preserve
the frozen source controller and prior sealed sources and evidence.

Evaluate each candidate through ordinary continuous execution and matched
recovery starts at 1.0 and 1.7 seconds, signed 150/300 N X/Y forces, reserved
poses, sampled core pushes and sixty-second deterministic quiet Stand. Retain
the established 45-trace / 720-case protocol, including exact baseline controls.
Record actual sampled actor means and verify innovations against the fixed RNG.

Independently audit physics, memory, delay, foot geometry, outcomes, training
accounting, probability evidence and frozen source weights. Compare case gains
and losses to both the corresponding full-noise learner and zero correction.
No Arena promotion or production configuration changes. No longer run follows
automatically from better training reward. Evaluate early pushed-minus-quiet
ankle response before declaring reactive recovery.

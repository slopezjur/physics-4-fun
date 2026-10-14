# Frozen exploration and credit check - 2026-10-09

Use source-prefix recovery starts at training onset frames 44 and 74, poses 0-5
only (worlds 0-5 and 8-13), signed 150 N X for six force intervals and matching
quiet starts. Restore immediately after the first interval; preserve native state,
queue and history, and continue to the original six-second endpoint.

For each of the four timing/quiet-push combinations, run zero correction and the
two established +/-0.075 rad, 12-interval ankle pulses. Also run the initial head
and both final recovery-start heads in deterministic mean mode and under four
fixed Gaussian innovation seeds (951001-951004). Use identical innovations for
matched policy comparisons. There are 72 continuations x 12 training worlds =
864 cases: 576 sampled, 144 mean-policy, and 144 diagnostic controls. No reserved
pose participates and no optimizer runs.

Record full physical/action/reward traces, the conditional mean and innovation,
and values from a zero initial critic and both saved final critics. Retain policy
features for the first 64 intervals and their endpoint for independent model
verification. Compute the existing gamma=.995, lambda=.98 GAE recurrence on each
64-interval segment, stopping at a physical termination and bootstrapping its
nonterminal boundary. Report raw advantages and within-segment normalized values
over live samples. Dead worlds remain frozen; this diagnostic does not recreate
PPO's immediate autoresets or the historical mixed-episode training batches.

Compare recovery and reward to the identical zero-correction start, and compare
the first-state GAE with the zero continuation under the same critic. Because the
starting state/value is identical, the difference measures the critic's ranking
of the first 64-interval continuations. It is not a causal advantage for one
isolated action or a reconstruction of historical policy gradients.

Measure early actual command coverage in the first 30 intervals (about 0.5 s):
maximum magnitude and longest run in the diagnostic-helpful sign above 0.025 or
0.060 rad. Helpful sign is negative for +X and positive for -X, based on prior
probes; this label is analysis-only and never reaches an actor. Thresholds describe
coverage relative to known probes, not necessary or sufficient recovery criteria.

Check exact restored starts, immutable model weights, generated Gaussian noise,
conditional action composition, real limits/delay, histories, critic predictions,
GAE arithmetic, force windows and independently audited physical outcomes. Keep
reward, exploration variance and physical/optimizer settings unchanged. No model
export, checkpoint creation or Arena replacement.

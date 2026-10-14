# Recovery noise cutoff diagnostic

Approved continuation of RECOVERY_EXPLORATION_PROGRESS.md, 2026-10-09.

## Question and intervention

Does continued random exploration destroy a recovery or quiet stance that was
still viable after its first 30 continuation intervals (0.50004 seconds)?
Reproduce the previous sampled trajectories through interval 29, then set only
the Gaussian innovation to zero. Continue evaluating the same frozen actor;
retain latent memory, history, physical state and pending commands. There is no
reset or target clamp at the cutoff. The cutoff is a diagnostic timing signal,
not a proposed deployable impact detector.

## Fixed scope

- Actors: initial, 921009 and 931009; frozen source controller and two critics.
- Onsets: frames 44 and 74, quiet and signed 150 N X push branches.
- Noise seeds: 951001 through 951004; unchanged Gaussian amplitude.
- Training poses 0 through 5, both signs: 12 worlds per trajectory.
- 48 continuations, 576 cases, original six-second endpoint.
- Reuse saved continuously noisy and zero-correction comparators from
  recovery-exploration-20261009-01. No optimizer updates, checkpoints or Arena changes.

## Acceptance and analysis

Seal source, model and comparator hashes. Require exact matching starts, every
recorded transition field through interval 29, feature inputs through interval
30, and physical qpos/qvel/active-target prefixes through interval 31 (the
two-interval action delay). Independently check Gaussian draws before cutoff,
zero innovations afterwards, actor means, memory continuity, target composition,
delay, applied forces, foot geometry, outcomes and 64-interval GAE.

Compare paired settled gains and losses, survival and discounted reward against
continuous noise and zero correction. Separate quiet and pushed cases. Decompose
first-window reward and frozen-critic credit to assess whether quiet ranking
errors remain. Do not claim historical PPO causality or learned stepping.

# Continuous recovery exploration amplitude comparison

Approved successor to RECOVERY_NOISE_CUTOFF_PROGRESS.md, 2026-10-09.

## Intervention

Freeze the initial coordinated ankle actor and its source Stand controller.
Compare continuous Gaussian innovations at 0.5 and 0.25 times the established
innovation standard deviation. Keep the same standardized draws at every
interval, latent recurrence, action limits, physical model, reward and starts.
There is no noise cutoff or new impact-timing input to the actor.

The initial actor has zero learned output but retains its sampled latent memory
with coefficient 0.9. Reducing innovation amplitude therefore changes the
exploration process, not a trained correction. The existing tanh, slew, joint
and torque limits still determine actual physical actions.

## Scope

- Onset frames 44 and 74; quiet and signed 150 N X force branches.
- Training poses 0 through 5 in both signs, 12 worlds per trajectory.
- Noise seeds 951001 through 951004; original six-second endpoint.
- Two amplitudes, 32 new trajectories, 384 cases.
- Reuse initial-actor full-amplitude and 30-interval-cutoff comparators and
  zero-correction controls from the preceding audited experiments.
- No optimization, checkpoints, heldout admission or Arena changes.

## Verification and decision

Seal source, model and comparator hashes. Verify identical starts and the two
initial delayed physical intervals, exact scaled Gaussian draws at all times,
live actor means, latent/history memory, command composition and delay, force
windows, independent foot geometry, reward totals, survival and settled outcomes.
Check saved 64-interval GAE independently with the established rounding tolerance.

Compare paired gains/losses and full discounted return against full noise,
cutoff and zero correction. Record early helpful-sign coverage, actual ankle
amplitudes and audited footsteps. Inspect onset and sign splits rather than
choosing by aggregate success alone. The repeated starts are not independent
heldout samples. If quieter exploration also removes useful corrections, do not
claim it solves learning. Use the result to select the next bounded experiment.

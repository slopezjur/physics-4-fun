# Recovery noise cutoff - completed 2026-10-09

Decision: continued exploration noise causes substantial quiet-stance damage
and some additional recovery failures. Stopping innovations after 0.5 seconds
helps, but does not produce a useful learned recovery policy. No promotion.

Protocol: [RECOVERY_NOISE_CUTOFF_PLAN.md](RECOVERY_NOISE_CUTOFF_PLAN.md).

## Matched outcomes

All 48 continuations / 576 training-pose cases passed the independent audit.
Each actor has 96 pushed and 96 quiet cases. The saved continuously noisy
comparators were reused; no baseline runs or training were repeated.

- Initial actor: pushed recovery increased from 16/96 to 23/96 (7 gains,
  no losses); quiet standing increased from 59/96 to 89/96 (31 gains, 1 loss).
- Actor 921009: pushes increased from 18/96 to 23/96 (6 gains, 1 loss);
  quiet standing increased from 58/96 to 91/96 (33 gains, no losses).
- Actor 931009: pushes increased from 18/96 to 23/96 (5 gains, no losses);
  quiet standing increased from 60/96 to 89/96 (31 gains, 2 losses).

The three cutoff actors recovered exactly the same 23 pushed case identities.
Against repeated zero correction (20/96), each gained 13 cases but lost 10.
Quiet zero correction retained 96/96, versus 89/91/89 with early exploration.
All 576 cases had zero audited corrective footsteps.

The 96-case totals repeat 24 physical starts across four noise seeds; they are
not 96 distinct initial states or heldout certification. Both onset timings are
training timings and poses 6/7 remain excluded.

## What the intervention establishes

The first 30 intervals were identical, including physical states, actions,
rewards, values and controller history. Inputs to interval 30 were also exact;
qpos, qvel and active targets matched through interval 31 because commands take
two intervals to reach the actuators. Subsequent Gaussian innovations alone
were removed. The actor kept running with its actual latent memory and live
observations; neither targets nor memory were reset.

Therefore later innovations caused the paired outcome changes in these cases.
Removing them rescued 31/33/31 quiet cases, but also lost 1/0/2 quiet cases;
some random continuations had been helpful. No case had physically failed
before the cutoff. Remaining failures occurred afterwards, so survival at
0.5 seconds was not evidence that the early command sequence was safe.

Full discounted reward improved versus continuous noise in 90/96 quiet cases
for each actor and 62/61/62 pushed cases. However, versus zero correction,
average full return remained lower for all actors on both branches. Only one
quiet case per actor exceeded the zero-control return. The extra three pushed
successes therefore do not establish a better overall recovery objective.

This is an ankle-only exploration diagnostic. It has not taught weight transfer,
foot placement, stronger-hit recovery or a new Arena behavior.

## Credit estimates

Against matched zero correction, the learned critics reversed the sign of the
same-window lambda-weighted reward difference in 44/96 and 28/96 quiet cases;
41 and 25 were negative reward differences turned into positive GAE differences.
The preceding continuously noisy comparison had 39 and 24 quiet reversals.
Stopping later noise therefore does not remove this critic sensitivity.

On pushes, the corresponding sign reversals were 1/96 for each learned critic.
Against continuously noisy tails instead, push reversals were 43/96 and 47/96;
these are different counterfactual comparisons and must not be conflated with
the zero-control ranking. Per-case reward and critic contributions are saved.

These are frozen-critic decompositions, not proof of past PPO gradient errors.
A value contribution can legitimately disagree with a truncated reward window.
Nor does the initial zero critic ensure agreement with a full six-second outcome:
early local rewards and eventual recovery remain different quantities.

## Verification and evidence

The audit checked 725 input hashes and 102 output hashes; frozen weights,
matched starts and prefixes; Gaussian draws and cutoff; actor means and critic
values; latent/history continuity; bounded target composition and delay; force
windows; independent foot geometry, survival, settling and step events.

Independent float64 GAE differed from stored float32 recurrence by at most
9.9241e-5 (absolute tolerance 1e-4, relative 2e-6). First-state credit decomposition
error was 9.7830e-5. No optimizer updates or new checkpoints. Arena unchanged.

The physical comparison took 192.89 seconds after setup. Evidence occupies
about 152.1 MiB, including source copies, reports and traces. The successful
audit required no runner or verifier repair and no repeated physical rollouts.

Evidence: `logs/mimickit-training/recovery-noise-cutoff-20261009-01/`.
See `audit.json`, `paired-credit.json`, `completed.json`, and individual reports.

## Next priority

Test whether lower continuous exploration can retain useful push corrections
while preserving quiet standing, without an impact-timed cutoff. Use the initial
frozen actor, the same two onsets, branches, four standardized noise streams and
training poses. Compare half and quarter innovation amplitude against the saved
full-amplitude and cutoff results: 32 new trajectories / 384 cases total.

Keep temporal correlation, reward, command bounds and all models fixed. Change
only innovation amplitude, continuing to sample at every interval. Compare
paired case identities and reward, not aggregate success alone. This tests a
simple exploration setting that can use the existing PPO probability model,
without treating deterministic tail actions as Gaussian samples or introducing
an impact clock into deployed behavior.

If reducing amplitude removes the useful corrections along with quiet damage,
do not call it a solution: that would motivate structured exploration or critic
work rather than a longer run. Select any subsequent bounded fresh-seed training
comparison only after this result; full Perturb and learned stepping remain open.


## Successor completed

The proposed continuous-amplitude comparison is complete. See
[RECOVERY_NOISE_SCALE_PROGRESS.md](RECOVERY_NOISE_SCALE_PROGRESS.md)
for 384 audited cases and the next bounded training comparison.

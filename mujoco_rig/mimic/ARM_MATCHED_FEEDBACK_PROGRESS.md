# Matched feedback and private queue investigation

2026-10-08. Completed and audited. Continues ARM_SAMPLING_BALANCE_PROGRESS.md.

## Finding and decision

The selected early failure states are recoverable under the existing physics
and command constraints: the physical teacher completed all 24 targeted cases
from exact learned prefixes. The learned controllers remain sensitive to small
command disturbances and to the history preceding a switch.

There is now a measured input/target mismatch. Both task heads mask the queued
motor commands. In 96 private counterfactuals with identical current body state,
goal and task-visible features, changing one queued target changed the teacher's
raw target every time, while both heads' raw outputs stayed bitwise identical.
The frozen Stand anchor still sees the queue, so the complete issued command
is not queue-blind. Its indirect response did not reproduce the teacher's
queue-dependent correction in these probes.

This identifies withheld information needed to reproduce the teacher mapping
exactly. It does not prove a queue-aware learner will pass the skill, that the
teacher's target is the only successful action, or that this is the sole cause
of the failures. No new model was trained or admitted. Arena remains unchanged.

## Protocol and artifacts

- ARM_MATCHED_FEEDBACK_PLAN.md: 32 continuous eight-pose episodes, 256 cases.
  Evidence: `logs/mimickit-training/arm-matched-feedback-20261008-01`.
- ARM_MATCHED_QUEUE_PROBE_PLAN.md: 120 private teacher queries from 24 saved
  switch states, with no additional continuation rollouts.
  Evidence: `logs/mimickit-training/arm-private-queue-20261008-01`.

Switch frames were frozen at the prior run's first issued-command difference
above 0.001 rad. Both seeds and both timings were used. Each origin controller
was continued, switched to the other head, or handed to the teacher on poses
0, 4 and 6. Other poses retained their origin policy in teacher episodes.
Teacher use on heldout pose 6 was diagnostic; no labels were admitted to fitting.

Each physical branch reached its switch through deterministic native replay.
No body state, delayed queue or clock was overwritten during continuations.
All archived controls and per-world prefixes reproduced exactly. Switch-state
fingerprints included integration state, pending commands, control history,
task state, observations and gate history. All teacher queries left live state
unchanged. Existing bounds, source policy, delay, hold and return gates remained
in force.

## Controller crossovers

Full eight-pose results below are continue / switch to the other head / teacher
on the three focus poses. Teacher totals include five untouched learned cases.

- Seed 711028, nominal, replay prefix: **5 / 4 / 7**.
- Seed 711028, nominal, 5% prefix: **5 / 4 / 7**.
- Seed 711028, varied, replay prefix: **5 / 5 / 7**.
- Seed 711028, varied, 5% prefix: **4 / 5 / 6**.
- Seed 721028, nominal, replay prefix: **3 / 1 / 5**.
- Seed 721028, nominal, 5% prefix: **3 / 4 / 5**.
- Seed 721028, varied, replay prefix: **2 / 4 / 5**.
- Seed 721028, varied, 5% prefix: **4 / 3 / 7**.

Across the focus cases alone, uninterrupted learned control completed 6/24,
controller swaps completed 7/24, and teacher continuations completed 24/24.
These are paired deterministic cases, not independent statistical trials.

For the first seed's nominal heldout pose 6, replay completed the original run.
Switching replay to the 5% head failed. The 5% head's original run also failed,
and switching it back to replay at the designated frame did not restore a full
pass. Teacher continuation completed the case from either prefix. Prefix and
controller choice both matter; choosing one head globally is insufficient.

## Physical response to small command pulses

Eight diagnostic episodes applied one requested +/-0.005-rad correction to
UpperArm_L_ry or UpperArm_R_ry at the switch, under each post-switch head.
Only poses 0, 4 and 6 received pulses, giving 24 actual pulse probes. The
existing authority/fade/slew projection transmitted all 24, with target changes
of 0.001050-0.002813 rad. Native positions remained exactly equal to the
unperturbed continuation for the first two control intervals, as required by
the unchanged command delay.

Compared with their matched unperturbed continuation:

- 9 formerly incomplete cases became complete.
- 2 formerly complete cases became incomplete.
- 2 probes introduced native physical failure in pose 0, which was already
  incomplete under the unperturbed full-skill gate. These are separate from
  the two complete-to-incomplete regressions.
- In all 24 probes, motor-position RMS differences were larger at 60 control
  frames after the pulse than at 12 frames.

Median motor-position RMS difference grew from 0.001014 rad at about 0.2 s to
0.01437 rad at about 1 s. The median within-case ratio was 14.49. This is a
finite disturbance response during a changing task, not an eigenvalue,
Lyapunov estimate or proof that every direction is unstable. Improvements as
well as regressions occurred; the result demonstrates consequential sensitivity.

## Instantaneous input probes

There were 288 central-difference probe pairs over both controllers, both
upper-arm axes, all selected switch states and three input types. These were
input-only probes; they did not move the physical body.

The own-axis issued-target response to a position input perturbation was
positive but below 0.14 rad/rad in all probes. Its response to a velocity input
perturbation was negative, from approximately -0.75 to -0.065 seconds. There
was no simple reversed-sign own-axis velocity response in the sampled states.
These instantaneous responses do not establish stability under delay and
cross-joint coupling.

Changing the oldest queued target input produced exactly zero change in each
task head's raw output, as expected from its columns-95:229 mask. The frozen
Stand anchor's target and the final issued target did respond. The input mask
was an intentional earlier ablation; these results do not justify blindly
reverting to the old raw-history representation, which had feedback problems.

## Private queue counterfactuals

Saved integration states were reconstructed in a separate scratch simulator,
solely for private teacher predictions. All 24 zero-change reconstructions
reproduced the executed teacher's effective raw target exactly: maximum error
0.0 rad. This scratch reconstruction was not used to generate or improve any
physical continuation result.

Each state had four counterfactuals: +/-0.005 rad in either upper arm's oldest
queued target. Current body state, velocity, goal, authority and previous
addition stayed fixed. The queue changed only in a copied control object; the
source-target change was recomputed from the corresponding Stand observation.
The original scratch state and original queue remained unchanged by queries.

Across the 96 counterfactuals:

- Teacher raw target changed in 96/96. Median maximum-coordinate change:
  0.001220 rad; maximum: 0.055394 rad. 66/96 exceeded 0.001 rad.
- Teacher issued target changed in 96/96. Median maximum-coordinate change:
  0.000520 rad; maximum: 0.021962 rad. 20/96 exceeded 0.001 rad.
- Both task heads' raw outputs were bitwise unchanged in all 96 cases.
- The heads' issued targets changed indirectly through the Stand source:
  median maximum-coordinate change 0.000162 rad; maximum 0.000430 rad.

All counterfactual queued targets remained within joint limits. These are
local synthetic queue alternatives; the study did not establish that every
alternative is reached by the current learned policy with precisely the same
body state. The teacher may also have multiple physically successful targets.
The demonstrated mismatch is with exact teacher imitation, not a proof that
every successful policy must reproduce these exact outputs.

## Next priority

Design a bounded representation of pending-command effects, then validate its
timing and frozen prediction parity before another fit. Candidate information
includes pending target errors or the body state predicted after the two
committed commands; choose the smallest justified addition. Preserve explicit
separation between the Stand source target and the task correction.

If that representation passes admission checks, compare it with the current
masked head using the same initialization capacity, fitting budget, complete
quiet data, paired seeds and heldout poses. Evaluate closed-loop hold/return,
small disturbances and timing changes, rather than selecting on teacher loss.
Keep the proven state/queue reconstruction and exact replay controls.

No more sampling-ratio search or longer PPO run is the next action. Neither
current head is ready to progress to weight transfer, single-foot support or
learned recovery stepping. The Arena continues to use its earlier recovery
brain; this block did not create an updated Arena policy.

## Verification

- Matched-branch `audit.json`: passed; 32 episodes, 256 physical cases,
  8,340 executed teacher queries, exact prefixes/states and command projection.
- Matched-branch `summary.json`: controller matrix, pulse responses and input
  sensitivities, with source/evidence hashes.
- Private-queue `audit.json`: passed; 120 queries, exact reconstruction parity,
  unchanged masked outputs and exact saved command projection.
- No training updates, PPO transitions, new fitting dataset, C# compilation,
  spring/rig changes or Arena changes. All experiment processes completed.

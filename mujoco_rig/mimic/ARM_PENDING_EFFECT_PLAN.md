# Pending-command effect comparison

Status: predeclared, 2026-10-08. Successor to ARM_MATCHED_FEEDBACK_PROGRESS.md.

## Question and input contract

Does exposing bounded motor effort from the two committed commands improve
learned arm hold/return at the same fitting budget? This addresses the observed
teacher dependence on information masked from the task head. It does not assume
that exact teacher imitation is necessary for every successful controller.

Keep the inherited normalized features outside columns 95:229. In each of the
two chronological 67-column queue slots, replace the first 33 columns with
`clip(4*(queued_q-current_q) + 0.08*(queued_v-current_v), -1, 1)*valid`,
zero the next 33 columns, and retain the valid bit. Decode observed hinge
velocity with `current_v = observation[48+motor_v_index]*10`; observations
store hinge velocities at one tenth of native units. Position is at
`observation[8+motor_q_index]`. The gain and damping match DelayedTargetControl33.

These are present-state normalized torque demands, not predicted future body
states. The second slot is evaluated at the same current body state; actual
torque will change as the body evolves before that slot executes. No teacher,
physics predictor, state restoration or assistance executes during evaluation.
Saturation deliberately discards target-error distinctions beyond torque limits.

## Admission before fitting

- Verify dataset bounds, invalid-slot zeros, outside-queue feature identity,
  initial zero-output identity and fixed 421-128-128-33 capacity.
- At all 24 audited switch states, compare both slot features against the actual
  native PD control method on isolated controls. Verify chronological queue,
  native position/velocity decoding and unchanged original state/control.
- Probe the same 96 bounded queue alternatives used in the prior private study;
  record changed features and clipped alternatives. Do not add their labels.

## Matched comparison

- Four fresh fits: masked and pending-effect inputs, seeds 711028 and 721028.
- Use only the original 9,072 teacher rows: 6,804 training rows from poses 0:6,
  2,268 validation rows from poses 6:8. Preserve all 2,256 quiet training rows.
- Identical initialization, 4,000 Adam updates at 1e-4, batch 512 and draw stream
  per seed. No correction mixture, capacity, goal, spring, limit or reward change.
- The masked fits must reproduce archived final weights and nominal/varied
  physical traces exactly. Save only initial/final weights, no interval checkpoints.
- Report training/heldout raw residual errors; do not select by regression loss.

## Frozen physical evaluation and decision

Evaluate all four heads on nominal and varied schedules, then an unused
`onset=0.9, ramp=1.4, hold=3.2` schedule and nominal raw-proposal noise of
0.005 rad with streams 710083 and 710084. Noise is a diagnostic disturbance,
not a projectile or proof of external-hit recovery. Save its exact stream.
Evaluate both pending-effect seeds for 60 seconds of quiet Stand against the
archived exact baseline.

The original full gate remains: final two seconds of hold with support,
motor RMS <=0.05 rad, COM error <=0.02 m, each foot drift <=0.02 m, survival,
and a final two-second return to Stand. No widening or per-case override.
Rank core results by worst seed/schedule, then total successes; report challenge
results separately. Research improvement requires a better core rank without a
worse challenge rank and exact quiet Stand. Complete admission requires every
core and challenge case for both seeds. Otherwise retain the existing baseline.

Budget: 16,000 optimizer updates, 8,192,000 sample presentations, 192 physical
evaluation cases; zero PPO transitions and zero new teacher-label queries.
Independently reconstruct features, sample streams, gates, replay provenance
and artifact hashes. No Arena update or learned stepping claim in this block.

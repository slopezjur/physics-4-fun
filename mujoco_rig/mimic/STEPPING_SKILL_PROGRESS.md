# Learned stepping curriculum progress

Date: 2026-10-09. Protocol: STEPPING_SKILL_PLAN.md.

## Current priority

The scalar ankle branch is no longer the main path to learned foot placement.
The new controller has independent target corrections for all 33 joints, explicit
commanded-skill observations, and a measured transfer/lift/place/land/settle
curriculum. This remains a commanded movement learner, not an autonomous response
to a ball impact. The Arena selection is unchanged.

## Implementation and checks

- Added stepping_curriculum.py, stepping_skill_core.py and train_stepping_skill.py.
- Eighteen focused checks passed, including four short native physics checks.
  They cover independent left/right commands, real delayed target application,
  zero initial correction, structural quiet retention, history/reset isolation,
  measured dwell, permitted single support, rejection of sliding, and terminal
  return semantics. No C# build was needed.
- A twelve-second native zero-head comparison exactly reproduces the original
  controller's physical state and targets across sixteen worlds. The unchanged
  controller stands throughout all cases and completes zero commanded transfers.

## Bounded experiment

Status: completed; both candidates rejected at the first curriculum gate.
Evidence: logs/mimickit-training/stepping-skill-20261009-01.

Each seed completed 49,152 transitions and 576 accepted actor updates, with no
KL rollback: 98,304 transitions and 1,152 accepted actor updates in total.
Both started from exactly the same zero-output actor. Fitting took 85.29 and
85.40 seconds; the complete experiment took 464.76 seconds (about 7 minutes
45 seconds), excluding preflight checks, independent audit and saved-trace analysis.

Frozen commanded-transfer outcomes across eight poses and both swing-side commands:

- Seed 941009: 0/16 transfer-and-return successes; 0/16 survived.
- Seed 951009: 0/16 transfer-and-return successes; 4/16 survived and retained Stand.
- Zero-head baseline: 0/16 transfer successes; 16/16 retained Stand.
- Each candidate and the baseline passed all sixteen sixty-second quiet cases.
  These use the exact disabled-command fallback. They establish retention of the
  original controller, not learned active-task stability.
- Neither seed passed the prospective 9/12 training-pose gate. No later onset
  sweep, lift/replant level, placement level, ball impact trial or Arena export ran.
  No corrective step was measured. Survival without moving is not task success.

The independent audit passed all 8 traces / 128 evaluation cases, 662 protected
inputs and 34 sealed outputs. It independently reconstructs whole-foot geometry,
Stand criteria, command timing/phase/dwell, reward components, target composition,
delay, physical bounds, the initial zero-head parity and curriculum decisions.
It also checks identical initial models, first/last collection probabilities and
GAE, and recorded update/sample accounting. It does not replay intermediate
optimizer updates or claim that two retained collections describe all training.

Six model files were retained: initial/final actors and final critic per seed.
There is no periodic checkpoint series. Evidence is approximately 138 MiB before
small final analysis additions. Existing experiment sources and Arena assets
were not edited. The bounded experiment completed on its first execution.

Metadata detail: evaluation report distance_m records the runner's default
0.04-m argument. Level 0 normalizes the requested FOOT displacement to zero;
the separate COM transfer target is 0.04 m. Saved task commands and independent
phase reconstruction confirm that no level-0 foot displacement was requested.

The earlier 13/16 scripted lift/replant successes establish partial physical
feasibility only. They are not used as runtime control or as a substitute for
the new learner's measured outcome.

## What the saved commands show

Both deterministic candidates produce zero valid transfer-gate intervals. The
first seed fails physically at 2.65-4.40 seconds. For the second seed, all twelve
training-pose cases fail at 3.00-3.52 seconds; the four survivors are reserved
poses 6/7 under both side commands. Neither survivor group completes the task.

First-500-ms applied correction RMS is 0.04621 and 0.02690 rad. These corrections
are appreciable; this is not a repeat of the near-zero ankle response result.
At the first active decision, paired opposite-side requests have exactly matching
physical state, history, source actions and previous correction memory. Their
action-vector cosine similarities are 0.99949-0.99997 and 0.99415-0.99966:
the initial responses are almost parallel despite different requested sides.
That identifies weak useful side differentiation at entry, not proof that the
policy ignores every goal input or that one implementation detail caused failure.

Every failing case crosses the 2 cm stance-drift threshold before its native
physical failure. In the first seed, apparent progress along the transfer direction
also has large perpendicular COM target error and insufficient stance loading.
The failure is therefore not explained solely by a strict completion threshold.

Saved first/last training batches contain only WAIT and TRANSFER observations,
with no completed-transfer or settled-success observations. They cover 1,536 of
49,152 transitions per seed (2/64 collections), so no all-training visitation
rate is inferred. See diagnostics.json and its separately hashed
diagnostics-inputs.json; analysis uses archived arrays without physics or model
execution. A Windows manifest-separator issue in the analysis was fixed before
its outputs were written; no experiment was rerun.

## Next priority

Obtain physically verified, side-specific transfer-and-return examples through
the exact current observation, action-envelope, slew and delay interface. Start
with opposite commands from matched initial states, and require actual COM/load
transfer and returned Stand. The old successful teacher used a different envelope
and approximate reconstructed observations; its success cannot certify this
new interface or supply exact current labels without revalidation.

If both directions are feasible, use exact issue-time examples and states near
those successful trajectories to initialize a goal-conditioned transfer learner.
Judge the neural learner alone in closed-loop operation across the two seeds and
the existing progression gate before returning to PPO or the lift level. Teacher
motor commands are training-data evidence only and must not enter the evaluated
learner or Arena. If the examples are not feasible under current bounds, report
that physical limitation before another fit.

Do not extend this rejected PPO run, weaken its recorded success gate, resume
scalar ankle sweeps, or increase ball strength. The current result establishes
an implemented and checked stepping-learning task, not learned stepping.

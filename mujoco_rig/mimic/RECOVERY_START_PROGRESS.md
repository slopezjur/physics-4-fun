# Matched recovery starts - completed 2026-10-09

Decision: admit the exact native recovery-start mechanism for a bounded learning
comparison. It preserves the archived physical continuations and their bounded
ankle-pulse responses. No policy was trained or promoted in this validation step.

Protocol: [RECOVERY_START_PLAN.md](RECOVERY_START_PLAN.md).

## What changed

`recovery_start_bank.py` stores complete native MjData copies and explicit control,
task and observation-history state. It restores delayed commands in oldest-first
order relative to the live queue cursor. Selected worlds can be restored without
resetting unrelated worlds. Cached solved-contact forces, previous actions/COM,
three balance frames and scalar command memory are retained.

Snapshots are exact in-memory copies tied to the same native model instance.
The saved arrays document their contents; they are not a portable exact native
snapshot loader. Task mode, model and episode horizon remain fixed in this study.
World indices preserve pose identity, including the reserved poses.

For each onset, the frozen source first advances without correction. Quiet and
pushed branches share the same state immediately before impact. After one control
interval, each branch supplies a recovery start. The pushed start is now disturbed
and retains the remaining five force intervals. No optimization sees or controls
the preceding stance in this proposed training setup.

## Physical results

Twelve rollouts covered two timings, quiet/pushed starts, three commands and 16
worlds: 192 cases. Push scores below combine 32 signed 150 N X cases per command.

- Zero correction: 7/32 settled, reserved poses 2/8; +X 2/16, -X 5/16.
- Negative 0.075 rad ankle pulse: 12/32, reserved poses 2/8; +X 12/16, -X 0/16.
  Compared with zero correction: ten rescues and five losses.
- Positive 0.075 rad ankle pulse: 15/32, reserved poses 4/8; +X 0/16, -X 15/16.
  Compared with zero correction: ten rescues and two losses.
- Quiet starts: each command retained whole-episode Stand and settled recovery
  in all 32/32 six-second cases. This is not a new sixty-second admission.
- Audited corrective steps: zero in all 192 cases.

The pulses last 12 command intervals and use the existing slew, envelope, joint
limits and delay. They are diagnostic interventions, not learned execution.
Choosing their sign using the known force direction would be privileged control;
the future actor must choose from its live physical observations.

These outcomes reproduce earlier evidence rather than establish a new controller
improvement. They demonstrate that exact recovery starts retain the known ability
to recover some failing cases with bounded ankle action. One fixed pulse cannot
handle both signs; neither pulse produces an audited recovery step.

## Verification

The independent audit passed 12 traces / 192 cases, 563 input hashes and 34 output
hashes. Eight traces reproduced every shared archived field exactly: both quiet
zero continuations, both pushed zero continuations and all four pushed pulses.
Quiet zero traces matched the first six seconds of the archived sixty-second
source trace. There were no native solver warnings.

Both onset checks passed full restore, isolated restores of worlds 0 and 8,
unchanged other worlds, an odd queue-cursor shift, immutable snapshots and four
identical subsequent intervals against an independent environment copy. The
427-input features were preserved exactly; changing force/timing metadata alone
did not change them. Quiet and pushed branches had identical pending and active
commands at takeover, with different physical states after the first force interval.

Saved-trace checks independently verified start history/memory, queued commands,
the two-interval physical delay, force and pulse windows, target composition,
foot geometry, survival, settled outcomes and footstep events.

The validation took 48.35 seconds after setup. Evidence including source copies
occupies about 39.6 MB. No optimizer updates or model checkpoints were generated.
Arena selection and the earlier sealed sources remain unchanged.

Evidence: `logs/mimickit-training/recovery-starts-20261009-01/`.
Use `audit.json`, `checks.json`, the common/start inspection archives and individual
traces/reports. `inputs.json` and `completed.json` seal provenance and outputs.

## Next priority

Run the bounded recovery-start learning comparison: two fresh zero-output scalar
ankle heads, the previous 24,576-transition budget per seed, and the existing
physical action limits. Sample frozen-source disturbed states at training-only
timings and poses 0-5, plus paired quiet starts. Fit only continuation transitions;
preserve the native queue and real history on every reset. Keep force labels,
episode timing and diagnostic commands outside learned inputs.

Keep the current reward for the first comparison so the recovery-start change
can be interpreted separately. Quiet episodes and the existing correction cost
provide the initial quiet-retention pressure; do not claim that they enforce
exact quiet preservation. If reward changes are later necessary, compare them
separately rather than changing both reset distribution and reward at once.

Evaluate both matched recovery-start continuations and ordinary execution with
the actor active from episode start. The latter is required because an actor
trained only after a disturbance can still alter its stance when run continuously.
Retain reserved poses, both unseen evaluation timings, both signs, broader X/Y
and stronger pushes, quiet sixty-second Stand and the independent step audit.
Reject improvements that exist only behind a privileged takeover gate. Do not
export or replace the Arena brain without admission under ordinary execution.

## 2026-10-09 - Recovery-start learning completed; candidates rejected

See RECOVERY_START_LEARNING_PROGRESS.md. Two fresh heads trained for 24,576 continuation transitions each from exact source states after the first impact interval, with quiet starts and unchanged architecture, PPO, reward and limits. Both accepted all 288 actor updates without KL rollback. Core 150 N X recovery remained 7/32 with identical source success identities in both matched-start and continuously active execution. Overall scores were 70/128 and 69/128 versus source 70/128; both lost one reserved stronger-sideways case. Sixty-second Stand remained 16/16 each; no corrective steps. Independent audit passed 45 traces/720 cases. Early learned corrections were small (first-500ms RMS 0.001408 and 0.000364 rad). Reject promotion and longer continuation; Arena unchanged. Next: use saved training-pose probe reward traces to check whether the exact discounted objective prefers independently successful corrections over doing nothing, excluding prefixes and post-failure frames. Establish reward alignment before changing reward, exploration or training duration. Evidence: logs/mimickit-training/recovery-start-learning-20261009-01/audit.json and response_diagnostics.json.

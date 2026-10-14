# Learned coordinated ankle pilot - completed 2026-10-09

Decision: reject both candidates for promotion or longer continuation. Core X
scores improved, but broader recovery regressed and the saved commands do not
yet establish a useful learned reaction after impact. Arena selection is unchanged.

Protocol: [LEARNED_ANKLE_RECOVERY_PLAN.md](LEARNED_ANKLE_RECOVERY_PLAN.md).

## What ran

The source controller stayed frozen. A new scalar actor coordinated both ankle
pitch additions using 427 inputs: normalized source observations, measured
balance/momentum/support history, previous addition and previous latent command.
The conditional Gaussian supplied persistent exploration. No force label, hit
clock, diagnostic pulse, teacher or source-action replay entered execution.

Two fresh seeds, 921009 and 931009, received identical zero-output initial heads
and 24,576 transitions each. Both accepted all 288 actor optimizer updates; no
KL rollback limited either run. Training took 29.85 and 29.29 seconds. The full
training/evaluation experiment took 369.14 seconds. Evidence occupies about
174 MB, with initial/final heads and final critics only.

Training used poses 0-5, 25% quiet episodes and otherwise signed 150 N X pushes,
with randomized onset frames excluding the evaluation times. The reward remained
the prior direct-recovery reward. Evaluation covered both timings, all eight
poses, both signs, both axes and 150/300 N forces, plus sampled execution and
sixty-second quiet Stand.

## Frozen evaluation

All counts below require survival and the final two seconds continuously meeting
the established balance/posture criteria with both feet loaded at least 30 N.

- 150 N X: source 7/32; seed 921009 11/32; seed 931009 9/32.
- Reserved poses within that group: source 2/8; candidates 2/8 and 0/8.
- X signs separately: source +X 2/16 and -X 5/16; first candidate 8/16 and 3/16;
  second candidate 4/16 and 5/16.
- 150 N Y: source 32/32; candidates 30/32 and 32/32.
- 300 N X: all three 0/32.
- 300 N Y: source 31/32; candidates 8/32 and 20/32.
- Overall deterministic pushes: source 70/128; candidates 49/128 and 61/128.
- Sampled-policy 150 N X checks: untrained head 6/32; candidates 0/32 and 3/32.
- Sixty-second quiet Stand: all three 16/16, covering eight repeated poses.
- Audited corrective steps: zero throughout all 528 evaluation cases.

Quiet gate retention is not exact equality of the trained policies' trajectories.
The zero-output baseline did reproduce every shared saved field in the previously
sealed crossed push and sixty-second quiet references exactly.

## What changed before and after impact

The learned commands were already active before the push. In the preceding
200 ms, ankle-addition RMS was 0.03042 rad for seed 921009 and 0.01647 rad for
seed 931009. RMS in the following 200 ms was almost unchanged: 0.03043 and
0.01644 rad. The near-zero overall means hide pose-dependent positive and negative
corrections; these are not one constant command shared across poses.

Comparing each pushed rollout with its own sixty-second quiet rollout at matching
times, the additional hit-specific command difference over the first 200 ms was
only 0.0000504 and 0.0002104 rad RMS. Over 500 ms it was 0.000565 and 0.000395 rad.
The complete prefixes before impact matched the respective quiet trajectories
exactly, including qpos, qvel, addition, latent command and history.

This is evidence of substantial pre-impact stance modification and weak early
command changes after impact. It does not by itself prove the causal source of
the score gains. We must not describe the increased X score as established
reactive recovery or learned stepping.

## Verification

Preflight checked COM velocity against native COM Jacobian times generalized
velocity (maximum error 2.09e-9), history ordering, idempotent reads, isolated
resets, zero initial output, controller memory and exclusion of force/timing data.

Independent saved-trace audit passed all 33 traces / 528 cases, 556 input hashes
and 85 output hashes. It checked history and sampled-action memory, scalar target
composition, unchanged delay, ankle-only additions, force windows, independently
reconstructed geometry/balance features, outcome counts, identical initial heads,
training budgets and exclusion of reserved poses/times. Maximum independently
reconstructed balance-feature error was 1.77e-6. No evaluation solver warnings.

The original verifier was slow because it repeatedly decompressed NPZ fields
inside its reconstruction loop. It completed successfully. A separate cached
reader then reproduced the same audit with the sealed verification logic
unchanged; array equality and harness hashes are recorded. No training or
physical evaluation was repeated for this audit improvement.

Evidence root: `logs/mimickit-training/learned-ankle-20261009-01/`.
See `summary.json`, `checks.json`, `audit.json`, `audit-harness.json`, individual
reports/traces, training histories and reset schedules. The later
`command_diagnostics.json` contains its own input hashes and is outside the
completion manifest.

## Next priority

Run a frozen causal comparison using identical learned prefixes before impact:
normal learned continuation versus replaying that policy's quiet ankle commands
after impact. Keep the source controller's live feedback in both branches.
This preserves the altered stance while removing the learned correction's
post-impact response, allowing us to measure whether that response contributes
to recovery. Two seeds, two timings and eight poses with both signs require
64 replay cases; the normal continuations are already saved.

Use that result to choose the next correction to the learning objective or
representation. Do not extend PPO, increase physical limits, promote a candidate,
or claim reactive recovery from the current aggregate X score. Broader-direction
retention, reserved-pose recovery, actual stepping and projectile admission remain
unresolved.

## 2026-10-09 - Frozen ankle response comparison completed

See ANKLE_RESPONSE_CAUSAL_PROGRESS.md. Preserving learned prefixes and replaying each policy's quiet ankle commands after the first impact interval changed settled recovery from 11/32 to 10/32 for seed 921009 and left seed 931009 at 9/32 with identical case identities. Only reserved pose 6, +X, onset near 1.0 seconds benefited from normal learned continuation; there were no replay-only successes. All other 63 outcomes matched. This removes only the ankle head's impact response: altered stance, continuing quiet commands and live frozen-source feedback remain. No corrective steps. Independent audit passed all 64 cases, exact prefixes, two queued intervals, command replay, geometry and outcomes. No training or Arena change. This supersedes the previous next priority: validate recovery training starts from identical frozen-source states with preserved queues/history, paired quiet/pushed continuations and exact zero-correction parity, then a bounded comparison that credits post-disturbance recovery rather than stance preparation. Evidence: logs/mimickit-training/ankle-response-causal-20261009-01/audit.json.

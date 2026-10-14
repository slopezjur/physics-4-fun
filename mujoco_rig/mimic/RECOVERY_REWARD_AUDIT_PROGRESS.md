# Saved recovery reward audit - completed 2026-10-09

Decision: retain the current reward. On the tested training-pose continuations,
it consistently ranks rescuing ankle corrections above doing nothing and harmful
corrections below it. This audit does not establish why PPO failed to learn those
corrections; exploration and estimated advantages are the next priority.

Protocol: [RECOVERY_REWARD_AUDIT_PLAN.md](RECOVERY_REWARD_AUDIT_PLAN.md).

## What was measured

The audit reused sealed reward/state traces from `recovery-starts-20261009-01`.
It used only poses 0-5: 24 signed pushed starts with three continuations each,
plus 24 matching quiet starts with three continuations each. That is 144 saved
trajectories and 96 pulse-versus-zero comparisons. Reserved poses 6/7 did not
enter rankings or decisions. Quiet signed-world pairs are duplicates, so the
48 quiet comparisons contain 24 unique pose/timing/pulse comparisons.

Scoring begins at frames 61 or 103, the first controllable decision after the
initial force interval. The return uses gamma 0.995, includes the first failure's
-2 reward, excludes all frozen post-failure frames, and ends at six seconds.
No pre-impact reward is included. Survival and final-two-second settling labels
come from the independently audited physical outcomes, not the reward score.

## Does the reward prefer recovery?

- All 16 pulses that rescued a zero-correction failure received higher measured
  discounted reward. Gains ranged from +86.55 to +114.85 reward units.
- All five pulses that turned a zero-correction success into failure received
  lower reward. Differences ranged from -121.34 to -110.51.
- The 27 pushed comparisons with unchanged settled status had eight higher and
  19 lower scores. Reward also distinguishes trajectory quality within the same
  binary outcome; this category alone does not establish an alignment failure.
- All 48 quiet comparisons scored below zero correction, by 0.23-4.22 units,
  although all retained the six-second Stand criterion.

Choosing the highest measured reward among the three known diagnostic options
recovered 21/24 pushed training starts. Those are exactly the 21 starts for which
any of the three options recovered. This is a retrospective oracle over fixed
interventions, not a learned controller, an untouched evaluation result, or an
Arena improvement. The diagnostic timings inform this investigation; future
changes need fresh timing checks.

## How early is the signal?

Every one of the 16 rescues already has a positive cumulative discounted reward
difference within 0.2 seconds of takeover. Differences are +0.62 to +1.05 by that
point, +3.57 to +5.63 by 0.5 seconds, +11.30 to +17.53 by one second and +33.98 to
the +50.04 maximum by two seconds. Thus the measured benefit is not solely a
distant survival bonus. These are cumulative task-reward differences, not PPO
advantages for individual actions.

Mean full-horizon return differences across rescuing pulses decompose into
approximately +24.39 from posture baseline, +51.59 from capture, +26.30 from
bilateral settling, +1.10 from avoiding the terminal penalty, -0.016 from swing
and -0.0018 from the action cost. The action penalty does not outweigh the reward
for these successful bounded corrections. No reward term was changed.

## Time-limit bootstrap sensitivity

PPO uses a critic bootstrap at time limits, while physical failures have none.
The audit therefore also added the discounted endpoint value from each saved
final critic. Both preserved the desired ranking in all 16 rescues and all five
losses. The observed full-horizon reward comparison is not being presented as
the exact historical PPO target.

The first critic reversed four of the 48 small quiet comparisons and the second
reversed two. These correspond to two and one unique quiet pose/timing/pulse
comparisons respectively. Their resulting positive margins were small (up to
0.317 and 0.027 units). This identifies critic sensitivity in quiet cases; it
does not overturn the pushed recovery rankings or prove a learning bottleneck.

These checks use the final frozen critics on diagnostic continuations. Actual
PPO used evolving critics, 64-interval collections and GAE lambda 0.98. Historical
rollout advantages were not saved, so the audit cannot reconstruct their action
credit or establish that successful pulses would have received the correct
policy gradient during fitting.

## Verification and artifacts

The reward was independently reconstructed into posture, capture, settling,
swing, action-cost and terminal components using saved state and native
kinematics/COM Jacobians. No physics integration occurred. Maximum per-interval
reward reconstruction error was 1.62e-7.

Post-interval 427-input critic features were reconstructed from saved state,
pending commands, real history, latent memory and source normalization. They
matched the saved recovery-start features within 3.58e-7 on the analyzed worlds.
The final critics and source normalization came from the sealed checkpoints.

A separate verifier checked input/output hashes, scanned each first failure,
recomputed returns by backward recursion, checked outcome rankings and oracle
selection, and confirmed reserved-pose exclusion. It passed all 144 trajectories
and 96 paired comparisons. Maximum independent return difference was 1.71e-13.
There are 592 protected input hashes. Evidence occupies about 4.6 MB including
source copies; no new model checkpoints, training or physics runs were created.

Evidence: `logs/mimickit-training/reward-alignment-20261009-01/`.
`results.json` contains every trajectory, reward decomposition, bootstrap score,
paired comparison and temporal window. `audit.json`, `inputs.json` and
`completed.json` record verification and provenance. Arena selection is unchanged.

## Next priority

Run a bounded frozen-policy exploration and credit check on training-only starts,
using the initial and both final ankle heads. Record conditional means and noise,
sampled latent actions, realized ankle additions and duration/sign of corrections,
plus critic values and the same 64-interval GAE calculation used by PPO.

Compare sampled continuations with matched zero-correction controls and the
established bounded pulse reference. Determine whether useful responses are
rarely sampled, whether sampled improvements receive poor estimated advantages,
or whether another explanation is needed. Keep reward, physical limits and
optimizer settings fixed, and perform no optimizer updates in this check.
Do not increase noise, scale outputs or extend training based on the small mean
commands alone. Neither exploration coverage nor historical policy-gradient
quality has yet been measured directly.

## 2026-10-09 - Frozen exploration and credit check completed

See RECOVERY_EXPLORATION_PROGRESS.md. Audited 72 frozen continuations/864 training-pose cases, with no optimizer updates. Sampled pushes recovered 16/96, 18/96 and 18/96 for initial/921009/931009 versus repeated zero-control 20/96; they rescued 8/10/10 cases but each lost 12 baseline successes. Quiet recovery fell to 59/96, 58/96 and 60/96 versus 96/96. Paired own-critic GAE favored every sampled rescue. Same-window reward/critic decomposition found quiet sign reversals in 39/96 and 24/96 for the learned critics, versus 3/96 and 0/96 on pushes. Strong early helpful offsets were rare (2/96 with >=0.060 rad for six intervals), but most rescues did not require that threshold. No corrective steps. Verification passed after repairing the verifier's actor batch shape and accounting for float32 GAE rounding; no physics was repeated. Next: preserve identical sampled first 30 intervals, then remove further innovations while retaining conditional-mean memory and all native state, to isolate damage from continued noise before changing exploration or critic calibration. No training or Arena change. Evidence: logs/mimickit-training/recovery-exploration-20261009-01/audit.json and credit-decomposition.json.

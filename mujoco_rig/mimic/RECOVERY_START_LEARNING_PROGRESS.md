# Recovery-start learning comparison - completed 2026-10-09

Decision: reject both candidates for promotion or longer continuation. The new
reset mechanism worked, but neither learner improved core recovery. Quiet Stand
was retained, while one reserved stronger-sideways case regressed in both seeds.
The existing Arena controller remains selected.

Protocol: [RECOVERY_START_LEARNING_PLAN.md](RECOVERY_START_LEARNING_PLAN.md).

## What ran

Two fresh scalar ankle heads began at exactly the same zero-output weights as
the earlier coordinated-ankle pilot. Seeds 921009 and 931009 each received
24,576 continuation transitions and accepted all 288 actor updates, with no
KL rollbacks. Architecture, Gaussian exploration, PPO settings, reward, source
controller and physical limits were unchanged.

Only the starts changed. A separate frozen-source environment generated each
requested prefix without ankle correction. After the first force interval, the
full native state, pending targets, cached contacts and real controller/history
state were transferred into the chosen training world. Quiet starts used the
matching construction without force. Five push intervals remained; the original
six-second endpoint was preserved. No prefix transition entered PPO.

Training used poses 0-5 and the same allowed onset frames as the preceding pilot.
The first fit made 146 exact state transfers, including 33 quiet starts, and
simulated 9,905 additional source-prefix world intervals. The second made 142
transfers, including 37 quiet starts, and simulated 9,620 prefix intervals.
The difference between configured 25% quiet probability and these realized
counts is finite sampling, not a changed quiet allocation rule.

Generating starts on demand avoided a persistent native snapshot bank. Training
took 102.84 and 100.79 seconds, including subsequent reset reconstruction inside
the fit loop; the reported fit timer excludes its initial reset. The complete
experiment took 533.62 seconds after initial setup. Evidence occupies about
198.8 MB, including evaluation traces and source copies. Only initial/final
actors and final critics were saved.

## Frozen outcomes

Each deterministic group has 32 cases: eight poses, both signs and two timings.
Counts require survival and the final two seconds continuously satisfying the
existing posture/balance criteria with both feet loaded at least 30 N.

- 150 N X: source 7/32; both new heads 7/32. The successful case identities were
  unchanged. Reserved poses remained 2/8; signs remained +X 2/16 and -X 5/16.
- 150 N Y: source and both heads 32/32.
- 300 N X: source and both heads 0/32.
- 300 N Y: source 31/32; heads 31/32 and 30/32.
- All deterministic pushes: source 70/128; heads 70/128 and 69/128.
- Sixty-second quiet Stand: both heads 16/16, covering eight repeated poses.
- Sampled-policy core pushes: 7/32 and 6/32, versus the earlier zero-head 6/32.
- Audited corrective steps: zero across all 720 evaluation cases.

Matched-prefix continuations produced the same deterministic success identities
as continuously active execution for each candidate, not merely the same totals.
Both also retained six-second Stand in all 32/32 matched quiet cases.

The first head's unchanged aggregate score hides a trade: it rescued training
pose 4 under positive 300 N Y at the later timing, but lost reserved pose 7 under
negative 300 N Y at the earlier timing. The second head lost that same reserved
case and gained none. Neither branch supplies evidence of better core recovery.

Compared with the preceding pilot, the large broad-direction regression is
reduced, and the earlier core gains disappear. This is a useful comparison of
learning setups, not a demonstrated improvement in the dummy's recovery skill.
There is no basis here for a full perturbation training run or Arena replacement.

## Response size

Saved matched-start traces show very small learned ankle additions during the
first 500 ms after takeover:

- Seed 921009: RMS 0.001408 rad, maximum absolute 0.003242 rad.
  Difference from its paired quiet commands: RMS 0.000297 rad.
- Seed 931009: RMS 0.000364 rad, maximum absolute 0.001115 rad.
  Difference from its paired quiet commands: RMS 0.000180 rad.

The successful diagnostic pulse family used a desired 0.075 rad addition.
That comparison establishes a substantial command-size gap; it does not prove
that simply scaling the learned output would work, or establish why PPO learned
such small corrections. Both fits changed their weights and accepted all updates.

## Verification

The independent audit passed 45 traces / 720 cases, 570 input hashes and 133
output hashes. It checked unchanged initial weights, training budgets and reset
schedules, exclusion of reserved poses/times, continuation-only accounting,
history/action memory, scalar target composition, force windows, physical delay,
independent geometry and balance features, survival, settling and step events.
Maximum reconstructed balance-feature error was 2.04e-6. No solver warnings.

Mixed asynchronous reset checks preserved unselected worlds exactly. Initial
matched zero-correction runs reproduced both archived core timings and quiet
continuation exactly. Every candidate matched-prefix run preserved the source
prefix and the two additional queued physical intervals.

Evidence: `logs/mimickit-training/recovery-start-learning-20261009-01/`.
See `audit.json`, `checks.json`, fit histories/reset records/start accounting and
individual traces. `response_diagnostics.json` records case-level differences
and early command measurements with its own source/input hashes; it was produced
after completion and is separate from the original output manifest.

## Next priority

Audit the existing reward against the already saved successful ankle probes on
training poses. From the same recovery starts, compare zero correction and both
pulse directions using the exact task reward, discount, failure treatment and
six-second endpoint used by this pilot. Exclude pre-impact rewards and frozen
post-failure frames. Report per-case preference and its relation to independently
audited recovery, instead of relying on mean reward alone.

This can use the saved `recovery-starts-20261009-01` reward traces; it needs no new
physics, model training or checkpoints. Keep reserved-pose outcomes out of
decisions about changing the objective. If the reward does not prefer successful
corrections, isolate that problem before another fit. If it does, inspect
exploration coverage and action/advantage sensitivity before changing the reward
or extending compute. The present evidence does not establish either cause.

## 2026-10-09 - Saved recovery reward audit completed

See RECOVERY_REWARD_AUDIT_PROGRESS.md. Analyzed 144 saved training-pose trajectories and 96 pulse-versus-zero comparisons, with no physics integration or training. All 16 rescuing pulses scored above zero correction; all five pulses that destroyed recovery scored below it. Every rescue had positive cumulative reward benefit within 200 ms. Both frozen final critics preserved those pushed rankings after time-limit bootstrap. Quiet measured returns preferred zero in 48/48 comparisons, though final-critic bootstrap reversed four and two small comparisons (including duplicate quiet worlds). Reward reconstruction and independent return/termination verification passed; reserved poses were excluded. Retain the reward. Next: bounded frozen-policy exploration/credit checks with actual sampled commands and the same 64-interval GAE, to distinguish poor useful-action coverage from poor advantage estimates before changing noise, objective or training duration. The audit did not reconstruct historical PPO advantages. Arena unchanged. Evidence: logs/mimickit-training/reward-alignment-20261009-01/results.json and audit.json.

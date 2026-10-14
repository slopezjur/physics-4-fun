# Frozen ankle response comparison - completed 2026-10-09

Decision: the learned ankle response contributes one additional settled recovery
across 64 matched cases. Most successful outcomes persist when it is replaced
by the policy's quiet command sequence. The previous candidates remain rejected.

Protocol: [ANKLE_RESPONSE_CAUSAL_PLAN.md](ANKLE_RESPONSE_CAUSAL_PLAN.md).

## Results

- Seed 921009: normal learned continuation 11/32 settled; quiet replay 10/32.
  Survival was 12/32 versus 11/32. The only changed outcome was reserved pose 6,
  positive X, onset near 1.0 seconds: normal recovered, replay failed.
  Reserved-pose settled recovery was 2/8 versus 1/8.
- Seed 931009: 9/32 settled and survived in both branches, with identical case
  identities. Reserved-pose recovery remained 0/8.
- Combined: 20/64 normal versus 19/64 replay. There were no replay-only successes.
  All 63 other case outcomes matched, including every training-pose outcome.
- No audited corrective steps in the 64 new cases.

Recovery requires survival and the final two seconds continuously satisfying
the existing balance/posture criteria with both feet loaded at least 30 N.
These are six-second direct-force trials, not projectile certification.

## What the intervention establishes

The learned actor ran normally through the first force interval. At frames 61
or 103, its scalar latent was replaced by the matching frame from its own quiet
rollout. This preserved the altered stance, existing commands and controller
memory before intervention. The source controller kept responding to live state;
only the additional ankle head lost its impact-dependent response.

Normal and replay branches matched every saved field exactly before intervention.
Physical states and active targets also matched through the next two queued
intervals. Replayed commands passed through the same tanh, slew, bounds and delay.
Quiet additions matched while worlds were active. The intervention actually
changed commands: post-switch ankle-addition differences against normal execution
had RMS 0.00085-0.00114 rad and maximum 0.00274-0.00842 rad across the four groups.

This supports a narrow causal conclusion: nearly all observed settled successes
do not require the ankle head's post-impact response in these cases. One reserved
case does benefit from it, so the response is not universally irrelevant.

It does not isolate pre-impact stance from the continuing quiet ankle commands:
both remain in the replay branch. Nor does it show recovery without feedback,
because the original source controller still reacts. We should describe the
earlier score gains as mostly compatible with a changed stance and quiet command
trajectory, not as established general reactive recovery or learned stepping.

## Verification and cost

The independent saved-trace audit passed four traces / 64 cases, 564 input hashes
and 11 output hashes. It verified exact prefixes, physical delay, replayed latent
commands, history and action memory, ankle-only corrections, target composition,
force windows, independent foot geometry, survival, settling and step events.
No solver warnings occurred. NPZ arrays were cached once during verification.

The four physical rollouts took 19.20 seconds after setup. Evidence occupies
about 17.6 MB including the source snapshot. No training transitions or new model
checkpoints were generated. Previous sealed Python sources and Arena selection
were preserved.

Evidence: `logs/mimickit-training/ankle-response-causal-20261009-01/`.
`audit.json` includes every paired case, the one changed outcome and command
differences. `summary.json`, individual reports/traces, `inputs.json` and
`completed.json` preserve the experimental contract and provenance.

## Next priority

Correct the learning setup so it must learn to handle a disturbance from a
shared Stand state, instead of improving its score by changing its stance before
the hit. First validate a training harness that starts recovery episodes from
the frozen source's physical states, with the native pending-command queue and
controller history preserved. Pair quiet and pushed starts, and check that zero
correction exactly reproduces the source continuations.

Compare the existing bounded ankle probes from these states as a reachability
check before a new learning run. Then use a bounded two-seed comparison with
post-disturbance training credit and quiet-retention pressure. Force sign,
timing and replay commands must not enter learned execution. A test of existing
heads from the new starts alone would be a distribution-shift diagnostic, not
proof that the representation cannot learn.

This is the next proposed experiment, not work already completed. Keep the
reserved poses and timing checks, broader-direction retention, and independent
step audit. Do not extend the rejected heads or increase physical limits based
on the current core score. Full perturbation training remains unqualified.

## 2026-10-09 - Matched recovery starts admitted

See RECOVERY_START_PROGRESS.md. Full native snapshots now preserve physical/solver state, cached contacts, task state, delayed commands and balance/action history, including isolated restores under an odd shared queue-cursor shift. Independent audit passed 12 rollouts/192 cases; eight traces exactly reproduced archived zero and pulse continuations. At signed 150 N X, zero correction recovered 7/32, negative ankle pulse 12/32 and positive pulse 15/32. Each pulse favors one sign and harms the other. Each command retained six-second quiet Stand in 32/32 cases; no corrective steps. These validate the start mechanism and reproduce reachability, not new learned recovery. No training or Arena update. Next: two fresh bounded recovery-start learners, 24,576 continuation transitions each, paired quiet starts and unchanged reward; evaluate both matched starts and continuously active execution, reserved poses/timings, broader directions/strengths and sixty-second quiet Stand. Reject any improvement dependent on a privileged takeover gate. Evidence: logs/mimickit-training/recovery-starts-20261009-01/audit.json.

# Weighted actor quiet-command replay - completed 2026-10-09

Decision: the two extra core recoveries do not require this ankle head's
impact-dependent response in the tested intervention. The weighted candidates
remain rejected for promotion. Arena is unchanged; no training was performed.

Protocol: [RECOVERY_WEIGHTED_QUIET_REPLAY_PLAN.md](RECOVERY_WEIGHTED_QUIET_REPLAY_PLAN.md).

## Results

Frozen weighted seed 931009 was tested at the two existing signed 150 N X
timings. Its live prefix continued through the first force interval, then its
ankle latent was replaced with the corresponding saved quiet-trial sequence.

- Normal execution and replay both recovered 9/32: 4/16 at the first timing,
  5/16 at the second. All 32 individual survival and settled outcomes matched.
- The two extra successes over zero correction persisted: +X poses 3 and 4
  at approximately 1.7 seconds. Both are training poses.
- Reserved-pose recovery remained 2/8. Matched-start execution and zero
  correction each scored 7/32, also with 2/8 reserved-pose recoveries.
- The additional six-second quiet control matched every saved trace field
  exactly and retained whole-episode Stand in 16/16 cases.
- No corrective steps in the 32 pushed or 16 quiet-control cases.

The intervention changed ankle commands. Post-switch addition differences from
normal execution had RMS 0.000890/0.000893 rad and maxima 0.004750/0.004583 rad.
Equal recovery outcomes therefore did not result from replaying identical
post-impact corrections.

Recovery still requires survival and the final two seconds meeting the existing
posture/balance and two-foot loading criteria. These are six-second direct-force
trials, not evidence of recovery from arbitrary projectiles or repeated hits.

## What this establishes

The extra successes survive removal of the ankle head's impact-dependent
continuation. They cannot be used as evidence that this learned response was
necessary. Preparation, the continuing quiet ankle command trajectory and the
frozen Stand controller's live feedback remain together in the replay branch.
This comparison does not separate those remaining contributions.

It also does not show recovery without feedback: the source Stand controller
continued to respond to the actual pushed body. Independently reconstructed
source targets agreed within 2.98e-8 rad across 13,258 active world-intervals
from frame 2 onward. The matched-start and zero comparisons have different
prefixes and are descriptive references, not the causal intervention control.

## Verification and cost

The independent audit passed three traces / 48 cases: 32 diagnostic push cases
and 16 quiet-parity controls. Every saved prefix field matched normal execution
for the first 61/103 intervals; physical state and active targets remained exact
for 63/105 intervals because of the two queued control intervals. Quiet latent suffixes,
active ankle additions, memory/history, limits, force timing, native foot
geometry, source targets and individual outcomes were verified.

The audit checked 606 input and 12 sealed output hashes. A supplemental lineage
check independently confirmed all 588 reused historical inputs against the
candidate training manifest, including foundation/controller/helper files.
Both policy state dictionaries remained unchanged; no checkpoints were created.

The first verifier used an overly strict 1e-10-second tolerance for reported
hit-time metadata computed through float32 tensors. Observed errors were
1.06e-8 and 5.37e-8 seconds. A successor uses the existing causal verifier's
1e-6-second metadata tolerance; exact force-frame/trace checks remain unchanged.
It also verifies the supplemental lineage manifest and audit harness hashes.
The original source is preserved. No physical trial was repeated for this repair.

Physical rollouts took 15.53 seconds after setup. Evidence occupies about
13.2 MiB. There were zero optimizer updates and zero new learned policies.

Evidence: `logs/mimickit-training/recovery-weighted-quiet-replay-20261009-01/`.
See audit.json, paired-outcomes.json, historical-input-audit.json,
audit-harness.json, completed.json and the three physical traces/reports.

## Next priority: paired first-action credit at identical recovery states

Run a bounded frozen diagnostic using both weighted actors and their own final
critics, training onsets 44/74, poses 0-5, quiet/+X/-X branches and four fixed
future-noise seeds. From each identical source-derived post-impact state,
compare the first conditional action plus HALF_SIGMA against the same action
minus HALF_SIGMA. Thereafter use each frozen policy with identical future
standardized noise draws in the paired worlds. Preserve native state, delayed
commands, history and action memory; do not reset at later decision points.

This is 2 actors x 2 timings x 6 poses x 3 branches x 2 first-action choices x
4 future-noise seeds = 576 continuations, or 288 paired comparisons. Count each
quiet pose once per condition; do not duplicate it as two signed worlds. Do
not use reserved poses or evaluation timings to select actions or settings.

Record compact per-transition rewards, terminal/failure masks, values, means,
innovations and actual corrections. Compare the first-state 64-interval GAE
ranking with measured full-episode discounted returns and physical outcomes.
Separate reward and value terms; also report the 64-interval discounted return
with its endpoint bootstrap, so horizon/lambda effects are visible.

The purpose is to check local action credit, which earlier whole-pulse and
whole-trajectory reward comparisons do not establish. Persistent ranking
disagreement calls for a targeted credit investigation; correct useful local
rankings shift attention to how PPO converts them into conditional updates.
GAE and full-return disagreement alone does not prove critic error. Keep reward,
noise amplitude, weights, physical limits and Arena fixed; no fitting or new
policy checkpoints in this proposed diagnostic.


## Successor completed

The paired local-action diagnostic is complete. All 192 pushed pairs and all
40 changed pushed recovery outcomes had useful GAE rankings. See
[RECOVERY_LOCAL_ACTION_CREDIT_PROGRESS.md](RECOVERY_LOCAL_ACTION_CREDIT_PROGRESS.md)
for the 576-case audit, quiet discrepancies and the next controlled offline
actor-update comparison.

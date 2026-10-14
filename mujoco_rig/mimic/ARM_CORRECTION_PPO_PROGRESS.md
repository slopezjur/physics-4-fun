# Feasible-correction PPO pilot - 2026-10-08

## Decision

Both correction-policy candidates regressed from the retained reference's 4/8 supported holds and 4/8 returns to 3/8 holds and 3/8 returns. Complete arms-clear skill success remains 0/8. Reject both candidates for promotion and retain the existing Arena brain.

The verified action path is usable, but this pilot did not demonstrate useful learning through it. These are two short, matched-budget runs; they do not establish that correction-policy PPO cannot work with a different justified learning setup or budget.

## What ran

The f832bab8 Stand anchor and queue-masked seed-451028 proposal learner stayed frozen. A separate correction policy received 488 inputs: the full normalized 421-value state/goal input, previous addition, nominal slew-limited proposal and episode frame. Two 128-unit ELU hidden layers produce 33 correction means. The final actor layer started at exact zero; its shared initial seed was 521027. Fresh optimizers and sampling seeds were 521028 and 531028.

Each run collected 24,576 transitions across 12 worlds containing two instances of training poses 0-5, with 32 collections of 64 frames. Entire poses 6 and 7 were excluded from actor and critic losses. Actor rate was 3e-6, critic rate 3e-4, Gaussian std 0.005 rad, three epochs and minibatch 256. PPO stored unprojected correction samples and their Gaussian log probabilities. Quiet frames were excluded from actor loss. The critic used the same 488 inputs with its own parameters.

The physical adapter applied corrections around the nominal slew-limited proposal, then enforced the same +/-0.02-rad addition change, +/-1-rad addition envelope, joint limits, two-command queue and quiet bypass. Passive springs remained zero. Broad motor/COM training reward scales were 0.10 rad/0.04 m; final physical admission remained 0.05-rad motor RMSE, 0.02-m COM error and 0.02-m foot drift, with supported hold and return to Stand.

This introduces a new trainable correction network over a frozen proposal. Earlier raw-policy PPO runs are useful context, but are not a matched architecture comparison proving that action placement alone caused the difference.

## Baseline and retention checks

Before learning, zero correction reproduced the retained nominal qpos, issued commands and alive masks exactly across all 540 frames. It reproduced the expected 4/8 holds, 4/8 returns and 0/8 full skills. All sixteen initial 30-second quiet cases passed, with qpos/commands identical to the original Stand anchor trace.

Both final policies also passed sixteen 30-second quiet cases each, again with exact anchor qpos/commands through the quiet bypass. Thus 48 quiet cases passed across the initial and two final evaluations. The safeguard preserves quiet Stand; it does not certify active-arm control.

## Frozen physical outcomes

- Seed 521028: supported poses 1, 5 and 7; returned poses 0, 1 and 5. Counts: 3/8 holds, 3/8 returns, 0/8 complete skills.
- Seed 531028: supported poses 0, 1 and 6; returned poses 0, 1 and 7. Counts: 3/8 holds, 3/8 returns, 0/8 complete skills.

Starting supported and returned poses were 0, 1, 5 and 6. Both candidates therefore traded individual successes and lost aggregate retention. Neither held-out pose achieved the complete skill. Every pose in both final evaluations exceeded both the motor and COM hold-error gates. The lowest peak motor RMSE across each seed's eight poses was 0.08908 and 0.08083 rad; the lowest peak COM error was 0.03334 and 0.03177 m.

Neither candidate qualified for changed-timing evaluation. No admission threshold was relaxed, and no candidate was exported or loaded into Arena.

## Learning and command diagnostics

Seed 521028 performed 214 actor gradient updates; seed 531028 performed 224. Both stopped actor updates early in 27 of 32 cycles. Maximum observed whole-batch post-update analytic KL was 0.01425 and 0.01299. This remains an observation on collected states, not a universal hard bound.

Approximately 81.40% and 80.90% of eligible sampled correction coordinates changed the rate-limited addition by more than 1e-7 rad compared with the nominal proposal on the same state. This measures transmission through the correction projection; final source/joint clipping can still affect the physical target. The prior one-joint intervention study separately verified joint response. The statistic does not mean 81% of samples were helpful.

Mean weighted hold tracking reward was 0.1326 and 0.1370. Only 1.01% and 0.88% of hold samples had tracking reward below 1% of its maximum. Thus near-zero tracking rewards and universally erased corrections are not sufficient explanations for this pilot's failure. No training episode earned the full-skill bonus; physical-failure counts were 18 and 22 over the fixed transition budget.

During alive active nominal evaluation frames, learned raw correction means had RMS magnitudes 0.001271 and 0.000965 rad, with maximum absolute values 0.004976 and 0.005025 rad. This shows that the new head changed, while the resulting deterministic trajectories still failed the skill.

## Next priority: stochastic versus deterministic correction behavior

Before another training sweep, evaluate whether projection creates a material mismatch between noisy training and mean-policy execution. At a slew boundary, symmetric Gaussian corrections can have an asymmetric executed distribution because outward samples are clipped while inward samples pass. Consequently, projecting the Gaussian mean is not generally equal to the average projected command. This is a mechanism to measure, not an established cause of the observed regression.

Use the frozen zero-correction reference and both final actors. Compare deterministic evaluations with a small predeclared set of stochastic evaluations on the same eight nominal poses, retaining whole-skill gates and physical bounds. Measure raw means, average executed corrections and resulting physical outcomes. Include the untrained zero-mean head so noise-only behavior is not mistaken for learned improvement. Keep optimizer state frozen and do not increase training duration in this diagnostic.

If noisy execution and mean execution differ materially, resolve policy/execution semantics before further fitting. If both fail similarly, use the same fixed policies and critics for a matched action/advantage-direction check rather than guessing another reward or learning-rate setting. The two correction policies from this pilot remain rejected while that diagnosis proceeds.

## Accounting and evidence

49,152 training transitions and 438 actor gradient updates, approximately 71.24 seconds of learning. New physical evaluations: eight initial nominal cases, sixteen initial quiet cases, sixteen final nominal cases and thirty-two final quiet cases, totaling 72. No varied-timing evaluation, new teacher data, long training run, Arena change, export or C# compilation. Run evidence occupies 30,860,397 bytes, about 30.86 MB, plus the review files.

- Frozen plan: `ARM_CORRECTION_PPO_PLAN.md`.
- Actor, adapter, PPO loop and deterministic evaluation: `train_arm_correction_ppo.py`.
- Sealed run: `logs/mimickit-training/arm-correction-ppo-20261008-01`.
- Independent physical/projection/provenance audit: `logs/mimickit-training/arm-correction-ppo-review-20261008-01/report.json`.
- Verifier: `verify_arm_correction_ppo.py`.

Audit passed sealed source/input/output hashes, identical initial correction weights, zero initial output layer, finite final weights, explicit holdout indices, transition/update counts, reward reconciliation, reconstructed final correction projection, slew/envelope/joint limits, quiet identity and independently recomputed physical gates. The in-memory proposal weights were checked unchanged at the end. Raw training trajectories were not retained for independent replay.

Arena remains on anchor f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d. The correction network is an experimental checkpoint and is not installed in Arena.


## 2026-10-08 - Thirty-step execution and critic investigation completed

See ARM_RECOVERY_30_PROGRESS.md. All 30 research steps completed and audited. Frozen noise, timing, projection and matched action/critic comparisons produced no complete arms-clear skill. A real projection bias was measured. Critic warm-up passed fresh-episode validation and improved partial holds over two matched controls, but all four bounded PPO candidates remained at zero complete skills. Four fits: 98,304 transitions, 913 actor updates; all 64 quiet cases exactly retained Stand. Dominant left-arm error was already present in issued commands (roughly 0.51-0.57 rad RMS for warm candidates). Next: structural proposal/target-representation ablation against direct goal tracking and the verified physical teacher, before more PPO. Arena unchanged. Evidence: logs/mimickit-training/arm-recovery-30-20261008-01/final-audit.json. One aborted harness attempt is preserved and excluded from successful fit totals.

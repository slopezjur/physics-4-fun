# Matched upper-arm response diagnostic - 2026-10-08

## Finding

The current raw-action exploration can be completely erased by the command slew limiter. At the sampled hold and return states, both +0.005 and -0.005 rad one-frame interventions on UpperArm_L_ry produced exactly the baseline issued commands and complete qpos trajectories in all eight poses throughout the following second. These are 32 unchanged intervention cases across two phases and two signs.

This is direct evidence of locally ineffective exploration at those states. It does not prove that all actions, all joints or all training states are blocked. No training occurred and complete learned arms-clear admission remains 0/8 from the preceding study.

## Matched-state checks

The subject was the retained queue-masked seed-451028 auxiliary actor with the same f832bab8 frozen Stand anchor. Fifteen branches covered ramp frame 90, hold frame 210 and return frame 360; each phase used an unchanged continuation plus positive/negative 0.005-rad interventions lasting one or twelve frames. Every branch continued the normal closed-loop actor for 60 frames, approximately one second.

Each branch replayed the deterministic prefix from reset. The complete MuJoCo integration state including warm start, cached contact forces, queued/applied commands, torque, reference, task step/offset/action/COM memory, adapter slew memory, observations and success-gate history matched exactly at the intervention point. No live pose was overwritten to create a branch. All unchanged prefixes and baseline continuations matched the archived physical qpos/issued traces bit for bit.

The native engine's existing substep recorder captured all four physics updates per control frame. The independent audit exactly reconstructed recorded torques from applied targets, joint state, velocity and the existing PD law: maximum reconstruction difference was zero. Command composition and the two-slot delay passed exact checks; configured torque bounds were respected.

## Command transmission

At the ramp intervention, authority was approximately 0.500225. The raw 0.005-rad change therefore changed the first issued target by approximately 0.002501 rad. Both signs reached the joint in every pose. The applied command changed two control frames later, at 33.336 ms; the first changed post-step qpos sample was at 50.004 ms. This is the expected physical delay and sampling schedule.

At the hold and return intervention instants, the requested addition was already more than 0.02 rad away from the previous addition in every pose. The adapter clips each change to +/-0.02 rad. A +/-0.005-rad raw perturbation remained on the same clipped side, producing the same issued command. With no state change, subsequent deterministic policy outputs also stayed identical. The zero response was not a missing motor or an unexplained simulator delay.

Across the baseline one-second continuations, the left-upper-arm slew limiter was active in 60.0% of live ramp samples, 85.625% of hold samples and 80.324% of return samples. The preceding +/-1-rad addition clamp was active in only 2.917% of hold samples and zero ramp/return samples. This motor reached its torque limit in zero recorded live physics substeps. These fractions describe the frozen reference continuations, not measured PPO training visitation.

## Sustained interventions and feedback

Keeping the same 0.005-rad bias for twelve frames (0.200016 seconds) gave it opportunities to reach the command when the slew limiter stopped masking it. During hold, six of eight poses responded for each sign; poses 0 and 2 remained exactly unaffected. First applied-target changes occurred at offsets 3-11 frames, depending on pose and sign. During return, all eight responded, with onsets at offsets 3-12 frames. These longer onset times combine limiter masking with the unchanged two-frame physical queue.

For the hold-phase positive sustained bias, the broad same-state discounted reward improved in all six responsive poses. The negative bias improved it in only one responsive pose. Narrow reward improved in five and two poses respectively. This demonstrates a measurable directional signal in this local sample; it is not a trained recovery skill or a globally preferable action direction.

Response was less consistent during ramp and return. Small ramp pulses eventually produced left-arm deviations up to 0.1553 rad (positive) or 0.1445 rad (negative) from the unchanged trajectory within one second, despite a first issued difference near 0.0025 rad. These are closed-loop trajectory differences, not direct actuator gain estimates. They show that small early changes can be amplified by subsequent policy/physics feedback.

All ramp and hold branches remained physically alive through their one-second continuation. In every return branch, including the unchanged baseline, pose 7 failed; the other seven survived. Paired reward comparisons use common alive, non-failure frames and omit unmatched tails. They therefore measure local signal rather than complete episode return or the value of a fall.

## Temporal credit interpretation

Existing PPO uses 64-frame collections (1.066752 seconds), gamma 0.995 and GAE lambda 0.98. The reward-only gamma-lambda weight is approximately 0.951 after the two-frame delay, 0.739 after twelve frames and 0.220 after one second; its half-life is 0.458 seconds.

The direct observed actuator response fits inside that collection length and retains substantial short-lag weight. These measurements do not identify the two-frame delay alone as the learning blocker. Later closed-loop consequences receive less reward-only weight, but actual GAE also includes critic predictions, bootstrapping and terminal handling. This diagnostic did not estimate actual critic-based advantages, so changing lambda or rollout length is not justified by these numbers alone.

Both narrow and broad rewards were compared on each same physical trajectory without influencing actions. Broad values were recovered from the fourth root of the saved float32 narrow kernels. Sixty common-alive component entries rounded to zero in the narrow kernels; the upper bound on omitted broad reward from zero rounding is 4.37e-12 per frame, or 2.62e-10 across sixty frames. Ordinary float32 roundoff still applies; these reward reconstructions are not asserted bitwise equal to a separate broad-kernel computation.

## Next bounded correction check

Preserve the 0.02-rad-per-frame slew limit and every existing joint/torque bound. In a diagnostic successor, place the small correction around the nominal command after its first slew limiting, then project the corrected command back into the same feasible interval around the previous issued addition. At zero correction it must reproduce the unchanged reference exactly. A boundary may still block outward corrections, but an inward correction should remain observable instead of both signs being erased.

Compare raw-output and feasible-command interventions at the same ramp/hold/return states, without training or promoting a policy. Verify exact zero-correction parity, limits, queue timing and directional reward response. This is a proposed change to action placement, not a request to raise physical speed or add springs. Only after that check should a bounded learned correction policy be considered; its action/log-probability definition must match the executed exploration.

## Accounting and evidence

Fifteen batches with eight candidate worlds each: 120 one-second physical continuations, including 24 unchanged controls and 96 interventions. All worlds were alive at the intervention points. Deterministic prefix replays add simulation work but are not additional independent skill evaluations. No optimizer update, training checkpoint, new whole-skill admission, Arena modification, export or C# build occurred. Run evidence occupies 23,773,799 bytes (about 23.77 MB), plus small review files.

- Frozen plan: `ARM_ACTION_RESPONSE_PLAN.md`.
- Collector and physical trace assertions: `probe_arm_action_response.py`.
- Saved common states, substep traces and provenance: `logs/mimickit-training/arm-action-response-20261008-01`.
- Independent audit and per-pose paired responses: `logs/mimickit-training/arm-action-response-review-20261008-01/report.json`.
- Limiter/credit summaries: `logs/mimickit-training/arm-action-response-review-20261008-01/mechanism.json`.
- Verifier: `verify_arm_action_response.py`.

The audit confirmed sealed source/input/output hashes, exact branch-state fingerprints, expected queue behavior, command composition, injected-action identity, torque reconstruction and reward reconciliation. Arena remains on anchor f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d.


## 2026-10-08 - Feasible-command correction diagnostic passed

See ARM_FEASIBLE_RESPONSE_PROGRESS.md. Moving the correction after nominal slew limiting and reapplying the same bounds transmitted inward +/-0.005-rad pulses in all eight hold and all eight return poses. Outward pulses stayed blocked. Zero correction exactly reproduced the prior state/trace; slew, joint, torque and delay checks passed. This verifies an action path, not learned recovery. No training or Arena change. Next: two bounded runs of a zero-output correction policy over the frozen proposal learner, using raw pre-projection PPO likelihoods and unchanged skill gates. Audit: logs/mimickit-training/arm-feasible-response-review-20261008-01/report.json.

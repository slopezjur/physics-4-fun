# Direct leg-response screen - completed 2026-10-09

Outcome: the current plant and correction limits permit useful recovery from
many failed 150 N X pushes. The next learning comparison should target coordinated,
state-dependent ankle recovery. This screen produced no trained policy or
corrective step. Arena selection remains unchanged.

Protocol: [DIRECT_LEG_RESPONSE_PLAN.md](DIRECT_LEG_RESPONSE_PLAN.md).

## Direction and timing

The crossed frozen-source baseline recovered 70/128 cases:

- X pushes: 7/64, across both signs, strengths and timings.
- Y pushes: 63/64 under the same crossed conditions.
- At 150 N X, results were 4/16 for onset 1.0 seconds and 3/16 for 1.7 seconds.
- At 300 N X, both timings produced 0/16 recovery.

The direction difference persists after crossing timing. All 64 cases matching
the previous pilot reproduced exactly.

## Bounded leg interventions

Six predefined combinations, four signed amplitudes and both timings produced
768 diagnostic cases. Selection used only poses 0-5. The selected combination was
bilateral ankle pitch +0.075 rad, requested for 12 control intervals (0.200016 s)
starting one interval after force onset. Existing slew, joint and torque limits
and the two-interval delay were preserved.

At 150 N X, that profile recovered 15/32 cases versus 7/32 for the source:

- It rescued 10 previously failing cases and lost 2 previous successes.
- Selection poses: 5/24 to 11/24 recovery.
- Reserved poses: 2/8 to 4/8 recovery; these outcomes did not affect selection.
- With +0.075 rad, +X recovery was 0/16 and -X recovery was 15/16.
- The opposite ankle pulse, -0.075 rad, produced +X 12/16 and -X 0/16.

The correct response must depend on state. A universal fixed pulse is unsuitable.
Across the entire predefined probe library, 20 of the 25 baseline failures had
at least one rescuing intervention. The union including baseline successes is
27/32. This is a posthoc reachability result using different interventions per
case, not the performance of one controller or a heldout validation score.

None of the tested interventions produced an audited corrective placement.
There were also no qualifying unloaded, clear-foot frames in the 768 probe cases.
These rescues demonstrate ankle balance recovery, not stepping or human naturalness.

## Does the source fight the correction?

For the selected profile, the source's paired nominal ankle-target response had
an integrated projection of approximately +0.19 onto the added correction at
both timings. It did not predominantly cancel that correction.

Replaying the unmodified baseline source actions on all 16 leg motors during
the pulse/return/delay window produced the same 15/32 settled cases, with exactly
the same success/failure identities. Zero-addition replay reproduced every saved
baseline field exactly. This short feedback intervention found no recovery benefit
from suppressing the source's leg response; it does not settle other timings,
longer interventions or stepping strategies.

The selected pulse retained all 16 quiet cases throughout a six-second rollout.
It recovered 0/32 stronger 300 N X cases. Neither projectile nor repeated-hit
admission has been established.

## Implications for the next learner

The previous 33-output PPO pilot produced small corrections and failed to exploit
these available rescues. A +0.075-rad request corresponds to a latent value of
atanh(0.75), or 9.73 initial standard deviations under that pilot's Gaussian.
This does not prove the cause of failure, but motivates testing coordinated,
larger, temporally sustained exploration within the existing physical bounds.

Do not infer the needed response from instantaneous pelvis velocity alone. After
the first hit interval, its X sign agreed with the prescribed force in only
20/32 cases; interval-average whole-body COM velocity agreed in 27/32. Neither
is a validated direction classifier. Pre-existing motion, support geometry and
recent motion history matter; the target remains recovering balance, not merely
classifying a hit.

Next comparison: a small learned correction with a shared bilateral ankle-pitch
action, live balance/momentum/support information and short measured-state history.
The learner must choose its own amplitude, sign and duration. Include quiet
episodes, both push signs, randomized timing, reserved poses and a matched frozen
source. Check observation/control consistency, then use two fresh bounded fits
before considering longer training. No hit clock, prescribed force label, stored
action replay or fixed recovery pulse belongs in that learned controller.

## Evidence and verification

Completed 63 rollouts / 1,008 six-second physical cases in 270.47 seconds, with
zero optimizer updates. Evidence occupies approximately 203 MB; no model
checkpoints were produced.

Independent audit passed: 553 input hashes, 194 output hashes, force windows,
correction formulas, exact intervention prefixes and delay, target limits,
independently reconstructed foot geometry, recovery/step outcomes, zero-replay
parity and selection without reserved-pose outcomes. No evaluation solver warnings.

Evidence root: `logs/mimickit-training/direct-leg-response-20261009-01/`.
Core results: `crossed_matrix.json`, `candidates.json`, `selected.json`,
`completed.json`, `audit.json`, and per-case reports/traces/descriptors.
Later saved-trace analyses `analysis.json` and `direction_observation.json` carry
their own input hashes and are outside the completion manifest.

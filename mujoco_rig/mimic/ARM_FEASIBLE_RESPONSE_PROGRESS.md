# Feasible-command correction diagnostic - 2026-10-08

## Decision

The proposed command placement passed its bounded transmission check. At the same hold and return states where raw-output pulses were erased in both directions, a 0.005-rad inward correction now reached the joint in every pose. Outward corrections remained blocked at the existing boundary. No physical limit was increased.

This admits a bounded correction-policy learning pilot. It does not establish learned balance or recovery: no optimizer ran here, and full learned arms-clear admission remains 0/8 from the preceding training studies. Arena is unchanged.

## What changed

The original auxiliary learner still proposes its raw target. The adapter computes the original nominal addition and limits its change to +/-0.02 rad per control frame. The experimental correction is added at that point, scaled by the original authority and goal-displacement fade, then projected back into the same slew interval and +/-1-rad addition envelope. The original final joint-target clipping and quiet bypass remain.

Projection is applied only to corrected coordinates. Zero correction retains the original arithmetic. The slew limit constrains auxiliary addition; it is not a new claim that the frozen Stand actor's entire composed target changes by at most 0.02 rad.

## Verification

The experiment repeated the earlier protocol: ramp frame 90, hold frame 210 and return frame 360, eight poses each, unchanged continuation plus +/-0.005-rad corrections lasting one or twelve frames. Every continuation lasted sixty control frames, approximately one second, with the actor responding normally afterward.

All branch-state fingerprints exactly matched the earlier raw-action experiment, including MuJoCo integration/warm-start state, contact cache, command queues, controller memory, task state and adapter memory. The new correction buffer was verified zero before each branch. Prefix qpos/commands and zero-correction continuations matched the archived reference exactly. The verifier also matched every common baseline trace array except broad reward, which this run computes directly rather than reconstructing through roots of float32 kernels.

The audit independently reconstructed the new projection, checked the actual addition slew on alive non-quiet frames, +/-1-rad envelope, joint bounds and original two-command queue, and exactly reconstructed substep torques. Maximum PD reconstruction error was zero. Quiet and terminated-world handling retain the original bypass behavior. No speed, torque, stiffness, spring or delay setting changed.

## Transmission result

At hold, the positive pulse was inward in poses 0, 3 and 4 and transmitted in all three. The negative pulse was inward in poses 1, 2, 5, 6 and 7 and transmitted in all five. At return, positive was inward in poses 0, 2, 3, 4 and 6; negative was inward in poses 1, 5 and 7. Every inward correction changed the issued target by approximately 0.005 rad, with less than 1e-6-rad rounding error.

Across those two phases, all sixteen inward cases changed the applied target at frame offset 2 and the first changed post-step joint sample appeared at the same offset. This means target application after 33.336 ms and a changed sampled joint position at 50.004 ms. All sixteen outward one-frame cases retained exactly baseline qpos and issued commands throughout the one-second continuation. Previously all thirty-two raw-output cases were unchanged.

At ramp, authority was about 0.500225, so both directions transmitted roughly 0.002501 rad in all eight poses. The expected two-frame queue was preserved. This stage had already transmitted small raw-output changes in the preceding diagnostic.

## Physical feedback remains a learning problem

Of the eight inward one-frame hold corrections, seven improved the measured narrow and broad discounted reward over the common alive interval. At return, only three of eight improved either reward. The permitted direction therefore is not automatically the useful direction for every state or phase.

Twelve-frame positive corrections improved broad local reward in seven of eight hold cases and all eight return cases. The opposite direction improved it in three hold cases and four return cases. These local counterfactuals are not a full-skill success rate, and positive reward change does not guarantee final pose accuracy.

All ramp and hold continuations remained physically alive. Pose 7 failed during every return continuation, including the zero control, while the other seven survived. Reward comparisons omit unmatched/failure frames and do not represent full episode return. Small changes can still produce larger subsequent closed-loop trajectory differences; the new action placement does not remove that sensitivity.

## Next bounded learning pilot

Keep the Stand anchor and retained auxiliary proposal learner frozen. Add a zero-output-initialized correction policy whose sampled actions are corrections at the verified feasible-command placement. Start each independent sampling seed from the same initial correction weights and fresh optimizer state. Mean zero must first reproduce the unchanged nominal trace and pass quiet retention.

Expose the actual previous addition and nominal feasible proposal so the correction policy can distinguish inward and outward directions. Use a Gaussian in correction units and save the raw pre-projection sample/log probability for PPO; do not substitute clipped commands into the Gaussian likelihood. Projection belongs to physical execution. Preserve quiet bypass, all bounds, fixed whole-pose holdouts and the original complete-skill admission gates.

Use two bounded 24,576-transition runs with the existing broader training reward and small-update diagnostics. No simultaneous reward, spring, torque, delay or long-duration sweep. Evaluate final deterministic policies on all eight nominal poses and sixteen 30-second quiet cases per run. Only actual complete-skill improvement without losing the retained support/return floor warrants changed-timing evaluation or any later promotion. This diagnostic itself did not start that learning pilot.

## Accounting and evidence

Fifteen batches: 24 unchanged controls and 96 interventions, totaling 120 one-second physical continuations, plus deterministic prefix replay. No training updates, policy checkpoint creation, Arena changes, export or C# compilation. Run evidence occupies 24,656,472 bytes (about 24.66 MB), plus small review files.

- Frozen plan: `ARM_FEASIBLE_RESPONSE_PLAN.md`.
- Adapter and collector: `probe_arm_feasible_response.py`.
- Run: `logs/mimickit-training/arm-feasible-response-20261008-01`.
- Independent projection/physics audit: `logs/mimickit-training/arm-feasible-response-review-20261008-01/report.json`.
- Explicit inward-transmission/outward-blocking admission: `logs/mimickit-training/arm-feasible-response-review-20261008-01/admission.json`.
- Verifier: `verify_arm_feasible_response.py`.

Sealed input/source/output hashes passed. Arena remains on anchor f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d. The verified correction path is experimental and is not installed in the active Arena controller.


## 2026-10-08 - Feasible-correction PPO pilot completed

See ARM_CORRECTION_PPO_PROGRESS.md. Two fresh correction-policy runs over the frozen proposal, 24,576 transitions each, both regressed from 4/8 holds and returns to 3/8 and 3/8; complete skills remain 0/8. About 81% of eligible sampled coordinates changed the rate-limited addition, so transmission alone did not yield useful learning. Initial/final quiet retention passed all 48 cases with exact anchor traces. Both candidates rejected for promotion; Arena unchanged. Next: frozen stochastic-versus-mean evaluation including the untrained zero-mean head to test projection-induced execution mismatch. Audit: logs/mimickit-training/arm-correction-ppo-review-20261008-01/report.json.

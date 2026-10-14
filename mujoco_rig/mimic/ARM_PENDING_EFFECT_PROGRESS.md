# Pending-command effect comparison completed

2026-10-08. Protocol: ARM_PENDING_EFFECT_PLAN.md.
Evidence: `logs/mimickit-training/arm-pending-effect-20261008-01`.

## Decision

Reject this pending-effect head. Retain the previous masked research references.
The new input contract passed physical-controller checks, but learned arm
hold/return regressed in both seeds. Neither recipe passes universal skill
admission. No Arena policy was replaced and no learned recovery step was added.

## What changed

The task head now has an experimental input option exposing bounded normalized
PD effort and validity from both pending commands, oldest first. Motor effort
uses the existing gain 4 and damping coefficient 0.08. Joint velocity is decoded
from the observation's one-tenth scaling before subtraction. Each queue slot's
33 remaining columns are zero. All other normalized inputs are unchanged.

This is present-state effort, not a future body prediction. Both slots use the
current position/velocity; the body will evolve before the second executes.
The network remains 421-128-128-33. Active feature count changes from 287 to 355;
parameter count and initialization remain identical. No springs, gains, limits,
rewards, reference poses or command projections changed.

## Feature admission

- Both chronological queue slots matched the actual native PD control method
  at all 24 previously audited switch states, including 96 alternatives.
  Maximum normalized-effort difference: 2.98023224e-8.
- All 96 alternatives changed the new input. The original teacher dataset had
  zero saturated valid motor-effort features; clipping did not erase the tested
  queue distinctions. This does not establish coverage of every future state.
- Invalid slots were zero, validity was binary, effort was bounded, and features
  outside the queue were identical to the masked control.
- Initial outputs and all initial model tensors were identical within each
  seed. Isolated native queries preserved the reconstructed integration state
  and original control queue. No new teacher labels were collected.

## Same-budget fitting

Four fresh supervised fits, seeds 711028 and 721028. Every fit used the same
6,804 original training rows, including 2,256 quiet rows, and the same 2,268
validation rows. Poses 6 and 7 never entered fitting. Each seed's two fits had
identical batch streams: 4,000 Adam updates, batch 512, learning rate 1e-4.
Only initial/final weights were retained. Total fitting time was 24.495 seconds;
this excludes physical evaluation, setup and auditing.

Raw-target residual RMSE, in radians:

- Seed 711028: train 0.005477 masked vs 0.005424 pending effect;
  heldout 0.028239 vs 0.023632, a 16.3% heldout-error reduction.
- Seed 721028: train 0.005520 vs 0.005436;
  heldout 0.024190 vs 0.021051, a 13.0% heldout-error reduction.

Both masked fits reproduced their archived final weights exactly. Their four
nominal/varied physical trajectories reproduced the archived qpos, issued
targets, alive flags and native failure flags exactly.

## Frozen physical results

The original support, motor RMS, COM, foot-drift, survival and return gates were
reconstructed independently. No threshold was widened.

- Masked core: seed 711028 nominal 5/8, varied 5/8; seed 721028 nominal 3/8,
  varied 2/8. Total 15/32; worst condition 2/8.
- Pending-effect core: seed 711028 nominal 0/8, varied 0/8; seed 721028 nominal
  3/8, varied 1/8. Total 4/32; worst condition 0/8.
- Core training poses: masked 8/24 vs pending effect 4/24. Heldout poses:
  masked 7/8 vs pending effect 0/8.
- Unused timing (onset 0.9, ramp 1.4, hold 3.2 seconds): masked 4/8 and 2/8;
  pending effect 1/8 and 2/8.
- Nominal 0.005-rad raw-proposal noise, streams 710083 and 710084: masked
  5/8, 5/8 for seed 711028 and 2/8, 3/8 for seed 721028; pending effect
  0/8, 0/8 and 0/8, 1/8 respectively. Noise was inserted before the original
  authority, fade and slew projection. It was not a physical ball impact.
- Combined timing/noise challenges: masked 21/48, worst condition 2/8;
  pending effect 4/48, worst condition 0/8.
- Quiet Stand: both pending-effect seeds retained the exact archived 60-second
  qpos, issued-target and alive trajectories, 32/32 cases. The task addition is
  disabled during quiet Stand; this verifies preservation of the Stand anchor,
  not a newly learned arm or stepping skill. Quiet reports' arm-skill `complete`
  counts are zero and are excluded from active-skill scores.

## Local response diagnosis

After the physical audit, compare the frozen heads with the 120 teacher queries
already saved by the private-queue study. No new teacher queries or fitting.
Each seed is evaluated on its own 48 queued-target alternatives.

- Masked raw task output is unchanged in all alternatives, as expected.
- Pending-effect raw output changes in all 96 alternatives. Information is
  available and used; that alone did not produce a successful feedback policy.
- Median full-vector response cosine with the teacher: 0.229 for seed 711028,
  0.240 for seed 721028. Eighteen of 96 response vectors point into the opposite
  half-space from the corresponding teacher response.
- Median own-joint raw-target derivative per queued-target change:
  teacher -0.140 vs learned -0.0207 for seed 711028; teacher -0.227 vs learned
  -0.0332 for seed 721028. These are medians, not per-case gain ratios.
- Own-joint derivative sign disagrees with the teacher in 0/48 and 9/48 cases.

These probes are local synthetic queue alternatives on archived masked-policy
states. They are not a physical stability certificate, proof that exact teacher
responses are uniquely necessary, or proof of the sole cause of regression.
They show that exposing the missing information did not teach the corresponding
teacher response well. Lower teacher error on its original trajectories was
insufficient to establish closed-loop performance.

## Next priority

Before another fit, admit response supervision on physically reachable disturbed
training states. Use bounded command pulses through the actual delay queue on
poses 0:6, query the verified teacher after those commands execute, and verify
the paired state/queue/goal timing and subsequent physical teacher continuation.
Compare changes in teacher target, rather than only absolute target error on
the teacher's original trajectory. Preserve exact quiet draw streams, the
existing raw-target contract, both fresh seeds and untouched whole-pose holdouts.
Do not fit the heldout private-queue diagnostics.

Choose any representation or response-loss change from those measurements
before fitting; do not combine another input change with an unmeasured sampling
change. A longer PPO run is not the next step. Weight transfer, single-foot
support and learned foot placement remain gated on a reliable preceding skill.

## Verification and resources

- `audit.json`: passed; formula, dataset exclusion, identical initialization,
  draw streams, exact masked replay, noise streams, 192 physical cases and gates.
- `queue-response-diagnostic.json`: passed; archived teacher-query alignment,
  frozen response comparisons and source/artifact hashes.
- 16,000 supervised optimizer updates; 8,192,000 sample presentations;
  zero PPO transitions, new teacher queries or C# compilations.
- Experiment files approximately 78.2 MB before this progress document/review.
  Existing sealed experiments were preserved. All experiment processes finished.
- Arena ONNX remains the earlier 374-input recovery brain. These research
  heads use 421 inputs and are not Arena replacements. Learned recovery foot
  placement is still absent.

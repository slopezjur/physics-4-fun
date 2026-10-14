# Exact-interface transfer examples

Date: 2026-10-09. Prospective successor to the rejected independent stepping
pilot. Existing experiment sources, model, rig, Arena and task gates are frozen.

## Question and physical contract

Can a feedback teacher complete the current level-0 task using the same commands
available to the learner? This is a feasibility/data-collection experiment, not
a learned recovery controller or ball-impact evaluation.

Use StepSkillEnvironment without changes: original resets and source actor,
528 issue-time inputs, 33 latent outputs, 0.5-rad leg / 0.1-rad other-joint
correction envelope, 0.02-rad per-interval slew, original two-command delay,
torque limits, PD gains and zero passive springs. No force, root injection,
prepared pose/queue, arm-clearing prerequisite or hidden motor assistance.

The task requests 4 cm COM transfer toward the opposite foot, zero commanded
foot displacement, measured transfer dwell and return to Stand. Keep all
existing tolerances, load-share, permanently latched stance-drift and final
two-second settling requirements. Survival or a plausible IK pose is not success.

## Teacher

Private IK holds both reset feet and targets the exact two-dimensional COM
goal, retaining reference arms and a four-degree private root lean in the
transfer direction. Only resulting bounded joint commands reach live physics.
Use the existing delayed prediction and interval-COM controller. Optimize
inside the exact next issued-target box obtained by intersecting the correction
envelope, previous correction plus/minus slew, and physical joint limits.
The midpoint and halfwidth of that box parameterize the existing box solver.
Recover a finite latent from a feasible residual, then issue it exclusively via
StepSkillEnvironment.step. Verify actual issued targets and unchanged live
state/queue during every private query. Finite inverse-tanh requires an explicit
one-part-per-million inset from the correction envelope boundary.

The measured FSM controls progression. During TRANSFER, smooth from reference
to transfer pose over the declared ramp. During SETTLE, smooth from transfer
pose back to reference over the same ramp, then request zero latent through
the ordinary slew limiter. There is no time-only transfer completion.

## Bounded physical panel and stop rules

1. Evaluate training poses 0 and 1 under both side commands (worlds 0,1,8,9),
   matched original resets, onset 1 s, full 12 s horizon, ramp 2 s.
2. If all four do not succeed, run the same four cases once with ramp 4 s.
   These are the only two pilot variants. No retrospective gate/config changes.
3. A variant must show at least one complete success on each side before
   expansion. Choose the highest success count; ties prefer the first variant.
   If none qualifies, stop before fitting and report the teacher's limitation;
   this does not establish impossibility for every controller under these bounds.
4. If qualified, evaluate all twelve training cases (poses 0-5, both sides) using
   the chosen variant. Require at least 9/12 complete successes, including 4/6
   on each side, before collecting nearby-state training data or fitting.
5. If admitted, collect two additional twelve-case teacher panels with onset
   0.8 and 1.2 s. These vary the actual source-policy state at command entry.
   Use successful trajectories only. No reserved poses 6/7 enter fitting,
   normalization or physical-variant selection. Maximum 44 teacher cases.

## Conditional neural initialization

Only after the physical admission above, fit the unchanged SteppingHead against
exact issue-time latent labels using its complete conditional mean, including
the 0.9 previous-latent term. Use two fresh shuffle seeds 961009/971009 and the
same zero-output initialization (seed 960009), 64 epochs, minibatch 256,
Adam 3e-4, gradient norm cap 0.5. Balance active TRANSFER/SETTLE and side groups;
WAIT observations do not supply a nonzero target. No PPO extension or parameter
sweep is included. Retain initial/final actors only.

Evaluate each neural controller alone at onset 1 s over all sixteen cases and
in sixteen 60-second disabled-command quiet cases. Keep the existing progression
gate: at least 9/12 training-case successes, at least 4/6 each side, both seeds,
plus all quiet cases. Reserved cases are reported separately. Only a passing
pair receives the onset-1.7-s check. Lift, translated placement, ball perturbations
and Arena export remain subsequent tasks, even if this transfer pilot succeeds.

## Evidence

Save exact pre-step features/raw observations, issued latent, source action,
nominal/planned/issued targets, prior correction and latent memory, measured
phase/geometry/load/Stand states, initial states, per-case outcomes and actual
control/private-query counts. Seal source/input/output hashes. Independently
reconstruct target bounds/delay, features, foot geometry and measured task gates.
No approximate reconstruction of old teacher observations is used as training
data. Preserve failures and skipped-stage reasons. This experiment has no
periodic model checkpoint series and changes no deployed controller.

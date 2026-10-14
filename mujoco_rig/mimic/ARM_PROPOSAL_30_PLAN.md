# Thirty-step proposal foundations investigation - 2026-10-08

Continue autonomously under the user's explicit thirty-step instruction and
existing authorization for checks and bounded experiments. The preceding
ARM_RECOVERY_30_PROGRESS.md identified an issued left-arm error near 0.5 rad.
No long run, compilation, or Arena change is planned.

## Physical contract

Freeze the f832bab8 Stand anchor, rig, zero passive springs, two-command delay,
33 motors, torque/joint limits, +/-1 rad source-relative addition envelope,
+/-0.02 rad addition slew, and exact quiet bypass. All active comparisons use
the existing eight poses, whole-pose holdouts 6/7, and unchanged full hold/return
gates. Report whether a controller is learned, scripted, or a physical teacher.
Do not count edits, reports, or routine checks as research steps.

## Prospective sequence

1. Audit archived teacher labels, normalization and proposal errors on teacher states.
2. Reproduce the inherited proposal and decompose its commands on its own states.
3. Compare direct joint-goal targets with the inherited proposal.
4. Compare static support feedforward with direct goals.
5. Revalidate the successful physical-teacher reference and its target semantics.
6. Compare direct arm goals while leaving leg targets to the frozen Stand anchor.
7. Compare arm-only static feedforward with the same anchor-leg control.
8. Remove only the largest-error left-arm proposal coordinate.
9. Remove the learned proposal on all arm coordinates, retaining its leg outputs.
10. Bound learned goal-relative residuals and compare physical tracking.
11. Query the physical teacher on identical learner states to measure off-policy target error.
12. Trace when issued bias first appears and separate proposal, slew and joint clipping.
13. Compare sensitivity to physical state, goal and inherited normalization.
14. Choose and physically validate the smallest justified replacement representation.
15. Fit a fresh task head only if the target/representation audit supports it.
16. Evaluate the first fit on the complete nominal skill and withheld poses.
17. Check changed movement timing against matched initialization/control.
18. Reproduce fitting and physical outcomes with an independent fresh head.
19. Verify exact quiet Stand retention for candidate heads.
20. Diagnose remaining learner-state error against feasible teacher targets.
21. Run a bounded correction-data or curriculum comparison only if justified.
22. Replicate that comparison with a fresh optimizer and independent seed.
23. Evaluate changed timing and whole-pose generalization after correction.
24. Compare smaller motion amplitudes to localize any remaining curriculum gap.
25. Challenge the full movement amplitude without relaxing admission gates.
26. Test a slower or otherwise unused movement timing for robustness.
27. Assess body support and foot stability over complete motion phases.
28. Challenge the leading explanation with a matched controller counterfactual.
29. Verify sixty-second quiet retention and independently repeat a decisive condition.
30. Perform final learned-policy admission and decide whether weight-transfer work can resume.

Adapt later steps to observed results and record concrete bounds before running
them. Maximum six new task-head fits, at most 4,000 supervised updates or
24,576 PPO transitions each. Prefer a smaller budget if an intervention fails
its physical prerequisite. Shared supervised data must exclude poses 6/7 from
normalization and fitting. No extra run to conceal a negative result.

Evidence root: logs/mimickit-training/arm-proposal-30-20261008-01.
Completed-step ledger: ARM_PROPOSAL_30_PROGRESS.md.

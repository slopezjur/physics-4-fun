# Autonomous recovery investigation: 30 bounded steps

Authorized on 2026-10-08: continue autonomously through thirty research steps.
The previous explicit authorization to run checks and bounded experiments applies.
No new C# compilation, Arena installation, or long training run is planned.

## Fixed contract

Preserve the f832bab8 Stand anchor, frozen queue-masked proposal, rig, passive
springs, delay, torque/joint limits and whole-skill admission thresholds. Include
the untrained zero-mean correction head in comparisons. Frozen actors 521028
and 531028 are experimental, rejected policies. Success means supported,
accurately tracked arms-clear hold AND return to Stand; partial counts are not
learned stepping or impact recovery. Poses 6/7 remain held out from fitting.

Use predeclared evaluation seeds 610081-610084 and independent replication
seeds 610091-610094, paired across actors. Training std is 0.005 radians.
All evaluation policies are frozen. Retain compact reproducible evidence,
check hashes and physical gates, and report negative outcomes.

## Research steps

1. Compare mean and training-noise execution for all three frozen actors.
2. Repeat that comparison with independent evaluation seeds.
3. Compare mean/noisy execution under changed movement timing.
4. Test half-sized independent noise against matched full-noise controls.
5. Test quarter-sized noise against matched half/full-noise controls.
6. Quantify same-state projected mean versus mean projected command.
7. Execute expected projected commands to separate clipping bias from noise.
8. Check expected-command execution with changed timing.
9. Compare three-frame correlated noise with independent noise.
10. Isolate ramp, hold and return noise to localize sensitivity.
11. Measure matched ramp interventions and subsequent physical reward.
12. Measure matched hold interventions and subsequent physical reward.
13. Measure matched return interventions and subsequent physical reward.
14. Compare frozen critic action rankings with ramp physical outcomes.
15. Compare frozen critic action rankings with hold physical outcomes.
16. Compare frozen critic action rankings with return physical outcomes.
17. Compare short and longer physical reward horizons for those interventions.
18. Separate immediate reward, delayed reward and critic-bootstrap contributions.
19. Compare learned correction directions with matched physical improvements.
20. Check the resulting diagnosis on held-out poses and a second horizon.
21. Physically validate the smallest execution/learning correction justified by
    steps 1-20, against the frozen control, before fitting anything.
22. Run one bounded fit only if that intervention supports it; otherwise run a
    targeted frozen diagnostic selected and documented from the evidence.
23. Independently replicate that fit or diagnostic.
24. Compare final candidates against their shared initialization and control.
25. Evaluate candidates under changed movement timing.
26. Verify thirty-second quiet Stand retention with exact anchor comparison.
27. Evaluate active-skill tracking and foot stability by phase and pose.
28. Challenge the leading explanation with a matched counterfactual.
29. Repeat the decisive comparison with fresh seeds or an unused condition.
30. Run the final physical admission comparison and record promotion/rejection.

Later steps are conditional research decisions, not a promise to fit thirty
models. Record the concrete hypothesis and bounds before each adaptive block.
Do not count file edits, documentation, or routine audits as research steps.
No threshold relaxation to manufacture success. No more than four new fits,
each at most 24,576 transitions, without a further documented decision; avoid
training entirely when the diagnosis does not support it.

## Progress ledger

Results and exact completed-step evidence: ARM_RECOVERY_30_PROGRESS.md.

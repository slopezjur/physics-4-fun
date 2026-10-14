# Steps 20-30: elbow localization and active-only task supervision

Step 14's complete eight-arm-motor ablation passed 6/8 nominal and 7/8 varied
full skills, with 8/8 supported holds and returns in both timings. The earlier
six-upper-arm-only variant passed 0/8 nominal and 1/8 varied; therefore include
elbows explicitly in causal comparisons. Static arm offsets and direct goals
with anchor non-arm control remained at zero full skills.

The fresh 128-unit inherited-feature head passed 5/8 full skills at both timings.
The task-relative head failed both seeds despite better held-out regression loss.
Reject that representation for this sequence. The fresh inherited head differs
from the old 256-unit continued-training proposal in width and initialization;
its success cannot be attributed uniquely to either change.

20. Compare removal of left-elbow, right-elbow and both-elbow residuals in the
    original frozen proposal; also compare static feedforward on all eight arms
    while retaining the original non-arm outputs. Nominal and changed timing.
21. Fit and evaluate a fresh inherited-feature 128-unit replica, seed 721028,
    using the same complete teacher data and 4,000-update budget.
22. Fit seed 711028 with the same initialization/architecture/settings but remove
    quiet samples, whose commands are bypassed in execution. Keep active teacher
    examples including ramp, hold and return. Evaluate nominal and changed timing.
23. Repeat active-only fitting and physical evaluations with seed 721028.

This reaches six total new fits including the three from steps 15/18. Stop
fitting after step 23. All fitting/normalization still excludes poses 6/7. The
quiet filter is applied before train/validation splits; common-evaluation-set
regression metrics must be used when comparing losses across these fits.

24. Compare half-amplitude task goals for all four inherited-feature heads.
25. Test full amplitude with 0.005-rad sampled action noise, seeds 710081/710082,
    to measure robustness of the actual proposal outputs, not post-slew correction.
26. Compare unused slower timing: onset 1.5 s, ramp 2.0 s, hold 4.0 s, unchanged
    1.5 s return and 2 s settling. Use all four heads and original proposal control.
27. Compare common-data regression, phase failures, motor/COM/foot gates and
    worst-pose results; select the best replicated training recipe by the minimum
    nominal/varied complete count across its two seeds, tie-break by summed full
    skills, then returns. Retain ties as inconclusive rather than using quiet data.
28. Matched counterfactual: remove learned elbow outputs in each new head and
    repeat nominal/varied cases. Do not count patched output as fully learned.
29. Run sixty-second quiet retention for both seeds of the selected recipe plus
    a zero-correction anchor baseline, checking complete trace equality.
30. Final whole-skill challenge for the selected pair: a distinct faster timing
    (onset 0.8 s, ramp 1.2 s, hold 2.5 s), plus its already measured nominal,
    varied, slower and noisy results. Keep all admission gates unchanged.

Select no Arena candidate unless the learned policy passes all nominal and
changed-timing cases, independently replicated with quiet retention. Report
partial progress without relabeling arm-goal tracking as learned foot placement.
No Arena update or compilation in this bounded research sequence.

# Recovery across launch distances - 2026-10-08

Continue the user-authorized bounded learning work from the selected learned
preview f832bab8. Its failed clear-distance repeat case is pose 7: the first hit
ends at 5.183748 s, horizontal speed first exceeds 0.2 m/s at 6.817212 s, and the
dummy falls at 7.600608 s, before the second throw at 9 s. This is a first-hit
generalization failure, not evidence that the second hit itself caused the fall.

Compare two fresh optimizer continuations from exactly the same full checkpoint:
fixed 0.25 m flight versus mixed 0.25/0.46 m flight. Both retain 5 kg, 1.6 m/s,
+X Chest aim, first onsets 1/3/5/9 s, second shot four seconds later, 20-second
episodes, all eight original poses, 64 worlds x 64 steps, three actor epochs,
actor LR 3e-5 and quiet-action rehearsal weight 100 / 256 examples per actor
optimizer call. Each replica cycles all eight onset/distance combinations over
resets in the mixed arm. Optimization seed 391028, 100 PPO updates per arm,
420 training-second cap each; maximum 200 updates / 819,200 transitions.
No physical, reward, success-threshold, passive-spring or target-delay changes.
No scripted teacher and no periodic checkpoint copies. Matched start admission
must pass within and across arms; an initialization fault is not a result.

Reuse the selected parent's sealed evaluation baseline. Assess both new endpoints
on native/GPU: eight poses, 30-second quiet Stand, four single-hit controls
(1.4/1.6 m/s, +/-X, launch 5 s, flight 0.25 m), and 1.6 m/s +X two-hit trials
(launch 5/9 s, flight 0.46 m, end 16 s). This is 192 new cases. A candidate must
retain every parent primary recovery count, pass insertion/body/backend gates,
and improve the two-hit result from 7/8 to 8/8 on both backends. If both qualify,
prefer mixed only if it beats fixed; otherwise fixed establishes an extra-practice
gain without evidence for distance mixing. If neither qualifies, keep Arena.

For a qualifier only: compare it and the parent on held-out 3/7-second throws at
0.46 m, 1.6 m/s, end 14 s (32 cases), requiring no loss, then verify candidate
120-second quiet Stand on both backends (16 cases). Maximum 240 new cases.
Only a passing successor may replace the current experimental Arena selection.
One optimization seed does not establish repeatable recipe improvement.

After evaluators verify full-resolution control equations and all gates, retain
compact traces with recovery/kinematic/observation evidence and a saved audit
record; discard newly generated redundant substep control arrays from those
traces before final sealing. Existing evidence and the running Arena stay intact.
No C# changes or compilation. No stronger-hit escalation in this comparison.

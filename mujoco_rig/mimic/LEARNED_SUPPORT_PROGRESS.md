# Learned support progress: 20/20 completed

2026-10-08. Four fresh supervised learners, 16 frozen evaluation batches (256 physical cases). None passed; the current Arena parent f832bab8 remains selected. These are support-prerequisite experiments, not successful foot-placement or impact-recovery training.

## Executed ledger

1. `source-421028`: 4,000 supervised minibatch updates; train RMSE 0.002236 rad; whole-pose held-out RMSE 0.017890 rad.
2. `goal-421028`: 4,000 supervised minibatch updates; train RMSE 0.001961 rad; whole-pose held-out RMSE 0.008890 rad.
3. `source-431028`: 4,000 supervised minibatch updates; train RMSE 0.002483 rad; whole-pose held-out RMSE 0.017402 rad.
4. `goal-431028`: 4,000 supervised minibatch updates; train RMSE 0.001905 rad; whole-pose held-out RMSE 0.008135 rad.
5. `source-421028-onset1`: Command onset 1 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
6. `goal-421028-onset1`: Command onset 1 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
7. `source-431028-onset1`: Command onset 1 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
8. `goal-431028-onset1`: Command onset 1 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
9. `source-421028-onset3`: Command onset 3 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
10. `goal-421028-onset3`: Command onset 3 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
11. `source-431028-onset3`: Command onset 3 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
12. `goal-431028-onset3`: Command onset 3 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
13. `source-421028-onset9`: Command onset 9 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
14. `goal-421028-onset9`: Command onset 9 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
15. `source-431028-onset9`: Command onset 9 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
16. `goal-431028-onset9`: Command onset 9 s: 0/16 lift/hold/replant/return passes; 0/16 valid 50 ms lifts; all failed physically.
17. `source-421028-quiet30`: 30-second quiet protocol: 0/16 passes; all terminated early by physical failure.
18. `goal-421028-quiet30`: 30-second quiet protocol: 0/16 passes; all terminated early by physical failure.
19. `source-431028-quiet30`: 30-second quiet protocol: 0/16 passes; all terminated early by physical failure.
20. `goal-431028-quiet30`: 30-second quiet protocol: 0/16 passes; all terminated early by physical failure.

## Findings and limits

Training used 12,530 reconstructed examples from 13 complete successful teacher sequences: 8,697 training examples and 3,833 held out by whole pose. Right-foot teacher poses 2, 3 and 4 were excluded from training but included in every physical evaluation. This is an explicit successor experimental-data policy; the historical all-sixteen export gate remains failed.

Goal-relative prediction improved held-out errors from 0.01789/0.01740 rad to 0.00889/0.00813 rad across the two seeds. It did not produce stable closed-loop control. All 192 entry cases and all 64 enabled-adapter quiet cases failed. No foot placement, natural recovery, impact timing robustness, or stronger-hit capacity was demonstrated.

The four training runs took about 19.5 measured seconds in total, each 4,000 optimizer updates. These are supervised updates, not PPO cycles or new environment transitions. The frozen f832bab8 actor was not trained. Only initial/final learner weights were retained.

The unchanged anchor passed a separate 30-second quiet check in all 16 worlds of the same harness. A repeated goal-relative handoff failed again. At activation, sampled live states differed from teacher states by 1.7-15.4 mm in root position and up to 5.18 training standard deviations in an input feature. This supports investigating state-distribution mismatch and feedback stability; it does not uniquely establish a cause.

Command replay confirmed issued-target arithmetic within 5.96e-8 rad. Reconstructed observations stayed within 1e-5 through frame 405 (~6.75 s), whereas learned controllers failed around 0.47-2.10 s after activation. Full recorded-target replay did NOT pass: accumulated trajectory divergence exceeded 1e-4 at frame 500, and at frame 535 the current-anchor offset required to replay the old teacher exceeded the 1-radian envelope. Therefore no full replay equivalence is claimed. Approximate reconstruction and anchor transfer remain explicit limitations.

The first two audit attempts aborted without admitted results; audit -03 recorded the replay failure rather than treating it as success. The final review independently reconciled all 20 experiment entries, case failures, and protected input/output hashes. Two extra batches (32 cases) checked the unchanged anchor and repeated the handoff; replay diagnostics are separate from the 256 planned evaluation cases.

## Next priority

Establish stable learned-controller activation before more timing sweeps: preserve exact zero residual for a quiet goal, validate a bounded transition from the current anchor, and collect corrective supervision on states visited by the learner. Require quiet retention and successful learned entry before extending to foot displacement and hit-triggered recovery. Do not spend a longer PPO run on these rejected adapters.

## Evidence

- Plan: `LEARNED_SUPPORT_PLAN.md` (frozen input).
- Experiments and final hashes: `logs/mimickit-training/learned-support-20261008-01`.
- Failed full replay audit: `logs/mimickit-training/learned-support-audit-20261008-03/report.json`.
- Reconciliation, anchor control, and handoff observations: `logs/mimickit-training/learned-support-review-20261008-01/report.json`.
- No Arena selection, ONNX export, physics limits, springs, or C# build changed. Godot PID 10144 remained responsive at closure.


## 2026-10-08: bounded learned handoff follow-up

See LEARNED_HANDOFF_PLAN.md and LEARNED_HANDOFF_PROGRESS.md. Added an exact
quiet-goal bypass and gradual, rate-limited learned correction. Three 30-second
quiet batches passed 16/16 with issued commands and qpos exactly matching the
unchanged f832bab8 Stand anchor. This is a structural safeguard, not newly
learned Stand. Live-only rate limiting was independently verified.

Two matched correction rounds used 277 learner-state labels and 6,000 supervised
updates. All support attempts failed before lifting; the later-state treatment
regressed and is rejected. Better fit to labels did not imply better control.
Completed 144 planned physical cases and 32 frozen audit cases. Arena unchanged.

Next: isolate fixed arms-clear holding and then small weight transfer before
another full support sequence. Do not extend the failed correction treatments
or advance to foot placement/impacts. Evidence: learned-handoff-20261008-01,
learned-handoff-late-20261008-01 and learned-handoff-review-20261008-01.

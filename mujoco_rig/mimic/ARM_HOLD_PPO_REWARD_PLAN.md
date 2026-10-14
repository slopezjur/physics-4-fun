# Bounded PPO reward-width comparison - 2026-10-08

Compare distance-reward width multipliers 1 and 2, keeping motor/COM weights 0.45/0.20 and all other rewards unchanged. Width 1 uses motor scale 0.05 rad and COM scale 0.02 m; width 2 uses 0.10 rad and 0.04 m. These scales affect training reward only. The fixed full-skill admission gates remain 0.05 rad motor RMSE, 0.02 m COM error and 0.02 m foot displacement with supported hold and return to Stand.

Both arms start from the retained queue-masked seed-451028 actor, with independent fresh optimizers for seeds 501028 and 511028. Use actor rate 3e-6 consistently, unchanged critic and PPO settings, fixed exploration std 0.005 rad, 32 batches of 64 frames across 12 active training worlds: 24,576 transitions per run. Poses 6 and 7 remain excluded from actor/critic losses. Exactly four runs; no longer run or additional reward sweep.

Reuse the sealed rate-study training loop and corrected success gate. Retain the actual Stand anchor, physical targets, zero passive springs, two-command delay, quiet bypass, fade and slew. Before training, verify both reward adapters exactly reproduce the unchanged actor's 540-frame physical trace. Replay old successful and failed traces through the frozen success gate. The width-1 controls must reproduce the preceding lower-rate actor weights and nominal traces exactly.

Keep full-batch post-update analytic KL and phase-separated reward diagnostics. Reconcile reward component sums. A higher broad-kernel reward is not evidence of better control; judge physical outcomes with the same frozen evaluator. Evaluate every final actor on eight nominal poses and sixteen 30-second quiet cases. Only improved complete-skill success with support and return counts at least 4/8 warrants changed-timing evaluation. Preserve per-pose regressions and whole-pose holdouts in reporting.

No teacher fitting, stronger hits, lifting, Arena change, export or C# build. If broadening fails, move next to action/state responsiveness and temporal credit assignment rather than another parameter sweep. Record this bounded comparison and retain compact initial/final checkpoints and evaluation traces only.

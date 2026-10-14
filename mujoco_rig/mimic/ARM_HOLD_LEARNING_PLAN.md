# Sustained arms-clear learning pilot

Continue the authorized bounded checks and experiments. Freeze the current f832bab8 Stand anchor, physical limits, zero passive springs and two-command delay. Arena remains unchanged.

Collect exact issue-time 421-input observations and raw, unblended 33-joint physical target labels. The actual commands retain the same authority/displacement fade and 0.02 rad/frame handoff. Save the raw labels, source targets, previous addition and issued targets so composition can be independently checked. Do not reconstruct observations from older traces or apply blending twice to labels.

Use all eight original poses in two physical teacher runs: nominal onset 1 s / ramp 1.5 s / hold 3 s; variation onset 1.2 s / ramp 1.7 s / hold 3.5 s. Vary command timing, not live physical state. Return over 1.5 s and settle for 2 s. Train only complete successful trajectories from poses 0-5; retain entire poses 6 and 7 for prediction and physical holdouts. Failures are explicit exclusions, never successes. These diagnostic teacher runs are not learned results.

Compare the unchanged goal-relative 421028 neural learner, a matched 4,000-update old-example replay control, and a 4,000-update exact arms-hold learner. Both training arms start from the same weights with fresh Adam state, seed 451028, learning rate 0.0001, batch 512 and fixed normalization. Arms-hold training is intentionally specialized; old skills are not assumed retained.

Evaluate every frozen model on all eight nominal pose cases using the same handoff. Hold gate: final two seconds of the commanded hold continuously supported, motor tracking RMSE <=0.05 rad, horizontal COM error <=0.02 m and foot drift <=0.02 m. Return gate: final two seconds of original Stand. Check 30-second quiet identity for the new learner. Only after 8/8 learned hold/return success, evaluate varied timing and replicate the training seed before transfer work. Retain initial/final states and compact evidence. No export, compilation, hit tests or Arena promotion in this pilot.

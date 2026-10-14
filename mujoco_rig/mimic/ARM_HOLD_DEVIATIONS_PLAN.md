# Recoverable deviations around the held pose

Continue authorized bounded experiments. Keep f832bab8, physical limits, zero springs, two-frame latency, queue-masked learner and the existing handoff. Arena remains unchanged.

Collect one eight-pose physical teacher sequence. Only during the 2.5-4.5 second hold interval, add bounded, smoothed joint-target disturbances: per-world RNG seeded from 471028, uniform innovations within +/-0.012 rad and recurrence noise = 0.85*noise + 0.15*innovation. Clamp the resulting raw target to original joint limits and the existing source-relative 1-radian envelope. The normal authority/fade/slew pipeline remains in force. Outside that interval the disturbance is zero. Never write live positions or velocities. These motor-command deviations are not impact training.

Save both the applied noisy raw target and the clean corrective target queried from the physical controller at the actual visited state. Train only whole successful teacher trajectories; retain poses 6 and 7 as holdouts. Restrict the new correction pool to 2.5-5.5 seconds, including the disturbance-free recovery portion.

Both learning arms start from the same queue-masked seed-451028 checkpoint, with fresh Adam state, seed 481028, 4,000 updates, batch 512, learning rate 0.0001 and fixed normalization. Control replays the existing exact hold examples; treatment uses half existing examples and half new correction examples. Data calls and initial weights are matched. The control also consumes the correction-pool RNG draw so common old-example draws align.

Evaluate both frozen learners on all eight ordinary-start nominal holds. Preserve supported hold, motor RMSE <=0.05 rad, COM error <=0.02 m, foot drift <=0.02 m and final two-second Stand gates. Compare error magnitudes as well as pass counts. Check 30-second quiet identity for the treatment. If it gains any full hold-and-return success over control, replicate the paired comparison with seed 491028; if all eight pass, additionally test the changed timing. Otherwise report the failed comparison and stop this bounded pilot. No transfer, Arena promotion, export or C# build.

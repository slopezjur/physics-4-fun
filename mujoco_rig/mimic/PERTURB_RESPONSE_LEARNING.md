# Prospective learning comparison: delayed recovery credit

Sealed after the response audit and the frozen velocity sensitivity probe,
before any continuation in this block. The parent is f832bab8. No saturated
actor outputs or target-limit clipping precedes pose 7's first departure.
Actor output changes on the next control frame; the two-frame queue is exact.
By departure its COM x is 0.208 m and the forward foot/toe box bound is about
0.160 m, with only millimeters of foot repositioning. These are descriptive
geometry bounds, not a contact support polygon or a proof of inevitable falling.
Velocity gain 1.25 does not improve the parent; gain .75 at onset 2 fails backend
admission and is excluded. The probe is diagnostic, never an export candidate.

Hypothesis: more persistent TD credit may improve this delayed instability.
Control TD-lambda .98; treatment .995. Discount remains .99. Within a 64-frame
rollout, the multiplier (discount * lambda)^63 is about .149 versus .387.
This does not connect an entire 1.6-second delay directly through a 1.067-second
rollout: value bootstrapping is still required. The observation motivates a
bounded hypothesis test, not a diagnosis that TD-lambda caused the failure.

Two independent optimizer seeds, 401028 and 411028; both arms start from the
same complete f832bab8 checkpoint with fresh optimizers and matched state/RNG.
100 PPO updates per arm, 64 worlds x 64 steps, three actor epochs, actor SGD
LR 3e-5, quiet rehearsal weight 100 with 256 examples per actor optimizer step.
Same mixed .25/.46 m training flights, 5 kg / 1.6 m/s +X Chest impacts,
first onsets 1/3/5/9 seconds, second +4 seconds, 20-second episodes, eight poses.
No other config, reference, reward, model, controller or success-gate changes.
Bound: 400 updates / 1,638,400 transitions total; 420 measured training seconds
per arm. Initial/final checkpoints only. Do not extend an unsuccessful arm.

Each seed: 192 primary frozen native/GPU cases over both arms. Each actor gets
30-second quiet Stand, four original single-hit conditions (1.4/1.6 m/s, +/-X,
launch 5 seconds, .25 m flight), and repeated 1.6 m/s +X (5/9 seconds, .46 m
flight, 16-second end). Primary qualifier requires all 8/8 in every condition,
including improved repeated recovery and valid body/insertion/backend gates.

After both seeds finish, test the parent and every qualifier at untrained
4/8-second repeated onsets, .46 m flight, end 15 seconds. Candidate must not
regress versus parent. Every remaining qualifier gets 120-second quiet Stand
on both backends, 8/8 required. If no primary qualifier exists, characterize
only the parent's untrained timing and retain Arena. Maximum new cases in
this whole block: 96 diagnostic + 384 primary + 80 timing + 64 retention = 624.

For selection, prefer candidates with higher untrained-timing recovery count;
break ties by earlier optimizer seed, then control before treatment. A control
improvement is extra-practice evidence, not evidence favoring TD-lambda .995.
A recipe improvement requires treatment to beat its matched control at both
seeds. Export only a fully qualified learned policy with unchanged Arena
contract and checked ONNX actions. Keep f832bab8 otherwise. No C# build.

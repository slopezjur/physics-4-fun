# Saved recovery reward audit - 2026-10-09

Use the sealed zero, negative and positive ankle-pulse traces in
`recovery-starts-20261009-01`. Analyze poses 0-5 only, at both existing diagnostic
timings, both force signs and paired quiet continuations. This is 24 pushed
starts x three continuations and 24 quiet starts x three continuations: 144
training-pose trajectories. Reserved poses 6/7 do not enter rankings or decisions.

Start scoring at the first controllable decision after the initial impact
interval (frames 61 and 103). Include the first failing transition with its -2
reward, exclude all frozen post-failure frames, and end at the original six-second
endpoint. Use gamma=0.995. Compare each pulse with the zero continuation from the
same start. Classify rescues, losses and unchanged outcomes using the previously
audited final-two-second settled criterion. Report every paired ranking, not just
an average. Check how reward differences accumulate over 0.2, 0.5, 1 and 2 seconds.

Reconstruct reward components from saved state, with native kinematics/COM
Jacobians only and no integration. Check reconstruction against the saved reward.
Separate posture baseline, capture, bilateral settling, swing, action cost and
terminal penalty. Also independently recompute discounted sums by backward
recursion and termination scanning.

PPO bootstraps time-limit states, so finite-horizon reward alone is not its full
training target. Reconstruct the saved 427-input endpoint observations, verify
the same reconstruction against saved recovery-start features, then add the
discounted endpoint values from each saved final critic for nonfailed episodes.
These are frozen-critic sensitivity checks on diagnostic trajectories, not the
actual on-policy GAE estimates used during learning. No historical rollout
advantages or evolving critics were saved, so do not claim to reconstruct them.

Confirm that no-fit physics probes are diagnostic interventions rather than a
deployable policy. A per-start best-return choice is an oracle over three known
continuations, not learned recovery. Report quiet costs separately. Do not tune
the reward, train, integrate physics, or change Arena in this step. The diagnostic
timings inform this investigation; use fresh timing checks for any later change.

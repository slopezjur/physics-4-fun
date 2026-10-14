# Learned coordinated ankle recovery - 2026-10-09

Test whether a small reactive learner can exploit the physically demonstrated
ankle rescues. No diagnostic pulse, source-action replay, teacher, prescribed
force label or hit clock is part of learned execution. No Arena promotion is
included in this pilot.

## Fixed controller and observations

Keep the deployed Stand/Perturb source frozen. One learned scalar controls both
Foot_L_rx and Foot_R_rx target additions. The existing 0.1-rad envelope,
0.02-rad/interval slew, joint/torque limits and two-interval delay are unchanged.
All other target additions are zero; the original source still controls all motors.

The 427 inputs are the source's 374 normalized observations, three frames of
17 measured balance features (current, previous, older), previous applied ankle
addition / 0.1, and previous sampled latent action. Balance features are COM XY
relative to foot midpoint / 0.1, COM XY velocity / 0.5, root angular velocity / 2,
capture-point XY relative to midpoint / 0.1, each foot load / body weight, pelvis
height error / 0.15, pelvis up, and each foot XY relative to COM / 0.2.
Physical-history features are clipped to +/-20. Reset fills all three frames
with the actual reset measurement; observation queries never advance history.
Only completed physical intervals advance history and controller memory.

Actor and critic: 427 -> 128 ELU -> 128 ELU -> 1, zero final layers. The actor's
conditional latent mean is 0.9 * previous latent + 0.1 * learned output. Gaussian
innovation standard deviation is 0.4 * sqrt(1 - 0.9^2). This gives persistent
exploration with initial stationary latent standard deviation 0.4. Evaluation
uses the conditional mean, retaining the same controller memory. PPO likelihoods
and exact KL use the conditional Gaussian with the recorded previous latent.
The learned output determines sign, amplitude and persistence from live state.

## Bounded learning and evaluation

Two seeds: 921009 and 931009, identical initial actor seed 920009. Each receives
32 collections of 64 frames across 12 worlds: 24,576 transitions. Reuse the prior
PPO implementation: three epochs, minibatches 256, actor LR 0.0001, critic LR
0.0003, gamma 0.995, lambda 0.98, ratio clip 0.2, exact conditional Gaussian KL
ceiling 0.01 with rollback. The direct-recovery reward is unchanged.

Train only poses 0-5, six-second episodes, 25% quiet; other episodes use 150 N
X pushes with equiprobable signs and six-interval duration. Choose hit frames
uniformly from 40-90 excluding 55-65. Thus evaluation frame 60 (about 1.0 s)
and frame 102 (about 1.7 s) are absent from training. No heldout poses affect
optimization, normalization or checkpoint selection. Retain initial/final heads
and final critics only; no intermediate checkpoints or parameter sweep.

For the initial zero head and both final candidates, evaluate both timings across
all eight poses, both signs, X/Y axes and 150/300 N (128 cases), sampled execution
on 150 N X at both timings (32 cases), and sixty-second quiet Stand (16 cases).
No checkpoint selection uses these outcomes. The zero-mean baseline must exactly
reproduce the previously sealed crossed baseline and quiet reference.

## Checks and decision

Before training, verify history ordering, idempotent observation reads, reset
isolation, source/input dimensions, native COM velocity against a Jacobian
calculation, force/timer exclusion and target composition. Require exact
zero-correction replay, preserved slew/delay, ankle-only additions and finite
inputs. Independently audit saved physical recovery and foot-placement evidence.

Compare final-two-second settled recovery, each push sign, reserved poses and
quiet Stand. A higher training reward alone does not qualify. Require useful
core recovery gains in both seeds without quiet or reserved-pose regression
before considering extension. Stronger/sideways pushes are reported separately.
Any mechanical footstep still requires the existing landing/settling audit.
These direct-force results cannot certify projectile or human-natural recovery.

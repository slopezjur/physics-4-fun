# Direct recovery pilot - 2026-10-09

## Decision

The accepted next priority is direct push recovery over the frozen deployed
Stand/Perturb controller. Complete arm-motion tracking is no longer an admission
prerequisite for this experiment. Previous arm results remain diagnostic evidence;
they do not establish a necessary prerequisite for recovery.

## Fixed experiment

- Frozen source: `perturb-preview-learning-20261008-01/single_ball/final/model.pt`.
- Original 33 motors, joint limits, torque limits, PD gains and two-interval delay.
  Passive springs remain zero. No root support, IK, teacher, scheduled foot motion,
  or future hit information enters the controller.
- New 407-input correction: the source's normalized 374 observations and the
  previous 33 target additions. Two 128-unit ELU layers, zero final layer.
- Addition: `0.1*tanh(latent)` radians, limited to 0.02 rad change per interval,
  added after the original clipped motor targets and clipped to joint limits.
  The frozen source continues reacting to the live state and pending commands.
- PPO uses stored Gaussian latent actions (standard deviation 0.1), so likelihoods
  describe the sampled actions before deterministic transformation and limiting.
  Deterministic mean and sampled execution are evaluated separately.
- Two fresh seeds, identical initial actor: 901009 and 911009. Each receives
  32 collections of 64 frames across 12 worlds = 24,576 transitions. Three epochs,
  minibatches 256, actor LR 0.0001, critic LR 0.0003, gamma 0.995, lambda 0.98,
  clip 0.2, exact latent-Gaussian batch KL ceiling 0.01 with rollback.
- Reset poses 0-5 train; poses 6-7 are never optimized. Training episodes last
  six seconds, with 25% quiet episodes, randomized cardinal direction, force
  150 or 300 N, onset 0.8 or 1.2 seconds, duration six control intervals.
- Evaluation: all eight poses, four directions, both strengths (64 cases), onset
  1.0 seconds for X and 1.7 seconds for Y. Also 16 sampled-policy cases with fixed
  noise and 16 quiet cases lasting 60 seconds, per final candidate and baseline.
- Pulses act at the Chest body COM in world coordinates. Force is explicitly set
  to zero outside its window, because this native engine retains external force.
  Approximately 15 and 30 N s of applied impulse are force-pulse tests, not a
  certification of a 10 kg projectile at 5 m/s.

## Objective and evidence

Use measured upright height, momentum capture over loaded feet, useful unloading
and lift toward capture, and low-velocity bilateral settling. No foot-position
anchor or bilateral-support failure during recovery. Native fall/contact failure
remains authoritative; strict Stand balance and posture apply to final settling.
The source imitation reward is not used in the new correction objective.

Verify zero correction against the original controller on quiet and pushed native
trajectories before training. Save compact physical traces, initial/final heads,
training histories and source/input hashes; no intermediate model checkpoints.
Independently reconstruct foot clearance from collision boxes. Report survival,
last-two-second settled recovery, unilateral support/lift, and audited foot
placement separately. Existing `corrective_step_audit` rejects rigid sliding and
requires landing support and subsequent settling; mechanical steps are not a
human-naturalness score.

## Decision rule

Do not promote on training reward. A promising correction must improve direct
recovery in both seeds without losing quiet retention or heldout-pose recovery.
Claim learned corrective stepping only with audited placement plus settling.
If both bounded runs fail, report the failure and diagnose the direct-recovery
action/reward evidence before extending compute. Ball contact, varied and repeated
hits, broader initial states, and Arena parity remain later admission gates.

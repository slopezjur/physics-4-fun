# Learned stepping curriculum - 2026-10-09

## Priority and scope

The user approved moving the main recovery effort from scalar ankle diagnostics
to independent learned leg movement. This prospective pilot supersedes the
unexecuted quiet-Jacobian diagnostic as the next priority. Previous experiments,
their results and sealed sources remain unchanged.

The first capability is commanded transfer, lift/replant and short foot placement.
Commands describe a task, not motor trajectories. This is not autonomous impact
recovery: selecting a foot and placement from the measured aftermath of a hit is
a later milestone. No candidate from this pilot is automatically exported to Arena.
Perfect arm-pose tracking is not a prerequisite.

## Fixed controller

- Frozen source: perturb-preview-learning-20261008-01/single_ball/final/model.pt.
- Original 33 motors, physical joint/torque limits, active PD, two-command delay,
  contacts and zero passive springs remain unchanged.
- A new zero-initialized head produces 33 independent target corrections. Each
  Thigh/Shin/Foot/Toe motor has a 0.5-rad correction envelope; other motors retain
  0.1 rad. All corrections retain the 0.02-rad/control-interval slew limit and
  final joint-limit clipping. These are experimental envelopes, not demonstrated
  requirements or changes to the source action scale.
- Zero correction preserves the original clipped source targets exactly. Verify
  native zero-head parity before learning; changing the envelope must not rescale
  the original policy's actions.
- Features: the frozen source's normalized 374 observations, three frames of 17
  measured balance features, 33 previous additions normalized by their envelopes,
  33 previous latent actions, and explicit task/phase/goal/timer observations.
- Two 128-unit ELU layers, zero final layer. Conditional actor mean is
  0.9 * previous latent + 0.1 * network output. Independent Gaussian innovations
  have standard deviation 0.3 * sqrt(1 - 0.9^2). Exploration persists over time;
  spatial coordination must be learned. Critic has a scalar output.
- Disabled commands use exact zero additions and cleared command memory. Quiet
  retention therefore validates a structural Stand fallback, not newly learned
  quiet behavior or a disturbance detector. No external force, live IK, teacher
  takeover, root support, scripted joint sequence or prepared-pose reset is used.

## Curriculum and objective

Level 0 requests a 4 cm COM transfer toward the stance foot, then return to Stand.
Level 1 adds an unloaded swing foot with 3 cm requested whole-foot clearance,
then landing in place. Level 2 adds a signed 4 cm world-X placement before landing.
Both swing sides are trained. Later levels are conditional on the earlier level's
closed-loop admission; completing a level is not inferred from training reward.

The task scheduler advances only after measured progress. Its phases are WAIT,
TRANSFER, LIFT, PLACE, LAND and SETTLE. The learner receives explicit goal, phase,
dwell and time information. The scheduler issues no motor targets. Single-foot
support is allowed during swing; bilateral Stand applies to the final settling
window. A lift in place cannot count as a displaced step.

The objective uses phase-specific geometric/load progress, bounded potential
shaping, one-time measured phase bonuses, a final settled-success bonus and
finite failure penalties. The old capture-risk-gated push reward is not reused:
a quiet commanded step must receive a useful learning signal. Per-interval active
task cost is 0.01, phase completion bonus 1, first settled success bonus 5,
physical failure penalty 5 and incomplete deadline penalty 2. Record reward
components and achieved physical stages separately. Reward design is a hypothesis;
it is not evidence that useful movement has been learned.
Applied normalized joint additions also incur a mean-squared cost of 0.005.

Episodes last 12 seconds. Success may latch before the deadline, but the physical
rollout continues and final admission still requires survival and the final two
seconds of original Stand with each foot carrying at least 30 N. Episodic PPO
bootstraps zero at every terminal, including a successful deadline. Time remaining
is observable. Physical failure and an incomplete deadline are reported separately.

## Bounded learning

Two fresh seeds, 941009 and 951009, share initial actor seed 940009. Native CPU
physics uses sixteen worlds; only pose IDs 0-5 in each side's eight-world block
enter training. Entire poses 6/7 are reserved. Training resets randomize requested
side and command onset in 0.8-1.2 seconds; level 2 also randomizes signed placement.
The frozen source always starts from the original reference reset.

Each admitted level receives at most 64 collections * 64 intervals * 12 worlds =
49,152 transitions per seed. Three PPO epochs, minibatch 256, actor LR 1e-4,
critic LR 3e-4, gamma 0.995, GAE lambda 0.98, ratio clip 0.2 and exact joint
conditional-Gaussian KL ceiling 0.01 with actor/optimizer rollback. There is no
learning-rate sweep, run extension or periodic checkpoint series. Save initial
and final actors and the final critic per level reached. Maximum across three
levels and both seeds: 294,912 transitions.
An admitted next level starts from that seed's preceding final actor, with a
fresh critic and fresh actor/critic optimizers.

Advancement requires at least 9/12 training-pose successes, including at least
4/6 for each swing side, in BOTH seeds, and all sixteen sixty-second quiet cases
passing for each. This is a prospective intermediate curriculum threshold, not
certification of reliable Perturb recovery. Report reserved poses without using
them in normalization, optimization or the curriculum decision. If the gate
fails, stop the curriculum and preserve the failed level's evidence. Do not
automatically spend the budgets for later levels.

## Evaluation and evidence

Before fitting, check reset isolation, observation history, independent left/right
action mapping, zero-head source parity, delayed target composition, finite reward,
single-foot acceptance, measured-only phase transitions and terminal semantics.
Evaluate the zero head on the same commands as the learned candidates.

For each trained level evaluate both swing sides and all eight poses, commanded
at 1.0 seconds. Level 2 evaluates both displacement signs. After the nominal
gate passes, assess unseen 1.7-second command timing before moving to the next
level. Final step admission additionally uses the existing corrective_step_audit:
unload below 5 N, clearance at least 15 mm for 50 ms, opposite support at least
30 N, loaded landing for 100 ms, at least 30 mm world and relative displacement,
stance-foot drift at most 20 mm, then original settled Stand. The audit window
begins at command onset, explicitly labelled as a command rather than a ball hit.

Save compact native qpos/qvel, targets, corrections, loads, clearance, foot/COM
positions, phase/timer/goal evidence and physical/task failure masks. Independently
reconstruct foot geometry and outcome counts, source/zero-head identity, physical
bounds and update accounting. Keep inputs and used source hashes. Broader impact
evaluations and Arena changes remain conditional on actual learned placement.

## Following milestone

Once the commanded movement is reproducible, teach observable-state selection
of foot, placement and timing after modest disturbances. Then evaluate actual
projectiles across directions, contact locations, unseen timing and repeated hits,
with native/GPU/Arena parity before increasing toward 10 kg at 5 m/s. A mechanical
step does not by itself establish human-natural movement quality.

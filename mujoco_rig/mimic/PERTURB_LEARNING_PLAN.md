# Stand-to-Perturb continuation

**Second autonomous batch complete (2026-10-07): 20/20 substantive experiments.** The first focused-varied pair retains natural Stand for 120 seconds and improves primary, timing, admitted directional, and repeated-hit recovery. Independent optimization repeat fails to reproduce the gain on both source parents, so the candidate remains unpromoted. No training/evaluations running; original selected pair and Arena unchanged. Full results and accounting: `PERTURB_AUTONOMOUS_PLAN.md`.

Started: 2026-10-06 (Europe/Madrid).

**Follow-up 21 completed (2026-10-07):** 512 frozen cases at 5 kg / 1.4 m/s /
0.25 m, with no new learning. The selected pair passes **128/128 repeated-hit
trials** and **124/128 delayed-only controls** (original sources: 102/128 and
100/128). All contact/backend checks and the final review seal pass. One fixed
pose of parent 301027 falls after a single +X throw at 15 or 25 seconds, on both
backends, while it succeeds with earlier hits. Parent 301028 passes every new
condition. The remaining gap is launch-time/state coverage. Next documented
comparison: a 192-case onset sweep at 1/3/9 seconds for the fixed selected pair,
reusing Stage 13 source baselines; it is not executed yet. Detailed results:
[PERTURB_AUTONOMOUS_PLAN.md](PERTURB_AUTONOMOUS_PLAN.md), Follow-up 21.

**Autonomous Perturb batch completed 2026-10-07: 20/20 substantive stages.**
The selected pair is the original-reward, corrected-geometry impact continuation
at **200 PPO updates** from Stand parents 301027/301028. Each continuation uses
64 environments x 64 rollout steps = 4,096 transitions/update, or 819,200
transitions per arm. These are warm starts from verified Stand brains.

Both selected brains retain 120-second Stand (32/32 cases). Their controlled
follow-ups pass 64/64 target-dose recovery cases through 60 seconds, 256/256
directional cases with clear launch geometry, 192/192 one/two/three-hit cases,
and 128/128 delayed-only controls. These checks use eight fixed starts per
parent and both native/GPU backends; they do not establish arbitrary Arena
impact robustness. Repeated throws use one reused projectile, 5 kg at 1.3 m/s,
0.25 m distance, ten-second spacing and a 32-second horizon. Directional
coverage uses a distinct 0.46 m distance because shorter angled launches
penetrated the character; side contacts include an upper arm.

Training reproducibility remains incomplete: independent optimization seeds
produce target +X scores 6/8 and 7/8, versus sources 5/8 and 6/8, with retained
Stand; however, the second ball-trained brain ties its matched quiet control.
The selected 200-update pair scores 8/8 and 8/8, while its 300-update endpoints
regress to 5/8 and 7/8. More training is not a reliable improvement rule.
Active target PD is retained and passive springs remain zero.

The original sources lose repeated-hit recovery even when settled before the
next hit. Matched delayed-single controls and identical native pre-impact
prefixes distinguish earlier-impact state from launch-time difficulty. The
selected pair passes both sets of controls. The subsequent repeated **1.4 m/s**
comparison is complete as Follow-up 21 above. Arena's 10 kg / 4 m/s, random-target,
persistent-ball setup remains outside this controlled evidence.

Current verified checkpoint paths, complete protocols, results and limitations:
[PERTURB_AUTONOMOUS_PLAN.md](PERTURB_AUTONOMOUS_PLAN.md).
Final evidence review: `logs/mimickit-training/stand-ball-batch-final-review-20261007-01`.
Perturb Arena still selects the user-requested Stage 12 seed-301028 preview;
this continuation did not replace its selection or run a C# build.

**Pre-batch result (2026-10-06): the saved-trace audit examined 448 existing cases
without new simulation or learning.** Actions/targets do not clip before balance
departure; all 108 failed +X cases first exceed horizontal speed. Torque limits
are not reached during their first second after contact. Changing contact
impulses, particularly later contact from the same ball, accompany several
success/failure reversals. Next: a bounded frozen-policy comparison that removes
the projectile after its first contact cluster ends, to isolate later-contact
effects before changing training or control. That intervention is now complete
as Stage 1 of the completed batch; detailed historical findings follow below.

**Historical pre-batch intermediate-checkpoint assessment:**
256/256 physical cases with zero learning updates. Earlier checkpoints recover
more cases than the final endpoint, but seed 301028's ball and quiet arms tie
at every sampled update. No checkpoint shows replicated benefit from ball
practice. All evidence checks pass; neither checkpoint qualifies for conditional
Stand/easy-impact assessment. The subsequent saved-trace diagnostic and selected
next intervention are recorded below.

**Historical pre-batch physical-ball pilot:** all four arms and 480 evaluations.
Final 120-second Stand passes 64/64. At 5 kg / 1 m/s,
seed 301027 recovers 9/16 before continuation, 8/16 after quiet continuation and
11/16 after ball practice; seed 301028 scores 14/16 in all three conditions.
These counts agree on native and GPU. The evidence checks pass, but improvement
does not replicate, so no brain is promoted. The intermediate assessment below
extends this evidence without replacing the failed final-endpoint criterion.

**Earlier pulse pilot: all four arms and 896 evaluations completed; no promotion.**
Quiet 120-second Stand is retained in all 64 final native/GPU cases. Recovery
improves beyond quiet continuation on only one of two seeds, so the predefined
replication rule fails. An initial-state fingerprint guard also failed and
remains explicitly recorded; the reset diagnostic below identifies a numerical
mechanism without reconstructing the unsaved original start components.

The user explicitly authorized "Run the pilot and its evaluations." Output root:
`logs/mimickit-training/perturb-continuation-20261006-01`.
The first launch failed during import, before output creation or simulation:
`amp_recovery` was imported before the MimicKit runtime path was activated.
Its import now occurs after runtime activation; the corrected launcher started.
No export, C# build or Arena launch ran during the pilot. The subsequent
user-requested inspection preview is recorded below. The authoritative combined report is
`completion/comparison/report.json` under that root. All four arm evidence
seals and the combined comparison seal passed; this verifies recorded evidence,
not the failed performance or initial-state admission criteria.

The preceding twenty substantive experiments are complete. Their results and
failures remain in [STAND_LEARNING_PLAN.md](STAND_LEARNING_PLAN.md); this protocol
does not reopen that ledger or count implementation work as a training result.

## Admission and question

Quiet Stand is established on the original eight starts, with natural-posture
and balance gates checked throughout 120 seconds on native and GPU backends.
Seeds 301027 and 301028 supply independently trained quiet source brains:

- `logs/mimickit-training/stand-push-learning-301027-quiet-20261006-01/updates-1200/model.pt`
- `logs/mimickit-training/stand-push-learning-301028-quiet-20261006-01/updates-1200/model.pt`

Both include full `.amp.pt` state. Their milestone-16 assessments establish
120-second admission. Each arm will also reassess its source before learning;
any quiet admission failure stops that arm. The untouched source files remain
protected inputs, and each arm saves an identical initial full/portable pair.

The earlier fresh 10-to-180 N curriculum did not replicate an improvement.
This experiment asks whether an already competent Stand brain benefits from
small, isolated forward/backward recovery practice. It does not train a new
brain from scratch or resume the rejected curriculum.

## Frozen foundation and reward audit

Keep the 374-observation, 33-action target-PD representation, action delay,
torque limits, contacts, body and reference poses. Passive joint stiffness
remains zero. Active local joint PD remains enabled. No feet are anchored.

The active `reference_v9` task reward in `torque_stand_task.py` is
`0.5*pose + 0.1*joint_velocity + 0.15*root_pose + 0.1*root_velocity + 0.15*key_positions`.
AMP combines 80% task reward with 20% discriminator reward. Root-relative
keypoints and soft velocity/pose tracking prefer the standing reference;
they can make recovery motion costly. However, that is a hypothesis about
the learning limitation, not an established cause of the failures. The current
mode has no hard foot anchors, quiet-speed multiplier, capture-point term,
action-cost term, or posture-mismatch training termination. Physical
fall/contact and nonfinite-state termination remain active.

The first pilot changes force exposure only within each pair. Reward redesign
would be a separate experiment after inspecting the new saved failures.

## First pilot: two matched continuation pairs

The executable protocol is [perturb_pilot.json](perturb_pilot.json). Launch paths
are in [scripts/perturb-pilot-config.ps1](scripts/perturb-pilot-config.ps1).
The legacy `scripts/config.ps1` and its Perturb presets are not used.

- Seeds: 301027 and 301028. For each, quiet continuation and Perturb continuation
  load the same actor, critic, discriminator and learned normalization tensors.
  Optimizer momentum, rollout/discriminator replay buffers and counters start
  fresh in both arms: this is a warm start, not an exact optimizer resume.
- PPO: 64 environments x 64 control intervals = 4,096 transitions/update.
  Inherited actor learning rate 3e-4, three epochs, effective minibatch 256;
  48 actor optimizer steps/update. Critic: two epochs, effective minibatch 128;
  64 optimizer steps/update. Remaining AMP/PPO settings are inherited unchanged.
- Budget: 300 additional updates = 1,228,800 transitions per arm. Four arms
  total 1,200 updates / 4,915,200 transitions if all complete. A 600-second
  synchronized training cap per arm stops an incomplete experiment; it does
  not silently reduce the comparison budget or admit an over-cap endpoint.
  Setup, checkpointing, assessments and evidence checks are outside this
  training timer, so 40 training minutes is not a wall-clock estimate.
- Both arms use 12-second episodes instead of the source's 60-second training
  episodes. This shared change increases exposure frequency. Only applied
  force differs within a pair; source-to-final differences alone cannot
  isolate the force effect, which is why quiet continuations are required.
- Perturb arm: a single world-X Chest pulse, 60 N for 0.1 seconds starting at
  episode second 5. First 16 worlds receive +X, next 16 receive -X, remaining
  32 stay quiet. Each sign contains two copies of all eight starts. Forces
  are held through the four physics substeps and cleared outside the pulse.
  Onset and duration round up to control frames: at dt=0.016668, six pulse
  frames last 0.100008 seconds, giving impulse 6.00048 N s.
- Quiet arm: identical task with zero external force. Counterbalanced order:
  seed 301027 quiet then Perturb; seed 301028 Perturb then quiet. Each arm runs
  in its own sequential process. Sampling seeds match within each pair;
  this does not guarantee identical GPU trajectories after learning diverges.
- No escalation or best-checkpoint selection. The final 300-update endpoint
  is the candidate. Updates 100/200/300 also save recovery checkpoint pairs
  for diagnosis if execution fails; they are not promotion candidates.

`EvaluationInterval` from legacy config is irrelevant here. `steps_per_iter=64`
sets rollout length. Behavioral assessments occur at initialization and the
final endpoint. Increasing rollout length would change PPO's batch and update
frequency; it would not merely reduce evaluation overhead.

## Assessments and decision

At initialization and the final endpoint of every arm, use frozen actor copies
and separate simulation worlds. Preserve learner tensors, optimizer state,
training environment and RNG across assessment. Record and verify full traces.

Each endpoint contains 112 cases: 16 quiet cases lasting 120 seconds, and 96
12-second cases covering 30/60/90 N x +X/-X x eight starts x two backends.
The complete pilot has 896 physical assessment cases. The 30 and 90 N levels
are absent from training; this remains narrow force generalization over the
original starts, body, onset and directions.

Quiet Stand must pass every pre/post control-frame balance and natural-posture
gate. Push recovery permits temporary motion and foot unloading, but requires
valid posture before the disturbance, survival, a two-second valid interval
beginning within two seconds after the pulse (one control-frame tolerance),
and a valid final two seconds. Merely surviving or preserving the reset pose
does not count as recovery.

To qualify for a separately bounded next stage, require **all** of the following
for **both seeds on both backends**, without selecting the better seed:

1. Repeated source recovery cases agree across the two matched arms.
2. Both final arms retain 8/8 full-interval quiet 120-second Stand.
3. Perturb recovery counts, summed over +X/-X separately at each force level,
   are no lower than either the untouched source or quiet continuation.
4. Combined 60/90 N recovery count is strictly higher than both comparators.

Report all per-start/direction outcomes even when the count gate passes;
aggregate counts can conceal individual regressions. A ceiling tie is not
evidence of improvement. A failure, cap, nonfinite value, provenance mismatch,
or unstable repeated baseline prevents promotion. Inspect preserved traces
and revise one factor in a successor protocol; do not automatically extend
training or increase force.

## Next stages and Arena

After replicated pilot improvement, separately define an increase in force or
timing/direction diversity, retain quiet controls, and repeat the same retention
gate. Repeated pushes and physical ball impacts require their own assessments.
No stronger curriculum is scheduled by this launcher.

The legacy Perturb Arena uses an older direct-torque brain, a different
observation contract and ball-training path. It is not a valid viewer for these
new continuation checkpoints. A later physical-ball integration must preserve
the 374/33 representation, append only projectile physics, record actual
contacts, and validate export/runtime behavior before selecting an Arena bundle.
The currently selected Stand brain remains seed 301026. This pilot does not
change scenes, viewer selections, production config or sealed research sources.

## Execution and progress

Command used after explicit authorization to run training **and its assessments**
(the output now exists and must not be overwritten):

```powershell
& .\mujoco_rig\mimic\scripts\train-perturb-pilot.ps1 `
    -Out logs/mimickit-training/perturb-continuation-20261006-01
```

Output must not exist. The launcher refuses another detected Mimic experiment.
Incomplete runs remain preserved; there is no automatic retry or overwrite.
Each arm writes `progress.json`, `updates.jsonl`, checkpoint pairs, force/reward
records, assessment traces and a provenance seal. The parent contains per-arm
console logs and progress; `comparison/report.json` records the decision.
Completion is evidence verification, not proof the candidate learned recovery.

- Completed: source/reward/legacy compatibility inspection; separate pilot
  protocol, warm-start runner, matched comparison launcher and documentation.
- Completed: all four 300-update arms, 896 scheduled assessments, reset-only
  diagnosis and the combined evidence report.
- Pending and gated: stronger disturbances, ball transfer and Arena promotion.
- Result: replication rule failed; initial-state guard failure preserved;
  no checkpoint selected for promotion and Arena unchanged.

### Completed results: seed 301027

The first arm completed 300 updates / 1,228,800 transitions in 392.972318
synchronized training seconds. Both backends retain 8/8 full-interval
120-second Stand. Recovery remains 8/8 for both signs at 30 and 60 N.
Forward 90 N recovery improves from 1/8 to 4/8 on each backend; backward
90 N remains 8/8. The arm's evidence seal passed. This improvement occurred
without training pushes and is the matched control, not a Perturb result.
The perturbation arm also completed 300 updates, in 397.866053 synchronized
training seconds. It retained 16/16 quiet Stand and the same aggregate recovery
counts as the quiet control: 8/8 at 30/60 N in both signs, 4/8 forward and 8/8
backward at 90 N on each backend. It did not beat quiet continuation. The
equal forward counts conceal a trade: the quiet control recovers starts
0/3/4/5, while Perturb recovers 0/4/5/7. Start identifiers are zero-based.

### Initial-state guard failure and bounded completion

The original orchestrator stopped after the first pair because its
`training-start.json` equality assertion failed. Starting learner, optimizer
and RNG fingerprints match; the environment fingerprint differs. Both arms
completed and passed their individual force/reward/trajectory/provenance checks.
The failed comparison assertion and original `failure.json` remain preserved.

`reset-diagnosis` made three resets in each of two fresh 64-world task instances,
with zero PPO updates, policy steps or physics integration steps. It captured
the raw components missing from the original training-start evidence. All poses,
velocities, delayed commands, controls, counters and reference offsets are
bit-identical; only observation columns 93/94 (foot loads scaled by 0.001) vary.
The maximum difference across this probe is 1.6391277313232422e-7 observation
units, equivalent to 0.00016391277313232422 N. Even repeated resets of the same
quiet instance change these low bits. The reset diagnostic passed its seal.
It demonstrates a repeatable numerical mechanism but cannot reconstruct the
unsaved initial components of the original training arms.

`finish_perturb_pilot.py` executed only the two remaining authorized
seed-301028 arms, in the original Perturb-then-quiet order, with the unchanged
sealed trainer and original budgets. Output is `completion` under the original
pilot root. No first-seed arm was repeated, and no extra learning was added.
The completion report keeps the original admission gate failed and prohibits
promotion. It reports all four arm outcomes descriptively without silently
waiving the failed comparison guard. Both seeds have matching learner,
optimizer and RNG fingerprints, differing environment fingerprints, and
identical repeated baseline recovery records within each backend.

### Completed results: seed 301028

Perturb completed 300 updates in 401.792040 training seconds; quiet continuation
completed 300 in 391.077904 seconds. Both final actors retain 16/16 continuous
120-second Stand. Both recover all 30/60 N cases and all backward 90 N cases.
At 60 N, the source initially recovered 15/16 sign/start cases per backend;
both continuations improve that to 16/16. An earlier live progress message
overstated that initial 60 N result as all passes; these saved counts govern.

For forward 90 N, the source and quiet continuation recover 6/8 on each backend,
while Perturb recovers 7/8. Quiet recovers starts 0/1/2/3/4/5; Perturb adds start 7.
The behavioral improvement rule passes for this seed on both backends, but
fails for seed 301027, so there is no replicated improvement across the pair.

### Pilot conclusion and next work

Actual new training: **1,200 updates / 4,915,200 transitions /
1,583.708315 synchronized training seconds (26.4 minutes)**. This excludes setup,
evaluations and evidence checks. All four arms completed the declared equal
update budget within their individual 600-second training caps. There were
**896 scheduled evaluation cases**, plus the separate reset-only probe with
zero policy or physics integration steps. The import failure added no cases
or training. No historical source checkpoint or sealed scientific module changed.

**Keep the current Stand selection; do not escalate or promote this pilot.**
Standing survived continuation reliably on the tested starts. Small-push
practice produced a modest extra recovery gain for one seed and an aggregate
tie with a case tradeoff for the other. More training or stronger forces are
not automatically authorized or justified by this result.

Before the next comparison, save raw training-start components and predeclare
a narrow numeric tolerance specifically for GPU foot-load observations, while
retaining exact checks on poses, velocities, delayed controls, counters,
learner, optimizers and RNG. Check the pair before spending its training budget.
The old failed gate must remain in this record. Then define a separate recovery
intervention around the remaining 90 N forward failures; changing the reward,
force distribution or episode schedule requires its own controlled comparison.

Evidence: [combined report](../../logs/mimickit-training/perturb-continuation-20261006-01/completion/comparison/report.json),
[comparison seal](../../logs/mimickit-training/perturb-continuation-20261006-01/completion/comparison/final-verification.json),
[reset diagnosis](../../logs/mimickit-training/perturb-continuation-20261006-01/reset-diagnosis/report.json).

## User-requested Perturb Arena inspection (2026-10-06)

The user requested loading the brain into Perturb Arena for inspection.
Selected seed 301028's final Perturb checkpoint `87bdc325`, from
`completion/301028-perturb/final/model.pt`, in the separate bundle
`logs/mimickit-training/perturb-arena-301028-20261006-01`.
This is an inspection selection, not promotion under the failed pilot rule.

The existing compiled adapter already supports the 374-observation/33-target
contract. Export conversion agrees with the saved actor within 2.086163e-7
on 2,055 saved observations; no new behavioral rollouts or C# build were run.
The existing projectile world was copied unchanged after checking its character
hash against the training model. Active local PD and zero passive stiffness
are preserved. The Godot log confirms the selected bundle loaded successfully.

Perturb Arena retains 10 kg balls at 4 m/s and its independent projectile pool.
Automatic throws start disabled: B launches, R restarts and clears balls, P
pauses. The scene identifies the 60 N pulse training separately from this
physical-ball preview. Ball-impact recovery remains unvalidated. Stand Arena
continues to select seed 301026; the prior Perturb scene and viewer selection
are backed up in the new bundle.

## Ball-recovery boundary preparation (2026-10-06)

The user reports falls after one or two Arena hits. The pulse pilot does not
establish ball recovery: its 60 N force lasts about 0.1 seconds, whereas the
Arena projectile has 40 kg m/s of horizontal momentum before collision.
Projectile momentum is not the impulse actually transferred to the dummy.
Repeated hits, random target bodies and uncleared projectiles add further
differences. More pulse-training time alone is not the next intervention.

**Initial preparation (execution results follow below):** `stand_ball_boundary.json`,
`evaluate_stand_ball_boundary.py`, `stand_ball_native.py`, and
`scripts/evaluate-stand-ball-boundary.ps1`. The first executable stage performs
zero training updates. It measures frozen final Perturb actors from seeds
301027 and 301028, including the brain currently selected in Perturb Arena.
Using these actors for diagnosis does not waive the old failed admission gate.

The bounded native screen contains 400 cases: two actors, eight starting poses,
12 mass/speed combinations, both horizontal X directions, plus one parked-ball
quiet control per actor. Each case lasts up to 12 seconds; launch is at 5 seconds.
Masses are 1/2/5/10 kg and nominal horizontal speeds are 1/2/4 m/s. The existing
ballistic launcher compensates gravity at the launch-time chest target from
0.65 m away; full launch velocity is saved and differs from horizontal speed.
The ball radius and contact properties come from the current Arena template.

The character XML must match after removing the appended ball/keyframe metadata.
The controller remains 33-target delayed PD, 374 observations and zero passive
stiffness. Projectile state, ground contacts and mass are excluded from character
observations, balance gates and center of mass. Every physics substep contributes
actual ball-to-character contact impulses to the saved trace. A missed shot is
never a recovery pass. Recovery requires survival, valid standing before contact,
two continuously valid seconds beginning within two seconds after the final
contact (one control-frame tolerance), and a valid final two seconds.

The parked-ball control must first retain all eight poses. Saved evidence checks
include control timing/formula, balance/posture gates, actor immutability,
physical state continuity, solver warnings and protected source hashes. The
screen proposes a boundary only when all 16 shots hit and recovery contains
both successes and failures for each seed. Among eligible levels, select the
lowest nominal horizontal momentum, then horizontal kinetic energy. If none
qualifies, revise the screen from measured misses/failures; do not escalate
automatically. A native candidate is not GPU admission or a training result.

`stand_start_guard.py` supplies raw snapshot capture and comparison for the next
trainer. It checks learner, optimizer, RNG and training mode fingerprints
exactly, and checks raw pose, velocity, delayed commands, controls and counters
bitwise. Only observation columns 93/94 allow absolute tolerance 1e-6 (0.001 N),
with zero relative tolerance. All other observation columns remain bitwise exact.
The old sealed trainer is unchanged; integrating this helper into the successor
ball trainer and checking both actual starts before PPO remain required work.

The future training design is recorded separately in the JSON and is not executed
by the assessment launcher: 64 environments, 64 rollout steps, 300 updates per arm,
600-second training cap, half quiet practice, matched quiet/ball continuations,
unchanged reference reward and PPO settings. First admit the GPU ball adapter and
the matched starts. Then compare single-hit learning on both seeds, preserving
120-second Stand on both backends. Require gains over source and quiet continuation
without easier-hit regressions. Only then assess two hits six seconds apart,
followed by separate four- and three-second interval stages with sufficient
post-contact recovery time. Random body targets and Arena's persistent ball pool
remain later held-out checks.

Execution entry point, from the project root:

```powershell
& ./mujoco_rig/mimic/scripts/evaluate-stand-ball-boundary.ps1 -Out logs/mimickit-training/stand-ball-boundary-20261006-01
```

No tests, compilation, training or behavioral evaluations were run during this
preparation. The user's global test/compile constraint requires explicit execution
authorization; `Proceed` authorized these edits only. Arena selections are unchanged.

## Single-ball boundary results (2026-10-06)

The user answered `Proceed` to the explicit request to run the 400-case ball
assessment. The completed output is
`logs/mimickit-training/stand-ball-boundary-20261006-02`.
All 400 scheduled native cases completed: 16 quiet controls and 384 impact cases.
All 384 throws made physical contact and every impact case was valid before
contact. Both frozen actors passed all eight 12-second quiet controls. This
screen did not repeat the historical 120-second Stand evaluation.

Seed 301027 recovered 113/192 impacts; seed 301028 recovered 117/192.
Every recovery failure in this screen was also a survival failure. Per-seed
recovery counts below pool the eight starts in each of the two X directions:

- 1 kg at 1, 2 or 4 m/s: 16/16 for both seeds at every speed.
- 2 kg at 1 or 2 m/s: 16/16 for both seeds at each speed.
- 2 kg at 4 m/s: 8/16 and 10/16. The +X counts are 0/8 and 2/8;
  both actors recover all eight -X hits.
- 5 kg at 1 m/s: 9/16 and 14/16. The +X counts are 1/8 and 6/8;
  both actors recover all eight -X hits.
- 5 kg at 2 m/s: 8/16 and 5/16. Both +X groups fail all eight;
  -X recovery differs between actors.
- 5 kg at 4 m/s: 0/16 for both seeds.
- 10 kg at 1 m/s: 8/16 for both seeds, entirely from -X recovery.
- 10 kg at 2 or 4 m/s: 0/16 for both seeds at each speed.

**The predeclared selection rule chooses 5 kg at 1 m/s.** It is the lowest
nominal horizontal-momentum level with actual hits and a mixture of successes
and failures for both actors. This is a candidate for a controlled training
comparison, not a demonstrated learning improvement. Ballistic vertical velocity
and measured contact impulses matter: mass times horizontal speed alone does
not order the actual difficulty of all these impacts.

At Arena's 10 kg / 4 m/s setting, even one fixed chest hit causes failure in all
32 combined cases. This reproduces insufficient impact recovery under a simpler
condition than Arena's random targets and repeated throws. It supports training
gentler physical impacts first, rather than adding more time at the old 60 N
pulse recipe or immediately training repeated heavy hits. Active PD and passive
stiffness remain unchanged.

The first output folder, `stand-ball-boundary-20261006-01`, remains preserved.
It contains eight successful quiet controls and an interrupted eight-world
trial that stopped at its first launch, before any impact, because the launcher
received one shared velocity vector where it expected one vector per world.
The isolated adapter now broadcasts finite launch vectors to the selected worlds.
The failed attempt's source snapshots and failure record remain intact. The
completed successor reran the full declared screen; those preliminary controls
and interrupted worlds are not included in its 400 accepted cases.

The process exited successfully. Its verification seal confirms 11,350 protected
inputs and 96 sources unchanged, plus the saved trajectory/control/gate checks.
No evaluation process remains. Training updates: **zero**. No brain was promoted,
and no Arena selection or C# assembly changed.

Next: implement and admit the GPU ball adapter at the selected level, integrate
the raw matched-start guard before either arm spends PPO updates, then run a
separately authorized two-seed quiet-versus-single-ball pilot with 50% quiet
practice and 120-second Stand retention checks. Repeated hits remain gated on
replicated single-hit improvement.

Evidence: [completed report](../../logs/mimickit-training/stand-ball-boundary-20261006-02/report.json),
[verification seal](../../logs/mimickit-training/stand-ball-boundary-20261006-02/final-verification.json),
[preserved launch failure](../../logs/mimickit-training/stand-ball-boundary-20261006-01/failure.json).

## GPU ball admission implementation (2026-10-06)

The next implementation is isolated in new modules. The sealed native screen,
its ball adapter, configuration and evidence remain unchanged.

- `stand_ball_gpu.py` extends the admitted 33-target GPU controller with one
  appended ball. It retains the pinned Newton substep order and reads solved
  MuJoCo-Warp contact forces at every substep. It records summed positive normal
  impulse and signed world-frame impulse from ball/character contacts only.
  Projectile state and ground contacts remain outside the character observation,
  reward, center-of-mass and termination views.
- `stand_ball_task.py` prepares the evaluation task and the future AMP task.
  Both training arms have the same 5 kg ball world, initial state and reference
  reward. At five seconds, the single-ball arm throws at the chest in 16 +X and
  16 -X worlds; 32 worlds remain quiet. Each direction covers all eight poses
  twice. Speed is 1 m/s horizontally, distance 0.65 m, with the same ballistic
  gravity compensation. Reset reparks the ball. Quiet continuation never throws.
- `evaluate_stand_ball_gpu.py` saves GPU trajectories, contact impulses and
  substep control evidence, and reuses the sealed native recovery definition.
  Terminated worlds' later contacts do not count. GPU warning counters are not
  available through this backend; the report states that explicitly and checks
  finite trajectories instead of claiming a native warning check.
- `prepare_stand_ball_pair.py` constructs both live learners before either may
  train. It uses identical full AMP weights and learned normalizers, fresh
  optimizers and matched random streams. The existing raw guard permits only
  the predeclared foot-load tolerance. Additional exact checks cover ball state,
  AMP history, previous COM/action, action cost, native controls and impulses.
  Both live states are rechecked after constructing the pair. A failure raises
  before the helper returns either learner to a future training loop.

The executable admission stage is `scripts/admit-stand-ball-gpu.ps1`:

```powershell
& ./mujoco_rig/mimic/scripts/admit-stand-ball-gpu.ps1 -Out logs/mimickit-training/stand-ball-gpu-admission-20261006-01
```

Its bounded work is **80 GPU evaluation cases, zero PPO updates**: each of two
frozen brains has eight quiet controls, 16 easy hits at 2 kg / 2 m/s and 16 hits
at the selected 5 kg / 1 m/s level. Results are compared case-by-case against the
already saved native screen. Admission requires identical recovery, survival,
hit and pre-contact validity outcomes; first/last contact may differ by at most
two control frames, and total normal impulse by 0.25 N s plus 10% of the native
value. These tolerances are declared before running. A mismatch preserves the
evidence and stops admission rather than relaxing a threshold automatically.

Before those cases, an eight-world isolation probe takes three zero-action
control intervals, then launches and resets worlds 0 and 4. It checks that peer
states, observations and delayed commands remain unchanged, while selected
worlds return to the reference with cleared controls and a parked, stopped ball.
After behavioral admission, both seeds undergo the 64-world matched-start probe
for both arms, with no explicit physics or learning steps in that probe.

The admission runner does not train, export, change Arena or persist resumable
environments. A future training runner must call `prepare_pair` again on its
actual live training tasks and restore each returned RNG stream when switching
arms; this probe cannot substitute for admission of later recreated states.
The 300-update matched trainer, isolated 120-second retention evaluations and
training comparisons remain the next implementation after GPU admission.

At preparation time, these modules had not been executed or compiled. The user
subsequently answered `Proceed` to the explicit request to run GPU admission;
the completed results follow.

## GPU ball admission results (2026-10-06)

**Passed on the first run**, without code repairs or changed tolerances.
Output: `logs/mimickit-training/stand-ball-gpu-admission-20261006-01`.
The process exited successfully and no admission process remains.

All **80/80 GPU cases match the saved native outcomes case by case**, including
survival, actual hit, pre-contact validity and recovery. The 16 quiet controls
pass; all 32 easy hits at 2 kg / 2 m/s recover. At 5 kg / 1 m/s, seeds 301027
and 301028 reproduce 9/16 and 14/16 recovery respectively: +X passes 1/8 and
6/8, while both -X groups pass 8/8. These are frozen-policy reproductions,
not new learning gains. The quiet controls last 12 seconds, not 120 seconds.

Contact timing passes the two-control-frame tolerance in every case. Maximum
total normal-impulse difference is **0.03582602764745424 N s**, within the
predeclared 0.25 N s plus 10% tolerance. Every saved GPU control trace passes
the delayed-target/torque formula checks and the balance/posture recomputation.

The separate eight-world, three-control-interval isolation probe passes every
check. Launching/resetting worlds 0 and 4 leaves peer physical state, projectile
state, observations and queued controls unchanged. Selected worlds reset to
their reference with cleared commands and parked, stopped balls.

Both 64-world matched-arm pairs pass the raw start guard before any PPO update.
Learner, fresh optimizer, RNG and training-mode fingerprints match exactly.
Pose, velocity, delayed commands, control, counters, projectile state, AMP
history, previous COM/action and impulse buffers match exactly. Rechecking each
live task after constructing its partner also passes. Maximum observation
differences are 1.341104507446289e-7 and 1.6391277313232422e-7 for seeds 301027
and 301028, confined to the permitted foot-load columns and below 1e-6.
The original pulse pilot's failed comparison remains unchanged in its own record.

The verification seal confirms **11,613 protected inputs and 103 sources
unchanged**, GPU ball admission and matched-start probe success. Training
updates: **zero**. No brain promotion, Arena change or C# build occurred.

**Next implementation:** the bounded two-seed quiet-versus-single-ball trainer
using these admitted modules. Use the selected 5 kg / 1 m/s level, 50% quiet
worlds, unchanged PPO/reference reward, 300 updates per arm and a 600-second
training cap per arm. Call `prepare_pair` again on the actual training instances;
this successful probe does not authorize bypassing live admission. Evaluate
initial/final recovery in isolated worlds and retain 120-second Stand on both
backends. The training runner and its execution remain pending; repeated-hit
training still requires replicated single-hit improvement.

Evidence: [GPU admission report](../../logs/mimickit-training/stand-ball-gpu-admission-20261006-01/report.json),
[verification seal](../../logs/mimickit-training/stand-ball-gpu-admission-20261006-01/final-verification.json),
[reset isolation](../../logs/mimickit-training/stand-ball-gpu-admission-20261006-01/reset-probe/report.json),
[seed 301027 starts](../../logs/mimickit-training/stand-ball-gpu-admission-20261006-01/301027-starts/report.json),
[seed 301028 starts](../../logs/mimickit-training/stand-ball-gpu-admission-20261006-01/301028-starts/report.json).

## Matched ball-training pilot protocol (2026-10-06; executed)

Prepared `run_stand_ball_pilot.py`, `stand_ball_training_evidence.py`,
`stand_ball_pilot.json` and `scripts/train-stand-ball-pilot.ps1`. The admitted
physics, tasks, evaluators, matched-start helper and all sealed historical
sources remain unchanged. The new recorder wraps calls to the existing task
and launcher, records their actual behavior, and restores those methods when
the training block exits; it does not replace rewards or the throw schedule.

The fixed budget is four arms: quiet and single-ball continuation for each of
seeds 301027 and 301028, warm-started from the same respective final pulse-pilot
parent used by ball admission. Each arm receives 300 PPO updates at 64 worlds
by 64 rollout steps, for **1,200 updates / 4,915,200 transitions total**. Fresh
optimizers and inherited weights/normalizers are used in both arms. PPO and AMP
configuration are inherited unchanged. The first seed runs quiet then ball;
the second runs ball then quiet. The single-ball arm has 32 quiet worlds and
16 worlds for each X direction, using 5 kg / 1 m/s chest throws at five seconds
in 12-second episodes. Passive springs and active delayed PD are unchanged.

Each arm has a 600-second CUDA-synchronized training cap. Timing includes PPO
and live evidence capture, but excludes evidence serialization, checkpoints,
setup and evaluation. A cap overrun or incomplete arm stops the run and leaves
failure/progress/checkpoints for inspection; budgets are not silently extended.
Thus 40 minutes is the maximum combined declared training budget, not a wall-time
estimate for the complete run.

Before constructing either live arm, each seed receives one shared frozen-source
assessment. The source must again retain 120-second Stand on both backends before
its pair may train. The runner then calls the admitted `prepare_pair` on the
actual live tasks and binds their actor hashes to the shared source assessment.
Immediately before each arm starts, its random stream is restored and its live
raw state is compared again with its paired admission, including the waiting arm
after its partner has trained. Neither a prior probe nor an approximate learner
fingerprint substitutes for this check.

Per-update compressed evidence records episode steps, launch masks, actual launch
position/velocity errors, contact impulses, task rewards and termination flags.
Checks require the declared throw schedule, finite values, correct launch state,
and zero projectile contacts in quiet worlds. The ball arm must actually launch
and receive impacts. Transition and optimizer counters must remain at 4,096
transitions, 48 actor steps and 64 critic steps per update. Checkpoints are saved
at updates 100/200/300 plus explicit initial/final actor/full-AMP pairs.

The complete run schedules **480 evaluation cases**: two shared source assessments
and four final assessments, each with 16 continuous 120-second Stand cases plus
64 physical-ball cases. Impact cases cover eight poses, both X directions,
2 kg / 2 m/s and 5 kg / 1 m/s, on native and GPU. Frozen actor copies run in separate
worlds. Final assessments must preserve both live learners' model/optimizer,
environment, projectile/history and random-stream fingerprints exactly.

Advancement requires both seeds and both backends to retain 120-second Stand,
all evaluation throws to hit, no recovery-count regression in any tested
direction/level relative to either source or quiet continuation, and strictly
more total 5 kg / 1 m/s recoveries than both comparators. Per-case outcomes remain
saved so aggregate counts can be inspected for trades between starts. Failure of
the behavioral rule is a valid negative result, not an evidence-check failure.
Even a positive result does not automatically select a brain in Arena.

Execution entry point:

```powershell
& ./mujoco_rig/mimic/scripts/train-stand-ball-pilot.ps1 -Out logs/mimickit-training/stand-ball-pilot-20261006-01
```

The user answered `Proceed` to the explicit request to run the pilot and all
evaluations. Execution completed successfully in
`logs/mimickit-training/stand-ball-pilot-20261006-01`; the process has exited.

## Matched ball-training pilot results (2026-10-06)

All four arms completed their declared 300 updates and 1,228,800 transitions:
**1,200 PPO updates, 4,915,200 transitions and 480 evaluation cases** in total.
Combined synchronized training time was **1,673.889325900098 seconds (27.90 min)**,
excluding setup, evaluation, serialization and checkpoints. Every arm remained
under its 600-second cap; no extension or restart was needed.

- Seed 301027 quiet: 408.42638850002913 training seconds, zero throws/contacts.
- Seed 301027 single ball: 428.34625129998676 seconds, 914 throws and 3,216
  control intervals with a world receiving projectile contact.
- Seed 301028 single ball: 428.211807600077 seconds, 886 throws and 3,076
  control intervals with a world receiving projectile contact.
- Seed 301028 quiet: 408.904878500005 seconds, zero throws/contacts.

The contact counts above count world/interval pairs, not distinct hits or
episodes. Training retained 50% quiet worlds in each ball arm. Every arm
performed 14,400 actor optimizer steps and 19,200 critic optimizer steps.

Both repeated sources pass all 32 native/GPU 120-second Stand cases. All four
final brains pass all **64/64** corresponding Stand cases, with every sampled
pre/post control-frame balance and natural-posture gate valid throughout.
All **384/384 evaluation throws physically hit**. All six assessed actors pass
every 2 kg / 2 m/s recovery case in both directions and backends (192/192).

At the selected **5 kg / 1 m/s** level, counts below are per backend; native
and GPU agree exactly:

- **Seed 301027:** source 9/16 (+X 1/8, -X 8/8); quiet continuation 8/16
  (+X 0/8, -X 8/8); ball continuation 11/16 (+X 3/8, -X 8/8).
  Ball practice retains source success at zero-based pose 5 and adds poses
  3 and 4. This seed passes its predefined improvement rule.
- **Seed 301028:** source, quiet continuation and ball continuation each
  recover 14/16 (+X 6/8, -X 8/8). Poses 6 and 7 still fail the +X hit.
  There is no strict gain over either comparator, so this seed fails the rule.

Read-only inspection of the saved native traces for seed 301028 confirms
actual falls in both remaining cases. The first post-step gate departure is
horizontal speed: source 5.717/5.700 seconds versus ball-trained 6.834/6.567
seconds for poses 6/7. Nevertheless, survival ends at 7.534/7.667 seconds for
the source versus 7.634/7.451 for the ball-trained brain. Delayed departure
does not establish successful recovery or a consistent survival improvement.
The traces identify a failure sequence, not its causal explanation.

All four actual training-start guards and extra-state comparisons passed.
All four final evaluations preserved both live learners/environments exactly
and restored their random streams. The final seal verifies **11,786 protected
inputs and 106 source files** unchanged. `final-verification.json` records
`verified=true`, `replicated_improvement=false`, and `promoted=false`.
This successor's passing checks do not erase the earlier pulse pilot's failed
initial-state admission.

Authoritative evidence under the output root:

- `report.json`: combined counts, timing and predefined comparison result.
- `final-verification.json`: protected-input/source checks and output hashes.
- `<seed>/report.json`: source, both endpoints and per-backend comparisons.
- `<seed>/<arm>/start-admission.json` and `evaluation-isolation.json`:
  actual live starting state and isolation evidence.
- `<seed>/<arm>/rollouts`, `updates.jsonl`, `training-report.json`, and
  `evaluation`: persisted training and assessment evidence.
- `<seed>/<arm>/updates-100`, `updates-200`, and `final`: saved actor/full-AMP
  checkpoint pairs for a bounded successor assessment.

**Decision:** partial learning on one seed; no replicated benefit from this
300-update recipe. Preserve the established Stand foundation. No Arena bundle,
passive spring, controller, or production configuration changed in this run.
This result does not establish repeated-hit recovery or resistance to the
Arena's 10 kg / 4 m/s balls.

**Subsequent assessment, now completed below:** compare the saved 100/200-update checkpoints
with the completed endpoint using the same two seeds, matched quiet controls,
5 kg / 1 m/s cases and both backends. This requires no new PPO training and can
distinguish a temporary recovery gain from a flat or regressing trajectory.
Declare the assessment budget and interpretation before execution; treat any
checkpoint selected after this comparison as exploratory. A candidate must
also retain 120-second Stand and easy-impact recovery before further validation.
Do not extend training, increase impact strength, add repeated hits or change
springs based on this single-seed improvement.

## Intermediate ball-checkpoint assessment (2026-10-06)

The user answered `Proceed` to the proposed saved-checkpoint assessment.
The successor uses `stand_ball_checkpoint_curve.json`,
`evaluate_stand_ball_checkpoint_curve.py`, and
`scripts/evaluate-stand-ball-checkpoints.ps1`. The prior pilot and its sealed
modules/outputs remain unchanged. No PPO update or new training transition is
scheduled. Source and update-300 results are reused from the verified pilot.

The fixed first stage contains **256 new physical cases**: updates 100 and 200,
both seeds, both quiet/ball arms, eight starts, both X directions and both
native/GPU backends, using 5 kg / 1 m/s throws with the existing 12-second
episode, five-second launch, geometry and recovery definition.

A checkpoint qualifies for conditional assessment only if that **same update**
passes on both seeds and backends: every shot hits, neither direction regresses
relative to the source or its matched quiet checkpoint, and total recovery is
strictly greater than both comparators. No different checkpoint may be selected
for each seed. All checkpoints and per-case outcomes remain reported.

For each qualifying update, assess all four corresponding actors on both
backends: 16 quiet 120-second Stand cases and 32 easy 2 kg / 2 m/s impact cases
per actor, adding **192 cases per qualifying update**. With at most two such
updates, the complete budget is **256 to 640 cases**, including at most 384
conditional cases. Conditional success requires every Stand/easy-impact case
to pass. Any selected checkpoint is exploratory and requires further validation;
this assessment cannot replace the historical failed 300-update replication.
No automatic promotion, Arena change, stronger impact or extra training follows.

The runner checks checkpoint/full-AMP hashes against the prior seal, freezes
loaded actors, verifies saved traces through the existing admitted evaluators,
and seals inputs/sources/outputs on completion. Failures preserve their evidence
and stop execution; no silent budget extension or overwrite is allowed.

Execution:

```powershell
& ./mujoco_rig/mimic/scripts/evaluate-stand-ball-checkpoints.ps1 -Out logs/mimickit-training/stand-ball-checkpoints-20261006-01
```

### Completed checkpoint results

Output: `logs/mimickit-training/stand-ball-checkpoints-20261006-01`.
The process exited successfully. All **256 new cases** completed, every throw
hit (256/256), and 200/256 recovered. No additional PPO update or training
transition occurred. No conditional stage ran because neither checkpoint
passed the two-seed improvement rule. Consequently, 120-second Stand and
easy-impact retention at updates 100/200 remain unassessed by this successor.

Recovery totals below are out of 16, per backend, at **source / update 100 /
update 200 / update 300**. Native and GPU agree on every per-case outcome:

- Seed 301027, ball continuation: **9 / 12 / 11 / 11**.
- Seed 301027, quiet continuation: **9 / 8 / 9 / 8**.
- Seed 301028, ball continuation: **14 / 15 / 15 / 14**.
- Seed 301028, quiet continuation: **14 / 15 / 15 / 14**.

Every -X group remains 8/8. All variation is in +X recovery. For seed 301027,
the successful +X poses (zero-based) are source `{5}`, ball update 100
`{4,5,6,7}`, update 200 `{3,4,7}`, and update 300 `{3,4,5}`. Thus the count
advantage includes turnover: update 200 loses source pose 5 despite gaining
other cases. Its quiet update 200 recovers pose 7; quiet updates 100/300
recover none. These outcomes constrain any interpretation of an aggregate gain.

For seed 301028, both intermediate quiet and ball actors recover exactly
`{0,1,2,3,4,5,7}` in +X. Pose 6 fails throughout all sampled conditions;
pose 7 improves temporarily in both arms and fails again at update 300.
The earlier 15/16 result is therefore not evidence of benefit specific to
ball practice. Neither update provides a qualifying exploratory candidate.

The final seal records `verified=true`, **13,392 protected inputs and 108
sources unchanged**, 228 hashed output files, `training_updates=0`, and
`promoted=false`. There is no failure record. The prior 300-update result,
selected Arena bundles, physics, controller and springs remain unchanged.

Evidence within the output root:

- `protocol.json`: frozen budget, candidate rule, sources and input lineage.
- `report.json`: all selected-level assessments and comparisons at updates
  100/200/300; empty qualifying/candidate lists.
- `final-verification.json`: integrity checks and output hashes.
- `<update>/<seed>/<arm>/selected`: per-direction/backend reports and complete
  saved physical traces. The source/update-300 traces remain in the sealed pilot.

**Interpretation:** the curve is not monotonically improving. The weaker seed
shows a partial, fluctuating benefit from impact practice; the stronger seed's
sampled outcome curve is identical under quiet and ball continuation. These
observations do not identify whether the cause is reward design, policy
optimization, available control authority, or inadequate impact exposure.
They do not justify a longer run or harder/repeated impacts by themselves.

**Subsequent diagnostic, completed below:** compare saved traces for seed 301028 pose 6 (persistent
failure), pose 7 (temporary success in both arms), and seed 301027's changing
successes. Inspect post-contact actions, delayed targets, joint/torque limits,
contact impulses and first gate departures. Use that read-only comparison to
form a specific hypothesis and propose one bounded intervention. No new
simulation, training or controller change is part of that diagnostic proposal.

## Saved ball-trajectory audit (2026-10-06)

The user authorized the saved-trajectory comparison with `Proceed`.
`analyze_stand_ball_traces.py` reads NumPy arrays and JSON only; it does not
import a physics runtime, run policy inference, train, compile or execute tests.
Output: `logs/mimickit-training/stand-ball-trace-audit-20261006-01`.

Scope: **56 saved traces / 448 existing cases**, covering both seeds, source
plus quiet/ball updates 100/200/300, both X directions, eight poses and both
backends at 5 kg / 1 m/s. These are reused observations, not additional physical
evaluations; native/GPU counterparts are not independent training replications.
All 168 source trace/report/contract files match the checkpoint assessment's
sealed lineage. The audit also hashes its own source and the prior protocol/seal:
**171 inputs unchanged**, with `verified=true` in `verification.json`.

### Control authority and failure sequence

The audit measures three windows: first contact through the first pre/post
balance or posture departure (inclusive), the first second after contact, and
all live post-contact intervals. For a case with no departure, the first window
includes its complete remaining live episode. Thresholds and exact definitions
are saved in `protocol.json`; all per-case results are in `report.json`.

- Before the first departure, **zero of 448 cases** exceeds the policy action
  range or clamps a commanded joint target to its joint bounds. No controlled
  joint exceeds its configured position limits by more than 0.0001 rad.
  The largest absolute action in these windows is **0.681**, below the clip
  boundary of 1. The tested residual target range is therefore not being
  exhausted; merely enlarging it has no demonstrated basis here.
- Of **108 failed +X cases** (54 native plus 54 GPU), only **six** saturate a
  motor by the first departure. They are three paired conditions for seed
  301027, pose 7: quiet update 100, quiet update 300 and ball update 300.
  In each, the saturated motor is `Toe_L_rx`. This local exception remains
  relevant, but does not explain the other 102 failures.
- **None of the 108 failures saturates torque in the first second after first
  contact.** All eventually saturate a motor if the whole trajectory to failure
  is included. Reporting only peak torque over that entire window would blur
  the distinction between the early response and the developing fall.
- All 108 failures first exceed the **0.2 m/s horizontal-speed gate**, followed
  later by posture/height departures. This is an observed ordering, not proof
  that changing the speed reward alone would solve recovery.
- Issued targets reconstruct exactly, and the recorded active targets match
  the specified **two-control-interval delay (0.033336 s)** exactly. Maximum
  reconstructed PD torque discrepancy is **0.00000190735 N m**, consistent
  with floating-point arithmetic. There is no evidence here of an extra delay,
  stale target or incorrect torque formula.

For seed 301028 poses 6/7, the largest first-second action changes are usually
the two ankle flexion motors (`Foot_L_rx` / `Foot_R_rx`). Their peak target
changes implied by action changes span approximately 0.0345-0.0552 rad relative
to the 0.5-second pre-contact action mean. This is a measurement of the policy's
requested residual change, not actual joint motion or proof of an optimal
response. The policy is responding; these traces alone cannot establish that
its response direction, timing or coordination is sufficient.

### The delivered disturbance changes with the actor

A fixed ball mass, launch speed and aiming rule do not guarantee identical
contact impulses. The character's pre-impact pose and motion, and its response
during contact, can change later collisions with that same projectile.
Contiguous control intervals with nonzero impulse are grouped as contact
clusters; these groups do not identify individual substep contacts or body parts.

For seed 301028 pose 7 on native, all conditions first contact at **5.50044 s**,
with the same recorded pre-contact ball velocity, rounded to
**[1, 0, -1.554] m/s**. Yet total normal impulses differ:

- Failed source: **12.550 N s**, including **5.778 N s** in later clusters.
- Successful update 100: quiet **8.551 N s** (later 1.387), ball **8.588 N s**
  (later 1.403).
- Successful update 200: quiet **7.944 N s** (later 0.656), ball **7.627 N s**
  (no later cluster).
- Failed update 300: quiet **11.959 N s** (later 5.162), ball **11.407 N s**
  (later 4.691).

The first contact cluster is actually larger in those successful intermediate
conditions (about 7.164-7.627 N s) than in the failed source/final conditions
(about 6.716-6.797 N s). Much of the total difference is in subsequent contact.
Chest tilt before contact also changes: approximately 2.50-3.05 degrees at the
successful intermediate checkpoints versus 3.34-4.03 degrees at the failed
source/final checkpoints. These are associations, not isolated causal effects.

Seed 301027's reversals show a similar pattern. Its pose 6 ball checkpoint at
update 100 recovers with **9.233 N s** total / **1.935 N s** later contact;
the update-300 failure receives **11.325 / 4.325 N s**. Pose 7 recovers at
update 100 with **8.854 / 1.515 N s**, then fails at update 300 with
**11.101 / 4.129 N s**. Conversely, impulse alone does not perfectly classify
the complete dataset: successful and failed +X impulse ranges overlap.

Reducing or deflecting a later collision can be a legitimate useful policy
behavior. The existing ball evaluation remains valid for the complete encounter.
It simply cannot by itself separate improved balance after an identical
disturbance from changes to the disturbance delivered during that encounter.

### Selected next intervention: remove later ball contact

**Proposed only; not implemented or run.** Before altering PPO, rewards, springs,
motor limits or training duration, perform a frozen-policy comparison that
parks the projectile after its first contact cluster has ended. Preserve the
original physics until the first entire control interval with zero ball-character
impulse following actual contact, then park only that world's ball before the
next interval. This isolates the effect of later contact from the same throw.

Budget: **160 new cases**, no learning. Use +X, 5 kg / 1 m/s, the original launch
schedule, eight poses and both backends, with both seeds' source plus matched
quiet/ball checkpoints at updates 100 and 300 (ten actors total). Reuse their
sealed complete-ball traces as comparators. Update 100 supplies the observed
early improvement and update 300 the later regression; no per-seed checkpoint
selection or stronger-impact expansion is allowed.

Admission must reproduce the corresponding saved character/controller trajectory
through the removal point, preserve character state when parking the ball,
and verify no subsequent projectile impulse. Cases with no original later
contact provide unchanged-outcome controls. Any mismatch stops interpretation.
Keep the established survival/recovery definition, with recovery measured from
the final actual contact of the modified encounter. Also report survival and
time to sustained valid posture so changed contact timing is explicit.

If the formerly failed cases recover after later contacts are removed, that
supports later-contact robustness as the next curriculum target. Persistent
failures would direct attention to the response to the initial collision.
The result would not prove a generally improved brain or validate Arena's
stronger balls. No checkpoint promotion follows this diagnostic.








## Corrective-step pilot closeout (2026-10-07)

See [CORRECTIVE_STEP_PLAN.md](CORRECTIVE_STEP_PLAN.md): 400 updates, 1,638,400
transitions, 512 frozen cases. No qualifying recovery steps. Naive doubled leg
targets fail quiet Stand; the recovery reward does not consistently outperform
matched continuation. No promotion or Arena change. Next: behavior-preserving
control changes and an explicit weight-transfer/single-support/short-step curriculum.

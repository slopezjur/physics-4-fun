# Stateful recovery residual training

Historical experiment: the motion-driven controller below is superseded for
current reactive recovery work. See [REACTIVE_STEPPING.md](REACTIVE_STEPPING.md)
for the stationary policy, either-foot objective and current Arena selection.
Statements below about the default scene describe the historical experiment.

The default 5 kg scene uses `scene-bundle-5kg-cached`. Its export includes a
hash-checked support schedule computed once from the selected motion. The
Python adapter loads that schedule on each scene start; older bundles still
compute it at startup. Measured headless Godot time to `Loaded` is 2.81 s with
the cached bundle versus 10.68 s with the older 5 kg bundle. The 586-step +X
trial has exactly equal observations and actions with both bundles.

The default 5 kg Godot trial now waits for confirmed ball contact before
advancing the short step reference. In the measured +X trial, both feet stay
within 3 mm of their starting positions before impact at about 1.63 s. The
left foot lifts 8.5 cm after impact; two-foot load is continuous for the final
4.83 s of the full 8.15 s recovery reference. Four cardinal 5 kg scene trials
pass the measured recovery gate in 2/4
directions (+X and +Y); -X and -Y fall. The original 0.25 kg bundle passes the
same contact-triggered scene gate in 4/4 directions. This changes deployment
timing only. The actor was trained on steps already in progress with 0.25 kg
hits, so this scene result is a diagnostic, not evidence of reactive training
or four-direction 5 kg recovery.

## Validated starting point

`step-recovery-residual-ball-01` completes 8,192 native MuJoCo transitions through
MimicKit PPO. Its actor parameters change, all 16 completed training episodes
survive, and 12 episodes contain measured projectile contact. Its final
evaluation passes nominal loaded touchdown and four directional physical-ball
recovery cases. The projectile is **0.25 kg at 1 m/s**. These results do not imply
recovery from the original 8 kg projectile.

The trained hybrid also passes **13/13** original velocity-offset/force-pulse
recovery cases (`step-residual-push-validation-01`). At 0.5 kg it initially
passes 4/5 nominal/early-impact cases; +Y settles too late. A further 16,384
transitions at 0.5 kg yield 31/32 survived training episodes but only 3/5 final
validation passes (`step-recovery-residual-ball-05kg-01`). That continuation is
rejected. The accepted curriculum remains 0.25 kg, and the earlier checkpoint
is preserved. Survival alone does not satisfy the recovery gate.

## Fifteen-minute continuation result

`step-recovery-residual-ball-15min-01` resumes the accepted 0.25 kg checkpoint
and trains for 900.078 seconds. It collects 302,080 transitions over 590 PPO
iterations, including 469 confirmed-hit episodes. Of 631 completed episodes,
570 survive. The final native ball gate falls from 5/5 at initialization to
3/5: nominal landing fails after 7.60 s and the early negative-X impact falls
after 4.57 s. Training return falls from 555.48 near the start to 527.86 at
the end. The final checkpoint is rejected.

Saved checkpoints show non-monotonic behavior. Iteration 83 passes 3/5 ball
cases. Iteration 255 passes 5/5 ball cases but 11/13 original recovery cases.
Iteration 404 passes 4/5 ball cases and loses nominal landing. No checkpoint
tested from this continuation beats the accepted checkpoint on both gates.
The PPO KL limit is per update, so many locally bounded updates still drift
from the known good action map.

## Anchored continuation and native checkpoint gate

`step-recovery-anchored-pilot-03` resumes the accepted 0.25 kg actor and
collects 12,288 transitions over 24 iterations. The first eight iterations
keep the actor fixed while complete recovery episodes enter the training
buffer. Subsequent training samples quiet, physical-ball, and 10 N force-pulse
episodes at 25%, 37.5%, and 37.5%; the pulse onset is near 1.75 or 2.75 s.
The actor is constrained to a maximum 0.02 KL from its starting action
distribution on recorded full-sequence ball-recovery observations.

The iteration-24 actor passes all five fixed 0.25 kg ball cases and all 13
original velocity/force cases. The trainer saves it as `best.pt` only after
both suites pass. Earlier pilots with ball-only training fail the late +Y
force case (12/13) despite passing 5/5 ball cases. A 0.002 anchor permits no
further actor updates after the first iteration; 0.02 with force exposure and
one-episode warmup permits a passing learned update. This is native 0.25 kg
training admission, not 8 kg or Godot deployment validation.

The independent `step-recovery-anchored-pilot-03-independent` probe confirms
13/13 recovery cases. In `step-recovery-anchored-rollback-01`, a further
16,384-transition continuation passes both gates at iteration 16. Its
iteration-32 actor passes 5/5 ball but only 12/13 velocity/force cases. The
trainer rejects that candidate, restores the last selected actor and Adam
state, and retains the iteration-16 `best.pt`. Gate history records the
rejection and rollback; `latest.pt` remains the rejected diagnostic candidate.

## Fifteen-minute anchored training and scene selection

`step-recovery-anchored-15min-01` resumes the passing rollback checkpoint and
trains for **900.328 seconds**. It collects 99,328 native transitions over 194
iterations: 202 completed episodes, 84 confirmed projectile hits, 194 episode
survivals, and 42 accepted actor updates. The iteration-32 candidate fails its
native gate and is rolled back; every gate from iteration 64 through the final
iteration 194 passes **5/5 ball** and **13/13 velocity/force** recovery cases.
The selected checkpoint is `best.pt` at iteration 194, hash
`51e47dba0b49adc15b560a4570a8e6dfacdeed9310a6800a28efc39d9c81dd15`.
Mean training return falls from 555.58 after the first eight iterations to
545.44 at the end; native gate retention supports this selection, while these
returns do not establish an improvement over the starting controller.

The scene bundle is `step-recovery-anchored-15min-01/scene-bundle`. The default
`MimicPerturbArena.tscn` loads this bundle, starts the short recovery motion at
phase zero, and launches a 0.25 kg ball at the chest at 1 m/s for impact near
1.75 s. A headless Godot trial completes all 8.151 s with confirmed impact.
Across 489 matching control steps, maximum Godot/native observation error is
0.000884 (a foot-load channel) and action error is 0.0000335. The scene parity
report passes its 0.005 observation and 0.0001 action limits.

The controller is a hybrid: the observation-based recovery controller handles
transfer, swing, touchdown and two-foot balance; PPO learns a bounded additive
correction. The learned correction alone cannot control the body. It uses native
MuJoCo and the pinned external MimicKit PPO implementation. All movement still
comes from the original delayed joint motors and ground contact.

The 8.15 s episode includes roughly 5.6 s after nominal touchdown. The nominal
gate requires retained first loaded contact. An impact may briefly unload a foot,
but must recover continuous loaded support by 3.55 s, retain it for at least two
seconds, and finish upright and slow. Projectile misses fail impact validation.

## Why the previous learning route failed

- Frozen Stand normalization clips 68% of future-reference values on the step
  dataset; 33 reference channels are clipped in every sample. A calibrated extra
  input path fixes that information loss, but action cloning still fails landing.
- The teacher remembers capture and contact decisions during swing. A plain
  feedforward actor trained on sparse trajectories does not reliably reproduce
  that feedback. A motion-plus-neural-feedback prototype retains nominal landing
  but fails all 12 disturbed cases.
- The force probes apply 10 N for 0.1 s. An 8 kg ballistic projectile at 1 m/s is
  substantially stronger and includes a vertical impact component. The teacher
  fails all four tested early 8 kg impacts. Mass progression is explicit in each
  run; the generated rig files and viewer defaults are unchanged.

## Contract

`mimic_stateful_recovery_residual_v1` has 400 observations:

- The existing 362 support-v2 observations.
- The baseline controller's 30 current actions.
- Eight remembered values: landed flag, selected ankle offset, two capture
  offsets, coronal proportional/derivative gains, and landing proportional/
  derivative gains.

The residual's normalizer is calibrated from the complete step dataset. The
baseline is evaluated once per control step, with independent memory and reset
handling in each world. Training uses full episodes, 25% unforced trials,
37.5% physical-ball trials and 37.5% force-pulse trials, four directions,
and early/late impact timing jitter. It rewards observed
upright posture, foot placement, loading and settling. The authored motion is
a joint-target guide; it is **not admitted as an AMP/imitation reference**.

The residual scale is 0.05 in normalized target action units. Actor updates use
the local KL guard and a fixed starting-policy anchor on native recovery states.
The run saves `initial.pt`, `latest.pt`, `best.pt`, source hashes, configuration,
per-iteration progress, and every native gate result. The selected `best.pt`
must pass both five-case ball and 13-case velocity/force gates. A warm start
requires that hash-matching checkpoint from a passing run.

## Train or continue

From the repository root, using the existing recorded teacher artifacts:

```powershell
$python = 'mujoco_rig/mimic/.venv/Scripts/python.exe'
$mimickit = 'D:/Proyectos/Juegos/Tools/MimicKit'
& $python -m mujoco_rig.mimic.train_recovery_residual `
  --mimickit $mimickit `
  --teacher-run logs/mimickit-steps/short-loaded-step-36 `
  --teacher-search logs/mimickit-steps/landing-action-forward-com-01 `
  --teacher-dataset logs/mimickit-steps/step-conditional-arm-dataset-01 `
  --contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --resume logs/mimickit-steps/step-recovery-anchored-15min-01/best.pt `
  --ball-mass 0.25 --envs 8 --iterations 32 `
  --out logs/mimickit-steps/recovery-residual-next
```

Omit `--resume` for a zero-residual initialization. Every output directory must be
new. Increase ball mass only as an explicit next curriculum stage; a failed stage
does not replace the earlier passing checkpoint. `latest.pt` is the last update;
`best.pt` is the gate-selected, hash-bound warm-start checkpoint.

## Validate the composed controller

```powershell
& $python -m mujoco_rig.mimic.probe_recovery_robustness `
  --mimickit $mimickit `
  --teacher-run logs/mimickit-steps/short-loaded-step-36 `
  --observation-search logs/mimickit-steps/landing-action-forward-com-01 `
  --contract logs/mimickit-training/stand-support-v2-15min-01/export/contract.json `
  --adaptive-ankle --negative-x-recovery `
  --negative-y-roll-kp 12 --negative-y-roll-kd 2 `
  --right-arm-clearance --right-arm-clearance-scale 0.25 `
  --residual-checkpoint logs/mimickit-steps/step-recovery-anchored-15min-01/best.pt `
  --out logs/mimickit-steps/recovery-residual-validation
```

This runs the original 13 velocity-offset/force-pulse cases. Add `--ball-speed 1
--ball-mass 0.25` to replace force pulses with real ball launches. Impact reports
include contacted bodies and accumulated world-space impulse. The mass changes
only the isolated projectile's mass and inertia; it does not modify the character.

## Godot

`MimicPerturbArena.tscn` now loads a recovery scene bundle with the short step
reference, hash-checked projectile world, scripted feedback controller,
and selected 400-input residual. `MjMimicStandDriver` continues to construct
the existing 362-input support-v2 observation. The local Python controller
reconstructs the same COM and remembered contact decisions as native training,
then adds the residual action before the existing delayed target-PD motors.
The local MuJoCo Python environment is required when the scene runs.

The selected 15-minute 0.25 kg bundle completed the full 8.15 s Godot trial
with a confirmed 1 m/s chest impact. Its 489-step native comparison passes.

The default scene now points to `scene-bundle-5kg`, which keeps the selected
0.25 kg-trained controller and changes only the projectile to **5 kg**. Its
headless Godot +X chest trial confirms impact and survives 8.151 s. Godot and
native MuJoCo agree over 489 steps to maximum observation error 2.33e-5 and
action error 4.13e-5. The native episode fails the recovery gate: continuous
two-foot loading resumes at 4.83 s, later than the 3.55 s limit. In the four
early cardinal 5 kg native impacts, all four balls hit, only +X survives, and
none passes the recovery gate. The scene shows the ball mass measured from the
loaded MuJoCo model; this 5 kg setting is a stress test, not a trained or
admitted mass stage.

Export a passing selected checkpoint with `export_recovery_scene.py`, then set
the scene's `BundleDirectory` and `ModelPath` to that bundle and its
`dummy_ball.xml`. The default scene paths point to the current 15-minute run's
`scene-bundle-5kg`. The passing 0.25 kg bundle remains at `scene-bundle`.

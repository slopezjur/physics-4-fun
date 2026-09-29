# Reactive recovery learning audit

## Measured failure

The admitted 2 kg actor was frozen and replayed against eight forward 5 kg,
1.5 m/s torso shots on native MuJoCo and Newton/MuJoCo-Warp. Both backends
confirmed all eight hits, activated the post-contact objective, and fell in
the same control intervals. The native reward averaged 0.8289 after contact;
the GPU reward averaged 0.8291. There was no action clipping on these traces.
Observation clipping was zero except for 0.044% of hinge-velocity values.
The 0.25-radian target range therefore was not a measured saturation blocker.
This does not establish that it permits every large recovery step.

Artifacts: `logs/mimickit-training/recovery-learning-audit-{native,gpu}-01`.
`audit_reactive_learning.py` saves the physical state, actions, observations,
rewards, contact flags, and normalization diagnostics for reproduction.

The old objective rewarded posture and slow motion without testing whether
momentum could be caught by the loaded feet. Its sustained bonus was based on
a hidden settling counter, absent from the actor and critic observations.
The previous hard quiet-state KL limit stopped all actor updates after update
16, but the trainer continued collecting samples. That run's additional GPU
time did not change its actor.

## Corrections

- `capture_v3` computes character COM from the authored body masses and
  transformed inertial centres; projectile mass is excluded. COM velocity is
  measured over the control interval. The reward evaluates a projected capture
  point against a conservative 0.06 m capsule joining the loaded feet. An
  unloaded foot cannot enlarge this support region. This is a reward proxy,
  not a proof of dynamic stability.
- Post-contact imitation weight is 0.05. Upright posture, capture and settling
  supply immediate recovery credit. Either foot may change placement. The
  two-second uninterrupted recovery requirement remains in the measured gate,
  rather than being a hidden reward clock. Quiet rewards and actor inputs are
  unchanged.
- Optional `--quiet-replay-weight` adds a differentiable KL penalty on measured
  quiet states from the frozen starting actor. It does not use the saturating
  hard anchor. Native standing and per-direction checks still reject regression.
- Every update records actor parameter displacement. A configured sequence of
  zero-change actor updates stops training with `actor_update_stalled`, excluding
  the explicit critic warmup. Saved candidate snapshots allow review without
  pausing GPU work for every checkpoint.
- Optional longer rollouts, lambda, and forward-hit mixture are recorded in the
  protocol. The first pilot retains 2,048 environments and evaluation interval
  64, uses 64-step rollouts and lambda 0.98, and draws half of first shots within
  20 degrees of forward. Other first shots and repeated shots cover all angles.

On the identical frozen native trajectories, `capture_v3` changes no qpos value
and reduces mean post-contact reward from 0.8289 to 0.5809. In the final half
second before failure, the reward falls from 0.4578 to 0.1651. Synthetic tests
check early momentum warning, either-foot capture, airborne-foot exclusion,
translation/yaw invariance, and exact quiet reward/observation retention. Native
COM reconstruction is checked against MuJoCo subtree COM.

The revised reward is an opt-in experiment. Successful tests of its mechanics
do not establish a robust recovery actor; independent physical-hit and standing
gates still control deployment.

## Training and exploration evidence

`reactive-ball-capture-v3-5min-01` completed 304.531 training-iteration seconds,
24 updates and 3,145,728 transitions. All 20 post-warmup updates changed the
actor. Short quiet standing passed, but the final cardinal result remained
23/32: direction 0 stayed 0/8, direction 2 regressed 8/8 to 7/8, and direction
3 improved 7/8 to 8/8. The checkpoint was rejected. Its `unseen_report.json`
reuses a previously inspected seed and must be treated as a regression suite,
not fresh admission. The trainer now draws and records a fresh validation seed
unless one is explicitly supplied.

`recovery-exploration-assay-02` tests a frozen actor with 128 forward episodes
per action-noise level. With no noise it recovers 0/128; the existing 0.05 noise
produces 8/128 recoveries. Noise 0.15 and 0.30 produces no recoveries and harms
standing. Increasing global action noise is therefore not the selected fix.
These stochastic successes demonstrate reachable recovery outcomes under the
current control range; they are not a deployable deterministic policy.

The earlier `recovery-exploration-assay-01` and `recovery-exploration-no-reset-01`
are invalid as forward-only assays: MimicKit's observation-space query reset
their launch configuration during agent construction. The corrected probe
installs and checks the launch directions after construction. No evidence of
the hypothesized batch/reset physics fault was established. Training selection
uses the correct ordering and was unaffected.

`recovery-post-hit-resets-01/recovery_resets.npz` contains 98 native states
between control steps 120 and 168 after confirmed forward torso impacts, with
the projectile separated and no failure. It includes all eight postures, qpos,
qvel, solved foot loads and the complete pending/active motor command state.
The optional reset curriculum samples these states only in training and only
among non-quiet episodes. The spent ball is parked, contact history is restored,
and later scheduled shots remain physical. Test episodes always start standing.
This gives the critic and actor more recovery-state experience without adding
a chosen foot, reference step, or recovery clock.

`reactive-ball-capture-resets-5min-01` completed 304.297 training-iteration
seconds, 16 updates and 2,097,152 transitions. All 12 post-warmup updates changed
the actor. Its selected checkpoint improves the fixed suite from 23/32 to
24/32, with direction counts `[0, 8, 8, 8]` and no directional regression.
Native and GPU short standing pass. Fresh validation seed 270951 yields 57/64
single-hit recoveries; the fixed repeated-hit suite yields 18/32. Forward
recovery remains unsolved, so this checkpoint is a training continuation source,
not an Arena admission. Reset-state restoration has an explicit native/GPU
parity test, including delayed motor commands.

Its independent `quiet-600s/report.json` passes all eight uninterrupted
600-second native trials. Minimum foot load after startup is 240.35 N; maximum
final-minute foot motion is 0.776 mm and maximum tail root drift is 0.255 mm.
The tested checkpoint SHA256 is
`867ff594586850da1474e124990bccdd6f49db3d04e74069fe14d8a355423246`.
This validates standing retention over that horizon, not forward-hit recovery.

Reproduce the configured experiment with
`pwsh -File mujoco_rig/mimic/scripts/train-reactive.ps1 -Minutes 5`.
The wrapper reads the shared `config.ps1` values (`Envs = 2048`,
`EvaluationInterval = 64`) and its `ReactiveTraining` section. Each new run
records an independent validation seed and copies the hashed training sources.
Use `-DryRun` to inspect the exact invocation without creating a run.

## Ten-minute continuation

`reactive-ball-capture-resets-10min-02` resumes the selected 24/32 actor with
learning rate 2.5e-5. It completes 616.344 synchronized training-iteration
seconds, 32 updates and 4,194,304 transitions. All 28 post-warmup updates change
the actor. Evaluation interval remains 64, so there is no intermediate native
evaluation pause during this run; checkpoint snapshots are saved every eight
updates. Total measured run time, including evaluations, is 945.265 seconds.

The final actor passes short quiet standing and recovers 25/32 fixed hits.
Forward recovery improves from 0/8 to 3/8, but opposite-direction recovery
regresses from 8/8 to 6/8. The checkpoint gate rejects it. The three successful
forward cases settle for 6.72–10.20 seconds and move either foot at most
6.44–9.60 mm. This is evidence of learned balance recovery, not yet a learned
recovery step or a robust all-angle controller.

The retained 24/32 actor scores 55/64 with fresh validation seed 270952 and
18/32 on repeated hits. The new randomized score is a separate suite, not a
like-for-like comparison with seed 270951. No Arena export is produced.

Offline snapshot review finds no earlier admissible forward gain: updates
8, 16 and 24 score 23/32, 23/32 and 22/32 respectively, with forward counts
still 0/8. All pass short standing. The runner and full per-case results are
saved as `review_snapshots.py` and `snapshot_review.json` in the continuation
directory. The retained training source remains the five-minute run's best
checkpoint with its 600-second standing evidence.

Verification: 23 CPU/unit tests pass across capture reward, reactive task,
recovery reward, evaluation and update guard; the separate GPU reset/parity
test also passes. Remaining work is simultaneous forward and opposite-direction
retention, followed by complete unseen/repeated-hit admission. More training
with the current settings is not established to solve that trade-off.

## Explicit Arena preview and five-minute continuation

The user explicitly requested loading the ten-minute final actor, then another
five minutes of training from that exact actor and loading the new final actor.
The ten-minute preview passed ONNX parity (3.73e-7 maximum action error); Godot
loaded it and confirmed eight contacts over a 20-second headless run without
falling. Preview export records experimental status and does not claim admission.

`reactive-ball-arena-continue-5min-03` starts from that `model.pt` (SHA256
`c5862805bb43d98a000abd0e8c532697883d29bb7827009d90e0ed68b537c8da`).
The shared PowerShell config supplies 2,048 environments and evaluation interval
64; learning rate stays 2.5e-5. It completes 306.922 training-iteration seconds,
19 updates and 2,490,368 transitions, with all 15 post-warmup actor updates active.

The final actor improves total fixed-hit recovery from 25/32 to 26/32, with
direction counts `[5, 8, 5, 8]` versus `[3, 8, 6, 8]`. All eight quiet cases
survive 30 seconds, but posture 2 only holds the final stable state for 0.317
seconds versus the two-second requirement. The checkpoint gate retains the
starting actor; its unseen score of 59/64 does not describe the final actor.

As explicitly requested, the Arena instead loads run 03's final `model.pt`,
SHA256 `881f01ff19c00abc3e0b900ab0a7796d7ca4406c15fd5535ead5ad5b36eddd6f`,
from `arena-preview`. ONNX parity is 3.58e-7 maximum action error. The scene
retains unlimited 5 kg balls at 1 m/s every 2.5 seconds and labels this actor
experimental. This manual selection does not change the standing, directional,
or unseen admission criteria.

The run 03 headless Arena check loaded the exact preview bundle and confirmed
eight physical contacts during 20 simulated seconds without a fall. The saved
log is `arena-preview/godot-smoke.log`; this smoke check is not recovery admission.

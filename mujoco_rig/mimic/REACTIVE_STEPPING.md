# Reactive stepping at 2 m/s

## Objective and current Arena

Learn quiet standing, physical 5 kg impact recovery, and a useful step with
either foot when needed. Finish upright with sustained two-foot support; a
recovery step may change the final world location. No fixed stepping leg,
motion reference progression or step clock is a recovery objective.

The Arena remains on the explicitly requested experimental run 03 final actor,
now firing at 2 m/s every 2.5 seconds. Its earlier 30-second quiet-settling
regression is still recorded. New training starts from
`reactive-ball-capture-resets-5min-01/best.pt`, which passed all eight 600-second
quiet trials, rather than treating the preview as admitted standing evidence.

## Authority and discovery probes

`step-authority-01` solves kinematic placement probes with 12 cm horizontal
displacement, 3 cm clearance, approximately level feet, and COM near the
supporting foot. At the current +/-0.25 radian target range, 29/64 probes meet
the declared tolerances; +/-0.5 admits 47/64. Forward/backward placement is much
less restricted than lateral placement. These are optimized poses, not motor
tracking or recovery proofs. No action range was changed.

`step-native-baseline-2mps-01` measures the frozen standing-retaining source:
18/32 physical-hit recoveries, direction counts `[0, 8, 2, 8]`, and zero measured
recovery steps. It captures 369 real post-hit states.

The current Arena actor is separately measured in
`step-native-arena-baseline-2mps-01`: 15/32 recoveries and zero qualifying steps
under the same fixed 2 m/s suite. Manual successes under random Arena shots do
not imply passing all postures and directions in this suite.

`step-exploration-2mps-01` applies correlated leg noise throughout the post-hit
episode. It discovers lift and landing but no settled recoveries under continued
noise. `step-exploration-2mps-02` limits exploration to 0.5 seconds following
contact, then restores deterministic feedback. Of 512 cases, one recovers with
a step (setting 7, replica 1, posture 2, direction 1). This is a diagnostic
exploration burst, not a deployed action script or recovery timing objective.

`step-native-exploration-2mps-01` replays selected exploration signals while
retaining native closed-loop actor feedback. It reproduces the successful
8.45 cm left-foot placement followed by 9.58 seconds settled. The 32 selected
exploratory trials are a discovery set, not a success-rate benchmark. They
provide 312 physically simulated reset states, including 57 single-support
states: 29 with the left foot unloaded and 28 with the right foot unloaded.
The bank preserves qpos, qvel, pending/active delayed motor commands and solved
contact observations. Native/GPU reset and transition parity passes.

## Learning experiment

`step_v4` is a state-based objective. It retains loaded-foot capture scoring,
adds limited credit for an unloaded foot approaching the momentum capture point
with clearance and an oppositely loaded foot, and reduces settling credit while
capture lies outside current support. Settled balance remains more rewarding
than swinging. There is no bonus per step, preferred leg, or required swing
duration in the reward. The quiet reward and deterministic policy contract are
unchanged.

Recovery resets are stratified by posture, impact direction, and support state.
Training uses all first-hit angles uniformly, 25% quiet episodes, a 1–2 m/s
physical-ball curriculum, and repeated hits in half the nonquiet episodes.
Half of nonquiet starts sample the measured reset bank. The quiet-state penalty
uses 512 source observations and weight 1.0, up from 128 and 0.1.

The first run is `reactive-step-v4-2mps-20min-01`. The explicit preset is
`ReactiveStepTraining` in `scripts/config.ps1`:

```powershell
pwsh -File mujoco_rig/mimic/scripts/train-reactive.ps1 `
  -Preset ReactiveStepTraining -ResumeSelectedSameMass `
  -Out logs/mimickit-training/<new-run>
```

It retains 2,048 environments, evaluation interval 64, fixed action noise 0.05,
learning rate 2.5e-5, 64-step rollouts and lambda 0.98. The 20-minute budget
counts synchronized training iterations; initialization and native validation
are additional. No discovery noise signal or exploratory action sequence is
executed by the trained policy.

## Evaluation

`RecoveryStepTracker` separately measures unloading below 5 N for at least
50 ms, at least 1.5 cm foot-and-toe clearance, at least 3 cm placement change,
and loaded landing for 50 ms. A recovered step also requires the existing
survival, upright posture, low velocity, and final two-second two-foot-settling
gate. Loaded sliding and brief bounces are not counted as steps. These diagnostic
thresholds do not prescribe a policy motion clock.

Before admission: preserve all quiet and directional gates, repeat the
600-second standing check, validate fresh unseen and repeated 2 m/s hits, and
verify the exported actor in Godot. A single discovered step is feasibility
evidence; it is not a robust learned recovery policy.

## First learning result and next exploration test

`reactive-step-v4-2mps-20min-01` completes 1,200.219 training-iteration seconds,
63 updates and 8,257,536 transitions. Its selected update 63 passes short quiet
standing and retains `[0, 8, 2, 8]` cardinal recoveries: 18/32, with no steps.
The fresh randomized suite scores 48/64 and the repeated-hit suite 4/32, also
without steps. Update 32 independently scored the same fixed results. The
standing penalty stayed soft and all 55 post-warmup updates changed the actor;
this is not another frozen-actor run.

The selected checkpoint also passes all eight uninterrupted 600-second native
quiet trials, with maximum final-minute foot motion 0.603 mm. This permits it
to seed the next exploration experiment while retaining long-standing evidence.

Swing guidance is nonzero on 55 of the 57 captured single-support states; its
maximum pre-posture contribution is 0.1081 and its mean over single-support
states is 0.0200. The term is active, but that alone has not taught initiation.
`step-bilateral-native-01` explores scaled and mirrored versions of the successful
signal over all postures: two left-foot successes, no right-foot success. These
are recorded as discovery data and are not used as a prescribed stepping target.

`risk_exploration.py` adds an optional state-conditioned exploration distribution.
Only leg variance increases, smoothly with observed horizontal root speed from
0.1 to 0.4 m/s; low-speed and upper-body variance are unchanged. Both PPO sampling
and its likelihood use the same conditioned Gaussian. It changes neither the
mean network nor deterministic export, and has no hidden phase or hit clock.
The next preset, `ReactiveStepRiskTraining`, tests maximum leg standard deviation
0.2 with learning rate 1e-4 and a per-iteration rollout KL bound of 0.01. This
bound resets against each iteration's behavior distribution; it is separate
from the earlier saturating cumulative quiet anchor.

## Risk exploration result and feedback discovery

`reactive-step-risk-2mps-10min-02` completes 601.751 training seconds, 31 updates
and 4,063,232 transitions. Native and GPU short quiet gates pass. Fixed recovery
remains 18/32; fresh randomized recovery is 51/64 and repeated recovery 4/32.
All have zero qualifying recovery steps. The per-iteration guard accepts all
80 actor optimizer steps in the inspected active updates. An independent native
check of update 16 also scores 18/32; its maximum absolute action is 0.5614,
without action clipping. More exploration has not yet taught step initiation.

`search_recovery_feedback.py` is a discovery probe for coordinated corrections.
It searches gains from four observable root-velocity components to both legs,
added to the frozen actor with an 0.8 normalized correction limit. The physical
action limits and controller remain unchanged. It receives no contact flag,
chosen foot, reference phase or recovery clock. CEM ranks 64 candidates over all
32 fixed cases per round, using measured survival and settled recovery. This is
an auxiliary search, not a replacement for MimicKit PPO or an Arena controller.

`step-feedback-search-01` improves its six-second discovery score from 18/32 to
29/32 in eight rounds, with direction counts `[7, 8, 6, 8]`. A saved intermediate
candidate independently recovers 26/32 over 12 seconds in native MuJoCo
(`step-native-feedback-round2-01`). The final feedback independently recovers
29/32 over 12 seconds (`step-native-feedback-final-01`). Neither produces a
qualifying step. These results establish useful leg-feedback corrections under
the current plant; quiet standing and unseen-hit admission remain outstanding.

`fit_recovery_feedback.py` fits measured feedback reactions into the existing
MimicKit actor's final linear readout, using ridge regularization and strong
rehearsal of the source actor's quiet outputs. The MLP architecture, normalization,
observation and action contracts are unchanged. Native quiet and directional
retention gates select any resulting initialization before further PPO training.
This transfers state-based reactions, not an action sequence or timed step.

## Protecting the standing network

All three `step-feedback-fit-01` readout fits fail native quiet standing, despite
quiet action RMSE of only 0.00028–0.00055. Preserving outputs along the measured
quiet trajectory was insufficient to retain the surrounding feedback behavior.
All three are rejected. This is measured evidence against deploying that fitting
method, not an action-limit problem.

A state-gated feedback probe activates corrections smoothly above horizontal
root speed 0.2 m/s or horizontal angular speed 0.5 rad/s, reaching full weight
at 0.4 m/s or 1 rad/s. `step-native-feedback-gated-01` recovers 22/32 with direction
counts `[2, 8, 4, 8]`, and `step-feedback-gated-quiet-01.json` passes all eight
30-second quiet cases. There is no hit flag, elapsed time, or selected leg in
this activation rule.

`gated_recovery_net.py` registers the local MimicKit actor architecture
`stationary_gated_recovery_v1` without editing the pinned upstream checkout.
It freezes the validated standing layers, standing readout and normalizer, and
adds a trainable velocity map plus nonlinear leg branch. The observation and
physical action contracts remain unchanged. The deterministic quiet output is
exactly the frozen base whenever the observable gate is zero. This property does
not replace physical long-standing validation, and normalizer clipping still
applies to the branch's reconstructed features.

`seed_gated_recovery.py` validates the migration and imports the discovered
velocity gains. `step-gated-seed-01` first reproduces the base 18/32 with a zero
branch, then selects the initialized 22/32 branch with all short quiet cases
passing and no directional regression. Unit tests verify that optimizer updates
cannot change the frozen base or quiet outputs while the gate is inactive.

`ReactiveGatedTraining` uses this seed for a 20-minute MimicKit PPO continuation,
with the common 2,048 environments and evaluation interval 64. Its actor learning
rate is 1e-3 with per-iteration KL bound 0.02; only the recovery branch learns.
The native 600-second quiet and fresh 33-second fixed-2-m/s unseen suites run
independently on the seed. The Arena remains unchanged pending admission.

## Measured limits of the velocity gate

`reactive-gated-2mps-20min-01` completes 1,210.298 training seconds, 59 updates
and 7,733,248 transitions. The selected actor recovers 23/32 fixed 2 m/s hits,
55/64 randomized 1–2 m/s hits, and 11/32 repeated-hit cases. An independent
33-second suite at strictly 2 m/s recovers 43/64. Eight uninterrupted 600-second
quiet trials pass. Frozen standing tensors remain byte-identical to the seed.
There are still **zero qualifying recovery steps**; this actor is not admitted.

`step-feedback-gated-search-02` finishes 16 discovery rounds at 26/32 over six
seconds, with no steps. This short search score is not deployment evidence.
`step-discounted-recovery-audit.json` compares actual transition rewards:
the four newly recovered cases score 86–98 after impact at gamma 0.99 versus
38–44 for the failed base trajectories. The reward already prefers these
recoveries; the evidence does not justify blaming the discount factor.

## Instantaneous balance observation

The native capture-risk probe recovers 27/32 fixed hits and passes eight quiet
30-second trials. Maximum quiet capture error is 0.514 mm. The new versioned
contract `mimic_stationary_capture_target_pd_v2` appends one scalar to the
unchanged 362-channel core: instantaneous character COM capture distance outside
the capsule joining loaded feet. It excludes the projectile and uses foot-plus-toe
loads from the existing last-solve contact contract. No hit flag or clock is added.

`stationary_capture_recovery_v2` freezes the standing network and activates its
learned leg correction smoothly between 1 and 20 mm of capture error. A zero
correction branch reproduces the original actor exactly. This is a training
architecture; it does not prescribe a foot, step sequence or landing time.

Newton's exposed body velocities lagged the current generalized state during
stepping. The new input therefore computes COM twists through forward kinematics
into private scratch buffers. Native/GPU recovery-state and transition parity
passes without changing the original observation or simulation buffers. Godot
matches 128 physical-state/load fixtures with maximum scalar error 3.09e-8 m
(`capture-godot-parity-01`). Quiet standing and unseen-hit gates still apply.

`ReactiveCaptureTraining` is a five-minute diagnostic continuation with the
existing 2,048 environments, evaluation interval 64, 5 kg ball, and 1–2 m/s
training distribution. Fixed selection uses 2 m/s. Its outcome must be measured;
neither the new input nor the auxiliary feedback seed demonstrates learned steps.

`step-capture-seed-01` scores 27/32 fixed recoveries, passes all eight uninterrupted
600-second standing trials, and recovers 51/64 independent strictly 2 m/s hits
over 33 seconds (`step-capture-seed-unseen-01`). It still has no recovery steps.
`reactive-capture-2mps-5min-01` completes 321.361 training seconds and 2,097,152
transitions. Its final actor remains 27/32 with no steps and does not improve
the checkpoint rank. The retained seed scores 59/64 in that run's different
randomized 1–2 m/s suite. These two unseen scores are not comparable benchmarks.

## Load-transfer reward experiment

The v4 swing term is zero until a foot has both unloaded and lifted. Its mean
over the last training batch remained approximately 0.00006–0.0007. The separate
`step_v5` experiment supplies 30% of swing guidance through continuous reduction
of a foot's load share, with opposite-foot support and capture-location alignment.
The remaining 70% still requires lift and unloading. Guidance is risk weighted;
it never rewards a step count or chooses a foot. Settling credit now additionally
requires each foot to carry at least 30 N, avoiding equal credit for indefinitely
holding single support. Undisturbed rewards are bit-identical to the prior task.

The physical discovered step has lower discounted return than its already
successful no-step baseline (`step-discovered-step-return-audit.json`). That is
not evidence of incorrect credit: that particular hit does not require a step.
No physically demonstrated stepping recovery from an otherwise failed case has
yet been learned or admitted.

`ReactiveTransferTraining` compares v5 from the same capture seed, seed number,
physical plant and exploration settings, with a ten-minute training budget.
One full test-suite run exceeded the existing contact-load parity tolerance by
0.659 N on a 1,235 N load. Isolated v4 and v5 parity reruns passed unchanged
tolerances; the initial failure is retained in `logs/step-v5-tests.txt`.

The completed v5 training budget is 611.375 seconds, 35 updates and 4,587,520
transitions (`reactive-transfer-2mps-10min-01`). The final actor passes short quiet
standing but recovers 26/32 fixed hits, versus the seed's 27/32, and is rejected.
Two cases lift a foot, neither completing a recovered step. Update 16 also scores
26/32 with one unsuccessful right-foot lift. Graded preparation credit has not
established useful foot placement or landing. The standing network, readout,
normalizers and fixed branch buffers remain byte-identical to the seed.

Keep v5 opt-in for investigation; this result does not justify promoting it or
assuming that another identical short continuation will solve stepping. The
best physically validated experimental seed remains `step-capture-seed-01`.
Its failed unseen cases and zero learned recovery steps leave the milestone
unmet. The Arena and its 2 m/s speed are unchanged by these experiments.

The retained seed passes the v5 run's GPU quiet check, scores 59/64 in its
randomized 1–2 m/s suite and 22/32 in the repeated-hit suite. These are evaluations
of the retained seed, not improvements attributable to the rejected final actor.
The five PPO experiments total 3,945.004 synchronized update seconds and
26,738,688 transitions, excluding evaluation and discovery-probe time. Consolidated
checkpoint hashes and outcomes are saved in
`logs/mimickit-training/reactive-stepping-milestone-status.json`.

## September 28 bounded recovery loop

`recovery-loop-authority-01` compares 16 settings in 2,048 GPU worlds:
0.25/0.5 rad target ranges, capture-only/support-aware activation, and four
correlated exploration amplitudes. Baseline replicas reproduce 27/32 recoveries.
Six stochastic trials recover with measured steps; increasing the physical
target range provides no aggregate recovery benefit. All six are cases the
deterministic baseline already recovers. Production motor limits remain unchanged.

Native replay at the original 0.25 rad range
(`recovery-loop-authority-native-01`) verifies a right-foot 8.48 cm recovery and
a left-foot 6.79 cm recovery, with 26/32 total recoveries. These are stochastic
discoveries, not learned stepping. A third GPU discovery does not survive native
replay. Successful native observations and actually applied, clipped actions
provide a bilateral initialization dataset.

The previous branch's internal +/-0.8 correction cannot represent those actions:
over 86 branch-active demonstration frames its minimum possible action RMSE is
0.06057. A +/-2 internal correction with support activation reduces that bound
to numerical precision (2.61e-8), while physical actions remain clipped to +/-1
and motor targets to +/-0.25 rad. This motivates the separately versioned
`stationary_support_recovery_v3`; existing v1/v2 inference remains unchanged.
The new gate is the maximum of capture risk and observable unloading with
opposite-foot support. It takes no step clock, hit-age or selected-foot input.

`recovery-loop-demonstration-fit-01` fits only the trainable recovery branch,
retaining the frozen standing network, normalization and upper-body output.
It rehearses quiet and baseline recovery observations and evaluates deterministic
native rollouts after bounded fit stages. Before fitting, the migrated actor
passes eight 30-second quiet trials and retains 27/32 recoveries with zero steps.
Fitting error alone never establishes closed-loop recovery or admission.

The completed 4,096-update fit uses 16.14 seconds of supervised GPU optimization
plus native evaluation time. Its deterministic final actor recovers 26/32 hits,
including learned left-foot steps in postures 6 and 7 under direction-2 impacts
(7.15 and 7.95 cm placements; over 8.5 seconds subsequently settled). Posture 6
was not one of the two fitted demonstration cases. The native short quiet gate
passes, but posture 0 under direction 2 regresses from recovered to failed.
Both learned stepping and that regression are retained in the evidence; this
candidate is not selected or loaded in Godot. A second bounded fit increases
baseline reaction rehearsal weight from 1 to 10 to test whether the regression
can be removed while retaining stepping. Long quiet and fresh unseen gates
remain outstanding for any improved candidate.

The loop also prepares a paired native difficulty audit using 5/6/8 kg balls
at 9 cm radius, and 5 kg balls at 6/12 cm radius. Each isolated model is checked
against the authored character; only projectile mass, radius and derived sphere
inertia vary. The 96-case suite covers twelve target body parts, eight postures,
four directions and randomized angular offsets, recording actual contact bodies
as well as intended targets. This is curriculum diagnosis, not held-out validation.

The stronger rehearsal fit (`recovery-loop-demonstration-fit-02`) completes
4,096 supervised updates in 22.03 GPU optimization seconds. Its final actor
retains 27/32 recoveries with no directional regressions, passes all eight
30-second standing cases, and recovers with one learned left-foot step in
posture 7 / direction 2 (10.58 cm placement, then 9.72 seconds settled).
`recovery-loop-support-seed-01` selects that exact checkpoint, SHA-256
`e8ba945bd5890f9ba2fcb02ccf4d50507171bafad388fda49487d8272bcf0650`.
It is a training seed, not Arena admission. Its independent 600-second standing
and 64 fresh 33-second, strictly 2 m/s tests are running; the original capture
seed receives the same fresh impact cases for a paired comparison.

`ReactiveSupportTraining` starts a ten-minute PPO continuation from this seed
using the same 2,048 environments, evaluation interval 64, and v5 task settings.
The changed hypothesis is that the broader support-aware branch and measured
stepping initialization make useful leg reactions reachable by PPO. A recovered
step breaks checkpoint ties only after quiet-standing, survival and recovery
counts. Per-direction regression rejection remains active. No scripted step
trajectory, foot choice or clock is supplied to the deployed actor.

The completed five-condition curriculum audit confirms all 96 shots hit in each
condition. At 9 cm radius the original capture seed recovers 87/96 at 5 kg,
82/96 at 6 kg, and 68/96 at 8 kg. At 5 kg, 6 cm and 12 cm radii yield 88/96
and 89/96. These changed contact conditions are not directly comparable to the
fixed 32-case torso suite. Intended-target contacts range from 55 to 67 per
96 shots; intervening body parts often receive the hit. The 6 kg suite includes
a spontaneous 10.45 cm left-foot recovery after head/chest contact, followed by
9.42 seconds settled. This shows stronger hits can elicit stepping in the
retained policy, not that its stepping is generally robust. Direction 0 remains
the dominant failure sector; mass alone is not a monotonic difficulty measure
for each case because contact and foot-placement outcomes change.

The selected support seed passes all eight uninterrupted 600-second native
standing trials (`recovery-loop-support-quiet600-01`). Its fresh strictly
2 m/s, 33-second suite recovers 53/64 with zero qualifying recovered steps
(`recovery-loop-support-unseen-01`, seed 280930), so Arena admission still fails.
The original capture seed's matched suite is required before claiming an unseen
improvement. PPO reaches active actor updates after the configured four critic
warmup iterations; update 6 accepts four actor steps, rejects one, and records
zero quiet-rehearsal KL. Updates and parameter changes are saved in `updates.jsonl`.

The completed paired unseen comparison gives the original capture seed 55/64
versus the support seed's 53/64 on exactly the same seed-280930 cases. Neither
has a recovered step in that suite. The support seed therefore remains an
experimental training initialization despite its fixed-case step and retained
standing; the original capture checkpoint remains the fallback. The comparison
is saved in `recovery-loop/paired-unseen-comparison-01.json`.

The support PPO run completes 605.392 synchronized training seconds, 32 updates
and 4,194,304 transitions. Its final native result is 22/32 recovered hits with
zero recovered steps; the short quiet gate passes. The final actor is rejected,
retaining the original support seed. A CPU midpoint assay of update 16 measures
23/32 and zero steps without interrupting GPU training.

`recovery-loop/regularization-audit-01.json` compares native trajectories and
the seed's first three seconds of post-hit states. The successful learned step
has discounted task return 93.98 versus 42.34 after update 16 loses that recovery.
All four lost midpoint recoveries also have lower actual task return. Reward
therefore already prefers those successful outcomes on these measured paths.
The action-bound regularizer has weight 10; its mean seed-state loss is 0.08859,
entirely from trainable leg outputs. Update 16 reduces that loss on identical
states to 0.07468. Its parameter change has cosine alignment 0.371 with the
negative bound-loss gradient. This motivates a controlled penalty ablation;
it is not yet evidence that the penalty caused the regression.

`ReactiveSupportNoBoundTraining` keeps the same initialization, seed, task,
reward, 2,048 environments, evaluation interval 64 and ten-minute budget, and
sets only optimizer `action_bound_weight` to zero. Hard action clipping,
0.25 rad target bounds and motor limits remain unchanged. The torso/head
curriculum is deliberately a separate later experiment.

Training now supports explicit `--target-set torso` (unchanged default),
`torso_head` (75% torso, 25% head), and `all` (twelve body targets). First and
second shots use the chosen set; the observation contract is unchanged.
A native test verifies default RNG equivalence, coverage of both shot sets,
and contract consistency. Fixed torso selection remains the common retention
gate; broader target coverage must receive its own physical validation.

The completed support PPO report retains the seed's weights: the retained actor
passes GPU quiet standing, scores 56/64 in its different randomized 1–2 m/s suite
and 19/32 repeated hits. These are seed evaluations, not improvements from the
rejected final actor. All 18 frozen standing/readout/normalization tensors in
the final actor remain identical to the source. The 32 PPO updates have at most
16 ms overhead between measured training intervals: there are no periodic
30-second pauses in this ten-minute training block. This measures wall-time
cadence, not GPU kernel occupancy. The controlled no-bound-penalty run starts
only after the preceding process and its final validations exit.

The no-bound run's update-16 native assay recovers 25/32 with zero recovered
steps, versus 23/32 and zero for weight 10. Both trail their 27/32, one-step
seed. Removing the penalty has not established retention at this midpoint;
the bounded final result is still required.

A separate CPU experiment (`recovery-loop-demonstration-rawmean-fit-03`) fits
unclipped action means to the same bounded demonstration commands, instead of
clipping predictions inside the supervised loss. This tests a way to avoid
unpenalized raw-mean overshoot during initialization. Execution still clips
actions and preserves the same physical target range. CPU minibatch sampling
differs from the earlier GPU fits, so this is an exploratory initialization
test, not a strict single-variable reproduction of their learning curves.
The unchanged quiet gate, fixed native recovery suite and frozen-base check
remain required at each saved fit stage.


The no-bound ablation completes 612.875 synchronized training seconds, 29 updates
and 3,801,088 transitions. Its final actor retains 27/32 recoveries and quiet
standing but loses the learned step; selection retains the one-step source.
The penalty-10 run recovered 22/32 after 605.392 seconds and 32 updates. Removing
the penalty improves this measured retention, but does not preserve stepping.
Training budgets are matched by wall time, not identical update counts. The
selected source scores 61/64 in this run's randomized 1-2 m/s test and 19/32
repeated hits; these are source results, not improvements in the rejected actor.

The CPU raw-mean fit ends at 26/32 with zero recovered steps and four directional
metric regressions despite reducing demonstration RMSE to 0.01938 and mean-bound
loss to 0.00002343. Its 11.3557 seconds are CPU optimization time. It is rejected.

The next bounded PPO experiment adds successful-impact teacher KL rehearsal to
the no-bound configuration. Quiet-only replay is often exactly zero because the
recovery branch is inactive in quiet states, leaving learned reactions unprotected.
The native trace must match source checkpoint, reference, ball and rig hashes.
Only active observations within three seconds after confirmed hits in successful
cases are eligible. Up to half the sample budget is reserved for measured stepped
recoveries: this source yields 180 stepped-case and 256 other recovery states.
These are training data; fresh unseen cases remain mandatory. The actor receives
only its existing state observation, with no foot choice or step clock.
`ReactiveSupportReplayTraining` preserves source, seed, task, 2,048 environments,
evaluation interval 64, ten-minute budget and physical controls, adding replay
weight 1. Selection tests exclude failed/pre-hit/inactive states; nine relevant
unit tests and the launch dry-run pass.

The impact-replay run reaches active PPO updates: iterations 5 and 6 accept
80 actor steps each, change actor parameters, and report nonzero impact-replay
KL (0.00114 and 0.00111), while quiet replay remains zero. This verifies the
new loss is active, not that recovery improved; native evaluation is pending.
A 60-second, one-second-resolution nvidia-smi sample during updates 5-8 averages
44.58% device GPU utilization (9-73%, no zero-use samples). Desktop consumers
also use the GPU, so these are device-wide activity samples, not per-process
kernel occupancy. They show sustained activity within the measured window.

A separate CPU paired audit now evaluates the support seed on the original
seed-280929 96-case body/angle suite at 5/6/8 kg and 6/9/12 cm radius. The
hypothesis is that the support-aware fitted step may change broader robustness,
which the fixed 32-case torso selection cannot determine. The original capture
seed's existing paired results provide the baseline; neither suite is held out
for promotion. The GPU continues only the existing impact-replay PPO experiment.


Impact-replay update 16 recovers 26/32 with zero qualifying steps. The source's
p7/d2 stepped recovery remains recovered without a measured step; its discounted
return is 94.21 versus 93.98 for the stepped source. A step itself is not required
when stable recovery succeeds. The regression is p7/d0, which falls at 9.63 s
after contact at 1.72 s. At discount 0.99, the successful source's return exceeds
that delayed fall by only 2.445. Recomputing the same native trajectories gives
gaps of 18.469 at 0.995, 46.809 at 0.997, and 123.921 at 0.999. This diagnoses
weak weighting of distant outcomes; it is not proof a discount change will train
better. `recovery-loop/replay-midpoint-return-diagnostic-01.json` binds the evidence.

`ReactiveSupportLongHorizonTraining` is prepared as a separate experiment changing
only discount to 0.997 versus the impact-replay preset. Wrapper dry-run confirms
2,048 environments, evaluation interval 64, 600 seconds, unchanged source and
physical controls. It must wait for the current GPU run to finish and its final
physical result; no trainer is launched concurrently. Critic adaptation with the
changed horizon remains a measured risk.

The support seed's paired broader audit currently gives 86/96 at 5 kg, 82/96 at
6 kg, and 62/96 at 8 kg (9 cm radius), with 1/2/4 recovered steps. Original
capture results are 87/82/68 with 0/1/1 steps. More measured steps have not
improved aggregate robustness. The support seed's new 5 kg head/chest case has
a 3.72 cm left placement and 8.37 s subsequently settled; a 6 kg forearm/pelvis
case also adds a 3.81 cm left step. Neither was a fitted demonstration. Smaller
6 cm radius gives 88/96, matching the original. The final radius case is pending.

The replay PPO final actor recovers 27/32 with one qualifying step, passes short
quiet standing, and has no directional metric regressions. It completes 623.469
synchronized training seconds, 29 updates and 3,801,088 transitions. This is the
first tested PPO continuation in this loop that retains the seeded step at its
final checkpoint. Its rank equals the source, so automatic selection retains
the source; this is retention, not aggregate improvement. All 19 frozen actor,
readout and normalization tensors are unchanged. The final actor SHA-256 is
21659b959817f60dc52bfa5bc99c53646b2bec49b551d11c37a4d500247a3acc.
Fresh paired 64-case native tests (seed 280946, strictly 2 m/s, 33 seconds) now
compare that final actor to the original capture fallback. These evaluate the
final actor, unlike the trainer's post-selection tests which evaluate its source.

The support-seed broader audit is complete. At 5 kg and 12 cm radius it recovers
87/96 versus the original's 89/96. Across all five paired conditions it never
improves aggregate recovery despite extra measured steps. Full case differences
and report hashes are saved in recovery-loop/support-curriculum-paired-comparison-01.json.

The replay trainer completes its selected-source checks: 61/64 mixed-speed
unseen cases and 19/32 repeated recoveries. The final actor's independent test
initially stops at the validator's selected-checkpoint guard because it tied the
source. An explicit `--candidate-sha256` mode now permits evaluation of that exact
unselected actor, records its status, and leaves the default guard intact. Two
identity tests pass. The failed attempt has no evaluation result; its separate
retry uses output recovery-loop-replay-final-unseen-02. The original capture
fallback completes the same fresh seed-280946 strict 33-second suite at 53/64,
zero steps. Its archived runner is restored to the recorded execution hash after
a concurrent validator source edit; the repair record documents the source and
Windows newline normalization. Neither policy is promoted.

Only after the preceding trainer exits, recovery-loop-longhorizon-10min-01 starts
from the same support seed with discount 0.997, 2,048 environments, evaluation
interval 64 and 600 synchronized training seconds. The final replay actor's CPU
validation continues independently. Physical limits and Arena remain unchanged.


The final replay actor completes fresh seed-280946 strictly 2 m/s, 33-second
validation at 49/64, zero recovered steps, versus the original capture fallback's
53/64 and zero steps. It gains one case and loses five. By direction, recovery
counts are 6/14/14/15 versus 7/16/15/15. The native fixed-suite retention does not
generalize into improved robustness; this actor is rejected for promotion.
The paired, hash-bound comparison is recovery-loop/replay-final-paired-unseen-01.json.
No additional 600-second admission test is needed for this failed candidate.

A read-only exploration audit examines 5,524 saved native source states in the
first three seconds after impact. Of 1,146 with support-aware activation above
0.25, 370 (32.3%) still receive the base exploration standard deviation because
root speed is at most 0.1 m/s. Mean normalized action standard deviation across
these risk states is 0.0998; only 16.7% reach the configured 0.2 maximum. This
suggests the older speed-only exploration gate misses some support instability.
These are native source states, not measurements of current training occupancy.
Aligning exploration with observable support/capture risk is a possible separate
experiment after the current discount test, not a change to its running process.

The validator now snapshots its own source bytes before evaluation and archives
those exact bytes afterward, avoiding concurrent-edit and Windows newline drift.
Both fresh paired reports have archived runners matching their execution hashes;
repair records preserve the reason for the prior mismatch. Validation outcomes
and policy-selection guards are unchanged by this archival fix.


The gamma-0.997 run's native update-16 audit recovers 25/32 with zero qualifying
steps, versus gamma-0.99 replay's 26/32 at the same update. Its final physical
result is pending; no improvement is established by the midpoint.

An optional support-aware exploration mode is now prepared. It takes the maximum
of the existing root-speed gate and the actor's observable capture/unloading
gate, keeping maximum normalized leg standard deviation at 0.2, unchanged means,
upper-body distribution and physical limits. Sampling and PPO likelihood use the
same state-conditioned distribution. Speed-only remains the default. Ten risk
exploration and gated-network tests pass, including either-foot unloading,
capture risk, unsupported feet and quiet/upper-body preservation. Both modes
produce exactly identical distributions on 512 saved source quiet observations.
This preflight does not replace independent quiet standing validation.

`ReactiveSupportExplorationTraining` clones the earlier gamma-0.99 replay preset,
changing only exploration activation. Its dry-run retains 2,048 environments,
evaluation interval 64, source, seed, replay and ten-minute budget. It is not
launched while the gamma-0.997 experiment runs; the two hypotheses stay separate.


The discount experiment completes 601.390 synchronized training seconds, 31
updates and 4,063,232 transitions. Final native recovery is 27/32 with zero
qualifying steps; short quiet standing passes and directional metrics do not
regress, but the source's one-step rank is retained. All 19 frozen standing,
readout and normalization tensors remain identical. The longest overhead between
training updates is 16 ms; the remaining post-training checks evaluate the
retained source. The next exploration run waits for this process to exit.

The exploration preflight also records a coupled effect: increasing conditional
standard deviation reduces KL cost for a fixed actor-mean displacement. Across
the saved support-risk states, the relative mean-displacement KL weight averages
0.36093 and reaches 0.0625. Sampling, PPO likelihood, actor KL and replay use the
same consistent distribution. This is a single gate configuration change, but
its outcome cannot be attributed to added action noise independently of the
changed KL weighting. Maximum noise, quiet distribution and motor limits remain
unchanged. The experiment uses the earlier gamma-0.99 replay configuration.

The discount run's remaining checks complete: retained-source GPU standing
passes, mixed-speed unseen recovery is 59/64 and repeated recovery 19/32. These
are source measurements, not results of the rejected final actor. After that
process exits, recovery-loop-support-exploration-10min-01 starts. Agent-config
comparison verifies only risk_exploration_gate changes from speed to support
relative to the earlier gamma-0.99 replay run. Initial native selection again
passes quiet standing and recovers 27/32 with one step.

Closer inspection of the broader audit identifies a useful exception to its
aggregate regression: at 8 kg, Chest/posture-3/direction-2 is failed by the
original capture policy but recovered by the support seed using both feet
(13.69 cm left, 8.76 cm right), followed by 8.97 s settled in the 12 s trial.
This is a measured recovery gain involving stepping, not merely a step added
to an already recovered case. A paired native neighborhood test now extends
that exact case to 33 s and angles -5/0/+5/+10 degrees across all eight postures.
This is a diagnostic neighborhood, not held-out admission. Both policies run
on CPU while the single exploration PPO run occupies the GPU.
The first neighborhood launches fail before evaluation because construction
allows at most 30 seconds. The harness now follows the existing validator:
construct at 30 seconds, then let the evaluator set the full 33-second case
horizon. Fresh output directories ending02 contain the actual running tests;
failed01 logs are preserved. Model/reference provenance and original case are
recorded before evaluation.

The paired 8 kg neighborhood finishes at 2/32 recoveries with one stepped
recovery for the support seed versus 1/32 and zero steps for the original.
The exact Chest/posture-3/direction-2 discovery remains upright for the full
33 seconds, with 29.97 seconds finally settled after moving both feet 13.69 cm
and 8.76 cm. The original falls at 4.23 seconds. Nearby angles for that posture
do not recover, so this is a narrow success, not robust stepping. A separate
12-second exact-angle replay over eight postures captures the successful
state/action trajectory (recovery-loop-8kg-demonstration-01); one case recovers
with both feet. This is data harvesting from a verified success, not a repeat
of an unchanged failed training setup.

A possible next supervised test is to transfer this necessary-recovery example
into the original capture fallback while rehearsing its 5 kg successes and quiet
standing. The teacher checkpoint and 8 kg model differ from the student's source
and 5 kg retention model; both pairs must have explicit, independently checked
provenance. The fitter currently assumes they are identical and must not be
bypassed by relabeling files. Any extension needs closed-loop 8 kg demonstration
assessment at each stage as well as the existing 5 kg and quiet retention tests.
The current GPU exploration experiment remains unchanged; its update-16 native
assay is now running on CPU.

The support exploration run completes 610.516 synchronized training seconds,
28 updates and 3,670,016 transitions. Its final actor retains 27/32 recoveries
and one stepped case. Selection changes only because mean survival rises from
10.739922 to 10.744609 seconds; this is not a recovery-count gain. The p7/d2 case
now moves both feet (10.21/5.50 cm) but settles only 2.984 seconds versus the
source's 9.717 seconds. Long trials are necessary. Selected GPU quiet passes;
mixed-speed recovery is 61/64 and repeated recovery20/32. A fresh paired native
seed280958 suite now tests this actor and the original fallback at exactly2m/s
for33 seconds. No Arena promotion occurs.

The demonstration fitter now accepts independently verified teacher-checkpoint
and demonstration-projectile provenance, preserving the student's separate
retention model. A provenance unit test covers teacher, model and trace mismatch.
The crossmass-fit-01 CPU experiment starts from the original capture policy,
uses the support teacher's verified8kg bilateral demonstration, and rehearses
5kg quiet standing and reactions. Every saved stage gets physical8kg tests as
well as5kg retention. Eligibility requires an8kg recovery gain, quiet standing
and no5kg directional regression; the seed selector respects this eligibility.
At stage256 it retains27/32 at5kg and shortquiet while recovering2/8 with steps
at8kg, versus0/8 for the starting policy. These are preliminary12-second results;
fitting error alone and demonstration success do not establish generalization.

Fresh seed280958 strict2m/s33-second comparison rejects the support exploration
actor:51/64 versus original53/64, no gained cases and two losses. Its one unseen
left-foot step moves12.74cm and leaves27.32s settled, but the original already
recovers that case without a step. The physical behavior is real; aggregate
robustness has not improved. The paired comparison retains both report hashes.

Crossmass fitting finishes after11.245 CPU optimization seconds. All three fitted
stages retain5kg27/32 and short quiet standing without directional regression.
At8kg, stages256/1024/4096 recover2/2/1 cases; fitting error keeps falling while
closed-loop success declines. The earliest eligible stage256 is chosen for
independent assessment, without using unseen scores:33-second8kg neighborhood,
paired5kg seed28095833-second validation, and8x600-second quiet standing. Its
initial8kg successes use both feet on postures1/2, while stage1024 succeeds on
postures2/3. Finalstage4096 is not preferred merely for its smaller fit error.

The independent impact validator now supports hash-bound fit candidates whose
directory has no best.json. Such candidates remain explicitly unselected; missing
metadata without an explicit hash still fails, and changed hashes fail. Three
identity tests pass. Existing selected-policy validation semantics are unchanged.

The crossmass256 33-second 8 kg neighborhood completes with 4/32 recoveries,
all involving measured steps, versus original1/32 and teacher2/32. Relative to
the original it gains three cases and loses none. Two successes use the right
foot and two use both feet, followed by28.52-30.15 seconds settled. This verifies
useful foot movement without delayed falls in these cases. It is still a narrow
diagnostic neighborhood of the demonstration angle, not fresh8kg generalization.
The separate5kg unseen and8x600-second quiet gates remain running. Comparison:
recovery-loop/crossmass256-neighborhood-comparison-01.json.

Crossmass256 passes all eight native600-second quiet trials, but its strict5kg
seed280958 test recovers51/64 versus original53/64, with no gains and two losses.
Posture5/Spine at0.5604rad falls after7.017s; posture6/Chest at3.3196rad falls
at4.717s. Neither records a step. The original survives both33-second trials.
The8kg benefit is real but does not satisfy retention, so this actor is rejected
for promotion. Hash-bound comparison:crossmass256-unseen-comparison-01.json.

The next controlled CPU fit tests rehearsal coverage. Previous source rehearsal
used only32 cases at four fixed angles. Optional randomized rehearsal now adds
64 independently seeded5kg cases at2m/s for12s (training seed281010), retaining
only successful, branch-active source observations. Total rehearsal weight10 is
split equally between fixed and random sets; demonstration/quiet weights,
source, teacher, physical bounds and optimizer stay the same. Every saved stage
gets closed-loop tests on both retention sets and the8kg demonstrations.
Eligibility also rejects random-set directional regressions. The new dataset
is archived with its hash; seed280958 validation states are not used for fitting.
Four existing provenance/identity tests pass and the CLI loads. Experiment
recovery-loop-crossmass-random-rehearsal-02 starts on CPU; it is supervised
initialization, not PPO. A new held-out seed is required for subsequent admission.

Random-rehearsal fit stage0 reproduces its source:fixed27/32, random50/64,
8kg0/8 and shortquiet pass. Its archived training data contains163 demonstration,
1,299 fixed-rehearsal and1,254 successful random-rehearsal branch-active frames.
The experiment remains running.

A parallel CPU diagnostic now evaluates rejected crossmass256 on the existing
paired broad curriculum (seed280929):12 body targets, randomized nearby angles,
5/6/8kg and6/9/12cm radius conditions. This measures whether its narrow8kg gains
extend beyond the demonstration neighborhood and can identify future training
examples. It does not override its failed5kg retention or authorize promotion.
No GPU trainer is running while these candidate diagnostics complete.

Randomized-rehearsal stages256/1024 retain fixed27/32, shortquiet and two8kg
stepped recoveries. Random training recovery is51/64 then50/64 versus source50;
neither has directional regressions. The earliest eligible stage256 is frozen
for fresh paired seed2810235kg/2m/s/33s validation and the8kg33s neighborhood.
Those results are pending; no selection or promotion has occurred.

A separate causal diagnostic tests the pre-optimization architecture conversion.
All tensors in crossmass-fit-01/candidate_000000.pt are exactly equal to the
original capture checkpoint, but the network changes capture-only activation
and0.8 correction limit to support activation and2.0. Fixed32-case parity does
not establish unseen parity. The known seed28095833-second suite now tests this
unfitted conversion, isolating architecture effects from optimization. If it
causes losses, a distinct zero-initialized support branch on top of the frozen
original recovery actor would preserve initialization behavior; implementation
waits for this result. This is diagnostic reuse, not new held-out admission.

Random rehearsal completes14.5966 CPU fitting seconds. Stages256/1024/4096 retain
fixed27/32 and shortquiet, with random-training51/50/50 versus source50 and8kg
demonstration2/2/1 recoveries. Earliest eligible stage256 matches the original
on fresh seed2810235kg/2m/s/33s:both recover53/64, exactly the same cases, zero
steps. Its8kg33-second neighborhood improves to5/32 with four gains and no losses
versus original1/32. Four cases use the right foot and one uses both feet, with
29.19-30.17s finally settled. This is useful neighborhood transfer, not yet broad
heavy-hit robustness. Hash-bound comparisons are archived in recovery-loop.

The unfitted support conversion matches original53/64 on known seed280958 with
no changed outcomes. The earlier fit's two losses appear after optimization;
conversion alone does not explain them on this set. The proposed extra frozen
recovery branch is not implemented without supporting evidence.

Four native tests now run for random-fit stage256:8x600-second quiet retention,
fresh paired8kg/2m/s/33s seed281037 against the original, and the previously
regressed5kg seed280958 as an explicitly reused diagnostic. The candidate stays
unselected pending these results. Arena still fails the all-hit admission gate.

Random-fit256 passes8/8 native600-second quiet trials, but fresh8kg seed281037
recovers33/64 versus original36/64, both zero steps (one gain, four losses).
The known5kg regression set improves from the preceding fit's51/64 to52/64,
still below original53/64; case54 remains lost. Neighborhood stepping gains
therefore do not establish broad robustness. This candidate is rejected.

The next experiment changes experience rather than fitting the same narrow
example: direct8kg/2m/s PPO from the original capture fallback. The new explicit
--initialize-from-quiet-report mode checks selected checkpoint identity, rig and
reference hashes, all eight600-second physical standing cases and at least25%
quiet episodes. It labels eligibility as experimental curriculum initialization;
it does not claim impact admission. A guard test covers valid evidence, each
provenance mismatch, short/incomplete trials and false-passing foot unloading.
The trainer archives the guard and standing-validator source with its protocol.

ReactiveHeavyCurriculumTraining in config.ps1 retains2048 environments, evaluation
interval64 and600 synchronized training seconds. It uses step_v5, uniform torso
angles,25% quiet episodes, quiet replay weight1,50% second shots, speed-based leg
exploration up to0.2, and no optimizer bound penalty at unchanged physical limits.
The existing5kg recovery-reset bank is disabled instead of relabeled for8kg;
all episodes start standing. Source architecture stays capture_v2. This changes
mass, speed distribution and reset mix together, so it is a curriculum experiment,
not a single-variable mass ablation. One GPU job starts as
recovery-loop-heavy-ppo-10min-01; final5kg retention and fresh heavy-hit validation
remain mandatory before promotion. Arena is unchanged.

The heavy run advances continuously through checkpoint16 (2,097,152 transitions,
254.907 synchronized training seconds). A40-second device-wide GPU sample averages
40.225%, range14-77%, with no zero-use samples; this includes desktop activity,
not just training kernels. The maximum between-update overhead through update13
is16ms. Native checkpoint16 tests now compare8kg recovery and5kg retention on
the same32 fixed cases as their18/32 and27/32 source baselines.

An independent native source replay captures373 valid8kg post-impact states,
covering all eight postures and four directions. It reproduces18/32 recoveries
with zero steps. The dataset retains exact source/reference/rig/projectile hashes
in recovery-loop-heavy-source-resets-01/recovery_resets.npz. It is reserved for
a separate later curriculum; the current run still starts standing and uses no
recovery reset bank. No Arena selection changes.

Heavy checkpoint16 native results are unchanged in recovery counts:18/32 at8kg
and27/32 at5kg, both with zero qualifying steps. Mean8kg survival rises only from
8.2106 to8.2220 seconds; forward0/8 and rear2/8 still fail most cases. This is
retention without meaningful recovery improvement. The full600-second PPO budget
continues; final results remain pending.

The new8kg reset bank passes a native restoration preflight:all eight postures
restore a matching recorded qpos and pending/active motor-control history within
1e-6, observations remain finite, and a5kg task rejects its projectile hash.
The bank hash and selected rows are archived in heavy-reset-bank-preflight-01.json.
This verifies native loading, not GPU restoration parity. The active trainer still
uses no reset bank; any reset curriculum remains a separate later experiment.

Direct heavy PPO completes611.875 synchronized seconds,35 updates and4,587,520
transitions. Final selected37e407e467549664202e1e9695cbf8a0d9e44e76b7dee05dbc8fc5b37d451151
improves8kg fixed recovery18/32 to19/32, gaining rear posture1 without a step and
without losses. All19 frozen standing/readout/normalization tensors match the
source. Actual final5kg fixed recovery remains27/32. Selected GPUquiet passes;
12-second unseen recovery is38/64 and repeated recovery7/32 (one stepped case).

Fresh paired8kg/2m/s/33s seed281101 improves from original30/64 to final34/64,
with zero qualifying steps in either actor. The hash-bound comparison is
heavy-final-paired-unseen-01.json. This is useful recovery improvement, not proof
of natural stepping or full admission. Long quiet and paired fresh5kg retention
are now running for the selected actor.

A new local reset-parity runner restores identical measured post-hit states into
native MuJoCo and the GPU adapter, then applies identical native-policy actions
for12 control steps. Initial qpos/contacts match exactly; maximum qpos error stays
6.54e-6. Transient contact error reaches0.3872N and observation error4.88e-4,
failing the strict0.1N/1e-4 gates. The final step is closer, but the failed gate is
preserved. Before reset training, measure action/target sensitivity and loaded-foot
threshold disagreements; do not assume either a serious defect or harmless noise.
The GPU test has exited and no reset bank is used in the trained policy.

The selected heavy actor passes all eight600-second native standing cases. On
fresh5kg seed281101 it recovers54/64 versus original52/64, gaining cases23/28
with no losses. Combined with fresh8kg34/64 versus30/64 (four gains, no losses),
it becomes the best tested research initialization; original3093c26b... remains
preserved as fallback. Neither fresh suite contains qualifying steps and full
Arena admission is still unmet.

Instrumented reset sensitivity on the first bank row per posture keeps target
error below1.372e-5rad and has no5/30/50N load classification disagreements,
despite strict force/observation parity failure. Selecting the least-supported
row per posture exposes larger divergence:maximum contact error20.2516N, qpos
error0.001313, observation error0.04044, action error0.006849 and target error
0.001712rad across12 steps. Load classifications still agree, but this is not
accepted as parity. The reset bank remains excluded; the cause is unresolved.

After that GPU diagnostic exits, recovery-loop-heavy-continue-20min-02 starts
from selected37e407e467549664202e1e9695cbf8a0d9e44e76b7dee05dbc8fc5b37d451151.
ReactiveHeavyContinueTraining uses1200 synchronized PPO seconds,2048 environments,
evaluation interval64,8kg at2m/s,25% quiet episodes and standing starts. It removes
the old source's quiet-report initialization option and validates selected same-mass
continuation. This extends a measured successful curriculum; no reset states or
physical limits change. Weights/normalization warm-start; optimizer/RNG restart
according to the established wrapper. Arena is unchanged.

The frozen heavy 10-minute best is now under the paired broad curriculum audit
recovery-loop-heavy-best-curriculum-01 (seed280929, five96-case conditions,
12 body targets,2m/s,12-second native trials). Initial5kg and6kg counts are
89/96 and86/96 versus original87/96 and82/96. Full case/radius comparison is
pending. This CPU audit runs alongside the single20-minute GPU continuation;
it evaluates its source checkpoint, not the still-changing training actor.

Inspection of low-load reset parity locates the divergence after a matching
initial restore: qpos/contact errors start at zero, remain small through step8,
and grow sharply at step10 (20.2516N contact error,0.001313 qpos error).
Both adapters reset solver history and forward restored states. No specific
engine defect is confirmed. Dynamic contact continuation needs further diagnosis
before admitting this bank; current standing-start PPO remains independent of it.

Heavy continuation checkpoint32 is saved after4,194,304 transitions and604.889
synchronized training seconds. Independent CPU32-case native assays now check8kg
recovery against19/32 and5kg retention against27/32. The GPU run continues toward
1200 seconds with its existing2048 environments and evaluation interval64.

The heavy source broad audit finishes:5/6/8kg at radius0.09m recover89/86/72
of96 versus original87/82/68;5kg at radius0.06/0.12m recovers90/90 versus88/89.
Paired gains are2/4/4/2/1 with no lost cases. Steps remain0/1/1/0/0 and occur
in the same previously stepped head/chest case. Thus this shows broader recovery
improvement, not new stepping ability. At8kg, direction0 remains weakest9/24.
The hash-bound comparison is heavy-best-curriculum-comparison-01.json. Audit and
evaluator sources match; archived task/network differences only add a default-
preserving target option and a training rehearsal helper, respectively. This is
a known diagnostic suite, not fresh admission evidence.

Checkpoint32 native assays complete:8kg improves19/32 to21/32 and5kg improves
27/32 to29/32. Each gains two cases without losses and has zero qualifying
steps. Exact case pairs and checkpoint/report hashes are saved in
heavy-continue-update32-comparison-01.json. These are intermediate fixed-suite
results; retain this checkpoint while the1200-second continuation finishes,
then require fresh paired impacts and600-second quiet retention for its final
candidate. No Arena promotion.

Before the continuation finishes, fresh paired validation seed281135 is fixed
for64 cases per mass,5kg and8kg, strictly2m/s and33-second native trials.
The immutable10-minute source37e407... is now being evaluated on both CPU suites.
The final continuation actor will be tested after fixed-suite retention, with
its identity frozen before reading candidate fresh results; checkpoint32 remains
a separately preserved fallback. Quiet8x600 remains required. No new GPU job
is launched while continuation2398 is active.

Fresh source baselines seed281135 complete:8kg42/64 and5kg56/64, both zero
qualifying steps over33 seconds at2m/s. These evaluate the frozen10-minute
source; final continuation results must be compared on the exact same cases.

Heavy continuation completes1210.421 synchronized PPO seconds,62 updates and
8,126,464 transitions. Final fixed8kg recovery rises19/32 to22/32;5kg rises
27/32 to30/32; both have zero qualifying steps. Short quiet passes. All19 frozen
standing/readout/normalizer tensors are unchanged. The final model357c484d...
and selected24d9c64a... serialize differently but all30 stored tensors are
exactly identical (selected-final-equivalence.json). Selected identity is frozen
before fresh results. Native quiet8x600 and paired seed2811355kg/8kg33-second
validation now run on CPU. Trainer posttests are still completing; no new GPU
job or Arena promotion. Exact fixed-case gains/losses are archived.

The continuation trainer exits cleanly. Selected GPU quiet passes; internal
12-second unseen recovery is52/64 and repeated recovery13/32 with one stepped
case. These do not replace the independent33-second and600-second gates still
running. A paired broad body/mass/radius audit now tests selected24d9... against
prior heavy source89/86/72/90/90 using unchanged diagnostic seed280929.

A small native saved-state probe checked whether the extra post-step mj_forward
rewrites solver warmstart: qacc_warmstart changed by exactly zero in that fixture
(tolerance1e-8,100 iterations). This does not identify the low-load parity cause;
no production engine was changed and reset-bank training remains excluded.

Independent continuation validation passes all8 native600.014664-second quiet
cases, including physical load/settling gates and exact checkpoint/reference/rig
provenance. Fresh paired seed281135 recovers62/64 at5kg versus source56/64 and
52/64 at8kg versus42/64, with6/10 gains and no lost cases. Both contain zero
qualifying steps. Selected24d9c64a... becomes the best tested research seed;
prior37e407... and original3093... remain preserved. Full admission still fails.

A40-minute continuation starts as recovery-loop-heavy-continue-40min-03 using
train-reactive.ps1 with ReactiveHeavyContinueTraining, explicit source24d9...,
Minutes40 and ResumeSelectedSameMass. Config remains2048 environments/evaluation64,
8kg2m/s,25% quiet,50% second shots,uniform torso angles and standing starts.
The hypothesis is further improvement from continued experience in a curriculum
that now shows substantial independent gains. No reward/physical limit/reset-bank
change. Weights/normalization warm-start; optimizer/RNG restart as before.
Only this trainer uses the GPU; broad CPU validation of its source continues.

The20-minute candidate broad audit completes:5/6/8kg radius0.09m96/92/85 of96,
versus its source89/86/72;5kg small/large radii94/93 versus90/90. Steps remain
0/1/1/0/0. All conditions improve in aggregate, but6kg loses two previously
recovered posture5 cases (Spine andUpperArm_R targets). Both still survive12s
with zero final settled time. Other conditions lose no cases. Exact comparisons
are archived in heavy-continue-curriculum-comparison-02.json; do not claim
universal non-regression from aggregate gains.

The neighborhood diagnostic adds explicit --allow-nonstep-case for known failed
or nonstep cases; the default still requires a discovered stepped recovery.
The flag is archived in its protocol and grants no admission. CPU paired replays
now extend each lost6kg angle to33 seconds acrossall8postures, source37e407 then
candidate24d9, recording observations/actions/rewards. This distinguishes slow
settling from a delayed fall and preserves the exactposture5 comparisons. The
40-minute GPU curriculum continues unchanged as the only GPU training job.

Both lost6kg posture5 cases survive33 seconds under both policies but fail the
final2-second settling criterion; neither produces a qualifying step. Acrossall
8 postures, candidate improves2/8 to6/8 for theSpine angle and1/8 to6/8 forArm.
Trace analysis of the final10seconds shows capture error exactlyzero and the
trainable capture branch inactive100% of the time. Candidate horizontal root
speed exceeds0.2m/s in44.7%/41.3% of these frames, despite foot loads staying
above195/228N and angular speed below1rad/s. Thus the learned branch cannot damp
this observed capture-safe motion. The source also has late oscillation; its
12-second pass was not sustained recovery. Preserve both short/long results.

A process-local CPU diagnostic now tests max(capture gate, a0.15..0.30m/s root
speed gate), retaining the checkpoint, correction limit and physical limits.
It first requires8x30-second quiet standing, then replays the two known33-second
8-posture suites. No actor metadata, training code or GPU process is changed.
Quiet600 traces have maximum speed0.1423m/s after1second (startup max0.2121), so
startup still requires direct validation. This probe is not training or admission;
report and implementation are archived in recovery-loop-settling-activation-probe-01.

Velocity/capture activation probe01 passes all8x30-second quiet cases. TheSpine
33-second diagnostic gains posture4 (6/8 to7/8) with no losses;Arm stays6/8.
Bothposture5 cases still oscillate and fail final settling;zero qualifying steps.
This limited gate intervention does not repair the problem without learning in
newly activated states. A600-second quiet-only probe now checks the changed gate
throughout long standing before considering a versioned training architecture.
No hit replay is repeated in this run; the production40-minute trainer remains
unchanged. Any serialized policy variant must retain explicit architecture and
checkpoint provenance and pass its own admission gates before Arena use.

Training continuity through update60 is measured at1071 synchronized seconds
with at most16ms between updates (94ms total overhead). Initial native assessment
and setup took64.031s separately. This establishes continuous iterations, not
GPU kernel occupancy. Evaluation64 now performs the configured native8kg and
shortquiet check; a separate CPU5kg audit of evaluation_000064.pt tests retention
against30/32. The velocity-gate longquiet probe is still running.

Evaluation64 of the40-minute run improves8kg22/32 to26/32, with27survivors,
zero qualifying steps and shortquiet passing. It is selected within that run;
5kg retention is still being evaluated and this is not final admission.

The process-local velocity/capture gate passesall8x600.014664-second native
standing trials, including physical load/settling/drift criteria. This supports
preparing a separately versioned learning experiment for capture-safe moving
states; it does not establish a settling fix or Arena admission. The current
trainer remains capture-only and will finish its bounded40-minute experiment.

Evaluation64 also improves5kg retention30/32 to31/32 with zero steps. Its
independent native report preserves the exact checkpoint identity; this is a
midpoint result, not fresh or longquiet admission.

A separately registered stationary_motion_recovery_v4 actor now combines the
existing capture gate with horizontal root speed0.15..0.30m/s. It retains the
capture-v2 correction limit0.8, frozen standing base and leg-only correction.
Thresholds are a serialized buffer, making checkpoint identity distinct and
preventing loading it silently as the old architecture. Quiet rehearsal selects
states under the actual motion gate. Nine gated-network tests pass, including
legacy behavior, restored identity, quiet invariance and nonzero learning
gradients for supported moving states.

seed_motion_recovery now builds a CPU candidate from24d9 with all30 original
tensors unchanged and only that gate buffer added. It checks action parity
against the tested process-local intervention, archives architecture/config/source
provenance, and requires measured shortquiet and every directional8kg retention
group before writing selected metadata. No optimizer runs. Serialized longquiet
and unseen admission remain separate; current40-minute GPU training is unchanged.

The serialized motion-v4 gate-only seed is rejected:shortquiet passes and8kg
recovery remains22/32, but forward survival drops3to2. Directional retention
therefore prevents selected metadata; it will not train or reach Arena. Existing
weights had only learned under the capture gate; applying them in new moving
states changed a previously surviving outcome.

The changed hypothesis preserves the complete capture policy and learns a new
small damping layer. stationary_capture_damping_v5 adds a zero-initialized4-to30
linear layer (leg outputs only), activated by horizontal speed0.15..0.30m/s,
both-foot loading up to30N and capture-safe support. Its gate fades tozero as
capture risk rises. Existing standing and recovery weights are frozen; the new
layer starts atzero, so original actions remain exactly equal. Correction and
physical action limits remain explicit and unchanged. Gate specification is
serialized to bind architecture behavior to checkpoint identity.

Ten network tests pass, including exact source identity at initialization,
source tensors unchanged after optimization, and learning confined to supported
moving leg outputs. CPU damping-seed conversion from24d9 now checks1024 measured
observations against the original actor and runs nativequiet/directional8kg
retention. Serialized longquiet remains required before training. The current
40-minute capture-v2 trainer continues unchanged.

Damping-v5 seed creation completes:shortquiet passes, native8kg recovery22/32
matches its source with no directional regressions andzero qualifying steps.
All30 original tensors are preserved;1024 measured observations produce exactly
the original actions. A selected seed is saved, but no optimizer has run. Its own
serialized eight600-second standing trials are now running before training.

Fresh validation seed281225 is fixed for the40-minute run:64 cases per mass,
5kg and8kg,2m/s and33seconds. Baseline24d9 tests run now; final candidate identity
will be frozen after native retention and before reading its fresh results.
Evaluation64 remains independently preserved as a successful intermediate.

The serialized damping-v5 seed passes all eight600.014664-second native quiet
trials. A controlled10-minute PPO test is planned after the current GPU job exits:
same8kg2m/s,12-second episodes,25%quiet,50%second shots,2048envs/eval64,
with only the new damping layer trainable. This isolates learning in supported
moving states; it does not assume natural stepping or change physical limits.

Fresh281225 source20min baselines finish:5kg62/64 recovered (63survivors),
8kg49/64 (54survivors),zero qualifying steps. These are comparison baselines,
not40-minute candidate results. Final candidate remains to be frozen/tested.

The40-minute continuation finishes126updates/16,515,072transitions in2413.202
synchronized PPOseconds. Final native8kg recovery improves22/32 to30/32,
including three recovered steps: rightfoot4.28/4.31cm(postures0/6), leftfoot
3.73cm(posture3), all direction2. Finalsettling9.55/9.50/5.97seconds.
Independent5kg fixedsuite improves30/32 to32/32,zero steps. Shortquiet passes
and no directional retention metric regresses. Selected81b0ce... has exactly
the same30stored tensors as model.pt; all19 frozen tensors match source24d9.

Selected identity is frozen before fresh281225 candidate results. Native8x600
quiet and paired5/8kg64case33-second validation now run independently onCPU.
This is measured stepping on fixed cases; unseen stepping and Arena admission
remain unproven. Detailed provenance: final-checkpoint-audit.json in runfolder.

The40-minute trainer exits successfully: internal12-second unseen59/64 with
three steps; repeated20/32 with4 stepped cases. GPUquiet passes.
Independent33-second fresh tests and native600-second standing remain pending.

The separately planned damping-only10-minute PPO experiment now starts through
train-reactive.ps1 / ReactiveHeavyContinueTraining, after confirming the previous
trainer exited. Source31ae50 is the zero-layer conversion of20-minute24d9;
this controlled branch experiment does not include the subsequent40-minute gains.
Only the damping layer is trainable;2048envs/eval64 and same8kg curriculum remain.

Fresh281225 strict2m/s33-second validation completes:
- 5kg: 64/64 versus62; 2 gains,0 losses; 0 qualifying stepped recoveries versus0.
- 8kg: 58/64 versus49; 10 gains,1 losses; 2 qualifying stepped recoveries versus0.

Quiet600 remains pending. Incomplete8kg and repeated-hit admission still
prevents Arena promotion. Exactpairedcases and hashes are archived in
recovery-loop/heavy40-paired-unseen-03.json.

All eight600.014664-second native quiet trials pass for selected81b0ce. It is
the best aggregate tested research seed;24d9 remains preserved and noArena
promotion occurs. The fresh8kg regression(case29,posture5,Chestangle5.17038)
survives33s but has no finalsettling. Atrace replay now diagnoses that loss.

Fresh8kg stepping includes leftfoot12.41cm(case33,alreadyrecoverablebefore)
and rightfoot3.37cm(case53,oldsourcefellat4.22s,newactorholds33s with30.35s
finalsettling). The latter is a useful stepped recovery gain. Source/candidate
33-second neighborhood tests now use thisangle3.78330 at[-5,0,5,10]degree
offsets andall8postures. This replaces the queuedfixedangle diagnostic with
a stronger failure-to-recovery case; nearbytests are diagnostic,notfresh.

The 33-second neighborhood around useful unseen step case53 completes:
81b0ce recovers32/32 versus24d9 22/32, ten gains and no losses; two stepped
recoveries versus one. These nearby-angle/posture tests support local robustness
but are diagnostic after discovery, not fresh admission.

The isolated8kg settling loss reproduces at posture5. In the final10seconds,
capture error is zero and the trainable capture branch is inactive throughout.
Root speed exceeds0.2m/s in33.3% of frames (mean0.166,max0.255m/s), while
feet remain above252/262N. The proposed damping gate would be active62.3% of
this window. Supported oscillation is again outside the current branch; this
does not yet prove that the separate damping experiment fixes it.

Damping update8 changes only damping.weight among noncritic tensors; all
original standing/capture/readout/normalization tensors remain identical.
The new layer has nonzero updates. A paired broad CPU diagnostic now evaluates
81b0ce across12 body targets,5/6/8kg and0.06/0.09/0.12m radii using the existing
280929 cases,96 percondition and12seconds. Compare against20-minute source.

The first three broad conditions complete:5/6/8kg recoveries94/93/90 versus
20-minute source96/92/85. The5kg losses are a Head/posture0 settling failure
and a Chest/posture7 fall at5.67s. The6kg gains include two right-foot steps
after head/chest contact, but another chest case loses final settling.
Both radius conditions remain running. Paired33-second Head5kg source/candidate
traces now distinguish slow settling from continued oscillation/delayedfall.

ReactiveHeavyRangeTraining is prepared in config.ps1 as a20-minute continuation
from81b0ce with8kg speeds1.25..2m/s, retaining the other settings and physical
limits. Hypothesis: include milder impulses after fixed-speed heavy training
regressed actual5kg cases. This is not an assertion that slower8kg collisions
equal5kg collisions;5kg retention still requires direct testing. Dry-run verifies
2048envs/eval64. It will wait for the current dampingtrainer to exit. New fresh
seed281305 is reserved for paired5/8kg33-second2m/s validation of this next run.

The damping-only PPO reaches34updates,4,456,448transitions and607.484
synchronized seconds. Final393e325... changes only damping.weight among all
noncritic tensors (norm0.016642); standing, originalcapture, normalization and
readout remain identical. Two33-second native probes now replay known6kg Spine
and Arm settling failures acrossall8postures on this actual finalmodel.pt.
Finalnative selection/posttests remain running; nextGPUtraining waits for exit.

The complete broad comparison is saved:5/6/8kg94/93/90 versus96/92/85;
small/large5kg balls96/95 versus94/93. Case losses percondition are2/1/2/0/1,
so aggregate gains do not satisfy retention. Stepped recoveries1/2/2/0/2.

The5kg Head33-second replay confirms persistent oscillation after a4.91cm
right-foot step: posture0 staysupright but neverfinalsettles; sourceholds29.60s
settled. Candidate final10s speedmean0.211,max0.344m/s versus0.069/0.122;
capture branch is inactive88.8%. Acrossall8postures,6/8 versus7/8 recover.
The measured step is not counted as recovered because settling fails.

Damping PPO exits: fixed22/32,zero steps,short/GPUquiet passing; internal12s
unseen51/64 and repeated13/32. Known6kg Spine/Arm33s stay6/8 each.
Arm late-speed mean drops0.181 to0.159m/s and above-threshold fraction41.3%
to26.3%, but final2s settling still fails. The learned damping contribution
before action clipping is below0.0007rad in these late traces. This is partial
motion reduction, not a solved recovery failure; further work needs a changed
hypothesis instead of repeating the same PPO configuration.

After confirming the damping process exited, ReactiveHeavyRangeTraining starts
for20 synchronized minutes from81b0ce,2048envs/eval64 and8kg speeds1.25..2m/s.
Only the speed range changes relative to the prior capture continuation.
Independent source baselines on precommitted fresh281305 run at actual5/8kg,
strict2m/s and33seconds, so future improvements can be compared case by case.

Fresh281305 frozen81b0ce source baselines finish: actual5kg64/64,8kg61/64
(strict2m/s,33seconds). These define the paired comparison for the active
variable-speed continuation, not improvement from that still-running trainer.

A bounded CPU damping-amplitude diagnostic now scales only the copied policy
damping layer by4 or8. The checkpoint, original capture/standing branches and
physical clipping remain unchanged. Each intervention first requires eight
30-second quiet trials, then replays the two known6kg33-second settling suites.
Hypothesis: the learned direction has insufficient amplitude (<0.0007rad in
measured late traces). No newPPO or deployment occurs; longquiet and independent
impact admission would still be required for a useful serialized candidate.

Damping-amplitude4x and8x both pass8x30quiet but leave the knownSpine/Arm
suites at6/8,with no gains or losses. Neither posture5 reaches2sfinalsettling
(Spine0.15/0.82s,Arm0/0s). Increasing amplitude alone has not solved the failures.

A zero-initialized damping layer is now being prepared on the newer81b0ce
stepping policy, using its exact600-second quiet evidence and measured traces
for action parity. This CPU seed conversion must preserve native8kg30/32with
three steps and shortquiet; serialized8x600 follows before any training.
The changed learning hypothesis is recovery-state coverage:6kg,30-second
single-hit episodes,all12bodytargets and25%quiet,with only damping trainable.
The earlier12-second torso/doublehit curriculum does not observe12..30s tails.
This next experiment waits for both seed gates and the currentGPUjob to finish.

The newer zero-damping seed retains8kg30/32withthree steps, passes shortquiet,
and has zero action-parity error with81b0ce. No directional regression occurs.
Its own serialized eight600-second nativequiet trials now run. The active
variable-speed GPU trainer remains unchanged; an independentCPU update32
fixed8kg check also measures retention of maximum training difficulty.

Variable-speed update32 retains the aggregate8kg count30/32 and31survivors,
with two steps versus three. It gains posture5/direction0 but loses posture1/
direction0; posture3/direction2 still recovers without its priorstep. This is
incomplete case retention, not evidence of a final improvement.

ReactiveDampingSettlingTraining is prepared and parsed in config.ps1:
6kg2m/s,30-second episodes,alltargets,25%quiet,no secondhit,2048envs/eval64.
It initializes from the newer zero-damping seed via exact longquiet evidence
because projectile mass changes. The standing validation is still running;
wrapper dry-run and GPU launch remain pending that gate and currenttrainer exit.

Variable-speed evaluation64 is rejected:8kg28/32 withfoursteps versus source
30/32 withthree; direction0 drops6to4 recoveries. Quiet passes, but selection
retains the source. Extra stepping does not compensate for failed recovery.
The bounded run continues to its final assessment.

The new zero-damping seed028d7f passes all8x600.014664s nativequiet cases.
The full settling-preset wrapper dry-run now succeeds with the actualhash-bound
quiet report,2048envs/eval64,6kg30s/alltargets/no secondshot. It is ready once
the current GPUjob exits. Paired CPU baselines now replay the6kg Spine/Arm
cases on its81b0ce parent; the older24d9 traces are not the correct comparison
for the planned newer damping actor.

The variable-speed run finishes82updates/10,747,904transitions with1213.188
synchronized seconds. Final8kg29/32withthree steps fails directional survival
and recovery retention; selectedsource preserves all30tensors exactly. All19
frozen tensors in finalactor remain unchanged. Final5kg32/32 withzero steps
is unchanged. The failedcandidate will not spend longquiet/fresh admission
resources. Trainer posttests assess retainedsource and must be labeled so.

Correct81b0ce-parent6kg33s baselines recover7/8 for bothSpine andArm;posture5
still oscillates. These replace24d9 as comparator for the newer damping seed.

The broad diagnostic now accepts12..60seconds while retaining its12s default.
A33-second baseline on81b0ce uses newly fixed seed281399 acrossall12body
targets,5/6/8kg andsmall/large radii,96cases percondition. It records actual
contacts and uninterrupted settling; it is a paired challenge baseline,not
Arena promotion evidence. This addresses the demonstrated12-second horizon gap.

The range trainer exits. Retained-source posttests are GPUquiet pass, mixed-speed
12-second unseen61/64 and repeated20/32; these are not rejectedfinal-actor
results. After checking no trainer remains, the20-minute damping-settling run
starts with the admitted028d7f seed:6kg2m/s,30s singlehit,alltargets,25%quiet,
2048envs/eval64. Standing andcapture branches remain frozen.

Fresh seed281425 is fixed for source/candidate comparisons at actual5/6/8kg,
2m/s,33seconds and64cases each. Candidate identity must be frozen before its
fresh results. Existing broad33s CPU baseline281399 continues independently.

Frozen028d7f source comparisons on fresh281425 complete:5/6/8kg62/63/59
of64 at2m/s33s. These are baselines, not the active damping learner results.

Training-health inspection finds sparse damping updates: update8 changes only
damping.weight, but many later iterations accept none. Rejections come from
the same-batch loss-decrease floor (1e-7*max(1,abs(loss))), not KL constraints.
The configured8-iteration stall window spans8.53simseconds, while an episode
lasts30seconds. A quiet gated branch can therefore exhaust that window before
fresh episode impacts arrive. sparse-update-audit.json records currentcounts;
monitor actual stop behavior before deciding a changed follow-up experiment.

The first30-second damping run stops on actor_update_stalled after22updates,
225.016 synchronized seconds and2,883,584 transitions. Native6kg recovery
remains29/32 with32survivors andzero steps; shortquiet passes. Only damping
weights change outside the critic (L2 .000133766). Finalrank does not improve;
selected-source internal12-second unseen64/64 is not trained-actor improvement.

The changed follow-up increases the stall window from8 to32iterations, covering
34.14simseconds instead of8.53 for30-second episodes. This allows fresh impacts
before declaring sparse gated updates stalled. Loss-decrease floor,KL,rewards,
learning rates and curriculum stay unchanged. Damping gate active fraction and
mean now accompany update metrics. Source syntax was compiled and wrapper
dry-run resolved2048envs/eval64/stall32 before launch. The prior trainer exited
before recovery-loop-damping-settling-window-20min-03 started (session93268).
The new run restarts from the same zero-damping seed028d7f to isolate the
stop-window change; it does not continue the tiny rejected final actor.

The independent33-second broad baseline (81b0ce, seed281399) completes480
confirmed physical impacts. Recovery is95/94/90 of96 at5/6/8kg with.09m radius,
and96/95 at5kg with.06/.12m radius. Total470 recoveries,475 survivors andthree
stepped recoveries. Five surviving cases fail final settling. Four of the five
falls follow8kg Head-target shots contacting Chest+Head; the other is a6kg
Arm-target shot contacting Chest. Exact failure conditions are saved in
recovery-loop-broad33-source-01/failure-classification.json. These are baseline
diagnostics, not admission evidence or results from the active damping run.

A CPU-only33-second, eight-posture trace replays the8kg Head p0 angle and launch
to inspect capture activation, foot unloading and clipping before any severe-hit
curriculum change. Current GPU work remains the single damping-window run. Its
new telemetry executes successfully; first12 iterations show active damping
states in3.6..12.8 percent of samples. The actor still updates sparsely; conclusions
about the extended stall window await complete episode coverage.

The8kg Head p0 exact-angle replay recovers1/8 postures (only posture7),with
zero recovered steps. Seven falls occur1.13..1.30seconds after impact. Their
capture branch is active in94.4..98.7 percent of post-hit samples; both feet
lift, and several landings still fail to stabilize. This failure is not explained
by an inactive capture gate. Leg action clipping affects10.3..12.2 percent of
post-hit values, which alone does not establish inadequate physical limits.
response-diagnostic.json binds these measurements to the captured trace.
The20-minute parent24d9 now replays identical cases to distinguish regression
from pre-existing severe-hit weakness; the single damping GPU run continues.

The extended stall window permits renewed learning after episode resets. At
update30, reported num_eps reaches2048 and damping parameter change is.00527759
L2, versus.000133766 total in the old22-update stopped run. This establishes
continued optimization after the old stop point, not recovery improvement.
stall-window-progress.json preserves the interim activation/update measurements;
native settling and retention gates remain necessary.

The earlier20-minute parent24d9 fails all8 identical severe head-angle cases;
newer81b0ce recovers posture7 (1/8), with no recovered-case losses. Thus this
challenge exposes a pre-existing weakness, not a new40-minute recovery
regression. Exact cases and report hashes are bound in
recovery-loop/heavy-head-paired-comparison-01.json.

Damping-window update32 (SHA27acc1b6a268e0dafcd2d09a1b79c935d4215ee3c73a3a937e7868032e5027b4)
changes only damping.weight outside the critic, with cumulative L2.0109068.
Two native33-second diagnostics now replay the known6kg Spine andArm settling
suites, compared with the correct81b0ce parent7/8 each. This tests whether
resumed optimization yields physical improvement while the bounded trainer runs.

Damping update32 does not fix the known settling failures: Spine remains7/8,
Arm drops7/8 to6/8,with zero recovered steps. Arm posture0 changes from30.27s
settled to0.10s; late mean horizontal speed doubles.0784 to.1579m/s and exceeds
.2m/s for24.2 percent of the last10seconds, versuszero before. Posture5 remains
unsettled in both suites. This midpoint is not admitted. The smallest8x30s
quiet check now distinguishes impact-specific regression from lost standing.
The soleGPU run continues its bounded optimization and nativeevaluation64;
renewed weight changes alone are not evidence of better recovery.

Damping update32 passes8/8x30.0024s native quiet standing despite its new
Arm posture0 impact oscillation. Update64 also passes shortquiet and preserves
29/32 fixed6kg recovery with32survivors andzero steps, but rank does not improve.
The source remains selected while bounded optimization continues.

A separate next experiment, ReactiveHeavyHeadTraining, is prepared in config.ps1
and its wrapper dry-run passes2048envs/eval64/1200trainingseconds. Source81b0ce
was trained on torso targets only. This follow-up adds Head to that target mix
(torso_head), preserving8kg2m/s,12s episodes,25 percent quiet,50 percent second
hits, rewards and optimizer settings. It trains the original capture branch
from the quiet-validated81b0ce actor; the damping branch experiment stays
separate. It has NOT started and must wait for the soleGPU trainer to exit.
Fixed5/8kg and longquiet retention, head challenge improvement and independent
unseen impacts remain required before Arena promotion.

Independent unseen validation now accepts explicit --target-bodies, including
Head, and records actual contacted body parts, target-contact counts and their
recoveries. The original torso case generator remains byte-for-byte equivalent
for the frozen281425 fixture (SHA32ced2e88a09688d7af12a5986c1d6137010a4ed56bb46634ac9fa10efd55e16).
Eight focused tests pass, covering fixture retention, all32 posture/direction
combinations for head-only tests, invalid targets and checkpoint identity guards.
The active trainer already imported and archived its original evaluator.

Frozen81b0ce now establishes head-only source baselines at actual5/8kg,2m/s,
33seconds and64cases per mass using predeclared seed281610. The head-training
plan was fixed before these results. Candidate identity must be frozen before
its paired test; these validation trajectories are not training data. Actual
contacts must support any claim about head recovery; aiming at Head is not
sufficient. Existing torso retention and longquiet gates remain necessary.

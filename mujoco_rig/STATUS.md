# MuJoCo track — status

## 2026-09-21: MimicPerturb scene and controlled-push baseline

Open `Scenes/RL/Isaac3/MuJoCo/MimicPerturb.tscn` with **F6**. This is live physics
using the retained `guarded-finetune-01/export` Stand policy, not a trained Perturb
actor. The main F5 scene remains MimicStand. Default: 20 N at the Chest COM for six
control intervals (0.100008 s, 2.00016 N·s), starting at interval 60 (1.00008 s).
Direction buttons restart the trial; R repeats, P pauses. Force, timing, duration
and reference phase are Inspector settings. Completion or falling holds the result.

`MimicTrial` shares scene lifecycle, controls and rendering between both scenes.
`MjMimicPerturbTrial` shares force application and measurements between the viewer
and batch replay. Pulses use world-frame force at body COM, expire after each
control interval, and are cleared on reset without affecting peer environments.
The push schedule is not included in policy observations. No rig, reward, actor,
normalization or motor-control change was made; no training was run in this step.

The fixed benchmark contains **52 five-second trials**: 10/20/40 N pulses in four
horizontal directions, at two onset times and two reference phases, plus four
matching unforced trials. Evidence: `logs/mimickit-perturb/baseline-02/`.

- Native MuJoCo and Godot: **4/4 unforced, 16/16 at 10 N, 16/16 at 20 N, 13/16 at
  40 N survived**. All three falls followed forward (+X) pushes.
- Newton: identical unforced/10/20 N survival; **14/16 at 40 N** in the final run.
  The earlier run gave 13/16. `40N-+x-p0.8-t60` is near the five-second survival
  boundary; do not claim exact full-horizon GPU/native outcome parity there.
- Short forced-rollout Newton/native pose discrepancy is 8.22e-6, root velocity
  discrepancy 4.34e-6 m/s. Force expiry, partial-reset isolation and reset replay pass.
- All **nine Godot/native parity checks** pass over the 52 trials, including pulse
  timing, survival, recovery and movement metrics. Pose discrepancy in the push
  windows is 4.75e-7; survival-time difference is at most 4e-7 s.
- Native aggregate foot travel is 8.38 cm unforced, 8.45 cm at 10 N, 8.74 cm at
  20 N and 11.78 cm at 40 N. This includes normal reference motion and is not a
  standalone realism score or a penalty against necessary recovery steps.

Settled recovery is a separate diagnostic: the final uninterrupted 0.5 s must have
root height >=0.75 m, tilt <=15 degrees, horizontal speed <=0.2 m/s, angular speed
<=1 rad/s and both foot loads >5 N. Only 2/4 unforced trials satisfy it; native
counts are 7/16, 7/16 and 8/16 under 10/20/40 N. Newton also differs on four settled
classifications. **Do not use this diagnostic as a training/promotion gate yet.**
First define recovery relative to the matching unforced/reference behavior, then
use mild-push fine-tuning with separate no-push retention and held-out push checks.
The measurements identify a recovery task, not a demonstrated need to redesign the rig.

Validation: 28 Python tests, 133 managed tests (one optional native fixture test
skipped), build, both interactive scene checks, and actual Godot batch replay pass.
Viewer checks cover pause, identical-repeat determinism, four directions, stopping
at completion, and visible rejection of invalid pulse timing. All 14 protected
baseline hashes and the retained Stand checkpoint hash remain unchanged.

## 2026-09-21: fifteen-minute guarded Stand run holds balance but does not improve

The requested run completed **901.156 seconds, 273 iterations, 1,118,208 samples**
with 128 worlds, seed 210921, frozen normalization and actor KL bound 0.02.
Output: `logs/mimickit-stand/guarded-15min-01/`. It initialized weights and
normalization from `guarded-finetune-01/best.pt` (local iteration 33), with fresh
optimizer/rollout state. The source hash was verified against its saved metadata.
Reward, rig, controller and training settings were held fixed throughout the run.

- All **18 periodic evaluations passed 8/8** three-second trials; no crash or
  abrupt standing collapse occurred. Their root errors ranged from 0.02855 to 0.03590 m.
- Final native and Newton policies both passed 8/8, with root errors **0.03406 m**
  and **0.03405 m**. Initial native error was **0.02833 m**, so no later evaluated
  checkpoint improved on the starting policy.
- The final policy also passed 8/8 five-second native trials, root error **0.03620 m**.
- Automatic selection retained the initial policy: `best.pt` tensors exactly match
  `initial.pt`. Its five-second native test passes 8/8 at **0.03324 m**. Its export
  passes all five actual Godot parity gates; ONNX max action error is **2.38e-7**.
- All saved observation-normalizer tensors remained exactly unchanged. Maximum
  accepted rollout KL was **0.01280**; 10,920 actor steps were accepted, none rejected.
  This run therefore does not measure a benefit from activating rollback.

Evidence: `report.json`, `learning_curve.json`, `updates.jsonl`,
`extended-validation.json`, and `godot-parity.json` in the run directory.
All 14 protected baseline hashes match. The interactive scene continues to use its
previously accepted iteration 833 bundle; training does not promote policies automatically.

**Decision: keep the retained candidate; another unchanged Stand run is not justified
by these results.** Next, evaluate that candidate under small controlled pushes to
measure recovery before defining the Perturb curriculum. This run establishes
standing retention on the same finite clip and phases, not general reactive balance
or proof that future optimization cannot regress.

## 2026-09-21: guarded fine-tuning and automatic best-policy export validated

The late-regression ablation implicates both actor updates and normalization drift.
Iteration 1025 weights with their original normalization pass 8/8; using the final
normalization reduces this to 6/8. Final weights fail 0/8 with either normalization.
Evidence: `logs/mimickit-stand/ppo-regression-ablation-01.json`. This separates the
components but does not identify the first damaging update or prove a unique cause.

The local `GuardedPPO` extension freezes loaded observation statistics, clips actor
gradient norm to 1, and rejects minibatches exceeding cumulative mean rollout KL
0.02, restoring both weights and optimizer momentum. Fine-tuning requires explicit
checkpoint/contract paths; it loads weights and normalization with fresh optimizer,
rollout state and local counters. The external MimicKit checkout is unchanged.

The trainer now retains the initial and best evaluated candidate as `best.pt` with
hashed metadata and exports that selection. `model.pt` still records the final
training state. Selection uses the fixed eight native phases and the existing
standing gate, not held-out data. The viewer bundle is never replaced automatically.

Validation: `logs/mimickit-stand/guarded-finetune-01/`, initialized from the accepted
iteration 833 checkpoint, completed **181 seconds, 53 iterations, 217,088 samples**:

- Every periodic evaluation and the final state passed 8/8 three-second trials.
- Best local iteration 33: native root error **0.02833 m**, Newton **0.02827 m**,
  compared with initial native **0.02968 m**. Both engines passed 8/8.
- Selected five-second native test: **8/8**, root error **0.03324 m**.
- ONNX max action discrepancy **2.38e-7**; all five actual Godot parity gates pass.
- All three saved normalizer tensors remained exactly unchanged. Maximum accepted
  rollout KL **0.01275**; 2,120 accepted actor steps, no rejection at the normal bound.
- A separate one-iteration CPU integration check intentionally used KL limit 1e-12.
  Its first update was rejected (attempted KL 7.68e-6), leaving actor weights,
  normalization and optimizer state exactly unchanged. Seven safety unit tests pass;
  all 23 Python regression tests passed during implementation.

Reports: `report.json`, `invariants.json`, `godot-parity.json`, and
`rollback-check/report.json` inside that run directory. Generated rig, shipped
policies and the protected production checkpoint match all 14 baseline hashes.
The interactive scene still uses the previously accepted iteration 833 bundle.

**Next: a longer guarded Stand fine-tune to test stability over a comparable budget.**
The short run establishes working safeguards and modest tracking improvement; it
does not establish sustained stability, unseen-motion balance, or Perturb readiness.
The KL bound applies per rollout on sampled states, not cumulative change across
training. Do not treat best-checkpoint retention as a solution to generalization.

## 2026-09-21: interactive MimicStand scene ready for F5

`Scenes/RL/Isaac3/MuJoCo/MimicStand.tscn` is now the main scene. It runs live
policy-controlled native physics with the selected `evaluation_000833` bundle,
configured through Inspector properties. Its MuJoCo library directory points to
the isolated Mimic environment; no environment variables are required.

The default trial lasts five seconds, then holds the resulting pose for inspection.
It also stops on a fall, without automatically resetting. R/button restarts;
P/button pauses or resumes. The existing CameraController provides right-drag look,
WASD/Q/E movement and wheel speed adjustment. Trial duration and reference start
phase are configurable within the finite reference clip. This scene does not claim
continuous indefinite standing or add a Perturb controller.

`MimicStandReplay.tscn` is now batch-only, retaining the existing Python parity
command and shared `MjMimicStandDriver`. Build, default main-scene startup without
environment settings, five-second completion, button/keyboard pause and restart,
and all five batch parity gates pass. No policy weights or controller settings changed.

## 2026-09-21: 30-minute target/PD run learns Stand, then regresses late

**Decision: retain checkpoint `evaluation_000833.pt` as the experimental Stand
candidate. Investigate late PPO update stability and add best-checkpoint protection
before another long run or Perturb training.** The selected actor passes the
three-second gate and an additional five-second test; the final actor fails.
This establishes learned standing on this reference, not general reactive balance.

The requested fresh, unchanged-setting run completed **1,800.063 seconds**, 1,071
iterations and 4,386,816 samples with 128 worlds, seed 210921 and fault tracing.
No native crash occurred. Output: `logs/mimickit-stand/target-pd-30min-01/`.
Controller, rig, reference, reward and PPO configuration were unchanged.

Learning progressed from 0/8 successes to 8/8 at about 17.4 minutes. Both original
gates first passed at 19.2 minutes, then passed at seven consecutive saved evaluation
checkpoints through minute 29.0. The lowest root error among those checkpoints was
at iteration 833 (24.15 minutes, 3,411,968 samples):

- Re-evaluated native: **8/8 three-second successes**, mean root error **0.02968 m**.
- Newton: **8/8 three-second successes**, mean root error **0.02972 m**.
- Extended native ONNX test: **8/8 five-second successes**, mean root error **0.03407 m**.
- Extended Newton test: **8/8 five-second successes**, mean root error **0.03378 m**.
- Actual Godot replay: all five parity gates pass, **8/8 three-second successes**;
  maximum prefix action error 7.49e-6 and episode-duration difference 2e-7 s.

Selection uses the existing eight evaluation phases: among checkpoints passing
8/8 survival and <=0.05 m mean root error, choose the smallest root error. This is
checkpoint selection, not independent generalization evidence. The five-second
test uses eight evenly spaced starts in `[0, reference_duration - 5]` within the
same non-looping motion, with unchanged failure conditions and no further training.
No longer hold, new motion, perturbation, or visual realism gate is established.

**Late regression:** logged test return falls from 169.57 at iteration 1024 to
50.00 at 1032, while PPO clip fraction rises from 0.506 to 0.933. The final actor
has 0/8 successes, 1.652 s mean survival and 0.182 m root error in both native and
Newton evaluation. It also passes Godot transfer checks, confirming this loss is
present before deployment. The exact optimization/normalization cause is not yet
isolated. The final model and failed export remain intact for diagnosis.

Use **`target-pd-30min-01/selected-000833/export/`** for experimental replay, not
the run's top-level `export/`. The selected `.pt` is the original saved checkpoint;
its separate export and report are in `selected-000833/`, alongside
`godot-parity.json` and `newton-evaluation.json`. The final actor's results remain
in the run's `report.json`, `godot-parity.json`, and `five-second-final.json`.

No training/controller source was changed for this run, no production policy was
replaced, and no commit or push was made. The generated model and shipped policy
hashes remain unchanged.

## 2026-09-21: target/PD learning path implemented; five-minute comparison complete

**Decision: retain the rig and bridge, keep this actor experimental, and use a
fixed-setting 15-minute Stand benchmark as the next bounded learning test. Do not
start Perturb or an overnight run yet.** The new control path transfers correctly;
the short PPO run shows modest improvement but does not solve standing.

The opt-in `--control target_pd` contract (`mimic_stand_target_pd_v1`) uses 30
reference-relative joint targets and physics-rate bounded PD feedback. Targets and
reference velocities wait two control intervals; the first two intervals apply
zero policy torque. Residual scale is 0.25 rad, with joint-limit clipping. Feedback
is `limit * clip(4 * angle_error + 0.08 * velocity_error, -1, 1)` on every native or
Newton physics substep. Original torque authority, passive mechanics, head control,
reference, reward and PPO YAML are retained. No support-force oracle is used.
The queue's two position/velocity/validity tuples replace pending torque actions,
giving 362 observations. Raw-torque 300-channel actors remain supported separately.

Fresh run `logs/mimickit-stand/target-pd-5min-02` completed 177 iterations and 724,992
samples in 302.344 seconds, with 128 worlds and seed 210921. Native mean survival
improves from **1.808 to 2.013 s**; best final trial is 2.400 s. Native and Newton
agree on all eight survival durations. **0/8 reaches three seconds**, and mean root
tracking error is **0.162 m**, above the unchanged 0.05 m gate. Periodic native means
are 1.786, 1.671, 1.781 s at 4k/266k/528k samples, followed by 2.013 s at the end.
The initial policy has small random residuals; the separate exactly-zero residual
baseline survives 1.786 s. More training is a hypothesis to test, not a demonstrated
solution. The old raw-torque run used a different budget and is not a matched trial.

Verification:

- 16 Python regressions, 130 managed tests and the opt-in native foundation test pass.
- Target native/Newton preflight passes 14 gates: short pose error 1.85e-6,
  motor torque error 0.00034 Nm, including reset isolation.
- Zero-residual and trained ONNX actors both pass all five actual Godot replay
  gates. Trained prefix observation/action/pose errors are 3.61e-6 / 3.35e-6 /
  2.76e-7; maximum episode-duration difference is 2e-7 s.
- The existing raw-torque Godot replay still passes. Generated MJCF, shipped ONNX
  files and the accepted production checkpoint match their baseline hashes.

The first attempt (`target-pd-5min-01`) crashed in native code after roughly 25 s
with Windows access violation 0xc0000005 and no Python traceback; it is not a
completed training result. The identical-setting retry enabled unbuffered output
and Python fault tracing and completed. The native crash's cause is unresolved;
retain the failed run's `failure.json` and monitor any subsequent bounded run.

Reports: `target-pd-preflight.json`, `target-pd-zero-01/godot-parity.json`, and
`target-pd-5min-02/{report,godot-parity,learning_curve}.json` under
`logs/mimickit-stand/`. See `mimic/README.md` for commands. No production actor was
replaced, and no commit or push was made.

## 2026-09-21: reference-controller diagnostics narrow the problem to balance/control design

**Decision: preserve the rig and bridge; do not extend the current raw-torque PPO
run. Prototype reference-relative joint targets with a physics-rate PD loop and
learned balance corrections.** MimicKit's pinned `data/engines/newton_engine.yaml`
uses `control_mode: pos`; our compatibility adaptation retained delayed raw torque.
Keeping the skeleton/sim bridge does not require keeping that action representation.
This is the next experiment, not a validated production change.

Bounded reference-controller tests are complete. Each controller family had 12
predeclared gain settings, calibrated on phases 0/3/7; selected parameters were then
evaluated across all eight phases, including five unused for selection. Physical
torque limits, passive joints, model and reference stayed unchanged. The primary
PD tests retain the exact 60 Hz / two-step delayed-torque contract.

- Primary normalized PD: 0/8 three-second successes; mean survival 1.002 s.
  Inertia-scaled PD with optional joint-bias compensation also fails (0/8); some
  cases become numerically unstable. Canonical corrected reports are
  `reference-control-normalized-02` and `reference-control-inertia-02`.
- Diagnostic zero-delay controls improve normalized PD to 1.502 s, still 0/8.
  Thus delay contributes, but removing it alone does not produce balance.
- A **different control contract**, with targets delayed two control intervals but
  PD feedback evaluated every physics step, reaches 1.875 s, still 0/8.
- Adding quasi-static support feedforward to that alternative reaches **7/8 at
  three seconds**, 0.0147 rad mean joint RMSE and 0.29% clipped torques. It still
  fails the root/foot tracking gates (0.0581 m / 0.0323 m) and **0/8 survive ten-second
  fixed-pose holds**; mean hold survival is 3.586 s. Joint tracking alone is not balance.

The support calculation uses prospective sole contacts within 1 mm of the floor,
friction pyramids and the same motor torque limits. Over all 180 reference frames,
the refined solution needs at most 14.4% of the available 60%-authority motor budget.
It balances the free base and controlled joints, allowing passive joints to settle
(up to 0.795 Nm residual at their exact reference angles). This is conditional
quasi-static evidence, **not proof of dynamic feasibility or a fault-free rig**.
Strict checks using only the reference's initial contacts fail because one foot is
slightly lifted. Native stepping establishes both-foot support within 16.7 ms,
before the first delayed motor command, so this transient does not explain the
later multi-second collapse by itself.

Diagnostic code: `mimic/reference_controller.py`, `probe_reference_control.py`,
`probe_target_pd.py`, `static_support.py`. Reports live under
`logs/mimickit-stand/reference-*` and `static-support.json`; the support-feedforward
result is `reference-target-support-01/report.json`. Twelve Python regressions pass.
The early `reference-control-01` selection minimized error among failed episodes,
which favored early termination; version 2 ranks survival before error and detects
numerical failures explicitly. Use the `*-02` primary reports above.

No training, production controller replacement, body-strength increase, motion
rewrite, commit or push occurred during this diagnostic. The alternative controller
has only native-MuJoCo evidence so far; it needs a matching Godot control contract
and transfer tests before an RL experiment or promotion.

## 2026-09-21: fixed-setting 15-minute Stand run and actual Godot replay complete

**Decision: keep the actor experimental.** The predeclared gate was eight of eight
three-second successes and mean root tracking error at most 0.05 m. The final
actor fails both gates; it is not promoted to Stand/Perturb production scenes.

The fresh run used the same reference, plant, torque limits/delay, PPO YAML and
reward as the smoke test: 128 worlds, 486 iterations, 1,990,656 samples in 900.515
seconds (time budget includes periodic output/evaluation). Mean native standing
time rises from 0.448 to 1.663 seconds; Newton gives 1.667 seconds. Best final trial
is 2.134 seconds, **zero of eight reach three seconds**, and mean root tracking
error is 0.117 m. Periodic checkpoint means progress from 1.20 s at 528k samples to
1.46 s at 1.58M and 1.53 s at 1.84M. This is learning progress, not stable standing.
Rendered native poses still show forward collapse.

Added an isolated `MimicStandReplay.tscn` and hash-validated reference/control
adapter using the existing native bridge. Actual headless Godot replay passes all
five parity gates for the final ONNX actor. Across the first 16 steps of each phase,
maximum observation/action/pose differences are 1.713e-5 / 3.540e-6 / 1.611e-6;
all eight full episode outcomes agree and duration differences are below 1.01e-7 s.
ONNX versus Torch action error is 1.79e-7. These bounded checks do not identify a
Godot transfer mismatch as the cause of the falls.

Validation: 127 managed C# tests, the opt-in native foundation sensor regression,
and nine Python motion/control regressions pass. The generated ABI adds the contact
frame offset; existing bridge APIs retain their semantics. Build input now excludes
saved C# source snapshots under `logs/`. The MJCF, shipped actor and accepted old
checkpoint match their preserved hashes. No commit or push was made.

Artifacts: `logs/mimickit-stand/ppo-15min-01/{report,learning_curve,godot-parity}.json`,
`model.pt`, periodic checkpoints, `export/` and `trained-poses.png`.

**Next:** check whether reference tracking is achievable under the exact torque
limits and action delay with a reference controller, before committing to another
long PPO run or changing rewards. The current result does not distinguish slow
direct-torque learning from an unsuitable tracking/control setup. Perturb remains gated.

## 2026-09-21: licensed motion-guided Stand reaches its first PPO smoke test

The isolated [MimicKit experiment](mimic/README.md) now uses a six-second neutral-idle
excerpt from Ian Mason's 100STYLE, under CC BY 4.0 with attribution. Source hashes,
license and changes accompany the adapted asset. Fixed-foot collision-aware IK
preserves the original dummy; reference validation passes (maximum foot error
0.355 mm, penetration 0.378 mm). No noncommercial source was used.

Implemented native-hinge reference observations, partial resets, upstream DeepMimic
reward and PPO, deterministic native evaluation and ONNX export. Nine regression
tests and all 12 task preflight checks pass. Production rig, bridge, reward code
and accepted/shipped policies are unchanged.

**First run:** 128 worlds, 32 PPO iterations, 131,072 samples, 67.8 seconds. Across
eight fixed phases, native MuJoCo standing time rises from 0.448 to 1.006 seconds;
Newton gives 1.010 seconds. **All eight still fail the three-second episode.**
Tracking error over the longer trajectories increases, so this is pipeline and
early learning evidence, not a claim of realistic movement. ONNX action error is
5.78e-8. Results/checkpoints/export are under `logs/mimickit-stand/ppo-smoke-01/`.

**Next:** bounded Stand learning-curve evaluation with fixed settings; matching
Godot observation/reference adapter and replay before policy promotion. No Perturb
or long unattended training yet. The exported 300-channel actor is experimental
and cannot replace the existing 105/140-channel Godot policies.

## 2026-09-21: MimicKit compatibility setup passes; Stand task is next

Implemented setup stages 1–5 in [mimic/README.md](mimic/README.md). A separate
Python environment and project-local MimicKit Newton adapter preserve the native
MuJoCo rig, offset hinge joints, passive mechanics, 33 actuators / 30 policy actions,
60% torque authority and two-step action delay. The existing Godot bridge and
accepted/shipped policies are unchanged. The original working tree and accepted
checkpoint are preserved under `logs/mimickit-compat/baseline/`.

Five regression tests and the GPU/native compatibility gates pass: 65 poses,
matching collision eligibility and sampled contacts, delayed controls, a two-world
0.5-second open-loop comparison and reset checks. Pose mapping error is below
1.2e-6 m; the recorded rollout root/joint differences are 1.714e-6 m / 1.749e-5 rad.
This establishes a bounded compatibility result, not learned Stand behavior or
long-horizon sim-to-sim equivalence. GPU reset replay is tolerance-based, not bitwise.

**Next:** select a motion with verified training/commercial-use permissions,
retarget standing/weight shifting into native hinge coordinates, then implement
the Stand reference observations, action space, per-world resets and Godot policy
contract. No external motion/pretrained archive was downloaded and no training or
policy export occurred. Do not resume the rejected demonstration candidate below.

## 2026-09-21: demonstration cloning calms standing but regresses recovery

**Decision: reject the demonstration candidate.** Keep the accepted training actor
`contact_v3_full5m/model_78.pt` and the shipped balance policy. Do not extend this
candidate with a long training session. Rewards, PPO settings, plant, torque limits
and action delay were unchanged throughout this pilot. Nothing was exported.

Added `scripts/recovery_demonstrations.py`, `scripts/learn_recovery_demonstrations.py`
and five data/initialization contract tests. The protocol was written before new
teacher outcomes: first three accepted-policy falls in each Head/Chest approach
stratum, first two for training and third for whole-trajectory validation. Original
benchmark prefixes replayed exactly. Previously saved oracle traces were hash-checked
and reused; new cases used the same bounded CEM/MPC configuration as the feasibility
probe. All commands replayed exactly before sensor/command pairs were retained.

**Teacher coverage:** 16 of 24 cases recover and settle under both metrics, including
the earlier feasibility cases. Ten successful training trajectories provide 1,200
oracle-command pairs from their first 120 control steps. Six successful validation
trajectories provide 720 independent pairs. Failed/unsettled attempts are retained in
the report but excluded from imitation labels. The actor sees only the existing
140-channel `foundation_v2` observation before each command; no projectile state,
future state or optimizer state enters its input. Successful handback states augment
retention targets. Independent references contain 4,096 quiet states (16/16 successful
20-second trajectories, seed 28101) and 4,096 impact states (39/64 successful six-second
trajectories, seed 28103). No validation trajectory enters any training group.

**Fixed fit:** zero-extend the accepted actor/critic inputs, retain exploration, and
optimize only actor weights for 2,000 Adam updates at 3e-5, seed 29021. Each update
samples 256 examples per group; demonstration/quiet/impact MSE weights are 1/2/1.
Critic and exploration parameters were verified unchanged. Scheduled candidates at
250/500/1000/2000 updates are ranked by whole-trajectory validation MSE subject to
quiet MSE <= 1e-4 and impact retention MSE <= 4e-4. None passes these filters.
The predeclared fallback locks `model_2000.pt` for diagnostic evaluation and rejection;
its hash was saved before any closed-loop result or fresh benchmark.

Training MSE drops from 0.131908 to 0.028276 and validation MSE from 0.131243 to
0.072097. Since commands are clipped when executed, the corresponding executed-action
MSE is also measured: training 0.093943 -> 0.026949, validation 0.096965 -> 0.068001.
The improvement therefore is not solely removal of harmless out-of-range output.
However, quiet retention MSE is 0.000145 and impact retention MSE 0.002294. From the
same 24 captured contact states, the learned actor produces **0 survivors/recoveries**,
including all ten successful training demonstrations. The optimizer recovers 16;
the learned policy does not reproduce those feedback trajectories.

**Standard CPU gates, survival / historical recovery / measured-contact recovery:**

- Accepted actor: **307 / 281 / 296** out of 512.
- Shipped actor: **302 / 277 / 288** out of 512.
- Demonstration candidate: **236 / 193 / 194** out of 512.

All three pass quiet standing: 16/16 survive and settle over 40 seconds. The candidate
loses 71 survivors and 88 recoveries versus accepted; recovery gains 39 worlds and
loses 127, paired z=-6.830. Against shipped, recovery z=-6.481. Both comparisons reject
continuation/promotion through the existing survival/recovery gates.

**Fresh balanced development bank:** seed 155921, 768 trials, all 48 target/approach
strata, eight 96-world shards, fixed accepted-policy projectile launches. Accepted
versus candidate: survival **465 vs 358**, historical recovery **427 vs 289**, and
measured-contact recovery **449 vs 294**. Historical recovery gains 65 worlds and
loses 203, z=-8.430. Contact is observed at control boundaries in 768 accepted and
766 candidate trials; no trials are excluded. The final held-out seed **196613 remains
unused**, because prerequisite gates already reject this candidate. The development
seed is now inspected and must not be represented as unseen in future experiments.

**Movement:** quiet mean action change decreases from 0.03910 to 0.00300 and contact
sliding from 0.33245 to 0.27505 m/world. Among the 293 fresh-bank worlds both policies
survive, action change decreases from 0.05691 to 0.03824 and contact sliding from
0.68848 to 0.63229 m/world. These are narrower improvements in motion metrics, not a
successful recovery policy or proof of perceptually realistic movement. All-world
sliding totals alone would be misleading because the candidate falls more often.

**Interpretation and next direction:** the expanded oracle evidence supports physical
recoverability in more sampled states. Direct offline cloning of these trajectories
does not transfer the recovery skill to the current reactive actor. Reduced action
error on teacher states is insufficient evidence of stable closed-loop behavior.
This does not isolate observation insufficiency, residual fitting error, limited
coverage or divergence onto states absent from demonstrations as the sole cause.
A useful next experiment would collect expert corrections on the student's own
rollouts, protect the accepted standing/recovery behavior, and require successful
closed-loop reproduction before spending time on a broad benchmark. Further reward
tweaks or more epochs of this rejected fit are not supported by this result.

Seven targeted tests pass, covering snapshot/physics parity, persistent native state,
pre-action labels, disjoint trajectory splits, input migration and retention selection.
`git diff --check` passes. Artifacts are in `logs/recovery-demonstrations-20260921/`:
protocol, immutable dataset inventory, teacher traces, dataset, fit history, locked
checkpoint, contact-state rollouts, all three standard scorings, fresh benchmark,
matched-survivor motion comparison and final decision. No commit or push.

## 2026-09-21: bounded trajectory search establishes five feasible recoveries

The frozen accepted actor is still `contact_v3_full5m/model_78.pt`. Added the offline
`scripts/probe_recovery_feasibility.py` diagnostic and two contract tests. No rig,
actuator authority, action delay, observation contract, reward, PPO setting or policy
weight changed in this experiment.

Eight cases were selected before optimization: the first recorded accepted-policy
fall in each Head/Chest x four approach sectors of the previous benchmark's selection
split. Original 96-world inference batches were replayed. A read-only observer
captured the first control boundary after positive-force ball/body contact, including
both queued commands and the native derived state. Every benchmark output array
matched the saved uninstrumented result exactly. All four head shots contacted Head;
chest +X/-X contacted Chest, +Y contacted UpperArm_R, and -Y contacted UpperArm_L.

One fixed full-state CEM/MPC configuration used a 60-control-step horizon, seven
residual knots, 96 candidates, four iterations and replanning every six steps.
Existing normalized action bounds, 0.6 torque authority, two-command latency and
projectile gravity schedule were retained. Only joint actuator controls changed.
After 120 control steps (about two seconds), control returned to the accepted actor.
Trials ended at first fall or approximately six seconds after contact. The physical
search surrogate was separate from the unchanged RL reward and recovery gates.

**Results:** both original batch-action replay and singleton actor baselines fall
in all eight cases. Optimized control yields **6/8 survivors and 5/8 settled
recoveries**, agreeing under both historical and measured-contact recovery metrics.

- Head +X, trial 0: recovered; final measured-contact settled hold 2.15 seconds.
- Head +Y, trial 17: recovered; settled hold 1.32 seconds.
- Head -X, trial 32: falls 2.28 seconds after contact, after policy handback.
- Head -Y, trial 48: recovered; settled hold 3.87 seconds.
- Chest +X, trial 66: falls 3.68 seconds after contact, after policy handback.
- Chest +Y, trial 80, actual right-arm contact: survives but does not finish settled.
- Chest -X, trial 96: recovered; settled hold 3.67 seconds.
- Chest -Y, trial 112, actual left-arm contact: recovered; settled hold 4.57 seconds.

All eight optimized trajectories independently replayed every saved state and applied
control exactly. All traces were finite, with peak absolute body generalized velocity
75.85 or less, below the existing 300 divergence ceiling. Every case completed all
20 plans within its fixed 80-second budget: 61,440 candidate sequences and 158.4
seconds of optimization/evaluation wall time in total, excluding prefix capture.
Two targeted tests pass: exact snapshot/observation/continuation preservation, and
threaded native-rollout parity across delayed/clipped commands, body contact and
projectile gravity transition. No long training run or policy export was launched.

**Interpretation:** the current body can recover from five sampled impact states
that defeat the accepted policy, without stronger motors or external assistance.
Control/learning is therefore part of the limitation. This does not clear every
structural or sensor issue: the oracle uses full simulator state and offline compute,
and the three cases without settled recovery remain inconclusive. The failure-only
sample is diagnostic, not an estimate of population success. Realistic movement and
learned-policy improvement are not established by this result. Motion metrics for
early-falling baselines and six-second survivors have different exposure lengths.

**Next recommendation:** a bounded demonstration-learning experiment using successful
optimized recoveries, retaining quiet-standing behavior and evaluating through the
existing CPU gates plus a newly sealed balanced bank. Demonstration quantity,
observation sufficiency and motion quality must be addressed before that experiment;
five successful traces alone are not a production training set. Keep rewards and PPO
settings frozen rather than launch another long continuation on the rejected seeds.

Artifacts: `logs/recovery-feasibility-20260921/` contains the frozen protocol, case
snapshots, baseline/optimized command and state traces, per-case results, summary and
SHA-256 inventory. `case-0-comparison.mp4` shows accepted versus optimized head +X
recovery; the baseline frame is held after its first fall. README documents reproduction.
No commit or push.

## 2026-09-21: balanced CPU benchmark rejects a reliable restart improvement

Added `scripts/benchmark_perturb.py` and `rl/test_impact_benchmark.py`. The reusable
benchmark saves a protocol before execution, checkpoint/plant hashes, fixed projectile
launches, per-trial outcomes and all 48 target/approach breakdowns. Default splits each
contain **768 trials: 12 intended targets x four approach sectors x 16 samples**.
Selection seed is 104729, held-out seed 130363. Each of eight 96-world shards includes
two samples of every stratum; four CPU workers run independently. The accepted actor
generates launches, which candidates replay exactly. Physics/rewards, observation
contracts and existing continuation/promotion gates are unchanged.

The registered selection rule chooses most historical recoveries, then most survivors,
then label. Only the selected candidate and accepted actor see the held-out split;
`selection-lock.json` is written first. Supplementary confirmation requires no survival
regression and more recoveries with paired z >= 2 on both splits. It cannot promote a
policy. The tool now also accepts distinct custom split seeds for future untouched
confirmation sets; the seeds used here are no longer unseen after inspecting results.

**Selection, survival / recovery out of 768:**

- Accepted `contact_v3_full5m/model_78.pt`: **466 / 420**.
- Protected training seed 443: **468 / 424**.
- Protected training seed 457: **455 / 414**.

Seed 443 is selected, but its recovery gain is only 4 worlds: 82 gained, 78 lost,
z=0.316. Measured-contact recovery ties accepted at 445/768. Mean native episode reward
changes by +0.614. Seed 457 has recovery z=-0.487 and 11 fewer survivors; it is not
evaluated on the held-out set.

**Held-out, accepted versus selected seed 443:** survival **476 versus 480**, historical
recovery **438 versus 430**, measured-contact recovery **457 versus 457**. Historical
recovery gains 85 worlds and loses 93 (z=-0.600). Mean native reward rises by 6.399,
but neither recovery definition improves. Mean action change is 0.06562 versus 0.06565
and contact sliding is 0.73643 versus 0.73851 m/world. No realism improvement is established.
Both selection and held-out confirmation flags are false.

**Failure distribution:** upper-body targets are consistently difficult. For the
accepted policy, selection/held-out recovery is Head 6/64 and 9/64, Chest 11/64 and
14/64, Spine 13/64 and 17/64. Left-thigh recovery is 55/64 in both splits; right-thigh
recovery is 54/64 and 49/64. These labels describe intended targets, not necessarily
the first contacted body. Seed 443's per-body changes are mixed: head recovery drops
in both splits, chest falls in selection but rises on held-out, and spine does the
reverse. No individual 16-trial target/sector cell should drive a reward-weight change.

Contact presence at control boundaries is observed in 767/768 accepted selection trials
and all other trials. An unobserved control-boundary contact is not proof of a miss;
no trials were removed after observing outcomes. All 3,840 policy trials completed
with finite outputs. Artifact hashes, complete unique trial IDs, aggregate counts,
and the frozen selection lock were verified. Four targeted tests pass, covering
stratification, split independence, ranking, trial aggregation and physical launch replay.
`git diff --check` passes. See README's stratified-benchmark section for reproduction.

**Decision:** keep the accepted training actor and shipped policy. The strong gains on
the earlier 128-scenario diagnostic did not generalize to this larger, balanced bank.
Do not extend either protected candidate on the strength of that diagnostic, or make
restart warm-up a production default. This benchmark localizes weak impact classes;
it does not establish whether their cause is physical recoverability, missing policy
information or the learning objective. No further training was launched.

Run artifacts: `logs/perturb-benchmark-20260921/{protocol,selection-lock,decision}.json`,
both split directories' `summary.json`, scenario NPZs, per-policy shard JSONs and hash
inventories. All processes finished. No policy export, commit or push.

## 2026-09-21: matched CPU/Warp impacts do not show GPU-only learning gains

Completed a frozen-policy transfer audit on **128 new scenarios, seed 977**, for the
accepted `contact_v3_full5m/model_78.pt` and protected restart policies from training
seeds 443 and 457. No training or production configuration changes were made.

**Isolation:** all runs receive identical initial qpos, zero qvel, launch steps,
projectile positions and velocities. Shared inputs are stored at float32 precision;
both backends report zero initial-qpos error against that manifest. The accepted CPU
rollout defines world-space launches at 1.5–1.8 seconds; every other rollout replays
them, with one 6 m/s projectile per world, six-second episodes and no automatic
resets. All policy inference runs on CPU using deterministic means. The accepted
actor consumes its original 105 channels; learned actors consume foundation_v2's
140 channels. Both simulators expose the same 140-channel environment contract.
All 128 projectiles register body-contact presence at physics-substep resolution in
each of the six closed-loop runs. All recorded arrays are finite and Warp constraint
buffer checks pass at every control step.

**Results, survival / legacy recovery out of 128:**

- Accepted: CPU **81 / 66**, Warp **87 / 77**.
- Protected seed 443: CPU **88 / 80**, Warp **93 / 78**.
- Protected seed 457: CPU **90 / 85**, Warp **92 / 84**.

Loaded-contact recovery counts are CPU 73/85/88 and Warp 82/85/87, respectively.
On CPU, learned recovery gains/losses versus accepted are 24/10 (z=2.401) and 27/8
(z=3.212). Warp gives 16/15 (z=0.180) and 17/10 (z=1.347). These diagnostic comparisons
are not promotion gates: they use a new scenario bank with fixed world-space launches,
and omit the required quiet-room gate. Prior continuation failures remain unchanged.

**Reward:** mean native episode reward is CPU 1432.06/1527.64/1542.35 and Warp
1510.39/1531.69/1542.12 (accepted/443/457). Reward is counted through the first native
training termination, including its terminal penalty, with subsequent rewards masked
out. Post-launch reward changes have the same direction. With gamma=0.99 discounting,
CPU means are 453.58/459.42/460.39 and Warp 460.10/459.77/461.16: the Warp discounted
changes are small relative to paired standard errors. These finite deterministic
returns are not estimates of the full stochastic PPO objective or bootstrapped value
targets. This sample does not demonstrate that the reward systematically prefers
worse recovery, nor that learned gains exist only on the training engine.

**Numerical controls:** an eight-world CPU repeat reproduces all recorded trajectory
arrays exactly; replaying its fixed action sequence on CPU also reproduces them exactly.
A full 128-world Warp repeat returns 84 survivors / 79 recoveries versus its original
87/77. Fourteen individual recovery outcomes flip (8 gains, 6 losses). CPU versus
original Warp flips 23 accepted-policy recovery outcomes (17 gains, 6 losses).
Thus GPU repeat variability is material and a cross-backend trajectory difference
alone is not evidence of a sensor or physics implementation bug.

Before launches, the accepted closed-loop CPU/Warp pelvis gap at step 89 is median
0.0000061 m, p95 0.00162 m; by the last step it is median 0.0987 m. Warp-versus-Warp
final median gap is 0.0662 m. Replaying the exact CPU action sequence on Warp without
feedback produces a final median gap of 1.217 m and no survivors. The replayed actions
match exactly; this is an open-loop sensitivity diagnostic, not a deployable-policy
score or proof that a controller cannot transfer.

**Decision:** the simple explanation that training improves Warp recovery but loses
those gains on CPU is not supported here. Both learned policies improve recovery and
undiscounted reward on the fresh CPU scenarios, despite their earlier gate regressions.
Behavior varies substantially by scenario and numerical trajectory; no new foundation
defect or reward-weight correction is established. Retain the accepted training actor
and shipped policy. Before more PPO tuning or a longer run, establish a larger fixed
CPU validation bank stratified by target body and heading, with held-out scenarios,
to measure which impacts improve and regress. Do not select a new policy from this
single favorable diagnostic batch. No further experiment has been launched.

Reproduction: `logs/perturb_transfer_audit.py --worlds 128 --out logs/transfer-audit`,
then `logs/analyze_transfer_audit.py` and `logs/transfer_repeat_controls.py`.
Evidence is in `logs/transfer-audit/`: hashed manifest and scenarios, six policy/backend
JSON results and NPZ trajectories, `comparison.json`, `repeat-controls.json`, the Warp
repeat and fixed-action replay. CPU integrity evidence is under
`logs/transfer-audit-smoke/cpu-repeat/`. All runs completed; `git diff --check` passes.
No production edits, policy export, commit or push.

## 2026-09-20: second restart seed does not reproduce the continuation win

Repeated the immediate/protected restart comparison with training seed **457** from
the same `foundation_v2_pilot/model_50.pt` source. Control:
`2026-09-20_23-18-15_foundation_restart457_control/model_25.pt`; protected:
`2026-09-20_23-21-29_foundation_restart457_protected/model_50.pt`.
Settings remain 4096 worlds x 16 steps, 6 m/s and reference coefficient 1. The control
runs 25 actor-learning rounds; protected runs 25 critic-only plus 25 actor-learning
rounds in one process. Both finish below their five-minute caps (1.7/3.5 minutes).

Initial networks, teacher and reference targets match. Control settings differ from
the seed-443 control only in seed and run name. Within the new pair, only warm-up,
total rounds and run name differ. The protected round-25 actor and exploration weights
are byte-identical to the source, actor optimizer state remains empty, and readiness
passes with pre-fit/post-fit EV 0.904/0.905. It then performs 468 actor updates versus
the control's 467. Critic optimizer counts are 2000/1500 including the source's 1000.
Final protected pre-fit/post-fit EV is 0.910/0.912. Thus the intended intervention ran
correctly; this is not a failed or incomplete warm-up experiment.

**Both new checkpoints fail continuation against the accepted actor.** On the unchanged
512-world seed-17 gate, control returns **304 survivors / 282 recoveries**, protected
**303 / 283**, versus accepted **307 / 281**. Protected recovery pairs versus control
are 52 gained and 51 lost (z=0.099); versus accepted, 54/52 (z=0.194). Promotion also
fails: protected recovery z versus shipped is 0.567, below the required 2.
Measured-contact recovery is control 296/512, protected 294/512, accepted 296/512.

On evaluation seed 71 (128 worlds), control returns **76 / 71**, protected **70 / 65**,
versus accepted **73 / 71**. Protected recovery versus control gains 9 and loses 15
(z=-1.225). Both retain perfect 16-world quiet survival/recovery over 40 seconds.
Quiet action change is control 0.04726, protected 0.05379, accepted 0.03910; protected
main impact action change/contact slip is 0.06712/0.71559 m per world versus accepted
0.06617/0.70827. This repeat provides no improved realism signal either.

**Conclusion and next decision:** seed 443's protected 311/287 main result did not
repeat; seed 457 gives 303/283. Both protected runs lose independent-seed recovery
(66 and 65 versus accepted 71). Two training seeds do not establish population harm,
and the reused evaluation worlds must not be pooled as independent samples. They
also do not support making restart warm-up a production default or extending these
runs. Keep `contact_v3_full5m/model_78.pt` as the accepted training actor and the
shipped policy unchanged. The earlier seed-443 checkpoint retains its recorded main
gate pass, but is not validated as a reliably better starting point.

Before another parameter change, the next recommended investigation is matched,
deterministic impact rollouts on CPU and Warp for the accepted and learned policies:
compare recovery and accumulated reward to separate simulator transfer from an
objective that fails to improve recovery. Existing state-level observation/reward
parity checks do not establish whole-trajectory equivalence. This is an investigation
proposal, not evidence that either backend is defective; no further run was launched.

Evidence: `logs/compare_protected_restart.py` now accepts explicit control and artifact
prefixes; `logs/foundation-restart457-{control,protected}-{train,512,seed71}.log`,
the corresponding evaluation JSON files, both initial fingerprint files, and
`logs/foundation-restart457-protected-{integrity,comparison}.json`. All six training
and evaluation processes finished successfully. No production code/default changes,
policy export, commit or push. `git diff --check` passes.

## 2026-09-20: protected restart passes continuation, evidence remains mixed

Completed the approved restart experiment from
`2026-09-20_22-19-35_foundation_v2_pilot/model_50.pt`: 25 critic-only rounds followed
by 25 actor-learning rounds, in one uninterrupted process. Run:
`2026-09-20_22-58-46_foundation_protected_restart/model_50.pt`.
Seed 443, 4096 worlds x 16 steps, fixed 6 m/s, reference coefficient 1 and all other
training settings match the immediate-restart retention-on control. Only run name,
total rounds (25 to 50) and explicit warm-up (inherited zero to 25) differ.
Training finished in 3.2 minutes, below the five-minute cap.

**Integrity and critic diagnostics:** initial model, reference targets and teacher match
the control. At round 25, all actor and exploration tensors remain byte-identical to
the source, actor Adam state is empty, and critic Adam advances from 1000 to 1500 steps.
Warm-up ends on schedule with pre-fit/post-fit EV 0.895/0.897, above the existing 0.5
readiness gate. The first fresh rollout again has pre-fit EV -4.945; the actor receives
no updates during this mismatch. Rounds 26–50 perform 461 actor optimizer updates and
500 further critic updates. Final pre-fit/post-fit EV is 0.910/0.912. These EV values
describe bootstrapped targets, not ground-truth value accuracy. Warm-up also advances
environment age and random streams before actor learning; this is not an isolated
proof that critic fitting alone causes any behavioral difference.

**Unchanged CPU gates:** quiet survival and recovery remain 16/16 over 40 seconds.
Main seed 17 returns **311 survivors and 287 recoveries out of 512**, versus accepted
307/281 and immediate-restart control 294/276. Measured-contact recovery is 301/512
versus accepted 296/512. Thus the existing main continuation gate passes. Promotion
does not: paired recovery z versus shipped is 0.990, below 2 (56 gained, 46 lost).
Versus accepted, recovery gains/losses are 60/54 (z=0.562); versus immediate restart,
59/48 (z=1.063). These single-run differences do not establish a reliable improvement.

**Independent seed 71 is worse:** 69 survivors and 66 recoveries out of 128, versus
accepted 73/71 and immediate restart 71/69. Recovery versus accepted gains 10 worlds
and loses 15 (z=-1.0). Impact action change is 0.06535 versus accepted 0.06617, and
contact slip is 0.69853 versus 0.70827 m/world on the main benchmark. However, quiet
action change rises to 0.06513 from accepted 0.03910 (control 0.04620). These mixed
results do not establish improved visual realism or generalization.

**Decision:** retain this checkpoint as a continuation-eligible experimental candidate,
but keep the previously accepted actor as the stable reference and leave the shipped
policy unchanged. Do not infer a skeleton defect, change reward weights, or start an
hours-long run from this result. The next recommended experiment is a second training
seed for the same protected/immediate restart comparison, with the same CPU gates,
to check whether the main improvement repeats without the independent-seed loss.
No additional experiment has been launched. No production default was changed.

Evidence: `logs/compare_protected_restart.py`, `logs/foundation-restart-train.log`,
`logs/foundation-restart-{initial,integrity,comparison}.json`, and
`logs/foundation-restart-{512,seed71}.{json,log}`. All training and evaluation processes
completed successfully. No policy export, commit or push.

## 2026-09-20: reference-retention ablation and restart audit

Two bounded runs start from `2026-09-20_22-19-35_foundation_v2_pilot/model_50.pt`,
after its native 16-world, 40-second quiet gate passes. Each performs 25 training rounds,
4096 worlds x 16 steps, seed 443, 6 m/s, the existing reward and 0.1 temporal smoothing.
Only `reference_coef` differs: 1 or 0. The original teacher, data, actor, exploration,
critic and optimizer state are retained. This is a settings-matched comparison; disabling
retention does not disable entropy or temporal smoothing. Both finish in under two training
minutes, with 1,638,400 transitions each. Normal KL stops produce 467 actor updates with
retention and 455 without; both perform 500 additional critic updates.

Runs: `2026-09-20_22-35-39_foundation_ref1_ablation/model_25.pt` and
`2026-09-20_22-38-13_foundation_ref0_ablation/model_25.pt`.

**Isolation audit:** fingerprints confirm identical initial networks, teacher and reference
targets. Saved settings differ only in run name and reference coefficient. First-rollout
reset masks match, but numerical trajectories are not bit-identical. An independent frozen
GPU repeat confirms identical initial legacy inputs and small initial load differences
(maximum 0.000102 kN). Over 16 steps, observation RMS difference is 0.000816 and action RMS
difference 0.000552, with occasional larger excursions (maximum action difference 0.148).
Consequently a small score difference between these single runs is not conclusive evidence
about the retention coefficient.

**Identical-batch objective check:** both objectives are also applied to one shared frozen
rollout with the same minibatch random seed. Initial actor-mean gradient norms are PPO 3.118,
full-corpus retention 1.176 and weighted smoothing 0.0226. PPO/retention cosine is -0.073;
PPO/combined cosine is 0.933. This first, pre-impact batch does not show retention dominating
the actor direction; it cannot establish the relationship during recovery. Critic updates
are identical between the two common-batch arms.

**Restart finding:** the source ends with pre-fit critic EV 0.912, but its first fresh rollout
has EV approximately -4.95, signed value bias -37.3 and RMSE 61.7. Both arms update the actor
immediately because the saved warm-up counter is zero. Pre-fit EV remains negative at the
logged rounds 5, 10 and 15, then reaches about 0.87 at round 20. These are bootstrapped-target
diagnostics, not ground-truth value accuracy. Each new process recreates environments at rest;
clock staggering also offsets the shot schedule, so initial shots still wait 4–7 physical
seconds. Twenty-five rounds cover only 6.667 seconds per world across resets. This motivates
testing actor-frozen settling/critic adaptation on resume rather than inferring that an
already-adapted saved critic is ready for newly reset states.

**CPU result: both rejected.** Quiet-room survival and recovery remain 16/16 over 40 s.
On the unchanged 512-world seed-17 impact benchmark, retention-on returns 294 survivors
and 276 recoveries; retention-off returns 298/278. The accepted actor remains 307/281.
Both fail the no-regression continuation gate and neither qualifies for promotion.
Retention-off versus on has 52 newly recovered worlds and 50 lost recoveries (z=0.198):
no persuasive recovery benefit from disabling the teacher. Against the accepted actor,
recovery z is -0.488 with retention and -0.302 without; these are gate failures, not proof
of statistically conclusive harm. Independent seed 71 returns 71/69 with retention and
72/66 without, versus the accepted 73/71 (survival/recovery counts, 128 worlds).

Quiet action change is 0.04620 with retention and 0.02597 without, versus accepted 0.03910.
The smoother quiet actions without retention do not establish improved impact recovery or
overall visual realism. Impact action change is 0.06751/0.06564 (on/off), and measured contact
slip is 0.71847/0.72138 m/world versus accepted 0.70827.

**Next decision:** keep the accepted actor and do not extend either ablation checkpoint.
Keep retention enabled pending better evidence. Test a protected restart from the same
foundation model 50: use the existing `--critic_warmup_iters 25` to freeze the actor while
the new environments settle and the critic adapts, then allow 25 actor-learning rounds.
Use `--iterations 50`, the same seed 443, and the existing EV readiness gate (extend the
critic-only phase if it does not pass). Compare against this retention-on control with
the unchanged CPU gates. This isolates a concrete lifecycle hypothesis before changing
rewards, geometry or committing to a longer run; it is not yet a production-default change.

Artifacts: `logs/perturb_reference_ablation.py`, `logs/compare_reference_ablation.py`,
`logs/perturb_reference_signal.py`, `logs/foundation-ref{0,1}-{initial,train,512,seed71}.*`,
`logs/foundation-reference-signal.{json,log}` and `logs/foundation-reference-comparison.json`.
Production training defaults, reward weights, model geometry and shipped policy are unchanged.

## 2026-09-20: foundation observations and protected migration

The foundation audit found an observation gap, not evidence that the skeleton needs rebuilding.
The policy still saw foot-origin height as contact after rewards had moved to measured floor
load. It also lacked the oldest action in the two-step actuation queue. The existing linear
velocity channel uses MuJoCo's subtree-COM spatial reference, rather than the pelvis origin.
These are now addressed by the opt-in `foundation_v2` contract (140 observations); the original
105 channels remain intact for legacy weights. See README for the migration and validation commands.

**Physical checks:** body mass 69.6259 kg; positive, physically valid principal inertias;
no rest-pose self-contact; all 30 policy actuators have the expected force and acceleration
direction. The unpowered body crosses the shared fall-height threshold after 0.996 s without
non-finite state. Maximum rest-pose joint gravity demand is 3.4% of available policy torque;
this scale check is not a proof of balance feasibility under impact. Pelvis-origin velocity
matches native `mj_objectVelocity` with `mjOBJ_XBODY`.

**Implementation checks:** all 80 Python tests and all 126 managed/native C# tests pass.
CPU/GPU checks cover measured loads, body-origin velocity, action ordering and partial reset.
C# native fixtures compare all 140 observations across 21 consecutive states in three initial
poses, plus independent flat/heel/toe load fixtures. Scratch ONNX export agrees with PyTorch
within 1.192e-7. ABI offsets were regenerated against MuJoCo 3.8.1, not hand-entered.

**Game/training audit:** the scene's 3 s interval is applied after a 2 s shot-resolution window,
so successful shots launch roughly 5 s apart, within training's 4–7 s range. Game aim lead
(capped at 0.5 m), speed jitter (6 +/- 0.25 m/s) and minimum launch height (0.1 vs 0.15 m)
remain distribution differences. Neither the projectile profile nor the fixed CPU benchmark
was changed during this sensor experiment.

**Pilot:** `2026-09-20_22-19-35_foundation_v2_pilot`, 4096 worlds x 16 steps, seed 431,
6 m/s, five minutes. The source is the accepted actor's quarter-noise frozen checkpoint
`2026-09-20_20-37-48_contact_v3_hard6_quarter/model_50.pt`. New actor columns start at zero;
the critic and both optimizers reset. The original incumbent remains the reference teacher,
with new physical observations recollected on CPU seeds 101/103. Model 50 retains every old
actor weight and exploration value bit-for-bit and all new actor columns are zero; it completes
critic warm-up at EV 0.913. Model 75 completes 25 actor-learning rounds (427 actor optimizer
steps; 1500 total critic steps), with nonzero weights on the added inputs.

**CPU result: rejected.** The migrated frozen actor (model 25, identical actor to model 50)
exactly reproduces the accepted actor's 512-case outcomes: 307 survived, 281 recovered,
296 measured-contact recoveries, and identical action-change/slip diagnostics. No paired
survival or recovery outcome changes, so migration itself preserves this baseline.
The learned model 75 keeps quiet-room survival/recovery at 16/16 over 40 s, but returns
297/512 survival, 268/512 recovery and 278/512 measured-contact recovery. Paired recovery
gains/losses versus the accepted actor are 53/66 (z=-1.192); this is not statistically
conclusive harm, but it fails the established no-regression continuation gate. Independent
seed 71 is also lower: 68/128 survived and 64/128 recovered, versus 73/71. Quiet action change
rises from 0.03910 to 0.05310; impact action change rises from 0.06617 to 0.06848 and measured
slip from 0.70827 to 0.72620 m/world. No shipped policy or model has been replaced.

**Next decision:** do not extend model 75 or launch an overnight chain. The physical checks
support retaining the current rig while investigating learning. The new sensor contract is
implemented and tested, but one short pilot does not establish its learning benefit. The next
bounded experiment should isolate the PPO objective from the fixed reference-retention loss
using identical collection settings and the actor-frozen, critic-adapted v2 model 50 as the
common start. Its normal quiet-room seed gate must still pass before training. Keep the original
teacher fixed and apply the existing CPU gates; do not change reward weights or geometry at
the same time. The accepted legacy training checkpoint remains `contact_v3_full5m/model_78.pt`.

Artifacts: `logs/foundation-audit.json`, `foundation-*-tests.log`, `foundation-dotnet-all.log`,
`foundation-v2-reference.pt`, `foundation-v2-{train,migration-integrity,pilot-integrity}.*`,
`foundation-v2-frozen-512.*`, `foundation-v2-pilot-{512,seed71}.*`,
`foundation-v2-comparison.json` and `foundation-export/`.

## 2026-09-20: PPO horizon and gradient investigation

The accepted training actor is still `2026-09-20_19-14-28_contact_v3_full5m/model_78.pt`.
No replacement has been exported. The PPO surrogate, Gaussian log probabilities/KL and
timeout-aware GAE review did not identify an arithmetic defect. This investigation measures
the quality of the learning signal before changing more reward weights.

**Observability correction.** PPO now reports pre-fit `value_ev_before`, signed
`value_bias_before` (prediction minus target), `value_rmse_before` and raw `advantage_std`.
The trainer prints these beside the existing post-fit `ev`. These are diagnostics against
bootstrapped batch targets, not held-out critic accuracy; neither is proof of useful recovery
gradients. Warm-up/optimizer behavior is unchanged. A regression test demonstrates that EV
can equal one while values have a constant error of five. All 22 training-safety and ten
policy-reference tests pass.

**Frozen trajectory probe.** Source: the quarter-noise, critic-adapted update-50 checkpoint
`2026-09-20_20-37-48_contact_v3_hard6_quarter/model_50.pt`. Its mean actor is exactly the accepted
actor; no rejected learned actor is reused. Seed 419, 512 GPU worlds, 768 control steps
(12.80 seconds/world), 6 m/s impacts every 4–7 seconds, no optimizer updates. Actual ball/body
contact is measured each physics substep by the existing exposure probe. The same trajectory
is evaluated with GAE cuts every 16 or 128 steps and with full-trajectory GAE; gamma/lambda
remain 0.99/0.95. The final 128 steps are excluded from the comparison of advantages.

Within two seconds of contact, 16-step advantage signs disagree with full-trajectory GAE
on 18.30% of samples; 128 steps reduce that to 3.61% (correlations 0.844 and 0.976).
This establishes sensitivity to truncation, not that full-trajectory estimates are ground
truth. Longer lambda-one return diagnostics also bootstrap at timeouts and the final boundary.
The [GAE formulation](https://arxiv.org/abs/1506.02438) trades bias against variance; a longer
trace alone need not improve a learned controller.

At three 65,536-transition windows, initial PPO gradient norms were 3.49–3.66 for 16-step
targets, versus 0.011–0.014 for weighted temporal smoothing and 1.08–1.12 for the fixed
reference. Smoothing does not dominate these instantaneous gradients. Norms do not measure
the eventual Adam update or establish whether an auxiliary objective helps behavior.

**Matched horizon experiment.** Both arms use 512 worlds x 128 collected steps, 12 PPO rounds,
seed 419, the same source/optimizer/teacher, quarter noise, fixed 6 m/s, 20-second episodes,
LR ceiling `1e-6`, KL target 0.001 and smoothing 0.1. Only GAE truncation differs: the control
cuts at 16-step boundaries; treatment uses all 128. Both finish below the five-minute cap.
The KL guard permits 211 and 224 actor optimizer steps respectively; both take 240 critic
steps. Teacher targets, variance and provenance remain exact. GPU trajectories are not
bitwise identical across processes; this is one training seed, not a replicated result.

- Control: `2026-09-20_21-09-13_contact_v3_gae16_control/model_12.pt`.
  Main survival/recovery 293/512 and 273/512, versus accepted 307/512 and 281/512.
  Rejected. Independent seed 71: 75/128 and 72/128, versus accepted 73/128 and 71/128.
- Treatment: `2026-09-20_21-12-16_contact_v3_gae128/model_12.pt`.
  Main survival/recovery 306/512 and 287/512. Rejected by the unchanged survival gate,
  despite six additional recoveries. Recovery pairs versus accepted: 52 gained, 46 lost,
  z = 0.606; versus the control: 56 gained, 42 lost, z = 1.414. Neither establishes a
  population improvement. Measured-contact recovery is 294/512 versus accepted 296/512.
  Independent seed 71: 78/128 survived, 71/128 recovered. Quiet survival/recovery stays
  16/16 in both arms. Treatment quiet action change is 0.0436 versus accepted 0.0391;
  hard-hit sliding is 0.702 versus 0.708 m/world.

**Gradient agreement.** A separate frozen repeat splits each window by disjoint worlds,
using common whole-batch advantage normalization. For 512 worlds x 128 steps, the three
half-gradient cosines are 0.249/-0.068/0.061 with 16-step cuts and 0.191/-0.220/-0.035 with
128-step cuts. A further frozen probe at the normal 4,096-world x 16-step batch shape gives
0.011/-0.082/0.119 at control steps 384/576/768, even though pre-fit EV is 0.910/0.898/0.906.
These few snapshots show weak agreement in sampled gradient directions at both shapes;
they do not estimate a population signal-to-noise ratio or prove a specific critic defect.
No learning occurs in these probes, and all constraint-buffer checks pass.

**Larger-batch validation also rejected.** The final trial uses 4,096 worlds x 128 steps
(524,288 transitions/update), retaining the same frozen source and other training settings.
`2026-09-20_21-25-04_contact_v3_gae128_large/model_11.pt` finishes 11 rounds at the end of
the iteration crossing its five-minute budget: 190 actor optimizer steps and 220 critic
steps. Original teacher targets/variance/provenance remain exact; final LR is `6.67e-7`.
Main survival/recovery is 295/512 and 274/512, versus accepted 307/512 and 281/512.
Recovery pairs: 54 gained, 61 lost, z = -0.653. Measured-contact recovery is 287/512.
Quiet survival/recovery remains 16/16, but quiet action change rises to 0.0565; hard-hit
contact sliding falls to 0.690 m/world. Seed 71 gives 75/128 survivors and 70/128 recoveries.
Neither earlier rejected candidate was used as its seed. This is a bounded configuration
validation, not a matched replication of the 512-world experiment: it finishes 11 rather
than 12 rounds and its KL guard permits a different number of optimizer steps.

**Decision.** All three learned candidates fail the unchanged main continuation gate.
Keep the accepted `contact_v3_full5m/model_78.pt`; do not resume or export these trials.
Longer traces and a larger batch have not established a recovery improvement, and these
results do not justify an hours-long run or another reward-weight change. Weak sampled
PPO gradient agreement is measured, but its cause is not yet isolated. The next controlled
test should separate the PPO reward gradient from the fixed-reference retention objective,
using the same validated source and CPU gates, before changing either objective. Reference
gradient magnitude alone is not evidence that removing the reference is safe.

Production defaults, reward weights, physics, actor observations and all acceptance/promotion
thresholds remain unchanged. All processes have finished. No export, commit or push.
`git diff --check` passes; the 32 targeted tests cover the only production change, additional
diagnostics. Frozen probes are separate from the three bounded learning trials.

Evidence: `logs/perturb_ppo_probe.py`, `logs/perturb_ppo_worlds_probe.py`,
`logs/perturb_gae_ablation.py`, `logs/compare_gae_ablation.py`,
`logs/contact-v3-ppo-{probe,signal,worlds}.{json,log}`,
`logs/contact-v3-gae{16,128}-{train,512,seed71}.log`,
`logs/contact-v3-gae{16,128}-{512,seed71}.json`,
`logs/contact-v3-gae-{comparison,training-integrity}.json`,
`logs/contact-v3-gae128-large-{train,512,seed71}.log`,
`logs/contact-v3-gae128-large-{512,seed71,integrity}.json`.

## 2026-09-20: impact exposure measured; hard-stage and reduced-noise trials rejected

The accepted actor remains `2026-09-20_19-14-28_contact_v3_full5m/model_78.pt`.
No replacement is accepted or exported. This investigation used separate diagnostic seeds
and two bounded five-minute training trials; neither passed the existing CPU gate.

**Difficulty and exposure.** A matched CPU sweep on diagnostic seed 223, 128 worlds,
six seconds and one shot measured recovery 125/128 at 2 m/s, 120/128 at 4 m/s, and 75/128
at 6 m/s (97.66%, 93.75%, 58.59%). Survival was 127/128, 124/128 and 81/128. The policy
already handles substantially more than the earlier 2–2.2 m/s training stages. This sweep
is diagnostic; promotion continues to use the unchanged seed-17 benchmark.

`logs/perturb_exposure_probe.py` observes the frozen accepted policy under the GPU training
reset, random episode phase, 4–7 second launch interval and Gaussian action sampling.
Seed 307, 256 worlds, 1,200 control steps (20.0016 seconds per world), no learning.
Ball–body contacts require more than 1 N normal force and are checked at every physics
substep, excluding floor/ball self-pairs and stale contact slots. A kernel fixture verified
those exclusions, and all probes passed the constraint-buffer check. A hit is counted once
per launched ball; some launches can still be pending or miss at the measurement boundary.

At 400 steps, only 5.79%/9.05%/9.39% of transitions were within two seconds of a detected
impact at 2/4/6 m/s. At 1,200 steps these shares were 21.74%/21.91%/19.89%; all 256 worlds
had experienced contact. Full runs therefore do obtain impact experience. Pre-launch
transitions still occupied 46.9%/50.7%/61.7%, including repeated waits after resets.
Detected hit counts were 629/640/651 from 690/671/678 launches. Fall counts were 6/68/249,
with 3/66/246 following contact on the most recent shot. These are repeated-episode counts,
not survival percentages or the deterministic CPU acceptance metric.

**Exploration control.** No completed sampled episode met the strict settled-recovery hold
at the original per-joint noise, even at 2 m/s. Additional frozen-actor 6 m/s controls used
the same seed and probe with one-quarter noise and zero noise. Settled episode endings:
0/414 at full noise, 37/380 at quarter noise, 38/381 at zero noise; fall counts 249, 207, 219.
All controls retain the original mean actor and consume the same initial sampling stream;
subsequent resets and shots diverge, so these aggregate counts are not paired trials.
They support investigating reduced exploration, not a claim of improved learned recovery.

**Training support added.** `train.py --exploration_scale <factor>` explicitly scales a
seed's learned per-joint standard deviations, preserving actor means and mean-parameter
Adam state. Only std Adam state is cleared. Perturb forces at least 50 critic-only updates
and retains the explained-variance gate; ordinary resumes inherit the saved noise without
reapplying the factor. The original teacher variance/targets remain unchanged. Invalid
factors, missing seeds and mixing with `--reset_std` are rejected. Startup logging now shows
actual policy std rather than the unused fresh-policy `init_std`. Defaults are unchanged.
All 21 training-safety and 10 policy-reference tests pass; three new tests cover scaling,
optimizer isolation, warm-up, identity scaling, normal resume and invalid requests.

**Trial A: game-strength impacts with original exploration.**
`2026-09-20_20-21-15_contact_v3_hard6/model_78.pt` used fixed 6 m/s, all other settings
unchanged, 50 critic-only updates then 28 actor updates (560 actor optimizer steps) within
five minutes. The update-50 actor/std were verified byte-identical to the accepted seed;
critic explained variance was 0.904 at that boundary. Main survival/recovery declined
307/512 -> 290/512 and 281/512 -> 272/512 (recovery pairs 42 gained, 51 lost, z = -0.93).
Seed 71 declined 73/128 -> 69/128 and 71/128 -> 63/128. Quiet remained 16/16, but action
change increased 0.0391 -> 0.0989. Rejected.

**Trial B: game-strength impacts with quarter exploration.**
`2026-09-20_20-37-48_contact_v3_hard6_quarter/model_79.pt` started from Trial A's frozen
update-50 checkpoint, not its rejected updated actor. Noise mean changed 0.1771 -> 0.0443,
preserving its per-joint pattern; no joint reached the existing 0.01 floor. Fifty further
critic-only updates adapted to the new sampling distribution, reaching explained variance
0.910. The remaining 29 updates took 550 actor optimizer steps; KL protection reduced the
learning rate during adaptation (final `6.67e-7`). The original teacher was verified intact.
Main survival/recovery declined 307/512 -> 297/512 and 281/512 -> 272/512 (50 gained,
59 lost, z = -0.86). Seed 71 declined 73/128 -> 68/128 and 71/128 -> 65/128. Quiet remained
16/16; action change was 0.0556, better than Trial A but worse than the accepted seed.
Measured main contact sliding rose 0.708 -> 0.723 m/world. Rejected.

Both trials used matched CPU settings/metric versions against the saved accepted policy
and incumbent. They also differ in starting critic and warm-up history, so their learning
results are not a strict noise-only ablation. Neither establishes a population improvement;
the point-estimate regressions trigger the unchanged stop rule. Increased difficulty and
reduced exploration alone have not resolved learning. The quarter-noise update-50 checkpoint
retains the exact accepted mean actor with an adapted critic, for diagnostics only. Further
work should examine how PPO updates affect impact recovery and standing, not assume that
more hours or another reward-weight guess will solve the problem. All processes finished;
no export, commit or push. The two training trials consumed ten GPU minutes, separate from
the frozen-policy probes and CPU evaluations. `git diff --check` passes.

Evidence: `logs/contact-v3-exposure-speed{2,4,6}.{json,log}`,
`logs/contact-v3-exposure-{gpu,quarter,mean}.{json,log}`,
`logs/contact-v3-hard6{-quarter}-train.log`,
`logs/contact-v3-hard6{-quarter}-{512,seed71,comparison}.json` and matching evaluation logs.

## 2026-09-20: curriculum resume defects fixed; harder-stage trial rejected

Two concrete resume defects are fixed. `overnight.py` discarded the starting seed's CPU
stage recommendation; it now applies that recommendation before the first chunk, just as
it does after an accepted chunk. `train.ps1` always supplied the configured 2 m/s start,
overriding the checkpoint's saved difficulty; resumed Perturb chains now inherit that
difficulty. Deliberate overrides remain available through `overnight.py --ball_speed`.
The configuration comment and README distinguish resumed stages from direct/fresh starts.
The previous manual fixed-2 m/s ablations bypassed these entry points: these defects do not
establish the cause of their score regressions.

The initial-stage regression test failed before the fix (launched at 2.0 despite a 2.2
recommendation) and passes afterward. A PowerShell launch test captures actual wrapper
arguments and verifies that the seed is forwarded without overriding its stage. All
18 training-safety and 18 architecture tests pass. No policy, reward, physics or observation
implementation changed, and `git diff --check` passes.

The accepted `contact_v3_full5m/model_78.pt` was evaluated at its current 2 m/s stage:
512/512 survived and 508/512 recovered (99.22%). That exceeds the existing 85% advancement
threshold. The next trial used the existing 10% increment, 2.2 m/s, rather than another
fixed-2 m/s continuation. The launch applied the measured decision directly, reusing the
saved quiet/hard-hit baselines; the corrected runner and wrapper paths are covered by tests.

`2026-09-20_19-55-27_contact_v3_curriculum22/model_78.pt` completed five minutes / 78
additional updates from the accepted checkpoint, with seed 0 and every other training setting
unchanged. Its saved difficulty is 2.2 m/s, actor optimizer count 3,120, and original teacher
targets/scales/provenance remain exact. There were no reported KL stops or numerical failures.
The last logged explained variance was 0.770. Quiet survival and recovery remained 16/16.

**The learning regression is not resolved.** At the fixed 6 m/s, seed-17 benchmark,
survival declined 307/512 -> 300/512 (59.96% -> 58.59%) and recovery 281/512 -> 273/512
(54.88% -> 53.32%). Recovery pairs: 54 gained, 62 lost, z = -0.74. Measured-contact recovery
also declined, 57.81% -> 55.66%. The seed-71 batch declined 73/128 -> 70/128 in survival
and 71/128 -> 61/128 in recovery (8 gained, 18 lost, z = -1.96). All hard-hit comparisons
used matching evaluation settings and metric versions; the unchanged gate rejects this trial.

At its new 2.2 m/s stage the candidate still survived 511/512 and recovered 507/512 (99.02%).
The stage scores use different speeds and must not be read as a paired learning comparison.
They show that the tested easy stages remain far easier than the game's impacts. Under-fire
action change was nearly unchanged from the accepted seed, while measured contact sliding
rose 0.708 -> 0.719 m/world; quiet action change rose 0.0391 -> 0.0587. Raising the easy-stage
speed alone did not produce robust hard-impact recovery.

Keep the resume fixes, retain `contact_v3_full5m/model_78.pt` as the accepted training seed,
and do not resume the rejected `contact_v3_curriculum22` candidate. Strong-impact training
coverage remains the next hypothesis to test, not an established reward defect or a reason
to weaken acceptance gates. No further chunk or export was launched. All processes finished;
no commit or push was performed.

Evidence: `logs/contact-v3-stage2-512.{json,log}`,
`logs/contact-v3-curriculum22-{train,512,seed71,stage}.log`,
`logs/contact-v3-curriculum22-{512,seed71,stage,comparison}.json`.

## 2026-09-20: guarded continuation stopped after the first five-minute chunk

The approved budget was up to three five-minute chunks, with CPU evaluation before each
continuation and a stop on regression. Only chunk 1 ran. It started from the accepted
`2026-09-20_19-14-28_contact_v3_full5m/model_78.pt` and produced
`2026-09-20_19-28-30_contact_v3_chain1/model_79.pt` after 79 additional updates.
Training used the same seed 0, 4096 worlds, 16 steps, 20-second episodes, fixed 2 m/s shots
every 4–7 seconds, actor ceiling `1e-6`, KL target 0.001, smoothing 0.1 and reference
coefficient 1. The embedded original incumbent teacher was restored from the checkpoint;
targets, scales, provenance and coefficient were verified unchanged. The optimizer reached
3,140 cumulative actor steps, exactly 1,580 above the accepted seed. Reward/training versions
matched, warm-up remained complete, and no KL stops or numerical failures were reported.

**Rejected by the unchanged main continuation gate.** On 512 seed-17 hard-hit worlds,
survival fell from 307/512 to 297/512 (59.96% -> 58.01%), and settled recovery from
281/512 to 277/512 (54.88% -> 54.10%). Recovery pairs: 51 gained, 55 lost, z = -0.39;
survival pairs: 40 gained, 50 lost, z = -1.05. These point-estimate regressions trigger
the stop rule without establishing statistically significant population regressions.
Against the shipped incumbent, recovery tied 277/512 and survival fell 302/512 -> 297/512.
Measured-contact recovery also declined against the accepted seed, 57.81% -> 56.25%.

Quiet survival/recovery stayed 16/16 over 40 seconds. Under-fire mean action change fell
0.0662 -> 0.0630 and measured contact sliding 0.708 -> 0.696 m/world, but quiet action
change rose 0.0391 -> 0.0482. The smaller seed-71 diagnostic held survival at 73/128 and
lost two recoveries, 71/128 -> 69/128 (paired recovery z = -0.38). Neither calmer impact
motion nor intact quiet standing compensated for the main gate failure.

All comparisons used matching evaluation settings and metric versions. Chunks 2 and 3
were not launched. Retain `contact_v3_full5m/model_78.pt` as the accepted training seed;
do not resume `contact_v3_chain1/model_79.pt`. This trial provides no basis for an
unattended longer run under the current setup. No reward/configuration or production code
changed, no policy was exported, and no commit or push was performed. All processes finished.

Evidence: `logs/contact-v3-chain1-train.log`,
`logs/contact-v3-chain1-{512,seed71}.{json,log}` and
`logs/contact-v3-chain-comparison.json`.

## 2026-09-20: full five-minute v3 trial passes continuation, not promotion

`2026-09-20_19-14-28_contact_v3_full5m/model_78.pt` completed the full five-minute
training budget from `2026-09-20_18-39-12_contact_v3_critic_extend/model_25.pt`, whose
actor is the unchanged incumbent. The initial CPU quiet gate passed. This trial retained
seed 0 and all previous settings: 4096 worlds, 16 rollout steps, 20-second episodes,
2 m/s shots every 4–7 seconds, actor ceiling `1e-6`, KL target 0.001, smoothing 0.1,
and fixed-reference coefficient 1. Only the iteration ceiling was raised to let the
five-minute limit control duration. No reward, policy architecture or production code changed.

The final checkpoint, selected by elapsed training time before evaluation, contains 78
updates / 1,560 actor optimizer steps. There were no reported KL stops or numerical failures.
The saved teacher targets, scales and provenance match the original corpus exactly.
Training covered 20.80 simulated seconds per world, including resets, versus 6.67 in the
short trials. Actual impact exposure was not counted. The last logged critic explained
variance was 0.618 at update 75; the final saved checkpoint is update 78.

**Main CPU continuation gate passes.** On the same 512 seed-17 hard-hit worlds, survival
rose from 302/512 to 307/512 (58.98% -> 59.96%), and historical settled recovery from
277/512 to 281/512 (54.10% -> 54.88%). All 16 quiet worlds survived and settled over
40 seconds. Recovery pairs: 59 gained, 55 lost, z = +0.37. This is not a statistically
established recovery improvement and fails the unchanged promotion threshold of z >= 2.
The incumbent remains shipped; this candidate is eligible for guarded continuation only.

Quiet mean action change fell 52%, from 0.0814 to 0.0391. Under fire, it fell 17%, from
0.0795 to 0.0662, and measured contact sliding fell 6.2%, from 0.755 to 0.708 m/world.
Measured-contact recovery rose from 56.25% to 57.81%. Historical rapid replants increased
slightly, 0.174 -> 0.186; measured load replants declined 9.65 -> 8.84. These improvements
in aggregate motion do not establish visually realistic recovery.

The independent seed-71, 128-world diagnostic recovered 71/128 versus 66/128 (55.47%
versus 51.56%), paired z = +1.00, but survival declined by one world, 74/128 -> 73/128.
This mixed result supports keeping continuation bounded and evaluated; it is not a second
full gate pass. Evaluation settings and both metric versions matched the saved incumbent
results before comparison. No new gate thresholds were introduced.

Next recommendation: continue from `contact_v3_full5m/model_78.pt` in five-minute chunks,
retaining the original fixed teacher and current settings, evaluating each chunk against
the accepted checkpoint and incumbent, and stopping on regression. No further chunk was
launched in this trial. All training/evaluation processes finished. No export, commit or push.

Evidence: `logs/contact-v3-full5m-train.log`,
`logs/contact-v3-full5m-{512,seed71}.{json,log}` and
`logs/contact-v3-full5m-comparison.json`.

## 2026-09-20: two additional training seeds; neither qualifies for continuation

Repeated the v3 fixed-reference trial with training seeds 1 and 2, each starting from
`2026-09-20_18-39-12_contact_v3_critic_extend/model_25.pt`. Its actor and exploration
parameters were verified byte-identical to the shipped incumbent; warm-up was complete.
Both trials used the same settings as seed 0: 25 updates, 4096 worlds, 16 rollout steps,
20-second episodes, 2 m/s shots every 4–7 seconds, actor ceiling `1e-6`, KL target 0.001,
smoothing 0.1 and reference coefficient 1. The saved arguments differ only in seed and run
name. Both took 500 actor optimizer steps with no KL stops, about 1.7 and 1.8 GPU minutes.
The saved reference targets, scales and provenance match the original corpus exactly.

- Seed 1: `2026-09-20_18-56-58_contact_v3_seed1/model_25.pt`.
  Main seed-17 benchmark: 303/512 survived (59.18%), 273/512 recovered (53.32%), versus
  incumbent 302/512 and 277/512. Recovery pairs: 55 gained, 59 lost, z = -0.37.
  Independent seed-71 batch: 70/128 survived, 65/128 recovered, versus 74/128 and 66/128;
  recovery z = -0.19. Rejected for lower main-benchmark recovery.
- Seed 2: `2026-09-20_19-00-12_contact_v3_seed2/model_25.pt`.
  Main benchmark: 298/512 survived (58.20%), 275/512 recovered (53.71%);
  recovery pairs: 48 gained, 50 lost, z = -0.20. Independent batch: 75/128 survived,
  69/128 recovered; recovery z = +0.54. Rejected for lower main survival and recovery.

Both retained 16/16 quiet survival and recovery over 40 seconds. Quiet action change fell
from 0.0814 to 0.0429/0.0436 (47%/46%). Main-benchmark action change fell 13%/13%, and
measured contact sliding fell 5.2%/3.9%. Historical height-based replants were 0.170/0.217
versus 0.174; calmer action does not imply every foot-motion metric improved. Measured
contact recovery was 56.25%/55.66% versus 56.25%. All evaluation settings and metric
versions matched the saved incumbent evaluations before paired comparisons.

Neither new candidate satisfies the unchanged continuation gate. No five-minute extension
was launched; keep the incumbent. The differences do not establish statistically significant
population regressions, and no seed establishes a recovery improvement. All training and
evaluation processes finished. No policy export, commit or push was performed.

**Trial-design limitation:** 25 updates × 16 steps × 4 physics substeps × 0.004167 seconds
provide only 6.67 simulated seconds per world, including any resets. Fresh shots wait 4–7
seconds; nominal flight from 2 m at 2 m/s takes another second. Randomizing episode clocks
also offsets the shot schedule, so it does not supply immediate impact experience. These
trials predominantly test retention and early adaptation, not sustained recovery learning;
actual impact exposure was not counted in their logs. Their results do not justify declaring
the reward wrong or ruling out longer training. Before more reward changes or hours of
training, the next proposed experiment is a full five-minute trial from the validated incumbent
with the warmed v3 critic, followed by the same CPU gates. Do not resume a rejected candidate.

Evidence: `logs/contact-v3-seed{1,2}-train.log`,
`logs/contact-v3-seed{1,2}-{512,seed71}.{json,log}` and
`logs/contact-v3-multiseed-comparison.json`.

## 2026-09-20: measured floor contact implemented; short trial rejected

`loaded_contact_recovery_v3` replaces reward grounding by foot-origin height with
floor-contact normal load from the foot and toe geometries. The shared hysteresis enters
above 5 N and exits at or below 1 N. Sliding is the per-foot load-weighted mean squared
tangential speed at contact points: a stationary heel/toe pivot is no longer charged as
sliding merely because the foot origin moves, and opposite velocities during twisting
cannot cancel. CPU and GPU adapters read the final physics substep. Reward coefficients,
policy observations, action mapping/latency, physics and PPO defaults are unchanged.

The scorer owns a separate `height_proxy_v2` contact history. Its gate and historical
movement metrics remain unchanged; `contact_*` fields are additional `loaded_contact_v3`
diagnostics. Re-evaluating the incumbent reproduced every historical survival/recovery
outcome and all three movement aggregates on the 512-world seed-17 benchmark and its
16-world, 40-second quiet test. New contact scoring alone raises the unchanged incumbent's
reported recovery from 54.1% to 56.3%; it must not be mistaken for learning progress.

Validation: **68 unit tests pass**, including eight new contact tests. Those exercise flat
loading, raised feet, heel/toe pivots, true sliding, twisting, ball-contact exclusion,
NumPy/Torch hysteresis, partial resets, frozen scoring, and physical CPU/GPU contact parity
with empty-buffer/world isolation. Both environment parity scripts pass, as does Perturb
preflight with zero failures/warnings. Standing reward parity error was below `6.2e-5`.
The original severely penetrating fixtures produce loads up to 22 kN and independent
CPU/GPU force distributions: raw reward differences reach 0.023 there. These differences
are reported; reward arithmetic is separately checked with identical slip measurements,
while physical flat/pivot/sliding fixtures compare native load/slip without substitution.
No simulation solver settings were changed to make the tests pass. `git diff --check` passes.

The incumbent actor was re-warmed under the new reward for 50 + 25 critic-only iterations:
`2026-09-20_18-34-01_contact_v3_critic/model_50` and
`2026-09-20_18-39-12_contact_v3_critic_extend/model_25`. Explained variance rose from 0.345
after the first chunk to 0.560 after the extension. Actor and exploration parameters were
verified byte-identical to the incumbent at both boundaries; the fit threshold stayed 0.5.

The only policy-training trial, `2026-09-20_18-43-16_contact_v3_trial/model_25`, then ran
25 iterations (500 actor Adam steps, about 1.6 GPU minutes), using seed 0, 4096 worlds,
16 rollout steps, 20-second episodes, fixed 2 m/s shots every 4–7 seconds, actor ceiling
`1e-6`, KL target 0.001, smoothing 0.1 and the unchanged all-phase reference corpus with
coefficient 1. There were no KL-stop events. Saved reference observations, means and scales
remain byte-identical to the original corpus. Critic warm-up consumed about 4.9 additional
GPU minutes; CPU validation time is separate.

**Rejected by the unchanged main gate.** On 512 seed-17 hard-hit worlds, survival declined
from 302/512 (59.0%) to 291/512 (56.8%), and settled recovery from 277/512 (54.1%) to
272/512 (53.1%). Paired recovery: 51 gained, 56 lost, z = -0.48. This fails the point-estimate
safety gate without establishing a statistically significant population regression.
Measured-contact recovery also declined, 56.3% to 55.3%, so the negative result is not
merely an artifact of preserving the old gate.

Quiet survival/recovery remained 16/16 over 40 seconds. Quiet mean action change fell
0.0814 -> 0.0488 (40%), while measured contact slip fell 0.357 -> 0.338 m/world. Under fire,
action change fell 0.0795 -> 0.0704 and measured contact slip 0.755 -> 0.728 m/world (3.6%).
Measured load replants fell 9.65 -> 9.06 per world; historical height replants fell
0.174 -> 0.150. Different measurement definitions must not be compared as one series.

The independent seed-71, 128-world batch improved survival 74/128 -> 76/128 (57.8% -> 59.4%)
and recovery 66/128 -> 71/128 (51.6% -> 55.5%), paired z = +0.93. This smaller, inconclusive
improvement does not override the main gate. The contact defect is fixed and movement is
calmer, but better recovery and visually realistic behavior have not been established.
Keep the incumbent; do not seed a longer chain from this rejected candidate. No policy
was exported and no commit or push was performed.

Evidence: `logs/contact-v3-{critic,critic-extend,trial}.log`,
`logs/contact-v3-{incumbent,candidate}-{512,seed71}.json`,
`logs/contact-v3-comparison.json`, and the contact/parity/preflight tests.

## 2026-09-20: contact measurement defect confirmed in matched replays

The [recovery diagnosis](PERTURB_DIAGNOSIS.md) reproduces all 1,024 incumbent/candidate
survival and recovery outcomes from the existing 512-world evaluations and records actual
floor loads at every physics substep. The reward's foot-origin-height proxy calls loaded,
tilted feet airborne: 47.9% of airborne-labelled incumbent foot samples still carry floor
load during the first half-second after impact, rising to 61.1% over the next second.
Both proxy feet are airborne despite actual support in 40.3%/46.5% of those response samples,
forcing support reward to zero. The candidate exhibits the same defect.

Correct contact measurement before tuning movement penalties or running longer training.
The report includes matched replays, reward-term attribution, a contact-only reward
counterfactual, and a scoped CPU/GPU correction plan. This identifies a real measurement
defect; it does not establish that correcting it alone will solve the recovery problem.
No production configuration or policy changed during this diagnosis.

## 2026-09-20: fixed-reference retention experiment

The optional `--reference_data` / `--reference_coef` path adds an actor-only penalty against
frozen incumbent mean actions. `policy_reference.py` owns balanced standing/impact sampling,
the fixed teacher variance scale, drift reporting and serialization. The critic and reward
kernel are unchanged. Checkpoints embed the original reference and its sampling state; a new
training chunk does not silently replace the teacher with the latest student. Current plant
hashes must match. Reference sampling does not consume rollout/PPO randomness.

The CPU corpus uses seeds 101 and 103, separate from evaluation seed 17. All 16 quiet trajectories
and 39/64 hard-hit trajectories survived and settled. Failed recoveries are not imitation targets.
The collector now samples uniformly across all control steps, capped at 4096 states per group,
to avoid aliasing alternating actions with a fixed temporal stride. A regression test explicitly
checks both phases of an alternating controller.

The experiment holds `grounded_recovery_v2`, temporal smoothing 0.1, actor ceiling `1e-6`, KL target
0.001, seed 0, 4096 worlds, 16 steps, 20-second episodes and 2 m/s training shots fixed. Each run
starts from the same warmed `2026-09-20_16-29-23_recovery_safe_critic/model_50.pt` and takes 25
updates (about 1.6 minutes of GPU training). The fixed teacher is the shipped `p0911i_s1/model_503`.

The initial, strided-corpus comparison used coefficient 1 versus 0. The zero-penalty control
survived 56.8% and recovered 52.5% of 512 hard-hit worlds. Anchoring reached 58.8% and 53.9%,
respectively, against the incumbent's 59.0% and 54.1%. That is one fewer surviving and recovered
world than the incumbent, so the unchanged strict gate rejects it. This point difference does
not establish a population-level regression. Quiet survival/settling stayed 16/16; anchored mean
action change fell from 0.0814 to 0.0443 and quiet sliding from 0.295 to 0.284 m/world. Under fire,
action change fell from 0.0795 to 0.0721, while sliding rose from 0.367 to 0.371 m/world.

The corrected all-phase corpus is evaluated separately. The zero-coefficient control is independent
of corpus contents (covered by an exact update-equivalence test). These are short, single-training-
seed experiments, not evidence that long training or visually realistic recovery is solved.
The reused incumbent benchmark has identical CPU evaluation settings; no plant, observation,
reward or scorer changes were made for this experiment.

**Final corrected-corpus result: rejected.** `2026-09-20_17-38-23_recovery_reference_dense/model_25`
survived 295/512 (57.6%) and recovered 267/512 (52.1%), versus 302/512 and 277/512 for the incumbent.
Paired recovery z = -0.94; the point estimates fail the unchanged acceptance gates, without
establishing a statistically significant population-level regression. Quiet survival and settling
remained 16/16. Quiet action change was 0.0739 (9.2% below the incumbent), and under-fire action
change was 0.0734 (7.6% lower). Under-fire sliding was slightly higher, 0.369 versus 0.367 m/world,
and rapid replants increased from 0.174 to 0.191. Less action oscillation is therefore not evidence
of better foot placement or recovery. No tested candidate qualifies to seed a longer chain or
replace the incumbent. Reference retention remains opt-in; no reward weights or acceptance gates
were relaxed to obtain a passing result.

Validation: 10 reference tests plus the existing 50 regression tests pass. The tests cover balanced
weighting, target immutability, actor gradients, independent random streams, warm-up isolation,
zero-penalty equivalence, checkpoint continuation, repeated-artifact continuation, wrong-plant
rejection and phase coverage.
The actual trained checkpoint also retained byte-identical reference observations, targets and
scales. No policy has been exported.

Evidence: `logs/recovery-reference-{anchor,control,dense}.log`, matching `*-512.json` evaluations,
`logs/recovery-reference-dense.pt`, and the `2026-09-20_*_recovery_reference_*` checkpoints.

## 2026-09-20: reject collapsed seeds and protect fine-tuning

The subsequent 15-, 60- and 90-minute checkpoints all failed quiet standing (0/64 survivors
over six seconds). More training from those checkpoints is not justified. The seeded 15-minute
run also collapsed, so starting from scratch was not the only problem. Its earlier three-minute
smoke result did not establish that longer training was safe.

The trainer had discarded time-limit value targets as if every timeout were a fall. Both
environments now preserve final observations before reset, and GAE bootstraps genuine timeouts
without propagating advantages into the reset episode. Actor and critic now have separate Adam
optimizers and gradient clipping. Perturb uses unclipped value regression, actor-only KL stopping,
a `1e-6` actor learning-rate ceiling, and a frozen-actor critic warm-up after target-version changes.
Warm-up lasts at least 50 iterations and until explained variance reaches 0.5. All continuation
state is checkpointed; incompatible target versions retain only the behavior seed.

Perturb requires an explicit checkpoint that survives and settles in all 16 CPU quiet-room
worlds over 40 seconds. The 90-minute checkpoint was rejected by this gate before GPU setup.
`train.ps1` splits budgets into chunks of at most five training minutes. Each chunk must pass
quiet standing and CPU recovery/survival comparisons before it can seed the next one. A rejection
stops the chain with exit code 2 and retains the accepted checkpoint; repeating an identical failed
experiment is no longer automatic. Evaluation time is additional to the training budget.

Controlled trials start from `2026-09-20_16-29-23_recovery_safe_critic/model_50.pt`: 50 critic-only
iterations from the shipped `p0911i_s1/model_503`, explained variance 0.556, actor and exploration
verified byte-identical to the incumbent. Each smoothing trial uses the same warmed critic,
optimizer state and seed, 4096 worlds, 16 rollout steps, 20-second episodes, fixed 2 m/s shots
every 4–7 seconds, and 50 PPO iterations. The initial sweep used a `3e-5` ceiling, KL target
0.005 and the earlier `1e-5` adaptive floor; the final implementation also lets the adaptive
floor fall below a small ceiling. CPU comparisons use seed 17, 16 quiet worlds for
40 seconds and 64 worlds for a single 8 kg, 6 m/s shot over six seconds.

Validation: 16 training-safety tests, 16 recovery tests and 18 architecture tests pass. Both
task parity scripts pass; Perturb also exercises actual terminal observations and reset masks
on CPU and GPU. PowerShell parsing and task override defaults pass. The shipped ONNX and contract
are unchanged.

At a `3e-5` actor ceiling and KL target 0.005, smoothing weights 0, 0.1 and 0.5 all produced
0/16 quiet-room survivors after 50 updates. Under one hard shot they survived 1/64, 1/64 and 3/64
respectively, with 0/64 settled recoveries. The incumbent survived 43/64 and recovered 40/64.
Iterations 25 of the 0 and 0.5 trials also failed quiet standing. All are rejected. This isolates
smoothing strength as insufficient to prevent degradation, rather than showing that one of these
weights is ready for a longer run.

A follow-up kept smoothing at 0.1, reduced the actor ceiling to `1e-6` and KL target to 0.001.
Both iterations 25 and 50 passed 16/16 quiet survival and settling for 40 seconds. Mean action
change fell from 0.0814 to 0.0050 and 0.0070; no quiet-room steps or rapid replants were recorded.
These smaller-step settings are now the defaults. A 64-world impact pilot at iteration 25 was
42/64 survival and 39/64 recovery, one world below the incumbent on both. The standard 512-world
comparison determines acceptance; improved quiet behavior alone cannot authorize promotion.

**Final 512-world comparison: neither candidate is accepted.** The incumbent survived 59.0%
and recovered 54.1%. The smaller-step iteration 25 survived 56.8% and recovered 52.3%
(paired recovery z = -0.83); iteration 50 survived 52.1% and recovered 37.3% (z = -6.60).
Both lose survival and recovery, so the gate retains the incumbent. At iteration 25,
under-fire action change fell from 0.0795 to 0.0730, but foot sliding rose from 0.367 to
0.389 m/world. Calmer quiet standing has not yet translated into more reliable, realistic
impact recovery. The remaining problem is behavior degradation during objective adaptation;
these experiments do not establish its exact cause. Do not resume the rejected checkpoints or
launch an unattended long run. The smaller defaults limit the damage while future experiments
remain subject to the same acceptance gates.

Evidence: `logs/recovery-safe-{critic,smooth0,smooth01,smooth05,slow}.log`, the matching evaluation
JSON files, and checkpoints in the `2026-09-20_*_recovery_safe_*` run directories.
The final matched evaluations are `logs/recovery-safe-{incumbent,slow-early,slow}-512.json`.

## 2026-09-20: perturb recovery learning corrections

The shipped balance checkpoint remains `p0911i_s1/model_503`. Its historical 92.4% result below
used a **5 kg ball at 6 m/s (30 N.s)**. Today's generated model uses **8 kg at 6 m/s (48 N.s)**.
The historical success rate is not a measurement of the current scene.

The old reward paid for fast swing-foot motion and proximity to the capture point while still
airborne. It did not require a useful landing. Its action-change penalty averaged 30 joints,
making isolated ankle reversals cheap. `grounded_recovery_v2` now rewards grounded support and
settling, penalises sliding and rapid replants, and allows a comfortable staggered stance.
CPU and GPU use one reward kernel. PPO regularises changes in deterministic mean actions and
uses a lower perturb learning-rate ceiling. Changed reward versions reset critic/optimizer
state while retaining the actor seed. See [README.md](README.md#perturb-recovery-objective-september-2026).

Promotion now measures a stable final half-second hold as well as survival; improved recovery
cannot trade away survival on the comparison batch. The curriculum checks settled episode
endings, and overnight sessions advance on a CPU recovery score of at least 85%.

Validation: 16 recovery tests and 18 architecture tests passed, both task parity scripts passed,
and perturb preflight reported no failures or warnings. Perturb reward parity error was below
0.000002. A three-minute GPU run completed 49 iterations without nonfinite-reward reports; a
separate two-iteration run verified the final training integration and same-version resume.

CPU smoke comparison: 32 worlds, seed 17, six seconds, one uniformly targeted shot launched
at 1.5–1.8 s, current 8 kg ball at 6 m/s. Both policies survived all quiet-room trials.
The incumbent survived 16/32 under fire and settled in 15/32. The smoke candidate survived and
settled in 14/32. Mean clipped action change per joint per step fell from 0.0617 to 0.0140 in
the quiet room, and from 0.0769 to 0.0530 under fire. These small samples establish a working
training path and movement smoothing, not improved robustness or visually validated realism.
The candidate fails the survival gate and was **not exported**. New rewards do not alter an
existing ONNX policy; a longer curriculum run and CPU/scene validation are still required.

Local evidence: `logs/recovery-v2-{incumbent,candidate}.json`,
`logs/recovery-v2-training.log`, `logs/recovery-v2-integration.log` and
`logs/mujoco/2026-09-20_11-19-31_recovery_v2_smoke/model_49.pt`.

## Historical handover — 2026-09-11

Written as a handover. `README.md` explains how the pieces work; this says where the project
actually is, what is known, and what the next person (or session) should do first.

## The one-paragraph version

The dummy's physics lives in MuJoCo; Godot renders it and drives it through P/Invoke. The body is
1.750 m, 69.6 kg, 18 bodies, 33 actuators of which a policy drives 30, built by `build_mjcf.py`
from Godot's own rig dump. It is a **ragdoll with muscles**: `motor` actuators over passive
ligaments, so zero policy output is zero torque and an unpowered body crumples. **Balance is
solved** — 100% upright indefinitely in a quiet room, actively, on under 10% of human strength.
**Perturbation survives 92.4% of a single 30 N·s hit** (68.8% at dusk on 2026-09-10), after four
measured jumps, each from a change aimed at a measured failure: more training shots at the
high hits it fell to, a step reward that keeps growing to 1 m/s of swing, and a term that pays where
the foot lands against the capture point - then a rest stance it is pulled back to once balanced,
which it ends in. **Walk follows most of a joystick at an amble** — straight,
arcs both ways, and it stops when told to — but it cannot turn on the spot, and measured from rest
its gait is a shuffle. **Neither brain has been seen in the scenes since it changed.**

## What is shipping

| file | from | what it does |
|---|---|---|
| `balance_policy.onnx` | `p0911i_s1/model_503` | stands, and takes hits: survives **92.4%** of a single 30 N·s hit (512 envs); the brain it replaced survives 87.5% (paired z +2.93). Ends back in its stance: feet at the rest width (0.31 m), 0% crossed, leg error 0.24 rad. Under a hit every 3 s, 53% last 20 s |
| `locomotion_policy.onnx` | `w0911f_s1/model_253` | **46.3% of the joystick** (32 envs, five manoeuvres): +6.0 m on a 0.25 m/s straight line, arcs −807° / +367°, stands still to 0.05 m; falls turning on the spot, and in some runs of the left arc. Its contract carries the heading hold; `MujocoWalk.tscn` commands 0.25 m/s |

Promotion is automatic and measured: `scripts/promote.py` scores challenger against the incumbent
read from the shipped contract, and exports only on a win. `overnight.py` calls it at the end of a
run. Do not export by hand — doing so once shipped a policy that walked **backwards**. (Re-exporting
the checkpoint a contract already names, to regenerate the contract, is not choosing a checkpoint.)

**Perturb is promoted on one real hit at 6 m/s (30 N·s), 512 envs, both checkpoints scored at once
on the same shots, and shipped only if the paired McNemar z reaches 2.** The gates before it failed
twice: upright over 40 s at 2.2 m/s read ~100% for everything, and a "one-hit" test that actually
fired two shots shipped on noise. **Walk is still promoted on 32 envs with no paired test**, on
the joystick score (all five of `eval_walk`'s manoeuvres, `scoring.WalkScorer`) — and on 2026-09-11
a +0.9-point win shipped a brain that falls more on one arc than the one it replaced.

## Run this before any long session

```
python mujoco_rig/scripts/preflight.py --task perturb --speed_end 6.0 --seed <checkpoint>
python mujoco_rig/scripts/preflight.py --task walk --seed <checkpoint>
```

Three minutes, no GPU. It checks the start pose does not self-intersect; that the reward ranks
standing / stepping / airborne / fallen in the intended order; for perturb, that **a capture step
earns more than a stomp and a step away from the falling COM earns nothing**; for walk, that
**walking out-earns a statue at every commanded speed**; that **one exploding world stays
harmless**; that the projectile connects and the curriculum's end is physically survivable; and that
the seed's exploration suits the task. **Six multi-hour failures on 2026-09-09/10 were task-level
errors these checks now catch before the first session.** A task that fails preflight cannot be
fixed by training.

## Numbers worth not re-deriving

**The plant**
- Torque actuators over ligaments (5% of peak torque per rad). Zero output ≈ `dummy_limp.xml`.
- `JOINT_ARMATURE = 0.02` is load-bearing. Without it, 10 of 10 random-torque rollouts diverge with
  `Nan, Inf or huge value in QACC`, and **MuJoCo silently resets the state** when that happens.
- `qpos0` is NOT a valid pose — the arms hang inside the legs there. Every model carries a `rest`
  keyframe; the envs and `MjBridge.ResetData` use it.
- The ball is **8 kg** since 2026-09-11 (48 N·s at 6 m/s); every perturb score before that is at
  **5 kg** (30 N·s), including the shipped brain's 92.4%. At 15 kg a single hit was survived 0 times in 12. A stepping recovery
  arrests about 1.0 m/s of COM velocity; ankles alone 0.3–0.5.
- **The body is left/right symmetric** — bodies, joints, actuators and geoms all checked. A gait
  that veers is a learned handedness, not the plant.
- Fastest body DOF over 1.2M world-steps, falls included: 94 rad/s walking, 97 under fire. A world
  past `QVEL_CEILING` (300) is diverged, not falling.

**Measurement traps, each of which produced a wrong conclusion**
- **MuJoCo `cvel` is referenced to the subtree centre of mass**, not to the body. Raw
  `cvel[foot, 3:6]` read −0.14 m/s for a foot moving at +0.70 — the wrong sign. `com_velocity`
  now shifts it: summed raw, the whole-body COM velocity was off by a median 0.32 m/s whenever the
  body moved (fixed 2026-09-11; preflight's "137% delivered" was the same error - it is 93%). The
  pelvis velocity the walk reward and the observation read is still raw.
- **Worlds that reset together time out together.** The curriculum promoted on that clock, and for
  the first 75 iterations the logged episode length counts only worlds that FELL.
  `randomize_episode_phase` now staggers every task.
- **The "one hit" test fired two** for most of 2026-09-10. `eval.py` now prints `hits per env`.
- **"Survived when it stepped" is confounded by how hard the hit was.** Never decide on it.
- **Survival near 30–40% is noisy.** Compare checkpoints only paired, on the same shots.
- **A calibration is only valid for the range it was made on.** The ball's mass silently re-based
  perturb's reference; the walk's command range silently re-based its tracking kernel.
- **The training log's walk vx is pelvis-frame.** It read +0.16–0.22 for a policy walking in
  circles. Judge walk on eval_walk's `along` and `heading swept`.
- **One exploding world poisons the whole batch** unless its reward is bounded AND sanitised. The
  `nefc overflow` printed before each walk crash was that world, not the gait (72 rows of 256).
- **A behaviour the env computes must exist in Godot too.** The heading hold lives in the walk env's
  `step()`; `MjPolicyDriver` wrote the yaw command raw. It now applies the same correction, driven
  by the contract's `command` block, with a heading formula identical to the env's over 10,000
  random rotations.
- **Measure a foot's lift from where it rests.** The walk env read an absolute 6 cm while the foot
  origin rests at 4.4 cm, so a 1.6 cm shuffle counted as a step and every gait term paid it.
- **A score clipped at 1.0 cannot see overshoot.** The joystick score read 43% too fast as a perfect
  straight line; each part is now 1 - |achieved / commanded - 1|.

**The trainer**
- Batch: **4,096 envs × 16 steps**; the split matters more than the batch.
- `ms per policy step = 35.8 + 0.04157 × envs`. Envs are nearly free; steps are sequential.
- A seeded start needs the promotion cooldown: without it the first update ran KL to +5.97.
- **Exploration does not travel with the weights.** `--reset_std` for any cross-task seed, even one
  whose std passes preflight — the perturb brain seeded into walk at 0.173 settled into a statue;
  reset to 0.4 with entropy 0.005, it stepped within two sessions.
- Scoring 512 envs takes ~70 s per checkpoint; the gates run their scorings concurrently.

**Reward brackets, measured**
- Perturb stance term must be **gated on being balanced**; `recover_step` must pay for the swing
  foot moving **toward** the escaping COM, not for single support (a stomp satisfied that).
- Walk tracking must be **relative to the command** (a fixed width paid a statue 44–86% at 0.15–0.35
  m/s); a **zero yaw command must hold the heading**, not just penalise yaw rate (a circle cost a
  flat 0.95/step); the velocity-squared penalties are clamped at `PENALTY_CAP`.
- Perturb: **aim the training shots at the failures** (head, chest and spine were half the falls on a
  quarter of the shots); pay the step up to **1 m/s** of swing (`CAPTURE_V`); pay **where the foot
  lands** against the capture point (`CAPTURE_PLACE_SIGMA` 0.15); pull a balanced body back to its
  **rest stance** measured in the pelvis frame and signed, legs included (`STANCE_SIGMA`,
  `LEG_POSE_SCALE`) - world axes with abs() read crossed feet as a normal width.
- Walk: **pay both feet down on a stand command** (`STAND_PLANTED` 1.5) — without it every brain
  walked away from a stand. The shuffle's payment is what keeps it walking: every `WALK_FOOT_CLEAR`
  above 1.6 cm tried (2, 2.5, 3, 6 cm) stopped the walk within one session.

## Where the tasks stand

**Perturb** — one real 30 N·s hit, 512 envs, the same shots for every row. The night is written up
in `isaac_lab_3/OVERNIGHT-2026-09-11.md`.

| checkpoint | survived | paired z vs row above | what changed |
|---|---|---|---|
| `perturb_z_s4/model_253` | 68.8% | — | shipped at dusk on 2026-09-10 |
| `p0911b_s1/model_249` | 74.2% | +2.60 | training shots weighted toward head, chest and spine |
| `p0911c_s1/model_255` | 80.9% | +3.08 | `CAPTURE_V` 0.5 -> 1.0 |
| `p0911e_s1/model_256` | 87.5% | +3.18 | a correct COM velocity + a capture-point placement term |
| `p0911i_s1/model_503` **(shipped)** | **92.4%** | +2.93 | a signed, pelvis-frame rest stance with the legs, paid while balanced (30 min) |

A continuation after each change did not improve (80.3%, 85.7%): the next gain needs a change, not
minutes. What is left: head 63% and chest 65%; pushed-right 79% against ~90% for every other
direction (a learned handedness - the body is symmetric); the first step still lands ~0.17 m short
of the capture point, so the last jump came from something the step probe does not see.

**The rest stance** (the morning after): watched in the scenes, the dummy stood oddly after hits, and
the stance term could not see it - world axes with `abs()` read crossed feet as a normal width, and
`pose` averaged all 30 joints. It is now the feet in the pelvis's own heading frame, signed, times
the legs against the rest pose, still only while balanced. It gained survival as well as posture:

| | before (`model_256`) | after (`model_503`) |
|---|---|---|
| one hit: survived / end stance | 87.5% / 0.29 | **92.4% / 0.37** |
| one hit: width (rest 0.31 m) / split / leg error | 0.34 m / 0.06 m / 0.27 rad | 0.31 m / 0.04 m / 0.24 rad |
| a hit every 3 s, 20 s: survived / end stance | 41.4% / 0.24 | **53.1% / 0.29** |

Two other changes that morning each cost ~4 points and did not ship: training at 6.6 m/s (83.0%)
and weights refreshed from the brain's own profile (83.4%).

**Walk** — stage 4 (the amble with turning and stopping), scored on the whole joystick:

| | `walk_g_s1/model_254` (dusk) | `w0911f_s1/model_253` **(shipped)** |
|---|---|---|
| joystick, 32 envs | 17.6% | **46.3%** |
| straight line, 40 s at 0.25 m/s | +8.1 m | +6.0 m |
| turn left / right | −7° / −158° | +367° / −807° |
| turn on the spot | +26° | falls |
| stand still | drifts 3.7 m | drifts 0.05 m |

Measured from rest, it takes **no real steps**: its higher foot rises a median 2.4 cm and never
6 cm, and the walk env had counted 1.6 cm as a step all along. Every clearance that stopped paying that shuffle - 6, 3 and 2.5 cm, and 2 cm once training continued - turned the walk into a statue within one session instead of making it lift higher, so `WALK_FOOT_CLEAR` is left at 1.6 cm above rest: what the env always used, now written relative to rest.

## What I would do next

1. **Look at the three scenes.** Both brains changed on 2026-09-11. Walk should print `heading hold
   gain 0.50` and amble straight; Perturb should take most hits with a step.
2. **Give walk the paired gate perturb has.** `w0911f` shipped on +0.9 points.
3. **A stronger reason to walk, before real steps.** Every `WALK_FOOT_CLEAR` above 1.6 cm tried
   turned the walk into a statue within a session. Turning on the spot needs real steps; without
   them it falls.
4. **The pelvis velocity** the walk reward and the observation read is still raw `cvel`. Change it in
   the env and in Godot's observation together, or not at all.
5. **Perturb:** probe every step rather than the first; the pushed-right handedness.

## Tooling

```
scripts/preflight.py     is the task winnable, does the reward pay for the right thing, is one bad world harmless
scripts/promote.py       ship only on a paired, significant win against the incumbent (perturb: one real 30 N.s hit)
scripts/overnight.py     chained scored sessions, difficulty from measurement, never chains a crash, auto-promote
scripts/scoring.py       how each task is scored; the one reader of the scorers' --json output
scripts/batch_table.py   where the useful batch is, on THIS plant
scripts/watch.ps1        a window on the scorer's engine, with live telemetry
rl/watch.py              the viewer itself
rl/env_config.py         the constants that define the task, shared by both backends
rl/body_env*.py          the base environment each task extends, CPU and GPU
```

Restructured on 2026-09-11 with no change in behaviour, every step gated: a ball-free base
environment (bit-identical perturb traces, walk in lockstep), scorers that write JSON read by one
`scoring.py` (same verdicts), per-task tables instead of task branches, and the trainer, exporter,
model generator and C# driver split into named parts (identical old-vs-new training runs,
identical contracts, byte-identical models, 0 build errors).

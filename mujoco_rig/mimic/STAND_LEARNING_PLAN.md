# Stand learning: fresh experiment series

**Second autonomous batch complete (2026-10-07): 20/20 substantive experiments.** The first focused-varied pair retains natural Stand for 120 seconds and improves primary, timing, admitted directional, and repeated-hit recovery. Independent optimization repeat fails to reproduce the gain on both source parents, so the candidate remains unpromoted. No training/evaluations running; original selected pair and Arena unchanged. Full results and accounting: `PERTURB_AUTONOMOUS_PLAN.md`.

Started: 2026-10-01 (Europe/Madrid).

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

**Earlier result (2026-10-06): the requested twenty substantive experiments are
complete and verified.** All four new final brains (two matched arms across
two independent seeds) maintain natural Stand for 120 seconds in all 64 native/
GPU cases. Initial-motion recovery passes 320/320. The push curriculum fails
its predefined improvement rule on both seeds: moderate-push gains reverse
between seeds, and every 180 N x 0.1-second forward/backward case still fails.
The combined pose/repeated-push test passes 492/576 cases, including its controls;
recovery is not solved. Retain the quiet training recipe and investigate the
forward/backward recovery boundary before another training change. Passive
springs remain zero; active local target PD is unchanged. Arena still selects
seed 301026's previously verified bundle. All experiment processes have ended.

The batch contains 4,208 accepted physical cases, 7,200 new PPO cycles and
29,491,200 new transitions across six fresh training arms. Verification repairs,
backend halves, checkpoints and interrupted attempts are not separate milestones.
All 44 result-root verification seals passed. Full ledger and completion audit:
`logs/mimickit-training/stand-milestones-20261005-01/ledger.json` and
`completion-verification.json` in the same directory. The detailed protocols,
failed criteria, preserved attempts and per-seed outcomes follow below.

## Current direction and behavioral milestones (2026-10-06)

**Ball-recovery screen completed:** all 400 native cases passed evidence checks,
with zero learning updates. Quiet controls pass 16/16 for 12 seconds; all 384
throws hit. Both actors fail every 10 kg / 4 m/s chest hit (0/32 combined recovery).
The predeclared candidate is 5 kg / 1 m/s, recovered in 9/16 and 14/16 cases.
See [PERTURB_LEARNING_PLAN.md](PERTURB_LEARNING_PLAN.md) for complete results,
the preserved launcher failure, GPU admission and subsequent training results.
**GPU ball admission passed:** all 80 GPU cases match native outcomes and meet
contact timing/impulse tolerances. Launch/reset isolation and both live matched-start
pairs pass. Zero training updates were performed during admission.

**Matched ball-training pilot completed:** four 300-update arms and all 480
evaluations finished in `logs/mimickit-training/stand-ball-pilot-20261006-01`.
The run used 1,200 PPO updates, 4,915,200 transitions and 27.90 minutes of measured
training time; each arm stayed within its 600-second cap. All final brains retain
120-second Stand (64/64 native/GPU cases), and all 384 evaluation throws hit.
At 5 kg / 1 m/s, seed 301027 scores 9/16 at source, 8/16 after quiet continuation
and 11/16 after ball practice, per backend. Seed 301028 remains 14/16 in all three
conditions. Native and GPU agree; easy-impact recovery is retained throughout.
All actual start/isolation checks and the final integrity seal pass, but the
predefined improvement rule fails replication. No brain is promoted or loaded
into Arena. Passive springs and active target PD remain unchanged.
**Intermediate checkpoints assessed:** 256 further physical cases completed
with zero new learning updates in `stand-ball-checkpoints-20261006-01` under
the same log directory. At source/100/200/300 updates, ball-trained recovery
is 9/12/11/11 out of 16 for seed 301027 and 14/15/15/14 for seed 301028, per
backend. The latter exactly matches its quiet control, including which poses
recover. Neither checkpoint qualifies for conditional Stand/easy-impact
assessment or promotion. Native/GPU outcomes agree and the evidence seal passes.
**Saved-trace diagnostic completed:** 448 existing cases examined with zero new
simulation or learning in `stand-ball-trace-audit-20261006-01`. All 108 failed
+X cases first exceed horizontal speed; none saturates motor torque in the first
second after contact. Action/target clipping does not precede balance loss;
recorded delay and PD control match their specified formulas. Several temporary
successes receive substantially less impulse from later contact with the same
ball. This does not isolate whether the policy balances better after the same
disturbance or changes the collision itself. Next, proposed only: a 160-case
frozen-policy comparison that removes the ball after its first contact cluster
ends, preserving the original trajectory until removal. Use source plus matched
quiet/ball checkpoints at 100 and 300, both seeds/backends, eight poses and +X.
No springs, controller, training or Arena change has been made. Details and
evidence are in [PERTURB_LEARNING_PLAN.md](PERTURB_LEARNING_PLAN.md).

**First Perturb pilot executed (2026-10-06):** all four continuation arms and
896 assessments in [PERTURB_LEARNING_PLAN.md](PERTURB_LEARNING_PLAN.md) are
complete. Final 120-second quiet Stand passes 64/64. Forward 90 N recovery
is 4/8 for both seed-301027 arms; seed 301028 reaches 6/8 with quiet continuation
and 7/8 with 60 N push practice, on each backend. The improvement does not
replicate across both seeds. A failed initial-environment fingerprint guard
also remains recorded; a reset-only probe isolated tiny GPU foot-load
differences but cannot reconstruct unsaved original start components. Keep
the Stand selection and fix initial-state evidence before another comparison.
This successor does not reopen the completed twenty-step ledger or promote
either the earlier curriculum or this pilot.

This section governs the next work. The numbered phases below remain the
historical experiment record; their completion does not imply learned Stand.
The user approved this planning correction, then authorized the next steps.
Phases 42-44 record the completed offline recovery-data comparison. Phases 45-46
record the control-representation implementation and two fresh PPO screens.
Phase 47 records the fresh duration curve, stopped at six minutes by its rule.
The user subsequently authorized ten autonomous next steps without waiting for
another Proceed. The bounded sequence below records that authorization.
All ten steps are now complete, including diagnosis (Phase 48), a two-seed
matched-update reward comparison and final verification (Phase 49).
The subsequent twenty-step batch is also complete: frozen-command diagnostics
(Phase 50), input/source audit (Phase 51), replicated actor-rate comparison
(Phase 52) and a fresh third-seed duration check (Phase 53).
The second twenty-step batch is complete: independent duration replication
(Phase 54), saved-trace comparison (Phase 55), a conditional fresh extension
(Phase 56), and batch verification/documentation (Phase 57). Phase 56 achieved
the first autonomous 12-second learned Stand milestone.

### Verified status

- **Physical feasibility established:** Phase 6's predictive teacher passed
  all eight native and eight GPU starts for 60 seconds with balance and natural
  posture, without passive springs or external support.
- **Autonomous 12-second learned Stand achieved on one fresh seed:** the final
  Phase 56 actor (seed 301024) passed **16/16** at **600 PPO cycles / 2,457,600
  transitions**, after **1,000.970159 synchronized training seconds** (16.7 min).
  Every original native/GPU start remained balanced and naturally posed for
  the entire **12.000960 seconds**, with no sampled departure. Both pre/post
  control-frame gates passed throughout. No teacher takeover or passive springs.
  The subsequent frozen 60-second assessment passed only **2/16** (one per
  backend), with mean survival about 33.3 seconds. Thus sustained Stand is
  not established for that 600-cycle actor.
- **Autonomous 60-second learned Stand reproduced on two fresh seeds:** active
  milestones 3-4 trained seeds **301025 and 301026** to **1,200 cycles /
  4,915,200 transitions each**, using **1,775.571491 / 1,759.730805 synchronized
  training seconds** respectively. Both final actors passed **16/16
  full-interval 60-second assessments**, including every pre/post balance and
  natural-posture sample. Both source/trajectory verifications passed. This
  admits bounded robustness assessments; it does not establish recovery or
  broad initial-pose generalization. Milestone 5 subsequently passed all 160
  small spine-reset cases. Milestones 6-7 passed all 160 small horizontal
  chest-push/control cases across both actors. The resumed milestones 8-20
  are now complete; the latest results and limitations are summarized above.
- **Actor-rate improvement replicated:** 3e-4 beat 1e-4 on both fresh seeds
  under the predeclared aggregate rule at equal 100-cycle budgets. Use 3e-4
  for the next isolated experiment, retaining reference_v9 and the same AMP/
  target-PD recipe. This is not a threefold learning-speed claim.
- **The 100-to-300 duration gain replicated across three fresh seeds:**
  301022/301023/301024 pass the fixed aggregate rule on both backends. Effects
  are uneven across starts: the latter two seeds improved 5/8 and 4/8 survival
  pairs respectively. The **300-to-600 improvement is observed on one seed**:
  mean survival 3.539866 -> 12.000960 seconds, minimum 2.833560 -> 12.000960,
  longest valid posture 2.683548 -> 12.000960, and Stand 0/16 -> 16/16.
  Six hundred is the first assessed passing checkpoint, not a general learning
  threshold; this does not imply linear gains from more transitions or time.
- **Changing policy commands contribute:** all four Phase 49 actors had
  longer mean survival with live commands than with their initial action held
  constant. Local joint PD feedback remained active in both conditions. This
  supports useful changing commands on these starts, not sufficient balance.
- **The quiet-speed reward change did not replicate:** seed 301018 regressed
  versus its matched reference arm (2.811 vs 3.029 seconds native mean), while
  seed 301019 improved (3.644 vs 3.402 seconds). Both backends agree on the
  direction of each seed's result. Retain reference_v9; do not adopt the
  quiet-speed multiplier or select its best seed as evidence of success.
- **Failure diagnosis narrowed the question:** 46/48 Phase 47 rollouts first
  violated horizontal speed, with no preceding torque saturation; all later
  failed the pelvis-height threshold. This identifies the observed failure
  sequence, not its cause or an established reward/actuator remedy.
- **The earlier 1e-4 duration screen was mixed:** in that fresh
  seed, three-to-six-minute minimum survival increased by 0.233 seconds, but
  mean survival decreased by 0.113 seconds on both backends. Mean continuously
  balanced/natural posture decreased from 2.240 to 2.173 native / 2.177 GPU.
  The predeclared rule stopped the run at six minutes; ten minutes was not run.
- **The short-run gain replicated across two fresh seeds:** in Phases 45 and 46,
  target-PD PPO beat both its zero-action baseline and trained torque on every
  original start on each backend. Phase 46's longest continuously balanced/
  natural-posture interval averaged 2.317 seconds native / 2.319 seconds GPU,
  compared with the zero-action PD baseline's 1.025 seconds. Sustained Stand
  and unseen-pose robustness are still unestablished.
- Phase 42 admitted 29/32 teacher recoveries. Phase 43 used the 10,080 successful
  native pairs as added supervision; its 10,800 GPU pairs stayed validation-only.
- Phases 40 and 41 used frozen actors and made no training updates. Phase 41
  rejected removing velocity clipping for these actors; retain clip 10.
- The offline recovery-data intervention is closed at its tested budget: its
  fitting improved on both seeds while its survival effect reversed. Phase 45
  instead trains PPO with a different control representation. Its active local
  feedback is explicit; passive springs remain zero. The representation's short
  screen has replication. The first 1e-4 duration curve was mixed; the 3e-4
  100-to-300 gain has three fresh-seed instances, and one 600-cycle actor now
  passes the autonomous 12-second gate. Reliability still requires the
  sustained-duration check and another independently trained successful actor.

### Success gates

1. **Autonomous 12-second Stand:** a final learned actor must pass all eight
   original starts independently on native and GPU physics (16/16). Use the
   existing survival and final two-second balance/natural-posture gates, with
   no teacher takeover. Report first departure and longest continuously valid
   interval as well, so a final-window pass cannot conceal earlier instability.
   **Achieved in Phase 56 on seed 301024; entire sampled interval valid.**
2. **Sustained 60-second Stand:** the same frozen actor must pass those starts
   for 60 seconds on both backends, maintaining balance and natural posture
   throughout the sampled control frames. Replicate the successful learning
   configuration with an additional fresh seed before declaring it reliable.
   These original-start results do not establish unseen-pose robustness.
3. **Small perturbations:** only after the autonomous standing gates pass,
   define a bounded disturbance curriculum and its recovery criteria. Teacher
   recovery and a natural reset pose cannot satisfy the learned-actor gates.

### Next bounded work and decision

**Current progress (2026-10-06): 20/20 substantive experiments complete and verified.**
Milestone 11: 144/208 continuous Stand and 176/208 recovery. Milestone 12:
129/208 continuous Stand and 161/208 recovery. Seed 301026 recovers from 90 N forward pushes in only 1/8 starts per
backend, versus 8/8 for seed 301025; its backward result is 7/8 native and
8/8 GPU. Strong 180 N forward/backward
pushes defeat both actors; lateral 180 N pushes allow recovery but interrupt
continuous balance. These frozen assessments made zero learning updates.

**Milestone 13 verification amendment, declared before retry:** the original
attempt is preserved. Its initial GPU qpos/qvel matched, but foot-load sensors
differed by up to 0.01855 N and actions by 2.68594e-6, exceeding the original
1e-6 cross-run action criterion. Own-observation replay already passed. The
successor retains 1e-6 initial qpos/qvel checks, verifies each actual observation
through the same actor at 2e-6, reports sensor/action differences, and requires
exact held-command constancy. This does not retroactively pass the original
criterion. Retry both actor matrices in new `20261006-02` roots; 160 accepted
cases plus 16 disclosed attempted cases, zero training. Repair is not a milestone.

**Resumed by user (2026-10-06):** complete the original twenty substantive
experiments without another approval. Seven are complete. Resume milestone
8's second actor in `stand-diagonal-push-301026-20261006-01`, preserving the
interrupted October 5 root unchanged. The resumed matrix has an additional
80-case budget; interrupted cases remain disclosed rather than counted as
completed evidence. Continue the predeclared milestones 9-10, then select
bounded amplitude/generalization or learning experiments from their results.
Arena selection stays on the verified seed-301026 actor during research.

**Paused by user (2026-10-05):** stop research and continue tomorrow. Seven
substantive experiments are complete. Milestone 8's seed-301025 diagonal
matrix finished (80/80 continuous Stand and recovery passes, verified).
Its seed-301026 matrix was stopped after the quiet native/GPU reports; the
partial root remains preserved and is not a completed experiment. Resume
that actor in a new output root using the sealed seed-301025 diagonal root
as provenance, then continue the remaining milestones. No trainer remains
running. Working ledger: `stand-milestones-20261005-01/ledger.json`.

The user separately requested loading the latest best verified Stand actor
into Arena. Select seed 301026's final 1,200-update actor: it ties seed 301025
on the completed sustained, reset and axial-push gates and is the latest
verified replication. Its matching 374-input / 33-target-PD Arena adapter
and export are complete. The user explicitly approved "Build and load Arena".
The build succeeded with zero errors (four existing ONNX-addon nullable
warnings), and Godot confirmed loading the selected bundle in the primary
Stand Arena. This deployment work does not count as an eighth research
experiment. No standalone tests or further research runs were started.

Arena bundle: `stand-arena-301026-20261005-01`; checkpoint SHA-256
`96e786ca67938cca8582d535bce98c75e1b0fa97c26d7358ff1ab6bcf016de76`.
ONNX conversion maximum action difference was **1.7881393e-7** over 2,055
saved observations. Scene case 0 uses a constant copy of its original reference
pose, with zero reference velocity and phase. The original model, weights,
normalizers, target-PD settings, actuator limits and zero passive springs are
preserved. Primary Arena and Stand viewer now select this bundle; the older
direct-torque Arena remains available separately. Previous scene/viewer and
adapter files are backed up in the bundle's `backup` directory. Evidence:
`arena-selection.json`, `export.log` and `godot.log` in that bundle. The
60-second admission comes from the sealed training assessments; loading the
interactive viewer is not an additional independent admission experiment.

**Active authorization: twenty substantive experiment milestones (2026-10-05).**
The user clarified that a step means an experiment-level advance, such as
verifying 60-second Stand or independently reproducing learning. Preparation,
individual checkpoints, native/GPU halves, verification and documentation do
not count separately. The previous two bookkeeping-based ledgers remain
historical; this batch uses the clarified definition. No further Proceed is
needed within these twenty milestones. Experiments include their required
implementation, behavioral assessments and saved-result verification.

The intended sequence is: (1) frozen 60-second Stand, (2) an independently
trained replication including sustained assessment, (3) unseen natural reset
poses, (4) feedback dependence of the stable actors, (5) small horizontal
disturbance recovery, (6) independent-seed recovery replication, (7) recovery
across disturbance directions, (8) recovery at later standing times,
(9) repeated disturbances, (10) disturbance-amplitude boundary,
(11) held-out disturbance combinations, (12) longer repeated-disturbance
endurance, (13) matched no-disturbance retention after the stress screens,
(14) a bounded recovery-learning intervention if the frozen actors fail,
(15) its matched training control, (16) independent learning replication,
(17) held-out recovery generalization, (18) quiet-Stand retention of the
resulting actors, (19) repeated-disturbance retention, and (20) a final
independent stress validation. Each item is a complete experiment, not a
single direction, seed checkpoint or file operation. Detailed magnitudes,
budgets and decision rules are recorded before each experiment. This is an
adaptive research queue: failed prerequisites replace downstream items with
specific diagnosis/correction/replication experiments; passed screens advance
the boundary without unnecessary retraining. Replacements and actual counts
must be explicit. Do not claim all twenty until twenty experiments finish.

**Milestone 1 protocol:** evaluate the frozen seed-301024, 600-cycle actor for
60 seconds on all original eight starts on both native/GPU backends. Exact
same 33-motor target PD, observations, normalizers, reference, model, actuator
limits and passive-spring-free plant. Require every pre/post sampled balance
and natural-posture gate to remain valid throughout; stop each case at
physical FAIL or the duration limit, recording any earlier departure. At most
16 cases; no training, rescue resets or teacher control. New duration-capable
runner/verifier preserve all sealed 12-second sources. Compare initial state
and action to the sealed successful assessment with absolute tolerances
1e-6 for initial state/action (no relative tolerance).

**Milestone 1 completed:** the frozen 60-second assessment passed **2/16**
(1/8 native, 1/8 GPU). Native mean survival 33.325582 s; GPU 33.329749 s.
The shortest native episode survived 15.817932 s; its first departure was
15.017868 s. Thus the existing 12-second result remains valid, but sustained
standing failed. Evidence: `stand-sustained-301024-20261005-01`. All saved
gates/controls and actor immutability passed verification. No perturbations
are admitted. Milestone 2 substitutes a causal controller diagnostic before
another learning investment; the fresh-seed reproduction remains queued.

**Milestone 2 protocol:** compare the sealed live actor with (a) its first
action held constant and (b) normal live actions until 10 seconds, then that
action held constant. Keep local motor PD and every physical setting fixed.
Each new condition uses eight original starts on both backends, 60 seconds
or physical FAIL, maximum 32 new cases. The sealed milestone-1 live traces
are the control. Verify live commands before the hold and exact constancy
after it, with initial state/action tolerance 1e-6 and one-control-frame
survival tolerance for early deaths before the hold. A held command is a
diagnostic ablation, never a learned feedback solution. Compare per-start
survival and full-interval Stand; do not promote a held-command wrapper.

Milestone 2 verification amendment, before replay: all 32 physical cases
completed in `stand-sustained-feedback-20261005-01`, but its original cross-run
prehold action tolerance 2e-5 failed on GPU (max 0.000577614 over ten seconds).
Native prehold actions matched exactly; initial GPU difference was 2.98e-8.
Keep the attempted runner and outputs unchanged. A separate verifier will
replay each saved prehold observation through the sealed actor on CPU and
require action agreement within 2e-6, then exact held-command constancy.
The independent-rollout prefix difference remains reported, not relabeled as
meeting the original tolerance. This checks whether the intended controller
actually generated each rollout without further physics or training.

**Milestone 2 completed:** replay errors were <=8.95e-8, below the predeclared
2e-6 tolerance; all held commands were exactly constant after the hold point.
Live mean survival was 33.325582 native / 33.329749 GPU seconds, versus
2.181424/2.181424 with the initial action held and 13.440658/13.515665 holding
after ten seconds. Every survival pair regressed under both holds; neither
hold passed any sustained case. Changing neural commands are beneficial,
although insufficient for reliable 60-second standing. The original failed
cross-run-prefix criterion remains explicitly failed. Accepted verification:
`stand-sustained-feedback-verified-20261005-01`.

**Milestone 3 protocol:** independently train fresh seed **301025** (sampling
301726), with exactly the seed-301024 recipe. Assess initial/100/300/600/1200
updates for 12 seconds, and at 600/1200 for 60 seconds (also at an earlier
12-second all-pass checkpoint if one occurs). Stop on 16/16 full-interval
60-second Stand; otherwise stop at 1,200 updates or a 2,400-second synchronized
training cap. Maximum **1,200 updates / 4,915,200 transitions / 144 assessment
cases**, including possible early sustained assessments. No weights,
normalizers, discriminator or optimizer state are reused. No coefficient,
reward, controller or passive spring changes. Evaluation must preserve all
learner/world/optimizer/RNG state. No perturbations.

This replaces the previously queued 600-cycle-only replication after the
60-second failure. It tests whether the observed duration improvement extends
to sustained balance; it does not assume 600 cycles suffice. The 100-to-300
rule is reported but is not an early-stop gate here: the bounded experiment
targets sustained behavior after the existing three-seed short-duration
evidence. Select the last reached checkpoint, never the best checkpoint.
If the synchronized time cap interrupts before an endpoint, preserve it as
incomplete rather than silently extending the cap. A second fresh seed is
required after the first sustained success before any perturbation work.

**Milestone 3 completed:** fresh seed 301025 reached **16/16 continuous
60-second Stand** at 1,200 cycles / 4,915,200 transitions / **1,775.571491
synchronized training seconds** (29.59 min). Every native/GPU case survived
60.004800 seconds with all pre/post balance and posture gates valid. At 600,
short Stand was 16/16 but sustained Stand only 4/16, mean survival 34.811118
native / 34.813202 GPU. The later checkpoint is the selected final result.
No reward or controller change was needed for this seed. This is one
independently trained sustained success; milestone 4 must reproduce it.
Evidence: `stand-sustained-learning-301025-20261005-01`; saved-result and
provenance verification passed. Actual assessments: 112 cases, no extra
verification physics or optimizer updates.

**Milestone 4 conditional protocol, recorded before execution:** if milestone
3 passes 16/16 full-interval 60-second Stand, repeat its entire bounded
learning protocol with fresh seed **301026**, sampling **301727**, using
`train_stand_sustained.py`. The prior run supplies configuration and provenance
only, never initialization tensors. Same initial/100/300/600/1200 assessments,
60-second assessments at 600/1200 or earlier 12-second all-pass checkpoints,
stop at the first sustained 16/16 success, at most 1,200 cycles / 4,915,200
transitions / 2,400 synchronized training seconds / 144 assessment cases.
Report any failed replication instead of replacing its seed. Two successful
independent actors are required for the subsequent bounded disturbance screens.

**Milestone 4 completed:** seed 301026 passed **16/16 continuous 60-second
Stand** at 1,200 cycles / 4,915,200 transitions / **1,759.730805 synchronized
training seconds** (29.33 min). At 600 cycles it passed 14/16 short cases and
12/16 sustained cases, with mean sustained survival 49.641471 native /
49.635221 GPU seconds. Both failures first departed through horizontal speed.
The final checkpoint passes every sampled balance/posture gate for all eight
starts on both backends. Evidence and provenance verification passed; actual
assessment count was 112. This supplies the second independent sustained actor
and admits the conditional robustness experiments below. Evidence:
`stand-sustained-learning-301026-20261005-01`.

**Milestone 5 conditional protocol:** after milestone 4 supplies a second
independently trained sustained success, assess unseen upper-body reset poses
on both admitted actors. Use a quiet control and four deviations: Spine_rx
plus/minus 2 degrees and Spine_ry plus/minus 2 degrees, with all other initial
coordinates unchanged. The reference remains the original natural pose;
these are physical initial deviations, not new reference targets. Each
condition uses the original eight starts on both backends for 12 seconds:
maximum 160 cases total, zero optimizer updates. Preserve all initial contact
and posture failures instead of filtering starts. Report strict continuous
Stand separately from recovery: survive the interval, begin two continuous
valid seconds within two seconds of initialization, and finish with two valid
seconds. Compare every disturbed case to its same-actor quiet control. This
tests local reset generalization; it cannot establish broad pose robustness.

**Milestone 5 completed:** both actors passed every quiet and perturbed reset
condition: **160/160 continuous Stand and recovery passes** across the two
seeds and native/GPU backends. All saved initial-state transformations, gates,
control formulas, force schedules (zero here), actor hashes and provenance
verified. This establishes local generalization to these four 2-degree spine
deviations only. Evidence: `stand-reset-pose-301025-20261005-01` and
`stand-reset-pose-301026-20261005-01`. No training updates.

The new challenge runner will also support subsequent force experiments.
Before each one, record force vectors, application body, onset, duration,
repeat schedule, horizon and case budget. For pulses, recovery additionally
requires valid standing before the first pulse and a two-second valid window
beginning within two seconds after each pulse, before the next pulse. Strict
continuous Stand remains a separate measure. Onset and duration are rounded
up to control frames; recovery timestamps have one control-frame tolerance.
Original passive springs stay
off; the existing local target PD remains unchanged. The successful seed-301025
quiet trace has native minima/maxima of 0.9001 m pelvis height, 8.7341 degrees
pelvis tilt, 0.10494 m/s horizontal speed, 268.36 N individual foot load and
2.337 degrees whole-body reference error. These are observed quiet margins,
not evidence that perturbations are already solved.

**Milestone 6 conditional protocol:** with two admitted sustained actors,
measure the first actor (seed 301025) under a quiet control and four chest
COM pushes: world +X/-X/+Y/-Y, 10 N for 0.1 seconds beginning at 5 seconds.
Use 12-second episodes, eight original starts on both backends, maximum 80
cases and no learning. This is approximately 1 N s per push after discrete
time rounding, on a 69.626 kg dummy. Apply the recovery criteria above and
report all failures. A quiet-control failure invalidates interpreting the
corresponding disturbed case as a push-specific regression.

**Milestone 6 completed:** seed 301025 passed **80/80 continuous Stand and
recovery cases**, including quiet controls and all four horizontal push
directions. Verification passed. A saved-trace causal check on native +X
found an exact pre-push state match to quiet, actual impulse **1.00008 N s**,
maximum root-velocity difference **0.01491635 m/s**, and maximum actor-command
difference **0.03111476**. The applied disturbance changed motion and neural
commands; these cases did not merely record an unused force. Evidence:
`stand-small-push-301025-20261005-01`. No training updates.

**Milestone 7 conditional protocol:** repeat milestone 6 with the independent
seed-301026 actor, preserving every condition, horizon and decision rule.
Maximum 80 cases and zero learning. This is an independent learned-policy
replication, not a second backend counted as another experiment. Results of
milestones 5-7 determine whether subsequent work measures a robustness
boundary or introduces a bounded learning correction.

**Milestone 7 completed:** seed 301026 independently reproduced the small-push
result: **80/80 continuous Stand and recovery passes**, with every quiet
control valid and all evidence/provenance checks passing. Evidence:
`stand-small-push-301026-20261005-01`. Together milestones 6-7 establish the
specified 10 N, 0.1-second axial-push tolerance on both actors. They do not
establish larger-force recovery or a general disturbance policy.

**Milestones 8-10 conditional protocols:** if both small-push experiments
pass every recovery case, retain the frozen actors and broaden one property
at a time. Each milestone evaluates both independent actors and both backends;
the actor/backend halves do not count separately.

- **8, direction coupling:** quiet plus four diagonal chest pushes with
  components (+/-10/sqrt(2), +/-10/sqrt(2), 0) N, preserving 10 N magnitude,
  0.1-second duration, 5-second onset and 12-second horizon. Maximum 160 cases.
- **9, onset dependence:** quiet plus +X chest pushes of 10 N for 0.1 seconds
  at 0.5, 20 and 45 seconds, with a 60-second horizon. Maximum 128 cases.
  This compares initial settling, established standing and late standing
  without changing push magnitude or direction.
- **10, repeated disturbances:** quiet plus three-pulse schedules at 5, 10
  and 15 seconds, each 10 N for 0.1 seconds, over 20 seconds. Compare repeated
  +X, alternating +X/-X/+X, and changing directions +X/+Y/-X. Maximum 128
  cases. Recovery must pass after each pulse, not merely at episode end.

All three have zero optimizer updates. Use the same continuous-Stand and
recovery criteria, keep every case, and require quiet controls. Record the
outcomes before selecting any amplitude or recovery-learning experiment.

**Milestone 8 completed (2026-10-06):** diagonal recovery generalized across
both actors: **160/160 continuous Stand and recovery passes**, with quiet
controls included. All source/trajectory checks passed. Accepted second-actor
root: `stand-diagonal-push-301026-20261006-01`; first-actor root remains
`stand-diagonal-push-301025-20261005-01`. The interrupted second-actor attempt
is retained separately and contributes no accepted result or extra milestone.
Milestone 9 now measures the same small push at early, middle and late onsets
over a full 60-second episode.

**Milestone 9 completed (2026-10-06):** **128/128 continuous Stand and recovery
passes** across both actors and all early/middle/late push conditions. Every
case survived 60.0048 seconds; both quiet controls passed. Evidence roots:
`stand-push-onset-301025-20261006-01` and
`stand-push-onset-301026-20261006-01`. Source and saved-trajectory verification
passed, with zero training updates. No delayed failure was observed within
the tested minute after the specified small push.

**Milestone 10 completed (2026-10-06):** **128/128 continuous Stand and
per-pulse recovery passes** over the repeated-push schedules on both actors.
Every quiet, same-direction, alternating-direction and changing-axis case
survived 20.0016 seconds. Source, force schedule, delayed control and physical
gate verification passed. Evidence: `stand-repeated-push-301025-20261006-01`
and `stand-repeated-push-301026-20261006-01`. Zero training updates. Ten of
the twenty substantive milestones are now complete; the amplitude matrices
are admitted by these small-push results.

**Milestones 11-12 conditional amplitude protocols (2026-10-06):** after the
small-push temporal and repeated-push matrices, measure a bounded amplitude
response instead of assuming small-force success transfers to stronger pushes.
Milestone 11 uses seed 301025; milestone 12 repeats the complete matrix with
independent seed 301026. Each includes quiet control plus 30, 90 and 180 N
chest pushes along world +X/-X/+Y/-Y, lasting 0.1 seconds at 5 seconds in a
12-second episode. Maximum 208 cases per milestone, zero optimizer updates.
Nominal impulses are 3, 9 and 18 N s (actual discretized values are recorded).
Report survival, strict continuous Stand and timed recovery separately, with
all failures retained. These are boundary measurements, not an admission rule
requiring every strength to pass. Any failed quiet control limits the causal
interpretation of its paired push cases. A failed small-push prerequisite will
trigger a documented amendment before these larger-force matrices execute.

These replace the earlier unspecialized amplitude/held-out placeholders with
a full response curve and its independent learned-policy replication. Results
will determine the remaining causal, retention and recovery-learning work;
individual force levels and directions are not separate counted milestones.

**Milestone 13 planned causal protocol:** compare the live actors from the
amplitude matrices against holding the neural command issued at two seconds,
while retaining the exact same local joint PD. Both seeds receive quiet
control and the four 180 N, 0.1-second chest pushes at five seconds over a
12-second horizon: maximum 160 new cases, no training. The matched live
conditions are reused from milestones 11-12. Replay each saved pre-hold
observation through the corresponding actor (absolute tolerance 2e-6), require
exact held-command constancy thereafter, verify initial state/action pairs,
and retain all control failures. This distinguishes ongoing neural feedback
from the local actuator response after standing has actually been learned;
milestone 2 only assessed the earlier unsuccessful 600-update actor. A held
controller is a diagnostic ablation and will not replace the learned policy.

**Milestone 13 completed:** successor verification passed all 160 held-command
cases; none passed continuous Stand or recovery. Quiet held-command survival
was 5.688/5.696 seconds (seed 301025, native/GPU) and 4.680/4.682 seconds
(seed 301026), versus 12.00096 seconds with live feedback. Local PD stayed
active. This demonstrates the usefulness of continuing neural corrections
for quiet standing. The early quiet failures, especially before the push in
seed 301026, prevent a clean push-specific causal conclusion. Original failed
attempt remains preserved with its additional 16 cases; no learning occurred.
Evidence: both `stand-push-feedback-30102{5,6}-20261006-02` roots.

**Milestone 14 partial result:** seed-301027 quiet control completed and verified
1,200 cycles / 4,915,200 transitions in 1,792.117300 measured training seconds.
Final quiet 60-second Stand: 16/16. Push recovery: 50/64 at 90 N, 32/64 at
180 N. Forward 90 N: 1/8 per backend; forward/backward 180 N: 0/8. The fresh
curriculum arm is running; this arm alone does not complete the milestone.

**Milestones 14-15 planned matched learning correction:** the first amplitude
matrix found 180 N forward/backward failures despite 90 N success. After its
independent replication and the causal ablation are recorded, compare fresh
quiet training against gradual push exposure. Milestone 14 uses seed 301027,
quiet arm first; milestone 15 uses seed 301028, curriculum arm first. Each
milestone includes both arms; initialization, normalization and all optimizer
state start from scratch in every arm. Require equal initial full tensor
hashes within each seed pair. No checkpoint or optimizer state is reused.

Both arms retain the existing 64x64 sampling, three actor epochs, 3e-4 actor
rate, reference_v9/AMP rewards, 60-second episodes, 33-motor delayed target PD
and spring-free plant. The only training difference is force exposure. Both
train quietly for 600 updates. Curriculum updates 601-900 ramp from 10 N
toward 180 N; updates 901-1200 use 180 N. Exactly, for zero-based update u,
amplitude is zero below 600, then 10+170*min((u-600)/300,1). First 32 worlds
cover all eight reference poses in each of the four axial directions; the
other 32 remain quiet. Chest pulses start at the first control frame at or
after five seconds and recur every ten seconds for 0.1 seconds. The quiet
arm uses zero forces throughout. Actual force buffers and episode step counts
are recorded and independently reconstructed for every collected transition.

Each arm stops at 1,200 updates or a 2,400-second synchronized training cap;
an interrupted budget is incomplete, never silently extended. Assess initial,
600 and 1,200 checkpoints for 12-second quiet Stand, the final checkpoint for
60-second quiet Stand, and final 90/180 N pushes in four directions for 12
seconds. Maximum 192 assessment cases and 4,915,200 transitions per arm,
384 cases and 9,830,400 transitions per matched experiment. Reward identity,
force schedule, checkpoint pairs, evaluation isolation, physical gates and
delayed controls must verify. Always select the final checkpoint.

The curriculum improves only if each backend retains 8/8 quiet 60-second
Stand, loses no 90 N recovery cases and gains 180 N recovery cases against its
matched quiet arm. Report the full result even if this rule fails. The second
seed repeats the comparison without selecting a favorable replacement seed.
These two complete matched experiments replace the earlier separate learning,
control and replication placeholders; their arms/checkpoints are not counted
as extra milestones. Later generalization/retention work follows the actual
comparison outcomes. Arena remains on the existing verified actor.

**Milestone 14 completed:** both fresh matched arms and their comparison are
verified. Quiet 60-second Stand is 16/16 in both. Recovery at 90 N improves
50/64 to 60/64; 180 N stays 32/64. The predefined improvement rule fails
because stronger recovery did not increase. Forward 90 N rises from 1/8 to
6/8 per backend; forward/backward 180 N remain 0/8. Total learning: 2,400
cycles, 9,830,400 transitions, 3,644.306448 synchronized seconds; 384 assessment
cases. Both arms had identical fresh initial tensors. Their 600-cycle full
hashes differ despite identical quiet warm-up recipes, so GPU trajectory
variation remains a source of within-seed noise. Do not attribute the modest
90 N difference to curriculum alone without the independent comparison.
Evidence: `stand-push-learning-pair-301027-20261006-01` and its two arm roots.
Milestone 15 now runs seed 301028 with curriculum first, quiet second.

**Milestone 15 completed:** the independent seed-301028 matched comparison
is verified. Both arms pass 16/16 quiet 60-second Stand. Recovery at 90 N
regresses from quiet 60/64 to curriculum 54/64; 180 N remains 32/64. Forward
90 N falls from 6/8 to 3/8 on each backend. Forward/backward 180 N remain
0/8 in both arms. Total: 2,400 cycles, 9,830,400 transitions, 3,626.781235
synchronized training seconds, 384 physical assessment cases. Fresh initial
tensors matched within the pair and differed from seed 301027. Warm-up hashes
again differed before force exposure. The predefined curriculum improvement
rule fails in both independent comparisons; the 90 N effect changes sign.
Retain the quiet recipe and reject promoting this curriculum at this budget.
All four new final actors pass the existing sustained-standing gate. This
supports quiet standing, not stronger forward/backward recovery. No Arena
change. Evidence: `stand-push-learning-pair-301028-20261006-01` and its arms.
The remaining five preregistered experiments now assess frozen actors only.

**Milestones 16-20 preregistered before either matched learning comparison
finishes:** retain all four final 1,200-update actors (quiet/curriculum, seeds
301027/301028). No favorable actor or seed selection, extra learning, reward
change or Arena replacement. Evaluate eight starts on both backends. Original
seeds 301025/301026 provide the already-established sustained-Stand prerequisite;
any new actor failing quiet retention remains in the results as a regression,
not evidence of successful perturbation learning. Specifications are saved in
`stand-milestones-20261005-01/specifications/milestone-{16..20}-*.json`.

16. **Longer retention:** quiet 120-second Stand, all four actors; 64 cases.
    Require every balance/natural-posture sample to remain valid. Report each
    actor separately; success is 16/16 per actor, not an aggregate hiding a seed.
17. **Initial-motion generalization:** quiet and initial horizontal velocity
    +/-0.1 m/s on each world X/Y axis, 12 seconds; 320 cases. Reference velocity
    remains zero. Use the declared initial-recovery and strict Stand measures.
18. **Direction generalization:** quiet and four diagonal 90 N Chest pushes
    (vector magnitude 90 N), 0.1 seconds at 5 seconds, 12-second horizon;
    320 cases. These directions never occur in curriculum training.
19. **Duration versus impulse:** quiet and four axial Chest directions for
    30 N x 0.3 seconds and 180 N x 0.05 seconds, at 5 seconds, 12-second horizon;
    576 cases. Both deliver the same discrete 9.00072 N s as the existing
    90 N x 0.1-second assessments. Compare to those saved baselines; report
    actual applied force schedules rather than equating force with impulse.
20. **Combined generalization:** quiet; four individual +/-2-degree Spine_rx/
    Spine_ry initial offsets; and those same offsets combined with 90 N axial
    reversing pushes at 5/10/15 seconds, lasting 0.1 seconds each. Directions
    are +X/-X/+X, -X/+X/-X, +Y/-Y/+Y, -Y/+Y/-Y paired with the four offsets.
    Horizon 20 seconds, 576 cases. Pose-only controls distinguish reset failure
    from failure added by the sequence. Require recovery after every pulse.

Total new physical-case budget for 16-20: 1,856, zero optimizer updates. Each
number is one complete behavioral experiment across both seeds and both arms;
individual actors/backends, verification and bookkeeping never add milestones.
For 17-20, compare curriculum-minus-quiet recovery separately by seed/backend;
claim replicated improvement only if both seeds retain quiet controls and
improve their aggregate disturbed recovery on both backends. Report ties and
regressions. If preceding learning fails verification, preserve its evidence
and repair verification without counting a new behavioral milestone.

**Milestone 16 completed:** all four final actors passed all eight starts on
both backends for 120 seconds: **64/64 continuous natural Stand**. Every
pre/post balance and posture sample passed. Evidence and frozen-source
verification succeeded for all four `stand-generalization-16-*-20261006-01`
roots. No learning updates. This extends retention beyond the 60-second
training episode horizon; it is not a push-recovery result.

**Milestone 17 completed:** **320/320 continuous Stand and recovery passes**
across all four actors, quiet controls and +/-0.1 m/s X/Y initial velocities.
Every actor passed 80/80 cases; all evidence and source checks passed.
This local initial-motion generalization is shared by both recipes; it is a
ceiling tie and does not establish curriculum improvement. Zero learning
updates. Evidence: four `stand-generalization-17-*-20261006-01` roots.

**Milestone 18 completed:** **308/320 continuous Stand and recovery passes**
for quiet controls plus four unseen 90 N diagonal directions. Seed 301027:
quiet 76/80, curriculum 80/80. Seed 301028: quiet 76/80, curriculum 76/80.
All four result roots verified. The first-seed advantage becomes a tie on the
independent seed, so the preregistered replicated-improvement rule fails.
Report this as limited directional generalization, not universal robustness.
Zero learning updates. Evidence: four `stand-generalization-18-*-20261006-01`
roots. Milestone 19 now isolates force-duration effects at equal impulse.

**Milestone 19 completed:** **522/576 continuous Stand and recovery passes**,
all evidence verified. Seed 301027: quiet 122/144, curriculum 138/144. Seed
301028: quiet 136/144, curriculum 126/144. The curriculum effect again changes
sign; the replicated-improvement rule fails. For forward pushes, recovery
counts per backend for [30 N x 0.3 s, 90 N x 0.1 s, 180 N x 0.05 s] are
[4,1,1] for seed-301027 quiet, [7,6,6] for its curriculum arm, [6,6,6] for
seed-301028 quiet, and [4,3,3] for its curriculum arm. Every profile has the
same discrete 9.00072 N s impulse. The middle profile reuses the already
verified matched-learning assessments. Equal impulse does not guarantee an
equal recovery outcome, and peak force alone is insufficient to characterize
the disturbance. Zero learning updates. Evidence: four
`stand-generalization-19-*-20261006-01` roots.

**Milestone 20 completed:** **492/576 continuous Stand and recovery passes**,
all evidence verified. Seed 301027: quiet 118/144, curriculum 120/144. Seed
301028: quiet 132/144, curriculum 122/144. All quiet and pose-only controls
pass (320/320); combined pose/push sequences pass 172/256. The small first-seed
curriculum gain reverses on the second seed, so the replicated-improvement
rule fails. In seed-301027 curriculum's native first sequence, the per-pulse
recovery counts are 6/8, 6/8, then 2/8: surviving/recovering from one push does
not establish repeated recovery. All four `stand-generalization-20-*-20261006-01`
roots are verified. Zero new learning updates.

**Twenty-experiment batch closed:** all numbers 1-20 are complete. Ledger totals
are 4,208 accepted physical cases, 7,200 new PPO updates and 29,491,200 new
transitions. There are 44 verified result roots; comparison/verification roots
are not extra physical cases or milestones. The interrupted and failed original
attempts remain disclosed and preserved. Final completion audit checks the
20 completed ledger entries, all result seals, paired generalization decisions
and unchanged Arena scene hash without rerunning physics or training. No further
experiment is running or queued. The bounded push curriculum is not promoted.
A subsequent research decision should address stronger/repeated forward-backward
recovery; no additional compute is authorized by this completed batch itself.

**Completed authorization: second twenty-step batch (2026-10-05).** The user
authorized another twenty autonomous steps. This batch tests duration
repeatability with fixed settings; new artifacts use new roots and historical
sources remain sealed. Training and behavioral assessments are the requested
experiments; no general test suite, build, Arena promotion or perturbation is
included. The concrete sequence is:

1. Reconcile rules, current milestones and the proposed duration replication.
2. Verify the preceding batch's sealed provenance and completion state.
3. Predeclare the fresh seed-301023 replication and conditional follow-up.
4. Implement isolated duration execution and saved-result verification.
5. Initialize seed 301023 from scratch and assess its initial actor.
6. Train to 100 PPO updates with the unchanged 3e-4 recipe.
7. Assess all eight native/eight GPU starts at 100 updates.
8. Continue to 300 updates unless all 16 Stand cases pass at 100.
9. Assess the final checkpoint on both backends.
10. Verify saved gates, controls, initialization, isolation and update counts.
11. Compare the 100-to-300 effect with Phase 53, including paired starts.
12. Diagnose first departures and terminal failures from saved traces.
13. Compare early/late optimizer and reward logs against behavioral outcomes.
14. Select the predefined continuation or diagnostic branch from that evidence.
15. Record the branch decision and concrete follow-up protocol.
16. Prepare the follow-up using sealed helpers and isolated output paths.
17. Execute the bounded follow-up, respecting its stop rules.
18. Verify its outcome and report per-start behavior, not only averages.
19. Verify all new artifacts and summarize actual compute and assessment costs.
20. Update milestones, completed ledger and next decision.

**All twenty steps are complete.** Steps 1-10 produced the seed-301023
replication; 11-14 compared and diagnosed it; 15-18 ran and verified the
conditional seed-301024 extension; 19-20 verified the batch and updated this
record. The extension reached 16/16 Stand at 600 cycles. No training remains
running. The following branch specification is retained as the predeclared
historical protocol, not an instruction to launch another run.

**Predeclared conditional follow-up:** if seed 301023 and Phase 53 both pass
the unchanged 100-to-300 continuation rule, run fresh seed **301024** (sampling
**301725**) with checkpoints at 0/100/300/600 updates. Cap synchronized training
at **1,500 seconds**, maximum **600 updates / 2,457,600 transitions / 64 assessment
cases**. Stop immediately at an assessment with 16/16 Stand; at 300, continue
to 600 only if that seed passes the same 100-to-300 rule. This tests whether
gains continue beyond 300 with no coefficient change. Do not substitute a
best intermediate actor for the final reached checkpoint. If replication
fails or ends with an early Stand pass, use saved-trace/learning-log diagnosis
as the follow-up instead of more training; record the result and next protocol.
No automatic 60-second gate or additional training follows within this batch.

**Completed authorization: twenty autonomous steps (2026-10-05).** The user
authorized continuation without further Proceed prompts within this batch.
The order is evidence-dependent; each new simulation/training protocol will
be recorded before execution. Historical artifacts remain sealed. No general
test suite, build, hour-long run or Arena promotion is included. Perturbations
remain gated by learned quiet Stand.

1. Reconcile instructions, current plan and historical provenance.
2. Predeclare the four-actor live-versus-held-command diagnostic.
3. Implement isolated execution and saved-result checks.
4. Reassess all four frozen actors with normal live commands.
5. Assess the same actors holding their first action constant.
6. Compare paired behavioral outcomes across both backends.
7. Verify repeatability, actor immutability, command traces and physical gates.
8. Diagnose the policy input/output changes in the saved trajectories.
9. Audit the relevant normalization, timing and training implementation.
10. Select one evidence-backed next hypothesis; reject unsupported alternatives.
11. Predeclare its controlled experiment, budgets and decision rules.
12. Implement the isolated intervention or diagnostic instrumentation.
13. Execute the first controlled comparison.
14. Verify its physics/learning contract and saved outcomes.
15. Execute an independent-seed or independent-actor replication as applicable.
16. Verify the replicated comparison.
17. Resolve whether the effect replicates and which mechanism it supports.
18. Perform a bounded follow-up selected by those findings, without a coefficient sweep.
19. Verify the complete batch and preserve its inputs and outputs.
20. Update the behavioral milestones, outcome and next direction.

All twenty steps are complete. Step 18 took the replicated-gain branch;
steps 19-20 verified the batch and recorded the outcome. The prior ten-step batch below
remains the historical completed record.

**Completed authorization: ten autonomous steps (2026-10-03).** This superseded
the separate-approval wording of the earlier Phase 47 proposal for this bounded
sequence only. No general test suite, build, Arena promotion or hour-long run
was included. The sequence below is complete.

1. Verify the saved duration experiment's input and output hashes. **Done.**
2. Diagnose first failed gates and terminal physical failures per start. **Done.**
3. Check preceding motor saturation and joint-limit proximity. **Done.**
4. Compare PPO/AMP logs and per-start changes between duration checkpoints. **Done.**
5. Predeclare one intervention, matched comparison and selection rule. **Done.**
6. Implement isolated reward/experiment instrumentation and saved-result verification. **Done.**
7. Run the fresh seed-301018 paired comparison. **Done; candidate did not win.**
8. Verify its saved gates, control, initialization, reward application and decision. **Done; passed.**
9. Repeat the same paired comparison with seed 301019 and reversed arm order. **Done; candidate won this seed only.**
10. Verify the complete replication, record the result and choose the next direction. **Done; verified, gain not replicated.**

The saved-trace analysis is completed as Phase 48. Its single reward hypothesis
was compared in Phase 49 and did not produce a replicated improvement. No
further reward intervention was selected within these ten steps.

The isolated 33-motor target-PD representation retains its replicated short-run
advantage. Phase 47 does not support extending this run automatically: both
backends failed the six-minute continuation rule, while remaining above their
fresh initial behavior. Keep the six-minute checkpoint as the declared final
result; the three-minute checkpoint remains a labeled intermediate assessment.

**Original handoff proposal, superseded by the active substantive-milestone batch above:** first assess the frozen Phase 56
`stand-duration-extension-20261005-01/updates-600/model.pt` actor for **60
seconds** on the same eight native/eight GPU starts. Keep its normalizers,
33-motor target-PD controller, model, reference and gates unchanged. Require
continuous balance/natural posture over every sampled frame, with no teacher,
reset rescue or training. Preserve paired checkpoints and control/provenance
checks in a new duration-capable verifier; existing 12-second helpers enforce
a 720-frame limit and must remain sealed. Stop on physical FAIL and report
every departure even if the actor later recovers. At most 16 assessment cases.

If that sustained gate passes, independently replicate the successful learning
protocol with fresh seed **301025**, sampling **301726**: actor rate 3e-4,
all other settings unchanged, initialization/100/300/600 assessments,
1,500-second synchronized-training cap, stop on 16/16 Stand and the same
300-cycle continuation rule. Assess a passing final actor for 60 seconds
before claiming reliability. If the first frozen 60-second check fails,
diagnose its saved traces before investing in another training run. Original
starts do not establish unseen-pose robustness; perturbations remain gated.

Both twenty-step batches are finished. No
hour-long run, coefficient sweep, passive springs, Arena promotion or
perturbations follows automatically. The offline recovery-data intervention
remains closed. Production configuration, including `scripts/config.ps1`,
was not changed by these isolated experiments.

Hour-long training runs remain deferred until a short comparison demonstrates
autonomous behavioral improvement and a specific longer-run hypothesis is
approved. Preserve historical artifacts and keep Arena promotion separate.

## Scope

This document is the plan and progress record for the new Stand learning series.
Earlier attempts are historical evidence only. No learned checkpoint, optimizer,
discriminator or normalization from those attempts initializes this series.
Existing artifacts and Arena selections remain preserved.

The first experiment screens PPO rollout size with three minutes of training per
configuration. It compares final native Stand behavior, not the best intermediate
checkpoint or training return alone.

## Phase 1: parallel environments and PPO cadence

Only environment count changes:

- `envs-2048`: 2,048 environments x 64 control steps = 131,072 transitions/update.
- `envs-256`: 256 environments x 64 control steps = 16,384 transitions/update.
- `envs-64`: 64 environments x 64 control steps = 4,096 transitions/update.

Each configuration constructs a fresh actor, critic, AMP discriminator and
normalization with seed **301010**. The seed is reset immediately before learner
construction so environment allocation cannot change network initialization.
Canonical SHA-256 hashes over all initial state tensors must match. All eight
initial native cases must also match. Each arm starts with zero updates and samples.
Optimizers, experience buffers and RNG streams are constructed anew.

Fixed settings:

- 180 synchronized training-update seconds per configuration. Finish the current
  update before stopping; report actual duration and overshoot. Setup, assessments,
  logging and checkpoint writes are outside this budget.
- 64 control steps per rollout; approximately 1.067 simulated seconds per world.
- 33 bounded direct-torque motors, including neck control; two-step command delay.
- Zero passive joint stiffness and no position servo. Existing damping, contacts,
  torque limits and joint limits are preserved.
- Stationary eight-pose reference: `logs/mimickit-steps/continuous-stand-02/stationary_reference.npz`.
- `reference_v9`: 80% task imitation and 20% AMP style reward.
- Existing licensed Stand and weight-shift style demonstrations; no learned AMP state.
- Actor/critic learning rate 0.0001; discriminator learning rate 0.00025.
- SGD optimizers, gradient clip 1.0, fixed 1,024-unit two-layer networks.
- Actor epochs 5 and minibatch multiplier 4; critic epochs 2 and multiplier 2.
- Discriminator epochs 1 and multiplier 8; replay capacity 200,000 histories.
- Discount 0.99, TD lambda 0.98, PPO clip 0.2, fixed action standard deviation 0.1.
- Training episode limit and native assessment limit: 60 simulated seconds.
- Initial and final assessments only, using the same eight deterministic native
  MuJoCo cases. Physical failures terminate early. Periodic assessment interval
  is 1,000,000,000 updates so cadence cannot favor an arm in this short screen.
- Sequential GPU runs. No overlapping trainer, Arena export/selection, Godot run,
  standalone unit test or C# build is part of this experiment.

MimicKit scales minibatches with environment count. This screen therefore tests
the complete existing environment-count configuration: smaller rollouts and smaller
minibatches together. Actor optimizer steps/update remain 80; critic steps/update
remain 64; discriminator steps/update remain 8. It does not isolate rollout size
from absolute minibatch size or reproduce the full original DeepMimic algorithm.

## Assessment and decision

Rank **final** actors by:

1. Number of surviving cases that finish with the existing whole-body standing gate.
2. Number of surviving cases that finish with the existing balance gate.
3. Number of 60-second survivors.
4. Minimum survival duration across the eight cases.
5. Mean survival duration across the eight cases.

Also record per-body tilt/pose errors, termination reasons, updates, transitions,
actual training seconds, trainer elapsed seconds and training throughput. Report
pose values from failed trajectories as failure diagnostics, not sustained natural
posture. Starting assessments are a fairness check, not training experience.

Equal training time is the primary comparison requested here. Experience counts
will differ. One seed and three minutes select a provisional next configuration;
they do not establish convergence, unseen-state robustness or natural movement.

## Later phases

1. Repeat promising environment counts across additional fresh seeds and compare
   both equal-time and equal-transition learning curves.
2. Isolate absolute minibatch size and epoch count. The released DeepMimic
   4,096/256/one-epoch proportions are a comparison arm, not an assumed optimum.
3. Compare active target/PD control against direct torque using the selected PPO
   schedule. Passive springs are a separate physical change, not a prerequisite.
4. Validate exact reference feasibility on the spring-free plant and inspect
   applied torques, pose drift and joint-limit bracing before changing rewards.
5. Require sustained native standing with measured posture before motion tracking
   or physical-hit recovery. Future execution is separate from this three-run screen.

## Reproduction

Use the isolated Mimic virtual environment and pinned clean MimicKit checkout.
Run each command sequentially with a new output directory; do not supply
`--initialize-from`.

```powershell
& mujoco_rig/mimic/.venv/Scripts/python.exe -B -u -m mujoco_rig.mimic.train_torque_stand `
  --mimickit D:/Proyectos/Juegos/Tools/MimicKit `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --out logs/mimickit-training/stand-fresh-config-screen-20261001-01/envs-2048 `
  --seconds 180 --envs 2048 --seed 301010 --reward-mode reference_v9 `
  --episode-seconds 60 --assessment-seconds 60 --evaluation-interval 1000000000 --comparison
```

Repeat with `--envs 256` / `envs-256`, then `--envs 64` / a fresh `envs-64` directory.
The completed series uses `envs-64-retry-01` for the valid 64-environment run;
the original `envs-64` directory records an excluded startup failure.
Each run saves initial, final and best checkpoints, full AMP sidecars, immutable
physics/reward provenance, initial/final learning-curve entries and update logs.
No historical policy initializes these runs.

## Progress

- Phase 1 completed on 2026-10-01. No historical checkpoint was loaded.
- `envs-2048`: completed, 31 updates / 4,063,232 transitions in 182.906 training seconds; 0/8 final standing passes. Final survival range: 0.817-1.383 seconds.
- `envs-256`: completed, 59 updates / 966,656 transitions in 182.905 training seconds; 0/8 final standing passes. Final survival range: 0.817-1.400 seconds.
- `envs-64`: completed in `envs-64-retry-01`, 71 updates / 290,816 transitions in 180.266 training seconds; 0/8 final standing passes. Final survival range: 0.900-1.150 seconds. The first launch had a mistyped reference path and stopped before learner construction or training; its request and error log are retained separately.
- Initial state equality: verified across all three valid runs. Fresh tensor SHA-256: `a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`. The complete initial eight-case native assessment is identical, with zero starting updates/samples and no loaded checkpoint.
- Final assessment comparison: completed. Ranking under the declared criteria: 64, then 2,048, then 256 environments.
- All saved physics, reference, demonstration, reward, code hashes and learner settings match; `envs` is the only differing protocol field. Historical documentation, global configuration and all three protected Arena scenes have unchanged SHA-256 hashes.

## Phase 1 results

All three final policies have **0/8 whole-body standing passes, 0/8 balance passes,
and 0/8 survivors at 60 seconds**. The shared untrained baseline survives
0.317-0.383 seconds. Training delays failure, but none establishes sustained Stand
or a natural standing posture.

- **2,048 environments:** 31 PPO updates; 4,063,232 transitions; 182.906 training
  seconds; 183.922 trainer elapsed seconds. Final survival minimum/mean/maximum:
  **0.817 / 1.061 / 1.383 seconds**. Average update: 5.900 seconds; throughput:
  22,215 transitions per training second. Four cases terminate on low pelvis
  height and four on non-foot ground contact.
- **256 environments:** 59 PPO updates; 966,656 transitions; 182.905 training
  seconds; 184.219 trainer elapsed seconds. Final survival minimum/mean/maximum:
  **0.817 / 0.998 / 1.400 seconds**. Average update: 3.100 seconds; throughput:
  5,285 transitions per training second. One case terminates on low pelvis height
  and seven on non-foot ground contact.
- **64 environments:** 71 PPO updates; 290,816 transitions; 180.266 training
  seconds; 181.438 trainer elapsed seconds. Final survival minimum/mean/maximum:
  **0.900 / 1.017 / 1.150 seconds**. Average update: 2.539 seconds; throughput:
  1,613 transitions per training second. All eight cases terminate on non-foot
  ground contact.

The training budgets overshoot by 2.906, 2.905 and 0.266 seconds respectively
because an update is completed before stopping. Trainer elapsed time includes
assessments and checkpoint/log work after learner construction; it excludes setup.

Posture diagnostics confirm failure rather than stable standing. Mean body tilt
at termination, in Spine / Chest / Head order, is:

- 2,048 environments: 79.5 / 82.0 / 93.6 degrees.
- 256 environments: 89.6 / 103.7 / 103.7 degrees.
- 64 environments: 96.6 / 113.9 / 123.0 degrees.

These values are sampled at different failure times. They describe the resulting
falls, not posture quality at a common time. Each report retains all individual
case criteria, trajectory tilt summaries and joint/body reference errors.

## Decision and next experiment

**64 environments is the provisional candidate under the declared ranking:**
standing, balance and full-duration survival all tie at zero, but its worst case
lasts 0.900 seconds instead of 0.817. The 2,048-environment arm has the highest mean
survival, so retain it as the comparison baseline. The 256-environment arm leads
neither minimum nor mean survival in this screen.

Reducing the rollout from 131,072 to 4,096 transitions is a 32-fold reduction,
but produces only 71 versus 31 updates in this budget: approximately 2.3 times as
many updates, with about 14 times fewer transitions. The smaller minibatches and
fixed work per update change GPU efficiency. This result does **not** imply
2.3 times faster learning, and it does not support a linear parameter-to-learning
speed multiplier. The eight fixed cases and single seed are insufficient to choose
a long-run optimum; run order was not counterbalanced.

The initial proposal was to repeat 64 and 2,048 environments across fresh seeds.
The user subsequently prioritized other PPO settings and approved the actor-epoch
comparison described below. Additional-seed checks remain necessary before a
long-run selection. Active target/PD control remains a separate later experiment.
No long training run, spring change, global default change or Arena selection was
applied in Phase 1.

## Saved evidence

Series root: `logs/mimickit-training/stand-fresh-config-screen-20261001-01`.

- [Protocol and protected-file hashes](../../logs/mimickit-training/stand-fresh-config-screen-20261001-01/comparison-protocol.json).
- [Comparison results, ranking and fairness checks](../../logs/mimickit-training/stand-fresh-config-screen-20261001-01/comparison-results.json).
- [2,048-environment report](../../logs/mimickit-training/stand-fresh-config-screen-20261001-01/envs-2048/report.json).
- [256-environment report](../../logs/mimickit-training/stand-fresh-config-screen-20261001-01/envs-256/report.json).
- [64-environment report](../../logs/mimickit-training/stand-fresh-config-screen-20261001-01/envs-64-retry-01/report.json).

Use each arm's `model.pt` and `model.amp.pt` for the final assessed actor and AMP
state; `initial.pt` and `initial.amp.pt` preserve the fresh baseline. Log artifacts
are Git-ignored; this document is the durable project record. Comparison mode
skips Arena export/selection and historical progress-document updates. The completed
work consists of the three authorized training runs and their built-in assessments;
no standalone tests or compilation were run.

## Phase 2: actor PPO epochs

Authorized: 2026-10-01. Compare **1, 3 and 5 actor epochs** with 64 environments,
64 rollout steps and actor minibatches of 256 transitions. These configurations
perform 16, 48 and 80 actor optimizer steps per PPO update respectively.

Each arm receives 180 synchronized training-update seconds and starts from the
same fresh seed 301010, with no loaded checkpoint. Rerun the five-epoch baseline
in this series so all arms use the same trainer revision. Initial full-state
fingerprints and all eight baseline assessment cases must match both each other
and the Phase 1 baseline. Final actors use the same ranking defined above.

Only `actor_epochs` changes. Critic/discriminator epochs, all learning rates,
fixed action standard deviation 0.1, rewards, demonstration files, spring-free
torque control, physics and reference remain as in Phase 1. Evaluation interval
remains 1,000,000,000 updates; only initial and final assessments run. Evaluation
time is outside the training budget, so increasing this interval cannot buy more
training seconds in these comparisons.

Additive trainer option: `--actor-epochs`; omitting it preserves the pinned agent
default. No global PowerShell configuration or third-party MimicKit source changes.
Learning-rate and exploration-standard-deviation comparisons are subsequent
proposals, not part of this three-arm experiment.

Series root: `logs/mimickit-training/stand-fresh-epoch-screen-20261001-01`.

```powershell
& mujoco_rig/mimic/.venv/Scripts/python.exe -B -u -m mujoco_rig.mimic.train_torque_stand `
  --mimickit D:/Proyectos/Juegos/Tools/MimicKit `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --out logs/mimickit-training/stand-fresh-epoch-screen-20261001-01/epochs-1 `
  --seconds 180 --envs 64 --actor-epochs 1 --seed 301010 --reward-mode reference_v9 `
  --episode-seconds 60 --assessment-seconds 60 --evaluation-interval 1000000000 --comparison
```

Repeat sequentially with `--actor-epochs 3` / `epochs-3`, then
`--actor-epochs 5` / `epochs-5`, using new output directories.

### Phase 2 progress

- Phase 2 completed on 2026-10-01; no historical weights were loaded.
- `epochs-1`: completed, 76 updates / 311,296 transitions in 182.121 training seconds; 0/8 final standing passes. Survival range: 0.850-1.283 seconds.
- `epochs-3`: completed, 73 updates / 299,008 transitions in 181.812 training seconds; 0/8 final standing passes. Survival range: 1.033-1.200 seconds.
- `epochs-5`: completed, 71 updates / 290,816 transitions in 180.740 training seconds; 0/8 final standing passes. Survival range: 0.800-1.100 seconds.
- Initial state and native baseline equality: verified across all three arms and Phase 1. All start with zero updates/samples, no loaded checkpoint, and fresh-state SHA-256 `a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
- Final assessment comparison: completed. Ranking: 3, then 1, then 5 actor epochs.
- Saved learner configurations differ only in `actor_epochs`; all other protocol fields match. Protected scene, global configuration and historical-document hashes remain unchanged.

### Phase 2 results

Every final actor scores **0/8 standing passes, 0/8 balance passes and 0/8
60-second survivors**. Final survival minimum / mean / maximum:

- **1 epoch:** **0.850 / 1.035 / 1.283 seconds**. 76 PPO updates;
  311,296 transitions; 1,216 actor optimizer steps; 182.121 training seconds;
  183.516 trainer elapsed seconds; 2.396 seconds/update. Six cases terminate on
  non-foot ground contact and two on low pelvis height.
- **3 epochs:** **1.033 / 1.123 / 1.200 seconds**. 73 PPO updates;
  299,008 transitions; 3,504 actor optimizer steps; 181.812 training seconds;
  183.157 trainer elapsed seconds; 2.491 seconds/update. Seven cases terminate on
  non-foot ground contact and one on low pelvis height.
- **5 epochs:** **0.800 / 0.940 / 1.100 seconds**. 71 PPO updates;
  290,816 transitions; 5,680 actor optimizer steps; 180.740 training seconds;
  181.969 trainer elapsed seconds; 2.546 seconds/update. All eight cases terminate
  on non-foot ground contact.

Reducing five actor epochs to one removes 80% of actor optimizer steps per update,
but average update duration decreases only about 6%. Physics collection, critic
and discriminator work remain fixed. The one-epoch arm completes only five more
PPO cycles than the five-epoch arm in this time budget. Less optimizer work does
not translate proportionally into faster feedback or better final survival.

Three epochs leads both minimum and mean survival in this screen. It is the
**provisional actor-epoch candidate**, with 64 environments and the existing actor
learning rate 0.0001. All actors still fall rapidly; no standing or natural-posture
milestone was achieved, and no global default or Arena selection was changed.

Mean body tilt at failure, Spine / Chest / Head:

- 1 epoch: 70.7 / 95.8 / 116.5 degrees.
- 3 epochs: 109.5 / 107.2 / 129.3 degrees.
- 5 epochs: 99.7 / 108.4 / 124.5 degrees.

Failure occurs at different times; these are fall diagnostics, not a posture-quality
ranking. Full per-case pose errors and trajectory summaries remain in the reports.

### Repeated-baseline variability

The five-epoch arm repeats Phase 1's 64-environment configuration with identical
learner settings, physics/reference hashes, initial tensors, seed, initial native
assessment and 71 final updates / 290,816 transitions. Only the trainer source hash
changes between series, following the additive actor-epoch option. Its minimum/mean
survival is 0.800/0.940 seconds versus Phase 1's 0.900/1.017 seconds.

The logged training metrics already differ slightly at the first update. Identical
initialization is verified; deterministic training trajectories are not established
by this pipeline. The cause of the variation has not been isolated. This makes the
single-seed ranking provisional and warrants repeat runs before long training or a
permanent configuration choice.

### Follow-up decision

The user chose one-hour training with the current winner instead of another short
parameter screen. Phase 3 below executes that decision. Actor learning-rate and
action-standard-deviation comparisons remain optional future experiments; neither
was executed in this epoch screen.

### Phase 2 evidence

- [Protocol and protected-file hashes](../../logs/mimickit-training/stand-fresh-epoch-screen-20261001-01/comparison-protocol.json).
- [Comparison, fairness checks and ranking](../../logs/mimickit-training/stand-fresh-epoch-screen-20261001-01/comparison-results.json).
- [One-epoch report](../../logs/mimickit-training/stand-fresh-epoch-screen-20261001-01/epochs-1/report.json).
- [Three-epoch report](../../logs/mimickit-training/stand-fresh-epoch-screen-20261001-01/epochs-3/report.json).
- [Five-epoch report](../../logs/mimickit-training/stand-fresh-epoch-screen-20261001-01/epochs-5/report.json).

All arms preserve fresh initial and final checkpoints plus full AMP sidecars and
per-update logs. Only the authorized three training runs and their built-in
assessments were executed; no standalone tests, compilation or Godot run.

## Phase 3: fresh one-hour training with the selected configuration

Authorized: 2026-10-01. Start from a new untrained policy, critic, AMP discriminator
and normalization with seed **301010**; no checkpoint continuation. Use the selected
**64 environments, 64 rollout steps, three actor epochs and 256-transition actor
minibatches**. Actor learning rate remains 0.0001, fixed action standard deviation
remains 0.1, and every other learner/physics/reward setting matches Phase 2.

Budget: **3,600 synchronized training-update seconds**. Initial, periodic and final
native assessments are outside this budget, so actual elapsed time will exceed
one hour. Finish the current PPO update before stopping and record the overshoot.

Evaluate every **64 completed PPO updates**, plus initial and final assessments.
Each assessment uses the same eight cases and 60-second simulated limit, with early
physical-failure termination. Preserve the learning curve, measured intermediate
actors, separate best actor and final actor. Periodic assessments do not change the
reward or advance a curriculum in `reference_v9`.

Zero passive stiffness and direct torque control remain fixed. Fresh-state tensors
and the initial native assessment must match earlier phases. The run uses comparison
mode to preserve existing Arena selections and historical documents. No standalone
tests, compilation, Godot launch or additional parameter screen is part of this run.

Run directory: `logs/mimickit-training/stand-fresh-winner-60min-20261001-01`.

```powershell
& mujoco_rig/mimic/.venv/Scripts/python.exe -B -u -m mujoco_rig.mimic.train_torque_stand `
  --mimickit D:/Proyectos/Juegos/Tools/MimicKit `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --out logs/mimickit-training/stand-fresh-winner-60min-20261001-01 `
  --seconds 3600 --envs 64 --actor-epochs 3 --seed 301010 --reward-mode reference_v9 `
  --episode-seconds 60 --assessment-seconds 60 --evaluation-interval 64 --comparison
```

### Phase 3 progress

- Status: completed; 1839 PPO updates / 7,532,544 transitions / 60.010 training minutes.
- Initial state and baseline verification: passed. Fresh tensors and the eight initial native cases match Phase 2; the learner configuration matches except evaluation/output cadence.
- Latest assessment: update 1839; standing 0/8, balance 0/8, 60-second survivors 0/8; survival minimum/mean/maximum 1.717/1.986/2.400 seconds.
- Final assessment: standing 0/8, balance 0/8, 60-second survivors 0/8; survival minimum/mean/maximum 1.717/1.986/2.400 seconds.

### Phase 3 outcome

Completed 3600.590 synchronized training seconds (0.590 seconds overshoot), 1839 PPO updates and 7,532,544 transitions. Trainer elapsed: 3639.547 seconds, excluding setup. Recorded 30 initial/periodic/final assessments.

Final actor: **0/8 standing passes**, 0/8 balance passes, 0/8 full 60-second survivors. Survival minimum/mean/maximum: **1.717/1.986/2.400 seconds**.

Best measured intermediate: update 1088; standing 0/8, balance 0/8, survivors 0/8; survival 1.800/2.061/2.817 seconds. Best and final checkpoints remain separate; no Arena selection was applied. Natural movement is not validated by this assessment.

Protected global configuration, historical documents and scenes unchanged: True. Fresh initialization and selected learner settings verified: True.

Evidence: [hour results](../../logs/mimickit-training/stand-fresh-winner-60min-20261001-01/hour-results.json), [final report](../../logs/mimickit-training/stand-fresh-winner-60min-20261001-01/report.json), [learning curve](../../logs/mimickit-training/stand-fresh-winner-60min-20261001-01/learning_curve.json).

### Arena selection after training

The user requested checking the finished run and loading it into the Arena.
Exported the **final actor at update 1,839**, checkpoint SHA-256
`6cbe95a5452b8bcc8b0488932eb0f1d799d93396617c4b076278c00e8b6010b1`.
The best intermediate remains separately saved; it was not substituted for final.

Both `MimicStandArena.tscn` and `MimicStandTorqueArena.tscn`, plus the Stand viewer
selection, now point to this run's `arena-preview` bundle. Previous scenes and viewer
selection are preserved in `arena-selection-backup`. Physics and trained policy
weights are unchanged. Export conversion maximum action difference: 8.94e-8.

Opened the primary Stand Arena using the existing compiled assembly. Godot confirmed
loading the new bundle and reported the displayed case falling at **2.400 simulated
seconds**. Final native assessment remains 0/8 standing passes, with survival
minimum/mean/maximum 1.717/1.986/2.400 seconds. `R` restarts; `P` pauses.

This selection occurred after training completion. The hour report's unchanged-Arena
flags describe the training stage before this separately authorized export. No new
training, standalone tests, C# compilation or historical-document update occurred.

Evidence: [Arena selection](../../logs/mimickit-training/stand-fresh-winner-60min-20261001-01/arena-preview/arena-selection.json),
[Godot log](../../logs/mimickit-training/stand-fresh-winner-60min-20261001-01/arena-preview/godot.log).

## Phase 4: foundation audit before further training

The user approved investigating reference feasibility, reward scoring and backend
consistency, then explicitly authorized the physical diagnostics with "Run those
tests". Artifact inspection and the bounded physical tests are completed. No new
training, production physics/reward/configuration changes, C# compilation or Arena
selection changes occurred. Simulations below are diagnostic measurements only.

### Established findings

1. **Reference validation does not cover the current plant.** Its recorded model
   hash is the original `e2b9d6c1...`, whereas the spring-free training/Arena plant
   is `7d4a37ea9...`. The reference metadata itself states that static feasibility
   is not closed-loop stability. Its inverse-dynamics/action-window screening is
   conditional on root support and excludes passive equilibrium, initial command
   queue fill and between-frame dynamics. The root-wrench subcheck uses unbounded
   joint torques; this must not be read as current bounded-motor hold admission.
2. **All 39 named hinges lost passive stiffness; six have no motor.** Both wrists'
   three axes lost stiffness 1.93 and remain unactuated. The original three neck
   position actuators became motors, producing 33 direct-torque channels. Reference
   snapshots are finite, have zero velocity and fit the saved hinge-angle limits;
   those checks do not establish a dynamically controllable hold.
3. **Balance is possible on this plant in some postures.** A historical measured
   run with the exact same physics hash has six 60-second survivors and five balance
   passes, but zero standing passes. This is evidence only; no old checkpoint was
   loaded or reused. Intended natural-reference stability remains unestablished.
4. **The reward and the pass gate serve different purposes.** Training combines
   0.8 times dense imitation with 0.2 times AMP style reward. These coefficients
   are not guaranteed shares of the realized reward. Direct imitation excludes
   unactuated wrist rotations, while AMP features include every hinge position and
   velocity. Any practical conflict or unreachable style preference remains a
   hypothesis requiring attribution, not a confirmed cause.
5. **A 60-second episode limit does not provide equal reward weight over 60 seconds.**
   At approximately 60 Hz, discount 0.99 has a reward-weight half-life of about
   1.150 seconds; the 0.99 x 0.98 GAE trace has a half-life of about 0.382 seconds.
   These are discount/trace properties, not maximum learnable behavior durations:
   value bootstrapping can represent longer consequences. Changing them is not yet
   justified by this artifact audit.
6. **The export is numerically bound to the final actor.** ONNX conversion action
   difference is 8.94e-8, and saved checkpoint/model/ONNX hashes match the selection.
   This proves export consistency; it does not establish GPU/native physics parity.
   All eight final native failures are low-pelvis termination at approximately
   0.63-0.65 m, rather than non-foot contact termination.

### Prepared physical diagnostic protocol

Keep the current plant, reference, motor limits, two-step action delay, checkpoint,
reward and Arena selection fixed. Do not optimize any network.

1. Inspect bounded torque/contact feasibility at all eight stationary reference
   states, including the floating base and six unactuated wrist hinges.
2. Attempt five simulated seconds per case with a bounded deterministic controller,
   alongside zero-command and frozen-final-policy baselines. Gravity stays enabled;
   no root anchoring, hidden springs or forced poses. Report startup queue effects,
   torque saturation, joint/pose drift, loads and physical termination. A failed
   simple controller does not establish impossibility of the pose.
3. Attribute task and AMP rewards on those same traces. Compare controlled holds,
   crouching/distortion and falling using comparable time windows. Do not compare
   one last-step task diagnostic with a whole-rollout discriminator mean.
4. Replay identical commands from identical initial states in GPU and native
   backends for up to one simulated second per case. Report joint/root/load/input
   differences over time, not only identical configuration hashes. Any discrepancy
   must be tied to its implementation before changing the training settings.

The bounded diagnostic results determine whether the next change concerns reference
feasibility, controller design, reward attribution or backend consistency. No longer
training run is proposed until that evidence is available.

### Phase 4 evidence and progress

- [Saved artifact audit and diagnostic protocol](../../logs/mimickit-training/stand-foundation-artifact-audit-20261001-01/artifact-audit.json).
- Artifact audit: completed.
- Physical tests and replay: completed after explicit test authorization; results below.
- Training/reward/physics settings and selected Arena: unchanged.

### Authorized physical diagnostics: results

Accepted full diagnostic output:
[foundation summary](../../logs/mimickit-training/stand-foundation-tests-20261001-03/foundation-summary.json),
[full report](../../logs/mimickit-training/stand-foundation-tests-20261001-03/report.json).
The reusable diagnostic is `diagnose_stand_foundations.py`. All tests use the selected
final actor and the current spring-free 33-motor plant. Actor/discriminator/normalizer
tensors and checkpoint/model/reference/ONNX file hashes remained unchanged.

**Static equilibrium:** zero of eight exact full-body targets satisfy the bounded
torque/contact equations. Each target requires nonzero gravity compensation at the
six unactuated wrist hinges. Foot/toe contact-force Jacobians provide zero authority
over those wrist coordinates; wrist stiffness and friction loss are both zero, and
targets are inside their joint limits. This establishes an exact stationary wrist
target inconsistency, not an inability of the legs/body to balance in a nearby pose.

With wrist equilibrium equations excluded, all eight targets pass the rigid contact
force-distribution screen. Minimum peak motor use is **9.97%-13.21%** of authority.
This uses independent friction-pyramid rays at existing foot/toe contacts, no root
support and no fictitious joint-limit reactions. It does not prove the compliant
contacts will realize those forces or that a controller will maintain them.
Restoring the original wrist stiffness would permit a gravity-induced deflection;
it would not automatically make the exact zero-angle reference an equilibrium.

**Attempted five-second holds:** each case starts from its unchanged reference with
zero velocity and a zero-filled two-step command queue. Stop scoring a case at its
first physical failure; terminated native worlds are no longer integrated.

- Zero command: survival **0.317-0.400 seconds**, zero of eight survivors.
- Bounded joint PD plus static feedforward: **0.383-0.717 seconds**, zero survivors.
  Gains use a fixed 1.5 Hz natural frequency, critical damping and each reference's
  diagonal joint inertia; feedforward comes from the wrist-relaxed bounded screen.
  This is a diagnostic joint servo, without COM feedback or policy optimization.
  Its failure does not establish that a better balance controller cannot hold.
- Frozen final actor: **1.717-2.400 seconds**, zero survivors, reproducing the hour
  run's eight failure times. Maximum commanded/delayed torque is **73.38%** of motor
  authority; no command reaches the 98% saturation threshold. Wrist deviations
  reach approximately **12.6-28.9 degrees** before termination. The actor therefore
  is not continuously failing against the motor-torque ceiling.
- Accepted traces have no BADQPOS/BADQVEL/BADQACC solver warnings. None supplies a
  stable five-second natural hold; a successful controller baseline remains absent.

**Reward attribution:** every recorded post-step state uses the exact current
`reference_v9` implementation. Five one-hot reward evaluations reproduce the
production task reward within 2e-6 for nonterminal samples. Task reward is zeroed
at physical failure; AMP uses the same four-frame trace and the frozen final
discriminator/normalizer. Common-window comparisons use the first 0.233 seconds
of all eight cases, with no termination or unequal sample counts.

Across the final actor's 953 live/terminal control samples, task reward averages
**0.5844** and AMP reward **0.4997**. Weighted contributions are **0.4675** and
**0.09995**: AMP supplies **17.61%** of measured combined reward on this trace.
The raw task component means are pose 0.4352, joint velocity 0.8910, root pose
0.9518, root velocity 0.7760 and key positions 0.3996. Thus some components remain
high while the actor distorts and collapses; that alone does not establish a
specific reward exploit or justify replacing coefficients.

The small set of balance-gate states consists of the first two delayed, zero-command
frames per case: combined reward averages 1.5321 there, versus 0.2672 in the actor's
low-pelvis subset. Stable-looking initial states score higher than the observed
collapse states. No sustained good-hold trace exists for a long-duration comparison.

**GPU/native replay:** eight worlds start from exactly equal float32 qpos and zero
velocities, and receive the same recorded commands for **0.983412 simulated seconds**.
All existing topology, collision, motor, passive-property and solver-option checks
pass. Native solver state uses float64; the GPU solver uses float32. The pending
command queue and future-reference observations match exactly. No replay case fails
within this interval; this does not establish parity through the eventual fall.

Two controlled repeats in one process use seed 301010 and two Torch CPU threads.
Maximum GPU/native root position difference is **0.055 mm**, hinge difference
**0.000985 radians (0.0565 degrees)**, and per-body ground-force difference
**41.48 N**. Native qpos repeats exactly; GPU qpos differs by up to **0.0006083**
in mixed position/quaternion/hinge coordinates between repetitions. GPU ground
forces differ by up to **2.55 N**, and observations by up to **0.00667**.

A separate-process replay exhibited larger differences: **8.884 mm** root position,
**0.08949 radians (5.13 degrees)** hinge angle and **393.85 N** per-body force.
Its native trace was still identical. Keep this result as a reproducibility concern;
do not discard it because controlled repeats were closer. GPU float32 arithmetic
and atomic contact/constraint reductions in the installed implementation are candidate
mechanisms, not an isolated root cause. Their effect on closed-loop actor actions
and long-term learning remains unmeasured. Backend equivalence is **not cleared**.

Evidence: [controlled replay 1](../../logs/mimickit-training/stand-foundation-replay-controlled-20261001-01/backend-replay.json),
[controlled replay 2](../../logs/mimickit-training/stand-foundation-replay-controlled-20261001-02/backend-replay.json),
[repeat comparison](../../logs/mimickit-training/stand-foundation-replay-controlled-20261001-02/repeat-comparison.json),
[separate-process replay](../../logs/mimickit-training/stand-foundation-replay-20261001-01/backend-replay.json).

### Interpretation and next decision

Do not invest another long run on the assumption that additional transitions alone
resolve the present failure. The diagnostics establish a wrist/reference mismatch,
reproduce the learned collapse, and identify backend/contact reproducibility as an
unresolved issue. They do not identify a single cause of the pelvis collapse.

The next proposed work is to align passive wrist behavior with reference/AMP features,
establish a bounded delayed controller that actually holds a natural posture, and
isolate GPU contact reproducibility and its policy-input effects. Keep these decisions
separate from PPO throughput tuning. Re-enabling stiffness at every joint is not
supported by these measurements; no such change was made.

The initial arbitrary-gain controller attempt in `stand-foundation-tests-20261001-01`
developed numerical instability and is superseded for hold conclusions. Attempt `02`
failed on an incorrect `mj_fullM` Python binding call before completing a trace. Both
directories are retained; accepted hold/reward results come from `03` only.

## Phase 5: foundation follow-up

The user approved the next implementation, then authorized execution in response
to the explicit follow-up physics-test question. The diagnostic
`stand_foundation_followup.py` completed successfully. No new training, production
configuration changes, C# compilation or Arena selection changes occurred.

### Prepared changes

1. **Isolated passive-wrist reference candidate.** Optimize only the six unactuated
   wrist coordinates within existing joint limits. Require gravity/passive-force
   residual at most 1e-6 Nm and no unstable local potential-energy direction;
   neutral rotation around gravity is permitted. Preserve every other coordinate
   and all physical parameters. Retain dataset attribution, record derivation and
   discard inherited optimistic admission. Screen all eight derived poses against
   the current bounded motor/contact equations. A static pass is not a hold pass.
2. **Predictive bounded controller.** Run a separate native controller against an
   admitted candidate. Predict through the two committed command intervals on
   private simulation data, then use whole-body acceleration feedback and the
   existing compliant-contact inverse map to choose 33 bounded motor torques.
   Validate proposed torques through native forward dynamics. Gravity, physical
   limits, zero passive stiffness and the real two-step action queue remain in
   place. No root anchoring or external support force. Stop each case at its first
   physical/numerical failure and report a five-second hold with final-window
   balance/posture measurements. Sixty-second admission remains a separate gate.
3. **GPU numerical comparison.** Replay the same saved actor commands twice per
   variant: unchanged baseline; graph capture disabled; solver tolerance 1e-6;
   solver warm-start disabled. These numerical changes apply only to probe engine
   instances, never the source XML, training settings or Arena. Record pose, force,
   observation, constraint-count and solver-iteration differences. Read native
   counters before the evaluator's extra ending-state `mj_forward`.
4. **Policy-input sensitivity.** On each replay's recorded states, compare frozen
   actor outputs for native inputs, GPU inputs, and native inputs with only the
   two foot-load channels replaced by GPU values. This isolates input sensitivity;
   it does not establish a closed-loop failure or training effect.

The existing AMP demonstrations and discriminator feature layout are unchanged.
Any future wrist-feature omission would require consistent live/demo mapping and
a fresh discriminator/normalizer; the current AMP checkpoint must not be loaded
into a changed feature layout. Choose that change only after the reference/controller
measurements establish what is physically achievable.

Executed command:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.stand_foundation_followup `
  --run logs/mimickit-training/stand-fresh-winner-60min-20261001-01 `
  --out logs/mimickit-training/stand-foundation-followup-20261001-01
```

Outputs were created in a fresh directory. Source model/reference/checkpoint/export
and command-trace hashes remained unchanged. No network updates, automatic promotion
of candidates, production configuration changes or Arena selection changes occurred.

### Phase 5 results

Evidence: [summary](../../logs/mimickit-training/stand-foundation-followup-20261001-01/summary.json),
[full report](../../logs/mimickit-training/stand-foundation-followup-20261001-01/report.json),
[predictive controller](../../logs/mimickit-training/stand-foundation-followup-20261001-01/predictive-controller.json),
[GPU numerical comparison](../../logs/mimickit-training/stand-foundation-followup-20261001-01/gpu-numerics.json).

**A short physical standing baseline is established.** All eight candidate cases
survive **4.983732 simulated seconds** with the predictive controller. Every case
satisfies the existing balance and posture criteria throughout the final two seconds.
There are no solver warnings, external support forces, root anchors or added springs.
The controller uses the original bounded 33 motors and real zero-filled two-step
command queue. Private prediction never overwrites live physical state.

During the final two seconds, actuated hinge errors relative to the candidate average
**0.0507 degrees**, with a maximum of **3.164 degrees**. Pelvis heights remain
**0.9013-0.9031 m**. These are actual controller measurements on the current native
physics, not a neural policy result, GPU hold, or sixty-second standing admission.

**The candidate reference passes all eight bounded static screens.** Only the six
wrist angles changed; the largest individual adjustment is **11.14 degrees**.
Other pose coordinates, target velocities, gravity, damping, motor authority and
stiffness are unchanged. Wrist residual torques are below 1e-6 Nm and potential
curvature has no unstable direction; neutral gravity-axis rotations are allowed.
The candidate remains isolated at `passive-wrist-reference.npz` under the result
directory. The active reference and AMP demonstrations were not replaced.

**No GPU numerical variant reliably clears backend consistency.** Each variant was
repeated twice with identical saved commands, seed and Torch CPU thread count.
Disabling graph capture reproduces the large divergence. Changing tolerance or
disabling solver warm-start produces one close replay and one large-divergence replay;
neither is a supported fix from this sample.

- Baseline: maximum root difference **8.890 mm**, hinge difference **0.08953 rad**,
  per-body ground-force difference **393.78 N**, frozen-actor action difference
  **0.01394** in normalized torque units.
- Graph capture disabled: **8.894 mm**, **0.08961 rad**, **393.89 N**, **0.01395**.
- Tolerance 1e-6: **8.891 mm**, **0.08959 rad**, **393.89 N**, **0.01395** across the
  two replays. The second replay alone was closer; do not select it and ignore the first.
- Warm-start disabled: **8.891 mm**, **0.08959 rad**, **393.83 N**, **0.01395** across
  the two replays. Again, one close repetition does not establish improvement.

At the baseline's peak force divergence, **case 5 / Foot_L / 0.100008 seconds**, the
last solved constraint counts differ: native **45 rows**, GPU **57 rows**. Thus the
large discrepancy accompanies different active constraint sets, not merely different
force reporting. This does not yet isolate which contact-generation, limit, numerical
ordering or state-perturbation mechanism initiates that difference.

Replacing only the two foot-load observation channels changes the frozen actor's
commands by up to **0.01300 normalized torque units** in the baseline. This directly
measures sensitivity to contact inputs. These counterfactual outputs are not applied
during the replay; no closed-loop failure cause or training effect is established.
Native replay qpos remains exactly reproducible; GPU repetitions still differ.

### Revised foundation decision

The current spring-free motorized body can maintain a nearby natural reference with
a suitable physical controller for this short native test. This removes the need
to assume that every joint requires a spring to permit standing. It does not show
that the current learned actor can stand or that wrist adjustments alone cure its
pelvis collapse.

Keep the new reference and controller as diagnostic candidates. The next proposed
admission is a longer native hold and a GPU counterpart, together with targeted
contact/constraint-set tracing. Once those pass, collect physically valid controller
traces for a fresh learning experiment and align the reference/AMP feature contract.
Do not promote a numerical setting based on its best repetition or start another
hour run while backend transfer remains unresolved.

## Phase 6: sixty-second controller admission and contact tracing

Completed 2026-10-01. The user's continuation authorized the proposed longer
native/GPU holds and targeted contact diagnostics. No neural training was run.

Commands:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.admit_stand_controller `
  --out logs/mimickit-training/stand-controller-admission-60s-20261001-01
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.trace_stand_contacts `
  --out logs/mimickit-training/stand-contact-trace-20261001-04
```

Evidence: [controller admission](../../logs/mimickit-training/stand-controller-admission-60s-20261001-01/report.json),
[contact report](../../logs/mimickit-training/stand-contact-trace-20261001-04/report.json),
[historical matched poses](../../logs/mimickit-training/stand-contact-trace-20261001-04/historical-matched-state.json).

### Controller admission

The isolated passive-wrist candidate and unchanged predictive controller pass
**8/8 standing cases on native physics and 8/8 on GPU physics** for
**60.0048 simulated seconds**. Every case passes balance and posture at every
sampled control frame, including the complete final two-second window.

- Native final-window mean motor-joint errors per case: **0.0045-0.0143 degrees**;
  largest individual motor-joint error: **0.1230 degrees**.
- GPU final-window mean errors: **0.0045-0.0141 degrees**;
  largest individual error: **0.1224 degrees**.
- Final pelvis height across both backends: **0.9014-0.9031 m**.
- Native warning counters remain clear. GPU states stay finite and pass the same
  physical gates; native MuJoCo warning counters are unavailable on that backend.
- Each backend begins independently from the eight candidate poses and an empty
  two-step command queue. The same controller reads its own backend's live state
  and produces its own feedback commands. This is not an identical-command replay.
- The controller predicts privately with native MuJoCo on both backends. Actual
  GPU evolution still uses the production GPU solver and original bounded motors.
  No private prediction overwrites live state.

No passive springs, external support, root anchors or additional motor authority
were introduced. The candidate changes only the six passive wrist angles.
One sixty-second admission per backend establishes this operating point; disturbance
recovery, additional poses and repeated long-hold robustness remain unmeasured.

The CPU controller and its native predictive solves are an offline physical
baseline. These timings are not PPO training throughput measurements. This result
does not change the existing neural actor's approximately two-second collapse.

### Contact tracing

Two repetitions replay the old frozen actor's first nine saved commands on the
original reference: **36 physics substeps / 0.150012 seconds / eight cases**.
Graph capture is disabled only in the diagnostic instance so that every substep
can be observed. Source physics and solver settings are unchanged.

The accepted instrumented repetitions remain close: maximum root-coordinate
difference **0.000201 mm** and hinge difference **0.0000401 rad**. Their constraint
type counts and active body-pair contact counts agree at every traced substep.
Detailed host reads synchronize the GPU after each substep; these instrumented
results do not supersede the prior minimally instrumented divergent replays.

**All 576 matched-state probes agree on constraint type counts and active contact
pair counts:** 288 from the new substep traces and 288 from the earlier baseline
replays' saved ending poses. Each probe supplies exactly the same float32 pose to
both backends, with fresh solver history. Native MuJoCo promotes those values to
float64. Historical traces contain no velocities, so their probes use zero
velocity and compare constraint generation, not historical forces.

The historical case 5 pose at **0.08334 seconds** shows a concrete threshold:

- Its saved native pose generates **32 rows** on either backend: four joint-limit
  rows plus 28 pyramidal-contact rows, including one left-foot contact.
- Its saved GPU pose generates **28 rows** on either backend: four joint-limit
  rows plus 24 contact rows, with no left-foot contact.
- The saved poses differ by only **1.19e-7 m** in root coordinates and
  **1.52e-6 rad** in hinges. The native pose's left-foot contact penetration is
  about **4.50e-8 m** in the native matched-state probe.

These are ending-pose forward probes, not the subsequent step's original solved
45-versus-57 row counts. They establish that a tiny trajectory difference can
change contact membership near zero penetration. They support contact-threshold
amplification and do not establish whether precision, numerical ordering, solver
history or another mechanism first produced the trajectory difference.

The contact harness's first output directory retained a serialization failure.
Directory 02 completed the initial trace; 03 added historical probes; 04 is the
accepted result using the final harness. None are training runs or model candidates.

### Learning decision after admission

Spring-free standing is now demonstrated on both simulation backends at the
candidate reference. Adding springs everywhere is not required to make this
standing operating point physically achievable. The admission also removes the
need to demand exact native/GPU trajectory equality before preparing useful
standing demonstrations: a suitable feedback controller stands on both.

Next prepare an isolated teacher dataset from admitted physical rollouts, preserving
observations, bounded actions, delayed-control history, pose/velocity and gate
results. The AMP demonstration/reference contract must represent the physically
realizable passive wrists consistently. Start any new actor, critic, discriminator,
normalizer and optimizer fresh; do not reuse the failed hour checkpoint. Validate
a short neural imitation pilot and its native/GPU standing behavior before another
long PPO run. The existing traces cover these eight standing cases only and do not
establish a recovery curriculum.

No production reference, AMP assets, training defaults or Arena selection were
changed. The current learned checkpoint and Arena ONNX retain SHA-256
`6cbe95a5452b8bcc8b0488932eb0f1d799d93396617c4b076278c00e8b6010b1` and
`51e5094c9287237e8f582304b0d325f1b953b771b968a13eed4c5fb6eba8cc8b` respectively.

## Phase 7: physical teacher dataset and fresh neural imitation pilot

The user authorized this phase with "Proceed" after the Phase 6 next-step proposal.
Scope: collect physical demonstrations, align the candidate reference and AMP
motion data in an isolated bundle, train a fresh actor for three minutes and
evaluate its final autonomous behavior on native and GPU physics. No hour-long
PPO continuation or Arena promotion is part of this phase.

Commands:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.collect_stand_teacher `
  --out logs/mimickit-training/stand-teacher-20261001-01
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.train_stand_imitation `
  --dataset logs/mimickit-training/stand-teacher-20261001-01 `
  --out logs/mimickit-training/stand-imitation-pilot-20261001-01
```

### Dataset admission and feature contract

Dataset [report](../../logs/mimickit-training/stand-teacher-20261001-01/report.json)
and [protocol](../../logs/mimickit-training/stand-teacher-20261001-01/protocol.json).
Native and GPU recordings independently pass **all eight cases for 60.0048 seconds**,
with zero sampled balance/posture violations. Each backend records **28,800 unique
transitions**: 3,600 control intervals across eight cases. Every observation includes
the actual oldest-first delayed action queue, whose recorded values match the
observation's queue channels exactly.

The bundle contains the byte-identical spring-free plant, copied passive-wrist
stationary reference, pre-action observations/actions/pose/velocity, post-action
pose/velocity, delayed actions and gate results. It contains eight physical AMP
clips per backend, each with 3,601 states including the initial state. Manifests
bind model and clip hashes and preserve the original 100STYLE / Ian Mason /
CC-BY-4.0 attribution, with the physical-controller adaptation documented.

The native clips supply AMP training demonstrations. GPU clips remain separate
validation evidence. All **39 hinge positions and velocities**, including the six
passive wrist hinges, remain in the discriminator feature mapping: four history
frames of 91 features = **364 inputs**. No mirrored examples are introduced because
their physical admission was not measured. A finite 256-history sample is checked
through the existing AMP demonstration loader. This phase initializes but does
not train the discriminator; it validates the data/feature interface.

### Fixed pilot protocol

- Fresh actor, critic, discriminator, observation normalizer and optimizer state.
  Seed **301010**. Initial complete tensor hash:
  `a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`, matching
  the original fresh experiment initialization. No previous learned weights load.
- Existing two-layer 1,024-unit actor and 306-input / 33-output policy contract.
  The policy already observes both queued actions; no privileged controller inputs
  or additional observation channels are required for the recorded targets.
- **180 seconds of supervised actor optimization**, batch size 512, fresh Adam
  at 1e-4, gradient norm cap 1.0 and normalized-action mean squared error.
  This optimizer is specific to imitation; production PPO SGD settings are unchanged.
- Fit observation moments on native training observations only, using the existing
  StableNormalizer and clip of 10. GPU observations never fit normalization or gradients.
- Half of each minibatch samples the first two seconds to retain seating/transient
  examples; the other half samples the entire native trajectory uniformly.
- Evaluate the **final budget checkpoint**, without best-checkpoint selection.
  Teacher-state action errors and autonomous sixty-second native/GPU gate results
  are reported separately. GPU validation uses the same eight poses and does not
  establish unseen-pose or disturbance generalization.
- Repeated supervised sample presentations are not new physical transitions.
  PPO, critic and discriminator receive **zero optimization updates**. Critic and
  discriminator tensor hashes must remain unchanged; portable actor and AMP sidecar
  must match after saving.

### Pilot results and decision

Completed. [Full report](../../logs/mimickit-training/stand-imitation-pilot-20261001-01/report.json),
[action metrics](../../logs/mimickit-training/stand-imitation-pilot-20261001-01/action-metrics.json),
[native evaluation](../../logs/mimickit-training/stand-imitation-pilot-20261001-01/native/report.json),
[GPU evaluation](../../logs/mimickit-training/stand-imitation-pilot-20261001-01/gpu/report.json).

**The physical dataset is admitted; the neural pilot fails autonomous standing.**
After **180.0058 optimization seconds**, the final actor has **0/8 survivors and
0/8 standing passes on both backends**. Native survival ranges **0.7167-1.0334 s**,
mean **0.8001 s**; GPU survival ranges **0.7167-1.0501 s**, mean **0.8021 s**.
This is worse than the earlier hour-trained actor's survival, not a standing
improvement. The learned actor is not promoted to the Arena.

Teacher-state normalized torque RMSE falls from **0.06431 to 0.001102** on the
native dataset. Independent GPU teacher-state RMSE is **0.01826**, with
first-two-second RMSE **0.05091**. Native first-two-second RMSE is **0.001708**.
Maximum individual errors remain **0.5657 native / 1.1581 GPU**; a small aggregate
RMSE must not hide rare large command errors or imply closed-loop stability.

There are **60,983 supervised optimizer updates** and **31,223,296 repeated sample
presentations**, drawn from only **28,800 unique native physical transitions**.
These counts cannot be compared directly with PPO transitions or collection cycles.
Critic and discriminator hashes remain unchanged; both have zero updates, as does
PPO. The saved portable actor matches its full AMP sidecar. Final model SHA-256:
`9817402185bc88e899c34584f0df8339b842e62f57422c0d422c7d922340f218`.

Autonomous observations hit the existing normalizer's clip limit on an average
**17.5-22.0% of features per case**, versus **0.0756%** across native teacher data
and **0.1175%** across GPU teacher data. The corresponding teacher first-two-second
fractions are **1.666% native / 2.765% GPU**. Thus the autonomous actor quickly
visits inputs poorly represented by the narrow successful trajectories. This is
evidence of a distribution gap; it does not isolate clipping, sparse large action
errors or another factor as the unique cause of falling.

The next proposed experiment is teacher-labelled coverage of controlled departures
and learner-visited states, with recovery clips admitted physically and observation
moments fitted on the broader training distribution. Keep the plant, reference,
action queue and full wrist feature mapping fixed. Run another bounded fresh
imitation pilot and compare its autonomous native/GPU survival before any long
PPO run. Merely extending the current narrow-dataset fitting budget is not supported
by this result. No production reference, AMP defaults, checkpoint selection or
Arena assets changed during Phase 7.

## Phase 8: admitted teacher corrections on departure states

The user authorized this experiment with "Proceed" after the Phase 7 proposal.
Scope: query the same physical controller on states visited by the failed imitation
actor and by bounded torque departures, admit actual teacher recoveries, then run
another fresh three-minute imitation pilot with autonomous native/GPU evaluation.

Commands:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.collect_stand_corrections `
  --out logs/mimickit-training/stand-corrections-20261001-01
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.train_stand_imitation `
  --dataset logs/mimickit-training/stand-teacher-20261001-01 `
  --corrections logs/mimickit-training/stand-corrections-20261001-01 `
  --out logs/mimickit-training/stand-correction-pilot-20261001-01
```

### Fixed collection and pilot protocol

- Every collection episode resets to the same eight candidate poses with an empty
  two-step queue. Plant, reference, motor limits, controller and feature layout
  remain fixed. No external forces, springs or root-state edits are introduced.
- Four schedules execute the previous imitation actor for **0.1, 0.2, 0.4 or
  0.6 seconds**, then hand control to the teacher. Two schedules add independent
  uniform normalized torque noise of **+/-0.005 or +/-0.02** to teacher commands
  for the first two seconds. All executed motor commands remain bounded to [-1,1].
- Each episode lasts **12 seconds**. A case enters the correction dataset only
  if it survives without solver warnings and satisfies balance and posture
  throughout the final two seconds. Failed recovery cases are retained as
  diagnostic traces and excluded from training.
- Record pre-action observations, queried teacher actions, actually executed
  actions, native pose/velocity and pending queue. Departure-state labels are
  counterfactual teacher commands; their execution is not claimed. Subsequent
  recovery uses the actual teacher commands on live physics.
- Only a final four-second tail which passes every balance/posture sample becomes
  an additional AMP clip. Learner-controlled or failed motion never becomes AMP
  style data. Preserve all six passive wrist coordinates in the original 364-input
  history mapping. No mirroring or discriminator optimization occurs.
- Native admitted correction samples train the new actor; independent GPU
  correction samples are validation only. Fit observation moments on the original
  native data plus admitted native corrections, never GPU data.
- Construct all learner tensors fresh with seed 301010. The previous failed actor
  generates departure states only; its weights do not initialize this new learner.
  Keep the same 1,024-unit architecture, Adam rate 1e-4, batch 512 and **180-second
  optimization budget**. Half each minibatch samples native corrections, a quarter
  samples the original first two seconds, and a quarter samples the full original
  native teacher data. Production PPO settings and Arena selection stay unchanged.
- Report final teacher-state errors and final autonomous sixty-second native/GPU
  gate results separately. There is no best-checkpoint selection or long PPO run.

### Correction collection results

Collection completed. [Report](../../logs/mimickit-training/stand-corrections-20261001-01/report.json)
and [protocol](../../logs/mimickit-training/stand-corrections-20261001-01/protocol.json).

- Native: **26/48 case/schedule recoveries admitted**, producing **18,720 labels**,
  including **1,344 departure-state queries**, and 26 settled AMP tails.
- GPU: **23/48 admitted**, producing **16,560 validation labels**, including
  **1,098 departure-state queries**, and 23 settled AMP tails.
- Learner prefix 0.1 seconds: native **8/8**, GPU **7/8** admitted. Prefix 0.2 seconds:
  **8/8 on both**. Prefixes 0.4 and 0.6 seconds: **0/8 on both**.
- Torque noise +/-0.005: **8/8 on both**. Noise +/-0.02: native **2/8**, GPU **0/8**.
  Stronger perturbations therefore do not establish a robust teacher recovery basin.

Admission means surviving the twelve-second episode and passing the final
two-second standing window. It does not require balance during the departure.
Only admitted cases train the actor. Failed or unsettled cases remain trace evidence;
no labels from those cases enter either training or the admitted validation pool.

The new learner has **47,520 native training pairs** in total: 28,800 original
teacher samples plus 18,720 admitted correction samples. This is one aggregation
round against the previous failed actor, not repeated on-policy correction rounds
against the new actor. Initial full tensor hash still matches
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
Its AMP loader uses the eight original native clips plus 26 admitted settled tails;
the feature layout and fresh discriminator initialization are unchanged.

### Fresh correction-pilot results

Completed. [Full report](../../logs/mimickit-training/stand-correction-pilot-20261001-01/report.json),
[regression metrics](../../logs/mimickit-training/stand-correction-pilot-20261001-01/action-metrics.json),
[native admission](../../logs/mimickit-training/stand-correction-pilot-20261001-01/native/report.json),
[GPU admission](../../logs/mimickit-training/stand-correction-pilot-20261001-01/gpu/report.json).

**The new actor still has 0/8 standing passes and 0/8 survivors on both backends.**
Mean survival increases from Phase 7's **0.8001 to 1.0272 seconds native** and
**0.8021 to 1.0022 seconds GPU**. Native case survival ranges **0.7834-1.3168 s**;
GPU ranges **0.8001-1.2334 s**. This is a modest increase in this single fixed-seed
comparison, not a successful standing policy, replicated learning advantage or
improvement over the earlier hour-trained actor's approximately two-second survival.

Optimization lasts **180.0064 seconds**, with **50,515 updates** and **25,863,680
repeated sample presentations**. Training still uses only the 47,520 admitted
native state/action pairs. Critic, discriminator and PPO receive zero updates.
Critic/discriminator hashes remain unchanged and portable actor/AMP sidecar agree.
Final actor checkpoint SHA-256:
`a3572b0f26a4152af9e4ee6df5aa31867d20d92118885e82d879fc74b4ec1285`.

Final normalized torque RMSE:

- Original native teacher states: **0.001708**; first two seconds **0.003121**.
- Admitted native corrections: **0.003770**.
- Original GPU teacher states: **0.01132**; first two seconds **0.04732**.
- Admitted GPU corrections: **0.01795**.

Maximum individual command errors are still large: **0.4831 / 0.5200** on original
native / native correction states and **1.2944 / 1.2341** on corresponding GPU pools.
The improved GPU regression RMSE does not establish stable autonomous feedback.

Mean autonomous observation clipping is **15.86% native / 15.66% GPU**, with
per-case ranges **12.7-19.5% / 12.9-19.1%**. Broader moments reduce clipping compared
with the first pilot, but the actor continues to leave the demonstrated input
distribution. Data coverage, moment fitting and minibatch sampling all changed
together; this experiment does not isolate their individual causal effects.

Source snapshots are saved as `runner.py` in both Phase 8 output directories and
match their recorded protocol hashes. Protected input hashes remain unchanged.
The failed candidate is not exported or selected in the Arena.

### Next foundation measurements before more training

Use a fixed-weight observation-clipping ablation to determine whether relaxing
the current normalizer clip changes autonomous survival. Keep weights, stored
means/standard deviations, plant, reference and queue fixed; report every setting,
including failures. This isolates inference clipping without another optimization run.

Also measure the admitted physical teacher under the actual Gaussian exploration
used by PPO, with repeated fixed seeds and unchanged motor bounds. Current config
uses **fixed action standard deviation 0.1**, sampled during training; deployed
deterministic evaluation uses the mean action. The Phase 8 uniform perturbations
have standard deviations **0.002887** for +/-0.005 and **0.01155** for +/-0.02.
Thus PPO's configured pre-clamp noise standard deviation is approximately **35x**
the mild admitted perturbation and **9x** the stronger perturbation which admitted
few/no cases. The distributions differ, so these are scale comparisons, not a
measured Gaussian failure threshold or learning-speed ratio. No stochastic PPO
actions were used in either imitation-pilot evaluation.

These proposed diagnostics separate a remaining inference limitation from
training exploration relative to the teacher's measured recovery range. Do not
infer that more environments, more fitting time, lower exploration or added springs
will solve standing from the present results. No further training is started here.

## Phase 9: fixed-weight clipping and PPO exploration tolerance

The user authorized execution with "Proceed" after the Phase 8 proposal.
This phase performs physics diagnostics only: no actor, critic, discriminator or
optimizer updates and no Arena selection changes.

Command:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.probe_stand_inference_noise `
  --out logs/mimickit-training/stand-inference-noise-20261001-01
```

### Protocol

- Clipping ablation uses the final Phase 8 actor and the same stored observation
  means/standard deviations. Change only the in-memory inference clip: **10, 20,
  50 or off**. Motor outputs still clamp to [-1,1]. Eight poses per setting, a
  sixty-second assessment, one native trial and two GPU repetitions. Validate
  that every actor/normalizer tensor remains unchanged. Report actual clipping
  and departure beyond the original clip-10 envelope separately.
- Gaussian tolerance uses the unchanged physical predictive teacher on the
  admitted passive-wrist reference. Test **sigma 0, 0.0025, 0.005, 0.01, 0.025
  and 0.1**, each with seeds **301010, 301011 and 301012** on both backends.
  Each trial starts from the same eight poses with an empty two-step queue and
  cleared solver history. Total: 36 eight-case trials.
- Add independent per-motor Gaussian noise to the freshly computed teacher mean
  at **every control interval throughout a twelve-second episode**. This tests
  continuing exploration, not a two-second burst followed by quiet recovery.
  Commands use the existing bounds, action-to-torque mapping and delayed queue.
- Precompute common float32 standard-normal Torch draws on CPU and save them.
  Use identical draws across backends and noise scales. This matches PPO's
  Gaussian marginal, independent motor channels and action mapping; it does not
  reproduce the old CUDA RNG sequence or substitute the learned actor's mean.
- Standing admission still requires survival and the complete final two-second
  balance/posture window. Report surviving-but-unsettled cases separately.
  Twelve-second noisy admission is not a sixty-second robustness certificate or
  evidence that PPO can learn at that exploration setting.
- Save protocol, source snapshots, per-trial reports/traces and common noise.
  Check protected plant/reference/checkpoint hashes after completion.

### Completed results

[Summary](../../logs/mimickit-training/stand-inference-noise-20261001-01/summary.json),
[full report](../../logs/mimickit-training/stand-inference-noise-20261001-01/report.json)
and [protocol](../../logs/mimickit-training/stand-inference-noise-20261001-01/protocol.json).
All twelve clipping trials and 36 Gaussian trials completed. Counts below include
every assessed case, including failed and unsettled repetitions.

**Relaxing observation clipping does not rescue the frozen Phase 8 actor.**
Every clipping setting has zero standing passes on both backends. Mean survival
for clips 10, 20, 50 and off, respectively:

- Native, eight cases per setting: **1.0272, 0.8626, 0.8230 and 0.8855 seconds**.
- GPU, sixteen cases per setting across two repetitions: **1.0167, 0.8834,
  0.8376 and 0.8917 seconds**.

Clip 10 performs best in this fixed-weight comparison. Keep it for this actor.
The high fraction of inputs outside its fitted observation distribution remains
evidence of departure from the demonstrated states; lifting the bound alone makes
survival worse. This does not isolate the effects of normalization, training-data
coverage or supervised optimization on a newly trained policy.

**The physical teacher tolerates substantially less continuous Gaussian action
noise than PPO's configured standard deviation of 0.1.** Each count below is out
of 24 cases: eight initial poses repeated with three common noise seeds. Native
and GPU standing-pass counts, respectively:

- Sigma **0**: **24/24 and 22/24**; survivors **24/24 and 23/24**.
- Sigma **0.0025**: **24/24 and 21/24**; all standing passes also survive.
- Sigma **0.005**: **14/24 and 11/24**; survivors **14/24 on both**.
- Sigma **0.01**: **0/24 on both**; one native survivor remains unsettled,
  and no GPU cases survive.
- Sigma **0.025**: **0/24 on both**, with no survivors. Mean survival is
  **1.7418 seconds native / 1.7265 seconds GPU**.
- Sigma **0.1**: **0/24 on both**, with no survivors. Mean survival is
  **0.8320 seconds native / 0.8313 seconds GPU**.

Noise is added before the existing motor command clamp. At sigma 0.1, mean
per-case applied-noise RMS after clamping is approximately **0.09536 native /
0.09532 GPU**. The action bounds therefore do not remove the large perturbation.
The 0.0025 candidate has 40 times lower configured standard deviation than 0.1,
or 1,600 times lower variance. Neither number is a learning-speed multiplier.

### GPU repeatability and interpretation

The zero-noise GPU repetitions differ despite identical reset poses and the same
deterministic controller. Seed labels identify trial slots here;
at sigma zero they do not introduce an exploration perturbation:

- Trial labelled 301010: case 1 falls at **2.766888 seconds**; seven cases pass.
- Trial labelled 301011: all eight cases pass.
- Trial labelled 301012: case 3 survives twelve seconds but violates the final
  posture window for **0.066672 seconds**; seven cases pass.

The previous single 60-second GPU teacher admission remains a recorded result,
but these repetitions show that it does not establish repeatable GPU admission.
The source of this backend variation is not isolated by the noise sweep. Do not
attribute it to a particular solver setting or velocity mapping without evidence.
GPU reports do not expose MuJoCo's native warning counters; their false
`solver_warning` entries are placeholders, not proof of warning-free execution.
Survival, finite-state and sampled physical gates are still assessed.

This experiment uses the physical teacher's mean action, not a learned PPO mean.
It establishes a narrow measured exploration tolerance for that controller under
these twelve-second cases. It does not prove that PPO cannot learn with sigma
0.1, that reducing sigma solves the failed neural feedback policy, or that lower
noise improves convergence speed. Inference clipping and exploration noise are
different mechanisms: the clipping actor evaluations are deterministic.

The final report confirms **zero training updates**, unchanged frozen actor
tensors and unchanged protected plant, reference, checkpoint and agent-config
hashes. No Arena selection changes occur. Saved common-noise SHA-256:
`688ab176c41be35f45a64ac8013734748e24923bada16636ebe9a9f16a979911`.

### Proposed next comparison; not executed

First isolate the noiseless GPU repeatability failures with repeated fixed-pose
teacher holds and recorded reset/contact state. Keep the spring-free plant and
teacher unchanged while identifying the cause. A single successful GPU hold
must not replace the failed repetitions.

Then compare **fresh three-minute PPO runs at action_std 0.1 versus 0.0025**.
Use the same physical reference and admitted AMP clips for both, 64 environments,
64 rollout steps (**4,096 transitions per update**), three actor epochs and the
same evaluation interval, rewards, optimizers and deterministic assessment.
Initialize each learner from scratch with a paired seed; do not initialize from
either imitation pilot or the hour-trained checkpoint. Shared actor, critic,
discriminator and initial normalizer tensors must match between configurations;
the deliberately different fixed action log-standard-deviation tensor is excluded
from that common-initialization comparison and reported separately.

Compare final deterministic survival and standing admission alongside transitions
and PPO updates completed within the training budget. A reduced exploration
standard deviation is a candidate to measure, not a declared winning setting.
Neither production defaults nor a longer training budget change in Phase 9.

## Phase 10: noiseless repeatability and fresh PPO exploration comparison

The user authorized execution with "Proceed" after the Phase 9 results and
proposed sequence. This phase runs the named diagnostics and two fresh
three-minute PPO experiments. No production-default or Arena changes are planned.

### Execution protocol

- Record full-reset qpos/qvel, action queue, contact loads and solver state.
  Run two twelve-second native teacher holds and three twelve-second GPU holds
  on reused engines, then a sixty-second teacher hold on a fresh GPU engine.
  Every hold uses the unchanged controller, spring-free plant and passive-wrist
  reference. Report failures and unsettled cases rather than selecting a winner.
- Compare the first different state/action/contact frame across repetitions.
  Also replay exactly the same first one second of recorded GPU commands three
  times without teacher feedback, distinguishing solver variation from feedback
  sensitivity. Raw contact order and geometry-sorted reset contacts are saved
  separately; numerical divergence does not by itself identify a specific kernel.
- First diagnostic launch, `stand-repeatability-20261001-01`, stops before any
  rollout because the new recorder references `data` instead of the native
  engine's `datas` property. Preserve its failure record. Correct the recorder
  and restart as `stand-repeatability-20261001-02`; plant/controller are unchanged.
- PPO arms: `action_std=0.1` and `0.0025`, paired seed **301010**, each receiving
  **180 synchronized training-update seconds** excluding setup and evaluation.
  Each arm uses a fresh actor, critic, discriminator and normalizers, with no
  supervised fitting or pretrained initialization. Shared initial tensor hashes
  must match after excluding only the intentionally different fixed logstd.
- Keep **64 environments, 64 rollout steps, three actor epochs, evaluation every
  64 PPO updates**, `reference_v9`, the source optimizer/reward configuration and
  a sixty-second training horizon. Use the same eight original native teacher
  clips plus 26 admitted native correction tails for both arms, full 364-input
  AMP history, no mirroring. GPU teacher clips never enter training.
- Compare final budget checkpoints, not best intermediate checkpoints. Evaluate
  deterministic mean actions on native and GPU for sixty seconds/eight poses;
  failed cases stop contributing after termination. Save every intermediate
  native evaluation, updates, transitions, hashes and resource measurements.
  This is a single-seed short screening comparison, not replicated convergence
  evidence or a controlled comparison against the historical hour-trained run.

Commands:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.probe_stand_repeatability `
  --out logs/mimickit-training/stand-repeatability-20261001-02
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.1 --out logs/mimickit-training/stand-ppo-noise-0p1-20261001-01
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.0025 --paired-with logs/mimickit-training/stand-ppo-noise-0p1-20261001-01 `
  --out logs/mimickit-training/stand-ppo-noise-0p0025-20261001-01
```

### Repeatability diagnostic results

Completed. [Full report](../../logs/mimickit-training/stand-repeatability-20261001-02/report.json).
All six teacher holds pass all eight cases: two native twelve-second holds,
three GPU twelve-second holds and one fresh GPU sixty-second hold. These new
passes do not invalidate Phase 9's failed GPU repetitions or establish their
specific failure cause.

- Native repeated resets and all recorded trajectories match exactly, including
  qpos, qvel, teacher commands and foot loads throughout 720 control frames.
- GPU resets match exactly for qpos, qvel, empty pending actions, queue cursor,
  controls, time and zero warm-start accelerations. Geometry-sorted contact
  records also match exactly, while raw contact allocation order differs.
- GPU forward-solved reset values vary: ground-force component differences reach
  **9.1553e-5 N**, constraint generalized forces **2.4414e-4**, and generalized
  acceleration coordinates **3.3379e-4** in their respective physical units.
  This is measured after reset/forward, not evidence of retained nonzero warm-start
  accelerations from the previous episode.
- GPU state and foot loads first diverge on control frame **1 (0.016668 s)**.
  Teacher commands first diverge on frame **2 (0.033336 s)**. Thus the initial
  variation precedes differences in controller commands.
- Open-loop replays use exactly identical commands in all three trials, yet GPU
  states diverge on frame 1. First qpos-coordinate differences are approximately
  **3e-8 to 4.5e-8**. Over one second they grow to up to **0.03992 m** in root
  coordinates and **0.66562 rad (38.14 degrees)** in a hinge coordinate. These are
  replay sensitivity measurements, not autonomous teacher-hold failures.
- Closed-loop teacher repetitions retain standing despite up to **0.00717 m**
  root-coordinate and **0.39124 rad (22.42 degrees)** hinge-coordinate differences.
  Standing admission does not require identical trajectories.

The controlled replay localizes a source of variation to GPU physics/forward
solves rather than exploration RNG, mismatched reset poses or different commands.
Different contact ordering and float32 parallel reductions are consistent with
the measurements; this phase does not isolate a particular GPU kernel. No solver
or plant setting is changed to suppress it. The following short PPO comparison
uses native deterministic assessment as its primary outcome and reports GPU
assessment separately.

### Fresh PPO comparison results

Completed. [Paired comparison](../../logs/mimickit-training/stand-ppo-noise-0p0025-20261001-01/comparison.json),
[sigma 0.1 report](../../logs/mimickit-training/stand-ppo-noise-0p1-20261001-01/report.json)
and [sigma 0.0025 report](../../logs/mimickit-training/stand-ppo-noise-0p0025-20261001-01/report.json).

**Reducing action_std alone makes this fresh three-minute PPO result worse.**
Both arms start at mean native survival **0.33544 seconds**. Both final actors
have **0/8 standing passes and 0/8 survivors on native and GPU**. Final native
and GPU survival durations match at control-frame resolution in each arm:

- **action_std 0.1:** mean survival **1.15843 seconds**, range **0.86674-1.43345 s**.
  Training completes **77 PPO updates / 315,392 transitions** in **181.20365 s**,
  approximately **1,740.54 transitions per training second**.
- **action_std 0.0025:** mean survival **0.34794 seconds**, range **0.31669-0.45004 s**.
  Training completes **74 PPO updates / 303,104 transitions** in **181.45515 s**,
  approximately **1,670.41 transitions per training second**.

The nominal 180-second budgets finish after the current complete PPO update,
producing the small reported overshoots. Throughput differences do not establish
an inherent performance effect of action_std from two sequential single-seed runs.
Native assessment after update 64 is **1.12301 s** at sigma 0.1 and **0.34586 s**
at sigma 0.0025, so the unfavorable low-noise result also appears at the same
262,144-transition checkpoint. Neither comparison establishes convergence speed
or extrapolates to successful standing after more time.

Shared initial tensor SHA-256 matches in both arms:
`6245a080af66ae42bdbe96f71c3c27172fe6555fae73460a1d709b903f696fc9`.
Only `_model._action_dist._logstd_net` is excluded; both arms construct all other
tensors fresh. No teacher or imitation actor initializes either learner. Final
checkpoint SHA-256 values:

- Sigma 0.1: `41179e3833d4db09dfe24db1fbde5486775fb3217d4e0c83cd4fa2cd182570fa`.
- Sigma 0.0025: `2194f870da0d8b98821c073fdc5143d3c6b2f0948bc6c58cff80a1030dc66825`.

Average sampled GPU utilization is **29.97% / 27.12%**, mean power **56.29 /
56.20 W**, and peak total GPU memory **3,347 / 3,350 MiB**, respectively.
The sampler includes intermediate/final assessments and excludes setup/initial
assessment. These utilization percentages measure device activity, not learning
quality or speed. Protected input hashes remain unchanged, portable actor/AMP
sidecar validation passes, and Arena selection remains unchanged.

### Exploration and PPO update size are coupled

PPO's mean `clip_frac` over optimizer minibatches is **4.83% at sigma 0.1** and
**66.90% at sigma 0.0025**. Last-sixteen-update means are **4.67% / 67.03%**.
Upstream defines this as the fraction of sampled probability ratios outside
**[0.8, 1.2]**; it is not a count of rejected updates or dropped transitions.
Three epochs and actor SGD learning rate **1e-4** remain fixed in both arms.

For two independent Gaussian action distributions with the same fixed sigma,
the same observation and a mean-action change `delta_mean`, their KL divergence
is `sum(delta_mean_i^2) / (2*sigma^2)`. Thus the identical mean change would be
**1,600 times larger in KL** when sigma is reduced 40 times. Actual learned mean
changes need not match, and gradients/normalization also evolve. The observed
ratio-clipping increase indicates a substantial change in PPO update behavior;
it does not by itself prove why the low-noise learner fails.

This probability-ratio clipping is separate from observation clipping. Final
native mean observation clip fractions are only **0.00751% / 0.00151%** for the
two arms. The high input-clipping problem in the supervised imitation pilots
therefore does not describe these fresh PPO evaluations.

The physical teacher starts near a working feedback solution; fresh PPO starts
with an untrained mean that falls in roughly one third of a second. A noise scale
which preserves the teacher's standing is not automatically adequate exploration
or an appropriate optimizer scale for learning from the fresh mean. Low-noise
exploration and changed PPO update behavior are both candidate explanations;
this experiment changes them together and does not separate their causal effects.

### Decision and next bounded experiment; not executed

Keep sigma 0.1 as the better measured baseline for this short fresh comparison.
Do not adopt sigma 0.0025 alone, add springs, select either failed actor in the
Arena or start another hour-long run from these findings.

The next focused comparison would hold **action_std=0.0025** and all Phase 10
settings fixed while testing fresh three-minute actor learning rates **1e-5**
and **2.5e-6**, against the recorded **1e-4** arm. The latter is a 40-fold smaller
parameter step scale motivated by the Gaussian relation, not a guaranteed KL
match or learning-speed improvement. Monitor probability-ratio clipping alongside
deterministic survival, Stand gates and throughput. This would measure whether
reducing actor step size relieves the observed clipping pressure; if it does but
standing still fails, that separates update size from the remaining exploration
and objective questions. No such follow-up training runs occur in Phase 10.

## Phase 11: actor step size at fixed low exploration

The user authorized execution with "Proceed" after the Phase 10 proposal.
Run two fresh **180-second PPO training budgets**, keeping **action_std=0.0025**
fixed and changing only actor SGD learning rate to **1e-5** and **2.5e-6**.
Compare against the recorded Phase 10 low-noise arm at **1e-4**. These are fresh
learners, not continuations from any recorded actor.

### Protocol

- Keep paired seed **301010**, 64 environments, 64 rollout steps, 4,096 transitions
  per PPO update, three actor epochs and evaluation every 64 updates.
- Keep critic/discriminator rates, reward weights, normalization, spring-free
  model, passive-wrist reference, two-step torque queue, sixty-second horizons
  and all 34 admitted native AMP clips fixed. No mirroring or GPU teacher data.
- Verify **full initial tensor hashes**, including fixed logstd, match the Phase
  10 low-noise arm. Read its configuration/initialization metadata only; never
  load its checkpoint. Expected full initial SHA-256:
  `0ac238a0223a15b1e4946760fe0265132fb87032536cb38dfc98596216421e24`.
- Extend the isolated comparison runner with an explicit actor-rate option,
  finite-positive validation, paired input/demo/config checks and full initial
  state matching when sigma is unchanged. Save its source in each output bundle.
- Report final deterministic native/GPU survival and Stand admission, PPO ratio
  clipping, updates and transitions within each synchronized training budget.
  Setup/evaluations remain outside the budget; finish the current full update.
  Assess update 64 at equal transition counts if a run reaches it.
- Neither production defaults nor Arena selection change. No general test suite,
  Godot build or additional physics diagnosis is included. GPU repeatability
  findings from Phase 10 remain unresolved by this optimizer comparison.

Commands:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.0025 --actor-learning-rate 1e-5 `
  --paired-with logs/mimickit-training/stand-ppo-noise-0p0025-20261001-01 `
  --out logs/mimickit-training/stand-ppo-low-noise-lr1em5-20261002-01
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.0025 --actor-learning-rate 2.5e-6 `
  --paired-with logs/mimickit-training/stand-ppo-noise-0p0025-20261001-01 `
  --out logs/mimickit-training/stand-ppo-low-noise-lr2p5em6-20261002-01
```

### Completed results

[Three-rate comparison](../../logs/mimickit-training/stand-ppo-low-noise-lr2p5em6-20261002-01/comparison.json),
[actor rate 1e-5](../../logs/mimickit-training/stand-ppo-low-noise-lr1em5-20261002-01/report.json)
and [actor rate 2.5e-6](../../logs/mimickit-training/stand-ppo-low-noise-lr2p5em6-20261002-01/report.json).
Both new runs complete, with no nonfinite-update failures. All three low-noise
learners have **0/8 standing passes and 0/8 survivors on native and GPU**.
Mean final survival on each backend:

- Recorded **1e-4** baseline: **0.34794 s**, range **0.31669-0.45004 s**;
  average PPO probability-ratio clipping **66.90%**.
- New **1e-5**: **0.32711 s**, range **0.31669-0.36670 s**;
  average probability-ratio clipping **14.26%**.
- New **2.5e-6**: **0.32294 s**, range **0.31669-0.35003 s**;
  average probability-ratio clipping **4.01%**.

Each learner begins at **0.33544 s** mean native survival. Lowering the learning
rate strongly reduces ratio clipping, but produces no measurable standing
improvement within these budgets. The small survival differences should not be
treated as replicated learning advantages. Native and GPU final survival
durations match at control-frame resolution for both new actors.

At **update 64 / 262,144 transitions**, native mean survival is **0.34586 /
0.32503 / 0.32294 s**, respectively. The same lack of improvement therefore also
appears at an equal-transition checkpoint. Mean ratio-clipping fractions over
the last sixteen updates are **67.03% / 14.65% / 4.49%**.

Training work and synchronized elapsed time:

- Actor rate **1e-5**: **69 PPO updates**, **282,624 transitions**, **181.40149 s**;
  approximately **1,558.00 transitions per training second**.
- Actor rate **2.5e-6**: **71 PPO updates**, **290,816 transitions**, **181.15057 s**;
  approximately **1,605.38 transitions per training second**.

The nominal 180-second budgets finish their current update. Sequential-run
throughput differences do not establish a learning-rate effect on compute speed.
Average sampled GPU activity is **24.53% / 24.35%**, including intermediate/final
assessments but excluding setup/initial assessment. Device activity remains
separate from the measured task outcome.

### Verification and interpretation

All three **full fresh initialization hashes match**, including fixed logstd.
Configuration comparison confirms the sole differing setting is actor learning
rate. Protected source inputs and all 34 admitted AMP clips match, input hashes
remain unchanged, source snapshots match their recorded hashes, and portable
actors match their AMP sidecars. Final new checkpoint SHA-256 values:

- Actor rate 1e-5: `7b52c517fd8ea1d472294314309cd871e86c3cdf984a0146488165be3ac4d880`.
- Actor rate 2.5e-6: `41ec30a7cf8c29dbbead012458ddfc8e2f3510e50dbdd241b371600a4d38f4e7`.

At 2.5e-6, average ratio clipping is close to the Phase 10 **sigma 0.1 / actor
rate 1e-4** arm's **4.83%**, yet survival is **0.32294 s** versus **1.15843 s**.
Both still fail Stand. Thus matching this optimizer diagnostic alone does not
reproduce the higher-noise arm's early task progress. This does not isolate low
exploration as the sole cause: smaller actor steps also reduce absolute progress
per training budget, and critic/discriminator dynamics evolve with the rollouts.
Longer convergence and other seeds are not assessed.

Observation clipping is negligible: mean final native fractions are **0.00186%**
at 1e-5 and **0%** at 2.5e-6. Do not confuse these with the PPO probability-ratio
fractions above. Changing the inference clip is not supported by these results.

**Decision:** do not adopt the low-noise configurations or extend them into an
hour run. Retain **sigma 0.1 / actor rate 1e-4** as the better measured short-run
baseline, with its failed Stand admission stated explicitly. No production
configuration or Arena selection is changed.

### Proposed next bounded comparison; not executed

Screen intermediate exploration scales using fresh three-minute learners:
**sigma 0.025 / actor rate 2.5e-5** and **sigma 0.01 / actor rate 1e-5**.
Keep the remaining Phase 11 settings and seed fixed. These maintain the baseline's
actor-rate/sigma ratio as a starting heuristic; neither equal clipping nor equal
learning speed is guaranteed. This is a comparison of coupled configurations,
not an isolation of noise from learning-rate effects. Judge final deterministic
survival and Stand gates before resource utilization or clipping percentages.
Neither follow-up run is executed in Phase 11.

## Phase 12: intermediate exploration with scaled actor steps

The user authorized execution with "Proceed" after the Phase 11 proposal.
Run fresh three-minute learners at **action_std 0.025 / actor rate 2.5e-5** and
**action_std 0.01 / actor rate 1e-5**. Compare with the Phase 10 **0.1 / 1e-4**
baseline and Phase 11 **0.0025 / 2.5e-6** arm. All four configurations use the
same actor-rate/sigma ratio of **0.001**; this is a heuristic, not guaranteed
equivalent PPO updates or learning speed.

### Protocol

- Keep seed **301010**, 64 environments, 64 rollout steps, three actor epochs,
  evaluation every 64 PPO updates, source critic/discriminator optimizers and
  rewards, fresh normalizers, all 34 admitted native AMP clips, no mirroring,
  spring-free plant, passive-wrist reference and two-step action queue fixed.
- Construct each learner from scratch. Match initial actor, critic, discriminator
  and normalizer hashes against the Phase 10 baseline; exclude only the explicitly
  differing fixed logstd. No checkpoint or fitted normalizer initializes a run.
- Change only the two named settings in each paired configuration. Verify input,
  reference/demo and remaining configuration hashes. The runner receives only
  the two additional authorized sigma choices; PPO update logic is unchanged.
- Give each learner **180 synchronized training-update seconds**, excluding setup
  and deterministic assessments, completing the current update at budget expiry.
  Compare final budget checkpoints on native/GPU sixty-second eight-pose tests,
  intermediate native update-64 results, ratio clipping, transitions and compute
  measurements. Do not select a best checkpoint or extend the budget.
- This is a paired single-seed comparison of coupled settings. It does not
  isolate sigma from actor rate, assess longer convergence or resolve GPU physics
  repeatability. No production configuration or Arena selection changes occur.

Commands:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.025 --actor-learning-rate 2.5e-5 `
  --paired-with logs/mimickit-training/stand-ppo-noise-0p1-20261001-01 `
  --out logs/mimickit-training/stand-ppo-mid-noise-0p025-20261002-02
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.01 --actor-learning-rate 1e-5 `
  --paired-with logs/mimickit-training/stand-ppo-noise-0p1-20261001-01 `
  --out logs/mimickit-training/stand-ppo-mid-noise-0p01-20261002-01
```

The first sigma-0.025 launch (`stand-ppo-mid-noise-0p025-20261002-01`)
is interrupted when another user turn arrives: **nine PPO updates / 36,864
transitions / 23.49731 training seconds**, with no final assessment. Preserve
its artifacts and exclude it from completed-budget comparisons. Restart from
scratch as `stand-ppo-mid-noise-0p025-20261002-02`, using an independent hidden
process with redirected logs so chat updates do not terminate the experiment.
This is not checkpoint continuation or a selected successful restart.

### Completed results

[Four-configuration comparison](../../logs/mimickit-training/stand-ppo-mid-noise-0p01-20261002-01/comparison.json),
[sigma 0.025 report](../../logs/mimickit-training/stand-ppo-mid-noise-0p025-20261002-02/report.json)
and [sigma 0.01 report](../../logs/mimickit-training/stand-ppo-mid-noise-0p01-20261002-01/report.json).
Both complete-budget runs finish without nonfinite updates. The interrupted
partial launch is preserved with `interruption.json` and is not a third assessed arm.

Both new actors have **0/8 standing passes and 0/8 survivors on native and GPU**.
Native and GPU final survival durations match at control-frame resolution:

- **Sigma 0.025 / actor rate 2.5e-5:** mean **0.47295 s**, range **0.36670-0.78340 s**.
  Mean PPO probability-ratio clipping **4.89%**, last-sixteen-update mean **6.09%**.
  **66 PPO updates / 270,336 transitions / 182.43315 training seconds**;
  approximately **1,481.84 transitions per training second**.
- **Sigma 0.01 / actor rate 1e-5:** mean **0.35420 s**, range **0.33336-0.46670 s**.
  Mean probability-ratio clipping **4.66%**, last-sixteen-update mean **5.25%**.
  **66 PPO updates / 270,336 transitions / 180.84671 training seconds**;
  approximately **1,494.84 transitions per training second**.

All learners start at **0.33544 s** mean native survival. At **update 64 /
262,144 transitions**, the two new native means are **0.47087 / 0.35628 s**.
Thus the same ranking appears at an equal-transition checkpoint, not only at
budget completion. The small budget overshoots finish the active PPO update.

For context, the paired Phase 10 **sigma 0.1 / actor rate 1e-4** configuration
reaches **1.15843 s** final survival and **1.12301 s** at update 64, with **4.83%**
mean ratio clipping. Phase 11's **sigma 0.0025 / actor rate 2.5e-6** reaches
**0.32294 s** final/update-64 survival and **4.01%** clipping. All four fail Stand.

Average sampled GPU activity is **23.71% / 23.57%**, mean power **55.22 / 54.05 W**,
respectively. Sampling includes intermediate/final assessments and excludes
setup/initial assessment. Utilization and throughput differences between
sequential runs are not measured learning-speed multipliers.

### Verification and interpretation

Common initial actor, critic, discriminator and normalizer tensor SHA-256 matches
in all four compared arms:
`6245a080af66ae42bdbe96f71c3c27172fe6555fae73460a1d709b903f696fc9`.
Fixed logstd intentionally differs with sigma. Remaining configuration fields,
protected source inputs and all admitted demonstration records match. Source
snapshots match their recorded hashes; both new snapshots match the current
runner. Protected input hashes remain unchanged and portable actor/AMP sidecar
validation succeeds. Final checkpoint SHA-256 values:

- Sigma 0.025: `2e0353e7ffca2ef3af6a11c84279edb2b102b4d60f555712033a66fcae96645d`.
- Sigma 0.01: `282fead2dbd620ce32b2d63831c7e84f250b06dd448819a07a7b27e90d917050`.

The constant actor-rate/sigma heuristic produces similar average probability-ratio
clipping, approximately **4-5%**, across a 40-fold range of sigma. It does not
produce comparable early survival. Larger sigma and larger actor rate accompany
better early survival in this one-seed screen, but they change together; neither
the exploration effect nor the absolute actor step-size effect is isolated.
The interrupted restart adds no evidence of reproducible learning. Native
assessment remains the primary outcome, and no GPU repeatability fix is claimed.

Final observation clipping is **0% on both backends for both new actors**.
Input clipping does not explain these failures. No arm approaches the sixty-second
Stand gate, and these budgets do not assess eventual convergence or natural walking.

**Decision:** retain **sigma 0.1 / actor rate 1e-4** as the best measured short-run
baseline in this comparison. Do not promote either intermediate configuration,
select a failed actor in the Arena or claim successful/faster Stand learning.
Production defaults and Arena selection remain unchanged.

### Proposed next isolation; not executed

One fresh three-minute run at **sigma 0.025 / actor rate 1e-4** would hold sigma
fixed against this phase's **0.025 / 2.5e-5** arm and actor rate fixed against the
Phase 10 **0.1 / 1e-4** arm. Keep the same seed, physical reference, admitted AMP
clips and all other settings. This completes a useful comparison corner to
separate the observed coupled trend: measure whether sigma 0.025 still makes
early progress when the absolute actor step scale is restored, while reporting
any resulting ratio clipping. It is still a short single-seed screen, not a
convergence test. No additional run is executed in Phase 12.

## Phase 13: restore actor step size at intermediate exploration

The user authorized execution with "Proceed" after the Phase 12 proposal.
Run one fresh **180-second PPO training budget** at **action_std=0.025** and
**actor SGD learning rate 1e-4**. No runner implementation change is required.
Use an independent hidden process with redirected logs, preserving the run
across chat updates.

### Protocol

- Compare against Phase 12 **0.025 / 2.5e-5** with sigma held fixed, and Phase 10
  **0.1 / 1e-4** with actor rate held fixed. These pairwise comparisons each
  change one configured parameter; they remain sequential single-seed screens.
- Construct all learner tensors fresh with seed **301010**. Full initial tensor
  SHA-256 must match the sigma-0.025 arm:
  `6d19c526fb0b65b30bdd7216281acace88f9ab9f9c9f8a510b3e5a82873ed5d7`.
  Read the paired run's metadata only; never load its checkpoint or fitted moments.
- Preserve 64 environments, 64 rollout steps, 4,096 transitions per update,
  three actor epochs, evaluation every 64 PPO updates, source critic/discriminator
  settings and rewards, fresh normalizers, 34 admitted native AMP clips, no
  mirroring, spring-free plant, passive-wrist reference and two-step torque queue.
- Keep the synchronized update-time budget, final-checkpoint selection, native/GPU
  sixty-second eight-pose assessment and update-64 native comparison unchanged.
  Report ratio clipping separately from observation clipping and task admission.
- Verify fresh initialization, remaining configuration, input/demo and saved
  source hashes. No production defaults, Arena selection, Godot build, unrelated
  tests or extra training runs are included.

Command:

```powershell
mujoco_rig/mimic/.venv/Scripts/python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.025 --actor-learning-rate 1e-4 `
  --paired-with logs/mimickit-training/stand-ppo-mid-noise-0p025-20261002-02 `
  --out logs/mimickit-training/stand-ppo-mid-noise-0p025-lr1em4-20261002-01
```

### Completed results

[Pairwise comparison](../../logs/mimickit-training/stand-ppo-mid-noise-0p025-lr1em4-20261002-01/comparison.json)
and [new run report](../../logs/mimickit-training/stand-ppo-mid-noise-0p025-lr1em4-20261002-01/report.json).
The new learner completes without nonfinite updates and has **0/8 standing
passes and 0/8 survivors on both native and GPU**. Native and GPU final survival
durations match at control-frame resolution:

- New **sigma 0.025 / actor rate 1e-4:** mean **0.46462 s**, range
  **0.35003-0.73339 s**. Mean PPO probability-ratio clipping **14.42%**,
  last-sixteen-update mean **15.80%**.
- Fixed-sigma comparison, **0.025 / 2.5e-5:** mean **0.47295 s**;
  mean ratio clipping **4.89%**. Restoring the four-times-larger actor rate
  increases ratio clipping but does not improve this short-budget survival.
- Fixed-rate comparison, **0.1 / 1e-4:** mean **1.15843 s**;
  mean ratio clipping **4.83%**. At the same actor rate, sigma 0.025 still
  produces less early task progress than sigma 0.1 in this paired seed.

At **update 64 / 262,144 transitions**, native mean survival is **0.46254 s**
for the new learner, versus **0.47087 s** for the fixed-sigma comparison and
**1.12301 s** for the fixed-rate baseline. These comparisons do not depend only
on differing counts of completed updates within the wall-time budget.

The new run completes **65 PPO updates / 266,240 transitions** in **180.31504
training seconds**, approximately **1,476.53 transitions per training second**.
The current update finishes after the nominal 180-second limit. Average sampled
GPU activity is **24.62%**, mean power **54.97 W**; the sampler includes
intermediate/final assessments and excludes setup/initial assessment.

### Verification and interpretation

The new full initialization hash matches the fixed-sigma arm exactly. Initial
actor, critic, discriminator and normalizer tensors also match the sigma-0.1
baseline after excluding only its intentionally different fixed logstd.
Pairwise configuration comparison confirms only actor rate changes at fixed
sigma, and only sigma changes at fixed actor rate. Actor-rate YAML strings and
numbers are compared by their effective numeric values. Protected source inputs
and demonstration records match, saved source hashes verify, input hashes remain
unchanged and portable actor/AMP sidecar validation succeeds.

Final checkpoint SHA-256:
`ba470072ee44c80ccd727e2205479f405c152fd71a235099c61d4e9e56f72daa`.
Final observation clipping is **0% on both backends**. It is separate from the
reported PPO probability-ratio clipping.

This additional corner shows that the poorer sigma-0.025 result is not recovered
by restoring actor rate 1e-4 within this budget. It does not prove a universally
optimal sigma, statistical equivalence between the two sigma-0.025 arms or
eventual convergence. No policy passes Stand, and the earlier GPU repeatability
issues remain unresolved. The ratio between two episode-survival measurements
is not a ratio of learning speeds.

**Decision:** retain **sigma 0.1 / actor rate 1e-4** as the better measured
three-minute baseline. Stop selecting configurations by low PPO clipping alone.
No production defaults or Arena selection change; no longer run is started.

### Proposed next foundation audit; not executed

Pause the noise/rate sweep and measure whether the learning objective actually
ranks physically successful standing above the observed failed behavior.
Use fixed controllers on the same eight native reset poses: the admitted physical
teacher, zero commands, the fresh initial actor and the Phase 10 sigma-0.1 final
actor. Give each a twelve-second assessment through the same delayed torque queue.
Freeze the Phase 10 critic, discriminator and all normalizers for scoring; perform
zero optimizer updates and no checkpoint selection.

Record per-step task reward, AMP reward and their configured weighted combination,
both mean live-step reward and discounted accumulated reward, survival and final
Stand gates. Distinguish complete episodes, failures and time truncation rather
than comparing unequal-length averages as equivalent returns. Report frozen
critic estimates separately, including the teacher's unfamiliar states. Inspect
FAIL/TIME handling and whether pre-reset terminal observations feed the existing
PPO return calculation. This audit tests objective ordering and terminal credit
before investing more compute; it does not change rewards or claim a training fix.
No audit trials are executed in Phase 13.

## Phase 14 - frozen objective ordering and terminal credit (2026-10-02)

The user approved the Phase 13 proposal with **Proceed**. This isolated audit
uses `audit_stand_rewards.py` and the final Phase 10 sigma-0.1 snapshot. Four
controllers start from identical eight-case native resets: the admitted physical
teacher, zero commands, the saved fresh initial actor and the trained baseline
actor. Actor means retain their own checkpoint normalizers; every trajectory is
scored with the same frozen final critic, discriminator and scoring normalizers.
There are no optimizer updates, normalizer samples or checkpoint selection.

Each assessment has a twelve-second horizon, existing two-command delay and
spring-free torque plant. The runner records task/AMP/weighted reward, live-step
means, discounted observed returns, independent Stand gates and terminal flags.
It additionally compares each controller over the same per-pose early prefix.
TIME is a truncation of continuing Stand, not successful task completion. Frozen
critic bootstraps and TD-lambda residuals are separate diagnostics; unfamiliar
teacher trajectories do not establish critic accuracy on its training distribution.

The audit preserves and hashes the exact local/upstream terminal-credit sources,
all input clips and checkpoints, all frozen network/normalizer tensors, and checks
identical reset coordinates and velocities. Production configuration, reward
implementation and Arena selection remain unchanged.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_rewards `
  --out logs/mimickit-training/stand-reward-audit-20261002-01
```

### Completed results

[Audit report](../../logs/mimickit-training/stand-reward-audit-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-reward-audit-20261002-01/protocol.json),
[terminal implementation audit](../../logs/mimickit-training/stand-reward-audit-20261002-01/terminal-source-audit.json)
and [buffer storage inspection](../../logs/mimickit-training/stand-reward-audit-20261002-01/terminal-buffer-storage.json).
Every controller directory contains its complete per-frame `trace.npz` and report.

The teacher achieves **8/8 Stand passes and survivors** through **12.00096 s**.
All other controllers have **0/8 Stand passes and survivors**. Mean survival is
**0.33961 s** for zero commands, **0.33544 s** for the fresh actor and **1.15843 s**
for the trained actor. The two actor survival results reproduce Phase 10 at
control-frame resolution. All trials have finite scores and no solver warning.

Per-case means, averaged equally over the eight poses:

- **Teacher:** task reward **0.99937**, AMP reward **2.93884**, combined reward
  **1.38726** per live step; discounted observed combined return **130.35142**.
- **Zero commands:** task **0.66055**, AMP **0.93833**, combined **0.71611**;
  discounted observed return **13.60760**.
- **Fresh actor:** task **0.66242**, AMP **0.94771**, combined **0.71948**;
  discounted observed return **13.51896**.
- **Trained actor:** task **0.55390**, AMP **0.48062**, combined **0.53924**;
  discounted observed return **29.16341**.

The successful teacher has a larger observed discounted return than every failed
controller **on every pose**. Teacher returns range **120.41860-133.72208**;
trained-actor returns range **21.35104-32.74649**. Teacher traces terminate with
TIME, so these are measured twelve-second prefixes, not complete infinite-horizon
returns. Failed traces have zero continuation value. Applying the frozen critic
timeout bootstrap raises teacher mean return only to **130.36070** at discount
0.99; it leaves failed-controller returns unchanged.

The common early comparison covers **19-23 frames / 0.31669-0.38336 s** per pose.
Combined reward means over those identical per-pose prefixes are **1.19675** for
teacher, **0.72399** for zero, **0.71948** for fresh actor and **0.83395** for
trained actor. The trained actor improves early behavior and total return despite
having a lower full-episode mean reward: it spends additional live time in degraded
states that the fresh actor never reaches. A live-step mean alone reverses this
comparison and must not be treated as a learning-success metric.

### Terminal credit and frozen critic

Source inspection confirms:

- `BaseAgent._rollout_train`, lines 255-265, records post-step data **before**
  `_reset_done_envs`. `_record_data_post_step`, lines 316-320, saves next_obs,
  reward and done. `ExperienceBuffer.record`, lines 58-66, copies values into
  the buffer tensor, rather than retaining a reference that a reset could replace.
- `PPOAgent._build_train_data`, lines 117-131, evaluates the critic on those
  next observations. FAIL receives `get_reward_fail()/(1-discount) = 0`;
  TIME keeps the value of the pre-reset terminal observation. Every measured
  failure bootstrap is zero; every measured teacher timeout keeps its raw value.
- `compute_td_lambda_return`, lines 9-23, sets lambda to zero on **all non-NULL
  done flags**, stopping recurrence across episode resets while preserving TIME
  bootstrapping. No terminal/reset ordering defect is identified in this path.
- `StandTask.step`, lines 140-143, zeros failed task reward. AMP subsequently
  adds its discriminator reward without a failure mask. The trained actor's
  mean terminal combined reward is **0.00824**, range **0.00045-0.01891**.
  This is one terminal contribution per failed episode; the measured objective
  still strongly prefers the standing teacher. Its presence alone is not evidence
  that terminal reward masking will fix learning.

The frozen critic gives the identical reset states mean value **14.18256**,
regardless of which controller will act next. It estimates the return of its
training policy, not the best physically possible controller. Its initial
TD-lambda target-minus-value residual means are **36.09198** for teacher,
**0.34309** for zero, **0.22148** for fresh actor and **12.80241** for deterministic
trained actor. These are off-policy diagnostics using episode-length recorded
prefixes, not the original 64-frame PPO batches or their normalized advantages.
The difference between deterministic policy return and critic value does not yet
establish poor critic fitting: training samples the sigma-0.1 Gaussian policy.

### Verification, decision and next bounded audit

All four trials start from exactly identical qpos/qvel arrays. Final/initial
portable checkpoints match their AMP sidecars. Frozen actor, critic, discriminator
and serialized normalizer tensor hashes are unchanged; pending scoring-normalizer
samples remain zero. Protected plant, reference, demonstrations and checkpoints
remain unchanged, and captured source hashes verify. Zero learning updates occur.
No general test suite, compilation, reward change or Arena update is performed.

**Decision:** the final frozen objective ranks admitted standing above the measured
failures. Continue using survival, fixed Stand gates and accumulated return; do not
select a configuration from mean live reward or PPO clipping alone. This audit
does not show that fresh PPO can reach the high-return teacher trajectories, or
that reward ordering stayed identical throughout discriminator training. The
teacher already demonstrates feasibility without re-enabling passive springs.

**Proposed next audit; not executed:** compare the actual frozen sigma-0.1 training
policy against its deterministic assessment mean. Assess both fresh and trained
Phase 10 policies over the same eight native poses with three paired random seeds
(301010, 301011, 301012), twelve seconds maximum per case. Use the upstream actor
distribution/sample/action-normalization path and existing delayed torque queue;
freeze final scoring weights/normalizers and perform zero optimizer updates.
Record survival, Stand gates, task/AMP/combined return and frozen critic residuals,
along with sampled/executed commands. Compare with this deterministic audit to
separate exploration-induced failures from a critic mismatch on sampled behavior.
No additional training, reward edit, production configuration or Arena change is
included in that proposal. Longer training remains paused pending this distinction.

## Phase 15 - frozen Gaussian sampling versus deterministic assessment (2026-10-02)

The user approved the Phase 14 proposal with **Proceed**. The isolated runner
`audit_stand_sampling.py` calls upstream `PPOAgent._decide_action` in TRAIN mode
for saved fresh and final Phase 10 sigma-0.1 policies. Every trial uses the same
eight native reference resets and twelve-second maximum horizon. Seeds 301010,
301011 and 301012 are paired across policies; all eight worlds consume draws
every frame, including worlds whose physics has already stopped after failure.
The task resets in TEST mode to preserve deterministic pose assignment; the actor
remains in TRAIN sampling mode. No optimizer, replay or normalizer update runs.

The final Phase 10 critic/discriminator/scoring normalizers remain frozen for
every trajectory. Each actor retains its own checkpoint observation/action
normalizers. The runner reuses the Phase 14 reward/terminal assessment, records
raw samples, deterministic means, Gaussian draws, log probabilities, bounded
commands and actual held motor torques. It checks exact paired draw prefixes,
identical sampled/deterministic resets and the two-slot delayed torque mapping.
Captured source and input hashes are retained. Production defaults, reward code
and Arena selection remain unchanged.

This is a native fixed-policy diagnostic, not historical GPU rollout replay.
The final critic predicts its sampled policy; comparing it with fresh-policy
returns remains off-policy. All rewards use the final frozen AMP snapshot, which
differs from discriminator snapshots encountered during training. Three seeds
provide a bounded measurement, not a broad convergence result.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_sampling `
  --out logs/mimickit-training/stand-sampling-audit-20261002-01
```

### Completed results

[Sampling audit report](../../logs/mimickit-training/stand-sampling-audit-20261002-01/report.json)
and [protocol](../../logs/mimickit-training/stand-sampling-audit-20261002-01/protocol.json).
Six trial directories contain complete frame traces and per-case reports: two
actors, three seeds, eight cases each. All **48 episodes terminate with FAIL**;
there are **0/24 Stand passes and survivors per actor**, no TIME truncations and
no solver warnings.

- **Fresh sampled actor:** mean survival **0.34308 s**, range **0.28336-0.50004 s**;
  mean discounted observed combined return **10.69081**. Deterministic assessment
  is **0.33544 s / 13.51896 return**. Small survival differences coexist with a
  lower sampled reward; neither mode achieves Stand.
- **Trained sampled actor:** mean survival **1.04106 s**, range **0.68339-1.36678 s**;
  mean discounted observed combined return **21.14018**. Deterministic assessment
  is **1.15843 s / 29.16341 return**. Sampling reduces mean survival by approximately
  **10.13%** and mean observed return by **27.51%** in this measurement.
- Trained sampled per-seed means are **1.06259 s / 20.84881 return**,
  **1.03550 s / 21.34809**, and **1.02508 s / 21.22364**. All three seeds have the
  same qualitative outcome; these are repeated assessments of one trained seed,
  not three independently trained policies.

The trained sampled policy retains an approximately **threefold survival**
improvement over the fresh sampled policy, while still falling in roughly one
second. That ratio measures episode duration, not learning speed. Removing
sampling does not recover Stand; the deterministic controller already fails.
This result does not support treating action noise as the entire failure cause.

### Critic discrepancy and verification

The final frozen critic predicts the same reset-state mean **14.18256** for every
trial. Trained sampled observed return is **21.14018**, a mean excess of **6.95762**
over the prediction, approximately **32.91% of measured return**. Initial
TD-lambda residual mean is **6.82424**, versus **12.80241** under deterministic
trained assessment. Per-seed sampled residual means are **6.48621**, **7.09900**
and **6.88750**. The Gaussian policy therefore explains part of the deterministic
critic discrepancy, but a positive discrepancy remains for sampled behavior.

Fresh sampled initial residual mean is **-2.49124** and return-minus-value mean
is **-3.49175**. The final critic is not expected to predict the fresh policy.
For trained sampled behavior, residuals are a native fixed-policy calibration
diagnostic. They do not establish that critic fitting is the sole PPO bottleneck:
actor/discriminator changes between updates, the final AMP scoring snapshot and
native-versus-training-backend differences can contribute. Episode-length audit
TD targets also differ from original 64-frame PPO targets. No optimizer update or
actual PPO advantage calculation is performed in this audit.

The upstream sampler produces random actions on **100% of live steps**, fixed
normalized sigma **0.1**, and observed applied-noise RMS **0.09860-0.10036** across
trials. **No raw sampled command requires actuator clipping**. The Gaussian draw
prefixes match exactly across fresh/trained actors at each seed; recorded reset
qpos/qvel match the Phase 14 deterministic assessment. Every held motor torque
matches the bounded command from two intervals earlier exactly, including the
two initial zero-torque intervals.

All frozen tensor and protected input hashes remain unchanged. Source snapshots
verify and pending normalizer samples remain zero. The approved six trials are
the only new physics assessments; no general tests, compilation, production
configuration, reward implementation or Arena selection change occurs.

### Decision and proposed next bounded comparison; not executed

Keep the sigma-0.1 baseline. The physical teacher can stand, the frozen objective
prefers standing, the terminal path stores appropriate observations, and the
baseline improves both sampled and deterministic short-episode behavior. These
findings narrow the remaining problem to reaching and learning stable feedback,
with a measurable value-estimation discrepancy; they do not establish a single
cause or an eventual convergence budget. Do not restore passive springs or start
another hour run from this evidence.

**Proposed next comparison:** one fresh **180-second PPO/AMP pilot** changes only
critic SGD learning rate from **1e-4 to 1e-3**, paired with the saved Phase 10
baseline. Keep critic epochs **2**, actor rate **1e-4**, actor epochs **3**, fixed
sigma **0.1**, 64 environments, 64 rollout steps, evaluation interval 64, seed
301010, all demonstrations, reward weights, normalizers and physical contracts.
Verify matching initial actor/critic/discriminator/normalizer tensors, and assess
at update 64 as well as the final completed update to control differing throughput.

Record ordinary native/GPU deterministic Stand assessments, then the same three
seed native sampled-policy audit on the new final snapshot. Compare survival,
fixed Stand gates and discounted return alongside critic discrepancy and finite
optimization. A smaller critic residual alone is not a winning result. This is
an intervention to check whether stronger critic fitting helps policy learning,
not an assumption that ten times the critic rate means ten times faster learning.
No critic optimizer type, actor/discriminator setting, reward or spring changes
are included. The runner would gain an explicit critic-rate override; production
defaults and Arena selection remain unchanged. No pilot has been started.

## Phase 16 - isolated critic learning-rate pilot (2026-10-02)

The user approved the Phase 15 proposal with **Proceed**. One fresh 180-second
PPO/AMP run changes only effective critic SGD learning rate **1e-4 -> 1e-3**.
The saved Phase 10 sigma-0.1 run is the paired baseline. All initialization
tensors, reference, demonstrations and remaining effective settings must match.
The comparison runner gains an optional critic-rate override, without changing
its default or any production configuration.

Settings remain actor SGD rate 1e-4, actor epochs 3, critic epochs 2, fixed sigma
0.1, 64 environments, 64-frame rollouts, evaluation interval 64 and training
seed 301010. The three-minute budget counts synchronized training-update time;
the current update completes after the nominal limit. Native assessment occurs
at update 64, and final ordinary native/GPU assessments use the existing
60-second horizon. The final checkpoint is used without best-checkpoint selection.

After training, an isolated native audit scores the deterministic final actor and
the same three seeded Gaussian trials (301010, 301011, 301012), eight poses per
trial, with twelve-second horizons. Scoring uses the new final frozen critic,
discriminator and normalizers. Each policy's own final AMP snapshot is used for
its critic calibration; cross-run raw AMP return is therefore not an identical
reward function. Report task return separately, and judge Stand gates and survival
alongside calibration. Verify paired draw prefixes, identical resets, two-slot
torque delay, frozen tensors, portable/AMP sidecars and protected input hashes.
No longer run, reward edit, spring change or Arena selection is included.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.1 --actor-learning-rate 1e-4 --critic-learning-rate 1e-3 `
  --paired-with logs/mimickit-training/stand-ppo-noise-0p1-20261001-01 `
  --out logs/mimickit-training/stand-ppo-critic-lr1em3-20261002-01
```

### Completed training and frozen audit

[Pilot report](../../logs/mimickit-training/stand-ppo-critic-lr1em3-20261002-01/report.json),
[paired comparison](../../logs/mimickit-training/stand-critic-audit-20261002-01/comparison.json),
[frozen audit report](../../logs/mimickit-training/stand-critic-audit-20261002-01/report.json)
and [checkpoint verification](../../logs/mimickit-training/stand-critic-audit-20261002-01/checkpoint-verification.json).
The runner `audit_stand_critic.py` scores the candidate final deterministic actor
and its three Gaussian trials. The existing Phase 15 sampling routine was
extracted into a shared `sampled_trial` helper with the same trace/sampling path.

The pilot completes **74 PPO updates / 303,104 transitions** in **182.77602
training seconds**, approximately **1,658.34 transitions per training second**.
The saved baseline completes 77 updates / 315,392 transitions in 181.20365 s.
All candidate updates and parameters remain finite. Average sampled GPU activity
is **25.84%**, power **54.86 W**, using the same assessment-inclusive sampler.
These throughput measurements are not evidence that critic learning rate changes
the physical compute cost of an otherwise identical optimizer step.

Ordinary sixty-second final assessments have **0/8 Stand passes and 0/8 survivors
on both native and GPU**. Mean survival is **1.24177 s**, range **1.00008-1.55012 s**;
native/GPU episode durations match at control-frame resolution. Baseline mean
survival is **1.15843 s**, range **0.86674-1.43345 s**. The approximately **7.19%**
final duration improvement is modest and comes from one paired training seed.

At **update 64 / 262,144 transitions**, native mean survival is **1.12717 s**
versus baseline **1.12301 s**. Both have zero Stand passes. This matched-update
comparison is nearly unchanged; the final difference does not establish a large
sample-efficiency improvement or a learning-speed multiplier.

### Sampled behavior and value calibration

All **24 sampled candidate episodes fail**, with zero Stand passes, survivors,
solver warnings or time truncations. Mean sampled survival is **1.15704 s**,
range **0.83340-1.53346 s**, versus baseline **1.04106 s**. Per-seed candidate
means are **1.19593 s**, **1.21468 s** and **1.06050 s**; each exceeds the paired
baseline assessment mean, but these are inference seeds for one trained policy.

Frozen critic calibration improves substantially:

- Baseline critic predicts **14.18256**, sampled observed combined return is
  **21.14018**, signed return-minus-value mean **+6.95762**, and mean absolute
  per-episode error **6.95762**.
- Candidate critic predicts **23.08168**, sampled observed combined return is
  **22.81965**, signed mean **-0.26203**, and mean absolute per-episode error
  **2.18420**. The near-zero signed mean does not mean every episode is accurately
  predicted; absolute error still measures the remaining variation.
- Initial TD-lambda residual mean falls from **+6.82424** to **-0.25495**.
  Candidate per-seed residual means are **-0.43036**, **+0.18074** and **-0.51524**.
  The candidate deterministic actor retains residual **+6.87752**, consistent
  with the critic evaluating sampled behavior rather than deterministic control.

Mean sampled discounted **task-only return**, scored identically across the two
physical tasks, rises from **22.29018** to **23.73118**. AMP component returns are
**16.54016** and **19.17351**, and weighted combined returns **21.14018** and
**22.81965**; these latter comparisons use different learned final discriminator
snapshots and must not be treated as a single fixed reward function. Candidate
deterministic observed combined return is **30.88650** under its own snapshot.

Mean training critic loss is **10.10547** versus baseline **20.59276**; final
sixteen-update means are **3.32985** versus **9.90434**. Mean PPO ratio clipping is
**6.04%** versus **4.83%**. Training losses use changing on-policy data and AMP
scores, so the frozen assessment supplies the more direct calibration evidence.

### Verification and decision

Only effective critic SGD rate differs in the paired YAML configurations.
Actual saved initial AMP tensors for both runs match the recorded full hash:
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
All initial/final portable checkpoints validate against their AMP sidecars.
Protected training input mappings and demonstration records match exactly;
input hashes and saved training-runner hashes verify.

Audit Gaussian draw prefixes match the baseline Phase 15 draws exactly at all
three seeds. Native qpos/qvel resets match, two-slot delayed torque mapping
verifies, no raw command requires clipping, and applied-noise RMS is
**0.09899-0.10032**. Candidate actor/critic/discriminator/normalizer tensor hashes
remain frozen through the audit; pending normalizer samples are zero. Audit
sources and protected input hashes verify. The frozen audit performs zero
learning updates; no general test suite, compilation or Arena update is run.

Candidate final portable checkpoint SHA-256:
`048ed1dc7fbe21d28a5813bed441f3865edd1057654d574d0feebf11bf768cfb`.

**Decision:** critic rate **1e-3** is a viable experimental candidate with better
measured value calibration and modest final survival gains. It has not solved
Stand and is not adopted as a production winner. The critic-rate intervention
reduces the measured value discrepancy without removing the standing failure;
it does not isolate every source of that discrepancy. Preserve both candidates
and keep production defaults/Arena unchanged.
The earlier teacher GPU repeatability issue is not resolved by this experiment.

### Proposed replication; not executed

Repeat the **1e-4 versus 1e-3 critic-rate comparison** with **two additional
training seeds, 301011 and 301012**: **four fresh 180-second pilots**, twelve
minutes of nominal training in total. For each seed, both arms must start from
identical fresh network/normalizer tensors and differ only in critic rate.
Keep all other Phase 16 settings and inputs; assess matched update 64 and the
final native/GPU Stand gates, then apply the same three-seed sampled calibration
audit to each final policy. Distinguish training seeds from inference RNG seeds.

Combine those results with training seed 301010 before promoting the critic rate
or committing to longer training. Require a consistent physical performance
benefit alongside finite updates and calibration; low value error alone is not
sufficient. No extra hyperparameter sweep, reward edit, springs, longer run or
Arena selection is included. No replication pilots have been started.

## Phase 17 - paired critic-rate replication across training seeds (2026-10-02)

The user approved the Phase 16 replication proposal with **Proceed**. Four fresh
180-second PPO/AMP pilots cover training seeds 301011 and 301012, each with critic
SGD rates 1e-4 and 1e-3. Run the lower-rate arm first for each seed and bind the
higher-rate arm to that seed's saved initialization/configuration. Training runs
execute sequentially on the same GPU. No warm-start or checkpoint selection occurs.

All other Phase 16 settings, physical inputs and demonstrations remain fixed.
Ordinary native assessment occurs at update 64 and final native/GPU assessment
uses sixty seconds. After each policy finishes, a native frozen audit assesses
its deterministic mean plus three Gaussian inference seeds 301010-301012 on the
same eight reset poses, twelve seconds maximum. Inference RNG seeds are distinct
from the training-seed comparison. Frozen own-snapshot critic/AMP calibration
and task-only returns are recorded alongside Stand gates and episode survival.

New runs use names `stand-ppo-critic-lr1em4-seed301011-20261002-01`,
`stand-ppo-critic-lr1em3-seed301011-20261002-01`, and corresponding names for
training seed 301012. Aggregate these with the already completed training seed
301010. Verify saved checkpoint tensors, within-seed initialization pairing,
only-critic-rate effective config differences, copied source hashes, physical
input hashes, paired inference draw prefixes and frozen scoring tensors.
No extra sweep, longer training, general test suite, compilation, reward edit,
springs or Arena change is included.

### Completed replication

[Aggregate report](../../logs/mimickit-training/stand-critic-replication-20261002-01/report.json),
[compact summary](../../logs/mimickit-training/stand-critic-replication-20261002-01/summary.json)
and [completed progress](../../logs/mimickit-training/stand-critic-replication-20261002-01/progress.json).
The four new run directories contain their training inputs, initialization/final
checkpoints, update logs, native/GPU assessments and training-source snapshots.
The aggregate directory contains one `audit-seed<seed>-lr<rate>` directory per
new policy, with deterministic and three sampled frame traces, reports and source
snapshots. `audit_stand_policy.py` reuses the Phase 15 sampling path;
`summarize_stand_critic_replication.py` verifies saved artifacts and aggregates
the six policies without executing additional physics.

All four new pilots complete with finite updates, totaling **725.40278 synchronized
training seconds / 1,183,744 transitions**. Completed update counts are **72 and
74** for training seed 301011 (rates 1e-4 and 1e-3), and **70 and 73** for seed
301012. Each has an update-64 assessment. Fresh final checkpoints are selected
by budget completion, with no best-policy or best-training-seed selection.

Mean native survival per training seed, **critic 1e-4 -> 1e-3**:

- **301010, already completed:** final **1.15843 -> 1.24177 s**;
  update 64 **1.12301 -> 1.12717 s**; sampled **1.04106 -> 1.15704 s**.
- **301011, new:** final **1.02300 -> 1.17718 s**;
  update 64 **0.97924 -> 1.18760 s**; sampled **1.02369 -> 1.16051 s**.
- **301012, new:** final **1.01883 -> 1.23968 s**;
  update 64 **1.03133 -> 1.28552 s**; sampled **0.98341 -> 1.14662 s**.

The higher rate improves final native survival, matched-update survival and
sampled survival in **all three paired training seeds**. Aggregate means give:

- Final native survival **1.06675 -> 1.21954 s**.
- Update-64 native survival **1.04453 -> 1.20010 s**, at exactly 262,144 transitions.
- Sampled survival **1.01605 -> 1.15472 s**.
- Sampled discounted task-only return **21.85528 -> 24.18303**, under the same
  physical/reference task reward.
- Mean absolute sampled return prediction error **6.94725 -> 2.39504**.

All policies still fail Stand: **0/24 final native passes, 0/24 final GPU passes
and 0/72 sampled passes for each critic rate**, combining three trained policies.
There are no survivors in these assessments. Every new frozen audit episode ends
with physical FAIL rather than TIME, and no solver warnings are recorded.
The inference seeds are repeated assessments of each policy; the independent
training replication count is three, not 72.

### Calibration, backend differences and verification

Sampled prediction-error means per training seed, **1e-4 -> 1e-3**, are
**6.95762 -> 2.18420**, **6.67052 -> 2.33490** and **7.21362 -> 2.66604**.
This supports a repeatable calibration improvement. Errors are measured against
each policy's own frozen final AMP snapshot; raw combined returns across policies
are not a single fixed scoring function. Task-only return is separately comparable.

For new lower-rate policies, native/GPU mean durations differ slightly:
seed 301011 **1.02300 / 1.01883 s**, seed 301012 **1.01883 / 1.01050 s**.
Higher-rate new policies have equal native/GPU mean durations at control-frame
resolution. The native frozen deterministic audit reproduces each new ordinary
native assessment; recorded command comparisons for seed 301011 match exactly.
Small backend differences do not resolve the earlier teacher repeatability issue.

Saved initial network/normalizer tensors match exactly within each rate pair,
and differ across training seeds. Full initial tensor hashes are:

- 301010: `a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
- 301011: `627738dde1a4250390662ac799bcad04f1023719e0c845f14130d79ba397f06e`.
- 301012: `14f7c630332765ea373853e8a15b41c49de20733eb0c0c47d9cf2eaff4500f98`.

Effective baseline configurations match across seeds; within each seed only
critic SGD learning rate differs. Protected original physical/training inputs
and demonstration records match. Saved source hashes, recorded initial tensor
hashes, final checkpoint file hashes and all initial/final portable/AMP sidecars
verify. Every audit verifies paired Gaussian draw prefixes, identical native
resets, the two-command delayed torque mapping and frozen scoring tensors, with
zero pending normalizer samples or optimizer updates. Protected artifacts remain
unchanged during aggregation. No general test suite, standalone compilation,
production configuration or Arena selection is changed.

New final portable checkpoint SHA-256 values:

- 301011 / 1e-4: `79eada1c9bf298a231bcb29c2066684ae8b806c8d4c8f082eefe29ce9502ca37`.
- 301011 / 1e-3: `890e50db91acf56a7b20b9465f50e78792d31f33fe8a3dbbb98b36cfc470e29b`.
- 301012 / 1e-4: `5e1be146596feac75a06ece21e915358a983d5634e116a095042cd5fa438def0`.
- 301012 / 1e-3: `3351e32cbce3e6f3be6404f361b7efb7c407b03caf1b915b605a8f98dc20e666`.

### Decision and proposed next bounded duration check; not executed

Use **critic SGD rate 1e-3 as the experimental research baseline** for subsequent
controlled runs. It improves measured physical performance and value calibration
across all three training seeds; the matched-update comparison supports progress
beyond a wall-time throughput artifact. This does not establish statistical
certainty, globally optimal settings or a solution to Stand. A tenfold critic
rate increase has not produced tenfold learning speed. Production defaults and
Arena remain unchanged, and no longer training starts in Phase 17.

**Proposed next run:** one fresh **600-second / ten-minute PPO/AMP duration check**,
training seed **301010**, using the Phase 16/17 higher-rate settings and identical
initial tensors. Keep actor rate 1e-4, actor epochs 3, critic rate 1e-3, critic
epochs 2, sigma 0.1, 64 environments, 64 rollout steps, evaluation interval 64,
all rewards/demonstrations/normalizers and the spring-free delayed-torque plant.
Assess native Stand at updates 64, 128, 192 and every later completed multiple
of 64, plus final native/GPU Stand and the three-seed frozen sampled audit.

Use the learning curve to distinguish continued survival gains from a plateau;
report fixed Stand gates, accumulated task reward and value calibration, without
promising a steps-to-standing threshold. The final budget checkpoint is used
without best-checkpoint selection. This changes duration only, starts from fresh
weights and includes no new hyperparameter sweep, reward/physics adjustment,
Arena loading or deployment. No ten-minute run has been started.

## Phase 18 - fresh ten-minute duration check (2026-10-02)

The user approved the Phase 17 proposal with **Proceed**. This run changes only
the synchronized training-update budget from 180 to **600 seconds**, using the
experimental critic-rate 1e-3 configuration and training seed **301010**. Actor
rate 1e-4, actor epochs 3, critic epochs 2, sigma 0.1, 64 environments, 64-frame
rollouts, evaluation interval 64 and all physical/reward/demo inputs remain fixed.
The learner starts from fresh tensors matching the Phase 16 seed-301010 pilot;
it does not resume that pilot or load teacher/BC weights.

Use the existing runner without code changes. Native Stand assessment occurs at
every completed multiple of 64 updates; final native/GPU assessment has the usual
sixty-second horizon. The current update finishes after the nominal time limit.
The final checkpoint is used without best-checkpoint or best-training-seed
selection. Afterwards, use the shared frozen-policy audit for the deterministic
mean and Gaussian inference seeds 301010-301012 on eight native reset poses,
twelve seconds maximum per audit case. Score with the new final frozen critic,
discriminator and normalizers; report task component and value errors separately.

Compare the native learning curve, matched update 64 and final/sample assessments
with the Phase 16 three-minute run. Verify identical initialization/configuration
and input hashes, paired inference draws, native resets, delayed torque mapping,
checkpoint sidecars and frozen audit tensors. This is one duration experiment,
not additional training-seed replication or a steps-to-standing guarantee.
No reward/physics edit, springs, production default, Arena loading, general test
suite or standalone compilation is included.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.1 --actor-learning-rate 1e-4 --critic-learning-rate 1e-3 `
  --seconds 600 --seed 301010 `
  --paired-with logs/mimickit-training/stand-ppo-critic-lr1em3-20261002-01 `
  --out logs/mimickit-training/stand-ppo-critic-lr1em3-10min-20261002-01
```

### Completed execution and artifacts

Training completed with **601.47099 synchronized update seconds**, **270 PPO
updates** and **1,105,920 transitions**. The timed loop including intermediate
and final assessments took **608.86543 seconds**, excluding earlier setup and
the initial assessment. Finishing the current update explains the budget overrun.

- Training: `logs/mimickit-training/stand-ppo-critic-lr1em3-10min-20261002-01`.
- Frozen audit: `logs/mimickit-training/stand-duration-audit-20261002-01`.
- Derived comparison: `logs/mimickit-training/stand-duration-audit-20261002-01/duration-comparison.json`.
- Final portable checkpoint SHA-256:
  `a9a00a28e62cc88c2a3d36c8d61d6343942cfcdc86ecf51c99b5fcdf980e26a6`.

### Native learning curve and final Stand assessments

Mean native survival, with zero Stand passes and zero survivors at every point:

- Fresh initial policy: **0.33544 s**.
- Update 64 / 262,144 transitions / 157.19 training seconds: **1.14384 s**.
- Update 128 / 524,288 transitions / 303.19 training seconds: **1.45428 s**.
- Update 192 / 786,432 transitions / 439.21 training seconds: **1.70639 s**.
- Update 256 / 1,048,576 transitions / 571.32 training seconds: **1.80014 s**.
- Final update 270 / 1,105,920 transitions: **1.92099 s**.

Final native and GPU ordinary assessments both give **0/8 Stand**, **0/8
survivors**, mean **1.92099 s**, minimum **1.65013 s**, maximum **2.65021 s**.
The separate frozen deterministic audit reproduces the native mean exactly.
No solver warnings occur. These are falls well before the sixty-second ordinary
assessment horizon, not near-passes at the final Stand gates.

The earlier three-minute higher-critic-rate run had 74 updates, 303,104 transitions,
182.77602 training seconds and **1.24177 s** final native survival. At matched
update 64 it gave **1.12717 s**, compared with **1.14384 s** here. Identical
initial tensors and RNG seeds do not establish bitwise GPU rollout reproduction;
the small early difference must remain visible in this duration comparison.

### Frozen sampled policy and value calibration

The new final policy's three Gaussian inference seeds on eight reset poses give
**0/24 Stand**, **0/24 survivors**, mean survival **1.92515 s**, minimum
**1.51679 s**, maximum **2.45020 s**. Each episode ends in physical FAIL.
Per-inference-seed means are **1.91474**, **1.93974** and **1.92099 s**.
This remains one independently trained policy, not three training replications.

Compared with the three-minute policy's sampled assessment:

- Mean survival: **1.15704 -> 1.92515 s**, an increase of **66.39%**.
- Mean discounted task return: **23.73118 -> 32.99113**, an increase of **39.02%**.
- Mean initial critic prediction: **23.08168 -> 30.17747**.
- Mean observed combined return: **22.81965 -> 31.47521**.
- Mean absolute observed return/prediction error: **2.18420 -> 1.99585**.
- Mean observed return minus prediction: **-0.26203 -> 1.29774**.
- Mean initial TD-lambda residual: **-0.25495 -> 0.92272**.

Each combined return and calibration error uses that policy's own final frozen
AMP discriminator and normalizers. Combined returns therefore do not represent
one fixed reward function across runs. The task-only component is separately
comparable. The deterministic new policy's task return is **39.97697**, combined
return **38.76723**, while its initial critic mean is **30.17747**; the critic is
trained for the sampled policy, so deterministic mismatch alone is not a critic
training failure.

Mean GPU utilization is **26.85%** and sampled board power **55.80 W**, compared
with **25.84% / 54.86 W** previously. These samples include intermediate and final
assessments. Throughput is **1,838.69 transitions/training second**, compared with
**1,658.34** over the shorter run, so 3.29 times the training seconds produced
3.65 times the transitions. Native survival improved **54.70%**, not proportionally
to compute. GPU utilization alone does not measure learning efficiency.

### Verification and decision

Effective configurations, full fresh initial tensors, saved training runner,
protected input hashes and demonstration records match the three-minute run.
The shared initial tensor SHA-256 is
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
All initial/final checkpoint sidecars, final model hash, source snapshots,
paired Gaussian draw prefixes, native resets, delayed torque mapping and frozen
audit tensors verify. There are zero pending normalizer samples and zero audit
optimizer updates. Protected training artifacts remain unchanged.

Keep the same experimental configuration. Both deterministic and sampled
survival improve beyond three minutes, with rising task-only return and a
reasonably calibrated sampled-policy critic. The recorded checkpoints support
continued progress over this interval; they do not establish sustained Stand,
a universal transitions threshold, or a linear learning-speed multiplier.

**Proposed next bounded run; not executed:** one fresh **1,800-second / thirty-minute
duration check**, training seed 301010, with the same initial tensors and all
Phase 18 settings. Retain native assessments every 64 updates, final sixty-second
native/GPU assessment, the three-seed twelve-second frozen sampled audit and
final-budget checkpoint selection. Compare the curve with this ten-minute run
to determine whether survival keeps increasing or plateaus before investing in
a longer run. This proposal requires the user's approval before execution.

No implementation code, production configuration, reward, plant, springs or
Arena selection changed. No general test suite or standalone compilation ran.

## Phase 19 - fresh thirty-minute duration check (2026-10-02)

The user approved the Phase 18 proposal with **Proceed**. Run one fresh policy
for **1,800 synchronized training-update seconds**, training seed **301010**,
paired with the ten-minute Phase 18 run. Keep actor SGD rate 1e-4 and three
epochs, critic SGD rate 1e-3 and two epochs, sigma 0.1, 64 environments, 64-frame
rollouts and native evaluation every 64 updates. All reward, demonstration,
normalizer and spring-free delayed-torque physical settings remain fixed.
Use the existing training and frozen-audit runners without implementation edits.

Verify the full initial tensors, effective configuration, source and input hashes
against the ten-minute run. Record the complete native learning curve, final
sixty-second native/GPU Stand assessments and final-budget checkpoint. Afterwards
run the shared frozen deterministic and sampled audit, paired with the Phase 18
audit's Gaussian draws and reset states: inference seeds 301010-301012, eight
native reset poses, twelve-second maximum per case. Report task-only return and
value calibration separately from each policy's own final AMP scoring snapshot.

Compare against both three-minute and ten-minute results, retaining matched
update comparisons and the limitation that CUDA rollout paths are not proven
bitwise repeatable. This is one duration experiment, not training-seed replication.
No production defaults, Arena selection, physical plant, rewards or springs
change. No general tests or standalone compilation are authorized.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_exploration `
  --action-std 0.1 --actor-learning-rate 1e-4 --critic-learning-rate 1e-3 `
  --seconds 1800 --seed 301010 `
  --paired-with logs/mimickit-training/stand-ppo-critic-lr1em3-10min-20261002-01 `
  --out logs/mimickit-training/stand-ppo-critic-lr1em3-30min-20261002-01
```

### Completed execution and artifacts

The run completed **1,801.05256 synchronized training-update seconds**, **859
PPO updates** and **3,518,464 transitions**. The timed loop including intermediate
and final assessments took **1,819.04480 seconds**, excluding earlier setup and
the initial assessment. The final update completed after the nominal limit.

- Training: `logs/mimickit-training/stand-ppo-critic-lr1em3-30min-20261002-01`.
- Frozen audit: `logs/mimickit-training/stand-duration30-audit-20261002-01`.
- Three/ten/thirty-minute comparison and full curves:
  `logs/mimickit-training/stand-duration30-audit-20261002-01/duration-comparison.json`.
- Final portable checkpoint SHA-256:
  `ba5ea24f75b6f06094e72a361c313858c93a9eb4149b9f72a02120921abd9129`.

### Native learning curve and final assessments

Every recorded assessment has **0/8 Stand passes**, **0/8 survivors** and zero
solver warnings. Mean survival at each native checkpoint:

- Initial: **0.33544 s**.
- Update 64 / 161.59 training seconds: **1.20635 s**.
- Update 128 / 308.86 training seconds: **1.54596 s**.
- Update 192 / 442.88 training seconds: **1.79389 s**.
- Update 256 / 575.46 training seconds: **1.85223 s**.
- Update 320 / 707.63 training seconds: **2.01891 s**.
- Update 384 / 838.18 training seconds: **1.74806 s**.
- Update 448 / 968.08 training seconds: **1.80431 s**.
- Update 512 / 1,100.53 training seconds: **1.91474 s**.
- Update 576 / 1,231.75 training seconds: **1.81473 s**.
- Update 640 / 1,359.87 training seconds: **1.91890 s**.
- Update 704 / 1,489.46 training seconds: **1.95224 s**.
- Update 768 / 1,619.08 training seconds: **1.89182 s**.
- Update 832 / 1,747.52 training seconds: **1.97932 s**.
- Final update 859 / 1,801.05 training seconds: **1.95016 s**.

Each update collects 4,096 transitions. After update 320, mean native survival
fluctuates between **1.74806 and 2.01891 seconds**. The final policy remains below
that earlier maximum; the approved final-budget policy is used regardless.

Final native survival is **1.95016 s**, minimum **1.78348 s**, maximum **2.20018 s**.
Final GPU survival is **1.95224 s**, with the same minimum and maximum. One pose
survives one additional control frame on GPU; all cases still fail. This small
backend difference does not resolve the earlier teacher repeatability issue.
The separate frozen deterministic native audit reproduces **1.95016 s** exactly.

At matched updates 64/128/192/256, the ten-minute run gave **1.14384 / 1.45428 /
1.70639 / 1.80014 s**, compared with **1.20635 / 1.54596 / 1.79389 / 1.85223 s**
here. Identical initial tensors and seeds do not guarantee bitwise CUDA rollout
paths. This is a controlled duration comparison with visible early variation,
not an exact continuation of the ten-minute trajectory.

### Frozen sampled policy, calibration and compute comparison

The three paired Gaussian inference seeds yield **0/24 Stand**, **0/24 survivors**,
mean survival **2.02933 s**, minimum **1.46678 s**, maximum **2.60021 s**. All
episodes terminate with physical FAIL and no solver warnings. Per-inference-seed
means are **2.06058**, **2.07100** and **1.95641 s**. These are repeated assessments
of one trained policy; the training replication count for this duration is one.

Compared with the ten-minute final policy:

- Native mean survival: **1.92099 -> 1.95016 s**, **+1.52%**.
- GPU mean survival: **1.92099 -> 1.95224 s**, **+1.63%**.
- Sampled mean survival: **1.92515 -> 2.02933 s**, **+5.41%**.
- Sampled discounted task return: **32.99113 -> 36.55267**, **+10.80%**.
- Mean initial critic prediction: **30.17747 -> 34.14930**.
- Sampled mean observed combined return: **31.47521 -> 34.69187**.
- Mean absolute observed return/prediction error: **1.99585 -> 2.03633**.
- Mean observed return minus prediction: **1.29774 -> 0.54257**.
- Mean initial TD-lambda residual: **0.92272 -> 0.50954**.

Calibration remains close to the ten-minute level. Each calibration measurement
uses its policy's own final frozen AMP discriminator and normalizers; raw combined
returns are not one fixed scoring function across policies. Task-only returns
are separately comparable. For the new deterministic policy, task return is
**43.65844**, combined return **43.23391**, and initial critic mean **34.14930**;
deterministic return is not the sampled-policy critic's training target.

Training time is **2.99441 times** the ten-minute budget and transitions are
**3.18148 times** as many. Mean sampled GPU utilization is **26.96%**, board power
**56.40 W**, versus **26.85% / 55.80 W** for ten minutes. Samples include intermediate
and final assessments. Mean throughput is **1,953.56 transitions/training second**,
versus **1,838.69** for ten minutes. Mean critic loss drops from **4.92768 to
3.07426**, and the final sixteen-update mean drops from **2.84991 to 1.68256**,
without a corresponding major physical survival gain. Mean PPO clipping fraction
is **5.02%**, versus **4.94%**. These optimization metrics do not establish Stand.

### Verification, decision and proposed next diagnostic

Full fresh initial tensors, effective configurations, saved runner source,
protected input hashes and admitted demonstration records match across the
three-, ten- and thirty-minute runs. The shared initial tensor SHA-256 remains
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
Initial/final sidecars, final checkpoint hashes, paired Gaussian draw prefixes,
native resets, delayed torque mapping, source snapshots and frozen audit tensors
verify. There are zero pending normalizer samples and zero audit optimizer updates.
Training and audit stderr files are empty; neither creates a failure report.

**Do not extend training unchanged based on these results.** There is a practical
plateau in assessed native survival after roughly twelve minutes, while sampled
survival and task return still improve modestly. This is not proof of convergence
across seeds or of a globally optimal policy. It does show diminishing returns
for this configuration over the approved window: roughly three times the compute
produces **+1.52% native survival** and still no Stand. The higher critic rate's
early improvement remains valid, but its benefit is not a solved Stand controller.

**Proposed next bounded diagnostic; not executed:** analyze existing frozen
deterministic and sampled traces for the three-, ten- and thirty-minute policies,
with the recorded teacher trajectory as a feasible reference. Locate the first
balance/posture violations; compare pelvis height/tilt, foot loads, joint motion
and command saturation through the approach to failure. Use the saved held-torque
arrays for the two-command delay. Report what is directly observable and which
failure/contact details are absent from the traces. Separate evidence from
hypotheses, then propose one targeted experiment. This is an offline trace audit,
with no new training, physical rollout, reward/plant/default change or Arena load.
It requires the user's approval before creating its report or documentation.

No implementation code, production configuration, reward, plant, springs or
Arena selection changed. No general test suite or standalone compilation ran.

## Phase 20 - offline recorded-failure audit (2026-10-02)

The user approved the Phase 19 diagnostic with **Proceed**. Analyze the recorded
teacher trajectory and the final three-, ten- and thirty-minute deterministic
and Gaussian sampled policy traces. Use live episode prefixes only; exclude
frozen padding after termination. Record first and sustained balance/posture
violations, pelvis height/tilt and root velocities, foot loads, joint motion and
limit proximity, command saturation and the recorded two-command delayed torques.
Compare common early windows and the approach to failure with the feasible teacher.

The offline analyzer reads NumPy arrays, saved metadata and XML joint definitions;
it does not construct a simulator or learner, load a policy, step physics, run
training, execute general tests or compile. Protect input/source hashes and check
reset correspondence, trace dimensions/timing, report consistency and saved torque
mapping. Identify missing measurements explicitly. Save an auditable report and
propose one causal experiment from the findings, without executing that experiment.

Execution and findings will be recorded below after this approved audit completes.

### Completed execution and artifacts

The offline analyzer reads **13 recorded traces / 104 episodes**: eight teacher
episodes, 24 deterministic learner episodes and 72 sampled learner episodes.
It constructs no model, simulator or agent and performs **zero physics rollouts
and zero training updates**. The three policy budgets use the same training seed;
the inference seeds do not add independent training replications.

- Analyzer: `mujoco_rig/mimic/audit_stand_failures.py`.
- Final report, input/source manifest and analyzer snapshot:
  `logs/mimickit-training/stand-failure-traces-20261002-02`.
- Compact derived findings: `logs/mimickit-training/stand-failure-traces-20261002-02/summary.json`.
- The exploratory `stand-failure-traces-20261002-01` report precedes the addition
  of outward delayed-torque measurements; `-02` is the final report.

The approved diagnostic command is:

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_failures `
  --out logs/mimickit-training/stand-failure-traces-20261002-02
```

### First loss of balance is unchanged by longer training

All **96 learner episodes**, deterministic and sampled, first violate the full
balance gate at **0.050004 seconds**. The directly identifiable violated component
is root angular speed above **1 rad/s**. The initial two zero-torque physics
intervals reproduce the teacher's recorded qpos/qvel **exactly in every trace**.
With the two-command delay, the first policy command is physically applied in
the third interval, ending at 0.050004 seconds. This locates the divergence at
initial torque application, rather than at a different reset state.

At that ending frame, mean deterministic root angular speeds are:

- Teacher: **0.57600 rad/s**.
- Three-minute policy: **2.91506 rad/s**.
- Ten-minute policy: **5.01548 rad/s**.
- Thirty-minute policy: **6.14540 rad/s**.

The teacher maintains the complete balance and posture gates throughout all
**12.00096 recorded seconds**, with no violating frame. Each deterministic
learner is balanced only for the first **0.033336 seconds**, and never recovers
the complete balance gate. Sampled policies have four episodes with any later
balanced frame, out of 72; the longest balanced stretch in the thirty-minute
sampled cases is **0.150012 seconds**. Their mean total balanced time is only
**0.0409755 seconds**, despite **2.02933 seconds** mean time before FAIL.

In sampled cases, first pelvis-tilt violations average **0.08334 / 0.06806 /
0.06667 seconds** for three/ten/thirty minutes. First aggregate upper-body
posture violations average **0.26322 / 0.21252 / 0.21807 seconds**. Longer training
delays terminal falling here; it does not extend continuous balanced standing.

### Joint-limit bracing, not command saturation

Every learner trace has **zero command clipping** and **zero normalized command
samples at or above 0.95 magnitude**. Maximum raw commands are **0.25727 / 0.33694 /
0.46414** for the three deterministic policies, and at most **0.68737** across
the sampled learners. There is no observed exhaustion of normalized motor authority.

The thirty-minute deterministic policy instead spends most live frames near
joint limits while the held delayed torque points farther toward that limit.
Using a five-degree proximity margin, the measured outward-drive fractions are:

- `Spine_rx`, lower limit: **97.44%**.
- `Thigh_L_rx`, lower limit: **96.90%**.
- `Thigh_R_rx`, lower limit: **96.69%**.
- `Shin_L_rx`, upper limit: **96.90%**.
- `Shin_R_rx`, upper limit: **96.47%**.
- `Foot_R_ry`, upper limit: **97.22%**.

Position means also sit near the actual boundaries: `Spine_rx` **-0.40037 rad**
versus lower limit **-0.4**, `Thigh_L_rx` **-0.49683 rad** versus **-0.5**, and
`Shin_L_rx` **0.09945 rad** versus upper limit **0.1**. Slight excursions beyond
the XML bounds are recorded; these physical joint limits are soft constraints.
The ten-minute policy shows a similar pattern. The teacher has **zero outward
near-limit drive on every actuated joint** and its largest actuated hinge RMS
drift from reset is **1.02753 degrees**. Passive wrists are analyzed separately
and are not evidence of a failed motor-controlled posture.

Initial mean `Thigh_L_rx` normalized commands are **+0.0697** for the teacher,
versus **-0.0859 / -0.1655 / -0.2270** for the learner budgets. `Thigh_R_rx` shows
the same sign reversal: teacher **+0.0555**, learners **-0.0512 / -0.1814 / -0.2187**.
These are signs in the model's actual joint coordinates; they are not inferred
anatomical labels. The command mismatch and subsequent limit bracing are directly
observable, but the contribution of an individual motor is not isolated here.

### Approach to terminal failure and missing measurements

Pelvis height is still approximately **0.90 m** during the initial loss of balance.
The thirty-minute deterministic episodes later terminate below **0.65 m** in
**8/8** cases, and its sampled episodes do so in **24/24**. Ten-minute counts are
**8/8** and **22/24**; three-minute counts are **3/8** and **11/24**. These are
observed height-threshold crossings, not claims that height is the only condition
violated on the terminal step.

First two-foot support violations do not universally precede the angular-speed
event. All sampled cases eventually violate the five-newton-per-foot support
gate, with mean first times **0.36531 / 0.38128 / 0.48893 seconds**. The recorded
foot loads can later recover even while the full balance gate remains violated.

The traces do not contain non-foot contact forces, contacting body IDs, individual
upper-body tilts/clearances or separate posture-criterion flags. For FAIL cases
above the pelvis-height threshold, non-foot contact is a plausible explanation
under the reviewed termination code, but is not directly confirmed. Observed
joint drifts are hinge-coordinate changes from the paired reset, not reconstructed
body-orientation errors. Teacher/deterministic held torques are reconstructed
from the verified control contract; sampled held torques are directly recorded
and exactly match the two-command mapping. Associations in these trajectories
do not prove a particular reward defect or guarantee that a torque edit solves Stand.

### Verification and decision

Protected input and inherited source/snapshot hashes remain unchanged. The
analyzer verifies identical resets, model/observation/control contracts, XML hinge
ordering and radians, positive unit motor gearing, live episode prefixes, finite
arrays, frame timing, saved report durations and sampled delayed torques. Padding
after termination is excluded. Per-frame joint/command summaries weight longer
episodes more heavily; episode event means weight poses equally. Only the offline
analyzer, diagnostic outputs and this documentation are added. No production
configuration, physical plant, reward, springs or Arena selection changes, and
no general tests or standalone compilation run.

Prioritize **balance duration and joint-stop behavior**, alongside time-to-FAIL,
in the next comparison. The longer training policy increasingly uses joint-limit
bracing and has a stronger initial angular disturbance. Its extra survival is
not evidence of acquiring sustained natural Stand. The feasible teacher trajectory
shows that standing is achievable under this spring-free plant, without proving
why PPO is settling on these commands.

### Proposed next causal experiment; not executed

Use the **frozen thirty-minute policy** in a paired native comparison. In the
intervention arm only, execute the teacher's recorded **first six commands
(0.100008 seconds)** for each matched reset pose, then hand control to the
unchanged learned policy. Preserve the two-command queue: the first learned
command takes effect in the interval ending at **0.150012 seconds**. The control
arm executes the same learned policy from the first command onward.

Run deterministic mean control and Gaussian inference seeds 301010-301012 on
eight poses in both arms: **64 episodes**, twelve-second maximum per case. Call
the frozen policy and consume the paired Gaussian draws during the prefix too,
then override only the executed commands in the intervention arm. Verify identical
resets, baseline reproduction, paired noise, held-torque mapping and frozen tensors.
Report survival, longest balanced/posture-valid duration, initial angular speed
and delayed outward joint-limit drive, including the handoff interval separately.

This isolates whether a physically balanced initial command prefix materially
changes the failure trajectory, or whether the learned feedback quickly returns
to the same limit-bracing strategy. It does not constitute Stand training or a
deployable teacher-assisted policy. The prefix trajectory is a different behavior
policy, so critic calibration is off-policy there. No optimizer updates, rewards,
plant, springs, termination gates, production defaults or Arena artifacts change.
Implement and execute this bounded diagnostic only after the user's approval.

## Phase 21 - paired recorded-teacher prefix and frozen handoff (2026-10-02)

The user approved the Phase 20 causal experiment with **Proceed**. Execute
64 native episodes: eight reset poses, deterministic control plus Gaussian
inference seeds 301010-301012, for the unmodified control arm and the six-command
recorded-teacher-prefix arm. Use the frozen thirty-minute actor/critic/AMP snapshot,
unchanged reference_v9 rewards, twelve-second maximum and original physical gates.
No learning, production defaults, springs, plant or Arena artifacts change.

Evaluate the learned policy during the prefix and consume the same Gaussian
draws, overriding only executed commands. Preserve the two-command torque queue
and capture proposals, overrides, executed commands and held torque. Verify
control-arm reproduction of the Phase 19 audit, paired draws/resets, teacher-state
reproduction through the delayed prefix, action mapping, source/input hashes and
frozen tensors. Compare post-handoff balance/posture duration, survival, angular
disturbance and outward torque near joint limits. Treat critic scores on the
intervention trajectory as off-policy diagnostics, not calibration of its behavior.

Execution and findings will be recorded below after this approved diagnostic completes.

### Completed execution and artifacts

All **64 approved native episodes** completed: 32 control and 32 prefix-arm
episodes, each comprising eight deterministic cases and 24 Gaussian sampled
cases. No optimizer updates or new training occur.

- Diagnostic runner: `mujoco_rig/mimic/audit_stand_handoff.py`.
- Report, protocol, source snapshots and eight per-trial traces:
  `logs/mimickit-training/stand-handoff-20261002-01`.
- Derived paired comparison:
  `logs/mimickit-training/stand-handoff-20261002-01/comparison.json`.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_handoff `
  --out logs/mimickit-training/stand-handoff-20261002-01
```

### A stable teacher prefix does not preserve learned balance after handoff

The control arm reproduces **every array of all four Phase 19 frozen audit
traces exactly**, including deterministic physical states and sampled Gaussian
draws/actions. The intervention arm reproduces the recorded teacher's qpos/qvel
for the first **eight completed physics-control intervals**, through
**0.133344 seconds**, in every case. The six-command prefix lasts 0.100008 seconds;
two further teacher commands remain in the original delay queue. All cases are
balanced and posture-valid throughout these first eight intervals.

The first learned command is selected from a balanced teacher-prefix state at
0.100008 seconds and takes effect in the interval ending at **0.150012 seconds**.
In **all 32 intervention episodes**, the complete balance gate fails in that
very interval. Observation clipping at selection of that first learned command
is **zero for all poses and inference modes**.

At the end of the preceding interval, mean root angular speed is **0.11340 rad/s**.
When learned torque takes effect, it rises to **5.93952 rad/s** deterministically
and **6.07822 rad/s** across the Gaussian cases. The fixed balance limit is
**1 rad/s**. This is direct evidence that this frozen policy cannot maintain the
balanced handoff state; a six-command stable prefix is insufficient.

The runner's handoff metrics use the same scheduled interval (the ninth interval)
in both arms. Fields named `first_learned_torque_interval_*` refer to the first
learned interval in the intervention arm; for control they are measurements at
the same comparison time, not its first learned interval. Control's first learned
torque already acts in interval three, as established in Phase 20.

### Survival, genuine balanced duration and joint-limit behavior

Both arms have **0/8 deterministic Stand passes**, **0/24 sampled Stand passes**
and zero survivors. Every episode ends in physical FAIL, with no solver warnings.

- Deterministic mean survival: **1.95016 -> 1.60221 s**, **-17.84%**.
- Sampled mean survival: **2.02933 -> 1.63902 s**, **-19.23%**.
- Deterministic post-handoff balanced time: **0 -> 0 s**.
- Sampled mean total post-handoff balanced time: **0.00764 -> 0.00625 s**.
- Longest post-handoff duration satisfying both balance and posture: **zero**
  for every case in both arms.

The intervention's deterministic longest balanced/posture-valid stretch is
**0.133344 seconds**, entirely from teacher-controlled physical intervals.
Its sampled longest balanced stretch is also 0.133344 seconds in every case.
That increase over control's initial 0.033336 seconds is teacher assistance,
not acquired learned balance. Prefix-arm sampled survival means by inference
seed are **1.65638 / 1.62096 / 1.63971 seconds**; each is below its paired control.

The post-handoff deterministic fraction of frames near a joint limit with held
torque directed farther toward that limit remains high in the intervention:

- `Spine_rx`: **98.86%**.
- `Thigh_L_rx`: **96.08%**.
- `Thigh_R_rx`: **97.86%**.
- `Shin_L_rx`: **98.12%**.
- `Shin_R_rx`: **96.99%**.

These fractions use the five-degree margin and delayed held torque at the ending
joint position, excluding the prefix from both arms. In control these five motors
have 100% outward near-limit drive over that later window. Sampled intervention
fractions remain **97.09 / 97.34 / 97.31 / 87.51 / 93.99%**, respectively.
The policy quickly resumes its limit-bracing pattern despite the stable handoff.

### Verification and interpretation

The final portable/AMP sidecar validates. All previous control traces reproduce
exactly; Gaussian draw prefixes match between intervention and control, resets
match the teacher, and actual held torques exactly match the executed commands
with the original two-command delay. Actor, critic, discriminator and normalizer
tensors remain unchanged, with zero pending normalizer samples. Protected inputs
and source snapshots verify. No production configuration, reward, plant, springs,
termination gates or Arena selection changes. No general tests or standalone
compilation run; the approved native diagnostic executes eight paired trials.

The intervention changes initial executed commands only. It establishes that
this particular stable prefix does not solve the frozen controller's behavior.
It does not prove that every teacher-assisted initialization would fail, or that
a particular reward penalty will solve PPO learning. Prefix critic/AMP scores are
off-policy diagnostics; they are not on-policy value calibration measurements.

Do not extend unchanged training or use this prefix as a deployment workaround.
The remaining problem is the learned torque strategy at balanced states and
through subsequent feedback, rather than merely the initial reset pose. Target
the measured outward joint-limit drive in a controlled learning experiment.

### Proposed next reward experiment; not executed

Run **two fresh 180-second PPO/AMP pilots**, training seed 301010, with identical
full initial tensors and the Phase 16-19 settings: actor SGD 1e-4/three epochs,
critic SGD 1e-3/two epochs, sigma 0.1, 64 environments, 64-frame rollouts, evaluation
interval 64, same spring-free plant, reference and admitted demonstrations.
One arm retains reference_v9; the other changes only the task reward by adding
an outward joint-limit torque cost on **all 33 actuated hinges**.

For each completed control interval, let `u` be the actual held motor torque
divided by its unchanged torque scale, and `q` the interval's ending hinge angle.
Use lower/upper proximity ramps clamped to [0,1] within **five degrees** of the
XML limits. Define `cost = mean_j(max(lower_proximity_j * max(-u_j, 0),
upper_proximity_j * max(u_j, 0)))`. The candidate task reward is
**reference_v9_reward - 1.0 * cost**. Apply the existing physical FAIL handling
afterward, preserving zero failed task reward and unchanged terminal bootstrap.
Keep AMP weighting at task 0.8 / discriminator 0.2 and its current reward scale.
This adds reward feedback; it does not suppress, redirect or stiffen a motor.

Use a diagnostic-only task wrapper so existing reward/plant implementations and
production defaults remain unchanged. First verify the cost on the recorded
teacher and learner traces, including delayed-torque alignment. The teacher
should have zero actuated outward-drive cost in the recorded trajectory. This
test uses the current reference_v9 foundation and all actuated hinges; it is not
an unchanged repeat of the historical nine-upper-hinge bracing reward.

Assess update 64 and final native/GPU Stand, then frozen deterministic and paired
three-seed Gaussian twelve-second trials for each final-budget policy. Compare
longest balance-and-posture-valid duration, actual balanced time, outward-drive
occupancy and time-to-FAIL. Preserve raw reference_v9 and penalty components
separately, since candidate task returns use a different reward function.
Do not select a policy merely for a lower penalty or longer unbalanced survival.
This coefficient is a bounded diagnostic choice, not a claimed optimum. No
additional sweep, long run or Arena loading is included. Implement and execute
only after the user's approval.

## Phase 22 - paired fresh outward-limit reward pilots (2026-10-02)

The user approved the Phase 21 proposal with **Proceed**. Verify the specified
five-degree, all-33-motor outward held-torque cost on recorded teacher and learner
traces, then execute two fresh **180 synchronized training-second** PPO/AMP pilots
with penalty weights **0 and 1**. Both use seed 301010, identical initial tensors,
actor SGD 1e-4/three epochs, critic SGD 1e-3/two epochs, sigma 0.1, 64 environments,
64-frame rollouts and evaluation every 64 updates. All plant, reference, demos,
normalizers, termination handling and AMP weights remain fixed.

Use a diagnostic task wrapper and narrowly scoped runner hooks. Capture the raw
reference reward and applied penalty separately in frozen audit traces. Assess
update 64, final sixty-second native/GPU Stand, and frozen deterministic/three-seed
Gaussian twelve-second behavior. Compare balanced/posture-valid duration and
outward delayed-torque drive as well as survival. Validate input/source hashes,
initial/final sidecars, shared initialization/configuration, paired draws/resets
and frozen tensors. No general tests, standalone compilation, production defaults,
springs, plant changes, extra sweep, long run or Arena loading are authorized.

### Completed execution and reproducible artifacts

**Decision: do not promote the weight-1 reward or extend it into a long run.**
Neither arm learns Stand, and weight 1 does not reduce the measured outward-limit
drive. The main immediate failure still occurs when the first learned torque
reaches the plant, before substantial limit cost is present.

Experiment root:
[stand-limit-reward-20261002-01](../../logs/mimickit-training/stand-limit-reward-20261002-01).
The [comparison](../../logs/mimickit-training/stand-limit-reward-20261002-01/comparison.json)
contains training reports, all 64 frozen learner episodes, per-hinge diagnostics,
raw reference/penalty components and equal-update assessment results.
[Comparison input hashes](../../logs/mimickit-training/stand-limit-reward-20261002-01/comparison-inputs.json)
and [offline aggregation source](../../logs/mimickit-training/stand-limit-reward-20261002-01/summarize.py)
preserve the analysis. Each arm has its full checkpoint, initialization, config,
update log, assessment traces and source snapshots; each frozen audit has four
complete traces and reports.

The implementation adds `outward_limit_reward.py` and
`compare_stand_limit_reward.py`, with optional task/source/trace hooks in the
existing diagnostic runners. Original task reward and physical control code are
unchanged. Weight 0 returns the original reward directly. Recorded reference
reward and applied penalty use the existing FAIL mask; recorded cost also retains
the terminal frame for physical diagnosis. Their difference reconstructs the
audited task reward. Historical audit source snapshots remain historical; the
current runner files intentionally include the new hooks.

The final protected-source preflight is
[preflight-verified/preflight.json](../../logs/mimickit-training/stand-limit-reward-20261002-01/preflight-verified/preflight.json).
It repeats the initial check after correcting the diagnostic comparison-scope
metadata. Both preflights use saved traces only, with no simulator rollouts or
optimizer updates. The independent NumPy and Torch formulas agree, sampled
held-torque arrays match the two-command queue, and historical trace/report hashes
verify. Teacher cost is exactly **zero on all eight episodes**. Previous 3/10/30
minute deterministic learners have mean live costs **0.014269 / 0.034431 /
0.059360** respectively. This validates the requested measurement; it does not
establish that its addition will improve policy learning.

Executed commands, in order, with the same project virtual environment:

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_limit_reward `
  --mode preflight --out logs/mimickit-training/stand-limit-reward-20261002-01/preflight-verified
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_limit_reward `
  --mode train --weight 0 --out logs/mimickit-training/stand-limit-reward-20261002-01/weight-0
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_limit_reward `
  --mode train --weight 1 --out logs/mimickit-training/stand-limit-reward-20261002-01/weight-1 `
  --paired-with logs/mimickit-training/stand-limit-reward-20261002-01/weight-0
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_limit_reward `
  --mode audit --weight 0 --run logs/mimickit-training/stand-limit-reward-20261002-01/weight-0 `
  --out logs/mimickit-training/stand-limit-reward-20261002-01/audit-weight-0 `
  --paired-audit logs/mimickit-training/stand-critic-audit-20261002-01
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_limit_reward `
  --mode audit --weight 1 --run logs/mimickit-training/stand-limit-reward-20261002-01/weight-1 `
  --out logs/mimickit-training/stand-limit-reward-20261002-01/audit-weight-1 `
  --paired-audit logs/mimickit-training/stand-limit-reward-20261002-01/audit-weight-0
$env:PYTHONPATH = (Get-Location).Path
.\mujoco_rig\mimic\.venv\Scripts\python.exe logs/mimickit-training/stand-limit-reward-20261002-01/summarize.py
```

Training/audits execute sequentially in hidden processes. The budget counts only
synchronized training updates; it excludes setup and deterministic assessments
and completes the last update. Aggregation initially failed because direct script
execution lacked the project import path; the shown command supplies it. No
training or physics rerun was required for that analysis invocation.

### Physical results

Both fresh full initial states have SHA256
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
PPO configs, reference, plant, demonstrations and source hashes match. Fresh
initial native survival is **0.335444 s** for both arms.

- **Weight 0:** **181.965682 s**, **71 updates**, **290,816 transitions**.
  Native survival at update 64 is **1.093838 s**; final native and GPU are both
  **1.241766 s**, with **0/8 Stand passes** each. Frozen Gaussian survival is
  **1.072308 s**, with **0/24 Stand passes**.
- **Weight 1:** **181.874510 s**, **73 updates**, **299,008 transitions**.
  Native survival at update 64 is **1.220931 s**; final native and GPU are both
  **1.202180 s**, with **0/8 Stand passes** each. Frozen Gaussian survival is
  **1.063974 s**, with **0/24 Stand passes**.

The candidate's survival advantage at update 64 reverses at the final budgets.
Final deterministic survival decreases **3.19%**, sampled survival decreases
**0.78%**. Final policies have different update counts because wall time is fixed;
the equal-update assessment is recorded separately. These results do not establish
a general reward effect or learning-speed ratio from one training seed.

Both deterministic policies have mean total and longest continuous
**balance-and-posture-valid duration 0.033336 s**, consisting of the two intervals
before learned torque delivery. All 16 deterministic cases lose balance at
**0.050004 s** and never recover balance. Gaussian balanced/posture-valid duration
is **0.033336 s** without the penalty and **0.0340305 s** with it. The latter adds
one control interval in one of 24 cases: that case first loses balance at
0.066672 s. All other Gaussian cases first lose balance at 0.050004 s. This is
not sustained standing or a meaningful recovery.

Mean live outward cost increases from **0.013208 to 0.014050** in deterministic
traces (**+6.37%**) and from **0.020423 to 0.020821** in sampled traces (**+1.95%**).
The sampled statistic averages three per-trial live-frame means. Outward torque
occupancy across the 33 motors increases from **26.673% to 27.640%** deterministic
and **21.198% to 21.599%** sampled. Longer live traces receive more weight within
each per-frame statistic; episode time measures average cases equally.

At the first held learned torque, the cost is **zero in seven of eight poses in
every trial of both arms**. Deterministic first-held raw reference rewards remain
**0.903965 / 0.912067** despite all cases failing the angular-speed/balance gate.
First-held mean angular speed decreases **2.879782 to 2.698835 rad/s**, still far
above the fixed **1 rad/s** Stand limit. The limit penalty is therefore largely
absent at the immediate observed failure. This explains its sparse immediate
coverage; it does not prove that a different reward will teach balance.

Discounted original reference returns remain separately comparable: deterministic
**32.552779 / 31.663653**, sampled **23.045234 / 22.983676**, weight 0 / 1.
Candidate applied penalty returns are **0.665265** deterministic and **0.911939**
sampled; sampled penalized task return is **22.071737**. Each policy has its own
final discriminator/critic, so combined scores and critic calibration are assessed
against its own objective rather than used as a common ranking metric.

### Verification and next bounded proposal; not executed

Portable/AMP sidecars, saved fresh initialization, paired reset states and Gaussian
draw prefixes verify. Frozen actor/critic/discriminator/normalizer hashes remain
unchanged and pending normalizer samples stay zero. Protected inputs and source
snapshots verify; final native/GPU survival matches case by case. All 64 frozen
learner episodes have no solver warning; all four training/audit stderr logs are
empty. No general tests or standalone compilation run. Production configuration,
physics, springs, termination and Arena selection remain unchanged.

Keep weight 0 as the experimental baseline. Do not increase the limit coefficient
or training duration based on these results: seven reset poses initially receive
no limit cost even when balance is lost. More weight cannot add missing coverage
at those specific states.

**Proposed next step:** audit a direct, smooth root-angular-speed cost on the saved
teacher and both new frozen policies, before spending more training compute.
Use `speed = norm(qvel[3:6])`,
`cost = 1 - exp(-(max(speed - 1 rad/s, 0)/(1 rad/s))**2)` and compare original
reference task reward with **reference_reward - 0.25*cost**, retaining zero FAIL
task reward. This is one explicitly fixed diagnostic coefficient, not a sweep or
chosen optimum. Rescore all nine existing traces (72 episodes), verify the teacher
remains unpenalized, measure coverage/magnitude at first balance loss, and report
same-prefix and full-live discounted task returns. Preserve original reward and
new cost separately. Do not compare different frozen discriminators as a common
objective. Perform **zero simulator rollouts and zero optimizer updates**; do not
change production rewards or launch another pilot. This checks whether the proposed
feedback reaches the observed failure states, not whether PPO will learn from it.
Execute only after the user's approval.

## Phase 23 - offline angular-speed reward coverage (2026-10-02)

The user approved the Phase 22 proposal with **Proceed**. Execute the specified
offline rescoring of the eight-case teacher and both Phase 22 frozen policies,
including deterministic and three-seed Gaussian traces: **nine traces / 72
episodes**. No new trajectory, checkpoint, optimizer update or normalizer sample
is produced. The original traces and physical behavior remain fixed.

**Result: the proposed angular-speed cost reaches first balance loss in all 64
learner episodes while leaving every teacher frame unpenalized.** This passes the
feedback-coverage check and justifies a bounded training comparison, not a claim
that the dummy has improved or will learn Stand.

### Formula, implementation and artifacts

`audit_stand_angular_reward.py` uses NumPy and standard-library code only. It reads
saved end-of-control-interval `qvel[3:6]`, in rad/s, and computes
`cost = 1 - exp(-(max(norm(qvel[3:6]) - 1, 0)/1)**2)`.
The rescored task reward is **original reference_v9 reward - 0.25*cost**, with
the existing FAIL mask applied afterward. The previous outward-limit penalty is
removed when reading weight-1 traces; the two penalties are not stacked. Failed
task reward remains zero, TIME retains its ordinary per-frame reward, and no
critic timeout bootstrap or AMP contribution is rescored. The fixed discount is
0.99, verified against both training configs.

[Report](../../logs/mimickit-training/stand-angular-reward-audit-20261002-01/report.json)
and [protocol](../../logs/mimickit-training/stand-angular-reward-audit-20261002-01/protocol.json)
contain per-case full-live and common-prefix returns, first balance-loss scores,
cost coverage and negative-reward fractions. Each of the nine output directories
contains `scores.npz` with original reference reward, angular speed, raw cost,
applied penalty, rescored task reward, live mask, times and terminal flags. Inputs
and exact local/upstream source snapshots are hashed and preserved.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_angular_reward `
  --out logs/mimickit-training/stand-angular-reward-audit-20261002-01
```

The first invocation stops before output creation because the existing manifest
mixes absolute and relative paths. Resolving manifest keys and lookup paths fixes
that guard; the completed invocation verifies all historical hashes. No input
artifact is changed and no simulation is run during either invocation.

### Coverage and measured scores

- **Teacher:** angular speed never exceeds **0.807321 rad/s**, below the fixed
  1 rad/s gate. Cost and penalty are exactly zero in all eight trajectories;
  original and rescored rewards match exactly. Stand remains **8/8** in these
  saved traces.
- **Learners:** angular cost is positive at first balance loss in **64/64**
  episodes. The previous limit cost is zero at that same event in **56/64**.
  Mean applied angular penalty at first loss is **0.227379**, lowering mean raw
  task reward **0.874299 to 0.646920**. Individual first-loss penalties range
  **0.000746-0.250000**; coverage alone does not imply uniform signal strength.
- **Later live behavior:** cost is positive in about **76.27% / 74.98%** of
  Gaussian live frames, averaging cases equally, for Phase 22 weight 0 / 1.
  About **23.34% / 22.89%** of those frames have cost at least 0.99, close to its
  cap. The proposed term distinguishes speeds most strongly near the threshold;
  much larger speeds produce nearly the same maximum penalty.

Full-live discounted **task-only** observed returns, original to rescored:

- Teacher: **99.667860 to 99.667860**.
- Weight-0 deterministic: **32.552779 to 28.422221**.
- Weight-1 deterministic: **31.663653 to 27.634929**.
- Weight-0 Gaussian, 24 cases: **23.045234 to 17.950278**.
- Weight-1 Gaussian, 24 cases: **22.983676 to 17.735962**.

The teacher's full-live value is a twelve-second truncated observed prefix,
whereas learner trajectories end at physical FAIL. The audit also compares all
nine controllers over identical per-pose **45-58 frame / 0.750060-0.966744 s**
prefixes, independent of their eventual episode lengths. Common-prefix teacher
return stays **39.861189**. Deterministic original/rescored returns are
**28.523000 / 25.928091** for weight 0 and **28.076848 / 25.551324** for weight 1;
Gaussian means are **21.494735 / 17.613214** and **21.486379 / 17.345523**.
The teacher also has greater rescored return on **every** learner pose when scored
over that learner's own measured episode duration. The original objective already
favored the teacher; this audit shows more immediate feedback at the observed
angular-speed failure, rather than discovering a previously reversed ranking.

The specified subtraction permits negative task rewards in degraded live states:
**9.46% / 7.76%** of deterministic frames and **12.24% / 11.80%** of Gaussian
frames, averaging cases equally. No extra clamp is introduced. FAIL still has
zero task reward and continuation value, so negative future reward can affect
termination incentives. The offline trace audit does not determine the new
on-policy behavior or establish that the coefficient is optimal.

Historical trace/report hashes, identical reset coordinates/velocities, parsed
terminal flag values, failed reward zeroing, independent equivalent cost formulas
and source snapshots verify. Previous weight-1 task reward reconstructs correctly
from its recorded reference minus limit penalty. **Zero physics rollouts and zero
optimizer updates** occur. No tests, standalone compilation, production defaults,
plant, springs, termination, existing checkpoints or Arena changes occur.

### Proposed next paired pilot; not executed

Run two fresh **180 synchronized training-second** PPO/AMP pilots, seed **301010**,
with identical full initial tensors and PPO configs. Compare original
reference_v9 (**angular weight 0**) against **angular weight 0.25**, using exactly
the threshold/scale/formula above and the existing FAIL mask. Use no outward-limit
penalty in either arm. Keep actor SGD **1e-4 / three epochs**, critic SGD
**1e-3 / two epochs**, fixed sigma **0.1**, **64 environments x 64-frame rollout**,
evaluation interval **64**, same spring-free 33-motor plant, two-command delay,
reference, admitted demonstrations, normalizers and AMP weights. Implement a
diagnostic task wrapper using the existing runner hooks; preserve production
reward implementations and the Phase 22-23 sources/artifacts.

First verify wrapper cost against the saved NumPy rescoring, including actual
end-of-interval root angular speed and failed reward composition. Assess native
Stand at update 64 and final native/GPU sixty-second behavior, then frozen
deterministic and paired three-seed Gaussian twelve-second traces. Select the
final budget checkpoint, not the best assessment. Compare actual balanced and
posture-valid duration, first balance-loss time, angular speed, survival, limit
drive and negative-reward/terminal behavior. Preserve raw reference, angular
penalty and own AMP/critic components separately; do not rank different objectives
by their combined return alone. Require measurable physical progress to consider
any follow-up. No coefficient sweep, long run, stiffness changes or Arena loading
is included. Execute only after the user's approval.

## Phase 24 - paired fresh angular-speed reward pilots (2026-10-02)

The user approved the Phase 23 paired pilot with **Proceed**. Compare fresh
reference_v9 PPO/AMP with angular-speed penalty weights **0 / 0.25**, seed 301010,
**180 synchronized training-update seconds per arm**, actor SGD 1e-4/three epochs,
critic SGD 1e-3/two epochs, sigma 0.1, 64 environments, 64-frame rollouts and
evaluation interval 64. Both arms retain the same full initialization, spring-free
plant, two-command torque queue, reference, demonstrations, normalizers, physical
FAIL mask and AMP weights. Neither includes the Phase 22 outward-limit penalty.

The new diagnostic wrapper `angular_speed_reward.py` and driver
`compare_stand_angular_reward.py` use the existing runner hooks without modifying
the previous implementations or artifacts. The preflight compares the exact
Torch cost on CPU/GPU against all nine saved Phase 23 NumPy score arrays, verifying
the teacher receives zero cost and the failed task reward composition is zero.
Maximum cost error is **2.05e-7** and maximum reward error **5.64e-8**; input and
source hashes verify. This preflight creates no simulator rollout or training
update. Root-angular-speed magnitude is invariant under the native/world frame
rotation; frozen traces additionally compare engine speed/cost against saved
native qvel for the new actual trajectories.

Run native update-64 and final native/GPU sixty-second assessments, then frozen
deterministic and three paired Gaussian-seed twelve-second audits. Record original
reference, angular cost and penalty separately. Analyze balanced/posture-valid
time, including time after the two-command delay, angular speed, joint-limit drive,
negative rewards, terminal behavior and survival. No best-checkpoint selection,
parameter sweep, longer run, springs or Arena changes are included. No general
tests or standalone compilation are authorized.

### Completed comparison and decision

**Do not promote angular weight 0.25 or extend it into a longer training run.**
The term reaches the failure states, but this pilot does not establish useful
balance learning. Initial angular speed decreases modestly while deterministic
balance still fails at the first learned torque, and final survival decreases.

Experiment root:
[stand-angular-reward-pilot-20261002-01](../../logs/mimickit-training/stand-angular-reward-pilot-20261002-01).
[Comparison](../../logs/mimickit-training/stand-angular-reward-pilot-20261002-01/comparison.json),
[input hashes](../../logs/mimickit-training/stand-angular-reward-pilot-20261002-01/comparison-inputs.json)
and [aggregation source](../../logs/mimickit-training/stand-angular-reward-pilot-20261002-01/summarize.py)
preserve all training/frozen reports, physical event analysis, per-hinge drive,
negative task rewards and separate original/penalty returns. Each run contains
fresh initialization, final portable/AMP checkpoints, config, updates,
assessments and copied diagnostic sources. Each frozen audit contains four full
native traces with independently verified angular speed/cost and reward components.

Executed commands use `compare_stand_angular_reward`:

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_angular_reward `
  --mode preflight --out logs/mimickit-training/stand-angular-reward-pilot-20261002-01
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_angular_reward `
  --mode train --weight 0 --out logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_angular_reward `
  --mode train --weight 0.25 --out logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0.25 `
  --paired-with logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_angular_reward `
  --mode audit --weight 0 --run logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0 `
  --out logs/mimickit-training/stand-angular-reward-pilot-20261002-01/audit-weight-0
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.compare_stand_angular_reward `
  --mode audit --weight 0.25 --run logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0.25 `
  --out logs/mimickit-training/stand-angular-reward-pilot-20261002-01/audit-weight-0.25 `
  --paired-audit logs/mimickit-training/stand-angular-reward-pilot-20261002-01/audit-weight-0
$env:PYTHONPATH = (Get-Location).Path
.\mujoco_rig\mimic\.venv\Scripts\python.exe logs/mimickit-training/stand-angular-reward-pilot-20261002-01/summarize.py
```

All four training/audit processes run sequentially in hidden windows and finish.
The baseline audit pairs its Gaussian draw prefixes against the Phase 22 weight-0
audit; the candidate audit pairs against the new baseline. Final checkpoint
selection uses training budget only. Setup and assessments are excluded from
synchronized training time, and the last update completes before stopping.

### Stand and physical behavior

Both arms have identical fresh full-state SHA256
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`;
PPO configs and protected source/plant/reference/demo inputs match. Initial native
survival is **0.335444 s** in both arms.

- **Weight 0:** **180.631594 s**, **71 updates**, **290,816 transitions**.
  Update-64 native survival is **1.181344 s**; final native/GPU survival is
  **1.258434 s**, with **0/8 Stand passes** each. Frozen Gaussian survival is
  **1.096615 s**, with **0/24 Stand passes**.
- **Weight 0.25:** **181.706962 s**, **72 updates**, **294,912 transitions**.
  Update-64 native survival is **1.302187 s**; final native/GPU survival is
  **1.193845 s**, with **0/8 Stand passes** each. Frozen Gaussian survival is
  **1.048695 s**, with **0/24 Stand passes**.

Again the candidate survival advantage at update 64 does not persist at the final
budget. Final deterministic survival decreases **5.13%**, sampled survival
decreases **4.37%**. Final update counts differ because wall time is the budget;
the equal-update assessment is preserved separately. The repeated fresh baseline
also differs from Phase 22 despite the same full initialization/seed/settings,
so a small difference in one pair is not a replicated reward effect.

All 16 deterministic cases first lose balance at **0.050004 s**, the first
delivered learned torque. Both deterministic mean total and longest continuous
balanced/posture-valid times remain **0.033336 s**, with **zero** such time after
the two-command delay and no balance recovery. Gaussian balanced/posture-valid
time is **0.033336 / 0.0340305 s**, baseline/candidate. Candidate improvement is
one extra interval in one of 24 cases; that case first loses balance at
**0.066672 s**. All other 47 Gaussian cases first lose balance at 0.050004 s.
Candidate has no later balance recovery. One baseline sampled case has one later
balanced frame without valid posture, contributing no combined valid time.

The candidate does reduce **first-held** root angular speed:
deterministic **2.911051 to 2.689001 rad/s (-7.63%)**, Gaussian
**3.188792 to 2.981618 rad/s (-6.50%)**. These remain far above the **1 rad/s**
gate. Across entire live traces, deterministic mean angular speed increases
**1.509640 to 1.602715 rad/s (+6.17%)** and cost increases **0.288604 to 0.300211**.
Gaussian mean angular speed decreases **2.051894 to 1.941017 rad/s (-5.40%)**
and cost decreases **0.463513 to 0.405023 (-12.62%)**. These live-frame means
depend on each policy's trajectory and survival; they do not measure matched-state
controller improvement.

Outward joint-limit drive across all 33 motors is **27.960% / 29.251%**
deterministic and **22.400% / 22.192%** sampled, baseline/candidate. There is no
command clipping or near-saturation command in any of the frozen traces.
Candidate negative task reward occupies **8.19%** of deterministic live frames
and **9.20%** of sampled live frames, averaging episodes equally. All FAIL task
rewards remain exactly zero. Different mixtures of height and non-height failure
occur; the traces do not identify individual non-foot contacting bodies.

### Reward accounting and verification

Original reference discounted returns remain comparable: deterministic
**32.487408 / 31.225881**, Gaussian **23.048441 / 22.626739**. Candidate applied
angular penalty returns are **3.406034** deterministic and **4.409300** Gaussian;
its Gaussian penalized task return is **18.217439**. Its frozen critic/discriminator
scores use its own trained objective and are not a shared policy ranking.

Saved initial tensors, portable/AMP sidecars, paired Gaussian prefixes and native
reset states verify. Frozen actor, critic, discriminator and normalizer tensors
remain unchanged; pending normalizer samples are zero. Protected inputs and
source snapshots verify. New native traces independently reproduce angular
speed/cost from qvel, and task rewards reconstruct as reference minus applied
penalty. Native/GPU final survival agrees on every case. All 64 frozen learner
episodes have no solver warning; all four stderr logs are empty. No general test
suite or standalone compilation runs. Production reward defaults, physical
plant, passive springs, termination and Arena selection remain unchanged.

Keep the original reward as the experimental baseline. Neither limit feedback nor
this angular coefficient has demonstrated sustained balance. Do not continue
these reward variants for an hour or select them from lower loss/cost alone.

### Proposed next action-effect diagnostic; not executed

Before another learning run, isolate the physical effect of the **first torque
vector** at the balanced reset. Reuse the saved teacher first commands and the
two Phase 24 deterministic actors' first commands; no networks need to run or
update. For each actor, interpolate
`command = teacher_command + fraction*(actor_command - teacher_command)` at
fixed fractions **0, 0.25, 0.5, 0.75, 1**. Reuse the identical teacher endpoint,
giving **nine command settings x eight poses = 72 short native cases**.

Each case starts from the same saved reset, uses the original two-command queue
and spring-free torque plant, and runs **exactly three control intervals**. Put
the interpolated first command into interval 1; use saved teacher commands for
intervals 2 and 3, which cannot reach the plant before this probe ends. Verify
actual held torque alignment and unchanged zero-torque first two intervals.
The pure teacher and pure actor endpoints must reproduce their saved first-three
interval states, with exact/tolerance checks appropriate to stored precision.
Report endpoint angular speed, root linear velocity, foot support, posture and
all fixed balance gates per pose and fraction, plus command distance/torque
components. This measures whether small movement from successful torque toward
the learned vector crosses a sharp local balance boundary, and how that boundary
varies by pose. It does not test later feedback or promise a better action scale.

This is a causal command intervention on the first effect, distinct from the
Phase 21 delayed teacher-to-policy handoff and from offline trajectory association.
Perform **zero optimizer updates**, retain all existing policies/normalizers,
and make no training-config, reward, spring or Arena change. No additional
fractions, long trajectory or parameter sweep is included. Execute only after
the user's approval.

## Phase 25 - causal first-torque interpolation (2026-10-02)

The user approved the Phase 24 action-effect diagnostic with **Proceed**.
`audit_stand_first_torque.py` executes **nine command settings x eight poses =
72 native cases**, each exactly **three control intervals / 0.050004 s**. This is
**216 control intervals**, with zero training updates, no checkpoint loading,
no optimizer creation and no neural-network inference. All commands come from
the saved teacher and Phase 24 deterministic traces.

**Finding: a 25% movement from the teacher toward either learned first command
already violates the angular-speed gate on all eight poses.** Every other balance
gate and every posture criterion still passes at that instant. This isolates an
immediate angular-control effect of the command vector, preceding the later
physical fall or contact failure.

### Execution, controls and artifacts

[Report](../../logs/mimickit-training/stand-first-torque-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-first-torque-20261002-01/protocol.json)
and [completion verification](../../logs/mimickit-training/stand-first-torque-20261002-01/completion-verification.json)
preserve all 72 per-case states/gates, command and torque components, per-pose
tested brackets, source snapshots and input/output hashes. Each setting directory
contains its three-frame trace and detailed report.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_first_torque `
  --out logs/mimickit-training/stand-first-torque-20261002-01
```

For each learned actor, use fractions **0, 0.25, 0.5, 0.75, 1** in
`first_command = teacher_first + fraction*(actor_first - teacher_first)`.
Fraction 0 is shared between the two actor families, giving nine settings.
Endpoints use the original recorded commands directly to preserve floating-point
identity. Fraction 0.25 contains 75% of the teacher command and 25% of the learned
command; it does not change actuator torque authority.

Every case starts from the same saved reference pose and zero velocity. The first
command is queued at interval 1; saved teacher commands are queued at intervals
2 and 3. Only the first command reaches the motors before the probe ends. The
plant retains its original normalized torque mapping, two-command delay,
spring-free 33-motor/six-passive-wrist configuration, native timestep and last-solve
contact sampling. A plain `TorqueStandTask` provides the original physical gates
without creating AMP scorers or neural policies; its physical engine path is the
same as the previous audits.

### Endpoint measurements

Mean root angular speed at **0.050004 s**, in fraction order
**0 / 0.25 / 0.5 / 0.75 / 1**:

- Toward the original-reward actor: **0.576000 / 1.162566 / 1.750429 /
  2.333909 / 2.911051 rad/s**.
- Toward the angular-penalty actor: **0.576000 / 1.104602 / 1.636239 /
  2.165020 / 2.689001 rad/s**.

The teacher endpoint meets balance and posture gates on **8/8** poses. Every
nonzero tested fraction meets them on **0/8**. At fraction 0.25, angular speed
ranges **1.062318-1.357125 rad/s** toward the original actor and
**1.008241-1.229692 rad/s** toward the penalty actor, all above the **1 rad/s**
gate. In all **64 nonteacher cases**, angular speed is the only failing balance
criterion. Height, pelvis uprightness, horizontal speed, two-foot support,
non-foot ground contact and finite state all pass. All **72** endpoint postures
are valid and all terminal flags are **NULL**; these are momentary balance-gate
results, not successful Stand episodes or physical FAIL terminations.

For every pose in both command families, the observed first gate crossing lies
between **fraction 0 and 0.25**. The tested balance-pass sequence and angular-speed
sequence are monotonic on this grid. No extra fractions were executed; this does
not establish the exact threshold, behavior between samples or a sharp physical
discontinuity. Angular speed grows progressively at the measured blends. No
claim of a universally safe action scale or exploration sigma follows from these
two coordinated directions.

Mean per-pose normalized command RMS distance from teacher at fraction 0.25 is
**0.011160 / 0.010300**, original/penalty actor. At fraction 1 it is
**0.044639 / 0.041201**. These average each pose's RMS equally, rather than taking
one RMS over all poses. The large torque differences include `Thigh_L_rx`
(**19.34 / 16.33 Nm RMS**) and `Thigh_R_rx` (**12.67 / 13.81 Nm RMS**), measured
against the teacher at the pure actor endpoint. Their mean teacher commands are
**+0.069703 / +0.055524**; mean original actor commands are **-0.090056 /
-0.049733**, and penalty actor commands **-0.066015 / -0.059125**. These recorded
component differences identify candidate channels for investigation, not their
individual causal contributions: the interpolation changes all motors together.

### Verification and interpretation

The pure teacher and both pure actor endpoints reproduce their original saved
first-three-frame **qpos, qvel, foot loads, pelvis height/tilt, balance, posture
and done flags exactly**, with zero numerical error. Every setting reproduces
the teacher's first two zero-torque qpos/qvel intervals exactly. Recorded held
torques are exactly zero for those two intervals and exactly equal the first
normalized command times the original motor scale in interval 3. Resets match
the saved teacher exactly. All state arrays are finite and there are no solver
warnings. The whole original Phase 24 manifest verifies separately, along with
all diagnostic source snapshots and final output hashes.

This intervention demonstrates that the first learned torque vector is sufficient
to cause the measured angular-speed gate failure in the original spring-free
plant. The teacher provides a passing first-command counterexample on every
pose. It does not identify a defective motor, establish a global stability region,
test subsequent feedback, or show that more reward weighting/longer PPO would
correct the command. No existing policy/normalizer, training configuration,
reward, physical plant, springs or Arena selection changes. No general test suite
or standalone compilation runs; only the approved short native probe executes.

### Proposed next motor-sensitivity diagnostic; not executed

Measure which torque channels drive this initial angular response before choosing
another learning intervention. Around the same saved teacher first command,
perturb **each of 33 actuated channels by +0.001 and -0.001 normalized units**,
one channel at a time, plus the unchanged teacher. This gives **67 settings x
eight poses = 536 native cases**, each the same three control intervals, for
**1,608 control intervals**. Retain saved teacher commands in intervals 2 and 3,
the same resets/plant/delay/contact sampling, and no policy/optimizer inference.

Use paired perturbations to estimate per-pose local derivatives of endpoint root
angular and linear velocity vectors with respect to normalized command and Nm.
Record all balance/posture gates, actual held torques and midpoint/nonlinearity
residuals. Rank channel sensitivity and each channel's linear contribution to
the two recorded actor-minus-teacher command vectors. Compare the summed local
prediction against the actual Phase 25 pure-actor endpoint vectors, reporting
prediction error rather than assuming the derivative remains valid for large
command differences. Teacher replay, zero-torque prefixes, immutable inputs and
source hashes must verify. This is a local physical diagnostic, not a chosen
exploration scale or training recipe. Perform **zero optimizer updates** and no
reward, action-scale, spring, model or Arena change. No extra perturbation sizes
or long trajectories are included. Execute only after the user's approval.

## Phase 26 - per-motor initial velocity sensitivity (2026-10-02)

The user approved the Phase 25 motor-sensitivity proposal with **Proceed**.
`audit_stand_motor_sensitivity.py` performs the unchanged teacher plus
**33 channels x two perturbation signs**, each across eight poses: **67 settings /
536 native cases / 1,608 control intervals**. Each case lasts exactly three
control intervals, **0.050004 s**, with zero optimizer updates, neural-network
inference or checkpoint loading. Previous code and experiment artifacts remain
unchanged; the new driver reuses the Phase 25 verified physics replay.

**Finding: `Thigh_L_rx`, `Thigh_R_rx` and `Spine_rx` dominate the local model's
angular-velocity error for both learned first commands.** The local model predicts
the actual full actor command's angular-velocity change within **1.85% / 1.54%**
relative vector error, original-reward/angular-penalty actor. This supports the
channel attribution, but a three-channel correction has not yet been simulated.

### Execution and artifacts

[Report](../../logs/mimickit-training/stand-motor-sensitivity-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-motor-sensitivity-20261002-01/protocol.json),
[dominant-channel projection accounting](../../logs/mimickit-training/stand-motor-sensitivity-20261002-01/dominant-motor-projections.json)
and [completion verification](../../logs/mimickit-training/stand-motor-sensitivity-20261002-01/completion-verification.json)
preserve per-case physical gates, derivative ranks, prediction errors, input/source
and output hashes. All 67 setting directories contain detailed reports and
three-frame traces. `jacobians.npz` stores per-pose angular/linear velocity
derivative matrices in normalized-command and Nm units, plus symmetric midpoint
residuals. `actor-0-prediction.npz` and `actor-0.25-prediction.npz` retain actual,
predicted and residual vectors and each motor's contribution.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_motor_sensitivity `
  --out logs/mimickit-training/stand-motor-sensitivity-20261002-01
```

The driver runs in one hidden native process and completes all settings. Each
perturbation changes only one first-command coordinate by **+0.001 or -0.001**;
other first-command coordinates remain exactly equal to the teacher. Intervals
2 and 3 use the saved teacher commands, which cannot reach the plant within the
probe. All original torque scales, six passive wrists, zero stiffness/bias,
latency, reset states, timestep and last-solve contact sampling remain fixed.

The centered derivative uses the **actual float32 plus-minus command gap for
each pose**, not an assumed exact 0.002. Per-Nm units divide by the original motor
torque scale. Velocity components are evaluated in the world frame. Derivatives
are estimated independently per pose; no averaging of states precedes the
finite difference.

### Local physics and dominant channels

All **536/536** endpoints pass balance and posture and have terminal flag NULL.
Angular speed ranges **0.524808-0.645990 rad/s**, below the fixed 1 rad/s gate.
Thus these small individual perturbations stay within the measured first-effect
balance region. This does not establish the result for simultaneous noise or
larger torque deviations.

Highest equal-pose mean angular-vector sensitivity, in
**rad/s per unit normalized command**:

- `Spine_rx`: **8.624333**, or **0.071869 rad/s per Nm**.
- `Thigh_R_rx`: **7.853868**, or **0.065449 rad/s per Nm**.
- `Thigh_L_rx`: **7.664893**, or **0.063874 rad/s per Nm**.
- `Shin_R_rx`: **3.512613**, followed by `Spine_ry` **3.310357** and
  `Shin_L_rx` **3.200599**.

Sensitivity alone is not the actor's error attribution; multiply each local
derivative vector by the actor's recorded command error on that channel. Mean
norms of the dominant predicted angular contributions are:

- Original-reward actor: left hip **1.221737**, right hip **0.826593**, spine
  **0.407404 rad/s**.
- Angular-penalty actor: left hip **1.040048**, right hip **0.900565**, spine
  **0.278002 rad/s**.

Both hips' learned first commands differ from the teacher in the negative
direction; the penalty actor reduces the left-hip and spine errors but increases
the right-hip error. The model therefore explains the modest improvement in
initial angular speed without a change to the three main error channels.

Project contributions onto each pose's **observed actor-minus-teacher angular
velocity change direction**, then average the eight poses. The three channels
contribute **2.381053 of 2.383021 rad/s (99.92%)** of net model-projected change
for the original actor and **2.148706 of 2.155718 rad/s (99.67%)** for the penalty
actor. These are signed, additive model projections. Other channels can cancel
one another; the percentages are not variance explained, energy fractions or
measured outcomes of replacing those three torques. The largest opposing
projected contribution is `Chest_rx`, **-0.052229 / -0.021670 rad/s**.

### Prediction accuracy and approximation limits

For each actor, predict endpoint velocity as teacher endpoint velocity plus the
sum of local derivatives times its full recorded command-minus-teacher vector.
Compare against the saved Phase 25 pure-actor endpoint without new actor
rollouts or policy inference.

- Original actor angular speed: **2.947932 rad/s predicted / 2.911051 actual**.
  Mean angular-vector error **0.045122 rad/s**, maximum **0.086794 rad/s**;
  mean error relative to the observed change-vector norm **1.8475%**.
- Penalty actor angular speed: **2.717527 predicted / 2.689001 actual**.
  Mean angular-vector error **0.033506 rad/s**, maximum **0.067845 rad/s**;
  mean relative change-vector error **1.5441%**.

Linear velocity prediction is less accurate proportionally: mean vector errors
**0.005188 / 0.004124 m/s**, or **13.18% / 12.40%** of the respective observed
change. The local model must not be treated as an exact contact-dynamics model.
Maximum symmetric midpoint residual is **2.745e-6 rad/s** angular and
**4.074e-7 m/s** linear. Ratios against a very small response can be dominated by
numerical resolution; the maximum angular midpoint/half-span ratio is **1.06%**.
No second perturbation size or additional physics sweep is executed.

Teacher replay is exact for saved qpos/qvel, foot loads, pelvis height/tilt and all
gates/terminal flags. Every probe verifies identical resets, an exact two-frame
zero-torque prefix, correct actual delayed torque and finite states. All original
Phase 25 protected inputs/source snapshots verify before and after execution.
All new sources, snapshots and output hashes verify; stderr is empty and there
are no solver warnings. No general tests or standalone compilation run. Policies,
normalizers, training configs, reward, action authority, plant, springs and Arena
remain unchanged.

### Proposed next three-channel causal substitution; not executed

Check the model's dominant-channel attribution directly before designing another
training change. For each Phase 24 deterministic actor, compare its saved first
command against the same command with **only `Thigh_L_rx`, `Thigh_R_rx` and
`Spine_rx` replaced by the saved teacher values**. Keep all other 30 coordinates
exactly equal to that actor's saved first command. This gives **four settings x
eight poses = 32 native cases**, each exactly three control intervals,
**96 control intervals total**. Use saved teacher commands in intervals 2 and 3,
the original reset/plant/delay/contact sampling and the same physical gates.

The two unchanged actor controls must reproduce their saved Phase 25 endpoints
exactly. Record actual held torques, command substitutions, root angular and
linear velocities, foot support and every balance/posture criterion. Compare
corrected endpoints against both uncorrected actors and the local model's
prediction after removing the three channel errors. This tests whether those
channels are sufficient to explain the **first-effect** failure; it does not
test later policy feedback or deploy teacher-assisted control. Perform **zero
optimizer updates**, no neural inference and no reward, config, spring, action
scale or Arena changes. No extra channel groups, trajectories or training run
is included. Execute only after the user's approval.

## Phase 27 - Three-channel first-torque substitution (2026-10-02)

The user approved Phase 26's proposed causal substitution with **Proceed**.
Execute exactly **four settings x eight poses = 32 native cases**, each three
control intervals, **96 intervals total**. No further physics sweep or learning
run is included.

**Finding: replacing only `Thigh_L_rx`, `Thigh_R_rx` and `Spine_rx` with teacher
values restores balance and valid posture in 16/16 corrected endpoints. The
original commands pass balance in 0/16 endpoints.** This establishes that the
three-channel substitution is sufficient to prevent the measured first-effect
balance failure for these two actors and eight reset poses. It does not show
that every channel in the group is individually necessary, that the other 30
channels remain adequate later, or that either actor has learned Stand.

### Execution and artifacts

New driver: `mujoco_rig/mimic/audit_stand_three_motors.py`. It reuses the existing
Phase 25 saved-command execution and replay checks and Phase 26 local velocity
prediction. Neither existing helper is modified.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_three_motors `
  --out logs/mimickit-training/stand-three-motor-substitution-20261002-01
```

[Report](../../logs/mimickit-training/stand-three-motor-substitution-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-three-motor-substitution-20261002-01/protocol.json)
and [completion verification](../../logs/mimickit-training/stand-three-motor-substitution-20261002-01/completion-verification.json)
preserve per-pose gates, command/torque differences, exact control replay,
local-model comparisons and protected input/source/output hashes. Each of
`actor-0-control`, `actor-0-three-teacher`, `actor-0.25-control` and
`actor-0.25-three-teacher` contains `report.json`, `trace.npz` and
`prediction.npz`.

Use the two Phase 24 deterministic actors' **saved first commands**; no neural
network or optimizer is instantiated. Only the selected three first-command
coordinates are changed. All other **30 coordinates are exactly equal to the
corresponding actor command**. Intervals 2 and 3 contain the saved teacher
commands; the two-interval latency prevents these commands from reaching the
plant before the probe ends. The endpoint is **0.050004 seconds**, after the
first delivered command has acted for one control interval.

### Measured causal effect

Original-reward actor:

- Original command mean angular speed **2.911051 rad/s**, range
  **2.501157-3.531175**; endpoint balance and posture **0/8**.
- Three-channel correction mean **0.591097 rad/s**, range
  **0.503938-0.687842**; endpoint balance and posture **8/8**.
- Mean angular-speed reduction **79.6947%**.

Angular-penalty actor:

- Original command mean angular speed **2.689001 rad/s**, range
  **2.289854-3.066037**; endpoint balance and posture **0/8**.
- Three-channel correction mean **0.580646 rad/s**, range
  **0.501284-0.665848**; endpoint balance and posture **8/8**.
- Mean angular-speed reduction **78.4066%**.

The saved full-teacher first-effect mean is **0.576000 rad/s**. Both corrected
actors therefore approach its initial angular speed while retaining their
remaining 30 command coordinates. Every corrected endpoint passes all fixed
physical balance gates; corrected horizontal speeds remain below **0.038 m/s**
against the **0.2 m/s** gate. All original endpoints fail **only angular speed**.
All **32/32** endpoint postures pass and all terminal flags are **NULL**.
Endpoint balance at 50 ms is not the two-second stability window required for
a final Stand result.

### Local prediction and verification

For corrected commands, the Phase 26 local model predicts:

- Original actor angular magnitude **0.594618 predicted / 0.591097 actual
  rad/s**. Mean angular-vector error **0.006940 rad/s**, maximum
  **0.018803**; mean relative error against the corrected-minus-teacher velocity
  change is **5.6780%**.
- Penalty actor angular magnitude **0.583016 predicted / 0.580646 actual
  rad/s**. Mean angular-vector error **0.004937 rad/s**, maximum
  **0.013417**; mean relative change-vector error **8.0259%**.
- Mean linear-vector prediction errors **0.000890 / 0.000728 m/s**, maximum
  **0.002137 / 0.002002 m/s**, original/penalty actor. Relative change-vector
  errors are **4.2305% / 7.0108%**.

Relative errors use the small remaining corrected-minus-teacher change as the
denominator, not the original actor's much larger failure. Angular prediction
metrics use the recorded float32 world velocity vectors; the primary speed
comparison uses double-precision native qvel. Their magnitude differences are
below the displayed precision.

Both unchanged actor controls reproduce their saved Phase 25 qpos/qvel, foot
loads, height/tilt, balance, posture, terminal flags, world root velocity vectors
and actual held torques **exactly**. Every case verifies the same reset, exact
two-interval zero-torque prefix and actual delayed torque equal to clipped first
command times the original motor torque scale. Corrected selected torques are
exactly equal to the saved teacher torques. All **644 protected inputs**, **14
sources and their snapshots**, and recorded output hashes verify. No solver
warnings occur. Completion verification adds **zero physics rollouts**.

There are **zero training updates, neural networks loaded or optimizers
created**. No general tests or standalone compilation run. Training configs,
reward, normalizers, action authority, plant, springs and Arena remain unchanged.

### Proposed next continuous three-channel feedback diagnostic; not executed

Determine whether the same three-channel correction remains sufficient once
later commands and observations enter the loop. Use each **frozen Phase 24
actor** in two arms: original deterministic control, and deterministic control
with only `Thigh_L_rx`, `Thigh_R_rx` and `Spine_rx` replaced at **every interval**
by the existing `PredictiveStandController`'s live command. This gives **two
actors x two arms x eight poses = 32 native episodes**, each up to **12 seconds**
or physical FAIL, at most **720 plant control intervals per episode**. No
Gaussian trials, new teacher parameters or training run are included.

The live teacher reads the actual current state and actual pending mixed-command
queue; replaying a saved full-teacher trajectory would give incorrect feedback
after states diverge. Actor observations must contain the actual pending mixed
commands. The other 30 coordinates must remain exactly equal to the actor's
command from that arm's current observation; later commands need not equal the
unassisted trajectory's commands because the states differ. Retain the original
plant, torque scales, two-interval delay, reset poses, physical FAIL handling and
fixed balance/posture gates. Teacher prediction/refinement uses private MuJoCo
states and must be recorded separately from plant intervals.

Verify unchanged model/normalizer tensors and sample counts, protected source
and input hashes, paired resets, first-effect consistency with Phase 27 and
unassisted deterministic replay against the saved Phase 24 rollout. Record
actor-only, teacher-only and mixed commands, actual delayed torques, first loss
and recovery of every physical gate, valid-posture duration, survival, final
two-second Stand window and solver warnings. This is a **teacher-assisted causal
diagnostic**, not learned Stand or an Arena policy. Its result should determine
whether a learning change can focus on these three channels or needs broader
feedback correction. Perform zero optimizer updates and no config, reward,
spring, action-scale or Arena changes. Execute only after the user's approval.



## Phase 28 - Continuous three-channel teacher feedback (2026-10-02)

The user approved Phase 27's proposed feedback diagnostic with **Proceed**.
Execute exactly **two frozen actors x two arms x eight poses = 32 native
episodes**, each up to 12 seconds or physical FAIL. No Gaussian evaluation,
additional intervention group or training run is executed.

**Finding: continuous live correction of the three channels preserves the
first-effect improvement but does not sustain Stand. All 16 assisted episodes
fall, as do all 16 original controls.** The assisted trajectories next lose
balance through angular speed at **0.066672-0.200016 seconds**. Initial channel
attribution does not establish a sufficient whole-trajectory feedback recipe.

### Execution and artifacts

New isolated driver: `mujoco_rig/mimic/audit_stand_three_motor_feedback.py`.
It reuses the existing frozen-policy trial, native torque recording, angular
reward factory and admitted `PredictiveStandController`. Existing code and
training defaults are unchanged.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_three_motor_feedback `
  --out logs/mimickit-training/stand-three-motor-feedback-20261002-01
```

[Report](../../logs/mimickit-training/stand-three-motor-feedback-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-three-motor-feedback-20261002-01/protocol.json),
[offline first-loss command differences](../../logs/mimickit-training/stand-three-motor-feedback-20261002-01/first-loss-command-differences.json)
and [completion verification](../../logs/mimickit-training/stand-three-motor-feedback-20261002-01/completion-verification.json)
preserve exact replays, all gate loss/recovery events, commands, frozen hashes,
normalizer counts and source/input/output hashes. Each arm saves `trace.npz`,
`report.json` and `teacher-private-diagnostics.json`.

At every interval both arms evaluate the frozen deterministic actor and the
existing live teacher. The unassisted arm applies the actor's command. The
assisted arm replaces only `Thigh_L_rx`, `Thigh_R_rx` and `Spine_rx` with live
teacher values; its other 30 commands remain exactly equal to its actor's
proposal. Actor observations are verified against the actual pending mixed
commands. The teacher predicts through that actual queue on private MuJoCo
states; its evaluations are verified not to mutate the plant state, motor
control tensor or queue. Teacher computation in the unassisted arm is recorded
but does not affect its applied commands.

### Closed-loop results

Original-reward actor:

- Original: **0/8 Stand**, mean survival **1.258434 s**, range
  **0.950076-1.400112 s**; mean longest balanced and valid-posture interval
  **0.033336 s**.
- Three-channel assistance: **0/8 Stand**, mean survival **0.914657 s**, range
  **0.533376-1.233432 s**; mean longest balanced and valid-posture interval
  **0.066672 s**, per-pose range **0.050004-0.083340 s**.
- First later balance loss occurs at **0.066672-0.100008 s**. Two poses briefly
  regain all balance gates after that loss; neither maintains a stable window.

Angular-penalty actor:

- Original: **0/8 Stand**, mean survival **1.193846 s**, range
  **0.983412-1.500120 s**; mean longest balanced and valid-posture interval
  **0.033336 s**.
- Three-channel assistance: **0/8 Stand**, mean survival **1.010498 s**, range
  **0.800064-1.400112 s**; mean longest balanced and valid-posture interval
  **0.095841 s**, per-pose range **0.050004-0.183348 s**.
- First later balance loss occurs at **0.066672-0.200016 s**. Six poses briefly
  regain all balance gates after that loss; none maintains a stable window.

Every original command loses balance at **0.050004 s**, exclusively through
angular speed, without subsequent complete balance recovery. Every assisted
case's first later loss is also exclusively angular speed. Posture and other
physical gates fail later; the trace records each separately. Mean total
valid-posture duration is **0.252104 -> 0.314609 s** for the original actor and
**0.308358 -> 0.287523 s** for the penalty actor. These short durations and
brief recoveries are not successful standing. All **32 terminal flags are
physical FAIL**, with zero survivors and zero final two-second Stand windows.
The intervention improves the initial balance interval but worsens mean
survival for both frozen actors.

### Verification and accounting

All unassisted traces replay every saved Phase 24 deterministic trace field
**exactly**, including observations, commands, physical states, rewards and
terminal handling. Every arm's first three physical endpoints replay its Phase
27 counterpart exactly, including root world velocities and actual held
motor torques. The first-effect mean angular speeds remain
**2.911051 -> 0.591097 rad/s** and **2.689001 -> 0.580646 rad/s**.

Actual delayed torques match clipped mixed commands times original scales after
exactly two intervals. All 30 retained coordinates match the live actor exactly,
all three assisted coordinates match the live teacher exactly, and the actual
pending mixed queue matches the actor observation. Paired reset states match
Phase 27. No solver warnings or nonfinite recorded values occur.

Execute **2,101 active plant control intervals**, or **8,404 plant physics
steps**. The teacher performs **2,101 active-world evaluations** and **16,808
private prediction physics steps**, counted separately; its inverse-contact
linearization and forward refinement do not advance time. Private simulated
states are not training transitions. All cases reach physical FAIL before the
720-interval per-episode bound.

Both scorer and portable actor tensor hashes match the Phase 24 frozen snapshots
before and after each arm. Observation normalizer sample counts remain
**290,816 / 294,912**, discriminator normalizer counts **581,632 / 589,824**,
original/penalty actor; action normalizer counts remain zero. Pending samples
remain zero. All **685 protected inputs**, **31 sources and their snapshots**,
and recorded output hashes verify. Completion verification and command-difference
analysis execute zero extra physics episodes.

There are **zero optimizer updates or Gaussian trials**. The existing AMP agent
is instantiated only to load the frozen scorer and actor; no optimizer is
stepped. No general tests or standalone compilation run. Reward, plant, springs,
action authority, training configuration and Arena remain unchanged.

### Interpretation and next proposed conditional correction; not executed

The full-body teacher selects all **33 torques jointly** to track a desired
acceleration. Applying three coordinates from that solution while applying 30
actor coordinates produces a different acceleration. Predicting through the
actual queue handles already committed commands; it does not constrain the
teacher's newly proposed 30 unused torques to match the actor. This experiment
therefore rejects this particular partial application of the full-body teacher
as a sustained controller. It does not establish that three appropriately
chosen torques can never stabilize these trajectories, or identify the remaining
channels' independent causal contributions.

At the first later loss, the largest remaining actor-versus-teacher RMS torque
disagreements are `Thigh_L_rz` **49.032 / 70.659 Nm** and `Thigh_R_rz`
**41.332 / 54.502 Nm**, original/penalty actor. Arm and chest differences also
increase. These compare different proposed joint solutions at the actual mixed
state; torque magnitude alone is not a sensitivity or causal ranking. The local
Phase 26 derivatives at reset are not valid attribution at these later states.

Next proposed diagnostic: compare **the Phase 28 unconstrained three-channel
assistance** against **conditional three-channel assistance**, for both frozen
actors and eight reset poses: **32 native episodes**, each up to 12 seconds or
physical FAIL, at most 720 plant intervals. Replay the unconstrained arm exactly
against Phase 28; reuse its original-actor results as the unassisted reference
without extra control episodes.

In a new isolated controller variant, retain the admitted teacher's desired
acceleration, queue prediction, full inverse-contact map, regularizer, two
forward-refinement iterations and selected-channel bounds. Set both lower and
upper bounds on each of the other 30 channels to the current actor proposal
before the bounded solve and throughout refinement. Only the three selected
channels remain free in [-1,1]. Record the resulting full conditional proposal
and verify its other 30 values equal the actor exactly. Apply the same
three-channel substitution and keep actor observations tied to the actual mixed
queue. This asks whether accounting for the actor's committed torque choices
changes the partial controller's balance result.

Verify paired resets, unassisted-input/source and frozen model/normalizer hashes,
actual delayed torques, all gate loss/recovery events, survival, longest valid
balance/posture interval and final two-second Stand window. Count private teacher
prediction separately from plant steps. No new reward, springs, action scaling,
Gaussian trials, optimizer updates or Arena changes. A failed conditional arm
would reject this restricted controller for the current actors; it would not
prove a general impossibility result. Execute only after the user's approval.


## Phase 29 - Conditional three-channel feedback (2026-10-02)

The user approved Phase 28's proposed conditional-correction diagnostic with
**Proceed**. Execute **two frozen actors x two arms x eight poses = 32 native
episodes**, each up to 12 seconds or physical FAIL, with no additional original
actor, Gaussian or training episodes.

**Finding: conditioning the three-channel solve on the other 30 actual actor
commands improves assisted survival, but all 16 conditional episodes still fail
Stand.** The longest interval satisfying both balance and valid posture in any
conditional case is **0.266688 seconds**. This particular three-channel
controller is insufficient for the current frozen actors. The result does not
prove that all possible three-channel controllers are incapable of stabilization.

### Execution and artifacts

New isolated driver/controller: `mujoco_rig/mimic/audit_stand_conditional_feedback.py`.
Existing Phase 28 instrumentation, native engine, fixed reward wrappers, frozen
policy trial and inverse-contact/forward-refinement helpers are reused unchanged.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_conditional_feedback `
  --out logs/mimickit-training/stand-conditional-feedback-20261002-01
```

[Report](../../logs/mimickit-training/stand-conditional-feedback-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-conditional-feedback-20261002-01/protocol.json),
[comparison](../../logs/mimickit-training/stand-conditional-feedback-20261002-01/comparison.json)
and [completion verification](../../logs/mimickit-training/stand-conditional-feedback-20261002-01/completion-verification.json)
preserve the intervention, per-pose events, exact replays, frozen normalizers,
private solver diagnostics and source/input/output hashes. Each setting stores
`trace.npz`, `report.json` and `teacher-private-diagnostics.json`.

The unconstrained arm repeats Phase 28's three-channel assistance. In the
conditional arm, both lower and upper solver bounds on the other **30 channels**
equal the **current actor proposal**; only `Thigh_L_rx`, `Thigh_R_rx` and
`Spine_rx` remain free in [-1,1]. The same fixed bounds apply during forward
refinement. All **1,332 active conditional solves** verify exact retained
coordinates before and after refinement. The actor is evaluated once per
interval and its proposal is bound to the controller for that interval.

Desired acceleration, one-Hz feedback, the full inverse-contact map, tracking
weights, regularizer, original torque scales, two forward-refinement iterations
and actual two-command queue prediction are unchanged. The conditional controller
retains the same objective; the only optimization difference is the 30 fixed
coordinates. All controller work occurs on private state, with no mutation of
the actual plant or queue. Applied mixed commands and pending observations are
verified against the current actor/controller proposals.

### Results

Original-reward actor:

- Unconstrained assistance: **0/8 Stand**, mean survival **0.914657 s**;
  mean longest balanced and valid-posture interval **0.066672 s**.
- Conditional assistance: **0/8 Stand**, mean survival **1.300104 s**, range
  **1.000080-1.783476 s**; mean longest balanced and valid-posture interval
  **0.195849 s**, per-pose range **0.100008-0.250020 s**.
- Survival improves **42.14%** against unconstrained assistance and **3.31%**
  against the saved unassisted actor's **1.258434 s**.
- First balance loss occurs at **0.116676-0.266688 s**. Horizontal speed is a
  first-loss gate in **5/8** poses, angular speed in **4/8**, including one tie.
  Three poses subsequently regain all balance gates briefly.

Angular-penalty actor:

- Unconstrained assistance: **0/8 Stand**, mean survival **1.010498 s**;
  mean longest balanced and valid-posture interval **0.095841 s**.
- Conditional assistance: **0/8 Stand**, mean survival **1.475118 s**, range
  **0.550044-2.133504 s**; mean longest balanced and valid-posture interval
  **0.206267 s**, per-pose range **0.100008-0.266688 s**.
- Survival improves **45.98%** against unconstrained assistance and **23.56%**
  against the saved unassisted actor's **1.193846 s**.
- First balance loss occurs at **0.116676-0.283356 s**. Horizontal speed is a
  first-loss gate in **7/8** poses, angular speed in **3/8**, and foot support
  in **1/8**; simultaneous gate losses make these counts overlap. Seven poses
  subsequently regain all balance gates briefly.

Thus **12/16 conditional cases first lose balance through horizontal speed**,
compared with exclusively angular-speed first losses in every unconstrained
case. Rotation is better controlled initially, but translation and support are
not fully controlled by this restricted solver. First-effect mean angular speeds
change from **0.591097 to 0.559130 rad/s** and **0.580646 to 0.550679 rad/s**,
original/penalty actor. These first-effect results are new commands, so exact
Phase 27 endpoint reproduction is not required of the conditional arm.

Conditional mean total balanced duration is **0.250020 / 0.383364 s**;
mean longest balanced duration is **0.195849 / 0.293774 s**. Valid posture is
also required: mean total valid-posture duration is **0.272939 / 0.268772 s**,
and every conditional pose first loses posture within **0.233352-0.350028 s**.
Penalty-actor pose 3 briefly sustains balance alone for **0.800064 s**, but its
longest balanced-and-valid-posture interval is only **0.133344 s**. Survival,
balance alone and natural standing must remain separate metrics.

All **32 terminal flags are physical FAIL**, with zero survivors and zero final
two-second Stand passes. The conditional arm improves short-term control under
teacher assistance; neither actor receives training or becomes an autonomous
standing policy.

### Verification and accounting

Both unconstrained arms reproduce **every stored Phase 28 trace field exactly**.
Conditional and unconstrained resets and their initial two-frame zero-torque
physical prefixes are identical. All retained 30 applied commands equal the
live actor proposal, all selected commands equal the live controller, actual
delayed torques match clipped commands times original scales, and pending mixed
commands match actor observations exactly. Conditional full controller proposals
also equal the actor on all fixed coordinates.

There are **2,256 active plant control intervals**, or **9,024 plant physics
steps**, and **18,048 private teacher prediction physics steps**. Controller
inverse-contact linearization and forward refinement do not advance time. No
solver warnings or nonfinite recorded values occur. All episodes terminate
before the 720-interval bound. Reusing saved unassisted reference metrics and
completion verification adds zero physical episodes.

Frozen actor/scorer hashes match Phase 28 before and after each arm. Observation,
discriminator and action normalizer counts are unchanged; pending samples stay
zero. All **734 protected inputs**, **32 sources and their snapshots**, and
recorded output hashes verify. There are **zero optimizer updates or Gaussian
trials**. No general tests or standalone compilation run. Reward, configuration,
plant, springs, motor authority and Arena remain unchanged.

### Decision and next proposed whole-body blend diagnostic; not executed

Do not turn the three-channel reset attribution into a standing training recipe.
Even after correctly accounting for the other actor torques, this controller
cannot maintain horizontal balance and natural posture. Whole-body coordination
must be accounted for before another long training run. This does not identify
an optimal larger free-channel group or justify changing springs.

Next proposed measurement: retain the existing full-body teacher and blend its
**entire 33-channel live command** with each frozen actor's live command:

`executed = teacher + actor_fraction * (actor - teacher)`

Use fixed actor fractions **0, 0.01, 0.05 and 0.10** throughout each episode.
This gives **two actors x four fractions x eight poses = 64 native episodes**,
each up to 12 seconds or physical FAIL, at most 720 plant intervals. No additional
fractions, Gaussian trials, conditional-controller variants or training runs are
included. The zero-fraction arms are full-teacher controls and must reproduce the
saved native teacher trajectory and final Stand outcome exactly. Both zero arms
must also agree physically despite using different frozen reward/scorer snapshots.

The full teacher sees the actual mixed queue and current physical state; the
actor sees the corresponding current observation. Both proposals are bounded,
so their convex blend remains within the original normalized motor limits.
Record full actor/teacher/executed commands, actual held torques, command error
from the live teacher, all physical gate events, survival, longest balanced and
valid-posture interval and the final two-second Stand window. Verify paired resets,
protected sources/inputs, frozen model/normalizer counts and exact delayed control.
Count teacher-private prediction steps separately from plant intervals.

This replaces motor-group attribution with a direct measurement of tolerance to
whole-body actor error along these specific feedback trajectories. Any successful
positive fraction is still **teacher-assisted control**, not learned Stand or
proof of tolerance to arbitrary errors/noise. A failure does not imply that more
optimizer updates or a particular PPO setting will fix it. Perform zero optimizer
updates and no reward, spring, motor-scale, configuration or Arena changes.
Execute only after the user's approval.


## Phase 30 - Whole-body live teacher/actor blend (2026-10-02)

The user approved Phase 29's proposed whole-body blend diagnostic with
**Proceed**. Execute exactly **two frozen actors x four fractions x eight poses
= 64 native episodes**, each up to 12 seconds or physical FAIL. Fractions remain
fixed throughout each episode: **0, 0.01, 0.05 and 0.10**.

**Finding: 0% and 1% actor contribution each pass Stand in 16/16 cases; 5% and
10% each pass in 0/16 cases.** Every 1% episode satisfies balance and valid
posture throughout the full **12.000960-second** trace. This is successful
**99% teacher-assisted control**, not a learned autonomous standing policy.

### Execution and artifacts

New isolated driver: `mujoco_rig/mimic/audit_stand_whole_body_blend.py`. It reuses
Phase 28's mixed-state queue/plant guards, frozen-policy trial, physical recording
and existing full-body teacher. The wrapper blends all 33 coordinates:

`executed = teacher + actor_fraction * (actor - teacher)`

Both commands are bounded, and the executed float32 convex blend remains within
the original normalized motor limits. The teacher predicts through the actual
mixed command queue on private state. The actor sees that same queue in its
current observation. Neither frozen policy nor teacher parameter is changed.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_whole_body_blend `
  --out logs/mimickit-training/stand-whole-body-blend-20261002-01
```

[Report](../../logs/mimickit-training/stand-whole-body-blend-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-whole-body-blend-20261002-01/protocol.json),
[comparison](../../logs/mimickit-training/stand-whole-body-blend-20261002-01/comparison.json)
and [completion verification](../../logs/mimickit-training/stand-whole-body-blend-20261002-01/completion-verification.json)
retain commands, teacher errors, all gate events, per-pose outcomes, exact
teacher replays and protected hashes. Eight setting directories contain detailed
`report.json`, `trace.npz` and `teacher-private-diagnostics.json` files.

The first launch stopped before imports or simulation because of a duplicated
protocol keyword. That metadata field was corrected before source snapshots
were taken; the successful execution includes exactly the approved 64 episodes.
No extra physical episodes result from the correction.

### Native control tolerance

Full-teacher controls, actor fraction 0:

- Original and angular-penalty actor arms each pass **8/8 Stand**, with all gates
  valid throughout **12.000960 s**.
- Both controls reproduce the saved Phase 14 native teacher physical trajectory
  **exactly** and agree with one another physically, including observations,
  commands, qpos/qvel, contact loads, gate/terminal flags, AMP input features,
  and task reward. Their actual held torques also agree exactly with one another
  and with the delayed command-to-torque mapping. Frozen critic/discriminator
  outputs are policy-specific and are not required to match one another.

Actor fraction 0.01:

- Both actors pass **8/8 Stand**, with no balance or posture loss at any interval.
  All episodes end with TIME rather than physical FAIL.
- Mean executed-minus-live-teacher normalized command RMSE is
  **0.000774 / 0.000734**, original/penalty actor; mean per-pose torque RMSE is
  **0.074695 / 0.068506 Nm**.
- Mean per-pose maximum absolute command error is **0.009733 / 0.009865**.
  Aggregate RMSE and largest coordinate errors remain separate metrics.
- First-effect mean angular speed is **0.599243 / 0.596898 rad/s**, versus
  **0.576000 rad/s** for the full teacher.

Actor fraction 0.05:

- Both actors pass **0/8 Stand**, despite **6/8 survivors** each. Mean survival
  is **9.569516 / 9.567432 s**.
- Every pose first loses balance through **pelvis tilt**, original actor at
  **0.233352-0.366696 s**, penalty actor at **0.266688-0.383364 s**.
- Mean longest balanced-and-valid-posture interval is **0.279189 / 0.306274 s**.
  Mean total valid-posture duration is **9.259074 / 9.325746 s**; preserving
  upper-body posture or surviving does not imply full physical balance.
- Mean normalized command RMSE is **0.013178 / 0.012705**, torque RMSE
  **1.172755 / 1.111373 Nm**.

Actor fraction 0.10:

- Both actors pass **0/8 Stand**, with **2/8 / 4/8 survivors** and mean survival
  **4.581616 / 7.077650 s**.
- First balance loss occurs at **0.066672-0.200016 s** for the original actor,
  **0.100008-0.200016 s** for the penalty actor. Across the 16 cases, angular
  speed is the first failing gate in **11**, pelvis tilt in **5**.
- Mean longest balanced-and-valid-posture interval is **0.118759 / 0.139594 s**.
- Mean normalized command RMSE is **0.036685 / 0.030964**, torque RMSE
  **3.493085 / 2.843284 Nm**.

The sampled grid separates successful 1% mixtures from failing 5% mixtures for
these actors, reset poses, teacher and native dynamics. It does not identify an
exact boundary, prove monotonicity at unmeasured fractions or establish tolerance
to arbitrary command errors, simultaneous Gaussian noise or GPU dynamics.
No autonomous actor is newly evaluated, trained or deployed in this phase.

### Feedback error and learning implications

At a fixed state, executed error scales linearly with the chosen actor fraction.
The observed trajectories do not remain at the same states. Mean raw
actor-minus-teacher command RMSE changes from **0.077378 / 0.073362** at the 1%
mixture to **0.263562 / 0.254091** at 5%, and **0.366851 / 0.309641** at 10%.
Larger intervention changes the state distribution and can increase subsequent
actor disagreement. Therefore a fivefold change in actor fraction does not
produce only a fivefold change in measured trajectory error. These means also
cover different live durations because some episodes fail earlier.

No gradient or policy update occurs here. Successful assisted control is a
measurement of the plant/controller trajectory, not evidence that a PPO batch,
learning rate, environment count or training duration will produce a proportional
learning gain. Teacher input remains essential to the successful 1% mixture.

### Verification and accounting

All 64 episodes verify paired resets, identical initial two-frame zero-torque
physical prefixes, exact all-channel blend computation, actual pending commands
in actor observations, original motor bounds and actual delayed torques. The
teacher is verified not to mutate plant state, control tensors or the queue.

Execute **37,821 active plant control intervals**, or **151,284 plant physics
steps**. Teacher prediction performs **302,568 private physics steps**, counted
separately; inverse-contact linearization and forward refinement do not advance
time. No solver warnings or nonfinite recorded values occur. Comparisons and
completion verification add zero physics episodes.

Frozen actor/scorer tensors and every normalizer count match Phase 29 before
and after all settings; pending normalizer samples remain zero. All **785
protected inputs**, **33 sources and their snapshots**, and recorded output
hashes verify. There are **zero optimizer updates or Gaussian trials**. No general
tests or standalone compilation run. Training configuration, reward, plant,
springs, action scaling and Arena remain unchanged.

### Proposed next offline student-error comparison; not executed

Before selecting another training recipe, compare the four already available
frozen neural actors on the newly recorded live teacher-labelled observations:
Phase 7's original imitation actor, Phase 8's correction-imitation actor, and
both Phase 24 PPO actors. This links the earlier small regression errors and
failed autonomous episodes to the current physically measured feedback states.
Do not rerun their old training or alter their normalization.

Use **seven unique Phase 30 trace groups**: one zero-fraction teacher control
(the other is physically identical), plus both actors' 1%, 5% and 10% traces.
Mask inactive worlds. This provides **32,061 recorded live observations** and
**128,244 frozen actor predictions**, four per observation, with **zero simulator
steps or optimizer updates**. All models use their own saved mean/std/clip and
action normalization, verified against their checkpoint/sidecar contracts.

For each actor, trace group and pose, record teacher-command normalized/Nm RMSE,
maximum coordinate error, per-motor errors, upper error quantiles and observation
clipping. Separate reset, first two seconds and later live observations; also
separate balanced/valid-posture observations from departures. Verify each PPO
actor's predictions exactly against its stored proposals on its own traces.
Keep per-pose equal weighting and show live sample counts; do not treat duplicate
zero controls or these same eight poses as independent generalization evidence.

Preserve checkpoint, source, trace and normalizer hashes. Save aggregate metrics
and per-frame predicted commands/error/clipping arrays for reproducibility.
These are counterfactual command predictions on recorded states, not new policy
trajectories or causal clipping ablations. The stable mixture's error measurements
are context, not a universal regression admission threshold. No model fitting,
new physical episodes, PPO setting changes, reward, springs or Arena changes are
included. Execute only after the user's approval.


## Phase 31 - Offline student errors on live teacher-labelled states (2026-10-02)

The user approved Phase 30's proposed offline comparison with **Proceed**.
Compare the four existing frozen actors: Phase 7 original imitation, Phase 8
correction imitation, and both Phase 24 PPO actors. No new actor, simulator,
optimizer, physical episode or training update is created.

**Finding: both imitation actors approximate the pure-teacher trajectory much
better than the PPO actors, but their errors increase sharply on the successful
1% mixture states. The correction-imitation actor retains this error increase
even on entirely unclipped observations.** The recorded states broaden the
measured approximation problem beyond the earlier teacher-only regression
metrics; clipping alone does not account for the remaining discrepancy.

### Execution and artifacts

New isolated driver: `mujoco_rig/mimic/audit_stand_student_errors.py`. It builds
only the saved actor network and normalizers through the pinned upstream network
factory and existing `StandPolicy`. It loads checkpoint tensors directly, with
no environment or AMP agent construction. Clip=10 is the existing inference
setting documented by the training protocols, not a new ablation.

```powershell
.\mujoco_rig\mimic\.venv\Scripts\python.exe -m mujoco_rig.mimic.audit_stand_student_errors `
  --out logs/mimickit-training/stand-student-errors-20261002-01
```

[Report](../../logs/mimickit-training/stand-student-errors-20261002-01/report.json),
[protocol](../../logs/mimickit-training/stand-student-errors-20261002-01/protocol.json),
[compact comparison](../../logs/mimickit-training/stand-student-errors-20261002-01/comparison.json),
[unclipped-observation analysis](../../logs/mimickit-training/stand-student-errors-20261002-01/unclipped-analysis.json)
and [completion verification](../../logs/mimickit-training/stand-student-errors-20261002-01/completion-verification.json)
preserve aggregate/per-pose errors, quantiles, clipping and all protected hashes.
Each of 28 model/trace combinations saves a detailed report and `predictions.npz`
with commands, teacher errors, clipping flags and subset/live masks.

Use seven unique Phase 30 trace groups: one zero-fraction teacher control plus
both actors' 1%, 5% and 10% mixtures. The duplicate zero control is excluded.
There are **32,061 live recorded observations** and **128,244 retained frozen
actor predictions**. Eight-row inference batches match the original PPO kernels;
**33,036 inactive padding outputs** are computed and discarded. Inactive saved
prediction/error/clipping rows are zero and never enter statistics.

Each checkpoint uses its own saved observation/action mean, std and count.
All four actors share the original 306-input, 33-output contract and saved
1024/512 ReLU architecture. Portable checkpoints and AMP sidecars agree exactly.
PPO reconstructed actor tensor hashes match the Phase 30 actors exactly, and all
seven corresponding own-policy trace groups reproduce their stored live actor
proposals **bit for bit**.

### Approximation changes on successful mixed states

Reported values are equal-pose means of normalized-command RMSE, over live
observations within each recorded group. They are not a pooled metric across
different episode lengths or the original full 60-second training dataset metric.

On the pure-teacher 12-second trajectory:

- Original imitation: **0.001556** RMSE; reset-only **0.000652**.
- Correction imitation: **0.002501** RMSE; reset-only **0.001334**.
- Original PPO: **0.083126** RMSE; reset-only **0.044639**.
- Angular-penalty PPO: **0.079556** RMSE; reset-only **0.041201**.

On the two successful 1% actor mixture trajectories, original-PPO/penalty-PPO
trajectory respectively:

- Original imitation RMSE: **0.032894 / 0.032941**. First-two-second errors
  are **0.052221 / 0.060458**.
- Correction imitation RMSE: **0.028499 / 0.029390**. First-two-second errors
  are **0.050158 / 0.059084**.
- Original PPO RMSE: **0.077378 / 0.077431**.
- Angular-penalty PPO RMSE: **0.073195 / 0.073362**.

Thus the supervised actors retain an accuracy advantage on these common states,
but are roughly **11-21 times** worse on the mixed-state groups than on the
pure-teacher group. Reset observations remain identical, so reset errors do not
change across groups. Large errors emerge after feedback evolves.

Mean per-pose largest coordinate errors also increase: original imitation from
**0.114027** on the pure teacher to **0.848488 / 0.896147** on 1% mixtures;
correction imitation from **0.123610** to **0.868767 / 0.904957**. These are
averages of each pose's maximum, not one global maximum. Low overall RMSE does
not eliminate rare large torque-coordinate errors.

The supervised actors' largest per-motor RMSE on the pure-teacher trajectory is
`UpperArm_R_rz`, followed by other right-arm coordinates. On mixed trajectories,
right-arm, hip-yaw and toe/foot errors also increase. These are regression-error
ranks, not physical sensitivity or causal failure attribution. The reset
three-channel attribution from Phase 26 is not a sufficient whole-body loss.

### Clipping and completely unclipped observations

Correction-imitation clipped feature fractions are **0.046637%** on the pure
teacher, **0.043403% / 0.045786%** on the two 1% mixtures. Only
**3.6979% / 4.3229% of frames** have any clipped feature in those mixed groups.
This actor's broader fitted moments therefore largely cover these observations
without clipping, yet its command error remains large.

Condition on frames with **no clipped feature at all**:

- Pure teacher: **5,574 observations**, equal-pose RMSE **0.002412**.
- Original-PPO 1% mixture: **5,547 observations**, RMSE **0.027147**.
- Penalty-PPO 1% mixture: **5,511 observations**, RMSE **0.025259**.

The discrepancy persists at approximately **10-11 times** the pure-teacher
unclipped-frame error. Conditioning selects different observations; this is
not a causal clipping ablation or proof that changing clipping has zero effect.
It shows that non-clipped observations also produce substantial errors.

The original imitation actor clips more often: **1.349911% / 0.988619%** of
features in the successful mixtures, with any clipping in **89.0625% / 67.0486%**
of frames. The distinction between feature fraction and frame fraction matters.
Its entirely unclipped subsets still show RMSE **0.017326 / 0.012271**, versus
**0.001128** on entirely unclipped pure-teacher observations.

Both PPO actors have negligible clipping on these groups but much larger command
errors. In the original-PPO 5% trajectory they have **zero clipped features**,
with RMSE **0.263562 / 0.264035**, original/penalty PPO predictor. Throughput,
GPU utilization and clipping alone therefore do not describe the measured
command-quality gap.

### Partitions and limits

Reset is the first recorded observation. First-two-second membership uses action
issue time, `ending_time - control_timestep < 2`; later observations use issue
time >=2. Pre-command balanced/valid-posture membership comes from the previous
saved ending state. Reset is excluded from this physical partition and reported
separately. Departure observations use the complementary previous-state mask.
Every subset reports live counts and number of populated poses; averages weight
those poses equally. Empty subsets do not contribute fabricated zero errors.

The labels are existing live teacher proposals on the actual mixed-state queue.
Only the original PPO actor's proposal was used in each recorded mixture; the
other predictions are counterfactual. No imitation-policy trajectory is rolled
out. Accuracy ordering does not establish autonomous stability, causal effects
of normalization, tolerance to arbitrary errors or unseen-pose generalization.
The Phase 30 successful mixture's small executed error is useful context, not a
universal neural regression threshold.

### Verification and accounting

All four portable/sidecar pairs and physical/observation contracts verify.
Actor hashes and observation/action normalizer moments/counts are unchanged;
pending samples stay zero. Observation counts are **28,800**, **47,520**,
**290,816** and **294,912**, original imitation, correction imitation, original
PPO and penalty PPO. Action normalizer counts remain zero.

All **868 protected inputs**, **41 sources and their snapshots**, and saved
output hashes verify. There are **zero simulator instances, simulation steps,
optimizers created or training updates**. Verification and unclipped-frame
analysis use saved arrays only. No general tests or standalone compilation run.
Training configuration, reward, plant, springs, action authority and Arena remain
unchanged.

### Proposed next fresh three-minute state-coverage comparison; not executed

Test whether adding the newly measured successful mixed-state teacher labels
improves a fresh student's feedback behavior. This is a controlled supervised
comparison, not another PPO learning-rate sweep or a warm start from a failed
checkpoint. Train **two fresh actors for 180 seconds of optimization each**,
sequentially on the same GPU, then evaluate their final autonomous policies.
No longer training run, extra configuration or Arena promotion is included.

Use the original **28,800 native teacher pairs** and **18,720 admitted native
correction pairs** as the common 47,520-pair pool. The augmented pool adds the
**11,520 recorded observations/live full-teacher proposals** from both successful
1% Phase 30 mixtures. These are supervised action labels on physically valid
assisted states; they are not new admitted full-teacher physical motion clips
and must not be added to AMP demonstration assets.

Initialize both complete actors from the same fresh seed **301010**, with
identical initial tensors and fresh independent Adam states. Fit common
observation moments once on all **59,040 available training observations**
(original pool plus the new mixed states), then use those identical frozen
mean/std/count values and existing clip=10 in both arms. This deliberately gives
both arms the same broader input normalization and isolates the additional
supervised state/action sampling; it does not replicate the earlier Phase 8
normalizer exactly. Do not fit moments on autonomous evaluation observations or
GPU validation labels. Action normalization and 306/33 network remain unchanged.

Keep the existing supervised settings: **Adam 1e-4**, **batch 512**, **gradient
clip 1**, normalized full-33-coordinate command MSE. Control sampling remains
half admitted corrections, quarter original first-two-second teacher pairs and
quarter uniform original teacher pairs. In the augmented arm, allocate **half
of each batch to new mixture pairs** (half early, half uniform; both generating
actors and poses equally sampled), with the other half retaining the control
pool's sampling proportions. Shared budget, initial tensors, normalization,
architecture, optimizer and validation are fixed; sample allocation is the
experimental difference. Record actual updates, sample presentations, timing
and unique data counts rather than equating repeated samples to physics steps.

After each pilot, evaluate exactly **eight native and eight GPU autonomous cases**
for up to **12 seconds or physical FAIL**, with the original plant, reference,
latency and gates: **32 final evaluation cases total** across both actors. Apply
only the neural actor command; no teacher blend, corrective controller or external
support is used during these evaluations. Report survival, longest balanced and
valid-posture interval, final two-second Stand passes and solver warnings, plus
frozen inference errors on the seven Phase 30 groups. Errors on augmented training
states are fitting measurements, not held-out generalization. Native/GPU autonomous
results determine practical progress; lower training RMSE alone does not.

Keep critic/discriminator tensors unchanged from their shared fresh initialization;
perform zero PPO/discriminator updates and preserve demonstration inputs. Verify
checkpoint/sidecar parity, common initialization and normalizer hashes, source
snapshots and protected input hashes. Save both outcomes, including failures,
and do not select or export either to the Arena. This small paired pilot tests
state coverage for these eight poses; it does not establish seed-robust superiority
or a proportional learning-speed law. Execute only after the user's approval.

## Phase 32 — Fresh supervised state-coverage comparison (2026-10-02)

**Status: completed after approval. Neither final actor learned autonomous Stand.**
The Phase 31 proposed comparison above is now executed. Its results do not
justify an hour-long training extension or Arena promotion.

Artifacts: `logs/mimickit-training/stand-state-coverage-20261002-01`.
Driver: `mujoco_rig/mimic/compare_stand_state_coverage.py`.
The root `protocol.json`, `comparison.json`, `report.json`,
`completion-verification.json` and `final-verification.json` preserve the setup,
measurements and verification. Each arm has initial/final portable and AMP
checkpoints, saved Adam state, training curve, native/GPU autonomous traces and
seven frozen-inference groups. `verify_saved_results.py` verifies saved artifacts
without creating a simulator, running inference or optimizing.

### Matched setup and accounting

Both actors started fresh with seed **301010**, identical initial actor tensors
and separate empty Adam states. The full fresh tensor hash is
`a5675cf2c1be6d126fbbb9ac6a04dd5759469ecb30fe24919e1696d20f498ec1`.
Both used the same observation moments fitted once on **59,040** training
observations and then frozen at clip=10. Observation count remained 59,040;
action count and pending counts remained zero. The common normalizer tensor hash
is `9a791b568d5c637a3162a54df921053031a7584b4adc2596ca0bf1d29f070832`.

Architecture, Adam **1e-4**, batch **512**, gradient clip **1**, normalized
full-33-coordinate MSE, plant, reference, latency and evaluation gates were
identical. Control drew 256 admitted corrections, 128 early original teacher
pairs and 128 uniform original teacher pairs per batch. Augmented drew 128
corrections, 64 early original and 64 uniform original teacher pairs, plus 128
early and 128 uniform successful-mixture pairs. Each 128-row added stratum drew
exactly eight samples per generating actor and pose, at randomly sampled times.
Added labels were recorded **full-teacher proposals**, never executed blend
commands. The existing 34 admitted AMP motion clips remained the demonstration
inputs; the mixture labels were not added as AMP motion clips.

Each arm received **180 seconds of optimization**, excluding setup, artifact
writing and evaluation. Actual budgets were **180.000614 / 180.000687 seconds**,
control/augmented. Control performed **102,458 updates / 52,458,496 supervised
sample presentations**; augmented performed **88,611 / 45,368,832**. The augmented
sampling path processed **13.51% fewer updates** in the same wall-time budget.
This is a matched-time comparison, not a matched-update comparison.

All **47,520** available original labelled pairs were sampled by both actors;
augmented also sampled all **11,520** added pairs. Repeated presentations are
re-use of those fixed pairs, not millions of newly simulated transitions. There
were **zero PPO/discriminator updates and zero training rollouts**. Final
evaluation recorded **1,948 live control intervals** across 32 autonomous cases;
these counts exclude GPU integration of inactive worlds.

### Autonomous final results

Each actor received exactly eight native and eight GPU cases, with a 12-second
horizon or physical FAIL. Only the learned actor issued commands through the
unchanged two-step queue. All **32 cases physically failed before two seconds**;
there were **zero survivors, final-two-second balance passes or Stand passes**.

- **Control:** native mean survival **0.943826 s**, GPU **0.939659 s**. Mean
  longest uninterrupted balanced and valid-posture interval **0.418784 s /
  0.420867 s**, native/GPU.
- **Augmented:** native mean survival **1.091754 s**, GPU **1.083420 s**. Mean
  longest uninterrupted balanced and valid-posture interval **0.422950 s /
  0.425034 s**.

Augmentation increased measured mean survival by **15.67% native / 15.30% GPU**,
about **0.15 seconds**. The balanced, natural-posture interval increased only
about **4 milliseconds**. Survival is not time spent successfully standing.
The two backends agree on the overall outcome. No native live-world
BADQPOS/BADQVEL/BADQACC warning was recorded. GPU solver warnings are
**unmeasured**, not verified absent; the inherited GPU trace's false warning
field is explicitly marked as a placeholder in its reports.

### Frozen command regression

Both final actors were evaluated on the same seven saved Phase 30 groups:
**32,061 live observations per actor / 64,122 live predictions total**. All
normalizer tensors and counts remained frozen. Clipping fractions on each saved
group are identical between the two predictors, as expected from the shared
normalizer and observations. The following RMSE values are normalized motor
commands, averaged equally across populated poses, control -> augmented:

- Pure teacher: **0.001946 -> 0.002489**.
- Successful 1% mixture from original PPO: **0.020463 -> 0.002056**.
- Successful 1% mixture from angular-penalty PPO: **0.024370 -> 0.001172**.
- Original PPO 5% mixture: **0.144411 -> 0.192975**.
- Angular-penalty PPO 5% mixture: **0.162024 -> 0.202213**.
- Original PPO 10% mixture: **0.301152 -> 0.316303**.
- Angular-penalty PPO 10% mixture: **0.249855 -> 0.265657**.

Added-state errors improved roughly **10x / 21x**, but those are fitting
measurements: augmented trained on these observations and labels. Later-than-two-
second errors in those groups improved **0.012116 -> 0.001798** and
**0.010707 -> 0.000644**, on frames with zero clipped features for both predictors.
That measured regression improvement does not depend on changing normalization.

Errors increased on the pure-teacher and all four more-departed 5%/10% groups.
Those latter groups were excluded from both training pools. Their results do not
show broader recovery generalization. This remains a single-seed, final-checkpoint
comparison over the same eight reference poses, not proof of seed-robust
superiority, an arbitrary-error tolerance or a proportional learning-speed law.

### Interpretation and verification

The new labels substantially improved fitting in the narrow successful assisted
state distribution. They did not establish a stable autonomous feedback policy.
The observed survival increase is small relative to the 12-second goal, and
balanced natural posture remains near 0.42 seconds. More accurate commands on
recorded successful states do not certify corrective commands once the actor's
own errors change the state and queued actions. The worse 5%/10% regression
results support investigating that coverage gap, but do not identify the first
causal autonomous error: these final autonomous traces do not store complete
observations or simultaneous teacher proposals.

All **981 protected inputs**, **43 source/snapshot pairs** and **114 initially
saved output hashes** verify. All six saved portable/AMP checkpoint pairs match.
Both initial full tensor states equal the common normalized fresh state;
critic/discriminator tensors remain equal to the original fresh initialization;
normalizers equal the common frozen state. Saved Adam step counters match the
reported update counts. Offline verification recomputed survival durations,
longest valid intervals, physical FAIL outcomes, prediction bounds and live
sample accounting from saved arrays. It generated no new simulations, forward
passes or training updates.

No general tests or standalone compilation were run. No longer training run,
extra configuration, springs/stiffness change, export or Arena update occurred.

### Proposed next bounded autonomous-state capture; not executed

Before spending more optimization time, locate the feedback error on the final
actors' own trajectories. Freeze both Phase 32 final checkpoints and their
normalizers. Capture exactly **eight native and eight GPU autonomous cases per
actor**, up to **12 seconds or physical FAIL**: 32 diagnostic cases total.
Save complete pre-command observations, actual pending queue, actor commands,
physical state/gates and simultaneous full-teacher proposals queried on that
same state. Execute only the actor's command; querying the teacher must not
alter the executed action, engine, queue or reference.

Report each pose's first balanced/posture departure, teacher-command error
before that departure, subsequent error growth, motor/observation-feature
contributions and clipping. Compare actual autonomous-state coverage against
the fixed original and successful-mixture training pools. Save teacher proposals
as diagnostic labels with physical-validity/warning masks; do not automatically
admit them as successful recovery samples or AMP clips. Verify frozen tensors,
normalizers, checkpoint/sidecar parity and historical source/input hashes.

This proposed capture includes no optimizer, new training budget, controller
blend, checkpoint selection, export or Arena change. It would determine whether
to design iterative student-state supervision or a different objective; neither
is authorized or executed in Phase 32. Execute the capture only after approval.

## Phase 33 — Frozen autonomous-state and teacher-label capture (2026-10-02)

**Status: completed after approval. The feedback error grows before balance is
lost; added successful-mixture states barely cover the actors' own drift.**
The Phase 32 proposed capture is now executed. No training or recovery-data
admission occurred.

Artifacts: `logs/mimickit-training/stand-student-states-20261002-01`.
Driver: `mujoco_rig/mimic/capture_stand_student_states.py`.
Each of the four settings has a complete `trace.npz`, `capture-report.json`,
`analysis.json`, `diagnostics.npz` and private teacher diagnostics. Root
`comparison.json`, `prefix-analysis.json`, `report.json` and
`final-verification.json` preserve the comparison and verification. The saved
`verify_and_summarize.py` operates on existing arrays/checkpoints only.

### Executed scope and labels

Both final Phase 32 actors were frozen with their own saved normalization and
evaluated on **eight native and eight GPU cases each**, up to **12 seconds or
physical FAIL**: exactly **32 autonomous cases**. Native inference used CPU;
GPU inference used the GPU. Only bounded deterministic actor commands were
executed, with zeros for inactive cases. No teacher command was substituted or
blended into the plant.

At each pre-command state, the unchanged full controller queried private native
MuJoCo data initialized from that world's actual q/v. GPU states used the
existing world-to-local free-joint angular-velocity conversion. It predicted
through the **actual two pending actor commands**, then used the existing
inverse-contact objective and two forward refinements. Per-query guards verify
unchanged live q/v, held controls, queue/cursor and complete policy observations.
Observed pending controls and actual delayed torques also verify exactly.

Capture recorded **1,956 live observations / teacher queries**, with **zero
query failures and zero private BADQPOS/BADQVEL/BADQACC warnings**. All 1,956
proposals were finite, bounded and met the recorded numerical-validity mask.
That mask does not certify recovery: these are diagnostic proposals, not newly
admitted successful teacher actions or AMP motion clips. Native plant warnings
were monitored and absent; GPU plant solver warnings remain unmeasured.

There were **7,824 live plant physics steps** and **15,648 private teacher
prediction physics steps**, four and eight per live observation respectively.
The latter advance private temporary states only; forward refinements do not
advance time. Live-step counts exclude GPU integration of inactive worlds.
No learning optimizer was created; there were **zero training updates and zero
new normalizer samples**. Observation counts stayed **59,040**, action and
pending counts zero.

### Trajectory replay and terminal results

Both native trajectories exactly replayed all 13 comparable Phase 32 fields,
including commands, physical measurements, gates, done/alive masks and clipping.
Teacher queries therefore preserved the recorded autonomous behavior exactly on
native MuJoCo. All **32 captures physically failed**, with **zero Stand passes**.

Mean survival, control -> augmented:

- Native: **0.943826 -> 1.091754 seconds**, exact Phase 32 replay.
- GPU: **0.947993 -> 1.091754 seconds** in this capture.

The GPU captures are **not bit-exact Phase 32 replays**. Initial actor commands
differ by at most **2.980232e-8**, initial pelvis heights match, and later
trajectories diverge. The largest per-case survival change from Phase 32 is
**0.066672 s control / 0.083340 s augmented**; shared-live command differences
can become large later. Read-only teacher guards passed, but these observations
do not establish the cause of GPU replay differences. GPU measurements here
describe these captured trajectories; do not silently substitute them for the
Phase 32 traces or claim GPU bit reproducibility.

### Where balance is first lost

The first combined balance/posture departure is a **balance departure in every
case**; natural-posture failure comes later. First balance-loss ranges:

- Control native **0.350028–0.566712 s**, GPU **0.350028–0.533376 s**.
- Augmented native **0.333360–0.533376 s**, GPU **0.333360–0.533376 s**.

Across 32 cases, root angular speed exceeds the **1 rad/s** gate at first
departure in **28 cases**. Horizontal speed exceeds **0.2 m/s** in **five**;
one augmented native case violates both gates together. Falling below the
physical root-height threshold is later, not the first Stand-gate failure.

The analysis separates observation issue time from ending-state gate time and
identifies which actor command was already applied. A command issued in the
departure interval is still queued; the held torque comes from **two decisions
earlier**. Motor-error measurements are differences from the unexecuted teacher
proposal, not causal actuator interventions.

### Command error is already large before departure

Normalized full-33-command RMSE, averaged equally across poses,
control -> augmented:

- Reset, native: **0.000987 -> 0.001271**.
- First 50 ms of issued commands, native: **0.003052 -> 0.003499**.
- Commands already applied by first balance departure, native:
  **0.121126 -> 0.167897**; GPU **0.115158 -> 0.162007**.
- Commands issued before the first ending-state departure, native:
  **0.140450 -> 0.191687**; GPU **0.137567 -> 0.182715**.
- Commands issued after that departure, native: **0.491833 -> 0.557817**;
  GPU **0.485750 -> 0.550710**.

For already-applied commands, corresponding native full-coordinate torque RMSE
is **10.49 -> 15.96 Nm**, GPU **10.00 -> 15.40 Nm**. The largest equal-pose
squared torque-error contributions in both actors/backends are **Thigh_R_rz**
and **Thigh_L_rz**. Control next emphasizes **UpperArm_R_rz / Chest_rz**;
augmented next emphasizes **Thigh_R_rx / Chest_rz / Spine_rz**. These rankings
identify proposal disagreement, not proof that replacing those motors causes
recovery.

The augmented actor's slightly longer survival coexists with **larger command
error before balance loss**. Phase 32's 10–21x improvement on recorded mixture
states does not transfer to these autonomous states. Survival time and fitting
RMSE therefore cannot be used interchangeably as progress toward learned Stand.

### State coverage and the early clipping prefix

Coverage compares each observation with the nearest **same-pose** observation
in the original 47,520-pair pool, the added 11,520-pair pool, and their union.
Distance is RMS over the **161 dynamic features**, using the unchanged common
mean/std **without clipping**; constant future-reference features are excluded.
This is a descriptive distance, not a universal out-of-distribution threshold
or a causal feature attribution. Per-feature squared-distance and clipping
contributions are saved for each pose and temporal subset.

Native mean nearest-state distance to the original -> combined pool:

- First 50 ms: control **0.026934 -> 0.025647**, augmented
  **0.028333 -> 0.027201**.
- Before first departure: control **3.645606 -> 3.617528**, augmented
  **7.046481 -> 7.012253**.
- After first departure: control **140.932226 -> 140.931604**, augmented
  **147.930612 -> 147.930227**.

Adding the successful-mixture pool reduces this measured pre-departure distance
by **less than 1%** for both autonomous actors. GPU measurements show the same
pattern. Many dominant discrepancies are hinge/root velocities, including toe
velocity coordinates with narrow training variance. The new successful mixture
observations cover a narrow assisted distribution rather than these later
autonomous feedback states.

Clipping first appears at **issue time 0.016668 s in all 32 cases**, notably
the **Toe_L_rx / Toe_R_rx velocity features** (observation indices 84/92).
This precedes the first neural torque's ending-state effect at **0.050004 s**.
Crucially, both native actors' **first three pre-command q/v states, physical
observations 0:95 and full clipping masks exactly equal the admitted original
teacher data**. Their pending-command observations differ, because the actor
already issued its own proposals. Thus early clipping also occurs in the
training examples and does not by itself locate the feedback failure. This
capture does not isolate clipping's causal effect on subsequent predictions.

### Verification and conclusion

All **1,145 protected inputs**, **46 source/snapshot pairs** and **73 initially
saved output hashes** verify. The final manifest binds **77 saved files**,
including offline analysis and verification source. Checkpoint/sidecar parity,
frozen actor tensors, full checkpoint hashes and normalizer counts verify.
Saved arrays independently verify 32 physical FAIL outcomes, departure times,
issued/applied command alignment, bounds, label masks and episode accounting.
Offline verification creates no simulator, advances no physics, runs no forward
passes and performs no training.

No general tests or standalone compilation were run. Training configuration,
plant, springs, action authority and Arena remain unchanged. The measured gap
now concerns teacher-response accuracy on the actors' own valid-but-drifting
states. Whether the existing teacher can actually recover those states with
their committed queue is still unmeasured; finite proposals alone do not answer
that question.

### Proposed next bounded teacher-recovery admission; not executed

Validate that the new diagnostic targets correspond to viable recovery before
using them for more supervised training. For each actor/pose/backend, select
the **last recorded pre-command state which is balanced and posture-valid
before its first departure**. Preserve that state's reference and both pending
actor commands in oldest-first order. Start one independent recovery evaluation
from its saved q/v and queue, then execute only the unchanged full teacher.

Run exactly **eight native and eight GPU recovery starts per actor**, up to
**12 seconds or physical FAIL**: 32 cases total. Use each backend's own captured
states. Restoring q/v resets solver/contact history; report initial measurements
and their differences from the source instead of claiming an exact continuation
of the original solver state. Do not clear the committed actor queue to make
recovery easier. The controller retains its existing objective, prediction,
authority and refinements; no spring, support or reward change is included.

Measure survival, final-two-second balance/posture Stand gates, full recovery
trajectories and solver warnings. Bind each result to the source state, actor,
pose and queue. Preserve successful and failed outcomes; admit recovery samples
only under the existing numerical/physical/provenance requirements, and never
automatically turn all finite Phase 33 proposals into successful motion clips.
Keep final actors/checkpoints/normalizers frozen and verify historical inputs
and source snapshots.

This proposed admission evaluates teacher feasibility, with **zero learning
optimizers or training updates**. It includes no longer training run, extra
starts, parameter sweep, export or Arena promotion. Successful recoveries would
support a subsequent matched fresh student-state-supervision pilot; that pilot
is a separate decision and is not authorized here. Execute the recovery
evaluation only after approval.

## Phase 34 — Teacher recovery from last valid student states (2026-10-02)

**Status: completed after approval. The unchanged teacher recovers 1/32 starts;
31 physically fail. Late finite teacher proposals are usually not backed by a
successful recovery in this experiment.** The proposed Phase 33 admission is
now executed. This is teacher feasibility, not newly learned actor Stand.

Artifacts: `logs/mimickit-training/stand-student-recovery-20261002-01`.
Driver: `mujoco_rig/mimic/admit_stand_student_recovery.py`.
Each setting preserves `source-states.npz`, `restoration.json`, full recovery
`trace.npz`, private teacher diagnostics and `report.json`. Root `comparison.json`,
`admission-metadata.json`, `report.json` and `final-verification.json` preserve
the admission, limits and verification. `verify_saved_results.py` verifies
existing arrays/checkpoints only.

### Starts, restoration and committed controls

Exactly **eight native and eight GPU cases per final Phase 32 actor** were run:
32 recovery starts, each up to **12 seconds or physical FAIL**. Each start was
the last live, balanced and posture-valid pre-command state at or before its
first ending-state departure in Phase 33. Selected source issue times span
**0.316692–0.550044 seconds**. Every source frame, pose, trace hash, q/v,
reference goal, pending queue and previous held torque is bound to its result.

Restoration preserved q/v and both committed actor commands, in oldest-first
order, plus the previous held torque. Native q/v were restored **bit-exactly**;
GPU qpos was exact, with maximum qvel differences **1.973517e-7 control /
1.203089e-7 augmented** from free-joint conversion/rounding. Reference goals and
pending-command observations verify exactly.

Solver/contact history was reset. The initial contact snapshot was recomputed
under the restored held torque, without advancing time. Initial foot-load
differences from the recorded source reach **16.220 / 11.567 N** for control
native/GPU and **133.046 / 133.167 N** for augmented native/GPU. Native initial
observation differences affect only foot-load indices 93/94; GPU also has small
world-angular-velocity differences at 12:15. Nevertheless **all 32 restored
starts still pass both balance and natural-posture gates**. These are restored
recovery starts, not exact continuations of the original contact/solver history.

Only the unchanged full teacher issued new commands. The **first two held
torques remained the saved actor commands**; new teacher commands first affect
the third interval. Queue observations, issued commands and delayed held
torques verify exactly. Per-query guards verify unchanged live q/v, controls,
queue/cursor and observations during private teacher computation. The existing
one-Hz objective, authority, inverse-contact solve and two refinements were
unchanged; there was no spring, support, reward or controller-parameter change.

### Outcomes

- **Control native:** **1/8 survives and passes Stand**, pose **7**. The other
  seven physically fail. Mean duration **3.358602 s**, range
  **1.250100–12.000960 s**.
- **Control GPU:** **0/8** survives/passes. Mean duration **2.264764 s**, range
  **1.316772–4.417020 s**.
- **Augmented native:** **0/8** survives/passes. Mean duration **1.516788 s**,
  range **0.866736–2.216844 s**.
- **Augmented GPU:** **0/8** survives/passes. Mean duration **1.887651 s**,
  range **0.866736–2.700216 s**.

All 32 lose a balance/posture gate at the **first ending interval, 0.016668 s**,
while the teacher's first issued command is still queued. The successful native
case recovers after **0.050004 s of balance violations**, has no posture
violation, remains balanced/posture-valid for **11.950956 s**, and passes the
final two-second Stand window. There were **zero controller-command errors,
zero private prediction warnings and zero native plant warnings**. GPU plant
solver warnings remain unmeasured. The 31 failures are physical, not rejected
labels or numerical solver errors.

For all 16 native starts, the first recovering teacher proposal exactly equals
the Phase 33 unexecuted proposal at that source state. Thus the finite labels
were reproduced even when their subsequent recovery failed. Most GPU first
proposals differ only slightly; one control-pose-4 difference reaches
**0.010396 normalized command** despite tiny qvel restoration differences.
Do not infer a causal numerical-sensitivity law from this single comparison.

GPU starts use their own captured states and queues, not the identical native
starts. The native/GPU outcome difference is therefore not an isolated backend
comparison. No GPU start produced a successful validation recovery here.

### Admission and accounting

Only control-native pose 7 meets the complete physical/numerical admission:
TIME at the full horizon, final two seconds valid, finite observations/q/v,
bounded commands and no native/private warning. Its **720 decision records**
are saved as an **isolated native supervised candidate** in
`control-native/admitted-dataset.npz`, with source/license, trace and dataset
hashes in `admission-metadata.json`. The other 31 cases contribute **zero
admitted records**. GPU admitted validation records: **zero**.

Dataset `action`, `executed_action` and `teacher_executed` describe **newly
issued teacher commands**, not the already-held delayed torque. The first two
physical intervals retain committed actor commands; `held_torque_nm` records
that distinction. The dataset has not been merged into training. No failed
recovery is promoted to supervision, and no Phase 33 finite proposal is
automatically admitted. **Zero new AMP clips** were created; original motion
demonstrations and their selection remain unchanged.

The evaluations recorded **4,333 live control intervals**, **17,332 live plant
physics steps**, and **34,664 private teacher prediction steps**. Private
refinements do not advance time. GPU inactive-world integration is excluded
from these live counts. Frozen actors were not called: **zero neural forward
passes, zero learning optimizers, zero training updates and zero new normalizer
samples**. Actor checkpoint/sidecar tensors and normalization remain unchanged.

### Verification and interpretation

All **1,271 protected inputs**, **48 source/snapshot pairs**, **80 initially
saved output hashes** and **84 final manifest files** verify. Offline checks
recompute source selection, restored state/reference/queue binding, the first
two actor torques, subsequent teacher delay, terminal flags, final-window gates,
numerical admission and dataset equality to the accepted trace. They verify
frozen full checkpoint hashes, portable/sidecar parity and observation/action
counts **59,040 / 0**, without new simulation or inference. No general tests
or standalone compilation were run. Arena and production training configuration
remain unchanged.

The last state which passes our Stand gates is usually already outside this
teacher's demonstrated recovery region **with its committed queue**. This does
not prove those states are physically irrecoverable or that another controller
could not recover. It does show why treating every finite late-state proposal
as a successful expert target would be unsupported. One admitted pose and no
GPU recovery validation do not justify another long training run or an
all-pose student-state-supervision claim.

### Proposed next earlier teacher-recovery comparison; not executed

Repeat the same bounded feasibility evaluation with an **earlier fixed source
frame 6**, issue time **0.100008 seconds**, from each Phase 33 captured actor,
pose and backend. Saved arrays already show that all 32 such source states are
live, balanced, posture-valid and have numerically valid teacher proposals;
their maximum measured root angular speed is **0.344 rad/s**. This is source
inspection only; no earlier recovery has been simulated yet.

Run exactly **eight native and eight GPU starts per actor**, up to **12 seconds
or physical FAIL**: 32 cases. Preserve each earlier source's q/v, original
reference, held torque and both pending actor commands. Use the same restoration,
initial-difference reporting, unchanged controller, physical/numerical admission
and separate native training-candidate/GPU-validation handling as Phase 34.
Bind the earlier source time and results explicitly; no empty-queue reset or
automatic retry at a different time is included.

Compare teacher recovery rates with Phase 34's last-valid starts. This tests
whether earlier intervention supplies a broader feasible supervision region,
before allocating more student optimization. Keep actors/normalizers frozen,
perform zero neural forward passes, learning updates or optimizer creation,
and preserve protected source/input hashes. There is no parameter sweep, extra
start, AMP clip creation, training extension, export or Arena promotion.
Any subsequent fresh three-minute supervised pilot requires a separate approved
plan based on the measured admission. Execute only after approval.

## Phase 35 — Earlier teacher recovery at 0.100008 seconds (2026-10-02)

**Status: completed after approval. The unchanged teacher passes 30/32 earlier
starts, compared with 1/32 last-valid starts in Phase 34.** This measures teacher
feasibility on restored student states, not improvement of a learned actor.

Artifacts: `logs/mimickit-training/stand-early-recovery-20261002-01`.
Driver: `mujoco_rig/mimic/admit_stand_early_recovery.py`.
The new wrapper reuses the sealed Phase 34 restoration, simulation, teacher,
queue, gates and admission implementation. It changes only source selection and
provenance. The historical implementation and artifacts remain hash-protected.
Each setting saves source states, restoration measurements, complete recovery
arrays, controller diagnostics and a report. Root `comparison.json` binds the
paired early/late outcomes; `admission-metadata.json` binds successful datasets.

### Starts and restoration

Exactly **eight native and eight GPU cases per frozen Phase 32 actor** were
evaluated: 32 starts, up to **12 seconds or physical FAIL**. Each uses its own
Phase 33 pre-command **frame 6**, issue time **0.100008 seconds**. All source
states are live, balanced, posture-valid and have numerically valid proposals.
No alternate source time or extra start was attempted.

Each start preserves q/v, reference goals, previous held torque and both pending
actor commands in oldest-first order. All 32 restored starts pass balance and
posture gates. Native q/v and GPU qpos restore exactly. Maximum GPU qvel
differences are **5.966817e-8 control / 5.962655e-8 augmented**. The reference and
queue observation fields verify exactly. Native initial observation differences
affect only foot-load indices 93/94; GPU also differs slightly at root angular
velocity indices 12:15.

Contact/solver history is reset and forward contacts are recomputed under the
saved held torque without advancing time. Maximum initial foot-load differences
are **4.141 / 4.135 N** for control native/GPU and **17.176 / 4.073 N** for
augmented native/GPU. These are restored recoveries, not exact continuations of
the original solver history. Both first physical intervals use the committed
actor torques; the teacher's first new command takes effect in interval three,
at recovery ending time **0.050004 seconds**. No queue clearing, spring, support,
reward, action-authority or controller change was made.

### Outcomes and comparison with Phase 34

- **Control native: 8/8 Stand**, previously **1/8**. All survive **12.000960 s**.
  Pose 4 briefly violates balance for **0.066672 s**, first at **1.700136 s**, then
  remains balanced/posture-valid for **10.250820 s**. Other poses never violate
  either gate.
- **Control GPU: 8/8 Stand**, previously **0/8**. All survive **12.000960 s** and
  never violate either gate.
- **Augmented native: 6/8 Stand**, previously **0/8**. Accepted poses are
  **0, 1, 2, 5, 6, 7**, without a gate violation. Pose **3** first loses a gate at
  **8.917380 s**, physically failing at **11.900952 s**. Pose **4** first loses a
  gate at **2.566872 s**, physically failing at **3.600288 s**. Mean duration is
  **10.938375 s**, range **3.600288–12.000960 s**.
- **Augmented GPU: 8/8 Stand**, previously **0/8**. All survive **12.000960 s**
  and never violate either gate.

There are **29 newly successful actor/pose/backend cases**, with no loss of the
one previously successful case. Overall teacher success increases from
**3.125% to 93.75%**. These are two specified intervention times; they do not
establish an optimal handoff time or a universal recovery boundary. The earlier
and later starts differ in state, queued commands and reset contact snapshot.
GPU uses its own captured states, so native/GPU differences are not isolated
same-state backend effects.

All 16 first native teacher proposals reproduce the Phase 33 proposal exactly;
maximum GPU difference is **2.980232e-8 normalized command**. There are **zero
controller-command errors, zero private prediction warnings and zero native
plant warnings**. GPU plant solver warnings remain unmeasured. Both rejected
cases fail physically despite finite, warning-free native/private computations.

### Admission, accounting and verification

Accepted native data total **10,080 decision records**: **5,760** from all eight
control poses and **4,320** from six augmented poses. Separate successful GPU
validation data total **11,520 records**. The two failed native cases contribute
zero admitted records. Datasets remain isolated artifacts; nothing was merged
into production training, and no new AMP clips were created. Native sources
retain the existing 100STYLE/CC-BY-4.0 attribution and source hashes.

Dataset `action`, `executed_action` and `teacher_executed` mean the newly issued
teacher command. The first two held torques remain the committed actor commands;
the full trace records actual `held_torque_nm`. Admission still requires TIME
at the full horizon, valid final-two-second balance/posture, finite live states
and observations, bounded commands and native/private warning checks. Successful
GPU data are validation only, excluded from the native training pool.

The experiment records **22,530 live control intervals**, **90,120 live plant
physics steps** and **180,240 private teacher prediction steps**. These counts
exclude inactive GPU worlds; private refinements do not advance time. There
were **zero neural forward passes, zero learning optimizers or updates, zero
new normalizer samples and zero new AMP clips**. Arena and production training
configuration remain unchanged.

Offline saved-array verification passes for **1,357 protected inputs**, **49
source/snapshot pairs**, **85 initial output hashes** and **88 final manifest
files**. Checks reproduce frame-six source binding, q/v/reference/queue
restoration, committed torques, subsequent teacher delay, terminal flags,
final-window gates, admitted dataset equality, frozen checkpoint tensor hashes,
portable/sidecar parity and observation/action counts **59,040 / 0**. Phase 34
files remain unchanged. Verification adds no simulation, inference or optimizer.
No general tests or standalone compilation were run.

The teacher can keep most of these mildly drifting student states standing
when it intervenes early, with the actual committed queue. The late-state
failures therefore do not imply that the teacher is unusable for student-state
supervision. The remaining question is whether supervised learning from these
successful branches reduces the student's autonomous drift before it leaves
that demonstrated region.

### Proposed next fresh three-minute student pilot; not executed

Compare two **fresh, identically initialized actors**, each with a separate empty
Adam state and **180 seconds of supervised optimization**. Use seed **301011**,
the existing actor architecture, learning rate **1e-4**, batch **512**, gradient
clip **1**, normalized full-33-coordinate MSE and the shared **frozen Phase 32
observation/action normalizer** (counts **59,040 / 0**, observation clip **10**).
Do not resume either trained actor or refit normalization. Verify common initial
weights and normalizer hashes; preserve critic/discriminator parameters.

Use the original native teacher/correction pools as the matched baseline. The
control batch contains **256 correction**, **128 original early-teacher** and
**128 original uniform-teacher** samples. The recovery-augmented batch contains
**128 correction**, **64 original early-teacher**, **64 original uniform-teacher**,
**128 early recovery** and **128 uniform recovery** samples. Early sampling uses
the first **120 control frames**, matching Phase 32's two-second stratum. For
recovery sampling, choose uniformly among the **14 admitted native generating
actor/pose cases**, then uniformly among eligible frames, with replacement.
Use all 10,080 accepted native records; exclude failed cases and all GPU records.
Do not add Phase 30 mixture data or Phase 34 late-recovery data to either arm.

Record actual wall time, optimizer updates, sample presentations, unique coverage,
normalization clipping and teacher-command error on both native training pools
and separate GPU validation pools. Fixed wall time can yield different update
counts; report them rather than assuming identical optimization budgets. After
each arm, evaluate exactly **eight native and eight GPU autonomous starts**,
up to **12 seconds or physical FAIL**, with the same original eight poses and
Stand gates: **32 autonomous cases total**. Report first departure, survival,
posture and final Stand, including failures. Bind checkpoints, datasets and
protected historical inputs; verify saved results and actor/normalizer integrity.

This is an isolated supervised transfer pilot, with no PPO or discriminator
update, controller deployment, new AMP clips, configuration sweep, extra starts,
longer run, export or Arena promotion. A teacher recovery pass does not count as
a learned actor pass. Execute the pilot only after approval.

## Phase 36 — Fresh students with earlier recovery supervision (2026-10-02)

**Status: completed after approval. Both three-minute arms produce 0/16 learned
Stand passes. Recovery augmentation improves fitting on accepted native recovery
states by 20.31 times, without improving autonomous balance or survival materially.**
Phase 35's 30/32 teacher passes remain teacher feasibility results; they did not
become learned actor passes in this experiment.

Artifacts: `logs/mimickit-training/stand-recovery-supervision-20261002-01`.
Driver: `mujoco_rig/mimic/compare_stand_recovery_supervision.py`.
Each arm preserves fresh initialization, final portable/full checkpoints, its
separate imitation optimizer, training curve, sample accounting, eight native
and eight GPU autonomous traces, and recorded-state predictions. Root
`comparison.json` and `final-verification.json` bind the saved outcomes.

### Fair initialization and fixed budgets

Both arms start from **fresh identical weights**, seed **301011**, with separate
empty Adam states. The fresh full tensor hash is
`627738dde1a4250390662ac799bcad04f1023719e0c845f14130d79ba397f06e`.
The common actor hash is
`e0ccd90d1656b0db92719dcf9c5886fd07f7c515e50c3cd7c0f1b90426ac3a21`.
Only the Phase 32 observation/action normalizer tensors are copied: counts
**59,040 / 0**, clip **10**, hash
`9a791b568d5c637a3162a54df921053031a7584b4adc2596ca0bf1d29f070832`.
All other common tensors equal the new fresh initialization exactly. No trained
actor weights or optimizer are resumed, and no normalization moments are refit.

Both use the existing actor architecture, **Adam 1e-4**, **512 samples per
update**, gradient clip **1**, and normalized full-33-coordinate MSE. Only actor
layers and the action-mean network are optimized. Critic/discriminator tensors
and normalizers remain unchanged.

- **Control:** **180.001491 seconds**, **101,992 updates**, **52,219,904 sample
  presentations**. Each batch has 256 original correction, 128 original early
  teacher and 128 original uniform teacher samples. All **47,520** available
  original pairs are presented; no recovery pair is used.
- **Recovery augmented:** **180.000950 seconds**, **91,298 updates**,
  **46,744,576 sample presentations**. Each batch has 128 correction, 64 original
  early teacher, 64 original uniform teacher, 128 early recovery and 128 uniform
  recovery samples. All **57,600** available native pairs are presented,
  including all **10,080** accepted Phase 35 recovery records.

Early sampling uses the first **120 frames**. Recovery sampling chooses uniformly
with replacement among **14 admitted native generating actor/pose cases**, then
among the eligible frames. Failed cases, GPU data, Phase 30 mixture labels and
Phase 34 late recovery labels are excluded from optimization. The normalizer
retains its previously fitted moments; retaining those moments does not mean
Phase 30 labels are sampled by either optimizer.

The augmented sampler completes **10.49% fewer updates** within the equal wall
time. This is a fixed-time comparison, not equal update counts. Total repeated
supervised presentations are **98,964,480**, with **zero new training physics
transitions**. Repeated label presentations are not PPO environment transitions
or independent motion trials. There are no PPO/discriminator updates, new AMP
clips, authority, stiffness, spring, controller or reward changes.

### Autonomous results

Exactly **eight native and eight GPU original-pose starts per final actor** were
evaluated, up to **12 seconds or physical FAIL**: 32 cases total. The final
elapsed-time checkpoints are used without best-checkpoint selection, teacher
intervention or external support.

- **Control native: 0/8 Stand**, mean survival **0.872986 s**, range
  **0.750060–1.083420 s**. Mean first balance/posture departure **0.400032 s**;
  mean longest valid balanced posture **0.383364 s**.
- **Control GPU: 0/8 Stand**, mean survival **0.881321 s**, range
  **0.750060–1.100088 s**. Mean first departure **0.400032 s**; mean longest
  valid balanced posture **0.383364 s**.
- **Augmented native: 0/8 Stand**, mean survival **0.879237 s**, range
  **0.750060–1.216764 s**. Mean first departure **0.400032 s**; mean longest
  valid balanced posture **0.383364 s**.
- **Augmented GPU: 0/8 Stand**, mean survival **0.875070 s**, range
  **0.750060–1.150092 s**. Mean first departure **0.402115 s**; mean longest
  valid balanced posture **0.385447 s**.

All 32 cases physically fail, with finite bounded commands and no native solver
warnings. GPU plant warnings remain unmeasured. Evaluations record **1,684 live
control intervals**. A single matched seed and these bounded evaluations do not
establish a general scaling law; they do not support extending either arm for
an hour or promoting a checkpoint to Arena.

### Fitting versus independent recorded-state validation

Frozen inference evaluates each final actor on **114,480 saved labelled records**:
native teacher **28,800**, native corrections **18,720**, native recovery **10,080**,
GPU teacher **28,800**, GPU corrections **16,560**, and GPU recovery **11,520**.
GPU records are never used for optimization. These are independent backend
trajectories on the same original poses, not unseen-pose validation. Metrics
pool labelled records; recovery cases all have equal full-horizon length.

Control → augmented normalized command RMSE:

- Original native teacher: **0.001247 → 0.001510**.
- Original native corrections: **0.002540 → 0.003189**.
- Accepted native recovery: **0.028732 → 0.001415**, a **20.31-fold reduction**.
  First 120 recovery frames: **0.058719 → 0.002203**, a **26.65-fold reduction**.
  Full recovery torque RMSE: **2.203 → 0.121 N m**.
- GPU teacher: **0.012111 → 0.011040**.
- GPU corrections: **0.017119 → 0.017020**.
- GPU recovery: **0.024768 → 0.018904**, a **1.31-fold reduction**.
  First 120 recovery frames: **0.051655 → 0.034853**. Full recovery torque RMSE:
  **1.745 → 1.357 N m**.

Native recovery is a fitting measurement for augmentation and withheld from
control. Original native pools are fitting measurements for both. These saved
states are the original teacher/correction trajectories and Phase 35's successful
teacher branches, not either new actor's autonomous states. Low error there
does not show that the actor reproduces the teacher along its own rollout.

Feature clipping fractions are identical between arms on each saved pool because
normalizers are identical and frozen. Full native recovery clips **0.07525%** of
feature values and **4.9206%** of frames; its first 120 frames clip **0.36317%** of
features and **20.2381%** of frames. Full GPU recovery clips **0.03611%** of
features and **3.2118%** of frames. These measurements do not identify clipping
as a cause of autonomous failure. Frozen inference totals **228,960 predictions**,
with zero simulation or training steps.

### Saved verification and limits

Offline verification passes for **1,497 protected inputs**, **50 source/snapshot
pairs**, **118 original output hashes**, **121 final manifest files**, and six
portable/full checkpoint pairs. It verifies identical fresh initialization,
exact historical normalization, unchanged critic/discriminator and normalizer
tensors, saved Adam step counts, sample/coverage accounting, all terminal flags
and final Stand windows, first departures, and recorded prediction errors and
clipping against the source pools. Historical Phase 32–35 files remain unchanged.

The first verifier had an incorrect fixed prediction-count expectation: it
assumed GPU corrections also contained 18,720 rows. Their saved count is 16,560.
The original completion-bound verifier is preserved; `verify_saved_results_v2.py`
derives the expectation from the bound pool sizes and passes. No training,
simulation or evaluation was repeated to repair that offline assertion. No
general tests or standalone compilation were run. Arena and production training
configuration remain unchanged.

The admitted recovery data improve fitting strongly, but this sampling change
does not prevent autonomous drift. Increasing presentations alone is not yet
supported by these results. The next useful measurement is command accuracy and
coverage on saved autonomous student states, separately from successful teacher
branches, before choosing another optimization run.

### Proposed next frozen autonomous-state error/coverage audit; not executed

Load both final Phase 36 actors and the **1,956 live saved Phase 33 autonomous
states** with their private teacher proposals and actual two-command queue.
Perform frozen inference only, preserving the original eight-row frame batches
and discarding inactive outputs. Evaluate each model on all four recorded
generating-actor/backend groups: **3,912 live predictions total**. These are
counterfactual commands on the older actors' trajectories, not newly captured
Phase 36 trajectories or proof of current-model closed-loop behavior.

Separate initial frames, issue time below 0.100008 seconds, later valid states
before first departure, and departed states. Report command/torque error, motor
contributions and feature clipping. Compare nearest normalized-observation RMS
distance to the original native pools and the admitted early-recovery pool,
including physical observation, pending-command and reference fields separately.
Use the bound frozen normalizer and record which source/case/frame supplies each
nearest label. Distances and error are descriptive; do not infer an irreducible
error or causal clipping threshold from nearest neighbors alone.

Keep both checkpoints/normalizers and all historical sources hash-frozen.
Verify saved prediction arrays, pool/state/proposal binding and numerical metrics.
Finite late Phase 33 proposals remain diagnostic targets; most last-valid states
failed recovery in Phase 34, so no diagnostic label is automatically admitted to
training or AMP motion. This audit creates **zero simulators, physics steps,
optimizers or training updates**. No fresh rollout, dataset admission, parameter
sweep, additional three-minute run, export or Arena change is included. Execute
the frozen audit only after approval.

## Phase 37 — Frozen recovery-transfer error and coverage audit (2026-10-02)

**Status: completed after approval. Recovery augmentation improves pooled command
RMSE on later valid autonomous states by only 0.55%, despite Phase 36's 20-fold
fitting improvement on native recovery branches.** Added data cover the selected
0.1-second handoff closely, but do not substantially cover subsequent autonomous
drift. This audit makes no new learned Stand or teacher recovery claim.

Artifacts: `logs/mimickit-training/stand-recovery-transfer-audit-20261002-02`.
Driver: `mujoco_rig/mimic/audit_stand_recovery_coverage.py`.
It preserves per-model/group prediction arrays and partition metrics, common
nearest-pool arrays, a row/case/frame/episode catalog and source hashes. Root
`comparison.json` contains pooled errors, named motor contributions and coverage
at selected source frames; `final-verification.json` binds all saved results.

### Frozen inputs and interpretation

Exactly **1,956 live Phase 33 recorded autonomous states** are evaluated by each
final Phase 36 actor: **3,912 live counterfactual predictions**. Each saved frame
uses the original eight-row batch, with inactive outputs discarded. Sources are
the older control/augmented actors' native/GPU trajectories, with actual queued
commands and finite private teacher proposals. These are not new Phase 36
rollouts and do not establish how either current actor behaves on those states
in a closed loop. Checkpoints, observation/action normalizers and all historical
inputs remain frozen.

The disjoint live partitions per actor contain:

- **Initial frame 0:** 32 records.
- **Early noninitial frames 1–5:** 160 records, issue time below **0.100008 s**.
- **Later valid states before first departure:** 648 records. The pre-command
  state passes balance and posture, frame is at least 6, and occurs no later than
  the first ending-state departure frame.
- **Departed remainder:** 1,116 records. These include all remaining live states;
  no later re-entry is reclassified as a pre-first-departure state.

An additional overlapping all-valid-before-first-departure partition contains
**840 records**. Private proposals remain diagnostic labels. Phase 34 established
that most last-valid starts fail recovery, so finite late proposals are not
automatically admitted to training or AMP motion.

### Command error on autonomous states

Pooled normalized command RMSE, control → recovery augmented:

- Initial: **0.001343 → 0.001180**.
- Early noninitial: **0.032107 → 0.030853**.
- Later valid before first departure: **0.186662 → 0.185634**, only **0.55% lower**.
- Departed: **0.533304 → 0.537521**, slightly worse.
- All valid before first departure: **0.164545 → 0.163599**.

Per-group metrics additionally weight each populated pose equally. Later-valid
equal-pose command RMSE is **0.162948 → 0.155213** on older control-native states,
**0.160836 → 0.152300** on control-GPU, **0.212201 → 0.215310** on augmented-native,
and **0.205073 → 0.208026** on augmented-GPU. The aggregate result is not a broad
20-fold improvement on autonomous states.

The difference is visible within the older control-native trajectory. At source
frame **6**, issue time **0.100008 s**, pooled eight-pose RMSE drops from
**0.026283 to 0.001369**. At frame **12**, issue time **0.200016 s**, it instead
increases from **0.092517 to 0.124446**. GPU control states show the same pattern:
**0.026297 → 0.001381** at frame 6, **0.091733 → 0.123092** at frame 12. Accurate
commands at the sampled handoff do not imply accurate commands after another
six autonomous intervals.

The two thigh `rz` motors dominate later-valid torque-error contributions,
together accounting for **38.6–44.2%** of total equal-pose torque MSE across the
eight model/group combinations. Contributions use per-pose mean squared torque
errors, averaged across populated poses before normalization. This identifies
large disagreements, not a proven causal motor ablation or justification to
change action authority.

Later-valid equal-pose feature-clipping fractions span **3.02–4.66%**, identical
between the two new actors on each saved group. Initial states have no clipping;
departed states reach **22.23–24.60%**. Identical frozen normalizers explain the
identical clipping masks, but these measurements do not identify clipping as
the cause of failure or prove an irreducible regression error.

### Coverage and source binding

Nearest searches compare **47,520 original native teacher/correction pairs**,
**10,080 admitted native early-recovery pairs**, and their **57,600-pair union**.
Distances are RMS in the shared frozen **clipped normalized input space**, not
physical units. Exact float64 squared-Euclidean searches minimize independently
over full observations **0:306**, physical fields **0:95**, pending commands
**95:161**, and reference fields **161:306**. Separate field matches can refer
to different source rows; they are not a decomposition of one joint nearest row.

For the 648 later-valid records, pooled mean nearest RMS is:

- Full observations: original **1.969293**, recovery **1.960304**, union
  **1.908387**. Union reduces original distance by only **3.09%**.
- Physical fields: original **3.122701**, union **3.002343**.
- Pending commands: original **1.260336**, union **1.232281**.
- Reference fields: **zero** for each pool on every live record. Matching
  reference inputs already exist; they do not establish a match in physical state
  or queue, nor that every joint nearest row uses that matching reference.

In the older control-native trajectory, mean full-observation distance changes
from original **0.262099** to union **0.003510** at frame 6. At frame 12 it changes
only **0.998323 → 0.940879**; at frame 18, **2.244135 → 2.233684**. Thus a strong
coverage improvement at the handoff coexists with little improvement on later
autonomous states. The recovery branch follows the teacher after handoff and
does not automatically cover what the student would do afterward.

Every nearest result records its pool catalog index. The catalog binds the
original source file/hash, flat source row, case, frame and episode. Nearest-label
command differences are also saved; neither proximity nor target agreement alone
certifies a viable recovery or a physically causal explanation.

### Verification, preflight correction and scope

Offline verification passes for **1,723 protected inputs**, **52 source/snapshot
pairs**, **80 initial output hashes** and **83 final manifest files**. It verifies
full checkpoints/portable sidecars, exact shared normalizers, trace/proposal/queue
binding, disjoint partitions, inactive-output exclusion, prediction errors,
clipping and all regression metrics. Catalog rows bind directly to their source
arrays. All saved nearest distances and label errors recompute; **1,024 exhaustive
candidate checks** additionally verify nearest selection for every pose at source
frames **0, 6, 12 and 18**, for both pools and all four fields.

The first attempt stopped **before actor construction or inference** because its
catalog used relative recovery paths while the protected manifest used absolute
paths. The files' hashes matched. A separate corrected driver resolves path
identities before comparison; the original driver, failed preflight snapshots
and `stand-recovery-transfer-audit-20261002-01/failure.json` are preserved and
protected. No historical source was edited and no inference was repeated from
that preflight failure.

The successful audit and offline verifier create **zero simulators, physics
steps, optimizers, training updates, admitted datasets or AMP clips**. Verification
performs no neural forward passes. There are no new autonomous evaluations,
general tests, standalone compilation, exports, production configuration changes
or Arena changes.

The added successful branches mostly teach the selected takeover and subsequent
teacher-controlled states. They do not fill the demonstrated gap in later
student-controlled state/queue coverage. This supports measuring a later viable
takeover on the current actors before allocating another supervised run; it does
not establish that more optimization, altered normalization or springs would
solve the remaining problem.

### Proposed next live takeover admission on current actors; not executed

Evaluate the two **frozen final Phase 36 actors**, on the original eight poses
in each backend: **eight native and eight GPU starts per actor**, 32 total.
Run each actor for exactly **12 control intervals**, **0.200016 seconds**, then
switch new issued commands to the **unchanged full teacher** for **12 seconds
or physical FAIL**. Preserve the live q/v, reference, held torque and both pending
actor commands at takeover. The teacher's first new command remains delayed by
two intervals, first affecting the interval ending at **0.250020 seconds** from
reset. If a prefix physically fails, retain that failed case with no substitute
start or retry.

Use the same task instance across the prefix and takeover, retaining live
contact/solver history. This deliberately measures a live takeover rather than
Phase 35's restored starts. Save the complete prefix and teacher segment, takeover
state/queue and measurements, actor/checkpoint hashes, private prediction and
native plant warnings, survival and final-two-second Stand gates. Bind commands
to the actual delay and verify that private teacher queries do not mutate the
live plant. GPU uses its own prefix states; no identical-state backend claim.

This measurement addresses current-actor states rather than extending supervision
from the older Phase 33 actors. As read-only planning evidence, all 32 older
source frame-12 states pass balance/posture and numerical-label checks, with
maximum root angular speed **0.366287 rad/s**; current prefixes and recovery
outcomes remain unmeasured. A comparison with Phase 35 must identify the different
actors, takeover time and preserved solver history, not attribute every outcome
difference to time alone.

Admit only complete successful numerical/physical teacher segments as isolated
native supervised candidates, with separate GPU validation records. Prefix actor
commands and failed branches are not expert targets. Keep checkpoints/normalizers
frozen, preserve licensed source attribution, and verify historical inputs and
saved arrays. There is **no learning optimizer or update**, reward, normalization,
spring, support or controller change; no alternative time, extra start, AMP clip,
training extension, export or Arena promotion. Any subsequent fresh three-minute
training comparison is a separate decision after admission. Execute the live
takeover evaluation only after approval.

## Phase 38 — Live teacher takeover on current actors at 0.200016 seconds (2026-10-03)

**Status: completed after approval. All 32 live teacher takeovers pass Stand,
without any balance or natural-posture violation.** The final Phase 36 actors
are frozen and control only the first twelve intervals. The unchanged teacher
then controls the remaining twelve seconds. This establishes viable teacher
continuations from current-actor states; it is not new autonomous learned Stand.

Artifacts: `logs/mimickit-training/stand-live-takeover-20261003-01`.
Driver: `mujoco_rig/mimic/admit_stand_live_takeover.py`.
Each setting preserves the complete actor-prefix/teacher `trace.npz`, live
`takeover-state.npz`, private diagnostics, errors, reports and an isolated admitted
dataset. Root `comparison.json`, `admission-metadata.json` and
`final-verification.json` bind outcomes, provenance and verification.

### Live prefix and takeover contract

Exactly **eight native and eight GPU starts per final Phase 36 actor** were run,
on the original eight poses: **32 starts total**. Every actor prefix completes
**12 control intervals**, **0.200016 seconds**, with no physical failure. All
takeover states pass balance and posture. Maximum root angular speed at takeover
is **0.290858 / 0.290705 rad/s** for control native/GPU and
**0.169717 / 0.169704 rad/s** for augmented native/GPU. Maximum horizontal speeds
span **0.038285–0.043944 m/s**.

The same task instance continues through takeover. q/v, reference, previous held
torque, both pending actor commands, contact/solver history and counters are
preserved. No state restoration, contact forward recomputation, queue clearing
or physical support occurs at the handoff. Private teacher queries verify that
they leave the live state and queue unchanged.

The first two teacher-segment physical intervals still execute the committed
actor commands. The first newly issued teacher command affects the third
interval, ending at **0.250020 seconds from reset**. The teacher segment lasts
**720 intervals**, **12.000960 seconds**; total reset-to-terminal duration is
**12.200976 seconds**. The one-Hz predictive inverse-contact objective, torque
authority, two refinements, spring-free plant, reward and normalization are
unchanged.

Both native prefixes reproduce Phase 36's first twelve commands, balance/posture
flags, heights, tilts and foot loads **bit-exactly**. GPU balance/posture flags
match, but GPU prefixes are not exact replays. First GPU command differences are
**3.352761e-8 control / 3.725290e-8 augmented**; maximum prefix command differences
grow to **0.002034 / 0.001107**. Maximum height differences remain below
**4.77e-7 m**, tilt differences below **0.000290 degrees**, and foot-load
differences below **0.078 N**. These measured differences do not establish their
numerical cause. GPU takeovers use their own live prefix states, not identical
native states.

### Outcomes and admission

- **Control native: 8/8 teacher Stand**, accepted poses 0–7.
- **Control GPU: 8/8 teacher Stand**, accepted poses 0–7.
- **Augmented native: 8/8 teacher Stand**, accepted poses 0–7.
- **Augmented GPU: 8/8 teacher Stand**, accepted poses 0–7.

All cases reach TIME after the full teacher horizon. Balance and natural-posture
gates pass throughout both the actor prefix and teacher segment, including the
final two seconds. There are **zero teacher-command errors, zero private
prediction warnings and zero native plant warnings**. GPU plant solver warnings
remain unmeasured. There are no failed or replacement starts, alternate takeover
times, retries or parameter configurations.

Accepted native supervised candidates total **11,520 teacher decision records**:
**5,760 per generating actor**. Separate GPU validation records also total
**11,520**. Every accepted case supplies all 720 teacher frames. **Zero actor-prefix
commands** are admitted as expert labels; failed branches would also be excluded.
Datasets remain isolated artifacts and have not been merged into training.

Dataset `action`, `executed_action` and `teacher_executed` describe the new teacher
command issued into the delayed queue, not the torque already held during that
interval. The trace records `held_torque_nm` separately. Dataset frame zero is
the takeover observation at reset issue time 0.200016 seconds; dataset steps
0–719 belong only to the teacher segment. Source metadata retain the existing
100STYLE/CC-BY-4.0 reference attribution, rig/reference hashes, frozen actor hashes
and live trace/handoff hashes. **Zero new AMP clips** are created.

Phase 35 had 30/32 successful restored starts at 0.100008 seconds from older
actors. Phase 38 has 32/32 live takeovers at 0.200016 seconds from current actors.
Different actors, intervention times and preserved versus reset solver history
prevent attributing this difference solely to later intervention. It establishes
success on these current prefixes, not an optimal takeover time, a universal
recovery region or unseen-pose generalization.

### Accounting and verification

The run records **384 live actor predictions**, **23,424 live control intervals**,
**93,696 live plant physics steps** and **184,320 private teacher prediction
steps**. Teacher queries occur only after the prefix: 23,040 live queries,
each predicting eight private physics steps. Private refinements do not advance
time, and inactive GPU integration is excluded from live counts. There are
**zero learning optimizers or updates**, no new normalization samples, PPO or
discriminator updates, configuration changes, exports or Arena promotion.

Offline saved-array verification passes for **1,860 protected inputs**, **53
source/snapshot pairs**, **83 initial output hashes** and **87 final manifest
files**. It verifies frozen full checkpoints, portable sidecars, reconstructed
actor tensor hashes and normalizer counts **59,040 / 0**. It also verifies prefix
replay metrics, exact q/v continuity across every interval and handoff, the live
pending queue and held torque, the two-interval delay, finite bounded commands,
private/native warnings, full-horizon TIME, final Stand gates and admitted records
matching only the teacher segment. No new simulation, inference or optimization
is performed by verification. No general tests or standalone compilation ran.

The teacher can maintain natural Stand after these current policies have acted
for 0.2 seconds, despite their autonomous failures around 0.88 seconds in Phase
36. We now have admitted labels from a later point on the current actors' own
prefixes. Whether a fresh student can transfer this supervision to autonomous
Stand remains unmeasured and requires a separate bounded training comparison.

### Proposed next matched fresh three-minute live-recovery pilot; not executed

Compare two **fresh identically initialized actors**, seed **301012**, each with
an independent empty Adam state and **180 seconds of supervised optimization**.
Keep the existing architecture, learning rate **1e-4**, batch **512**, gradient
clip **1**, normalized full-33-coordinate MSE and the exact shared frozen Phase
32 normalizers (counts **59,040 / 0**, observation clip **10**). No trained weights
or optimizer are resumed and no moments are refit. Preserve critic/discriminator
tensors and verify common initial weights/normalizer hashes.

The baseline uses Phase 36's recovery-augmented native sampling: **128 correction**,
**64 original early teacher**, **64 original uniform teacher**, **128 Phase 35 early
recovery**, **128 Phase 35 uniform recovery** records per batch. Its available
pool contains **57,600 pairs**.

The live-recovery arm adds only the **11,520 admitted native Phase 38 records**,
for **69,120 available pairs**. Each batch contains **64 correction**, **32 original
early teacher**, **32 original uniform teacher**, **64 Phase 35 early recovery**,
**64 Phase 35 uniform recovery**, **128 Phase 38 early live recovery**, and
**128 Phase 38 uniform live recovery** records. Early strata use the first
**120 teacher frames**. Sample recovery generating actor/pose cases uniformly
with replacement (**14 Phase 35 cases / 16 Phase 38 cases**), then eligible
frames uniformly. Exclude GPU records, actor prefixes, failed branches, Phase 30
mixture labels and Phase 34 late-recovery labels from both optimizers.

Record actual wall time, update/sample counts, unique coverage, clipping and
command/torque errors separately for fitting pools and GPU validation pools.
Equal wall time does not guarantee equal update counts. After each final
elapsed-time checkpoint, evaluate exactly **eight native and eight GPU autonomous
starts**, up to **12 seconds or physical FAIL**: **32 cases total**. Report first
departure, survival, posture and final Stand, including all failures. Verify
protected inputs, source snapshots, saved checkpoints/optimizer accounting,
normalizer integrity and saved evaluation gates.

This is a supervised transfer comparison, with no teacher at autonomous inference,
PPO/discriminator update, new AMP clip, reward, normalization, spring, support,
action-authority or controller change. No sweep, extra starts, longer run, export
or Arena promotion is included. Teacher takeover success does not count as learned
actor success. Execute the matched training/evaluation pilot only after approval.

## Phase 39 - Matched fresh live-recovery supervision (2026-10-03)

Executed the approved Phase 38 follow-up: two fresh actors, 180 seconds of
supervised optimization each, then exactly eight native and eight GPU autonomous
12-second-or-FAIL starts per actor. **Both actors passed 0/16 Stand cases. All
32 starts physically failed.** Adding live-recovery labels improved regression
on those records but did not transfer to autonomous Stand in this comparison.

### Artifacts and unchanged contract

Independent driver: `mujoco_rig/mimic/compare_stand_live_recovery_supervision.py`.
Artifact root: `logs/mimickit-training/stand-live-recovery-supervision-20261003-01`. Its `protocol.json`, `initialization.json`,
`report.json`, `comparison.json`, `completion-verification.json` and
`final-verification.json` record the comparison and integrity checks. Each arm
contains initial/final portable checkpoints and AMP sidecars, its fresh
supervised Adam state, training curve, native/GPU evaluation traces, and saved
predictions/clipping flags for all eight labelled-data groups.

Both actors used seed **301012**, the existing architecture, Adam **1e-4**, batch
**512**, gradient clip **1**, and normalized full-33-coordinate MSE. Only the
exact Phase 32 observation/action normalizer tensors were copied: counts
**59,040 / 0**, zero pending samples, observation clip **10**. No trained model
weights or optimizer state were resumed; no normalizer moments were refit.

- Fresh full-model tensor hash: `14f7c630332765ea373853e8a15b41c49de20733eb0c0c47d9cf2eaff4500f98`.
- Common normalized initial full-model hash: `0b5573bcbaed20e7d6401f037e21e92c3d3ee498eec2cec31653aa8ddf057ef7`.
- Common initial actor hash: `deebf83b0733e3e0b36208bb689069937ad3d743780d6239b19b71eca24ed2b1`.
- Common frozen normalizer hash: `9a791b568d5c637a3162a54df921053031a7584b4adc2596ca0bf1d29f070832`.

The two initial checkpoints were tensor-identical. Each supervised Adam began
empty. Critic, discriminator and discriminator-normalizer tensors remained
equal to the fresh model. The plant still has 33 direct torque motors, six
passive wrist joints, zero stiffness, four physics steps per control interval
and the original two-interval action queue. The controller was absent from
autonomous inference. Reward, authority, reference, reset poses and original
34 AMP demonstrations were unchanged; PPO/discriminator updates, new AMP clips
and Arena/export changes were all **zero**.

### Exact data mix and optimization accounting

The `control` label now means the approved Phase 36 recovery-augmented baseline,
not Phase 36's original control arm. It used **57,600 available native pairs**:
28,800 original teacher, 18,720 corrections and 10,080 Phase 35 recoveries.
Each batch drew 128 corrections, 64 early/64 uniform original teacher, and
128 early/128 uniform Phase 35 recovery records.

The `augmented` arm added **11,520 native Phase 38 live-recovery pairs**, giving
**69,120 available pairs**. Each batch drew 64 corrections, 32 early/32 uniform
original teacher, 64 early/64 uniform Phase 35 recovery, and 128 early/128
uniform Phase 38 live recovery. Recovery sampling used uniform replacement
over generating actor/pose cases (14 Phase 35 / 16 Phase 38), then uniform
eligible frames. Early strata contain the first 120 teacher frames.

GPU labels, actor-prefix commands, failed recovery branches, Phase 30 mixture
labels and Phase 34 late-recovery labels were excluded from both optimizers.
The new labels are newly issued teacher commands into the unchanged delayed
queue; the first two teacher intervals still physically execute pending actor
commands. These queued commands are not substituted as training targets.

- Baseline: **180.001278 s**, **93,593 Adam updates**,
  **47,919,616 repeated supervised presentations**.
  All **57,600** available native pairs were seen; Phase 38 pairs seen: **0**.
- Live-recovery arm: **180.002121 s**, **88,913 Adam updates**,
  **45,523,456 repeated supervised presentations**.
  All **69,120** available native pairs were seen, including all **11,520** new pairs.

The added-data arm performed **5.00% fewer updates** within the same wall-time
budget. This is the approved equal-time comparison, not an equal-update or
single-variable label-count ablation: the sampling weights also change.
Repeated presentations are reused saved data, **not new physics transitions**.
The supervised loops collected zero new physics transitions.

### Autonomous results

Each result below uses eight original licensed reset poses. Stand requires
12-second survival and a valid final two-second balance/natural-posture window.
First departure is the first recorded balance or posture gate failure; it is
distinct from the later physical FAIL.

- Baseline, native: **0/8 survivors, 0/8 balance passes, 0/8 Stand passes**;
  mean survival **0.800064 s**, mean first departure
  **0.368779 s**, mean longest
  balanced/natural-posture interval **0.352111 s**.
- Baseline, gpu: **0/8 survivors, 0/8 balance passes, 0/8 Stand passes**;
  mean survival **0.795897 s**, mean first departure
  **0.366696 s**, mean longest
  balanced/natural-posture interval **0.350028 s**.
- Live-recovery arm, native: **0/8 survivors, 0/8 balance passes, 0/8 Stand passes**;
  mean survival **0.762561 s**, mean first departure
  **0.379197 s**, mean longest
  balanced/natural-posture interval **0.362529 s**.
- Live-recovery arm, gpu: **0/8 survivors, 0/8 balance passes, 0/8 Stand passes**;
  mean survival **0.762561 s**, mean first departure
  **0.379197 s**, mean longest
  balanced/natural-posture interval **0.362529 s**.

Native mean survival changed **0.800064 -> 0.762561 s**; GPU mean survival changed
**0.795897 -> 0.762561 s**. First departure moved slightly later on average,
but survival did not improve. Individual cases also varied in both directions.
No winner or positive learning-speed multiplier is established. This is one
matched seed and these original poses, without unseen-pose validation.

Native solver warnings: **0**. GPU plant solver-warning measurement remains
**unavailable** (`null`); its inherited false trace placeholder is not evidence
of zero warnings. All saved neural commands were finite and bounded to [-1, 1].

### Saved-state fitting and GPU validation

Numbers below are baseline -> live-recovery arm: full-horizon normalized
33-command RMSE, then torque RMSE in N m after multiplying by per-motor scales.
These compare frozen actor predictions against saved teacher labels; they are
not autonomous rollouts. GPU groups are excluded from training and share the
original poses; they measure backend/state-transfer validation.

- `teacher-native` (28,800 records): **0.001778 -> 0.002110**;
  torque **0.136302 -> 0.159827 N m**.
- `corrections-native` (18,720 records): **0.003928 -> 0.005508**;
  torque **0.319045 -> 0.427069 N m**.
- `recovery-native` (10,080 records): **0.001929 -> 0.002379**;
  torque **0.165485 -> 0.205075 N m**.
- `live-recovery-native` (11,520 records): **0.024341 -> 0.001877**;
  torque **1.778677 -> 0.146111 N m**.
- `teacher-gpu` (28,800 records): **0.010695 -> 0.010446**;
  torque **0.843965 -> 0.818352 N m**.
- `corrections-gpu` (16,560 records): **0.017477 -> 0.016755**;
  torque **1.282093 -> 1.222951 N m**.
- `recovery-gpu` (11,520 records): **0.019997 -> 0.019208**;
  torque **1.421616 -> 1.359432 N m**.
- `live-recovery-gpu` (11,520 records): **0.024136 -> 0.016996**;
  torque **1.791928 -> 1.210883 N m**.

Native Phase 38 live-recovery error fell **12.97x (92.29%)**; on its first 120
frames, command RMSE fell **0.054650 -> 0.002601** and torque RMSE
**4.080835 -> 0.221114 N m**. This is fitting error only for the augmented arm;
the baseline never trained on those records. GPU Phase 38 error fell **29.58%**,
a much smaller improvement. Original native teacher/correction/Phase 35
regression worsened under the changed sampling weights. Better imitation of
the new recorded recoveries therefore did not produce better autonomous Stand.

### Clipping evidence and interpretation

With the identical frozen normalizers, saved-data clipping fractions are
identical for both actors. Across all native labelled groups, clipped feature
fractions range from **0.0102% to 0.1589%**; Phase 38 native live-recovery records
are **0.0546%** overall (**0.3249%** over the first 120 teacher frames).

On the autonomous trajectories, pooling all live frames gives:

- Baseline: **15.8505%** of observation features clipped on native and
  **15.7427%** on GPU.
- Live-recovery arm: **14.2371%** on native and **14.1648%** on GPU.

These averages include the drift toward physical failure. They do not establish
when clipping began, which observation blocks dominate, whether information
was lost before the first departure, or whether clipping caused the failure.
The current evaluation traces lack complete live observations, velocities and
action-queue states, so those questions cannot be reconstructed from these
artifacts alone. New labels came from Phase 36 policies, while these fresh
Phase 39 policies produce their own trajectories. The gap between saved-state
fitting and current autonomous behavior remains unresolved; more elapsed
optimization alone is not justified by this comparison.

### Verification

Offline saved-artifact verification passed: **2,002 protected inputs**,
**54 source/snapshot pairs**, **130 completion-manifest hashes**, and
**133 final-manifest file hashes**. It checked six portable/full checkpoint
pairs, common fresh initialization, exact Phase 32 normalizers, unchanged
critic/discriminator tensors, saved actor/normalizer hashes, Adam step counts,
stratum presentation totals and source-bound recorded labels/clipping/errors.

All 32 saved terminal outcomes and final Stand windows were recomputed from
the saved live/done/balance/posture arrays, with survival, first departure,
longest valid intervals and violation durations checked against reports.
The trace stores evaluated gate booleans; this does not independently recompute
every gate from complete underlying physical state.

The autonomous evaluations used **1,498 live control intervals / 5,992 nominal
plant physics steps**. Frozen saved-state inference evaluated **275,040 records**
across both actors. Verification created no simulators, forward passes or
optimizer updates. No general tests or standalone compilation ran.

### Proposed next current-policy boundary audit; not executed

Freeze both Phase 39 final actors. Capture exactly eight native and eight GPU
original starts per actor (**32 cases total**), for at most **36 control intervals
(0.600048 s) or physical FAIL**. Keep the unchanged plant, resets, reference,
normalizers, deterministic actor inference and action queue. No teacher command
may drive the live plant. Preserve critic/discriminator, checkpoints and all
existing artifact hashes.

For each live interval save complete observations before action, unclipped
standardized observations, per-feature clip flags, qpos/qvel, pending/issued/
executed commands, complete physical balance/posture measurements and gate
flags. Query the unchanged teacher privately on the reached state to record
a counterfactual newly issued command and its diagnostics. Report unsuccessful
queries; do not admit labels or claim physical recovery from a private query.

Compare current-policy command error and nearest recorded-state distances
against the existing native training pools, separately for physical state,
pending queue and reference. Report clipping and disagreement by observation
block/motor over time, especially reset, 0.2 s, first departure and the remaining
live prefix. Keep native/GPU and each actor/pose separate before pooling. This
will distinguish clipping present before departure from clipping after drift,
and measure whether the current actor reaches states absent from its labels.

No training, normalizer refit, new recovery admission, AMP/PPO update, springs,
support, authority/reward/controller change, longer rollout, sweep, export or
Arena promotion is included. Execute this bounded audit only after approval.

## Phase 40 - Current-policy boundary, clipping and coverage audit (2026-10-03)

Executed the approved bounded audit of both frozen Phase 39 actors: eight native
and eight GPU original starts per actor, exactly **32 captures**. All cases
reached the **36-control-interval budget (0.600048 s)** without physical FAIL.
Every case had already violated balance before the budget ended. This short
capture is not Stand admission; Phase 39 remains **0/16 learned Stand passes
per actor**. No model was trained or promoted.

### Artifacts and bounded execution

Independent driver: `mujoco_rig/mimic/audit_stand_current_boundary.py`.
Artifact root: `logs/mimickit-training/stand-current-boundary-20261003-01`. It contains `protocol.json`, `report.json`,
`comparison.json`, source snapshots, native pool provenance/catalog, saved
coverage arrays and completion/final verification manifests. Each actor/backend
setting contains complete `trace.npz`, private teacher diagnostics/errors,
`capture-report.json`, `diagnostics.npz` and `analysis.json`. Reports preserve
each pose and backend before pooled/equal-pose summaries, plus every frame.

Captured before each action: raw 306-feature observations, unclipped standardized
values, feature clip flags, exact native or converted GPU qpos/qvel, both pending
commands, queue cursor, issued and held/executed commands. Saved complete pre-
and post-step balance/posture measurements and individual gate booleans. Actor
weights, normalizer tensors/counts, critic/discriminator and historical inputs
were frozen. The normalizers remain exactly Phase 32, **59,040 / 0**, clip **10**.

The task retained its prior 12-second termination setting and was externally
bounded to 36 steps; no artificial TIME flag was introduced at 0.6 seconds.
Only the actor drove the plant through the unchanged two-step queue. The
original full one-Hz predictive teacher was queried privately on every reached
state, replaying the actual pending commands and using its unchanged torque
objective/refinements. Live qpos/qvel, queue, control and observations were
checked unchanged after each query. Private proposals were not executed,
admitted as data or claimed to recover the dummy.

- **1,152** live control intervals / **4,608** live plant physics steps.
- **1,152 / 1,152** numerically valid private proposals, **0** query exceptions
  and **0** private solver-warning labels.
- **9,216** private prediction physics steps, recorded from private simulation
  time; these are separate from the live plant.
- Training/PPO/discriminator updates, new optimizer state, normalizer samples,
  dataset admissions, AMP clips, spring/support/authority/reward changes and
  Arena/export changes: **0**.

### First departure is excessive angular speed

The first violated ending-state gate was **angular speed > 1 rad/s in all
32 cases**. No posture gate was the first departure. Native first-departure
ending times by pose 0..7:

- `control`: **0.333360, 0.350028, 0.383364, 0.383364, 0.316692, 0.333360, 0.400032, 0.450036 s**.
- `augmented`: **0.366696, 0.350028, 0.400032, 0.366696, 0.416700, 0.433368, 0.416700, 0.283356 s**.

GPU first-departure times matched the corresponding native times in this
audit. These are balance departures at **0.283356-0.450036 s**, not the later
physical failures observed in Phase 39. At a first-departure decision, the
actor sees the state at ending time minus one control interval; the torque
physically executing during that interval was issued two frames earlier.

### Clipping begins early, but early clipping is already in teacher states

At reset, **zero features were clipped** in all 32 starts. The first clipped
observation arrived at issue time **0.016668 s in every case**, before any
actor torque had passed through the two-step queue. The first clipped features
were root vertical velocity and hinge-velocity entries, including entries
30/38 in all poses. At that frame the native physical-state nearest distance
to the training pool is **exactly zero**. It is also zero at issue time
**0.033336 s**: both intervals still have zero held torque.

For both native actors, mean feature clipping at issue times 0.016668 and
0.033336 s was **0.7761%** and **3.8807%**. Clipping therefore occurs on early
physical states already represented in the labels. Pending actor commands
can differ even when the physical state is identical. Early clipping alone
does not identify the cause of failure or establish an out-of-distribution
physical state.

Later clipping grows while the pre-action balance/posture gates still pass.
For decisions from frame 12 (0.200016 s) through the first ending departure,
conditional on valid pre-action posture/balance:

- `control-native`: **81** decisions; **6.6408%** of features clipped;
  root angular-velocity block **37.04%**, hinge-velocity
  block **33.40%**.
- `augmented-native`: **86** decisions; **6.4865%** of features clipped;
  root angular-velocity block **39.92%**, hinge-velocity
  block **34.68%**.
- `control-gpu`: **81** decisions; **6.6126%** of features clipped;
  root angular-velocity block **36.21%**, hinge-velocity
  block **33.43%**.
- `augmented-gpu`: **86** decisions; **6.4789%** of features clipped;
  root angular-velocity block **40.70%**, hinge-velocity
  block **34.59%**.

These are block clipping frequencies, not causal attribution. The entire
physical state, queue and reference remain available in the saved arrays for
a controlled follow-up.

### Good saved-state fitting does not hold on reached trajectories

Full-33-command disagreement against the private teacher grows rapidly on
each actor's own states. The following pooled native numbers retain eight poses
at fixed times; late-valid partitions have different per-pose sample counts
and also have separate equal-pose summaries in the artifacts:

`control-native`:

- `reset`: command RMSE **0.001294**,
  torque RMSE **0.111466 N m**, clipping **0.0000%**.
- `at_0_2s`: command RMSE **0.157605**,
  torque RMSE **17.281008 N m**, clipping **1.1029%**.
- `late_valid_before_departure`: command RMSE **0.262432**,
  torque RMSE **26.157291 N m**, clipping **6.6408%**.
- `first_departure_decision`: command RMSE **0.414395**,
  torque RMSE **41.077901 N m**, clipping **13.2353%**.

`augmented-native`:

- `reset`: command RMSE **0.001754**,
  torque RMSE **0.178538 N m**, clipping **0.0000%**.
- `at_0_2s`: command RMSE **0.185022**,
  torque RMSE **14.498391 N m**, clipping **2.0833%**.
- `late_valid_before_departure`: command RMSE **0.248467**,
  torque RMSE **24.944239 N m**, clipping **6.4865%**.
- `first_departure_decision`: command RMSE **0.423896**,
  torque RMSE **45.699297 N m**, clipping **12.9493%**.

The motor commands already held at first departure came from earlier
decisions with pooled native command RMSE **0.330155 / 0.350836** and torque
RMSE **37.918345 / 36.779082 N m** (control / augmented), rather than from the
new command issued on the departure interval. The thigh-rz pair contributes
**52.19% / 52.21%** of torque-weighted disagreement in the late-valid native
partitions. This is a disagreement ranking, not an isolated motor intervention.
The teacher proposals are counterfactual diagnostics; their physical recovery
success at these reached states remains untested.

GPU command RMSE at 0.200016 s was **0.156904 / 0.141943**, and in the late-valid
partitions **0.264237 / 0.245620**. These have the same broad drift pattern but
are not interchangeable with native values.

### Adding older-policy recovery labels barely covers the current trajectories

The native catalog exactly matches the actual Phase 39 training pools:
**57,600 baseline pairs**, **11,520 added Phase 38 pairs**, **69,120 union pairs**.
It preserves original source file, row, pose, frame and episode provenance.
Exact nearest searches use all training poses and independently minimize full
(306), physical (0:95), pending queue (95:161) and reference (161:306) blocks.
Distances are RMS in the shared frozen standardization, computed in float64
after the original float32 standardization. Both unclipped and inference-clip-10
spaces are reported; there is no universal OOD cutoff. Independent block minima
need not come from the same training record.

At native reset, nearest full/physical/pending/reference distances to baseline
and union are exactly zero. The reference block remains exactly represented
at every captured decision. At issue time 0.200016 s, nearest unclipped physical
RMS to the union is **2.424985 / 4.155793**, and queue RMS **1.116787 / 1.158509**.
The changed physical state and queue, rather than the stationary reference,
dominate the current-state gap.

In late-valid native decisions, adding Phase 38 labels reduces mean nearest
unclipped full-state RMS only:

- `control-native`: **9.153900 -> 9.034291**, a **1.31%** reduction.
- `augmented-native`: **8.336958 -> 8.284495**, a **0.63%** reduction.

At the first-departure decision, unclipped nearest full-state RMS to the union
is **27.275688 / 21.092001**; clipping compresses these to **3.823226 / 3.934091**.
Using only clipped distances therefore hides much of the raw standardized
drift. Large standardized distances can also reflect narrow frozen moments;
neither space by itself proves the corrective action or causal mechanism.

Phase 39 reduced fitting error on older-policy live-recovery labels about
13x. Phase 40 shows that the new students' own trajectories still accumulate
much larger teacher disagreement and move away from the labelled state/queue
combinations before departure. More repeated presentations of the same saved
labels are not equivalent to learning corrective feedback on these reached
states. This audit identifies the transfer gap; it does not show that either
normalization changes or additional labels would solve it.

### Replay and verification limits

All saved native commands, gates, heights, tilts, foot loads and clipping
fractions were **bit-exact Phase 39 prefixes**. The teacher queries did not
change the native trajectories. GPU captures were not bit-exact replays:
maximum command differences from prior GPU prefixes reached **0.841419**
(control) and **1.067758** (augmented), with height differences up to
**0.016964 / 0.011830 m**, tilt differences **2.261063 / 3.098635 degrees** and
foot-load differences **839.718934 / 482.247314 N**. Private query nonmutation
checks passed, but the numerical source of GPU replay divergence was not
isolated. Native results provide the exact-replay evidence. GPU plant solver
warnings remain unavailable; native and private warning measurements were zero.

Offline verification passed **2,191 protected inputs**, **55 source/snapshot
pairs**, **88 completion hashes** and **92 final-manifest file hashes**.
It verified checkpoint/sidecar and frozen-normalizer integrity, source-bound
catalog rows, queue/delayed torque accounting, saved standardization/clipping,
pre/post gate thresholds and aggregate booleans, query masks, regression and
coverage metrics, all nearest distances and selected source labels. Independent
exhaustive nearest searches made **2,048 block/pool checks** at reset, 0.2 s,
first departure and last captured decisions. Verification created no simulator,
forward pass or optimizer update. No general tests or standalone compilation ran.

### Proposed next controlled velocity-clipping probe; not executed

Use the **same two frozen Phase 39 final actors** and identical frozen means,
standard deviations and counts. Compare exactly two inference conditions:
baseline clip 10 on every feature, versus clip 10 everywhere except the root
linear/angular-velocity blocks (9:15) and hinge-velocity block (54:93), which
remain unclipped. Change only this velocity clipping rule inside an isolated
diagnostic inference path; leave the saved policies and global normalizer
implementation untouched. Do not change weights, action clamp/torque scale,
queue, physical plant, reset/reference or reward.

Run exactly **eight native original starts per actor per condition: 32 cases**.
Bound each to **36 control intervals (0.600048 s) or physical FAIL**; keep the
original 12-second task setting and mark budget stops explicitly. Baseline
captures must reproduce the corresponding Phase 40 native prefixes exactly.
Native-only execution uses the demonstrated exact replay and avoids treating
the current GPU replay divergence as an isolated clipping intervention.

Save raw observations, standardized values, original would-clip flags versus
actual applied clipping, pending/issued/held actions, full pre/post measurements
and gates. Keep private unchanged teacher queries unexecuted to measure command
and torque disagreement. Report unavailable queries and any nonfinite state/
command as failures; do not omit them. Compare first angular/balance/posture
departure, valid duration, angular-speed growth, command saturation and teacher
disagreement by actor/pose, then pool.

This tests whether velocity clipping changes the bounded physical drift; it
does not test new training or establish learned Stand. No training, moment
refit, recovery admission, AMP/PPO update, spring/support/authority/controller
change, GPU sweep, longer rollout, export or Arena promotion is included.
Execute this controlled probe only after approval.

## Phase 41 - Controlled velocity-clipping inference probe (2026-10-03)

Executed the approved native-only comparison: the same two frozen Phase 39
actors, eight original poses per actor per condition, **32 starts total**,
each capped at **36 control intervals (0.600048 s) or physical FAIL**. The
intervention removed clipping only from root linear/angular velocities (9:15)
and hinge velocities (54:93); every other input still used clip 10.

**Removing velocity clipping worsened mean balance-departure time and increased
command saturation and angular-speed peaks for both actors. Retain clip 10
for these existing policies.** This is a bounded inference result, not a new
learned Stand evaluation. Phase 39 still has **0/16 Stand passes per actor**.

### Artifacts and isolation

Independent driver: `mujoco_rig/mimic/probe_stand_velocity_clipping.py`.
Artifact root: `logs/mimickit-training/stand-velocity-clipping-20261003-01`.
Root protocol/report/comparison files, source snapshots and completion/final
verification manifests bind the run. Each actor/condition contains complete
`trace.npz`, private teacher query diagnostics/errors, `capture-report.json`,
`diagnostics.npz` and `analysis.json`.

The saved policies and global normalizer implementation were untouched.
Model weights, frozen Phase 32 means/std/counts (**59,040 / 0**), architecture,
action unnormalization/clamp, torque scales, queue, plant, reference/reset
poses and reward were identical between paired conditions. The original
12-second task setting was retained; capture stopped externally at the budget.

Both baseline traces reproduced **every saved Phase 40 native trace field
bit-exactly**, including private teacher proposals and full pre/post states
and measurements. Paired reset qpos/qvel, observations and issued commands
were identical. The first issued-command difference appeared at **0.016668 s**
in both actors; the first ending-state difference appeared at **0.066672 s**,
after that changed command passed through the original two-step queue.

Saved fields distinguish raw observations, original unclipped standardized
values, original would-clip flags, actual applied-clip flags and the exact
input passed to the network. Only the selected velocity slices bypassed
clipping. All actual neural commands retained the original [-1, 1] bounds.
Invalid input/output handling records a failure rather than integrating a
nonfinite command or dropping a case; no such failure occurred in this run.

### Paired physical-prefix results

All **32** cases reached the budget without physical FAIL, nonfinite failure
or native solver warning. All had already lost balance through angular speed
above 1 rad/s. A budget stop does not mean the dummy stood successfully.

- Control mean first angular/balance departure: **0.3687795 -> 0.304191 s**,
  **64.5885 ms earlier**. Mean longest balanced/natural-posture interval:
  **0.3521115 -> 0.287523 s**. Paired poses: **7 earlier, 1 unchanged, 0 later**.
- Augmented mean first angular/balance departure: **0.379197 -> 0.3187755 s**,
  **60.4215 ms earlier**. Mean longest balanced/natural-posture interval:
  **0.362529 -> 0.3021075 s**. Paired poses: **5 earlier, 0 unchanged, 3 later**.

The augmented actor's three later departures were small (16.668, 16.668 and
50.004 ms); its five earlier departures included a **200.016 ms** regression.
Every pose in both actors had a higher angular-speed peak under the unclipped
velocity condition, including the poses whose first departure moved later.
First-departure improvement in those few poses therefore did not imply an
overall quieter or recovered trajectory.

First-departure ending times by pose 0..7, baseline -> velocity-unclipped:

- `control-baseline`: **0.333360, 0.350028, 0.383364, 0.383364, 0.316692,
  0.333360, 0.400032, 0.450036 s**.
- `control-velocity-unclipped`: **0.300024, 0.283356, 0.350028, 0.250020,
  0.250020, 0.333360, 0.333360, 0.333360 s**.
- `augmented-baseline`: **0.366696, 0.350028, 0.400032, 0.366696, 0.416700,
  0.433368, 0.416700, 0.283356 s**.
- `augmented-velocity-unclipped`: **0.383364, 0.366696, 0.316692, 0.283356,
  0.216684, 0.283356, 0.366696, 0.333360 s**.

### Command saturation, angular speed and private teacher disagreement

These values pool the same **288 live decisions** per setting over the bounded
prefix. Saturation means |issued normalized command| >= 0.98, averaged across
33 motors and all live decisions. Peak angular speed is the largest measured
post-step root angular speed across the eight poses; it is not an average.
Teacher disagreement is a counterfactual private proposal on each condition's
own reached state and actual pending queue, not a shared-state comparison or
proof of teacher recovery.

- Control saturation: **1.3784% -> 33.6911%**; peak angular speed
  **9.833137 -> 18.410023 rad/s**; command RMSE **0.398310 -> 0.748265**;
  torque RMSE **38.430069 -> 71.643052 N m**.
- Augmented saturation: **1.0206% -> 30.3451%**; peak angular speed
  **10.451740 -> 24.376751 rad/s**; command RMSE **0.363556 -> 0.720980**;
  torque RMSE **36.176553 -> 69.249161 N m**.

At issue time 0.200016 s, teacher disagreement also increased:

- Control: command RMSE **0.157605 -> 0.255120**; torque RMSE
  **17.281008 -> 27.239712 N m**.
- Augmented: command RMSE **0.185022 -> 0.307761**; torque RMSE
  **14.498391 -> 28.215932 N m**.

At the last captured decision, saturation reached **72.35% / 79.17%** for the
unclipped control/augmented actors, versus **1.52% / 1.14%** in baseline.
Disabling a subset of clipping did not even reduce the pooled overall applied
clipping frequency: changed trajectories drove additional nonvelocity features
beyond clip 10. Overall applied feature clipping changed **11.7749% -> 13.9286%**
for control and **10.6867% -> 12.9845%** for augmented. All selected velocity
features were nevertheless verified unclipped throughout the intervention.

The paired intervention establishes that this specific clipping removal makes
mean bounded drift worse for these fixed policies and original poses. It does
not establish that all normalization choices are optimal, test retraining with
broader moments, or remove the state/queue coverage gap observed in Phase 40.
Longer optimization or a broader clipping sweep is not justified by this
negative result.

### Integrity and accounting

Execution used **1,152** live/integrated control intervals, **4,608** live plant
physics steps, **1,152 / 1,152** valid unexecuted teacher proposals and **9,216**
private prediction physics steps. Query exceptions, private warning labels,
nonfinite failures and native solver warnings were all **zero**. No GPU world
was created. No training/PPO/discriminator updates, optimizer creation, moment
refit, data admission, AMP clip, spring/support/authority/reward change, export
or Arena promotion occurred.

Offline verification passed **2,340 protected inputs**, **56 source/snapshot
pairs**, **83 completion hashes** and **87 final-manifest file hashes**. It checked
full/portable checkpoint parity and hashes, unchanged normalizers, paired
initial states/reset commands, exact baseline full-trace reproduction, every
saved actual network-input clipping rule, raw-mean action unnormalization,
finite/query masks, queue/torque delays, physics counts and complete saved
pre/post gate thresholds. Per-pose, temporal, pooled and paired analysis was
recomputed from arrays. Verification created no simulator, physics step,
network forward pass or optimizer update. No general tests or standalone
compilation ran.

### Proposed next current-policy live-recovery admission; not executed

Keep ordinary clip **10 on all inputs**. Freeze the same two Phase 39 final
actors and their exact weights/normalizers. Run exactly **eight native and
eight GPU original starts per actor (32 cases)**. Each actor drives **12 control
intervals (0.200016 s)**, then the unchanged full teacher takes over on the
**live state and live queue** for up to **12 seconds or physical/command FAIL**.
Maximum reset-to-end duration is **12.200976 s / 732 control intervals**.
Do not restore a snapshot, reset/forward the plant at handoff, edit its queue,
change controller gains/authority, or add springs/support. The first two teacher
intervals retain queued actor torques.

Require the native actor prefix to reproduce Phase 40's first 12 intervals
exactly. Record actual GPU prefix differences and warning availability without
claiming exact replay. Save pre/post handoff qpos/qvel, observations, pending/
issued/held commands, full measurements/gates, teacher diagnostics and every
failed case. Actor and normalizer hashes/counts must remain unchanged.

Admit only numerically valid teacher suffixes that survive all 12 teacher
seconds and pass the unchanged final two-second balance/natural-posture Stand
window. Save newly issued teacher commands on successful **native** suffixes
as an isolated candidate dataset; save successful GPU suffixes separately
for validation. Exclude actor prefixes and failed branches from candidates.
Private finite proposals alone do not qualify for admission.

This directly tests recovery from the **current Phase 39 policies' own prefixes**,
where Phase 40 observed poor coverage; the already admitted Phase 38 labels
came from older Phase 36 policies. Teacher takeover success will not count as
learned actor success. No training, normalizer refit, optimizer/PPO/discriminator
update, AMP clip generation, longer run, parameter sweep, export or Arena
promotion is included. Execute this bounded admission only after approval.

## Phase 42 - Current-policy live recovery admission (2026-10-03)

Completed under the user's instruction to proceed with the next steps of the
corrected plan. Independent runner: `admit_stand_current_policy_recovery.py`.
Artifacts: `logs/mimickit-training/stand-current-policy-recovery-20261003-01`.
The Phase 39 actors remained frozen. All 32 starts completed their 12-interval
actor prefix, followed by live teacher takeover with the original pending queue.

Teacher Stand/admission results:

- Control native: 6/8; cases 2 and 4 physically failed after 9.300744 and
  3.983652 teacher seconds respectively.
- Control GPU: 7/8; case 4 physically failed after 2.683548 teacher seconds.
- Augmented native and GPU: 8/8 each.
- Total: **29/32 teacher passes**, **10,080 native candidate labels** and
  **10,800 separate GPU validation labels**. Failed branches and actor prefixes
  are excluded from these datasets. No actor training occurred in this phase.

All native prefix states, observations, commands, pending queues and measured
pre/post gates reproduced Phase 40 exactly. GPU replay was not exact: maximum
prefix command differences were 0.001038 control and 0.014606 augmented.
These differences are recorded, not treated as native replay failures.

Saved-array verification passed: 2,485 protected inputs and 57 source/snapshot
pairs; frozen actor/normalizer and portable checkpoint parity, continuous
handoff, delayed torque application, pre/post gate thresholds and admitted
teacher-only records were checked. Native and private solver warnings were
zero; GPU plant warning counters remain unavailable. No new simulation was
used for offline verification. Teacher recovery does not change the latest
learned Stand result of 0/16 per actor.

## Phase 43 - Fresh current-recovery comparison: protocol (2026-10-03)

The valid Phase 42 labels permit the next bounded comparison under the user's
instruction to proceed. This protocol is recorded before training.
Independent runner: `compare_stand_current_recovery_supervision.py`.
Output: `logs/mimickit-training/stand-current-recovery-supervision-20261003-01`.

- Fresh seed **301013**, identical initial actor weights, independent empty Adam
  optimizers; neither arm loads a previously trained actor. Only the identical
  frozen Phase 32 normalization tensors are reused (59,040 observations, zero
  action samples, clip 10), with no refit.
- **180 seconds per arm**, sequential, final checkpoint only; Adam 0.0001,
  batch 512, gradient clip 1, full 33-coordinate normalized command MSE.
  Setup and evaluations are outside the budget. No PPO/discriminator updates.
- Control uses all 69,120 native pairs available to Phase 39's augmented arm:
  teacher 28,800; corrections 18,720; Phase 35 recovery 10,080; Phase 38 recovery
  11,520. This is a fresh control, not the old Phase 39 control checkpoint.
- Augmented adds the 10,080 Phase 42 native pairs, for 79,200 total.
  All GPU labels remain validation-only; no new AMP clips are generated.
- Control batch allocation: corrections 64; teacher 32 early + 32 uniform;
  Phase 35 recovery 64 early + 64 uniform; Phase 38 recovery 128 early + 128
  uniform. Augmented halves each allocation and adds Phase 42 recovery
  128 early + 128 uniform. Early means the first 120 frames; recovery sampling
  selects a generating actor/pose case uniformly, then a frame with replacement.
  This tests the added-data sampling recipe, not dataset size in isolation.
- Final autonomous evaluation: eight native and eight GPU original starts per
  actor, 12 seconds or physical FAIL, unchanged natural Stand gates, no teacher.
- Rank Stand passes first. If both arms have zero passes, call survival improvement
  consistent only if augmented minimum AND mean survival improve by at least
  one control interval (0.016668 s) on EACH backend and more paired starts improve
  than worsen on EACH backend. Also report first departure and longest valid
  posture interval. This practical screening rule is not statistical significance.
- If that criterion fails, close this intervention at the tested budget and
  return to a design decision. No automatic longer run or further diagnostic.
  Even a positive short result does not satisfy the 60-second or perturbation gate.

### Phase 43 results and decision

Completed both fresh arms with the predeclared seed and identical initial
weights/normalization. Control trained for 180.000695 seconds, making 82,502
Adam updates / 42,241,024 reused sample presentations. Augmented trained for
180.002362 seconds, making 75,578 updates / 38,695,936 presentations. All
available pairs for each arm were presented; control consumed zero Phase 42
pairs. These counts are reused supervision, not newly collected transitions.

Final autonomous results (eight starts per backend and arm):

- **Control native:** 0/8 Stand, minimum survival 0.633384 s, mean 0.825066 s,
  mean first departure 0.360446 s, mean longest valid posture interval 0.343777 s.
- **Control GPU:** 0/8 Stand, minimum 0.633384 s, mean 0.814648 s,
  mean first departure 0.360446 s, longest valid interval mean 0.343777 s.
- **Augmented native:** 0/8 Stand, minimum 0.766728 s, mean 0.897988 s,
  mean first departure 0.418784 s, longest valid interval mean 0.402115 s.
- **Augmented GPU:** 0/8 Stand, minimum 0.766728 s, mean 0.897988 s,
  mean first departure 0.416700 s, longest valid interval mean 0.400032 s.

All 32 autonomous starts eventually physically failed. Native mean survival
improved by 0.072923 s, GPU by 0.083340 s; minimum survival improved by 0.133344 s
on each backend. Six native pairs improved and two worsened; seven GPU pairs
improved and one worsened. Case 4 regressed by 0.466704 s native and 0.500040 s
GPU. Thus the predeclared intermediate survival rule **passes on both backends**,
while the 12-second learned Stand gate remains **0/16 per actor**.

Saved Phase 42 command RMSE decreased from 0.027717 to 0.002109 on native labels
and from 0.035770 to 0.031523 on separate GPU labels. These fitting measurements
support interpretation only; they are not the reason for the behavioral decision.
One fresh seed and the original reset poses do not establish statistical
significance, generalization or a training-speed multiplier.

Saved-artifact verification passed: 2,635 protected inputs, 58 source/snapshot
pairs, six portable/full checkpoint pairs, exact fresh initialization and frozen
normalization, unchanged critic/discriminator, optimizer-step/sample accounting,
source-label binding, and all 32 saved autonomous case outcomes. It checked
316,800 saved inference predictions and 1,649 live evaluation control intervals,
without simulation, network inference or optimization. Native solver warnings
were zero; GPU warning counters remain unavailable. No Arena promotion, springs,
physics/reward change, generic test suite or standalone compilation occurred.

Evidence: [comparison](../../logs/mimickit-training/stand-current-recovery-supervision-20261003-01/comparison.json),
[predeclared decision](../../logs/mimickit-training/stand-current-recovery-supervision-20261003-01/decision.json),
[verification](../../logs/mimickit-training/stand-current-recovery-supervision-20261003-01/final-verification.json).

**Decision:** provisional behavioral progress, not learned Stand. Keep the
coverage intervention open for one exact fresh-seed replication (301014), as
specified in Current direction. That replication is proposed, not executed.
No hour-long extension, new recovery collection or perturbations follow
implicitly from this result.

## Phase 44 - Fresh-seed replication: protocol (2026-10-03)

Authorized by the user's next "Proceed" after the Phase 43 result. Repeat the
exact Phase 43 comparison with **fresh seed 301014** (sampling seed 301715).
Runner: `replicate_stand_current_recovery_supervision.py`.
Output: `logs/mimickit-training/stand-current-recovery-replication-20261003-01`.

Both actors start from identical new weights and independent empty Adam state.
Retain 180 seconds per arm, the same frozen Phase 32 normalization, the same
69,120 control / 79,200 augmented native pairs, batch 512, learning rate 0.0001,
stratum allocation and final autonomous 8-native/8-GPU evaluations per arm.
Only initialization/sampling seed changes; Phase 43 artifacts are protected
inputs for comparison, never actor initialization. No new recovery collection.

Apply the exact Phase 43 survival rule, report every paired result and compare
both seeds. If zero Stand passes and the survival rule fails, report mixed-seed
evidence and return to a design decision. No automatic seed sweep, extended
training, perturbations or Arena promotion is included.

### Phase 44 results and cross-seed decision

Completed both arms with fresh seed 301014. Control: **180.001949 seconds**,
75,711 Adam updates / 38,764,032 reused presentations. Augmented:
**180.002370 seconds**, 75,913 updates / 38,867,456 presentations. Each arm saw
all its available pairs; control saw zero Phase 42 pairs. Optimization collected
no new physics transitions. Equal-time budgets produced almost equal update
counts in this replication; they were not forced to match.

Final autonomous evaluations:

- **Control native and GPU:** each 0/8 Stand, minimum survival 0.733392 s,
  mean survival 0.860486 s, mean first departure 0.379197 s,
  mean longest continuously valid balance/posture interval 0.362529 s.
- **Augmented native:** 0/8 Stand, minimum 0.700056 s, mean 0.806314 s,
  mean first departure 0.366696 s, longest valid interval mean 0.350028 s.
- **Augmented GPU:** 0/8 Stand, minimum 0.700056 s, mean 0.810481 s,
  mean first departure 0.364613 s, longest valid interval mean 0.347944 s.

All 32 starts physically failed. Native mean survival changed by **-0.054171 s**,
GPU by **-0.050004 s**; minimum survival decreased **0.033336 s** on both.
Only two native pairs improved (six worsened) and three GPU pairs improved
(five worsened). The exact predeclared survival criterion **fails on both**.

Paired survival differences, augmented minus control, by original case 0-7:

- Native: -0.050004, -0.050004, -0.150012, -0.050004, +0.183348,
  +0.133344, -0.050004, -0.400032 seconds.
- GPU: -0.050004, +0.016668, -0.166680, -0.050004, +0.183348,
  +0.133344, -0.050004, -0.416700 seconds.

Case 4, which regressed in Phase 43, improved in this replication. Case 7 had
this seed's largest regression. The failure is not consistently confined to the
same reset pose across seeds.

Native Phase 42 saved-command RMSE nevertheless improved **0.027299 -> 0.002182**;
separate GPU RMSE improved **0.035715 -> 0.030992**. The improved recorded-state
fit again did not imply improved autonomous behavior.

Verification passed: **2,840 protected inputs**, **59 source/snapshot pairs**, six
portable/full checkpoint pairs, all 32 autonomous outcomes, 1,602 live evaluation
control intervals and 316,800 saved inference predictions. It also checked that
all protocol settings/data match Phase 43 apart from seed and provenance,
normalization is identical across seeds, fresh actor weights differ across
seeds, and both arms share the same new initial weights. Source comparison
confirmed only the seed, documentation and provenance roots changed. No new
simulation, inference or training was performed by offline verification.
Native solver warnings were zero; GPU warning counters remain unavailable.

Evidence: [replication comparison](../../logs/mimickit-training/stand-current-recovery-replication-20261003-01/comparison.json),
[decision](../../logs/mimickit-training/stand-current-recovery-replication-20261003-01/decision.json),
[cross-seed comparison](../../logs/mimickit-training/stand-current-recovery-replication-20261003-01/cross-seed-comparison.json),
[verification](../../logs/mimickit-training/stand-current-recovery-replication-20261003-01/final-verification.json).

**Decision:** seed 301013's intermediate gain did not replicate in seed 301014.
Across the two comparisons there are **0/64 autonomous Stand passes**. Close
this coverage intervention at the tested budget and return to the design decision
in Current direction. No seed sweep, further teacher collection, longer run,
perturbation training or Arena promotion was launched. Historical sources and
artifacts remain preserved; no generic tests or standalone compilation ran.

## Design decision after Phase 44 (2026-10-03)

Completed the requested source/document review. No training, simulation, tests,
compilation, controller implementation or production configuration change ran.

### Findings that determine the next comparison

1. **Recent trials optimized saved torque labels, not standing returns.**
   `compare_stand_current_recovery_supervision.py` samples fixed observations
   and minimizes normalized 33-coordinate command MSE with Adam. It collects
   no states during optimization and leaves critic/discriminator unchanged.
   The actor does observe state at deployment, so it is a feedback policy;
   the limitation is that its fitting objective does not optimize the state
   trajectory resulting from its own errors. `policy.py` maps normalized
   observations through an MLP to bounded commands, with no predictive rollout.
2. **The teacher supplies structure that command regression need not recover.**
   `PredictiveStandController` predicts through both pending commands, forms
   whole-body acceleration feedback and solves bounded inverse-contact control
   with forward refinements. Its successful holds establish physical feasibility,
   not successful transfer of that control law to the neural actor.
3. **PPO itself has already been tried.** Phase 19's fresh thirty-minute
   direct-torque PPO/AMP run reached about 1.950 seconds native mean survival,
   with 0/8 Stand at every assessment. Therefore merely returning to the old PPO
   settings is not a new solution. It is an appropriate simultaneous control arm.
4. **An existing PD path is incompatible with the current contract.**
   `target_control.py` requires 30 actions, stores 61 values per queue slot,
   and uses fixed slices 30/60. `newton_adapter.py` also hard-codes 30 motors,
   velocity offset 30 and valid flag 60 in its PD kernel. `TorqueStandTask`
   explicitly selects torque mode and requires all 33 motors. The legacy
   `scripts/config.ps1` target-PD defaults do not govern the recent standalone
   imitation runners. Do not flip that flag or load old 30-action checkpoints.
5. **Active feedback and passive springs are separate design choices.** Current
   torque commands wait two control intervals and are held during the physics
   substeps. Target PD instead delays the desired angles, then recomputes bounded
   motor torque from current joint position/velocity every physics step. This
   changes local feedback timing and action meaning, while leaving the mechanical
   plant unchanged. It is not a pure action rescaling or an isolated PPO setting.

### Selected hypothesis

Test whether a fixed local joint feedback loop plus learned target corrections
allows fresh PPO to improve autonomous natural standing within the short budget.
Keep the existing reference_v9/AMP objective in both arms so reward changes are
not mixed with the control comparison. This hypothesis is supported as a useful
alternative by the implementation contrast, not proven by the existing results.
Do not restart offline recovery-data accumulation or change passive stiffness.

A predictive-teacher residual controller would confound success with a teacher
that already stands. Changing rewards, controller gains and representation in
one comparison would obscure the outcome. Both are deferred.

### Proposed bounded representation comparison; not executed

**Plant and target controller:** use the same spring-free 33-motor model and
passive-wrist reference as Phase 44. Preserve gravity, contacts, damping, limits,
force authority, reset poses, two-interval command delay and four physics substeps.
All six wrist hinges remain unactuated. No root force, body anchor, foot-position
constraint, inverse-contact teacher or gravity feedforward is added.

For the target arm, normalized action a defines a joint target sampled at issue:

- q_target = clip(q_reference + 0.25 * clip(a, -1, 1), joint_lower, joint_upper).
- v_target = 0 for the stationary reference.
- After the existing two-interval delay, every physics substep applies
  tau = torque_limit * clip(4 * (q_target - q) + 0.08 * (v_target - v), -1, 1).
- Queue targets and validity flags; begin with invalid slots and zero motor torque.
  No action or feedback bypasses the documented startup queue semantics.

The 0.25-radian range, normalized gain 4 and damping time 0.02 reproduce the
existing local-PD formula as a predeclared starting choice. They are not validated
optimal gains for this 33-motor rig. Keep them fixed during this screen. Mark
active pose feedback explicitly in the new contract; passive stiffness stays zero.

**Matched network inputs:** both arms expose 374 observations: 95 physical
features, 134 queue features, then the same 145 reference/phase features. Target
queues contain two slots of 33 positions + 33 velocities + one validity flag.
The torque arm places its existing 66 pending torque values first in the queue
region and fills the remaining 68 values with zeros. This preserves all torque
information while allowing identical fresh actor/critic dimensions and tensors.
AMP physical features remain unchanged. Version both observation contracts;
existing 306-input checkpoints and Arena adapters cannot load these actors.

**Implementation scope:** use isolated controller, task and GPU-adapter modules
and a bounded paired runner. Derive motor counts/indices from the rig. Preserve
sealed historical sources. Reuse physical measurements and reward computation;
implement matching native/GPU substep control and partial-reset queue behavior.
Do not route a 33-channel array through the old hard-coded kernel. Save distinct
control/observation contracts and source snapshots for each arm.

**Controller baseline and admission:** before training, evaluate zero action for
both control modes on eight native and eight GPU starts, up to 12 seconds or
physical FAIL (32 cases). Zero action means zero torque in the torque arm and
reference targets in the PD arm; this difference must be reported. Check native
and GPU formulas on identical saved q/v/target arrays, motor bounds, finite state,
queue timing, reference sampling and partial-reset isolation. These checks belong
to the proposed experiment; none ran during this review. Stop for contract,
nonfinite, native solver-warning or timing failures; a finite physical fall is a
behavioral result, not proof of an implementation fault. GPU warning availability
must be reported accurately.

If zero-action PD already passes all 16 short Stand cases, stop before PPO and
report controller-provided standing. A subsequent 60-second validation and the
purpose of a learned policy must then be decided explicitly. Do not call that
outcome learned Stand or proceed directly to perturbations.

**Fresh learning comparison if the baseline does not already solve Stand:**

- Seed 301015 for both arms; fresh actor, critic, discriminator, normalizers,
  optimizer state and sampling streams. Verify identical initial full tensors.
  Normalizers may subsequently diverge with the arms' different visited states.
- 180 synchronized training-update seconds per arm, sequential, excluding setup
  and evaluation. Finish the current update and report any budget overshoot.
- 64 environments x 64 rollout frames = 4,096 transitions per PPO update.
  Actor SGD 1e-4, three epochs, batch multiplier 4; critic SGD 1e-3, two epochs,
  multiplier 2. Fixed action sigma 0.1, PPO clip 0.2, discount 0.99, lambda 0.98.
- Use Phase 19's archived agent configuration for all other settings, including
  80% reference_v9 task / 20% AMP reward, discriminator settings and networks.
  Use the same admitted native teacher/correction AMP clips; no Phase 42 torque
  labels or prior learned weights. Identical normalized sigma does not imply
  identical physical exploration after the different control mappings.
- Keep 60-second training episode limits. Evaluate initial and final deterministic
  actors only on eight native + eight GPU original starts, up to 12 seconds or
  physical FAIL. Disable periodic assessments explicitly for this equal-time
  screen. Baseline plus initial/final assessments total at most 96 cases.
- Save the final elapsed-budget checkpoint, updates, transitions, wall time,
  all cases, source hashes, zero-action results and per-substep controller traces.
  No best-checkpoint selection, parameter sweep or Arena export.

**Decision rule:** rank final Stand passes first. A learned-controller gain must
beat both the matched direct-torque actor and its own zero-action PD baseline.
If there are no Stand passes, require minimum and mean survival to improve by
at least one control interval on each backend and more paired cases improving
than worsening against each comparator. Also report first departure and longest
valid posture interval. If there is no such gain, close this chosen formulation
at this budget; do not tune gains or extend time automatically. A positive short
result requires fresh-seed replication before considering a longer learning curve.
Controller-only standing, one learned pass and reliable 16/16 Stand are distinct
outcomes. The 60-second natural-standing gate still precedes perturbations.

## Phase 45 - Matched control-representation screen (2026-10-03)

The user authorized implementation and execution of the documented bounded
representation comparison with "Proceed". Isolated modules:
`target_control33.py`, `representation_engine.py`, `representation_task.py`,
and `compare_stand_representations.py`. No sealed historical source is edited.
Both new observation contracts have 374 inputs and 33 actions. Native and GPU
adapters implement the same rig-sized feedback, with evaluation substep tracing.

Run destination: `logs/mimickit-training/stand-representations-20261003-01`.
Admission precedes training: formula/queue/reset checks, then 32 zero-action
native/GPU cases. A 16/16 zero-action PD Stand result stops before PPO. Otherwise
run seed 301015, 180 synchronized update seconds per arm, with the Phase 19
agent configuration, no periodic evaluations, and 32 initial + 32 final cases.
All planned gates and stopping criteria are those in Design decision after
Phase 44. No gains, physical parameters, rewards, springs or Arena selection
change beyond the approved isolated active-feedback representation.

### Phase 45 execution and results

The first launch (`stand-representations-20261003-01`) stopped before any
simulation or training: the standalone formula probe omitted `warp.init()`
before Torch/Warp CUDA interoperation. Its original source snapshots and failure
record are retained. Adding that initialization was the only startup correction;
no gain or experiment parameter changed. The accepted run is
`logs/mimickit-training/stand-representations-20261003-02`.

Admission passed. Shared-input CPU/GPU formula discrepancy was zero in the
no-physics probe. All evaluation substep torques matched their delayed command
or active target formula within the predeclared 0.0002 Nm tolerance. Startup
invalid targets produced zero torque, targets were sampled at issue, and queued
actions took effect after two intervals. Partial reset checks preserved other
worlds' states, controls, queues and contact measurements. Native warnings were
zero; GPU solver-warning measurement remains unavailable.

Zero-action baselines on both backends:

- Torque: **0/8 Stand**, minimum/mean survival **0.316692 / 0.339610 s**;
  mean longest balanced/natural-posture interval **0.033336 s**.
- Target PD: **0/8 Stand**, minimum/mean survival **1.833480 / 1.895985 s**;
  mean first departure **1.041750 s**, mean longest valid interval **1.025082 s**.

Because PD did not already solve Stand, both fresh PPO arms ran. Full initial
actor, critic, discriminator and normalizer tensors matched exactly, including
374-input network dimensions and the same normalized action standard deviation.
No historical weights or fitted normalizers were loaded. The active local
controller differs by design; identical action sigma does not imply identical
physical exploration in torque and angle-target units.

- Torque: **181.143493 synchronized training seconds**, **76 PPO updates**,
  **311,296 transitions**. The last complete update accounts for the overshoot.
- Target PD: **180.665364 seconds**, **68 PPO updates**, **278,528 transitions**.
- No intermediate assessments, supervised optimization, reward/gain changes or
  checkpoint selection by best score occurred. Each final checkpoint is the
  final elapsed-budget checkpoint.

Final autonomous outcomes:

- **Torque native:** 0/8 Stand, minimum **0.950076 s**, mean **1.262601 s**.
- **Torque GPU:** 0/8 Stand, minimum **0.950076 s**, mean **1.256351 s**.
- **Target PD native and GPU:** each 0/8 Stand, minimum **2.100168 s**,
  mean **2.902316 s**, mean first balance/posture departure **1.939738 s**,
  mean longest continuously valid interval **1.923070 s**.

Every target-PD final start improved survival against both its zero-action PD
baseline and the paired trained torque actor on each backend. Mean gains over
zero-action PD were **1.006331 s** and minimum gains **0.266688 s**. Compared with
torque, mean gains were **1.639715 s native / 1.645965 s GPU**, with minimum gain
**1.150092 s**. All four comparisons satisfy the predeclared intermediate rule.

The target-PD actor's own untrained initial mean was **1.918903 s** (minimum
1.850148 s, longest valid interval mean 1.045917 s). Its final result therefore
also improves beyond its random initial policy, not merely beyond zero torque.
All 96 baseline/initial/final cases physically failed before 12 seconds; no
learned Stand admission, 60-second hold or perturbation readiness is established.
Identical native/GPU aggregate PD scores do not claim bit-exact physics replay.

Offline verification passed **3,052 protected inputs**, **63 source/snapshot
pairs** and **132 completion hashes**, including every delayed queue and recorded
physics-substep torque, pre/post gate reconstruction, partial resets, formula
probe, checkpoint sidecars, shared fresh initialization, PPO transition/update
counts, budget cutoff and comparison decisions. It checked 96 assessment cases
and **8,309 live assessment intervals**, with no additional physics, network
inference or training. Production `config.ps1`, old sources, physics assets,
rewards and Arena selections remain unchanged. Warp compiled the new CUDA
kernel for the approved experiment; no standalone build or general test suite ran.

Evidence: [comparison](../../logs/mimickit-training/stand-representations-20261003-02/comparison.json),
[full results](../../logs/mimickit-training/stand-representations-20261003-02/report.json),
[verification](../../logs/mimickit-training/stand-representations-20261003-02/final-verification.json).

**Decision:** this control formulation shows a larger short-run behavioral gain,
including longer valid natural posture, but remains a one-seed result with
**0/16 learned Stand passes per actor**. Propose one exact fresh-seed replication
(301016) as specified in Current direction. No further training ran after this
screen, and no speed multiplier, convergence or unseen-pose claim is made.

## Phase 46 - Fresh-seed representation replication (2026-10-03)

Authorized by the user's "Proceed" after the Phase 45 result. Repeat the exact
screen with seed **301016**, sampling seed **301717**, and 180 synchronized PPO
training seconds per arm. Runner: `replicate_stand_representations.py`.
Output: `logs/mimickit-training/stand-representations-replication-20261003-01`.

Only initialization/sampling seed and provenance/output roots change. Reuse the
unchanged admitted 33-motor controller, native/GPU engines, 374-input tasks,
plant, reference, agent settings, AMP clips, baseline and assessment protocol.
No trained weights or fitted normalizers initialize either arm. Repeat formula
and partial-reset checks, all 32 zero-action cases, and the 16/16 zero-action PD
stop before training. Otherwise run both fresh arms and their 64 initial/final
cases. Evaluate the same gain rule against both comparators on each backend.
No gain sweep, longer run, perturbations or Arena promotion is included.

### Phase 46 results and replication decision

Completed the exact screen with seed 301016. All settings and datasets match
Phase 45; the runner differs only in its description, fresh seed and provenance
roots. Both new arms have identical full initial tensors; those tensors differ
from Phase 45's seed 301015. No previously trained network or fitted normalizer
was loaded. Formula/queue/partial-reset admission passed again.

Zero-action baselines reproduced the prior aggregate results on both backends:
torque mean/minimum survival **0.339610 / 0.316692 s**; target-PD
**1.895985 / 1.833480 s**. Both have 0/16 Stand passes. The target-PD actor's
untrained initial mean also measured **1.895985 s** on both backends.

Training:

- Torque: **181.508231 seconds**, **72 PPO updates**, **294,912 transitions**.
- Target PD: **181.362421 seconds**, **106 PPO updates**, **434,176 transitions**.

These are equal requested time budgets, not equal transition/update budgets.
The target-PD arm completed more updates in this replication; it completed fewer
than torque in Phase 45. Do not attribute the cross-seed score difference solely
to the seed, or claim a constant throughput/training-speed multiplier.

Final autonomous results:

- **Torque native:** 0/8 Stand, minimum **0.950076 s**, mean **1.181344 s**.
- **Torque GPU:** 0/8 Stand, minimum **0.950076 s**, mean **1.183428 s**.
- **Target PD native:** 0/8 Stand, minimum **2.450196 s**, mean **3.254427 s**,
  mean first departure **2.333520 s**, mean longest valid posture interval
  **2.316852 s**.
- **Target PD GPU:** 0/8 Stand, minimum **2.450196 s**, mean **3.256510 s**,
  mean first departure **2.335604 s**, mean longest valid posture interval
  **2.318935 s**.

Every target-PD start improves survival against both the trained torque actor
and zero-action PD on both backends. The minimum gain over zero-action PD is
**0.616716 s**, with mean gain **1.358442 s native / 1.360525 s GPU**. Against
torque, minimum gain is **1.500120 s** and mean gain **2.073083 s** on each
backend. All four comparisons pass the unchanged intermediate decision rule.
All 96 baseline/initial/final cases still physically failed before 12 seconds.

This confirms the direction of Phase 45's short-screen result in a second
fresh seed: trained target-PD means of **2.902 s** and **3.254-3.257 s**, both
above the **1.896 s** zero-action PD baseline and their paired torque arms.
It does not establish convergence, unseen-pose robustness, 60-second Stand,
or perturbation recovery. Each final actor remains **0/16 learned Stand**.

Saved-artifact verification passed **3,251 protected inputs**, **64 source/
snapshot pairs**, **133 completion hashes**, all 96 assessment outcomes and
**8,558 live assessment intervals**. It verified matching recipes/data, source
changes limited to seed/provenance, fresh cross-seed weight differences,
within-seed initial equality, sidecars, gates, queues, all recorded substep
torques, budgets and decision calculations. No additional simulation, network
inference or training occurred during verification. Native warnings remained
zero; GPU solver-warning counters remain unavailable. Controller sources,
production defaults, passive stiffness, rewards and Arena selections are unchanged.

Evidence: [replication comparison](../../logs/mimickit-training/stand-representations-replication-20261003-01/comparison.json),
[cross-seed comparison](../../logs/mimickit-training/stand-representations-replication-20261003-01/cross-seed-comparison.json),
[verification](../../logs/mimickit-training/stand-representations-replication-20261003-01/final-verification.json).

**Decision:** short-run behavioral gain replicated. Propose the single fresh,
at-most-ten-minute target-PD learning-curve run with the explicit checkpoint
and early-stop rules in Current direction. It is not executed by Phase 46.
No further training, gain sweep, export or perturbations ran after replication.

## Phase 47 - Fresh target-PD duration curve (2026-10-03)

Authorized by the user's Proceed after Phase 46. The bounded protocol used
fresh seed 301017, sampling seed 301718, assessments at
initialization and the first completed PPO updates crossing 180/360/600 seconds.
Only synchronized training iterations count against the budget. Each assessment
uses eight original starts on each backend, up to 12 seconds or physical FAIL.
Stop immediately at an assessment with 16/16 Stand passes. At 360 seconds,
continue to 600 only if each backend improves over 180 seconds: more Stand
passes, or equal Stand passes together with minimum survival, mean survival
and mean longest valid posture interval each gaining at least one control
interval (0.016668 seconds). Otherwise stop at 360. Stop on nonfinite training/
state, contract failure or native solver warnings. Keep the Phase 45-46 physical
plant, reference, AMP clips, gains, objective, PPO settings and fresh
initialization unchanged. No torque arm or recovery-label collection is included.

Implementation: `measure_stand_learning_curve.py` reuses the sealed Phase 46
assessment functions and representation components. It saves each actor/full AMP
pair and verifies that evaluation leaves learner tensors, optimizer state and
the sampled training-world state unchanged. Python, NumPy, Torch CPU/CUDA and
task/demo RNG states are restored, with before/after evidence saved. No trained
checkpoint or fitted normalization initializes the run. Configuration, model,
reference, demonstrations and all historical protected inputs are hash-bound.

`verify_stand_learning_curve.py` recomputes saved gates, control traces, checkpoint
hashes, update counts and stopping decisions without further simulation or
inference. The final result is the last planned checkpoint reached. No automatic
60-second assessment, Arena export, perturbations or extension is included.

Output: `logs/mimickit-training/stand-target-pd-curve-20261003-01`.

### Phase 47 results and duration decision

Completed and saved-artifact verification passed. A startup import-order error
was corrected before output-directory creation, simulation or training; no
training budget or candidate selection was affected. The accepted run used
the source snapshots in its protocol and completed without solver warnings
on native assessments. GPU solver-warning counters remain unavailable.

- **Fresh initialization:** 0/16 Stand; minimum survival 1.833480 seconds and
  mean 1.895985 seconds on both backends. Mean longest valid posture interval
  was 1.018832 seconds on both. All observation/action normalizer counts were
  zero and initial weights differed from Phase 46.
- **Three-minute checkpoint:** 181.135912 synchronized training seconds,
  109 PPO updates, 446,464 transitions. Native/GPU Stand 0/8 each, minimum
  survival 2.383524 seconds each, mean survival 3.171087 / 3.173170 seconds.
  Mean first departure was 2.256430 seconds and mean longest valid posture
  interval 2.239763 seconds on both backends.
- **Six-minute final checkpoint:** 360.177786 synchronized training seconds,
  223 PPO updates, 913,408 transitions. Native/GPU Stand 0/8 each, minimum
  survival 2.616876 seconds each, mean survival 3.058578 / 3.060661 seconds.
  Mean first departure was 2.189759 / 2.193925 seconds; mean longest valid
  posture interval 2.173091 / 2.177258 seconds.

The six-minute checkpoint improved minimum survival by 0.233352 seconds, but
mean survival declined by 0.112509 seconds on both backends. Mean longest valid
posture declined by 0.066672 native / 0.062505 GPU seconds. Neither backend
passed the predeclared continuation rule, so training stopped automatically
at six minutes. No ten-minute checkpoint or 60-second assessment was run.
The final checkpoint is `seconds-360/model.pt`; the earlier checkpoint is
preserved as an intermediate measurement, not substituted as the final result.

Verification covered 3,454 protected inputs, 66 source/snapshot pairs,
110 completion hashes, all 48 assessment cases and 7,802 live intervals.
It recomputed gates, delayed-target queues and every recorded substep torque,
checked portable/full AMP checkpoint pairs and fresh initialization, and
recomputed the training budgets and stop decision. All three assessments
preserved learner tensors, optimizer state and recorded training-world state;
all tracked RNG streams were restored. Verification made no additional physics
steps, network forward passes or training updates.

**Decision:** early learned benefit remains, but the first duration extension
produced a mixed result and still 0/16 autonomous Stand. Do not infer that
twice the training time yields twice the learning, that this is a global
convergence limit, or that all seeds would plateau here. Diagnose existing
failure traces before proposing another configuration or longer compute run.
Arena selections, production configuration and passive stiffness are unchanged.

Evidence: [learning curve](../../logs/mimickit-training/stand-target-pd-curve-20261003-01/comparison.json),
[continuation decision](../../logs/mimickit-training/stand-target-pd-curve-20261003-01/continuation-decision.json),
[verification](../../logs/mimickit-training/stand-target-pd-curve-20261003-01/final-verification.json).

## Phase 48 - Saved duration-failure diagnosis (2026-10-03)

Steps 1-4 of the autonomous sequence are complete. The saved artifacts and
their protected inputs passed hash checks before and after analysis. No new
physics, inference or learning occurred. `audit_stand_duration_failures.py`
records every original start at initialization, three minutes and six minutes
on both backends in `stand-duration-diagnosis-20261003-01/report.json`.

Horizontal speed is the first failed gate in **46/48 cases**; pelvis tilt is
first in the other two (three-minute case 7, both backends). No motor reaches
99% of its torque bound in the preceding 31 control frames, approximately
half a second. The audit records per-motor proximity to joint limits rather
than assuming torque headroom rules out every authority limitation.

From three to six minutes, six original starts improve survival on each
backend, but cases 4 and 7 regress: native changes -0.517 and -1.150 seconds.
This distribution explains the improved minimum but reduced mean. All starts
still fail, and gate departure consistently precedes physical collapse.

The last 20 updates at three/six minutes have mean PPO clip fractions about
0.045/0.049 and importance ratios about 0.9994/0.9997; critic loss decreases
from 7.38 to 3.82. Mean discriminator reward stays near 0.295, with agent/demo
classification accuracy near 99%. These logs provide no obvious explosion
signal, but neither prove optimizer adequacy nor identify a causal bottleneck.
Training episode length increases while deterministic assessment mean falls.

Source inspection shows reference_v9's root-velocity component has weight 0.1
and scale 1. With other velocity errors zero, reaching the 0.2 m/s horizontal
speed gate reduces task reward by only **0.003921**. This is a mathematical
sensitivity comparison, not a reconstruction of the total reward on live
frames. The task reward has other changing terms and AMP is weighted separately.

**Hypothesis:** a stronger quiet-speed preference may improve the feedback
learned before drift develops into a fall. This motivates one reward ablation;
it does not establish that the existing reward caused the plateau or justify
changing motor stiffness, actuator authority or PD gains.

## Phase 49 - Matched-update quiet-speed reward comparison (2026-10-03)

Steps 5-6 are complete. Predeclared before execution:

- Two fresh seeds, **301018 and 301019**, sampling seeds +701. No checkpoint,
  fitted normalization or optimizer reuse. Within each seed, both arms must
  have identical full initial weights; the seeds must differ from each other.
- Reference arm: unchanged Phase 47 target-PD/AMP recipe. Candidate arm:
  multiply the reference_v9 task reward by
  `exp(-sum(root_velocity_xy**2) / 0.2**2)` using post-physics root velocity.
  This factor is 1 at rest and about 0.368 at 0.2 m/s. Task/AMP mixture remains
  0.8/0.2; AMP itself is unchanged. Physical failure still zeros task reward.
- All controller gains, passive stiffness (zero), limits, physics, resets,
  reference, demonstrations, observations, PPO settings and evaluation gates
  remain unchanged. This is a training reward change only.
- Each arm receives **100 PPO updates = 409,600 transitions**, with 64 worlds
  and 64 rollout steps. Report actual synchronized training time. Equal update
  budgets remove the earlier equal-time update-count confound; they do not
  establish equal wall-clock efficiency. A 300-second training safety cap per
  arm aborts an incomplete comparison, rather than extending or comparing
  unequal update counts. Total planned learning is 400 updates; no continuation.
- Seed 301018 runs reference then candidate. Seed 301019 reverses that order.
  Evaluate fresh and final actors on eight native/eight GPU original starts,
  up to 12 seconds or physical FAIL. Save all four fresh/final checkpoint pairs
  and the 128 assessment cases. Assessments are excluded from training time.
- Candidate wins a seed only on BOTH backends: more Stand passes, or equal
  passes plus minimum survival, mean survival and mean longest continuously
  valid posture each increasing by at least one control interval (0.016668 s),
  with more paired survival improvements than regressions. Replicated gain
  requires both seeds to win. Final budget actors determine results.
- Record base/shaped reward, factor and speed statistics for every rollout
  frame. Check reference identity and candidate formula application during
  training, preserve evaluation RNG isolation, and recompute all saved gates,
  queue/torque traces, checkpoint pairs, budgets and decisions. Verify seed 1
  before proceeding to seed 2. No standalone tests or builds are included.

Implementation: `compare_stand_quiet_speed.py` and
`verify_stand_quiet_speed.py`; historical runners/controllers are sealed and
unchanged. Planned output: `logs/mimickit-training/stand-quiet-speed-20261003-01`.
Execution and verification completed. Perturbations and Arena promotion remain gated.

### Phase 49 results and autonomous-batch decision

All four arms completed exactly 100 PPO updates and 409,600 transitions, with
no safety-cap abort. Total learning was **400 updates / 1,638,400 transitions**
and **691.389396 synchronized training seconds** (about 11.5 minutes), excluding
setup, checkpoint saving, assessment, verification and other orchestration.

Seed 301018, reference then quiet-speed:

- Reference: **169.712402 training seconds**, 0/16 Stand, minimum survival
  **2.350188 s** on both backends, native/GPU mean **3.029409 / 3.031492 s**,
  mean longest valid posture **2.091834 / 2.093918 s**.
- Quiet-speed: **182.240892 training seconds**, 0/16 Stand, minimum survival
  **2.350188 s**, mean **2.810641 s** and longest valid posture **1.920987 s**
  on both backends.
- Candidate mean survival declined by **0.218768 native / 0.220851 GPU s**;
  minimum did not improve. Two starts improved, four regressed and two tied on
  each backend. The candidate failed the predeclared rule. The seed's saved
  verification passed before seed 301019 started.

Seed 301019, quiet-speed then reference:

- Quiet-speed: **169.032899 training seconds**, 0/16 Stand, minimum survival
  **2.466864 s**, mean **3.644041 s** and longest valid posture **2.662713 s**
  on both backends.
- Reference: **170.403202 training seconds**, 0/16 Stand, minimum survival
  **2.350188 s**, mean **3.402355 s** and longest valid posture **2.466864 s**
  on both backends.
- Candidate minimum survival improved by **0.116676 s**, mean by **0.241686 s**
  and longest valid posture by **0.195849 s** on both backends. Seven starts
  improved and one regressed; the regression was **1.850148 s** on case 4.
  This seed passed the predeclared aggregate rule, without establishing
  uniform improvement across starts.

The direction reverses between two fresh seeds at equal update/transition
budgets. **Replicated gain is false.** Do not adopt the quiet-speed multiplier,
choose seed 301019 as proof of success, or infer that reward shaping generally
cannot help. Retain the unchanged target-PD recipe as the comparison control.
All 128 initial/final assessment cases still physically failed before 12 seconds.
No actor has passed autonomous Stand, sustained Stand or perturbation readiness.

Final saved-artifact verification passed **3,572 protected inputs**, **69
source/snapshot pairs**, **195 completion hashes**, all **128 cases** and
**19,695 live intervals**. It verified equal fresh initial weights within each
seed, different weights across seeds, fresh normalizers, actor/full AMP pairs,
100-update budgets, reference reward identity, candidate formula checks and
recorded reward summaries, evaluation isolation/RNG restoration, all physical
gates and delayed substep controls, and both decision calculations. Native
solver warnings were zero; GPU warning counters remain unavailable. No new
physics, inference or learning occurred during saved-result verification.

The completed ten-step batch produced a narrower diagnosis and a rejected
replication claim, not a solved behavior. The next proposed mechanism check is
specified in Current direction. No further simulation or training was launched.
Historical sources/artifacts, passive springs, production defaults and Arena
selections remain unchanged.

Evidence: [diagnosis](../../logs/mimickit-training/stand-duration-diagnosis-20261003-01/report.json),
[matched comparison](../../logs/mimickit-training/stand-quiet-speed-20261003-01/comparison.json),
[verification](../../logs/mimickit-training/stand-quiet-speed-20261003-01/final-verification.json).

## Phase 50 - Changing policy commands versus initial offsets (2026-10-05)

Authorized within the twenty-step batch. Execute the previously proposed
128-case diagnostic: all four final Phase 49 actors, live versus held first
deterministic action, eight original starts on native and GPU, 12 seconds or
physical FAIL. No optimization, fitted normalization changes or checkpoint
selection. The same active local PD controller and two-interval delay operate
in both conditions; passive springs remain zero.

`probe_stand_policy_feedback.py` reuses the sealed evaluation function. The
held-action wrapper caches a cloned first output and returns independent copies
so masking dead worlds cannot change live commands. Actor hashes must remain
unchanged. Saved traces must show identical first actions across conditions
and constant held actions for every live frame. Live per-start survival must
repeat Phase 49 within one control interval; otherwise stop and diagnose
repeatability before interpreting an ablation. Saved gates and substep torques
are recomputed using `stand_experiment_evidence.py`.

Output: `logs/mimickit-training/stand-policy-feedback-20261005-01`.
Execution completed; attempt and accepted-repeat results follow. No inference about general feedback sufficiency is made
from this comparison; local motor feedback is explicitly retained.

Phase 50 startup/verification record: a syntax error was corrected before any
output creation or simulation. Attempt `stand-policy-feedback-20261005-01`
then completed the first actor's 32 cases but stopped on bitwise equality of
initial GPU actions between independently constructed worlds. The observed
maximum differences were 1.192093e-7 in initial observations and 2.980232e-8
in actions; native differences were zero. No physical/solver failure caused
this verification stop. Preserve the attempt and original source snapshots.

Before repeating, declare absolute cross-rollout tolerances of 2e-6 for initial
observations and 1e-6 for initial actions, with zero relative tolerance. Held
actions must still be exactly constant within each live rollout. Other gates,
control checks and live-survival repeatability requirements remain unchanged.
The isolated successor `probe_stand_policy_feedback_repeat.py` binds the failed
attempt's artifacts and runs the full 128-case comparison anew under
`stand-policy-feedback-20261005-02`. Total diagnostic accounting includes the
32 interrupted-attempt cases; they are not substituted for accepted cases.

Phase 50 accepted repeat completed. Native mean survival, live versus held:
301018/reference **3.029409 / 2.223095 s**; 301018/quiet-speed
**2.810641 / 2.291850 s**; 301019/quiet-speed **3.644041 / 2.443946 s**;
301019/reference **3.402355 / 2.562705 s**. GPU results have the same direction.
All live cases repeated their sealed per-start survival within the declared
tolerance. All 128 accepted cases, actor hashes, held commands, saved gates
and substep controls passed verification. No training occurred. This supports
a contribution from changing policy commands for all four actors' mean
survival, but does not establish a sufficient stabilizing feedback law.

## Phase 51 - Frozen-policy input sensitivity and implementation audit (2026-10-05)

Steps 8-9: inspect 16 saved native observations per actor (initial and last
frame before first gate departure for each of eight starts), all four Phase 49
actors. Reproduce saved actions within 1e-6; compute input Jacobians weighted
by the actor's fitted observation standard deviations, grouped into physical
state, queued targets and reference inputs. Query symmetric +/-0.02 m/s root
horizontal-velocity changes on these saved observations. This is a local
network query, not a physical perturbation or evidence of closed-loop recovery.
No fitted tensors or optimizer states change, and no simulation is run.

Audit source behavior for stationary phase, reference inputs, normalization,
PPO update size and actor optimizer clipping. `audit_stand_feedback_inputs.py`
saves selected observations, derivatives, finite query responses and input/
source hashes under `stand-feedback-inputs-20261005-01`. Policy weights must
remain unchanged; all derived values must be finite. Results pending.

Phase 51 completed: 64 saved-state actions reproduced, all derivatives/queries
were finite and actor hashes stayed unchanged. A +/-0.02 m/s velocity query
produced maximum target half-differences of **0.000352-0.000373 radians** across
the four actors. Root pose/velocity and reference inputs were unclipped on the
saved live trajectories; queued-target clipping fractions were about
0.000034-0.000044. No stationary phase clock was present (phase always zero).
Jacobian group energy roughly followed group dimensionality; it must not be
interpreted as proof that reference/queue inputs dominate behavioral control.
Finite sensitivity and the Phase 50 ablation support some useful policy
dependence, not a sufficient stabilizing controller.

The pinned optimizer is SGD with momentum 0.9 and gradient-norm clipping 1.0.
At 64 environments, 64 rollout steps, actor batch multiplier 4 and three
epochs, each PPO cycle performs 48 actor optimizer steps. The unchanged critic
uses 64 steps. The earlier clip fractions near 0.05 and small measured policy
responses do not prove a learning-rate bottleneck, but motivate a controlled
actor-rate comparison without changing reward or mechanics. No evidence here
justifies springs, a phase feature, removing normalization or more reward terms.

## Phase 52 - Matched-update actor learning-rate comparison (2026-10-05)

Steps 10-17: compare **actor SGD rate 1e-4 versus 3e-4**, retaining reference_v9,
AMP mixture 0.8/0.2, target-PD gains, all observations, three actor epochs,
critic/discriminator settings, plant and datasets. The Phase 49 quiet-speed
multiplier is not used. The learning-rate hypothesis is unproven.

Predeclared protocol: fresh seeds **301020/301021**, sampling seeds +701,
identical full initial weights within each pair, distinct seeds across pairs,
100 PPO cycles and 409,600 transitions per arm. Seed 301020 runs reference then
3e-4; seed 301021 reverses order. A 300-second synchronized-training safety cap
aborts an incomplete arm; no extension or unequal-budget comparison. Save
initial/final 12-second-or-FAIL assessments on eight native/eight GPU starts
for each arm (128 cases). Track actual actor and critic optimizer-step counts,
unchanged reward identity, checkpoint pairs and evaluation/RNG isolation.

Use the same Phase 49 paired win rule: on each backend, more Stand passes, or
equal passes plus minimum/mean survival and mean longest valid posture each
improving by one control interval, with more paired improvements than regressions.
Both seeds must win for replicated gain. No favorable-checkpoint selection.
Implementation: `compare_stand_actor_rate.py`, `verify_stand_actor_rate.py`.
Output: `logs/mimickit-training/stand-actor-rate-20261005-01`.

Step 18 is bounded in advance: if the rate improvement replicates, run one
fresh third seed (301022) with the winning rate to 300 PPO updates, assessing
at 100 and 300, with a 900-second training cap and immediate stop on 16/16
Stand at the 100-update assessment. This tests duration on a new seed, not
resumption or another coefficient. Otherwise, use all four final Phase 52
actors in the live-versus-held-initial-action diagnostic (128 cases, no
training) to distinguish offset and changing-command effects. No hour-long
run, automatic 60-second gate, Arena change or perturbations is included.

Phase 52 completed and verified: the actor-rate gain replicated under the
predeclared aggregate rule. Seed 301020 reference/candidate native means
**3.035659 / 3.229425 s** (candidate GPU 3.233592), minimum
**2.433528 / 2.533536 s**, longest valid interval **2.152255 / 2.346021 s**.
All eight starts improved on each backend. Seed 301021 reference/candidate
means **3.123167 / 3.383604 s**, minimum **2.500200 / 2.616876 s**, native
longest valid interval **2.158506 / 2.498117 s** (candidate GPU 2.500200).
Five starts improved and three regressed, so this is not uniform dominance.
All four actors still passed 0/16 Stand cases.

Each arm completed 100 cycles/409,600 transitions, 4,800 actor optimizer steps
and 6,400 critic steps. Training seconds were 199.944638/188.890593 in seed
301020 and 192.180002/191.858051 for reference/candidate in seed 301021.
The full 128-case verification passed; reward identity, one-key config changes,
fresh paired weights, distinct seeds, RNG isolation, gates and controls were
checked. This supports the 3e-4 rate for the next experiment, not a threefold
learning-speed claim or production/Arena promotion.

## Phase 53 - Conditional fresh third-seed duration check (2026-10-05)

Step 18 takes the replicated-gain branch declared before Phase 52. Fresh seed
**301022**, sampling seed **301723**, actor rate **3e-4**, all other Phase 52
settings unchanged. No learned checkpoint, fitted normalization or optimizer
is reused. Execute 300 PPO cycles (1,228,800 transitions), with initial,
100-cycle and 300-cycle native/GPU assessments: at most 48 cases, 12 seconds
or physical FAIL. Stop at 100 if both backends pass all eight Stand cases;
otherwise complete 300, without applying Phase 47's six-minute screen to this
different predeclared duration protocol. The total synchronized-training safety
cap is 900 seconds. A cap hit makes the protocol incomplete, not a winner.

`extend_stand_actor_rate.py` saves the initial and each planned checkpoint,
unchanged reward records, optimizer counts and evaluation isolation evidence.
`verify_stand_actor_extension.py` recomputes saved gates/control traces,
checkpoint pairing, update budgets and the stop decision. Final selection is
the last planned checkpoint reached. A 100-to-300 gain requires more Stand
passes, or equal passes plus minimum/mean survival and longest valid posture
each improving by at least a control interval on both backends; otherwise
report the actual mixed/regressed outcome without choosing the earlier actor.

Output: `logs/mimickit-training/stand-actor-rate-duration-20261005-01`.
No further coefficient, training extension, 60-second gate, perturbation or
Arena action follows within this twenty-step batch. Execution completed; results follow.

### Phase 53 results and twenty-step batch completion

The fresh third-seed run completed **300 PPO cycles**, **1,228,800 transitions**,
**14,400 actor optimizer steps** and **19,200 critic steps**, using
**523.932803 synchronized training seconds**. No safety cap was reached.

- At 100 cycles (**183.644404 training seconds**): 0/16 Stand; minimum survival
  **2.416860 s**, mean **2.883564 s** on both backends; mean first departure
  **2.006411 native / 2.010577 GPU s**; mean longest valid posture
  **1.989742 native / 1.993909 GPU s**.
- At 300 cycles: 0/16 Stand; minimum **3.050244 s**, mean
  **3.525282 native / 3.527365 GPU s**; mean first departure **2.696049 s**
  and mean longest valid posture **2.679381 s** on both backends.

The final versus 100-cycle gains are minimum **+0.633384 s**, native/GPU mean
**+0.641718 / +0.643802 s**, and longest valid posture
**+0.689639 / +0.685472 s**. Both backends pass the predeclared duration rule.
The final selected artifact is `updates-300/model.pt`; no earlier or best
checkpoint replaces it. This is a positive within-seed duration result, not
replication of that duration result or solved Stand. The next proposed work
replicates this exact curve with a fourth fresh seed, as specified above.

Phase 53 verification passed **4,301 protected input entries**, **77 source/
snapshot pairs**, all **48 assessment cases** and **8,011 live intervals**.
It verified unchanged reward/configuration, fresh initialization, optimizer
counts, sidecars, evaluation isolation/RNG restoration, gates, delayed controls
and stopping behavior. No additional physics or network inference was used
for saved-result verification.

Steps 19-20 completed with `verify_stand_autonomous_batch.py`. The batch check
verified **3,549 unique canonical protected file paths** (fewer than raw input
entries because relative/absolute aliases are deduplicated), recomputed the
saved sensitivity aggregates and duration decision, and bound the phase
manifests. Accepted physical assessments total **304 cases**, plus the **32
preserved interrupted-attempt cases** in Phase 50. Phase 51 queried 64 saved
states without new simulation. Total learning across Phases 52-53 was
**700 cycles / 2,867,200 transitions / 1,296.806089 synchronized training seconds**
(about 21.6 minutes), excluding setup, evaluations, file work and verification.

All twenty authorized steps are complete. The learned behavior is still below
the 12-second Stand gate, but this batch establishes useful changing commands,
a replicated modest actor-rate gain, and a positive longer-duration result on
one fresh seed. No springs, new reward terms, production defaults, Arena
selection, 60-second assessment or perturbation curriculum were enabled.
No simulation or training remains running.

Evidence: [feedback comparison](../../logs/mimickit-training/stand-policy-feedback-20261005-02/report.json),
[input sensitivity](../../logs/mimickit-training/stand-feedback-inputs-20261005-01/report.json),
[actor-rate comparison](../../logs/mimickit-training/stand-actor-rate-20261005-01/comparison.json),
[duration result](../../logs/mimickit-training/stand-actor-rate-duration-20261005-01/report.json),
[batch verification](../../logs/mimickit-training/stand-autonomous-batch-20261005-01/report.json).

## Phase 54: independent replication of the actor-rate duration gain

Protocol recorded before execution, 2026-10-05. Steps 1-4 of the second
twenty-step batch are complete: global instructions and the plan were read;
the previous batch's 3,552 protected/output hash entries passed a fresh
read-only comparison; no training process was active. Historical sources are
unchanged. A source-generation guard rejected an incomplete in-memory
conversion before writing any file or executing any experiment; the successor
runner was then written explicitly.

`replicate_stand_duration.py --mode replication` uses fresh seed 301023,
sampling seed 301724 and the exact Phase 53 recipe: 64 environments, 64 rollout
steps, 4,096 transitions per cycle, actor SGD rate 3e-4, three actor epochs,
reference_v9/AMP reward and the same 33-motor target-PD controller. No warm
start. Assess at initialization/100/300 cycles on eight native and eight GPU
starts, maximum 12 seconds or physical FAIL. Stop at 100 only for 16/16 Stand;
otherwise select 300 as the final checkpoint, with a 900-second synchronized
training safety cap. A cap interruption is an incomplete protocol, not a pass.

The unchanged continuation rule requires, on both backends, more Stand passes
or equal passes with minimum survival, mean survival and mean longest valid
posture each improving by at least one control interval (0.016668 seconds).
Report per-start changes and compare with Phase 53. Replication here refers
to this duration effect, not to reliable Stand.

`verify_stand_duration_replication.py` recomputes saved gates/controls, checks
fresh initial normalizers, checkpoint pairing, RNG/learner/world isolation,
unchanged reward/configuration, optimizer steps and the declared stop rule.
The same isolated successor supports the predeclared conditional seed-301024
extension; it checks both prior duration outcomes before entering that mode.

Output: `logs/mimickit-training/stand-duration-replication-20261005-01`.
Execution completed: **300 updates / 1,228,800 transitions / 536.952423
synchronized training seconds**. Initial native/GPU mean survival was
1.895985 seconds, 0/16 Stand. At 100 cycles: minimum 2.533536, mean 3.325266
and longest valid posture 2.441862 seconds on both backends. At 300: minimum
3.116916, mean 3.496113 native / 3.500280 GPU, longest valid 2.656463 seconds
on both backends. Stand remains **0/16**. Select `updates-300/model.pt`.

The duration rule passed on both backends: minimum +0.583380 seconds,
mean +0.170847 native / +0.175014 GPU, longest valid +0.214600 seconds.
Five starts improved survival and three regressed on each backend. Native
per-start changes: [-0.050004, +0.100008, +0.583380, +0.483372, -0.066672,
+0.566712, +0.650052, -0.900072] seconds. The gain is not uniform.

Verification passed **4,425 protected input entries**, **79 source/snapshot
pairs**, **48 cases / 8,370 live intervals**. Training cap not reached;
normalizers were fresh, configuration/reward unchanged, optimizer counts and
assessment isolation verified. Steps 5-10 are complete.

## Phase 55: saved duration comparison and failure diagnosis

Protocol recorded while Phase 54 is training. Once its result is sealed,
`audit_stand_duration_replication.py` will analyze the saved Phase 53/54
initial/100/300 assessments. Recompute existing physical gates and motor
control formulas; compare paired per-start survival/longest-valid changes;
apply the unchanged continuation rule. Count the first failed gates and
terminal physical failure conditions per stage, and motor saturation/joint
limit proximity in the preceding 31 control frames. These associations do
not establish a causal controller or reward defect.

Compare first-20 and last-20-update windows ending at each training checkpoint:
optimizer/discriminator statistics, unmodified task reward and horizontal
speed. Logged stochastic training behavior and deterministic assessment are
different measurements; neither training return nor fitting loss replaces
the Stand criteria. No simulation, policy inference or training is added by
this analysis. Any conditional duration extension uses the previously
declared rule; no coefficient is chosen from these diagnostic correlations.

Output: `logs/mimickit-training/stand-duration-comparison-20261005-01`.

Results: steps 11-14 completed. Both seeds 301022/301023 pass the same
100-to-300 rule on both backends. All **96** saved cases first violated the
horizontal-speed gate; all later failed pelvis height below 0.65 m, without
terminal non-foot contact/nonfinite state or >=99% motor saturation in the
predeparture window. Saved gates and delayed motor controls were recomputed.

From the windows ending at 100 to 300, critic loss fell about 8.4-8.5 -> 4.81;
PPO clip fraction remained about 0.103-0.108. Mean task reward rose roughly
0.927-0.929 -> 0.939, and sampled training horizontal speed fell about
0.220-0.225 -> 0.176 m/s. Discriminator reward remained about 0.28-0.29 and
discriminator classification accuracy stayed high. These logs do not establish
a discriminator defect or justify a reward change. The modest behavioral
duration gain replicated; sustained Stand remains unsolved.

Decision: take the predeclared conditional duration branch. This does not
establish convergence, guarantee another gain or justify an hour-long run.

## Phase 56: conditional fresh duration extension to 600 updates

Steps 15-16, protocol recorded before execution. Reuse the sealed
`replicate_stand_duration.py --mode extension` and verifier with fresh seed
**301024**, sampling **301725**, unchanged settings and no warm start.
Initialization/100/300/600 checkpoints, at most **64 assessment cases**,
**600 cycles / 2,457,600 transitions / 1,500 synchronized training seconds**.
Stop at any assessed checkpoint with 16/16 Stand. At 300, stop unless its
100-to-300 outcome passes the same rule on both backends. At 600, assess
300-to-600 by that rule and report every paired start. Always select the last
checkpoint reached. No further training, 60-second gate or perturbations is
part of this batch. The runner verifies both prior duration gains before entry.

Output: `logs/mimickit-training/stand-duration-extension-20261005-01`.
If completed, repeat the saved-trace diagnosis across all three duration
seeds in `stand-duration-comparison-20261005-02`; this adds no simulation or
training. Step 19 verifies canonical hashes and batch totals; step 20 updates
the current milestones and next direction.

Phase 56 intermediate decision: fresh initial mean survival 1.887651 seconds,
0/16 Stand. At 100 updates (189.579621 training seconds): minimum 2.416860,
mean 3.335684, longest valid posture 2.452279 native / 2.454363 GPU seconds.
At 300 (542.482806 training seconds): minimum 2.833560, mean 3.539866,
longest valid posture 2.683548 seconds on both backends, still 0/16 Stand.
Both backends pass the same continuation rule, so the already-declared
600-update stage executes. The 100-to-300 duration gain now has three positive
fresh-seed instances; this says nothing yet about the 300-to-600 effect.

### Phase 56 final result: first autonomous 12-second Stand

The fresh seed-301024 run completed **600 PPO cycles / 2,457,600 transitions**,
**28,800 actor optimizer steps / 38,400 critic steps**, using **1,000.970159
synchronized training seconds**. Its final `updates-600/model.pt` actor passed
**8/8 native and 8/8 GPU starts**. Every case reached the time limit at
**12.000960 seconds**, with all balance and natural-posture gates valid
throughout the sampled interval; no first departure, physical failure or
teacher takeover occurred. The later batch check also verified every saved
pre-control-frame gate, including the initial state. Passive springs remained
zero; the existing active local target-PD controller remained explicit.

Compared with 300 updates, all eight survival pairs improved on each backend,
with no regressions. Minimum survival increased **2.833560 -> 12.000960 s**,
mean **3.539866 -> 12.000960 s**, longest valid posture **2.683548 -> 12.000960
s**, and Stand **0/16 -> 16/16**. These endpoints are capped observations:
the actor stood for the full assessment; its eventual failure time is unknown.
The successful interval was first observed at 600, not measured at every
intervening update. This is one successful independent actor, not proof that
600 cycles generally suffice or that learning speed scales linearly.

Verification passed **4,549 protected input entries**, **79 source/snapshot
pairs**, **64 assessment cases / 19,932 live control intervals**. All reward,
configuration, checkpoint pairing, optimizer-count, RNG/isolation, gate and
delayed-control checks passed. The run stopped with `stand_16_of_16`; no
additional training followed. The 1,500-second cap was not reached.

The follow-up saved-trace analysis covered **160 cases** across all three
duration seeds without additional simulation or inference. Every initial,
100-cycle and 300-cycle case still had horizontal speed as its first departure;
all sixteen 600-cycle cases had none. The original seed-301024 100-to-300
comparison improved only four survival pairs and regressed four on each
backend, despite passing the aggregate rule. Its later success therefore
does not erase those earlier regressions.

In the final 20 training updates, mean task reward was 0.958689 and sampled
horizontal speed 0.118175 m/s; PPO clip fraction was 0.111788, discriminator
reward 0.282779 and mean sampled episode length 303.39 control intervals.
The stochastic training episode length is not the deterministic 12-second
assessment result. Success is established by the frozen behavioral traces,
not by those training statistics.

Selected checkpoint:
`logs/mimickit-training/stand-duration-extension-20261005-01/updates-600/model.pt`
with its `.amp.pt` sidecar. This is the isolated 33-motor controller artifact;
production configuration and Arena selection remain unchanged.

## Phase 57: second twenty-step batch verification and completion

`verify_stand_duration_batch.py` completed step 19. Its first read-only attempt
identified that the two diagnostic roots had saved reports, source snapshots
and sealed output hashes but no standalone `protocol.json`. Before any batch
output was created, the verifier was corrected to reconstruct each diagnostic
input/source manifest from its recorded parent run and sealed numbered source
snapshots. Reconstructed input/source counts match the original seal, every
expected hash is checked, and the report explicitly identifies both recovered
manifests. No historical source or artifact was edited and no training or
physics was repeated for this repair.

The completed batch verification checked **3,985 unique canonical protected
file paths**, including the preceding batch, both new training runs and both
diagnostic reports. It checked copied model/reference hashes and unchanged
configurations/demonstrations, recomputed both duration decisions, and checked
the final actor's continuous pre/post balance and posture flags independently
of the final-window pass count. Both backends report full-interval validity.

Actual new compute: **900 PPO cycles / 3,686,400 transitions / 1,537.922582
synchronized training seconds** (25.63 min), plus setup, assessments and file
verification. There were **112 new physical assessment cases**, with 16 Stand
passes at the final checkpoint. Analysis of historical saved traces is not
counted as new physical cases. No verification rollouts, policy forwards or
optimizer updates were added.

Step 20 updated the top-level milestones and next protocol. **All twenty
authorized steps are complete.** The autonomous 12-second milestone is now
achieved for one fresh seed. Sustained 60-second Stand and a second successful
training seed remain to be demonstrated before perturbations. No simulation
or training is running; no hour-long run, extra coefficient change or Arena
promotion was performed.

Evidence: [replication](../../logs/mimickit-training/stand-duration-replication-20261005-01/report.json),
[first comparison](../../logs/mimickit-training/stand-duration-comparison-20261005-01/report.json),
[successful extension](../../logs/mimickit-training/stand-duration-extension-20261005-01/report.json),
[three-seed diagnosis](../../logs/mimickit-training/stand-duration-comparison-20261005-02/report.json),
[batch verification](../../logs/mimickit-training/stand-autonomous-batch-20261005-02/report.json).








## Corrective-step pilot closeout (2026-10-07)

See [CORRECTIVE_STEP_PLAN.md](CORRECTIVE_STEP_PLAN.md): 400 updates, 1,638,400
transitions, 512 frozen cases. No qualifying recovery steps. Naive doubled leg
targets fail quiet Stand; the recovery reward does not consistently outperform
matched continuation. No promotion or Arena change. Next: behavior-preserving
control changes and an explicit weight-transfer/single-support/short-step curriculum.

## Support-skill successor: twenty comparisons completed (2026-10-07)

See [SUPPORT_SKILL_AUTONOMOUS_PLAN.md](SUPPORT_SKILL_AUTONOMOUS_PLAN.md) for the
completed weight-transfer/single-support/placement batch. Its separate totals
are 20 substantive comparisons, 1,848 accepted frozen cases, 700 PPO updates /
2,867,200 training transitions, and 1,228,800 offline supervision presentations.
Previous batches retain their own accounting. Infrastructure retries and invalid
screens are excluded; the Stage15 finalization exception remains disclosed.

Both source parents retain 120-second quiet Stand with the added component
on/off, on native/GPU (64/64 full-interval cases). Both successive transfer
commands succeed in only 4/16 cases per parent/backend, although all 64 repeated
command cases retain Stand. No learned corrective step or sustained single-foot
skill is established. Reactive transfer regresses the first parent's mild-hit
recovery from 6/8 to 4/8 and offers no consistent cross-parent improvement.
No Arena promotion, export or C# build. The next prerequisite is a physically
verified controlled single-support transition and return, before more impact
training. Reuse and assess the existing whole-body feedback teachers first.

Final accounting: logs/mimickit-training/support-skill-batch-review-20261007-01/report.json.
Checkpoint/trace retention and the separately requested 39.458 GiB storage
cleanup are documented in docs/STORAGE_RETENTION.md. Future long retention and
broad impact suites should follow demonstrated prerequisite skill improvement.

## Single-support controller follow-up (2026-10-07)

The separately authorized foundation follow-up completed four focused comparisons
and 36 small native cases, with zero learning updates. See the Single-support
feedback follow-up section in SUPPORT_SKILL_AUTONOMOUS_PLAN.md.

The 33-motor predictive target adapter exactly reproduces committed delayed PD
commands in its transport check. Scripted feedback reaches the 11 cm COM transfer
goal within about 2 mm while retaining upright Stand in 4/4 pilot cases. Releasing
the other foot still fails the one-second support gate. Starting directly in an
admitted lifted pose isolates the issue: the best feedback objective holds valid
single support only 0.250-0.350 seconds, then all four cases fail; explicit
COM/foot tracking performs worse. No entry/hold/return skill or training examples
were admitted. Arena selection and mechanical settings remain unchanged.

Review: logs/mimickit-training/single-support-feedback-review-20261007-01/report.json.
Next: diagnose actual delayed PD-target support evolution and the short-horizon
control objective, then demonstrate hold and entry/return before more learning.

## Delayed-target response follow-up (2026-10-07)

Completed eight native physical cases and twenty delayed-response forecasts.
Forecasts match live execution within tolerance. A bounded correction using the
actual four-substep PD response reduces its local objective but still passes
0/4 one-second prepared single-support holds (best 0.383 s). Two survive four
seconds without valid final balance; two fall. No training, brain promotion or
Arena change. Next: identify the first support-loss event in the saved traces
before another controller intervention. Details and evidence limitations are in
SUPPORT_SKILL_AUTONOMOUS_PLAN.md; independent reconciliation is recorded at
logs/mimickit-training/support-target-response-review-20261007-01/report.json.
This follow-up is separate from the completed twenty-comparison batch.

## Support-goal feasibility correction (2026-10-07)

Found a missing foundations check: all four 11 cm transfer / 2 cm lift goals are
static-feasible under full motor torque, but infeasible under the actual bounded
position-target controller. The stance ankle is near its roll limit; legal targets
provide only 8-9 percent torque in the required direction versus 30-38 percent in
the old static solution. Constraint isolation confirms that the stance-ankle bound
alone causes infeasibility in the existing static model. Future teacher-goal
admission now includes the target-PD torque envelope. No physical limits or gains
were widened. Prior full-torque admission of these four goals is superseded.

Eight archived traces were diagnosed without advancing physics. A proposed warm
initial-queue comparison stopped at its preflight bound guard, before scientific
cases. No training or Arena change. Next: redesign feasible single-support goal
geometry before another controller or PPO run. Full protocol, timing and limits:
SUPPORT_SKILL_AUTONOMOUS_PLAN.md. Verified constraint evidence:
logs/mimickit-training/support-target-bound-admission-20261007-02/report.json.

## Feasible support geometry progress (2026-10-07)

The revised static screen found a common feasible goal: 10 cm COM shift with
12-degree lean, 2 cm lift, and >=0.11875 rad support-ankle angle margin. Screened
64 private geometry cases, then ran eight frozen native physical cases.
With two valid initial support commands, all four cases hold valid single-foot
support for 1.57-2.30 seconds, but three later fall: only 1/4 passes the complete
four-second prepared trial. The matched cold queue fails 4/4 after only one valid
control interval. This demonstrates improved scripted physical support, not learned
entry or stepping. No PPO or Arena change. Next: stabilize the later warm-hold
failure, then verify entry/hold/replant/return from standing. Full details are in
SUPPORT_SKILL_AUTONOMOUS_PLAN.md. Reconciled evidence:
logs/mimickit-training/support-geometry-review-20261007-01/report.json.

## Scripted single-support milestone (2026-10-07)

COM feedback plus the already feasible 12 cm COM shift / 12-degree lean goal now
passes all four warm prepared holds for four seconds. Fresh source/feedback quiet
checks also pass. Crucially, all four ordinary-start cases complete weight
transfer, lift, 1.47-1.50 s valid single support, replant, return and two seconds
of original Stand. These ordinary-start cases use no lifted reset or preloaded
support queue. Existing physical limits, delay, gains and success gates remain.

This is a verified scripted teacher sequence on poses 0/1 and both feet, not a
trained recovery brain or corrective forward step. No PPO or Arena change.
Next: replicate across all eight original starting poses, then collect successful
teaching examples and validate the learner contract before bounded training.
Protocol and negative comparisons: SUPPORT_SKILL_AUTONOMOUS_PLAN.md.
Independent review of the twenty physical cases in this follow-up:
logs/mimickit-training/support-com-review-20261007-01/report.json.

## Full-pose scripted support replication (2026-10-07)

The frozen 12 cm / 12-degree COM-feedback teacher passes all sixteen quiet
checks and 13/16 ordinary-start transfer/lift/hold/replant/return cases across
eight original poses and both feet. Left swing passes 8/8; right swing passes
5/8. Right swing in poses 2-4 never achieves valid single support and falls.
Maximum whole-foot clearance while posture/support remains valid is only
7.6-9.4 mm versus the unchanged 15 mm gate; chest tilt fails before stance
loading/drift. The original four pilot trajectories reproduce exactly.

The all-sixteen teacher-data gate remains closed: no demonstration dataset,
training updates, checkpoint or Arena promotion. Candidate goal-conditioned
collection code exists but successful end-to-end dataset verification is pending.
Independent review covers 32 physical cases, ten output hashes and 80 source
snapshot entries. Compact run evidence is 12.80 MiB. Next: isolate right-swing
height/posture tracking, correct one controller term at a time, then repeat all
sixteen cases before teaching-data export or learning. Details and evidence are
in SUPPORT_SKILL_AUTONOMOUS_PLAN.md and
logs/mimickit-training/support-all-poses-review-20261007-01/report.json.

## Updated learned Perturb Arena preview (2026-10-08)

User correction: deliver an updated learned recovery brain for interactive hits.
The scripted support teacher's 16/16 export gate is no longer a prerequisite for
all recovery learning. Its 13/16 result and rejected teaching dataset remain
unchanged. Stage33 was already a newer trained candidate than Arena's Stage12.

Completed a bounded continuation from Stage33 parent 301028, optimization seed
381028: 100 PPO updates with repeated impacts and 100 matched quiet-control
updates, 819,200 total transitions, 243.757365 measured training seconds.
Both arms use quiet-action rehearsal (2,457,600 total example presentations,
not physics transitions). Impact practice: 378 first and 341 second throws,
covering all 32 pose/onset pairs for both throws; 5 kg, 1.6 m/s, +X Chest aim,
0.25 m flight, first onset 1/3/5/9 s and second four seconds later. No scripted
teacher takeover, mechanical/gain changes, or intermediate checkpoint copies.

Initial frozen evaluation exposed an invalid Stage33 repeated condition: one
second projectile insertion overlaps the dummy, with failed native/GPU impulse
admission. Preserve its raw 7/8 count but exclude that 16-case backend group from
accepted comparisons. Initial finalization stopped at the admission guard.
A prospective completion protocol repeated all four actors at the same clear
0.46 m flight distance; same success thresholds, mass, speed, direction and
5/9-second timing. Do not compare the two different flight geometries as if they
were identical impacts. The original Stage12 4/8 versus Stage33 7/8 repeat
comparison is not an admitted conclusion.

Final matched results, eight original poses, identical native/GPU counts:
- Stage12 Arena: single +X 1.4/1.6 m/s = 7/8 and 5/8; clear two-hit = 3/8.
- Stage33 parent: single +X = 8/8 and 7/8; clear two-hit = 5/8.
- New quiet continuation: single +X = 8/8 and 6/8; clear two-hit = 3/8.
- New repeated-hit continuation: single +X = 8/8 and 8/8; clear two-hit = 7/8.
Every -X single-hit condition passes 8/8. Singles launch at 5 s from 0.25 m;
corrected repeats launch at 5/9 s from 0.46 m. All four actors retain 30-second
quiet Stand. The selected new impact endpoint additionally passes all sixteen
120-second native/GPU quiet cases. Body attribution and original recovery,
control-formula, insertion and backend gates pass the final comparison.

Accounting: 464 physical evaluation cases, including 16 excluded original
Stage33 repeat cases; 448 admitted cases. No additional learning during repair.
This is one optimization seed, not replicated recipe success or proof of
natural corrective stepping / 10 kg high-speed recovery.

Selected experimental preview checkpoint:
logs/mimickit-training/perturb-preview-learning-20261008-01/single_ball/final/model.pt
SHA256 f832bab84d392ff5839c265bb21228ce21608bc26c6306a1e17fd8befa0a143d.
Export: logs/mimickit-training/perturb-arena-learned-20261008-01.
ONNX agreement over 2,053 observations: maximum action error 2.384186e-7.
Arena scene and ball viewer selection now point to this export. Previous scene
and selection are backed up inside it. Existing Godot build loaded the exact
bundle successfully; no C# compilation. Manual B throws, R resets, P pauses.
Arena retains its existing 10 kg / 4 m/s random-target persistent-ball setup,
with automatic throws off; that harder setting remains experimentally unproven.

Evidence: perturb-preview-training-review-20261008-01/report.json and
perturb-preview-evaluation-review-20261008-01/report.json under
logs/mimickit-training. The initial and completion protocols are both retained.
Storage: training 78,129,606 bytes; detailed evaluation evidence 1,912,107,134
bytes; export approximately 3.8 MB. No previous run was deleted.

## Launch-distance continuation result (2026-10-08)

Continue from the learned Arena checkpoint f832bab8. Saved-trace diagnosis
shows its remaining clear-distance failure is pose 7 after the FIRST hit:
contact ends 5.183748 s; horizontal speed exceeds 0.2 m/s at 6.817212 s;
physical failure occurs 7.600608 s, before the second throw at 9 s.

Matched fixed 0.25 m versus mixed 0.25/0.46 m training completed 100 PPO
updates per arm from identical full parent weights, fresh matched optimizers
and optimization seed 391028. Total 200 updates / 819,200 transitions;
226.987631 measured training seconds and 2,457,600 quiet rehearsal presentations.
Fixed received 371 first and 338 second throws, covering all 32 pose/onset
combinations. Mixed received 377 first and 339 second throws, covering all 64
pose/onset/distance combinations for both throws. Start, launch and integrity
checks passed. Keep gains, limits, reward, delay and zero passive springs.

All 192 new frozen cases are admitted with matching native/GPU outcomes:
- Parent f832bab8: stronger +X single 8/8, clear-distance two-hit 7/8 (reused).
- Fixed-distance continuation: stronger single 7/8, two-hit 6/8.
- Mixed-distance continuation: stronger single 8/8, two-hit 7/8.
All easier and reverse-direction singles pass 8/8; both endpoints pass all
30-second quiet cases. Fixed loses poses 6 and 7 in the repeat trial. Mixed
still loses pose 7 after the first hit, at 7.833960 s, only 0.233352 s later
than the parent. Delay of a fall is not successful recovery.

Neither endpoint qualifies. Keep f832bab8 loaded and selected in Perturb Arena;
no export, restart, promotion or C# compilation. Conditional timing and
120-second candidate retention checks were not run and add zero cases.
Before evaluation, the conditional timing protocol was corrected from 3/7 s
(already trained) to genuinely untrained 2/6 s. That correction is recorded in
the prospective evaluation protocol; the original training plan stays sealed.

Full control equations and gates were checked before compacting only the new
substep-control arrays. Compact traces retain observations, kinematics, all
recovery/posture gates, contacts and decisions; original full-trace hashes and
audit records disclose which arrays were removed. Independent review recomputes
all retained gates and recovery outcomes; full substep control equations cannot
be independently recomputed from the compact traces. Existing evidence unchanged.
Storage: training 78,456,311 bytes; evaluation 348,497,537 bytes.

Plan: PERTURB_DISTANCE_PLAN.md. Evidence under logs/mimickit-training:
perturb-distance-training-review-20261008-01/report.json and
perturb-distance-evaluation-review-20261008-01/report.json.
Next priority: inspect the learned policy's action/target saturation and delayed
response in the failing pose before prescribing another continuation. The mixed
curriculum prevents the fixed-arm regression but does not resolve that failure.

## Learned recovery response: twenty-step block closed (2026-10-08)

The complete substantive-step ledger is PERTURB_RESPONSE_PROGRESS.md. Prospective
protocols are PERTURB_RESPONSE_PLAN.md and PERTURB_RESPONSE_LEARNING.md; sealed
inputs were preserved. Administrative setup, audit and documentation are not
counted as substantive experiments in the final ledger.

Outcome: keep learned brain f832bab8 in Perturb Arena. No export, Arena restart,
C# compilation, passive-spring change, target-range change or delay change.

Diagnostics: failed pose 7 has no actor-output or target-limit clipping before
its 6.817212-second departure. Maximum raw action magnitude is .553. The exact
2-frame command queue and all recorded PD substeps check out. Comparison with
quiet motion confirms policy response on the first frame after contact, 16.668
ms later. Both feet remain loaded while COM advances beyond the foot/toe box
bounds; these bounds describe geometry, not a force support polygon. Velocity
input gains .75/1.25 do not provide an admitted improvement. Of 96 diagnostic
cases, the 16 gain-.75 / onset-2 cases fail backend admission and are excluded.

Matched learning: TD-lambda .98 control versus .995 treatment, optimizer seeds
401028 and 411028, each starting from the same full f832bab8 checkpoint with
fresh optimizers. Mixed .25/.46 m flight curriculum, 5 kg / 1.6 m/s +X Chest,
1/3/5/9-second first throws and second +4 s, 64 worlds x 64 frames, three actor
epochs, actor LR 3e-5, discount .99, quiet rehearsal 100/256. No other learning
or physical setting changed. All starts, coverage and accounting checks pass.
Four arms x 100 updates = 400 updates / 1,638,400 transitions, 485.114 measured
training seconds. Only initial/final checkpoints were saved.

First optimizer seed: both arms 6/8 repeated recovery. Second optimizer seed:
control 7/8, treatment 8/8. All four actors retain all four single-hit controls
and 30-second quiet Stand at 8/8, with native/GPU agreement. This is one improved
primary candidate, not a repeated recipe gain at both optimizer seeds.

Final untrained timing, 4/8-second throws: parent 7/8, candidate 53da704a 6/8,
with valid insertion/body/backend admission. Candidate pose 7 still falls after
the first hit; pose 6 fails after the second. Reject the candidate. Its gated
120-second retention check was not run because timing already disqualified it.

Foot-motion audit: zero recovered lifted-step signatures in parent or any new
actor. Successful trials move feet at most 2.084-2.472 mm; the 8/8 candidate's
maximum is 2.330 mm. Improvement at one schedule is standing balance, not learned
corrective foot placement or evidence of natural recovery from 10 kg / 5 m/s.

Total: 512 new physical evaluation cases, 496 admitted and 16 explicitly
excluded. Independent audits recomputed outcomes from compact evidence. Full
substep controls were verified before documented compaction; they cannot be
recomputed from compact traces alone. About 1.075 GB of new evidence, no old
results deleted. Godot process 10144 remains responsive on f832bab8.

Evidence under logs/mimickit-training:
- perturb-response-audit-20261008-01 (including matched quiet-response.json)
- perturb-response-probe-20261008-01 and its review
- perturb-response-learning-20261008-01 / -02 and training reviews
- perturb-response-evaluation-20261008-01 / -02 and evaluation reviews
- perturb-response-final-20261008-01 and final review
- perturb-response-foot-audit-20261008-01
- perturb-response-block-review-20261008-01/report.json (final reconciliation)

Next priority: a learned recovery task targeting timing robustness and actual
foot placement, retaining quiet Stand and final recovery admission. No evidence
here supports changing springs or widening target limits, or extending the same
TD-lambda comparison. Scripted diagnostics must remain explicitly separate from
learned Arena behavior.

## 2026-10-08: learned support block completed (20/20)

See LEARNED_SUPPORT_PLAN.md and LEARNED_SUPPORT_PROGRESS.md for the frozen
protocol, complete ledger, and limitations. Four fresh supervised learners and
16 frozen batches (256 cases) finished. Goal-relative labels halved held-out
prediction error, but every learned entry and enabled-adapter quiet case failed.
No learned foot placement or hit-timing robustness was achieved. Current Arena
parent f832bab8 is unchanged; it passed a separate 16/16 30-second quiet control.

The historical physical teacher's all-sixteen export gate remains failed. This
successor explicitly used only its 13 complete successes as experimental data,
held out whole poses, and evaluated all sixteen combinations. Full old-command
replay failed later despite close early observation reconstruction; see the
progress document. No script or playback was promoted as a learned brain.

Next priority supersedes the earlier next-action paragraph: establish stable
learned handoff and zero-residual quiet retention, then collect corrective labels
on learner-visited states. Actual displacement and hit recovery remain gated on
successful learned support. Avoid another long run or timing sweep of the failed
adapters. Evidence: learned-support-20261008-01, learned-support-audit-20261008-03,
and learned-support-review-20261008-01 under logs/mimickit-training.


## 2026-10-08: bounded learned handoff follow-up

See LEARNED_HANDOFF_PLAN.md and LEARNED_HANDOFF_PROGRESS.md. Added an exact
quiet-goal bypass and gradual, rate-limited learned correction. Three 30-second
quiet batches passed 16/16 with issued commands and qpos exactly matching the
unchanged f832bab8 Stand anchor. This is a structural safeguard, not newly
learned Stand. Live-only rate limiting was independently verified.

Two matched correction rounds used 277 learner-state labels and 6,000 supervised
updates. All support attempts failed before lifting; the later-state treatment
regressed and is rejected. Better fit to labels did not imply better control.
Completed 144 planned physical cases and 32 frozen audit cases. Arena unchanged.

Next: isolate fixed arms-clear holding and then small weight transfer before
another full support sequence. Do not extend the failed correction treatments
or advance to foot placement/impacts. Evidence: learned-handoff-20261008-01,
learned-handoff-late-20261008-01 and learned-handoff-review-20261008-01.


## 2026-10-08: fixed intermediate-goal diagnosis completed

See FIXED_SUPPORT_GOALS_PROGRESS.md. Twelve native cases compared identical
fixed goals and handoff: physical label-source feedback passed arms-clear 2/2
and small transfer 4/4; the frozen neural learner passed 0/2 and 0/4. Physical
success included supported tracked holding and return to Stand. This is a small
pose-0/pose-6 diagnostic panel, not learned recovery or all-pose admission.

The old training pool has only 72 near-arms-clear goal frames, at most 0.133344
continuous seconds, and no near-matching 4 cm transfer goals. Sustained holds
are a missing curriculum component; these results do not uniquely establish
an architecture defect. Next: exact-state sustained arms-clear training with
closed-loop validation and quiet retention, before transfer or lift. Arena
f832bab8 unchanged; no training updates or C# compilation in this diagnosis.
Evidence: fixed-support-goals-20261008-01 and its review under logs/mimickit-training.


## 2026-10-08: exact sustained arm-hold learning and queue ablation

See ARM_HOLD_LEARNING_PLAN.md and ARM_HOLD_LEARNING_PROGRESS.md. Both physical
teacher timing variants passed 8/8 and supplied 9,072 exact issue-time/raw-target
examples (6,804 training, 2,268 whole-pose held out). Full-input supervised
learners still achieved 0/8 supported holds. A matched learner with only its
queued-command inputs masked achieved 4/8 supported holds across two seeds;
returned Stand was 4/8 and 3/8. All remained 0/8 on the full pose-tracking gate.

This is a replicated partial support improvement, not an admitted arms-clear
skill or foot placement. Physical delay and the frozen Stand anchor's full
observations remain unchanged. The evaluated full and masked learners retained
16/16 exact quiet-Stand traces each. Completed 20,000 supervised updates and
96 physical cases. Arena f832bab8 unchanged; no export or C# compilation.

Next: teach the masked learner corrections for small recoverable deviations
around the held pose, requiring accurate tracking as well as support. Advance
to transfer only after the full learned hold-and-return gate passes. Evidence
is under logs/mimickit-training/arm-hold-*-20261008-01.


## 2026-10-08: deviation-supervision comparison rejected

See ARM_HOLD_DEVIATIONS_PROGRESS.md. The first disturbed teacher collection
passed only 1/8 and lacked a successful held-out pose, so it ran zero training
updates. A four-times-smaller innovation bound passed 8/8 and supplied 1,440
exact clean-target correction examples with whole-pose holdouts preserved.

Matched 4,000-update continuations both regressed from the earlier 4/8-supported
experimental learner: replay retained 1/8 supported holds; correction training
retained 0/8. Both were 0/8 complete and are rejected. The treatment still
retained 16/16 exact quiet-Stand cases. Completed 48 physical cases and 8,000
supervised updates; Arena f832bab8 unchanged.

Next: bounded on-policy supported pose-tracking/return learning from the prior
partial checkpoint, after zero-update adapter parity. Do not extend these
rejected supervised fits or advance to transfer/hits. Keep the quiet safeguard,
physical limits, full tracking gate, and whole-pose holdouts.


## 2026-10-08 - Bounded arms-clear PPO pilot

See ARM_HOLD_PPO_PROGRESS.md for the latest completed step. Two independent 24,576-transition runs improved supported holds from 4/8 to 5/8 and 6/8; both returned Stand in 5/8, but complete skills remain 0/8. Quiet retention passed 16/16 per run with exact anchor traces. Every PPO cycle hit the actor KL stop; correct update size and inspect reward contributions before extending training. Arena remains unchanged. Audit: logs/mimickit-training/arm-hold-ppo-review-20261008-01/report.json.


## 2026-10-08 - PPO learning-rate comparison completed

See ARM_HOLD_PPO_RATE_PROGRESS.md. Four matched runs compared actor rates 1e-5 and 3e-6, 24,576 transitions each. Controls exactly reproduced prior weights/traces. Smaller updates reduced KL spikes but returned Stand in only 3/8 and 4/8 versus control 5/8 in both seeds; complete skills remain 0/8. All 64 quiet cases passed with exact anchor traces. Tracking reward was below 1% of its maximum in 55-58% of hold samples. Next bounded priority: broader training-only distance rewards with unchanged final gates. Arena unchanged. Audit: logs/mimickit-training/arm-hold-ppo-rate-review-20261008-01/report.json.


## 2026-10-08 - PPO reward-width comparison completed

See ARM_HOLD_PPO_REWARD_PROGRESS.md. Four matched runs compared motor/COM reward width multipliers 1 and 2 at actor rate 3e-6, 24,576 transitions each. Broader reward improved returns from 3/8 to 4/8 and 4/8 to 6/8; holds changed 5/8 to 4/8 and 4/8 to 5/8. Complete skills remain 0/8. Near-zero tracking rewards fell from 55-58% to below 1.5%. All 64 quiet cases passed exact anchor traces. Left upper-arm error closely matches an issued-command offset; next priority is a matched action-to-joint-response diagnostic, not another sweep. Arena unchanged. Audit: logs/mimickit-training/arm-hold-ppo-reward-review-20261008-01/report.json.


## 2026-10-08 - Upper-arm command-response diagnostic completed

See ARM_ACTION_RESPONSE_PROGRESS.md. Matched +/-0.005-rad one-frame probes at hold and return were completely erased by the existing slew limiter in all eight poses. Baseline left-arm slew saturation was 85.6% in hold and 80.3% in return, with no torque saturation. Twelve-frame biases reached six hold poses and all return poses; the two-frame queue and PD torque reconstruction passed exactly. No training or Arena change. Next: test corrections around the already rate-limited command while reapplying the same feasible bounds; require zero-correction parity before any learner change. Audit: logs/mimickit-training/arm-action-response-review-20261008-01/report.json.


## 2026-10-08 - Feasible-command correction diagnostic passed

See ARM_FEASIBLE_RESPONSE_PROGRESS.md. Moving the correction after nominal slew limiting and reapplying the same bounds transmitted inward +/-0.005-rad pulses in all eight hold and all eight return poses. Outward pulses stayed blocked. Zero correction exactly reproduced the prior state/trace; slew, joint, torque and delay checks passed. This verifies an action path, not learned recovery. No training or Arena change. Next: two bounded runs of a zero-output correction policy over the frozen proposal learner, using raw pre-projection PPO likelihoods and unchanged skill gates. Audit: logs/mimickit-training/arm-feasible-response-review-20261008-01/report.json.


## 2026-10-08 - Feasible-correction PPO pilot completed

See ARM_CORRECTION_PPO_PROGRESS.md. Two fresh correction-policy runs over the frozen proposal, 24,576 transitions each, both regressed from 4/8 holds and returns to 3/8 and 3/8; complete skills remain 0/8. About 81% of eligible sampled coordinates changed the rate-limited addition, so transmission alone did not yield useful learning. Initial/final quiet retention passed all 48 cases with exact anchor traces. Both candidates rejected for promotion; Arena unchanged. Next: frozen stochastic-versus-mean evaluation including the untrained zero-mean head to test projection-induced execution mismatch. Audit: logs/mimickit-training/arm-correction-ppo-review-20261008-01/report.json.


## 2026-10-08 - Thirty-step execution and critic investigation completed

See ARM_RECOVERY_30_PROGRESS.md. All 30 research steps completed and audited. Frozen noise, timing, projection and matched action/critic comparisons produced no complete arms-clear skill. A real projection bias was measured. Critic warm-up passed fresh-episode validation and improved partial holds over two matched controls, but all four bounded PPO candidates remained at zero complete skills. Four fits: 98,304 transitions, 913 actor updates; all 64 quiet cases exactly retained Stand. Dominant left-arm error was already present in issued commands (roughly 0.51-0.57 rad RMS for warm candidates). Next: structural proposal/target-representation ablation against direct goal tracking and the verified physical teacher, before more PPO. Arena unchanged. Evidence: logs/mimickit-training/arm-recovery-30-20261008-01/final-audit.json. One aborted harness attempt is preserved and excluded from successful fit totals.


## 2026-10-08 - Second thirty-step proposal investigation completed

See ARM_PROPOSAL_30_PROGRESS.md. All 30 steps completed and independently audited. Teacher labels and replay were exact; large late target offsets also appeared in matched teacher queries, so they were not a simple constant-label bug. Fresh inherited-feature task heads now complete 5/8 and 3/8 nominal skills, and 5/8 and 2/8 changed-timing skills. Task-relative inputs and removal of quiet samples failed both seeds. The selected pair retained exact sixty-second Stand. Six fresh supervised fits, 24,000 optimizer updates; no PPO. Scripted eight-arm ablations improved outcomes but are clearly distinguished from learned results. No candidate passes universal admission; Arena unchanged. Next: verified corrective teacher takeovers from the fresh heads' remaining motor/COM/support failures, then a matched corrective-data comparison retaining quiet data and whole-pose holdouts. Audit: logs/mimickit-training/arm-proposal-30-20261008-01/final-audit.json.

## 2026-10-08 - Corrective takeover and matched fitting completed

See ARM_CORRECTIVE_LEARNING_PROGRESS.md. The physical teacher recovered all 16 attempted failing training cases from exact learned prefixes, admitting 5,827 samples with no heldout labels. Four matched fresh fits then showed uniform correction augmentation regressed full learned skills from 15/32 to 5/32; original replay weights and trajectories reproduced exactly. Both augmented heads retained exact sixty-second quiet Stand, 32/32 cases. Both independent audits passed. Retain the previous complete-data heads; Arena unchanged. Next: isolate sampling balance (quiet share fell from 33.16% to 17.86%) with controlled correction weighting and matched physical evaluations, before more data or longer training. This is a confound to test, not an established cause. Evidence: logs/mimickit-training/arm-corrective-takeover-20261008-01/audit.json and logs/mimickit-training/arm-corrective-fit-20261008-01/audit.json.

## 2026-10-08 - Exact-quiet sampling comparison completed

See ARM_SAMPLING_BALANCE_PROGRESS.md. Eight matched fresh fits preserved every original quiet draw at its original batch position. Replay reproduced historical weights and traces exactly. Full skills: replay 15/32, 5% corrections 16/32, 15% corrections 15/32, restored-quiet high share 5/32. The selected 5% pair improved worst-condition success from 2/8 to 3/8 but lost heldout passes (7/8 to 5/8); it is not admitted. Both selected heads retained exact sixty-second Stand, 32/32. Sampling and physical audits passed. Command differences emerge during the ramp and grow as state/history diverge. Next: frozen matched-state feedback/input diagnosis before more fitting; retain previous references and Arena. No learned stepping. Evidence: logs/mimickit-training/arm-sampling-balance-20261008-01/audit.json and diagnostics.json.

## 2026-10-08 - Matched feedback and queue-input diagnosis completed

See ARM_MATCHED_FEEDBACK_PROGRESS.md. Across 24 targeted matched-prefix cases, learned continuation passed 6/24, swapping heads 7/24, and physical teacher continuation 24/24. All 24 small bounded pulses transmitted; 9 improved complete-skill outcomes, 2 regressed, and 2 introduced native failures in an already-incomplete pose. Position differences grew from 0.2 to 1 second in all pulse probes. Private queue diagnosis reproduced teacher targets exactly in 24 reconstructed states; 96 queue-only variants changed every teacher target while both masked task heads' raw outputs stayed identical. The Stand source still responds to the queue. This is an exact-imitation input mismatch, not proof of the sole failure cause. Both audits passed. No training or Arena update. Next: validate a bounded representation of pending-command effects before matched fitting; do not blindly restore raw history or continue sampling-ratio search. Evidence: logs/mimickit-training/arm-matched-feedback-20261008-01 and logs/mimickit-training/arm-private-queue-20261008-01.


## 2026-10-08 - Bounded pending-command input comparison completed

See ARM_PENDING_EFFECT_PROGRESS.md. The new input matches native PD control at all 24 saved states and distinguishes all 96 queue alternatives. Four same-budget fresh fits reproduced masked controls exactly. Pending-effect inputs reduced heldout teacher error by 13-16% but regressed full core skills from 15/32 to 4/32 and timing/noise challenges from 21/48 to 4/48. Both new heads preserved exact sixty-second quiet Stand, 32/32. The independent 192-case audit passed; the candidate is rejected. Frozen response diagnosis shows information is used but teacher responses remain poorly matched. Next: validate response supervision on physically reachable disturbed training states before another fit, preserving quiet draws and whole-pose holdouts. No longer PPO run, Arena replacement or learned stepping. Evidence: logs/mimickit-training/arm-pending-effect-20261008-01/audit.json and queue-response-diagnostic.json.


## 2026-10-09 - Reachable response admission and matched supervision completed

See ARM_RESPONSE_SUPERVISION_PROGRESS.md. Real delayed pulses on training poses 0 and 4 produced 40/40 successful targeted teacher continuations and 160 admitted response pairs; no heldout teacher queries. Four matched fresh fits preserved original quiet draws and used identical paired examples. Added response teaching reduced fitting-set response loss by about 57% but worsened original target MSE by 43-46%; core skills fell from the matched absolute control's 3/32 to 0/32, and challenges from 2/48 to 0/48. The retained masked references remain stronger at 15/32 and 21/48. Both new response heads retained exact sixty-second quiet Stand, 32/32. Collection and learning audits passed, covering 352 physical cases. Reject the new heads. Next: freeze the retained controller and validate an initially zero, bounded correction with exact replay before fitting. No Arena replacement or learned stepping. Evidence: logs/mimickit-training/arm-reachable-response-20261008-01 and logs/mimickit-training/arm-response-supervision-20261008-01.

## 2026-10-09 - Direct recovery pivot authorized

See DIRECT_RECOVERY_PLAN.md. Complete arm-motion tracking is no longer a prerequisite for direct push-recovery experiments. Freeze the existing deployed Stand/Perturb source and train a zero-initialized bounded joint correction, judged on direct force-pulse recovery, quiet retention, heldout poses and independently audited foot placement. First verify exact zero-correction native parity. Two bounded PPO seeds, 24,576 transitions each. Force pulses are not projectile certification; no Arena promotion is implied. Earlier arm evidence is retained as diagnostics. This decision supersedes prior arm-gate instructions to block all hit training.
## 2026-10-09 - Direct recovery pilot completed; both candidates rejected

See DIRECT_RECOVERY_PROGRESS.md. Two zero-initialized bounded correction heads over the frozen deployed source trained for 24,576 transitions each. Direct force-pulse settled recovery regressed from 35/64 to 30/64 in both seeds; heldout-pose results fell from 10/16 to 8/16 and 7/16. Both retained sixty-second Stand, 16/16 each. No audited corrective steps. Zero correction reproduced the original controller exactly; the source also matched its archived sixty-second trace. Independent audit passed 22 traces/352 evaluation cases. Mean corrections did not saturate the envelope or slew bound. Direction and onset are confounded in this first screen. Next: cross direction with timing, then test bounded leg-response feasibility in matched failing push states before another PPO run. These are force pulses, not projectile certification. Arena unchanged. Evidence: logs/mimickit-training/direct-recovery-20261009-01/audit.json.
## 2026-10-09 - Direct leg-response screen completed

See DIRECT_LEG_RESPONSE_PROGRESS.md. Crossed frozen baselines recovered X 7/64 versus Y 63/64, confirming the difference at both hit times. A selected bounded bilateral ankle pulse improved 150 N X recovery from 7/32 to 15/32 (10 rescues, 2 losses); reserved poses improved 2/8 to 4/8. Across the fixed library, 20/25 baseline failures had some rescuing intervention, a posthoc reachability result rather than one controller's performance. Replaying baseline leg commands during the selected pulse did not change recovery identities or count. No corrective steps; stronger X pushes remained 0/32. No training or Arena update. Audit passed 63 traces/1,008 cases. Next: verify balance-state/history observations and test a small learned coordinated ankle correction with its own sign, amplitude and duration, quiet retention, both push signs and heldout timing/poses. Fixed diagnostic pulses and force labels must not enter learned execution. Evidence: logs/mimickit-training/direct-leg-response-20261009-01/audit.json.
## 2026-10-09 - Learned coordinated ankle pilot completed; promotion rejected

See LEARNED_ANKLE_RECOVERY_PROGRESS.md. Two fresh scalar ankle learners with live balance/history inputs completed 24,576 transitions each. Core 150 N X recovery improved from 7/32 to 11/32 and 9/32, but reserved-pose recovery was 2/8 and 0/8 versus 2/8, and overall recovery regressed from 70/128 to 49/128 and 61/128. Both retained sixty-second Stand, 16/16. No corrective steps. Commands were substantially active before impact and changed little during the first 200 ms relative to matched quiet rollouts, so the score gains do not yet establish reactive recovery. Input/control and independent 528-case audits passed. Arena unchanged. Next: frozen matched-prefix continuation versus quiet-command replay after impact to isolate stance preparation from actual learned reaction, before extending PPO. Evidence: logs/mimickit-training/learned-ankle-20261009-01/audit.json.
## 2026-10-09 - Frozen ankle response comparison completed

See ANKLE_RESPONSE_CAUSAL_PROGRESS.md. Preserving learned prefixes and replaying each policy's quiet ankle commands after the first impact interval changed settled recovery from 11/32 to 10/32 for seed 921009 and left seed 931009 at 9/32 with identical case identities. Only reserved pose 6, +X, onset near 1.0 seconds benefited from normal learned continuation; there were no replay-only successes. All other 63 outcomes matched. This removes only the ankle head's impact response: altered stance, continuing quiet commands and live frozen-source feedback remain. No corrective steps. Independent audit passed all 64 cases, exact prefixes, two queued intervals, command replay, geometry and outcomes. No training or Arena change. This supersedes the previous next priority: validate recovery training starts from identical frozen-source states with preserved queues/history, paired quiet/pushed continuations and exact zero-correction parity, then a bounded comparison that credits post-disturbance recovery rather than stance preparation. Evidence: logs/mimickit-training/ankle-response-causal-20261009-01/audit.json.

## 2026-10-09 - Matched recovery starts admitted

See RECOVERY_START_PROGRESS.md. Full native snapshots now preserve physical/solver state, cached contacts, task state, delayed commands and balance/action history, including isolated restores under an odd shared queue-cursor shift. Independent audit passed 12 rollouts/192 cases; eight traces exactly reproduced archived zero and pulse continuations. At signed 150 N X, zero correction recovered 7/32, negative ankle pulse 12/32 and positive pulse 15/32. Each pulse favors one sign and harms the other. Each command retained six-second quiet Stand in 32/32 cases; no corrective steps. These validate the start mechanism and reproduce reachability, not new learned recovery. No training or Arena update. Next: two fresh bounded recovery-start learners, 24,576 continuation transitions each, paired quiet starts and unchanged reward; evaluate both matched starts and continuously active execution, reserved poses/timings, broader directions/strengths and sixty-second quiet Stand. Reject any improvement dependent on a privileged takeover gate. Evidence: logs/mimickit-training/recovery-starts-20261009-01/audit.json.

## 2026-10-09 - Recovery-start learning completed; candidates rejected

See RECOVERY_START_LEARNING_PROGRESS.md. Two fresh heads trained for 24,576 continuation transitions each from exact source states after the first impact interval, with quiet starts and unchanged architecture, PPO, reward and limits. Both accepted all 288 actor updates without KL rollback. Core 150 N X recovery remained 7/32 with identical source success identities in both matched-start and continuously active execution. Overall scores were 70/128 and 69/128 versus source 70/128; both lost one reserved stronger-sideways case. Sixty-second Stand remained 16/16 each; no corrective steps. Independent audit passed 45 traces/720 cases. Early learned corrections were small (first-500ms RMS 0.001408 and 0.000364 rad). Reject promotion and longer continuation; Arena unchanged. Next: use saved training-pose probe reward traces to check whether the exact discounted objective prefers independently successful corrections over doing nothing, excluding prefixes and post-failure frames. Establish reward alignment before changing reward, exploration or training duration. Evidence: logs/mimickit-training/recovery-start-learning-20261009-01/audit.json and response_diagnostics.json.

## 2026-10-09 - Saved recovery reward audit completed

See RECOVERY_REWARD_AUDIT_PROGRESS.md. Analyzed 144 saved training-pose trajectories and 96 pulse-versus-zero comparisons, with no physics integration or training. All 16 rescuing pulses scored above zero correction; all five pulses that destroyed recovery scored below it. Every rescue had positive cumulative reward benefit within 200 ms. Both frozen final critics preserved those pushed rankings after time-limit bootstrap. Quiet measured returns preferred zero in 48/48 comparisons, though final-critic bootstrap reversed four and two small comparisons (including duplicate quiet worlds). Reward reconstruction and independent return/termination verification passed; reserved poses were excluded. Retain the reward. Next: bounded frozen-policy exploration/credit checks with actual sampled commands and the same 64-interval GAE, to distinguish poor useful-action coverage from poor advantage estimates before changing noise, objective or training duration. The audit did not reconstruct historical PPO advantages. Arena unchanged. Evidence: logs/mimickit-training/reward-alignment-20261009-01/results.json and audit.json.

## 2026-10-09 - Frozen exploration and credit check completed

See RECOVERY_EXPLORATION_PROGRESS.md. Audited 72 frozen continuations/864 training-pose cases, with no optimizer updates. Sampled pushes recovered 16/96, 18/96 and 18/96 for initial/921009/931009 versus repeated zero-control 20/96; they rescued 8/10/10 cases but each lost 12 baseline successes. Quiet recovery fell to 59/96, 58/96 and 60/96 versus 96/96. Paired own-critic GAE favored every sampled rescue. Same-window reward/critic decomposition found quiet sign reversals in 39/96 and 24/96 for the learned critics, versus 3/96 and 0/96 on pushes. Strong early helpful offsets were rare (2/96 with >=0.060 rad for six intervals), but most rescues did not require that threshold. No corrective steps. Verification passed after repairing the verifier's actor batch shape and accounting for float32 GAE rounding; no physics was repeated. Next: preserve identical sampled first 30 intervals, then remove further innovations while retaining conditional-mean memory and all native state, to isolate damage from continued noise before changing exploration or critic calibration. No training or Arena change. Evidence: logs/mimickit-training/recovery-exploration-20261009-01/audit.json and credit-decomposition.json.


## 2026-10-09 - Matched exploration noise cutoff completed

See RECOVERY_NOISE_CUTOFF_PROGRESS.md. Independent audit passed 48 frozen continuations / 576 training-pose cases, exact first-30-interval command/state prefixes and two delayed physical intervals. Removing only later Gaussian innovations raised pushed recovery from 16/18/18 to 23/23/23 out of 96 and quiet standing from 59/58/60 to 89/91/89. All actors recovered the same pushed case identities. Against zero correction (20/96 pushes, 96/96 quiet), each still lost 10 pushed successes and 5-7 quiet successes; average full return remained lower on both branches. No failures occurred before cutoff; no corrective footsteps. Learned-critic quiet reward-sign reversals persisted (44/96 and 28/96 versus zero), without establishing historical PPO causality. No training, checkpoints or Arena update. Next: frozen initial-actor comparison of half and quarter continuous innovation amplitude, same standardized draws and training starts, 32 trajectories / 384 cases. This tests quiet retention and useful correction coverage without a deployed impact clock, before selecting another training configuration. Evidence: logs/mimickit-training/recovery-noise-cutoff-20261009-01/audit.json and paired-credit.json.


## 2026-10-09 - Continuous exploration amplitude comparison completed

See RECOVERY_NOISE_SCALE_PROGRESS.md. Independent audit passed 32 frozen initial-actor trajectories / 384 training-pose cases. Half and quarter continuous noise both retained 96/96 quiet whole-episode Stand; push recovery was 23/96 and 21/96 versus full-noise 16/96, cutoff 23/96 and zero-control 20/96. Half rescued 9 zero-control failures but lost 6 successes; quarter rescued 3 and lost 2. Average full reward remained below zero correction on both branches, with quarter carrying the smaller penalty. No corrective steps or new learning. Nominate half amplitude for the next bounded learning comparison because it preserves quiet standing while retaining more useful corrections, not as a proven optimal setting. Next: two fresh recovery-start PPO seeds, 24,576 transitions each, identical initial actor and unchanged reward/hyperparameters except half innovation amplitude. Verify consistent sampling/log-probability/KL/evaluation scale; evaluate matched and ordinary execution, heldout timing/poses, signed 150/300 N X/Y and sixty-second quiet Stand. No Arena or production configuration change. Evidence: logs/mimickit-training/recovery-noise-scale-20261009-01/audit.json and paired-outcomes.json.


## 2026-10-09 - Half-noise recovery learning completed; both candidates rejected

See RECOVERY_HALF_NOISE_LEARNING_PROGRESS.md. Two fresh seeds completed 24,576 continuation transitions each at half innovation amplitude, with 288 accepted actor updates and zero KL rollbacks per seed. Independent audit passed 45 traces / 720 cases, probability and sampled-evaluation scale checks, initial-policy parity, source immutability and physical outcomes. Ordinary deterministic core recovery was 7/32 and 5/32 (heldout 1/8 and 0/8), and wider recovery 68/128 and 65/128 versus corresponding full-noise learners 70/128 and 69/128. Both retained sixty-second quiet Stand, 16/16. Matched recovery totals were 68/128 and 67/128. No corrective steps. Early pushed-minus-quiet ankle RMS remained small (0.000137 and 0.000329 rad), despite larger offsets in the second seed. First/last PPO batches show local surrogate improvements, not heldout recovery gains. Reject both; no Arena or production configuration change. Next: reconstruct and audit actual first-collection PPO updates, separating quiet/+X/-X credit and gradients to distinguish useful response from shared drift before changing learning rate or training duration. Evidence: logs/mimickit-training/recovery-half-noise-learning-20261009-01/audit.json, paired-outcomes.json and response_diagnostics.json.


## 2026-10-09 - Actual first-collection PPO replay and gradients audited

See RECOVERY_PPO_GRADIENT_PROGRESS.md. Reconstructed both first collections (1,536 saved samples), exact reset assignments, Gaussian actions, minibatch permutations, nine actor/critic updates per seed, recorded KL and bit-identical saved post-update outputs. Independent analytic gradients matched autograd within 1.82e-8. No new physics or policy checkpoints. The first update is not uniformly a shared offset: its constant component accounts for 0.83% and 53.99% of squared conditional-mean change in the two seeds. Branch gradient sums retain 88.0% and 62.6% of summed branch norms; quiet gradients do not dominate/cancel pushes consistently. Equal branch weighting leaves gradient directions similar (cosines 0.996/0.966), so imbalance alone is not a demonstrated fix. Initial critics were zero; this cannot diagnose later value-estimate effects. Next: reproduce the same bounded half-noise streams with compact per-update labels, gradients, fixed-state actor probes and critic statistics; require exact historical final weights, reset records and numerical history. Do not try a new configuration or repeat the physical evaluation suite if parity passes. Half-noise candidates remain rejected; Arena unchanged. Evidence: logs/mimickit-training/recovery-ppo-gradients-20261009-01/audit.json, results.json and weighting-sensitivity.json.


## 2026-10-09 - Full per-update historical replay completed

See RECOVERY_UPDATE_TRACE_PROGRESS.md. Both half-noise streams reproduced exactly: final actor/critic tensors, all numerical histories/resets/accounting, and first/last complete batches. Independent audit passed 64 collections / 49,152 replayed transitions. Early recovery (<30 continuation intervals) supplied only 16.99%/17.11% of samples; later gradients were larger in 30/32 updates for both seeds and opposed early gradients in 17/15 updates. Quiet occupancy was 39.39%/38.09%, but quiet did not consistently dominate branch gradients. Fixed pushed-minus-quiet conditional response fluctuated and ended small; larger intermediate probe response is not proven useful recovery. No new candidate, checkpoints or evaluation suite; historical rejections and Arena selection remain. A successor verifier separated GAE recurrence rounding from advantage normalization; no physics repeated for that repair. New evidence about 5.5 MiB. Next: two fresh bounded fits changing only actor-surrogate weights to raw5 for early continuation and1 later, normalized to whole-collection mean1. This gives about48% early mass on saved streams while avoiding sparse-group32x weights; preserve critic/reward/noise/full episodes and perform the established recovery/quiet/footstep evaluations. Evidence: logs/mimickit-training/recovery-update-trace-20261009-01/audit.json, interpretation.json and age-weighting-proposal.json.


## 2026-10-09 - Early-recovery actor weighting completed; no promotion

See RECOVERY_AGE_WEIGHTED_LEARNING_PROGRESS.md. Two fresh seeds completed 24,576 transitions and 288 accepted actor updates each, with zero KL rollbacks. Raw 5:1 actor weighting raised early effective mass to 48.51%/46.53% from 16.83%/16.18% of samples; critic/reward/noise/full episodes remained unchanged. Independent audit passed 64 collection objectives and 45 traces/720 cases, including exact initial batch and post-critic parity. Ordinary core recovery was 7/32 and 9/32, overall 67/128 and 70/128, versus unweighted 7/32 and 5/32, overall 68/128 and 65/128. Both retained 60-second Stand 16/16. Both matched-start core scores were 7/32 and totals 68/128; no corrective steps. Seed 931009's two added core successes over zero (training +X poses 3/4 at 1.7 s) disappeared from matched starts; its overall gains were offset by three reserved stronger-sideways losses. Reject both for promotion; Arena unchanged. A successor verifier corrected only float32 scalar-report rounding tolerance, with no physical rerun. Next: frozen quiet-command replay for seed 931009 at the two core timings/32 cases, preserving its live prefix through the first force interval, to test whether the two added recoveries require impact-dependent ankle response. This distinguishes preparation/quiet-command sufficiency from a training-start mismatch before another optimizer change. Evidence: logs/mimickit-training/recovery-age-weighted-learning-20261009-01/audit.json, paired-outcomes.json and response_diagnostics.json.


## 2026-10-09 - Weighted actor quiet-command replay completed

See RECOVERY_WEIGHTED_QUIET_REPLAY_PROGRESS.md. Frozen weighted seed 931009 retained 9/32 core recoveries when its ankle commands after the first force interval were replaced with its own quiet sequence. All 32 individual outcomes matched normal execution, including the two additional +X training-pose successes at 1.7 seconds. Reserved recovery remained 2/8; matched-start and zero correction each scored 7/32. A six-second quiet control matched every saved trace field and passed 16/16. Independent audit passed 48 cases, exact prefixes and queued intervals, command/memory/geometry/outcome checks, and independently reconstructed live source targets (max error 2.98e-8 rad). All 588 reused historical inputs matched the candidate lineage. A successor verifier repaired float32 hit-time metadata tolerance; no physics repeated. No corrective steps, training or new checkpoints; Arena unchanged. The two gains do not require this head's impact-dependent response; preparation, continuing quiet commands and live source feedback remain combined explanations. Next: 576 frozen training-only continuations (288 paired comparisons) across both weighted actors, onsets 44/74, six poses, quiet/+X/-X and four shared future-noise seeds, changing only the first innovation by +/- HALF_SIGMA. Compare 64-interval GAE components, bootstrapped truncated return, measured full return and recovery before another PPO change. Evidence: logs/mimickit-training/recovery-weighted-quiet-replay-20261009-01/audit.json and paired-outcomes.json.


## 2026-10-09 - Paired first-action credit completed

See RECOVERY_LOCAL_ACTION_CREDIT_PROGRESS.md. Independent audit passed 64 frozen rollouts / 576 continuations / 288 pairs, with exact historical starts, common future innovations, frozen predictions, physical gates and credit arithmetic. For both weighted actors, all 96 pushed pairs agreed across GAE, same-window reward and full episode return: +X favored negative first innovation, -X positive. All 24 pushed start conditions per actor retained that preference across four noise sequences. GAE preferred the recovering arm in all 40 changed pushed outcomes (16/24 by actor). Quiet GAE/full-return rankings disagreed in 12/48 and 10/48 individual pairs, and 5/12 and 6/12 averaged starts; the one quiet outcome reversal did not isolate critic error. No corrective steps, optimizer updates or new policies. Arena unchanged. This does not support a local pushed-credit failure; historical optimizer behavior remains unresolved. Next: controlled offline actor updates on the saved 288 first-decision samples per actor, comparing ordinary and pair-centered advantages with the same original standard deviation. Verify initial full-gradient invariance, then use identical fresh Adam/minibatches to measure conditional response, quiet drift and minibatch variance. Six diagnostic minibatch updates per variant, no physics/checkpoints. Evidence: logs/mimickit-training/recovery-local-action-credit-20261009-01/audit.json, paired-credit.json and results.json.


## 2026-10-09 - Offline paired-credit actor updates completed

See RECOVERY_LOCAL_ACTOR_UPDATES_PROGRESS.md. Both weighted actors were copied into ordinary and pair-centered objective arms using the same 288 saved first-decision samples, original advantage standard deviation, permutations and fresh Adam settings. All four arms accepted six updates; independent audit exactly replayed all 24, verified source data, probability/gradient arithmetic and 725 input hashes. Initial full-gradient relative differences were 4.25e-7 and 9.21e-7. Centering reduced initial minibatch deviation in all 12 comparisons, but increased quiet movement alongside useful response. All arms changed pushed-minus-quiet responses in the preferred direction at 24/24 starts; raw pushed changes favored only 12/24 except centered actor 931009 at 14/24. Centered response RMS grew 2.31/2.47 times, while quiet RMS grew 2.20/7.02 times. Opposing push gradients had cosine -0.9874/-0.9926; matched pushed/quiet Jacobians were highly aligned. Shared movement exists in the full gradient before minibatching or Adam, so pair centering alone is not a demonstrated fix. No physics, critic updates, new checkpoints or Arena changes; no recovery improvement claimed. Evidence occupies about 24.62 MiB. Next: reconstruct and independently verify quiet-Jacobian projection of the full gradient, then compare small original/projected scratch displacements at the same scalar step and KL guard, testing whether useful raw push changes survive with minimal quiet change. This remains a local feasibility diagnostic before bounded quiet-constrained learning. Evidence: logs/mimickit-training/recovery-local-actor-updates-20261009-01/audit.json and results.json.

## 2026-10-09 - Quiet-output projection feasibility completed

See RECOVERY_QUIET_ACTOR_PROJECTION_PROGRESS.md. Both actors passed the independently verified quiet-output projection and four finite scratch displacements. All three thresholds retained rank 12; projected gradient norms retained 36.96% and 68.17%. At the same step scale per actor, projected raw push changes favored 24/24 directions versus 12/24 for the original direction. Projected quiet RMS was 3.07e-9 and 2.71e-8; all quiet changes stayed below the 1e-7 latent sign band. Pushed RMS was 2.20e-5 and 3.29e-5. The audit exactly reconstructed float32 application, predictions and KL, verifying 605 input hashes. This establishes local fitted-state feasibility, not physical recovery or generalization. No optimizer, critic, physics or checkpoint; Arena unchanged. Evidence occupies about 7.28 MiB.

Next: two fresh seeds with 24,576 transitions each, starting from the original zero-correction head and holding the age-weighted recipe fixed. Project actual Adam displacements using 12 current quiet-anchor Jacobians, with cumulative maximum anchor drift 1e-5 and the existing KL limit 0.01. Anchor targets come from the fresh head. Compare the sealed matching controls and run the established 720-case recovery, quiet-standing and footstep evaluation. Evidence: logs/mimickit-training/recovery-quiet-actor-projection-20261009-01/audit.json and results.json.


## 2026-10-09 - Quiet-constrained recovery learning completed; no promotion

See RECOVERY_QUIET_CONSTRAINED_LEARNING_PROGRESS.md. Two fresh seeds completed 24,576 continuation transitions each. Independent audit passed 45 traces / 720 cases, exact first-collection actor/Adam/critic replay, all 32 matching minibatch permutations and 898 input hashes. Core recovery remains 7/32 each, with exact zero-control settled case membership. Continuous and matched totals are 70/128 and 69/128 versus zero 70/128: seed 921009 matches every settled outcome, while seed 931009 adds one failure at 300 N -Y, 1.0 seconds, reserved pose 7. Both retain 60-second quiet Stand 16/16 and matched quiet 32/32; sampled core is 10/32 each. No corrective steps. Both candidates rejected; Arena unchanged.

Only 83/82 of 288 planned actor updates were accepted; 24/25 rejected and 181 skipped each. All rejections are anchor-only. Independent projection disagreement is below 9e-16, but finite nonlinear quiet drift exhausts the cumulative 1e-5 bound. Seed 921009 accepts nothing after collection 10; seed 931009 resumes in collections 31/32. This diagnoses rejection, not proof that rejections caused the physical non-gain. Evidence is about 202.38 MiB, with initial/final actor and final critic only; no intermediate checkpoint series.

Next: exactly replay each first-collection ninth rejected candidate, then apply one minimum-norm Newton correction at its current quiet Jacobian toward original zero targets (same 1e-6 cutoff, explicit float32 application, no retries or parameter search). Compare actual anchor residual, KL, minibatch/full-collection weighted objective, correction size and pushed-versus-quiet probe movement. Independently verify before considering another learner. No new trajectories or automatic training continuation. Evidence: logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01/audit.json, comparison-results.json and response-diagnostics.json.


## 2026-10-09 - Finite quiet-anchor correction completed; mixed objective retention

See RECOVERY_QUIET_ANCHOR_RETRACTION_PROGRESS.md. Exactly replayed each seed's first-collection ninth rejected candidate and applied one minimum-norm correction toward original zero quiet targets. Both anchor maxima fall from about 1.1e-5 to below 5.3e-11, with KL below 2.4e-7. The independent audit passes 614 input hashes, 10 output hashes, all 18 actor/Adam attempts, candidate Jacobians, float32 application and predictions. No retries or source repair. Actual corrections are only 0.362%/0.287% of the rejected parameter step, yet seed 921009 loses its proposed full/minibatch objective gain (-3.64e-6/-7.03e-6 versus pre), while seed 931009 retains positive gains (+2.21e-6/+6.91e-6). Conditional pushed probe response remains; most repair movement is shared with quiet outputs. The correction resets accumulated drift from eight earlier accepted updates, not only new ninth-step drift. No physical recovery conclusion or training continuation; Arena unchanged. Evidence is about 12.46 MiB; runner 3.56 seconds, no deployable checkpoints or new trajectories.

Next: compare one correction toward the PRE-NINTH quiet outputs on the same two fixed candidates/Jacobians, retaining the original ZERO-referenced 1e-5 guard, KL limit, cutoff, float32 application and unchanged tentative Adam moments. Compare both full/minibatch objective and fixed-state probe changes against the sealed zero-target arm, with independent verification and no retries/search. This isolates accumulated-drift reset cost; it does not establish a better fresh-training algorithm. Decide before another learner. Evidence: logs/mimickit-training/recovery-quiet-anchor-retraction-20261009-01/audit.json and results.json.


## 2026-10-09 - Local quiet-target correction completed; bounded fresh learning next

See RECOVERY_QUIET_ANCHOR_TARGETS_PROGRESS.md. On the same two sealed rejected candidates and Jacobians, one correction toward pre-update quiet outputs preserves those outputs within 7.91e-11/1.05e-10 while passing the unchanged ORIGINAL ZERO anchor bound and KL limit. Both full/minibatch surrogate gains remain positive versus pre-update: 1.91e-6/1.67e-6 and 5.81e-6/8.79e-6. This avoids seed 921009's zero-target correction reversal. Conditional pushed probe response remains. Independent audit passed 617 input hashes, eight outputs, exact prior-state parity, independently reconstructed Jacobians, correction geometry and predictions/objectives. No optimizer replay or operations, physical rollout, new training or deployable checkpoint. Evidence about 6.04 MiB; runner 0.93 seconds. Arena unchanged.

Next: two fresh 24,576-transition seeds from the same original zero head, retaining the age-weighted recipe. After each projected Adam candidate, apply exactly one candidate-Jacobian correction toward that step's pre-update quiet outputs, then check the original zero-reference 1e-5 bound and collection KL 0.01. Preserve existing actor/Adam rollback, collection stop, critic updates and nominal actor slots; no retries or surrogate acceptance rule. Log local residual separately from cumulative drift; verify initial collection/critic parity, permutations and independent update accounting. Run the established 720-case recovery/quiet/footstep evaluation against projected-only, weighted and zero controls. Fresh correction starts near zero, so the offline result does not establish which target trains better or that more accepted updates improve recovery. Require reproducible physical gains before promotion. Evidence: logs/mimickit-training/recovery-quiet-anchor-targets-20261009-01/audit.json and results.json.


## 2026-10-09 - Retracted learning completed; numerical blocking fixed, no promotion

See RECOVERY_RETRACTED_LEARNING_PROGRESS.md. Both fresh seeds completed 24,576 transitions and accepted all 288 actor updates each, versus 83/82 in projected-only controls. No rejection or skipped update; maximum cumulative zero-anchor drift 1.13e-8 under the unchanged 1e-5 bound. Every-step finite correction prevented accumulation; median correction sizes were 0.0195%/0.0290% of projected steps. Independent audit passed 1,186 input hashes, 273 outputs, exact first-collection actor/Adam/critic replay, all permutations and 45 traces/720 cases. A preflight-only historical diagnostic-source mismatch was recorded and repaired in successor wrappers before any training; no physical rerun. Evidence: recovery-retracted-learning-20261009-02; -01 preserves the stopped source snapshot.

Physical results remain 7/32 core and 70/128 or 69/128 overall in both execution modes, with exact settled-case parity against projected-only controls including sampled 10/32 each. Seed 931009 retains the single zero-control loss at 300 N -Y, 1.0 seconds, reserved pose 7. Both pass sixty-second Stand 16/16 and matched quiet 32/32; no footsteps. Both candidates rejected; Arena unchanged. Early conditional command RMS is smaller despite more accepted updates. Median KL is 8.56e-8/1.28e-7, maximum below 8.77e-6 versus limit 0.01; no PPO ratio clipping. Fits took 124.01/122.82 seconds; successful evidence about 214.77 MiB, with six initial/final actor/critic files and no intermediate checkpoint series.

Next: one fixed actor learning-rate contrast, 1e-4 to 1e-3, using the same fresh seeds, original zero initialization and 24,576-transition budgets. Keep critic rate 3e-4, all other settings, projection plus one correction, original guards, rollback and 288 nominal actor slots unchanged. No rate grid, relaxed limits or compensating epochs. Audit initial samples/critic/permutation parity and optimizer math; compare actual KL, clipping, acceptance and all 720 physical cases against the sealed retracted and zero controls. A 10x actor rate does not imply 10x faster learning. Require reproduced physical gains with quiet retention before promotion. See audit.json, comparison-results.json, response-diagnostics.json and update-scale-diagnostics.json under logs/mimickit-training/recovery-retracted-learning-20261009-02.


## 2026-10-09 - Fixed actor-rate contrast completed; larger updates regress recovery

See RECOVERY_RETRACTED_ACTOR_RATE_PROGRESS.md. Two fresh seeds completed 24,576 transitions each with actor Adam 1e-3 versus sealed 1e-4 controls; critic remained 3e-4 and the complete recipe/guards/budget stayed fixed. Both accepted all 288 actor updates. Independent audit passed 1,465 input hashes, 273 outputs, exact first-collection actor/Adam/critic replay, matching initial data/critic/permutations and 45 traces/720 cases. No repairs or physical reruns. Evidence about 215.06 MiB; six initial/final actor/critic files, no intermediate checkpoint series.

Seed 921009 falls from 7/32 to 5/32 continuous core, overall 70/128 to 68/128; matched core 6/32, overall 69/128. Losses are reserved -X poses 6/7 at 1.0 seconds; pose 7 remains lost with matched starts. Seed 931009 retains core 7/32 but overall falls 69/128 to 67/128 in both modes, adding reserved pose-7 losses at 300 N +Y/1.0 seconds and -Y/1.7 seconds. No new deterministic successes over either low-rate or zero controls. Sampled core 9/32 and 10/32; quiet Stand 16/16 and matched quiet 32/32 each. No footsteps. Both candidates rejected; Arena unchanged.

Median KL rises about 275/286 times to 2.35e-5/3.65e-5; maxima remain below 0.00117 versus limit 0.01. PPO surrogate clipping remains rare (18/21 of 73,728 reused rows); gradient clipping is frequent (218/236 of 288 attempts). Finite correction keeps every final candidate within limits, although 247/234 intermediate candidates violate a guard. Maximum accepted anchor drift 1.856e-7/7.892e-6. First/last saved own-batch objective gains are negative for seed 921009, positive for seed 931009. Bigger accepted changes do not establish useful learning.

Matched first-500 ms conditional command RMS grows 7.79/9.65 times, but quiet command RMS grows 51.94/35.99 times. High-rate quiet RMS on reserved poses is 1.371e-3/5.731e-3 rad versus 4.16e-5/9.38e-5 on seen poses. All windows retain 30 live intervals. The 12 training anchors constrain only poses 0-5 at onsets 44/74 with zero memory; they do not guarantee zero quiet commands elsewhere. This measured generalization issue does not prove the cause of each failure.

Next: one frozen offline command-direction and quiet-generalization audit. Evaluate both final rates/seeds on the same existing 36 training probe inputs, reporting raw and pushed-minus-quiet means, reference sign agreement, amplitude and requested ankle additions. Separately decompose archived deterministic quiet/push commands across all eight poses and both existing core timings into learned innovation = latent - 0.9*previous_latent and memory, using the fixed first-30-decision windows. Keep same-input inference distinct from own-trajectory evidence and preserve held-out status. Independently verify hashes, pairing, arithmetic and action mapping. No new training/physics, checkpoints, rate sweep, longer continuation or Arena promotion. Evidence: logs/mimickit-training/recovery-retracted-actor-rate-20261009-01/audit.json, comparison-results.json, response-diagnostics.json and update-scale-diagnostics.json.

## 2026-10-09 - Frozen command direction and quiet generalization completed

See RECOVERY_COMMAND_GENERALIZATION_PROGRESS.md. Independent audit passed 684 input hashes, 44 outputs, four frozen heads/144 same-input predictions, and 16 archived windows/256 paired case windows/7,680 paired rows. The independent NumPy network differs by at most 4.09e-9. All probe pairing, sign arithmetic, action mapping, memory chronology and summaries pass. No repairs, optimizer updates, new physics or checkpoints. Runner 3.26 seconds; evidence about 11.85 MiB. Arena unchanged.

Both low-rate heads and high-rate seed 931009 match the historical pushed reference at 24/24 fixed states. High-rate seed 921009 matches only 12/24, reversing poses 2/3/4 at both onsets and signs, even after subtracting quiet means. Immediate conditional latent RMS grows only 1.61/6.91 times across rates; higher-rate values are 0.0576%/0.1419% of HALF_SIGMA. The older first-action reference is not proof of current optimality or optimal response throughout the later 500 ms.

All first matched decisions have exactly zero prior latent/memory. Reserved-pose first quiet RMS nevertheless grows from 3.470e-5 to 1.822e-3 and from 2.110e-4 to 7.603e-3 latent units, about 52.5/36.0 times. Higher-rate pose-7 quiet outputs are approximately -0.002288 for seed 921009 and +0.01027 to +0.01045 for seed 931009. The quiet bias therefore precedes action-memory accumulation. First matched raw/reference agreement is 29/32 and 27/32 at low rate versus 18/32 and 27/32 at high rate; subtracting quiet gives 32/32 and 32/32 versus 20/32 and 32/32. Later memory and innovation are correlated; energy identities retain their cross terms, so component RMS is not an independent causal attribution. No physical recovery improvement is claimed.

Next: a bounded 12-versus-48 training-only quiet-anchor feasibility comparison. Capture a frozen zero-head/source quiet prefix at onsets 40/44/48/54/66/74/82/90, poses 0-5, preserving exact old44/74 features and zero memory. Keep poses6/7 and evaluation onsets60/102 excluded. Exactly reconstruct each seed's first low-rate Adam proposal and apply the original12 and expanded48 projection plus one correction to that identical proposal, with unchanged cutoff/guards/arithmetic. Score BOTH on all48 quiet states and the24 pushed probes, checking rank, retained step, surrogate gain, residuals and reference directions independently. No PPO training or deployable checkpoint. This tests first-update compatibility, not later drift or reserved-pose recovery; a useful result would justify a later bounded fresh learning comparison. Evidence: logs/mimickit-training/recovery-command-generalization-20261009-01/audit.json, results.json and per-policy probe/window arrays.
## 2026-10-09 - Expanded quiet coverage feasibility passed

See RECOVERY_QUIET_COVERAGE_PROGRESS.md. One frozen 91-interval quiet prefix (1,456 world intervals) exactly reproduces all 24 archived trace fields. The expanded bank has 48 training-only states at onsets 40/44/48/54/66/74/82/90 and poses 0-5; original12 features at44/74 match exactly. Reserved poses6/7 and evaluation onsets60/102 remain excluded. Each seed's first low-rate Adam proposal was reconstructed once and shared by both constraint arms. Original12 actor/optimizer/hash/gradient/anchor/KL parity is exact.

Both arms pass own-bank and all48 guards. Expansion lowers maximum all48 quiet output from 9.693e-7/2.227e-6 to 4.002e-12/4.366e-12, while retaining 48.92%/53.92% of the Adam proposal norm versus 60.81%/68.87%. Projection and correction ranks are48 versus12, with the unchanged1e-6 cutoff. All full/minibatch surrogate gains remain positive, including independent float64 readback; gains shrink. Paired pushed response RMS increases1.05/1.43 times rather than collapsing, but requested additions remain only about1.26e-7/1.12e-7 rad after this first step. This is technical feasibility, not physical improvement.

All four first-step candidates oppose the historical direction reference24/24, raw and conditional. This disagreement exists in both banks; it is not an expansion-induced reversal. Both final lower-rate trained heads previously aligned24/24. Historical labels are not current zero-head optimality or a new training target. No new direction-tuning branch is warranted by this comparison alone.

Independent audit passed683 inputs,11 outputs, prefix phase/normalization/history, both exact Adam proposals, four candidates, independent QR/reduced-SVD projection/correction, objectives and probe signs. Runner14.64 seconds; evidence about40.99 MiB. Runner and audit each reconstruct two historical Adam steps; audit adds no physics. Zero new PPO transitions, critic updates or deployable checkpoints. No repairs or repeated physical runs. Arena unchanged.

Next: a bounded fresh two-seed learning comparison with the sealed48-state bank, original zero initialization, actor1e-4, critic3e-4 and24,576 transitions per seed. Keep every other training/control setting and guard fixed. Use sealed low-rate12-anchor and zero-head controls; verify exact first-collection/critic/permutation parity. Save compact per-collection probe summaries without intermediate checkpoint series or label tuning. Run the complete established physical suite and compare paired gains/losses plus quiet generalization, including reserved cases. Require reproduced physical gains with quiet retention before promotion. No rate sweep, longer continuation or automatic Arena promotion. Evidence: logs/mimickit-training/recovery-quiet-coverage-20261009-01/audit.json, results.json, bank.npz, prefix.npz and per-seed evidence/report files.

## 2026-10-09 - Expanded quiet coverage learning completed; no promotion

See RECOVERY_COVERAGE_LEARNING_PROGRESS.md. Both fresh seeds completed 24,576 transitions with 48 training-only quiet anchors, actor rate 1e-4 and critic rate 3e-4. All other training/control settings remained fixed. Both accepted 288/288 actor attempts with no rejection, skipped slot or active PPO clipping. The first collection/critic, every minibatch permutation and first candidate's feasibility parity passed independently.

Core recovery remains 7/32 for both candidates; sampled core remains 10/32 with unchanged case membership. Quiet Stand passes 16/16 sixty-second cases and 32/32 matched cases per candidate. No footsteps. Seed 921009 falls from 70/128 to 69/128 in both continuous and matched evaluations, losing reserved pose 7 under 300 N negative Y at 1.0 seconds. Seed 931009 regains that case only continuously (69/128 to 70/128, matching zero correction); matched stays 69/128. There is no new success beyond the zero head or reproduced gain. Neither candidate is promoted; Arena unchanged.

Final common-input conditional response RMS is only 19.8%/10.6% of the 12-anchor controls. Seed 921009 reverses all 24 historical reference directions at collections 28-32 after an aligned stretch; seed 931009 ends aligned. These labels are diagnostic, not current optimality. Seen-pose quiet bias improves at reserved times, but first-decision reserved-pose bias worsens 3.23 times for seed 921009 and improves to 29.4% of control for seed 931009, before memory accumulation. Fitting more quiet states is not a general quiet-policy guarantee.

Maximum accepted fitted-anchor error is 2.421e-9/1.781e-9. Median retained parameter displacement is about 99.9% for both seeds, close to controls, despite much weaker output response. Audit passed 1,036 inputs, 277 outputs, 45 traces/720 cases, constrained-update records, first actor/Adam/critic replay, and 33-boundary probe arithmetic/hash linkage. Initial/final probe predictions are independently reproduced; intermediate models were not retained or replayed. Analysis contains 1,280 paired records. Runner 695.05 seconds; evidence about 352.23 MiB; six initial/final actor/critic files, no intermediate checkpoint series. No repairs or repeated physical runs.

Next: one offline sensitivity audit of the four frozen final 12/48-anchor actors. On each SAME actor, compare both quiet banks using paired conditional Jacobians (pushed minus matching quiet), projected row norms and conditional spectra. Measure output sensitivity per unit parameter displacement, checking cutoff/subspace-nesting effects. A decrease under exact nested constraints is expected; its magnitude and consistency matter. Keep this a local geometry diagnostic, not a claim about optimal action signs or physical returns. No optimizer, physics, new checkpoint, extra gradient branch, rate sweep or longer training. Evidence: logs/mimickit-training/recovery-coverage-learning-20261009-01/audit.json, comparison-results.json, command-generalization.json, update-scale-diagnostics.json and fixed-probes files.

## 2026-10-09 - Independent stepping curriculum pilot completed; transfer gate failed

The user approved moving the main priority from scalar ankle diagnostics to learned leg/foot placement. STEPPING_SKILL_PLAN.md supersedes the unexecuted quiet-Jacobian diagnostic as the next action. Historical results and sealed sources remain unchanged. See STEPPING_SKILL_PROGRESS.md for the current result and next priority.

Implemented a commanded transfer/lift/placement curriculum with 33 independent joint corrections, measured phase progression, and exact zero-initial Stand behavior. This is a commanded movement learner, not an autonomous ball-response policy. All 18 focused checks passed. The original active PD, joint/torque limits, two-command delay and zero passive springs remain unchanged; no Arena export or C# build occurred.

Both fresh seeds completed 49,152 transitions and 576 accepted actor updates: 98,304 transitions total, with no KL rollbacks. Seed 941009 completed 0/16 transfer-and-return cases and survived 0/16; seed 951009 completed 0/16 and survived 4/16, all on reserved poses. Zero-head baseline retained Stand in 16/16 and completed zero transfers. Each candidate retained 16/16 sixty-second quiet cases through its explicit disabled-command fallback. Neither passed the prospective training-pose progression gate; lift, placement, changed-timing and ball evaluations did not run.

The independent audit passed 662 inputs, 34 sealed outputs and 8 traces/128 cases, including physical geometry, command/dwell/reward reconstruction, target limits/delay, exact initial native parity, first/last GAE/probability and update accounting. No intermediate optimizer replay is claimed. Runner duration was 464.76 seconds. Six initial/final actor/critic files and about 138 MiB of evidence were retained, without periodic checkpoints. Evidence: logs/mimickit-training/stepping-skill-20261009-01/audit.json and diagnostics.json.

Saved commands show appreciable corrections (first-500-ms RMS 0.04621/0.02690 rad) but almost parallel initial actions for opposite-side commands from matched physical states. Neither deterministic policy has a valid transfer interval. First/last saved batches show no completed transfer, covering only 2/64 collections per seed. This identifies missing useful coordination without establishing a unique cause or mechanical impossibility.

Next: establish physically successful, side-specific transfer-and-return examples through the exact current interface and bounds. If feasible, teach the neural learner from exact issue-time examples and nearby states, then require learned closed-loop transfer in both seeds before lift or further PPO. The teacher remains training evidence only. Do not extend the rejected run, restore passive springs, restart the arm-pose prerequisite, resume ankle sweeps or promote a scripted controller as learned recovery.

# Stationary Stand foundation

This experiment trains a time-independent balance actor before physical-ball
recovery. It uses `continuous-stand-02/stationary_reference.npz`: eight independent
two-foot postures, zero target velocity and acceleration, and phase fixed at zero.
There is no motion endpoint, step clock, prescribed stepping foot, or external
support force. The existing rig, contact-v2 observations, 30 residual targets,
two-step command delay, motor limits and physics-rate PD remain authoritative.

## Selected Stand foundation (2026-09-27)

Checkpoint: `logs/mimickit-training/stand-foundation-anchor-tight-01/best.pt`,
selected update 16, SHA-256
`9d6abf9a3c2227828243d264ce1778e498e23bd742544421d5cb498b21a0a28d`.
Use `best.pt`; the final training `model.pt` regressed and is rejected.
The selected actor passes all eight native and GPU 30-second gates. Its portable
bundle is `logs/mimickit-training/stand-foundation-anchor-tight-01/export/`;
`stand.onnx` matches PyTorch on 2,048 recorded observations with maximum absolute
action error 8.94e-8. `validation.json` binds both long/randomized reports to the
checkpoint, reference, plant and export contract. The final bounded run completed
128 updates / 8,388,608 transitions in 729.875 seconds including its checks.
The report corrects the runner's generic time-budget label to the actual
iteration-budget exit and preserves that correction explicitly.

Native evidence for this exact checkpoint:

- `stand-foundation-long-04`: **8/8 uninterrupted 600-second holds**. Every
  post-startup sample meets the standing gate. Minimum per-foot load 219.51 N;
  maximum foot error 4.447 mm; worst per-case mean root error 17.017 mm.
- Maximum foot motion over the final minute is **0.0138 mm**, and the maximum
  shift between consecutive final ten-second mean root positions is 0.00126 mm.
  The former creeping posture, case 7, has no measurable final-minute foot motion
  at the stored float32 position precision.
- `stand-foundation-unseen-04`: **64/64 fresh 60-second velocity starts**, seed
  270932, linear components +/-0.04 m/s and angular components +/-0.08 rad/s.
  Minimum per-foot load 214.18 N; maximum foot error 5.327 mm; maximum foot motion
  in the final ten seconds 0.0964 mm. This suite was first run after selection.

These results establish the stationary seed for reactive Perturb training. They
do not establish physical-impact recovery. The finite-reference Godot viewer
selection remains unchanged; a stationary adapter and unseen physical-hit
validation are required before loading the eventual recovery policy there.

## Diagnosed blockers

The selected finite-reference Stand v2 actor passes only three of eight stationary
30-second cases. A new native run on 2026-09-27 reproduced all prior per-case
measurements exactly (`standing-foundation-audit-20260927`). Passing cases carry
at least 232 N per foot in the stored control-rate load samples. Failed cases
also retain foot loads before losing posture; detecting contact is insufficient.

The subsequent native 600-second baseline (`stand-foundation-baseline-long-01`)
passes only case 6. Case 0 falls at 94.874 s and case 5 at 40.053 s, despite both
passing the earlier 30-second gate. The other five fall within 28.186 s. The
fixed reference has no endpoint in this test: these are closed-loop failures,
not clip expiry. One passing ten-minute case is positive evidence that the
unchanged plant can sustain a stationary stance, not proof of robustness across
all eight postures or initial states.

The finite-reference actor's frozen normalization clips 12–26 future-reference
channels at the stationary starting poses. This is a transfer mismatch, not a
proof that clipping alone causes the falls. Previous stationary continuations
also terminated after their first measured regression, at eight or 16 updates.

A fresh-training control exposed a separate numerical defect in the upstream
float32 running moments. For the eight target root heights, the actual standard
deviation is 0.5600105 mm; repeated moment updates report the 0.1 mm floor and
clip legitimate inputs. A matched isolated fixture reproduces this. The opt-in
`stable_moments_v1` normalizer uses centered, merged moments with double-precision
accumulation and retains the standard serialized mean/std/count and inference
mapping. It does not reinterpret an existing exported actor.

The fresh control (`stand-foundation-fresh-01`) was stopped after its saved
update-128 evaluation to test this correction. It remains 0/8, with 3.85–7.18 s
native survival. Its original runner and normalization diagnosis are preserved
in the run directory. This control has no final GPU validation or export.

## Training and validation

`train_stand_foundation` uses upstream PPO directly with a fresh actor, critic and
normalizer. It keeps physics episodes alive across logging and native evaluation.
Only falls and the 30-second collection timeout reset worlds. Native checkpoint
selection preserves every incumbent passing posture; a rejected evaluation cannot
overwrite `best.pt`. A bounded fresh learning run may continue after rejection.

The corrected run changes only observation-moment accumulation relative to the
fresh control: same seed, stationary targets, reward, PPO settings and physical
controller. Later training changes require their own recorded configuration.

The corrected run and its continuation (`stand-foundation-stable-01/02`) reach
8/8 native and GPU 30-second passes after 257 updates / 16,842,752 transitions.
All eight cases subsequently survive 600 seconds, but only seven pass the full
fixed-stance gate. Case 7 exceeds 3 cm foot error at 262.454 s and reaches 9.08 cm
by the end while both feet remain loaded. The first unseen velocity-start suite
passes 64/64 at 60 seconds. The candidate is rejected for foundation export;
neither survival nor the short randomized suite overrides the long foot-drift
failure. That first random suite is now observed development evidence.

The opt-in quiet-standing objective `standing_foot_anchor_sigma_m` multiplies
the existing tracking reward by `exp(-mean(foot_position_error_squared)/sigma^2)`.
The corrective run uses sigma=0.02 m, applies it equally to both feet, and changes
no physical or observation contract. This reward belongs to quiet standing;
future reactive Perturb training must permit foot placement changes after hits.
Changing this objective on a resumed actor requires frozen-actor critic
calibration. `stand-foundation-anchored-01` passes the existing calibration gate
(balanced validation MSE 28.54 to 4.50) before actor updates, retains 8/8 short
standing through 32 updates, and selects update 24. Its long validation and a
fresh unseen seed are separate from the previously observed suite. The result
still fails case 7 at 600 seconds: 9.11 cm maximum foot error, although it passes
all eight cases at 360 seconds. The other seven converge. Passing 64/64 fresh
60-second velocity starts does not override this failure.

`stand-foundation-late-capture-01` records real settled native states every thirty
seconds through 330 seconds (eleven states per posture). Maximum recorded motor
use is 29.6% of its torque limit, with actions within +/-0.247. The late curriculum
keeps 50% ordinary starts and draws 50% from the matching posture's captured
states. It restores joint positions/velocities, pending targets in chronological
order, the active target, and the last-solve contact observation. The subsequent
transitions are freshly simulated on-policy. No artificial age is exposed to the
actor. Native evaluation always starts from the original stationary poses.

A reset test verifies posture correspondence, contact observations and delay-ring
ordering across a changed destination cursor. The bank is bound to reference and
plant hashes; its own hash and source checkpoint provenance are retained.
The first 400-step critic fit was rejected because several posture groups
regressed despite lower average error. The 1,600-step continuation experiment
passes calibration with balanced validation MSE 56.44 to 6.25 before PPO begins.
`stand-foundation-late-02` completes 64 PPO updates and retains update 16. Updates
48 and 64 regress on original-start standing and are rejected. The retained actor
passes the eight-case native and GPU 30-second gates. It is evaluated separately
by `stand-foundation-long-03` and the new-seed `stand-foundation-unseen-03` suite.
Those runs reject it: all worlds survive, but only 5/8 long holds and 39/64
randomized starts satisfy the foot-convergence gate. Cases 6 and 7 reach about
10 cm foot error, and case 3 still moves 1.20 mm during the final minute.

The next run (`stand-foundation-anchor-tight-01`) returns to the earlier
seven-posture-converged actor, tightens the foot reward scale from 20 to 5 mm,
and reduces the actor learning rate from 1e-4 to 2.5e-5. It retains the same real
late-state curriculum and passes frozen-actor critic calibration before PPO.
Checkpoint ranking now prioritizes worst final-five-second foot motion and
worst mean foot error before root error, after all existing standing/survival
criteria. Losing any incumbent passing standing case still rejects a candidate.

```powershell
& $python -m mujoco_rig.mimic.train_stand_foundation `
  --mimickit $env:MIMICKIT_PATH `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --config logs/mimickit-training/stand-support-v2-15min-01/agent.yaml `
  --out logs/mimickit-training/<new-run> --seconds 900 `
  --stable-normalization --stop-on-pass
```

`validate_stand_foundation` adds stricter native validation after the existing
eight-case 30-second gate passes. Required evidence before foundation export:

- All eight quiet postures hold for 600 seconds in one uninterrupted episode.
- At least 64 unseen velocity-start cases hold for 60 seconds. Root linear
  velocity components are sampled within +/-0.04 m/s and angular components
  within +/-0.08 rad/s; positions and physical contact geometry are unchanged.
- Mean root error is at most 5 cm; the last ten seconds remain settled.
- After two seconds of startup, both feet including toes continuously carry
  at least 5 N at the control-rate sampling points, foot error stays within
  3 cm, and at least 99% of samples meet the existing settled posture/speed limits.
- The last two consecutive ten-second mean root positions differ by at most 1 cm.
- Foot motion during the final 60 seconds is at most 1 mm (final ten seconds
  for the 60-second randomized-start suite), to reject ongoing slow creep.
- Failed worlds never contribute again after a maintenance reset.

Reward-free validation advances the same native control and physics sequence.
An exact-prefix test compares observations, contact snapshots and termination
against ordinary task stepping. Validation records the reference, checkpoint,
model and runner hashes and saves scalar/position traces.

```powershell
& $python -m mujoco_rig.mimic.validate_stand_foundation `
  --mimickit $env:MIMICKIT_PATH --reference $reference --checkpoint $checkpoint `
  --seconds 600 --out logs/mimickit-training/<new-long-check>
& $python -m mujoco_rig.mimic.validate_stand_foundation `
  --mimickit $env:MIMICKIT_PATH --reference $reference --checkpoint $checkpoint `
  --seconds 60 --cases 64 --velocity-noise .04 --seed 270928 `
  --out logs/mimickit-training/<new-unseen-check>
```

`export_stand_foundation` requires matching passing reports and checks ONNX action
parity. Its bundle explicitly uses stationary reference semantics and is not
silently substituted into the finite-reference Godot adapter. Stand acceptance
does not establish impact recovery. Future Perturb training must begin standing,
randomize physical hits, permit either foot to respond, retain quiet standing as
a checkpoint gate, and validate a new unseen hit suite before Godot selection.

Ten-minute holds are empirical evidence of sustained standing, not a mathematical
proof of infinite stability.

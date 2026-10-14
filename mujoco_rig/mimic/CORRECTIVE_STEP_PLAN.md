# Corrective stepping from the current Stand policy

2026-10-07. Status: bounded pilot complete; no corrective stepping established.
Successor to the completed second Perturb batch, not another completion within
its 20 stages. Preserve sealed history and Arena selection. Passive springs
remain zero and active joint PD remains.

## Evidence and objective

Stage33 retains 120-second natural Stand and improves recovery at 5 kg / 1.6 m/s.
Offline inspection of all 28 successful native Stage40 two-hit cases found no
unloaded foot, no whole-foot clearance and less than 4 mm horizontal foot movement.
This establishes planted-foot recovery, not corrective stepping. The independent
optimization repeat did not reproduce gains on both original parents.

The objective is a lifted, translated and replanted foot followed by stable Stand.
Readiness for 10 kg / 5 m/s remains unestablished. Its nominal horizontal momentum
is 50 kg m/s and energy 125 J, versus 8 kg m/s and 6.4 J at 5 kg / 1.6 m/s. These
ratios predict neither learning time nor transferred collision impulse.

## Measurement

`corrective_step_audit.py` reads saved states and solved loads. MuJoCo kinematics
reconstruct the minimum height of both oriented Foot/Toe boxes above flat ground;
no new physics integration or policy actions. Either foot may qualify:

- Loaded at least 30 N immediately before post-contact unloading.
- Below 5 N during contiguous swing; whole-foot clearance at least 15 mm for
  50 continuous milliseconds.
- Opposite foot carries at least 30 N through swing and landing confirmation.
- Landing carries at least 30 N continuously for 100 milliseconds.
- Placement changes at least 30 mm in world coordinates and relative to the
  opposite foot, whose drift stays within 20 mm.
- A two-second window of original balance/posture gates starts within two seconds
  of touchdown, plus one control-frame tolerance, after landing confirmation.
- Existing encounter recovery and final survival must also pass.

Single-foot support during swing is allowed. Sliding, rocking, hopping, pre-hit
flight and settling across a subsequent launch earn no successful-step credit.
Report mechanical steps separately from recovered steps. These are prospective
engineering thresholds, not a validated human-naturalness score. Events shorter
than the control sampling interval cannot be resolved.

## Bounded comparison

`corrective_step_pilot.json` starts every arm from the same Stage33 checkpoint and
full AMP sidecar for each parent 301027/301028. Fresh optimizers and matched RNG.
Four arms: baseline; recovery reward; wider leg targets; both interventions.
Baseline range is 0.25 rad. Wider hip/knee/ankle/toe range is 0.5 rad, with upper
body unchanged. Keep joint/torque limits, two-frame delay, gains, plant and friction.
Identical weights with a wider range do not produce identical motor commands.
Require 12-second full-interval quiet Stand in both backends before training that
range. Failed ranges receive no training and are reported as negative preflights.

Post-contact reward reuses `TransferRecoveryReward`: upright support/capture,
state-based load transfer/swing and bilateral settling, with 0.05 imitation.
Measured contact activates it; reset clears it. Quiet/pre-contact task rewards
stay unchanged. No scripted trajectory or teacher motor takeover. AMP examples
and task/AMP weighting remain fixed; this may still constrain large deviations.
Quiet rehearsal uses weight 100 and 256 examples per existing actor optimizer
step, from the first 16 seconds of the matching Stage37 quiet trace.

Practice one 5 kg / 2 m/s Chest-directed +X projectile with launch times 1/3/5/9
cycling across resets and 16-second episodes. This is one modest dose increase,
not an unconditional ramp to the final target. Use 64 worlds x 64 rollout frames,
three actor epochs, actor LR 3e-5, 100 updates per admitted arm, 420-second training
cap. Maximum 800 updates / 3,276,800 transitions. Count rehearsal separately.

Initial baseline and final policies receive frozen 5 kg / 1.6 and 2 m/s checks,
both signs, eight starts, both backends, launch 5/end 12 seconds. Wider controller
gets zero-update quiet preflight. Maximum 832 discovery cases including quiet.
Require first-contact Chest inclusion and existing native/GPU outcome and impulse
admission. Record measured impulses, steps, recovery and quiet Stand separately.
No behavioral interpretation of failed physical admission and no automatic Arena
replacement. New runtime failures require diagnosis, not silent retries.

## Advancement

Only an admitted candidate with new recovered steps on both parents, retained
quiet Stand and no per-condition recovery-count regression against source and
baseline continuation advances to 120-second Stand (32 additional cases). Then
require independent optimization replication and held-out impact times before
stronger doses. Negative results motivate representation or demonstration changes,
not indefinite training. Current pilot cannot rule out stepping with other designs.

## Completed pilot

Eight focused checks passed: seven mechanical classification cases and the
baseline/wider-target formula, joint limits, delay and scoped restoration check.
The user explicitly authorized checks, bounded training and frozen evaluations.

Output: `logs/mimickit-training/corrective-step-pilot-20261007-01`.
Independent saved-evidence review:
`logs/mimickit-training/corrective-step-pilot-review-20261007-01/report.json`.
The separate Stage40 saved-trace audit is
`logs/mimickit-training/corrective-step-saved-audit-20261007-01/report.json`.

All 512 new frozen cases reconcile with saved reports, including both backends.
All impact comparisons pass body attribution and backend admission; every native
launch has positive character clearance (minimum 3.36 mm). Native/GPU categorical
step outcomes agree. Inputs and used sources retain their recorded hashes.

Wider targets fail the zero-update quiet gate on both parents: 3/8 and 2/8
full-interval Stand passes, respectively, identically on both backends. The range
and combined arms are therefore gated out without training. This rejects naive
range doubling with unchanged weights; it does not prove wider authority cannot
work with a behavior-preserving initialization and a different curriculum.

The four admitted runs complete 400 updates / 1,638,400 transitions total, plus
4,915,200 offline rehearsal presentations. Each run uses 100 updates from its own
source warm start. All final policies retain the 12-second quiet screen. New
120-second retention is not run because the advancement gate fails.

Native recovery counts below are reproduced by the GPU backend. Counts are out
of eight starts per parent and condition; these are not independent trained seeds.

- Parent301027: source, baseline continuation and recovery-reward continuation
  score 6/7/7 on +X 1.6 m/s, 8/8/8 on -X 1.6 m/s, 0/0/1 on +X 2 m/s, and
  6/4/3 on -X 2 m/s.
- Parent301028: source, baseline continuation and recovery-reward continuation
  score 7/8/8 on +X 1.6 m/s, 8/8/8 on -X 1.6 m/s, 0/4/1 on +X 2 m/s, and
  8/7/7 on -X 2 m/s.
- Every initial and final impact group has zero qualifying mechanical steps and
  zero recovered-step cases. All successful recoveries remain outside the new
  stepping milestone.

The recovery reward is not a consistent improvement over matched continuation.
Its apparent 0-to-1 gain against both source policies is insufficient: baseline
continuation reaches 4/8 for the second parent, and opposite-direction recovery
regresses against the sources. No promotion, strength escalation, conditional
120-second test or further optimization replication is justified by this pilot.
Arena selection is unchanged and all experiment processes have stopped.

## Next design milestone, not executed

Make weight transfer, temporary single-foot support and a short replanted step
learnable before another stronger-impact run. Preserve the source policy's initial
physical motor commands when adding leg authority; identical network weights are
not sufficient if action meanings change. Evaluate a behavior-preserving residual
or moving-reference design, with a separate quiet-Stand control and explicit step
examples only after their physical feasibility is established on the current
33-motor plant. Keep step mechanics and final settled recovery as separate gates.
The pilot does not isolate whether stationary AMP examples, rehearsal strength,
exploration or the reference-centered controller is the main remaining obstacle.

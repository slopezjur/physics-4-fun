# Direct recovery pilot - completed 2026-10-09

Decision: reject both new correction policies. The arm-motion prerequisite has
been removed from the direct-recovery plan, but this pilot did not improve recovery.
The existing Arena selection remains unchanged.

## What ran

See [DIRECT_RECOVERY_PLAN.md](DIRECT_RECOVERY_PLAN.md) for the prospective protocol.
The deployed 374-input controller stayed frozen; each new 407-input head learned
only bounded additions to its 33 motor targets. No teacher, IK, foot trajectory,
root support, passive spring, or future hit signal was added.

Two identical zero-output initial heads were trained with independent seeds
901009 and 911009. Each received 24,576 transitions; total 49,152. Accepted actor
updates were 166 and 167. Training took 27.35 and 27.01 seconds. The complete
physical collection, training and evaluation run took 286.51 seconds. Evidence
occupies about 134 MB, including compact traces and initial/final heads; there
are no intermediate model checkpoints.

## Physical outcomes

Across the same 64 deterministic push cases per controller:

- Frozen source: 35 survived and settled; heldout-pose recovery 10/16.
- Seed 901009: 30 survived and settled; heldout-pose recovery 8/16.
- Seed 911009: 30 survived and settled; heldout-pose recovery 7/16.
- All three: zero audited corrective placements and zero recovered-with-step cases.
- Each final candidate retained quiet Stand throughout all 16 sixty-second cases
  (the eight reference poses repeated). This is gate retention, not exact equality
  of the trained candidates' quiet trajectories.

Group order is 150 N X, 150 N Y, 300 N X, 300 N Y, with both signs and all eight
poses in every group. Settled counts out of 16 were 4/16/0/15 for the source,
1/16/0/13 for seed 901009 and 2/16/0/12 for seed 911009. X and Y use different
onsets, 1.0 and 1.7 seconds respectively, so direction and timing are confounded
in this first screen; a causal direction comparison must cross both timings.

With fixed sampled actions on the 150 N X group, settled counts were 3/16 for
the untrained zero-mean head, 2/16 and 3/16 for the trained heads. These limited
sampled tests also show no improvement.

Pulses lasted six 0.016668-second intervals: approximately 15.0012 or 30.0024 N s
at Chest COM. These are direct-force results, not 10 kg / 5 m/s ball admission.

## What the traces show

Mean-policy maximum additions were 0.01379 and 0.01214 radians, below the 0.1-rad
envelope. Neither envelope nor slew limiting saturated in their deterministic
evaluation traces. Only 27/539,462 and 29/540,549 nonzero addition coordinates were
altered by final joint-limit clipping. The action path is working; these traces
do not support increasing its limits as the immediate fix.

Addition RMS before versus after impact was 0.001996 versus 0.002132 rad for the
first seed and 0.001916 versus 0.001965 rad for the second. The heads mainly made
small corrections, with little change in aggregate magnitude after impact. This
does not prove their feedback is constant or identify the cause of failed learning.

One first-seed case briefly lifted an unloaded foot with opposite support for
16 frames, then failed. It produced no admitted landing/settling sequence. This
is not learned corrective stepping. The second seed produced no such lift.

## Verification

The independent audit passed: 22 saved traces, 352 evaluation cases including
parity controls and repeated quiet poses, 542 input hashes and 59 output hashes.
It checked source integrity, identical initial heads, exclusion of correction
training poses 6-7, update budgets/KL, force windows, target bounds, two-interval
delay, independently reconstructed collision-box clearance and step outcomes.

Zero correction exactly reproduced the original controller on both quiet and
pushed rollouts. Its sixty-second quiet qpos, issued targets and alive flags also
exactly matched the archived source reference. No solver warnings occurred in
the evaluation traces.

Evidence: `logs/mimickit-training/direct-recovery-20261009-01/`:
`summary.json`, `audit.json`, `archival-parity.json`, `diagnostics.json`, per-case
reports/traces, training histories and source snapshots. The later diagnostics
file records its own input hashes and is outside the completion manifest.

## Next priority

Before extending PPO, test direct recovery feasibility at the failing push
boundary: cross X/Y direction with both hit timings, then compare bounded leg
corrections on identical failing states. Measure unloading, clearance, landing
and settling, including whether the frozen source counteracts those corrections.
Those are diagnostic interventions, not learned behavior or Arena candidates.

Use demonstrated recoverable conditions to choose the next learning curriculum.
Do not restart the arm gate or infer that more training time, larger corrections,
or restored passive springs will solve the present failure. Varied/repeated ball
hits and Arena parity still follow actual learned recovery success.

# Reachable disturbed-state response admission

Predeclared 2026-10-08; successor to ARM_PENDING_EFFECT_PROGRESS.md.

## Purpose

Collect and validate paired teacher responses on physically reachable states
before choosing another learning objective. The bounded pending-effort input
passed its contract checks but regressed physical performance. Synthetic queue
probes revealed poor local teacher-response agreement; they were not admitted
as training data, and their heldout-pose queries remain diagnostic only.

## Bounded pilot

- Origins: retained masked heads, seeds 711028 and 721028; nominal and varied
  schedules. Reuse the previously sealed pulse/switch frames.
- Target training poses 0 and 4, the two training poses in the preceding matched
  feedback diagnosis. Other poses retain the original learner throughout.
  Poses 6 and 7 receive zero teacher queries and zero admitted labels.
- Five branches per origin/schedule: no pulse, left arm +/-0.005 rad, right arm
  +/-0.005 rad, each for one control frame before the existing authority/fade/
  slew projection. Measure the actual transmitted target change.
- Teacher takeover begins three control frames after pulse issue. The pulse
  waits two control intervals and acts during the third; the teacher's first
  query therefore sees the resulting native physical state. No state restore,
  direct queue write or body override occurs during these episodes.
- Twenty continuous episodes, 160 physical cases: 8 unpulsed and 32 pulsed
  targeted teacher cases; 120 unaffected learned-control cases. No fitting,
  PPO, Arena update, gain/spring/limit change or new checkpoint in this pilot.

## Required evidence

Unpulsed prefixes must reproduce archived learned traces exactly up to takeover.
Every pulsed branch must reproduce its unpulsed prefix and full pre-pulse state.
The first two post-issue physical positions/velocities must remain identical;
measure nonzero physical response during the third interval and verify both
pending slots in chronological observation order at takeover. Record native
integration state and qvel alongside issue-time observations/labels. Teacher
queries must preserve live state and control.

Reconstruct the original command projection and all physical gates independently:
supported final two-second hold, motor RMS <=0.05 rad, COM <=0.02 m, foot drift
<=0.02 m, survival and final two-second Stand return. No tolerance widening.

## Paired-data admission

For each pulsed targeted case, require both its own and the matching unpulsed
teacher case to pass the full gate, actual issued pulse magnitude >1e-5 rad,
and a nonzero native position response >1e-10 by the third interval.
At takeover offsets 0, 1, 3, 6 and 12 control frames, admit pairs only when both
branches have live teacher queries and identical goal/authority/timing. Save
both raw effective teacher targets, inputs, Stand sources and previous task
additions, with full provenance. Maximum 160 paired examples. Do not select
pairs by how easily the learner predicts their response.

Evaluate both frozen input recipes on these training-only pairs. Report absolute
target error and paired target-change error, response direction, magnitude and
variation across seed, timing, pose and offset. These are diagnostics on reachable
teacher continuations, not proof of learned external-hit recovery.

Use the admitted measurements to specify the next bounded comparison. Keep its
input representation fixed and compare response supervision against an absolute-
target control with exactly the same examples and fitting budget. Preserve the
original quiet draw stream and whole-pose holdouts. Seal that comparison before
any fitting; do not combine an input change with an unmeasured sampling change.

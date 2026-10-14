# Updated learned Perturb preview - 2026-10-08

User requested a current interactive recovery brain and authorized the bounded
training/evaluation/export plan. This supersedes the scripted-teacher 16/16 gate
as the next delivery prerequisite; its negative evidence remains intact.

Start with Stage33 varied parent 301028, not the earlier Stage12 Arena preview.
One optimization seed (381028), matched quiet and repeated-impact arms, 100 PPO
updates each, 64 environments x 64 steps, three actor epochs, actor rate 3e-5.
Maximum 200 updates / 819,200 transitions, 420 training seconds per arm. Preserve
reward, physical PD, motor limits, zero passive springs and two-frame delay.
Both arms rehearse the parent's first sixteen seconds of saved quiet actions
with weight 100 and 256 examples per actor optimizer call. No teacher takeover.

Impact arm: 5 kg at 1.6 m/s, +X Chest aim from 0.25 m. First onsets cycle
1/3/5/9 seconds across resets; second shot four seconds later; episode 20 seconds.
Reuse one physical projectile, explicitly distinct from Arena's persistent pool.
Quiet arm uses the same resets/optimizer budget without shots. Save only initial
and final learner states, compact per-update training evidence, no intermediate
checkpoint copies. Verify every actual launch against its declared schedule.

Frozen comparison: Stage12 Arena, Stage33 parent, new quiet and new impact
endpoints. Eight original poses, native and GPU, single hits at 5 seconds with
5 kg at 1.4/1.6 m/s in both X directions (12-second trials), plus two 1.6 m/s
+X hits at 5/9 seconds (16-second trials). 320 impact cases. All four actors get
30-second quiet screens (64 cases). Selected preview gets 120-second quiet
retention on both backends (16 cases). Maximum 400 frozen cases. Retain original
recovery criteria and report body contacts, insertion and backend agreement.

Select only candidates passing quiet retention and evaluation admission. Prefer
the new impact endpoint if it retains every primary recovery count relative to
Stage33 and improves repeated-hit recovery; otherwise retain Stage33 as the newer
experimental preview. Quiet control diagnoses training effects, not promotion.
Do not claim replicated recipe improvement from one optimization seed. Arena's
10 kg / 4 m/s random-target settings exceed these controlled tests; label the
preview accordingly, with automatic throws off. Export ONNX, verify action
agreement, update reversible selection/scene paths, and launch existing Godot
build. No C# changes or compilation planned.

# Private queue counterfactuals at the matched switch states

2026-10-08. Follow-up input diagnosis within the frozen matched-state block.

After the branch audit passes, inspect its 24 focus switch states (two origins,
two seeds, two timings, three poses). Load saved integration states into a
separate scratch simulator for read-only teacher predictions. This is not a
continuation rollout or an assisted recovery. Require the unmodified teacher
query to reproduce the saved executed teacher target within 2e-6 rad.

For each state, make four private pending-queue variants: +/-0.005 rad in the
oldest UpperArm_L_ry or UpperArm_R_ry target, preserving body state, velocity,
goal, authority and previous addition. Keep target bounds unchanged. Reflect
that queue change in the Stand observation, and recompute its source-target
change relative to the exact archived source. Evaluate both task heads on that
same changed observation and the teacher on a proxy control with only its
private queue changed. The original task and scratch integration state must be
unchanged by all queries. No live branch state or Arena state is modified.

Save baseline and counterfactual raw/effective targets and projected commands.
The bounded total is 120 teacher queries. A teacher difference with identical
masked task features demonstrates target dependence on withheld information;
it does not prove that this alone explains failures or that restoring raw
queue inputs will work. No labels are admitted for fitting, no training, and
no further physical evaluation episodes are added.

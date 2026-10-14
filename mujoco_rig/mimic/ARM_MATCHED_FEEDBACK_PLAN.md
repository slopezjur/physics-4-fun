# Frozen matched-prefix feedback investigation

2026-10-08. Continues ARM_SAMPLING_BALANCE_PROGRESS.md.

Freeze each pose's switch frame at the first >0.001-rad issued-command
difference recorded for the 5% head versus replay in the previous diagnostics.
Use both fitting seeds and nominal/varied timing. For each origin controller
(replay or 5%), execute three continuous episodes: continue the origin, switch
to the other frozen head, or switch poses 0, 4 and 6 to the physical teacher.
Other poses keep their origin controller in teacher branches. This gives 24
eight-pose episodes. Teacher queries on heldout pose 6 are diagnostic only;
export no new fitting labels and perform no fitting or model selection.

Every branch reaches its switch by deterministic native replay from reset,
never by overwriting physical state or clearing pending targets. Require exact
archived control trajectories, exact per-world prefixes and matching complete
issue-time state fingerprints, including integration state, command queue,
controller history, task state, observation and physical gate history.
Fingerprint live state before and after teacher queries. Keep the original
full hold/return interval, torque, target, slew, delay and physical gates.

At each control branch's switch for poses 0, 4 and 6, probe both frozen heads'
instantaneous response to input-only +/-0.001-rad UpperArm_L_ry/UpperArm_R_ry
angle errors and +/-0.01-rad/s velocity errors. Also probe +/-0.005-rad changes
in the oldest queued target input. Record raw task output, frozen anchor target
and the final projected target separately. These probes do not modify physical
state and are not a claim about dynamic stability or teacher information needs.

For seed 711028 at nominal timing, additionally run eight pulse episodes from
replay prefixes: each post-switch learned head, each of the two upper-arm axes,
and each +/- sign. Apply a 0.005-rad requested correction for one control frame
through the existing feasible-correction projection, only in poses 0, 4 and 6
at their own switch. No direct state perturbation. Record actual transmission,
then compare position/velocity/COM differences over the following 3, 12, 30
and 60 frames and the complete original horizon. A blocked pulse is a blocked
probe, not evidence of insensitivity. This gives 32 episodes / 256 cases total.

Use an independent saved-evidence audit for prefix/state identity, raw-target
projection, perturbation timing, teacher scope and full physical gates.
Distinguish diagnostic switches/pulses/teacher recovery from learned deployment.
No PPO, supervised updates, quiet re-evaluation, C# build or Arena changes.

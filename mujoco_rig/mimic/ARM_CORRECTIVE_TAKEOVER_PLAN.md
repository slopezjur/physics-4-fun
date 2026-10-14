# Corrective teacher admission after fresh proposal heads

Date: 2026-10-08. Continues ARM_PROPOSAL_30_PROGRESS.md.

## Bounded protocol

Replay the retained complete-data heads (711028 and 721028), nominal and varied
timing, against their archived trajectories. For training poses 0-5 that failed
the full skill, locate the first post-onset motor (>0.05 rad), COM (>0.02 m),
support, foot-drift (>0.02 m), or native survival violation. During the ramp,
motor and COM diagnostics use the current moving goal. Switch 0.5 seconds before
that violation, clipped to task onset. If no such violation exists, use onset.
Freeze these switch frames before executing teacher continuations.

Keep successful training cases and holdouts 6/7 under the original learned head.
They supply no corrective teacher labels. Replay controls must exactly match
archived commands, states and survival. Each takeover must match its control
prefix and issue-time integration state, queued commands, previous addition and
observation. The teacher uses private predictions; fingerprint the live state
before and after every batch of queries. Never restore state, clear queues,
change the goal, reset the clock or relax torque/target/slew/physical gates.

Use IntervalTargetSupport(interval_com) through the existing goal-conditioned
residual interface and delayed adapter. Record the teacher raw target, actual
residual and effective reconstructed raw target. Round-trip floating-point
differences must remain below 2e-7 rad. Label the effective raw target before
authority, fade, slew and final joint clipping, not the issued command.

Admit only live teacher-controlled samples from training cases that complete the
original entire final-two-second hold and final-two-second Stand return. Save
rejected trajectories separately as evidence. If a case fails this takeover,
allow exactly one predeclared earlier fallback at task onset, unless already
tested at onset. No adaptive timing search beyond that fallback.

Independently reconstruct admission gates and the command projection from saved
traces. Verify dataset provenance and exclude all heldout labels. Preserve the
original complete teacher dataset, including its quiet examples, for subsequent
matched corrective-data versus replay fitting. This block performs no fitting,
PPO, C# build, or Arena promotion. Teacher-assisted success is not learned skill.

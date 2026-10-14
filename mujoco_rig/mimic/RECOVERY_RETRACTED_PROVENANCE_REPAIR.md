# Historical-source accounting repair before retracted learning

The first runner stopped in preflight before creating inputs.json, collecting
trajectories or starting either fit. Its sources remain preserved under
recovery-retracted-learning-20261009-01/sources.

The new runner unnecessarily required all shared source files to match the
oldest zero-control pilot, in addition to both immediate controls. One
diagnostic file had changed before those immediate controls were sealed:
probe_recovery_exploration.py. The original pilot archived a different GPU
exploration assay under that name. Its old hash is
d3bd741568548f16a059a5f2b236161715c71f74c74563a5c48ea76370dd50d5;
the current hash is
2d0441a09deeaaabcf5cb9096cc01a2be1c2a3234fa91f359a8b5e46adec1a8e.

All shared source hashes match the projected-only and weighted controls.
Every consumed pilot output already passed its sealed hash check. The repair
records this exact historical source difference, validates the original
archived source against the old hash, and requires both modern controls to
agree with the current source. Every other shared input must match exactly.
It does not relax output, model, physics, current-source or numerical checks.

Successor runner/audit/analysis wrappers use evidence folder
recovery-retracted-learning-20261009-02. The learner, physical protocol and
budget remain unchanged. No earlier sealed source or evidence is edited, and
no physical experiment is repeated because none had begun.

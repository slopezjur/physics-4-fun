# Experiment storage retention

Cleanup performed on 2026-10-07 at the user's request. The project contained
149.8 GiB (about 161 GB) of logical file data: 103 GiB in root logs, 32.1 GiB in
Isaac Lab 3 logs, and 4.3 GiB in Isaac Lab logs. Mimic training alone contained
73.4 GiB of compressed numerical traces and 15.3 GiB of checkpoints.

Removed 8,004 unreferenced intermediate checkpoints from the retired Isaac,
MuJoCo RL and Godot RL experiments, reclaiming 36.499 GiB. Every run retains its
first and latest two numbered checkpoints, every explicitly named checkpoint
found in the reviewed documentation/configuration, and all named best/final
policies and exports. This intentionally retires intermediate resume points;
historical logs may still mention iterations whose checkpoint was pruned.

Consolidated 166 byte-identical archived NPZ copies into NTFS hard links,
reclaiming another 2.959 GiB. All original paths and SHA-256 contents remain.
These archived files are immutable: if regeneration is ever needed, replace a
file atomically rather than editing it in place, because identical copies share
storage. No Mimic trace, report, checkpoint, input, source snapshot, Arena bundle
or selection was removed. Hash-based verification continues to read the same
bytes. Hard links reduce physical disk use, not Explorer's sum of logical sizes.

The logical project size after cleanup was 113.290 GiB, and drive free space
increased from 3.789 to 43.248 GiB. A trial of additional disk compression saved
only 0.7 percent on one already-compressed trace; it was not applied broadly.
Automatic approval review rejected recursive removal of the unused legacy
`rl/.venv` environment with "blocked by policy"; that directory remains intact.
The current Mimic environment and the open Godot editor were left intact.

The exact inventories and completed operations are recorded under
`logs/storage-cleanup-20261007/`: `legacy-checkpoints.json`, `deleted.json`,
`duplicate-traces.json`, `consolidated.json`, and `summary.json`.

For new experiments, keep final/selected brains, their restoration inputs,
protocols, source snapshots and scientific summaries. Save compact traces with
the physical measurements needed for the prospective question. Avoid recording
every large observation tensor and every controller substep unless the stage
requires them. Do not delete referenced evidence solely because it is old or a
negative result. Review intermediate checkpoint retention after a batch ends;
these cleanup tools do not schedule or perform automatic future deletion.

"""Aligned loading of Godot and Isaac traces.

**One shared place for the one-policy-step offset, because doing it per script has failed three
times.** `IsaacPolicyDriver.DofTracePath` writes the observation computed at the START of a policy
step alongside body state read during it, so the Godot trace's rows sit one policy step behind the
Isaac trace produced from the same actions. Verified by cross-correlating the `act*` columns:
`godot[n+1] == isaac[n]` at mean |difference| of exactly 0.000000, while shift 0 gives 1.95.

Consequences of forgetting it, all real and all on record:

* `probe_divergence.py` reported `joint_Shin_R:0` diverging at 0.02 s and concluded "distal-led, look
  at contact". That was the offset. Aligned, the ARMS lead by 3x and nothing diverges at contact.
* A frequency-response comparison charged 16.7 ms of logging offset to the plant, turning a ~15 ms
  first-order actuator lag into a reported "32.1 ms pure transport delay" - and a phase-lead
  compensator was then tuned against it.
* A step test read 1 policy step of dead time in Godot against 0 in Isaac. True dead time is 0 in
  both.

Use `load_pair` for any Godot-vs-Isaac comparison. Do not hand-roll the shift again.
"""

from __future__ import annotations

import csv
import pathlib

# Godot's trace lags the Isaac trace produced from the same action sequence by this many rows.
GODOT_TRACE_LAG_STEPS = 1

# Godot writes `pos_Thigh_L.x`; Isaac writes `pos_joint_Thigh_L:0`. Same quantity, different naming.
_AXIS = {"x": "0", "y": "1", "z": "2"}


def godot_to_isaac_column(name: str) -> str | None:
    """Map a Godot trace column onto its Isaac equivalent, or None when there is no counterpart."""
    for prefix in ("pos_", "vel_"):
        if name.startswith(prefix) and "." in name:
            bone, _, axis = name[len(prefix):].rpartition(".")
            if axis in _AXIS:
                return f"{prefix}joint_{bone}:{_AXIS[axis]}"
    return name


def load_pair(godot_csv, isaac_csv, lag: int = GODOT_TRACE_LAG_STEPS):
    """Return (godot_rows, isaac_rows) trimmed so index n refers to the same policy step in both."""
    g = list(csv.DictReader(open(pathlib.Path(godot_csv), encoding="utf-8")))
    i = list(csv.DictReader(open(pathlib.Path(isaac_csv), encoding="utf-8")))
    g = g[lag:]
    n = min(len(g), len(i))
    return g[:n], i[:n]


def verify_alignment(godot_rows, isaac_rows, n_actions: int = 36, samples: int = 200) -> float:
    """Mean |action difference| after alignment. Near zero confirms the shift; call it before trusting
    any comparison, because a changed decimation or trace format silently breaks the assumption."""
    n = min(samples, len(godot_rows), len(isaac_rows))
    if n == 0:
        return float("inf")
    total = 0.0
    for k in range(n):
        for a in range(n_actions):
            key = f"act{a}"
            if key in godot_rows[k] and key in isaac_rows[k]:
                total += abs(float(godot_rows[k][key]) - float(isaac_rows[k][key]))
    return total / (n * n_actions)

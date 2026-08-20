"""Summarise the get-up phase of the most recent telemetry dumps.

Reads the CSVs written by Source/Ragdoll/Diagnostics/RagdollTelemetryRecorder.cs and prints how
chest height, chest pitch and arm pitch move across each Recovering segment - start, middle, end.

Run it from anywhere; Debug/ is resolved relative to this file, not the working directory. The
dumps themselves are gitignored, so a fresh clone has nothing to analyse until you record some.
"""

import csv
import os

# The columns this reads. Named explicitly so a rename in the recorder fails loudly here with the
# offending column, instead of the silent no-op this script used to do: its whole body sat inside
# `except Exception: return`, so any format drift made it print nothing and exit 0. Two sibling
# scripts died of exactly that - one lost its knee traces when KneeAngleL became KneeAngleL_Deg,
# the other indexed [-1] on the miss and reported an unrelated column as if it were the knee.
REQUIRED_COLUMNS = ("Time", "RecoveryProgress", "PelvisPosY", "Chest_PosY", "Chest_EulerX", "UpperArm_R_EulerX")

DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Debug")


def read_recovery(path):
    """Rows of the Recovering segment, as floats. Raises if a column the summary needs is gone."""
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)

        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"{os.path.basename(path)} is missing column(s): {', '.join(missing)}")

        for row in reader:
            if row["State"] == "Recovering":
                rows.append({
                    "Time": float(row["Time"]),
                    "Progress": float(row["RecoveryProgress"]),
                    "Height": float(row["PelvisPosY"]),
                    "ChestHeight": float(row["Chest_PosY"]),
                    "ChestPitch": float(row["Chest_EulerX"]),
                    "ArmPitch": float(row["UpperArm_R_EulerX"]),
                })
    return rows


def summarise(folder):
    path = os.path.join(DEBUG_DIR, folder, f"{folder}.csv")
    if not os.path.exists(path):
        print(f"--- {folder} --- no CSV at {path}")
        return

    rows = read_recovery(path)
    if not rows:
        # Say what the file DOES contain rather than just "nothing here". This summary only covers
        # the Recovering segment, so an RL or balance dump - where State reads
        # ReinforcementLearning throughout - legitimately has no rows, and reporting that as a bare
        # "no Recovering rows" reads like a broken recording.
        states = {}
        with open(path, "r", newline="") as handle:
            for row in csv.DictReader(handle):
                states[row["State"]] = states.get(row["State"], 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(states.items(), key=lambda kv: -kv[1]))
        print(f"--- {folder} --- no Recovering rows; states present: {summary or 'none'}")
        return

    start, mid, end = rows[0], rows[len(rows) // 2], rows[-1]
    print(f"--- {folder} --- {len(rows)} rows over {end['Time'] - start['Time']:.2f}s")
    for label, key in (("Chest Height", "ChestHeight"), ("Chest Pitch", "ChestPitch"), ("Arm Pitch", "ArmPitch")):
        print(f"{label}: Start {start[key]:.2f} -> Mid {mid[key]:.2f} -> End {end[key]:.2f}")


def main():
    if not os.path.isdir(DEBUG_DIR):
        print(f"No Debug/ directory at {DEBUG_DIR} - record a telemetry dump first.")
        return

    folders = sorted(
        (f for f in os.listdir(DEBUG_DIR) if f.startswith("telemetry_dump_")),
        reverse=True,
    )[:3]

    if not folders:
        print(f"No telemetry_dump_* folders under {DEBUG_DIR}.")
        return

    for folder in folders:
        summarise(folder)


if __name__ == "__main__":
    main()

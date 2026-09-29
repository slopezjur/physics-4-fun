"""Cross-language balance-input parity over physical recovery states and load boundaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .capture_reward import capture_distance
from .rig import Rig


def run(args):
    root = Path(__file__).resolve().parents[2]
    rig = Rig.load(root / "mujoco_rig/dummy.xml")
    data = mujoco.MjData(rig.model)
    feet = [rig.model.body(name).id for name in ("Foot_L", "Foot_R")]
    cases, expected = [], []
    with np.load(args.states) as bank:
        for i in np.linspace(0, len(bank["qpos"])-1, 32).astype(int):
            data.qpos[:], data.qvel[:] = bank["qpos"][i], bank["qvel"][i]
            mujoco.mj_forward(rig.model, data)
            mujoco.mj_subtreeVel(rig.model, data)
            for loads in ([100., 100.], [0., 5.], [5., 0.], [4.999, 0.]):
                error = capture_distance(*[torch.tensor(value, dtype=torch.float32)[None] for value in
                    (data.subtree_com[1], data.subtree_linvel[1], data.xpos[feet], loads)])
                expected.append(float(error[0]))
                cases.append(dict(qpos=data.qpos.tolist(), qvel=data.qvel.tolist(), foot_loads_n=loads))
    args.out.mkdir(parents=True, exist_ok=False)
    fixture = args.out / "fixtures.json"
    fixture.write_text(json.dumps(dict(schema="mimic_com_parity_fixtures_v1", cases=cases)))
    output = args.out / "godot.json"
    env = os.environ.copy()
    env.update(P4F_MIMIC_COM_PARITY_INPUT=str(fixture.resolve()), P4F_MIMIC_REPLAY_OUTPUT=str(output.resolve()),
               P4F_MUJOCO_LIBRARY=str(Path(mujoco.__file__).parent))
    with (args.out / "godot.log").open("w") as log:
        subprocess.run([str(args.godot), "--headless", "--path", str(root),
                        "res://Scenes/RL/Isaac3/MuJoCo/MimicStandReplay.tscn"], env=env,
                       stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    actual = json.loads(output.read_text())["cases"]
    if len(actual) != len(cases):
        raise ValueError("Incomplete Godot parity response")
    errors = np.abs(np.array([case["loaded_capture_error_m"] for case in actual]) - expected)
    report = dict(cases=len(cases), max_error_m=float(errors.max()), passed=bool(errors.max() < 2e-6),
                  states_sha256=sha256(args.states), model_sha256=sha256(rig.path),
                  scope="Reset-state balance input parity, including single support and the 5 N threshold")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("states", "godot", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)

"""Compare observation-derived COM against native MuJoCo and the Godot bridge."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .observation_com import ObservationCom
from .rig import Rig
from .runtime import activate


def observation_from_state(data, num_obs):
    rotation = data.xmat[1].reshape(3, 3)
    observation = np.zeros((1, num_obs), dtype=np.float32)
    observation[0, :3] = data.qpos[:3]
    observation[0, 3:6] = rotation[:, 0]
    observation[0, 6:9] = rotation[:, 2]
    observation[0, 9:12] = data.qvel[:3]
    observation[0, 12:15] = rotation @ data.qvel[3:6]
    observation[0, 15:54] = data.qpos[7:]
    observation[0, 54:93] = data.qvel[6:] * .1
    return observation


def run(args):
    activate(args.mimickit)
    from .support_transition import SupportTransitionReference

    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path(__file__).resolve().parents[2]
    rig = Rig.load(root / "mujoco_rig/dummy.xml")
    source = json.loads(args.source_contract.read_text())
    observer = ObservationCom(rig, source)
    reference = SupportTransitionReference(args.reference, rig, "cpu", mode="transition")
    seconds = (0., .5, .75, 1., 1.25, 1.5, 3., 6.)
    data = mujoco.MjData(rig.model)
    fixtures, expected = [], []
    for t in seconds:
        q, v = reference.sample(torch.tensor(t))
        q, v = q.numpy().astype(float), v.numpy().astype(float)
        for shift in (0., -.04, .04):
            shifted = q.copy()
            shifted[0] += shift
            data.qpos[:], data.qvel[:] = shifted, v
            mujoco.mj_forward(rig.model, data)
            mujoco.mj_subtreeVel(rig.model, data)
            native = np.array([data.subtree_com[1, 0], data.subtree_linvel[1, 0]])
            observed = observer(observation_from_state(data, source["num_obs"]))[0]
            fixtures.append(dict(seconds=t, root_shift_m=shift,
                                 qpos=shifted.tolist(), qvel=v.tolist()))
            expected.append(dict(native=native, observed=observed))
    args.out.mkdir(parents=True, exist_ok=False)
    fixture_path = args.out / "fixtures.json"
    fixture_path.write_text(json.dumps(dict(schema="mimic_com_parity_fixtures_v1",
                                            cases=fixtures), indent=2), encoding="utf-8")
    output = args.out / "godot.json"
    environment = os.environ.copy()
    environment.update(P4F_MIMIC_COM_PARITY_INPUT=str(fixture_path.resolve()),
                       P4F_MIMIC_REPLAY_OUTPUT=str(output.resolve()),
                       P4F_MUJOCO_LIBRARY=str(Path(mujoco.__file__).parent))
    with (args.out / "godot.log").open("w") as log:
        subprocess.run([str(args.godot), "--headless", "--path", str(root),
                        "res://Scenes/RL/Isaac3/MuJoCo/MimicStandReplay.tscn"],
                       env=environment, stdout=log, stderr=subprocess.STDOUT,
                       timeout=120, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    actual = json.loads(output.read_text())
    if actual.get("schema") != "mimic_godot_com_parity_v1" or len(actual["cases"]) != len(fixtures):
        raise ValueError("Godot COM parity output does not match fixtures")
    rows = []
    for fixture, item, godot in zip(fixtures, expected, actual["cases"]):
        position = np.asarray(godot["com_godot_m"])
        velocity = np.asarray(godot["velocity_godot_m_s"])
        godot_x = np.array([-position[2], -velocity[2]])
        rows.append(dict(seconds=fixture["seconds"], root_shift_m=fixture["root_shift_m"],
                         native_com_x_m=float(item["native"][0]),
                         native_com_vx_m_s=float(item["native"][1]),
                         observation_position_error_m=float(item["observed"][0] - item["native"][0]),
                         observation_velocity_error_m_s=float(item["observed"][1] - item["native"][1]),
                         godot_position_error_m=float(godot_x[0] - item["native"][0]),
                         godot_velocity_error_m_s=float(godot_x[1] - item["native"][1])))
    errors = {key: max(abs(row[key]) for row in rows) for key in (
        "observation_position_error_m", "observation_velocity_error_m_s",
        "godot_position_error_m", "godot_velocity_error_m_s")}
    report = dict(schema="mimic_com_cross_engine_parity_v1",
                  model_sha256=sha256(rig.path), source_contract_sha256=sha256(args.source_contract),
                  observer_contract=observer.contract(), cases=len(rows), errors=errors,
                  passed=all(value < 1e-5 for value in errors.values()), rows=rows,
                  scope="Reset-state COM parity only; no Godot controller or policy promotion")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(cases=len(rows), errors=errors, passed=report["passed"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args())["passed"] else 1)

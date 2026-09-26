"""Measure native COM response to small paired pitch-target residuals."""
import argparse
import json
from pathlib import Path

import mujoco
import torch

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference


PROBES = (("zero", (), 0.),
          ("ankle_positive", ("Foot_L_rx", "Foot_R_rx"), .25),
          ("ankle_negative", ("Foot_L_rx", "Foot_R_rx"), -.25),
          ("hip_positive", ("Thigh_L_rx", "Thigh_R_rx"), .25),
          ("hip_negative", ("Thigh_L_rx", "Thigh_R_rx"), -.25))


def run(args):
    activate(args.mimickit)
    from envs.base_env import EnvMode
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    names = [rig.model.actuator(int(a)).name for a in rig.actuators]
    cases = []
    for label, joints, amount in PROBES:
        task = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode="support_v2",
                         reference_factory=lambda p, r, d: SupportTransitionReference(
                             p, r, d, mode="transition"))
        task.episode_seconds = 6.
        task.set_mode(EnvMode.TEST)
        task.reset()
        records = []
        for step in range(round(args.seconds / task.dt)):
            action = torch.zeros((1, len(names)))
            if step * task.dt < args.pulse_seconds:
                for joint in joints:
                    action[0, names.index(joint)] = amount
            task.step(action)
            if step in (round(.25 / task.dt) - 1, round(.5 / task.dt) - 1,
                        round(1. / task.dt) - 1):
                data = task.engine.datas[0]
                mujoco.mj_subtreeVel(rig.model, data)
                records.append(dict(seconds=(step + 1) * task.dt,
                                    com_x_m=float(data.subtree_com[1, 0]),
                                    com_vx_m_s=float(data.subtree_linvel[1, 0]),
                                    root_x_m=float(data.qpos[0]),
                                    foot_x_m=[float(data.xpos[rig.model.body(name).id, 0])
                                              for name in ("Foot_L", "Foot_R")]))
        cases.append(dict(label=label, joints=joints, action=amount, samples=records))
    report = dict(schema="mimic_com_authority_probe_v1", model_sha256=sha256(rig.path),
                  reference_sha256=task.reference.sha256,
                  pulse_seconds=args.pulse_seconds, cases=cases,
                  scope="Zero-residual native target-PD interventions from the same reset; no learned policy or viewer change")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(cases), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--pulse-seconds", type=float, default=.5)
    run(parser.parse_args())

"""Compare real foot/toe contact fixtures on native MuJoCo and the GPU adapter."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .foundation_audit import sensor_poses
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate
from .target_control import DelayedTargetControl


class ContactTimingProbe(NativeDummyEngine):
    def _after_physics_step(self, world, data):
        # mj_step integrates q/qvel but its solved contact forces still describe
        # that last physics solve, before NativeDummyEngine's final mj_forward.
        self.last_solve_forces[world] = self._read_ground_forces(data)


def audit(mimickit, out, contact_mode="legacy"):
    activate(mimickit)
    from .newton_adapter import DummyNewtonEngine
    from .stand_task import StandTask
    torch.set_num_threads(1)
    rig=Rig.load(Path(__file__).resolve().parents[1]/"dummy.xml")
    native=ContactTimingProbe(rig,3,"cpu",DelayedTargetControl)
    native.last_solve_forces=np.zeros((3,rig.model.nbody-1,3))
    gpu=DummyNewtonEngine(rig,3,"cuda:0",DelayedTargetControl)
    reference=Path(__file__).parent/"assets/stand_reference.npz"
    tasks=[StandTask(rig,reference,3,device,engine_factory=lambda *args, current=engine, **kwargs:current,
                     control_mode="target_pd", contact_mode=contact_mode) for engine,device in ((native,"cpu"),(gpu,"cuda:0"))]
    layout=tasks[0].observation_contract()["observation_layout"]
    load_offset=layout[6]["offset"]
    q=sensor_poses(rig);v=np.zeros((3,rig.model.nv),dtype=np.float32)
    for engine in (native,gpu):engine.reset_envs(torch.arange(3),q,v)
    rows=[]
    for step in range(13):
        expected=native.get_ground_contact_forces(0).numpy()
        actual=gpu.get_ground_contact_forces(0).cpu().numpy()
        rows.append(dict(step=step,max_body_force_error_n=float(np.abs(actual-expected).max()),
                         last_solve_force_error_n=float(np.abs(actual-native.last_solve_forces).max()) if step else None,
                         max_qpos_error=float(np.abs(native.native_qpos().numpy()-gpu.native_qpos().cpu().numpy()).max()),
                         native_body_forces_n=expected.tolist(),gpu_body_forces_n=actual.tolist(),
                         actual_actor_load_inputs_n={name:(task.observations()[:,load_offset:load_offset+2]*1000).cpu().tolist()
                                                     for name,task in zip(("native","gpu"),tasks)},
                         native_last_solve_forces_n=native.last_solve_forces.tolist() if step else None))
        for engine,device in ((native,"cpu"),(gpu,"cuda:0")):
            engine.policy_control.set_reference(torch.tensor(q,dtype=torch.float32,device=device),
                                                 torch.tensor(v,device=device))
            # A short upward COM force unloads the toe fixture after quiet support.
            # This tests contact disappearance without teleporting or resetting it.
            force = np.zeros((3, 3), dtype=np.float32)
            if step >= 4:
                force[2, 2] = 2000
            engine.set_external_force(rig.model.body("Pelvis").id, force)
            engine.set_cmd(0,torch.zeros((3,30),device=device));engine.step()
    checks = dict(body_forces=max(row["max_body_force_error_n"] for row in rows) < .1,
                  qpos=max(row["max_qpos_error"] for row in rows) < 1e-4,
                  actor_loads=bool(max(np.abs(np.array(row["actual_actor_load_inputs_n"]["native"])-
                                        np.array(row["actual_actor_load_inputs_n"]["gpu"])).max() for row in rows) < .1))
    support = [rig.model.body(name).id - 1 for name in ("Foot_L", "Toe_L", "Foot_R", "Toe_R")]
    toe_initial = np.abs(np.array(rows[0]["native_body_forces_n"])[2, support, 2]).sum()
    toe_final = np.abs(np.array(rows[-1]["native_body_forces_n"])[2, support, 2]).sum()
    checks["toe_off_transition"] = bool(toe_initial > 10 and toe_final < 1e-4)
    report=dict(schema="mimic_foundation_sensor_parity_v1",model_sha256=sha256(rig.path),
                contact_mode=contact_mode, checks=checks, passed=all(checks.values()),
                body_names=[rig.model.body(i).name for i in range(1,rig.model.nbody)],rows=rows,
                scope="Three pitched floor-contact fixtures, reset, four quiet control intervals and eight upward-force intervals to unload the toe fixture. A local contact-force comparison, not long-horizon parity.")
    with out.open("x",encoding="utf-8") as file:json.dump(report,file,indent=2)
    print(json.dumps(dict(max_body_force_error_n=max(row["max_body_force_error_n"] for row in rows))))
    return report["passed"]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--contact-mode",choices=("legacy", "support_v2"),default="legacy")
    args=parser.parse_args()
    passed = audit(args.mimickit,args.out,args.contact_mode)
    raise SystemExit(1 if args.contact_mode == "support_v2" and not passed else 0)

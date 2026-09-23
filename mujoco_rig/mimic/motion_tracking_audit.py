"""Substep evidence for motion failures; no changes to training or deployment control."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate


class TraceEngine(NativeDummyEngine):
    def __init__(self, *args, prefill=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill = prefill
        self.records = [[] for _ in self.datas]
        self.soles = [self.rig.model.body(n).id - 1 for n in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]

    def reset_envs(self, ids, qpos, qvel):
        super().reset_envs(ids, qpos, qvel)
        if self.prefill:
            control = self.policy_control
            packet = torch.cat((qpos[:, control.q_indices], qvel[:, control.v_indices],
                                torch.ones((len(ids), 1))), dim=1)
            control.pending[:, ids] = packet
            control.active[ids] = packet

    def _after_physics_step(self, world, data):
        model = self.rig.model
        collision = 0.
        pair = np.array([-1, -1])
        wrench = np.zeros(6)
        for k, contact in enumerate(data.contact[:data.ncon]):
            if 0 not in contact.geom:
                mujoco.mj_contactForce(model, data, k, wrench)
                force = float(np.linalg.norm(wrench[:3]))
                if force > collision:
                    collision, pair = force, contact.geom.copy()
        self.records[world].append(dict(
            qpos_after=data.qpos.copy(), qvel_after=data.qvel.copy(),
            qacc_solve=data.qacc.copy(), control=data.ctrl[self.rig.actuators].copy(),
            ground_force_solve=self._read_ground_forces(data)[self.soles].copy(),
            self_contact_max_force_n=collision, self_contact_pair=pair))


def first_time(mask, dt):
    indices = np.flatnonzero(mask)
    return float((indices[0] + 1) * dt) if len(indices) else None


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    torch.set_num_threads(1)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads((args.bundle / "contract.json").read_text())
    if sha256(args.bundle / "stand.onnx") != contract["actor_sha256"]:
        raise ValueError("Actor hash mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for mode, prefill in (("zero_residual", False), ("zero_residual_prefilled_diagnostic", True),
                          ("standing_actor", False), ("zero_residual_no_self_collision_diagnostic", False)):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        def factory(*a, **kw):
            return TraceEngine(*a, prefill=prefill, **kw)
        task = StandTask(rig, args.reference, 8, "cpu", engine_factory=factory,
                         control_mode="target_pd", contact_mode=mode_from_contract(contract))
        validate_reference_substitution(contract, task.observation_contract())
        if mode == "zero_residual_no_self_collision_diagnostic":
            # This private model is discarded after this intervention. Floor collisions remain.
            rig.model.geom_contype[:] = 2
            rig.model.geom_conaffinity[:] = 1
            rig.model.geom_contype[0], rig.model.geom_conaffinity[0] = 1, 2
        actor = ((lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0]))
                 if mode == "standing_actor" else lambda obs: torch.zeros((len(obs), 30)))
        metrics, _ = evaluate(task, actor)
        cases = []
        for case, seconds in enumerate(metrics["survival_seconds"]):
            count = round(seconds / task.dt) * rig.decimation
            records = task.engine.records[case][:count]
            arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
            dt = rig.model.opt.timestep
            phase = case / 7 * (task.reference.duration - task.episode_seconds)
            q, v = task.reference.sample(torch.tensor(phase + np.arange(1, count + 1) * dt, dtype=torch.float32))
            error = np.linalg.norm(arrays["qpos_after"][:, :3] - q.numpy()[:, :3], axis=1)
            arrays.update(reference_qpos=q.numpy(), reference_qvel=v.numpy(), root_error_m=error)
            np.savez_compressed(args.out / f"{mode}-{case}.npz", **arrays)
            force = arrays["ground_force_solve"][:, :, 2]
            normalized = arrays["control"] / rig.action_scale
            saturation = (np.abs(normalized) >= .999).mean(axis=0)
            collision_frames = np.flatnonzero(arrays["self_contact_max_force_n"] > 5)
            first_pair = ([rig.model.geom(int(g)).name for g in arrays["self_contact_pair"][collision_frames[0]]]
                          if len(collision_frames) else None)
            cases.append(dict(case=case, phase_seconds=phase, survival_seconds=seconds,
                              first_root_error_5cm_seconds=first_time(error > .05, dt),
                              first_self_contact_over_5N_seconds=first_time(arrays["self_contact_max_force_n"] > 5, dt),
                              first_self_contact_pair=first_pair,
                              first_both_feet_unloaded_seconds=first_time(force.sum(axis=1) < 5, dt),
                              saturated_motor_sample_fraction=float((np.abs(normalized) >= .999).mean()),
                              most_saturated_motors=[dict(name=rig.model.actuator(int(rig.actuators[i])).name,
                                                          fraction=float(saturation[i]))
                                                     for i in np.argsort(saturation)[-3:][::-1]],
                              max_self_contact_force_n=float(arrays["self_contact_max_force_n"].max())))
        results.append(dict(mode=mode, metrics=metrics, cases=cases))
        print(json.dumps(results[-1]), flush=True)
    report = dict(schema="mimic_motion_tracking_audit_v1", reference_sha256=task.reference.sha256,
                  actor_sha256=contract["actor_sha256"], results=results,
                  sampling="qpos/qvel after integration; forces, acceleration and controls from the preceding physics solve. First episode only, truncated at failure.",
                  scope="Temporal diagnostics with isolated reset/contact interventions, not production controller changes. No-self-collision uses a private model and retains floor collisions.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())

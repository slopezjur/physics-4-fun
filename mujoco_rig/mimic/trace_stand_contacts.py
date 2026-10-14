"""Substep constraint tracing and matched-state collision probes; no learning."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import traceback

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


def constraint_counts(types):
    return dict(Counter(mujoco.mjtConstraint(int(value)).name for value in types))


def contact_record(model, geom, dist, pos, address):
    return dict(geom_ids=[int(i) for i in geom],
        bodies=[model.body(int(model.geom_bodyid[i])).name for i in geom],
        distance_m=float(dist), position_m=np.asarray(pos).tolist(),
        active=bool(np.any(np.asarray(address) >= 0)))


def native_snapshot(model, data):
    return dict(nefc=int(data.nefc), constraints=constraint_counts(data.efc_type[:data.nefc]),
        contacts=[contact_record(model, c.geom, c.dist, c.pos, c.efc_address)
                  for c in data.contact[:data.ncon]])


def gpu_snapshot(engine):
    data, model = engine._solver.mjw_data, engine.rig.model
    count = int(data.nacon.numpy()[0])
    fields = {key: getattr(data.contact, key).numpy()[:count]
              for key in ('geom', 'dist', 'pos', 'efc_address', 'worldid')}
    nefc, types = data.nefc.numpy(), data.efc.type.numpy()
    result = [dict(nefc=int(n), constraints=constraint_counts(types[w, :n]), contacts=[])
              for w, n in enumerate(nefc)]
    for i, world in enumerate(fields['worldid']):
        result[int(world)]['contacts'].append(contact_record(model, fields['geom'][i],
            fields['dist'][i], fields['pos'][i], fields['efc_address'][i]))
    return result


def pair_counts(snapshot):
    return dict(Counter('/'.join(c['bodies']) for c in snapshot['contacts'] if c['active']))


def comparison(native, gpu):
    return dict(native_nefc=native['nefc'], gpu_nefc=gpu['nefc'],
        native_types=native['constraints'], gpu_types=gpu['constraints'],
        native_active_pairs=pair_counts(native), gpu_active_pairs=pair_counts(gpu),
        same_constraint_types=native['constraints'] == gpu['constraints'],
        same_active_pairs=pair_counts(native) == pair_counts(gpu))


class ContactTraceNativeEngine(NativeDummyEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = [[] for _ in self.datas]

    def _before_physics_step(self, world, data):
        self.before = (data.qpos.copy(), data.qvel.copy())

    def _after_physics_step(self, world, data):
        self.records[world].append(dict(qpos_before=self.before[0].tolist(), qvel_before=self.before[1].tolist(),
            qpos_after=data.qpos.tolist(), **native_snapshot(self.rig.model, data)))


def raw_native_state(state, model):
    q = state.joint_q.numpy().reshape(-1, model.nq).copy()
    v = state.joint_qd.numpy().reshape(-1, model.nv).copy()
    q[:, 3:7] = q[:, [6, 3, 4, 5]]
    for w in range(len(q)):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q[w, 3:7].astype(np.float64))
        v[w, 3:6] = rotation.reshape(3, 3).T @ v[w, 3:6]
    return q, v


@torch.no_grad()
def replay(rig, reference, commands, out):
    from envs.base_env import EnvMode
    from .torque_stand_task import TorqueStandTask
    from .stand_foundation_followup import gpu_probe_engine
    native = TorqueStandTask(rig, reference, 8, 'cpu', engine_factory=ContactTraceNativeEngine,
                            reward_mode='reference_v9', episode_seconds=60.)
    gpu = TorqueStandTask(rig, reference, 8, 'cuda:0', engine_factory=gpu_probe_engine('uncaptured'),
                         reward_mode='reference_v9', episode_seconds=60.)
    for task in (native, gpu):
        task.set_mode(EnvMode.TEST)
        task.reset()
    gpu_records = []
    original_step = gpu.engine._solver.step
    def traced_step(state_in, state_out, control, contacts, dt):
        q, v = raw_native_state(state_in, rig.model)
        original_step(state_in, state_out, control, contacts, dt)
        after, _ = raw_native_state(state_out, rig.model)
        snapshots = gpu_snapshot(gpu.engine)
        gpu_records.append([dict(qpos_before=q[w].tolist(), qvel_before=v[w].tolist(), qpos_after=after[w].tolist(),
                                **snapshot) for w, snapshot in enumerate(snapshots)])
    gpu.engine._solver.step = traced_step
    for command in commands[:9]:
        native.step(torch.tensor(command))
        gpu.step(torch.tensor(command, device='cuda:0'))
    gpu.engine._solver.step = original_step
    native_records = [[native.engine.records[w][s] for w in range(8)]
                      for s in range(len(gpu_records))]
    # Snapshot contacts refer to the solve's pre-integration state, not qpos_after.
    comparisons = [dict(substep=s, solve_start_seconds=s*rig.model.opt.timestep, case=w,
                        **comparison(n, g))
                   for s, (ns, gs) in enumerate(zip(native_records, gpu_records))
                   for w, (n, g) in enumerate(zip(ns, gs))]
    write_json(out/'substep-trace.json', dict(native=native_records, gpu=gpu_records))
    write_json(out/'substep-comparison.json', comparisons)
    matched = matched_state_probes(gpu.engine, native_records, gpu_records)
    write_json(out/'matched-state.json', matched)
    qdiff = np.abs(np.array([[r['qpos_after'] for r in worlds] for worlds in native_records]) -
                   np.array([[r['qpos_after'] for r in worlds] for worlds in gpu_records]))
    result = dict(substeps=len(gpu_records), duration_seconds=len(gpu_records)*rig.model.opt.timestep,
        max_root_difference_m=float(qdiff[:, :, :3].max()),
        max_hinge_difference_rad=float(qdiff[:, :, 7:].max()),
        trajectory_constraint_mismatches=[c for c in comparisons if not c['same_constraint_types']],
        matched_state_count=len(matched),
        matched_state_constraint_mismatches=[c for c in matched if not c['same_constraint_types']],
        matched_state_pair_mismatches=[c for c in matched if not c['same_active_pairs']])
    write_json(out/'report.json', result)
    return result


def matched_state_probes(engine, native_records, gpu_records, sample_stride=4):
    """Use identical float32 coordinates and fresh solver history for both backends."""
    import mujoco_warp
    import warp as wp
    model = engine.rig.model
    data = engine._solver.mjw_data
    native = [mujoco.MjData(model) for _ in range(8)]
    result = []
    for source, records in (('native', native_records), ('gpu', gpu_records)):
        for step, worlds in enumerate(records):
            if step % sample_stride != sample_stride-1:
                continue
            q = np.array([r['qpos_before'] for r in worlds], dtype=np.float32)
            v = np.array([r['qvel_before'] for r in worlds], dtype=np.float32)
            mujoco_warp.reset_data(engine._solver.mjw_model, data)
            wp.to_torch(data.qpos).copy_(torch.tensor(q, device='cuda:0'))
            wp.to_torch(data.qvel).copy_(torch.tensor(v, device='cuda:0'))
            mujoco_warp.forward(engine._solver.mjw_model, data)
            snapshots = gpu_snapshot(engine)
            for w, snapshot in enumerate(snapshots):
                mujoco.mj_resetData(model, native[w])
                native[w].qpos[:], native[w].qvel[:] = q[w], v[w]
                mujoco.mj_forward(model, native[w])
                n = native_snapshot(model, native[w])
                result.append(dict(source=source, substep=step, case=w,
                    **comparison(n, snapshot), native_contacts=n['contacts'], gpu_contacts=snapshot['contacts']))
    return result


def historical_pose_probes(rig, paths):
    """Probe previously divergent poses; zero velocity isolates constraint generation."""
    from .stand_foundation_followup import gpu_probe_engine
    engine = gpu_probe_engine('uncaptured')(rig, 8, 'cuda:0')
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            records = {}
            for key in ('native_qpos', 'gpu_qpos'):
                # Existing traces store control-interval ending poses. No velocity
                # is saved, so this screen does not compare the original forces.
                records[key] = [[dict(qpos_before=q.tolist(), qvel_before=[0.]*rig.model.nv)
                                 for q in worlds]
                                for worlds in trace[key][:9]]
            probes = matched_state_probes(engine, records['native_qpos'], records['gpu_qpos'], sample_stride=1)
            for probe in probes:
                probe['trace'] = str(path)
                probe['control_sample'] = probe.pop('substep')
            result.extend(probes)
    return result


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {p: sha256(p) for p in (args.model, args.reference, args.commands)}
    rig = Rig.load(args.model, policy_exclude=())
    commands = np.load(args.commands, allow_pickle=False)['command']
    write_json(args.out/'protocol.json', dict(inputs={str(p): h for p, h in inputs.items()},
        source_sha256=sha256(Path(__file__)), repeats=2, graph_capture=False,
        matched_state_precision='Identical float32 qpos/qvel; native promotes these values to float64',
        training_updates=0))
    reports = []
    for trial in range(2):
        torch.manual_seed(301010)
        np.random.seed(301010)
        out = args.out/str(trial+1)
        out.mkdir()
        print('Tracing contacts, repetition', trial+1, flush=True)
        reports.append(replay(rig, args.reference, commands, out))
    paths = [Path('logs/mimickit-training/stand-foundation-followup-20261001-01')/
             f'baseline-{trial}'/'trace.npz' for trial in (1, 2)]
    historical_hashes = {str(p): sha256(p) for p in paths}
    print('Probing saved historical poses', flush=True)
    historical = historical_pose_probes(rig, paths)
    write_json(args.out/'historical-matched-state.json', historical)
    if any(sha256(Path(p)) != h for p, h in historical_hashes.items()):
        raise RuntimeError('Historical trace changed')
    if any(sha256(p) != h for p, h in inputs.items()):
        raise RuntimeError('Protected input changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        reports=reports, historical_trace_hashes=historical_hashes,
        historical_matched_pose_count=len(historical),
        historical_constraint_mismatches=[p for p in historical if not p['same_constraint_types']],
        historical_pair_mismatches=[p for p in historical if not p['same_active_pairs']],
        historical_scope='Identical float32 saved qpos; zero velocities; contact sets only, not historical force parity',
        inputs_unchanged=True, training_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('logs/mimickit-training/stand-fresh-winner-60min-20261001-01/dummy.xml'))
    parser.add_argument('--reference', type=Path, default=Path('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz'))
    parser.add_argument('--commands', type=Path, default=Path('logs/mimickit-training/stand-foundation-tests-20261001-03/final_actor-trace.npz'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

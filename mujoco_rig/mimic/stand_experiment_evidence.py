"""Provenance and saved-rollout checks shared by isolated Stand diagnostics."""
from pathlib import Path
import shutil
import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .replicate_stand_representations import check_protected, summarize_trace, verify_control_trace
from .verify_stand_learning_curve import verify_gates
from .train_stand_imitation import tensor_hash


def protect(prior, out, source_paths, extra_inputs=()):
    protocol, verification = read(prior/'protocol.json'), read(prior/'final-verification.json')
    inputs = {**protocol['inputs'], **verification['output_hashes']}
    for path in (prior/'final-verification.json', *extra_inputs):
        inputs[str(path)] = digest(path)
    paths = [Path(__file__), *source_paths]
    for source in protocol['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            inputs[source[key]] = source['sha256']
    paths = list({str(p.resolve()): p for p in paths}.values())
    for path in paths:
        expected = digest(path)
        assert str(path) not in inputs or inputs[str(path)] == expected
        inputs[str(path)] = expected
    check_protected(inputs, [])
    out.mkdir(parents=True, exist_ok=False)
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(prior/name, out/name)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return inputs, sources


def load_actor(rig, reference, protocol, checkpoint):
    from .representation_task import AMPRepresentationTask
    from .amp_recovery import validate_sidecar
    from .train_torque_stand import build_agent
    from .policy import StandPolicy
    task = AMPRepresentationTask(rig, reference, 8, 'cpu', representation='target_pd',
        reward_mode='reference_v9', episode_seconds=12., seed=301020,
        demonstrations=[Path(r['path']) for r in protocol['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    agent = build_agent(protocol['config'], task, 'cpu')
    full = torch.load(checkpoint.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(checkpoint, map_location='cpu', weights_only=True), full)
    agent.load_state_dict(full)
    assert tensor_hash(agent.state_dict()) == tensor_hash(full)
    return StandPolicy(agent).eval().requires_grad_(False)


def verify_rollout(folder):
    contract, report = read(folder/'contract.json'), read(folder/'report.json')
    with np.load(folder/'trace.npz', allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    live, dt = trace['alive_before'], contract['control_timestep']
    assert live.shape[1] == 8 and len(live) <= 720
    assert contract['representation'] == 'target_pd' and contract['num_obs'] == 374
    assert contract['passive_joint_stiffness'] == 0 and contract['standing_pose_servo']
    assert np.array_equal(trace['time_s'], (np.arange(len(live))+1)*dt)
    assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
    assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
    assert all(np.isfinite(trace[key][live]).all() for key in
        ('observation', 'action', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after'))
    assert not trace['solver_warning'][live].any()
    verify_gates(trace, '')
    verify_gates(trace, 'pre_')
    assert verify_control_trace(trace, contract) == report['max_control_formula_error_nm']
    assert all(report[k] == v for k, v in summarize_trace(trace, dt).items())
    for case in range(8):
        count = int(live[:, case].sum())
        assert live[:count, case].all() and not live[count:, case].any()
        assert (trace['done'][:count-1, case] == 0).all() and trace['done'][count-1, case] in (1, 3)
    return trace, report


def seal(out, inputs, sources, **summary):
    check_protected(inputs, sources)
    save(out/'final-verification.json', dict(protected_inputs=len(inputs), sources=len(sources),
        inputs_and_sources_unchanged=True, **summary,
        output_hashes={str(p): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'final-verification.json'}))

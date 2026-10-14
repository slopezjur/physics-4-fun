"""Analyze saved Stand traces without importing a simulator or learner."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def balance_gate(source):
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'BALANCE_GATE' for target in node.targets):
            return {entry.arg: ast.literal_eval(entry.value) for entry in node.value.keywords}
    raise ValueError('Balance gate definition missing')


def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.max(stops-starts)) if len(starts) else 0


def event(mask, times, dt):
    starts = np.flatnonzero(mask)
    required = int(np.ceil(.1/dt))
    sustained = np.flatnonzero(np.convolve(mask.astype(int), np.ones(required, int), 'valid') == required) if len(mask) >= required else []
    return dict(first_seconds=float(times[starts[0]]) if len(starts) else None,
                first_sustained_100ms_seconds=float(times[sustained[0]]) if len(sustained) else None,
                violating_seconds=float(mask.sum()*dt), longest_violation_seconds=float(longest(mask)*dt))


def state_summary(a, case, mask):
    loads = a['foot_loads_n'][mask, case]
    return dict(height_mean_m=float(a['pelvis_height_m'][mask, case].mean()),
                height_min_m=float(a['pelvis_height_m'][mask, case].min()),
                tilt_mean_degrees=float(a['pelvis_tilt_degrees'][mask, case].mean()),
                tilt_max_degrees=float(a['pelvis_tilt_degrees'][mask, case].max()),
                horizontal_speed_mean_m_s=float(np.linalg.norm(a['qvel'][mask, case, :2], axis=-1).mean()),
                angular_speed_mean_rad_s=float(np.linalg.norm(a['qvel'][mask, case, 3:6], axis=-1).mean()),
                foot_load_mean_n=loads.mean(0).tolist(),
                minimum_foot_load_n=float(loads.min()),
                balanced_fraction=float(a['balance'][mask, case].mean()),
                posture_fraction=float(a['posture'][mask, case].mean()))


def analyze(path, contract, joints, gate, teacher, sampled):
    with np.load(path/'trace.npz', allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    report = read(path/'report.json')
    dt, latency = contract['control_timestep'], contract['latency_steps']
    live, times = a['alive_before'], a['time_s']
    frames, worlds = live.shape
    assert worlds == 8 and latency == 2
    assert a['qpos'].shape == (frames, worlds, 46) and a['qvel'].shape == (frames, worlds, 45)
    assert a['action'].shape == (frames, worlds, 33)
    assert np.allclose(times, (np.arange(frames)+1)*dt, rtol=0, atol=1e-12)
    assert all(np.isfinite(value).all() for value in a.values())
    assert all(np.array_equal(a[key], teacher[key]) for key in ('initial_qpos', 'initial_qvel'))
    bounded = a['action'].clip(-1., 1.)
    scale = np.asarray(contract['torque_scale_nm'], dtype=np.float32)
    held = np.zeros_like(bounded)
    held[latency:] = bounded[:-latency]*scale
    if sampled:
        assert np.array_equal(a['bounded_command'], bounded)
        assert np.array_equal(a['held_torque_nm'], held)
        assert np.array_equal(a['raw_sampled_action'][live].clip(-1., 1.), bounded[live])
    cases = []
    for case in range(worlds):
        n = int(live[:, case].sum())
        assert live[:n, case].all() and not live[n:, case].any()
        assert n == report['cases'][case]['frames']
        assert np.isclose(times[n-1], report['cases'][case]['survival_seconds'])
        t = times[:n]
        angular = np.linalg.norm(a['qvel'][:n, case, 3:6], axis=-1)
        speed = np.linalg.norm(a['qvel'][:n, case, :2], axis=-1)
        violations = dict(balance=~a['balance'][:n, case], posture=~a['posture'][:n, case],
            height=a['pelvis_height_m'][:n, case] < gate['min_pelvis_height_m'],
            tilt=a['pelvis_tilt_degrees'][:n, case] > gate['max_pelvis_tilt_degrees'],
            horizontal_speed=speed > gate['max_horizontal_speed_m_s'],
            angular_speed=angular > gate['max_angular_speed_rad_s'],
            foot_support=a['foot_loads_n'][:n, case].min(-1) < gate['min_each_foot_load_n'])
        windows = {}
        for seconds in (.05, .1, .25, .5, 1.):
            mask = live[:, case] & (times <= seconds+1e-5)
            windows[str(seconds)] = dict(frames=int(mask.sum()), complete=bool(t[-1] >= seconds-1e-5),
                **state_summary(a, case, mask))
        tail = live[:, case] & (times > t[-1]-.25)
        endpoints = {key: a[key][n-1, case].tolist() for key in ('pelvis_height_m', 'pelvis_tilt_degrees', 'foot_loads_n')}
        cases.append(dict(case=case, survival_seconds=float(t[-1]), terminal_flag=report['cases'][case]['terminal_flag'],
            height_below_failure_threshold=bool(a['pelvis_height_m'][n-1, case] < .65),
            solver_warning=bool(a['solver_warning'][:n, case].any()),
            events={key:event(mask, t, dt) for key,mask in violations.items()},
            balanced_seconds=float(a['balance'][:n, case].sum()*dt),
            longest_balanced_seconds=float(longest(a['balance'][:n, case])*dt),
            balance_recovered_after_first_violation=bool(a['balance'][:n, case][np.flatnonzero(violations['balance'])[0]+1:].any()) if violations['balance'].any() else None,
            early_windows=windows, pre_terminal_250ms=state_summary(a, case, tail), endpoint=endpoints))
    names = contract['actuator_names']
    positions = a['qpos'][:, :, 7:]
    velocities = a['qvel'][:, :, 6:]
    drift = positions-a['initial_qpos'][None, :, 7:]
    joint_rows = []
    for index, joint in enumerate(joints):
        p, v, d = positions[:, :, index][live], velocities[:, :, index][live], drift[:, :, index][live]
        lower, upper = joint['range_rad']
        distance = np.minimum(p-lower, upper-p)
        joint_rows.append(dict(**joint, mean_drift_degrees=float(np.rad2deg(d).mean()),
            rms_drift_degrees=float(np.sqrt(np.mean(np.rad2deg(d)**2))), max_abs_drift_degrees=float(np.abs(np.rad2deg(d)).max()),
            rms_velocity_rad_s=float(np.sqrt(np.mean(v**2))), near_limit_5deg_fraction=float((distance <= np.deg2rad(5)).mean()),
            outside_limit_fraction=float(((p<lower) | (p>upper)).mean())))
    commands = []
    for index, name in enumerate(names):
        raw, b, h = a['action'][:, :, index][live], bounded[:, :, index][live], held[:, :, index][live]
        joint_index = contract['joint_names'].index(name)
        joint_position = positions[:, :, joint_index][live]
        lower, upper = joints[joint_index]['range_rad']
        near_lower = joint_position <= lower+np.deg2rad(5)
        near_upper = joint_position >= upper-np.deg2rad(5)
        outward = (near_lower & (h < -1e-6)) | (near_upper & (h > 1e-6))
        commands.append(dict(name=name, first_command_mean=float(a['action'][0, :, index].mean()),
            teacher_first_command_mean=float(teacher['action'][0, :, index].mean()),
            first_command_teacher_rmse=float(np.sqrt(np.mean((a['action'][0, :, index]-teacher['action'][0, :, index])**2))),
            bounded_command_rms=float(np.sqrt(np.mean(b**2))), max_abs_raw_command=float(np.abs(raw).max()),
            clipping_fraction=float((np.abs(raw)>1).mean()), near_saturation_095_fraction=float((np.abs(b)>=.95).mean()),
            near_lower_joint_limit_fraction=float(near_lower.mean()),
            near_upper_joint_limit_fraction=float(near_upper.mean()),
            near_limit_outward_held_torque_fraction=float(outward.mean()),
            mean_outward_normalized_torque=float((np.abs(h)*outward/scale[index]).mean()),
            held_torque_rms_nm=float(np.sqrt(np.mean(h**2))), held_torque_max_abs_nm=float(np.abs(h).max())))
    aggregate = dict(episodes=worlds, mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in cases])),
        height_threshold_failures=sum(c['terminal_flag']=='FAIL' and c['height_below_failure_threshold'] for c in cases),
        fail_without_height_threshold=sum(c['terminal_flag']=='FAIL' and not c['height_below_failure_threshold'] for c in cases),
        balance_recoveries=sum(c['balance_recovered_after_first_violation'] is True for c in cases),
        mean_balanced_seconds=float(np.mean([c['balanced_seconds'] for c in cases])),
        raw_command_clipping_fraction=float((np.abs(a['action'][live])>1).mean()),
        near_saturation_095_fraction=float((np.abs(bounded[live])>=.95).mean()),
        bounded_command_rms=float(np.sqrt(np.mean(bounded[live]**2))),
        first_command_teacher_rmse=float(np.sqrt(np.mean((a['action'][0]-teacher['action'][0])**2))),
        maximum_raw_command=float(np.abs(a['action'][live]).max()),
        mean_initial_angular_speed_rad_s=float(np.linalg.norm(a['qvel'][2, :, 3:6],axis=-1).mean()))
    return dict(path=str(path), sampled=sampled, cases=cases, aggregate=aggregate,
        joints=joint_rows, commands=commands, resets_identical=True, timing_verified=True,
        report_verified=True, delayed_torques_recorded_and_verified=sampled,
        torque_scope='Recorded held torques' if sampled else 'Reconstructed from recorded commands and verified control contract; not independently recorded')


def run(args):
    root = args.root
    runs = {3:root/'stand-ppo-critic-lr1em3-20261002-01', 10:root/'stand-ppo-critic-lr1em3-10min-20261002-01', 30:root/'stand-ppo-critic-lr1em3-30min-20261002-01'}
    audits = {3:root/'stand-critic-audit-20261002-01', 10:root/'stand-duration-audit-20261002-01', 30:root/'stand-duration30-audit-20261002-01'}
    teacher_dir = root/'stand-reward-audit-20261002-01/teacher'
    entries = [('teacher', teacher_dir, False)]
    for minutes, path in audits.items():
        entries.append((f'{minutes}min-deterministic', path/'deterministic-native', False))
        entries.extend((f'{minutes}min-sampled-{seed}', path/f'trained_actor-{seed}', True) for seed in (301010,301011,301012))
    protected = {}
    for path in list(audits.values())+[teacher_dir.parent]:
        protocol = read(path/'protocol.json')
        protected.update(protocol['inputs'])
        for filename in ('report.json', 'protocol.json'):
            protected[str(path/filename)] = digest(path/filename)
        for source in protocol.get('sources', []):
            for key in ('path', 'snapshot'):
                protected[source[key]] = source['sha256']
    for path in runs.values():
        for filename in ('contract.json', 'dummy.xml', 'initialization.json'):
            protected[str(path/filename)] = digest(path/filename)
    for _,path,_ in entries:
        for filename in ('report.json', 'trace.npz'):
            protected[str(path/filename)] = digest(path/filename)
    sources = [Path(__file__), Path('mujoco_rig/mimic/torque_stand_task.py'), Path('mujoco_rig/mimic/stand_task.py'), Path('mujoco_rig/mimic/native_engine.py'), Path('mujoco_rig/mimic/rig.py')]
    protected.update({str(path):digest(path) for path in sources})
    assert all(digest(Path(path))==expected for path,expected in protected.items()), 'Input/source hash mismatch'
    contract = read(runs[30]/'contract.json')
    assert contract['model_sha256']==digest(runs[30]/'dummy.xml')
    assert all(read(path/'contract.json')==contract for path in runs.values())
    xml = ET.parse(runs[30]/'dummy.xml').getroot()
    assert xml.find('compiler').get('angle')=='radian'
    hinges = list(xml.find('worldbody').iter('joint'))
    assert [node.get('name') for node in hinges]==contract['joint_names']
    joints = [dict(name=node.get('name'), range_rad=[float(x) for x in node.get('range').split()], actuated=node.get('name') in contract['actuator_names']) for node in hinges]
    assert all(node.get('type')=='hinge' for node in hinges)
    motors = list(xml.find('actuator'))
    assert [node.get('name') for node in motors] == contract['actuator_names']
    assert all(node.tag == 'motor' and node.get('joint') == node.get('name') and
               float(node.get('gear', '1').split()[0]) == 1 for node in motors)
    gate = balance_gate(Path('mujoco_rig/mimic/torque_stand_task.py'))
    with np.load(teacher_dir/'trace.npz', allow_pickle=False) as archive:
        teacher = {key:archive[key] for key in archive.files}
    results = {label:analyze(path, contract, joints, gate, teacher, sampled) for label,path,sampled in entries}
    assert all(digest(Path(path))==expected for path,expected in protected.items()), 'Protected artifact changed'
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.out/'runner.py')
    save(args.out/'protocol.json', dict(inputs=protected, balance_gate=gate, contract=contract,
        live_prefix_only=True, command_near_saturation_threshold=.95, joint_limit_margin_degrees=5,
        sustained_violation_seconds=.1, torque_delay_steps=2,
        limitations=['No non-foot contact forces or contacting body IDs in these traces.',
            'No per-body tilt/clearance or individual posture-criterion flags.',
            'Joint drifts are measured from the paired reset, not reconstructed body pose errors.',
            'Teacher has a different controller; offline trajectory associations are not causal interventions.',
            'Per-frame command/joint aggregates weight longer episodes more heavily. Event means weight episodes equally.',
            'Teacher/deterministic held torques are reconstructed; sampled held torques are recorded and verified.']))
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        traces=results, inputs_unchanged=True, source_snapshots_verified=True, resets_identical=True,
        traces_analyzed=len(results), episodes_analyzed=sum(v['aggregate']['episodes'] for v in results.values()),
        training_updates=0, physics_rollouts=0, simulator_imported=False, arena_changed=False))
    print(json.dumps({label:value['aggregate'] for label,value in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('logs/mimickit-training'))
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())

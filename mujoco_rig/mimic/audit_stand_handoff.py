"""Paired frozen-policy assessment with six recorded teacher commands."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .audit_stand_critic import effective_config, read_json
from .audit_stand_failures import analyze, balance_gate, longest
from .audit_stand_rewards import source_audit, trial
from .audit_stand_sampling import SamplingNativeEngine, SampledActor, SEEDS
from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent

PREFIX_COMMANDS = 6


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class PrefixActor:
    """Consume every policy proposal and override only executed prefix commands."""
    def __init__(self, actor, teacher_commands, enabled):
        self.actor, self.teacher_commands, self.enabled = actor, teacher_commands, enabled
        self.proposals, self.overrides = [], []

    @torch.no_grad()
    def __call__(self, observation):
        proposal = self.actor(observation)
        index = len(self.proposals)
        override = self.enabled and index < PREFIX_COMMANDS
        self.proposals.append(proposal.numpy().copy())
        self.overrides.append(override)
        return self.teacher_commands[index].clone() if override else proposal.clone()


def handoff_metrics(arrays, contract, joints):
    first_learned = PREFIX_COMMANDS+contract['latency_steps']
    indices = np.array([contract['joint_names'].index(name) for name in contract['actuator_names']])
    lower = np.array([joints[index]['range_rad'][0] for index in indices])
    upper = np.array([joints[index]['range_rad'][1] for index in indices])
    positions = arrays['qpos'][:, :, 7+indices]
    held = arrays['held_torque_nm']
    outward = ((positions <= lower+np.deg2rad(5)) & (held < -1e-6)) | ((positions >= upper-np.deg2rad(5)) & (held > 1e-6))
    rows = []
    for case in range(8):
        alive = arrays['alive_before'][:, case]
        count = int(alive.sum())
        assert count > first_learned
        mask = alive & (np.arange(len(alive)) >= first_learned)
        balance, posture = arrays['balance'][mask, case], arrays['posture'][mask, case]
        bad = np.flatnonzero(~balance)
        angular = np.linalg.norm(arrays['qvel'][:, case, 3:6], axis=-1)
        dt = contract['control_timestep']
        rows.append(dict(case=case, survival_seconds=float(arrays['time_s'][count-1]),
            total_balanced_seconds=float(arrays['balance'][alive, case].sum()*dt),
            longest_balanced_seconds=float(longest(arrays['balance'][alive, case])*dt),
            longest_balanced_and_posture_valid_seconds=float(longest((arrays['balance'] & arrays['posture'])[alive, case])*dt),
            post_handoff_balanced_seconds=float(balance.sum()*dt),
            longest_post_handoff_balanced_seconds=float(longest(balance)*dt),
            longest_post_handoff_balanced_and_posture_valid_seconds=float(longest(balance & posture)*dt),
            first_post_handoff_balance_violation_seconds=float(arrays['time_s'][first_learned+bad[0]]) if len(bad) else None,
            teacher_prefix_end_balanced=bool(arrays['balance'][first_learned-1, case]),
            teacher_prefix_end_angular_speed_rad_s=float(angular[first_learned-1]),
            first_learned_torque_interval_balanced=bool(arrays['balance'][first_learned, case]),
            first_learned_torque_interval_angular_speed_rad_s=float(angular[first_learned]),
            initial_torque_interval_angular_speed_rad_s=float(angular[2]),
            post_handoff_outward_limit_torque_fraction=float(outward[mask, case].mean()),
            post_handoff_outward_limit_torque_by_motor={name:float(outward[mask, case, index].mean()) for index,name in enumerate(contract['actuator_names'])}))
    return dict(cases=rows, means={key:float(np.mean([row[key] for row in rows])) for key in
        ('survival_seconds', 'total_balanced_seconds', 'longest_balanced_seconds',
         'longest_balanced_and_posture_valid_seconds', 'post_handoff_balanced_seconds',
         'longest_post_handoff_balanced_seconds', 'longest_post_handoff_balanced_and_posture_valid_seconds',
         'teacher_prefix_end_angular_speed_rad_s', 'first_learned_torque_interval_angular_speed_rad_s',
         'initial_torque_interval_angular_speed_rad_s', 'post_handoff_outward_limit_torque_fraction')})


def execute(task, scorer, mean_actor, teacher, contract, joints, gate, arm, seed, out):
    from learning.base_agent import AgentMode
    sampled = seed is not None
    scorer.set_mode(AgentMode.TRAIN if sampled else AgentMode.TEST)
    if sampled:
        torch.manual_seed(seed)
        np.random.seed(seed)
    base = SampledActor(scorer) if sampled else mean_actor
    actor = PrefixActor(base, torch.from_numpy(teacher['action'][:PREFIX_COMMANDS]), arm == 'teacher-prefix')
    out.mkdir()
    report = trial(task, scorer, actor, 'trained_actor', 12., out)
    arrays = load_trace(out/'trace.npz')
    arrays['policy_proposed_action'] = np.stack(actor.proposals)
    arrays['prefix_override'] = np.asarray(actor.overrides)
    arrays['bounded_command'] = arrays['action'].clip(-1., 1.)
    arrays['held_torque_nm'] = np.stack(task.engine.held_torques)
    if sampled:
        arrays.update({key:np.stack([row[key] for row in base.records]) for key in base.records[0]})
    expected = np.zeros_like(arrays['held_torque_nm'])
    expected[2:] = arrays['bounded_command'][:-2]*np.asarray(contract['torque_scale_nm'], dtype=np.float32)
    assert np.array_equal(expected, arrays['held_torque_nm']), 'Delayed executed torque mismatch'
    live = arrays['alive_before']
    override = np.broadcast_to(arrays['prefix_override'][:, None], live.shape)
    assert np.array_equal(arrays['action'][live & ~override], arrays['policy_proposed_action'][live & ~override])
    assert all(np.array_equal(arrays[key], teacher[key]) for key in ('initial_qpos', 'initial_qvel'))
    teacher_frames = PREFIX_COMMANDS+contract['latency_steps'] if arm == 'teacher-prefix' else 2
    assert all(np.array_equal(arrays[key][:teacher_frames], teacher[key][:teacher_frames]) for key in ('qpos', 'qvel'))
    if arm == 'teacher-prefix':
        assert np.array_equal(arrays['action'][:PREFIX_COMMANDS], teacher['action'][:PREFIX_COMMANDS])
    if sampled:
        assert np.array_equal(arrays['policy_proposed_action'], arrays['raw_sampled_action'])
    np.savez_compressed(out/'trace.npz', **arrays)
    diagnostics = analyze(out, contract, joints, gate, teacher, sampled=False)
    diagnostics['delayed_torques_recorded_and_verified'] = True
    diagnostics['torque_scope'] = 'Recorded held torques verified against executed commands, including the teacher prefix'
    report.update(arm=arm, inference_seed=seed, handoff=handoff_metrics(arrays, contract, joints),
        physical_diagnostics=diagnostics, delayed_torque_mapping_verified=True,
        teacher_state_reproduction_frames=teacher_frames,
        critic_scope='Frozen original Gaussian-policy critic; teacher-prefix trajectories are off-policy. No intervention calibration claim.')
    write_json(out/'report.json', report)
    return report, arrays


def summarize_reports(reports):
    cases = [case for report in reports for case in report['handoff']['cases']]
    return dict(episodes=len(cases), standing_passes=sum(report['standing_passes'] for report in reports),
        survivors=sum(report['survivors'] for report in reports),
        first_learned_interval_balanced_cases=sum(row['first_learned_torque_interval_balanced'] for row in cases),
        means={key:float(np.mean([report['handoff']['means'][key] for report in reports])) for key in reports[0]['handoff']['means']},
        outward_limit_torque_by_motor={name:float(np.mean([row['post_handoff_outward_limit_torque_by_motor'][name] for row in cases])) for name in cases[0]['post_handoff_outward_limit_torque_by_motor']})


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar
    from .torque_stand_task import AMPTorqueStandTask
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(1)
    training = read_json(args.run/'report.json')
    protocol = read_json(args.run/'protocol.json')
    paired_protocol = read_json(args.paired_audit/'protocol.json')
    paired_report = read_json(args.paired_audit/'report.json')
    assert paired_report['frozen_tensors_unchanged'] and paired_report['inputs_unchanged']
    assert paired_report['training']['model_sha256'] == training['model_sha256'] == sha256(args.run/'model.pt')
    protected = dict(paired_protocol['inputs'])
    protected.update(protocol['inputs'])
    for source in paired_protocol['sources']:
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    for path in (args.paired_audit, args.teacher.parent):
        for filename in ('report.json', 'protocol.json'):
            protected[str(path/filename)] = sha256(path/filename)
    for subdir in ['deterministic-native']+[f'trained_actor-{seed}' for seed in SEEDS]:
        for filename in ('report.json', 'trace.npz'):
            path = args.paired_audit/subdir/filename
            protected[str(path)] = sha256(path)
    for filename in ('report.json', 'trace.npz'):
        path = args.teacher/filename
        protected[str(path)] = sha256(path)
    for filename in ('contract.json', 'model.amp.pt'):
        path = args.run/filename
        protected[str(path)] = sha256(path)
    assert all(sha256(Path(path)) == value for path,value in protected.items()), 'Protected input/source mismatch'
    contract = read_json(args.run/'contract.json')
    teacher = load_trace(args.teacher/'trace.npz')
    assert read_json(args.teacher/'report.json')['standing_passes'] == 8
    state = torch.load(args.run/'model.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(args.run/'model.pt', map_location='cpu', weights_only=True), state)
    rig = Rig.load(args.run/'dummy.xml', policy_exclude=())
    assert contract['latency_steps'] == 2 and not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    assert sha256(rig.path) == contract['model_sha256']
    teacher_inputs = read_json(args.teacher.parent/'protocol.json')['inputs']
    original = Path('logs/mimickit-training/stand-teacher-20261001-01')
    for filename in ('dummy.xml', 'stationary_reference.npz'):
        assert sha256(args.run/filename) == teacher_inputs[str(original/filename)]
    xml = ET.parse(rig.path).getroot()
    hinges = list(xml.find('worldbody').iter('joint'))
    assert [node.get('name') for node in hinges] == contract['joint_names']
    joints = [dict(name=node.get('name'), range_rad=[float(x) for x in node.get('range').split()], actuated=node.get('name') in contract['actuator_names']) for node in hinges]
    gate = balance_gate(Path('mujoco_rig/mimic/torque_stand_task.py'))
    config = effective_config(args.run/'agent.yaml')
    task = AMPTorqueStandTask(rig, args.run/'stationary_reference.npz', 8, 'cpu', seed=protocol['seed'],
        reward_mode='reference_v9', episode_seconds=12., engine_factory=SamplingNativeEngine,
        demonstrations=[Path(record['path']) for record in protocol['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    scorer = build_agent(config, task, 'cpu')
    scorer.load_state_dict(state)
    scorer.eval().requires_grad_(False)
    actor = StandPolicy(scorer).eval().requires_grad_(False)
    before = dict(scorer=tensor_hash(scorer.state_dict()), actor=tensor_hash(actor.state_dict()))
    assert before == paired_protocol['frozen_tensor_hashes'], 'Frozen snapshot differs from paired audit'
    args.out.mkdir(parents=True, exist_ok=False)
    sources = source_audit(args.out, args.mimickit)['sources']
    extra = [Path(__file__), Path('mujoco_rig/mimic/audit_stand_sampling.py'),
        Path('mujoco_rig/mimic/audit_stand_failures.py'), Path('mujoco_rig/mimic/audit_stand_critic.py'),
        Path('mujoco_rig/mimic/rig.py'), Path('mujoco_rig/mimic/native_engine.py'), Path('mujoco_rig/mimic/policy.py'),
        args.mimickit/'mimickit/learning/distribution_gaussian_diag.py']
    for path in extra:
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=sha256(path)))
    write_json(args.out/'protocol.json', dict(inputs=protected, sources=sources, run=str(args.run),
        paired_audit=str(args.paired_audit), teacher=str(args.teacher), config=config,
        frozen_tensor_hashes=before, prefix_commands=PREFIX_COMMANDS,
        prefix_command_seconds=PREFIX_COMMANDS*task.dt,
        first_learned_torque_interval_end_seconds=(PREFIX_COMMANDS+contract['latency_steps']+1)*task.dt,
        inference_seeds=list(SEEDS), native_cases=8, arms=['control', 'teacher-prefix'],
        seconds=12., total_episodes=64, training_updates=0, arena_changed=False,
        scoring='Original frozen AMP snapshot; intervention critic scores are off-policy diagnostics.'))
    results = {arm:{} for arm in ('control', 'teacher-prefix')}
    for seed in (None,)+SEEDS:
        label = 'deterministic' if seed is None else f'sampled-{seed}'
        pair = {}
        for arm in results:
            report, arrays = execute(task, scorer, actor, teacher, contract, joints, gate,
                arm, seed, args.out/f'{arm}-{label}')
            if arm == 'control':
                old_label = 'deterministic-native' if seed is None else f'trained_actor-{seed}'
                paired = load_trace(args.paired_audit/old_label/'trace.npz')
                assert all(np.array_equal(arrays[key], value) for key,value in paired.items()), 'Control trace does not reproduce paired audit'
                report['paired_control_trace_reproduced'] = True
            else:
                if seed is not None:
                    frames = min(len(arrays['done']), len(pair['control']['done']))
                    assert np.array_equal(arrays['standard_normal_draw'][:frames], pair['control']['standard_normal_draw'][:frames]), 'Paired Gaussian draws differ'
                report['paired_gaussian_draws_verified'] = seed is not None
            pair[arm] = arrays
            results[arm][label] = report
            write_json(args.out/f'{arm}-{label}'/'report.json', report)
            completed = sum(len(value) for value in results.values())
            write_json(args.out/'progress.json', dict(completed_trials=completed, total_trials=8))
            print(dict(arm=arm, seed=seed, survival=report['mean_survival_seconds'],
                post_handoff_balanced_seconds=report['handoff']['means']['post_handoff_balanced_seconds']), flush=True)
    after = dict(scorer=tensor_hash(scorer.state_dict()), actor=tensor_hash(actor.state_dict()))
    assert after == before and not scorer._obs_norm._pending_count and not scorer._disc_obs_norm._pending_count
    assert all(sha256(Path(path)) == value for path,value in protected.items()), 'Protected input changed'
    assert all(sha256(Path(source[key])) == source['sha256'] for source in sources for key in ('path', 'snapshot'))
    summaries = {arm:dict(deterministic=summarize_reports([reports['deterministic']]),
        sampled=summarize_reports([reports[f'sampled-{seed}'] for seed in SEEDS])) for arm,reports in results.items()}
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        summaries=summaries, trials=results, final_sidecar_valid=True,
        paired_control_traces_reproduced=True, paired_gaussian_draws_verified=True,
        reset_states_identical=True, teacher_delayed_prefix_states_reproduced=True,
        delayed_torque_mapping_verified=True, frozen_tensors_unchanged=True,
        frozen_tensor_hashes_before=before, frozen_tensor_hashes_after=after,
        normalizer_pending_samples=0, inputs_unchanged=True, source_snapshots_verified=True,
        total_episodes=64, training_updates=0, arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('logs/mimickit-training/stand-ppo-critic-lr1em3-30min-20261002-01'))
    parser.add_argument('--paired-audit', type=Path, default=Path('logs/mimickit-training/stand-duration30-audit-20261002-01'))
    parser.add_argument('--teacher', type=Path, default=Path('logs/mimickit-training/stand-reward-audit-20261002-01/teacher'))
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

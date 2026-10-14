"""Bounded MimicKit AMP training on a spring-free, direct-torque body."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
from xml.etree import ElementTree as ET

import numpy as np
import torch
import yaml

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def state_sha256(agent):
    """Fingerprint network and normalization tensors independently of serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(agent.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def prepare_plant(source, destination):
    tree = ET.parse(source)
    root = tree.getroot()
    for joint in root.iter('joint'):
        if joint.get('name'):
            joint.set('stiffness', '0')
    for actuator in root.find('actuator'):
        if actuator.tag == 'position':
            force = actuator.get('forcerange')
            name, joint = actuator.get('name'), actuator.get('joint')
            actuator.tag = 'motor'
            actuator.attrib.clear()
            actuator.attrib.update(name=name, joint=joint, gear='1', forcerange=force,
                                  ctrlrange=force, ctrllimited='true', forcelimited='true')
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def sample_gpu(stop, path):
    from datetime import datetime, timezone
    while not stop.is_set():
        try:
            line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                            '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()[0]
            values = [float(v.strip()) for v in line.split(',')]
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), gpu_percent=values[0],
                                        memory_mib=values[1], power_w=values[2]))+'\n')
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(2)


def build_agent(config, task, device, initialize_from=None):
    """Construct the same AMP learner for training and isolated throughput measurements."""
    from learning.amp_agent import AMPAgent
    from .amp_recovery import actor_state_dict, amp_sidecar
    from .stable_normalizer import install, StableNormalizer

    class TorqueStandAMP(AMPAgent):
        def _build_normalizers(self):
            super()._build_normalizers()
            install(self)
            self._disc_obs_norm = StableNormalizer(self._disc_obs_norm.get_shape(), device=self._device, clip=10.)

        def save(self, path):
            torch.save(self.state_dict(), amp_sidecar(path))
            torch.save(actor_state_dict(self.state_dict()), path)

    agent = TorqueStandAMP(config, task, device)
    if initialize_from:
        contract = json.loads((initialize_from.parent/'contract.json').read_text())
        if contract['observation_contract'] == 'mimic_stationary_stand_torque_v1':
            from .torque_stand_transfer import migrate_state
            state = torch.load(amp_sidecar(initialize_from), map_location='cpu', weights_only=True)
            agent.load_state_dict(migrate_state(state, agent.state_dict(), contract, task.observation_contract()))
            agent._sync_optimizer()
        else:
            agent.load(str(amp_sidecar(initialize_from)))
    return agent


def export_preview(agent, task, args):
    """Convert final weights and prepare an unlimited Stand Arena, without running it."""
    import onnxruntime as ort
    out = args.out/'arena-preview'
    out.mkdir(exist_ok=False)
    actor = StandPolicy(agent).cpu().eval()
    sample = task.observations().cpu()
    torch.onnx.export(actor, sample[:1], str(out/'stand.onnx'), input_names=['observation'],
                      output_names=['action'], dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    # Conversion parity is part of exporting the actor, not a separate physics test.
    runtime = ort.InferenceSession(str(out/'stand.onnx'), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        error = float(np.abs(runtime.run(None, {'observation': sample.numpy()})[0]-actor(sample).numpy()).max())
    if not np.isfinite(error) or error>1e-5:
        raise RuntimeError(f'ONNX conversion difference {error}')
    shutil.copy2(task.rig.path, out/'dummy.xml')
    shutil.copy2(args.reference, out/'stand_reference.npz')
    q, v = task.sample_reference(torch.zeros(1), torch.zeros(1, dtype=torch.long))
    frames = dict(dt=60., reference_sha256=sha256(args.reference), qpos=[q[0].tolist()]*2, qvel=[v[0].tolist()]*2)
    write_json(out/'reference_frames.json', frames)
    contract = task.observation_contract()
    contract.update(actor_sha256=sha256(out/'stand.onnx'), reference_frames_sha256=sha256(out/'reference_frames.json'),
                    reference_duration=60., scene_posture=0)
    write_json(out/'contract.json', contract)
    selection = dict(schema='mimic_torque_stand_preview_v1', checkpoint=str(args.out/'model.pt'),
                     checkpoint_sha256=sha256(args.out/'model.pt'), onnx_sha256=contract['actor_sha256'],
                     model_sha256=contract['model_sha256'], onnx_max_action_error=error,
                     policy_actions=len(task.rig.actuators), observation_contract=contract['observation_contract'],
                     posture_reward=contract.get('posture_reward'),
                     experimental_preview=True, motor_control='direct joint torque', passive_joint_stiffness=0,
                     standing_pose_servo=False, standalone_tests_run=False, compiled=False, godot_run=False)
    write_json(out/'arena-selection.json', selection)
    # Separate Arena keeps all historical trained and passive-ball previews available.
    scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicStandTorqueArena.tscn')
    bundle = 'res://'+out.resolve().relative_to(Path.cwd().resolve()).as_posix()
    scene.write_text('[gd_scene load_steps=2 format=3]\n\n'
        '[ext_resource type="PackedScene" path="res://Scenes/RL/Isaac3/MuJoCo/MimicStandArena.tscn" id="1_base"]\n\n'
        '[node name="MimicStandTorqueArena" instance=ExtResource("1_base")]\n'
        f'ModelPath = "{bundle}/dummy.xml"\nBundleDirectory = "{bundle}"\n'
        'UseLatestExport = false\nRunIndefinitely = true\nStartPhaseSeconds = 0.0\n\n'
        '[node name="Title" parent="Hud/Margin/Controls" index="0"]\n'
        'text = "Mimic Stand — spring-free learned torques"\n\n'
        '[node name="Help" parent="Hud/Margin/Controls" index="3"]\n'
        'text = "R: restart · P: pause\\nDirect learned torques · no joint springs or standing-pose servo\\nExperimental final candidate · unlimited quiet standing"\n', encoding='utf-8')
    return selection


def train(args):
    comparison = getattr(args, 'comparison', False)
    actor_epochs = getattr(args, 'actor_epochs', None)
    if actor_epochs is not None and actor_epochs <= 0:
        raise ValueError('Actor epochs must be positive')
    if comparison and (args.initialize_from or args.reward_mode != 'reference_v9'):
        raise ValueError('Comparison requires fresh initialization and reference_v9')
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import actor_state_dict, amp_sidecar, validate_sidecar
    from .torque_stand_task import (TorqueStandTask, AMPTorqueStandTask, evaluate_standing,
                                   CONTRACT, POSTURE, STAGED_POSTURE, WORST_POSTURE,
                                   UPRIGHT_IMITATION, WHOLE_POSE_IMITATION, LIMIT_AWARE_IMITATION,
                                   LIMIT_OCCUPANCY_IMITATION, REFERENCE_IMITATION,
                                   STAGED_MODES, UPRIGHT_MODES, reward_contract)
    from .torque_stand_curriculum import CURRICULUM, PostureCurriculum, UPRIGHT_CURRICULUM, UprightHorizonCurriculum

    objective = reward_contract(args.reward_mode)
    if not np.isfinite(args.assessment_seconds) or args.assessment_seconds <= 0:
        raise ValueError('Assessment duration must be finite and positive')
    short_upright = args.reward_mode in UPRIGHT_MODES
    curriculum_protocol = UPRIGHT_CURRICULUM if short_upright else CURRICULUM if args.reward_mode in STAGED_MODES else None

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, 'cuda:0', None)
    args.out.mkdir(exist_ok=False, parents=True)
    source = Path(__file__).resolve().parents[1]/'dummy.xml'
    prepare_plant(source, args.out/'dummy.xml')
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Spring-free torque plant contains a spring or position actuator')
    demonstrations = [Path('mujoco_rig/mimic/assets/stand_reference.npz'),
                      Path('logs/mimickit-steps/weight-shift-reference-01/weight_shift_reference.npz')]
    config = yaml.safe_load((args.mimickit/'data/agents/amp_humanoid_agent.yaml').read_text())
    config.update(steps_per_iter=64, iters_per_output=args.evaluation_interval, test_episodes=0,
                  task_reward_weight=.8, disc_reward_weight=.2, disc_epochs=1, disc_batch_size=8,
                  td_lambda=.98, normalizer_samples=100000000)
    if actor_epochs is not None:
        config['actor_epochs'] = actor_epochs
    config['model']['action_std'] = .1
    for optimizer in ('actor_optimizer', 'critic_optimizer', 'disc_optimizer'):
        config[optimizer]['grad_clip'] = 1.
    continuation = None
    if args.initialize_from:
        parent = args.initialize_from.parent
        protocol = json.loads((parent/'protocol.json').read_text())
        contract = json.loads((parent/'contract.json').read_text())
        source_report = json.loads((parent/'report.json').read_text())
        if (contract['observation_contract'] not in ('mimic_stationary_stand_torque_v1', CONTRACT)
                or contract['model_sha256'] != sha256(rig.path)
                or contract['reference_sha256'] != sha256(args.reference)
                or protocol['demonstrations'] != [dict(path=str(p), sha256=sha256(p)) for p in demonstrations]):
            raise ValueError('Continuation physics, reference or AMP demonstrations do not match')
        source_reward = contract.get('posture_reward')
        reward_changed = source_reward != objective
        if contract['observation_contract'] == CONTRACT and reward_changed:
            allowed = args.allow_reward_change and (
                (source_reward == POSTURE and objective == STAGED_POSTURE) or
                (source_reward == STAGED_POSTURE and objective == WORST_POSTURE) or
                (source_reward in (STAGED_POSTURE, WORST_POSTURE) and objective == UPRIGHT_IMITATION) or
                (source_reward == UPRIGHT_IMITATION and objective == WHOLE_POSE_IMITATION) or
                (source_reward == WHOLE_POSE_IMITATION and objective == LIMIT_AWARE_IMITATION) or
                (source_reward == LIMIT_AWARE_IMITATION and objective == LIMIT_OCCUPANCY_IMITATION) or
                (source_reward == WHOLE_POSE_IMITATION and objective == REFERENCE_IMITATION))
            if not allowed:
                raise ValueError('Reward transition requires explicit --allow-reward-change for a recognized transition')
        if contract['observation_contract'] != CONTRACT and args.reward_mode in STAGED_MODES and not args.allow_reward_change:
            raise ValueError('Legacy-to-staged reward transition requires --allow-reward-change')
        source_config = yaml.safe_load((parent/'agent.yaml').read_text())
        if source_config != config:
            raise ValueError('Continuation must preserve the source AMP training configuration')
        portable = torch.load(args.initialize_from, map_location='cpu', weights_only=True)
        training = torch.load(amp_sidecar(args.initialize_from), map_location='cpu', weights_only=True)
        validate_sidecar(portable, training)
        prior_seconds = source_report.get('cumulative_training_seconds', source_report['training_iteration_wall_seconds'])
        source_iteration = source_report['iterations']
        source_metrics = source_report['final']
        if source_report['final_checkpoint_sha256'] != sha256(args.initialize_from):
            # Admit a measured intermediate only when both actor and full AMP state
            # exactly match the best state recorded by the completed source run.
            source_iteration = source_report['best_iteration']
            if args.initialize_from.name != f'evaluation_{source_iteration:06d}.pt':
                raise ValueError('Source is neither the recorded final actor nor the measured best evaluation')
            for state, best_path in ((portable, parent/'best.pt'), (training, parent/'best.amp.pt')):
                best_state = torch.load(best_path, map_location='cpu', weights_only=True)
                if state.keys() != best_state.keys() or any(
                        not torch.equal(value, best_state[key]) for key, value in state.items()):
                    raise ValueError('Intermediate actor/AMP tensors differ from the recorded best state')
            curve = json.loads((parent/'learning_curve.json').read_text())
            source_metrics = next(row['metrics'] for row in curve if row['iteration'] == source_iteration)
            if source_metrics != source_report['best']:
                raise ValueError('Best checkpoint assessment provenance differs')
            updates = [json.loads(line) for line in (parent/'updates.jsonl').read_text().splitlines()]
            source_update = next(row for row in updates if row['iteration'] == source_iteration)
            prior_seconds -= source_report['training_iteration_wall_seconds']-source_update['training_iteration_wall_seconds']
        del portable, training
        continuation = dict(checkpoint=str(args.initialize_from.resolve()),
            checkpoint_sha256=sha256(args.initialize_from), amp_sha256=sha256(amp_sidecar(args.initialize_from)),
            prior_training_seconds=prior_seconds, source_iteration=source_iteration,
            source_assessment=source_metrics,
            source_contract=contract['observation_contract'], destination_contract=CONTRACT,
            posture_reward_changed=reward_changed, source_posture_reward=source_reward,
            reward_change_explicitly_enabled=args.allow_reward_change,
            source_curriculum_stage=(source_report.get('posture_curriculum') or {}).get('final_stage', 0)
                if not reward_changed else 0,
            scope='Actor/critic/AMP/normalization warm start; v1 maps 30 outputs to 33 with zero neck means; fresh optimizer/RNG stream')
    write_json(args.out/'request.json', dict(schema='spring_free_torque_stand_request_v1', pid=os.getpid(),
        seconds=args.seconds, envs=args.envs, evaluation_interval=args.evaluation_interval, reward_mode=args.reward_mode,
        episode_seconds=args.episode_seconds, assessment_seconds=args.assessment_seconds,
        initialization=continuation or 'Fresh actor, critic, discriminator and normalization; no pose-PD checkpoint reuse',
        long_hold='No additional long-hold suite for reference_v9; historical modes retain their conditional 600-second assessment',
        arena=None if comparison else 'Scenes/RL/Isaac3/MuJoCo/MimicStandArena.tscn',
        comparison=comparison, standalone_tests_authorized=False))
    write_json(args.out/'protocol.json', dict(model_sha256=sha256(rig.path), source_model_sha256=sha256(source),
        reference_sha256=sha256(args.reference), passive_joint_stiffness=0., standing_pose_servo=False,
        neck_position_servo=False, motor_control='normalized_motor_torque', damping_preserved=True,
        torque_scale_nm=rig.action_scale.tolist(), seed=args.seed, seconds=args.seconds, envs=args.envs,
        evaluation_interval=args.evaluation_interval, mimickit_revision=MIMICKIT_REVISION,
        episode_seconds=args.episode_seconds, assessment_seconds=args.assessment_seconds,
        demonstrations=[dict(path=str(p), sha256=sha256(p)) for p in demonstrations],
        demonstration_scope='Licensed coordinate/style examples admitted on original physics; not new-plant dynamic admission',
        reward=objective, reward_mode=args.reward_mode,
        posture_curriculum=curriculum_protocol,
        policy_actions=len(rig.actuators), neck_control='learned joint torque',
        time_budget='Synchronized training-update wall seconds excluding built-in evaluations',
        training_sources={name: sha256(Path(__file__).with_name(name)) for name in
                          ('train_torque_stand.py', 'torque_stand_task.py', 'stand_task.py')},
        continuation=continuation, config=config))
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    task = AMPTorqueStandTask(rig, args.reference, args.envs, 'cuda:0', seed=args.seed, reward_mode=args.reward_mode,
                             episode_seconds=args.episode_seconds,
                             demonstrations=demonstrations, demonstration_rig=Rig.load(source))
    native = TorqueStandTask(rig, args.reference, engine_factory=NativeDummyEngine, reward_mode=args.reward_mode,
                            episode_seconds=args.episode_seconds)
    curriculum_class = UprightHorizonCurriculum if short_upright else PostureCurriculum
    curriculum = curriculum_class(stage=continuation['source_curriculum_stage'] if continuation else 0) \
        if curriculum_protocol else None

    def apply_curriculum():
        if short_upright:
            task.set_upright_horizon_stage(curriculum.stage)
        elif curriculum:
            task.set_posture_stage(curriculum.stage)
            native.set_posture_stage(curriculum.stage)

    apply_curriculum()
    write_json(args.out/'contract.json', task.observation_contract())

    if comparison:
        # Environment allocation must not change the fresh network initialization.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    agent = build_agent(config, task, 'cuda:0', args.initialize_from)
    initial_state_sha256 = state_sha256(agent) if comparison else None
    if comparison:
        write_json(args.out/'initialization.json', dict(seed=args.seed,
            state_sha256=initial_state_sha256, iterations=0, samples=0,
            training_seconds=0., initialize_from=None,
            scope='Fresh actor, critic, AMP discriminator and normalization tensors'))
    write_json(args.out/'completion-state.json', dict(status='training_running', pid=os.getpid(),
        requested_training_seconds=args.seconds, continuation=continuation))
    started = time.monotonic()
    training_seconds = 0.
    curve, best_rank, best_metrics, best_iteration = [], None, None, 0
    short_curve = []
    gpu_stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(gpu_stop, args.out/'gpu-samples.jsonl'), daemon=True)
    sampler.start()

    def assess():
        nonlocal best_rank, best_metrics, best_iteration
        metrics = evaluate_standing(native, StandPolicy(agent).cpu().eval(), seconds=args.assessment_seconds)
        rank = (metrics['passed'], metrics['settled_cases'], metrics['balance_settled_cases'],
                sum(c['survived'] for c in metrics['cases']), min(c['survival_seconds'] for c in metrics['cases']),
                sum(c['survival_seconds'] for c in metrics['cases']))
        if best_rank is None or rank>best_rank:
            best_rank, best_metrics, best_iteration = rank, metrics, agent._iter
            agent.save(str(args.out/'best.pt'))
        curve.append(dict(iteration=agent._iter, metrics=metrics))
        write_json(args.out/'learning_curve.json', curve)
        print(json.dumps(dict(iteration=agent._iter, settled_cases=metrics['settled_cases'],
                             balance_settled_cases=metrics['balance_settled_cases'],
                             posture_stage=metrics['assessed_stage'],
                             survival=[round(c['survival_seconds'], 3) for c in metrics['cases']])), flush=True)
        return metrics

    def save_retention(metrics, retention=None):
        checkpoint = args.out/'retention.pt'
        agent.save(str(checkpoint))
        write_json(args.out/'retention.json', dict(iteration=agent._iter, metrics=metrics,
            checkpoint_sha256=sha256(checkpoint), amp_sha256=sha256(amp_sidecar(checkpoint)),
            posture_stage=curriculum.stage, retention=retention,
            scope='Source quiet retention; saved separately, never silently selected as the final candidate'))

    def assess_short():
        metrics = evaluate_standing(native, StandPolicy(agent).cpu().eval(), seconds=curriculum.horizon)
        short_curve.append(dict(iteration=agent._iter, horizon_seconds=curriculum.horizon, metrics=metrics))
        write_json(args.out/'short_learning_curve.json', short_curve)
        print(json.dumps(dict(iteration=agent._iter, short_horizon_seconds=curriculum.horizon,
            upright_stable_seconds=[round(c['final_stable_seconds'], 3) for c in metrics['cases']])), flush=True)
        return metrics

    try:
        agent.save(str(args.out/'initial.pt'))
        initial = assess()
        if curriculum:
            curriculum.set_baseline(initial)
            save_retention(initial)
            if short_upright:
                assess_short()
            write_json(args.out/'posture-curriculum.json', curriculum.report())
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        agent._init_train()
        stop_reason = 'training_update_time_budget'
        while True:
            budget_path = args.out/'budget-change.json'
            budget = json.loads(budget_path.read_text())['training_seconds'] if budget_path.exists() else args.seconds
            if not np.isfinite(budget) or budget<=0:
                raise ValueError('Training budget must be finite and positive')
            if training_seconds>=budget:
                break
            before = torch.cat([p.detach().flatten() for p in agent._model._actor_layers.parameters()]+
                               [p.detach().flatten() for p in agent._model._action_dist._mean_net.parameters()]).clone()
            torch.cuda.synchronize()
            tick = time.monotonic()
            info = agent._train_iter()
            torch.cuda.synchronize()
            training_seconds += time.monotonic()-tick
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            after = torch.cat([p.detach().flatten() for p in agent._model._actor_layers.parameters()]+
                              [p.detach().flatten() for p in agent._model._action_dist._mean_net.parameters()])
            row = dict(iteration=agent._iter, samples=agent._sample_count, training_iteration_wall_seconds=training_seconds,
                       actor_parameter_change_l2=float(torch.linalg.vector_norm(after-before)),
                       **{k:float(v) for k,v in info.items()},
                       **{k:float(v) for k,v in task._diagnostics.items()})
            if not all(np.isfinite(v) for v in row.values()) or any(not torch.isfinite(p).all() for p in agent.parameters()):
                raise FloatingPointError('Nonfinite direct-torque training update')
            with (args.out/'updates.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n')
            if agent._iter%args.evaluation_interval==0:
                agent.save(str(args.out/f'evaluation_{agent._iter:06d}.pt'))
                metrics = assess()
                if curriculum:
                    event = curriculum.observe(assess_short(), metrics, agent._iter) if short_upright \
                        else curriculum.observe(metrics, agent._iter)
                    if event['retention']['passed']:
                        save_retention(metrics, event['retention'])
                    apply_curriculum()
                    write_json(args.out/'posture-curriculum.json', curriculum.report())
                    print(json.dumps(dict(curriculum=event)), flush=True)
                    if event['stop_requested']:
                        stop_reason = 'quiet_retention_regression'
                        break
            elif agent._iter%8==0:
                agent.save(str(args.out/'latest.pt'))
        agent.save(str(args.out/'model.pt'))
        final = assess()
        final_short = assess_short() if short_upright else None
        long_hold = evaluate_standing(native, StandPolicy(agent).cpu().eval(), seconds=600.) \
            if final['passed'] and args.reward_mode != 'reference_v9' else None
        report = dict(schema='spring_free_torque_stand_report_v1', stop_reason=stop_reason,
            iterations=agent._iter, samples=agent._sample_count, training_iteration_wall_seconds=training_seconds,
            requested_training_seconds=args.seconds, continuation=continuation,
            episode_seconds=task.episode_seconds, assessment_seconds=args.assessment_seconds,
            cumulative_training_seconds=training_seconds+(continuation['prior_training_seconds'] if continuation else 0.),
            elapsed_seconds=time.monotonic()-started, initial=initial, final=final,
            best_iteration=best_iteration, best=best_metrics, final_long_hold=long_hold,
            final_short_upright=final_short,
            natural_movement_validated=False, quiet_long_hold_passed=bool(long_hold and long_hold['passed']),
            posture_reward=objective, reward_mode=args.reward_mode,
            posture_curriculum=curriculum.report() if curriculum else None,
            final_quiet_retention=curriculum.retention(final) if curriculum else None,
            policy_actions=len(rig.actuators),
            model_sha256=sha256(rig.path), final_checkpoint_sha256=sha256(args.out/'model.pt'))
        if comparison:
            report.update(comparison=True, initial_state_sha256=initial_state_sha256,
                          arena_selection_changed=False, historical_document_updated=False)
        write_json(args.out/'report.json', report)
        if comparison:
            write_json(args.out/'completion-state.json', dict(status='comparison_assessed',
                report_path=str(args.out/'report.json'), initial_state_sha256=initial_state_sha256,
                arena_selection_changed=False, historical_document_updated=False))
            print(json.dumps(dict(complete=True, comparison=True, training_seconds=training_seconds,
                                  final_settled=final['settled_cases'])), flush=True)
            return
        preview = export_preview(agent, native, args)
        if args.reward_mode == 'reference_v9':
            from datetime import datetime
            initial_survivors = sum(case['survived'] for case in initial['cases'])
            final_survivors = sum(case['survived'] for case in final['cases'])
            initialization = (f'checkpoint `{continuation["checkpoint_sha256"][:8]}` and its full AMP state'
                              if continuation else 'a fresh actor, critic, discriminator and normalization')
            entry = (
                f'\n## Additive reference imitation outcome ({datetime.now().astimezone().date().isoformat()})\n\n'
                f'Run `{args.out.name}` started from {initialization} '
                'with a fresh optimizer/RNG stream. The task calls upstream DeepMimic additive tracking, '
                'including neck pose, with a 20% AMP style prior. No passive springs, pose servo, '
                'limit penalties or early posture resets are used. '
                f'Training episodes last {task.episode_seconds:g} seconds and built-in checks last {args.assessment_seconds:g} seconds; '
                'physical falls still terminate episodes.\n\n'
                f'Completed {training_seconds:.3f} synchronized training-update seconds '
                f'({training_seconds/60:.3f} minutes), {agent._iter} updates and {agent._sample_count:,} transitions. '
                f'Trainer elapsed: {report["elapsed_seconds"]:.3f} seconds. '
                f'Paired initial/final one-minute survival: {initial_survivors}/8 to {final_survivors}/8; '
                f'balanced cases: {initial["balance_settled_cases"]}/8 to {final["balance_settled_cases"]}/8; '
                f'upright cases: {initial["settled_cases"]}/8 to {final["settled_cases"]}/8. '
                f'Final `{report["final_checkpoint_sha256"][:8]}` is exported in `{args.out.name}/arena-preview` '
                'for MimicStandTorqueArena. Primary StandArena selection is preserved. '
                'Per-body reference errors and physical failures are recorded in report.json; '
                'natural movement is not admitted. No extra hold suite, standalone tests, compilation or Godot run occurred.\n')
            with Path(__file__).with_name('AMP_RECOVERY.md').open('a', encoding='utf-8') as document:
                document.write(entry)
        write_json(args.out/'completion-state.json', dict(status='final_preview_ready', report=report, arena_preview=preview,
            next_action='Open MimicStandTorqueArena.tscn to inspect the exported final candidate; no compilation or Godot run was performed.'))
        print(json.dumps(dict(complete=True, training_seconds=training_seconds, final_settled=final['settled_cases'],
                             arena='MimicStandTorqueArena.tscn')), flush=True)
    finally:
        gpu_stop.set()
        sampler.join(timeout=6)


def complete_saved_window(run, snapshot, update):
    """Complete this trainer's assessment/export from its saved full AMP state."""
    from types import SimpleNamespace
    request = json.loads((run/'request.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    mimickit = Path('D:/Proyectos/Juegos/Tools/MimicKit')
    activate(mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .torque_stand_task import TorqueStandTask, evaluate_standing, reward_contract, STAGED_MODES, UPRIGHT_MODES
    torch.set_num_threads(1)
    mp_util.init(0, 1, 'cpu', None)
    reference = Path('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz')
    rig = Rig.load(run/'dummy.xml', policy_exclude=())
    mode = protocol.get('reward_mode', 'whole_body_v2')
    native = TorqueStandTask(rig, reference, engine_factory=NativeDummyEngine, reward_mode=mode)
    curriculum_path = run/'posture-curriculum.json'
    curriculum = json.loads(curriculum_path.read_text()) if mode in (*STAGED_MODES, *UPRIGHT_MODES) and curriculum_path.exists() else None
    if curriculum:
        if mode in UPRIGHT_MODES:
            native.set_upright_horizon_stage(curriculum['final_stage'])
        else:
            native.set_posture_stage(curriculum['final_stage'])
    config = yaml.safe_load((run/'agent.yaml').read_text())
    agent = PPOAgent(config, native, 'cpu')
    agent.load(str(snapshot))
    shutil.copy2(snapshot, run/'model.pt')
    shutil.copy2(snapshot.with_suffix('.amp.pt'), run/'model.amp.pt')
    final = evaluate_standing(native, StandPolicy(agent).cpu().eval())
    quiet_retention = None
    if curriculum:
        from .torque_stand_curriculum import PostureCurriculum, UprightHorizonCurriculum
        controller_class = UprightHorizonCurriculum if mode in UPRIGHT_MODES else PostureCurriculum
        controller = controller_class(stage=curriculum['final_stage'])
        controller.baseline = {int(key): value for key, value in curriculum['baseline'].items()}
        quiet_retention = controller.retention(final)
    curve = json.loads((run/'learning_curve.json').read_text())
    curve.append(dict(iteration=update['iteration'], metrics=final))
    write_json(run/'learning_curve.json', curve)
    long_hold = evaluate_standing(native, StandPolicy(agent).cpu().eval(), seconds=600.) if final['passed'] else None
    report = dict(schema='spring_free_torque_stand_report_v1', stop_reason='user_shortened_training_window',
        iterations=update['iteration'], samples=update['samples'],
        training_iteration_wall_seconds=update['training_iteration_wall_seconds'],
        initial=curve[0]['metrics'], final=final, final_long_hold=long_hold,
        natural_movement_validated=False, quiet_long_hold_passed=bool(long_hold and long_hold['passed']),
        posture_reward=reward_contract(mode), reward_mode=mode, posture_curriculum=curriculum,
        final_quiet_retention=quiet_retention,
        policy_actions=len(rig.actuators),
        model_sha256=sha256(rig.path), final_checkpoint_sha256=sha256(run/'model.pt'),
        checkpoint_scope='Saved actor/critic/normalization plus full AMP sidecar; no additional training during completion')
    write_json(run/'report.json', report)
    request.update(seconds=900., requested_arena='Scenes/RL/Isaac3/MuJoCo/MimicStandArena.tscn')
    protocol.update(seconds=900., budget_change='User reduced active run from 30 to 15 training minutes')
    write_json(run/'request.json', request)
    write_json(run/'protocol.json', protocol)
    preview = export_preview(agent, native, SimpleNamespace(out=run, reference=reference))
    write_json(run/'completion-state.json', dict(status='final_preview_ready', report=report, arena_preview=preview,
        next_action='Rebuild C# adapter and restart MimicStandArena.tscn; no compilation was run.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--envs', type=int, default=2048)
    parser.add_argument('--evaluation-interval', type=int, default=64)
    parser.add_argument('--actor-epochs', type=int,
                        help='Override actor PPO epochs; omission preserves the pinned agent configuration')
    parser.add_argument('--episode-seconds', type=float)
    parser.add_argument('--assessment-seconds', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=300990)
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--reward-mode', choices=('whole_body_v2', 'staged_v3', 'worst_v4', 'short_upright_v5', 'whole_pose_v6', 'limit_aware_v7', 'limit_occupancy_v8', 'reference_v9'), default='whole_body_v2')
    parser.add_argument('--allow-reward-change', action='store_true')
    parser.add_argument('--comparison', action='store_true',
                        help='Fresh comparison only; save native assessments without changing Arena or historical docs')
    args=parser.parse_args()
    try:
        train(args)
    except Exception:
        if args.out.exists():
            write_json(args.out/'completion-state.json', dict(status='failed', error=traceback.format_exc()))
        raise

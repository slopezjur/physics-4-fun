"""Fresh fixed-budget PPO/AMP exploration comparison using admitted Stand clips."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import torch
import yaml

from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION
from .train_stand_imitation import evaluate_actor, tensor_hash
from .train_torque_stand import build_agent, sample_gpu


def run(args, *, task_factory=None, source_paths=(), reward_variant=None):
    activate(args.mimickit)
    from util import mp_util
    from .torque_stand_task import AMPTorqueStandTask
    from .amp_recovery import validate_sidecar
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    teacher = json.loads((args.dataset/'report.json').read_text())
    corrections = json.loads((args.corrections/'report.json').read_text())
    if not teacher['passed'] or not corrections['passed']:
        raise ValueError('Physically admitted Stand clips required')
    model, reference = args.dataset/'dummy.xml', args.dataset/'stationary_reference.npz'
    if corrections['model_sha256'] != sha256(model) or corrections['reference_sha256'] != sha256(reference):
        raise ValueError('Correction physics/reference differs from teacher')
    records = teacher['reports']['native']['clips'] + corrections['reports']['native']['amp_clips']
    clips = [Path(r['path']) for r in records]
    protected_paths = [model, reference, reference.with_suffix('.json'), args.config,
                       args.dataset/'report.json', args.corrections/'report.json']
    protected_paths.extend(source_paths)
    for clip, record in zip(clips, records):
        if sha256(clip) != record['sha256']:
            raise ValueError('Admitted AMP clip hash mismatch')
        protected_paths.extend((clip, clip.with_suffix('.json')))
    protected = {p: sha256(p) for p in protected_paths}
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(args.dataset/name, args.out/name)
    shutil.copy2(Path(__file__), args.out/'runner.py')
    if source_paths:
        (args.out/'extra-sources').mkdir()
        for path in source_paths:
            shutil.copy2(path, args.out/'extra-sources'/path.name)
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    reference = args.out/'stationary_reference.npz'
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Plant must remain spring-free with direct motor torques')
    config = yaml.safe_load(args.config.read_text())
    config['model']['action_std'] = args.action_std
    if args.actor_learning_rate is not None:
        config['actor_optimizer']['learning_rate'] = args.actor_learning_rate
    critic_override = getattr(args, 'critic_learning_rate', None)
    if critic_override is not None:
        config['critic_optimizer']['learning_rate'] = critic_override
    actor_learning_rate = float(config['actor_optimizer']['learning_rate'])
    critic_learning_rate = float(config['critic_optimizer']['learning_rate'])
    if config['steps_per_iter'] != 64 or config['actor_epochs'] != 3 or config['iters_per_output'] != 64:
        raise ValueError('Comparison requires the documented PPO update/evaluation cadence')
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    factory = task_factory or AMPTorqueStandTask
    task = factory(rig, reference, 64, 'cuda:0', seed=args.seed,
        reward_mode='reference_v9', episode_seconds=60., demonstrations=clips, demonstration_rig=rig)
    task.mirror_demos = False
    # Reset the initialization RNG after environment construction in every run.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent = build_agent(config, task, 'cuda:0')
    initial_state = agent.state_dict()
    std_keys = [k for k in initial_state if 'logstd' in k]
    if not std_keys:
        raise ValueError('Fixed action distribution parameter not found')
    common_hash = tensor_hash({k: v for k, v in initial_state.items() if k not in std_keys})
    initialization = dict(seed=args.seed, initialize_from=None, fresh=True,
        full_tensor_sha256=tensor_hash(initial_state), shared_tensor_sha256=common_hash,
        excluded_fixed_logstd_keys=std_keys,
        fixed_logstd={k: initial_state[k].cpu().tolist() for k in std_keys},
        fixed_logstd_sha256=tensor_hash({k: initial_state[k] for k in std_keys}))
    if args.paired_with:
        paired = json.loads((args.paired_with/'initialization.json').read_text())
        if paired['shared_tensor_sha256'] != common_hash or paired['seed'] != args.seed:
            raise ValueError('Paired fresh network/normalizer initialization differs')
        initialization['paired_initialization_matches'] = True
        same_logstd = paired['fixed_logstd_sha256'] == initialization['fixed_logstd_sha256']
        if same_logstd and paired['full_tensor_sha256'] != initialization['full_tensor_sha256']:
            raise ValueError('Paired full initial state differs at the same action_std')
        initialization['paired_full_initialization_matches'] = bool(same_logstd)
        paired_config = yaml.safe_load((args.paired_with/'agent.yaml').read_text())
        paired_config['model']['action_std'] = args.action_std
        if args.actor_learning_rate is not None:
            paired_config['actor_optimizer']['learning_rate'] = args.actor_learning_rate
        if critic_override is not None:
            paired_config['critic_optimizer']['learning_rate'] = critic_override
        if paired_config != config:
            raise ValueError('Paired configs differ beyond action_std and explicit actor/critic learning rates')
        paired_protocol = json.loads((args.paired_with/'protocol.json').read_text())
        if paired_protocol['demonstrations'] != records or any(
                paired_protocol['inputs'].get(str(p)) != h for p, h in protected.items()):
            raise ValueError('Paired plant/reference/config source or demonstrations differ')
    write_json(args.out/'initialization.json', initialization)
    agent.save(str(args.out/'initial.pt'))
    write_json(args.out/'contract.json', task.observation_contract())
    write_json(args.out/'protocol.json', dict(seed=args.seed, action_std=args.action_std,
        actor_learning_rate=actor_learning_rate, paired_with=str(args.paired_with) if args.paired_with else None,
        critic_learning_rate=critic_learning_rate,
        seconds=args.seconds, envs=64, rollout_steps=64, transitions_per_update=4096,
        actor_epochs=3, evaluation_interval=64, training_episode_seconds=60., assessment_seconds=60.,
        config=config, reward_mode='reference_v9', reward_variant=reward_variant,
        mirror_demos=False, amp_input_width=364,
        initialization='Fresh actor, critic, discriminator and normalizers; no fitted moments or checkpoints',
        demonstrations=records, inputs={str(p): h for p, h in protected.items()},
        timing='Synchronized PPO training-update seconds excluding setup and deterministic assessments',
        source_sha256=sha256(Path(__file__)), mimickit_revision=MIMICKIT_REVISION,
        arena_changed=False, comparison_scope=(
            'Only diagnostic reward weight differs; fixed PPO configuration, physical reference and AMP clips'
            if reward_variant is not None else
            'Only action_std and explicitly overridden actor/critic learning rates may differ between paired configs; same physical reference and AMP clips')))
    demo = task.fetch_disc_obs_demo(256)
    if demo.shape != (256, 364) or not torch.isfinite(demo).all():
        raise ValueError('AMP full hinge-history contract failed')
    assessments = []

    def assess(label, gpu=False):
        out = args.out/label
        out.mkdir()
        actor = StandPolicy(agent)
        result = evaluate_actor(rig, reference, actor, 'gpu' if gpu else 'native', 60., out)
        assessments.append(dict(label=label, iteration=agent._iter, metrics=result))
        write_json(args.out/'assessments.json', assessments)
        print(dict(assessment=label, iteration=agent._iter, standing_passes=result['standing_passes'],
            mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in result['cases']]))), flush=True)
        return result

    agent._init_train()
    initial = assess('initial-native')
    # Assessments use separate engines; synchronize the rollout RNG for both arms.
    torch.manual_seed(args.seed+701)
    np.random.seed(args.seed+701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    elapsed = 0.
    wall_start = time.perf_counter()
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, args.out/'gpu-samples.jsonl'), daemon=True)
    sampler.start()
    try:
        while elapsed < args.seconds:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            info = agent._train_iter()
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-tick
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
                **{k: float(v) for k, v in info.items()}, **{k: float(v) for k, v in task._diagnostics.items()})
            if not all(np.isfinite(v) for v in row.values()) or any(not torch.isfinite(p).all() for p in agent.parameters()):
                raise FloatingPointError('Nonfinite PPO update')
            with (args.out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            write_json(args.out/'progress.json', row)
            if agent._iter % 16 == 0:
                print(dict(action_std=args.action_std, iteration=agent._iter,
                    actor_learning_rate=actor_learning_rate,
                    critic_learning_rate=critic_learning_rate,
                    transitions=agent._sample_count, training_seconds=elapsed), flush=True)
            if agent._iter % 64 == 0:
                assess(f'evaluation-{agent._iter:06d}-native')
        agent.save(str(args.out/'model.pt'))
        validate_sidecar(torch.load(args.out/'model.pt', weights_only=True),
                         torch.load(args.out/'model.amp.pt', weights_only=True))
        final_native = assess('final-native')
        final_gpu = assess('final-gpu', gpu=True)
    finally:
        stop.set()
        sampler.join(timeout=6.)
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError('Protected training inputs changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        action_std=args.action_std, actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        seed=args.seed, requested_training_seconds=args.seconds,
        training_seconds=elapsed, wall_seconds=time.perf_counter()-wall_start,
        ppo_updates=agent._iter, transitions=agent._sample_count,
        initial=initial, final_native=final_native, final_gpu=final_gpu,
        shared_initial_tensor_sha256=common_hash, model_sha256=sha256(args.out/'model.pt'),
        protected_inputs_unchanged=True, arena_changed=False,
        selection='Final budget checkpoint; no best-checkpoint selection'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-std', type=float, required=True, choices=(.1, .025, .01, .0025))
    parser.add_argument('--actor-learning-rate', type=float)
    parser.add_argument('--critic-learning-rate', type=float)
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--seed', type=int, default=301010)
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--corrections', type=Path, default=Path('logs/mimickit-training/stand-corrections-20261001-01'))
    parser.add_argument('--config', type=Path, default=Path('logs/mimickit-training/stand-correction-pilot-20261001-01/agent.yaml'))
    parser.add_argument('--paired-with', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('Positive finite training budget required')
    if args.actor_learning_rate is not None and (not np.isfinite(args.actor_learning_rate) or args.actor_learning_rate <= 0):
        parser.error('Positive finite actor learning rate required')
    if args.critic_learning_rate is not None and (not np.isfinite(args.critic_learning_rate) or args.critic_learning_rate <= 0):
        parser.error('Positive finite critic learning rate required')
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

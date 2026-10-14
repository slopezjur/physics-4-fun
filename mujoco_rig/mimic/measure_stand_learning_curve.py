"""Bounded fresh target-PD learning curve with isolated checkpoint assessments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from . import replicate_stand_representations as screen
from .audit_stand_failures import digest, read, save
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


PRIOR = Path('logs/mimickit-training/stand-representations-replication-20261003-01')
SEED = 301017
CHECKPOINT_SECONDS = (180., 360., 600.)
BACKENDS = ('native', 'gpu')


def tree_hash(value):
    """Fingerprint nested optimizer/RNG evidence independently of serialization."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            result.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict')
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(str((type(item).__name__, len(item))).encode())
            for entry in item:
                visit(entry)
        else:
            result.update(repr(item).encode())
        result.update(b'\0')

    visit(value)
    return result.hexdigest()


def rng_state(task):
    state = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
        numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]),
        python=random.getstate(), demonstrations=task._demo_rng.get_state(), environment=task.rng.get_state())


def restore_rng(task, state):
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    kind, values, position, gaussian, cached = state['numpy']
    np.random.set_state((kind, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    random.setstate(state['python'])
    task._demo_rng.set_state(state['demonstrations'])
    task.rng.set_state(state['environment'])


def training_fingerprint(agent, task):
    q, v = screen.read_state(task, 'gpu')
    control = task.engine.policy_control
    environment = dict(q=torch.from_numpy(q), v=torch.from_numpy(v), pending=control.pending,
        active=control.active, ctrl=control.ctrl, cursor=control.cursor, steps=task.steps,
        offset=task.offset, current_observation=getattr(agent, '_curr_obs', None))
    optimizers = {name: getattr(agent, name)._optimizer.state_dict() for name in
        ('_actor_optimizer', '_critic_optimizer', '_disc_optimizer')}
    return dict(learner=tensor_hash(agent.state_dict()), optimizers=tree_hash(optimizers),
        environment=tree_hash(environment), training_mode=agent.training)


def assess(agent, task, rig, reference, out, threshold, elapsed):
    """Use actor copies and separate worlds; restore all training RNG streams."""
    from .amp_recovery import validate_sidecar
    out.mkdir()
    agent.save(str(out/'model.pt'))
    validate_sidecar(torch.load(out/'model.pt', weights_only=True),
        torch.load(out/'model.amp.pt', weights_only=True))
    state = rng_state(task)
    before = training_fingerprint(agent, task)
    started = time.perf_counter()
    try:
        results = {backend: screen.evaluate(rig, reference, 'target_pd', backend,
            StandPolicy(agent), out/backend) for backend in BACKENDS}
    finally:
        restore_rng(task, state)
    after = training_fingerprint(agent, task)
    restored = rng_state(task)
    torch.save(dict(before=state, restored=restored), out/'rng.pt')
    guard = dict(before=before, after=after, rng_before=tree_hash(state), rng_after=tree_hash(restored))
    save(out/'isolation.json', guard)
    assert before == after and guard['rng_before'] == guard['rng_after']
    result = dict(requested_training_seconds=threshold, training_seconds=elapsed,
        overshoot_seconds=elapsed-threshold, ppo_updates=agent._iter,
        transitions=agent._sample_count, assessment_seconds=time.perf_counter()-started,
        checkpoint=str(out/'model.pt'), full_tensor_hash=before['learner'], results=results,
        evaluation_isolated=True)
    save(out/'report.json', result)
    return result


def continuation(candidate, baseline, dt):
    result = {}
    for backend in BACKENDS:
        current, previous = candidate['results'][backend], baseline['results'][backend]
        gain = current['standing_passes']-previous['standing_passes']
        changes = {key: current[key]-previous[key] for key in
            ('minimum_survival_seconds', 'mean_survival_seconds', 'mean_longest_valid_seconds')}
        result[backend] = dict(stand_pass_gain=gain, changes_seconds=changes,
            continue_training=bool(gain > 0 or (gain == 0 and all(v >= dt-1e-12 for v in changes.values()))))
    return dict(backends=result, continue_training=all(v['continue_training'] for v in result.values()))


def protect(out):
    prior, verification = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    protected = {**prior['inputs'], **verification['output_hashes']}
    protected[str(PRIOR/'final-verification.json')] = digest(PRIOR/'final-verification.json')
    paths = [Path(__file__), Path(__file__).with_name('verify_stand_learning_curve.py')]
    for source in prior['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        expected = digest(path)
        assert str(path) not in protected or protected[str(path)] == expected
        protected[str(path)] = expected
    screen.check_protected(protected, [])
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .representation_task import AMPRepresentationTask
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    protected, sources = protect(args.out)
    prior = read(PRIOR/'protocol.json')
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(PRIOR/name, args.out/name)
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    assert len(rig.actuators) == 33 and not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    assert digest(rig.path) == prior['model_sha256'] and digest(reference) == prior['reference_sha256']
    records, config = prior['demonstrations'], prior['config']
    assert config == yaml.safe_load(screen.CONFIG.read_text())
    for record in records:
        assert digest(Path(record['path'])) == record['sha256']
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, seed=SEED,
        sampling_seed=SEED+701, initialize_from=None, config=config,
        model_sha256=digest(rig.path), reference_sha256=digest(reference), demonstrations=records,
        checkpoint_seconds=CHECKPOINT_SECONDS, maximum_training_seconds=600.,
        evaluation_seconds=12., training_episode_seconds=60., environments=64,
        rollout_steps=64, transitions_per_update=4096, observation_width=374,
        representation='target_pd', assessment_rng_restored=True,
        stop_rule='16/16 Stand at any assessment; at 360 require improvement on each backend over 180; otherwise stop',
        continuation_metrics=['minimum_survival_seconds', 'mean_survival_seconds', 'mean_longest_valid_seconds'],
        continuation_metric_minimum_gain_seconds=rig.model.opt.timestep*rig.decimation,
        maximum_assessment_cases=64, selection='Last planned checkpoint reached',
        springs=0., arena_changed=False, perturbations=False))
    task = AMPRepresentationTask(rig, reference, 64, 'cuda:0', representation='target_pd',
        reward_mode='reference_v9', episode_seconds=60., seed=SEED,
        demonstrations=[Path(record['path']) for record in records], demonstration_rig=rig)
    task.mirror_demos = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = build_agent(config, task, 'cuda:0')
    assert task.get_obs_space().shape == (374,) and task.get_disc_obs_space().shape == (364,)
    assert int(agent._obs_norm._count) == 0 and int(agent._a_norm._count) == 0
    assert tensor_hash(agent.state_dict()) != read(PRIOR/'report.json')['arms']['target_pd']['initial_full_tensor_hash']
    save(args.out/'contract.json', task.observation_contract())
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    assessments = [assess(agent, task, rig, reference, args.out/'initial', 0., 0.)]
    elapsed, next_output, decision = 0., 30., None
    reason = 'initial_16_of_16' if all(assessments[0]['results'][b]['standing_passes'] == 8 for b in BACKENDS) else None
    if reason is None:
        agent._init_train()
        torch.manual_seed(SEED+701)
        np.random.seed(SEED+701)
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        for threshold in CHECKPOINT_SECONDS:
            while elapsed < threshold:
                torch.cuda.synchronize()
                started = time.perf_counter()
                info = agent._train_iter()
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-started
                agent._sample_count = agent._update_sample_count()
                agent._iter += 1
                row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
                    **{key: float(value) for key, value in info.items()})
                assert all(np.isfinite(v) for v in row.values())
                assert all(torch.isfinite(value).all() for value in agent.state_dict().values())
                assert torch.isfinite(agent._curr_obs).all()
                with (args.out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                save(args.out/'progress.json', row)
                if elapsed >= next_output:
                    print(dict(training_seconds=elapsed, updates=agent._iter, transitions=agent._sample_count), flush=True)
                    next_output += 30.
            result = assess(agent, task, rig, reference, args.out/f'seconds-{int(threshold)}', threshold, elapsed)
            assessments.append(result)
            save(args.out/'learning-curve.json', assessments)
            if all(result['results'][b]['standing_passes'] == 8 for b in BACKENDS):
                reason = 'stand_16_of_16'
                break
            if threshold == 360.:
                decision = continuation(result, assessments[1], task.dt)
                save(args.out/'continuation-decision.json', decision)
                if not decision['continue_training']:
                    reason = 'plateau_or_mixed_at_360'
                    break
        reason = reason or 'maximum_budget_reached'
    save(args.out/'learning-curve.json', assessments)
    last = assessments[-1]
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), assessments=assessments,
        training_seconds=elapsed, ppo_updates=agent._iter, transitions=agent._sample_count,
        final_checkpoint=last['checkpoint'], stop_reason=reason, continuation_decision=decision,
        full_learned_stand=reason == 'stand_16_of_16', sustained_stand_verified=False,
        perturbations_ready=False, arena_changed=False, total_assessment_cases=16*len(assessments))
    screen.check_protected(protected, sources)
    save(args.out/'report.json', report)
    save(args.out/'completion-verification.json', dict(protected_inputs=len(protected), sources=len(sources),
        inputs_and_sources_unchanged=True,
        output_hashes={str(p): digest(p) for p in sorted(args.out.rglob('*')) if p.is_file()}))
    print(dict(completed=True, stop_reason=reason, training_seconds=elapsed), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        if arguments.out.is_dir():
            save(arguments.out/'failure.json', dict(error=traceback.format_exc()))
        raise

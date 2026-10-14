"""Bounded warm-start comparison on the admitted 374-observation Stand plant."""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .compare_stand_actor_rate import build_task_type, REWARD_COLUMNS
from .evaluate_stand_challenges import evaluate_challenge, verify_challenge
from .evaluate_stand_sustained import evaluate_duration, verify_duration
from .measure_stand_learning_curve import rng_state, restore_rng, training_fingerprint, tree_hash
from .policy import StandPolicy
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal
from .train_torque_stand import build_agent


BACKENDS = ('native', 'gpu')
OPTIMIZERS = ('_actor_optimizer', '_critic_optimizer', '_disc_optimizer')


def read_settings(path):
    settings = read(path)
    # A changed scientific protocol needs a successor, not an unnoticed resweep.
    expected = dict(schema='stand_to_perturb_pilot_v1', seeds=[301027, 301028], envs=64,
        updates=300, maximum_training_seconds_per_arm=600., episode_seconds=12.,
        quiet_retention_seconds=120., training_force_n=60., pulse_onset_seconds=5.,
        pulse_duration_seconds=.1, assessment_forces_n=[30, 60, 90])
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f'Pilot v1 requires {key}={value!r}')
    return settings


def scenarios(settings):
    return [dict(name=f'{force}N-{name}', body='Chest', pulses=[dict(
        onset_seconds=settings['pulse_onset_seconds'], duration_seconds=settings['pulse_duration_seconds'],
        force_world_n=[sign * force, 0, 0])])
        for force in settings['assessment_forces_n'] for name, sign in (('xp', 1), ('xm', -1))]


def build_task(settings):
    class ContinuationTask(build_task_type()):
        def __init__(self, *args, arm, **kwargs):
            self.arm, self.force_records = arm, []
            super().__init__(*args, **kwargs)
            self.push_body = self.rig.model.body('Chest').id
            self.start = int(np.ceil(settings['pulse_onset_seconds'] / self.dt))
            self.width = int(np.ceil(settings['pulse_duration_seconds'] / self.dt))
            # Two copies of all eight poses per sign; the other 32 worlds stay quiet.
            self.direction = torch.zeros((64, 3), device=self.device)
            self.direction[:16, 0] = 1.
            self.direction[16:32, 0] = -1.

        def step(self, action):
            active = (self.steps >= self.start) & (self.steps < self.start + self.width)
            strength = settings['training_force_n'] if self.arm == 'perturb' else 0.
            self.engine.set_external_force(self.push_body, self.direction * active[:, None] * strength)
            actual = self.engine._sim_state.body_force[0][:, self.push_body-1, :3]
            self.force_records.append(torch.cat((self.steps[:, None].float(), actual), dim=1).detach().clone())
            return super().step(action)

    return ContinuationTask


def verify_force(values, settings, arm, dt):
    assert values.shape == (64, 64, 4) and np.isfinite(values).all()
    steps = values[:, :, 0]
    assert np.array_equal(steps, np.rint(steps)) and (steps >= 0).all()
    start = int(np.ceil(settings['pulse_onset_seconds'] / dt))
    width = int(np.ceil(settings['pulse_duration_seconds'] / dt))
    active = (steps >= start) & (steps < start + width)
    direction = np.zeros((64, 3), dtype=np.float32)
    direction[:16, 0], direction[16:32, 0] = 1., -1.
    strength = settings['training_force_n'] if arm == 'perturb' else 0.
    expected = direction[None] * active[:, :, None] * np.float32(strength)
    assert np.array_equal(values[:, :, 1:], expected)
    return int(np.count_nonzero(np.any(expected != 0, axis=2)))


def assess(agent, task, rig, reference, out, settings):
    """Save immutable actor/full-state pairs and assess copies in separate worlds."""
    from .amp_recovery import validate_sidecar
    out.mkdir()
    shutil.copy2(reference, out/'stationary_reference.npz')  # Saved-trace verifier input.
    agent.save(str(out/'model.pt'))
    full = torch.load(out/'model.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(out/'model.pt', map_location='cpu', weights_only=True), full)
    before, state = training_fingerprint(agent, task), rng_state(task)
    assert tensor_hash(full) == before['learner']
    actor_hash = tensor_hash(StandPolicy(agent).state_dict())
    quiet, pushes = {}, {}
    try:
        for backend in BACKENDS:
            folder = out/f'quiet-{backend}'
            quiet[backend] = evaluate_duration(rig, reference, 'target_pd', backend,
                StandPolicy(agent), folder, seconds=settings['quiet_retention_seconds'])
            verify_duration(folder, settings['quiet_retention_seconds'])
            assert quiet[backend]['actor_hash'] == actor_hash
        for scenario in scenarios(settings):
            pushes[scenario['name']] = {}
            for backend in BACKENDS:
                folder = out/f"{scenario['name']}-{backend}"
                result = evaluate_challenge(rig, reference, 'target_pd', backend,
                    StandPolicy(agent), folder, seconds=settings['episode_seconds'], scenario=scenario)
                verify_challenge(folder, settings['episode_seconds'], scenario)
                assert result['actor_hash'] == actor_hash
                pushes[scenario['name']][backend] = result
            save(out/'push-progress.json', pushes)
    finally:
        restore_rng(task, state)
    after, restored = training_fingerprint(agent, task), rng_state(task)
    isolation = dict(before=before, after=after, rng_before=tree_hash(state), rng_after=tree_hash(restored))
    save(out/'isolation.json', isolation)
    torch.save(dict(before=state, restored=restored), out/'rng.pt')
    assert before == after and isolation['rng_before'] == isolation['rng_after']
    result = dict(checkpoint=str(out/'model.pt'), full_tensor_hash=tensor_hash(full), actor_hash=actor_hash,
        updates=agent._iter, transitions=agent._sample_count, quiet=quiet, pushes=pushes,
        full_interval_stand=all(r['standing_passes'] == 8 for r in quiet.values()), cases=112)
    save(out/'report.json', result)
    return result


def admit_source(settings, seed, inputs):
    source = settings['sources'][str(seed)]
    root, admission = Path(source['root']), Path(source['admission'])
    checkpoint = root/'updates-1200/model.pt'
    specification = read(admission/'protocol.json')['specification']
    protected = {str(Path(p).resolve()): value for p, value in inputs.items()}
    for path in (root/'protocol.json', checkpoint, checkpoint.with_suffix('.amp.pt'),
                 admission/'protocol.json', admission/'report.json', admission/'final-verification.json'):
        assert protected[str(path.resolve())] == digest(path)
    assert Path(specification['checkpoint']).resolve() == checkpoint.resolve()
    assert specification['seconds'] == 120.
    for result in read(admission/'report.json')['results']['quiet'].values():
        assert result['standing_passes'] == 8 and len(result['cases']) == 8
        assert all(c['standing_pass'] and c['first_departure_seconds'] is None
                   and c['survival_seconds'] >= 120. for c in result['cases'])
    source_protocol = read(root/'protocol.json')
    assert source_protocol['seed'] == seed and source_protocol['arm'] == 'quiet'
    return checkpoint, source_protocol


def run(args):
    if not __debug__:
        raise RuntimeError('Evidence checks require Python assertions enabled')
    settings = read_settings(args.protocol)
    assert args.seed in settings['seeds']
    activate(args.mimickit)
    from .amp_recovery import validate_sidecar
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    package = Path(__file__).parent
    sources_to_copy = [Path(__file__), package/'run_perturb_pilot.py',
        package/'scripts/train-perturb-pilot.ps1', package/'scripts/perturb-pilot-config.ps1']
    inputs, sources = protect(args.prior, args.out, sources_to_copy, extra_inputs=[args.protocol])
    checkpoint, source = admit_source(settings, args.seed, inputs)
    config = copy.deepcopy(source['config'])
    assert config['steps_per_iter'] == 64 and config['actor_epochs'] == 3
    assert config['actor_optimizer']['learning_rate'] == 3e-4
    assert config['actor_batch_size'] == 4 and config['critic_batch_size'] == 2 and config['critic_epochs'] == 2
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    assert digest(rig.path) == source['model_sha256'] and digest(reference) == source['reference_sha256']
    protocol = dict(inputs=inputs, sources=sources, config=config, settings=settings, seed=args.seed, arm=args.arm,
        initialize_from=str(checkpoint), sampling_seed=args.seed+1701, optimizer_state='fresh in both arms',
        demonstrations=source['demonstrations'], model_sha256=source['model_sha256'],
        reference_sha256=source['reference_sha256'], control_timestep=rig.model.opt.timestep*rig.decimation,
        comparison='Only force exposure differs within each matched continuation pair',
        maximum_cases=224, maximum_updates=settings['updates'])
    save(args.out/'protocol.json', protocol)
    task = build_task(settings)(rig, reference, settings['envs'], 'cuda:0', arm=args.arm,
        representation='target_pd', reward_mode='reference_v9', episode_seconds=settings['episode_seconds'],
        seed=args.seed, demonstrations=[Path(r['path']) for r in source['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent = build_agent(config, task, 'cuda:0')
    full = torch.load(checkpoint.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(checkpoint, map_location='cpu', weights_only=True), full)
    agent.load_state_dict(full)
    agent._sync_optimizer()
    assert tensor_hash(agent.state_dict()) == tensor_hash(full)
    for name in OPTIMIZERS:
        optimizer = getattr(agent, name)
        assert optimizer.get_steps() == 0 and not optimizer._optimizer.state_dict()['state']
    agent._init_train()
    save(args.out/'contract.json', task.observation_contract())
    initial = assess(agent, task, rig, reference, args.out/'initial', settings)
    if not initial['full_interval_stand']:
        raise RuntimeError('Source failed repeated 120-second admission; no training allowed')
    random.seed(args.seed+1701)
    torch.manual_seed(args.seed+1701)
    np.random.seed(args.seed+1701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    save(args.out/'training-start.json', dict(fingerprint=training_fingerprint(agent, task), rng=tree_hash(rng_state(task))))
    elapsed, rewards, forces, forced = 0., [], [], 0
    while agent._iter < settings['updates']:
        if elapsed >= settings['maximum_training_seconds_per_arm']:
            raise RuntimeError('Training cap reached; incomplete arm cannot be compared or promoted')
        task.reward_records.clear()
        task.force_records.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        info = agent._train_iter()
        torch.cuda.synchronize()
        elapsed += time.perf_counter()-started
        reward = torch.stack(task.reward_records).cpu().numpy()
        force = torch.stack(task.force_records).cpu().numpy()
        assert reward.shape == (64, 6) and np.isfinite(reward).all()
        assert np.array_equal(reward[:, 0], reward[:, 1]) and (reward[:, 2] == 1).all()
        forced += verify_force(force, settings, args.arm, task.dt)
        rewards.append(reward)
        forces.append(force)
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
            forced_world_control_intervals=forced,
            actor_optimizer_steps=agent._actor_optimizer.get_steps(),
            critic_optimizer_steps=agent._critic_optimizer.get_steps(), **{k: float(v) for k, v in info.items()})
        assert all(np.isfinite(v) for v in row.values())
        assert all(torch.isfinite(v).all() for v in agent.state_dict().values())
        assert row['transitions'] == 4096*agent._iter
        assert row['actor_optimizer_steps'] == 48*agent._iter and row['critic_optimizer_steps'] == 64*agent._iter
        with (args.out/'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        save(args.out/'progress.json', row)
        if agent._iter % 100 == 0:
            agent.save(str(args.out/f'recovery-{agent._iter}.pt'))
            print(dict(seed=args.seed, arm=args.arm, **row), flush=True)
    np.savez_compressed(args.out/'training-records.npz', reward=np.stack(rewards), force=np.stack(forces),
        reward_columns=np.asarray(REWARD_COLUMNS))
    if elapsed > settings['maximum_training_seconds_per_arm']:
        raise RuntimeError('Last update exceeded the training cap; preserve incomplete arm')
    assert (forced > 0) == (args.arm == 'perturb')
    final = assess(agent, task, rig, reference, args.out/'final', settings)
    report = dict(seed=args.seed, arm=args.arm, initial=initial, final=final,
        updates=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
        forced_world_control_intervals=forced, arena_changed=False)
    save(args.out/'report.json', report)
    seal(args.out, inputs, sources, verified=True, warm_start=True, fresh_optimizers=True,
        training_updates=agent._iter, cases=224, reward_identity_verified=True,
        force_schedule_verified=True, checkpoint_pairs_verified=True, evaluation_isolated=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(301027, 301028), required=True)
    parser.add_argument('--arm', choices=('quiet', 'perturb'), required=True)
    parser.add_argument('--mimickit', type=Path, required=True)
    arguments = parser.parse_args()
    existed = arguments.out.exists()
    try:
        run(arguments)
    except Exception:
        if not existed and arguments.out.is_dir():
            save(arguments.out/'failure.json', dict(error=traceback.format_exc()))
        raise

"""One fixed behavior-cloning pilot; CPU evaluation remains the acceptance authority."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import torch

import benchmark_perturb as benchmark
import probe_recovery_feasibility as probe
from observation_contract import FOUNDATION, LEGACY, checkpoint_version, observation_size
from ppo import ActorCritic


def widen_checkpoint(source):
    """Retain actor/std exactly; version the artifact so PPO resets its stale critic."""
    if checkpoint_version(source) != LEGACY:
        raise ValueError('This pilot expects the frozen legacy accepted actor')
    width = observation_size(source['num_actions'], FOUNDATION)
    state = copy.deepcopy(source['model'])
    for key in ('actor.0.weight', 'critic.0.weight'):
        old = state[key]
        new = torch.zeros(old.shape[0], width, dtype=old.dtype)
        new[:, :old.shape[1]] = old
        state[key] = new
    net = ActorCritic(width, source['num_actions']).eval()
    net.load_state_dict(state)
    return net


def training_groups(dataset, intervention_steps):
    train = [x for x in dataset['trajectories'].values() if x['case']['split'] == 'train']
    validation = [x for x in dataset['trajectories'].values() if x['case']['split'] == 'validation']
    if not train or not validation:
        raise ValueError('Whole-trajectory train and validation sets must both be nonempty')
    if {x['case']['trial'] for x in train} & {x['case']['trial'] for x in validation}:
        raise ValueError('Trajectory leakage')
    join = lambda rows, part: tuple(torch.cat([x[k][part] for x in rows]) for k in ('obs', 'mean'))
    groups = dict(demo=join(train, slice(None, intervention_steps)),
                  validation=join(validation, slice(None, intervention_steps)),
                  quiet=tuple(dataset['reference']['quiet'][k] for k in ('obs', 'mean')))
    # Successful handback states are retention examples, never additional oracle labels.
    groups['impact'] = tuple(torch.cat([dataset['reference']['impact'][k]] +
        [x[k][intervention_steps:] for x in train]) for k in ('obs', 'mean'))
    for name, (obs, target) in groups.items():
        if (obs.ndim != 2 or target.ndim != 2 or len(obs) != len(target) or not len(obs)
                or not torch.isfinite(obs).all() or not torch.isfinite(target).all()):
            raise ValueError(f'Invalid group {name}')
    return groups


@torch.no_grad()
def measure(net, groups):
    return {name: float((net.actor(obs)-mean).square().mean()) for name, (obs, mean) in groups.items()}


def choose_checkpoint(measurements, config):
    eligible = [x for x in measurements if x['quiet'] <= config['quiet_mse_limit']
                and x['impact'] <= config['impact_mse_limit']]
    chosen = min(eligible or measurements, key=lambda x: (x['validation'], x['update']))
    return chosen, bool(eligible)


def verify_artifact(folder, filename):
    hashes = json.loads((folder/'artifacts.json').read_text())
    if benchmark.digest(folder/filename) != hashes[filename]:
        raise ValueError(f'Experiment artifact changed: {filename}')


def train(folder):
    if (folder/'selection-lock.json').exists():
        raise ValueError('This experiment already has a locked candidate')
    verify_artifact(folder, 'protocol.json')
    verify_artifact(folder, 'dataset.pt')
    protocol = json.loads((folder/'protocol.json').read_text())
    config = protocol['training']
    if benchmark.digest(protocol['checkpoint']) != protocol['checkpoint_sha256']:
        raise ValueError('Accepted actor changed')
    torch.manual_seed(config['seed'])
    source = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=False)
    dataset = torch.load(folder/'dataset.pt', map_location='cpu', weights_only=False)
    groups = training_groups(dataset, protocol['search']['optimize_steps'])
    net = widen_checkpoint(source)
    initial_state = copy.deepcopy(net.state_dict())
    initial = measure(net, groups)
    benchmark.write_json(folder/'initial-fit.json', initial)
    optimizer = torch.optim.Adam(net.actor.parameters(), lr=config['learning_rate'])
    generator = torch.Generator().manual_seed(config['seed'])
    measurements = []
    for update in range(1, config['updates']+1):
        loss = torch.zeros(())
        for name, weight in config['weights'].items():
            obs, target = groups[name]
            indices = torch.randint(len(obs), (config['batch_per_group'],), generator=generator)
            loss = loss + weight*(net.actor(obs[indices])-target[indices]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite demonstration loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.actor.parameters(), 1.)
        optimizer.step()
        if update in config['checkpoints']:
            metrics = dict(update=update, **measure(net, groups))
            measurements.append(metrics)
            checkpoint = dict(model=copy.deepcopy(net.state_dict()), num_obs=observation_size(source['num_actions'], FOUNDATION),
                num_actions=source['num_actions'], observation_version=FOUNDATION, task='perturb',
                training_version='offline_recovery_bc_v1', reward_version=source['reward_version'],
                ball_speed=6., critic_warmup_remaining=50, demonstration_update=update,
                demonstration_protocol_sha256=benchmark.digest(folder/'protocol.json'),
                source_checkpoint_sha256=protocol['checkpoint_sha256'])
            torch.save(checkpoint, folder/f'model_{update}.pt')
            benchmark.write_json(folder/'fit.json', measurements)
            print('FIT', json.dumps(metrics), flush=True)
    for key, value in net.state_dict().items():
        if not key.startswith('actor.'):
            torch.testing.assert_close(value, initial_state[key], rtol=0, atol=0)
    selected, retained = choose_checkpoint(measurements, config)
    path = folder/f'model_{selected["update"]}.pt'
    benchmark.write_json(folder/'selection-lock.json', dict(checkpoint=str(path.resolve()),
        checkpoint_sha256=benchmark.digest(path), retention_filter_pass=retained, metrics=selected,
        rule=protocol['selection_rule'], note='Locked before any closed-loop candidate result or fresh benchmark.'))
    print('LOCKED', path, 'retention filter:', retained, flush=True)


def diagnose(folder):
    protocol = json.loads((folder/'protocol.json').read_text())
    lock = json.loads((folder/'selection-lock.json').read_text())
    if benchmark.digest(lock['checkpoint']) != lock['checkpoint_sha256']:
        raise ValueError('Locked candidate changed')
    net, width = probe.load_actor(lock['checkpoint'])
    out = folder/'closed-loop'
    out.mkdir(exist_ok=False)
    results = []
    for case in protocol['cases']:
        trial = case['trial']
        name = f'case-{trial}-snapshot.pkl'
        verify_artifact(folder, name)
        snapshot = pickle.loads((folder/name).read_bytes())
        result = probe.evaluate(snapshot, net, width, probe.SearchConfig(), 0, 'policy', out/f'case-{trial}')
        results.append(dict(case=case, result=result))
        benchmark.write_json(out/'results.json', results)
        print('CLOSED LOOP', trial, case['split'], result['survived'], result['recovered'], flush=True)


def development_benchmark(folder):
    """Evaluate the already locked actor; leave the final holdout sealed on a rejected pilot."""
    protocol = json.loads((folder/'protocol.json').read_text())
    lock = json.loads((folder/'selection-lock.json').read_text())
    checkpoints = dict(accepted=protocol['checkpoint'], demonstration=lock['checkpoint'])
    expected = dict(accepted=protocol['checkpoint_sha256'], demonstration=lock['checkpoint_sha256'])
    if any(benchmark.digest(path) != expected[name] for name, path in checkpoints.items()):
        raise ValueError('A locked checkpoint changed')
    out = folder/'fresh-benchmark'
    out.mkdir(exist_ok=False)
    benchmark.write_json(out/'protocol.json', dict(checkpoints=checkpoints, checkpoint_sha256=expected,
        seeds=protocol['benchmark_seeds'], repeats=protocol['benchmark_repeats'], shards=protocol['benchmark_shards'],
        plant_sha256=protocol['plant_sha256'], rule='Candidate locked before evaluation; selection split is diagnostic. Final holdout remains sealed unless all prerequisite gates pass.'))
    benchmark.run_split(out, 'selection', checkpoints, protocol['benchmark_repeats'],
                        protocol['benchmark_shards'], 4, protocol['benchmark_seeds']['selection'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('train', 'diagnose', 'benchmark'))
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    dict(train=train, diagnose=diagnose, benchmark=development_benchmark)[args.stage](args.experiment)


if __name__ == '__main__':
    main()

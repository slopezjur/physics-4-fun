"""Collect sensor-only training pairs from a bounded, independently verified oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_perturb as benchmark
import probe_recovery_feasibility as probe
from build_policy_reference import collect
from perturb_env import PerturbEnv


TRAINING = dict(seed=29021, updates=2000, checkpoints=[250, 500, 1000, 2000],
                learning_rate=3e-5, batch_per_group=256,
                weights=dict(demo=1., quiet=2., impact=1.),
                quiet_mse_limit=1e-4, impact_mse_limit=4e-4)
TEST_SEEDS = dict(selection=155921, heldout=196613)


def choose_cases(bank):
    """Split whole trajectories before observing any new optimizer outcome."""
    protocol = json.loads((bank/'protocol.json').read_text())
    repeats, shards = protocol['repeats'], protocol['shards']
    records = [json.loads((bank/'selection'/f'accepted-{s:02d}.json').read_text()) for s in range(shards)]
    cases = []
    for body in range(2):
        for heading in range(4):
            start = (body*4+heading)*repeats
            found = []
            for trial in range(start, start+repeats):
                shard = (trial-start) % shards
                local = records[shard]['indices'].index(trial)
                if not records[shard]['survived'][local]:
                    found.append(dict(trial=trial, shard=shard, local=local,
                        target=protocol['target_bones'][body], heading=protocol['approach_sides'][heading]))
                if len(found) == 3:
                    break
            if len(found) != 3:
                raise ValueError('The fixed pilot requires three recorded falls per target/sector')
            for rank, case in enumerate(found):
                case['split'] = 'validation' if rank == 2 else 'train'
            cases.extend(found)
    return cases


def sensor_pairs(snapshot, trace):
    """Read BEFORE each command; full-state traces never become actor inputs."""
    env = PerturbEnv(num_envs=1, episode_seconds=20, observation_version='foundation_v2')
    snapshot.restore(env)
    observations = []
    with np.load(trace) as data:
        actions = data['commands'].copy()
        for i, action in enumerate(actions):
            observations.append(env.get_observations()[0].numpy())
            env.step(torch.tensor(action[None], dtype=torch.float32))
            np.testing.assert_array_equal(probe.initial_state(env), data['state'][i])
            np.testing.assert_array_equal(env.datas[0].ctrl, data['applied_controls'][i])
    return dict(obs=torch.tensor(np.array(observations)), mean=torch.tensor(actions))


def save_exact_snapshot(snapshot, path):
    # Native MjData pickling retains derived state, unlike state+forward restoration.
    # This local artifact is loaded only from this experiment's hash inventory.
    import pickle
    path.write_bytes(pickle.dumps(snapshot, protocol=5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    original = json.loads((args.bank/'protocol.json').read_text())
    checkpoint = original['checkpoints']['accepted']
    if benchmark.digest(checkpoint) != original['checkpoint_sha256']['accepted']:
        raise ValueError('Accepted checkpoint changed')
    if benchmark.digest(probe.ROOT/'mujoco_rig/dummy_ball.xml') != original['plant_sha256']:
        raise ValueError('Plant changed')
    args.out.mkdir(parents=True, exist_ok=False)
    cases = choose_cases(args.bank)
    protocol = dict(cases=cases, checkpoint=checkpoint, checkpoint_sha256=benchmark.digest(checkpoint),
        plant_sha256=original['plant_sha256'], implementation_sha256=benchmark.digest(__file__),
        search=vars(probe.SearchConfig()), training=TRAINING, benchmark_seeds=TEST_SEEDS,
        benchmark_repeats=16, benchmark_shards=8, observation_version='foundation_v2',
        split_rule='First two falls per stratum train, third validates; only successful teacher recoveries become labels.',
        selection_rule='Lowest whole-trajectory validation MSE among scheduled checkpoints meeting quiet/impact retention MSE limits; ties choose earlier update. If none qualifies, evaluate lowest-validation checkpoint diagnostically and reject it.',
        acceptance='Existing CPU quiet/survival/recovery gates plus both fresh balanced splits; no automatic promotion. Motion quality must not regress.',
        reference=dict(quiet_seed=28101, impact_seed=28103, quiet_envs=16, quiet_seconds=20,
                       impact_envs=64, impact_seconds=6, capacity=4096))
    benchmark.write_json(args.out/'protocol.json', protocol)
    snapshots, _ = probe.capture_cases(args.bank, checkpoint, cases, args.out)
    net, width = probe.load_actor(checkpoint)
    prior = {x['case']['trial']: x for x in json.loads((args.probe/'results.json').read_text())}
    old_protocol = json.loads((args.probe/'protocol.json').read_text())
    if old_protocol['checkpoint_sha256'] != protocol['checkpoint_sha256'] or old_protocol['config'] != protocol['search']:
        raise ValueError('Prior oracle does not match this fixed experiment')
    source_hashes = {k.replace('\\', '/'): v for k, v in json.loads((args.probe/'sha256.json').read_text()).items()}
    dataset, results = {}, []
    for case in cases:
        trial = case['trial']
        snapshot = snapshots[trial]
        save_exact_snapshot(snapshot, args.out/f'case-{trial}-snapshot.pkl')
        if trial in prior:
            trace = args.probe/f'case-{trial}-optimized.npz'
            if benchmark.digest(trace) != source_hashes[str(trace).replace('\\', '/')]:
                raise ValueError('Prior demonstration trace changed')
            result = prior[trial]['optimized']
        else:
            trace = args.out/f'case-{trial}-optimized.npz'
            result = probe.evaluate(snapshot, net, width, probe.SearchConfig(), 23000+trial,
                                    'optimized', trace.with_suffix(''))
        pairs = sensor_pairs(snapshot, trace)
        successful = result['survived'] and result['recovered'] and result['contact_recovered']
        if successful:
            dataset[trial] = dict(case=case, **pairs)
        results.append(dict(case=case, successful=successful, result=result, trace=str(trace), exact_replay=True))
        benchmark.write_json(args.out/'teachers.json', results)
        print(f'teacher {trial} {case["split"]}: survived={result["survived"]} recovered={successful}', flush=True)
    # Retention uses independent training seeds; evaluation seeds never enter training.
    reference = dict(quiet=collect(net, ball=False, num_envs=16, seconds=20, seed=28101,
                                  observation_version='foundation_v2', teacher_width=width),
                     impact=collect(net, ball=True, num_envs=64, seconds=6, seed=28103,
                                   observation_version='foundation_v2', teacher_width=width))
    if not all(any(v['case']['split'] == split for v in dataset.values()) for split in ('train', 'validation')):
        raise RuntimeError('Insufficient successful trajectories for a disjoint training/validation pilot')
    torch.save(dict(trajectories=dataset, reference=reference), args.out/'dataset.pt')
    # Other evaluations can run beside collection; their growing logs are not dataset artifacts.
    owned = [args.out/name for name in ('protocol.json', 'dataset.pt', 'teachers.json')]
    owned.extend(p for p in args.out.glob('case-*') if p.is_file())
    benchmark.write_json(args.out/'artifacts.json', {p.name: benchmark.digest(p) for p in owned})
    print('DATASET READY', {s: sum(v['case']['split'] == s for v in dataset.values()) for s in ('train', 'validation')}, flush=True)


if __name__ == '__main__':
    main()

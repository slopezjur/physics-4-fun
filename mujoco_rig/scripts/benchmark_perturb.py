"""Stratified, replayable CPU impact benchmark with a locked held-out comparison.

This diagnostic supplements eval.py; it cannot export a policy or relax promotion gates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mujoco_rig/rl'))
from env_config import BALL_SPAWN_DISTANCE, FALL_FRACTION, TARGET_BONES
from observation_contract import checkpoint_version
from perturb_env import PerturbEnv
from ppo import ActorCritic
from recovery_metrics import HeightProxyRecoveryMetrics, RecoveryMetrics
from scoring import paired_z

HEADINGS = ('+X', '+Y', '-X', '-Y')  # Projectile approach side in world coordinates.
SEEDS = {'selection': 104729, 'heldout': 130363}
VERSION = 'stratified_impact_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False), encoding='utf-8')


def make_design(seed, repeats, num_actions):
    """Balanced strata, with independent pose, launch-time and within-sector jitter."""
    if repeats < 1:
        raise ValueError('repeats must be positive')
    rng = np.random.default_rng(seed)
    rows = np.array([(b, h, r) for b in range(len(TARGET_BONES))
                     for h in range(len(HEADINGS)) for r in range(repeats)])
    n = len(rows)
    return dict(body_index=rows[:, 0], heading=rows[:, 1], replicate=rows[:, 2],
                noise=rng.uniform(-.03, .03, (n, num_actions)),
                theta=rows[:, 1] * np.pi / 2 + rng.uniform(-np.pi/4, np.pi/4, n),
                launch_time=rng.uniform(1.5, 1.8, n))


def run_policy(checkpoint, scenario, generate=False):
    """Production CPU physics and metrics, with externally fixed projectile launches."""
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    checkpoint_version(ck)
    net = ActorCritic(ck['num_obs'], ck['num_actions']).eval()
    net.load_state_dict(ck['model'])
    n = len(scenario['body_index'])
    env = PerturbEnv(num_envs=n, seed=0, episode_seconds=6, ball_every=(100, 101),
                     ball_speed=6, observation_version='foundation_v2')
    env.auto_reset = False
    env.next_ball[:] = np.inf
    dt = env.dt * env.decimation
    flight = BALL_SPAWN_DISTANCE / 6 * 1.2
    for i, d in enumerate(env.datas):
        d.qpos[:] = scenario['qpos'][i]
        d.qvel[:] = 0
        mujoco.mj_forward(env.model, d)
    obs = env.get_observations()
    meters = [HeightProxyRecoveryMetrics(n, dt, flight, env.rest_foot_z), RecoveryMetrics(n, dt, flight)]
    fallen = np.zeros(n, bool)
    terminated = np.zeros(n, bool)
    contact_seen = np.zeros(n, bool)
    reward_sum = np.zeros(n)
    previous = np.zeros((n, env.num_actions))
    for t in range(env.max_episode_length):
        with torch.no_grad():
            action = net.actor(obs[:, :ck['num_obs']])
        for i in np.flatnonzero(scenario['launch_step'] == t):
            d = env.datas[i]
            if generate:
                aim = d.xpos[env.targets[scenario['body_index'][i]]].copy()
                theta = scenario['theta'][i]
                start = aim + BALL_SPAWN_DISTANCE * np.array([np.cos(theta), np.sin(theta), 0])
                start[2] = max(start[2], .15)
                velocity = (aim - start) / np.linalg.norm(aim - start) * 6
                scenario['start'][i], scenario['velocity'][i] = start, velocity
            d.qpos[env.ball_q:env.ball_q+3] = scenario['start'][i]
            d.qvel[env.ball_v:env.ball_v+3] = scenario['velocity'][i]
            env.ball_flight[i] = t * dt + flight
            env.shots_fired[i] = 1
        obs, reward, done, _ = env.step(action)
        if not torch.isfinite(reward).all() or not torch.isfinite(action).all():
            raise RuntimeError('Nonfinite rollout output')
        reward_sum += reward.numpy() * ~terminated
        terminated |= done.numpy()
        clipped = action.clamp(-1, 1).numpy()
        for i, d in enumerate(env.datas):
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError('Nonfinite physics state')
            f = env.recovery_features(i)
            fallen[i] |= f['pelvis_z'] < FALL_FRACTION * env.rest_pelvis_z or f['up'] < .5
            for meter in meters:
                meter.record(i, f, d.cvel[env.pelvis, :3], not fallen[i], env.shots_fired[i], clipped[i] - previous[i])
            # Presence sampled at control boundaries; absence does not prove a missed shot.
            if not contact_seen[i]:
                for contact in d.contact:
                    bodies = env.model.geom_bodyid[contact.geom]
                    if env.ball in bodies and all(b != 0 for b in bodies):
                        contact_seen[i] = True
                        break
        previous = clipped
    legacy, loaded = [m.result(~fallen, True) for m in meters]
    if not np.all(env.shots_fired == 1):
        raise RuntimeError('Every trial must receive exactly one launch')
    return dict(survived=~fallen, recovered=legacy['recovered_mask'],
                contact_recovered=loaded['recovered_mask'], reward_sum=reward_sum,
                contact_seen=contact_seen, action_sum=meters[1].action_change,
                alive_samples=meters[1].alive_samples, contact_sliding=meters[1].sliding)


def worker(job):
    torch.set_num_threads(1)
    folder, shard, design, indices, checkpoints = job
    folder = Path(folder)
    env = PerturbEnv(num_envs=1, episode_seconds=6, observation_version='foundation_v2')
    n = len(indices)
    qpos = np.repeat(env.rest_qpos[None], n, axis=0)
    qpos[:, env.qadr] += design['noise'][indices]
    qpos[:, env.ball_q:env.ball_q+3] = (40, 40, 2)
    scenario = {k: v[indices] for k, v in design.items()}
    scenario.update(qpos=qpos.astype(np.float32),
                    launch_step=np.ceil(scenario['launch_time']/(env.dt*env.decimation)).astype(int),
                    start=np.zeros((n, 3), np.float32), velocity=np.zeros((n, 3), np.float32))
    del env
    results = {}
    for label, checkpoint in checkpoints.items():
        print(f'{folder.name}/{shard}/{label}: started ({n} worlds)', flush=True)
        results[label] = run_policy(checkpoint, scenario, generate=label == 'accepted')
        if label == 'accepted':
            np.savez_compressed(folder/f'scenarios-{shard:02d}.npz', indices=indices, **scenario)
        write_json(folder/f'{label}-{shard:02d}.json', dict(indices=indices, **results[label]))
        print(f'{folder.name}/{shard}/{label}: finished', flush=True)
    return shard


def aggregate(folder, label, total, shards):
    combined = {}
    seen = np.zeros(total, bool)
    for shard in range(shards):
        record = json.loads((folder/f'{label}-{shard:02d}.json').read_text())
        ids = np.array(record.pop('indices'))
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids >= total).any():
            raise ValueError('Invalid trial IDs')
        if len(np.unique(ids)) != len(ids) or seen[ids].any():
            raise ValueError('Duplicate trial IDs')
        seen[ids] = True
        if combined and set(record) != set(combined):
            raise ValueError('Mismatched result fields')
        for key, values in record.items():
            values = np.asarray(values)
            if values.ndim != 1 or len(values) != len(ids):
                raise ValueError('Result length does not match trial IDs')
            combined.setdefault(key, np.empty(total, dtype=values.dtype))[ids] = values
    if not seen.all():
        raise ValueError('Missing trials')
    return combined


def paired(candidate, baseline, mask=None):
    mask = np.ones(len(baseline['survived']), bool) if mask is None else mask
    result = {'worlds': int(mask.sum())}
    for key in ('survived', 'recovered', 'contact_recovered'):
        new, old = np.asarray(candidate[key])[mask], np.asarray(baseline[key])[mask]
        gained, lost, z = paired_z(new, old)
        result[key] = dict(candidate=int(new.sum()), baseline=int(old.sum()), gained=gained, lost=lost, z=float(z))
    result['reward_delta'] = float(np.mean(candidate['reward_sum'][mask] - baseline['reward_sum'][mask]))
    return result


def choose_candidate(results):
    """Predeclared ranking; held-out data is never passed to this function."""
    candidates = [label for label in results if label != 'accepted']
    if not candidates:
        raise ValueError('At least one candidate is required')
    return sorted(candidates, key=lambda label: (-int(np.sum(results[label]['recovered'])),
                  -int(np.sum(results[label]['survived'])), label))[0]


def summarize(folder, design, results):
    baseline = results['accepted']
    summary = dict(policies={}, comparisons={})
    for label, data in results.items():
        summary['policies'][label] = dict(survived=int(data['survived'].sum()), recovered=int(data['recovered'].sum()),
            contact_recovered=int(data['contact_recovered'].sum()), observed_contacts=int(data['contact_seen'].sum()),
            mean_reward=float(data['reward_sum'].mean()),
            action_change=float(data['action_sum'].sum()/max(1, data['alive_samples'].sum())),
            contact_sliding=float(data['contact_sliding'].mean()))
        if label == 'accepted':
            continue
        cells = {}
        for b, name in enumerate(TARGET_BONES):
            for h, heading in enumerate(HEADINGS):
                mask = (design['body_index'] == b) & (design['heading'] == h)
                cells[f'{name}/{heading}'] = paired(data, baseline, mask)
        summary['comparisons'][label] = dict(overall=paired(data, baseline), cells=cells,
            by_body={name: paired(data, baseline, design['body_index'] == b) for b, name in enumerate(TARGET_BONES)},
            by_heading={name: paired(data, baseline, design['heading'] == h) for h, name in enumerate(HEADINGS)})
    write_json(folder/'summary.json', summary)
    return summary


def run_split(root, split, checkpoints, repeats, shards, workers, seed):
    folder = root/split
    folder.mkdir()
    ck = torch.load(checkpoints['accepted'], map_location='cpu', weights_only=False)
    design = make_design(seed, repeats, ck['num_actions'])
    total = len(design['body_index'])
    np.savez_compressed(folder/'design.npz', **design)
    jobs = [(folder, s, design, np.flatnonzero(design['replicate'] % shards == s), checkpoints)
            for s in range(shards)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            shard = future.result()
            print(f'{split}: shard {shard+1}/{shards} complete', flush=True)
    results = {label: aggregate(folder, label, total, shards) for label in checkpoints}
    summary = summarize(folder, design, results)
    print(split, json.dumps(summary['policies']), flush=True)
    write_json(folder/'artifacts.json', {p.name: digest(p) for p in sorted(folder.glob('*')) if p.is_file()})
    return results, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accepted', type=Path, required=True)
    p.add_argument('--candidate', action='append', required=True, help='Unique label=checkpoint path')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=16)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--selection_seed', type=int, default=SEEDS['selection'])
    p.add_argument('--heldout_seed', type=int, default=SEEDS['heldout'])
    args = p.parse_args()
    if args.repeats < 1 or args.shards < 1 or args.repeats % args.shards or args.workers < 1:
        p.error('repeats must be positive and divisible by positive shards; workers must be positive')
    if min(args.selection_seed, args.heldout_seed) < 0 or args.selection_seed == args.heldout_seed:
        p.error('Selection and held-out seeds must be nonnegative and distinct')
    checkpoints = {'accepted': str(args.accepted.resolve())}
    for spec in args.candidate:
        label, path = spec.split('=', 1)
        if not label or not all(c.isalnum() or c in '_-' for c in label) or label in checkpoints:
            p.error('Candidate labels must be unique, safe file names, excluding accepted')
        checkpoints[label] = str(Path(path).resolve())
    hashes = {label: digest(path) for label, path in checkpoints.items()}
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = {'selection': args.selection_seed, 'heldout': args.heldout_seed}
    write_json(args.out/'protocol.json', dict(version=VERSION, seeds=seeds, checkpoints=checkpoints,
        checkpoint_sha256=hashes, plant_sha256=digest(ROOT/'mujoco_rig/dummy_ball.xml'),
        implementation_sha256=digest(__file__),
        target_bones=TARGET_BONES, approach_sides=HEADINGS, repeats=args.repeats, shards=args.shards,
        worlds_per_split=len(TARGET_BONES)*len(HEADINGS)*args.repeats, seconds=6, speed=6,
        selection_rule='Most legacy recoveries, then most survivors, then label ascending; selection split only.',
        confirmation_rule='Selection AND heldout: no survival regression, more legacy recoveries, paired recovery z >= 2 versus accepted. Diagnostic only; existing promotion gates still apply.',
        launch_rule='Accepted CPU policy generates launches; exact world-space launch data is replayed for candidates.',
        heading_rule='World-space projectile spawn side, uniform +/-45 degrees around each axis.',
        contact_note='Contact presence sampled at control boundaries; absence is not proof of a miss.',
        mujoco_version=mujoco.__version__, torch_version=torch.__version__))
    selection, first_summary = run_split(args.out, 'selection', checkpoints, args.repeats, args.shards, args.workers,
                                         args.selection_seed)
    selected = choose_candidate(selection)
    write_json(args.out/'selection-lock.json', dict(selected=selected, selection_sha256=digest(args.out/'selection'/'summary.json'),
        checkpoint_sha256=hashes[selected], rule='Frozen before any held-out rollout'))
    print(f'SELECTION LOCKED: {selected}', flush=True)
    _, second_summary = run_split(args.out, 'heldout', {label: checkpoints[label] for label in ('accepted', selected)},
                                  args.repeats, args.shards, args.workers, args.heldout_seed)
    comparisons = [summary['comparisons'][selected]['overall'] for summary in (first_summary, second_summary)]
    passes = [r['survived']['candidate'] >= r['survived']['baseline'] and
              r['recovered']['candidate'] > r['recovered']['baseline'] and r['recovered']['z'] >= 2 for r in comparisons]
    write_json(args.out/'decision.json', dict(selected=selected, selection_pass=passes[0], heldout_pass=passes[1],
        confirmed=all(passes), promoted=False, note='Supplementary benchmark only; no quiet-room test or shipped-policy comparison here.'))
    print('DECISION', (args.out/'decision.json').read_text(), flush=True)


if __name__ == '__main__':
    main()

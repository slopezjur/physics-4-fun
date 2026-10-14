"""Audit reward rankings using sealed trajectories; no physics integration."""
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from util.torch_util import quat_to_tan_norm
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .train_coordinated_ankle import AnkleHead

SOURCE = Path('logs/mimickit-training/recovery-starts-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-start-learning-20261009-01')
OUT = Path('logs/mimickit-training/reward-alignment-20261009-01')
TRAIN = [i for i in range(16) if i % 8 < 6]
GAMMA = .995


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def next_features(z, frame, reference, mean, std):
    """Rebuild the post-interval critic input without advancing the simulator."""
    q, v = (torch.from_numpy(z[key][frame]) for key in ('qpos', 'qvel'))
    bf = torch.from_numpy(z['balance_features'][frame])
    pending = []
    for f in (frame-1, frame):
        pending.append(torch.cat((torch.from_numpy(z['issued'][f]), torch.zeros(16, 33), torch.ones(16, 1)), -1))
    future = torch.cat((reference[:, :3], quat_to_tan_norm(reference[:, [4, 5, 6, 3]]), reference[:, 7:]), -1).repeat(1, 3)
    raw = torch.cat((q[:, :3], quat_to_tan_norm(q[:, [4, 5, 6, 3]]), v[:, :3], 2*bf[:, 4:7],
        q[:, 7:], .1*v[:, 6:], .001*torch.from_numpy(z['loads'][frame]), *pending, future, torch.zeros(16, 1)), -1)
    assert raw.shape == (16, 374)
    # MimicKit BaseAgent uses a 10-unit observation-normalizer clip.
    normalized = ((raw-mean)/std).clamp(-10, 10)
    history = torch.cat((bf[:, None], torch.from_numpy(z['history_before'][frame, :, :2])), 1)
    x = torch.cat((normalized, history.flatten(1).clamp(-20, 20),
        torch.from_numpy(z['addition'][frame, :, 21:22])/.1, torch.from_numpy(z['latent'][frame])), -1)
    assert x.shape == (16, 427) and torch.isfinite(x).all()
    return x


def components(z, i, begin, model, data, jac):
    terms = {k: np.zeros(len(z['time'])) for k in ('posture', 'capture', 'settling', 'swing', 'action_cost', 'failure')}
    for f in range(begin, len(z['time'])):
        if not z['alive'][f, i]:
            continue
        if z['failed'][f, i]:
            terms['failure'][f] = -2.
            continue
        data.qpos[:], data.qvel[:] = z['qpos'][f, i], z['qvel'][f, i]
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSubtreeCom(model, data, jac, 1)
        velocity = jac@data.qvel
        bf = z['balance_features'][f, i].astype(np.float64)
        posture = np.exp(-bf[11]**2-((1-np.clip(bf[12], -1, 1))/.2)**2)
        risk = np.clip(z['capture'][f, i]/.06, 0, 1)
        settling = np.exp(-np.square(velocity/.5).sum()-np.square(bf[4:7]).sum())
        bilateral = np.clip(z['loads'][f, i].min()/30, 0, 1)
        terms['posture'][f] = .35*posture
        terms['capture'][f] = .45*posture*np.exp(-(z['capture'][f, i]/.1)**2)
        terms['settling'][f] = .2*posture*(1-risk)*settling*bilateral
        terms['swing'][f] = .15*posture*risk*z['swing'][f, i]
        terms['action_cost'][f] = -.005*np.square(z['addition'][f, i].astype(np.float64)/.1).mean()
    return terms


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py'))
    for p in paths:
        shutil.copy2(p, OUT/'sources'/p.name)
    external = Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit')
    paths += [PARENT, FOUNDATION/'dummy.xml', FOUNDATION/'stationary_reference.npz',
        Path(__file__).with_name('RECOVERY_REWARD_AUDIT_PLAN.md'), external/'learning/base_agent.py',
        external/'learning/normalizer.py', external/'util/torch_util.py', SOURCE/'audit.json',
        SOURCE/'completed.json', LEARNING/'audit.json', LEARNING/'completed.json']
    paths += [LEARNING/f'fit-{s}'/'critic-final.pt' for s in (921009, 931009)]
    for t in (0, 1):
        for branch in ('quiet', 'push'):
            paths.append(SOURCE/f't{t}-{branch}-start.npz')
            for action in ('zero', 'negative', 'positive'):
                paths += [SOURCE/f't{t}-{branch}-{action}'/name for name in ('trace.npz', 'report.json')]
    inputs = {str(p): digest(p) for p in paths}
    for root in (SOURCE, LEARNING):
        assert read(root/'audit.json')['passed']
        manifest = read(root/'completed.json')['outputs']
        for p in paths:
            if p.is_relative_to(root) and str(p.relative_to(root)) in manifest:
                assert digest(p) == manifest[str(p.relative_to(root))]
    save(OUT/'inputs.json', inputs)
    torch.set_num_threads(1)
    state = torch.load(PARENT, weights_only=True, map_location='cpu')
    mean, std = state['_obs_norm._mean'], state['_obs_norm._std']
    reference = torch.from_numpy(arrays(FOUNDATION/'stationary_reference.npz')['qpos'].astype(np.float32))[torch.arange(16) % 8]
    critics = {}
    for seed in (921009, 931009):
        critic = AnkleHead(1).eval().requires_grad_(False)
        critic.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/'critic-final.pt', weights_only=True, map_location='cpu'))
        critics[str(seed)] = critic
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    data, jac = mujoco.MjData(model), np.zeros((3, model.nv))
    rows, reward_error, feature_error = [], 0., 0.
    for timing in (0, 1):
        for branch in ('quiet', 'push'):
            start = arrays(SOURCE/f't{timing}-{branch}-start.npz')
            for action in ('zero', 'negative', 'positive'):
                folder = SOURCE/f't{timing}-{branch}-{action}'
                z, report = arrays(folder/'trace.npz'), read(folder/'report.json')
                begin = report['issue_frame']
                x = next_features(z, begin-1, reference, mean, std)
                error = float(np.abs(x.numpy()[TRAIN]-start['features'][TRAIN]).max())
                feature_error = max(feature_error, error)
                assert error < 2e-5, (folder, error)
                endpoint = next_features(z, len(z['time'])-1, reference, mean, std)
                values = {seed: critic(endpoint).detach().numpy() for seed, critic in critics.items()}
                weights = GAMMA**np.arange(len(z['time'])-begin)
                for i in TRAIN:
                    alive, failed = z['alive'][:, i], z['failed'][:, i]
                    assert alive[begin] and np.array_equal(alive[1:], (alive & ~failed)[:-1])
                    mask = alive[begin:]
                    terms = components(z, i, begin, model, data, jac)
                    reconstruction = sum(terms.values())[begin:]
                    err = float(np.abs(reconstruction[mask]-z['reward'][begin:, i][mask]).max())
                    reward_error = max(reward_error, err)
                    assert err < 2e-5, (folder, i, err)
                    score = float(np.dot(weights, z['reward'][begin:, i]*mask))
                    parts = {key: float(np.dot(weights, value[begin:])) for key, value in terms.items()}
                    assert abs(sum(parts.values())-score) < .002
                    survived = bool((alive & ~failed).all())
                    assert survived == report['cases'][i]['survived']
                    bonus = {seed: float(weights[-1]*GAMMA*values[seed][i]) if survived else 0. for seed in critics}
                    windows = {str(seconds): float(np.dot(weights[:int(np.ceil(seconds/report['dt']))],
                        (z['reward'][begin:, i]*mask)[:int(np.ceil(seconds/report['dt']))])) for seconds in (.2, .5, 1., 2.)}
                    rows.append(dict(timing=timing, branch=branch, action=action, case=i, pose=i % 8,
                        issue_frame=begin, source=str(folder), alive_intervals=int(mask.sum()),
                        survived=survived, settled=report['cases'][i]['settled'], observed_return=score,
                        components=parts, bootstrap_bonus=bonus,
                        bootstrapped_return={seed: score+b for seed, b in bonus.items()}, window_returns=windows))
                print('AUDIT', folder.name, '12 training poses', flush=True)
    lookup = {(r['timing'], r['branch'], r['case'], r['action']): r for r in rows}
    pairs = []
    for row in rows:
        if row['action'] == 'zero':
            continue
        base = lookup[(row['timing'], row['branch'], row['case'], 'zero')]
        outcome = 'rescue' if row['settled'] and not base['settled'] else 'loss' if base['settled'] and not row['settled'] else 'unchanged'
        pairs.append(dict(timing=row['timing'], branch=row['branch'], case=row['case'], pose=row['pose'],
            action=row['action'], outcome=outcome, delta=row['observed_return']-base['observed_return'],
            bootstrap_delta={s: row['bootstrapped_return'][s]-base['bootstrapped_return'][s] for s in critics},
            component_delta={k: row['components'][k]-base['components'][k] for k in row['components']},
            window_delta={k: row['window_returns'][k]-base['window_returns'][k] for k in row['window_returns']}))
    summary = {}
    for branch in ('push', 'quiet'):
        selected = [p for p in pairs if p['branch'] == branch]
        summary[branch] = {outcome: dict(count=len(group), positive=sum(p['delta'] > 1e-6 for p in group),
            negative=sum(p['delta'] < -1e-6 for p in group), min_delta=min([p['delta'] for p in group], default=None),
            max_delta=max([p['delta'] for p in group], default=None),
            bootstrapped_positive={s: sum(p['bootstrap_delta'][s] > 1e-6 for p in group) for s in critics})
            for outcome in ('rescue', 'loss', 'unchanged') if (group := [p for p in selected if p['outcome'] == outcome])}
    oracle = []
    for t in (0, 1):
        for i in TRAIN:
            candidates = [lookup[(t, 'push', i, a)] for a in ('zero', 'negative', 'positive')]
            best = max(candidates, key=lambda r: r['observed_return'])
            oracle.append(dict(timing=t, case=i, any_settled=any(r['settled'] for r in candidates),
                best_reward_action=best['action'], best_reward_settled=best['settled']))
    save(OUT/'results.json', dict(gamma=GAMMA, training_worlds=TRAIN, rows=rows, pairs=pairs,
        summary=summary, oracle=oracle, max_reward_error=reward_error, max_start_feature_error=feature_error,
        interpretation='Observed finite-horizon rewards and frozen-final-critic sensitivity; not historical PPO advantages.'))
    assert len(rows) == 144 and len(pairs) == 96 and all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(OUT/'completed.json', dict(new_physics_intervals=0, optimizer_updates=0, trajectories=144,
        input_hashes=len(inputs), inputs_unchanged=True,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'sources' not in p.parts}))
    print(json.dumps(dict(summary=summary, reward_error=reward_error, feature_error=feature_error,
        oracle_settled=sum(r['best_reward_settled'] for r in oracle), reachable=sum(r['any_settled'] for r in oracle))), flush=True)


if __name__ == '__main__':
    main()

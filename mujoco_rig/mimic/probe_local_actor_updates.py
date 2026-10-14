"""Offline paired-credit PPO diagnostic; no physics or policy checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.distributions import Normal

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead, SIGMA

BASE = Path('logs/mimickit-training/recovery-local-action-credit-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
OUT = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
ACTORS = (921009, 931009)
OBJECTIVES = ('ordinary', 'centered')
HALF_SIGMA = SIGMA/2
PERMUTATION_SEED = 961011
MOVEMENT_TOLERANCE = 1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_actor(seed):
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/'final.pt',
                                    weights_only=True, map_location='cpu'))
    return actor


def parameter_vector(actor):
    return torch.cat([p.detach().flatten() for p in actor.parameters()])


def parameter_hash(actor):
    return hashlib.sha256(parameter_vector(actor).numpy().tobytes()).hexdigest()


def make_dataset(seed):
    rows, metadata, probes, lookup = [], [], [], {}
    for onset in (44, 74):
        for kind, worlds in (('quiet', list(range(6))), ('push', [*range(6), *range(8, 14)])):
            for noise in (951011, 951012, 951013, 951014):
                names = [f'{onset}-{kind}-{seed}-{noise}-{arm}' for arm in ('negative', 'positive')]
                traces = [arrays(BASE/name/'trace.npz') for name in names]
                reports = [read(BASE/name/'report.json') for name in names]
                for j, world in enumerate(worlds):
                    branch = 0 if kind == 'quiet' else 1 if world < 8 else 2
                    key = (onset, branch, world % 8)
                    if key not in lookup:
                        lookup[key] = len(probes)
                        probes.append(dict(onset=onset, branch=branch, pose=world % 8))
                    pair = len(rows)//2
                    for arm, (z, report, name) in enumerate(zip(traces, reports, names)):
                        case = report['cases'][j]
                        assert case['world'] == world and z['alive'][0, j]
                        assert abs(case['gae0']-float(z['advantage64'][0, j])) < 1e-7
                        rows.append(dict(x=z['features64'][0, j], action=z['latent'][0, j],
                            old_mean=z['mean'][0, j], gae=z['advantage64'][0, j], value=z['value'][0, j],
                            return_full=np.float64(case['reward_return']), pair=np.int64(pair),
                            probe=np.int64(lookup[key]), branch=np.int64(branch), onset=np.int64(onset),
                            pose=np.int64(world % 8), noise=np.int64(noise), sign=np.int64(2*arm-1)))
                        metadata.append(dict(folder=name, world=world, index=j))
    z = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert len(rows) == 288 and len(probes) == 36
    assert np.array_equal(z['x'][::2], z['x'][1::2])
    assert np.array_equal(z['old_mean'][::2], z['old_mean'][1::2])
    av = torch.from_numpy(z['gae'])
    denominator = av.std().clamp_min(1e-6)
    z['ordinary'] = ((av-av.mean())/denominator).numpy()
    z['centered'] = ((av-av.reshape(-1, 2).mean(1).repeat_interleave(2))/denominator).numpy()
    z['advantage_std'] = denominator.numpy()
    first = np.array([np.flatnonzero(z['probe'] == i)[0] for i in range(36)])
    z['probe_x'], z['probe_old_mean'] = z['x'][first], z['old_mean'][first]
    for key in ('branch', 'onset', 'pose'):
        z['probe_'+key] = z[key][first]
    z['probe_quiet_index'] = np.array([lookup[(p['onset'], 0, p['pose'])] for p in probes])
    z['probe_return_delta'] = np.array([np.mean(z['return_full'][(z['probe'] == i) & (z['sign'] == 1)])
        -np.mean(z['return_full'][(z['probe'] == i) & (z['sign'] == -1)]) for i in range(36)])
    z['probe_preferred'] = np.where(np.abs(z['probe_return_delta']) > 1e-3,
                                  np.sign(z['probe_return_delta']), 0).astype(np.int64)
    assert (z['probe_preferred'][z['probe_branch'] == 1] == -1).all()
    assert (z['probe_preferred'][z['probe_branch'] == 2] == 1).all()
    rng = torch.Generator().manual_seed(PERMUTATION_SEED)
    z['permutations'] = torch.stack([torch.randperm(288, generator=rng) for _ in range(3)]).numpy()
    for i in range(36):
        assert np.array_equal(z['x'][z['probe'] == i], np.repeat(z['probe_x'][i:i+1], 8, axis=0))
    return z, dict(rows=metadata, probes=probes)


def tensors(z):
    t = {key: torch.from_numpy(z[key]) for key in
         ('x', 'action', 'old_mean', 'ordinary', 'centered', 'probe_x')}
    t['old_logp'] = Normal(t['old_mean'], HALF_SIGMA).log_prob(t['action']).sum(-1)
    return t


def surrogate(actor, t, advantage, ids=None):
    if ids is None:
        ids = torch.arange(len(advantage))
    ratio = (Normal(actor(t['x'][ids]), HALF_SIGMA).log_prob(t['action'][ids]).sum(-1)
             -t['old_logp'][ids]).exp()
    return torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids]).mean(), ratio


def gradient(actor, t, advantage, ids=None):
    objective, _ = surrogate(actor, t, advantage, ids)
    return torch.cat([g.flatten() for g in torch.autograd.grad(objective, actor.parameters())]).detach()


def comparison(left, right):
    a, b = left.double(), right.double()
    an, bn = a.norm(), b.norm()
    return dict(norm=float(an), reference_norm=float(bn), norm_ratio=float(an/bn.clamp_min(1e-30)),
        cosine=float(torch.dot(a, b)/(an*bn).clamp_min(1e-30)),
        relative_difference=float((a-b).norm()/bn.clamp_min(1e-30)),
        max_absolute_difference=float((a-b).abs().max()))


def split_counts(ids):
    counts = np.bincount(np.asarray(ids)//2, minlength=144)
    return dict(complete_pairs=int((counts == 2).sum()), unmatched_rows=int((counts == 1).sum()),
                samples=len(ids))


def movement(mean, z):
    delta = np.asarray(mean, dtype=float).flatten()-z['probe_old_mean'].astype(float).flatten()
    quiet = z['probe_branch'] == 0
    push = ~quiet
    response = delta-delta[z['probe_quiet_index']]
    preferred = z['probe_preferred']
    def rms(v):
        return float(np.sqrt(np.mean(v*v)))
    def ranked(v, mask):
        directed = v[mask]*preferred[mask]
        return dict(helpful=int((directed > MOVEMENT_TOLERANCE).sum()),
            harmful=int((directed < -MOVEMENT_TOLERANCE).sum()),
            tied=int((np.abs(directed) <= MOVEMENT_TOLERANCE).sum()),
            signed_mean=float(directed.mean()), rms=rms(v[mask]))
    energy = float(np.mean(delta*delta))
    return dict(quiet_mean=float(delta[quiet].mean()), quiet_rms=rms(delta[quiet]),
        all_rms=rms(delta), constant_energy_fraction=float(delta.mean()**2/max(energy, 1e-30)),
        raw_push=ranked(delta, push), response_push=ranked(response, push),
        by_branch={label: dict(mean_change=float(delta[z['probe_branch'] == b].mean()),
            response_mean=float(response[z['probe_branch'] == b].mean()),
            raw=ranked(delta, z['probe_branch'] == b), response=ranked(response, z['probe_branch'] == b))
            for b, label in ((1, '+X'), (2, '-X'))})


def initial_diagnostic(actor, t, z):
    n = len(z['gae'])
    grads = torch.stack([torch.stack([gradient(actor, t, t[name]*mask) for mask in
        [torch.ones(n), *[torch.from_numpy((z['branch'] == b).astype(np.float32)) for b in range(3)]]])
        for name in OBJECTIVES])
    invariant = comparison(grads[0, 0], grads[1, 0])
    assert invariant['relative_difference'] < 2e-4
    assert torch.allclose(grads[0], grads[1], atol=2e-6, rtol=2e-4)
    assert torch.allclose(grads[:, 1:].sum(1), grads[:, 0], atol=2e-6, rtol=2e-4)
    jacobian = []
    for x in t['probe_x']:
        jacobian.append(torch.cat([g.flatten() for g in
            torch.autograd.grad(actor(x[None]).sum(), actor.parameters())]).detach())
    jacobian = torch.stack(jacobian)
    gram = jacobian.double()@jacobian.double().T
    response = torch.einsum('sp,ogp->ogs', jacobian.double(), grads.double())
    minibatches = [torch.from_numpy(ids) for permutation in z['permutations'] for ids in
                   (permutation[:256], permutation[256:])]
    mini = torch.stack([torch.stack([gradient(actor, t, t[name], ids) for ids in minibatches])
                        for name in OBJECTIVES])
    batch_records = []
    for j, ids in enumerate(minibatches):
        counts = torch.bincount(ids//2, minlength=144)
        unmatched = counts[ids//2] == 1
        difference = (t['ordinary']-t['centered']).clone()
        mask = torch.zeros(n)
        mask[ids[unmatched]] = 1
        baseline_gradient = gradient(actor, t, difference*mask)*n/len(ids)
        residual = comparison(mini[0, j]-mini[1, j], baseline_gradient)
        assert residual['max_absolute_difference'] < 2e-6
        batch_records.append(dict(epoch=j//2, batch=j % 2, **split_counts(ids.numpy()),
            ordinary=comparison(mini[0, j], grads[0, 0]), centered=comparison(mini[1, j], grads[1, 0]),
            unmatched_baseline_max_error=residual['max_absolute_difference']))
    for objective in range(2):
        for epoch in range(3):
            weighted=(256*mini[objective, 2*epoch]+32*mini[objective, 2*epoch+1])/288
            assert torch.allclose(weighted, grads[objective, 0], atol=2e-6, rtol=2e-4)
    norm = grads[0].double().norm(dim=1)
    branch_gram = grads[0].double()@grads[0].double().T
    cosines = branch_gram/(norm[:, None]*norm[None, :]).clamp_min(1e-30)
    jnorm = gram.diag().sqrt()
    jcos = gram/(jnorm[:, None]*jnorm[None, :]).clamp_min(1e-30)
    return dict(gradient=grads.numpy(), jacobian_gram=gram.numpy(), response=response.numpy(),
                initial_minibatch_gradient=mini.numpy()), dict(invariant=invariant,
        branch_gradient_norm=norm.tolist(), branch_gradient_cosine=cosines.tolist(),
        branch_retention=float(norm[0]/norm[1:].sum()),
        matched_push_quiet_jacobian_cosines=[float(jcos[i, z['probe_quiet_index'][i]])
            for i in range(36) if z['probe_branch'][i] != 0],
        advantage_sign_changes=int(((t['ordinary'] > 0) != (t['centered'] > 0)).sum()),
        batches=batch_records)


def run_arm(initial, t, z, name):
    actor = deepcopy(initial)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    assert not optimizer.state
    av = t[name]
    with torch.no_grad():
        initial_mean = actor(t['probe_x']).numpy().copy()
    output = {key: [] for key in ('candidate_means', 'ratios', 'minibatch_gradient',
                                  'full_gradient', 'parameter_delta')}
    output['means'] = [initial_mean]
    records, stopped = [], False
    for epoch, order in enumerate(z['permutations']):
        for batch, selection in enumerate((order[:256], order[256:])):
            if stopped:
                break
            ids = torch.from_numpy(selection)
            full = gradient(actor, t, av)
            mini = gradient(actor, t, av, ids)
            before_params = parameter_vector(actor).clone()
            old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
            with torch.no_grad():
                _, ratio_all = surrogate(actor, t, av)
            objective, ratios = surrogate(actor, t, av, ids)
            optimizer.zero_grad(set_to_none=True)
            (-objective).backward()
            clip_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), .5))
            optimizer.step()
            with torch.no_grad():
                kl = float(((actor(t['x'])-t['old_mean']).square().sum(-1)/(2*HALF_SIGMA**2)).mean())
                candidate = actor(t['probe_x']).numpy().copy()
            delta = parameter_vector(actor)-before_params
            accepted = bool(np.isfinite(kl) and kl <= .01)
            if not accepted:
                actor.load_state_dict(old_actor)
                optimizer.load_state_dict(old_optimizer)
                assert torch.equal(parameter_vector(actor), before_params)
                stopped = True
            with torch.no_grad():
                mean = actor(t['probe_x']).numpy().copy()
            output['candidate_means'].append(candidate)
            output['means'].append(mean)
            output['ratios'].append(ratio_all.numpy().copy())
            output['minibatch_gradient'].append(mini.numpy())
            output['full_gradient'].append(full.numpy())
            output['parameter_delta'].append(delta.numpy())
            records.append(dict(epoch=epoch, batch=batch, ids=selection.tolist(),
                loss=float(-objective.detach()), gradient_norm=float(mini.norm()), clip_norm=clip_norm,
                gradient_compare=comparison(mini, full), split_counts=split_counts(selection),
                kl=kl if np.isfinite(kl) else None,
                accepted=accepted, clipped_ratio_fraction=float(((ratios < .8) | (ratios > 1.2)).float().mean()),
                surrogate_clipped_fraction=float(((ratios*av[ids]) >
                    ratios.clamp(.8, 1.2)*av[ids]).float().mean()),
                candidate_metrics=movement(candidate, z) if np.isfinite(candidate).all() else None,
                metrics=movement(mean, z)))
    return {key: np.stack(value) for key, value in output.items()}, dict(objective=name,
        records=records, accepted=sum(r['accepted'] for r in records),
        rejected=sum(not r['accepted'] for r in records), attempted=len(records),
        initial_parameter_sha256=parameter_hash(initial), final_parameter_sha256=parameter_hash(actor),
        final_summary=movement(output['means'][-1], z))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = sorted(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [Path(__file__).with_name('RECOVERY_LOCAL_ACTOR_UPDATES_PLAN.md'),
        BASE/'inputs.json', BASE/'completed.json', BASE/'audit.json', BASE/'results.json', BASE/'paired-credit.json',
        LEARNING/'completed.json', LEARNING/'audit.json']
    assert read(BASE/'audit.json')['passed'] and read(LEARNING/'audit.json')['passed']
    source_done = read(BASE/'completed.json')['outputs']
    source_inputs = read(BASE/'inputs.json')
    traces = sorted(BASE.glob('*/trace.npz'))+sorted(BASE.glob('*/report.json'))
    assert len(traces) == 128
    for path in traces:
        assert digest(path) == source_done[str(path.relative_to(BASE))]
    paths += traces
    paths += [LEARNING/f'fit-{seed}'/name for seed in ACTORS for name in ('final.pt', 'critic-final.pt')]
    inputs = {str(path): digest(path) for path in paths}
    common = sorted(set(source_inputs) & set(inputs))
    assert all(source_inputs[path] == inputs[path] for path in common)
    save(OUT/'inputs.json', inputs)
    save(OUT/'contract.json', dict(actors=list(ACTORS), objectives=list(OBJECTIVES), samples_per_actor=288,
        pairs_per_actor=144, unique_starts_per_actor=36, noise_repeats=4, sigma=HALF_SIGMA,
        permutation_seed=PERMUTATION_SEED, epochs=3, minibatch_sizes=[256, 32], max_attempted_updates=24,
        learning_rate=1e-4, adam_betas=[.9, .999], adam_epsilon=1e-8, gradient_clip=.5,
        ppo_ratio_clip=[.8, 1.2], kl_limit=.01, standard_deviation_correction=1,
        invariant_atol=2e-6, invariant_rtol=2e-4, invariant_relative_norm_tolerance=2e-4,
        movement_sign_tolerance=MOVEMENT_TOLERANCE, physical_rollouts=0, critic_updates=0,
        new_checkpoints=0, shared_historical_inputs=len(common), baseline_early_weights='all age zero, hence one',
        scope='Balanced first-decision diagnostic with fresh Adam, not historical optimizer replay or physical evaluation.'))
    torch.set_num_threads(1)
    results = {}
    for seed in ACTORS:
        folder = OUT/str(seed)
        folder.mkdir()
        z, metadata = make_dataset(seed)
        np.savez_compressed(folder/'dataset.npz', **z)
        save(folder/'metadata.json', metadata)
        t = tensors(z)
        actor = load_actor(seed)
        before = parameter_hash(actor)
        with torch.no_grad():
            assert torch.allclose(actor(t['x']), t['old_mean'], atol=2e-7, rtol=0)
        evidence, initial = initial_diagnostic(actor, t, z)
        np.savez_compressed(folder/'initial.npz', **evidence)
        save(folder/'initial.json', initial)
        results[str(seed)] = dict(initial=initial, arms={})
        for name in OBJECTIVES:
            evidence, report = run_arm(actor, t, z, name)
            np.savez_compressed(folder/(name+'.npz'), **evidence)
            save(folder/(name+'.json'), report)
            results[str(seed)]['arms'][name] = {key: report[key] for key in
                ('accepted', 'rejected', 'attempted', 'final_summary')}
            print('ARM', seed, name, json.dumps(results[str(seed)]['arms'][name]), flush=True)
        assert parameter_hash(actor) == before
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, inputs_unchanged=True,
        samples=576, pairs=288, actor_starts=72, physical_rollouts=0, new_checkpoints=0, critic_updates=0,
        attempted_updates=sum(arm['attempted'] for result in results.values() for arm in result['arms'].values()),
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

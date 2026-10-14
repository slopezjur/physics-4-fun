"""Paired sampling interventions preserving every historical quiet draw."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .train_arm_corrective_comparison import TAKEOVER, TEACHER, PRIOR_ROOT, ARENA, equal_weights
from .probe_arm_execution_distribution import setup, seal, episode
from .train_arm_relative_proposal import ProposalHead, load_head
from .train_arm_correction_ppo import CorrectionEnvironment
from .challenge_arm_proposals import HEADS, adapter_for
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-sampling-balance-20261008-01')
PLAN = Path(__file__).with_name('ARM_SAMPLING_BALANCE_PLAN.md')
SEEDS = (711028, 721028)
UPDATES, BATCH = 4000, 512


def recipes(original, correction):
    training = ~original['validation']
    quiet = float(original['quiet'][training].mean())
    active = int((training & ~original['quiet']).sum())
    high = (1-quiet)*len(correction['x'])/(active+len(correction['x']))
    return {'replay': 0., 'correction05': .05, 'correction15': .15, 'restored_high': high}


class PairedSampler:
    def __init__(self, quiet, correction_count, seed, share, device):
        self.quiet = torch.tensor(quiet, device=device)
        self.count, self.correction_count = len(quiet), correction_count
        self.threshold = share/(1-float(np.mean(quiet)))
        assert 0 <= self.threshold <= 1
        self.base_rng = torch.Generator(device=device).manual_seed(seed)
        self.swap_rng = torch.Generator(device=device).manual_seed(seed+1000000)
        self.device = device
        self.hashes = {k: hashlib.sha256() for k in ('base', 'quiet', 'effective', 'replaced')}
        self.counts = dict(quiet=0, original_active=0, correction=0)

    def draw(self):
        base = torch.randint(self.count, (BATCH,), generator=self.base_rng, device=self.device)
        uniform = torch.rand((BATCH,), generator=self.swap_rng, device=self.device)
        correction = torch.randint(self.correction_count, (BATCH,), generator=self.swap_rng, device=self.device)
        quiet = self.quiet[base]
        replace = ~quiet & (uniform < self.threshold)
        effective = torch.where(replace, self.count+correction, base)
        assert torch.equal(effective[quiet], base[quiet])
        arrays = dict(base=base, quiet=torch.where(quiet, base, -1), effective=effective, replaced=replace)
        for key, value in arrays.items():
            self.hashes[key].update(value.cpu().numpy().tobytes())
        q, c = int(quiet.sum()), int(replace.sum())
        self.counts['quiet'] += q
        self.counts['correction'] += c
        self.counts['original_active'] += BATCH-q-c
        return effective

    def evidence(self):
        return dict(counts=self.counts, hashes={k: h.hexdigest() for k, h in self.hashes.items()},
                    replacement_probability=self.threshold)


def fit(base, control, original, correction, seed, recipe, share, folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    net = ProposalHead(base.mean, base.scale, base.output_scale, control.q_indices, control.v_indices, 'inherited')
    folder.mkdir()
    torch.save(net.state_dict(), folder/'initial.pt')
    net.to(device).train()
    train = ~original['validation']
    assert set(original['world'][train]) == set(range(6))
    assert not correction['validation'].any() and (correction['world'] < 6).all()
    x = torch.tensor(np.concatenate((original['x'][train], correction['x'])), device=device)
    y = torch.tensor(np.concatenate(((original['raw_target']-original['goal'])[train],
                                    correction['raw_target']-correction['goal'])), device=device)
    sampler = PairedSampler(original['quiet'][train], len(correction['x']), seed, share, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    start, history = time.perf_counter(), []
    for update in range(UPDATES):
        ids = sampler.draw()
        loss = ((net(x[ids])-y[ids])/net.output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (update+1) % 1000 == 0:
            history.append(dict(update=update+1, loss=float(loss.detach())))
            print('FIT', recipe, seed, history[-1], flush=True)
    net.eval().cpu()
    torch.save(net.state_dict(), folder/'final.pt')
    save(folder/'architecture.json', dict(kind='inherited', seed=seed, inputs=421,
        active_features=287, architecture='421-128ELU-128ELU-33, zero final layer, inherited per-motor output scale'))
    result = dict(recipe=recipe, seed=seed, correction_share=share, updates=UPDATES, batch_size=BATCH,
                  lr=1e-4, device=device, wall_seconds=time.perf_counter()-start, sampler=sampler.evidence())
    save(folder/'report.json', result)
    save(folder/'history.json', history)
    return net, result


def scores_for(results, shares):
    scores = {}
    for recipe in shares:
        rows = [r for r in results if r['config']['recipe'] == recipe and not r['config'].get('quiet')]
        scores[recipe] = [min(r['complete'] for r in rows), sum(r['complete'] for r in rows)]
    return scores


def main():
    assert json.loads((TAKEOVER/'audit.json').read_text())['passed']
    original, correction = (dict(np.load(path/'training-samples.npz')) for path in (TEACHER, TAKEOVER))
    shares = recipes(original, correction)
    baseline_report = json.loads((PRIOR_ROOT/'step-29/report.json').read_text())
    baseline_path = PRIOR_ROOT/'step-29'/baseline_report['results'][0]['folder']/'trace.npz'
    extra = [PLAN, ARENA, TAKEOVER/'audit.json', TAKEOVER/'training-samples.npz',
             TEACHER/'training-samples.npz', baseline_path]
    for seed in SEEDS:
        extra += [HEADS[f'all-{seed}']/'final.pt']
    inputs = seal(OUT, extra)
    save(OUT/'protocol.json', dict(inputs=inputs, seeds=SEEDS, shares=shares, updates=UPDATES,
        batch_size=BATCH, quiet_fraction=float(original['quiet'][~original['validation']].mean()),
        collection_datasets=1, heldout_poses=[6, 7], quiet_baseline=str(baseline_path)))
    env, anchor, base, poses, actors = setup()
    training, results = [], []
    for seed in SEEDS:
        replay_sampler = None
        for recipe, share in shares.items():
            folder = OUT/f'{recipe}-{seed}'
            net, tr = fit(base, env.engine.policy_control, original, correction, seed, recipe, share, folder)
            if recipe == 'replay':
                assert equal_weights(folder/'final.pt', HEADS[f'all-{seed}']/'final.pt')
                replay_sampler = tr['sampler']
            else:
                assert equal_weights(folder/'initial.pt', OUT/f'replay-{seed}'/'initial.pt')
                for key in ('base', 'quiet'):
                    assert tr['sampler']['hashes'][key] == replay_sampler['hashes'][key]
                assert tr['sampler']['counts']['quiet'] == replay_sampler['counts']['quiet']
            training.append(tr)
            for variant in (False, True):
                name = folder.name+('-varied' if variant else '-nominal')
                adapter = CorrectionEnvironment(env, anchor, net, poses, variant=variant)
                r = episode(adapter, actors['zero'], dict(recipe=recipe, seed=seed, variant=variant,
                    mode='mean', teacher=False), OUT/name)
                results.append(dict(folder=name, **r))
                save(OUT/'progress.json', dict(training=training, results=results))
    scores = scores_for(results, shares)
    candidate = max((r for r in shares if r != 'replay'), key=lambda r: (*scores[r], -shares[r]))
    save(OUT/'selection.json', dict(scores=scores, candidate=candidate))
    from .run_single_support_feedback import make_environment
    env = make_environment(env.rig, anchor, 80.)
    baseline = np.load(baseline_path)
    for seed in SEEDS:
        net = load_head(OUT/f'{candidate}-{seed}')
        adapter = adapter_for(env, anchor, net, poses, quiet_seconds=60.)
        name = f'{candidate}-{seed}-quiet60'
        r = episode(adapter, actors['zero'], dict(recipe=candidate, seed=seed, quiet=True,
                    mode='mean', teacher=False), OUT/name)
        z = np.load(OUT/name/'trace.npz')
        assert r['quiet_passed'] == 16 and all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
        results.append(dict(folder=name, quiet_exact=True, **r))
        save(OUT/'progress.json', dict(training=training, results=results))
    selected = candidate if scores[candidate] > scores['replay'] else 'replay'
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(training=training, results=results, scores=scores, candidate=candidate,
        selected=selected, inputs_unchanged=True, arena_changed=False,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', scores, 'selected', selected, flush=True)


if __name__ == '__main__':
    main()

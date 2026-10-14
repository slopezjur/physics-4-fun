"""Independent sampler reconstruction, physical gates and replay provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_sampling_balance import OUT, SEEDS, UPDATES, BATCH
from .train_arm_corrective_comparison import TAKEOVER, TEACHER, equal_weights
from .challenge_arm_proposals import HEADS
from .analyze_arm_proposal_generalization import physical_results
from .verify_arm_proposal_30_rounding import gates
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR


def main():
    from .rig import Rig
    torch.set_num_threads(2)
    protocol = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    for path, h in {**protocol['inputs'], **report['outputs']}.items():
        assert digest(Path(path)) == h, path
    original, correction = (np.load(path/'training-samples.npz') for path in (TEACHER, TAKEOVER))
    train = ~original['validation']
    quiet_np = original['quiet'][train]
    assert set(original['world'][train]) == set(range(6))
    assert set(original['world'][~train]) == {6, 7}
    assert not correction['validation'].any() and (correction['world'] < 6).all()
    assert not correction['quiet'].any()
    active_count = int((train & ~original['quiet']).sum())
    expected_shares = {'replay': 0., 'correction05': .05, 'correction15': .15,
        'restored_high': (1-float(quiet_np.mean()))*len(correction['x'])/(active_count+len(correction['x']))}
    assert protocol['shares'] == expected_shares
    stream_results = []
    for seed in SEEDS:
        rows = {r['recipe']: r for r in report['training'] if r['seed'] == seed}
        assert len(rows) == 4
        device = rows['replay']['device']
        quiet = torch.tensor(quiet_np, device=device)
        base_rng = torch.Generator(device=device).manual_seed(seed)
        swap_rng = torch.Generator(device=device).manual_seed(seed+1000000)
        hashes = {r: {k: hashlib.sha256() for k in ('base', 'quiet', 'effective', 'replaced')} for r in rows}
        counts = {r: dict(quiet=0, original_active=0, correction=0) for r in rows}
        for _ in range(UPDATES):
            base = torch.randint(len(quiet), (BATCH,), generator=base_rng, device=device)
            uniform = torch.rand((BATCH,), generator=swap_rng, device=device)
            ids = torch.randint(len(correction['x']), (BATCH,), generator=swap_rng, device=device)
            q = quiet[base]
            previous_mask = torch.zeros_like(q)
            for recipe, share in expected_shares.items():
                mask = ~q & (uniform < share/(1-float(quiet_np.mean())))
                assert not (previous_mask & ~mask).any()
                previous_mask = mask
                effective = torch.where(mask, len(quiet)+ids, base)
                assert torch.equal(effective[q], base[q])
                arrays = dict(base=base, quiet=torch.where(q, base, -1), effective=effective, replaced=mask)
                for k, value in arrays.items():
                    hashes[recipe][k].update(value.cpu().numpy().tobytes())
                nq, nc = int(q.sum()), int(mask.sum())
                counts[recipe]['quiet'] += nq
                counts[recipe]['correction'] += nc
                counts[recipe]['original_active'] += BATCH-nq-nc
        for recipe, row in rows.items():
            folder = OUT/f'{recipe}-{seed}'
            assert row['sampler']['counts'] == counts[recipe]
            assert row['sampler']['hashes'] == {k: h.hexdigest() for k, h in hashes[recipe].items()}
            assert (row['updates'], row['batch_size'], row['lr']) == (4000, 512, 1e-4)
            assert equal_weights(folder/'initial.pt', OUT/f'replay-{seed}'/'initial.pt')
            state = torch.load(folder/'final.pt', weights_only=True)
            assert all(torch.isfinite(v).all() for v in state.values())
            initial = torch.load(folder/'initial.pt', weights_only=True)
            assert (initial['net.4.weight'] == 0).all() and (initial['net.4.bias'] == 0).all()
            assert any(not torch.equal(v, state[k]) for k, v in initial.items())
            stream_results.append(dict(seed=seed, recipe=recipe, **counts[recipe]))
        assert equal_weights(OUT/f'replay-{seed}'/'final.pt', HEADS[f'all-{seed}']/'final.pt')
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    poses = np.load(PRIOR/'goals.npz')['poses']
    prior = physical_results()
    baseline = np.load(protocol['quiet_baseline'])
    outcomes, quiet_cases = [], 0
    for r in report['results']:
        folder = OUT/r['folder']
        outcomes.append(dict(folder=folder.name, **gates(rig.model, poses, folder, r)))
        z = np.load(folder/'trace.npz')
        assert not z['correction'].any() and not r['config']['teacher']
        if r['config'].get('quiet'):
            assert r['quiet_passed'] == 16
            assert all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
            quiet_cases += 16
        elif r['config']['recipe'] == 'replay':
            archived = next(path for path, row in prior[f'all-{r["config"]["seed"]}']
                            if row['config'].get('variant', False) == r['config']['variant'])
            old = np.load(archived/'trace.npz')
            assert all(np.array_equal(z[k], old[k]) for k in ('qpos', 'issued', 'alive', 'done'))
    scores = {}
    for recipe in expected_shares:
        rows = [r for r in report['results'] if r['config']['recipe'] == recipe and not r['config'].get('quiet')]
        assert len(rows) == 4
        scores[recipe] = [min(r['complete'] for r in rows), sum(r['complete'] for r in rows)]
    candidate = max((r for r in scores if r != 'replay'), key=lambda r: (*scores[r], -expected_shares[r]))
    selected = candidate if scores[candidate] > scores['replay'] else 'replay'
    assert scores == report['scores'] and candidate == report['candidate'] and selected == report['selected']
    assert quiet_cases == 32
    save(OUT/'audit.json', dict(passed=True, scores=scores, selected=selected, candidate=candidate,
        quiet60_exact=quiet_cases, sampler_streams=stream_results, quiet_streams_exact=True,
        nested_correction_masks=True, heldout_training_rows=0, outcomes=outcomes,
        optimizer_updates=32000, sampled_presentations=16384000, physical_cases=160,
        arena_changed=False, auditor_sha256=digest(Path(__file__))))
    print('AUDIT PASSED', scores, 'selected', selected, flush=True)


if __name__ == '__main__':
    main()

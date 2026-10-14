"""Audit identical data streams, calibrated response weight and physical gates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_response_supervision import OUT, COLLECTION, PREVIOUS, TEACHER, SEEDS, RECIPES
from .train_arm_pending_effect import load_head
from .train_arm_corrective_comparison import equal_weights
from .analyze_arm_reachable_responses import response_metrics
from .verify_arm_proposal_30_rounding import gates
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR


def verify_calibration(original, pairs, report, calibration):
    train = ~original['validation']
    rows = []
    for seed in SEEDS:
        device = next(r['device'] for r in report['training'] if r['seed'] == seed)
        net = load_head(OUT/f'absolute-{seed}').requires_grad_(True).to(device)
        net.load_state_dict(torch.load(OUT/f'absolute-{seed}'/'initial.pt', weights_only=True))
        tensor = lambda v: torch.tensor(v, device=device)
        x, y = tensor(original['x'][train]), tensor((original['raw_target']-original['goal'])[train])
        p0, p1 = net(tensor(pairs['x0'])), net(tensor(pairs['x1']))
        y0, y1 = tensor(pairs['raw_target0']-pairs['goal']), tensor(pairs['raw_target1']-pairs['goal'])
        absolute = (((net(x)-y)/net.output_scale)**2).mean()+.025*((((p0-y0)/net.output_scale)**2).mean()+(((p1-y1)/net.output_scale)**2).mean())
        secant = (p1-p0-tensor(pairs['raw_target1']-pairs['raw_target0']))/net.output_scale/tensor(abs(pairs['issued_delta']).astype(np.float32)[:, None])
        magnitude = secant.abs()
        response = torch.where(magnitude < 1, .5*secant.square(), magnitude-.5).mean()
        parameters = tuple(net.parameters())
        ga = torch.autograd.grad(absolute, parameters, retain_graph=True)
        gr = torch.autograd.grad(response, parameters)
        a, r = (float(torch.cat([g.flatten() for g in values]).norm()) for values in (ga, gr))
        saved = next(v for v in calibration['seeds'] if v['seed'] == seed)
        assert np.isclose(a, saved['absolute_gradient_norm'], rtol=1e-5, atol=1e-6)
        assert np.isclose(r, saved['response_gradient_norm'], rtol=1e-5, atol=1e-6)
        rows.append(dict(seed=seed, absolute_gradient_norm=a, response_gradient_norm=r))
    weight = np.exp(np.mean(np.log([.1*r['absolute_gradient_norm']/r['response_gradient_norm'] for r in rows])))
    assert np.isclose(weight, calibration['response_weight'], rtol=1e-5)
    return rows


def main():
    from .rig import Rig
    torch.set_num_threads(2)
    protocol, report, calibration = (json.loads((OUT/name).read_text()) for name in ('protocol.json', 'report.json', 'calibration.json'))
    for path, h in {**protocol['inputs'], **report['outputs']}.items():
        assert digest(Path(path)) == h, path
    original, pairs = (np.load(p) for p in (TEACHER/'training-samples.npz', COLLECTION/'response-pairs.npz'))
    assert not pairs['validation'].any() and set(pairs['world']) <= {0, 4}
    train = ~original['validation']
    assert train.sum() == 6804 and (train & original['quiet']).sum() == 2256 and (~train).sum() == 2268
    assert set(original['world'][train]) == set(range(6)) and set(original['world'][~train]) == {6, 7}
    calibrated_gradients = verify_calibration(original, pairs, report, calibration)
    streams, diagnostics = [], []
    for seed in SEEDS:
        rows = {r['recipe']: r for r in report['training'] if r['seed'] == seed}
        assert set(rows) == set(RECIPES)
        device = rows['absolute']['device']
        rng, pair_rng = torch.Generator(device=device).manual_seed(seed), torch.Generator(device=device).manual_seed(seed+2000000)
        hashes = {k: hashlib.sha256() for k in ('original', 'pairs', 'quiet')}
        quiet, quiet_count = original['quiet'][train], 0
        for _ in range(4000):
            ids = torch.randint(6804, (512,), generator=rng, device=device).cpu().numpy()
            pair_ids = torch.randint(len(pairs['x0']), (32,), generator=pair_rng, device=device).cpu().numpy()
            hashes['original'].update(ids.tobytes());hashes['pairs'].update(pair_ids.tobytes())
            hashes['quiet'].update(np.where(quiet[ids], ids, -1).tobytes());quiet_count += int(quiet[ids].sum())
        expected = {k: h.hexdigest() for k, h in hashes.items()}
        old = json.loads((PREVIOUS/f'pending_effect-{seed}'/'report.json').read_text())
        assert expected['original'] == old['draw_hash']
        for recipe, r in rows.items():
            folder = OUT/f'{recipe}-{seed}'
            assert r['hashes'] == expected and r['quiet_draws'] == quiet_count
            assert (r['updates'], r['batch_size'], r['pair_batch'], r['lr'], r['pair_weight']) == (4000, 512, 32, 1e-4, .05)
            assert r['response_weight'] == (calibration['response_weight'] if recipe == 'response' else 0.)
            assert equal_weights(folder/'initial.pt', PREVIOUS/f'pending_effect-{seed}'/'initial.pt')
            s = torch.load(folder/'final.pt', weights_only=True)
            assert all(torch.isfinite(v).all() for v in s.values())
            assert json.loads((folder/'history.json').read_text())[-1]['update'] == 4000
            with torch.no_grad():
                net = load_head(folder)
                predicted = net(torch.tensor(original['x'])).numpy()
                error = predicted-(original['raw_target']-original['goal'])
                raw0 = pairs['goal']+net(torch.tensor(pairs['x0'])).numpy()
                raw1 = pairs['goal']+net(torch.tensor(pairs['x1'])).numpy()
                row = dict(recipe=recipe, seed=seed, train_rmse=float(np.sqrt(np.mean(error[train]**2))),
                    heldout_rmse=float(np.sqrt(np.mean(error[~train]**2))),
                    fitted_response=response_metrics(raw1-raw0, pairs['raw_target1']-pairs['raw_target0'], pairs['seed'] == seed))
                diagnostics.append(row)
        streams.append(dict(seed=seed, hashes=expected, quiet_draws=quiet_count))
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    poses = np.load(PRIOR/'goals.npz')['poses']
    baseline, outcomes, quiet_cases = np.load(protocol['quiet_baseline']), [], 0
    for r in report['results']:
        folder, c = OUT/r['folder'], r['config']
        outcomes.append(dict(folder=folder.name, **gates(rig.model, poses, folder, r)))
        z = np.load(folder/'trace.npz')
        assert not z['correction'].any() and not c['teacher']
        if c.get('quiet'):
            assert r['quiet_passed'] == 16 and all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
            quiet_cases += 16
        if c.get('proposal_std'):
            noise = np.load(folder/'noise.npz')['noise']
            generator = torch.Generator().manual_seed(c['noise_seed'])
            for row in noise:
                assert np.array_equal((torch.randn(row.shape, generator=generator)*.005).numpy(), row)
        if c.get('timeline'):
            assert c['timeline'] == [.9, 1.4, 3.2] and abs(r['hold_end']-5.5) < 1e-12
    core, challenge = {}, {}
    for recipe in RECIPES:
        for flag, target, count in ((False, core, 4), (True, challenge, 6)):
            group = [r for r in report['results'] if r['config']['recipe'] == recipe and r['config']['challenge'] == flag]
            assert len(group) == count
            target[recipe] = [min(r['complete'] for r in group), sum(r['complete'] for r in group)]
    assert core == report['core_scores'] and challenge == report['challenge_scores']
    assert (core['response'] > core['absolute'] and challenge['response'] >= challenge['absolute']) == report['research_improved']
    assert (core['response'] == [8, 32] and challenge['response'] == [8, 48]) == report['admitted']
    assert quiet_cases == 32 and sum(r['cases'] for r in outcomes) == 192
    save(OUT/'audit.json', dict(passed=True, core_scores=core, challenge_scores=challenge,
        admitted=report['admitted'], research_improved=report['research_improved'],
        calibrated_gradients=calibrated_gradients, response_weight=calibration['response_weight'],
        original_and_quiet_streams_preserved=True, pair_streams_identical=True, sampler_streams=streams,
        diagnostics=diagnostics, outcomes=outcomes, quiet60_exact=32, physical_cases=192,
        optimizer_updates=16000, original_presentations=8192000, extra_pair_presentations=1024000,
        total_presentations=9216000, heldout_training_rows=0, ppo_transitions=0, arena_changed=False,
        auditor_sha256=digest(Path(__file__))))
    print('FIT AUDIT PASSED', core, challenge, flush=True)


if __name__ == '__main__':
    main()

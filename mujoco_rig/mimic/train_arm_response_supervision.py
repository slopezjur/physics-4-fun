"""Matched absolute-target versus finite-response supervision on reachable pairs."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .collect_arm_reachable_responses import OUT as COLLECTION
from .train_arm_pending_effect import PendingEffectHead, load_head, SEEDS, OUT as PREVIOUS
from .train_arm_corrective_comparison import TEACHER, PRIOR_ROOT, ARENA, equal_weights
from .probe_arm_execution_distribution import setup, seal, episode
from .challenge_arm_proposals import adapter_for, OutputProbe
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-response-supervision-20261008-01')
PLAN = Path(__file__).with_name('ARM_RESPONSE_SUPERVISION_PLAN.md')
RECIPES = ('absolute', 'response')
UPDATES, BATCH, PAIR_BATCH = 4000, 512, 32
PAIR_WEIGHT, RESPONSE_GRADIENT_FRACTION = .05, .1
TIMELINE, NOISE_SEEDS = (.9, 1.4, 3.2), (710083, 710084)


def tensors(original, pairs, device):
    train = ~original['validation']
    assert set(original['world'][train]) == set(range(6))
    assert not pairs['validation'].any() and set(pairs['world']) <= {0, 4}
    values = dict(x=original['x'][train], y=(original['raw_target']-original['goal'])[train],
        x0=pairs['x0'], x1=pairs['x1'], y0=pairs['raw_target0']-pairs['goal'],
        y1=pairs['raw_target1']-pairs['goal'], delta=pairs['raw_target1']-pairs['raw_target0'],
        pulse=np.abs(pairs['issued_delta']).astype(np.float32)[:, None])
    assert values['pulse'].min() > 1e-5
    return {k: torch.tensor(v, device=device) for k, v in values.items()}


def losses(net, data, ids, pairs):
    prediction = net(data['x'][ids])
    p0, p1 = net(data['x0'][pairs]), net(data['x1'][pairs])
    original = ((prediction-data['y'][ids])/net.output_scale).square().mean()
    absolute = .5*(((p0-data['y0'][pairs])/net.output_scale).square().mean()
                    +((p1-data['y1'][pairs])/net.output_scale).square().mean())
    secant = (p1-p0-data['delta'][pairs])/net.output_scale/data['pulse'][pairs]
    response = F.smooth_l1_loss(secant, torch.zeros_like(secant), beta=1.)
    return original+PAIR_WEIGHT*absolute, response


def fresh(base, control, seed):
    torch.manual_seed(seed)
    return PendingEffectHead(base.mean, base.scale, base.output_scale, control.q_indices, control.v_indices, 'pending_effect')


def calibration(base, control, data, device):
    rows = []
    for seed in SEEDS:
        net = fresh(base, control, seed).to(device)
        absolute, response = losses(net, data, slice(None), slice(None))
        parameters = tuple(net.parameters())
        ga = torch.autograd.grad(absolute, parameters, retain_graph=True)
        gr = torch.autograd.grad(response, parameters)
        a = float(torch.stack([g.square().sum() for g in ga]).sum().sqrt())
        r = float(torch.stack([g.square().sum() for g in gr]).sum().sqrt())
        assert np.isfinite(a+r) and a > 0 and r > 0
        rows.append(dict(seed=seed, absolute_gradient_norm=a, response_gradient_norm=r,
                         ratio=RESPONSE_GRADIENT_FRACTION*a/r))
    weight = float(np.exp(np.mean(np.log([r['ratio'] for r in rows]))))
    return dict(response_weight=weight, initial_gradient_fraction=RESPONSE_GRADIENT_FRACTION,
                seeds=rows, training_only=True, physical_selection=False)


def fit(base, control, data, original, recipe, seed, coefficient, folder):
    device = data['x'].device
    net = fresh(base, control, seed)
    folder.mkdir();torch.save(net.state_dict(), folder/'initial.pt');net.to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    original_rng = torch.Generator(device=device).manual_seed(seed)
    pair_rng = torch.Generator(device=device).manual_seed(seed+2000000)
    hashes = {k: hashlib.sha256() for k in ('original', 'pairs', 'quiet')}
    quiet = original['quiet'][~original['validation']]
    quiet_count, history, start = 0, [], time.perf_counter()
    for update in range(UPDATES):
        ids = torch.randint(len(data['x']), (BATCH,), generator=original_rng, device=device)
        pair_ids = torch.randint(len(data['x0']), (PAIR_BATCH,), generator=pair_rng, device=device)
        index, pair_index = ids.cpu().numpy(), pair_ids.cpu().numpy()
        hashes['original'].update(index.tobytes());hashes['pairs'].update(pair_index.tobytes())
        hashes['quiet'].update(np.where(quiet[index], index, -1).tobytes())
        quiet_count += int(quiet[index].sum())
        absolute, response = losses(net, data, ids, pair_ids)
        loss = absolute+coefficient*response if recipe == 'response' else absolute
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (update+1) % 1000 == 0:
            history.append(dict(update=update+1, absolute_loss=float(absolute.detach()), response_loss=float(response.detach()), loss=float(loss.detach())))
            print('FIT', recipe, seed, history[-1], flush=True)
    net.cpu().eval();torch.save(net.state_dict(), folder/'final.pt')
    save(folder/'architecture.json', dict(kind='pending_effect', seed=seed, inputs=421, active_features=355,
        architecture='421-128ELU-128ELU-33', recipe=recipe))
    save(folder/'history.json', history)
    result = dict(recipe=recipe, seed=seed, updates=UPDATES, batch_size=BATCH, pair_batch=PAIR_BATCH,
        lr=1e-4, pair_weight=PAIR_WEIGHT, response_weight=coefficient if recipe == 'response' else 0.,
        device=str(device), original_rows=len(data['x']), paired_examples=len(data['x0']), quiet_draws=quiet_count,
        hashes={k: h.hexdigest() for k, h in hashes.items()}, wall_seconds=time.perf_counter()-start)
    save(folder/'report.json', result)
    return net, result


def scores(results, challenge):
    groups = {k: [r['complete'] for r in results if r['config']['recipe'] == k and r['config']['challenge'] == challenge] for k in RECIPES}
    return {k: [min(v), sum(v)] for k, v in groups.items()}


def main():
    admission = json.loads((COLLECTION/'audit.json').read_text())
    diagnostic = json.loads((COLLECTION/'response-diagnostic.json').read_text())
    assert admission['passed'] and diagnostic['passed'] and admission['paired_examples'] >= 80
    pairs = dict(np.load(COLLECTION/'response-pairs.npz'))
    assert set(zip(pairs['seed'], pairs['variant'], pairs['world'])) == {(s, v, w) for s in SEEDS for v in (False, True) for w in (0, 4)}
    baseline_report = json.loads((PRIOR_ROOT/'step-29/report.json').read_text())
    baseline_path = PRIOR_ROOT/'step-29'/baseline_report['results'][0]['folder']/'trace.npz'
    extra = [PLAN, ARENA, COLLECTION/'audit.json', COLLECTION/'response-diagnostic.json',
             COLLECTION/'response-pairs.npz', TEACHER/'training-samples.npz', PREVIOUS/'audit.json', baseline_path]
    extra += [PREVIOUS/f'pending_effect-{seed}'/name for seed in SEEDS for name in ('initial.pt', 'report.json')]
    inputs = seal(OUT, extra)
    env, anchor, base, poses, actors = setup()
    original = dict(np.load(TEACHER/'training-samples.npz'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = tensors(original, pairs, device)
    calibrated = calibration(base, env.engine.policy_control, data, device)
    save(OUT/'calibration.json', calibrated)
    save(OUT/'protocol.json', dict(inputs=inputs, seeds=SEEDS, recipes=RECIPES, updates=UPDATES,
        batch_size=BATCH, pair_batch=PAIR_BATCH, pair_weight=PAIR_WEIGHT, calibration=calibrated,
        heldout_poses=[6, 7], quiet_baseline=str(baseline_path), original_draws_preserved=True,
        collection_coupled_across_fitting_seeds=True, physical_cases=192, training_poses=[0, 1, 2, 3, 4, 5],
        new_pair_poses=[0, 4], response_loss='SmoothL1((prediction_delta-teacher_delta)/output_scale/abs(actual_issued_pulse), beta=1)'))
    training, results = [], []
    for seed in SEEDS:
        for recipe in RECIPES:
            folder = OUT/f'{recipe}-{seed}'
            net, tr = fit(base, env.engine.policy_control, data, original, recipe, seed, calibrated['response_weight'], folder)
            assert equal_weights(folder/'initial.pt', PREVIOUS/f'pending_effect-{seed}'/'initial.pt')
            previous = json.loads((PREVIOUS/f'pending_effect-{seed}'/'report.json').read_text())
            assert tr['hashes']['original'] == previous['draw_hash']
            if recipe == 'response':
                assert tr['hashes'] == training[-1]['hashes'] and tr['quiet_draws'] == training[-1]['quiet_draws']
            training.append(tr)
            configs = [dict(label='nominal', variant=False, challenge=False), dict(label='varied', variant=True, challenge=False),
                       dict(label='new-timing', variant=False, challenge=True, timeline=TIMELINE)]
            configs += [dict(label=f'noise-{s}', variant=False, challenge=True, proposal_std=.005, noise_seed=s) for s in NOISE_SEEDS]
            for c in configs:
                c.update(actor=folder.name, recipe=recipe, seed=seed, mode='mean', teacher=False)
                proposal = OutputProbe(net, env, std=c.get('proposal_std', 0.), seed=c.get('noise_seed', 710081))
                adapter = (CorrectionEnvironment(env, anchor, proposal, poses, variant=True) if c['variant']
                           else adapter_for(env, anchor, proposal, poses, timeline=c.get('timeline')))
                name = folder.name+'-'+c['label']
                r = episode(adapter, actors['zero'], c, OUT/name)
                if proposal.noises:
                    np.savez_compressed(OUT/name/'noise.npz', noise=np.stack(proposal.noises))
                results.append(dict(folder=name, **r));save(OUT/'progress.json', dict(training=training, results=results))
    from .run_single_support_feedback import make_environment
    env = make_environment(env.rig, anchor, 80.)
    baseline = np.load(baseline_path)
    for seed in SEEDS:
        name = f'response-{seed}-quiet60'
        net = load_head(OUT/f'response-{seed}')
        r = episode(adapter_for(env, anchor, net, poses, quiet_seconds=60.), actors['zero'],
                    dict(actor=name, recipe='response', seed=seed, quiet=True, challenge='quiet', mode='mean', teacher=False), OUT/name)
        z = np.load(OUT/name/'trace.npz')
        assert r['quiet_passed'] == 16 and all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
        results.append(dict(folder=name, quiet_exact=True, **r));save(OUT/'progress.json', dict(training=training, results=results))
    core, challenge = scores(results, False), scores(results, True)
    improved = core['response'] > core['absolute'] and challenge['response'] >= challenge['absolute']
    admitted = core['response'] == [8, 32] and challenge['response'] == [8, 48]
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(training=training, results=results, core_scores=core, challenge_scores=challenge,
        research_improved=improved, admitted=admitted, arena_changed=False, inputs_unchanged=True,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('FIT COMPARISON COMPLETE', core, challenge, 'admitted', admitted, flush=True)


if __name__ == '__main__':
    main()

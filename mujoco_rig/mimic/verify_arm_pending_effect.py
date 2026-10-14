"""Independent queue formula, pairing, stream, replay and physical-gate audit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_pending_effect import OUT, SEEDS, RECIPES, UPDATES, BATCH, load_head
from .train_arm_corrective_comparison import TEACHER, equal_weights
from .challenge_arm_proposals import HEADS
from .analyze_arm_proposal_generalization import physical_results
from .verify_arm_proposal_30_rounding import gates
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR


def expected_effects(x, qi, vi):
    slots = x[:, 95:229].reshape(-1, 2, 67)
    q, v = x[:, qi+8, None].transpose(0, 2, 1), x[:, vi+48, None].transpose(0, 2, 1)*10
    expected = np.zeros_like(slots)
    expected[:, :, :33] = np.clip(4*(slots[:, :, :33]-q)+.08*(slots[:, :, 33:66]-v), -1, 1)*slots[:, :, 66:]
    expected[:, :, 66] = slots[:, :, 66]
    return expected.reshape(-1, 134)


@torch.no_grad()
def main():
    from .rig import Rig
    torch.set_num_threads(2)
    protocol = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    validation = json.loads((OUT/'feature-validation.json').read_text())
    for path, h in {**protocol['inputs'], **report['outputs']}.items():
        assert digest(Path(path)) == h, path
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    qi, vi = rig.model.jnt_qposadr[joints], rig.model.jnt_dofadr[joints]
    poses = np.load(PRIOR/'goals.npz')['poses']
    data = np.load(TEACHER/'training-samples.npz')
    train = ~data['validation']
    assert len(data['x']) == 9072 and train.sum() == 6804
    assert (train & data['quiet']).sum() == 2256 and (~train).sum() == 2268
    assert set(data['world'][train]) == set(range(6)) and set(data['world'][~train]) == {6, 7}
    probes = np.load(OUT/'feature-probes.npz')
    assert probes['x'].shape == (120, 421)
    calculated = expected_effects(probes['x'], qi, vi)
    native_error = float(abs(calculated-probes['native_effects']).max())
    assert native_error < 2e-6 and validation['passed']
    assert validation['switch_states'] == 24 and len(validation['probes']) == 120
    changed, saturated = 0, 0
    for start in range(0, 120, 5):
        baseline = probes['x'][start]
        assert validation['probes'][start]['joint'] is None
        for i in range(start+1, start+5):
            x = probes['x'][i]
            difference = np.flatnonzero(x != baseline)
            assert len(difference) == 1 and 95 <= difference[0] < 128
            assert 0 < abs(float(x[difference[0]]-baseline[difference[0]])) <= .0050001
            altered = not np.array_equal(calculated[i], calculated[start])
            assert altered == validation['probes'][i]['changed_features']
            changed += altered
            saturated += not altered
    assert changed == validation['changed_counterfactuals']
    regression, streams = [], []
    for seed in SEEDS:
        rows = {r['recipe']: r for r in report['training'] if r['seed'] == seed}
        assert set(rows) == set(RECIPES)
        generator = torch.Generator(device=rows['masked']['device']).manual_seed(seed)
        stream = hashlib.sha256()
        for _ in range(UPDATES):
            ids = torch.randint(6804, (BATCH,), generator=generator, device=rows['masked']['device'])
            stream.update(ids.cpu().numpy().tobytes())
        for recipe, r in rows.items():
            folder = OUT/f'{recipe}-{seed}'
            assert (r['updates'], r['batch_size'], r['lr'], r['train_examples'], r['heldout_examples']) == (4000, 512, 1e-4, 6804, 2268)
            assert r['draw_hash'] == stream.hexdigest()
            assert equal_weights(folder/'initial.pt', OUT/f'masked-{seed}'/'initial.pt')
            initial = torch.load(folder/'initial.pt', weights_only=True)
            final = torch.load(folder/'final.pt', weights_only=True)
            assert all(torch.isfinite(v).all() for v in final.values())
            assert not initial['net.4.weight'].any() and not initial['net.4.bias'].any()
            assert any(not torch.equal(v, final[k]) for k, v in initial.items())
            assert json.loads((folder/'history.json').read_text())[-1]['update'] == 4000
            assert np.array_equal(final['qi'].numpy(), qi) and np.array_equal(final['vi'].numpy(), vi)
            net = load_head(folder)
            x = torch.tensor(data['x'])
            z = net.features(x).numpy()
            inherited = np.clip((data['x']-final['mean'].numpy())/final['scale'].numpy(), -20, 20)
            assert np.array_equal(z[:, :95], inherited[:, :95]) and np.array_equal(z[:, 229:], inherited[:, 229:])
            if recipe == 'masked':
                assert not z[:, 95:229].any()
                assert equal_weights(folder/'final.pt', HEADS[f'all-{seed}']/'final.pt')
            else:
                assert np.array_equal(z[:, 95:229], expected_effects(data['x'], qi, vi))
                assert abs(z[:, 95:229]).max() <= 1
            error = (net(x).numpy()-(data['raw_target']-data['goal']))**2
            for label, mask in (('train', train), ('heldout', ~train),
                                ('train-active', train & ~data['quiet']), ('heldout-active', ~train & ~data['quiet']),
                                ('train-quiet', train & data['quiet']), ('heldout-quiet', ~train & data['quiet'])):
                regression.append(dict(seed=seed, recipe=recipe, group=label, rows=int(mask.sum()),
                                       rmse=float(np.sqrt(error[mask].mean()))))
            for label, mask in (('train', train), ('heldout', ~train)):
                assert abs(float(np.sqrt(error[mask].mean()))-r[label+'_rmse']) < 1e-6
            net.load_state_dict(initial)
            assert not net(x).any()
        streams.append(dict(seed=seed, draw_hash=stream.hexdigest()))
    archived = physical_results()
    baseline = np.load(protocol['quiet_baseline'])
    outcomes, quiet_cases = [], 0
    for r in report['results']:
        folder, config = OUT/r['folder'], r['config']
        outcomes.append(dict(folder=folder.name, **gates(rig.model, poses, folder, r)))
        z = np.load(folder/'trace.npz')
        assert not z['correction'].any() and not config['teacher']
        if config.get('quiet'):
            assert r['quiet_passed'] == 16
            assert all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
            quiet_cases += 16
        elif config['recipe'] == 'masked' and not config['challenge']:
            old_folder = next(path for path, old in archived[f'all-{config["seed"]}']
                              if old['config'].get('variant', False) == config['variant'])
            old = np.load(old_folder/'trace.npz')
            assert all(np.array_equal(z[k], old[k]) for k in ('qpos', 'issued', 'alive', 'done'))
        if config.get('proposal_std'):
            noise = np.load(folder/'noise.npz')['noise']
            rng = torch.Generator().manual_seed(config['noise_seed'])
            for row in noise:
                assert np.array_equal((torch.randn(row.shape, generator=rng)*.005).numpy(), row)
        if config.get('label') == 'new-timing':
            assert config['timeline'] == [.9, 1.4, 3.2] and abs(r['hold_end']-5.5) < 1e-12
    core, challenge = {}, {}
    for recipe in RECIPES:
        for flag, destination, count in ((False, core, 4), (True, challenge, 6)):
            group = [r for r in report['results'] if r['config']['recipe'] == recipe and r['config']['challenge'] == flag]
            assert len(group) == count
            destination[recipe] = [min(r['complete'] for r in group), sum(r['complete'] for r in group)]
    assert core == report['core_scores'] and challenge == report['challenge_scores']
    improved = core['pending_effect'] > core['masked'] and challenge['pending_effect'] >= challenge['masked']
    admitted = core['pending_effect'] == [8, 32] and challenge['pending_effect'] == [8, 48]
    assert improved == report['research_improved'] and admitted == report['admitted']
    assert report['selected'] == ('pending_effect' if admitted else 'masked')
    assert quiet_cases == 32 and sum(r['cases'] for r in outcomes) == 192
    save(OUT/'audit.json', dict(passed=True, core_scores=core, challenge_scores=challenge,
        research_improved=improved, admitted=admitted, quiet60_exact=32,
        native_formula_max_error=native_error, changed_counterfactuals=changed,
        clipped_counterfactuals=saturated, sampler_streams=streams, regression=regression,
        paired_initializations_identical=True, masked_weights_and_traces_exact=True,
        heldout_training_rows=0, optimizer_updates=16000, sample_presentations=8192000,
        physical_cases=192, ppo_transitions=0, teacher_queries=0, outcomes=outcomes,
        arena_changed=False, auditor_sha256=digest(Path(__file__))))
    print('AUDIT PASSED', core, challenge, 'admitted', admitted, flush=True)


if __name__ == '__main__':
    main()

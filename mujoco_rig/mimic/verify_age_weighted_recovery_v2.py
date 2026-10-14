"""Independent physical and objective audit of the age-weighted PPO comparison."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .train_coordinated_ankle import AnkleHead, SIGMA
from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save

FULL = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')
OUT = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
PILOT = Path('logs/mimickit-training/learned-ankle-20261009-01')
WORLDS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def actor_weights(age):
    raw = np.where(age < 30, 5., 1.).astype(np.float32)
    return raw/raw.mean()


def check_objectives(folder, resets, history):
    batches = [arrays(folder/f'objective-{update}.npz') for update in range(1, 33)]
    assert len(list(folder.glob('objective-*.npz'))) == 32
    error = dict(weight=0., actor_loss=0., critic_loss=0.)
    exposure, raw_counts, early_raw, early_weighted = np.zeros(3), np.zeros(3, dtype=int), 0, 0.
    attempts, skipped = 0, 0
    for update, (batch, row) in enumerate(zip(batches, history), 1):
        for key in ('age', 'force', 'terminal', 'failed', 'actor_weight'):
            assert batch[key].shape == (64, 12), (update, key)
            assert np.isfinite(batch[key]).all(), (update, key)
        assert batch['terminal'].dtype == batch['failed'].dtype == np.bool_
        assert not np.any(batch['failed'] & ~batch['terminal'])
        assert np.issubdtype(batch['age'].dtype, np.integer) and (batch['age'] >= 0).all()
        weights = actor_weights(batch['age'])
        weight_error = float(np.abs(batch['actor_weight']-weights).max())
        error['weight'] = max(error['weight'], weight_error)
        assert np.allclose(batch['actor_weight'], weights, atol=1e-7, rtol=1e-7)
        assert abs(float(batch['actor_weight'].astype(float).mean())-1.) < 2e-7
        flat_weight, advantage = weights.ravel().astype(float), batch['normalized_advantage']
        assert advantage.shape == (768,) and np.isfinite(advantage).all()
        assert abs(float(advantage.astype(float).mean())) < 2e-6
        assert abs(float(advantage.astype(float).std(ddof=1))-1.) < 2e-6
        indices, ratio = batch['minibatch_indices'], batch['policy_ratio']
        assert indices.shape == ratio.shape == (9, 256)
        assert np.issubdtype(indices.dtype, np.integer)
        assert batch['actor_loss'].shape == batch['critic_loss'].shape == (9,)
        assert batch['critic_residual'].shape == (9, 256)
        for epoch in range(3):
            assert np.array_equal(np.sort(indices[epoch*3:epoch*3+3].ravel()), np.arange(768))
        actor_active = np.isfinite(batch['actor_loss'])
        assert actor_active[0] and np.all(np.diff(actor_active.astype(int)) <= 0)
        assert int(actor_active.sum()) == row['actor_updates']+row['rolled_back']
        assert row['rolled_back'] in (0, 1)
        for step, ids in enumerate(indices):
            if actor_active[step]:
                assert np.isfinite(ratio[step]).all() and (ratio[step] >= 0).all()
                raw = ratio[step].astype(float)*advantage[ids].astype(float)
                clipped = np.clip(ratio[step].astype(float), .8, 1.2)*advantage[ids].astype(float)
                expected = -np.mean(flat_weight[ids]*np.minimum(raw, clipped))
                delta = abs(float(batch['actor_loss'][step])-expected)
                error['actor_loss'] = max(error['actor_loss'], delta)
                assert np.isclose(batch['actor_loss'][step], expected, atol=3e-6, rtol=2e-6)
            else:
                assert np.isnan(batch['actor_loss'][step]) and np.isnan(ratio[step]).all()
            residual = batch['critic_residual'][step].astype(float)
            assert np.isfinite(residual).all() and np.isfinite(batch['critic_loss'][step])
            expected_critic = np.mean(residual**2)
            error['critic_loss'] = max(error['critic_loss'], abs(float(batch['critic_loss'][step])-expected_critic))
            assert np.isclose(batch['critic_loss'][step], expected_critic, atol=5e-5, rtol=2e-6)
        assert float(batch['critic_loss'][-1]) == row['critic_loss']
        assert int(batch['terminal'].sum()) == row['completed']
        assert int(batch['failed'].sum()) == row['failures']
        attempts += int(actor_active.sum())
        skipped += int((~actor_active).sum())
        force, early = batch['force'].ravel(), batch['age'].ravel() < 30
        recorded = row['weighting']
        assert recorded['early_samples'] == int(early.sum())
        assert recorded['late_samples'] == int((~early).sum())
        scalar_mean_tolerance = 4*np.finfo(np.float32).eps
        assert abs(recorded['mean_weight']-1.) < scalar_mean_tolerance
        assert abs(recorded['mean_weight']-float(batch['actor_weight'].astype(float).mean())) < scalar_mean_tolerance
        assert np.isclose(recorded['max_weight'], weights.max(), atol=1e-7, rtol=1e-7)
        assert np.isclose(recorded['early_weight_mass'], flat_weight[early].sum(), atol=1e-4, rtol=2e-6)
        assert np.isclose(recorded['late_weight_mass'], flat_weight[~early].sum(), atol=1e-4, rtol=2e-6)
        for i, mask in enumerate((force == 0, force > 0, force < 0)):
            exposure[i] += flat_weight[mask].sum()
            raw_counts[i] += int(mask.sum())
            branch = recorded['branches'][('quiet', 'positive_x', 'negative_x')[i]]
            assert branch['samples'] == int(mask.sum())
            assert np.isclose(branch['weight_mass'], flat_weight[mask].sum(), atol=1e-4, rtol=2e-6)
        early_raw += int(early.sum())
        early_weighted += float(flat_weight[early].sum())
        if update in (1, 32):
            collection = arrays(folder/f'collection-{update}.npz')
            for key in ('age', 'force', 'terminal', 'failed', 'actor_weight', 'normalized_advantage'):
                assert np.array_equal(collection[key], batch[key]), (update, key)
    force = np.concatenate([z['force'] for z in batches])
    age = np.concatenate([z['age'] for z in batches])
    terminal = np.concatenate([z['terminal'] for z in batches])
    assert force.shape == age.shape == terminal.shape == (2048, 12)
    assert all(row['world'] in WORLDS for row in resets)
    for j, world in enumerate(WORLDS):
        records = [row for row in resets if row['world'] == world]
        episode, expected_age = 0, 0
        for frame in range(2048):
            assert episode < len(records), (world, frame, episode)
            assert force[frame, j] == records[episode]['force_n'][0], (world, frame, 'force')
            assert age[frame, j] == expected_age, (world, frame, 'age')
            expected_age += 1
            if terminal[frame, j]:
                episode += 1
                expected_age = 0
        assert len(records) == episode+1
    assert len(resets) == len(WORLDS)+int(terminal.sum())
    assert raw_counts.sum() == 24576 and abs(exposure.sum()-24576) < .01
    return dict(collections=32, actor_attempts=attempts, skipped_actor_minibatches=skipped,
                critic_minibatches=288, max_absolute_errors=error,
                raw_early_fraction=early_raw/24576, weighted_early_fraction=early_weighted/exposure.sum(),
                branch_order=['quiet', '+X', '-X'], raw_branch_counts=raw_counts.tolist(),
                weighted_branch_fractions=(exposure/exposure.sum()).tolist(),
                chronological_age_and_force_verified=True, critic_objective_unweighted=True)


def main():
    inputs, done, contract = (read(OUT/p) for p in ('inputs.json', 'completed.json', 'contract.json'))
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    assert read(OUT/'checks.json')['passed'] and read(OUT/'checks.json')['probability']['passed']
    assert read(OUT/'checks.json')['weighting']['passed']
    assert contract['sigma'] == SIGMA/2 and done['frozen_source_unchanged']
    assert contract['baseline_sigma'] == contract['sigma']
    assert contract['critic_weighted'] is False and contract['advantage_normalization_changed'] is False
    assert contract['reward_changed'] is False and contract['actor_weighting']['passed']
    assert contract['actor_weighting']['early_age_intervals'] == 30
    assert contract['actor_weighting']['raw_early_weight'] == 5.
    assert contract['actor_weighting']['raw_late_weight'] == 1.
    assert done['inputs_unchanged'] is True and read(FULL/'audit.json')['passed'] is True
    torch.set_num_threads(1)
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    data, jac, rotation = mujoco.MjData(model), np.zeros((3, model.nv)), np.zeros(9)
    feet = [model.body(f'Foot_{side}').id for side in ('L', 'R')]
    reference = arrays(FOUNDATION/'stationary_reference.npz')['qpos'].astype(np.float32)
    mass = float(model.body_mass.sum())
    dt, ankles = contract['dt'], contract['ankles']
    lower, upper = (np.array(contract['control'][f'target_{side}_radians']) for side in ('lower', 'upper'))
    reports, event_count, max_feature_error = {}, 0, 0.
    for path in sorted(OUT.glob('*/trace.npz')):
        z, r = arrays(path), read(path.with_name('report.json'))
        name = path.parent.name
        reports[name] = r
        if r['sampled']:
            policy = arrays(path.with_name('sampled-policy.npz'))
            assert float(policy['sigma']) == contract['sigma']
            actor = AnkleHead().eval()
            actor.load_state_dict(torch.load(OUT/f'fit-{name.split(chr(45))[0]}'/'final.pt', weights_only=True, map_location='cpu'))
            with torch.no_grad():
                means = actor(torch.from_numpy(policy['features64']).reshape(-1, 427)).numpy().reshape(64, 16, 1)
            assert np.allclose(means, policy['mean'][:64], atol=2e-6, rtol=0)
            rng = torch.Generator().manual_seed(700009)
            noise = torch.stack([contract['sigma']*torch.randn((16, 1), generator=rng) for _ in z['time']]).numpy()
            assert np.array_equal(z['latent'], policy['mean']+noise)
        assert all(np.isfinite(value).all() for value in z.values()) and not z['warning'].any()
        assert np.allclose(z['time'], (np.arange(len(z['time']))+1)*dt, atol=1e-10, rtol=0)
        assert len(z['time']) == int(np.ceil(r['seconds']/dt))
        assert not z['previous_latent'][0].any()
        assert np.array_equal(z['history_before'][0, :, 0], z['history_before'][0, :, 2])
        history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
        history[~z['alive'][:-1]] = z['history_before'][:-1][~z['alive'][:-1]]
        assert np.array_equal(z['history_before'][1:], history)
        memory = np.where(z['alive'][:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
        assert np.array_equal(z['previous_latent'][1:], memory)
        addition = z['previous'][:, :, ankles[:1]] + np.clip(.1*np.tanh(z['latent'])-z['previous'][:, :, ankles[:1]], -.02, .02)
        addition[~z['alive']] = z['previous'][:, :, ankles[:1]][~z['alive']]
        assert np.allclose(z['addition'][:, :, ankles[0]], addition[:, :, 0], atol=1e-7, rtol=0)
        assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
        assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
        assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), atol=1e-7, rtol=0)
        assert not z['active_target'][:2, :, -1].any()
        assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
        xy, clearance = foot_geometry(model, z['qpos'])
        assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
        assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
        for frame in sorted(set([0, len(z['time'])-1, *range(0, len(z['time']), 30)])):
            for i in range(16):
                data.qpos[:], data.qvel[:] = z['qpos'][frame, i], z['qvel'][frame, i]
                mujoco.mj_forward(model, data)
                mujoco.mj_jacSubtreeCom(model, data, jac, 1)
                velocity = jac@data.qvel
                mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                angular = rotation.reshape(3, 3)@data.qvel[3:6]
                com, fp = z['com'][frame, i], data.xpos[feet]
                center = fp[:, :2].mean(0)
                capture = com[:2]+velocity[:2]*np.sqrt(max(.05, com[2]-fp[:, 2].min())/9.81)
                expected = np.r_[(com[:2]-center)/.1, velocity[:2]/.5, angular/2, (capture-center)/.1,
                    z['loads'][frame, i]/(mass*9.81), (data.qpos[2]-reference[i % 8, 2])/.15,
                    1-2*(data.qpos[4]**2+data.qpos[5]**2), ((fp[:, :2]-com[:2])/.2).ravel()]
                max_feature_error = max(max_feature_error, float(np.abs(expected-z['balance_features'][frame, i]).max()))
        alive = z['alive'] & ~z['failed']
        assert z['alive'][0].all() and np.array_equal(z['alive'][1:], alive[:-1])
        for c in r['cases']:
            i = c['case']
            assert c['pose'] == i % 8 and c['heldout'] == (i % 8 >= 6)
            assert c['survived'] == bool(alive[:, i].all())
            assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/dt)):, i].all())
            assert c['whole_stand'] == bool(alive[:, i].all() and z['stand'][:, i].all())
            force = np.zeros_like(z['force'][:, i])
            if not r['quiet']:
                hit = int(round(c['hit_seconds']/dt))
                force[hit:hit+6] = c['force_n']
                force[~z['alive'][:, i]] = 0
            assert np.array_equal(force, z['force'][:, i])
            events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive[:, i],
                z['settled'][:, i], hit_time=c['hit_seconds'])
            assert events == c['steps']
            assert c['recovered_with_step'] == (c['settled'] and any(e['settled_window_start'] is not None for e in events))
            event_count += len(events)
        for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
            assert r[key] == sum(c[key] for c in r['cases'])
        if name.startswith('matched-'):
            desc = read(path.with_name('descriptor.json'))
            baseline = arrays(Path(desc['baseline'])/'trace.npz')
            issue = desc['issue_frame']
            end = len(z['time']) if desc['entire_trace_exact'] else issue
            assert all(np.array_equal(z[key][:end], baseline[key][:end]) for key in z)
            assert all(np.array_equal(z[key][:issue+2], baseline[key][:issue+2]) for key in ('qpos', 'qvel'))
    initial = torch.load(PILOT/'fit-921009'/'initial.pt', weights_only=True, map_location='cpu')
    fits, results, objective_checks = {}, {}, {}
    for seed in contract['training_seeds']:
        folder = OUT/f'fit-{seed}'
        resets, history, report, account = (read(folder/name) for name in ('resets.json', 'history.json', 'report.json', 'start-accounting.json'))
        assert report['sigma'] == contract['sigma'] and report['max_independent_kl_error'] < 2e-7
        assert report['weighting'] == contract['actor_weighting']
        for update in (1, 32):
            batch = arrays(folder/f'collection-{update}.npz')
            assert float(batch['sigma']) == contract['sigma']
            lp = (-.5*((batch['action'].astype(float)-batch['mean'])/contract['sigma'])**2-np.log(contract['sigma'])-.5*np.log(2*np.pi)).sum(-1)
            assert np.allclose(lp, batch['logp'], atol=3e-6, rtol=1e-6)
            gae = np.zeros((12,), dtype=float)
            independent = np.zeros((64, 12), dtype=float)
            for t in reversed(range(64)):
                delta = (batch['reward'][t].astype(float)+.995*batch['next_value'][t].astype(float)
                         *(~batch['failed'][t])-batch['value'][t].astype(float))
                gae = delta+.995*.98*(~batch['terminal'][t])*gae
                independent[t] = gae
            assert np.allclose(independent, batch['advantage'], atol=1e-4, rtol=2e-6)
            assert np.array_equal((batch['advantage']+batch['value']).flatten(), batch['returns'])
            stored = batch['advantage'].astype(float).ravel()
            normalized = (stored-stored.mean())/max(stored.std(ddof=1), 1e-6)
            assert np.allclose(normalized, batch['normalized_advantage'], atol=3e-6, rtol=2e-6)
            if update == 1:
                baseline = arrays(FULL/f'fit-{seed}'/'collection-1.npz')
                for key in baseline:
                    if key != 'post_mean':
                        assert np.array_equal(batch[key], baseline[key]), (seed, 'initial_collection', key)
            for t in range(64):
                fresh = batch['age'][t] == 0
                assert not batch['x'][t, fresh, -2:].any()
                if t:
                    assert np.array_equal(batch['x'][t, ~fresh, -1], batch['action'][t-1, ~fresh, 0])
        assert report['transitions'] == account['collected'] == 24576 and len(history) == 32
        assert all(r['pose'] < 6 and r['world'] % 8 == r['pose'] and r['onset_frame'] in contract['training_onset_frames']
                   and r['start_frame'] == r['onset_frame']+1 for r in resets)
        assert all(r['force_n'][1:] == [0., 0.] and r['force_n'][0] in (-150., 0., 150.) for r in resets)
        assert account['optimizer_prefix_transitions'] == 0
        assert account['reset_records'] == account['exact_transfers'] == len(resets)
        assert account['source_prefix_intervals'] == sum(r['start_frame'] for r in resets)
        assert not ({60, 102} & set(contract['training_onset_frames']))
        assert all(h['transitions'] == (i+1)*768 for i, h in enumerate(history))
        assert all(0 <= kl <= .01 for h in history for kl in h['accepted_kl'])
        saved = torch.load(folder/'initial.pt', weights_only=True, map_location='cpu')
        final = torch.load(folder/'final.pt', weights_only=True, map_location='cpu')
        assert all(torch.equal(value, saved[key]) for key, value in initial.items())
        assert any(not torch.equal(value, final[key]) for key, value in saved.items())
        objective_checks[str(seed)] = check_objectives(folder, resets, history)
        fits[str(seed)] = dict(transitions=account['collected'], prefix_intervals=account['source_prefix_intervals'],
            resets=len(resets), quiet_resets=sum(r['force_n'][0] == 0 for r in resets),
            updates=sum(h['actor_updates'] for h in history), rollbacks=sum(h['rolled_back'] for h in history),
            wall_seconds=report['wall_seconds'])
        result = {}
        for mode in ('continuous', 'matched'):
            groups = {}
            for group in range(4):
                names = [f'{seed}-t{t}-g{group}' if mode == 'continuous' else f'matched-{seed}-t{t}-g{group}' for t in (0, 1)]
                cases = [c for name in names for c in reports[name]['cases']]
                groups[str(group)] = dict(settled=sum(c['settled'] for c in cases),
                    heldout_settled=sum(c['settled'] and c['heldout'] for c in cases),
                    by_sign=[sum(c['settled'] for c in cases if (c['case'] < 8) == (s == 0)) for s in (0, 1)])
                recorded = done['results'][str(seed)][mode]['groups'][str(group)] if mode == 'continuous' else done['results'][str(seed)][mode][str(group)]
                assert groups[str(group)]['settled'] == recorded['settled']
                assert groups[str(group)]['heldout_settled'] == recorded['heldout_settled']
            quiet = reports[f'{seed}-quiet']['whole_stand'] if mode == 'continuous' else sum(reports[f'matched-{seed}-t{t}-quiet']['whole_stand'] for t in (0, 1))
            result[mode] = dict(groups=groups, total_settled=sum(g['settled'] for g in groups.values()), quiet=quiet)
        result['sampled_settled'] = sum(reports[f'{seed}-sampled-t{t}']['settled'] for t in (0, 1))
        results[str(seed)] = result
    assert len(reports) == 45 and len(reports)*16 == done['cases'] == 720
    assert max_feature_error < 2e-5 and done['total_transitions'] == 49152
    audit = dict(passed=True, traces=45, cases=720, measured_steps=event_count,
        input_hashes=len(inputs), output_hashes=len(done['outputs']),
        max_independent_balance_feature_error=max_feature_error, fits=fits, results=results,
        objectives=objective_checks, exact_first_collection_and_post_critic=True,
        advantage_normalization_unweighted=True)
    pairs = []
    for seed in contract['training_seeds']:
        for mode in ('continuous', 'matched'):
            for timing in (0, 1):
                for group in range(4):
                    name = f'{seed}-t{timing}-g{group}'
                    if mode == 'matched':
                        name = 'matched-'+name
                    for comparator, root, previous in [('unweighted-half-noise', FULL, name),
                            ('zero', PILOT, f'baseline-t{timing}-g{group}')]:
                        baseline = read(root/previous/'report.json')
                        for c, old in zip(reports[name]['cases'], baseline['cases']):
                            pairs.append(dict(seed=seed, mode=mode, timing=timing, group=group, case=c['case'],
                                heldout=c['heldout'], comparator=comparator, settled=c['settled'],
                                previous_settled=old['settled'], outcome='gain' if c['settled'] and not old['settled']
                                else 'loss' if old['settled'] and not c['settled'] else 'unchanged'))
    audit['probability_and_sampled_scale_verified'] = True
    save(OUT/'paired-outcomes.json', pairs)
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()

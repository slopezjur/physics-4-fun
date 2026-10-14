"""Matched fresh fits exposing bounded effort from committed delayed targets."""
from copy import copy
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .train_arm_relative_proposal import ProposalHead
from .train_arm_corrective_comparison import TEACHER, PRIOR_ROOT, ARENA, equal_weights
from .probe_arm_execution_distribution import setup, seal, episode
from .probe_arm_matched_feedback import OUT as BRANCHES, FOCUS, JOINTS
from .challenge_arm_proposals import HEADS, adapter_for, OutputProbe
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-pending-effect-20261008-01')
PLAN = Path(__file__).with_name('ARM_PENDING_EFFECT_PLAN.md')
SEEDS = (711028, 721028)
RECIPES = ('masked', 'pending_effect')
UPDATES, BATCH = 4000, 512
TIMELINE = (.9, 1.4, 3.2)
NOISE_SEEDS = (710083, 710084)


def pending_effects(x, qi, vi):
    slots = x[:, 95:229].reshape(-1, 2, 67)
    q = x[:, qi+8, None].transpose(1, 2)
    v = x[:, vi+48, None].transpose(1, 2)*10
    encoded = torch.zeros_like(slots)
    encoded[:, :, :33] = (4*(slots[:, :, :33]-q)+.08*(slots[:, :, 33:66]-v)).clamp(-1, 1)*slots[:, :, 66:]
    encoded[:, :, 66] = slots[:, :, 66]
    return encoded.flatten(1)


class PendingEffectHead(ProposalHead):
    def features(self, x):
        if self.kind == 'inherited':
            return super().features(x)
        if self.kind != 'pending_effect':
            raise ValueError(self.kind)
        z = ((x-self.mean)/self.scale).clamp(-20, 20).clone()
        z[:, 95:229] = pending_effects(x, self.qi, self.vi)
        return z


def load_head(folder):
    kind = json.loads((folder/'architecture.json').read_text())['kind']
    s = torch.load(folder/'final.pt', weights_only=True)
    net = PendingEffectHead(s['mean'], s['scale'], s['output_scale'], s['qi'], s['vi'], kind)
    net.load_state_dict(s)
    return net.eval().requires_grad_(False)


@torch.no_grad()
def validate_features(env, base, data):
    c, model = env.engine.policy_control, env.rig.model
    assert c.gain == 4 and c.damping_time == .02 and len(c.pending) == 2
    x = torch.tensor(data['x'])
    torch.manual_seed(SEEDS[0])
    masked = PendingEffectHead(base.mean, base.scale, base.output_scale, c.q_indices, c.v_indices, 'inherited')
    torch.manual_seed(SEEDS[0])
    candidate = PendingEffectHead(base.mean, base.scale, base.output_scale, c.q_indices, c.v_indices, 'pending_effect')
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in masked.state_dict().items())
    a, b = masked.features(x), candidate.features(x)
    assert torch.equal(a[:, :95], b[:, :95]) and torch.equal(a[:, 229:], b[:, 229:])
    assert torch.equal(masked(x), candidate(x)) and not candidate(x).any()
    encoded = b[:, 95:229].reshape(-1, 2, 67)
    slots = x[:, 95:229].reshape(-1, 2, 67)
    assert torch.isfinite(b).all() and (encoded.abs() <= 1).all()
    assert ((slots[:, :, 66] == 0) | (slots[:, :, 66] == 1)).all()
    assert not encoded[slots[:, :, 66] == 0].any() and not encoded[:, :, 33:66].any()
    branch_report = json.loads((BRANCHES/'report.json').read_text())
    branch_protocol = json.loads((BRANCHES/'protocol.json').read_text())
    origins = [r for r in branch_report['results'] if r['config']['pulse'] is None
               and r['config']['origin'] == r['config']['destination']]
    rows, parity, inputs, expected = [], [], [], []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    original_pending = c.pending.clone()
    for row in origins:
        folder = BRANCHES/row['folder']
        states, commands = np.load(folder/'switch-states.npz'), np.load(folder/'commands.npz')
        for world in FOCUS:
            state = {k.split('__', 1)[1]: states[k] for k in states.files if k.startswith(f'{world}__')}
            frame = row['switches'][world]
            physical = env.engine.datas[world]
            mujoco.mj_setState(model, physical, state['integration'], spec)
            warmstart = physical.qacc_warmstart.copy()
            mujoco.mj_forward(model, physical)
            physical.qacc_warmstart[:] = warmstart
            before = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, physical, before, spec)
            assert np.array_equal(before, state['integration'])
            obs = torch.tensor(commands['x'][frame, world:world+1])
            q, v = (torch.tensor(getattr(physical, name)[None], dtype=torch.float32) for name in ('qpos', 'qvel'))
            assert torch.allclose(obs[:, c.q_indices+8], q[:, c.q_indices], atol=1e-7, rtol=0)
            assert torch.allclose(obs[:, c.v_indices+48], v[:, c.v_indices]*.1, atol=1e-7, rtol=0)
            chronological = np.stack([state['pending'][(int(state['cursor'])+i) % 2] for i in range(2)])
            assert np.array_equal(obs[:, 95:229].numpy().reshape(2, 67), chronological)
            baseline = pending_effects(obs, c.q_indices, c.v_indices)
            for joint, sign in [(None, 0)]+[(j, s) for j in JOINTS for s in (-1, 1)]:
                altered = obs.clone()
                if joint is not None:
                    motor = branch_protocol['motors'][joint]
                    altered[:, 95+motor] = (altered[:, 95+motor]+sign*.005).clamp(c.lower[motor], c.upper[motor])
                native = copy(c)
                native.ctrl = torch.zeros((1, model.nu))
                native_effects = torch.zeros((1, 2, 67))
                for slot in range(2):
                    native.active = altered[:, 95+67*slot:95+67*(slot+1)].clone()
                    native_effects[:, slot, :33] = native.physics_control(q, v)[:, native.indices]/native.scale
                    native_effects[:, slot, 66] = native.active[:, 66]
                actual = pending_effects(altered, c.q_indices, c.v_indices)
                error = float((actual-native_effects.flatten(1)).abs().max())
                assert error < 2e-6
                inputs.append(altered.numpy()[0]);expected.append(native_effects.flatten(1).numpy()[0])
                rows.append(dict(folder=folder.name, world=world, frame=frame, joint=joint, sign=sign,
                    native_max_error=error, changed_features=not torch.equal(actual, baseline)))
            after = np.empty_like(before)
            mujoco.mj_getState(model, physical, after, spec)
            assert np.array_equal(before, after) and torch.equal(c.pending, original_pending)
            parity.append(dict(folder=folder.name, world=world, frame=frame))
    assert len(parity) == 24 and len(rows) == 120
    np.savez_compressed(OUT/'feature-probes.npz', x=np.stack(inputs), native_effects=np.stack(expected))
    report = dict(passed=True, switch_states=24, probes=rows, initial_outputs_identical=True,
        invalid_slots_zero=True, outside_queue_identical=True, original_control_unchanged=True,
        saturation_fraction=float((encoded[:, :, :33][slots[:, :, 66] == 1].abs() == 1).float().mean()),
        changed_counterfactuals=sum(r['changed_features'] for r in rows if r['joint'] is not None),
        native_max_error=max(r['native_max_error'] for r in rows), training_labels_added=0)
    save(OUT/'feature-validation.json', report)
    print('FEATURE ADMISSION', report['native_max_error'], 'changed', report['changed_counterfactuals'], '/96', flush=True)


def fit(base, control, data, recipe, seed, folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    kind = 'inherited' if recipe == 'masked' else 'pending_effect'
    net = PendingEffectHead(base.mean, base.scale, base.output_scale, control.q_indices, control.v_indices, kind)
    folder.mkdir();torch.save(net.state_dict(), folder/'initial.pt');net.to(device).train()
    mask = ~data['validation']
    assert set(data['world'][mask]) == set(range(6))
    x = torch.tensor(data['x'][mask], device=device)
    y = torch.tensor((data['raw_target']-data['goal'])[mask], device=device)
    vx = torch.tensor(data['x'][~mask], device=device)
    vy = torch.tensor((data['raw_target']-data['goal'])[~mask], device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    start, history, draws = time.perf_counter(), [], hashlib.sha256()
    for update in range(UPDATES):
        ids = torch.randint(len(x), (BATCH,), generator=generator, device=device)
        draws.update(ids.cpu().numpy().tobytes())
        loss = ((net(x[ids])-y[ids])/net.output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (update+1) % 500 == 0:
            history.append(dict(update=update+1, loss=float(loss.detach())))
            print('FIT', recipe, seed, history[-1], flush=True)
    net.eval()
    with torch.no_grad():
        train_error = float((net(x)-y).square().mean().sqrt())
        heldout_error = float((net(vx)-vy).square().mean().sqrt())
    net.cpu();torch.save(net.state_dict(), folder/'final.pt')
    save(folder/'architecture.json', dict(kind=kind, seed=seed, inputs=421,
        active_features=287 if recipe == 'masked' else 355,
        architecture='421-128ELU-128ELU-33, zero final layer, inherited per-motor output scale',
        queue_contract='current-state normalized PD effort and validity, oldest first' if recipe != 'masked' else 'masked'))
    save(folder/'history.json', history)
    result = dict(recipe=recipe, seed=seed, updates=UPDATES, batch_size=BATCH, lr=1e-4,
        train_examples=len(x), heldout_examples=len(vx), draw_hash=draws.hexdigest(),
        train_rmse=train_error, heldout_rmse=heldout_error, device=device, wall_seconds=time.perf_counter()-start)
    save(folder/'report.json', result)
    return net, result


def scores(results, challenge):
    return {recipe: [min(r['complete'] for r in results if r['config']['recipe'] == recipe
                        and r['config']['challenge'] == challenge),
                     sum(r['complete'] for r in results if r['config']['recipe'] == recipe
                         and r['config']['challenge'] == challenge)] for recipe in RECIPES}


def main():
    assert json.loads((BRANCHES/'audit.json').read_text())['passed']
    baseline_report = json.loads((PRIOR_ROOT/'step-29/report.json').read_text())
    baseline_path = PRIOR_ROOT/'step-29'/baseline_report['results'][0]['folder']/'trace.npz'
    branches = json.loads((BRANCHES/'report.json').read_text())
    extra = [PLAN, ARENA, TEACHER/'training-samples.npz', baseline_path, BRANCHES/'audit.json',
             BRANCHES/'report.json', BRANCHES/'protocol.json']
    for row in branches['results']:
        if row['config']['pulse'] is None and row['config']['origin'] == row['config']['destination']:
            extra += [BRANCHES/row['folder']/'switch-states.npz', BRANCHES/row['folder']/'commands.npz']
    for seed in SEEDS:
        extra += [HEADS[f'all-{seed}']/'final.pt']
    inputs = seal(OUT, extra)
    save(OUT/'protocol.json', dict(inputs=inputs, seeds=SEEDS, recipes=RECIPES, updates=UPDATES,
        batch_size=BATCH, quiet_baseline=str(baseline_path), timeline=TIMELINE, noise_seeds=NOISE_SEEDS,
        training_poses=list(range(6)), heldout_poses=[6, 7], teacher_queries=0, physical_cases=192))
    env, anchor, base, poses, actors = setup()
    original = dict(np.load(TEACHER/'training-samples.npz'))
    validate_features(env, base, original)
    training, results = [], []
    for seed in SEEDS:
        for recipe in RECIPES:
            folder = OUT/f'{recipe}-{seed}'
            net, tr = fit(base, env.engine.policy_control, original, recipe, seed, folder)
            if recipe == 'masked':
                assert equal_weights(folder/'final.pt', HEADS[f'all-{seed}']/'final.pt')
            else:
                assert equal_weights(folder/'initial.pt', OUT/f'masked-{seed}'/'initial.pt')
                assert tr['draw_hash'] == training[-1]['draw_hash']
            training.append(tr)
            configs = [dict(label='nominal', variant=False, challenge=False),
                       dict(label='varied', variant=True, challenge=False),
                       dict(label='new-timing', variant=False, challenge=True, timeline=TIMELINE)]
            configs += [dict(label=f'noise-{s}', variant=False, challenge=True, proposal_std=.005, noise_seed=s) for s in NOISE_SEEDS]
            for config in configs:
                config.update(actor=folder.name, recipe=recipe, seed=seed, mode='mean', teacher=False)
                proposal = OutputProbe(net, env, std=config.get('proposal_std', 0.), seed=config.get('noise_seed', 710081))
                adapter = (CorrectionEnvironment(env, anchor, proposal, poses, variant=True) if config['variant']
                           else adapter_for(env, anchor, proposal, poses, timeline=config.get('timeline')))
                name = folder.name+'-'+config['label']
                r = episode(adapter, actors['zero'], config, OUT/name)
                if proposal.noises:
                    np.savez_compressed(OUT/name/'noise.npz', noise=np.stack(proposal.noises))
                results.append(dict(folder=name, **r))
                save(OUT/'progress.json', dict(training=training, results=results))
    from .run_single_support_feedback import make_environment
    env = make_environment(env.rig, anchor, 80.)
    baseline = np.load(baseline_path)
    for seed in SEEDS:
        net = load_head(OUT/f'pending_effect-{seed}')
        name = f'pending_effect-{seed}-quiet60'
        r = episode(adapter_for(env, anchor, net, poses, quiet_seconds=60.), actors['zero'],
            dict(actor=name, recipe='pending_effect', seed=seed, quiet=True, challenge='quiet', mode='mean', teacher=False), OUT/name)
        z = np.load(OUT/name/'trace.npz')
        assert r['quiet_passed'] == 16 and all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
        results.append(dict(folder=name, quiet_exact=True, **r))
        save(OUT/'progress.json', dict(training=training, results=results))
    core, challenge = scores(results, False), scores(results, True)
    improved = core['pending_effect'] > core['masked'] and challenge['pending_effect'] >= challenge['masked']
    admitted = core['pending_effect'] == [8, 32] and challenge['pending_effect'] == [8, 48]
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(training=training, results=results, core_scores=core, challenge_scores=challenge,
        research_improved=improved, admitted=admitted, selected='pending_effect' if admitted else 'masked',
        inputs_unchanged=True, arena_changed=False,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', core, challenge, 'admitted', admitted, flush=True)


if __name__ == '__main__':
    main()

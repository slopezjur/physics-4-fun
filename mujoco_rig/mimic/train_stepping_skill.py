"""Bounded independent-joint learning for commanded stepping prerequisites."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from . import run_direct_recovery as prior
from .corrective_step_audit import foot_geometry, measure_steps
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .stepping_skill_core import StepSkillEnvironment, SteppingHead

OUT = Path('logs/mimickit-training/stepping-skill-20261009-01')
SEEDS = (941009, 951009)
INITIAL_SEED = 940009
TRAIN = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
GAMMA, GAE_LAMBDA = .995, .98
SIGMA = .3 * (1 - .9**2)**.5
COLLECTIONS, HORIZON = 64, 64


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def setup(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    audit_tools = {'verify_stepping_skill.py', 'test_stepping_curriculum.py', 'test_stepping_skill_core.py'}
    sources = [path for path in sorted(Path(__file__).parent.glob('*.py')) if path.name not in audit_tools]
    sources.append(Path(__file__).with_name('STEPPING_SKILL_PLAN.md'))
    for path in sources:
        shutil.copy2(path, out/'sources'/path.name)
    inputs = sources + [PARENT, PARENT.with_suffix('.amp.pt'),
        FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
        FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json']
    for row in read(FOUNDATION/'protocol.json')['demonstrations']:
        path = Path(row['path'])
        assert digest(path) == row['sha256'], path
        inputs.append(path)
    hashes = {str(path): digest(path) for path in inputs}
    save(out/'inputs.json', hashes)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
                        read(FOUNDATION/'protocol.json'), PARENT)
    assert not any(parameter.requires_grad for parameter in anchor.parameters())
    return rig, anchor, hashes


def advantages(batch):
    """Finite task deadlines terminate returns, including successful deadlines."""
    advantage = torch.zeros_like(batch['reward'])
    running = torch.zeros(len(TRAIN))
    for frame in reversed(range(HORIZON)):
        continuation = ~batch['terminal'][frame]
        delta = (batch['reward'][frame] + GAMMA*batch['next_value'][frame]*continuation
                 - batch['value'][frame])
        running = delta + GAMMA*GAE_LAMBDA*continuation*running
        advantage[frame] = running
    returns = advantage + batch['value']
    normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    return advantage, returns, normalized


def learn(env, initial, folder, seed, level):
    folder.mkdir(parents=True)
    torch.manual_seed(seed + 1000*level)
    rng = torch.Generator().manual_seed(seed + 1000*level + 1)
    actor = deepcopy(initial).train().requires_grad_(True)
    critic = SteppingHead(input_width=env.features().shape[-1], outputs=1)
    ao = torch.optim.Adam(actor.parameters(), lr=1e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
    torch.save(actor.state_dict(), folder/'initial.pt')
    env.reset(rng=rng, level=level)
    active = torch.zeros(16, dtype=torch.bool)
    active[TRAIN] = True
    history, start = [], time.perf_counter()
    for update in range(COLLECTIONS):
        rows = []
        for _ in range(HORIZON):
            with torch.no_grad():
                x = env.features()
                mean = actor(x)
                action = mean + SIGMA*torch.randn(mean.shape, generator=rng)
                reward, terminal, info = env.step(action, active)
                rows.append(dict(x=x[TRAIN], action=action[TRAIN], mean=mean[TRAIN],
                    logp=Normal(mean, SIGMA).log_prob(action).sum(-1)[TRAIN],
                    value=critic(x)[TRAIN], reward=reward[TRAIN],
                    terminal=terminal[TRAIN], failed=info['failed'][TRAIN],
                    next_value=critic(env.features())[TRAIN]))
                ids = torch.nonzero(active & terminal).flatten()
                if len(ids):
                    env.reset(ids, rng=rng, level=level)
        batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        advantage, returns, av = advantages(batch)
        if update in (0, COLLECTIONS-1):
            tag = 'first' if update == 0 else 'last'
            np.savez_compressed(folder/f'{tag}-collection.npz',
                **{key: value.numpy() for key, value in batch.items()},
                advantage=advantage.numpy(), returns=returns.numpy(),
                normalized_advantage=av.numpy(), worlds=TRAIN.numpy(), sigma=np.array(SIGMA))
        flat = {key: value.flatten(0, 1) for key, value in batch.items()}
        target, av = returns.flatten(), av.flatten()
        accepted, rejected, stopped, kls = 0, 0, False, []
        for _ in range(3):
            for ids in torch.randperm(len(av), generator=rng).split(256):
                if not stopped:
                    old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(ao.state_dict())
                    logp = Normal(actor(flat['x'][ids]), SIGMA).log_prob(flat['action'][ids]).sum(-1)
                    ratio = (logp-flat['logp'][ids]).exp()
                    loss = -torch.minimum(ratio*av[ids], ratio.clamp(.8, 1.2)*av[ids]).mean()
                    ao.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    ao.step()
                    with torch.no_grad():
                        kl = float(((actor(flat['x'])-flat['mean']).square().sum(-1)/(2*SIGMA**2)).mean())
                    if not np.isfinite(kl) or kl > .01:
                        actor.load_state_dict(old_actor)
                        ao.load_state_dict(old_optimizer)
                        stopped, rejected = True, rejected+1
                    else:
                        accepted += 1
                        kls.append(kl)
                value_loss = (critic(flat['x'][ids])-target[ids]).square().mean()
                co.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                co.step()
        assert torch.isfinite(target).all() and torch.isfinite(value_loss)
        history.append(dict(update=update+1, transitions=(update+1)*HORIZON*len(TRAIN),
            reward=float(batch['reward'].mean()), failures=int(batch['failed'].sum()),
            completed=int(batch['terminal'].sum()), actor_updates=accepted,
            rolled_back=rejected, accepted_kl=kls, critic_loss=float(value_loss.detach())))
        save(folder/'history.json', history)
        if (update+1) % 8 == 0:
            print('TRAIN', seed, level, {k: v for k, v in history[-1].items() if k != 'accepted_kl'}, flush=True)
    torch.save(actor.state_dict(), folder/'final.pt')
    torch.save(critic.state_dict(), folder/'critic-final.pt')
    save(folder/'report.json', dict(seed=seed, level=level, train_worlds=TRAIN.tolist(),
        transitions=COLLECTIONS*HORIZON*len(TRAIN), collections=COLLECTIONS,
        accepted_actor_updates=sum(row['actor_updates'] for row in history),
        actor_rollbacks=sum(row['rolled_back'] for row in history),
        wall_seconds=time.perf_counter()-start, episodic_terminal_bootstrap=False,
        initial_hash=digest(folder/'initial.pt'), final_hash=digest(folder/'final.pt')))
    return actor.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(rig, anchor, actor, folder, *, level, onset=1., distance=.04, quiet=False):
    folder.mkdir()
    seconds = 60. if quiet else 12.
    env = StepSkillEnvironment(rig, anchor, seconds=seconds)
    env.reset(level=level, side=torch.arange(16)//8, distance=distance,
              onset=onset, enabled=not quiet)
    active = torch.ones(16, dtype=torch.bool)
    rows = []
    for frame in range(int(np.ceil(seconds/env.dt))):
        before = active.clone()
        reward, terminal, info = env.step(actor(env.features()), active)
        rows.append(dict(time=np.array((frame+1)*env.dt), alive=before.numpy().copy(),
            reward=reward.numpy().copy(), terminal=terminal.numpy().copy(),
            **{key: value.detach().numpy().copy() for key, value in info.items()
               if isinstance(value, torch.Tensor) and key not in ('reward', 'terminal')}))
        active &= ~terminal
    trace = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert not trace['warning'].any(), 'Native solver warning invalidates this evaluation'
    xy, clearance = foot_geometry(rig.model, trace['qpos'])
    assert np.allclose(xy, trace['xy'], atol=2e-6, rtol=0)
    assert np.allclose(clearance, trace['clearance'], atol=2e-6, rtol=0)
    cases = []
    tail = int(np.ceil(2/env.dt))
    for case in range(16):
        alive = trace['alive'][:, case] & ~trace['physical_failed'][:, case]
        survived = bool(alive.all())
        final_stand = bool(survived and trace['settled'][-tail:, case].all())
        events = measure_steps(trace['time'], xy[:, case], clearance[:, case],
            trace['loads'][:, case], alive, trace['settled'][:, case],
            hit_time=None if quiet else onset)
        step = final_stand and any(event['settled_window_start'] is not None for event in events)
        success = bool(survived and final_stand and trace['task_success'][-1, case])
        if level == 2 and not quiet:
            success &= step
        cases.append(dict(case=case, pose=case % 8, side=case//8, level=level,
            heldout=case % 8 >= 6, survived=survived, success=success,
            whole_stand=bool(survived and trace['stand'][:, case].all()),
            final_stand=final_stand, events=events, recovered_with_step=bool(step),
            phase_max=int(trace['phase_after'][:, case].max())))
    report = dict(level=level, onset_seconds=onset, distance_m=distance, quiet=quiet,
        seconds=seconds, dt=env.dt, cases=cases, successes=sum(row['success'] for row in cases),
        survived=sum(row['survived'] for row in cases),
        whole_stand=sum(row['whole_stand'] for row in cases),
        final_stand=sum(row['final_stand'] for row in cases),
        recovered_with_step=sum(row['recovered_with_step'] for row in cases),
        command_window='Command onset, not projectile contact',
        phase_labels=['QUIET', 'WAIT', 'TRANSFER', 'LIFT', 'PLACE', 'LAND', 'SETTLE'])
    np.savez_compressed(folder/'trace.npz', **trace)
    save(folder/'report.json', report)
    print('EVAL', folder.name, {key: value for key, value in report.items() if key != 'cases'}, flush=True)
    return report


def admission(reports):
    by_condition = []
    for report in reports:
        train = [row for row in report['cases'] if not row['heldout']]
        counts = [sum(row['success'] for row in train if row['side'] == side) for side in (0, 1)]
        by_condition.append(dict(training_successes=sum(counts), by_side=counts,
            passed=sum(counts) >= 9 and min(counts) >= 4))
    return dict(passed=all(row['passed'] for row in by_condition), conditions=by_condition,
                reserved_cases_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    rig, anchor, hashes = setup(args.out)
    out, start = args.out, time.perf_counter()
    env = StepSkillEnvironment(rig, anchor)
    torch.manual_seed(INITIAL_SEED)
    initial = SteppingHead(input_width=env.features().shape[-1]).eval()
    assert not initial(env.features()).any()
    contract = dict(schema='commanded_stepping_pilot_v1', task=env.contract(),
        input_width=env.features().shape[-1], action_outputs=33, initial_seed=INITIAL_SEED,
        seeds=list(SEEDS), sigma=SIGMA, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        collections_per_level=COLLECTIONS, horizon=HORIZON, train_worlds=TRAIN.tolist(),
        epochs=3, minibatch=256, actor_lr=1e-4, critic_lr=3e-4, kl_limit=.01,
        episodic_terminal_bootstrap=False, maximum_transitions=294912,
        admission=dict(training_cases=12, minimum_success=9, minimum_each_side=4,
            quiet_cases=16, quiet_seconds=60, heldout_in_decision=False),
        curriculum=asdict(env.curriculum.config))
    save(out/'contract.json', contract)
    prior.rollout(prior.Recovery(rig, anchor, bare=True, seconds=12.), None,
                  out/'parity-bare', quiet=True, seconds=12.)
    evaluate(rig, anchor, initial, out/'parity-zero', level=0)
    with np.load(out/'parity-bare/trace.npz') as bare, np.load(out/'parity-zero/trace.npz') as zero:
        fields = ['qpos', 'qvel', 'issued', 'active_target', 'addition', 'nominal', 'force']
        assert all(np.array_equal(bare[key], zero[key]) for key in fields)
        assert np.array_equal(bare['failed'], zero['physical_failed'])
    save(out/'parity.json', dict(exact=True, cases=16, seconds=12., fields=fields,
        source_targets_rescaled=False, command_enabled=True))
    actors = {seed: deepcopy(initial) for seed in SEEDS}
    summary = dict(levels=[], status='running', total_transitions=0)
    for level in range(3):
        distances = (.04, -.04) if level == 2 else (.04,)
        stage = dict(level=level, baseline=[], candidates={})
        for distance in distances:
            tag = 'positive' if distance > 0 else 'negative'
            stage['baseline'].append(evaluate(rig, anchor, initial,
                out/f'eval-baseline-l{level}-t0-d{tag}', level=level, distance=distance))
        stage['baseline_quiet'] = evaluate(rig, anchor, initial,
            out/f'quiet-baseline-l{level}', level=level, quiet=True)
        assert stage['baseline_quiet']['whole_stand'] == 16, 'Original Stand failed the quiet prerequisite'
        for seed in SEEDS:
            actors[seed] = learn(StepSkillEnvironment(rig, anchor), actors[seed],
                out/f'fit-{seed}'/f'level-{level}', seed, level)
            summary['total_transitions'] += COLLECTIONS*HORIZON*len(TRAIN)
            nominal = []
            for distance in distances:
                tag = 'positive' if distance > 0 else 'negative'
                nominal.append(evaluate(rig, anchor, actors[seed],
                    out/f'eval-{seed}-l{level}-t0-d{tag}', level=level, distance=distance))
            quiet = evaluate(rig, anchor, actors[seed], out/f'quiet-{seed}-l{level}', level=level, quiet=True)
            gate = admission(nominal)
            gate['passed'] &= quiet['whole_stand'] == 16
            stage['candidates'][str(seed)] = dict(nominal=nominal, quiet=quiet, nominal_gate=gate)
        stage['nominal_passed'] = all(row['nominal_gate']['passed'] for row in stage['candidates'].values())
        stage['advanced'] = False
        if stage['nominal_passed']:
            for seed in SEEDS:
                varied = []
                for distance in distances:
                    tag = 'positive' if distance > 0 else 'negative'
                    varied.append(evaluate(rig, anchor, actors[seed],
                        out/f'eval-{seed}-l{level}-t1-d{tag}', level=level, onset=1.7, distance=distance))
                stage['candidates'][str(seed)]['varied'] = varied
                stage['candidates'][str(seed)]['varied_gate'] = admission(varied)
            stage['advanced'] = all(row['varied_gate']['passed'] for row in stage['candidates'].values())
        summary['levels'].append(stage)
        save(out/'summary.json', summary)
        if not stage['advanced']:
            summary['status'] = 'stopped_at_failed_curriculum_gate'
            break
    else:
        summary['status'] = 'commanded_step_pilot_passed_pending_independent_audit'
    assert all(digest(Path(path)) == sha for path, sha in hashes.items())
    summary['arena_changed'] = False
    save(out/'summary.json', summary)
    save(out/'completed.json', dict(wall_seconds=time.perf_counter()-start,
        sources_unchanged=True, total_transitions=summary['total_transitions'],
        status=summary['status'], outputs={str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob('*')) if path.is_file() and 'sources' not in path.parts}))
    print('COMPLETE', summary['status'], summary['total_transitions'], flush=True)


if __name__ == '__main__':
    main()

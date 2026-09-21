"""Bounded CPU trajectory search from confirmed head/chest impact states.

Offline full-state oracle, not a game controller or a training configuration.
One fixed CEM/MPC setting; no pre-impact intervention and no additional body forces.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import mujoco
from mujoco.rollout import Rollout
import numpy as np
import torch

import benchmark_perturb as benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rl'))
from perturb_env import PerturbEnv
from ppo import ActorCritic
from env_config import FALL_FRACTION, QVEL_CEILING
from recovery_metrics import HeightProxyRecoveryMetrics, RecoveryMetrics

ROOT = Path(__file__).resolve().parents[2]
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
CONTROL = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED
HISTORY = ('prev_action', 'episode_length_buf', 'grounded', 'since_landing',
           'rapid_replants', 'settle_hold', 'ball_flight', 'shots_fired', 'commands')


@dataclass(frozen=True)
class SearchConfig:
    horizon: int = 60
    commit: int = 6
    optimize_steps: int = 120
    population: int = 96
    iterations: int = 4
    knots: int = 7
    sigma: float = .15
    elite: int = 12
    threads: int = 6
    seconds_per_case: float = 80


class Snapshot:
    def __init__(self, env, index=0):
        self.model = env.model
        self.data = mujoco.MjData(env.model)
        mujoco.mj_copyData(self.data, env.model, env.datas[index])
        self.history = {}
        for key in HISTORY:
            value = getattr(env, key)[index:index+1]
            self.history[key] = (value.detach().cpu().numpy() if torch.is_tensor(value) else value).copy()
        self.queue = [x[index:index+1].copy() for x in env.action_queue]

    def restore(self, env):
        # Copy derived quantities too: an extra forward call changes observation timing.
        mujoco.mj_copyData(env.datas[0], env.model, self.data)
        for key, value in self.history.items():
            target = getattr(env, key)
            target[:] = torch.as_tensor(value) if torch.is_tensor(target) else value
        env.action_queue = [x.copy() for x in self.queue]
        env.next_ball[:] = np.inf
        env.auto_reset = False

    def save(self, path):
        state = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(path, integration_state=state, queue=np.stack(self.queue), **self.history)


def load_actor(checkpoint):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = ActorCritic(ck['num_obs'], ck['num_actions']).eval()
    net.load_state_dict(ck['model'])
    return net, ck['num_obs']


def action_of(net, width, env):
    with torch.no_grad():
        return net.actor(env.get_observations()[:, :width]).clamp(-1, 1)


def select_cases(bank):
    cases = []
    protocol = json.loads((bank/'protocol.json').read_text())
    repeats, shards = protocol['repeats'], protocol['shards']
    for body in range(2):
        for heading in range(4):
            start = (body*4+heading)*repeats
            for trial in range(start, start+repeats):
                shard = (trial-start) % shards
                record = json.loads((bank/'selection'/f'accepted-{shard:02d}.json').read_text())
                local = record['indices'].index(trial)
                if not record['survived'][local]:
                    cases.append(dict(trial=trial, shard=shard, local=local,
                                      target=protocol['target_bones'][body], heading=protocol['approach_sides'][heading]))
                    break
            else:
                raise ValueError('No recorded fall in a requested stratum')
    return cases


def capture_cases(bank, checkpoint, cases, out):
    """Replay original batch shapes; instrument contact reads without changing steps."""
    snapshots, suffixes = {}, {}
    for shard in sorted({c['shard'] for c in cases}):
        wanted = {c['local']: c for c in cases if c['shard'] == shard}
        scenario = dict(np.load(bank/'selection'/f'scenarios-{shard:02d}.npz'))
        original_class, original_step = benchmark.PerturbEnv, mujoco.mj_step
        pending, captured = {}, set()
        actions = {i: [] for i in wanted}
        env_holder = []

        class Recorder(PerturbEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                env_holder.append(self)

            def step(self, action):
                was_captured = set(captured)
                result = super().step(action)
                for i in was_captured:
                    actions[i].append(action[i].detach().numpy().clip(-1, 1).copy())
                for i in pending.keys() - captured:
                    snapshots[wanted[i]['trial']] = Snapshot(self, i)
                    wanted[i].update(contact_body=pending[i], contact_control_step=int(self.episode_length_buf[i])-1)
                    captured.add(i)
                return result

        def observe_step(model, data):
            original_step(model, data)
            if not env_holder:
                return
            env = env_holder[0]
            i = data_ids.get(id(data))
            if i is None or i in pending:
                return
            force = np.zeros(6)
            for contact_id, contact in enumerate(data.contact):
                bodies = model.geom_bodyid[contact.geom]
                if env.ball not in bodies or 0 in bodies:
                    continue
                mujoco.mj_contactForce(model, data, contact_id, force)
                if force[0] > 0:
                    other = int(bodies[0] if bodies[1] == env.ball else bodies[1])
                    pending[i] = model.body(other).name
                    break

        # Build the ID map when the factory has constructed all native data objects.
        data_ids = {}
        def factory(*args, **kwargs):
            env = Recorder(*args, **kwargs)
            data_ids.update({id(env.datas[i]): i for i in wanted})
            return env
        benchmark.PerturbEnv, mujoco.mj_step = factory, observe_step
        try:
            result = benchmark.run_policy(checkpoint, scenario)
        finally:
            benchmark.PerturbEnv, mujoco.mj_step = original_class, original_step
        saved = json.loads((bank/'selection'/f'accepted-{shard:02d}.json').read_text())
        for key, value in result.items():
            np.testing.assert_array_equal(value, saved[key], err_msg=f'Prefix replay differs: {key}')
        if captured != set(wanted):
            raise RuntimeError('A selected shot has no confirmed body contact')
        for i, case in wanted.items():
            trial = case['trial']
            suffixes[trial] = np.array(actions[i])
            snapshots[trial].save(out/f'case-{trial}-initial.npz')
            print(f'captured {case}', flush=True)
    return snapshots, suffixes


def native_controls(env, commands):
    """Exactly the production two-command queue, torque scale, and ball force schedule."""
    commands = np.clip(commands, -1, 1).astype(np.float32)
    batch, horizon, _ = commands.shape
    queued = np.broadcast_to(np.stack(env.action_queue)[:, 0], (batch, len(env.action_queue), env.num_actions))
    applied = np.concatenate((queued, commands), axis=1)[:, :horizon]
    control = np.zeros((batch, horizon, mujoco.mj_stateSize(env.model, CONTROL)))
    control[:, :, env.act_idx] = applied * env.force_limit * env.authority
    t = (int(env.episode_length_buf[0]) + np.arange(horizon)) * env.dt * env.decimation
    control[:, :, env.model.nu + env.ball * 6 + 2] = np.where(t < env.ball_flight[0], env.ball_gravity_cancel, 0)
    return np.repeat(control, env.decimation, axis=1)


def initial_state(env):
    state = np.empty(mujoco.mj_stateSize(env.model, FULL))
    mujoco.mj_getState(env.model, env.datas[0], state, FULL)
    return state


def physical_cost(env, states, commands):
    """Offline search surrogate, separate from the unchanged RL reward and final gates."""
    q = states[:, env.decimation-1::env.decimation, 1:1+env.model.nq]
    v = states[:, env.decimation-1::env.decimation, 1+env.model.nq:1+env.model.nq+env.model.nv]
    up = 1 - 2*(q[:, :, 4]**2+q[:, :, 5]**2)
    height = q[:, :, 2]/env.rest_pelvis_z
    stage = (40*(1-up)**2 + 40*(height-1)**2 + 2*np.sum(v[:, :, :2]**2, axis=-1)
             + .3*np.sum(v[:, :, 3:6]**2, axis=-1))
    terminal = 60*(1-up[:, -1])**2 + 60*(height[:, -1]-1)**2
    terminal += 10*np.sum(v[:, -1, :3]**2, axis=-1) + 1.5*np.sum(v[:, -1, 3:6]**2, axis=-1)
    fallen = (height < FALL_FRACTION) | (up < .5)
    invalid = (~np.isfinite(states).all(axis=(1, 2)) | (np.abs(v[:, :, :env._body_nv]).max(axis=(1, 2)) > QVEL_CEILING))
    cost = stage.mean(1) + terminal + 10000*fallen.mean(1) + .01*np.mean(np.diff(commands, axis=1)**2, axis=(1, 2))
    return np.where(invalid, 1e12, np.nan_to_num(cost, nan=1e12, posinf=1e12))


class ShootingController:
    def __init__(self, env, net, width, config, seed):
        self.config, self.net, self.width = config, net, width
        self.pool = Rollout(nthread=config.threads)
        self.datas = [mujoco.MjData(env.model) for _ in range(config.threads)]
        self.predictor = PerturbEnv(num_envs=1, episode_seconds=20, observation_version='foundation_v2')
        self.rng = np.random.default_rng(seed)
        self.previous = None
        self.evaluations = 0
        self.logs = []

    def close(self):
        self.pool.close()

    def plan(self, env):
        c = self.config
        Snapshot(env).restore(self.predictor)
        nominal = []
        for _ in range(c.horizon):
            action = action_of(self.net, self.width, self.predictor)
            nominal.append(action[0].numpy())
            self.predictor.step(action)
        nominal = np.asarray(nominal)
        warm = nominal.copy() if self.previous is None else np.concatenate((self.previous[c.commit:], nominal[-c.commit:]))
        knots = np.linspace(0, c.horizon-1, c.knots).astype(int)
        mean = (warm-nominal)[knots]
        std = np.full_like(mean, c.sigma)
        best, best_cost = nominal.copy(), np.inf
        first_cost = None
        state = initial_state(env)
        for iteration in range(c.iterations):
            noise = self.rng.normal(size=(c.population, c.knots, env.num_actions))*std + mean
            residual = np.stack([np.stack([np.interp(np.arange(c.horizon), knots, x[:, j])
                                           for j in range(env.num_actions)], axis=-1) for x in noise])
            candidates = np.clip(nominal[None]+residual, -1, 1).astype(np.float32)
            candidates[0], candidates[1] = best, warm
            states, _ = self.pool.rollout(env.model, self.datas, state, native_controls(env, candidates),
                                         control_spec=CONTROL, initial_warmstart=env.datas[0].qacc_warmstart)
            costs = physical_cost(env, states, candidates)
            self.evaluations += c.population
            if first_cost is None:
                first_cost = float(costs[0])
            idx = int(np.argmin(costs))
            if costs[idx] < best_cost:
                best, best_cost = candidates[idx].copy(), float(costs[idx])
            elites = (candidates[np.argsort(costs)[:c.elite]]-nominal[None])[:, knots]
            mean, std = elites.mean(0), np.maximum(.015, elites.std(0))
        self.previous = best
        self.logs.append(dict(nominal_cost=first_cost, optimized_cost=best_cost))
        return best[:c.commit]


def evaluate(snapshot, net, width, config, seed, mode, out, recorded=None):
    env = PerturbEnv(num_envs=1, episode_seconds=20, observation_version='foundation_v2')
    snapshot.restore(env)
    dt = env.dt*env.decimation
    meters = [HeightProxyRecoveryMetrics(1, dt, .4, env.rest_foot_z), RecoveryMetrics(1, dt, .4)]
    planner = ShootingController(env, net, width, config, seed) if mode == 'optimized' else None
    started = time.monotonic()
    pending, actions, states, applied = [], [], [], []
    previous = env.prev_action[0].copy()
    fallen, capped, optimized_steps = False, False, 0
    max_steps = min(int(6/dt), len(recorded)) if mode == 'recorded' else int(6/dt)
    try:
        for t in range(max_steps):
            if planner and t < config.optimize_steps and not capped:
                if not len(pending):
                    if time.monotonic()-started >= config.seconds_per_case:
                        capped = True
                    else:
                        pending = list(planner.plan(env))
                if not capped:
                    action = torch.tensor(np.asarray(pending.pop(0))[None], dtype=torch.float32)
                    optimized_steps += 1
            if not planner or t >= config.optimize_steps or capped:
                action = (torch.tensor(recorded[t:t+1], dtype=torch.float32) if mode == 'recorded'
                          else action_of(net, width, env))
            _, _, _, _ = env.step(action)
            d = env.datas[0]
            f = env.recovery_features(0)
            fallen |= bool(f['pelvis_z'] < FALL_FRACTION*env.rest_pelvis_z or f['up'] < .5)
            clipped = action[0].numpy().clip(-1, 1)
            for meter in meters:
                meter.record(0, f, d.cvel[env.pelvis, :3], not fallen, 1, clipped-previous)
            previous = clipped
            actions.append(clipped.copy()); states.append(initial_state(env)); applied.append(d.ctrl.copy())
            assert np.isfinite(states[-1]).all()
            assert np.max(np.abs(d.ctrl[env.act_idx])/(env.force_limit*env.authority)) <= 1+1e-6
            assert np.count_nonzero(d.xfrc_applied[np.arange(env.model.nbody) != env.ball]) == 0
            if fallen:
                break
    finally:
        if planner:
            planner.close()
    legacy, loaded = [m.result(np.array([not fallen]), True) for m in meters]
    result = dict(mode=mode, survived=not fallen, recovered=bool(legacy['recovered_mask'][0]),
                  contact_recovered=bool(loaded['recovered_mask'][0]), settled_hold=float(loaded['settled_hold_s']),
                  simulated_seconds=len(actions)*dt, optimized_steps=optimized_steps, budget_capped=capped,
                  wall_seconds=time.monotonic()-started, searches=[] if planner is None else planner.logs,
                  evaluated_sequences=0 if planner is None else planner.evaluations,
                  action_change=loaded['action_change'], contact_sliding=loaded['foot_sliding_m'])
    np.savez_compressed(out.with_suffix('.npz'), commands=actions, state=states, applied_controls=applied)
    benchmark.write_json(out.with_suffix('.json'), result)
    return result


def verify_replay(snapshot, path):
    """Independently execute saved commands, including policy handback, on the CPU plant."""
    env = PerturbEnv(num_envs=1, episode_seconds=20, observation_version='foundation_v2')
    snapshot.restore(env)
    with np.load(path) as saved:
        for i, command in enumerate(saved['commands']):
            env.step(torch.tensor(command[None], dtype=torch.float32))
            np.testing.assert_array_equal(initial_state(env), saved['state'][i])
            np.testing.assert_array_equal(env.datas[0].ctrl, saved['applied_controls'][i])
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads((args.bank/'protocol.json').read_text())
    checkpoint = protocol['checkpoints']['accepted']
    assert benchmark.digest(checkpoint) == protocol['checkpoint_sha256']['accepted']
    assert benchmark.digest(ROOT/'mujoco_rig/dummy_ball.xml') == protocol['plant_sha256']
    args.out.mkdir(parents=True, exist_ok=False)
    config = SearchConfig()
    cases = select_cases(args.bank)
    benchmark.write_json(args.out/'protocol.json', dict(cases=cases, config=vars(config),
        checkpoint=checkpoint, checkpoint_sha256=benchmark.digest(checkpoint), plant_sha256=protocol['plant_sha256'],
        intervention='First control boundary after positive-force body contact; pending actions preserved.',
        decision='A previously failing case must never fall and settle after policy handback to establish a feasible recovery. Search failure is inconclusive.',
        scope='Offline, full-state oracle. No model, PPO, reward, torque authority or action-delay changes.'))
    snapshots, suffixes = capture_cases(args.bank, checkpoint, cases, args.out)
    net, width = load_actor(checkpoint)
    results = []
    for case in cases:
        trial = case['trial']
        entry = dict(case=case)
        for mode in ('recorded', 'policy', 'optimized'):
            entry[mode] = evaluate(snapshots[trial], net, width, config, 23000+trial, mode,
                                   args.out/f'case-{trial}-{mode}', recorded=suffixes[trial])
            print(trial, mode, json.dumps({k: v for k, v in entry[mode].items() if k != 'searches'}), flush=True)
        entry['optimized_replay_exact'] = verify_replay(snapshots[trial], args.out/f'case-{trial}-optimized.npz')
        entry['feasible_new_recovery'] = (not entry['recorded']['survived'] and not entry['policy']['survived']
                                         and entry['optimized']['recovered'] and entry['optimized']['contact_recovered'])
        results.append(entry)
        benchmark.write_json(args.out/'results.json', results)
    print('DONE feasible new recoveries:', sum(x['feasible_new_recovery'] for x in results), '/', len(results), flush=True)


if __name__ == '__main__':
    main()

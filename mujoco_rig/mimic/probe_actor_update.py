"""Isolate actor optimization from critic fitting and test matched policy branches."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import validate_contract
from .continuous_stand import evaluate_retention, passing_cases
from .evaluation_cache import actor_fingerprint
from .motion_features import install
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION
from .standing_readiness import prepare_episodes, age_counts, require_coverage


def frozen_state(agent):
    mutable = {id(p) for p in agent._actor_optimizer._param_list}
    names = {n for n,p in agent.named_parameters() if id(p) in mutable}
    return {n:v.detach().cpu().clone() for n,v in agent.state_dict().items() if n not in names}


def assert_frozen(agent, expected):
    actual = frozen_state(agent)
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k],v) for k,v in expected.items()):
        raise RuntimeError('Actor-only probe changed critic, normalization or fixed distribution state')


def paired_summary(baseline, candidate, seed=210924):
    """Conditional rollout uncertainty; resample paired replicas within each posture."""
    a, b = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 16 or len(a) % 8 or not np.isfinite([a,b]).all():
        raise ValueError('Require finite, matched trials with equal replicas for eight postures')
    delta = (b-a).reshape(-1,8).T
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(2000):
        indices = rng.integers(delta.shape[1], size=delta.shape)
        means.append(np.take_along_axis(delta, indices, axis=1).mean())
    return dict(mean_delta=float(delta.mean()), interval_95=np.quantile(means,[.025,.975]).tolist(),
                by_posture=delta.mean(1).tolist(),
                scope='Paired stratified bootstrap over rollout noise, conditional on one trained batch; not training-seed generalization')


@torch.no_grad()
def surrogate(agent, batch, chunk=4096):
    total, count = {}, len(batch['obs'])
    for start in range(0,count,chunk):
        part = {k:v[start:start+chunk] for k,v in batch.items()}
        info = agent._compute_actor_loss(part)
        for key,value in info.items():
            total[key] = total.get(key,0.) + float(value)*len(part['obs'])/count
    return total


@torch.no_grad()
def matched_trials(agents, task, trials, seed, discount):
    """All branches advance together; each receives the same per-trial Gaussian draw."""
    from envs.base_env import DoneFlags, EnvMode
    if task.get_num_envs() != len(agents)*trials:
        raise ValueError('Branch/world layout mismatch')
    generator = torch.Generator(device=task.device).manual_seed(seed)
    task.set_mode(EnvMode.TEST)
    obs, _ = task.reset()
    active = torch.ones(task.get_num_envs(), dtype=torch.bool, device=task.device)
    terminal_bootstrap = torch.zeros(task.get_num_envs(), device=task.device)
    data = {key:[] for key in ('reward','done','active')}
    # Record the common noise itself so pairing is independently inspectable.
    noises = []
    for step in range(math.ceil(30/task.dt)):
        noise = torch.randn((trials,task.get_action_space().shape[0]), generator=generator, device=task.device)
        noises.append(noise.cpu().numpy())
        actions = []
        for i,agent in enumerate(agents.values()):
            local = obs[i*trials:(i+1)*trials]
            dist = agent._model.eval_actor(agent._obs_norm.normalize(local))
            actions.append(agent._a_norm.unnormalize(dist.mean + dist.stddev*noise))
        following, reward, done, _ = task.step(torch.cat(actions))
        timeout = active & (done == DoneFlags.TIME.value)
        if timeout.any():
            fixed_critic = next(iter(agents.values()))
            terminal_bootstrap[timeout] = fixed_critic._model.eval_critic(
                fixed_critic._obs_norm.normalize(following[timeout])).flatten()
        data['reward'].append((reward*active).cpu().numpy())
        data['done'].append(done.cpu().numpy())
        data['active'].append(active.cpu().numpy().copy())
        active &= done == DoneFlags.NULL.value
        if (step+1)%300 == 0:
            print(json.dumps(dict(matched_control_steps=step+1, active_first_episodes=int(active.sum()))), flush=True)
        if not active.any():
            break
        obs, _ = task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
    if active.any():
        raise RuntimeError('Unfinished matched first episodes')
    data = {k:np.asarray(v) for k,v in data.items()}
    bootstrap = terminal_bootstrap.cpu().numpy()
    results = {}
    for i,name in enumerate(agents):
        sl = slice(i*trials,(i+1)*trials)
        reward, mask, done = (data[k][:,sl] for k in ('reward','active','done'))
        lengths = mask.sum(0)
        terminal = done[lengths-1,np.arange(trials)]
        returns = (reward.astype(np.float64)*discount**np.arange(len(reward))[:,None]).sum(0)
        results[name] = dict(returns=returns.tolist(), survival_seconds=(lengths*task.dt).tolist(),
            returns_with_fixed_critic_tail=(returns+discount**lengths*bootstrap[sl]).tolist(),
            survived=(terminal == DoneFlags.TIME.value).tolist(), mean_return=float(returns.mean()),
            successes=int((terminal == DoneFlags.TIME.value).sum()),
            mean_survival_seconds=float(lengths.mean()*task.dt))
    return results, {**data,'common_noise':np.asarray(noises), 'terminal_bootstrap':bootstrap}


def run(args):
    if args.envs < 64 or args.envs%32 or args.trials < 16 or args.trials%8:
        raise ValueError('Require training worlds in multiples of 32 and matched trials in multiples of eight')
    activate(args.mimickit)
    from .continuous_stand_task import ContinuousStandTask
    from .guarded_ppo import GuardedPPO
    from learning.base_agent import AgentMode
    from util import mp_util
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0,1,'cuda:0',None)
    config = yaml.safe_load((args.run/'agent.yaml').read_text())
    source_contract = json.loads((args.run/'contract.json').read_text())
    calibration = json.loads((args.run/'critic_calibration.json').read_text())
    parent = json.loads((args.run/'report.json').read_text())
    if not calibration['passed'] or not calibration['temporal_validation']:
        raise ValueError('Require admitted temporal calibration')
    if sha256(args.run/'critic_calibration.json') != parent['calibration_sha256']:
        raise ValueError('Calibration provenance mismatch')
    if config.get('exp_prob_beg',1.) != 1. or config.get('exp_prob_end',1.) != 1.:
        raise ValueError('Matched Gaussian probe currently requires fully stochastic training actions')
    rig = Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
    task = ContinuousStandTask(rig,args.reference,args.envs,'cuda:0',seed=args.seed)
    task.reward_scale = config['standing_objective']['reward_scale']
    validate_contract(source_contract,task.observation_contract())

    class ProbeAgent(GuardedPPO):
        def _build_model(self, config):
            super()._build_model(config)
            if 'motion_feature_indices' in config:
                install(self,config['motion_feature_indices'])

        def _update_model(self):
            raise RuntimeError('No combined actor/critic training in this probe')

        def _record_data_pre_step(self, obs, info, action, action_info):
            super()._record_data_pre_step(obs,info,action,action_info)
            self._exp_buffer.record('episode_age',task.get_env_time())

    agent = ProbeAgent(config,task,'cuda:0').eval()
    agent.load(str(args.run/'calibrated.pt'))
    invariant = frozen_state(agent)
    args.out.mkdir(parents=True,exist_ok=False)
    viewers = Path(__file__).resolve().parents[2]/'logs/mimickit-viewer'
    viewer_hashes = {str(p):sha256(p) for p in viewers.glob('*.json')}
    report = dict(schema='mimic_actor_direction_probe_v1', source_run=str(args.run),
        mimickit_revision=MIMICKIT_REVISION, seed=args.seed, evaluation_seed=args.seed+1,
        envs=args.envs, trials_per_branch=args.trials, config=config, contract=source_contract,
        input_hashes={str(args.run/name):sha256(args.run/name) for name in
            ('calibrated.pt','agent.yaml','contract.json','critic_calibration.json','report.json')},
        source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
            ('probe_actor_update.py','guarded_ppo.py','update_guard.py','standing_readiness.py','continuous_stand_task.py','stand_task.py')},
        viewer_hashes=viewer_hashes, variants={},
        scope='One fresh prepared rollout batch. Frozen critic/normalization/objective. Fresh actor optimizer. '
              'One SGD step and one normal 40-step actor phase start independently from identical state and sampler. '
              'Native deterministic retention plus simultaneous paired Gaussian GPU trials and a no-op repeat. '
              'Measured return is the discounted 30-second prefix with zero tail, not the full continuing objective; '
              'a separate estimate bootstraps timeouts with the frozen critic. '
              'Diagnostic weights only; no deployment or checkpoint selection.')
    (args.out/'protocol.json').write_text(json.dumps(report,indent=2))
    state, preparation = prepare_episodes(agent,task,passing_cases(parent['initial_native']))
    (args.out/'preparation.json').write_text(json.dumps(preparation,indent=2))
    agent._init_train()
    agent._curr_obs, agent._curr_info = state
    agent.set_mode(AgentMode.TRAIN)
    agent._init_iter()
    with torch.no_grad():
        agent._rollout_train(config['steps_per_iter'])
        agent._build_train_data()
    buffer = agent._exp_buffer
    count = buffer.get_sample_count()
    ages = buffer.get_data('episode_age')
    counts = age_counts(ages.flatten(), task.ids.expand_as(ages).flatten())
    require_coverage(counts,preparation['retained_cases'])
    report['batch'] = dict(samples=count, age_counts=counts.cpu().tolist())
    fields = ('obs','action','a_logp','rand_action_mask','adv','tar_val','reward','done','episode_age')
    batch = {key:buffer.get_data_flat(key) for key in fields}
    payload = {key:v.cpu() for key,v in batch.items()}
    payload['sample_permutation'] = buffer._sample_buf.cpu()
    torch.save(payload,args.out/'batch.pt')
    report['batch']['sha256'] = sha256(args.out/'batch.pt')
    weights = copy.deepcopy(agent.state_dict())
    optimizer = copy.deepcopy(agent._actor_optimizer._optimizer.state_dict())
    permutation, head = buffer._sample_buf.clone(), buffer._sample_buf_head
    cuda_rng = torch.cuda.get_rng_state()
    cpu_rng = torch.get_rng_state()
    before = surrogate(agent,batch)
    minibatch = int(math.ceil(config['actor_batch_size']*args.envs))
    phase_steps = math.ceil(count/minibatch)*config['actor_epochs']
    for label,steps in [('baseline',0),('one_step',1),('actor_phase',phase_steps)]:
        agent.load_state_dict(weights)
        agent._actor_optimizer._optimizer.load_state_dict(copy.deepcopy(optimizer))
        agent._actor_optimizer._steps = 0
        buffer._sample_buf.copy_(permutation)
        buffer._sample_buf_head = head
        torch.cuda.set_rng_state(cuda_rng)
        torch.set_rng_state(cpu_rng)
        agent.train()
        update = agent._update_actor(minibatch,steps) if steps else {}
        agent.eval()
        after = surrogate(agent,batch)
        assert_frozen(agent,invariant)
        path = args.out/f'{label}.pt'
        agent.save(str(path))
        report['variants'][label] = dict(requested_optimizer_steps=steps,
            update={k:float(v) for k,v in update.items()}, before=before, after=after,
            actor_loss_decrease=before['actor_loss']-after['actor_loss'], checkpoint_sha256=sha256(path))
        print(json.dumps(dict(variant=label, **report['variants'][label])),flush=True)
    # Each evaluation agent uses the same immutable critic. No source-run file is modified.
    eval_task = ContinuousStandTask(rig,args.reference,args.trials*4,'cuda:0',seed=args.seed+1)
    eval_task.reward_scale = task.reward_scale
    actors = {}
    for label in ('baseline','baseline_repeat','one_step','actor_phase'):
        candidate = ProbeAgent(config,eval_task,'cuda:0').eval()
        candidate.load(str(args.out/f'{"baseline" if label=="baseline_repeat" else label}.pt'))
        assert_frozen(candidate,invariant)
        actors[label] = candidate
    fingerprints = {k:actor_fingerprint(v) for k,v in actors.items()}
    results,traces = matched_trials(actors,eval_task,args.trials,args.seed+1,config['discount'])
    np.savez_compressed(args.out/'matched-trials.npz',**traces)
    report['matched'] = results
    report['matched_trace_sha256'] = sha256(args.out/'matched-trials.npz')
    report['comparisons'] = {name:{key:paired_summary(results['baseline'][key],result[key],args.seed+1)
        for key in ('returns','returns_with_fixed_critic_tail','survival_seconds','survived')}
        for name,result in results.items() if name!='baseline'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    native = ContinuousStandTask(rig,args.reference,engine_factory=NativeDummyEngine)
    report['native'] = {}
    for label in ('baseline','one_step','actor_phase'):
        metrics,traces = evaluate_retention(native,StandPolicy(actors[label]).cpu().eval())
        path = args.out/f'{label}-native.npz'
        np.savez_compressed(path,**traces)
        report['native'][label] = dict(metrics=metrics,trace_sha256=sha256(path))
        print(json.dumps(dict(native=label,passing_cases=sorted(passing_cases(metrics)))),flush=True)
    assert all(actor_fingerprint(actors[k])==v for k,v in fingerprints.items())
    assert all(sha256(Path(p))==v for p,v in {**viewer_hashes,**report['input_hashes']}.items())
    report.update(complete=True, critic_and_normalization_unchanged=True, viewers_unchanged=True, exported=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--envs',type=int,default=2048)
    parser.add_argument('--trials',type=int,default=256)
    parser.add_argument('--seed',type=int,default=210923)
    run(parser.parse_args())

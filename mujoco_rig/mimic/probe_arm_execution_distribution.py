"""Frozen, paired physical comparisons of correction execution distributions."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT
from .run_learned_handoff import PRIOR
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_correction_ppo import CorrectionEnvironment, CorrectionNetwork, INITIAL, OUT as PPO

ROOT = Path('logs/mimickit-training/arm-recovery-30-20261008-01')
ACTORS = {'zero': PPO / 'initial-correction.pt',
          '521028': PPO / '521028/final.pt', '531028': PPO / '531028/final.pt'}


def setup():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION / 'stationary_reference.npz',
                        json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)
    state = torch.load(INITIAL, weights_only=True)
    proposal = QueueBlindLearner(state['mean'], state['scale'], state['output_scale'])
    proposal.load_state_dict(state)
    proposal.eval().requires_grad_(False)
    env = make_environment(rig, anchor, 40.)
    poses = np.load(PRIOR / 'goals.npz')['poses']
    actors = {}
    for label, path in ACTORS.items():
        net = CorrectionNetwork()
        net.load_state_dict(torch.load(path, weights_only=True))
        actors[label] = net.eval().requires_grad_(False)
    return env, anchor, proposal, poses, actors


def seal(out, extra=()):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'sources').mkdir()
    sources = list(Path(__file__).parent.glob('*.py'))
    sources.append(Path(__file__).with_name('ARM_RECOVERY_30_PLAN.md'))
    inputs = sources + list(ACTORS.values()) + [PARENT, PARENT.with_suffix('.amp.pt'),
        INITIAL, PRIOR / 'goals.npz', FOUNDATION / 'dummy.xml', FOUNDATION / 'stationary_reference.npz',
        FOUNDATION / 'protocol.json'] + list(extra)
    for path in sources:
        shutil.copy2(path, out / 'sources' / path.name)
    return {str(p): digest(p) for p in inputs}


def projected_mean(mean, scale, previous, nominal, std):
    """Analytic expectation of the slew/envelope projection, before joint clipping."""
    m = nominal.double() + mean.double() * scale.double()[:, None]
    s = (scale.double()[:, None] * std).expand_as(m)
    lo = (previous.double() - .02).clamp_min(-1)
    hi = (previous.double() + .02).clamp_max(1)
    safe = s.clamp_min(1e-30)
    a, b = (lo - m) / safe, (hi - m) / safe
    ca, cb = .5 * (1 + torch.erf(a / 2**.5)), .5 * (1 + torch.erf(b / 2**.5))
    pa, pb = torch.exp(-a*a/2) / (2*np.pi)**.5, torch.exp(-b*b/2) / (2*np.pi)**.5
    expected = lo*ca + m*(cb-ca) + safe*(pa-pb) + hi*(1-cb)
    return torch.where(s > 0, expected, m.clamp(lo, hi)).float()


def action_for(adapter, mean, config, generator, frame, noise):
    p = adapter.physical
    _, goals, amount, quiet, _ = p.observation()
    scale = amount * ((goals[:, p.qi] - p.reference[:, p.qi]).abs().amax(-1) / .05).clamp(0, 1)
    mode, std = config.get('mode', 'sample'), config.get('std', .005)
    if mode == 'mean':
        return mean, noise
    if mode == 'expected':
        nominal = adapter.observation()[0][:, 454:487] * .1
        # Recompute without feature rescaling round-off.
        from .target_control33 import DelayedTargetControl33
        x = p.observation()[0]
        source = DelayedTargetControl33.target_positions(p.control, p.anchor(x[:, :374]))
        desired = (goals[:, p.qi] + adapter.proposal(x) - source).clamp(-1, 1) * scale[:, None]
        nominal = p.previous + (desired - p.previous).clamp(-.02, .02)
        expectation = projected_mean(mean, scale, p.previous, nominal, std)
        correction = (expectation - nominal) / scale[:, None].clamp_min(1e-20)
        correction[scale == 0] = 0
        return correction, noise
    if frame % config.get('repeat', 1) == 0:
        noise = torch.randn(mean.shape, generator=generator) * std
    t = frame * adapter.env.dt
    phase = config.get('phase', 'all')
    enabled = (phase == 'all' or
               phase == 'ramp' and adapter.onset <= t < adapter.onset + adapter.ramp or
               phase == 'hold' and adapter.onset + adapter.ramp <= t < adapter.hold_end or
               phase == 'return' and adapter.hold_end <= t < adapter.hold_end + 1.5)
    return mean + noise if enabled else mean, noise


@torch.no_grad()
def episode(adapter, actor, config, folder, keep_features=False):
    adapter.reset()
    adapter.physical.control.cursor = 0
    p, env = adapter.physical, adapter.env
    n = 16 if adapter.quiet else 8
    active = torch.arange(16) < n
    xy = env.measurements()['foot_xy_m'].numpy()[:n].copy()
    generator = torch.Generator().manual_seed(config.get('seed', 610081))
    noise = torch.zeros((16, 33))
    rows = []
    for frame in range(p.frames):
        x, goals, amount, quiet, _ = adapter.observation()
        mean = actor(x)
        action, noise = action_for(adapter, mean, config, generator, frame, noise)
        before = active.clone()
        reward, terminal, info = adapter.step(action, active)
        post = env.last
        values = dict(alive=before, qpos=info['qpos'], stand=info['stand'], support=info['supported'],
            com=post['com_m'], xy=post['foot_xy_m'], issued=info['issued'], addition=info['addition'],
            previous=info['previous_addition'], nominal=info['nominal_addition'], raw_mean=mean,
            correction=action, done=info['failed'], reward=reward, components=info['components'],
            quiet=quiet, scale=amount * ((goals[:, p.qi]-p.reference[:, p.qi]).abs().amax(-1)/.05).clamp(0,1))
        if keep_features:
            values['x'] = x
        rows.append({k: v[:n].numpy().copy() for k, v in values.items()})
        active &= ~terminal
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    z['time'] = (np.arange(p.frames)+1)*env.dt
    hold = (z['time'] > adapter.hold_end-2) & (z['time'] <= adapter.hold_end)
    cases = []
    for i in range(n):
        survived = bool(z['alive'][-1,i] and not z['done'][-1,i])
        returned = bool(survived and z['stand'][-int(np.ceil(2/env.dt)):,i].all())
        stable = bool(z['alive'][hold,i].all() and z['support'][hold,i].all())
        motor = np.sqrt(np.mean((z['qpos'][:,i,p.qi]-p.poses[i,1,p.qi])**2, axis=-1))
        com = np.linalg.norm(z['com'][:,i,:2]-p.gate.com[i],axis=-1)
        feet = np.linalg.norm(z['xy'][:,i]-xy[i],axis=-1).max(-1)
        tracked = bool(stable and (motor[hold] <= .05).all() and (com[hold] <= .02).all() and (feet[hold] <= .02).all())
        cases.append(dict(pose=i%8,heldout=i%8>=6,stable_hold=stable,returned_stand=returned,
            complete=tracked and returned,tracked_hold=tracked,survived=survived,
            whole_interval_stand=bool(z['alive'][:,i].all() and z['stand'][:,i].all() and survived),
            peak_motor=float(motor[hold].max()),mean_motor=float(motor[hold].mean()),peak_com=float(com[hold].max()),
            peak_feet=float(feet[hold].max()), reward_sum=float((z['reward'][:,i]*z['alive'][:,i]).sum())))
    result = dict(config=config,hold_end=adapter.hold_end,seconds=adapter.seconds,cases=cases,
        stable_holds=sum(c['stable_hold'] for c in cases),returned=sum(c['returned_stand'] for c in cases),
        complete=sum(c['complete'] for c in cases),quiet_passed=sum(c['whole_interval_stand'] for c in cases))
    folder.mkdir()
    np.savez_compressed(folder/'trace.npz',**z)
    save(folder/'report.json',result)
    print('EVAL',folder.name, 'hold',result['stable_holds'],'return',result['returned'],'full',result['complete'],flush=True)
    return result


def configs(step):
    rows=[]
    def add(mode='sample', seeds=(610081,610082), **kwargs):
        for actor in ACTORS:
            for seed in seeds:
                rows.append(dict(actor=actor,mode=mode,seed=seed,**kwargs))
    if step==1:
        add('mean', (610081,));add(seeds=range(610081,610085))
    elif step==2: add(seeds=range(610091,610095))
    elif step==3:
        add('mean',(610081,),variant=True);add(variant=True)
    elif step==4: add(std=.0025)
    elif step==5: add(std=.00125)
    elif step==7: add('expected',(610081,))
    elif step==8: add('expected',(610081,),variant=True)
    elif step==9: add(repeat=3)
    elif step==10:
        for phase in ('ramp','hold','return'):add(phase=phase)
    else: raise ValueError(step)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True)
    args=parser.parse_args();out=ROOT/f'step-{args.step:02d}'
    spec=configs(args.step);inputs=seal(out)
    save(out/'protocol.json',dict(step=args.step,configs=spec,inputs=inputs,training_updates=0))
    env,anchor,proposal,poses,actors=setup()
    adapters={variant:CorrectionEnvironment(env,anchor,proposal,poses,variant=variant) for variant in {c.get('variant',False) for c in spec}}
    results=[]
    for index,c in enumerate(spec):
        label=f'{index:02d}-{c["actor"]}-{c["mode"]}-{c["seed"]}'
        result=episode(adapters[c.get('variant',False)],actors[c['actor']],c,out/label,
                       keep_features=args.step==1 and c['mode']=='mean')
        if args.step==1 and c['mode']=='mean':
            prior=PPO/('zero-nominal' if c['actor']=='zero' else c['actor']+'-nominal')/'trace.npz'
            with np.load(prior) as old, np.load(out/label/'trace.npz') as new:
                assert all(np.array_equal(old[k],new[k]) for k in ('qpos','issued','alive'))
        results.append(dict(folder=label,**result));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()

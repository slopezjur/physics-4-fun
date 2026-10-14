"""Replay time-varying quiet commands to isolate impact-specific actor feedback."""
import argparse
import gc
from pathlib import Path
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .ball_study_support import FOUNDATION, PILOT, body_observer, checkpoint, protect_used, save_bodies
from .evaluate_stand_ball_boundary import evaluate as native_ball
from .evaluate_stand_ball_gpu import evaluate as gpu_ball, compare
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import load_actor, seal



class QuietReplayPolicy(torch.nn.Module):
    def __init__(self,actor,commands,switch):
        super().__init__()
        self.actor,self.switch,self.frame=actor,switch,0
        self.register_buffer('quiet_commands',torch.as_tensor(commands.copy(),dtype=torch.float32))

    def forward(self,observation):
        action=self.actor(observation) if self.frame<self.switch else self.quiet_commands[self.frame].clone()
        self.frame+=1
        return action


@torch.no_grad()
def verify_replay(trace,actor,commands,switch):
    error=0.
    for frame in range(min(switch,len(trace['action']))):
        expected=actor(torch.from_numpy(trace['observation'][frame])).numpy()
        live=trace['alive_before'][frame]
        error=max(error,float(np.abs(expected[live]-trace['action'][frame,live]).max()))
    assert error<=2e-6
    for frame in range(switch,len(trace['action'])):
        live=trace['alive_before'][frame]
        assert np.array_equal(trace['action'][frame,live],commands[frame,live])
    return error


def jobs(settings):
    trained = Path(settings['training_root'])
    baseline = Path(settings['baseline_root'])
    for seed in settings['seeds']:
        yield seed,'source',checkpoint(seed,'source'),PILOT/str(seed)/'source',baseline,PILOT
        for arm in ('quiet','single_ball'):
            root = trained/str(seed)/arm
            yield seed,arm,root/'final/model.pt',root/'evaluation',root/'evaluation',trained


def baseline_folder(seed,label,quiet,impacts,name,backend):
    if name=='quiet':
        return quiet/f'quiet-{backend}'
    sign = 1 if name=='positive' else -1
    return impacts/(f'{seed}-source-speed140-{sign:+d}-{backend}' if label=='source'
                    else f'5kg-1.4mps-{sign:+d}-{backend}')


def prefix(trace,baseline,hold,backend,envelope):
    fields = ('qpos_before','qvel_before','qpos_after','qvel_after','action','pending_before','active_targets')
    with np.load(baseline/'trace.npz',allow_pickle=False) as old:
        count=min(hold+1,len(trace['action']),len(old['action']))
        covered=count==hold+1
        errors = {k:float(np.abs(trace[k][:count]-old[k][:count]).max()) for k in fields}
    limits = {k:(1e-10 if backend=='native' else envelope[k]) for k in fields}
    return dict(passed=covered and all(errors[k]<=limits[k] for k in fields),
                prefix_covered=covered,errors=errors,limits=limits)


def run(args):
    settings=read(args.protocol)
    if (settings['schema']!='chest_quiet_replay_v1' or settings['maximum_cases']!=384
            or settings['seeds']!=[301027,301028] or settings['training_updates']!=0):
        raise ValueError('Unsupported feedback protocol')
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0,1,'cuda:0',None)
    torch.set_num_threads(1)
    all_jobs=list(jobs(settings))
    calibration=Path(settings['calibration_root'])
    calibration_seal=read(calibration/'final-verification.json')
    calibration_hashes={str(Path(p).resolve()):h for p,h in calibration_seal['output_hashes'].items()}
    if (not calibration_seal['verified'] or
            calibration_hashes.get(str((calibration/'report.json').resolve()))!=digest(calibration/'report.json')):
        raise ValueError('Replay envelope differs from its locked calibration')
    # Protect exactly the checkpoint and baseline files consumed below, against
    # their sealed hashes, in addition to the inherited live implementation.
    evidence=list(settings['evidence'])
    for seed,label,model,quiet,impacts,root in all_jobs:
        manifests={}
        for location in {root,Path(settings['baseline_root'])}:
            verification=read(location/'final-verification.json')
            if not verification['verified']:
                raise ValueError('Unverified baseline root')
            manifests.update({str(Path(p).resolve()):h for p,h in verification['output_hashes'].items()})
        paths=[model,model.with_suffix('.amp.pt')]
        for name in ('positive','negative'):
            for backend in ('native','gpu'):
                folder=baseline_folder(seed,label,quiet,impacts,name,backend)
                paths += [folder/'trace.npz',folder/'report.json']
        for path in paths:
            if manifests.get(str(path.resolve()))!=digest(path):
                raise ValueError(f'Changed baseline {path}')
            evidence.append(str(path))
    used=dict(settings,evidence=evidence)
    parent,inputs,sources=protect_used(args.out,[Path(__file__)],args.protocol,used)
    save(args.out/'protocol.json',dict(inputs=inputs,sources=sources,settings=used,
        config=parent['config'],demonstrations=parent['demonstrations'],automatic_promotion=False))
    rig,reference=Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    dt=rig.model.opt.timestep*rig.decimation
    hold=int(np.ceil(5./dt))-1
    world=args.out/'ball-5kg.xml'
    write_world(args.out/'dummy.xml',FOUNDATION/'ball-5kg.xml',world,5.)
    envelope=read(Path(settings['calibration_root'])/'report.json')['intervention_envelope']
    cfg=dict(seconds=12.,launch_seconds=5.,launch_distance_m=.25)
    results=[]
    switch=hold+1
    for seed,label,model,quiet,impacts,_ in all_jobs:
        actor=load_actor(rig,reference,parent,model)
        original=tensor_hash(actor.state_dict())
        assessments={name:dict(rows={}) for name in ('quiet','positive','negative')}
        live_quiet={}
        for backend,evaluate in (('native',native_ball),('gpu',gpu_ball)):
            reference_folder=args.out/f'{seed}-{label}-live-quiet-{backend}'
            live_quiet[backend]=evaluate(rig,reference,actor.cpu(),world,reference_folder,cfg,0.,1)
            assert live_quiet[backend]['actor_hash']==original
            with np.load(reference_folder/'trace.npz',allow_pickle=False) as data:
                commands=data['action'].copy()
            if len(commands)!=int(np.ceil(12./dt)) or live_quiet[backend]['standing_passes']!=8:
                raise RuntimeError('Live quiet ball-world reference failed; no replay interpretation')
            for name,sign in (('quiet',0),('positive',1),('negative',-1)):
                folder=args.out/f'{seed}-{label}-{name}-{backend}'
                wrapped=QuietReplayPolicy(actor.cpu(),commands,switch)
                journal={}
                if backend=='native':
                    with body_observer(journal,len(commands)):
                        report=evaluate(rig,reference,wrapped,world,folder,cfg,0. if sign==0 else 1.4,sign or 1)
                    bodies=save_bodies(folder,journal)
                else:
                    report=evaluate(rig,reference,wrapped,world,folder,cfg,0. if sign==0 else 1.4,sign or 1)
                    bodies=None
                with np.load(folder/'trace.npz',allow_pickle=False) as data:
                    trace={k:data[k] for k in data.files}
                own_error=verify_replay(trace,actor.cpu(),commands,switch)
                base=reference_folder if name=='quiet' else baseline_folder(seed,label,quiet,impacts,name,backend)
                before=read(base/'report.json')
                assert before['actor_hash']==original==tensor_hash(actor.state_dict())
                admission=prefix(trace,base,hold,backend,envelope)
                save(folder/'replay-admission.json',dict(prefix=admission,own_observation_action_error=own_error,
                    inner_actor_hash=original,switch_frame=switch,local_pd_preserved=True,
                    quiet_command_source=str(reference_folder/'trace.npz'),quiet_command_sha256=digest(reference_folder/'trace.npz')))
                assessments[name]['rows'][backend]=dict(report=report,baseline=before,prefix=admission,bodies=bodies)
                save(args.out/'progress.json',dict(completed_actors=results,active_seed=seed,
                    active_actor=label,live_quiet=live_quiet,assessments=assessments))
                del trace,wrapped
                gc.collect()
                torch.cuda.empty_cache()
        for assessment in assessments.values():
            rows=assessment['rows']
            assessment['parity']=compare(rows['native']['report'],rows['gpu']['report'],dt)
        live_parity=compare(live_quiet['native'],live_quiet['gpu'],dt)
        result=dict(seed=seed,actor=label,live_quiet=live_quiet,live_quiet_parity=live_parity,
            assessments=assessments,
            admitted=live_parity['passed'] and
                all(r['prefix']['passed'] for s in assessments.values() for r in s['rows'].values())
                and all(s['parity']['passed'] for s in assessments.values()),
            replay_quiet_retained=all(r['report']['standing_passes']==8 for r in assessments['quiet']['rows'].values()))
        results.append(result)
        del actor
    save(args.out/'report.json',dict(results=results,evaluation_cases=384,training_updates=0,
        all_actors_admitted=all(r['admitted'] for r in results),promoted=False))
    seal(args.out,inputs,sources,verified=True,evaluation_cases=384,training_updates=0,promoted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args=parser.parse_args()
    existed=args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise

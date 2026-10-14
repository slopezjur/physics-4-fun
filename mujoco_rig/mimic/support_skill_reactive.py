"""State-triggered weight-transfer composition under frozen projectile tests."""
import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .runtime import activate
from . import run_support_skill_study as study
from .support_skill_frozen import load_policy


class TriggeredTransfer(torch.nn.Module):
    def __init__(self,anchor,residual,basis,controls,settings,radius):
        super().__init__();self.anchor=deepcopy(anchor);self.residual=deepcopy(residual)
        self.register_buffer('basis',basis)
        self.controls=controls;self.settings=settings;self.radius=radius
        self.runtime={};self.journal=[]

    def forward(self,obs):
        from .support_skill_core import smooth
        n=len(obs);device=obs.device;dt=.016668
        if not self.runtime:
            self.runtime=dict(frame=0,start=torch.full((n,),-10000.,device=device),
                side=torch.zeros(n,device=device,dtype=torch.long),count=torch.zeros(n,device=device,dtype=torch.long))
        state=self.runtime;t=state['frame']*dt
        speed=obs[:,9:11].norm(dim=-1)
        state['count']=torch.where(speed>self.settings['trigger_speed'],state['count']+1,0)
        duration=self.settings['ramp']+self.settings['hold']+1.5
        trigger=(state['count']>=2)&(t-state['start']>duration+self.settings.get('cooldown',1.))
        state['start']=torch.where(trigger,torch.full_like(state['start'],t),state['start'])
        state['side']=torch.where(trigger,(obs[:,93]>obs[:,94]).long(),state['side'])
        elapsed=t-state['start'];active=(elapsed>=0)&(elapsed<duration)
        c=smooth(elapsed,0.,self.settings['ramp'])*(1-smooth(elapsed,self.settings['ramp']+self.settings['hold'],1.5))
        sign=torch.where(state['side']==0,-1.,1.)
        goals=torch.zeros((n,6),device=device);goals[:,0]=2*sign*self.settings['shift']/.31*c
        goals[:,-1]=active.float()
        action=self.residual(torch.cat((obs,goals),dim=-1))
        bases=self.basis[state['side'],torch.arange(n,device=device)]
        offset=torch.einsum('ni,nij->nj',action,bases).clamp(-self.radius,self.radius)*active[:,None]
        self.controls[-1].addition.copy_(offset)
        self.journal.append(dict(goal=goals.detach().cpu().numpy().copy(),action=action.detach().cpu().numpy().copy(),
            addition=offset.detach().cpu().numpy().copy(),trigger=trigger.cpu().numpy().copy(),speed=speed.cpu().numpy().copy()))
        state['frame']+=1
        return self.anchor(obs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.protocol.read_text())
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from util import mp_util
    from .support_skill_core import AddedTargetControl,SkillEnvironment,Goal
    from . import stand_ball_native,stand_ball_gpu
    from . import evaluate_stand_ball_boundary as native
    from . import evaluate_stand_ball_gpu as gpu
    from .replicate_stand_representations import verify_control_trace
    from .run_chest_repeated_screen import wrap_evaluator
    from .corrective_step_audit import audit_trace
    from . import ball_study_support as support
    mp_util.init(0,1,'cuda:0',None);torch.set_num_threads(1)
    rig,_,anchors=study.setup(spec,args.out)
    inputs={str(Path(spec['policy'])/name):study.digest(Path(spec['policy'])/name) for name in ('final.pt','protocol.json')}
    world=study.FOUNDATION/'ball-5kg.xml';inputs[str(world)]=study.digest(world)
    study.save(args.out/'composition-inputs.json',inputs)
    results={};admission={}
    for parent,anchor in anchors.items():
        residual,trained=load_policy(rig,anchor,spec['policy'])
        if trained.get('action_mode','basis')!='basis':raise ValueError('This composition explicitly requires the tested coordinated transfer policy')
        env=SkillEnvironment(rig,study.FOUNDATION/'stationary_reference.npz',deepcopy(anchor),
            [Goal(side=side,shift=spec['skill']['shift']) for side in (0,1) for _ in range(8)],'cpu')
        basis=env.basis.reshape(2,8,3,33).clone()
        for condition in spec['conditions']:
            for mode in ('source','triggered'):
                pair={}
                for backend,evaluator in [('native',native.evaluate),('gpu',gpu.evaluate)]:
                    label=f"{parent}-{mode}-{condition['name']}-{backend}";out=args.out/label
                    controls=[]

                    class RecordedControl(AddedTargetControl):
                        def __init__(self,*a,**kw):
                            super().__init__(*a,**kw);controls.append(self)

                    actor=(deepcopy(anchor) if mode=='source' else TriggeredTransfer(anchor,residual,basis,controls,spec['skill'],trained.get('radius',.5)))
                    actor=actor.to('cpu' if backend=='native' else 'cuda:0').eval().requires_grad_(False)

                    def verify(trace,contract):
                        if mode=='source':return verify_control_trace(trace,contract)
                        extra=np.stack([r['addition'] for r in actor.journal])
                        if extra.shape!=trace['action'].shape:raise RuntimeError('Composition journal/physics frame mismatch')
                        lower=np.asarray(contract['target_lower_radians']);upper=np.asarray(contract['target_upper_radians'])
                        raw=.25*np.clip(trace['action'],-1,1)
                        base=np.clip(trace['reference_motor_q']+raw,lower,upper)
                        expected=np.clip(base+extra,lower,upper)
                        assert np.allclose(trace['issued_entry'][...,:33],expected,rtol=0,atol=2e-7)
                        return verify_control_trace(dict(trace,reference_motor_q=base+extra-raw),contract)

                    settings=dict(seconds=condition['seconds'],launch_seconds=condition['launch_times_seconds'][0],launch_distance_m=.25)
                    journal={}
                    with ExitStack() as stack:
                        for module in (stand_ball_native,stand_ball_gpu):stack.enter_context(patch.object(module,'DelayedTargetControl33',RecordedControl))
                        for module in (native,gpu):stack.enter_context(patch.object(module,'verify_control_trace',verify))
                        if backend=='native':stack.enter_context(support.body_observer(journal,int(np.ceil(settings['seconds']/.016668))))
                        fn=wrap_evaluator(evaluator,backend,spec) if len(condition['launch_times_seconds'])>1 else evaluator
                        report=fn(rig,study.FOUNDATION/'stationary_reference.npz',actor,world,out,settings,condition['speed'],condition.get('sign',1))
                    if backend=='native':
                        bodies=support.save_bodies(out,journal)
                        report['chest_first_all']=all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies'] for c in bodies['cases'])
                        encounter_bodies=[]
                        for case,body_case in zip(report['cases'],bodies['cases'],strict=True):
                            checks=[]
                            for event in case.get('encounters',[case]):
                                first=event.get('first_contact_seconds')
                                frame=None if first is None else int(round(first/.016668))-1
                                cluster=next((c for c in body_case['clusters'] if frame is not None and c['first_frame']<=frame<=c['last_frame']),None)
                                checks.append(bool(cluster and 'Chest' in cluster['bodies']))
                            encounter_bodies.append(checks)
                        report['every_encounter_chest_first']=encounter_bodies
                    if mode=='triggered':
                        np.savez_compressed(out/'composition.npz',**{k:np.stack([r[k] for r in actor.journal]) for k in actor.journal[0]})
                        report['trigger_counts']=np.stack([r['trigger'] for r in actor.journal]).sum(0).tolist()
                    study.save(out/'report.json',report)
                    audit=audit_trace(rig.path,out);study.save(out/'step-audit.json',audit)
                    results[label]=dict(report=report,measured_steps=audit['measured_steps'],recovered_with_step=audit['recovered_with_step'])
                    pair[backend]=report
                    study.save(args.out/'progress.json',results)
                    print(label,'recovery',report['recovery_passes'],'recovered steps',audit['recovered_with_step'],flush=True)
                admission[f"{parent}-{mode}-{condition['name']}"]=gpu.compare(pair['native'],pair['gpu'],.016668)
    for path,expected in inputs.items():assert study.digest(path)==expected
    p=json.loads((args.out/'protocol.json').read_text())
    for path,expected in {**p['sources'],**p['inputs']}.items():assert study.digest(path)==expected
    study.save(args.out/'report.json',dict(stage=spec['stage'],completed=True,results=results,backend_admission=admission,
        transfer_component_only=True,learned_step_claim=False))
    study.save(args.out/'final-verification.json',dict(inputs_unchanged=True,outputs={str(p.relative_to(args.out)):study.digest(p)
        for p in sorted(args.out.rglob('*')) if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()

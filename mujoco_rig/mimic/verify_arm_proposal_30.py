"""Independent physical gates, provenance, paired initialization and budget audit."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .probe_arm_proposal_foundations import ROOT,TEACHER
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_handoff import PRIOR
from .predictive_target_support import blend_pose


def read(path):return json.loads(path.read_text())


def gates(model,poses,folder,r):
    z=np.load(folder/'trace.npz');dt=model.opt.timestep*4
    joints=np.array([model.actuator_trnid[i,0] for i in range(model.nu)])
    # Rig owns the explicit motor subset, including exclusion of passive joints.
    from .rig import Rig
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=());joints=model.actuator_trnid[rig.actuators,0]
    qi=model.jnt_qposadr[joints];bounds=model.jnt_range[joints]
    goals=poses.copy();config=r.get('config',{})
    if config.get('amplitude',1)!=1:
        for i in range(16):goals[i,1]=blend_pose(poses[i,0],poses[i,1],config['amplitude'],1.)
    assert len(z['time'])==int(np.ceil(r['seconds']/dt))
    assert np.allclose(z['time'],(np.arange(len(z['time']))+1)*dt,atol=1e-12,rtol=0)
    assert np.isfinite(z['qpos']).all()
    assert (z['issued']>=bounds[:,0]-1e-6).all() and (z['issued']<=bounds[:,1]+1e-6).all()
    if 'nominal' in z:
        scale=z['scale'][:,:,None];previous=z['previous']
        projected=np.clip(previous+np.clip(z['nominal']+z['correction']*scale-previous,-.02,.02),-1,1)
        expected=np.where(z['correction']!=0,projected,z['nominal']);expected[z['quiet']|~z['alive']]=0
        assert np.allclose(expected,z['addition'],atol=2e-7,rtol=0)
        live=z['alive']&~z['quiet']
        assert abs(z['addition'][live]-previous[live]).max(initial=0)<=.0200001
    assert abs(z['addition']).max()<=1.0000001
    hold=(z['time']>r['hold_end']-2)&(z['time']<=r['hold_end']);assert hold.sum()>=int(2/dt)-1
    data=mujoco.MjData(model)
    for i,c in enumerate(r['cases']):
        data.qpos[:]=poses[i,0];mujoco.mj_forward(model,data)
        xy=data.xpos[[model.body('Foot_L').id,model.body('Foot_R').id],:2].copy()
        data.qpos[:]=goals[i,1];mujoco.mj_forward(model,data);com=data.subtree_com[1,:2].copy()
        error=np.sqrt(np.mean((z['qpos'][:,i,qi]-goals[i,1,qi])**2,axis=-1))
        ce=np.linalg.norm(z['com'][:,i,:2]-com,axis=-1);fe=np.linalg.norm(z['xy'][:,i]-xy,axis=-1).max(-1)
        stable=bool(z['alive'][hold,i].all() and z['support'][hold,i].all())
        tracked=bool(stable and (error[hold]<=.05).all() and (ce[hold]<=.02).all() and (fe[hold]<=.02).all())
        # Teacher stores native done codes; new traces store a boolean failure.
        survived=bool(z['alive'][-1,i] and z['done'][-1,i]!=1)
        returned=bool(survived and z['stand'][-int(np.ceil(2/dt)):,i].all())
        assert (stable,tracked,returned,tracked and returned)==(c['stable_hold'],c['tracked_hold'],c['returned_stand'],c['complete'])
        if 'whole_interval_stand' in c:
            assert c['whole_interval_stand']==bool(z['alive'][:,i].all() and z['stand'][:,i].all() and survived)
        if 'peak_motor' in c:
            assert abs(c['peak_motor']-float(error[hold].max()))<1e-8
            assert abs(c['peak_com']-float(ce[hold].max()))<1e-8
            assert abs(c['peak_feet']-float(fe[hold].max()))<1e-8
    assert sum(c['complete'] for c in r['cases'])==r['complete']
    return dict(cases=len(r['cases']),complete=r['complete'],holds=r['stable_holds'],returned=r['returned'],
        quiet=config.get('quiet',False),actor=config.get('actor','physical-teacher'),config=config)


def main():
    from .rig import Rig
    torch.set_num_threads(2);rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    poses=np.load(PRIOR/'goals.npz')['poses'];reports={i:read(ROOT/f'step-{i:02d}/report.json') for i in range(1,31)}
    outcomes=[];hash_checks=0
    physical=(2,3,4,5,6,7,8,9,10,14,16,17,18,19,20,21,22,23,24,25,26,28,29,30)
    for step,r in reports.items():
        root=ROOT/f'step-{step:02d}';protocol=root/'protocol.json'
        inputs=read(protocol).get('inputs',{}) if protocol.exists() else r.get('inputs',{})
        for path,h in inputs.items():assert digest(Path(path))==h;hash_checks+=1
        for path,h in r.get('outputs',{}).items():assert digest(Path(path))==h;hash_checks+=1
        if step in physical:
            for item in r['results']:outcomes.append(dict(step=step,folder=item['folder'],**gates(rig.model,poses,root/item['folder'],item)))
    for step in (12,13):assert reports[step]['trace_hash']==digest(ROOT/'step-11/trace.npz')
    models=[ROOT/'step-15/inherited',ROOT/'step-15/relative',ROOT/'step-18/relative',ROOT/'step-21/model',ROOT/'step-22/model',ROOT/'step-23/model']
    training=[]
    for model in models:
        r=read(model/'report.json');initial=torch.load(model/'initial.pt',weights_only=True);final=torch.load(model/'final.pt',weights_only=True)
        assert r['updates']==4000 and r['batch_size']==512
        assert (initial['net.4.weight']==0).all() and (initial['net.4.bias']==0).all()
        assert all(torch.isfinite(v).all() for v in final.values())
        assert any(not torch.equal(v,final[k]) for k,v in initial.items())
        assert read(model/'history.json')[-1]['update']==4000
        training.append(dict(folder=str(model),**r))
    for left,right in ((models[0],models[1]),(models[0],models[4]),(models[2],models[3]),(models[3],models[5])):
        a=torch.load(left/'initial.pt',weights_only=True);b=torch.load(right/'initial.pt',weights_only=True)
        assert all(torch.equal(v,b[k]) for k,v in a.items())
    teacher=np.load(TEACHER/'training-samples.npz')
    assert set(teacher['world'][~teacher['validation']])==set(range(6))
    assert set(teacher['world'][teacher['validation']])=={6,7}
    assert len(teacher['x'])==9072 and (~teacher['validation']).sum()==6804
    for r in training:
        active='step-22' in r['folder'] or 'step-23' in r['folder']
        mask=(~teacher['quiet']) if active else np.ones(len(teacher['x']),dtype=bool)
        assert r['train_examples']==int((mask&~teacher['validation']).sum())
        assert r['heldout_examples']==int((mask&teacher['validation']).sum())
    # Check every saved raw-proposal noise stream against its dedicated generator.
    for r in reports[25]['results']:
        noise=np.load(ROOT/'step-25'/r['folder']/'noise.npz')['noise']
        generator=torch.Generator().manual_seed(r['config']['seed'])
        for row in noise:
            assert np.array_equal((torch.randn(row.shape,generator=generator)*.005).numpy(),row)
    baseline30=np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
    quiet30=quiet60=0
    for r in reports[19]['results']:
        z=np.load(ROOT/'step-19'/r['folder']/'trace.npz')
        assert all(np.array_equal(z[k],baseline30[k]) for k in ('qpos','issued'));assert r['quiet_passed']==16;quiet30+=16
    baseline60=None
    for r in reports[29]['results']:
        z=dict(np.load(ROOT/'step-29'/r['folder']/'trace.npz'))
        if baseline60 is None:
            baseline60=z
            assert all(np.array_equal(z[k][:len(baseline30[k])],baseline30[k]) for k in ('qpos','issued'))
        else:assert all(np.array_equal(z[k],baseline60[k]) for k in ('qpos','issued','alive'))
        quiet60+=r['quiet_passed']
    arena=Path('logs/mimickit-training/perturb-arena-learned-20261008-01/stand.onnx')
    assert digest(arena)=='c87c272a0cf60d857de45f186eb5c64b7c63652ca5acf7a5ba050afebe6ad172'
    report=dict(completed_steps=list(range(1,31)),hash_checks=hash_checks,physical_gates_and_projection=True,
        paired_initializations_identical=True,holdouts_excluded=True,noise_streams_verified=True,
        quiet30_exact=quiet30,quiet60_passed=quiet60,quiet60_cases=48,training=training,
        supervised_updates=sum(r['updates'] for r in training),supervised_draws=sum(r['updates']*r['batch_size'] for r in training),
        fit_wall_seconds=sum(r['wall_seconds'] for r in training),ppo_transitions=0,
        physical_evaluation_cases=sum(r['cases'] for r in outcomes),teacher_queries=reports[11]['queries'],
        diagnostic_inherited_replay_cases=8,outcomes=outcomes,selected_recipe=reports[27]['selected_recipe'],
        arena_changed=False,bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        report_hashes={str(ROOT/f'step-{i:02d}/report.json'):digest(ROOT/f'step-{i:02d}/report.json') for i in reports})
    save(ROOT/'final-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('outcomes','report_hashes')},indent=2))


if __name__=='__main__':main()

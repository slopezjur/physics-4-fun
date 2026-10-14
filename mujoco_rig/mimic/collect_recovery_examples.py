"""Harvest and replay physical recovery examples; no optimizer or policy clock."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def articulated_step_feet(case, quality):
    articulation = quality['articulation_first_5s']
    hip = np.asarray(articulation['hip']['range_rad']).reshape(2, 3)
    knee = np.asarray(articulation['knee']['range_rad'])
    stepped = np.asarray(case['steps_by_foot']) > 0
    return np.flatnonzero(stepped & (np.linalg.norm(hip,axis=-1)>=.20) & (knee>=.25)).tolist()


def example_checks(case, quality):
    return dict(recovered=case['recovered'], measured_step=case['recovered_with_step'],
        leg_excursion=bool(articulated_step_feet(case,quality)),
        bounded_contact_slip=sum(quality['contact_slip_distance_by_foot_m'])<=.05,
        bounded_reversals=sum(quality['rapid_contact_reversals_by_foot'])<=2,
        bounded_steps=sum(case['steps_by_foot'])<=2)


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate, selection_cases
    from .recovery_quality import RecoveryQuality
    from .rig import Rig
    torch.set_num_threads(1)
    mp_util.init(0,1,'cpu',None)
    rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='native_reactive_motion_examples_v1',
        reference_sha256=sha256(args.reference), ball_model_sha256=sha256(args.ball_model),
        character_model_sha256=sha256(rig.path), source_sha256=sha256(Path(__file__)),
        curriculum='Fixed development cardinal torso impacts at 2.5 and 3 m/s; one hit, 33-second episodes',
        thresholds=dict(hip_vector_range_rad=.20,knee_range_rad=.25,contact_slip_total_m=.05,
                        rapid_reversals=2,maximum_steps=2), teachers=[],clips=[],
        scope='Mechanically filtered physical reactions; naturalness requires visual review; no training or Arena selection')
    (args.out/'runner.py').write_bytes(Path(__file__).read_bytes())
    def save():
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for index,checkpoint in enumerate(args.checkpoint):
        contract=json.loads((checkpoint.parent/'contract.json').read_text())
        _,Task=task_classes(contract)
        count=32*len(args.speeds)
        task=Task(rig,args.reference,count,'cpu',ball_path=args.ball_model,
            speed_min=2.,speed_max=max(args.speeds),episode_seconds=30.,recovery_mode='step_v7')
        validate_contract(contract,task.observation_contract())
        agent=PPOAgent(yaml.safe_load((checkpoint.parent/'agent.yaml').read_text()),task,'cpu')
        agent.load(str(checkpoint))
        actor=StandPolicy(agent).eval()
        quality=RecoveryQuality(task,actor)
        cases=[dict(c,name=f'teacher-{index}-speed-{speed:g}-{c["name"]}')
            for speed in args.speeds for c in selection_cases(speed,33.,task.dt)]
        actions,observations,qpos,qvel,mask,plant_qpos=[],[],[],[],[],[]
        signals=[]
        @torch.no_grad()
        def policy(obs):
            action=quality(obs)
            observations.append(obs.numpy().copy());actions.append(action.numpy().copy())
            return action
        def record(task,step,active,done):
            quality.record(task,step,active,done)
            qpos.append(task.engine.native_qpos().numpy().copy())
            qvel.append(task.engine._array('qvel')[:,:rig.model.nv].numpy().copy())
            plant_qpos.append(task.engine._array('qpos').numpy().copy())
            mask.append(active.numpy().copy())
            # The task's own meter records every step. Rejections can overlap.
            signals.append(task.recovery_signals.drain())
        metrics,_=evaluate(task,policy,cases,record)
        metrics['motion_quality']=quality.report()
        summaries={key:sum(s[key] for s in signals) for key in signals[0] if key not in
            ('scope','mean_post_hit_slip_proxy_m_s','credited_events_per_1000_post_hit_steps')}
        teacher=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            metrics=metrics,reward_signals=summaries,candidates=[])
        report['teachers'].append(teacher)
        arrays=dict(action=np.asarray(actions),observation=np.asarray(observations),qpos=np.asarray(qpos),
                    qvel=np.asarray(qvel),active=np.asarray(mask),plant_qpos=np.asarray(plant_qpos))
        trace=args.out/f'teacher-{index}-trace.npz'
        np.savez_compressed(trace,**arrays)
        teacher['trace_sha256']=sha256(trace)
        selected=[]
        for i,case in enumerate(metrics['cases']):
            checks=example_checks(case,metrics['motion_quality']['cases'][i])
            row=dict(case=i,checks=checks,accepted=all(checks.values()))
            teacher['candidates'].append(row)
            if row['accepted']:selected.append(i)
        print(json.dumps(dict(teacher=index,recovered=metrics['recovered_hits'],
            stepped=metrics['recovered_with_step'],selected=selected,reward_signals=summaries)),flush=True)
        save()
        if not selected:
            continue
        # Replay complete eight-posture blocks to retain identical initial states.
        blocks=sorted({i//8 for i in selected})
        ids=[i for block in blocks for i in range(block*8,block*8+8)]
        replay=Task(rig,args.reference,len(ids),'cpu',ball_path=args.ball_model,
                    speed_min=2.,speed_max=max(args.speeds),episode_seconds=30.,recovery_mode='step_v7')
        replay_cases=[cases[i] for i in ids]
        errors=np.zeros(len(ids))
        replay_step=[0]
        def replay_actor(obs):
            step=replay_step[0]
            return torch.from_numpy(arrays['action'][step,ids])
        def replay_record(task,step,active,done):
            error=np.max(np.abs(task.engine.native_qpos().numpy()-arrays['qpos'][step,ids]),axis=-1)
            errors[:]=np.maximum(errors,np.where(active.numpy(),error,0.))
            replay_step[0]=step+1
        replay_metrics,_=evaluate(replay,replay_actor,replay_cases,replay_record)
        for i in selected:
            position=ids.index(i)
            if not replay_metrics['cases'][position]['recovered_with_step'] or errors[position]>1e-5:
                teacher['candidates'][i]['accepted']=False
                teacher['candidates'][i]['replay_failed']=True
                continue
            start=max(0,metrics['cases'][i]['first_hit_step']-1)
            # Keep a full transient and a settled tail, not 30 seconds of idle.
            end=min(len(arrays['qpos']),start+math.ceil(7./task.dt))
            clip=args.out/f'teacher-{index}-case-{i}-motion.npz'
            np.savez_compressed(clip,qpos=arrays['qpos'][start:end,i],qvel=arrays['qvel'][start:end,i],dt=task.dt)
            report['clips'].append(dict(path=str(clip),sha256=sha256(clip),teacher=index,case=i,
                moving_feet=articulated_step_feet(metrics['cases'][i],metrics['motion_quality']['cases'][i]),
                replay_max_qpos_error=float(errors[position]),replay_recovered=True,frames=end-start,
                source_license='Project-generated physical policy rollouts; underlying rig/source attribution retained'))
        save()
    report['both_feet_covered']=set(foot for clip in report['clips'] for foot in clip['moving_feet'])=={0,1}
    report['ready_for_motion_prior_review']=bool(report['clips']) and report['both_feet_covered']
    report['amp_loader_ready']=False
    save()
    print(json.dumps(dict(clips=len(report['clips']),both_feet=report['both_feet_covered'],
                         ready=report['ready_for_motion_prior_review'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','reference','ball-model','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,action='append',required=True)
    parser.add_argument('--speeds',type=float,nargs='+',default=[2.5,3.])
    run(parser.parse_args())

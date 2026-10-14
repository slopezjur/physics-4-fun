"""Matched learning variants, with sustained Stand evaluated only for recipe gains."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import traceback

import numpy as np
import torch

from .audit_stand_failures import read,save
from .ball_study_support import FOUNDATION,body_observer,checkpoint,protect_used,save_bodies
from .chest_ball_curriculum import curriculum
from .chest_reward_variant import reward_variant
from .evaluate_stand_ball_boundary import evaluate as native_ball
from .evaluate_stand_ball_gpu import evaluate as gpu_ball,compare as backend_compare
from .evaluate_stand_sustained import evaluate_duration,verify_duration
from .measure_stand_learning_curve import rng_state,restore_rng,tree_hash
from .policy import StandPolicy
from .prepare_stand_ball_pair import prepare_pair
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from .run_chest_ball_pilot import source_assessment,compare as compare_seed
from . import run_stand_ball_pilot as pilot
from .stand_experiment_evidence import load_actor,seal


def impacts(actor,rig,reference,world,out,settings):
    out.mkdir()
    actor.eval().requires_grad_(False)
    original=tensor_hash(actor.state_dict())
    results,parity,bodies={},{},{}
    frames=int(np.ceil(settings['seconds']/(rig.model.opt.timestep*rig.decimation)))
    for mass,speed in settings['assessment_levels']:
        assert mass==5
        for sign in (1,-1):
            name=f'{mass}kg-{speed:g}mps-{sign:+d}'
            results[name]={}
            for backend,evaluate in (('native',native_ball),('gpu',gpu_ball)):
                folder=out/f'{name}-{backend}'
                if backend=='native':
                    journal={}
                    with body_observer(journal,frames):
                        row=evaluate(rig,reference,actor.cpu(),world,folder,settings,speed,sign)
                    bodies[name]=save_bodies(folder,journal)
                else:
                    row=evaluate(rig,reference,actor,world,folder,settings,speed,sign)
                assert row['actor_hash']==original
                results[name][backend]=row
            parity[name]=backend_compare(results[name]['native'],results[name]['gpu'],rig.model.opt.timestep*rig.decimation)
            save(out/'impact-progress.json',results)
    chest=all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies'] and 'Head' not in c['clusters'][0]['bodies']
              for b in bodies.values() for c in b['cases'])
    result=dict(actor_hash=original,impacts=results,backend_parity=parity,chest_first=bool(chest),
                admitted=all(r['passed'] for r in parity.values()),full_interval_stand=None,
                quiet={},quiet_assessed=False,cases=64)
    save(out/'report.json',result)
    return result


def recipe_gain(initial,final,speed):
    target=[f'5kg-{speed:g}mps-{sign:+d}' for sign in (1,-1)]
    checks={}
    for backend in pilot.BACKENDS:
        before={name:r[backend]['recovery_passes'] for name,r in initial['impacts'].items()}
        after={name:r[backend]['recovery_passes'] for name,r in final['impacts'].items()}
        checks[backend]=dict(no_regressions=all(after[k]>=v for k,v in before.items()),
            strict_target_gain=sum(after[k] for k in target)>sum(before[k] for k in target),
            all_hits=all(r[backend]['hit_cases']==8 for r in final['impacts'].values()),before=before,after=after)
    eligible=final['admitted'] and final['chest_first'] and all(
        c['no_regressions'] and c['strict_target_gain'] and c['all_hits'] for c in checks.values())
    return dict(eligible=bool(eligible),backends=checks)


def run(args):
    if not __debug__:
        raise RuntimeError('Assertions required')
    s=read(args.protocol)
    fixed=dict(schema='chest_learning_screen_v1',seeds=[301027,301028],actors=['source'],
               arms=['quiet','single_ball'],envs=64,rollout_steps=64,training_mass_kg=5,
               quiet_fraction=.5,quiet_retention_seconds=120,assessment_levels=[[5,1],[5,1.4]],
               training_speed_m_s=1.4,seconds=12,launch_seconds=5,launch_distance_m=.25)
    if any(s.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Unsupported bounded learning screen')
    n=s['updates_per_arm']
    if not (0<n<=300 and s['maximum_updates']==4*n and s['maximum_transitions']==4*n*4096
            and s['maximum_evaluation_cases']==320 and 0<s['maximum_training_seconds_per_arm']<=600):
        raise ValueError('Incorrect budget')
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0,1,'cuda:0',None)
    torch.set_num_threads(1)
    dependencies=[Path(__file__).with_name(p) for p in ('chest_ball_curriculum.py','chest_reward_variant.py','run_chest_ball_pilot.py')]
    parent,inputs,sources=protect_used(args.out,[Path(__file__),*dependencies],args.protocol,s)
    config=deepcopy(parent['config'])
    for k,v in s.get('config_overrides',{}).items():
        if k not in ('discount','td_lambda','task_reward_weight','disc_reward_weight'):
            raise ValueError('Unsupported algorithm override')
        config[k]=v
    if 'actor_learning_rate' in s:
        if not 1e-5<=s['actor_learning_rate']<=1e-3:
            raise ValueError('Unbounded actor learning rate')
        config['actor_optimizer']['learning_rate']=s['actor_learning_rate']
    source=dict(config=config,demonstrations=parent['demonstrations'])
    save(args.out/'protocol.json',dict(inputs=inputs,sources=sources,settings=s,**source,
        source_baselines_reused=True,automatic_promotion=False,
        conditional_stand='Only an arm with replicated absolute impact gain receives fresh 120-second Stand assessment; unassessed retention remains null'))
    rig,reference=Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    world=args.out/'ball-5kg.xml'
    write_world(args.out/'dummy.xml',FOUNDATION/'ball-5kg.xml',world,5.)
    results={}
    training_settings=dict(s,launch_seconds=s.get('training_launch_seconds',s['launch_seconds']),
                           seconds=s.get('training_episode_seconds',s['seconds']))
    with curriculum(training_settings),reward_variant(s):
        for seed in s['seeds']:
            folder=args.out/str(seed)
            folder.mkdir()
            initial=source_assessment(seed,s)
            pair=prepare_pair(rig,reference,world,source,checkpoint(seed,'source'),seed,folder/'starts')
            assert all(tensor_hash(StandPolicy(a).state_dict())==initial['actor_hash'] for a,_,_ in pair.values())
            finals,training={},{}
            for arm in (('quiet','single_ball') if seed==301027 else ('single_ball','quiet')):
                training[arm]=pilot.train_arm(pair,arm,seed,folder/'starts',folder/arm,s)
                agent,task,_=pair[arm]
                state,before=rng_state(task),pilot.pair_fingerprint(pair)
                try:
                    finals[arm]=impacts(StandPolicy(agent),rig,reference,world,folder/arm/'evaluation',s)
                finally:
                    restore_rng(task,state)
                after=pilot.pair_fingerprint(pair)
                restored=tree_hash(state)==tree_hash(rng_state(task))
                save(folder/arm/'evaluation-isolation.json',dict(before=before,after=after,rng_restored=restored))
                assert before==after and restored
            results[str(seed)]=dict(initial=initial,final=finals,training=training,
                recipe={arm:recipe_gain(initial,final,s['training_speed_m_s']) for arm,final in finals.items()})
            save(folder/'report.json',results[str(seed)])
            save(args.out/'progress.json',dict(results=results))
            del pair,agent,task
            gc.collect()
            torch.cuda.empty_cache()
    eligible={arm:all(r['recipe'][arm]['eligible'] for r in results.values()) for arm in s['arms']}
    cases=256
    for arm,yes in eligible.items():
        if not yes:
            continue
        for seed in s['seeds']:
            folder=args.out/str(seed)/arm
            actor=load_actor(rig,reference,source,folder/'final/model.pt')
            result=results[str(seed)]['final'][arm]
            for backend in pilot.BACKENDS:
                path=folder/'evaluation'/f'quiet-{backend}'
                result['quiet'][backend]=evaluate_duration(rig,reference,'target_pd',backend,actor,path,seconds=120.)
                verify_duration(path,120.)
                assert result['quiet'][backend]['actor_hash']==result['actor_hash']
                cases+=8
            result.update(quiet_assessed=True,full_interval_stand=all(r['standing_passes']==8 for r in result['quiet'].values()),cases=80)
            save(folder/'evaluation'/'report.json',result)
            del actor
            gc.collect()
            torch.cuda.empty_cache()
    for seed,result in results.items():
        result['comparison']=compare_seed(result['initial'],result['final'],s['training_speed_m_s'])
        comparison=result['comparison']
        comparison['all_arms_stand_retained']=comparison.pop('stand_retained')
        comparison['source_and_ball_stand_retained']=(result['initial']['full_interval_stand']
            and result['final']['single_ball']['full_interval_stand'])
        for row in comparison['backends'].values():
            row['passed']=bool(comparison['source_and_ball_stand_retained'] and row['all_shots_hit']
                and row['no_direction_or_level_regressions'] and row['strict_selected_level_gain'])
        comparison['passed']=all(row['passed'] for row in comparison['backends'].values()) and all(
            f['admitted'] and f['chest_first'] for f in result['final'].values())
        save(args.out/seed/'report.json',result)
    gains={arm:bool(yes and all(r['final'][arm]['full_interval_stand'] for r in results.values())) for arm,yes in eligible.items()}
    report=dict(results=results,impact_eligible_arms=eligible,replicated_recipe_gain=gains,
        replicated_impact_practice_gain=all(r['comparison']['passed'] for r in results.values()),
        evaluation_cases=cases,updates=sum(a['updates'] for r in results.values() for a in r['training'].values()),
        transitions=sum(a['transitions'] for r in results.values() for a in r['training'].values()),
        training_seconds=sum(a['training_seconds'] for r in results.values() for a in r['training'].values()),promoted=False)
    assert cases<=320 and report['updates']==s['maximum_updates'] and report['transitions']==s['maximum_transitions']
    save(args.out/'report.json',report)
    seal(args.out,inputs,sources,verified=True,evaluation_cases=cases,training_updates=report['updates'],
         replicated_recipe_gain=gains,promoted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args=p.parse_args()
    existed=args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise

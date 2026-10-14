"""Bounded matched chest-impact learning with reused sealed source assessments."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .ball_study_support import FOUNDATION, PILOT, body_observer, checkpoint, protect_used, save_bodies
from .chest_ball_curriculum import curriculum
from .evaluate_stand_ball_gpu import compare as compare_backends
from .measure_stand_learning_curve import rng_state, restore_rng, tree_hash
from .policy import StandPolicy
from .prepare_stand_ball_pair import prepare_pair
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from . import run_stand_ball_pilot as pilot
from .stand_experiment_evidence import seal


def sealed_report(folder):
    path = Path(folder)/'report.json'
    verification = read(Path(folder)/'final-verification.json')
    expected = {str(Path(p).resolve()): h for p,h in verification['output_hashes'].items()}
    if not verification['verified'] or expected.get(str(path.resolve())) != digest(path):
        raise ValueError(f'Unverified baseline: {path}')
    return read(path)


def source_assessment(seed, settings):
    prior = sealed_report(PILOT)['results'][str(seed)]['initial']
    initial = dict(actor_hash=prior['actor_hash'], quiet=prior['quiet'],
        full_interval_stand=prior['full_interval_stand'], impacts={}, cases=0,
        reused=True)
    for baseline in settings['baseline_impacts']:
        report = read(Path(baseline['root'])/'report.json')
        verification_name = baseline.get('verification', 'final-verification.json')
        verification = read(Path(baseline['root'])/verification_name)
        if not verification.get('verified', verification.get('inputs_and_sources_unchanged', False)):
            raise ValueError('Baseline verification failed')
        # The geometry completion uses its own summary; its original progress is
        # explicitly protected and selection was admitted in that completion.
        if 'rows_root' in baseline:
            progress = Path(baseline['rows_root'])/'progress.json'
            hashes = {str(Path(p).resolve()): h for p,h in verification['inputs'].items()}
            if (verification['output_sha256'] != digest(Path(baseline['root'])/'report.json')
                    or hashes.get(str(progress.resolve())) != digest(progress)
                    or not report['conditions'][baseline['condition']]['admitted']):
                raise ValueError('Geometry completion integrity/admission failed')
            rows = read(progress)['results']
        else:
            rows = sealed_report(baseline['root'])['results']
        chosen = [r for r in rows if r['seed']==seed and r['actor']=='source'
                  and r['condition']['name']==baseline['condition']]
        if len(chosen) != 2:
            raise ValueError('Two source directions required')
        for row in chosen:
            if not row['parity']['passed'] or not all(c['clusters'] and
                'Chest' in c['clusters'][0]['bodies'] and 'Head' not in c['clusters'][0]['bodies']
                for c in row['bodies']['cases']):
                raise ValueError('Unadmitted source condition')
            mass,speed = row['condition']['mass_kg'], row['condition']['speed_m_s']
            name = f'{mass}kg-{speed:g}mps-{row["sign"]:+d}'
            if not all(a['actor_hash']==initial['actor_hash'] for a in row['assessments'].values()):
                raise ValueError('Baseline actor mismatch')
            initial['impacts'][name] = row['assessments']
    expected = {f'{mass}kg-{speed:g}mps-{sign:+d}' for mass,speed in settings['assessment_levels'] for sign in (1,-1)}
    if set(initial['impacts']) != expected or not initial['full_interval_stand']:
        raise ValueError('Incomplete retained source baseline')
    return initial


def assess(actor, rig, reference, worlds, out, settings):
    native = pilot.native_ball
    def observed(*args, **kwargs):
        folder, cfg = args[4], args[5]
        journal = {}
        frames = int(np.ceil(cfg['seconds']/(rig.model.opt.timestep*rig.decimation)))
        with body_observer(journal, frames):
            result = native(*args, **kwargs)
        save_bodies(folder, journal)
        return result
    with patch.object(pilot, 'native_ball', observed):
        result = pilot.assess(actor, rig, reference, worlds, out, settings)
    parity, chest = {}, True
    for name, rows in result['impacts'].items():
        parity[name] = compare_backends(rows['native'],rows['gpu'],rig.model.opt.timestep*rig.decimation)
        bodies = read(out/f'{name}-native'/'body-report.json')
        chest &= all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies']
                     and 'Head' not in c['clusters'][0]['bodies'] for c in bodies['cases'])
    result.update(backend_parity=parity, admitted=all(r['passed'] for r in parity.values()),
                  chest_first=bool(chest))
    save(out/'report.json',result)
    return result


def compare(initial, finals, speed):
    result = pilot.compare_seed(initial,finals)
    retained = result['stand_retained']
    for backend,row in result['backends'].items():
        target = {label: sum(level[f'5kg-{speed:g}mps-{sign:+d}'] for sign in (1,-1))
                  for label,level in row['counts'].items()}
        row['selected_level_totals'] = target
        row['strict_selected_level_gain'] = target['single_ball']>max(target['source'],target['quiet'])
        row['passed'] = bool(retained and row['all_shots_hit'] and row['no_direction_or_level_regressions']
                             and row['strict_selected_level_gain'])
    result['passed'] = all(r['passed'] for r in result['backends'].values()) and all(
        r['admitted'] and r['chest_first'] for r in finals.values())
    return result


def run(args):
    if not __debug__:
        raise RuntimeError('Assertions required')
    settings = read(args.protocol)
    if settings['schema'] != 'matched_chest_pilot_v1' or settings['seeds'] != [301027,301028]:
        raise ValueError('Unsupported protocol')
    fixed = dict(actors=['source'],arms=['quiet','single_ball'],envs=64,rollout_steps=64,
                 training_mass_kg=5,quiet_fraction=.5,quiet_retention_seconds=120)
    if any(settings.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Unsupported curriculum contract')
    updates = settings['updates_per_arm']
    if not (0 < updates <= 300 and settings['maximum_updates']==4*updates
            and settings['maximum_transitions']==4*updates*4096
            and settings['maximum_evaluation_cases']==320 and len(settings['assessment_levels'])==2):
        raise ValueError('Unmatched declared budget')
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0,1,'cuda:0',None)
    torch.set_num_threads(1)
    parent,inputs,sources = protect_used(args.out,[Path(__file__),Path(__file__).with_name('chest_ball_curriculum.py')],args.protocol,settings)
    config = deepcopy(parent['config'])
    for key,value in settings.get('config_overrides',{}).items():
        if key not in ('discount','td_lambda','task_reward_weight','disc_reward_weight'):
            raise ValueError('Unsupported independent variable')
        config[key] = value
    source = dict(config=config,demonstrations=parent['demonstrations'])
    save(args.out/'protocol.json',dict(inputs=inputs,sources=sources,settings=settings,**source,
        reused_source_assessments=True,automatic_promotion=False,
        optimizer_state='Fresh, full AMP weights and normalizers from the same source in both arms'))
    rig,reference = Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    worlds = {5:args.out/'ball-5kg.xml'}
    write_world(args.out/'dummy.xml',FOUNDATION/'ball-5kg.xml',worlds[5],5.)
    results = {}
    with curriculum(settings):
        for seed in settings['seeds']:
            current = dict(settings, **settings.get('per_seed',{}).get(str(seed),{}))
            if any(current[k]!=settings[k] for k in ('updates_per_arm','maximum_training_seconds_per_arm',
                'seconds','launch_seconds','quiet_retention_seconds')):
                raise ValueError('Only dose and corresponding baselines may vary by seed')
            folder = args.out/str(seed)
            folder.mkdir()
            initial = source_assessment(seed,current)
            pair = prepare_pair(rig,reference,worlds[5],source,checkpoint(seed,'source'),seed,folder/'starts')
            assert all(tensor_hash(StandPolicy(agent).state_dict())==initial['actor_hash'] for agent,_,_ in pair.values())
            finals,training = {},{}
            order = ('quiet','single_ball') if seed==301027 else ('single_ball','quiet')
            for arm in order:
                training[arm] = pilot.train_arm(pair,arm,seed,folder/'starts',folder/arm,current)
                agent,task,_ = pair[arm]
                state,before = rng_state(task),pilot.pair_fingerprint(pair)
                try:
                    finals[arm] = assess(StandPolicy(agent),rig,reference,worlds,folder/arm/'evaluation',current)
                finally:
                    restore_rng(task,state)
                after = pilot.pair_fingerprint(pair)
                restored = tree_hash(state)==tree_hash(rng_state(task))
                save(folder/arm/'evaluation-isolation.json',dict(before=before,after=after,rng_restored=restored))
                if before!=after or not restored:
                    raise RuntimeError('Evaluation mutated live training')
            result = dict(initial=initial,final=finals,training=training,
                          comparison=compare(initial,finals,current['training_speed_m_s']))
            save(folder/'report.json',result)
            results[str(seed)] = result
            save(args.out/'progress.json',dict(results=results))
            del pair,agent,task
            gc.collect()
            torch.cuda.empty_cache()
    report = dict(results=results,replicated_improvement=all(r['comparison']['passed'] for r in results.values()),
        promoted=False,arena_changed=False,evaluation_cases=320,
        updates=sum(a['updates'] for r in results.values() for a in r['training'].values()),
        transitions=sum(a['transitions'] for r in results.values() for a in r['training'].values()),
        training_seconds=sum(a['training_seconds'] for r in results.values() for a in r['training'].values()))
    assert report['updates']==settings['maximum_updates'] and report['transitions']==settings['maximum_transitions']
    save(args.out/'report.json',report)
    seal(args.out,inputs,sources,verified=True,evaluation_cases=320,training_updates=report['updates'],
        replicated_improvement=report['replicated_improvement'],promoted=False)


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

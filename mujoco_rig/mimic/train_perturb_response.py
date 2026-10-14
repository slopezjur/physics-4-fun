"""Matched TD-lambda continuation with fixed physics, reward and rehearsal."""
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import read, save
from .runtime import activate
from .run_perturb_distance import PARENT, ASSESSMENT, MODEL, distance_curriculum
from .run_perturb_preview import attach_rehearsal, sealed


def train(args):
    from . import ball_study_support as support, run_stand_ball_pilot as pilot
    from .prepare_stand_ball_pair import prepare_pair
    from .stand_start_guard import compare
    from .rig import Rig
    from .stand_ball_native import write_world
    from .stand_experiment_evidence import seal
    from .replicate_stand_representations import tensor_hash
    from .prepare_chest_normalizer_swap import policy_state
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    assert args.seed in (401028,411028)
    quiet=ASSESSMENT/'selected-retention-native'
    sealed(PARENT,[MODEL,MODEL.with_suffix('.amp.pt')])
    sealed(ASSESSMENT,[quiet/n for n in ('trace.npz','report.json','contract.json')])
    full=torch.load(MODEL.with_suffix('.amp.pt'),map_location='cpu',weights_only=True)
    assert tensor_hash(policy_state(full))==read(quiet/'report.json')['actor_hash']
    with np.load(quiet/'trace.npz') as z:
        keep=z['time_s']<=16
        assert (z['alive_before'][keep]&z['balance'][keep]&z['posture'][keep]&z['pre_balance'][keep]&z['pre_posture'][keep]).all()
        obs,action=z['observation'][keep].reshape(-1,374).copy(),z['action'][keep].reshape(-1,33).copy()
    settings=dict(seeds=[301028],actors=['parent'],updates_per_arm=100,maximum_training_seconds_per_arm=420,
        checkpoint_interval=1000,optimization_seed=args.seed,td_lambda_by_arm=dict(control=.98,long_return=.995),
        evidence=[str(PARENT/'final-verification.json'),str(ASSESSMENT/'final-verification.json'),
            *[str(quiet/n) for n in ('trace.npz','report.json','contract.json')],
            'logs/mimickit-training/perturb-response-audit-20261008-01/report.json',
            'logs/mimickit-training/perturb-response-probe-20261008-01/report.json'])
    paths=[Path(__file__),Path('mujoco_rig/mimic/run_perturb_distance.py'),Path('mujoco_rig/mimic/run_perturb_preview.py'),
        Path(pilot.__file__),Path('mujoco_rig/mimic/prepare_stand_ball_pair.py'),
        Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit/learning/rl_util.py'),
        Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit/learning/ppo_agent.py')]
    with patch.object(support,'checkpoint',lambda *_:MODEL):
        parent,inputs,sources=support.protect_used(args.out,paths,Path('mujoco_rig/mimic/PERTURB_RESPONSE_LEARNING.md'),settings)
    parent=deepcopy(parent)
    parent['config']['actor_optimizer']['learning_rate']=3e-5
    assert parent['config']['td_lambda']==.98 and parent['config']['discount']==.99
    save(args.out/'protocol.json',dict(settings=settings,inputs=inputs,sources=sources,config=parent['config'],
        demonstrations=parent['demonstrations'],parent=str(MODEL),factor='TD lambda only; control .98 versus treatment .995'))
    rig,ref=Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    world=args.out/'ball-5kg.xml'
    write_world(args.out/'dummy.xml',support.FOUNDATION/'ball-5kg.xml',world,5)
    results={}
    for label,coefficient in settings['td_lambda_by_arm'].items():
        local=deepcopy(parent)
        local['config']['td_lambda']=coefficient
        with distance_curriculum(True):
            starts=args.out/f'{label}-starts'
            pair=prepare_pair(rig,ref,world,local,MODEL,args.seed,starts)
            if label=='long_return':
                result=compare(args.out/'control-starts/single_ball',starts/'single_ball')
                assert result['passed']
                save(starts/'cross-arm-admission.json',result)
            agent=pair['single_ball'][0]
            assert agent._td_lambda==coefficient and agent._discount==.99
            rehearsal=attach_rehearsal(agent,obs,action)
            print('Training',args.seed,label,'lambda',coefficient,flush=True)
            results[label]=pilot.train_arm(pair,'single_ball',args.seed,starts,args.out/label,settings)
            assert rehearsal['calls']==4800
            results[label].update(rehearsal=rehearsal,td_lambda=coefficient)
            save(args.out/'progress.json',results)
            print('Complete',label,results[label],flush=True)
            del pair,agent
            torch.cuda.empty_cache()
    save(args.out/'report.json',dict(results=results,training_updates=200,transitions=819200,selected=False))
    seal(args.out,inputs,sources,verified=True,training_updates=200)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    train(args)

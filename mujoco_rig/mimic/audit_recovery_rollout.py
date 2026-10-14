"""Measure recovery credit on frozen GPU rollouts; never update policy weights."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from envs.base_env import EnvMode, DoneFlags
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .rig import Rig
    from .risk_exploration import install
    torch.set_num_threads(1)
    mp_util.init(0,1,'cuda:0',None)
    rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
    checkpoint_hash=sha256(args.checkpoint)
    contract=json.loads((args.checkpoint.parent/'contract.json').read_text())
    _,Task=task_classes(contract)
    task=Task(rig,args.reference,2048,'cuda:0',ball_path=args.ball_model,
        speed_min=2.,speed_max=2.5,quiet_fraction=.25,episode_seconds=30.,
        seed=args.seed,target_set='all',recovery_mode='step_v7')
    validate_contract(contract,task.observation_contract())
    config=yaml.safe_load((args.checkpoint.parent/'agent.yaml').read_text())
    agent=PPOAgent(config,task,'cuda:0')
    agent.load(str(args.checkpoint))
    actor=StandPolicy(agent).eval()
    install(agent,.2,'support')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'runner.py').write_bytes(Path(__file__).read_bytes())
    report=dict(schema='frozen_recovery_rollout_audit_v1',checkpoint_sha256=checkpoint_hash,
        reference_sha256=sha256(args.reference),ball_model_sha256=sha256(args.ball_model),
        character_model_sha256=sha256(rig.path),source_sha256=sha256(Path(__file__)),
        envs=2048,evaluation_interval=64,seed=args.seed,seconds=30.,target_set='all',
        speed_range_m_s=[2.,2.5],quiet_fraction=.25,cohorts=[],optimizer_updates=0,
        scope='Training distribution assay, not unseen validation; failed worlds parked; first episodes only')
    for mode in ('mean','support_exploration'):
        torch.manual_seed(args.seed)
        task.rng.manual_seed(args.seed)
        task.set_mode(EnvMode.TRAIN)
        obs,_=task.reset()
        task.recovery_signals.drain()
        active=torch.ones(2048,device='cuda:0',dtype=torch.bool)
        quiet=~task.shot_enabled.clone()
        failures=torch.zeros_like(active)
        confirmed=torch.zeros_like(active)
        events=torch.zeros(2048,device='cuda:0')
        placements=torch.zeros(2048,2,device='cuda:0')
        shot_parameters=dict(target_body=task.shot_target_body.cpu().tolist(),
            speed=task.shot_speeds.cpu().tolist(),direction=task.directions.cpu().tolist(),
            enabled=task.shot_enabled.cpu().tolist(),launch_step=task.shot_steps.cpu().tolist())
        torch.cuda.synchronize()
        started=time.monotonic()
        with torch.no_grad():
            for _ in range(math.ceil(30./task.dt)):
                if mode=='mean':
                    action=actor(obs)
                else:
                    distribution=agent._model.eval_actor(agent._obs_norm.normalize(obs))
                    action=agent._a_norm.unnormalize(distribution.sample()).clamp(-1,1)
                obs,_,done,_=task.step(action)
                confirmed |= task.engine.hit & active
                diagnostic=task.placement.last_diagnostics
                events += diagnostic['useful_events']*active*task.engine.hit
                placements += diagnostic['measured_placements']*active[:,None]*task.engine.hit[:,None]
                failures |= active & (done==DoneFlags.FAIL.value)
                finished=active & (done!=DoneFlags.NULL.value)
                active &= ~finished
                ids=(done!=DoneFlags.NULL.value).nonzero().flatten()
                if len(ids):
                    obs,_=task.reset(ids)
                    task.shot_enabled[ids]=False
                    task.second_shot_enabled[ids]=False
        torch.cuda.synchronize()
        row=dict(mode=mode,wall_seconds=time.monotonic()-started,
            quiet_worlds=int(quiet.sum()),quiet_failures=int((failures & quiet).sum()),
            hit_worlds=int((~quiet).sum()),confirmed_hits=int(confirmed.sum()),
            hit_failures=int((failures & ~quiet).sum()),event_worlds=int((events>0).sum()),
            events_by_world=events.cpu().tolist(),placements_by_world=placements.cpu().tolist(),
            signals=task.recovery_signals.drain(),shot_parameters=shot_parameters)
        report['cohorts'].append(row)
        if sha256(args.checkpoint)!=checkpoint_hash:
            raise RuntimeError('Frozen checkpoint changed during assay')
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:value for key,value in row.items() if key not in
            ('events_by_world','placements_by_world','shot_parameters')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','reference','ball-model','checkpoint','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=300953)
    run(parser.parse_args())

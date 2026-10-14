"""Strict long native quiet hold under the isolated recovery motor contract."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .rig import Rig
    from .validate_stand_foundation import evaluate
    torch.set_num_threads(1)
    mp_util.init(0,1,'cpu',None)
    rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
    contract=json.loads((args.checkpoint.parent/'contract.json').read_text())
    _,Task=task_classes(contract)
    legacy=Task(rig,args.reference,8,'cpu',ball_path=args.ball_model)
    validate_contract(contract,legacy.observation_contract())
    agent=PPOAgent(yaml.safe_load((args.checkpoint.parent/'agent.yaml').read_text()),legacy,'cpu')
    checkpoint_hash=sha256(args.checkpoint)
    agent.load(str(args.checkpoint))
    actor=StandPolicy(agent).eval()
    task=Task(rig,args.reference,8,'cpu',ball_path=args.ball_model,recovery_mode='step_v7',
              recovery_motor_radians=args.radius)
    gate_frames=torch.zeros(8,dtype=torch.long)
    @torch.no_grad()
    def policy(obs):
        # Foundation evaluate advances motors directly, so supply the same
        # pre-command observation as ReactiveBallTask.step would.
        task.engine.policy_control.observe(obs)
        gate_frames.add_((task.engine.policy_control.gate>0).long())
        return actor(obs)
    args.out.mkdir(parents=True,exist_ok=False)
    sources=('check_authority_quiet.py','validate_stand_foundation.py','target_control.py',
             'recovery_motor.py','reactive_ball_task.py')
    for name in sources:(args.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report,traces=evaluate(task,policy,args.seconds)
    if sha256(args.checkpoint)!=checkpoint_hash:raise RuntimeError('Source checkpoint changed')
    report.update(checkpoint_sha256=checkpoint_hash,character_model_sha256=sha256(rig.path),
        ball_model_sha256=sha256(args.ball_model),motor_contract=task.engine.policy_control.contract(),
        gate_active_frames_by_posture=gate_frames.tolist(),optimizer_updates=0,
        source_hashes={name:sha256(Path(__file__).with_name(name)) for name in sources},
        scope='Frozen actor with experimental motor semantics; no ball launched, no resets after failure')
    np.savez_compressed(args.out/'trace.npz',**traces)
    report['trace_sha256']=sha256(args.out/'trace.npz')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],gate_frames=gate_frames.tolist(),seconds=args.seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','checkpoint','reference','ball-model','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--radius',type=float,default=1.25)
    parser.add_argument('--seconds',type=float,default=600.)
    run(parser.parse_args())

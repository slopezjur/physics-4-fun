"""Frozen successor actors and optional direction/feedback interventions."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np

from .audit_stand_failures import digest, read, save
from . import ball_study_support as support
from . import evaluate_stand_ball_boundary as native_module
from . import evaluate_stand_ball_gpu as gpu_module
from . import run_frozen_ball_screen as screen
from .ball import launch_state
from .evaluate_stand_push_feedback_v2 import HeldCommandPolicy, verify_hold
from .replicate_stand_representations import tensor_hash


def run(args):
    settings = read(args.protocol)
    checkpoints = settings.get('checkpoints',{})
    if len(settings['actors'])!=1 or (not checkpoints and settings['actors']!=['source']):
        raise ValueError('One declared actor group required; custom groups need checkpoints')
    original_checkpoint = support.checkpoint
    seals = {}
    for item in checkpoints.values():
        root = Path(item['root'])
        verification = read(root/'final-verification.json')
        if not verification['verified']:
            raise ValueError('Unverified actor source')
        seals[str(root)] = verification
        known = {str(Path(p).resolve()): h for p,h in verification['output_hashes'].items()}
        model = Path(item['checkpoint'])
        for path in (model,model.with_suffix('.amp.pt')):
            if known.get(str(path.resolve())) != digest(path):
                raise ValueError('Checkpoint differs from its source seal')
        if str(root/'final-verification.json').replace('\\','/') not in [p.replace('\\','/') for p in settings['evidence']]:
            raise ValueError('Source seal must be a protected input')

    def checkpoint(seed, label):
        return Path(checkpoints[str(seed)]['checkpoint']) if checkpoints else original_checkpoint(seed,label)

    original_protect = screen.protect_used
    def protect(out, paths, protocol, cfg):
        return original_protect(out,[*paths,Path(__file__),Path(__file__).with_name('evaluate_stand_push_feedback_v2.py')],protocol,cfg)

    angle = float(settings.get('direction_degrees',0.))
    if not -180 <= angle <= 180:
        raise ValueError('Invalid direction')
    radians = np.deg2rad(angle)
    def aimed(target,direction,speed,gravity,distance):
        sign = float(direction[0])
        return launch_state(target,[sign*np.cos(radians),sign*np.sin(radians),0.],speed,gravity,distance)

    def evaluator(original):
        def evaluate(rig,reference,actor,world,out,cfg,speed,sign):
            before = tensor_hash(actor.state_dict())
            if settings.get('feedback','live') == 'hold_prelaunch':
                hold = int(np.ceil(cfg['launch_seconds']/(rig.model.opt.timestep*rig.decimation)))-1
                wrapped = HeldCommandPolicy(actor,hold)
            elif settings.get('feedback','live') == 'live':
                wrapped,hold = actor,None
            else:
                raise ValueError('Unsupported feedback intervention')
            result = original(rig,reference,wrapped,world,out,cfg,speed,sign)
            if before != tensor_hash(actor.state_dict()):
                raise RuntimeError('Inner frozen actor changed')
            evidence = dict(inner_actor_hash=before,feedback=settings.get('feedback','live'),direction_degrees=angle)
            if hold is not None:
                with np.load(out/'trace.npz',allow_pickle=False) as data:
                    trace = {k:data[k] for k in ('action','observation','alive_before')}
                evidence['held_command_prefix_error'] = verify_hold(trace,actor.cpu(),hold)
                evidence['hold_frame'] = hold
            save(out/'intervention.json',evidence)
            return result
        return evaluate

    with ExitStack() as stack:
        for module in (screen,support):
            stack.enter_context(patch.object(module,'checkpoint',checkpoint))
        stack.enter_context(patch.object(screen,'protect_used',protect))
        stack.enter_context(patch.object(screen,'native_evaluate',evaluator(screen.native_evaluate)))
        stack.enter_context(patch.object(screen,'gpu_evaluate',evaluator(screen.gpu_evaluate)))
        stack.enter_context(patch.object(native_module,'launch_state',aimed))
        stack.enter_context(patch.object(gpu_module,'launch_state',aimed))
        screen.run(args)


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

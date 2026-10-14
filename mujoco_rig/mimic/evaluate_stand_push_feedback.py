"""Causal frozen-command ablation of verified actors under matched chest pushes."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch

from .audit_stand_failures import read, save
from .evaluate_stand_challenges import evaluate_challenge, verify_challenge
from .replicate_stand_representations import tensor_hash
from .stand_experiment_evidence import protect, seal, load_actor
from .runtime import activate
from .rig import Rig


class HeldCommandPolicy(torch.nn.Module):
    def __init__(self, actor, hold_frame):
        super().__init__()
        self.actor = actor
        self.hold_frame = hold_frame
        self.frame = 0
        self.held = None

    def forward(self, observation):
        if self.frame <= self.hold_frame:
            action = self.actor(observation)
            if self.frame == self.hold_frame:
                self.held = action.clone()
        else:
            action = self.held.clone()
        self.frame += 1
        return action


@torch.no_grad()
def verify_hold(trace, actor, hold_frame):
    count = min(len(trace['action']), hold_frame+1)
    largest = 0.
    for frame in range(count):
        expected = actor(torch.from_numpy(trace['observation'][frame])).numpy()
        live = trace['alive_before'][frame]
        largest = max(largest, float(np.abs(expected[live]-trace['action'][frame,live]).max()))
    assert largest <= 2e-6
    if len(trace['action']) > hold_frame:
        for case in range(8):
            live = trace['alive_before'][hold_frame:,case]
            assert np.array_equal(trace['action'][hold_frame:,case][live],
                np.broadcast_to(trace['action'][hold_frame,case], (int(live.sum()),33)))
    return largest


def run(args):
    spec = read(args.specification)
    assert spec['seconds']==12. and spec['hold_after_seconds']==2.
    activate(args.mimickit)
    torch.set_num_threads(1)
    inputs,sources=protect(args.prior,args.out,[Path(__file__)],extra_inputs=(args.specification,))
    canonical={str(Path(p).resolve()) for p in inputs}
    checkpoint,baseline=Path(spec['checkpoint']),Path(spec['baseline_root'])
    assert str(checkpoint.resolve()) in canonical and str((baseline/'final-verification.json').resolve()) in canonical
    assert read(baseline/'final-verification.json')['verified']
    baseline_protocol=read(baseline/'protocol.json')
    assert Path(baseline_protocol['specification']['checkpoint']).resolve()==checkpoint.resolve()
    prior=read(args.prior/'protocol.json')
    protocol=dict(inputs=inputs,sources=sources,config=prior['config'],demonstrations=prior['demonstrations'],
        model_sha256=prior['model_sha256'],reference_sha256=prior['reference_sha256'],specification=spec,
        maximum_cases=16*len(spec['scenarios']),training_updates=0,
        intervention='Hold the command issued at two seconds; preserve local PD and every physical parameter',
        prefix_verification='Replay each own observation through the live actor; atol 2e-6; held commands exact')
    save(args.out/'protocol.json',protocol)
    rig=Rig.load(args.out/'dummy.xml',policy_exclude=())
    actor=load_actor(rig,args.out/'stationary_reference.npz',protocol,checkpoint)
    original_hash=tensor_hash(actor.state_dict())
    dt=rig.model.opt.timestep*rig.decimation
    hold_frame=int(np.ceil(spec['hold_after_seconds']/dt))
    results,comparisons={},{}
    for scenario in spec['scenarios']:
        name=scenario['name']
        assert scenario in baseline_protocol['specification']['scenarios']
        pair,comparison={},{}
        for backend in ('native','gpu'):
            folder=args.out/f'{name}-{backend}'
            controller=HeldCommandPolicy(copy.deepcopy(actor),hold_frame)
            pair[backend]=evaluate_challenge(rig,args.out/'stationary_reference.npz','target_pd',backend,
                controller,folder,seconds=12.,scenario=scenario)
            trace,_=verify_challenge(folder,12.,scenario)
            error=verify_hold(trace,actor,hold_frame)
            baseline_report=read(baseline/f'{name}-{backend}'/'report.json')
            with np.load(baseline/f'{name}-{backend}'/'trace.npz') as base:
                assert np.allclose(trace['qpos_before'][0],base['qpos_before'][0],rtol=0,atol=1e-6)
                assert np.allclose(trace['qvel_before'][0],base['qvel_before'][0],rtol=0,atol=1e-6)
                assert np.allclose(trace['action'][0],base['action'][0],rtol=0,atol=1e-6)
            comparison[backend]=dict(held_stand=pair[backend]['standing_passes'],held_recovery=pair[backend]['recovery_passes'],
                live_stand=baseline_report['standing_passes'],live_recovery=baseline_report['recovery_passes'],
                held_survival=pair[backend]['mean_survival_seconds'],live_survival=baseline_report['mean_survival_seconds'],
                replay_max_error=error)
        results[name],comparisons[name]=pair,comparison
        save(args.out/'progress.json',dict(results=results,comparisons=comparisons))
    assert tensor_hash(actor.state_dict())==original_hash
    save(args.out/'report.json',dict(results=results,comparisons=comparisons,cases=16*len(results),training_updates=0,
        diagnostic_ablation_only=True,controller_promoted=False,arena_changed=False))
    seal(args.out,inputs,sources,verified=True,cases=16*len(results),training_updates=0,
        held_commands_exact=True,own_observation_replay_verified=True,initial_pairs_verified=True)
    print(comparisons,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())

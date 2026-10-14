"""Matched continuation testing recovery generalization across flight distances."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate
from .run_perturb_preview import repeated_curriculum, attach_rehearsal, sealed

BASE = Path('logs/mimickit-training')
PARENT = BASE/'perturb-preview-learning-20261008-01'
ASSESSMENT = BASE/'perturb-preview-evaluation-20261008-01'
MODEL = PARENT/'single_ball/final/model.pt'


@contextmanager
def distance_curriculum(mixed):
    from . import stand_ball_task, run_stand_ball_pilot as pilot
    from .representation_task import RepresentationMixin
    with repeated_curriculum():
        base, evidence_base = stand_ball_task.StandBallAMPTask, pilot.TrainingEvidence

        class DistanceTask(base):
            def distances(self):
                combination = (self.ids//8+self.schedule_phase) % 8
                return torch.where((combination>=4) & mixed, .46, .25)

            def step(self, action):
                enabled = (self.steps==self.launch_step) | (self.steps==self.launch_step+self.repeat_gap)
                if self.arm=='quiet':
                    enabled[:] = False
                ids = self.ids[enabled]
                if len(ids):
                    target = self.engine.get_body_pos(0)[ids,self.rig.model.body('Chest').id-1]
                    direction = self.direction[ids]
                    distance = self.distances()[ids,None]
                    speed = self.launch_speed_m_s
                    gravity = torch.as_tensor(self.rig.model.opt.gravity,device=self.device,dtype=torch.float32)
                    self.engine.launch_ball(ids,target-distance*direction,speed*direction-.5*gravity*(distance/speed))
                self.launch_records.append(enabled.detach().clone())
                return RepresentationMixin.step(self,action)

        class Evidence(evidence_base):
            def step(self,action):
                distance = self.task.distances().detach().clone()
                result = super().step(action)
                self.rows[-1]['launch_distance_m'] = distance
                return result

            def launch(self,ids,position,velocity):
                assert not bool(self.launched[ids].any())
                self.launched[ids] = True
                target = self.task.engine.get_body_pos(0)[ids,self.task.rig.model.body('Chest').id-1]
                direction = self.task.direction[ids]
                d,v = self.task.distances()[ids,None],self.task.launch_speed_m_s
                gravity = torch.as_tensor(self.task.rig.model.opt.gravity,device=self.task.device,dtype=torch.float32)
                self.position_error[ids] = (position-(target-d*direction)).abs().amax(-1)
                self.velocity_error[ids] = (velocity-(v*direction-.5*gravity*(d/v))).abs().amax(-1)
                self.original_launch(ids,position,velocity)
                q,vel = self.task.engine.ball_state()
                self.position_error[ids] = torch.maximum(self.position_error[ids],(q[ids,:3]-position).abs().amax(-1))
                self.velocity_error[ids] = torch.maximum(self.velocity_error[ids],(vel[ids,:3]-velocity).abs().amax(-1))

            def flush(self,path):
                result = super().flush(path)
                with np.load(path) as z:
                    combination = (np.arange(64)[None]//8+z['schedule_phase'])%8
                    expected = np.where((combination>=4) & mixed,np.float32(.46),np.float32(.25))
                    assert np.array_equal(z['launch_distance_m'],expected)
                return result

        with patch.object(stand_ball_task,'StandBallAMPTask',DistanceTask),patch.object(pilot,'TrainingEvidence',Evidence):
            yield


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
    quiet = ASSESSMENT/'selected-retention-native'
    sealed(PARENT,[MODEL,MODEL.with_suffix('.amp.pt')])
    sealed(ASSESSMENT,[quiet/n for n in ('trace.npz','report.json','contract.json')])
    full = torch.load(MODEL.with_suffix('.amp.pt'),map_location='cpu',weights_only=True)
    assert tensor_hash(policy_state(full))==read(quiet/'report.json')['actor_hash']
    with np.load(quiet/'trace.npz') as z:
        keep = z['time_s']<=16
        assert (z['alive_before'][keep]&z['balance'][keep]&z['posture'][keep]&z['pre_balance'][keep]&z['pre_posture'][keep]).all()
        obs,action = z['observation'][keep].reshape(-1,374).copy(),z['action'][keep].reshape(-1,33).copy()
    settings = dict(seeds=[301028],actors=['parent'],updates_per_arm=100,maximum_training_seconds_per_arm=420,
        checkpoint_interval=1000,optimization_seed=391028,evidence=[str(PARENT/'final-verification.json'),
        str(ASSESSMENT/'final-verification.json'),*[str(quiet/n) for n in ('trace.npz','report.json','contract.json')]])
    paths=[Path(__file__),Path('mujoco_rig/mimic/run_perturb_preview.py'),Path(pilot.__file__),
           Path('mujoco_rig/mimic/prepare_stand_ball_pair.py'),Path('mujoco_rig/mimic/chest_ball_curriculum.py')]
    with patch.object(support,'checkpoint',lambda *_:MODEL):
        parent,inputs,sources = support.protect_used(args.out,paths,Path('mujoco_rig/mimic/PERTURB_DISTANCE_PLAN.md'),settings)
    parent=deepcopy(parent)
    parent['config']['actor_optimizer']['learning_rate']=3e-5
    save(args.out/'protocol.json',dict(settings=settings,inputs=inputs,sources=sources,config=parent['config'],
         demonstrations=parent['demonstrations'],parent=str(MODEL),training_variable='Fixed versus mixed launch distance only'))
    rig,reference=Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    world=args.out/'ball-5kg.xml'
    write_world(args.out/'dummy.xml',support.FOUNDATION/'ball-5kg.xml',world,5)
    results={}
    for label,mixed in (('fixed',False),('mixed',True)):
        with distance_curriculum(mixed):
            starts=args.out/f'{label}-starts'
            pair=prepare_pair(rig,reference,world,parent,MODEL,391028,starts)
            if mixed:
                common=compare(args.out/'fixed-starts/single_ball',starts/'single_ball')
                assert common['passed']
                save(starts/'cross-arm-admission.json',common)
            rehearsal=attach_rehearsal(pair['single_ball'][0],obs,action)
            print('Training',label,flush=True)
            results[label]=pilot.train_arm(pair,'single_ball',391028,starts,args.out/label,settings)
            results[label]['rehearsal']=rehearsal
            assert rehearsal['calls']==4800
            save(args.out/'progress.json',results)
            print('Complete',label,results[label],flush=True)
            del pair
            torch.cuda.empty_cache()
    save(args.out/'report.json',dict(results=results,training_updates=200,transitions=819200,selected=False))
    seal(args.out,inputs,sources,verified=True,training_updates=200)


def compact(folder):
    """Keep outcome evidence after the original evaluator verifies every control frame."""
    path=folder/'trace.npz'
    original_hash=digest(path)
    with np.load(path) as z:
        removed=[k for k in z.files if k.startswith('substep_') or k in ('pending_before','issued_entry','reference_motor_q','active_targets')]
        keep={k:z[k] for k in z.files if k not in removed}
    np.savez_compressed(path,**keep)
    save(folder/'trace-retention.json',dict(full_trace_sha256=original_hash,compact_trace_sha256=digest(path),
         removed_after_full_control_verification=removed,report_sha256=digest(folder/'report.json'),
         full_control_verification_completed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    train(args)

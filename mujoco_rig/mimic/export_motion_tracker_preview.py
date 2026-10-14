"""Explicit latest-weight preview with constant standing goals and active feedback."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import onnxruntime as ort
import torch
import yaml

from .baseline import sha256
from .motion_tracker import MotionTracker, encode_pose
from .runtime import activate


class FixedStandingPreview(torch.nn.Module):
    def __init__(self, tracker, stationary):
        super().__init__()
        self.tracker = tracker
        self.register_buffer('stationary_pose', encode_pose(stationary)[0])
        self.register_buffer('pose_scale', torch.tensor([.1]*3 + [1.]*6 + [.25]*39))

    def forward(self, core):
        # Three identical fixed goals represent every lookahead. No motion clock.
        goals = core[:, 217:361].reshape(-1, 3, 48)
        delta = (goals - self.stationary_pose) / self.pose_scale
        velocity = core.new_zeros(core.shape[0], 3, 45)
        command = torch.cat((delta, velocity), -1).flatten(1)
        requested = core.new_ones(core.shape[0], 1)
        return self.tracker(torch.cat((core, command, requested), -1)).clamp(-1., 1.)


def run(args):
    if args.out.exists():
        raise ValueError('Require a fresh preview bundle')
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from envs.base_env import DoneFlags
    from .balance_observation import task_classes
    from .perturb import RECOVERY
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .rig import Rig

    torch.set_num_threads(1)
    report = json.loads(args.checkpoint.with_name('report.json').read_text())
    protocol = json.loads(args.checkpoint.with_name('protocol.json').read_text())
    checkpoint_hash = sha256(args.checkpoint)
    if args.checkpoint.name != 'model.pt' or report['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Preview requires intact completed final weights')
    if protocol['source_checkpoint_sha256'] != sha256(args.source):
        raise ValueError('Tracker source mismatch')
    source_contract = json.loads(args.source.with_name('contract.json').read_text())
    QuietTask, _ = task_classes(source_contract)
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    task = QuietTask(rig, args.stationary, 1, 'cpu', engine_factory=NativeDummyEngine)
    source_agent = PPOAgent(yaml.safe_load(args.source.with_name('agent.yaml').read_text()), task, 'cpu')
    source_agent.load(str(args.source))
    source = StandPolicy(source_agent).eval()
    tracker = MotionTracker(source).eval()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    prefix = '_model._actor_layers.'
    tracker.load_state_dict({k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)}, strict=True)
    checks = {'_model._action_dist._mean_net.weight': torch.eye(30),
              '_model._action_dist._mean_net.bias': torch.zeros(30),
              '_a_norm._mean': torch.zeros(30), '_a_norm._std': torch.ones(30)}
    if any(not torch.equal(saved[k],v) for k,v in checks.items()):
        raise ValueError('Unsupported physical readout')
    if any(not torch.equal(v,tracker.source.state_dict()[k]) for k,v in source.state_dict().items()):
        raise ValueError('Frozen source tensors differ')
    actor = FixedStandingPreview(tracker, task.reference.qpos[:1]).eval()
    args.out.mkdir(parents=True)
    sample = task.observations()
    with torch.no_grad():
        torch.onnx.export(actor, sample, str(args.out/'stand.onnx'), input_names=['observation'],
            output_names=['action'], dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'}},
            opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(args.out/'stand.onnx'), providers=['CPUExecutionProvider'])
    traces = []
    max_error = 0.
    task.reset()
    with torch.no_grad():
        for step in range(1800):
            observation = task.observations()
            action = actor(observation)
            exported = session.run(None, {'observation':observation.numpy()})[0]
            max_error = max(max_error,float(np.abs(exported-action.numpy()).max()))
            _,_,done,_ = task.step(action)
            traces.append(dict(step=step, observation=observation[0].tolist(),
                               action=action[0].tolist(), qpos_after=task.engine.native_qpos()[0].tolist()))
            if done[0] == DoneFlags.FAIL.value:
                break
    if max_error > 1e-5:
        raise RuntimeError(f'ONNX action mismatch: {max_error}')
    for source_path,name in ((rig.path,'dummy.xml'),(args.ball_model,'dummy_ball.xml'),
            (args.stationary,'stand_reference.npz'),(args.stationary.with_suffix('.json'),'stand_reference.json'),
            (args.stationary.parent/'ATTRIBUTION.md','ATTRIBUTION.md')):
        shutil.copy2(source_path,args.out/name)
    pose=task.reference.qpos[0].tolist()
    frames=dict(dt=60.,reference_sha256=sha256(args.stationary),qpos=[pose,pose],qvel=[[0.]*45]*2)
    (args.out/'reference_frames.json').write_text(json.dumps(frames))
    contract=task.observation_contract()
    contract.update(actor_sha256=sha256(args.out/'stand.onnx'),reference_frames_sha256=sha256(args.out/'reference_frames.json'),
        reference_duration=60.,scene_posture=0,source_checkpoint_sha256=checkpoint_hash,
        deployment_status='Experimental latest tracker preview; constant standing request active; impact recovery unvalidated',
        preview_adapter=dict(schema='fixed_standing_motion_tracker_preview_v1',external_inputs=363,tracker_inputs=643,
            motion_request=1,selected_foot=False,phase=False,target_velocity='zero',
            stationary_baseline_sha256=sha256(args.stationary)))
    (args.out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    fixture=dict(schema='mimic_push_v1',force_frame='MuJoCo_world',application_point='body_COM',
        bundle=str(args.out.resolve()),model_sha256=sha256(rig.path),actor_sha256=contract['actor_sha256'],
        reference_sha256=sha256(args.stationary),control_timestep=task.dt,recovery=RECOVERY,
        cases=[dict(name='fixed-preview-quiet',phase=0.,force=[0.,0.,0.],body='Chest',start_step=60,
                    duration_steps=6,trial_steps=1800)])
    (args.out/'godot-protocol.json').write_text(json.dumps(fixture))
    env=os.environ.copy()
    import mujoco
    env.update(P4F_MIMIC_PUSH_PROTOCOL=str((args.out/'godot-protocol.json').resolve()),
        P4F_MIMIC_REPLAY_OUTPUT=str((args.out/'godot-replay.json').resolve()),P4F_MUJOCO_LIBRARY=str(Path(mujoco.__file__).parent))
    with (args.out/'godot-replay.log').open('w') as log:
        subprocess.run([str(args.godot),'--headless','--path',str(Path.cwd()),
            'res://Scenes/RL/Isaac3/MuJoCo/MimicPerturbReplay.tscn'],env=env,stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    replay=json.loads((args.out/'godot-replay.json').read_text())[0]
    errors={key:max(float(np.max(np.abs(np.asarray(row[key])-traces[row['step']][key])))
                    for row in replay['trace']) for key in ('observation','action','qpos_after')}
    if errors['action'] > 1e-4 or errors['qpos_after'] > 1e-4 or errors['observation'] > 1e-3:
        raise RuntimeError(f'Godot/native preview parity failed: {errors}')
    result=dict(schema='motion_tracker_fixed_preview_export_v1',checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash,source_checkpoint_sha256=sha256(args.source),
        onnx_max_action_error=max_error,godot_native_errors=errors,
        quiet_survived_30s=len(traces)==1800 and not bool(done[0]==DoneFlags.FAIL.value),
        godot_quiet_survived_30s=replay['metrics']['survived'],experimental_preview=True,
        full_tracker_admission=report['foundation_passed'],unseen_hits_validated=False,
        request='User explicitly requested latest final weights despite failed tracking selection',
        adapter_scope='Fixed standing goal only; feedback active; no moving reference or step clock',
        source_sha256=sha256(Path(__file__)))
    (args.out/'export_report.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if sha256(args.checkpoint)!=checkpoint_hash:
        raise RuntimeError('Checkpoint changed during export')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','checkpoint','source','stationary','ball-model','godot','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())

"""Prepare a verified sustained-Stand actor for the matching Godot adapter."""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
import onnxruntime as ort
import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate
from .rig import Rig
from .stand_experiment_evidence import load_actor


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    report = read(args.training / 'report.json')
    verification = read(args.training / 'final-verification.json')
    assert verification['verified'] and verification['sustained_stand_verified']
    assert report['sustained_stand_verified']
    checkpoint = Path(report['final_checkpoint'])
    protected = {str(Path(p).resolve()): h for p, h in verification['output_hashes'].items()}
    for path in (checkpoint, checkpoint.with_suffix('.amp.pt'), args.training / 'dummy.xml',
                 args.training / 'stationary_reference.npz', args.training / 'contract.json'):
        assert digest(path) == protected[str(path.resolve())]
    contract = read(args.training / 'contract.json')
    assert contract['observation_contract'] == 'mimic_stand_representation_target_pd_v1'
    assert contract['num_obs'] == 374 and contract['motor_count'] == 33
    assert contract['passive_joint_stiffness'] == 0
    rig = Rig.load(args.training / 'dummy.xml', policy_exclude=())
    actor = load_actor(rig, args.training / 'stationary_reference.npz',
                       read(args.training / 'protocol.json'), checkpoint)
    with np.load(args.training / f"sustained-{report['ppo_updates']}" / 'native' / 'trace.npz') as trace:
        observations = trace['observation'].reshape(-1, 374)
        indices = np.unique(np.r_[np.arange(8), np.linspace(0, len(observations)-1, 2048, dtype=int)])
        sample = torch.from_numpy(observations[indices].copy())
    assert args.out.is_dir() and not (args.out / 'stand.onnx').exists()
    torch.onnx.export(actor, sample[:1], str(args.out / 'stand.onnx'),
                      input_names=['observation'], output_names=['action'],
                      dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(args.out / 'stand.onnx'), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        error = float(np.abs(session.run(None, {'observation': sample.numpy()})[0] - actor(sample).numpy()).max())
    assert np.isfinite(error) and error <= 1e-5, f'Export conversion mismatch: {error}'
    for source, destination in [('dummy.xml', 'dummy.xml'), ('stationary_reference.npz', 'stand_reference.npz'),
                                ('stationary_reference.json', 'stand_reference.json')]:
        shutil.copy2(args.training / source, args.out / destination)
    with np.load(args.training / 'stationary_reference.npz') as reference:
        q = reference['qpos'][0].astype(np.float32).tolist()
        v = reference['qvel'][0].astype(np.float32).tolist()
    save(args.out / 'reference_frames.json', dict(dt=60., reference_sha256=contract['reference_sha256'],
                                                 qpos=[q, q], qvel=[v, v]))
    contract.update(actor_sha256=digest(args.out / 'stand.onnx'),
                    reference_frames_sha256=digest(args.out / 'reference_frames.json'),
                    reference_duration=60., scene_posture=0,
                    deployment_status='User-requested sustained Stand preview; matching 33-motor target-PD adapter required')
    save(args.out / 'contract.json', contract)
    bundle = 'res://' + args.out.resolve().relative_to(Path.cwd().resolve()).as_posix()
    scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicStandArena.tscn')
    text = scene.read_text(encoding='utf-8')
    text = re.sub(r'^ModelPath = ".*"', f'ModelPath = "{bundle}/dummy.xml"', text, flags=re.M)
    text = re.sub(r'^BundleDirectory = ".*"', f'BundleDirectory = "{bundle}"', text, flags=re.M)
    text = re.sub(r'^TrialSeconds = .*', 'TrialSeconds = 60.0', text, flags=re.M)
    text = text.replace('Mimic Stand — spring-free learned torques', 'Mimic Stand — learned natural standing')
    help_text = 'R: restart · P: pause\\nStand seed 301026 · 1,200 learning updates\\n60-second standing verified on eight starts and both backends'
    text = re.sub(r'(\[node name="Help"[^\]]*\]\s*)text = ".*"',
                  lambda match: match[1] + f'text = "{help_text}"', text)
    (args.out / 'MimicStandArena.ready.tscn').write_text(text, encoding='utf-8')
    save(args.out / 'viewer-selection.ready.json', dict(schema='mimic_viewer_selection_v1', task='stand',
        bundle=bundle, checkpoint_sha256=digest(checkpoint), actor_sha256=contract['actor_sha256'],
        experimental_preview=True, selected_policy='verified-final', previous=read(Path('logs/mimickit-viewer/stand.json'))['bundle']))
    save(args.out / 'arena-selection.json', dict(schema='mimic_sustained_stand_preview_v1',
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), seed=301026, updates=report['ppo_updates'],
        onnx_sha256=contract['actor_sha256'], onnx_max_action_error=error, conversion_observations=len(sample),
        model_sha256=contract['model_sha256'], reference_sha256=contract['reference_sha256'],
        observation_contract=contract['observation_contract'], policy_actions=33, policy_observations=374,
        motor_control='learned reference-relative targets with delayed local PD', passive_joint_stiffness=0,
        sustained_evidence=str(args.training / 'final-verification.json'), selected=False,
        compiled=False, godot_run=False, standalone_tests_run=False, bundle=bundle,
        preparation_source_sha256=digest(Path(__file__))))
    print(dict(prepared=str(args.out), seed=301026, export_max_error=error, selected=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())

"""Select a completed Perturb pilot checkpoint for user-requested Arena inspection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

import numpy as np
import onnxruntime as ort
import torch

from .audit_stand_failures import digest, read, save
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import load_actor


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    verification = read(args.training/'final-verification.json')
    assert verification['verified']
    protected = {str(Path(p).resolve()): h for p, h in verification['output_hashes'].items()}
    for name in ('report.json', 'protocol.json', 'contract.json', 'dummy.xml',
                 'stationary_reference.npz', 'stationary_reference.json', 'final/model.pt',
                 'final/model.amp.pt', 'final/quiet-native/trace.npz'):
        path = args.training/name
        assert digest(path) == protected[str(path.resolve())], name
    report = read(args.training/'report.json')
    assert report['seed'] == 301028 and report['arm'] == 'perturb' and report['updates'] == 300
    assert report['final']['full_interval_stand']
    checkpoint = args.training/'final/model.pt'
    assert Path(report['final']['checkpoint']).resolve() == checkpoint.resolve()
    contract = read(args.training/'contract.json')
    assert contract['num_obs'] == 374 and contract['motor_count'] == 33
    assert contract['observation_contract'] == 'mimic_stand_representation_target_pd_v1'
    assert contract['passive_joint_stiffness'] == 0
    previous_ball = read(args.ball_source/'arena-selection.json')
    assert previous_ball['model_sha256'] == contract['model_sha256']
    for name, expected in (
        ('dummy.xml', contract['model_sha256']),
        ('dummy_ball.xml', previous_ball['ball_model_sha256']),
        ('dummy_ball_arena.xml', previous_ball['arena_model_sha256']),
    ):
        assert digest(args.ball_source/name) == expected, name
    rig = Rig.load(args.training/'dummy.xml', policy_exclude=())
    actor = load_actor(rig, args.training/'stationary_reference.npz',
        read(args.training/'protocol.json'), checkpoint)
    with np.load(args.training/'final/quiet-native/trace.npz') as trace:
        values = trace['observation'].reshape(-1, 374)
        indices = np.unique(np.r_[np.arange(8), np.linspace(0, len(values)-1, 2048, dtype=int)])
        sample = torch.from_numpy(values[indices].copy())
    args.out.mkdir(parents=True, exist_ok=False)
    torch.onnx.export(actor, sample[:1], str(args.out/'stand.onnx'),
        input_names=['observation'], output_names=['action'],
        dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}},
        opset_version=17, dynamo=False)
    # Validate the format conversion on saved observations, without new rollouts.
    session = ort.InferenceSession(str(args.out/'stand.onnx'), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        error = float(np.max(np.abs(session.run(None, {'observation': sample.numpy()})[0]-actor(sample).numpy())))
    assert np.isfinite(error) and error <= 1e-5, f'ONNX conversion mismatch: {error}'
    for source, target in (('dummy.xml', 'dummy.xml'), ('stationary_reference.npz', 'stand_reference.npz'),
                           ('stationary_reference.json', 'stand_reference.json')):
        shutil.copy2(args.training/source, args.out/target)
    for name in ('dummy_ball.xml', 'dummy_ball_arena.xml'):
        shutil.copy2(args.ball_source/name, args.out/name)
    with np.load(args.training/'stationary_reference.npz') as reference:
        q, v = (reference[name][0].astype(np.float32).tolist() for name in ('qpos', 'qvel'))
    save(args.out/'reference_frames.json', dict(dt=60., reference_sha256=contract['reference_sha256'],
        qpos=[q, q], qvel=[v, v]))
    contract.update(actor_sha256=digest(args.out/'stand.onnx'),
        reference_frames_sha256=digest(args.out/'reference_frames.json'), reference_duration=60., scene_posture=0,
        deployment_status='User-requested pilot preview; physical ball recovery is unvalidated')
    save(args.out/'contract.json', contract)
    scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicPerturbArena.tscn')
    viewer = Path('logs/mimickit-viewer/ball.json')
    shutil.copy2(scene, args.out/'prior-perturb-arena.tscn')
    shutil.copy2(viewer, args.out/'prior-viewer-selection.json')
    previous = read(viewer)
    bundle = 'res://'+args.out.resolve().relative_to(Path.cwd().resolve()).as_posix()
    checkpoint_hash = digest(checkpoint)
    text = scene.read_text(encoding='utf-8')
    text = re.sub(r'^ModelPath = ".*"', f'ModelPath = "{bundle}/dummy_ball_arena.xml"', text, flags=re.M)
    text = re.sub(r'^BundleDirectory = ".*"', f'BundleDirectory = "{bundle}"', text, flags=re.M)
    text = re.sub(r'^AutoFire = .*', 'AutoFire = false', text, flags=re.M)
    text = text.replace('button_pressed = true', 'button_pressed = false')
    for node, value in (
        ('Title', 'Mimic Perturb — learned Stand + small-push practice'),
        ('Help', 'R: restart and clear balls · B: launch · P: pause\\n'
            '10 kg at 4 m/s · automatic throws off\\n'
            f'Seed 301028 · brain {checkpoint_hash[:8]} · active joint PD · no passive springs\\n'
            'Trained with 60 N chest pulses · ball impacts are an inspection preview'),
    ):
        text = re.sub(rf'(\[node name="{node}"[^\]]*\]\s*)text = ".*"',
            lambda match: match[1]+f'text = "{value}"', text)
    scene.write_text(text, encoding='utf-8')
    save(viewer, dict(schema='mimic_viewer_selection_v1', task='ball', bundle=bundle,
        actor_sha256=contract['actor_sha256'], checkpoint_sha256=checkpoint_hash,
        previous=previous['bundle'], experimental_preview=True))
    save(args.out/'arena-selection.json', dict(schema='mimic_perturb_pilot_preview_v1',
        bundle=bundle, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        seed=301028, continuation_updates=300, actor_sha256=contract['actor_sha256'],
        model_sha256=contract['model_sha256'], reference_sha256=contract['reference_sha256'],
        ball_model_sha256=digest(args.out/'dummy_ball.xml'), arena_model_sha256=digest(args.out/'dummy_ball_arena.xml'),
        source_evidence=str(args.training/'final-verification.json'), source_ball_bundle=str(args.ball_source),
        onnx_max_action_error=error, conversion_observations=len(sample),
        passive_joint_stiffness=0, standing_pose_servo=True, policy_observations=374, policy_actions=33,
        training_disturbance='60 N Chest pulses for 0.1 seconds', ball_mass_kg=10., ball_speed_m_s=4., auto_fire=False,
        selected=True, promoted=False, experimental_preview=True, physical_ball_recovery_validated=False,
        standalone_tests_run=False, compiled=False, godot_run=False,
        selected_at_utc=datetime.now(timezone.utc).isoformat(), scene_sha256=digest(scene),
        preparation_source_sha256=digest(Path(__file__))))
    print(dict(bundle=bundle, checkpoint=checkpoint_hash, onnx_max_action_error=error), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--ball-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())

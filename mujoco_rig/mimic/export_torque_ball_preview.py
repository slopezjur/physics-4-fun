"""Reuse a torque Stand export in a projectile Arena without changing its actor or character."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from xml.etree import ElementTree as ET

from .arena_ball_pool import build
from .baseline import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def export(source, projectile_template, out):
    contract = read(source/'contract.json')
    source_selection = read(source/'arena-selection.json')
    if contract['observation_contract'] not in ('mimic_stationary_stand_torque_v1', 'mimic_stationary_stand_torque_v2'):
        raise ValueError('Expected the spring-free stationary torque contract')
    for filename, expected in (('dummy.xml', contract['model_sha256']),
                               ('stand.onnx', contract['actor_sha256']),
                               ('reference_frames.json', contract['reference_frames_sha256'])):
        if sha256(source/filename) != expected:
            raise ValueError(f'Source export hash mismatch: {filename}')
    checkpoint = Path(source_selection['checkpoint'])
    if sha256(checkpoint) != source_selection['checkpoint_sha256']:
        raise ValueError('Source final checkpoint changed after conversion')
    template = ET.parse(projectile_template).find("./worldbody/body[@name='ball']")
    if template is None:
        raise ValueError('Projectile template has no ball')
    geom = template.find('geom')
    mass = float(geom.get('mass'))
    radius = float(geom.get('size'))
    out.mkdir(parents=True, exist_ok=False)
    for filename in ('dummy.xml', 'stand.onnx', 'reference_frames.json', 'stand_reference.npz', 'contract.json'):
        shutil.copy2(source/filename, out/filename)
    tree = ET.parse(out/'dummy.xml')
    root = tree.getroot()
    root.find('worldbody').append(deepcopy(template))
    keyframe = root.find('keyframe')
    if keyframe is not None:
        root.remove(keyframe)
    tree.write(out/'dummy_ball.xml', encoding='utf-8', xml_declaration=True)
    build(out/'dummy_ball.xml', out/'dummy_ball_arena.xml', count=64)
    bundle = 'res://'+out.resolve().relative_to(Path.cwd().resolve()).as_posix()
    scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicPerturbArena.tscn')
    shutil.copy2(scene, out/'prior-perturb-arena.tscn')
    viewer = Path('logs/mimickit-viewer/ball.json')
    previous = read(viewer)
    text = scene.read_text(encoding='utf-8')
    text = re.sub(r'^ModelPath = ".*"', f'ModelPath = "{bundle}/dummy_ball_arena.xml"', text, flags=re.M)
    text = re.sub(r'^BundleDirectory = ".*"', f'BundleDirectory = "{bundle}"', text, flags=re.M)
    text = text.replace('Mimic Perturb — empty brain / passive ragdoll',
                        'Mimic Perturb — spring-free Stand brain')
    short_hash = source_selection['checkpoint_sha256'][:8]
    help_text = ('R: restart and clear balls · B: launch and reset auto countdown · P: pause\\n'
                 f'{mass:g} kg at 4 m/s · up to 64 independent balls\\n'
                 f'Stand brain {short_hash} · learned torques · no springs or pose servo\\n'
                 'Experimental impact preview · random angles and body targets')
    text = re.sub(r'(\[node name="Help"[^\]]*\]\s*)text = ".*"',
                  lambda match: match[1]+f'text = "{help_text}"', text)
    scene.write_text(text, encoding='utf-8')
    selection = dict(schema='mimic_torque_ball_preview_v1', bundle=bundle,
        checkpoint=str(checkpoint), checkpoint_sha256=source_selection['checkpoint_sha256'],
        actor_sha256=contract['actor_sha256'], model_sha256=contract['model_sha256'],
        ball_model_sha256=sha256(out/'dummy_ball.xml'), arena_model_sha256=sha256(out/'dummy_ball_arena.xml'),
        source_bundle=str(source), projectile_template=str(projectile_template),
        ball_mass_kg=mass, ball_radius_m=radius, scene_ball_speed_m_s=4., capacity=64,
        experimental_preview=True, impact_recovery_trained=False, unseen_hits_validated=False,
        passive_joint_stiffness=0, standing_pose_servo=False, motor_actuation_enabled=True,
        standalone_tests_run=False, compiled=False, godot_run=False,
        previous_selection=previous, selected_at_utc=datetime.now(timezone.utc).isoformat(),
        scene_sha256=sha256(scene))
    write(out/'arena-selection.json', selection)
    write(viewer, dict(schema='mimic_viewer_selection_v1', task='ball', bundle=bundle,
        actor_sha256=contract['actor_sha256'], checkpoint_sha256=selection['checkpoint_sha256'],
        previous=previous['bundle'], experimental_preview=True))
    print(json.dumps(dict(bundle=bundle, checkpoint=short_hash, ball_mass_kg=mass,
                          radius_m=radius, recovery_trained=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--projectile-template', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.projectile_template, args.out)

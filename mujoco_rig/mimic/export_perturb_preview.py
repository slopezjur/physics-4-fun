"""Export the evaluated learned policy and select it for reversible Arena inspection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

import numpy as np
import onnxruntime as ort
import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate
from .run_perturb_preview import sealed
from .rig import Rig
from .stand_experiment_evidence import load_actor


def main(args):
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    sealed(args.evaluation, [args.evaluation/'report.json'])
    result = read(args.evaluation/'report.json')
    name = result['selected']
    root, model = Path(result['selected_root']), Path(result['checkpoint'])
    sealed(root, [model, model.with_suffix('.amp.pt')])
    assert result['results'][name]['admitted']
    assert all(row['standing_passes']==8 for row in result['retention'].values())
    trace_path = args.evaluation/name/'hit-strong-plus-native/trace.npz'
    sealed(args.evaluation,[trace_path])
    contract_path = model.parent.parent/'contract.json'
    sealed(root,[contract_path,root/'dummy.xml',root/'stationary_reference.npz',root/'stationary_reference.json',root/'protocol.json'])
    contract = read(contract_path)
    assert contract['num_obs']==374 and contract['motor_count']==33 and contract['passive_joint_stiffness']==0
    assert contract['observation_contract']=='mimic_stand_representation_target_pd_v1'
    old = read(args.ball_source/'arena-selection.json')
    assert old['model_sha256']==contract['model_sha256']
    for filename, key in (('dummy.xml','model_sha256'),('dummy_ball.xml','ball_model_sha256'),('dummy_ball_arena.xml','arena_model_sha256')):
        assert digest(args.ball_source/filename)==old[key]
    rig = Rig.load(root/'dummy.xml',policy_exclude=())
    actor = load_actor(rig,root/'stationary_reference.npz',read(root/'protocol.json'),model)
    with np.load(trace_path) as z:
        obs = z['observation'].reshape(-1,374)
        ids = np.unique(np.r_[np.arange(8),np.linspace(0,len(obs)-1,2048,dtype=int)])
        sample = torch.from_numpy(obs[ids].copy())
    args.out.mkdir(parents=True,exist_ok=False)
    torch.onnx.export(actor,sample[:1],str(args.out/'stand.onnx'),input_names=['observation'],output_names=['action'],
        dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'}},opset_version=17,dynamo=False)
    session = ort.InferenceSession(str(args.out/'stand.onnx'),providers=['CPUExecutionProvider'])
    with torch.no_grad():
        error = float(abs(session.run(None,{'observation':sample.numpy()})[0]-actor(sample).numpy()).max())
    assert np.isfinite(error) and error<=1e-5
    for source,target in (('dummy.xml','dummy.xml'),('stationary_reference.npz','stand_reference.npz'),('stationary_reference.json','stand_reference.json')):
        shutil.copy2(root/source,args.out/target)
    for filename in ('dummy_ball.xml','dummy_ball_arena.xml'):
        shutil.copy2(args.ball_source/filename,args.out/filename)
    with np.load(root/'stationary_reference.npz') as ref:
        q,v = (ref[key][0].astype(np.float32).tolist() for key in ('qpos','qvel'))
    save(args.out/'reference_frames.json',dict(dt=60.,reference_sha256=contract['reference_sha256'],qpos=[q,q],qvel=[v,v]))
    contract.update(actor_sha256=digest(args.out/'stand.onnx'),reference_frames_sha256=digest(args.out/'reference_frames.json'),
        reference_duration=60.,scene_posture=0,deployment_status='Experimental learned recovery preview; 120-second quiet retention passed; heavy random Arena hits unvalidated')
    save(args.out/'contract.json',contract)
    scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicPerturbArena.tscn')
    viewer = Path('logs/mimickit-viewer/ball.json')
    shutil.copy2(scene,args.out/'prior-perturb-arena.tscn')
    shutil.copy2(viewer,args.out/'prior-viewer-selection.json')
    previous = read(viewer)
    bundle = 'res://'+args.out.resolve().relative_to(Path.cwd().resolve()).as_posix()
    checkpoint_hash = digest(model)
    outcomes = result['results'][name]['impacts']
    single, repeated = (outcomes[key]['native']['recovery_passes'] for key in ('strong-plus','repeat'))
    label = 'Stage33' if name=='stage33' else 'Repeated-hit continuation'
    text = scene.read_text(encoding='utf-8')
    for key,value in (('ModelPath',f'"{bundle}/dummy_ball_arena.xml"'),('BundleDirectory',f'"{bundle}"'),
                      ('UseLatestExport','false'),('AutoFire','false')):
        text,count = re.subn(rf'^{key} = .*',f'{key} = {value}',text,flags=re.M)
        assert count==1
    text = text.replace('button_pressed = true','button_pressed = false')
    help_text = ('R: restart and clear balls · B: launch · P: pause\\n'
        'Arena: 10 kg at 4 m/s · automatic throws off\\n'
        f'{label} · learned brain {checkpoint_hash[:8]}\\n'
        f'Controlled 5 kg / 1.6 m/s: single {single}/8 · two hits {repeated}/8\\n'
        '120 s Stand passed · heavy random hits remain experimental')
    for node,value in (('Title','Mimic Perturb — updated learned recovery preview'),('Help',help_text)):
        text,count = re.subn(rf'(\[node name="{node}"[^\]]*\]\s*)text = ".*"',lambda m:m[1]+f'text = "{value}"',text)
        assert count==1
    scene.write_text(text,encoding='utf-8')
    save(viewer,dict(schema='mimic_viewer_selection_v1',task='ball',bundle=bundle,actor_sha256=contract['actor_sha256'],
        checkpoint_sha256=checkpoint_hash,previous=previous['bundle'],experimental_preview=True))
    save(args.out/'arena-selection.json',dict(schema='mimic_learned_recovery_preview_v1',bundle=bundle,
        checkpoint=str(model),checkpoint_sha256=checkpoint_hash,actor_sha256=contract['actor_sha256'],
        model_sha256=digest(args.out/'dummy.xml'),reference_sha256=contract['reference_sha256'],
        ball_model_sha256=digest(args.out/'dummy_ball.xml'),arena_model_sha256=digest(args.out/'dummy_ball_arena.xml'),
        evaluation=str(args.evaluation/'report.json'),evaluation_sha256=digest(args.evaluation/'report.json'),
        selected=name,experimental_preview=True,promoted=False,onnx_max_action_error=error,
        conversion_observations=len(sample),quiet_retention_seconds=120,arena_ball_mass_kg=10,arena_ball_speed_m_s=4,
        auto_fire=False,compiled=False,godot_run=False,selected_at_utc=datetime.now(timezone.utc).isoformat(),
        scene_sha256=digest(scene),exporter_sha256=digest(Path(__file__))))
    print(dict(bundle=bundle,selected=name,checkpoint=checkpoint_hash,onnx_max_action_error=error),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--ball-source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())

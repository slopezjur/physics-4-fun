"""Reconstruct response limits and support geometry from existing frozen evidence."""
from pathlib import Path

import mujoco
import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate

BASE = Path('logs/mimickit-training')
ROOT = BASE/'perturb-preview-evaluation-20261008-01'
TRAIN = BASE/'perturb-preview-learning-20261008-01'
OUT = BASE/'perturb-response-audit-20261008-01'


def run():
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .replicate_stand_representations import verify_control_trace
    from .run_perturb_preview import sealed
    rig = Rig.load(TRAIN/'dummy.xml', policy_exclude=())
    actor = load_actor(rig, TRAIN/'stationary_reference.npz', read(TRAIN/'protocol.json'),
                       TRAIN/'single_ball/final/model.pt')
    actor.cpu().eval()
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    names = [rig.model.joint(int(j)).name for j in joints]
    qi = rig.model.jnt_qposadr[joints]
    inputs, result = {}, {}
    for label, folder in [('repeat', 'hit-repeat-clear-native'), ('single', 'hit-strong-plus-native')]:
        path = ROOT/'repeated'/folder
        sealed(ROOT, [path/n for n in ('trace.npz', 'report.json', 'contract.json')])
        for name in ('trace.npz', 'report.json', 'contract.json'):
            inputs[str(path/name)] = digest(path/name)
        with np.load(path/'trace.npz') as z:
            z = {k:z[k] for k in z.files}
        contract = read(path/'contract.json')
        verify_control_trace(z, contract)
        obs = torch.from_numpy(z['observation'].reshape(-1, 374))
        with torch.no_grad():
            raw = actor.action_norm.unnormalize(actor.mean(actor.actor(actor.obs_norm.normalize(obs)))).numpy().reshape(*z['action'].shape)
        alive = z['alive_before']
        assert np.max(np.abs(raw.clip(-1, 1)[alive]-z['action'][alive])) < 2e-6
        cases = []
        for case in range(8):
            contacts = np.flatnonzero(z['normal_impulse_ns'][:,case]>0)
            first = int(contacts[0])
            first_end = int(contacts[np.flatnonzero(np.diff(contacts)>1)[0]]) if (np.diff(contacts)>1).any() else int(contacts[-1])
            valid = np.logical_and.reduce([z[k][:,case] for k in ('balance','posture','pre_balance','pre_posture')])
            failures = np.flatnonzero(alive[:,case]&~valid)
            end = int(failures[0]) if len(failures) else min(first+120, len(alive)-1)
            window = (np.arange(len(alive))>=first)&(np.arange(len(alive))<=end)&alive[:,case]
            action_saturation = (np.abs(raw[window,case])>=1).mean(0)
            target = z['reference_motor_q'][window,case]+.25*z['action'][window,case]
            clipped = (target < rig.model.jnt_range[joints,0]-1e-7)|(target > rig.model.jnt_range[joints,1]+1e-7)
            torque = np.abs(z['substep_torque'][window,:,case]/np.asarray(contract['torque_scale_nm']))
            # Post-contact geometric evidence: COM projection relative to foot geom bounds.
            data = mujoco.MjData(rig.model)
            foot_bodies = [rig.model.body(n).id for n in ('Foot_L','Foot_R')]
            support_bodies = foot_bodies+[rig.model.body(n).id for n in ('Toe_L','Toe_R')]
            foot_geoms = [i for i,b in enumerate(rig.model.geom_bodyid) if b in support_bodies]
            geometry = []
            for frame in sorted(set([first-1, first_end, min(first+30,end), min(first+60,end), end])):
                data.qpos[:] = z['qpos_after'][frame,case]
                data.qvel[:] = z['qvel_after'][frame,case]
                mujoco.mj_forward(rig.model,data)
                corners = []
                for g in foot_geoms:
                    if rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                        corners.extend(data.geom_xpos[g]+(signs*rig.model.geom_size[g])@data.geom_xmat[g].reshape(3,3).T)
                corners = np.asarray(corners)
                geometry.append(dict(time=float(z['time_s'][frame]), com=data.subtree_com[1].tolist(),
                    feet=data.xpos[foot_bodies].tolist(), loads=z['foot_loads_n'][frame,case].tolist(),
                    foot_bounds_xy=[corners[:,:2].min(0).tolist(),corners[:,:2].max(0).tolist()],
                    root_velocity=z['qvel_after'][frame,case,:3].tolist()))
            response = np.max(np.abs(z['action'][:,case]-z['action'][first-1,case]),axis=-1)
            changed = np.flatnonzero((np.arange(len(alive))>first)&(response>.01))
            cases.append(dict(case=case, first_contact=float(z['time_s'][first]),
                first_contact_end=float(z['time_s'][first_end]), first_departure=None if not len(failures) else float(z['time_s'][end]),
                response_over_001_seconds=None if not len(changed) else float(z['time_s'][changed[0]]-z['time_s'][first]),
                action_saturation={n:float(v) for n,v in zip(names,action_saturation) if v},
                target_limit_fraction={n:float(v) for n,v in zip(names,clipped.mean(0)) if v},
                torque_saturation={n:float(v) for n,v in zip(names,(torque>=1-1e-5).mean((0,1))) if v},
                maximum_raw_action=float(np.abs(raw[window,case]).max()), geometry=geometry))
        result[label] = dict(control_formula_and_two_frame_queue_verified=True,cases=cases)
    # Rehearsal and PPO telemetry are descriptive, not a causal attribution.
    import json
    telemetry = {}
    distance = BASE/'perturb-distance-learning-20261008-01'
    for label in ('fixed','mixed'):
        rows = [json.loads(x) for x in (distance/label/'updates.jsonl').read_text().splitlines()]
        keys = [k for k in rows[0] if any(s in k for s in ('loss','reward','clip','entropy','rehearsal'))]
        telemetry[label] = {k:dict(first=rows[0][k],last=rows[-1][k],mean=float(np.mean([r[k] for r in rows]))) for k in keys}
        coverage = []
        for p in sorted((distance/label/'rollouts').glob('*.npz')):
            with np.load(p) as z:
                ids = np.nonzero(z['launched'])
                coverage.extend(zip((ids[1]%8).tolist(), z['launch_distance_m'][ids].tolist()))
        telemetry[label]['throws_per_pose_distance'] = {f'{p}:{d:.2f}':coverage.count((p,d)) for p,d in sorted(set(coverage))}
    OUT.mkdir(exist_ok=True)
    save(OUT/'report.json',dict(inputs=inputs,results=result,training=telemetry,
        scope='Association from frozen traces; foot box bounds are descriptive, not a contact support polygon.'))
    for label, rows in result.items():
        c=rows['cases'][7]
        print(label,{k:v for k,v in c.items() if k!='geometry'})
        print('geometry',c['geometry'])
    print('telemetry',telemetry)


if __name__ == '__main__':
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    run()

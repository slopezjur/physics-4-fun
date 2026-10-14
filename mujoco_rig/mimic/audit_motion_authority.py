"""Compare AMP hip/knee examples with stationary commanded target intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import sha256
from .rig import Rig


def envelope_excess(poses, examples, limits, radius):
    lower=np.maximum(poses-radius,limits[:,0])
    upper=np.minimum(poses+radius,limits[:,1])
    return np.maximum(np.maximum(lower[None]-examples[:,None],examples[:,None]-upper[None]),0.)


def run(args):
    rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
    protocol=json.loads((args.checkpoint.parent/'protocol.json').read_text())
    contract=protocol['contract']
    if contract['model_sha256']!=sha256(rig.path):
        raise ValueError('Authority audit requires the same character model')
    if protocol['experiment']['reference_sha256']!=sha256(args.reference):
        raise ValueError('Stationary reference differs from training')
    radius=contract['residual_radians']
    with np.load(args.reference,allow_pickle=False) as arrays:
        poses=arrays['qpos'].copy()
    joints=[rig.model.joint(f'{part}_{side}_{axis}').id for side in ('L','R')
        for part,axis in [('Thigh','rx'),('Thigh','ry'),('Thigh','rz'),('Shin','rx')]]
    indices=rig.model.jnt_qposadr[joints]
    limits=rig.model.jnt_range[joints]
    rows=[]
    for demo in protocol['motion_prior']['demonstrations']:
        path=Path(demo['path'])
        if sha256(path)!=demo['sha256']:
            raise ValueError('Demonstration differs from training: '+str(path))
        with np.load(path,allow_pickle=False) as arrays:
            q=arrays['qpos'][:,indices].copy()
        excess=envelope_excess(poses[:,indices],q,limits,radius)
        rows.append(dict(path=str(path),sha256=sha256(path),frames=len(q),
            outside_fraction_by_posture=(excess.max(-1)>1e-6).mean(0).tolist(),
            max_excess_rad_by_joint=excess.max((0,1)).tolist(),joint_rom_rad=np.ptp(q,axis=0).tolist()))
    report=dict(schema='amp_command_envelope_audit_v1',checkpoint_sha256=sha256(args.checkpoint),
        reference_sha256=sha256(args.reference),character_model_sha256=sha256(rig.path),
        source_sha256=sha256(Path(__file__)),residual_radians=radius,
        joints=[rig.model.joint(j).name for j in joints],rows=rows,
        scope='Requested hip/knee pose compatibility with commanded target bounds; '
              'not a hard bound on dynamic realized poses or proof of trackability')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(path=r['path'],outside_fraction_range=[min(r['outside_fraction_by_posture']),
        max(r['outside_fraction_by_posture'])],max_excess_deg=float(np.rad2deg(max(r['max_excess_rad_by_joint']))))
        for r in rows],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())

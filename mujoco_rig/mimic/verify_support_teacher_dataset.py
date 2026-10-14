"""Audit successful support demonstrations before admitting them to a learner."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import activate
from .diagnose_support_loss import digest
from .support_teacher_dataset import SCHEMA, OBSERVATION_SIZE, ACTION_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_support_skill_study import FOUNDATION, PARENTS
    torch.set_num_threads(1)
    folder = args.run/'entry-feedback'
    report = json.loads((folder/'report.json').read_text())
    contract = json.loads((folder/'teacher-contract.json').read_text())
    if contract['schema'] != SCHEMA or contract['observation_size'] != OBSERVATION_SIZE or contract['action_size'] != ACTION_SIZE:
        raise ValueError('Unsupported dataset schema; never infer compatibility from filenames')
    assert len(report['cases']) == 16 and all(c['entry_hold_return'] for c in report['cases'])
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    model = rig.model
    joints = model.actuator_trnid[rig.actuators, 0]
    qi = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    original = json.loads((FOUNDATION/'protocol.json').read_text())
    checkpoint = PARENTS/'301027/varied/final/model.pt'
    anchor = load_actor(rig,FOUNDATION/'stationary_reference.npz',original,checkpoint).eval().requires_grad_(False)
    with np.load(FOUNDATION/'stationary_reference.npz') as ref:
        references = ref['qpos'].copy()
    with np.load(folder/'teacher-samples.npz') as sample, np.load(folder/'trace.npz') as trace:
        x, action = sample['observation'], sample['action']
        frame, case = sample['frame'], sample['case_id']
        assert x.shape == (len(action), OBSERVATION_SIZE) and action.shape[1] == ACTION_SIZE
        assert all(np.isfinite(sample[k]).all() for k in sample.files)
        assert np.array_equal(np.unique(case), np.arange(16))
        assert len(x) == int(trace['alive'].sum()) == contract['examples']
        assert np.all(trace['alive'][frame,case])
        assert len(np.unique(np.stack((frame,case),axis=1),axis=0)) == len(x)
        assert abs(action).max() <= 1 and np.allclose(x[:,374:420],trace['requested_qpos'][frame,case],rtol=0,atol=1e-7)
        q = trace['qpos'][np.maximum(frame-1,0),case].copy()
        q[frame==0] = references[case[frame==0]%8]
        assert np.allclose(x[:,:3],q[:,:3],rtol=0,atol=1e-7)
        assert np.allclose(x[:,15:54],q[:,7:],rtol=0,atol=1e-7)
        t = np.clip((frame*trace['time'][0]-1),0,1)
        authority = t**3*(10+t*(-15+6*t))
        assert np.allclose(x[:,-1],authority,rtol=0,atol=1e-7)
        assert not action[authority==0].any()
        reconstructed = np.clip(sample['source_targets']+action,lower,upper)
        target_error = float(abs(reconstructed-sample['issued_targets']).max())
        assert target_error < 2e-6
        assert np.array_equal(sample['issued_targets'],trace['issued'][frame,case])
        split = sample['split']
        assert np.array_equal(split,(case%8>=6).astype(np.uint8))
        train_poses, validation_poses = set(case[split==0]%8), set(case[split==1]%8)
        assert train_poses == set(range(6)) and validation_poses == {6,7} and not train_poses & validation_poses
        # Re-evaluate the frozen source from saved issue-time observations.
        selected = np.linspace(0,len(x)-1,min(512,len(x)),dtype=int)
        with torch.no_grad():
            predicted = anchor(torch.tensor(x[selected,:374])).numpy()
        source = np.clip(references[case[selected]%8][:,qi]+.25*np.clip(predicted,-1,1),lower,upper)
        source_error = float(abs(source-sample['source_targets'][selected]).max())
        assert source_error < 2e-6
        results = dict(schema=SCHEMA, verified=True, examples=len(x), complete_cases=16,
                       observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
                       maximum_target_error=target_error, frozen_source_samples=len(selected),
                       maximum_frozen_source_target_error=source_error,
                       observation_timing='Issue-time root/joints match previous physical frame; goal is application-time pose',
                       train_examples=int((split==0).sum()), validation_examples=int((split==1).sum()),
                       held_out_poses=[6,7], all_case_frame_pairs_unique=True,
                       old_380_input_trainer_compatible=False, learning_updates=0)
    paths = [folder/'teacher-samples.npz',folder/'teacher-contract.json',folder/'trace.npz',folder/'report.json',
             args.run/'protocol.json',checkpoint,checkpoint.with_suffix('.amp.pt'),
             Path(__file__),Path(__file__).with_name('support_teacher_dataset.py')]
    results['inputs'] = {str(p):digest(p) for p in paths}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.out/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in results.items() if k!='inputs'},indent=2))


if __name__ == '__main__':
    main()

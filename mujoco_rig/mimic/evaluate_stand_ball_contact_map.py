"""Native unchanged-physics replay with read-only per-body contact attribution."""
import argparse
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from .audit_stand_failures import read, save
from .calibrate_stand_ball_replay import jobs
from .evaluate_stand_ball_boundary import evaluate
from .evaluate_stand_ball_first_contact import PRIOR, SETTINGS
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor


def run(args):
    activate(args.mimickit)
    from . import stand_ball_native as module
    torch.set_num_threads(1)
    package = Path(__file__).parent
    inputs, sources = protect(PRIOR, args.out, [Path(__file__),
        package/'calibrate_stand_ball_replay.py', package/'evaluate_stand_ball_first_contact.py'])
    parent = read(PRIOR/'protocol.json')
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=SETTINGS,
        config=parent['config'], demonstrations=parent['demonstrations'], maximum_cases=80,
        backend='native', intervention='None; read solved contacts per substep',
        admission='Same actor, all outcomes identical, q/v/action/torque trajectories atol 1e-10',
        training_updates=0, promoted=False))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    world = args.out/'ball-5kg.xml'
    shutil.copy2(PRIOR/world.name, world)
    base = module.StandBallNative
    instances = []

    class MappedEngine(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.audit_frame = 0
            self.body_normal = np.zeros((720, 8, 18))
            self.body_vector = np.zeros((720, 8, 18, 3))
            instances.append(self)

        def _record_contacts(self, case, data):
            super()._record_contacts(case, data)
            for i, contact in enumerate(data.contact[:data.ncon]):
                a, b = self.rig.model.geom_bodyid[contact.geom]
                if a == self.ball_body and 0 < b < self.ball_body:
                    body, sign = b, 1.
                elif b == self.ball_body and 0 < a < self.ball_body:
                    body, sign = a, -1.
                else:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.rig.model, data, i, wrench)
                if wrench[0] > 0.:
                    dt = self.rig.model.opt.timestep
                    self.body_normal[self.audit_frame, case, body-1] += wrench[0]*dt
                    self.body_vector[self.audit_frame, case, body-1] += sign*(contact.frame.reshape(3, 3).T@wrench[:3])*dt

        def step(self):
            super().step()
            self.audit_frame += 1

    results = []
    for seed, update, arm, original, checkpoint in jobs():
        folder = args.out/f'{seed}-{update}-{arm}'
        actor = load_actor(rig, reference, parent, checkpoint)
        with patch.object(module, 'StandBallNative', MappedEngine):
            report = evaluate(rig, reference, actor, world, folder, SETTINGS, 1., 1)
        engine = instances.pop()
        baseline = original/'5kg-1mps-+1-native'
        old_report = read(baseline/'report.json')
        assert report['actor_hash'] == old_report['actor_hash']
        names = [engine.rig.model.body(i).name for i in range(1, 19)]
        count = engine.audit_frame
        np.savez_compressed(folder/'body-impulses.npz', normal=engine.body_normal[:count],
            vector=engine.body_vector[:count], body_names=np.asarray(names))
        with np.load(folder/'trace.npz') as a, np.load(baseline/'trace.npz') as b:
            errors = {k: float(np.abs(a[k]-b[k]).max()) for k in
                ('qpos_before','qpos_after','qvel_before','qvel_after','action','substep_torque','normal_impulse_ns')}
            normals = a['normal_impulse_ns']
            sum_error = float(np.abs(engine.body_normal[:count].sum(-1)-normals).max())
        same = all(all(a[k] == b[k] for k in ('hit','survived','prequalified','recovery_pass'))
            for a,b in zip(report['cases'],old_report['cases'],strict=True))
        passed = same and max(errors.values()) <= 1e-10 and sum_error <= 1e-10
        save(folder/'admission.json', dict(passed=passed, outcomes_identical=same,
            errors=errors, impulse_attribution_error_ns=sum_error))
        if not passed:
            raise RuntimeError('Read-only contact attribution changed the native trajectory')
        cases = []
        for case in range(8):
            indices = np.flatnonzero(normals[:, case] > 0.)
            groups = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
            clusters = []
            for group in groups:
                values = engine.body_normal[group, case].sum(0)
                vectors = engine.body_vector[group, case].sum(0)
                clusters.append(dict(first_frame=int(group[0]), last_frame=int(group[-1]),
                    normal_impulse_ns=float(values.sum()),
                    bodies={names[i]:dict(normal_impulse_ns=float(values[i]), vector_impulse_ns=vectors[i].tolist())
                            for i in np.flatnonzero(values > 0.)}))
            cases.append(dict(case=case, recovered=report['cases'][case]['recovery_pass'], clusters=clusters))
        results.append(dict(seed=seed, update=update, arm=arm, prefix_and_outcomes_passed=passed, cases=cases))
        save(args.out/'progress.json', dict(evaluation_cases=8*len(results), results=results))
        print(dict(seed=seed, update=update, arm=arm, passed=passed), flush=True)
        del actor, engine
    assert len(results) == 10
    save(args.out/'report.json', dict(results=results, evaluation_cases=80, training_updates=0))
    seal(args.out, inputs, sources, verified=True, evaluation_cases=80, training_updates=0, promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

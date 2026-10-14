"""Direct-input evidence and read-only native body attribution for successor studies."""
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

import mujoco
import numpy as np

from .audit_stand_failures import digest, read, save


FOUNDATION = Path('logs/mimickit-training/stand-ball-checkpoints-20261006-01')
PILOT = Path('logs/mimickit-training/stand-ball-pilot-20261006-01')


def checkpoint(seed, label):
    if label == 'source':
        return PILOT/str(seed)/'quiet'/'initial'/'model.pt'
    arm, update = label.rsplit('-', 1)
    if arm not in ('quiet', 'single_ball') or update not in ('100', '200', '300'):
        raise ValueError('Unknown admitted checkpoint')
    return PILOT/str(seed)/arm/f'updates-{update}'/'model.pt'


def protect_used(out, source_paths, settings_path, settings):
    parent, verification = read(FOUNDATION/'protocol.json'), read(FOUNDATION/'final-verification.json')
    if not verification['verified']:
        raise ValueError('Verified foundation required')
    manifest = {str(Path(k).resolve()): v for k,v in {**parent['inputs'], **verification['output_hashes']}.items()}
    paths = [FOUNDATION/n for n in ('protocol.json','final-verification.json','dummy.xml',
        'stationary_reference.npz','stationary_reference.json','ball-5kg.xml')]
    paths += [Path(d['path']) for d in parent['demonstrations']]
    for seed in settings['seeds']:
        for actor in settings['actors']:
            p = checkpoint(seed, actor)
            paths += [p, p.with_suffix('.amp.pt')]
    paths += [Path(p) for p in settings.get('evidence', [])]
    inherited = [Path(s['path']) for s in parent['sources']]
    source_paths = list({str(p.resolve()): p for p in [Path(__file__), *inherited, *source_paths]}.values())
    paths += source_paths + [settings_path]
    inputs = {}
    for path in paths:
        value = digest(path)
        expected = manifest.get(str(path.resolve()))
        if expected is not None and value != expected:
            raise ValueError(f'Sealed used input changed: {path}')
        inputs[str(path)] = value
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for i,path in enumerate(source_paths):
        snapshot = out/'sources'/f'{i:03d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    for name in ('dummy.xml','stationary_reference.npz','stationary_reference.json'):
        shutil.copy2(FOUNDATION/name, out/name)
    return parent, inputs, sources


@contextmanager
def body_observer(journal, frames):
    from . import stand_ball_native as module
    base = module.StandBallNative

    class ObservedEngine(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.audit_frame = 0
            self.body_normal = np.zeros((frames, 8, 18))
            self.body_vector = np.zeros((frames, 8, 18, 3))
            journal['engine'] = self

        def _record_contacts(self, case, data):
            super()._record_contacts(case, data)
            for i,c in enumerate(data.contact[:data.ncon]):
                a,b = self.rig.model.geom_bodyid[c.geom]
                if a == self.ball_body and 0 < b < self.ball_body:
                    body,sign = b,1.
                elif b == self.ball_body and 0 < a < self.ball_body:
                    body,sign = a,-1.
                else:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.rig.model, data, i, wrench)
                if wrench[0] > 0:
                    dt = self.rig.model.opt.timestep
                    self.body_normal[self.audit_frame,case,body-1] += wrench[0]*dt
                    self.body_vector[self.audit_frame,case,body-1] += sign*(c.frame.reshape(3,3).T@wrench[:3])*dt

        def step(self):
            super().step()
            self.audit_frame += 1

    with patch.object(module, 'StandBallNative', ObservedEngine):
        yield


def save_bodies(folder, journal):
    engine = journal.pop('engine')
    names = [engine.rig.model.body(i).name for i in range(1,19)]
    values = engine.body_normal[:engine.audit_frame]
    np.savez_compressed(folder/'body-impulses.npz', normal=values,
        vector=engine.body_vector[:engine.audit_frame], body_names=np.asarray(names))
    with np.load(folder/'trace.npz') as trace:
        normal = trace['normal_impulse_ns']
    error = float(np.abs(values.sum(-1)-normal).max())
    if error > 1e-10:
        raise RuntimeError('Body attribution does not sum to recorded impulse')
    cases = []
    for case in range(8):
        hits = np.flatnonzero(normal[:,case] > 0)
        groups = np.split(hits, np.flatnonzero(np.diff(hits)>1)+1) if len(hits) else []
        clusters = []
        for group in groups:
            impulse = values[group,case].sum(0)
            clusters.append(dict(first_frame=int(group[0]), last_frame=int(group[-1]),
                bodies={names[i]:float(impulse[i]) for i in np.flatnonzero(impulse>0)}))
        cases.append(dict(case=case, clusters=clusters))
    result = dict(cases=cases, impulse_sum_error_ns=error)
    save(folder/'body-report.json', result)
    return result

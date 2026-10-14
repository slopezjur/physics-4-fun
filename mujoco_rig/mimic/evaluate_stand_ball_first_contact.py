"""Frozen contact-removal intervention using the admitted trace evaluators."""
import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .evaluate_stand_ball_boundary import evaluate as native_evaluate
from .evaluate_stand_ball_gpu import evaluate as gpu_evaluate
from .replicate_stand_representations import cpu, read_state
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor


SETTINGS = dict(seconds=12., launch_seconds=5., launch_distance_m=.65)
PRIOR = Path('logs/mimickit-training/stand-ball-checkpoints-20261006-01')
PILOT = Path('logs/mimickit-training/stand-ball-pilot-20261006-01')


def snapshot(task, backend):
    q, v = read_state(task, backend)
    c = task.engine.policy_control
    return dict(q=q[:, :46].copy(), v=v[:, :45].copy(), pending=cpu(c.pending).copy(),
        active=cpu(c.active).copy(), ctrl=cpu(c.ctrl).copy(),
        steps=cpu(task.steps).copy(), offset=cpu(task.offset).copy(),
        observation=cpu(task.observations()).copy(), cursor=np.asarray(c.cursor))


@contextmanager
def removal_task(backend, journal):
    if backend == 'native':
        from . import stand_ball_native as module
        name = 'StandBallTask'
    else:
        from . import stand_ball_task as module
        name = 'StandBallGPUTask'
    base = getattr(module, name)

    class FirstContactTask(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen_contact = np.zeros(8, dtype=bool)
            self.removed = np.zeros(8, dtype=bool)
            self.removal_frame = np.full(8, -1, dtype=int)
            self.frame = 0
            journal['task'] = self

        def step(self, action):
            result = super().step(action)
            value = self.engine.normal_impulse
            impulse = cpu(value) if torch.is_tensor(value) else value.copy()
            if (impulse[self.removed] != 0.).any():
                raise RuntimeError('Projectile contacted character after removal')
            hit = impulse > 0.
            remove = self.seen_contact & ~hit & ~self.removed
            self.seen_contact |= hit
            if remove.any():
                before = snapshot(self, backend)
                ids = torch.as_tensor(np.flatnonzero(remove), device=self.device)
                self.engine.launch_ball(ids, [40., 40., .09], [0., 0., 0.])
                after = snapshot(self, backend)
                differences = {k: float(np.abs(before[k]-after[k]).max()) for k in before}
                # Exact physical/controller preservation; only admitted foot-load low bits may differ.
                for k in before:
                    if k != 'observation' and not np.array_equal(before[k], after[k]):
                        raise RuntimeError(f'Parking changed {k}')
                obs_error = np.abs(before['observation']-after['observation'])
                other = np.ones(obs_error.shape[-1], dtype=bool)
                other[93:95] = False
                if obs_error[:, other].any() or (obs_error[:, 93:95] > 1e-6).any():
                    raise RuntimeError('Parking changed character observations')
                journal['removals'].append(dict(frame=self.frame, cases=np.flatnonzero(remove).tolist(),
                    state_differences=differences))
                self.removed |= remove
                self.removal_frame[remove] = self.frame
            self.frame += 1
            return result

    with patch.object(module, name, FirstContactTask):
        yield


def compare_prefix(folder, baseline, frames, backend):
    # Tolerances declared before simulation, restricted to same-backend replay.
    tolerances = dict(qpos_before=1e-6, qpos_after=1e-6, qvel_before=1e-5,
        qvel_after=1e-5, action=1e-6, pending_before=1e-6, active_targets=1e-6,
        substep_q=1e-6, substep_v=1e-5, substep_torque=1e-3,
        normal_impulse_ns=1e-3, contact_impulse_ns=1e-3)
    if backend == 'native':
        tolerances = {k: 1e-10 for k in tolerances}
    checks = []
    with np.load(folder/'trace.npz', allow_pickle=False) as current, \
            np.load(baseline/'trace.npz', allow_pickle=False) as old:
        a, b = ({k: source[k] for k in tolerances} for source in (current, old))
        for case, frame in enumerate(frames):
            if frame < 0:
                raise RuntimeError(f'Case {case}: no first-contact removal')
            end = int(frame)+1
            errors = {}
            for key in tolerances:
                index = (slice(0, end), slice(None), case) if key.startswith('substep_') else (slice(0, end), case)
                errors[key] = float(np.abs(a[key][index]-b[key][index]).max())
            passed = all(errors[k] <= tolerances[k] for k in errors)
            checks.append(dict(case=case, prefix_frames=end, passed=passed, errors=errors))
            if not passed:
                save(folder/'prefix-admission.json', dict(passed=False, cases=checks, tolerances=tolerances))
                raise RuntimeError(f'Unmatched trajectory before removal: {backend}, case {case}')
            if (current['normal_impulse_ns'][end:, case] != 0.).any():
                raise RuntimeError('Saved trace contains later contact after parking')
    result = dict(passed=True, cases=checks, tolerances=tolerances)
    save(folder/'prefix-admission.json', result)
    return result


def run(args):
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    parent, prior_seal = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    if not prior_seal['verified']:
        raise ValueError('Verified predecessor required')
    inputs, sources = protect(PRIOR, args.out, [Path(__file__)])
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=SETTINGS,
        config=parent['config'], demonstrations=parent['demonstrations'],
        seeds=[301027, 301028], checkpoints=[0, 100, 300], maximum_cases=160,
        new_training_updates=0, intervention='Park after first zero-contact interval following contact',
        prefix_admission='Native absolute 1e-10; GPU q/action/targets 1e-6, v 1e-5, torque/impulse 1e-3',
        parking_admission='Exact character/controller state; observation foot-load columns atol 1e-6 only',
        promoted=False))
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    reference = args.out/'stationary_reference.npz'
    world = args.out/'ball-5kg.xml'
    shutil.copy2(PRIOR/'ball-5kg.xml', world)
    results = []
    for seed in (301027, 301028):
        for update in (0, 100, 300):
            for arm in (('source',) if update == 0 else ('quiet', 'single_ball')):
                base = (PILOT/str(seed)/'source' if update == 0 else
                    PILOT/str(seed)/arm/'evaluation' if update == 300 else
                    PRIOR/str(update)/str(seed)/arm/'selected')
                checkpoint = (PILOT/str(seed)/'quiet'/'initial'/'model.pt' if update == 0 else
                    PILOT/str(seed)/arm/f'updates-{update}'/'model.pt')
                actor = load_actor(rig, reference, parent, checkpoint)
                for backend, evaluate in (('native', native_evaluate), ('gpu', gpu_evaluate)):
                    baseline = base/f'5kg-1mps-+1-{backend}'
                    folder = args.out/f'{seed}-{update}-{arm}-{backend}'
                    journal = dict(removals=[])
                    with removal_task(backend, journal):
                        assessment = evaluate(rig, reference, actor, world, folder, SETTINGS, 1., 1)
                    frames = journal.pop('task').removal_frame
                    save(folder/'removals.json', dict(**journal, frames=frames.tolist()))
                    admission = compare_prefix(folder, baseline, frames, backend)
                    original = read(baseline/'report.json')
                    assert original['actor_hash'] == assessment['actor_hash']
                    cases = [dict(case=i, baseline_recovery=a['recovery_pass'],
                        removed_recovery=b['recovery_pass'], baseline_survival=a['survival_seconds'],
                        removed_survival=b['survival_seconds'], baseline_impulse=a['total_normal_impulse_ns'],
                        removed_impulse=b['total_normal_impulse_ns']) for i, (a, b) in enumerate(zip(original['cases'], assessment['cases'], strict=True))]
                    results.append(dict(seed=seed, update=update, arm=arm, backend=backend,
                        baseline_recoveries=original['recovery_passes'], recoveries=assessment['recovery_passes'],
                        prefix_passed=admission['passed'], cases=cases))
                    save(args.out/'progress.json', dict(evaluation_cases=8*len(results), results=results))
                    print(dict(seed=seed, update=update, arm=arm, backend=backend,
                        baseline=original['recovery_passes'], removed=assessment['recovery_passes']), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                del actor
    assert len(results) == 20
    save(args.out/'report.json', dict(results=results, evaluation_cases=160, new_training_updates=0, promoted=False))
    seal(args.out, inputs, sources, verified=True, evaluation_cases=160, training_updates=0, promoted=False)


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
            save(args.out/'failure.json', dict(error=traceback.format_exc(), promoted=False))
        raise

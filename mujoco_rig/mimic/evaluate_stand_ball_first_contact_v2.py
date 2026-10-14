"""Contact removal with independently calibrated, prospectively locked GPU limits."""
import argparse
import gc
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import read, save
from .calibrate_stand_ball_replay import jobs
from .evaluate_stand_ball_first_contact import removal_task, SETTINGS
from .evaluate_stand_ball_boundary import evaluate as native_evaluate
from .evaluate_stand_ball_gpu import evaluate as gpu_evaluate
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor


CALIBRATION = Path('logs/mimickit-training/stand-ball-replay-calibration-20261006-01')


def prefix(folder, baseline, frames, backend, envelope):
    tolerances = dict.fromkeys(envelope, 1e-10) if backend == 'native' else envelope
    checks = []
    with np.load(folder/'trace.npz') as a, np.load(baseline/'trace.npz') as b:
        current, old = ({k: z[k] for k in tolerances} for z in (a, b))
        old_impulse = old['normal_impulse_ns']
        for case, frame in enumerate(frames):
            if frame < 0:
                raise RuntimeError('No contact removal')
            errors = {}
            for key in tolerances:
                index = (slice(0, int(frame)+1), slice(None), case) if key.startswith('substep_') else (slice(0, int(frame)+1), case)
                errors[key] = float(np.abs(current[key][index]-old[key][index]).max())
            first = int(np.flatnonzero(old_impulse[:, case]>0.)[0])
            baseline_cut = first+int(np.flatnonzero(old_impulse[first:, case]==0.)[0])
            matched_cut = int(frame) == baseline_cut
            later_zero = not current['normal_impulse_ns'][int(frame)+1:, case].any()
            passed = matched_cut and later_zero and all(errors[k] <= tolerances[k] for k in errors)
            checks.append(dict(case=case, removal_frame=int(frame), baseline_cut=baseline_cut,
                passed=passed, errors=errors, no_later_impulse=later_zero))
    result = dict(passed=all(c['passed'] for c in checks), cases=checks, tolerances=tolerances)
    save(folder/'prefix-admission.json', result)
    if not result['passed']:
        raise RuntimeError('Intervention exceeds locked prefix admission; no widening permitted')
    return result


def run(args):
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    verification = read(CALIBRATION/'final-verification.json')
    if not verification['verified']:
        raise ValueError('Completed replay calibration required')
    parent, calibration = read(CALIBRATION/'protocol.json'), read(CALIBRATION/'report.json')
    envelope = calibration['intervention_envelope']
    inputs, sources = protect(CALIBRATION, args.out, [Path(__file__)])
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=SETTINGS,
        config=parent['config'], demonstrations=parent['demonstrations'], maximum_cases=160,
        gpu_prefix_envelope=envelope, native_prefix_atol=1e-10,
        caveat='Native same-backend prefix exact within 1e-10; GPU constrained by independent unchanged-replay envelope, not bitwise identity',
        training_updates=0, promoted=False))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    world = args.out/'ball-5kg.xml'
    shutil.copy2(CALIBRATION/world.name, world)
    results = []
    for seed, update, arm, base, checkpoint in jobs():
        actor = load_actor(rig, reference, parent, checkpoint)
        for backend, evaluate in (('native', native_evaluate), ('gpu', gpu_evaluate)):
            baseline = base/f'5kg-1mps-+1-{backend}'
            folder = args.out/f'{seed}-{update}-{arm}-{backend}'
            journal = dict(removals=[])
            with removal_task(backend, journal):
                report = evaluate(rig, reference, actor, world, folder, SETTINGS, 1., 1)
            frames = journal.pop('task').removal_frame
            save(folder/'removals.json', dict(**journal, frames=frames.tolist()))
            admission = prefix(folder, baseline, frames, backend, envelope)
            original = read(baseline/'report.json')
            assert original['actor_hash'] == report['actor_hash']
            with np.load(baseline/'trace.npz') as trace:
                old_impulse = trace['normal_impulse_ns']
            cases = []
            for i, (a, b) in enumerate(zip(original['cases'], report['cases'], strict=True)):
                no_later = not old_impulse[int(frames[i])+1:, i].any()
                unchanged_outcome = all(a[k] == b[k] for k in ('survived','recovery_pass'))
                if no_later and not unchanged_outcome:
                    raise RuntimeError('No-recontact control changed outcome')
                cases.append(dict(case=i, baseline_recovery=a['recovery_pass'], removed_recovery=b['recovery_pass'],
                    baseline_survival=a['survival_seconds'], removed_survival=b['survival_seconds'],
                    baseline_impulse=a['total_normal_impulse_ns'], removed_impulse=b['total_normal_impulse_ns'],
                    no_original_later_contact=no_later, unchanged_outcome=unchanged_outcome,
                    baseline_recovery_start=a['recovery_window_start'], removed_recovery_start=b['recovery_window_start']))
            results.append(dict(seed=seed, update=update, arm=arm, backend=backend,
                baseline_recoveries=original['recovery_passes'], recoveries=report['recovery_passes'],
                prefix_passed=admission['passed'], cases=cases))
            save(args.out/'progress.json', dict(evaluation_cases=8*len(results), results=results))
            print(dict(seed=seed, update=update, arm=arm, backend=backend,
                baseline=original['recovery_passes'], removed=report['recovery_passes']), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        del actor
    assert len(results) == 20
    save(args.out/'report.json', dict(results=results, evaluation_cases=160, training_updates=0, promoted=False))
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
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

"""Independent unchanged GPU replays to freeze a numerical admission envelope."""
import argparse
import gc
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import read, save
from .evaluate_stand_ball_gpu import evaluate
from .evaluate_stand_ball_first_contact import PRIOR, PILOT, SETTINGS
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor


# Prospective physical caps; observed differences may yield a stricter envelope.
CAPS = dict(qpos_before=.005, qpos_after=.005, qvel_before=.2, qvel_after=.2,
    action=.02, pending_before=.005, active_targets=.005, substep_q=.005,
    substep_v=.2, substep_torque=2., normal_impulse_ns=.1, contact_impulse_ns=.1)


def jobs():
    for seed in (301027, 301028):
        for update in (0, 100, 300):
            for arm in (('source',) if update == 0 else ('quiet', 'single_ball')):
                base = (PILOT/str(seed)/'source' if update == 0 else
                    PILOT/str(seed)/arm/'evaluation' if update == 300 else
                    PRIOR/str(update)/str(seed)/arm/'selected')
                checkpoint = (PILOT/str(seed)/'quiet'/'initial'/'model.pt' if update == 0 else
                    PILOT/str(seed)/arm/f'updates-{update}'/'model.pt')
                yield seed, update, arm, base, checkpoint


def run(args):
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    package = Path(__file__).parent
    failed = Path('logs/mimickit-training/stand-ball-first-contact-20261006-01')
    inputs, sources = protect(PRIOR, args.out, [Path(__file__), package/'evaluate_stand_ball_first_contact.py'],
        extra_inputs=[p for p in failed.rglob('*') if p.is_file()])
    parent = read(PRIOR/'protocol.json')
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=SETTINGS,
        config=parent['config'], demonstrations=parent['demonstrations'], maximum_cases=80,
        field_caps=CAPS, rule='All unchanged replay outcomes identical; differences through first contact-free frame within caps; intervention envelope = min(cap, max(1e-6, 2*observed_max))',
        no_intervention=True, counts_as_autonomous_stage=False))
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    reference = args.out/'stationary_reference.npz'
    world = args.out/'ball-5kg.xml'
    shutil.copy2(PRIOR/world.name, world)
    maxima = dict.fromkeys(CAPS, 0.)
    results = []
    for seed, update, arm, base, checkpoint in jobs():
        actor = load_actor(rig, reference, parent, checkpoint)
        folder = args.out/f'{seed}-{update}-{arm}-gpu'
        report = evaluate(rig, reference, actor, world, folder, SETTINGS, 1., 1)
        baseline = base/'5kg-1mps-+1-gpu'
        original = read(baseline/'report.json')
        assert report['actor_hash'] == original['actor_hash']
        same = all(all(a[k] == b[k] for k in ('hit', 'survived', 'prequalified', 'recovery_pass'))
            for a, b in zip(report['cases'], original['cases'], strict=True))
        with np.load(folder/'trace.npz') as a, np.load(baseline/'trace.npz') as b:
            current, old = ({k: z[k] for k in CAPS} for z in (a, b))
        errors = dict.fromkeys(CAPS, 0.)
        for case in range(8):
            impulses = old['normal_impulse_ns'][:, case]
            first = int(np.flatnonzero(impulses > 0.)[0])
            cut = first + int(np.flatnonzero(impulses[first:] == 0.)[0])
            for key in CAPS:
                index = (slice(0, cut+1), slice(None), case) if key.startswith('substep_') else (slice(0, cut+1), case)
                errors[key] = max(errors[key], float(np.abs(current[key][index]-old[key][index]).max()))
        passed = same and all(errors[k] <= CAPS[k] for k in CAPS)
        result = dict(seed=seed, update=update, arm=arm, outcomes_identical=same,
            prefix_errors=errors, passed=passed)
        results.append(result)
        save(args.out/'progress.json', dict(results=results, evaluation_cases=8*len(results)))
        if not passed:
            raise RuntimeError('Unchanged GPU replay exceeds prospective admission caps')
        maxima = {k: max(maxima[k], errors[k]) for k in CAPS}
        print(dict(seed=seed, update=update, arm=arm, passed=passed), flush=True)
        del actor
        gc.collect()
        torch.cuda.empty_cache()
    envelope = {k: min(CAPS[k], max(1e-6, 2*maxima[k])) for k in CAPS}
    save(args.out/'report.json', dict(results=results, observed_maxima=maxima,
        intervention_envelope=envelope, evaluation_cases=80, training_updates=0))
    seal(args.out, inputs, sources, verified=True, evaluation_cases=80, training_updates=0,
        counts_as_autonomous_stage=False)


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

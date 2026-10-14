"""Frozen final actors: changing commands versus held initial target offsets."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import traceback
import numpy as np
import torch

from .audit_stand_failures import read, save, digest
from .replicate_stand_representations import evaluate, comparison
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, load_actor, verify_rollout, seal
from .train_stand_imitation import tensor_hash

PRIOR = Path('logs/mimickit-training/stand-quiet-speed-20261003-01')


class HoldFirst(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.first = None

    def state_dict(self, *args, **kwargs):
        return self.actor.state_dict(*args, **kwargs)

    def forward(self, observation):
        if self.first is None:
            self.first = self.actor(observation).detach().clone()
        return self.first.clone()


def run(args):
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    inputs, sources = protect(PRIOR, args.out, [Path(__file__)])
    prior, previous = read(PRIOR/'protocol.json'), read(PRIOR/'report.json')
    rig, reference = Rig.load(PRIOR/'dummy.xml', policy_exclude=()), PRIOR/'stationary_reference.npz'
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, source_run=str(PRIOR),
        config=prior['config'], demonstrations=prior['demonstrations'], model_sha256=digest(rig.path),
        reference_sha256=digest(reference), conditions=['live', 'held_initial_action'],
        cases=128, seconds=12., training_updates=0, repeatability_tolerance_seconds=rig.model.opt.timestep*rig.decimation,
        selection='All four final Phase 49 actors', no_teacher=True, no_passive_springs=True,
        local_pd_feedback='Unchanged in both conditions'))
    results, decisions, intervals = {}, {}, 0
    for seed, arms in previous['arms'].items():
        for mode, arm in arms.items():
            key = f'{seed}-{mode}'
            folder = args.out/key
            folder.mkdir()
            actor = load_actor(rig, reference, prior, Path(arm['final_checkpoint']))
            actor_hash = tensor_hash(actor.state_dict())
            results[key], decisions[key] = {}, {}
            for condition in ('live', 'held_initial_action'):
                results[key][condition] = {}
                for backend in ('native', 'gpu'):
                    policy = copy.deepcopy(actor) if condition == 'live' else HoldFirst(copy.deepcopy(actor))
                    dest = folder/f'{condition}-{backend}'
                    result = evaluate(rig, reference, 'target_pd', backend, policy, dest)
                    trace, verified = verify_rollout(dest)
                    assert result == verified and result['actor_hash'] == actor_hash
                    for case in range(8):
                        live = trace['alive_before'][:, case]
                        if condition == 'held_initial_action':
                            assert np.array_equal(trace['action'][live, case],
                                np.broadcast_to(trace['action'][0, case], trace['action'][live, case].shape))
                            with np.load(folder/f'live-{backend}'/'trace.npz') as control:
                                assert np.array_equal(trace['action'][0], control['action'][0])
                    if condition == 'live':
                        differences = [abs(a['survival_seconds']-b['survival_seconds'])
                            for a, b in zip(result['cases'], arm['final']['results'][backend]['cases'])]
                        assert max(differences) <= rig.model.opt.timestep*rig.decimation+1e-12
                        save(dest/'repeatability.json', dict(max_survival_difference_seconds=max(differences),
                            sealed_actor_hash=arm['final']['results'][backend]['actor_hash'], actor_hash=actor_hash))
                        assert actor_hash == arm['final']['results'][backend]['actor_hash']
                    intervals += result['live_control_intervals']
                    results[key][condition][backend] = result
            for backend in ('native', 'gpu'):
                decisions[key][backend] = comparison(results[key]['live'][backend],
                    results[key]['held_initial_action'][backend], rig.model.opt.timestep*rig.decimation)
            save(folder/'comparison.json', decisions[key])
            print(dict(actor=key, live_mean=results[key]['live']['native']['mean_survival_seconds'],
                held_mean=results[key]['held_initial_action']['native']['mean_survival_seconds']), flush=True)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), results=results, decisions=decisions,
        training_updates=0, cases=128, perturbations_ready=False, arena_changed=False)
    save(args.out/'report.json', report)
    seal(args.out, inputs, sources, cases=128, live_intervals=intervals, frozen_actor_hashes_verified=True,
        held_commands_verified=True, live_repeatability_verified=True, saved_gates_and_controls_verified=True,
        training_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

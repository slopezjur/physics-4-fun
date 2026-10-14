"""Causal command-hold ablation of the first 12-second standing actor."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from .audit_stand_failures import read, save
from .evaluate_stand_sustained import evaluate_duration, verify_duration
from .runtime import activate
from .rig import Rig
from .stand_experiment_evidence import protect, seal, load_actor

PRIOR = Path('logs/mimickit-training/stand-sustained-301024-20261005-01')


class HoldAfter(torch.nn.Module):
    def __init__(self, actor, hold_frame):
        super().__init__()
        self.actor, self.hold_frame = actor, hold_frame
        self.frame, self.held = 0, None

    def state_dict(self, *args, **kwargs):
        return self.actor.state_dict(*args, **kwargs)

    def forward(self, obs):
        if self.frame <= self.hold_frame:
            self.held = self.actor(obs).detach().clone()
        self.frame += 1
        return self.held.clone()


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    inputs, sources = protect(PRIOR, args.out, [Path(__file__)])
    prior = read(PRIOR/'protocol.json')
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    dt = rig.model.opt.timestep*rig.decimation
    conditions = {'hold_initial': 0, 'hold_after_10s': int(np.ceil(10/dt))}
    protocol = dict(inputs=inputs, sources=sources, config=prior['config'],
        demonstrations=prior['demonstrations'], model_sha256=prior['model_sha256'],
        reference_sha256=prior['reference_sha256'], checkpoint=prior['checkpoint'],
        seconds=60., conditions=conditions, maximum_cases=32, training_updates=0,
        initial_tolerance=1e-6, prehold_action_tolerance=2e-5, held_constancy_tolerance=0.,
        scope='Counterfactual command hold, local PD unchanged; diagnostic only')
    save(args.out/'protocol.json', protocol)
    actor = load_actor(rig, reference, protocol, Path(prior['checkpoint']))
    results, comparisons = {}, {}
    for name, frame in conditions.items():
        results[name], comparisons[name] = {}, {}
        for backend in ('native', 'gpu'):
            wrapper = HoldAfter(copy.deepcopy(actor), frame)
            folder = args.out/f'{name}-{backend}'
            result = evaluate_duration(rig, reference, 'target_pd', backend, wrapper, folder, seconds=60.)
            trace, verified = verify_duration(folder, 60.)
            assert result == verified
            with np.load(PRIOR/backend/'trace.npz', allow_pickle=False) as original:
                for key in ('qpos_before', 'qvel_before', 'action'):
                    assert np.allclose(trace[key][0], original[key][0], atol=1e-6, rtol=0)
                n = min(frame+1, len(trace['action']), len(original['action']))
                live = trace['alive_before'][:n] & original['alive_before'][:n]
                prefix_error = float(np.abs(trace['action'][:n]-original['action'][:n])[live].max())
                assert prefix_error <= 2e-5
            for case in range(8):
                mask = trace['alive_before'][:, case] & (np.arange(len(trace['action'])) >= frame)
                if mask.any():
                    assert np.array_equal(trace['action'][mask, case],
                        np.broadcast_to(trace['action'][frame, case], trace['action'][mask, case].shape))
            baseline = read(PRIOR/backend/'report.json')
            assert result['actor_hash'] == baseline['actor_hash']
            comparisons[name][backend] = dict(max_prehold_action_difference=prefix_error,
                survival_changes_seconds=[c['survival_seconds']-b['survival_seconds'] for c, b in zip(result['cases'], baseline['cases'])],
                stand_pass_change=result['standing_passes']-baseline['standing_passes'])
            results[name][backend] = result
    save(args.out/'report.json', dict(results=results, comparisons=comparisons, cases=32,
        training_updates=0, deployable_candidate=False, perturbations_ready=False))
    seal(args.out, inputs, sources, verified=True, cases=32, actor_unchanged=True,
        command_holds_verified=True, physical_gates_and_controls_verified=True, training_updates=0)
    print({name: {b: (r['standing_passes'], r['mean_survival_seconds']) for b, r in pair.items()}
        for name, pair in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())

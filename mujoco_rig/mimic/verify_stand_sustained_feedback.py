"""Verify completed hold trajectories by replaying their own observations."""
from pathlib import Path
import numpy as np
import torch
from .audit_stand_failures import read, save
from .evaluate_stand_sustained import verify_duration
from .runtime import activate
from .rig import Rig
from .stand_experiment_evidence import protect, seal, load_actor

PRIOR = Path('logs/mimickit-training/stand-sustained-301024-20261005-01')
ATTEMPT = Path('logs/mimickit-training/stand-sustained-feedback-20261005-01')
OUT = Path('logs/mimickit-training/stand-sustained-feedback-verified-20261005-01')


@torch.no_grad()
def run():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    inputs, sources = protect(PRIOR, OUT, [Path(__file__), Path(__file__).with_name('probe_stand_sustained_feedback.py')],
        extra_inputs=tuple(path for path in ATTEMPT.rglob('*') if path.is_file()))
    prior = read(PRIOR/'protocol.json')
    protocol = dict(inputs=inputs, sources=sources, config=prior['config'], demonstrations=prior['demonstrations'],
        model_sha256=prior['model_sha256'], reference_sha256=prior['reference_sha256'], checkpoint=prior['checkpoint'],
        action_replay_tolerance=2e-6, original_prehold_cross_rollout_tolerance_failed=True,
        physics_rollouts=0, training_updates=0, source_attempt=str(ATTEMPT))
    save(OUT/'protocol.json', protocol)
    rig, reference = Rig.load(OUT/'dummy.xml', policy_exclude=()), OUT/'stationary_reference.npz'
    actor = load_actor(rig, reference, protocol, Path(prior['checkpoint']))
    results, comparisons = {}, {}
    for name, frame in read(ATTEMPT/'protocol.json')['conditions'].items():
        results[name], comparisons[name] = {}, {}
        for backend in ('native', 'gpu'):
            trace, result = verify_duration(ATTEMPT/f'{name}-{backend}', 60.)
            n = min(frame+1, len(trace['action']))
            observations = torch.from_numpy(trace['observation'][:n].reshape(-1, 374))
            predicted = actor(observations).numpy().reshape(n, 8, 33)
            live = trace['alive_before'][:n]
            error = float(np.abs(predicted-trace['action'][:n])[live].max())
            assert error <= 2e-6, (name, backend, error)
            for case in range(8):
                mask = trace['alive_before'][:, case] & (np.arange(len(trace['action'])) >= frame)
                if mask.any():
                    assert np.array_equal(trace['action'][mask, case], np.broadcast_to(trace['action'][frame, case], trace['action'][mask, case].shape))
            baseline = read(PRIOR/backend/'report.json')
            assert result['actor_hash'] == baseline['actor_hash']
            with np.load(PRIOR/backend/'trace.npz') as original:
                for key in ('qpos_before', 'qvel_before', 'action'):
                    assert np.allclose(trace[key][0], original[key][0], atol=1e-6, rtol=0)
                prefix_difference = float(np.abs(trace['action'][:n]-original['action'][:n])[live].max())
            comparisons[name][backend] = dict(replay_action_error=error,
                independent_rollout_prefix_action_difference=prefix_difference,
                survival_changes_seconds=[c['survival_seconds']-b['survival_seconds'] for c, b in zip(result['cases'], baseline['cases'])],
                stand_pass_change=result['standing_passes']-baseline['standing_passes'])
            results[name][backend] = result
    save(OUT/'report.json', dict(results=results, comparisons=comparisons, accepted_existing_cases=32,
        new_physics_cases=0, training_updates=0, deployable_candidate=False, perturbations_ready=False))
    seal(OUT, inputs, sources, verified=True, accepted_existing_cases=32, replay_and_holds_verified=True,
        new_physics_cases=0, training_updates=0)
    print({name: {b: (r['standing_passes'], r['mean_survival_seconds']) for b, r in pair.items()}
        for name, pair in results.items()})


if __name__ == '__main__':
    run()

"""Close the twenty-step batch using sealed evidence, without new simulation."""
from pathlib import Path
import shutil
import numpy as np

from .audit_stand_failures import digest, read, save
from .measure_stand_learning_curve import continuation

BASE = Path('logs/mimickit-training')
ROOTS = [BASE/name for name in ('stand-policy-feedback-20261005-02',
    'stand-feedback-inputs-20261005-01', 'stand-actor-rate-20261005-01',
    'stand-actor-rate-duration-20261005-01')]
OUT = BASE/'stand-autonomous-batch-20261005-01'


def run():
    hashes, metadata = {}, {}
    for root in ROOTS:
        protocol, verification = read(root/'protocol.json'), read(root/'final-verification.json')
        entries = {**protocol['inputs'], **verification['output_hashes'],
            str(root/'final-verification.json'): digest(root/'final-verification.json')}
        for source in protocol['sources']:
            entries.update({source[k]: source['sha256'] for k in ('path', 'snapshot')})
        for name, expected in entries.items():
            path = str(Path(name).resolve())
            assert path not in hashes or hashes[path] == expected, path
            hashes[path] = expected
        metadata[root.name] = {k: v for k, v in verification.items() if k != 'output_hashes'}
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__))
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), OUT/'verifier.py')
    duration = read(ROOTS[-1]/'report.json')
    stages = duration['assessments']
    dt = read(ROOTS[-1]/'contract.json')['control_timestep']
    decision = continuation(stages['updates-300'], stages['updates-100'], dt) if 'updates-300' in stages else None
    save(OUT/'duration-decision.json', dict(comparison='300 versus 100 PPO updates within fresh seed 301022',
        decision=decision, stopped_early=duration['ppo_updates'] == 100,
        full_learned_stand=duration['full_learned_stand'], sustained_stand_verified=False))
    sensitivity = read(ROOTS[1]/'report.json')
    groups = read(ROOTS[1]/'protocol.json')['groups']
    for key, result in sensitivity['results'].items():
        with np.load(ROOTS[1]/key/'sensitivity.npz', allow_pickle=False) as data:
            jac = data['normalized_jacobian']
            assert jac.shape == (16, 33, 374) and np.isfinite(jac).all()
            energy = np.maximum(np.square(jac).sum((1, 2)), 1e-20)
            for name, (lo, hi) in groups.items():
                fractions = np.square(jac[..., lo:hi]).sum((1, 2))/energy
                for label, values in (('initial', fractions[:8]), ('predeparture', fractions[8:])):
                    assert np.isclose(float(values.mean()), result['groups'][name][label+'_jacobian_energy_fraction'], atol=1e-6, rtol=0)
            assert np.isclose(float(abs(data['horizontal_velocity_half_difference']).max()*.25),
                result['max_half_difference_target_radians'], atol=1e-9, rtol=0)
    rate = read(ROOTS[2]/'report.json')
    assert rate['replicated_gain']
    training_updates = sum(a['ppo_updates'] for pair in rate['arms'].values() for a in pair.values()) + duration['ppo_updates']
    transitions = sum(a['transitions'] for pair in rate['arms'].values() for a in pair.values()) + duration['transitions']
    seconds = sum(a['training_seconds'] for pair in rate['arms'].values() for a in pair.values()) + duration['training_seconds']
    assert transitions == training_updates*4096
    summary = dict(completed_steps=list(range(1, 21)), evidence_roots=[str(p) for p in ROOTS],
        protected_files=len(hashes), phase_verification=metadata, selected_sensitivity_states=64,
        accepted_assessment_cases=128+128+16*len(stages), interrupted_attempt_cases=32,
        total_ppo_updates=training_updates, total_transitions=transitions, synchronized_training_seconds=seconds,
        actor_rate_gain_replicated=True, duration_decision=decision,
        final_stand_passes={b: list(stages.values())[-1]['results'][b]['standing_passes'] for b in ('native', 'gpu')},
        sustained_stand_verified=False, perturbations_ready=False, arena_changed=False,
        verification_new_physics_steps=0, verification_new_forward_passes=0, verification_training_updates=0)
    save(OUT/'report.json', summary)
    save(OUT/'verification.json', dict(protected_hashes=hashes, all_hashes_verified=True,
        sensitivity_aggregates_recomputed=True, duration_rule_recomputed=True,
        output_hashes={str(p): digest(p) for p in OUT.iterdir() if p.is_file()}))
    print({k: summary[k] for k in ('protected_files', 'accepted_assessment_cases', 'total_ppo_updates', 'total_transitions', 'final_stand_passes')})


if __name__ == '__main__':
    run()

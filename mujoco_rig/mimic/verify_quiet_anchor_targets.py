"""Independent audit of fixed zero-target versus local-target anchor correction."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_anchor_retraction import (
    actor_hash, compare, finite_number, linear_verify, probe_changes, quiet_jacobian,
    rms, stage, tensors,
)

OUT = Path('logs/mimickit-training/recovery-quiet-anchor-targets-20261009-01')
BASE = Path('logs/mimickit-training/recovery-quiet-anchor-retraction-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
PROBES = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
SEEDS = (921009, 931009)
CUTOFF = 1e-6
SIGMA = .08717797887081347
ANCHOR_LIMIT = 1e-5
KL_LIMIT = .01


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def actor_from_vector(parameters):
    assert parameters.dtype == np.float32 and parameters.ndim == 1 and np.isfinite(parameters).all()
    actor = AnkleHead().eval()
    assert parameters_to_vector(actor.parameters()).numel() == len(parameters)
    vector_to_parameters(torch.from_numpy(parameters.copy()), actor.parameters())
    return actor


def verify_provenance():
    inputs, done = read(OUT/'inputs.json'), read(OUT/'completed.json')
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    resolved = {Path(path).resolve(): sha for path, sha in inputs.items()}
    assert digest(Path(__file__)) == resolved[Path(__file__).resolve()]
    for path, sha in inputs.items():
        path = Path(path)
        if path.suffix == '.py' and path.resolve().parent == Path(__file__).resolve().parent:
            assert digest(OUT/'sources'/path.name) == sha
    required = [LEARNING/'quiet-anchors.npz']
    required += [BASE/str(seed)/name for seed in SEEDS for name in ('evidence.npz', 'report.json')]
    required += [LEARNING/f'fit-{seed}'/'collection-1.npz' for seed in SEEDS]
    required += [PROBES/str(seed)/'dataset.npz' for seed in SEEDS]
    assert all(path.resolve() in resolved for path in required)
    historical = {}
    for root in (BASE, LEARNING, PROBES):
        assert read(root/'audit.json')['passed']
        outputs = read(root/'completed.json')['outputs']
        sealed_outputs = 0
        for path, sha in resolved.items():
            if path.is_relative_to(root.resolve()):
                relative = str(path.relative_to(root.resolve()))
                if relative in outputs:
                    assert sha == outputs[relative], (root, relative)
                    sealed_outputs += 1
        old = {Path(path).resolve(): sha for path, sha in read(root/'inputs.json').items()}
        shared = sorted(set(old) & set(resolved))
        assert shared and all(old[path] == resolved[path] for path in shared)
        historical[str(root)] = dict(shared_inputs=len(shared), sealed_outputs=sealed_outputs)
    assert historical == read(OUT/'historical-inputs.json')
    return inputs, done, historical


def verify_actor(seed, features, zero):
    previous = arrays(BASE/str(seed)/'evidence.npz')
    previous_report = read(BASE/str(seed)/'report.json')
    evidence = arrays(OUT/str(seed)/'evidence.npz')
    report = read(OUT/str(seed)/'report.json')
    batch = tensors(arrays(LEARNING/f'fit-{seed}'/'collection-1.npz'))
    probes = arrays(PROBES/str(seed)/'dataset.npz')
    assert probes['probe_x'].shape == (36, 427)
    assert np.array_equal(probes['probe_x'][probes['probe_branch'] == 0], features.numpy())
    assert np.array_equal(evidence['minibatch_indices'], previous['minibatch_indices'])
    ids = torch.from_numpy(previous['minibatch_indices'])
    assert ids.shape == (256,) and len(torch.unique(ids)) == 256
    actors = [actor_from_vector(previous[key]) for key in
              ('parameters_pre', 'parameters_rejected', 'parameters_corrected')]
    states, errors = [], []
    for index, (name, actor) in enumerate(zip(('pre', 'rejected', 'zero'), actors)):
        expected, metrics, arithmetic = stage(actor, name, batch, ids, features, zero, probes)
        for key, value in expected.items():
            assert np.array_equal(value, previous[key][index]), (seed, name, 'old', key)
            assert np.array_equal(value, evidence[key][index]), (seed, name, 'new', key)
        compare(metrics, dict(previous_report['states'][index], name=name), name+' prior metrics', atol=0, rtol=0)
        compare(report['states'][index], metrics, name+' metrics', atol=0, rtol=0)
        states.append(metrics)
        errors.append(arithmetic)
    quiet = quiet_jacobian(actors[1], features)
    assert np.array_equal(quiet, previous['jacobian_candidate'])
    local_target = previous['anchor_predictions'][0].astype(float)
    residual = previous['anchor_predictions'][1].astype(float)-local_target
    assert np.array_equal(evidence['local_target'], local_target)
    assert np.array_equal(evidence['residual'], residual)
    left, singular, right = np.linalg.svd(quiet, full_matrices=False)
    keep = singular > CUTOFF*singular[0]
    correction = -right[keep].T@((left[:, keep].T@residual)/singular[keep])
    assert np.array_equal(singular, previous['singular_values'])
    assert np.array_equal(singular, evidence['singular_values'])
    assert np.array_equal(correction, evidence['correction'])
    independent = linear_verify(quiet, residual, correction, singular)
    compare(report['independent'], independent, 'independent local correction')
    rejected_parameters = torch.from_numpy(previous['parameters_rejected'])
    local_parameters = rejected_parameters+torch.from_numpy(correction.astype(np.float32))
    assert np.array_equal(local_parameters.numpy(), evidence['parameters_local'])
    local = actor_from_vector(local_parameters.numpy())
    actual = local_parameters.double().numpy()-rejected_parameters.double().numpy()
    assert np.array_equal(actual, evidence['actual_correction'])
    assert np.array_equal(quiet@correction, evidence['linear_prediction'])
    assert np.array_equal(quiet@actual, evidence['rounded_prediction'])
    expected, metrics, arithmetic = stage(local, 'local', batch, ids, features, zero, probes)
    for key, value in expected.items():
        assert np.array_equal(value, evidence[key][3], equal_nan=True), (seed, 'local', key)
    compare(report['states'][3], metrics, 'local metrics', atol=0, rtol=0)
    states.append(metrics)
    errors.append(arithmetic)
    assert states[0]['passes_guards'] and not states[1]['passes_guards'] and states[2]['passes_guards']
    proposal = previous['parameters_rejected'].astype(float)-previous['parameters_pre'].astype(float)
    local_step = local_parameters.double().numpy()-previous['parameters_pre'].astype(float)
    proposal_norm, local_norm = np.linalg.norm(proposal), np.linalg.norm(local_step)
    tentative_norm = previous_report['correction']['tentative_adam_norm']
    local_residual = evidence['anchor_predictions'][3].astype(float)-local_target
    nonlinear = (evidence['anchor_predictions'][3].astype(float)
                 -evidence['anchor_predictions'][1].astype(float)-quiet@actual)
    geometry = dict(rank=int(keep.sum()), threshold=CUTOFF,
        proposed_norm=float(np.linalg.norm(correction)), actual_norm=finite_number(np.linalg.norm(actual)),
        rounding_norm=finite_number(np.linalg.norm(actual-correction)),
        rejected_minus_pre_norm=float(proposal_norm), tentative_adam_norm=tentative_norm,
        correction_to_proposal_norm=float(np.linalg.norm(correction)/proposal_norm),
        actual_correction_to_proposal_norm=finite_number(np.linalg.norm(actual)/proposal_norm),
        correction_to_tentative_adam_norm=float(np.linalg.norm(correction)/tentative_norm),
        actual_correction_to_tentative_adam_norm=finite_number(np.linalg.norm(actual)/tentative_norm),
        local_minus_pre_norm=finite_number(local_norm),
        local_step_cosine_to_rejected=finite_number(local_step@proposal/(local_norm*proposal_norm)),
        linear_residual_max=float(np.abs(residual+quiet@correction).max()),
        rounded_linear_residual_max=finite_number(np.abs(residual+quiet@actual).max()),
        local_target_residual_max=finite_number(np.abs(local_residual).max()),
        nonlinear_rms=rms(nonlinear) if states[3]['finite'] else None,
        nonlinear_max=float(np.abs(nonlinear).max()) if states[3]['finite'] else None)
    compare(report['correction'], geometry, 'geometry')
    changes = {}
    comparisons = (('rejected_minus_pre', 1, 0), ('zero_minus_pre', 2, 0),
        ('local_minus_pre', 3, 0), ('local_minus_rejected', 3, 1), ('local_minus_zero', 3, 2))
    for label, end, start in comparisons:
        changes[label] = None if not (states[end]['finite'] and states[start]['finite']) else dict(
            full_surrogate=states[end]['full_surrogate']-states[start]['full_surrogate'],
            minibatch_surrogate=states[end]['minibatch_surrogate']-states[start]['minibatch_surrogate'],
            probes=probe_changes(evidence['probe_predictions'][end].astype(float)
                                -evidence['probe_predictions'][start].astype(float), probes))
    compare(report['changes'], changes, 'changes')
    assert report['seed'] == seed and report['collection'] == 1 and report['attempt'] == 9
    assert report['state_order'] == ['pre', 'rejected', 'zero', 'local']
    assert report['optimizer_operations'] == 0 and report['previous_states_exact']
    assert report['candidate_jacobian_reused']
    assert report['historical_tentative_optimizer_sha256'] == previous_report['optimizer_tentative_before']
    return dict(previous_three_states_exact=True, candidate_jacobian_independently_reconstructed=True,
        local_residual_uses_pre_update_target=True, guard_uses_original_zero_target=True,
        predictions_and_float32_application_exact=True, independent_minimum_norm=independent,
        float64_probability_objective_errors=errors, optimizer_operations=0,
        local_passes_guards=states[3]['passes_guards'], local_target_max_error=geometry['local_target_residual_max'],
        local_minus_pre_full_surrogate=changes['local_minus_pre']['full_surrogate'] if changes['local_minus_pre'] else None,
        local_minus_pre_minibatch_surrogate=changes['local_minus_pre']['minibatch_surrogate'] if changes['local_minus_pre'] else None,
        scope='Local fitted-state target comparison; feasibility, objective movement and physical benefit are distinct.')


def main():
    inputs, done, historical = verify_provenance()
    contract = read(OUT/'contract.json')
    assert contract['seeds'] == list(SEEDS) and contract['source'] == str(BASE)
    assert contract['state_order'] == ['pre', 'rejected', 'zero', 'local']
    assert contract['fixed_collection'] == 1 and contract['fixed_attempt'] == 9
    assert contract['corrections_per_actor'] == 1
    assert contract['correction_target'] == 'pre-ninth quiet outputs'
    assert contract['global_anchor_target'] == 'original zero outputs'
    assert contract['sigma'] == SIGMA and contract['singular_relative_cutoff'] == CUTOFF
    assert contract['anchor_limit'] == ANCHOR_LIMIT and contract['kl_limit'] == KL_LIMIT
    assert contract['candidate_jacobian_reused'] and contract['independent_linear_gate_before_application']
    assert not contract['line_search'] and not contract['correction_rescaled'] and contract['retries'] == 0
    assert contract['anchor_count'] == 12 and contract['probe_count'] == 36
    assert contract['saved_collection_samples'] == 768 and contract['rejected_minibatch_samples'] == 256
    for key in ('optimizer_operations', 'historical_actor_attempts_replayed', 'physical_rollouts',
                'new_training_updates', 'newly_collected_transitions', 'critic_updates', 'new_checkpoints'):
        assert contract[key] == done[key] == 0
    assert done['inputs_unchanged'] and done['previous_states_exact']
    assert done['actors'] == done['corrections'] == 2
    assert not list(OUT.rglob('*.pt'))
    bank = arrays(LEARNING/'quiet-anchors.npz')
    assert bank['features'].shape == (12, 427) and bank['target'].shape == (12, 1)
    assert bank['features'].dtype == bank['target'].dtype == np.float32
    assert np.isfinite(bank['features']).all() and not bank['features'][:, -2:].any() and not bank['target'].any()
    features, zero = torch.from_numpy(bank['features']), bank['target'].ravel()
    torch.set_num_threads(1)
    actors = {str(seed): verify_actor(seed, features, zero) for seed in SEEDS}
    results = read(OUT/'results.json')
    assert set(results) == {str(seed) for seed in SEEDS}
    assert all(results[str(seed)] == read(OUT/str(seed)/'report.json') for seed in SEEDS)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    audit = dict(passed=True, input_hashes=len(inputs), output_hashes=len(done['outputs']),
        historical=historical, actors=actors, corrected_actor_copies=2,
        optimizer_operations=0, historical_actor_attempts_replayed=0, new_physical_rollouts=0,
        scope='Audit success verifies the comparison; neither guard success nor objective improvement proves physical recovery.')
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()

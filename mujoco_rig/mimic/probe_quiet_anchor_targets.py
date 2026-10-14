"""Compare zero and pre-update quiet targets on fixed saved rejected candidates."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from . import probe_quiet_anchor_retraction as prior
from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_anchor_retraction import linear_verify

BASE = prior.OUT
LEARNING = prior.BASE
PROBES = prior.PROBES
OUT = Path('logs/mimickit-training/recovery-quiet-anchor-targets-20261009-01')
SEEDS = (921009, 931009)
STATES = ('pre', 'rejected', 'zero', 'local')
COMPARISONS = (('rejected_minus_pre', 1, 0), ('zero_minus_pre', 2, 0),
               ('local_minus_pre', 3, 0), ('local_minus_rejected', 3, 1),
               ('local_minus_zero', 3, 2))


def restore(parameters):
    assert parameters.dtype == np.float32 and np.isfinite(parameters).all()
    actor = AnkleHead().eval()
    vector_to_parameters(torch.from_numpy(parameters).clone(), actor.parameters())
    return actor


def experiment(seed, features, zero):
    folder = OUT/str(seed)
    folder.mkdir()
    saved = prior.arrays(BASE/str(seed)/'evidence.npz')
    old_report = prior.read(BASE/str(seed)/'report.json')
    batch = prior.batch_tensors(prior.arrays(LEARNING/f'fit-{seed}'/'collection-1.npz'))
    ids = torch.from_numpy(saved['minibatch_indices'])
    probes = prior.arrays(PROBES/str(seed)/'dataset.npz')
    assert np.array_equal(probes['probe_x'][probes['probe_branch'] == 0], features.numpy())
    actors = [restore(saved[key]) for key in
              ('parameters_pre', 'parameters_rejected', 'parameters_corrected')]
    measurements, states = [], []
    for index, (name, actor) in enumerate(zip(STATES[:3], actors)):
        evidence, state = prior.state_measurements(actor, name, batch, ids, features, zero,
                                                 torch.from_numpy(probes['probe_x']))
        for key, value in evidence.items():
            assert np.array_equal(value, saved[key][index]), (seed, name, key)
        expected = dict(old_report['states'][index], name=name)
        assert state == expected
        measurements.append(evidence)
        states.append(state)
    jacobian = saved['jacobian_candidate']
    local_target = measurements[0]['anchor_predictions'].astype(np.float64)
    residual = measurements[1]['anchor_predictions'].astype(np.float64)-local_target
    left, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    assert np.array_equal(singular, saved['singular_values'])
    keep = singular > prior.CUTOFF*singular[0]
    correction = -right[keep].T@((left[:, keep].T@residual)/singular[keep])
    independent = linear_verify(jacobian, residual, correction, singular)
    assert independent['passed']
    rejected_parameters = torch.from_numpy(saved['parameters_rejected'])
    local_parameters = rejected_parameters+torch.from_numpy(correction.astype(np.float32))
    local = restore(local_parameters.numpy())
    evidence, state = prior.state_measurements(local, 'local', batch, ids, features, zero,
                                             torch.from_numpy(probes['probe_x']))
    measurements.append(evidence)
    states.append(state)
    actual = local_parameters.double().numpy()-rejected_parameters.double().numpy()
    proposal = saved['parameters_rejected'].astype(float)-saved['parameters_pre'].astype(float)
    local_step = local_parameters.double().numpy()-saved['parameters_pre'].astype(float)
    proposal_norm, local_norm = np.linalg.norm(proposal), np.linalg.norm(local_step)
    tentative_norm = old_report['correction']['tentative_adam_norm']
    local_residual = evidence['anchor_predictions'].astype(float)-local_target
    nonlinear = evidence['anchor_predictions'].astype(float)-measurements[1]['anchor_predictions'].astype(float)-jacobian@actual
    stacked = {key: np.stack([row[key] for row in measurements]) for key in measurements[0]}
    stacked.update(parameters_local=local_parameters.numpy(), local_target=local_target,
        residual=residual, correction=correction, actual_correction=actual,
        singular_values=singular, linear_prediction=jacobian@correction,
        rounded_prediction=jacobian@actual, minibatch_indices=ids.numpy())
    np.savez_compressed(folder/'evidence.npz', **stacked)
    geometry = dict(rank=int(keep.sum()), threshold=prior.CUTOFF,
        proposed_norm=float(np.linalg.norm(correction)), actual_norm=prior.finite_number(np.linalg.norm(actual)),
        rounding_norm=prior.finite_number(np.linalg.norm(actual-correction)),
        rejected_minus_pre_norm=float(proposal_norm), tentative_adam_norm=tentative_norm,
        correction_to_proposal_norm=float(np.linalg.norm(correction)/proposal_norm),
        actual_correction_to_proposal_norm=prior.finite_number(np.linalg.norm(actual)/proposal_norm),
        correction_to_tentative_adam_norm=float(np.linalg.norm(correction)/tentative_norm),
        actual_correction_to_tentative_adam_norm=prior.finite_number(np.linalg.norm(actual)/tentative_norm),
        local_minus_pre_norm=prior.finite_number(local_norm),
        local_step_cosine_to_rejected=prior.finite_number(local_step@proposal/(local_norm*proposal_norm)),
        linear_residual_max=float(np.abs(residual+jacobian@correction).max()),
        rounded_linear_residual_max=prior.finite_number(np.abs(residual+jacobian@actual).max()),
        local_target_residual_max=prior.finite_number(np.abs(local_residual).max()),
        nonlinear_rms=prior.rms(nonlinear) if state['finite'] else None,
        nonlinear_max=float(np.abs(nonlinear).max()) if state['finite'] else None)
    changes = {}
    for label, end, start in COMPARISONS:
        changes[label] = None if not (states[end]['finite'] and states[start]['finite']) else dict(
            full_surrogate=states[end]['full_surrogate']-states[start]['full_surrogate'],
            minibatch_surrogate=states[end]['minibatch_surrogate']-states[start]['minibatch_surrogate'],
            probes=prior.probe_change(stacked['probe_predictions'][end].astype(float)
                                     -stacked['probe_predictions'][start].astype(float), probes))
    report = dict(seed=seed, collection=1, attempt=9, state_order=list(STATES), states=states,
        independent=independent, correction=geometry, changes=changes,
        historical_tentative_optimizer_sha256=old_report['optimizer_tentative_before'],
        optimizer_operations=0, previous_states_exact=True, candidate_jacobian_reused=True,
        scope='One local-target correction of the same saved rejected candidate; no physical recovery inference.')
    save(folder/'report.json', report)
    return report


def provenance():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    sources = sorted(Path(__file__).parent.glob('*.py'))
    for path in sources:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths = sources+[Path(__file__).with_name('RECOVERY_QUIET_ANCHOR_TARGETS_PLAN.md'),
        Path(__file__).with_name('RECOVERY_QUIET_ANCHOR_RETRACTION_PROGRESS.md')]
    for root in (BASE, LEARNING, PROBES):
        assert prior.read(root/'audit.json')['passed']
        paths += [root/'inputs.json', root/'completed.json', root/'audit.json']
    paths += [LEARNING/'quiet-anchors.npz']
    for seed in SEEDS:
        paths += [BASE/str(seed)/'evidence.npz', BASE/str(seed)/'report.json',
                  LEARNING/f'fit-{seed}'/'collection-1.npz',
                  LEARNING/f'fit-{seed}'/'objective-1.npz', PROBES/str(seed)/'dataset.npz']
    inputs = {str(path): digest(path) for path in paths}
    historical = {}
    for root in (BASE, LEARNING, PROBES):
        manifest = prior.read(root/'completed.json')['outputs']
        count = 0
        for path in paths:
            if path.is_relative_to(root) and str(path.relative_to(root)) in manifest:
                assert inputs[str(path)] == manifest[str(path.relative_to(root))]
                count += 1
        previous = prior.read(root/'inputs.json')
        shared = sorted(set(previous) & set(inputs))
        assert shared and all(previous[path] == inputs[path] for path in shared)
        historical[str(root)] = dict(sealed_outputs=count, shared_inputs=len(shared))
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', historical)
    return inputs


def main():
    started = time.perf_counter()
    inputs = provenance()
    torch.set_num_threads(1)
    bank = prior.arrays(LEARNING/'quiet-anchors.npz')
    assert bank['features'].shape == (12, 427) and not bank['features'][:, -2:].any()
    assert bank['target'].shape == (12, 1) and not bank['target'].any()
    features, zero = torch.from_numpy(bank['features']), bank['target'].ravel()
    save(OUT/'contract.json', dict(seeds=list(SEEDS), source=str(BASE), state_order=list(STATES),
        fixed_collection=1, fixed_attempt=9, corrections_per_actor=1,
        correction_target='pre-ninth quiet outputs', global_anchor_target='original zero outputs',
        singular_relative_cutoff=prior.CUTOFF, anchor_limit=prior.ANCHOR_LIMIT, kl_limit=prior.KL_LIMIT,
        sigma=prior.SIGMA, step_cast='cast correction once to float32, then add rejected float32 parameters',
        candidate_jacobian_reused=True, independent_linear_gate_before_application=True,
        line_search=False, retries=0, correction_rescaled=False, optimizer_operations=0,
        historical_actor_attempts_replayed=0, physical_rollouts=0, new_training_updates=0,
        newly_collected_transitions=0, critic_updates=0, new_checkpoints=0,
        anchor_count=12, probe_count=36, saved_collection_samples=768, rejected_minibatch_samples=256))
    results = {}
    for seed in SEEDS:
        results[str(seed)] = experiment(seed, features, zero)
        print('ACTOR', seed, json.dumps(dict(states=results[str(seed)]['states'],
            local_minus_pre=results[str(seed)]['changes']['local_minus_pre'],
            correction=results[str(seed)]['correction'])), flush=True)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, inputs_unchanged=True,
        actors=2, corrections=2, previous_states_exact=True,
        optimizer_operations=0, historical_actor_attempts_replayed=0, physical_rollouts=0,
        new_training_updates=0, newly_collected_transitions=0, critic_updates=0, new_checkpoints=0,
        outputs={str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and 'sources' not in path.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

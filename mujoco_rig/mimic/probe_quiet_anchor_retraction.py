"""One finite anchor retraction on each fixed, exactly reconstructed rejected actor."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .quiet_constrained_ppo import actor_hash, optimizer_hash, quiet_jacobian
from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_anchor_retraction import linear_verify

BASE = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
PROBES = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
OUT = Path('logs/mimickit-training/recovery-quiet-anchor-retraction-20261009-01')
SEEDS = (921009, 931009)
STATE_ORDER = ('pre', 'rejected', 'corrected')
SIGMA = .08717797887081347
CUTOFF = 1e-6
ANCHOR_LIMIT = 1e-5
KL_LIMIT = .01


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def vector(actor):
    return parameters_to_vector(actor.parameters()).detach().clone()


def batch_tensors(batch):
    return dict(x=torch.from_numpy(batch['x']).flatten(0, 1),
        action=torch.from_numpy(batch['action']).flatten(0, 1),
        old_mean=torch.from_numpy(batch['mean']).flatten(0, 1),
        old_logp=torch.from_numpy(batch['logp']).flatten(),
        advantage=torch.from_numpy(batch['normalized_advantage']),
        weight=torch.from_numpy(batch['actor_weight']).flatten())


def ratio_and_surrogate(mean, t, ids):
    ratio = (Normal(mean, SIGMA).log_prob(t['action'][ids]).sum(-1)-t['old_logp'][ids]).exp()
    advantage = t['advantage'][ids]
    surrogate = (t['weight'][ids]*torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).mean()
    return ratio, surrogate


def replay(seed, features, target):
    """Replay nine historical actor attempts; retain the rejected ninth copy before rollback."""
    folder = BASE/f'fit-{seed}'
    batch, objective = arrays(folder/'collection-1.npz'), arrays(folder/'objective-1.npz')
    historical = read(folder/'projection-1.json')
    assert len(historical) == 9 and all(row['accepted'] for row in historical[:8])
    assert historical[8]['reasons'] == ['anchor'] and not historical[8]['accepted']
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    with torch.no_grad():
        assert np.array_equal(actor(features).numpy().ravel(), target)
        assert not np.any(target)
    t = batch_tensors(batch)
    evidence = []
    selected = None
    for index, historical_row in enumerate(historical):
        ids = torch.from_numpy(objective['minibatch_indices'][index])
        assert (historical_row['update'], historical_row['epoch'], historical_row['batch'], historical_row['minibatch_index']) == (1, index//3, index % 3, index)
        hashes = dict(actor_pre=actor_hash(actor), optimizer_pre=optimizer_hash(optimizer))
        assert all(hashes[key] == historical_row[key] for key in hashes)
        ratio, objective_value = ratio_and_surrogate(actor(t['x'][ids]), t, ids)
        loss = -objective_value
        assert np.array_equal(ratio.detach().numpy(), objective['policy_ratio'][index])
        assert float(loss.detach()) == historical_row['actor_loss'] == float(objective['actor_loss'][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
        clipped_norm = float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in actor.parameters()])))
        assert gradient_norm == historical_row['gradient_norm'] and clipped_norm == historical_row['clipped_norm']
        old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
        before = vector(actor)
        if index == 8:
            pre = deepcopy(actor)
        jacobian = quiet_jacobian(actor, features)
        _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
        kept = right[singular > CUTOFF*singular[0]]
        assert np.array_equal(singular, historical_row['singular_values'])
        optimizer.step()
        hashes.update(actor_tentative=actor_hash(actor), optimizer_tentative=optimizer_hash(optimizer))
        assert all(hashes[key] == historical_row[key] for key in hashes)
        displacement = vector(actor).double().numpy()-before.double().numpy()
        projected = displacement-kept.T@(kept@displacement)
        candidate = before+torch.from_numpy(projected.astype(np.float32))
        vector_to_parameters(candidate, actor.parameters())
        hashes['actor_candidate'] = actor_hash(actor)
        assert hashes['actor_candidate'] == historical_row['actor_candidate']
        with torch.no_grad():
            anchor = actor(features).numpy().ravel()
            kl = float(((actor(t['x'])-t['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
        assert np.array_equal(anchor, historical_row['anchor_attempted']) and kl == historical_row['kl']
        accepted = bool(np.isfinite(kl) and kl <= KL_LIMIT and np.abs(anchor-target).max() <= ANCHOR_LIMIT)
        assert accepted is historical_row['accepted']
        if index == 8:
            selected = (pre, deepcopy(actor), deepcopy(optimizer.state_dict()), t, ids.clone())
        if not accepted:
            actor.load_state_dict(old_actor)
            optimizer.load_state_dict(old_optimizer)
        hashes.update(actor_post=actor_hash(actor), optimizer_post=optimizer_hash(optimizer))
        assert all(hashes[key] == historical_row[key] for key in hashes)
        evidence.append(dict(index=index, accepted=accepted, actor_loss=float(loss.detach()),
            gradient_norm=gradient_norm, clipped_norm=clipped_norm, kl=kl,
            anchor_max=float(np.abs(anchor-target).max()), exact_historical_hashes=True, **hashes))
    with torch.no_grad():
        assert np.array_equal(actor(t['x']).numpy(), batch['post_mean'])
    history = read(folder/'history.json')[0]
    assert actor_hash(actor) == history['actor_sha256'] and optimizer_hash(optimizer) == history['optimizer_sha256']
    assert selected is not None
    return (*selected, evidence)


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64)**2)))


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def probe_change(delta, probes):
    value = np.asarray(delta, dtype=np.float64).reshape(36)
    branch = probes['probe_branch']
    response = value-value[probes['probe_quiet_index']]
    def summary(mask):
        return dict(raw_mean=float(value[mask].mean()), raw_rms=rms(value[mask]),
                    response_mean=float(response[mask].mean()), response_rms=rms(response[mask]))
    return dict(quiet_rms=rms(value[branch == 0]), quiet_max=float(np.abs(value[branch == 0]).max()),
        push=summary(branch != 0), positive_x=summary(branch == 1), negative_x=summary(branch == 2))


def state_measurements(actor, name, t, ids, features, target, probe_x):
    with torch.no_grad():
        full_mean = actor(t['x'])
        minibatch_mean = actor(t['x'][ids])
        anchor = actor(features).numpy().ravel().copy()
        probe = actor(probe_x).numpy().copy()
        full_ratio, full_surrogate = ratio_and_surrogate(full_mean, t, torch.arange(len(t['x'])))
        minibatch_ratio, minibatch_surrogate = ratio_and_surrogate(minibatch_mean, t, ids)
        kl = float(((full_mean-t['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
    finite = bool(torch.isfinite(vector(actor)).all() and torch.isfinite(full_mean).all()
                  and torch.isfinite(minibatch_mean).all() and torch.isfinite(full_ratio).all()
                  and torch.isfinite(minibatch_ratio).all() and np.isfinite(anchor).all()
                  and np.isfinite(probe).all() and np.isfinite(kl)
                  and torch.isfinite(full_surrogate) and torch.isfinite(minibatch_surrogate))
    anchor_max = float(np.max(np.abs(anchor-target)))
    report = dict(name=name, actor_sha256=actor_hash(actor), finite=finite,
        full_surrogate=finite_number(float(full_surrogate)), minibatch_surrogate=finite_number(float(minibatch_surrogate)),
        kl=finite_number(kl), anchor_max=finite_number(anchor_max),
        passes_guards=finite and kl <= KL_LIMIT and anchor_max <= ANCHOR_LIMIT)
    return dict(collection_predictions=full_mean.numpy().copy(), minibatch_predictions=minibatch_mean.numpy().copy(),
        full_ratio=full_ratio.numpy().copy(), minibatch_ratio=minibatch_ratio.numpy().copy(),
        anchor_predictions=anchor, probe_predictions=probe), report


def experiment(seed, features, target):
    folder = OUT/str(seed)
    folder.mkdir()
    pre, rejected, tentative_state, t, ids, replay_evidence = replay(seed, features, target)
    probes = arrays(PROBES/str(seed)/'dataset.npz')
    assert probes['probe_x'].shape == (36, 427)
    quiet_indices = np.flatnonzero(probes['probe_branch'] == 0)
    assert np.array_equal(probes['probe_x'][quiet_indices], features.numpy())
    probe_x = torch.from_numpy(probes['probe_x'])
    with torch.no_grad():
        residual = rejected(features).numpy().ravel().astype(np.float64)-target.astype(np.float64)
    jacobian = quiet_jacobian(rejected, features)
    left, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    keep = singular > CUTOFF*singular[0]
    correction = -right[keep].T@((left[:, keep].T@residual)/singular[keep])
    independent = linear_verify(jacobian, residual, correction, singular)
    assert independent['passed']
    pre_parameters, rejected_parameters = vector(pre), vector(rejected)
    corrected = deepcopy(rejected)
    optimizer = torch.optim.Adam(corrected.parameters(), lr=1e-4)
    optimizer.load_state_dict(tentative_state)
    optimizer_before = optimizer_hash(optimizer)
    assert optimizer_before == replay_evidence[-1]['optimizer_tentative']
    corrected_parameters = rejected_parameters+torch.from_numpy(correction.astype(np.float32))
    vector_to_parameters(corrected_parameters, corrected.parameters())
    actual = corrected_parameters.double().numpy()-rejected_parameters.double().numpy()
    optimizer_after = optimizer_hash(optimizer)
    assert optimizer_before == optimizer_after
    measurements, states = [], []
    for name, actor in zip(STATE_ORDER, (pre, rejected, corrected)):
        evidence, state = state_measurements(actor, name, t, ids, features, target, probe_x)
        measurements.append(evidence)
        states.append(state)
    assert states[0]['actor_sha256'] == replay_evidence[-1]['actor_pre']
    assert states[1]['actor_sha256'] == replay_evidence[-1]['actor_candidate']
    assert states[0]['passes_guards'] and not states[1]['passes_guards']
    evidence = {key: np.stack([row[key] for row in measurements]) for key in measurements[0]}
    linear_prediction, rounded_prediction = jacobian@correction, jacobian@actual
    evidence.update(parameters_pre=pre_parameters.numpy(), parameters_rejected=rejected_parameters.numpy(),
        parameters_corrected=corrected_parameters.numpy(), jacobian_candidate=jacobian, residual=residual,
        correction=correction, actual_correction=actual, singular_values=singular,
        linear_prediction=linear_prediction, rounded_prediction=rounded_prediction,
        minibatch_indices=ids.numpy())
    np.savez_compressed(folder/'evidence.npz', **evidence)
    proposal = rejected_parameters.double().numpy()-pre_parameters.double().numpy()
    corrected_change = corrected_parameters.double().numpy()-pre_parameters.double().numpy()
    proposal_norm, corrected_change_norm = np.linalg.norm(proposal), np.linalg.norm(corrected_change)
    tentative_norm = read(BASE/f'fit-{seed}'/'projection-1.json')[-1]['tentative_norm']
    finite = states[2]['finite']
    nonlinear = evidence['anchor_predictions'][2].astype(float)-evidence['anchor_predictions'][1].astype(float)-rounded_prediction
    report = dict(seed=seed, collection=1, attempt=9, state_order=list(STATE_ORDER), states=states,
        optimizer_tentative_before=optimizer_before, optimizer_tentative_after=optimizer_after,
        tentative_optimizer_unchanged=True, independent=independent,
        correction=dict(rank=int(keep.sum()), threshold=CUTOFF, proposed_norm=float(np.linalg.norm(correction)),
            actual_norm=finite_number(np.linalg.norm(actual)), rounding_norm=finite_number(np.linalg.norm(actual-correction)),
            rejected_minus_pre_norm=float(proposal_norm),
            tentative_adam_norm=tentative_norm,
            correction_to_proposal_norm=float(np.linalg.norm(correction)/proposal_norm),
            actual_correction_to_proposal_norm=finite_number(np.linalg.norm(actual)/proposal_norm),
            correction_to_tentative_adam_norm=float(np.linalg.norm(correction)/tentative_norm),
            actual_correction_to_tentative_adam_norm=finite_number(np.linalg.norm(actual)/tentative_norm),
            corrected_minus_pre_norm=finite_number(corrected_change_norm),
            corrected_step_cosine_to_rejected=finite_number(corrected_change@proposal/(corrected_change_norm*proposal_norm)),
            linear_residual_max=float(np.abs(residual+linear_prediction).max()),
            rounded_linear_residual_max=finite_number(np.abs(residual+rounded_prediction).max()),
            nonlinear_rms=rms(nonlinear) if finite else None,
            nonlinear_max=float(np.abs(nonlinear).max()) if finite else None),
        changes={label: dict(full_surrogate=states[end]['full_surrogate']-states[start]['full_surrogate'],
                minibatch_surrogate=states[end]['minibatch_surrogate']-states[start]['minibatch_surrogate'],
                probes=probe_change(evidence['probe_predictions'][end].astype(float)-evidence['probe_predictions'][start].astype(float), probes))
            if states[end]['finite'] and states[start]['finite'] else None
            for label, end, start in (('rejected_minus_pre', 1, 0), ('corrected_minus_pre', 2, 0), ('corrected_minus_rejected', 2, 1))},
        scope='One offline correction on the fixed first rejected candidate. Probe changes are descriptive; '
              'no recovery benefit or optimal action direction is inferred.')
    save(folder/'report.json', report)
    save(folder/'replay.json', replay_evidence)
    assert actor_hash(pre) == replay_evidence[-1]['actor_pre']
    assert actor_hash(rejected) == replay_evidence[-1]['actor_candidate']
    assert optimizer_hash(optimizer) == optimizer_before
    return report


def provenance():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    sources = sorted(Path(__file__).parent.glob('*.py'))
    for path in sources:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths = sources+[Path(__file__).with_name('RECOVERY_QUIET_ANCHOR_RETRACTION_PLAN.md'),
        Path(__file__).with_name('RECOVERY_QUIET_CONSTRAINED_LEARNING_PROGRESS.md'),
        BASE/'inputs.json', BASE/'completed.json', BASE/'audit.json', BASE/'quiet-anchors.npz',
        PROBES/'inputs.json', PROBES/'completed.json', PROBES/'audit.json']
    paths += [BASE/f'fit-{seed}'/name for seed in SEEDS for name in
              ('initial.pt', 'collection-1.npz', 'objective-1.npz', 'projection-1.json', 'history.json')]
    paths += [PROBES/str(seed)/'dataset.npz' for seed in SEEDS]
    inputs = {str(path): digest(path) for path in paths}
    historical = {}
    for root in (BASE, PROBES):
        assert read(root/'audit.json')['passed']
        manifest = read(root/'completed.json')['outputs']
        for path in paths:
            if path.is_relative_to(root) and str(path.relative_to(root)) in manifest:
                assert digest(path) == manifest[str(path.relative_to(root))]
        prior_inputs = read(root/'inputs.json')
        shared = sorted(set(prior_inputs) & set(inputs))
        assert shared and all(prior_inputs[path] == inputs[path] for path in shared)
        historical[str(root)] = dict(count=len(shared), shared_paths=shared, passed=True)
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', historical)
    return inputs


def main():
    started = time.perf_counter()
    inputs = provenance()
    torch.set_num_threads(1)
    bank = arrays(BASE/'quiet-anchors.npz')
    features, target = torch.from_numpy(bank['features']), bank['target'].ravel()
    assert features.shape == (12, 427) and np.isfinite(bank['features']).all()
    assert np.count_nonzero(bank['features'][:, -2:]) == 0 and not np.any(target)
    save(OUT/'contract.json', dict(seeds=list(SEEDS), source=str(BASE), fixed_collection=1, fixed_attempt=9,
        selection='first rejected candidate after eight accepted updates for both seeds',
        state_order=list(STATE_ORDER), sigma=SIGMA, singular_relative_cutoff=CUTOFF,
        anchor_limit=ANCHOR_LIMIT, kl_limit=KL_LIMIT, corrections_per_actor=1,
        correction='minimum-norm candidate-Jacobian Newton correction toward original zero targets',
        step_cast='cast correction once to float32, then add rejected float32 parameters',
        line_search=False, retries=0, correction_rescaled=False, optimizer_moments_changed=False,
        independent_linear_gate_before_application=True, anchor_count=12, probe_count=36,
        saved_collection_samples=768, rejected_minibatch_samples=256, historical_actor_attempts_replayed=18,
        newly_collected_transitions=0, physical_rollouts=0, new_training_updates=0, critic_updates=0, new_checkpoints=0))
    results = {}
    for seed in SEEDS:
        results[str(seed)] = experiment(seed, features, target)
        print('ACTOR', seed, json.dumps(dict(states=results[str(seed)]['states'], correction=results[str(seed)]['correction'])), flush=True)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, inputs_unchanged=True,
        actors=2, historical_actor_attempts_replayed=18, exact_historical_hash_parity=True,
        corrections=2, independent_linear_checks_passed=True, tentative_optimizers_unchanged=True,
        newly_collected_transitions=0, physical_rollouts=0, new_training_updates=0, critic_updates=0, new_checkpoints=0,
        outputs={str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and 'sources' not in path.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

"""Fresh quiet-constrained ankle learning with the established physical suite."""
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from . import run_direct_recovery as prior
from . import train_age_weighted_recovery as controls
from . import train_coordinated_ankle as ankle
from .age_weighted_ppo import HALF_SIGMA, probability_checks, weighting_checks
from .quiet_constrained_ppo import learn
from .recovery_start_training import RecoveryStartTraining
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleHead, AnkleRecovery, ONSET_FRAMES, OUT as PILOT

OUT = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
WEIGHTED = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
DATA = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
PROJECTION = Path('logs/mimickit-training/recovery-quiet-actor-projection-20261009-01')
SEEDS = (921009, 931009)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def quiet_bank(initial):
    banks = []
    for seed in SEEDS:
        z = controls.arrays(DATA/str(seed)/'dataset.npz')
        quiet = z['probe_branch'] == 0
        assert quiet.sum() == 12
        assert np.array_equal(z['probe_onset'][quiet], np.repeat([44, 74], 6))
        assert np.array_equal(z['probe_pose'][quiet], np.tile(np.arange(6), 2))
        features = z['probe_x'][quiet]
        assert features.shape == (12, 427) and not features[:, -2:].any()
        banks.append(features)
    assert np.array_equal(banks[0], banks[1])
    features = torch.from_numpy(banks[0]).clone()
    with torch.no_grad():
        target = initial(features)
    assert not target.any()
    np.savez_compressed(OUT/'quiet-anchors.npz', features=features.numpy(), target=target.numpy(),
                        onset=np.repeat([44, 74], 6), pose=np.tile(np.arange(6), 2))
    return features


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = sorted(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
        FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_QUIET_CONSTRAINED_LEARNING_PLAN.md'),
        PILOT/'fit-921009'/'initial.pt', PROJECTION/'audit.json', PROJECTION/'completed.json',
        DATA/'audit.json', DATA/'completed.json', WEIGHTED/'inputs.json']
    for root in (WEIGHTED, PILOT):
        assert read(root/'audit.json')['passed']
        completed = read(root/'completed.json')
        paths += [root/'completed.json', root/'audit.json']
        for relative, sha in completed['outputs'].items():
            path = root/relative
            assert digest(path) == sha, path
            paths.append(path)
    assert read(PROJECTION/'audit.json')['passed'] and read(DATA/'audit.json')['passed']
    data_manifest = read(DATA/'completed.json')['outputs']
    for seed in SEEDS:
        path = DATA/str(seed)/'dataset.npz'
        assert digest(path) == data_manifest[str(path.relative_to(DATA))]
        paths.append(path)
    inputs = {str(path): digest(path) for path in paths}
    historical = read(WEIGHTED/'inputs.json')
    common = sorted(set(inputs) & set(historical))
    assert all(inputs[path] == historical[path] for path in common)
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', dict(passed=True, count=len(common), shared_paths=common))
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz', read(FOUNDATION/'protocol.json'), PARENT)
    torch.manual_seed(920009)
    initial = AnkleHead().eval()
    features = quiet_bank(initial)
    started = time.perf_counter()
    save(OUT/'checks.json', dict(**controls.checks(rig, anchor, initial),
        probability=probability_checks(), weighting=weighting_checks(),
        quiet_anchors=dict(count=12, width=427, identical_source_features=True,
                           zero_action_memory=True, fresh_initial_targets_zero=True)))
    frozen_anchor = {key: value.clone() for key, value in anchor.state_dict().items()}
    env = AnkleRecovery(rig, anchor)
    save(OUT/'contract.json', dict(dt=env.dt, ankles=env.ankles, control=env.task.observation_contract(),
        training_seeds=list(SEEDS), training_onset_frames=ONSET_FRAMES.tolist(), observation_width=427,
        nominal_transitions_per_seed=24576, preimpact_optimizer_transitions=0, reward_changed=False,
        initial_actor_seed=920009, sigma=HALF_SIGMA, baseline_sigma=HALF_SIGMA,
        actor_weighting=weighting_checks(), critic_weighted=False, advantage_normalization_changed=False,
        cases=720, matched_cases_per_candidate=160, continuous_cases_per_candidate=176,
        quiet_projection=dict(anchor_count=12, anchor_onsets=[44, 74], anchor_poses=list(range(6)),
            target='fresh initial head outputs', singular_relative_cutoff=1e-6,
            anchor_max_absolute_deviation=1e-5, kl_limit=.01,
            project='actual clipped-Adam parameter displacement at pre-step Jacobian',
            renormalized=False, rollback_actor_and_optimizer=True,
            stop_actor_for_collection_on_rejection=True),
        comparator=str(WEIGHTED), shared_historical_inputs=len(common)))
    del env
    results = {}
    with patch.object(controls, 'OUT', OUT):
        for timing in (0, 1):
            controls.matched(rig, anchor, initial, 'baseline', timing)
        controls.matched(rig, anchor, initial, 'baseline', 0, quiet=True)
        for seed in SEEDS:
            print('BEGIN_TRAIN', seed, flush=True)
            training = RecoveryStartTraining(rig, anchor)
            actor = learn(training, initial, OUT/f'fit-{seed}', seed, HALF_SIGMA, features)
            actor.requires_grad_(False)
            assert training.collected == 24576
            save(OUT/f'fit-{seed}'/'resets.json', training.reset_records)
            save(OUT/f'fit-{seed}'/'start-accounting.json', dict(collected=training.collected,
                source_prefix_intervals=training.prefix_intervals, exact_transfers=training.exact_transfers,
                reset_records=len(training.reset_records), optimizer_prefix_transitions=0))
            del training
            print('BEGIN_EVALUATION', seed, flush=True)
            with patch.object(ankle, 'SIGMA', HALF_SIGMA), patch.object(prior, 'rollout', controls.audited_rollout):
                continuous = ankle.evaluate(rig, anchor, actor, OUT, str(seed))
            restored = []
            for timing in (0, 1):
                for group in range(4):
                    restored.append(controls.matched(rig, anchor, actor, str(seed), timing, group))
                restored.append(controls.matched(rig, anchor, actor, str(seed), timing, quiet=True))
            results[str(seed)] = dict(continuous=continuous, matched={str(group): dict(
                settled=sum(row['settled'] for row in restored if not row['quiet'] and row['group'] == group),
                heldout_settled=sum(case['settled'] and case['heldout'] for row in restored
                    if not row['quiet'] and row['group'] == group for case in row['cases'])) for group in range(4)},
                matched_quiet=sum(row['whole_stand'] for row in restored if row['quiet']))
            save(OUT/'summary.json', results)
            print('EVALUATED', seed, json.dumps(results[str(seed)]), flush=True)
    assert all(torch.equal(value, anchor.state_dict()[key]) for key, value in frozen_anchor.items())
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, cases=720, traces=45,
        total_transitions=49152, frozen_source_unchanged=True, inputs_unchanged=True, results=results,
        outputs={str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and 'sources' not in path.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

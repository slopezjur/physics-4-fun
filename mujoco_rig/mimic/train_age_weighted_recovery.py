"""Fresh age-weighted recovery learning and matched frozen evaluations."""
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from . import run_direct_recovery as prior
from . import train_coordinated_ankle as ankle
from .age_weighted_ppo import HALF_SIGMA, learn, probability_checks, weighting_checks
from .recovery_start_bank import RecoveryStart
from .recovery_start_training import RecoveryStartTraining
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleHead, AnkleRecovery, SIGMA, ONSET_FRAMES, evaluate, OUT as PILOT

FULL = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')
OUT = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
SEEDS = (921009, 931009)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class MatchedContinuation:
    def __init__(self, env, actor, issue):
        self.env, self.actor, self.issue = env, actor, issue
        self.restored = False

    def __call__(self, features):
        frame = int(self.env.task.steps[0])
        if frame < self.issue:
            return torch.zeros(16, 1)
        if not self.restored:
            assert frame == self.issue
            start = RecoveryStart.capture(self.env)
            self.env.reset()
            start.restore(self.env)
            assert torch.equal(features, self.env.features())
            self.restored = True
        return self.actor(features)


def matched(rig, anchor, actor, name, timing, group=0, quiet=False):
    env = AnkleRecovery(rig, anchor)
    onset = (1., 1.7)[timing]
    issue = int(np.ceil(onset/env.dt))+1

    def configure(target, group, quiet):
        target.reset()
        target.force.zero_()
        if not quiet:
            strength, axis = (150., 300.)[group//2], group % 2
            target.force[:8] = strength*prior.DIRECTIONS[2*axis]
            target.force[8:] = strength*prior.DIRECTIONS[2*axis+1]
        target.onset[:] = issue-1

    folder = OUT/f'matched-{name}-t{timing}-{"quiet" if quiet else "g"+str(group)}'
    wrapper = MatchedContinuation(env, actor, issue)
    with patch.object(prior, 'configure', configure):
        report = prior.rollout(env, wrapper, folder, group=group, quiet=quiet)
    assert wrapper.restored
    reference = PILOT/('baseline-quiet' if quiet else f'baseline-t{timing}-g{group}')
    current, baseline = arrays(folder/'trace.npz'), arrays(reference/'trace.npz')
    end = len(current['time']) if name == 'baseline' else issue
    assert all(np.array_equal(current[key][:end], baseline[key][:end]) for key in current)
    save(folder/'descriptor.json', dict(mode='matched', baseline=str(reference), issue_frame=issue,
        prefix_verified=True, entire_trace_exact=name == 'baseline'))
    return report


def checks(rig, anchor, initial):
    old = torch.load(PILOT/'fit-921009'/'initial.pt', weights_only=True, map_location='cpu')
    assert all(torch.equal(value, old[key]) for key, value in initial.state_dict().items())
    env = RecoveryStartTraining(rig, anchor)
    env.reset(rng=torch.Generator().manual_seed(940009))
    assert not initial(env.features()).any()
    active = torch.zeros(16, dtype=torch.bool)
    active[prior.TRAIN] = True
    env.step(torch.full((16, 1), .2), active)
    before = RecoveryStart.capture(env).arrays()
    ids = torch.tensor([0, 9])
    other = [i for i in range(16) if i not in ids.tolist()]
    env.reset(ids, torch.Generator().manual_seed(950009))
    after = RecoveryStart.capture(env).arrays()
    for key in before:
        axis = 1 if key == 'pending_oldest_first' else 0
        assert np.array_equal(np.take(before[key], other, axis=axis), np.take(after[key], other, axis=axis)), key
    assert env.collected == 12 and env.exact_transfers == 14
    assert all(r['start_frame'] == r['onset_frame']+1 and r['pose'] < 6 for r in env.reset_records)
    return dict(passed=True, identical_previous_initial_actor=True, asynchronous_restores=True,
        unselected_worlds_unchanged=True, exact_feature_transfers=env.exact_transfers,
        prefix_excluded_from_collection=True, collected_check_transitions=env.collected,
        prefix_check_intervals=env.prefix_intervals)


_ORIGINAL_ROLLOUT = prior.rollout


def audited_rollout(env, actor, folder, **kwargs):
    if not kwargs.get('sampled', False):
        return _ORIGINAL_ROLLOUT(env, actor, folder, **kwargs)
    assert prior.STD == HALF_SIGMA
    means, features = [], []
    def recorded(x):
        result = actor(x)
        means.append(result.detach().numpy().copy())
        if len(features) < 64:
            features.append(x.detach().numpy().copy())
        return result
    report = _ORIGINAL_ROLLOUT(env, recorded, folder, **kwargs)
    z = arrays(folder/'trace.npz')
    rng = torch.Generator().manual_seed(700009)
    noise = torch.stack([HALF_SIGMA*torch.randn((16, 1), generator=rng) for _ in means]).numpy()
    assert np.array_equal(z['latent'], np.stack(means)+noise)
    np.savez_compressed(folder/'sampled-policy.npz', mean=np.stack(means), features64=np.stack(features),
                        sigma=np.array(HALF_SIGMA))
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
        FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_AGE_WEIGHTED_LEARNING_PLAN.md'), PILOT/'completed.json', PILOT/'audit.json',
        PILOT/'fit-921009'/'initial.pt', Path('logs/mimickit-training/recovery-starts-20261009-01/audit.json')]
    paths += list(PILOT.glob('baseline-*/trace.npz'))
    paths += list(PILOT.glob('baseline-*/report.json'))
    assert json.loads((FULL/'audit.json').read_text())['passed']
    full_done = json.loads((FULL/'completed.json').read_text())
    for relative, sha in full_done['outputs'].items():
        assert digest(FULL/relative) == sha, relative
    paths += [FULL/'completed.json', FULL/'audit.json', *[FULL/r for r in full_done['outputs']],
              Path('logs/mimickit-training/recovery-noise-scale-20261009-01/audit.json')]
    inputs = {str(path): digest(path) for path in paths}
    completed = json.loads((PILOT/'completed.json').read_text())
    assert json.loads((PILOT/'audit.json').read_text())['passed']
    for path in paths:
        if path.is_relative_to(PILOT) and str(path.relative_to(PILOT)) in completed['outputs']:
            assert digest(path) == completed['outputs'][str(path.relative_to(PILOT))]
    save(OUT/'inputs.json', inputs)
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    torch.manual_seed(920009)
    initial = AnkleHead().eval()
    start = time.perf_counter()
    save(OUT/'checks.json', dict(**checks(rig, anchor, initial), probability=probability_checks(), weighting=weighting_checks()))
    frozen_anchor = {k: v.clone() for k, v in anchor.state_dict().items()}
    env = AnkleRecovery(rig, anchor)
    save(OUT/'contract.json', dict(dt=env.dt, ankles=env.ankles, control=env.task.observation_contract(),
        training_seeds=list(SEEDS), training_onset_frames=ONSET_FRAMES.tolist(), observation_width=427,
        nominal_transitions_per_seed=24576, preimpact_optimizer_transitions=0, reward_changed=False,
        initial_actor_seed=920009, sigma=HALF_SIGMA, baseline_sigma=HALF_SIGMA,
        actor_weighting=weighting_checks(), critic_weighted=False, advantage_normalization_changed=False, cases=720, matched_cases_per_candidate=160, continuous_cases_per_candidate=176))
    for timing in (0, 1):
        matched(rig, anchor, initial, 'baseline', timing)
    matched(rig, anchor, initial, 'baseline', 0, quiet=True)
    results = {}
    for seed in SEEDS:
        training = RecoveryStartTraining(rig, anchor)
        actor = learn(training, initial, OUT/f'fit-{seed}', seed, HALF_SIGMA)
        actor.requires_grad_(False)
        assert training.collected == 24576
        save(OUT/f'fit-{seed}'/'resets.json', training.reset_records)
        save(OUT/f'fit-{seed}'/'start-accounting.json', dict(collected=training.collected,
            source_prefix_intervals=training.prefix_intervals, exact_transfers=training.exact_transfers,
            reset_records=len(training.reset_records), optimizer_prefix_transitions=0))
        del training
        with patch.object(ankle, 'SIGMA', HALF_SIGMA), patch.object(prior, 'rollout', audited_rollout):
            continuous = evaluate(rig, anchor, actor, OUT, str(seed))
        restored = []
        for timing in (0, 1):
            for group in range(4):
                restored.append(matched(rig, anchor, actor, str(seed), timing, group))
            restored.append(matched(rig, anchor, actor, str(seed), timing, quiet=True))
        results[str(seed)] = dict(continuous=continuous, matched={str(g): dict(
            settled=sum(r['settled'] for r in restored if not r['quiet'] and r['group'] == g),
            heldout_settled=sum(c['settled'] and c['heldout'] for r in restored
                if not r['quiet'] and r['group'] == g for c in r['cases'])) for g in range(4)},
            matched_quiet=sum(r['whole_stand'] for r in restored if r['quiet']))
        save(OUT/'summary.json', results)
    assert all(torch.equal(v, anchor.state_dict()[k]) for k, v in frozen_anchor.items())
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-start, cases=720, traces=45,
        total_transitions=49152, frozen_source_unchanged=True, inputs_unchanged=True, results=results,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()

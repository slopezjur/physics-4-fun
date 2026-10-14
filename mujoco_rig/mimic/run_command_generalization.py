"""Frozen common-input and archived-command analysis; no simulation or learning."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .command_generalization_probes import compute
from .command_generalization_traces import analyze
from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-command-generalization-20261009-01')
HIGH = Path('logs/mimickit-training/recovery-retracted-actor-rate-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
DATA = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
RATES = (('lower_rate', LOW), ('higher_rate', HIGH))
SEEDS = (921009, 931009)
COMMON_PROBE_KEYS = ('probe_x', 'probe_branch', 'probe_onset', 'probe_pose', 'probe_quiet_index')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Evidence:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT/'sources').mkdir()
        sources = sorted(Path(__file__).parent.glob('*.py'))
        for path in sources:
            shutil.copy2(path, OUT/'sources'/path.name)
        plan = Path(__file__).with_name('RECOVERY_COMMAND_GENERALIZATION_PLAN.md')
        shutil.copy2(plan, OUT/'sources'/plan.name)
        paths = sources+[plan]
        self.inputs = {str(path): digest(path) for path in paths}
        self.manifests = {}
        for root in (HIGH, LOW, DATA):
            for name in ('completed.json', 'audit.json'):
                path = root/name
                self.inputs[str(path)] = digest(path)
            assert read(root/'audit.json')['passed'] is True
            self.manifests[root] = read(root/'completed.json')['outputs']
            self.path(root, 'inputs.json')

    def path(self, root, relative):
        path = root/relative
        actual = digest(path)
        assert self.manifests[root][str(Path(relative))] == actual, path
        self.inputs[str(path)] = actual
        return path

    def read(self, root, relative):
        path = self.path(root, relative)
        if path.suffix == '.json':
            return read(path)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def finish(self):
        historical = {}
        for root in (HIGH, LOW):
            previous = read(root/'inputs.json')
            shared = sorted(set(previous) & set(self.inputs))
            assert shared and all(previous[path] == self.inputs[path] for path in shared)
            historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared)
        assert all(digest(Path(path)) == sha for path, sha in self.inputs.items())
        save(OUT/'inputs.json', self.inputs)
        save(OUT/'historical-inputs.json', historical)
        return historical


def report_pair(push, quiet, timing, mode, dt):
    assert push['sampled'] is False and quiet['sampled'] is False
    assert push['quiet'] is False and quiet['quiet'] is True
    assert push['group'] == quiet['group'] == 0
    assert push['dt'] == quiet['dt'] == dt and push['seconds'] == 6.
    assert quiet['seconds'] == (60. if mode == 'continuous' else 6.)
    assert len(push['cases']) == len(quiet['cases']) == 16
    for index, (pushed, unpushed) in enumerate(zip(push['cases'], quiet['cases'], strict=True)):
        for item in (pushed, unpushed):
            assert item['case'] == index and item['pose'] == index % 8
            assert item['heldout'] == (index % 8 >= 6)
        assert pushed['force_n'] == ([150., 0., 0.] if index < 8 else [-150., 0., 0.])
        assert abs(pushed['hit_seconds']-(60, 102)[timing]*dt) < 2e-7
        assert unpushed['force_n'] == [0., 0., 0.] and unpushed['hit_seconds'] is None


def main():
    started = time.perf_counter()
    evidence = Evidence()
    torch.set_num_threads(1)
    datasets = {seed: evidence.read(DATA, Path(str(seed))/'dataset.npz') for seed in SEEDS}
    for key in COMMON_PROBE_KEYS:
        assert np.array_equal(datasets[SEEDS[0]][key], datasets[SEEDS[1]][key]), key
    contracts = {rate: evidence.read(root, 'contract.json') for rate, root in RATES}
    assert contracts['higher_rate']['actor_learning_rate'] == 1e-3
    assert contracts['higher_rate']['control_actor_learning_rate'] == 1e-4
    assert contracts['higher_rate']['critic_learning_rate'] == 3e-4
    for key in ('sigma', 'dt', 'ankles', 'control', 'training_seeds', 'initial_actor_seed'):
        assert contracts['lower_rate'][key] == contracts['higher_rate'][key], key
    contract = dict(actor_rates={'lower_rate': 1e-4, 'higher_rate': 1e-3}, seeds=list(SEEDS),
        source_roots={rate: str(root) for rate, root in RATES}, dataset_root=str(DATA),
        common_probe_keys=list(COMMON_PROBE_KEYS), same_inputs_across_seeds=True,
        probe_count=36, quiet_probes=12, pushed_probes=24, probe_onsets=[44, 74], probe_poses=list(range(6)),
        probe_memory_zero=True, latent_sign_band=1e-7, sigma=contracts['higher_rate']['sigma'],
        dt=contracts['higher_rate']['dt'], ankles=contracts['higher_rate']['ankles'],
        rho=.9, action_radius=.1, action_slew=.02, modes=['continuous', 'matched'],
        onset_frames=[60, 102], issue_frames=[61, 103], window_intervals=30, case_slots=16,
        trace_poses=list(range(8)), reserved_poses=[6, 7], window_comparisons=16,
        paired_case_windows=256, paired_sample_rows=7680, unique_trace_sources=28,
        same_input_actor_evaluations=144, physical_rollouts=0, optimizer_updates=0,
        critic_updates=0, new_checkpoints=0, reserved_cases_remain_diagnostic=True,
        reference_polarity='Historical weighted-policy local continuations: +X negative; -X positive.',
        trajectory_interpretation='Own-trajectory command decomposition, not causal memory removal or same-input derivatives.',
        requested_addition_interpretation='Scalar correction after tanh/slew, before parent nominal target and joint-limit clipping.')
    save(OUT/'contract.json', contract)
    results, trace_paths = {}, set()
    for rate, root in RATES:
        results[rate] = {}
        for seed in SEEDS:
            folder = OUT/f'{rate}-{seed}'
            folder.mkdir()
            checkpoint = evidence.path(root, Path(f'fit-{seed}')/'final.pt')
            state = torch.load(checkpoint, weights_only=True, map_location='cpu')
            actor = AnkleHead().eval().requires_grad_(False)
            actor.load_state_dict(state)
            frozen = {key: value.clone() for key, value in actor.state_dict().items()}
            with torch.inference_mode():
                arrays, probes = compute(actor, datasets[seed])
            assert all(torch.equal(value, actor.state_dict()[key]) for key, value in frozen.items())
            probes = dict(probes, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                          frozen_parameters_unchanged=True)
            np.savez_compressed(folder/'probes.npz', **arrays)
            save(folder/'probes.json', probes)
            windows = {}
            for mode in contract['modes']:
                for timing in (0, 1):
                    prefix = 'matched-' if mode == 'matched' else ''
                    push_name = f'{prefix}{seed}-t{timing}-g0'
                    quiet_name = f'matched-{seed}-t{timing}-quiet' if mode == 'matched' else f'{seed}-quiet'
                    push_report = evidence.read(root, Path(push_name)/'report.json')
                    quiet_report = evidence.read(root, Path(quiet_name)/'report.json')
                    report_pair(push_report, quiet_report, timing, mode, contract['dt'])
                    push = evidence.read(root, Path(push_name)/'trace.npz')
                    quiet = evidence.read(root, Path(quiet_name)/'trace.npz')
                    trace_paths.update((str(root/push_name/'trace.npz'), str(root/quiet_name/'trace.npz')))
                    arrays, summary = analyze(push, quiet, timing, mode, seed, rate)
                    summary = dict(summary, pushed_source=str(root/push_name/'trace.npz'),
                                   quiet_source=str(root/quiet_name/'trace.npz'))
                    name = f'{mode}-t{timing}'
                    np.savez_compressed(folder/f'{name}.npz', **arrays)
                    save(folder/f'{name}.json', summary)
                    windows[name] = summary
            results[rate][str(seed)] = dict(probes=probes, windows=windows)
            print('ANALYZED', rate, seed, flush=True)
    assert len(trace_paths) == contract['unique_trace_sources']
    historical = evidence.finish()
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, inputs_unchanged=True,
        frozen_parameters_unchanged=True, common_probe_features_exact=True,
        physical_rollouts=0, optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        actors=4, common_probe_states=36, same_input_actor_evaluations=144,
        unique_trace_sources=len(trace_paths), window_comparisons=16, paired_sample_rows=7680,
        historical_shared_inputs={root: row['count'] for root, row in historical.items()},
        outputs={str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and 'sources' not in path.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

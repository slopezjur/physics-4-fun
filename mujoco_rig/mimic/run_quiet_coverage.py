"""Compare training-only quiet coverage on two fixed historical Adam proposals."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .quiet_coverage_bank import build, ONSETS, PREFIX_INTERVALS
from .quiet_coverage_math import compute
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-quiet-coverage-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
DATA = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
BASE = Path('logs/mimickit-training/recovery-command-generalization-20261009-01')
PILOT = Path('logs/mimickit-training/learned-ankle-20261009-01')
SEEDS = (921009, 931009)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Evidence:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT/'sources').mkdir()
        sources = sorted(Path(__file__).parent.glob('*.py'))
        plan = Path(__file__).with_name('RECOVERY_QUIET_COVERAGE_PLAN.md')
        for path in sources+[plan]:
            shutil.copy2(path, OUT/'sources'/path.name)
        external = Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit/learning')
        paths = sources+[plan, PARENT, PARENT.with_suffix('.amp.pt'),
            FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
            FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
            external/'normalizer.py', external/'base_agent.py']
        self.inputs = {str(path): digest(path) for path in paths}
        for demonstration in read(FOUNDATION/'protocol.json')['demonstrations']:
            path = Path(demonstration['path'])
            actual = digest(path)
            assert actual == demonstration['sha256'], path
            self.inputs[str(path)] = actual
        self.manifests = {}
        for root in (LOW, DATA, BASE, PILOT):
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

    def seal(self):
        historical = {}
        for root in (LOW, BASE):
            previous = read(root/'inputs.json')
            shared = sorted(set(previous) & set(self.inputs))
            assert shared and all(previous[path] == self.inputs[path] for path in shared)
            historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared)
        self.verify()
        save(OUT/'inputs.json', self.inputs)
        save(OUT/'historical-inputs.json', historical)
        return historical

    def verify(self):
        assert all(digest(Path(path)) == sha for path, sha in self.inputs.items())


def main():
    started = time.perf_counter()
    evidence = Evidence()
    torch.set_num_threads(1)
    old_bank = evidence.read(LOW, 'quiet-anchors.npz')
    previous_contract = evidence.read(LOW, 'contract.json')
    reference = evidence.read(PILOT, Path('baseline-quiet')/'trace.npz')
    reference_report = evidence.read(PILOT, Path('baseline-quiet')/'report.json')
    assert reference_report['quiet'] and not reference_report['sampled']
    assert reference_report['seconds'] == 60. and reference_report['dt'] == previous_contract['dt']
    states, batches, objectives, records, probes = {}, {}, {}, {}, {}
    for seed in SEEDS:
        folder = Path(f'fit-{seed}')
        states[seed] = torch.load(evidence.path(LOW, folder/'initial.pt'), weights_only=True, map_location='cpu')
        batches[seed] = evidence.read(LOW, folder/'collection-1.npz')
        objectives[seed] = evidence.read(LOW, folder/'objective-1.npz')
        records[seed] = evidence.read(LOW, folder/'projection-1.json')[0]
        probes[seed] = evidence.read(DATA, Path(str(seed))/'dataset.npz')
    assert states[SEEDS[0]].keys() == states[SEEDS[1]].keys()
    assert all(torch.equal(value, states[SEEDS[1]][key]) for key, value in states[SEEDS[0]].items())
    for key in ('probe_x', 'probe_branch', 'probe_onset', 'probe_pose', 'probe_quiet_index'):
        assert np.array_equal(probes[SEEDS[0]][key], probes[SEEDS[1]][key]), key
    historical = evidence.seal()
    contract = dict(seeds=list(SEEDS), source_root=str(LOW), dataset_root=str(DATA),
        baseline_root=str(PILOT), actor_learning_rate=1e-4, actor_gradient_clip=.5,
        initial_actor_seed=920009, sigma=previous_contract['sigma'], dt=previous_contract['dt'],
        ankles=previous_contract['ankles'], control=previous_contract['control'],
        original_anchors=12, expanded_anchors=48, poses=list(range(6)), onsets=list(ONSETS),
        original_onsets=[44, 74], excluded_poses=[6, 7], excluded_onsets=[60, 102],
        feature_width=427, zero_action_memory=True, zero_anchor_target=True,
        state_index='onset + 1', singular_relative_cutoff=1e-6,
        anchor_limit=1e-5, kl_limit=.01, finite_corrections_per_candidate=1,
        primary_algebra='float64 direct Jacobian SVD with float32 parameter application',
        independent_algebra='thin QR(J.T), then SVD(R.T)',
        spectrum_atol=1e-10, spectrum_rtol=2e-7, vector_atol=2e-8, vector_rtol=2e-5,
        full_samples=768, minibatch_samples=256, common_pushed_probes=24, latent_sign_band=1e-7,
        prefix_intervals=PREFIX_INTERVALS, native_worlds=16, native_world_intervals=PREFIX_INTERVALS*16,
        physical_rollouts=1, historical_adam_reconstructions=2, candidate_comparisons=4,
        new_ppo_transitions=0, new_learning_updates=0, critic_updates=0, new_checkpoints=0,
        reserved_states_in_bank=False, arena_changed=False)
    save(OUT/'contract.json', contract)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz', read(FOUNDATION/'protocol.json'), PARENT)
    initial = AnkleHead().eval().requires_grad_(False)
    initial.load_state_dict(states[SEEDS[0]])
    bank, prefix, bank_report = build(rig, anchor, initial, old_bank, reference)
    assert bank_report['dt'] == contract['dt']
    np.savez_compressed(OUT/'bank.npz', **bank)
    np.savez_compressed(OUT/'prefix.npz', **prefix)
    save(OUT/'bank-report.json', bank_report)
    print('CAPTURED', bank_report['anchors'], 'quiet states; archived prefix exact', flush=True)
    results = {}
    for seed in SEEDS:
        folder = OUT/str(seed)
        folder.mkdir()
        arrays, report = compute(states[seed], batches[seed], objectives[seed], records[seed], bank, probes[seed], seed)
        np.savez_compressed(folder/'evidence.npz', **arrays)
        save(folder/'report.json', report)
        results[str(seed)] = report
        print('COMPARED', seed, flush=True)
    evidence.verify()
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started,
        inputs_unchanged=True, frozen_source_unchanged=True, original_features_exact=True,
        physical_rollouts=1, prefix_intervals=PREFIX_INTERVALS, native_world_intervals=PREFIX_INTERVALS*16,
        historical_adam_reconstructions=2, candidate_comparisons=4,
        new_ppo_transitions=0, new_learning_updates=0, critic_updates=0, new_checkpoints=0,
        historical_shared_inputs={root: row['count'] for root, row in historical.items()},
        outputs={str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and 'sources' not in path.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

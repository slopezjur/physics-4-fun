"""Reproduce historical half-noise streams and retain only compact diagnostics."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .half_noise_ppo import HALF_SIGMA
from .recovery_start_training import RecoveryStartTraining
from .rig import Rig
from .run_direct_recovery import TRAIN
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .trace_half_noise_ppo import learn
from .train_coordinated_ankle import AnkleHead

BASE = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')
PROBE_BASE = Path('logs/mimickit-training/recovery-exploration-20261009-01')
OUT = Path('logs/mimickit-training/recovery-update-trace-20261009-01')


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(1)
    assert read(BASE/'audit.json')['passed'] and read(PROBE_BASE/'audit.json')['passed']
    inherited = read(BASE/'inputs.json')
    assert all(digest(Path(p)) == sha for p, sha in inherited.items())
    source_dir = Path(__file__).parent
    paths = [source_dir/name for name in ('trace_recovery_updates.py', 'trace_half_noise_ppo.py',
        'recovery_update_observer.py', 'verify_recovery_update_trace.py', 'RECOVERY_UPDATE_TRACE_PLAN.md')]
    paths += [BASE/n for n in ('completed.json', 'audit.json', 'inputs.json')]
    manifest = read(BASE/'completed.json')['outputs']
    for seed in (921009, 931009):
        for name in ('initial.pt', 'final.pt', 'critic-final.pt', 'collection-1.npz', 'collection-32.npz',
                     'resets.json', 'history.json', 'start-accounting.json', 'report.json'):
            path = BASE/f'fit-{seed}'/name
            assert digest(path) == manifest[str(path.relative_to(BASE))]
            paths.append(path)
    paths += [PROBE_BASE/n for n in ('completed.json', 'audit.json')]
    probe_manifest = read(PROBE_BASE/'completed.json')['outputs']
    probe_rows = []
    for onset in (44, 74):
        branches = []
        for branch in ('quiet', 'push'):
            path = PROBE_BASE/f'{onset}-{branch}-zero'/'trace.npz'
            assert digest(path) == probe_manifest[str(path.relative_to(PROBE_BASE))]
            paths.append(path)
            with np.load(path, allow_pickle=False) as z:
                features = z['features64'][:30].copy()
                assert z['alive'][:30].all() and not features[:, :, -2:].any()
                branches.append(features)
        probe_rows.append(np.stack(branches))
    probe_features = np.stack(probe_rows)
    inputs = {**inherited, **{str(p): digest(p) for p in paths}}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    for path in paths[:5]:
        shutil.copy2(path, OUT/'sources'/path.name)
    save(OUT/'inputs.json', inputs)
    np.savez_compressed(OUT/'probes.npz', features=probe_features, worlds=TRAIN.numpy(), onsets=np.array([44, 74]))
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz', read(FOUNDATION/'protocol.json'), PARENT)
    frozen = {k: v.clone() for k, v in anchor.state_dict().items()}
    initial = AnkleHead().eval()
    initial.load_state_dict(torch.load(BASE/'fit-921009'/'initial.pt', weights_only=True, map_location='cpu'))
    start = time.perf_counter()
    for seed in (921009, 931009):
        env = RecoveryStartTraining(rig, anchor)
        learn(env, initial, OUT/f'fit-{seed}', seed, HALF_SIGMA, BASE/f'fit-{seed}', torch.from_numpy(probe_features))
        del env
        print('PARITY', seed, 'exact', flush=True)
    assert all(torch.equal(v, anchor.state_dict()[k]) for k, v in frozen.items())
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert not list(OUT.rglob('*.pt'))
    save(OUT/'completed.json', dict(exact_historical_parity=True, source_unchanged=True,
        training_transitions_replayed=49152, new_candidate_configurations=0, evaluation_rollouts=0,
        seeds=[921009, 931009], updates_per_seed=32, sigma=HALF_SIGMA,
        wall_seconds=time.perf_counter()-start,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

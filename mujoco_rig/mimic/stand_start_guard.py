"""Raw, fail-closed admission evidence for future matched continuation arms."""
from pathlib import Path

import numpy as np

from .audit_stand_failures import digest, read, save
from .diagnose_perturb_reset import snapshot
from .measure_stand_learning_curve import rng_state, training_fingerprint, tree_hash


FOOT_LOAD_COLUMNS = (93, 94)
FOOT_LOAD_ATOL = 1e-6  # Scaled observation units; 0.001 N. No relative tolerance.
COMPONENTS = {'qpos', 'qvel', 'pending', 'active', 'ctrl', 'cursor', 'steps', 'offset', 'observation'}


def capture(folder, agent, task):
    """Call immediately before admission, before any PPO updates in either arm."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/'components.npz', **snapshot(agent, task))
    fingerprint = training_fingerprint(agent, task)
    fingerprint['rng'] = tree_hash(rng_state(task))
    save(folder/'manifest.json', dict(fingerprint=fingerprint,
        components_sha256=digest(folder/'components.npz'),
        foot_load_columns=list(FOOT_LOAD_COLUMNS), foot_load_atol=FOOT_LOAD_ATOL))


def compare(left, right):
    """Compare saved starts; a differing environment hash alone is not admissible."""
    folders = [Path(left), Path(right)]
    manifests = [read(p/'manifest.json') for p in folders]
    arrays = []
    for folder, manifest in zip(folders, manifests):
        if (digest(folder/'components.npz') != manifest['components_sha256']
                or manifest['foot_load_columns'] != list(FOOT_LOAD_COLUMNS)
                or manifest['foot_load_atol'] != FOOT_LOAD_ATOL):
            raise ValueError('Start evidence or predeclared tolerance changed')
        with np.load(folder/'components.npz', allow_pickle=False) as saved:
            arrays.append({key: saved[key] for key in saved.files})
    if any(set(value) != COMPONENTS for value in arrays):
        raise ValueError('Incomplete raw training-start components')
    left_hash, right_hash = (m['fingerprint'] for m in manifests)
    exact = {key: left_hash[key] == right_hash[key]
             for key in ('learner', 'optimizers', 'rng', 'training_mode')}
    components = {}
    for key in sorted(COMPONENTS):
        a, b = arrays[0][key], arrays[1][key]
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Invalid training-start component: {key}')
        identical = a.tobytes() == b.tobytes()
        passed = identical
        if key == 'observation':
            if a.ndim != 2 or a.shape[1] != 374:
                raise ValueError('Expected the 374-observation contract')
            strict = np.ones(374, dtype=bool)
            strict[list(FOOT_LOAD_COLUMNS)] = False
            passed = (a[:, strict].tobytes() == b[:, strict].tobytes()
                      and np.allclose(a[:, FOOT_LOAD_COLUMNS], b[:, FOOT_LOAD_COLUMNS],
                                      rtol=0., atol=FOOT_LOAD_ATOL))
        components[key] = dict(passed=bool(passed), bitwise_equal=identical,
            maximum_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
    return dict(passed=all(exact.values()) and all(c['passed'] for c in components.values()),
        exact_fingerprints=exact, components=components, foot_load_atol=FOOT_LOAD_ATOL,
        environment_hash_equal=left_hash['environment'] == right_hash['environment'])

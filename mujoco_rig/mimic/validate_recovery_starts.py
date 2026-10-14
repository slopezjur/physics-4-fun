"""Admit exact recovery starts against sealed continuous source trajectories."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .recovery_start_bank import RecoveryStart
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleRecovery, OUT as PILOT

OUT = Path('logs/mimickit-training/recovery-starts-20261009-01')
PROBES = Path('logs/mimickit-training/direct-leg-response-20261009-01')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


@torch.no_grad()
def advance(env, frame, active, latent=None):
    assert (env.task.steps == frame).all()
    latent = torch.zeros(16, 1) if latent is None else latent
    reward, terminal, info = env.step(latent, active)
    row = dict(time=np.array((frame+1)*env.dt), alive=active.numpy().copy(), reward=reward.numpy().copy(),
               **{key: value.numpy().copy() for key, value in info.items()})
    return row, active & ~terminal


def preflight(env, start):
    """Test selected-world restoration with the other ring slot currently oldest."""
    original = start.arrays()
    features = env.features().clone()
    start.restore(env)
    assert torch.equal(features, env.features())
    # The privileged schedule can change without changing policy inputs.
    env.force.fill_(321.)
    env.onset.add_(19)
    assert torch.equal(features, env.features())
    start.restore(env)
    for _ in range(3):
        env.step(torch.full((16, 1), .4), torch.ones(16, dtype=torch.bool))
    changed = RecoveryStart.capture(env).arrays()
    ids = torch.tensor([0, 8])
    other = torch.tensor([i for i in range(16) if i not in (0, 8)])
    cursor = env.engine.policy_control.cursor
    start.restore(env, ids)
    assert env.engine.policy_control.cursor == cursor
    restored = RecoveryStart.capture(env).arrays()
    for key, value in original.items():
        axis = 1 if key == 'pending_oldest_first' else 0
        assert np.array_equal(np.take(restored[key], ids, axis=axis), np.take(value, ids, axis=axis)), key
        assert np.array_equal(np.take(restored[key], other, axis=axis), np.take(changed[key], other, axis=axis)), key
    assert torch.equal(env.features()[ids], features[ids])
    # A new full restore must reproduce the next four intervals of an independent copy.
    clone = AnkleRecovery(env.rig, env.anchor)
    start.restore(clone)
    start.restore(env)
    for frame in range(4):
        latent = torch.full((16, 1), .3 if frame % 2 else -.2)
        a = env.step(latent, torch.ones(16, dtype=torch.bool))
        b = clone.step(latent, torch.ones(16, dtype=torch.bool))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert all(torch.equal(a[2][key], b[2][key]) for key in a[2])
    assert all(np.array_equal(value, start.arrays()[key]) for key, value in original.items())
    start.restore(env)
    return dict(passed=True, partial_worlds=[0, 8], unrelated_worlds_unchanged=True,
        odd_cursor_shift=True, independent_copy_intervals=4, snapshot_immutable=True,
        force_and_timer_excluded=True, exact_features=True)


def report_trace(env, rows, folder, *, quiet, onset, amplitude, timing):
    z = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert not z['warning'].any()
    xy, clearance = foot_geometry(env.rig.model, z['qpos'])
    assert np.allclose(xy, z['xy'], rtol=0, atol=2e-6)
    assert np.allclose(clearance, z['clearance'], rtol=0, atol=2e-6)
    hit = int(np.ceil(onset/env.dt))
    cases = []
    for i in range(16):
        alive = z['alive'][:, i] & ~z['failed'][:, i]
        survived = bool(alive.all())
        settled = bool(survived and z['settled'][-int(np.ceil(2/env.dt)):, i].all())
        events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive,
                               z['settled'][:, i], hit_time=None if quiet else hit*env.dt)
        cases.append(dict(case=i, pose=i % 8, heldout=i % 8 >= 6, survived=survived, settled=settled,
            whole_stand=bool(survived and z['stand'][:, i].all()), steps=events,
            recovered_with_step=settled and any(e['settled_window_start'] is not None for e in events)))
    folder.mkdir()
    np.savez_compressed(folder/'trace.npz', **z)
    reference = (PILOT/'baseline-quiet' if quiet and amplitude == 0 else
                 PILOT/f'baseline-t{timing}-g0' if amplitude == 0 else
                 PROBES/f'probe-ankles-a{0 if amplitude < 0 else 3}-t{timing}' if not quiet else None)
    verified_fields = []
    if reference is not None:
        old = arrays(reference/'trace.npz')
        verified_fields = sorted(set(z) & set(old))
        for key in verified_fields:
            assert np.array_equal(z[key], old[key][:len(rows)]), (folder.name, key)
    report = dict(timing=timing, onset=onset, hit_frame=hit, issue_frame=hit+1, quiet=quiet,
        amplitude=amplitude, dt=env.dt, cases=cases, settled=sum(c['settled'] for c in cases),
        survived=sum(c['survived'] for c in cases), whole_stand=sum(c['whole_stand'] for c in cases),
        measured_steps=sum(len(c['steps']) for c in cases), reference=str(reference) if reference else None,
        exact_archived_fields=verified_fields)
    save(folder/'report.json', report)
    print('EVAL', folder.name, report['settled'], report['survived'], report['measured_steps'], flush=True)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py'))
    for p in paths:
        shutil.copy2(p, OUT/'sources'/p.name)
    paths += [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
              FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
              Path(__file__).with_name('RECOVERY_START_PLAN.md'), PILOT/'audit.json', PILOT/'completed.json',
              PROBES/'audit.json', PROBES/'completed.json', PILOT/'baseline-quiet'/'trace.npz']
    for timing in (0, 1):
        paths += [PILOT/f'baseline-t{timing}-g0'/'trace.npz']
        paths += [PROBES/f'probe-ankles-a{amplitude}-t{timing}'/'trace.npz' for amplitude in (0, 3)]
    inputs = {str(path): digest(path) for path in paths}
    for root in (PILOT, PROBES):
        assert json.loads((root/'audit.json').read_text())['passed']
        old = json.loads((root/'completed.json').read_text())['outputs']
        for path in paths:
            if path.is_relative_to(root) and str(path.relative_to(root)) in old:
                assert digest(path) == old[str(path.relative_to(root))]
    save(OUT/'inputs.json', inputs)
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    assert not any(p.requires_grad for p in anchor.parameters())
    records, checks = [], []
    started = time.perf_counter()
    for timing, onset in enumerate((1., 1.7)):
        env = AnkleRecovery(rig, anchor)
        hit = int(np.ceil(onset/env.dt))
        env.onset.fill_(hit)
        active = torch.ones(16, dtype=torch.bool)
        prefix = []
        for frame in range(hit):
            row, active = advance(env, frame, active)
            prefix.append(row)
        assert active.all()
        common = RecoveryStart.capture(env)
        np.savez_compressed(OUT/f't{timing}-common.npz', **common.arrays(), features=env.features().numpy())
        checks.append(preflight(env, common))
        for quiet in (True, False):
            common.restore(env)
            env.force.zero_()
            if not quiet:
                env.force[:8, 0], env.force[8:, 0] = 150., -150.
            assert torch.equal(env.features(), torch.from_numpy(arrays(OUT/f't{timing}-common.npz')['features']))
            first, branch_active = advance(env, hit, active)
            assert branch_active.all()
            start = RecoveryStart.capture(env)
            branch = 'quiet' if quiet else 'push'
            np.savez_compressed(OUT/f't{timing}-{branch}-start.npz', **start.arrays(), features=env.features().numpy())
            for label, amplitude in (('zero', 0.), ('negative', -.075), ('positive', .075)):
                start.restore(env)
                rows = prefix + [first]
                running = branch_active.clone()
                for frame in range(hit+1, int(np.ceil(6/env.dt))):
                    latent = torch.atanh(torch.full((16, 1), amplitude if frame < hit+13 else 0.)/.1)
                    row, running = advance(env, frame, running, latent)
                    rows.append(row)
                records.append(report_trace(env, rows, OUT/f't{timing}-{branch}-{label}', quiet=quiet,
                    onset=onset, amplitude=amplitude, timing=timing))
    save(OUT/'checks.json', checks)
    save(OUT/'summary.json', [{k: v for k, v in report.items() if k != 'cases'} for report in records])
    save(OUT/'contract.json', dict(dt=env.dt, ankles=env.ankles, control=env.task.observation_contract(),
        prefix_optimizer_transitions=0, total_optimizer_transitions=0, cases=192,
        start='after first force interval; remaining five intervals preserved',
        snapshot='full native MjData plus explicit task/control/history state', training_poses=list(range(6))))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, cases=192, rollouts=12,
        optimizer_updates=0, inputs_unchanged=True,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

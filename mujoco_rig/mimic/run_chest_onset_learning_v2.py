"""Matched onset continuations with every pose cycling all prospective launch times."""
import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from . import ball_study_support as support
from .chest_ball_curriculum import curriculum
from .measure_stand_learning_curve import tree_hash
from .prepare_stand_ball_pair import prepare_pair
from .replicate_stand_representations import cpu
from .rig import Rig
from .runtime import activate
from . import run_stand_ball_pilot as pilot
from .stand_experiment_evidence import seal
from .stand_start_guard import compare


@contextmanager
def onset_curriculum(settings, schedule):
    # Reuse the sealed launch formula, changing only the prospective per-world time.
    base_settings = dict(settings, launch_seconds=5, seconds=12)
    with curriculum(base_settings):
        from . import stand_ball_task
        original_task = stand_ball_task.StandBallAMPTask
        original_evidence = pilot.TrainingEvidence

        class OnsetTask(original_task):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.episode_seconds = settings['episode_seconds']
                self.onset_frames = torch.tensor([int(np.ceil(t / self.dt)) for t in schedule], device=self.device)
                self.schedule_phase = torch.full((64,), -1, device=self.device, dtype=torch.long)
                self.launch_step = self.onset_frames[((self.ids // 8) % 2) % len(schedule)]

            def reset(self, env_ids=None):
                result = super().reset(env_ids)
                if hasattr(self, 'schedule_phase'):
                    ids = self.ids if env_ids is None else env_ids
                    self.schedule_phase[ids] += 1
                    indexes = ((ids // 8) % 2 + self.schedule_phase[ids]) % len(schedule)
                    self.launch_step[ids] = self.onset_frames[indexes]
                return result

        class OnsetEvidence(original_evidence):
            def step(self, action):
                frames, phase = self.task.launch_step.clone(), self.task.schedule_phase.clone()
                result = super().step(action)
                self.rows[-1].update(scheduled_launch_step=frames, schedule_phase=phase)
                return result

            def flush(self, path):
                if len(self.rows) != 64:
                    raise RuntimeError('Expected exactly one rollout')
                values = {k: cpu(torch.stack([r[k] for r in self.rows])) for k in self.rows[0]}
                self.rows.clear()
                np.savez_compressed(path, **values)
                with np.load(path, allow_pickle=False) as saved:
                    values = {k: saved[k] for k in saved.files}
                assert all(np.isfinite(v).all() for v in values.values())
                indexes = ((np.arange(64)[None] // 8) % 2 + values['schedule_phase']) % len(schedule)
                assert np.array_equal(values['scheduled_launch_step'], cpu(self.task.onset_frames)[indexes])
                expected = (values['steps'] == values['scheduled_launch_step']) & (np.arange(64)[None] < 32)
                if self.task.arm == 'quiet':
                    expected[:] = False
                assert np.array_equal(values['launched'], expected)
                assert (values['position_error'] <= 2e-6).all() and (values['velocity_error'] <= 2e-6).all()
                assert (values['normal_impulse'] >= 0).all()
                quiet = np.arange(64) >= (0 if self.task.arm == 'quiet' else 32)
                assert not values['normal_impulse'][:, quiet].any()
                assert not values['contact_impulse'][:, quiet].any()
                assert ((values['steps'][1:] == values['steps'][:-1] + 1) | (values['steps'][1:] == 0)).all()
                return dict(launches=int(values['launched'].sum()),
                            contact_world_intervals=int(np.count_nonzero(values['normal_impulse'] > 0)),
                            task_reward_mean=float(values['task_reward'].mean()))

        with patch.object(stand_ball_task, 'StandBallAMPTask', OnsetTask), patch.object(pilot, 'TrainingEvidence', OnsetEvidence):
            yield


def run(args):
    s = read(args.protocol)
    assert __debug__ and s['schema'] == 'chest_onset_learning_v1'
    assert s['seeds'] == [301027, 301028] and s['actors'] == ['candidate']
    assert s['arms'] == ['quiet', 'fixed', 'varied']
    assert s['envs'] == 64 and s['rollout_steps'] == 64 and s['quiet_fraction'] == .5
    assert s['training_mass_kg'] == 5 and s['launch_distance_m'] == .25
    assert .8 <= s['training_speed_m_s'] <= 1.4 and 0 < s['updates_per_arm'] <= 200
    assert s['maximum_updates'] == 6 * s['updates_per_arm']
    assert s['maximum_transitions'] == s['maximum_updates'] * 4096
    assert 0 < s['maximum_training_seconds_per_arm'] <= 600
    assert s['fixed_onsets'] == [5] and s['varied_onsets'] == [1, 3, 5, 9]
    assert s['episode_seconds'] == 16 and s['checkpoint_interval'] in (50, 100)
    assert 0 <= s['optimization_seed_offset'] <= 100000
    inputs_extra = []
    for item in s['checkpoints'].values():
        root, model = Path(item['root']), Path(item['checkpoint'])
        proof = read(root / 'final-verification.json')
        hashes = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
        assert proof['verified']
        for p in (model, model.with_suffix('.amp.pt')):
            assert hashes.get(str(p.resolve())) == digest(p)
        inputs_extra.append(str(root / 'final-verification.json'))
    assert all(p.replace('\\', '/') in [a.replace('\\', '/') for a in s['evidence']] for p in inputs_extra)
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    dependencies = [Path(__file__), Path(__file__).with_name('chest_ball_curriculum.py')]
    with patch.object(support, 'checkpoint', lambda seed, _: Path(s['checkpoints'][str(seed)]['checkpoint'])):
        parent, inputs, sources = support.protect_used(args.out, dependencies, args.protocol, s)
    assert parent['config']['steps_per_iter'] == 64 and parent['config']['actor_epochs'] == 3
    save(args.out / 'protocol.json', dict(settings=s, inputs=inputs, sources=sources,
        config=parent['config'], demonstrations=parent['demonstrations'],
        optimizer_state='Fresh per arm; identical full parent weights and learned normalizers.',
        evaluation_cases=0, automatic_promotion=False,
        scope='Training-only component; no behavioral conclusion without subsequent frozen assessments.'))
    rig, reference = Rig.load(args.out / 'dummy.xml', policy_exclude=()), args.out / 'stationary_reference.npz'
    world = args.out / 'ball-5kg.xml'
    write_world(args.out / 'dummy.xml', support.FOUNDATION / 'ball-5kg.xml', world, 5)
    results = {}
    for seed in s['seeds']:
        folder = args.out / str(seed)
        folder.mkdir()
        model = Path(s['checkpoints'][str(seed)]['checkpoint'])
        actual_seed = seed + s['optimization_seed_offset']
        results[str(seed)] = {}
        for variant, schedule in (('fixed', s['fixed_onsets']), ('varied', s['varied_onsets'])):
            with onset_curriculum(s, schedule):
                starts = folder / f'{variant}-starts'
                pair = prepare_pair(rig, reference, world, parent, model, actual_seed, starts)
                save(starts / 'schedule.json', dict(onsets_seconds=schedule, schedule_rule='Each pose/sign cycles every onset across resets; replica offset=(world//8)%2.',
                    schedule_by_world=cpu(pair['single_ball'][1].launch_step).tolist(),
                    episode_seconds=s['episode_seconds'], optimization_seed=actual_seed,
                    rng_hash=tree_hash(pair['single_ball'][2])))
                if variant == 'varied':
                    matched = compare(folder / 'fixed-starts' / 'single_ball', starts / 'single_ball')
                    assert matched['passed']
                    with np.load(folder / 'fixed-starts' / 'single_ball' / 'extra-state.npz', allow_pickle=False) as a, np.load(starts / 'single_ball' / 'extra-state.npz', allow_pickle=False) as b:
                        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
                    assert read(folder / 'fixed-starts' / 'schedule.json')['rng_hash'] == tree_hash(pair['single_ball'][2])
                    save(starts / 'cross_schedule-admission.json', matched)
                order = ('quiet', 'single_ball') if seed == 301027 else ('single_ball', 'quiet')
                for arm in order if variant == 'fixed' else ('single_ball',):
                    label = 'quiet' if arm == 'quiet' else variant
                    results[str(seed)][label] = pilot.train_arm(pair, arm, seed, starts, folder / label, s)
                    save(args.out / 'progress.json', dict(results=results, current_parent=seed, current_arm=label))
                del pair
                gc.collect()
                torch.cuda.empty_cache()
    updates = sum(a['updates'] for r in results.values() for a in r.values())
    assert updates == s['maximum_updates']
    save(args.out / 'report.json', dict(results=results, training_updates=updates,
        transitions=updates * 4096, evaluation_cases=0, promoted=False))
    seal(args.out, inputs, sources, verified=True, training_updates=updates,
         evaluation_cases=0, promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = p.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc(), promoted=False))
        raise

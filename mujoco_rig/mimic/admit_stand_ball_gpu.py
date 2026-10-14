"""Bounded GPU admission and zero-update paired starts; never trains an actor."""
import argparse
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .evaluate_stand_ball_gpu import evaluate, compare
from .prepare_stand_ball_pair import prepare_pair
from .replicate_stand_representations import cpu, read_state
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor


def probe_partial_reset(rig, reference, ball_path, out):
    from .stand_ball_task import StandBallGPUTask
    task = StandBallGPUTask(rig, reference, ball_path=ball_path, seconds=12.)
    for _ in range(3):
        task.step(torch.zeros((8, 33), device=task.device))
    engine, control = task.engine, task.engine.policy_control

    def snapshot():
        q, v = read_state(task, 'gpu')
        bq, bv = (cpu(t) for t in engine.ball_state())
        return dict(qpos=q, qvel=v, ball_qpos=bq, ball_qvel=bv,
            pending=cpu(control.pending).transpose(1, 0, 2), active=cpu(control.active),
            ctrl=cpu(engine.native_ctrl), observation=cpu(task.observations()))

    before = snapshot()
    ids = torch.tensor([0, 4], device=task.device)
    positions = engine.get_body_pos(0)[ids, rig.model.body('Chest').id-1].clone()
    positions[:, 0] -= .65
    engine.launch_ball(ids, positions, [1., 0., 3.18825])
    launched = snapshot()
    peers = np.ones(8, dtype=bool)
    peers[[0, 4]] = False
    launch_checks = {key: np.array_equal(before[key][peers], launched[key][peers]) for key in before}
    for key in ('qpos', 'qvel', 'pending', 'active', 'ctrl', 'observation'):
        launch_checks[f'all_{key}'] = np.array_equal(before[key], launched[key])
    task.reset(ids)
    reset = snapshot()
    reset_checks = {key: np.array_equal(launched[key][peers], reset[key][peers]) for key in launched}
    q, v = task.sample_reference(torch.zeros(2, device=task.device), ids)
    reset_checks['selected_pose'] = np.allclose(reset['qpos'][[0, 4]], cpu(q), rtol=0, atol=2e-6)
    reset_checks['selected_velocity'] = np.allclose(reset['qvel'][[0, 4]], cpu(v), rtol=0, atol=2e-6)
    reset_checks['selected_commands_cleared'] = all(not reset[k][[0, 4]].any() for k in ('pending', 'active', 'ctrl'))
    reset_checks['selected_ball_parked'] = np.allclose(reset['ball_qpos'][[0, 4]], engine.rig.rest[46:], rtol=0, atol=2e-6)
    reset_checks['selected_ball_stopped'] = not reset['ball_qvel'][[0, 4]].any()
    out.mkdir()
    for name, values in (('before', before), ('launched', launched), ('reset', reset)):
        np.savez_compressed(out/f'{name}.npz', **values)
    passed = all(launch_checks.values()) and all(reset_checks.values())
    save(out/'report.json', dict(passed=bool(passed), launch_checks=launch_checks,
        reset_checks={k: bool(v) for k, v in reset_checks.items()}, control_intervals_per_world=3,
        worlds=8, training_updates=0))
    if not passed:
        raise RuntimeError('Projectile launch/partial-reset isolation failed')


def run(args):
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    prior = args.prior
    boundary, parent = read(prior/'report.json'), read(prior/'protocol.json')
    if boundary['native_boundary_candidate'] != '5kg-1mps' or not read(prior/'final-verification.json')['verified']:
        raise ValueError('Requires the verified 5 kg / 1 m/s native boundary')
    settings = parent['settings']
    package = Path(__file__).parent
    files = [package/name for name in ('admit_stand_ball_gpu.py', 'evaluate_stand_ball_gpu.py',
        'stand_ball_gpu.py', 'stand_ball_task.py', 'prepare_stand_ball_pair.py', 'stand_start_guard.py',
        'scripts/admit-stand-ball-gpu.ps1')]
    inputs, sources = protect(prior, args.out, files)
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=parent['config'], demonstrations=parent['demonstrations'],
        model_sha256=parent['model_sha256'], reference_sha256=parent['reference_sha256'],
        maximum_evaluation_cases=80, maximum_training_updates=0,
        reset_probe=dict(worlds=8, control_intervals_per_world=3),
        start_probe=dict(seeds=2, arms_per_seed=2, worlds_per_arm=64, physics_steps=0),
        admission=dict(outcomes='Identical case-by-case recovery, survival and contact flags',
            contact_timing_frames=2, normal_impulse_atol_ns=.25, normal_impulse_rtol=.10),
        limitation='Admission prepares live pairs but does not persist a resumable training environment'))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    worlds = {}
    for mass in (1, 2, 5):
        path = args.out/f'ball-{mass}kg.xml'
        shutil.copy2(prior/path.name, path)
        worlds[mass] = path
    probe_partial_reset(rig, reference, worlds[5], args.out/'reset-probe')
    results, comparisons = {}, {}
    for seed in settings['seeds']:
        source_root = Path(settings['sources'][str(seed)])
        source = read(source_root/'protocol.json')
        checkpoint = source_root/'final/model.pt'
        expected = {str(Path(p).resolve()): h for p, h in inputs.items()}
        for path in (checkpoint, checkpoint.with_suffix('.amp.pt')):
            if expected.get(str(path.resolve())) != digest(path):
                raise ValueError('Source checkpoint differs from boundary evidence')
        actor = load_actor(rig, reference, source, checkpoint)
        cases = [('quiet', 1, 0., 1), ('2kg-2mps-+1', 2, 2., 1), ('2kg-2mps--1', 2, 2., -1),
                 ('5kg-1mps-+1', 5, 1., 1), ('5kg-1mps--1', 5, 1., -1)]
        for label, mass, speed, sign in cases:
            name = f'{seed}-{label}'
            result = evaluate(rig, reference, actor, worlds[mass], args.out/name, settings, speed, sign)
            baseline = read(prior/name/'report.json')
            results[name] = result
            comparisons[name] = compare(baseline, result, rig.model.opt.timestep*rig.decimation)
            save(args.out/'comparison-progress.json', comparisons)
        del actor
    behavior_passed = all(c['passed'] for c in comparisons.values())
    save(args.out/'report.json', dict(results=results, comparisons=comparisons,
        behavioral_admission_passed=behavior_passed, matched_starts_passed=False,
        training_updates=0, promoted=False))
    if not behavior_passed:
        raise RuntimeError('GPU/native ball admission differs; inspect saved comparisons before training')
    for seed in settings['seeds']:
        root = Path(settings['sources'][str(seed)])
        prepared = prepare_pair(rig, reference, worlds[5], read(root/'protocol.json'),
            root/'final/model.pt', seed, args.out/f'{seed}-starts')
        del prepared
    save(args.out/'report.json', dict(results=results, comparisons=comparisons,
        behavioral_admission_passed=True, matched_starts_passed=True,
        training_updates=0, promoted=False,
        next_gate='Build the matched trainer using prepare_pair again immediately before learning; retain 120-second Stand'))
    seal(args.out, inputs, sources, verified=True, gpu_ball_admitted=True,
        matched_start_probe_passed=True, training_updates=0, promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, required=True)
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc(), training_updates=0))
        raise

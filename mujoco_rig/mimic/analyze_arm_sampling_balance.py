"""Trace first command/physical divergence and compare same-state predictions."""
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_sampling_balance import OUT, SEEDS
from .train_arm_corrective_comparison import TAKEOVER, TEACHER
from .probe_arm_execution_distribution import setup
from .train_arm_relative_proposal import load_head
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest, save


def first(mask, times):
    ids = np.flatnonzero(mask)
    return None if not len(ids) else dict(frame=int(ids[0]), seconds=float(times[ids[0]]))


@torch.no_grad()
def main():
    report = json.loads((OUT/'report.json').read_text())
    assert json.loads((OUT/'audit.json').read_text())['passed']
    env, anchor, _, poses, _ = setup()
    original = dict(np.load(TEACHER/'training-samples.npz'))
    correction = dict(np.load(TAKEOVER/'training-samples.npz'))
    groups = {'corrections': correction}
    for heldout in (False, True):
        for quiet in (False, True):
            use = (original['validation'] == heldout) & (original['quiet'] == quiet)
            key = ('heldout' if heldout else 'train')+('-quiet' if quiet else '-active')
            groups[key] = {k: v[use] for k, v in original.items()}
    paths = [OUT/'report.json', OUT/'audit.json', Path(__file__),
             TEACHER/'training-samples.npz', TAKEOVER/'training-samples.npz']
    regression, comparisons = [], []
    for seed in SEEDS:
        for recipe in report['scores']:
            folder = OUT/f'{recipe}-{seed}'
            paths += [folder/'final.pt', folder/'architecture.json']
            net = load_head(folder)
            for name, data in groups.items():
                error = net(torch.tensor(data['x']))-torch.tensor(data['raw_target']-data['goal'])
                regression.append(dict(seed=seed, recipe=recipe, subset=name, rows=len(error),
                    rmse_rad=float(error.square().mean().sqrt()), normalized_mse=float((error/net.output_scale).square().mean())))
            for variant in (False, True):
                suffix = '-varied' if variant else '-nominal'
                folder = OUT/(f'{recipe}-{seed}'+suffix)
                control_path = OUT/(f'replay-{seed}'+suffix)/'trace.npz'
                label_path = TAKEOVER/(f'all-{seed}'+suffix+'-control')/'labels.npz'
                paths += [folder/'trace.npz', control_path, label_path]
                z, baseline, labels = (np.load(p) for p in (folder/'trace.npz', control_path, label_path))
                adapter = CorrectionEnvironment(env, anchor, net, poses, variant=variant)
                p, c = adapter.physical, env.engine.policy_control
                frames, worlds, _ = labels['x'].shape
                residual = net(torch.tensor(labels['x'].reshape(-1, 421))).reshape(frames, worlds, 33)
                raw = torch.tensor(labels['goal'])+residual
                source, previous = (torch.tensor(labels[k]) for k in ('source', 'previous'))
                fade = ((torch.tensor(labels['goal'])-p.reference[:8, p.qi]).abs().amax(-1)/.05).clamp(0, 1)
                desired = (raw-source).clamp(-1, 1)*p.authority[:-1, None, None]*fade[..., None]
                addition = previous+(desired-previous).clamp(-.02, .02)
                addition[torch.tensor(labels['quiet'] | ~baseline['alive'])] = 0
                projected = (source+addition).clamp(c.lower, c.upper).numpy()
                if recipe == 'replay':
                    # A larger inference batch can have tiny floating-point differences.
                    assert np.max(abs(projected-baseline['issued'])) < 2e-6
                details = []
                for i in range(8):
                    issued = abs(z['issued'][:, i]-baseline['issued'][:, i]).max(-1)
                    same = abs(projected[:, i]-baseline['issued'][:, i]).max(-1)
                    motion = abs(z['qpos'][:, i, p.qi]-baseline['qpos'][:, i, p.qi]).max(-1)
                    com_delta = np.linalg.norm(z['com'][:, i, :2]-baseline['com'][:, i, :2], axis=-1)
                    motor = np.sqrt(np.mean((z['qpos'][:, i, p.qi]-p.goals[:-1, i, p.qi].numpy())**2, -1))
                    com = np.linalg.norm(z['com'][:, i, :2]-p.target_com[:-1, i, :2].numpy(), axis=-1)
                    onset = z['time'] > adapter.onset
                    common_alive = z['alive'][:, i] & baseline['alive'][:, i]
                    ramp = onset & (z['time'] <= adapter.onset+adapter.ramp) & common_alive
                    delta = z['issued'][ramp, i]-baseline['issued'][ramp, i]
                    rms = np.sqrt(np.mean(delta**2, axis=0))
                    order = np.argsort(rms)[-3:][::-1]
                    names = [env.rig.model.joint(int(j)).name for j in env.rig.model.actuator_trnid[env.rig.actuators, 0]]
                    details.append(dict(pose=i, heldout=i >= 6,
                        first_command_1mrad=first((issued > .001) & common_alive, z['time']),
                        first_same_state_command_1mrad=first((same > .001) & baseline['alive'][:, i], z['time']),
                        first_joint_difference_1mrad=first((motion > .001) & common_alive, z['time']),
                        first_com_difference_1mm=first((com_delta > .001) & common_alive, z['time']),
                        first_moving_motor_violation=first((motor > .05) & onset & z['alive'][:, i], z['time']),
                        first_moving_com_violation=first((com > .02) & onset & z['alive'][:, i], z['time']),
                        first_support_loss=first(~z['support'][:, i] & onset & z['alive'][:, i], z['time']),
                        ramp_max_issued_delta=float(issued[ramp].max()),
                        ramp_max_same_state_issued_delta=float(same[ramp].max()),
                        ramp_max_com_delta=float(com_delta[ramp].max()),
                        largest_ramp_command_changes=[dict(joint=names[j], rms_rad=float(rms[j])) for j in order]))
                r = next(r for r in report['results'] if r['folder'] == folder.name)
                comparisons.append(dict(recipe=recipe, seed=seed, variant=variant,
                    complete=r['complete'], cases=details))
    save(OUT/'diagnostics.json', dict(regression=regression, physical=comparisons,
        inputs={str(path): digest(path) for path in paths},
        interpretation='Moving-goal thresholds are diagnostics, not replacements for full interval admission. Same-state projections use archived replay history.'))
    print('DIAGNOSTICS COMPLETE', len(comparisons), 'physical comparisons', flush=True)


if __name__ == '__main__':
    main()

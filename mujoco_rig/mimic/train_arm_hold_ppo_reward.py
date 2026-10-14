"""Isolate training reward width while reusing the sealed PPO implementation."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT, evaluate
from .run_learned_handoff import PRIOR, GuardedLearner
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_hold import episode
from .train_arm_hold_ppo import INITIAL, BASELINE, TRAIN_IDS, parity
from .train_arm_hold_ppo_rate import DiagnosticRollout, gate_replay, learn

OUT = Path('logs/mimickit-training/arm-hold-ppo-reward-20261008-01')


class RewardWidthRollout(DiagnosticRollout):
    def __init__(self, env, anchor, poses, width):
        assert width in (1., 2.)
        self.width = width
        super().__init__(env, anchor, poses)

    @torch.no_grad()
    def step(self, action, active):
        _, goals, _, _, frame = self.observation()
        reward, terminal, info = super().step(action, active)
        if self.width == 1.:
            return reward, terminal, info
        post = self.env.last
        error = (info['qpos'][:, self.qi] - goals[:, self.qi]).square().mean(-1)
        com_error = (post['com_m'][:, :2] - self.target_com[frame, self.ids, :2]).square().sum(-1)
        parts = info['components'].clone()
        parts[:, 0] = .45 * torch.exp(-error / (.05 * self.width)**2)
        parts[:, 1] = .20 * torch.exp(-com_error / (.02 * self.width)**2)
        dense = parts[:, 0] + parts[:, 1] + parts[:, 2] + parts[:, 3]
        parts[:, 4] = torch.where(info['failed'], -2 - dense, 0)
        reward = torch.where(info['failed'], -2., dense) + parts[:, 5]
        info['components'] = parts
        assert torch.allclose(parts.sum(-1)[active], reward[active], atol=1e-6, rtol=0)
        return reward, terminal, info


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    (OUT / 'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('ARM_HOLD_PPO_REWARD_PLAN.md')]
    inputs = {str(p): digest(p) for p in paths + [PARENT, PARENT.with_suffix('.amp.pt'), INITIAL, BASELINE,
        PRIOR / 'goals.npz', FOUNDATION / 'dummy.xml', FOUNDATION / 'stationary_reference.npz']}
    for p in paths:
        shutil.copy2(p, OUT / 'sources' / p.name)
    save(OUT / 'protocol.json', dict(inputs=inputs, seeds=[501028, 511028], width_multipliers=[1., 2.],
        actor_lr=3e-6, training_ids=TRAIN_IDS.tolist(), transitions_per_run=24576,
        reward_weights=dict(motor=.45, com=.20, feet=.10, support=.25),
        gate_scales=dict(motor_rmse_rad=.05, com_m=.02, feet_m=.02), arena_promotion=False))
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION / 'stationary_reference.npz', json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)
    env = make_environment(rig, anchor, 40.)
    poses = np.load(PRIOR / 'goals.npz')['poses']
    state = torch.load(INITIAL, weights_only=True)
    initial = QueueBlindLearner(state['mean'], state['scale'], state['output_scale'])
    initial.load_state_dict(state)
    initial.eval()
    adapters = {name: RewardWidthRollout(env, anchor, poses, width) for name, width in (('control', 1.), ('broad', 2.))}
    results = dict(parity={name: parity(adapter, initial) for name, adapter in adapters.items()},
        gate_replay=gate_replay(adapters['control']))
    save(OUT / 'checks.json', results)
    print('CHECKS', results, flush=True)
    for seed in (501028, 511028):
        for label, adapter in adapters.items():
            name = f'{label}-{seed}'
            net, training = learn(adapter, initial, OUT / name, seed, 3e-6)
            training['width_multiplier'] = adapter.width
            save(OUT / name / 'report.json', training)
            nominal = episode(env, anchor, net, poses, OUT / f'{name}-nominal')
            bank = {i: dict(poses=poses[i]) for i in range(16)}
            quiet = evaluate(env, anchor, bank, GuardedLearner(net, anchor, env), 'goal', 1., True, OUT / f'quiet-{name}')
            assert quiet['whole_interval_stand'] == 16
            results[name] = dict(training=training, nominal=nominal, quiet=quiet)
            if nominal['complete'] > 0 and nominal['stable_holds'] >= 4 and nominal['returned'] >= 4:
                results[name]['varied'] = episode(env, anchor, net, poses, OUT / f'{name}-varied', variant=True)
            save(OUT / 'progress.json', results)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT / 'report.json', dict(results=results, inputs_unchanged=True, arena_changed=False,
        output_hashes={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__ == '__main__':
    main()

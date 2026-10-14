"""Capture reset components to diagnose a failed initial-state fingerprint gate."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch

from .audit_stand_failures import read, save
from .measure_stand_learning_curve import training_fingerprint
from .replicate_stand_representations import read_state
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal
from .train_perturb_continuation import build_task, read_settings
from .train_torque_stand import build_agent


def snapshot(agent, task):
    q, v = read_state(task, 'gpu')
    control = task.engine.policy_control
    values = dict(qpos=q, qvel=v, pending=control.pending, active=control.active,
        ctrl=control.ctrl, cursor=np.asarray(control.cursor), steps=task.steps,
        offset=task.offset, observation=agent._curr_obs)
    return {key: (value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor)
                  else np.asarray(value).copy()) for key, value in values.items()}


def differences(left, right):
    result = {}
    for key in left:
        a, b = left[key], right[key]
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.isfinite(a).all() and np.isfinite(b).all()
        row = dict(bitwise_equal=a.tobytes() == b.tobytes(),
            unequal_values=int(np.count_nonzero(a != b)),
            max_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
        if key == 'observation':
            row['differing_columns'] = np.flatnonzero(np.any(a != b, axis=0)).tolist()
        result[key] = row
    return result


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    settings = read_settings(args.protocol)
    inputs, sources = protect(args.prior, args.out, [Path(__file__)], extra_inputs=[args.protocol])
    source_root = Path(settings['sources']['301027']['root'])
    source = read(source_root/'protocol.json')
    checkpoint = source_root/'updates-1200/model.pt'
    full = torch.load(checkpoint.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(checkpoint, map_location='cpu', weights_only=True), full)
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=source['config'], demonstrations=source['demonstrations'],
        model_sha256=source['model_sha256'], reference_sha256=source['reference_sha256'],
        checkpoint=str(checkpoint), scope='Two task instances, three full resets each; zero policy or physics steps'))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    captured, fingerprints = {}, {}
    for arm in ('quiet', 'perturb'):
        task = build_task(settings)(rig, reference, 64, 'cuda:0', arm=arm,
            representation='target_pd', reward_mode='reference_v9', episode_seconds=12.,
            seed=301027, demonstrations=[Path(r['path']) for r in source['demonstrations']], demonstration_rig=rig)
        task.mirror_demos = False
        random.seed(301027)
        torch.manual_seed(301027)
        np.random.seed(301027)
        agent = build_agent(source['config'], task, 'cuda:0')
        agent.load_state_dict(full)
        agent._sync_optimizer()
        agent._init_train()
        for reset in range(3):
            random.seed(302728)
            torch.manual_seed(302728)
            np.random.seed(302728)
            agent._curr_obs, agent._curr_info = agent._reset_envs()
            name = f'{arm}-{reset}'
            captured[name] = snapshot(agent, task)
            fingerprints[name] = training_fingerprint(agent, task)
            np.savez_compressed(args.out/f'{name}.npz', **captured[name])
        del agent, task
    comparisons = {name: differences(captured['quiet-0'], value) for name, value in captured.items()}
    save(args.out/'report.json', dict(fingerprints=fingerprints, differences_from_first_reset=comparisons,
        training_updates=0, policy_steps=0, physics_steps=0, explicit_world_resets=384,
        limitation='New diagnostic resets cannot reconstruct unsaved initial components of the training arms'))
    seal(args.out, inputs, sources, verified=True, training_updates=0, policy_steps=0, physics_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, required=True)
    run(parser.parse_args())

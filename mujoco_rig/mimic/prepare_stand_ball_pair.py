"""Prepare and admit both live warm-start arms before any PPO update."""
from pathlib import Path
import random

import numpy as np
import torch

from .audit_stand_failures import digest, save
from .measure_stand_learning_curve import rng_state, restore_rng
from .replicate_stand_representations import cpu, tensor_hash
from .stand_start_guard import capture, compare
from .train_torque_stand import build_agent


def extra_state(task):
    bq, bv = task.engine.ball_state()
    return dict(ball_qpos=cpu(bq), ball_qvel_world=cpu(bv), history=cpu(task._history.values),
        previous_com=cpu(task.previous_com), previous_action=cpu(task.previous_action),
        action_cost=cpu(task.action_cost), normal_impulse=cpu(task.engine.normal_impulse),
        contact_impulse=cpu(task.engine.contact_impulse), native_ctrl=cpu(task.engine.native_ctrl))


def identical(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.isfinite(left).all() and np.isfinite(right).all()
            and left.tobytes() == right.tobytes())


def prepare_pair(rig, reference, ball_path, source, checkpoint, seed, out):
    """Return the admitted live agents/tasks and isolated RNG streams for a future trainer."""
    from .amp_recovery import validate_sidecar
    from .stand_ball_task import StandBallAMPTask
    out.mkdir()
    full = torch.load(checkpoint.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(checkpoint, map_location='cpu', weights_only=True), full)
    prepared, extras = {}, {}
    for arm in ('quiet', 'single_ball'):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        task = StandBallAMPTask(rig, reference, ball_path=ball_path, seed=seed, arm=arm,
            demonstrations=[Path(r['path']) for r in source['demonstrations']])
        agent = build_agent(source['config'], task, 'cuda:0')
        agent.load_state_dict(full)
        agent._sync_optimizer()
        assert tensor_hash(agent.state_dict()) == tensor_hash(full)
        for name in ('_actor_optimizer', '_critic_optimizer', '_disc_optimizer'):
            optimizer = getattr(agent, name)
            assert optimizer.get_steps() == 0 and not optimizer._optimizer.state_dict()['state']
        agent._init_train()
        random.seed(seed+1701)
        np.random.seed(seed+1701)
        torch.manual_seed(seed+1701)
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        assert agent._iter == 0 and not task.steps.any()
        state = rng_state(task)
        capture(out/arm, agent, task)
        extras[arm] = extra_state(task)
        np.savez_compressed(out/arm/'extra-state.npz', **extras[arm])
        with np.load(out/arm/'extra-state.npz', allow_pickle=False) as saved:
            extras[arm] = {key: saved[key] for key in saved.files}
        prepared[arm] = (agent, task, state)
    result = compare(out/'quiet', out/'single_ball')
    exact = {key: bool(identical(extras['quiet'][key], extras['single_ball'][key]))
             for key in extras['quiet']}
    # Creating the second task must not mutate the first task or its CUDA buffers.
    for arm, (agent, task, state) in prepared.items():
        restore_rng(task, state)
        capture(out/f'{arm}-live', agent, task)
        live = compare(out/arm, out/f'{arm}-live')
        live_extra = extra_state(task)
        exact[f'{arm}_still_live'] = live['passed'] and all(
            identical(live_extra[k], extras[arm][k]) for k in live_extra)
    passed = result['passed'] and all(exact.values())
    save(out/'report.json', dict(passed=passed, common_start=result, exact_extra_state=exact,
        source_checkpoint_sha256=digest(checkpoint), ball_world_sha256=digest(ball_path),
        training_updates=0, physics_steps=0, arm_difference='Throw schedule only',
        extra_state_hashes={arm: digest(out/arm/'extra-state.npz') for arm in prepared}))
    if not passed:
        raise RuntimeError('Matched live ball starts failed; neither arm may train')
    return prepared

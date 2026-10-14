"""Frozen Stand objective and terminal-credit audit; no learning or Arena writes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, write_json
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


def clipping_fraction(normalizer, values):
    standardized = (values-normalizer.get_mean())/normalizer.get_std()
    return (standardized.abs() > normalizer._clip).float().mean(-1)


@torch.no_grad()
def trial(task, scorer, actor, label, seconds, out):
    from envs.base_env import DoneFlags, EnvMode
    from learning.rl_util import compute_td_lambda_return
    task.set_mode(EnvMode.TEST)
    task.reset()
    initial_qpos = task.engine.native_qpos().numpy().copy()
    initial_qvel = np.stack([data.qvel.copy() for data in task.engine.datas])
    controller = PredictiveStandController(task) if label == 'teacher' else None
    active = np.ones(task.get_num_envs(), dtype=bool)
    records = []
    started = time.perf_counter()
    for step in range(int(np.ceil(seconds/task.dt))):
        task.engine.active = active.copy()
        obs = task.observations()
        critic = scorer._model.eval_critic(scorer._obs_norm.normalize(obs)).squeeze(-1)
        action = (controller.command(active) if controller else
                  actor(obs) if actor is not None else torch.zeros((len(active), 33)))
        action[torch.from_numpy(~active)] = 0.
        next_obs, task_reward, done, info = task.step(action)
        warnings = task.engine.solver_warning.copy()
        done[torch.from_numpy(warnings)] = DoneFlags.FAIL.value
        task_reward[torch.from_numpy(warnings)] = 0.
        norm_disc = scorer._disc_obs_norm.normalize(info['disc_obs'])
        amp_reward = scorer._calc_disc_rewards(norm_disc)
        logits = scorer._model.eval_disc(norm_disc).squeeze(-1)
        combined = scorer._task_reward_weight*task_reward + scorer._disc_reward_weight*amp_reward
        next_critic = scorer._model.eval_critic(scorer._obs_norm.normalize(next_obs)).squeeze(-1)
        bootstrap = next_critic.clone()
        bootstrap[done == DoneFlags.FAIL.value] = scorer._compute_fail_val()
        bootstrap[done == DoneFlags.SUCC.value] = scorer._compute_succ_val()
        state = task.standing_measurements()
        values = dict(alive_before=active.copy(), obs=obs.numpy(), next_obs=next_obs.numpy(),
            qpos=task.engine.native_qpos().numpy(), qvel=np.stack([d.qvel.copy() for d in task.engine.datas]),
            action=action.numpy(), task_reward=task_reward.numpy(), amp_reward=amp_reward.numpy(),
            combined_reward=combined.numpy(), disc_logits=logits.numpy(), done=done.numpy(),
            critic=critic.numpy(), next_critic=next_critic.numpy(), bootstrap=bootstrap.numpy(),
            obs_clip_fraction=clipping_fraction(scorer._obs_norm, obs).numpy(),
            disc_clip_fraction=clipping_fraction(scorer._disc_obs_norm, info['disc_obs']).numpy(),
            disc_obs=info['disc_obs'].numpy(),
            balance=torch.stack(list(task.balance_criteria(state).values())).all(0).numpy(),
            posture=torch.stack(list(task.posture_criteria(state).values())).all(0).numpy(),
            pelvis_height_m=state['pelvis_height_m'].numpy(),
            pelvis_tilt_degrees=state['pelvis_tilt_degrees'].numpy(),
            foot_loads_n=state['foot_loads_n'].numpy(), solver_warning=warnings)
        if any(not np.isfinite(value).all() for value in values.values()):
            raise RuntimeError(f'Nonfinite audit values in {label} step {step}')
        records.append({key: value.copy() for key, value in values.items()})
        active &= done.numpy() == DoneFlags.NULL.value
        if (step+1) % 300 == 0:
            print(dict(controller=label, simulated_seconds=(step+1)*task.dt,
                       active_cases=int(active.sum())), flush=True)
        if not active.any():
            break
    arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
    arrays['time_s'] = (np.arange(len(records))+1)*task.dt
    arrays['initial_qpos'], arrays['initial_qvel'] = initial_qpos, initial_qvel
    cases = []
    for case in range(len(active)):
        live = arrays['alive_before'][:, case]
        count = int(live.sum())
        end = count-1
        flag = int(arrays['done'][end, case])
        survived = flag == DoneFlags.TIME.value
        window = live & (arrays['time_s'] > arrays['time_s'][end]-2.)
        balance = bool(survived and arrays['balance'][window, case].all())
        standing = bool(balance and arrays['posture'][window, case].all())
        discount = scorer._discount**np.arange(count)
        observed_return = float(discount @ arrays['combined_reward'][live, case])
        rewards = torch.from_numpy(arrays['combined_reward'][live, case])[:, None]
        bootstraps = torch.from_numpy(arrays['bootstrap'][live, case])[:, None]
        flags = torch.from_numpy(arrays['done'][live, case])[:, None]
        td_target = compute_td_lambda_return(rewards, bootstraps, flags,
                                            scorer._discount, scorer._td_lambda)
        cases.append(dict(case=case, frames=count, survival_seconds=float(count*task.dt),
            terminal_flag=DoneFlags(flag).name, time_truncated=survived,
            complete_task_episode=flag in (DoneFlags.FAIL.value, DoneFlags.SUCC.value),
            survived=survived, balance_pass=balance, standing_pass=standing,
            mean_live_rewards={key: float(arrays[key][live, case].mean())
                for key in ('task_reward', 'amp_reward', 'combined_reward')},
            discounted_observed_returns={key: float(discount @ arrays[key][live, case])
                for key in ('task_reward', 'amp_reward', 'combined_reward')},
            initial_critic=float(arrays['critic'][0, case]),
            terminal_critic_raw=float(arrays['next_critic'][end, case]),
            terminal_bootstrap_used=float(arrays['bootstrap'][end, case]),
            critic_bootstrapped_return=observed_return+float(scorer._discount**count*arrays['bootstrap'][end, case]),
            initial_td_lambda_target=float(td_target[0, 0]),
            initial_td_lambda_residual=float(td_target[0, 0]-arrays['critic'][0, case]),
            terminal_task_reward=float(arrays['task_reward'][end, case]),
            terminal_amp_reward=float(arrays['amp_reward'][end, case]),
            terminal_combined_reward=float(arrays['combined_reward'][end, case]),
            mean_obs_clip_fraction=float(arrays['obs_clip_fraction'][live, case].mean()),
            mean_disc_clip_fraction=float(arrays['disc_clip_fraction'][live, case].mean()),
            solver_warning=bool(arrays['solver_warning'][end, case])))
    np.savez_compressed(out/'trace.npz', **arrays)
    report = dict(controller=label, seconds=seconds, cases=cases,
        standing_passes=sum(c['standing_pass'] for c in cases), survivors=sum(c['survived'] for c in cases),
        mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in cases])),
        mean_live_rewards={key: float(np.mean([c['mean_live_rewards'][key] for c in cases]))
            for key in ('task_reward', 'amp_reward', 'combined_reward')},
        mean_discounted_observed_returns={key: float(np.mean([c['discounted_observed_returns'][key] for c in cases]))
            for key in ('task_reward', 'amp_reward', 'combined_reward')},
        mean_initial_critic=float(np.mean([c['initial_critic'] for c in cases])),
        mean_initial_td_lambda_residual=float(np.mean([c['initial_td_lambda_residual'] for c in cases])),
        critic_scope='Frozen trained-policy critic; teacher/zero/fresh trajectories are off-policy. Residuals are diagnostics, not training advantages.',
        wall_seconds=time.perf_counter()-started, training_updates=0)
    write_json(out/'report.json', report)
    print(dict(controller=label, standing_passes=report['standing_passes'],
               survival_seconds=report['mean_survival_seconds'], rewards=report['mean_live_rewards']), flush=True)
    return report


def source_audit(out, upstream):
    """Hash and preserve the exact implementation used for terminal credit."""
    paths = [Path(__file__), Path('mujoco_rig/mimic/stand_task.py'),
             Path('mujoco_rig/mimic/torque_stand_task.py'),
             Path('mujoco_rig/mimic/diagnose_stand_foundations.py'),
             Path('mujoco_rig/mimic/stand_foundation_followup.py'),
             Path('mujoco_rig/mimic/stable_normalizer.py'),
             Path('mujoco_rig/mimic/train_torque_stand.py')]
    paths += [upstream/'mimickit'/p for p in ('learning/base_agent.py', 'learning/ppo_agent.py',
              'learning/amp_agent.py', 'learning/rl_util.py', 'envs/base_env.py')]
    snapshots = out/'sources'
    snapshots.mkdir()
    sources = []
    for path in paths:
        destination = snapshots/path.name
        shutil.copy2(path, destination)
        sources.append(dict(path=str(path), snapshot=str(destination), sha256=sha256(path)))
    return dict(sources=sources,
        rollout_order='BaseAgent._rollout_train records next_obs/reward/done, then resets done environments.',
        reset_storage='BaseAgent._record_data_post_step stores pre-reset next_obs in ExperienceBuffer.',
        failure='PPO replaces V(next_obs) with env.get_reward_fail()/(1-discount); inherited fail reward is zero.',
        timeout='TIME keeps V(pre-reset next_obs); time limit truncates a continuing Stand task.',
        reset_boundary='compute_td_lambda_return sets lambda to zero at every non-NULL done flag, preventing cross-reset credit.',
        terminal_amp='StandTask zeros failed task reward. AMPAgent adds discriminator reward on the failure transition without masking it.',
        training_order='AMP computes rollout rewards before actor/critic updates, then updates discriminator. This audit freezes only the final snapshot.')


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from envs.base_env import EnvMode
    from learning.ppo_agent import PPOAgent
    from .amp_recovery import validate_sidecar
    from .stable_normalizer import install
    from .torque_stand_task import AMPTorqueStandTask
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(1)
    teacher = json.loads((args.dataset/'report.json').read_text())
    corrections = json.loads((args.corrections/'report.json').read_text())
    if not teacher['passed'] or not corrections['passed']:
        raise ValueError('Admitted physical clips required')
    model, reference = args.dataset/'dummy.xml', args.dataset/'stationary_reference.npz'
    if corrections['model_sha256'] != sha256(model) or corrections['reference_sha256'] != sha256(reference):
        raise ValueError('Correction plant/reference differs')
    clip_records = teacher['reports']['native']['clips'] + corrections['reports']['native']['amp_clips']
    clips = [Path(r['path']) for r in clip_records]
    paths = [model, reference, reference.with_suffix('.json'), args.dataset/'report.json',
             args.corrections/'report.json'] + [args.baseline/p for p in
                 ('agent.yaml', 'initial.pt', 'initial.amp.pt', 'model.pt', 'model.amp.pt')]
    for clip, record in zip(clips, clip_records):
        if sha256(clip) != record['sha256']:
            raise ValueError('Admitted clip hash mismatch')
        paths.extend((clip, clip.with_suffix('.json')))
    protected = {str(p): sha256(p) for p in paths}
    args.out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((args.baseline/'agent.yaml').read_text())
    rig = Rig.load(model, policy_exclude=())
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Spring-free torque plant required')
    task = AMPTorqueStandTask(rig, reference, 8, 'cpu', seed=301010, reward_mode='reference_v9',
        episode_seconds=args.seconds, engine_factory=DiagnosticNativeEngine,
        demonstrations=clips, demonstration_rig=rig)
    task.mirror_demos = False
    task.set_mode(EnvMode.TEST)
    scorer = build_agent(config, task, 'cpu')
    final_state = torch.load(args.baseline/'model.amp.pt', map_location='cpu', weights_only=True)
    final_portable = torch.load(args.baseline/'model.pt', map_location='cpu', weights_only=True)
    validate_sidecar(final_portable, final_state)
    scorer.load_state_dict(final_state)
    scorer.eval().requires_grad_(False)
    final_actor = StandPolicy(scorer).eval().requires_grad_(False)
    fresh = PPOAgent(config, task, 'cpu')
    install(fresh)
    initial_state = torch.load(args.baseline/'initial.pt', map_location='cpu', weights_only=True)
    initial_full = torch.load(args.baseline/'initial.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(initial_state, initial_full)
    fresh.load_state_dict(initial_state)
    fresh.eval().requires_grad_(False)
    initial_actor = StandPolicy(fresh).eval().requires_grad_(False)
    frozen = dict(scorer=scorer, final_actor=final_actor, initial_actor=initial_actor)
    hashes = {key: tensor_hash(value.state_dict()) for key, value in frozen.items()}
    audit = source_audit(args.out, args.mimickit)
    write_json(args.out/'terminal-source-audit.json', audit)
    write_json(args.out/'protocol.json', dict(inputs=protected, config=config, demonstrations=clip_records,
        frozen_tensor_hashes=hashes, mimickit_revision=MIMICKIT_REVISION, seconds=args.seconds,
        controllers=['teacher', 'zero', 'fresh_actor', 'trained_actor'], native_cases=8,
        delayed_command_slots=2, decimation=rig.decimation, control_dt=task.dt,
        scoring='Final Phase 10 critic/discriminator/normalizers for every controller; exact AMP reward method.',
        actors='Deterministic means using each actor checkpoint\'s own observation/action normalizers.',
        reward_weights=dict(task=scorer._task_reward_weight, amp=scorer._disc_reward_weight,
                            amp_scale=scorer._disc_reward_scale),
        discount=scorer._discount, td_lambda=scorer._td_lambda,
        fail_bootstrap=scorer._compute_fail_val(), success_bootstrap=scorer._compute_succ_val(),
        episode_scope='12-second TIME truncation of continuing Stand, not success; physical failure overrides TIME.',
        training_updates=0, arena_changed=False, sources=audit['sources']))
    reports = []
    traces = {}
    for label, actor in [('teacher', None), ('zero', None), ('fresh_actor', initial_actor),
                         ('trained_actor', final_actor)]:
        out = args.out/label
        out.mkdir()
        reports.append(trial(task, scorer, actor, label, args.seconds, out))
        with np.load(out/'trace.npz') as trace:
            traces[label] = {key: trace[key] for key in trace.files}
        write_json(args.out/'progress.json', dict(completed_controllers=len(reports), total_controllers=4))
    reset_matches = all(np.array_equal(traces['teacher'][key], trace[key])
                        for trace in traces.values() for key in ('initial_qpos', 'initial_qvel'))
    if not reset_matches:
        raise RuntimeError('Controller reset states differ')
    common_frames = [min(c['frames'] for r in reports for c in r['cases'] if c['case']==case)
                     for case in range(8)]
    shared = {}
    for label, trace in traces.items():
        shared[label] = {key: float(np.mean([trace[key][:count, case].mean()
            for case, count in enumerate(common_frames)]))
            for key in ('task_reward', 'amp_reward', 'combined_reward')}
    after = {key: tensor_hash(value.state_dict()) for key, value in frozen.items()}
    if after != hashes or any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise RuntimeError('Frozen weights or protected inputs changed')
    if scorer._obs_norm._pending_count or scorer._disc_obs_norm._pending_count:
        raise RuntimeError('Normalizer received pending samples')
    if any(sha256(Path(s['path'])) != s['sha256'] or sha256(Path(s['snapshot'])) != s['sha256']
           for s in audit['sources']):
        raise RuntimeError('Audit source snapshot mismatch')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        controllers=reports, common_prefix_frames=common_frames,
        common_prefix_seconds=[count*task.dt for count in common_frames],
        common_prefix_mean_rewards=shared, reset_states_identical=True,
        frozen_tensor_hashes_before=hashes, frozen_tensor_hashes_after=after,
        frozen_tensors_unchanged=True, normalizer_pending_samples=0, inputs_unchanged=True,
        source_snapshots_verified=True, training_updates=0, arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--corrections', type=Path, default=Path('logs/mimickit-training/stand-corrections-20261001-01'))
    parser.add_argument('--baseline', type=Path, default=Path('logs/mimickit-training/stand-ppo-noise-0p1-20261001-01'))
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 2 or not np.isfinite(args.seconds):
        parser.error('Require a finite horizon of at least two seconds')
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

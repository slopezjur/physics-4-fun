"""Bounded Stand physics diagnostics; never train or modify the Arena selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import linprog
import torch
import yaml

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


class DiagnosticNativeEngine(NativeDummyEngine):
    """Stop integrating terminated worlds and detect solver recovery warnings."""
    def __init__(self, rig, num_envs, device='cpu', **kwargs):
        super().__init__(rig, num_envs, device, **kwargs)
        self.active = np.ones(num_envs, dtype=bool)
        self.solver_warning = np.zeros(num_envs, dtype=bool)

    def reset_envs(self, ids, qpos, qvel):
        super().reset_envs(ids, qpos, qvel)
        self.active[ids.numpy()] = True
        self.solver_warning[ids.numpy()] = False

    def step(self):
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC)
        for i, data in enumerate(self.datas):
            if not self.active[i]:
                continue
            data.ctrl[:] = self.native_ctrl[i].numpy()
            for _ in range(self.rig.decimation):
                mujoco.mj_step(self.rig.model, data)
                if any(data.warning[warning].number for warning in bad):
                    self.solver_warning[i] = True
                    break
            self._ground_force_cache[i] = self._read_ground_forces(data)
            mujoco.mj_forward(self.rig.model, data)
            data.xfrc_applied[:] = 0


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def static_feasibility(rig, poses):
    """Rigid foot-contact force screen, including the unactuated DOFs.

    Contact forces are independent nonnegative friction-pyramid rays at the
    actual contact points. This screen does not certify compliant-contact holds.
    No fictitious root support or joint-limit reactions are supplied.
    """
    model = rig.model
    joints = model.actuator_trnid[rig.actuators, 0]
    dofs = model.jnt_dofadr[joints]
    wrist_dofs = np.array([model.jnt_dofadr[j] for j in range(1, model.njnt)
                          if j not in joints])
    motor_map = np.zeros((model.nv, len(dofs)))
    motor_map[dofs, np.arange(len(dofs))] = rig.action_scale
    allowed = {model.body(n).id for n in ('Foot_L', 'Foot_R', 'Toe_L', 'Toe_R')}
    results, commands = [], []
    for case, pose in enumerate(poses):
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        rays = []
        for contact in data.contact:
            if 0 not in contact.geom:
                continue
            body = model.geom_bodyid[contact.geom[1] if contact.geom[0] == 0 else contact.geom[0]]
            if body not in allowed:
                continue
            jac, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, jacr, contact.pos, body)
            normal, tangent1, tangent2 = contact.frame.reshape(3, 3)
            if normal[2] < 0:
                normal = -normal
            for direction in (normal + contact.friction[0]*tangent1,
                              normal - contact.friction[0]*tangent1,
                              normal + contact.friction[1]*tangent2,
                              normal - contact.friction[1]*tangent2):
                rays.append(jac.T @ direction)
        mapping = np.column_stack((motor_map, np.array(rays).T))
        required = data.qfrc_bias - data.qfrc_passive
        # Minimize peak normalized motor torque, rather than an arbitrary LP vertex.
        columns = mapping.shape[1]
        equality = np.column_stack((mapping, np.zeros(model.nv)))
        objective = np.zeros(columns + 1)
        objective[-1] = 1
        upper = np.zeros((2*len(dofs), columns + 1))
        for i in range(len(dofs)):
            upper[2*i, i], upper[2*i+1, i] = 1, -1
        upper[:, -1] = -1
        bounds = [(-1, 1)]*len(dofs) + [(0, None)]*len(rays) + [(0, 1)]
        solved = {}
        for name, rows in (('all_dofs', np.arange(model.nv)),
                           ('wrists_relaxed', np.setdiff1d(np.arange(model.nv), wrist_dofs))):
            solution = linprog(objective, A_ub=upper, b_ub=np.zeros(len(upper)),
                               A_eq=equality[rows], b_eq=required[rows], bounds=bounds,
                               method='highs')
            solved[name] = dict(feasible=bool(solution.success), status=int(solution.status),
                                message=solution.message)
            if solution.success:
                residual = mapping @ solution.x[:-1] - required
                solved[name].update(peak_normalized_torque=float(solution.x[-1]),
                    equality_residual_max_nm=float(np.abs(residual[rows]).max()),
                    wrist_residual_nm=residual[wrist_dofs].tolist(),
                    normalized_motor_torques=solution.x[:len(dofs)].tolist())
        commands.append(solved.get('wrists_relaxed', {}).get('normalized_motor_torques', [0.]*len(dofs)))
        results.append(dict(case=case, contact_ray_count=len(rays),
            unactuated_wrist_required_nm=required[wrist_dofs].tolist(),
            wrist_force_map_max=float(np.abs(mapping[wrist_dofs]).max()), **solved))
    return dict(scope='Rigid, force-distribution screen; no contact compliance or stability admission',
                friction_cone='Four nonnegative normal/tangent rays per actual support contact',
                joint_limit_reactions='Excluded; targets are inside joint limits',
                cases=results), np.asarray(commands, dtype=np.float32)


def reward_components(task):
    """Evaluate the exact production reward with each coefficient set to one."""
    from envs.deepmimic_env import compute_reward
    from .torque_stand_task import REFERENCE_IMITATION
    task._ensure_stationary_cache()
    target, engine = task._stationary_cache, task.engine
    q, v = target['reference_qpos'], target['reference_qvel']
    args = (engine.get_root_pos(0), engine.get_root_rot(0), engine.get_root_vel(0),
        engine.get_root_ang_vel(0), task.joint_rotations(engine.get_body_rot(0)), engine.get_dof_vel(0),
        engine.get_body_pos(0)[:, task.key_bodies], q[:, :3], q[:, [4, 5, 6, 3]], v[:, :3], v[:, 3:6],
        target['reference_joint_rotations'], v[:, 6:], target['reference_key_positions'],
        task.joint_weights, task.dof_weights, True, False)
    scales = tuple(REFERENCE_IMITATION['scales'].values())
    return torch.stack([compute_reward(*args, *np.eye(5)[i].tolist(), *scales)
                        for i in range(5)], -1)


@torch.no_grad()
def trace_controller(task, agent, actor, mode, feedforward, seconds):
    from envs.base_env import EnvMode, DoneFlags
    task.set_mode(EnvMode.TEST)
    task.episode_seconds = seconds + task.dt
    task.reset()
    joints = task.rig.model.actuator_trnid[task.rig.actuators, 0]
    qadr, dadr = task.rig.model.jnt_qposadr[joints], task.rig.model.jnt_dofadr[joints]
    target = task.reference.qpos[:, qadr]
    ff = torch.tensor(feedforward)
    # A quiet 1.5 Hz servo, critically damped using each pose's joint inertia.
    # Fixed diagnostic design; no controller parameter search or policy training.
    inertia = []
    for data in task.engine.datas:
        mass = np.zeros((task.rig.model.nv, task.rig.model.nv))
        mujoco.mj_fullM(task.rig.model, mass, data.qM)
        inertia.append(np.diag(mass)[dadr])
    inertia = torch.tensor(np.stack(inertia), dtype=torch.float32)
    authority = torch.tensor(task.rig.action_scale, dtype=torch.float32)
    kp = torch.minimum(inertia*(2*np.pi*1.5)**2, 4.*authority)
    kd = 2.*torch.sqrt(inertia*kp)
    active = torch.ones(8, dtype=torch.bool)
    rows = {key: [] for key in ('command', 'delayed_control_nm', 'qpos', 'qvel', 'observation',
        'alive_before', 'done', 'task_reward', 'disc_reward', 'components', 'pelvis_height_m',
        'foot_loads_n', 'balanced', 'posture_pass', 'pose_error_degrees', 'wrist_error_rad', 'solver_warning')}
    wrist_qadr = np.setdiff1d(np.arange(7, task.rig.model.nq), qadr)
    for step in range(int(np.floor(seconds/task.dt))):
        if mode == 'zero':
            command = torch.zeros((8, 33))
        elif mode == 'pd_feedforward':
            q, v = task.engine.native_qpos(), task.engine._array('qvel')
            command = (ff + (kp*(target-q[:, qadr])-kd*v[:, dadr])/authority).clamp(-1, 1)
        else:
            command = actor(task.observations())
        command[~active] = 0
        task.engine.active = active.numpy().copy()
        obs, task_r, done, info = task.step(command)
        numerical_failure = torch.tensor(task.engine.solver_warning)
        done[numerical_failure] = DoneFlags.FAIL.value
        task_r[numerical_failure] = 0
        components = reward_components(task)
        raw = components @ torch.tensor([.5, .1, .15, .1, .15])
        if not torch.allclose(raw[done != DoneFlags.FAIL.value], task_r[done != DoneFlags.FAIL.value], atol=2e-6):
            raise RuntimeError('Component attribution differs from production reward')
        disc_r = agent._calc_disc_rewards(agent._disc_obs_norm.normalize(info['disc_obs']))
        state = task.standing_measurements()
        balance = torch.stack(list(task.balance_criteria(state).values())).all(0)
        posture = torch.stack(list(task.posture_criteria(state).values())).all(0)
        row = dict(command=command, delayed_control_nm=task.engine.native_ctrl,
            qpos=task.engine.native_qpos(), qvel=task.engine._array('qvel'), observation=obs,
            alive_before=active.clone(), done=done, task_reward=task_r, disc_reward=disc_r,
            components=components, pelvis_height_m=state['pelvis_height_m'], foot_loads_n=state['foot_loads_n'],
            balanced=balance, posture_pass=posture, pose_error_degrees=state['whole_body_reference_error_degrees'],
            wrist_error_rad=task.engine.native_qpos()[:, wrist_qadr]-task.reference.qpos[:, wrist_qadr],
            solver_warning=numerical_failure)
        for key, value in row.items():
            rows[key].append(value.detach().cpu().numpy().copy())
        active &= done != DoneFlags.FAIL.value
        if not active.any():
            break
    result = {key: np.stack(values) for key, values in rows.items()}
    result['time_s'] = (np.arange(len(result['done']))+1)*task.dt
    return result


def summarize_trace(trace, dt):
    from envs.base_env import DoneFlags
    cases = []
    for case in range(8):
        mask = trace['alive_before'][:, case]
        end = np.flatnonzero(mask)[-1]
        failure = trace['done'][end, case] == DoneFlags.FAIL.value
        valid = mask & (trace['done'][:, case] != DoneFlags.FAIL.value)
        final_window = valid & (trace['time_s'] > trace['time_s'][end]-2)
        cases.append(dict(case=case, observed_seconds=float(trace['time_s'][end]), failed=bool(failure),
            solver_warning=bool(trace['solver_warning'][end, case]),
            survived_test=bool(not failure), end_pelvis_height_m=float(trace['pelvis_height_m'][end, case]),
            balance_fraction=float(trace['balanced'][valid, case].mean()) if valid.any() else 0.,
            final_two_seconds_balance_fraction=float(trace['balanced'][final_window, case].mean()) if final_window.any() else 0.,
            final_two_seconds_posture_fraction=float(trace['posture_pass'][final_window, case].mean()) if final_window.any() else 0.,
            peak_wrist_error_degrees=float(np.rad2deg(np.abs(trace['wrist_error_rad'][mask, case])).max()),
            command_saturation_fraction=float((np.abs(trace['command'][mask, case])>=.98).mean()),
            pose_error_mean_degrees=float(trace['pose_error_degrees'][mask, case].mean()),
            zero_initial_delayed_control=bool(not np.any(trace['delayed_control_nm'][:2, case]))))
    return dict(cases=cases, survivors=sum(c['survived_test'] for c in cases),
                scope='Five-second diagnostic; not the sixty-second standing admission')


def reward_summary(traces, task_weight, disc_weight):
    result = dict(scope='Frozen final discriminator and normalizers; no discriminator updates',
                  components=['pose', 'joint_velocity', 'root_pose', 'root_velocity', 'key_positions'],
                  component_weights=[.5, .1, .15, .1, .15], controllers={})
    for mode, trace in traces.items():
        time = trace['time_s'][:, None]
        regions = dict(common_first_quarter_second=trace['alive_before'] & (time<=.25),
            all_alive_including_terminal=trace['alive_before'],
            low_pelvis_alive=trace['alive_before'] & (trace['pelvis_height_m']<.75),
            balance_gate_met=trace['alive_before'] & trace['balanced'])
        summaries = {}
        for name, mask in regions.items():
            if not mask.any():
                summaries[name] = dict(samples=0)
                continue
            task = float(trace['task_reward'][mask].mean())
            disc = float(trace['disc_reward'][mask].mean())
            weighted_task, weighted_disc = task_weight*task, disc_weight*disc
            summaries[name] = dict(samples=int(mask.sum()), task_reward_mean=task,
                disc_reward_mean=disc, task_contribution_mean=weighted_task,
                disc_contribution_mean=weighted_disc, combined_mean=weighted_task+weighted_disc,
                realized_disc_fraction=weighted_disc/(weighted_task+weighted_disc),
                raw_components_mean=trace['components'][mask].mean(0).tolist())
        result['controllers'][mode] = summaries
    return result


@torch.no_grad()
def replay_gpu(rig, reference, commands, out):
    from envs.base_env import EnvMode
    from .torque_stand_task import TorqueStandTask
    from .model_contract import model_checks
    native = TorqueStandTask(rig, reference, 8, 'cpu', engine_factory=NativeDummyEngine,
                            reward_mode='reference_v9', episode_seconds=60.)
    gpu = TorqueStandTask(rig, reference, 8, 'cuda:0', reward_mode='reference_v9', episode_seconds=60.)
    backend_metadata = dict(native_qpos_dtype=str(native.engine.datas[0].qpos.dtype),
        gpu_solver_qpos_dtype=str(gpu.engine._solver.mjw_data.qpos.dtype),
        model_checks=model_checks(rig.model, gpu.engine._solver.mj_model),
        solver=int(rig.model.opt.solver), iterations=int(rig.model.opt.iterations),
        tolerance=float(rig.model.opt.tolerance), contact_sampling='Last completed solver substep')
    for task in (native, gpu):
        task.set_mode(EnvMode.TEST)
        task.reset()
    records = {key: [] for key in ('native_qpos', 'gpu_qpos', 'native_velocity', 'gpu_velocity',
        'native_loads', 'gpu_loads', 'native_obs', 'gpu_obs', 'native_done', 'gpu_done')}
    def capture(nd=None, gd=None):
        for prefix, task, done in (('native', native, nd), ('gpu', gpu, gd)):
            velocity = torch.cat((task.engine.get_root_vel(0), task.engine.get_root_ang_vel(0), task.engine.get_dof_vel(0)), -1)
            values = dict(qpos=task.engine.native_qpos(), velocity=velocity,
                          loads=task.engine.get_ground_contact_forces(0), obs=task.observations(),
                          done=torch.zeros(8, dtype=torch.int) if done is None else done)
            for key, value in values.items():
                records[prefix+'_'+key].append(value.cpu().numpy().copy())
    capture()
    for command in commands[:int(np.floor(1./native.dt))]:
        _, _, nd, _ = native.step(torch.tensor(command))
        _, _, gd, _ = gpu.step(torch.tensor(command, device='cuda:0'))
        capture(nd, gd)
    records = {key: np.stack(value) for key, value in records.items()}
    records['time_s'] = np.arange(len(records['native_qpos']))*native.dt
    np.savez_compressed(out/'backend-replay.npz', **records)
    qdiff_all = np.abs(records['native_qpos']-records['gpu_qpos'])
    force_diff = np.abs(records['native_loads']-records['gpu_loads'])
    force_index = np.unravel_index(force_diff.argmax(), force_diff.shape)
    actuated_qadr = rig.model.jnt_qposadr[rig.model.actuator_trnid[rig.actuators, 0]]
    reports = []
    layout = native.observation_contract()['observation_layout']
    for index, seconds in ((0, 0.), (1, native.dt), (min(6, len(records['time_s'])-1), .1),
                           (min(30, len(records['time_s'])-1), .5), (len(records['time_s'])-1, 1.)):
        qdiff = np.abs(records['native_qpos'][index]-records['gpu_qpos'][index])
        odiff = np.abs(records['native_obs'][index]-records['gpu_obs'][index])
        reports.append(dict(time_s=float(records['time_s'][index]),
            root_position_max_m=float(qdiff[:, :3].max()), hinge_max_rad=float(qdiff[:, 7:].max()),
            world_velocity_max=float(np.abs(records['native_velocity'][index]-records['gpu_velocity'][index]).max()),
            ground_force_max_n=float(np.abs(records['native_loads'][index]-records['gpu_loads'][index]).max()),
            observation_groups={g['name']:float(odiff[:, g['offset']:g['offset']+g['width']].max()) for g in layout}))
    return dict(scope='Eight worlds; identical recorded native actor commands, no feedback adaptation',
        backend_metadata=backend_metadata,
        duration_seconds=float(records['time_s'][-1]), initial_qpos_exact=bool(np.array_equal(records['native_qpos'][0], records['gpu_qpos'][0])),
        maxima_over_entire_replay=dict(root_position_m=float(qdiff_all[:, :, :3].max()),
            all_hinges_rad=float(qdiff_all[:, :, 7:].max()),
            actuated_hinges_rad=float(qdiff_all[:, :, actuated_qadr].max()),
            ground_force_n=float(force_diff.max()),
            ground_force_peak=dict(time_s=float(records['time_s'][force_index[0]]),
                case=int(force_index[1]), body=rig.model.body(force_index[2]+1).name),
            observation=float(np.abs(records['native_obs']-records['gpu_obs']).max())),
        checkpoints=reports,
        native_failures=int((records['native_done'] == 1).any(0).sum()),
        gpu_failures=int((records['gpu_done'] == 1).any(0).sum()))


@torch.no_grad()
def main(args):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar, amp_sidecar
    from .diagnose_torque_stand import load_bundle
    from .policy import StandPolicy
    from .torque_stand_task import AMPTorqueStandTask
    from .train_torque_stand import build_agent, state_sha256
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(2)
    torch.manual_seed(301010)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    bundle, hashes = load_bundle(args.run)
    protocol = json.loads((args.run/'protocol.json').read_text())
    reference = Path('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz')
    rig = Rig.load(bundle/'dummy.xml', policy_exclude=())
    source_rig = Rig.load(Path('mujoco_rig/dummy.xml'))
    if sha256(source_rig.path) != protocol['source_model_sha256']:
        raise ValueError('Demonstration source model changed')
    task = AMPTorqueStandTask(rig, reference, 8, 'cpu', engine_factory=DiagnosticNativeEngine,
        reward_mode='reference_v9', episode_seconds=60., demonstrations=[Path(d['path']) for d in protocol['demonstrations']],
        demonstration_rig=source_rig)
    config = yaml.safe_load((args.run/'agent.yaml').read_text())
    agent = build_agent(config, task, 'cpu')
    portable = torch.load(args.run/'model.pt', map_location='cpu', weights_only=True)
    training = torch.load(amp_sidecar(args.run/'model.pt'), map_location='cpu', weights_only=True)
    validate_sidecar(portable, training)
    agent.load_state_dict(training, strict=True)
    agent.eval()
    actor = StandPolicy(agent).eval()
    before = state_sha256(agent)
    print('Running bounded static feasibility', flush=True)
    static, ff = static_feasibility(rig, task.reference.qpos.numpy())
    write_json(args.out/'static-feasibility.json', static)
    traces, controllers = {}, {}
    for mode in ('zero', 'pd_feedforward', 'final_actor'):
        print('Running five-second diagnostic:', mode, flush=True)
        trace = trace_controller(task, agent, actor, mode, ff, 5.)
        np.savez_compressed(args.out/(mode+'-trace.npz'), **trace)
        traces[mode], controllers[mode] = trace, summarize_trace(trace, task.dt)
        write_json(args.out/'controller-holds.json', controllers)
    rewards = reward_summary(traces, float(config['task_reward_weight']), float(config['disc_reward_weight']))
    write_json(args.out/'reward-attribution.json', rewards)
    print('Running one-second GPU/native command replay', flush=True)
    parity = replay_gpu(rig, reference, traces['final_actor']['command'], args.out)
    write_json(args.out/'backend-replay.json', parity)
    if before != state_sha256(agent):
        raise RuntimeError('Diagnostic changed network or normalization tensors')
    report = dict(schema='stand_foundation_diagnostics_v1', completed_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic_source_sha256=sha256(Path(__file__)),
        source_run=str(args.run), hashes=hashes, amp_checkpoint_sha256=sha256(amp_sidecar(args.run/'model.pt')),
        elapsed_wall_seconds=time.perf_counter()-start, model_state_unchanged=True, training_updates=0,
        physics_changed=False, arena_changed=False, physics_dt=rig.model.opt.timestep, control_dt=task.dt,
        pd_controller=dict(natural_frequency_hz=1.5, damping_ratio=1.,
            gain_design='Kp=min(M_diagonal*(2*pi*f)^2,4*torque_authority); Kd=2*sqrt(M_diagonal*Kp)',
            feedforward='Min-peak bounded contact LP with wrist equilibrium relaxed; no root anchoring',
            control_rate='60 Hz, existing two-step delay and zero-filled queue',
            interpretation='Failure of this one controller is not a pose impossibility proof'),
        static_feasibility=static, controller_holds=controllers, reward_attribution=rewards, backend_replay=parity)
    write_json(args.out/'report.json', report)
    print('Completed:', args.out/'report.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    main(parser.parse_args())

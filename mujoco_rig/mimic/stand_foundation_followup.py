"""Isolated reference, predictive-control and GPU numerics follow-up diagnostics.

No training defaults, demonstration assets, checkpoints or Arena files are edited.
Running this module performs physics tests and requires explicit test authorization.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, static_feasibility, write_json
from .inverse_contact_control import inverse_contact_map, refine_forward_torque, solve_box
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate

GPU_VARIANTS = ('baseline', 'uncaptured', 'tolerance_1e_minus6', 'no_warmstart')


class SolverTraceNativeEngine(NativeDummyEngine):
    """Capture solver counters before the evaluator's ending-state mj_forward."""
    def __init__(self, rig, num_envs, device='cpu', **kwargs):
        super().__init__(rig, num_envs, device, **kwargs)
        self.last_iterations = np.zeros(num_envs, dtype=int)
        self.last_constraint_count = np.zeros(num_envs, dtype=int)

    def _after_physics_step(self, world, data):
        self.last_iterations[world] = data.solver_niter[0]
        self.last_constraint_count[world] = data.nefc


def passive_wrist_reference(rig, source, out):
    """Find interior wrist gravity equilibria while preserving every other coordinate."""
    model = rig.model
    motors = set(model.actuator_trnid[rig.actuators, 0])
    wrists = [j for j in range(1, model.njnt) if j not in motors]
    if len(wrists) != 6 or any(not model.joint(j).name.startswith('Hand_') for j in wrists):
        raise ValueError('Expected exactly six unactuated wrist hinges')
    qadr, dadr = model.jnt_qposadr[wrists], model.jnt_dofadr[wrists]
    with np.load(source, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    original_q = arrays['qpos'].copy()
    reports = []
    for case, pose in enumerate(arrays['qpos']):
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        def residual(angles):
            data.qpos[qadr] = angles
            mujoco.mj_forward(model, data)
            return (data.qfrc_bias-data.qfrc_passive)[dadr].copy()
        lower, upper = model.jnt_range[wrists].T
        solution = least_squares(residual, pose[qadr], bounds=(lower+1e-5, upper-1e-5),
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
        force = residual(solution.x)
        jacobian = np.empty((6, 6))
        for i in range(6):
            plus, minus = solution.x.copy(), solution.x.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            jacobian[:, i] = (residual(plus)-residual(minus))/2e-5
        # Gravity generalized force is the potential-energy gradient.
        curvature = np.linalg.eigvalsh((jacobian+jacobian.T)*.5)
        # Rotation about gravity can be neutral; require no unstable energy direction.
        admitted = bool(solution.success and np.max(np.abs(force)) <= 1e-6
                        and curvature.min() >= -1e-5)
        if admitted:
            pose[qadr] = solution.x
        reports.append(dict(case=case, admitted=admitted, residual_max_nm=float(np.abs(force).max()),
            potential_curvature_eigenvalues=curvature.tolist(), wrist_angles_rad=solution.x.tolist(),
            wrist_change_degrees=np.rad2deg(solution.x-original_q[case, qadr]).tolist()))
    path = out/'passive-wrist-reference.npz'
    np.savez_compressed(path, **arrays)
    metadata = copy.deepcopy(json.loads(source.with_suffix('.json').read_text()))
    # Original optimistic admission must not be inherited by a derived candidate.
    for field in ('physical_cases', 'action_window_consistency', 'passed'):
        metadata.pop(field, None)
    metadata.update(schema='stationary_passive_wrist_candidate_v1', reference_sha256=sha256(path),
        model_sha256=sha256(rig.path), derived_from=dict(path=str(source), sha256=sha256(source)),
        admission='Candidate only: wrist equilibrium screen; no closed-loop standing admission',
        scope='Only six wrist coordinates changed; neutral gravity-axis rotation is permitted',
        coordinate_changes=[model.joint(j).name for j in wrists], wrist_cases=reports)
    write_json(path.with_suffix('.json'), metadata)
    static, _ = static_feasibility(rig, arrays['qpos'].astype(np.float32))
    write_json(out/'candidate-static-feasibility.json', static)
    return path, dict(cases=reports, unchanged_non_wrist_coordinates=bool(np.array_equal(
        arrays['qpos'][:, np.setdiff1d(np.arange(model.nq), qadr)],
        original_q[:, np.setdiff1d(np.arange(model.nq), qadr)])),
        exact_static_feasible_count=sum(c['all_dofs']['feasible'] for c in static['cases']))


class PredictiveStandController:
    """Bounded whole-body acceleration feedback, predicting through the command queue."""
    def __init__(self, task, state_reader=None):
        self.task, self.model = task, task.rig.model
        self.actuators = task.rig.actuators
        self.dofs = self.model.jnt_dofadr[self.model.actuator_trnid[self.actuators, 0]]
        self.scale = task.rig.action_scale
        self.data = [mujoco.MjData(self.model) for _ in range(task.get_num_envs())]
        self.state_reader = state_reader or self._native_state
        self.target_qpos = task.reference.qpos.detach().cpu().numpy().astype(np.float64)
        weights = np.zeros(self.model.nv)
        weights[:6], weights[self.dofs] = 4., .6
        self.tracking = np.diag(weights)[weights>0]
        self.reports = []

    def _native_state(self):
        return (np.stack([data.qpos for data in self.task.engine.datas]),
                np.stack([data.qvel for data in self.task.engine.datas]))

    def command(self, active):
        controls = np.zeros((self.task.get_num_envs(), len(self.actuators)), dtype=np.float32)
        queue = self.task.engine.policy_control
        qpos, qvel = self.state_reader()
        pending = queue.pending.detach().cpu().numpy()
        for world in np.flatnonzero(active):
            data = self.data[world]
            mujoco.mj_resetData(self.model, data)
            data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
            # Replay already committed controls on private simulation state only.
            for lag in range(len(queue.pending)):
                data.ctrl[self.actuators] = pending[(queue.cursor+lag)%len(pending), world]*self.scale
                for _ in range(self.task.rig.decimation):
                    mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            error = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, error, 1., data.qpos,
                                      self.target_qpos[world])
            frequency = 2*np.pi*1.
            desired = np.clip(frequency**2*error-2*frequency*data.qvel, -20., 20.)
            mapping, offset, *_ = inverse_contact_map(self.model, data, self.dofs, self.scale)
            reduced = self.tracking @ mapping
            source = data.ctrl[self.actuators]/self.scale
            regularizer = .15**2
            hessian = reduced.T@reduced+np.eye(len(source))*(regularizer+1e-7)
            linear = -reduced.T@(self.tracking@(desired-offset))-regularizer*source
            lower, upper = -np.ones(len(source)), np.ones(len(source))
            proposed, rank = solve_box(hessian, linear, lower, upper, source)
            if rank < 0 or not np.isfinite(proposed).all():
                raise RuntimeError('Predictive motor QP failed')
            # Validate against actual compliant forward dynamics, not free force rays.
            control, acceleration, diagnostics = refine_forward_torque(self.model, data,
                self.actuators, self.dofs, self.scale, self.tracking, self.tracking@desired,
                proposed, lower, upper, iterations=2)
            controls[world] = control
            self.reports.append(dict(world=int(world),
                tracking_residual=float(np.linalg.norm(self.tracking@(acceleration-desired))), **diagnostics))
        return torch.tensor(controls)


@torch.no_grad()
def controller_hold(rig, reference, out, seconds=5.):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    task = TorqueStandTask(rig, reference, 8, 'cpu', engine_factory=DiagnosticNativeEngine,
                         reward_mode='reference_v9', episode_seconds=seconds+rig.model.opt.timestep*rig.decimation)
    task.set_mode(EnvMode.TEST)
    task.reset()
    controller = PredictiveStandController(task)
    active = np.ones(8, dtype=bool)
    samples = {key: [] for key in ('qpos', 'qvel', 'command', 'done', 'alive_before', 'balanced', 'posture', 'solver_warning')}
    for _ in range(int(np.floor(seconds/task.dt))):
        command = controller.command(active)
        task.engine.active = active.copy()
        _, _, done, _ = task.step(command)
        warnings = task.engine.solver_warning.copy()
        done[torch.tensor(warnings)] = DoneFlags.FAIL.value
        state = task.standing_measurements()
        balanced = torch.stack(list(task.balance_criteria(state).values())).all(0)
        posture = torch.stack(list(task.posture_criteria(state).values())).all(0)
        values = dict(qpos=task.engine.native_qpos().numpy(), qvel=task.engine._array('qvel').numpy(),
            command=command.numpy(), done=done.numpy(), alive_before=active.copy(),
            balanced=balanced.numpy(), posture=posture.numpy(), solver_warning=warnings)
        for key, value in values.items():
            samples[key].append(value.copy())
        active &= done.numpy()!=DoneFlags.FAIL.value
        if not active.any():
            break
    arrays = {key: np.stack(value) for key, value in samples.items()}
    arrays['time_s']=(np.arange(len(arrays['done']))+1)*task.dt
    np.savez_compressed(out/'predictive-controller-trace.npz', **arrays)
    cases=[]
    for case in range(8):
        mask=arrays['alive_before'][:,case]
        end=np.flatnonzero(mask)[-1]
        failed=arrays['done'][end,case]==DoneFlags.FAIL.value
        stable=mask & (arrays['time_s']>arrays['time_s'][end]-2.)
        cases.append(dict(case=case, survived=bool(not failed), duration_s=float(arrays['time_s'][end]),
            balance_final_window=bool(arrays['balanced'][stable,case].all()),
            posture_final_window=bool(arrays['posture'][stable,case].all()),
            pelvis_height_end_m=float(arrays['qpos'][end,case,2]),
            solver_warning=bool(arrays['solver_warning'][end,case]),
            command_saturation_fraction=float((np.abs(arrays['command'][mask,case])>=.98).mean())))
    report=dict(scope='Five-second diagnostic; natural pose admission still requires sixty seconds',
        external_forces=False, root_anchoring=False, passive_stiffness_changes=False,
        controller='One Hz acceleration feedback, full inverse-contact map, two-step prediction, bounded motor QP',
        cases=cases, survives=sum(c['survived'] for c in cases),
        solver_summary=dict(commands=len(controller.reports),
            max_tracking_residual=max(r['tracking_residual'] for r in controller.reports)))
    write_json(out/'predictive-controller.json', report)
    return report


def gpu_probe_engine(variant):
    from .newton_adapter import DummyNewtonEngine
    class NumericalProbe(DummyNewtonEngine):
        def _build_solver(self):
            super()._build_solver()
            if variant == 'tolerance_1e_minus6':
                self._solver.mj_model.opt.tolerance=1e-6
                self._solver.mjw_model.opt.tolerance.fill_(1e-6)
            elif variant=='no_warmstart':
                flags=int(self._solver.mj_model.opt.disableflags)|int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
                self._solver.mj_model.opt.disableflags=flags
                self._solver.mjw_model.opt.disableflags=flags
        def _build_graphs(self):
            if variant=='uncaptured':
                self._graph=None
            else:
                super()._build_graphs()
    return NumericalProbe


@torch.no_grad()
def gpu_replay_probe(rig, reference, commands, actor, variant, out):
    from envs.base_env import EnvMode
    from .torque_stand_task import TorqueStandTask
    native=TorqueStandTask(rig,reference,8,'cpu',engine_factory=SolverTraceNativeEngine,
                         reward_mode='reference_v9',episode_seconds=60.)
    gpu=TorqueStandTask(rig,reference,8,'cuda:0',engine_factory=gpu_probe_engine(variant),
                       reward_mode='reference_v9',episode_seconds=60.)
    for task in (native,gpu):
        task.set_mode(EnvMode.TEST)
        task.reset()
    records={key:[] for key in ('native_qpos','gpu_qpos','native_loads','gpu_loads',
        'native_obs','gpu_obs','native_solver_iterations','gpu_solver_iterations',
        'native_constraint_count','gpu_constraint_count','actor_full_input_action_delta','actor_load_input_action_delta')}
    load_group=next(g for g in native.observation_contract()['observation_layout'] if g['name']=='foot_and_toe_normal_load_kN_L_R')
    for command in commands[:int(np.floor(1./native.dt))]:
        native.step(torch.tensor(command))
        gpu.step(torch.tensor(command,device='cuda:0'))
        no,go=native.observations(),gpu.observations().cpu()
        loads_only=no.clone()
        start=load_group['offset']
        loads_only[:,start:start+2]=go[:,start:start+2]
        values=dict(native_qpos=native.engine.native_qpos().numpy(),gpu_qpos=gpu.engine.native_qpos().cpu().numpy(),
            native_loads=native.engine.get_ground_contact_forces(0).numpy(),gpu_loads=gpu.engine.get_ground_contact_forces(0).cpu().numpy(),
            native_obs=no.numpy(),gpu_obs=go.numpy(),
            native_solver_iterations=native.engine.last_iterations,
            gpu_solver_iterations=gpu.engine._solver.mjw_data.solver_niter.numpy(),
            native_constraint_count=native.engine.last_constraint_count,
            gpu_constraint_count=gpu.engine._solver.mjw_data.nefc.numpy(),
            actor_full_input_action_delta=(actor(go)-actor(no)).numpy(),
            actor_load_input_action_delta=(actor(loads_only)-actor(no)).numpy())
        for key,value in values.items():
            records[key].append(value.copy())
    arrays={key:np.stack(value) for key,value in records.items()}
    np.savez_compressed(out/'trace.npz',**arrays)
    qdiff=np.abs(arrays['native_qpos']-arrays['gpu_qpos'])
    report=dict(variant=variant,model_sha256=sha256(rig.path),duration_s=len(commands[:int(np.floor(1./native.dt))])*native.dt,
        max_root_difference_m=float(qdiff[:,:,:3].max()),max_hinge_difference_rad=float(qdiff[:,:,7:].max()),
        max_ground_force_difference_n=float(np.abs(arrays['native_loads']-arrays['gpu_loads']).max()),
        max_actor_action_difference=float(np.abs(arrays['actor_full_input_action_delta']).max()),
        max_actor_action_difference_loads_only=float(np.abs(arrays['actor_load_input_action_delta']).max()),
        native_solver_iteration_max=int(arrays['native_solver_iterations'].max()),
        gpu_solver_iteration_max=int(arrays['gpu_solver_iterations'].max()),
        interpretation='Open-loop replay; action differences are counterfactual frozen-actor sensitivity, not a closed-loop trial')
    write_json(out/'report.json',report)
    return report,arrays


@torch.no_grad()
def run(args):
    activate(args.mimickit)
    from util import mp_util
    from learning.ppo_agent import PPOAgent
    from .policy import StandPolicy
    from .stable_normalizer import install
    from .torque_stand_task import TorqueStandTask
    from .diagnose_torque_stand import load_bundle
    import yaml
    mp_util.init(0,1,'cpu',0)
    torch.set_num_threads(2)
    bundle, _ = load_bundle(args.run)
    args.out.mkdir(parents=True,exist_ok=False)
    rig=Rig.load(bundle/'dummy.xml',policy_exclude=())
    reference=Path('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz')
    protected={path:sha256(path) for path in (rig.path,reference,args.run/'model.pt',bundle/'stand.onnx',args.trace)}
    write_json(args.out/'protocol.json',dict(scope='Isolated follow-up physics diagnostics',
        original_model_sha256=sha256(rig.path),source_reference_sha256=sha256(reference),
        seed=301010,torch_threads=2,hold_seconds=5.,replay_seconds=1.,training_updates=0,
        diagnostic_source_sha256=sha256(Path(__file__)),
        command_trace_sha256=sha256(args.trace), gpu_variants=list(GPU_VARIANTS)))
    print('Generating passive wrist reference candidate',flush=True)
    candidate,wrist_report=passive_wrist_reference(rig,reference,args.out)
    write_json(args.out/'wrist-equilibrium.json',wrist_report)
    if all(case['admitted'] for case in wrist_report['cases']) and wrist_report['exact_static_feasible_count'] == 8:
        print('Running predictive controller hold', flush=True)
        controller_report = controller_hold(rig, candidate, args.out)
    else:
        controller_report = dict(skipped=True, reason='Candidate did not pass all eight equilibrium screens')
        write_json(args.out/'predictive-controller.json', controller_report)
    native=TorqueStandTask(rig,reference,8,'cpu',engine_factory=NativeDummyEngine,reward_mode='reference_v9')
    agent=PPOAgent(yaml.safe_load((args.run/'agent.yaml').read_text()),native,'cpu')
    install(agent)
    agent.load(str(args.run/'model.pt'))
    actor=StandPolicy(agent).eval()
    commands=np.load(args.trace,allow_pickle=False)['command']
    results={}
    for variant in GPU_VARIANTS:
        repeats=[]
        for trial in range(2):
            torch.manual_seed(301010)
            np.random.seed(301010)
            out=args.out/(variant+'-'+str(trial+1))
            out.mkdir()
            print('GPU replay',variant,trial+1,flush=True)
            report,arrays=gpu_replay_probe(rig,reference,commands,actor,variant,out)
            repeats.append((report,arrays))
        a,b=repeats[0][1],repeats[1][1]
        results[variant]=dict(reports=[r[0] for r in repeats],
            native_repeat_qpos_max=float(np.abs(a['native_qpos']-b['native_qpos']).max()),
            gpu_repeat_qpos_max=float(np.abs(a['gpu_qpos']-b['gpu_qpos']).max()),
            gpu_repeat_load_max_n=float(np.abs(a['gpu_loads']-b['gpu_loads']).max()))
        write_json(args.out/'gpu-numerics.json',results)
    if any(sha256(path)!=digest for path,digest in protected.items()):
        raise RuntimeError('Source model/reference/checkpoint/export changed')
    write_json(args.out/'report.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        wrist_equilibrium=wrist_report,predictive_controller=controller_report,gpu_numerics=results,
        production_inputs_unchanged=True,training_updates=0,arena_changed=False))
    print('Completed',args.out/'report.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--trace',type=Path,default=Path('logs/mimickit-training/stand-foundation-tests-20261001-03/final_actor-trace.npz'))
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    arguments=parser.parse_args()
    try:
        run(arguments)
    except Exception:
        if arguments.out.is_dir():
            write_json(arguments.out/'failure.json',dict(error=traceback.format_exc()))
        raise

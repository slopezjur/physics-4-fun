"""Sixty-second native/GPU admission of the isolated predictive Stand controller."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, write_json
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController


def gpu_native_state(task):
    """Read live GPU state, converting free-joint angular velocity to native local axes."""
    engine = task.engine
    qpos = engine.native_qpos().detach().cpu().numpy().astype(np.float64)
    linear = engine.get_root_vel(0).detach().cpu().numpy()
    angular = engine.get_root_ang_vel(0).detach().cpu().numpy()
    hinges = engine.get_dof_vel(0).detach().cpu().numpy()
    qvel = np.empty((len(qpos), task.rig.model.nv))
    qvel[:, :3], qvel[:, 6:] = linear, hinges
    for world, q in enumerate(qpos):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q[3:7])
        qvel[world, 3:6] = rotation.reshape(3, 3).T @ angular[world]
    return qpos, qvel


@torch.no_grad()
def admit(rig, reference, backend, seconds, out):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9',
                         episode_seconds=seconds, **options)
    task.set_mode(EnvMode.TEST)
    task.reset()
    reader = None if backend == 'native' else lambda: gpu_native_state(task)
    controller = PredictiveStandController(task, state_reader=reader)
    active = np.ones(8, dtype=bool)
    samples = {key: [] for key in ('qpos', 'qvel', 'command', 'done', 'alive_before',
        'balanced', 'posture', 'solver_warning', 'foot_loads_n', 'pelvis_tilt_degrees',
        'horizontal_speed_m_s', 'angular_speed_rad_s', 'non_foot_contact_max_n')}
    wall_start = time.perf_counter()
    count = int(np.ceil(seconds / task.dt))
    report_every = max(1, int(np.ceil(5. / task.dt)))
    for step in range(count):
        command = controller.command(active).to(device)
        if backend == 'native':
            task.engine.active = active.copy()
        _, _, done, _ = task.step(command)
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(warnings, device=device)] = DoneFlags.FAIL.value
        state = task.standing_measurements()
        balanced = torch.stack(list(task.balance_criteria(state).values())).all(0)
        posture = torch.stack(list(task.posture_criteria(state).values())).all(0)
        qpos, qvel = controller.state_reader()
        values = dict(qpos=qpos, qvel=qvel, command=command.cpu().numpy(), done=done.cpu().numpy(),
            alive_before=active.copy(), balanced=balanced.cpu().numpy(), posture=posture.cpu().numpy(),
            solver_warning=warnings, foot_loads_n=state['foot_loads_n'].cpu().numpy(),
            pelvis_tilt_degrees=state['pelvis_tilt_degrees'].cpu().numpy(),
            horizontal_speed_m_s=state['horizontal_speed_m_s'].cpu().numpy(),
            angular_speed_rad_s=state['angular_speed_rad_s'].cpu().numpy(),
            non_foot_contact_max_n=state['non_foot_ground_contacts_n'].amax(-1).cpu().numpy())
        for key, value in values.items():
            samples[key].append(value.copy())
        active &= done.cpu().numpy() == DoneFlags.NULL.value
        if (step + 1) % report_every == 0 or not active.any() or step + 1 == count:
            progress = dict(backend=backend, simulated_seconds=(step + 1)*task.dt,
                live_cases=int(active.sum()), wall_seconds=time.perf_counter()-wall_start)
            write_json(out/'progress.json', progress)
            print(progress, flush=True)
        if not active.any():
            break
    arrays = {key: np.stack(values) for key, values in samples.items()}
    arrays['time_s'] = (np.arange(len(arrays['done'])) + 1)*task.dt
    np.savez_compressed(out/'trace.npz', **arrays)
    cases = []
    motor_qadr = rig.model.jnt_qposadr[rig.model.actuator_trnid[rig.actuators, 0]]
    for case in range(8):
        live = arrays['alive_before'][:, case]
        end = np.flatnonzero(live)[-1]
        survived = arrays['done'][end, case] == DoneFlags.TIME.value
        final = live & (arrays['time_s'] > arrays['time_s'][end] - 2.)
        complete_window = arrays['time_s'][end] >= 2.
        balance = bool(complete_window and arrays['balanced'][final, case].all())
        posture = bool(complete_window and arrays['posture'][final, case].all())
        errors = np.rad2deg(np.abs(arrays['qpos'][final, case][:, motor_qadr] - controller.target_qpos[case, motor_qadr]))
        cases.append(dict(case=case, survived=bool(survived), balance_pass=bool(survived and balance),
            standing_pass=bool(survived and balance and posture), duration_seconds=float(arrays['time_s'][end]),
            pelvis_height_end_m=float(arrays['qpos'][end, case, 2]),
            final_foot_loads_n=arrays['foot_loads_n'][end, case].tolist(),
            motor_joint_error_final_mean_degrees=float(errors.mean()),
            motor_joint_error_final_max_degrees=float(errors.max()),
            command_saturation_fraction=float((np.abs(arrays['command'][live, case]) >= .98).mean()),
            solver_warning=bool(arrays['solver_warning'][end, case]),
            balance_violation_seconds=float((~arrays['balanced'][live, case]).sum()*task.dt),
            posture_violation_seconds=float((~arrays['posture'][live, case]).sum()*task.dt)))
    report = dict(backend=backend, requested_seconds=seconds, model_sha256=sha256(rig.path),
        reference_sha256=sha256(reference), cases=cases, survivors=sum(c['survived'] for c in cases),
        balance_passes=sum(c['balance_pass'] for c in cases), standing_passes=sum(c['standing_pass'] for c in cases),
        wall_seconds=time.perf_counter()-wall_start,
        controller='Unchanged one-Hz predictive motor QP; native private prediction on both live backends',
        external_forces=False, root_anchoring=False, passive_springs=False, training_updates=0,
        gpu_warning_scope='GPU admission checks finite state and physical gates; native MuJoCo warning counters do not exist here')
    write_json(out/'report.json', report)
    return report


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    torch.manual_seed(301010)
    np.random.seed(301010)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(args.model, policy_exclude=())
    protected = {path: sha256(path) for path in (args.model, args.reference)}
    write_json(args.out/'protocol.json', dict(seconds=args.seconds, seed=301010, torch_threads=2,
        backends=['native', 'gpu'], inputs={str(p): h for p, h in protected.items()},
        admission_source_sha256=sha256(Path(__file__)),
        controller_source_sha256=sha256(Path('mujoco_rig/mimic/stand_foundation_followup.py'))))
    reports = {}
    for backend in ('native', 'gpu'):
        out = args.out/backend
        out.mkdir()
        reports[backend] = admit(rig, args.reference, backend, args.seconds, out)
        write_json(args.out/'reports.json', reports)
    if any(sha256(path) != digest for path, digest in protected.items()):
        raise RuntimeError('Protected physics/reference input changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        reports=reports, inputs_unchanged=True, training_updates=0, arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('logs/mimickit-training/stand-fresh-winner-60min-20261001-01/dummy.xml'))
    parser.add_argument('--reference', type=Path, default=Path('logs/mimickit-training/stand-foundation-followup-20261001-01/passive-wrist-reference.npz'))
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise

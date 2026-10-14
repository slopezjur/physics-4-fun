"""Force the dynamic teacher to hold standing under native gravity, without impacts."""
import numpy as np
import torch

from .capture_step_teacher import bridge_source_action
from .continuous_stand import evaluate_retention
from .dynamic_support_teacher import DynamicSupportTeacher
from .recovery_motor import authority_gate


def evaluate_support(task, actor, *, contact_damping=18., contact_tracking_weight=0., all_motors=False,
                     actuation='target_pd', force_model='independent_rays', balance_tracking_weight=1.):
    if actuation not in ('target_pd','direct_physics_torque'):
        raise ValueError('Unknown diagnostic actuation mode')
    teacher = DynamicSupportTeacher(task, actor, contact_damping=contact_damping,
                                    contact_tracking_weight=contact_tracking_weight,all_motors=all_motors,
                                    force_model=force_model,balance_tracking_weight=balance_tracking_weight)
    anchors = rotations = centers = roots = None
    planned_torque, actual_torque, planned_acceleration, actual_acceleration, planned_load = [], [], [], [], []

    @torch.no_grad()
    def policy(observation):
        nonlocal anchors, rotations, centers, roots
        task.shot_enabled.zero_(); task.second_shot_enabled.zero_()
        if anchors is None:
            anchors = task.engine.get_body_pos(0)[:, task.feet].numpy().copy()
            centers = task.center_of_mass().numpy().copy()
            roots = task.engine.native_qpos()[:, :3].numpy().copy()
            rotations = np.array([[data.xmat[body].reshape(3, 3).copy() for body in teacher.body]
                                  for data in task.engine.datas])
        action = bridge_source_action(task, actor(observation), observation)
        gate = authority_gate(observation).numpy()
        qpos = task.engine.native_qpos().numpy()
        reference, _ = task.sample_reference(task.offset)
        reference = reference.numpy()
        control = task.engine.policy_control
        actual_torque.append(np.array([data.qfrc_actuator[teacher.dofs].copy() for data in task.engine.datas]))
        actual_acceleration.append(np.array([data.qacc[:teacher.model.nv].copy() for data in task.engine.datas]))
        for world in range(len(action)):
            radius = np.where(control.leg_mask[teacher.motors].numpy(),
                              .25 + (control.maximum_radians - .25) * gate[world], .25)
            source = reference[world, teacher.indices] + action[world, teacher.motors].numpy() * radius
            targets = teacher.command_targets(world, qpos[world], reference[world], source, radius,
                anchors[world], rotations[world], 1, centers[world], roots[world])
            action[world, teacher.motors] = torch.from_numpy(
                (targets - reference[world, teacher.indices]) / radius).float()
        planned_torque.append(np.array([s['torque'] if s else np.full(len(teacher.motors), np.nan)
                                       for s in teacher.last_by_world]))
        planned_acceleration.append(np.array([s['acceleration'] if s else np.full(teacher.model.nv, np.nan)
                                             for s in teacher.last_by_world]))
        planned_load.append(np.array([[sum(s['coefficients'][s['contact_feet'] == f]) * teacher.weight
                                      for f in (0, 1)] if s else [np.nan, np.nan] for s in teacher.last_by_world]))
        return action.clamp(-1, 1)

    control = task.engine.policy_control
    original_physics_control = control.physics_control

    def direct_physics_control(q, v):
        original_physics_control(q, v)
        reference, _ = task.sample_reference(task.offset)
        reference = reference.numpy()
        for world, live in enumerate(task.engine.datas):
            radius = np.where(control.leg_mask[teacher.motors].numpy(),
                              .25 + (control.maximum_radians - .25) * float(control.gate[world]), .25)
            source = (control.active[world,teacher.motors].numpy() if control.active[world,60]
                      else reference[world,teacher.indices])
            teacher.command_targets(world, live.qpos[:teacher.model.nq].copy(), reference[world], source,
                radius, anchors[world], rotations[world], 1, centers[world], roots[world])
            solved = teacher.last_by_world[world]
            if solved is not None:
                control.ctrl[world,control.indices[teacher.motors]] = torch.from_numpy(solved['torque']).float()
        return control.ctrl

    try:
        if actuation == 'direct_physics_torque':
            control.physics_control = direct_physics_control
        report, trace = evaluate_retention(task, policy)
    finally:
        control.physics_control = original_physics_control
    report.update(teacher_contract=teacher.contract(), teacher_diagnostics=teacher.diagnostics(),
                  scope='Dynamic controller forced active during gravity-only standing; '
                        'motor selection is explicit in teacher contract, no root writes or external stabilizing forces')
    trace.update(planned_torque=np.array(planned_torque),realized_torque=np.array(actual_torque),
                 planned_acceleration=np.array(planned_acceleration),realized_acceleration=np.array(actual_acceleration),
                 planned_foot_normal_load=np.array(planned_load))
    report['diagnostic_actuation'] = actuation
    report['deployable_motor_contract'] = actuation == 'target_pd'
    if actuation == 'direct_physics_torque':
        report['teacher_contract']['target_conversion'] = 'Diagnostic bypass: bounded torques refreshed at each physics step'
        report['teacher_contract']['latency'] = 'No PD target transport in this diagnostic'
        report['motion_prior_admission_allowed'] = False
    return report, trace

"""Deterministic rig, sensor and elementary support probes; no PPO or deployment."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from .baseline import sha256
from .model_contract import collision_pairs
from .native_engine import NativeDummyEngine
from .rig import Rig
from .static_support import solve_support
from .target_control import DelayedTargetControl


def structure(rig):
    model, data = rig.model, mujoco.MjData(rig.model)
    data.qpos[:] = rig.rest
    mujoco.mj_forward(model, data)
    inertias = model.body_inertia[1:]
    principal = np.sort(inertias, axis=1)
    bodies = []
    for i in range(1, model.nbody):
        bodies.append(dict(name=model.body(i).name, mass_kg=float(model.body_mass[i]),
                           inertia_kg_m2=inertias[i-1].tolist(), position_m=data.xpos[i].tolist()))
    actuated = set(model.actuator_trnid[rig.actuators, 0])
    joints = []
    for i in range(1, model.njnt):
        qi, vi = model.jnt_qposadr[i], model.jnt_dofadr[i]
        joints.append(dict(name=model.joint(i).name, axis_world=data.xaxis[i].tolist(),
                           pivot_world_m=data.xanchor[i].tolist(), limits_rad=model.jnt_range[i].tolist(),
                           rest_rad=float(rig.rest[qi]), policy_controlled=i in actuated,
                           stiffness_nm_rad=float(model.jnt_stiffness[i]),
                           damping_nm_s_rad=float(model.dof_damping[vi])))
    symmetry = []
    for body in range(1, model.nbody):
        name = model.body(body).name
        if not name.endswith("_L"):
            continue
        other = model.body(name[:-2] + "_R").id
        symmetry.append(dict(pair=name[:-2], mass_difference_kg=float(model.body_mass[body]-model.body_mass[other]),
                             mirrored_position_error_m=float(np.linalg.norm(data.xpos[body]*[1,-1,1]-data.xpos[other])),
                             principal_inertia_difference=float(np.abs(model.body_inertia[body]-model.body_inertia[other]).max())))
    checks = dict(finite=bool(np.isfinite(model.body_mass).all() and np.isfinite(inertias).all()),
                  positive_mass=bool((model.body_mass[1:] > 0).all()),
                  positive_inertia=bool((inertias > 0).all()),
                  inertia_triangle=bool((principal[:,2] <= principal[:,:2].sum(axis=1)+1e-10).all()),
                  rest_within_limits=bool(((rig.rest[7:] >= model.jnt_range[1:,0]) &
                                           (rig.rest[7:] <= model.jnt_range[1:,1])).all()))
    return dict(checks=checks, body_mass_kg=float(model.body_mass.sum()),
                center_of_mass_m=data.subtree_com[1].tolist(), bodies=bodies, joints=joints,
                bilateral_symmetry=symmetry,
                rest_contacts=[dict(geoms=[model.geom(int(g)).name for g in c.geom], distance_m=float(c.dist))
                               for c in data.contact[:data.ncon]])


def fit_pose(rig, foot_offsets, com_offset, joint_window=None):
    """Find a bounded pose without mocap or policy weights; this is not a controller."""
    model, data = rig.model, mujoco.MjData(rig.model)
    data.qpos[:] = rig.rest
    mujoco.mj_forward(model, data)
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    foot_targets = data.xpos[feet].copy() + foot_offsets
    rotation_targets = data.xmat[feet].copy()
    com_target = data.subtree_com[1,:2].copy() + com_offset
    joints = model.actuator_trnid[rig.actuators,0]
    qi = model.jnt_qposadr[joints]
    pairs = sorted(collision_pairs(model))
    initial = np.r_[rig.rest[:3], rig.rest[qi]]

    def residual(x):
        data.qpos[:] = rig.rest
        data.qpos[:3], data.qpos[qi] = x[:3], x[3:]
        mujoco.mj_kinematics(model,data)
        mujoco.mj_comPos(model,data)
        distance = [mujoco.mj_geomDistance(model,data,a,b,.01,None) for a,b in pairs]
        return np.r_[500*(data.xpos[feet]-foot_targets).ravel(),
                     10*(data.xmat[feet]-rotation_targets).ravel(),
                     200*(data.subtree_com[1,:2]-com_target),
                     8*(x[2]-rig.rest[2]), .5*(x[3:]-initial[3:]),
                     1000*np.minimum(distance,0)]

    lower = np.r_[rig.rest[:3] - [.3,.3,.25],model.jnt_range[joints,0]+1e-5]
    upper = np.r_[rig.rest[:3] + [.3,.3,.1],model.jnt_range[joints,1]-1e-5]
    if joint_window is not None:
        lower[3:] = np.maximum(lower[3:],initial[3:]-joint_window)
        upper[3:] = np.minimum(upper[3:],initial[3:]+joint_window)
    # A straight leg has a singular vertical foot Jacobian. Also try a bent-knee
    # seed before interpreting a failed local IK solve as a mechanical restriction.
    bent = initial.copy()
    for side in ("L", "R"):
        for name, value in ((f"Thigh_{side}_rx", .2), (f"Shin_{side}_rx", -.4), (f"Foot_{side}_rx", .2)):
            bent[3 + list(joints).index(model.joint(name).id)] = value
    fits = [least_squares(residual,seed.clip(lower,upper),bounds=(lower,upper),max_nfev=180)
            for seed in (initial,bent)]
    fit = min(fits,key=lambda value:value.cost)
    residual(fit.x)
    pose = data.qpos.copy()
    mujoco.mj_forward(model,data)
    return pose, dict(optimizer_converged=bool(fit.success), initializations=len(fits),
                      max_foot_error_m=float(np.linalg.norm(data.xpos[feet]-foot_targets,axis=1).max()),
                      com_error_m=float(np.linalg.norm(data.subtree_com[1,:2]-com_target)),
                      max_penetration_m=max([0., *[-float(c.dist) for c in data.contact[:data.ncon]]]),
                      max_joint_departure_from_rest_rad=float(np.abs(pose[qi]-rig.rest[qi]).max()))


def pose_cases(rig):
    cases = [("center",np.zeros((2,3)),np.zeros(2),("Foot_L","Foot_R","Toe_L","Toe_R"))]
    for side, sign in (("left",-1),("right",1)):
        cases.append(("shift_"+side,np.zeros((2,3)),np.array([0.,sign*.10]),cases[0][3]))
        support = ("Foot_L","Toe_L") if side == "left" else ("Foot_R","Toe_R")
        swing = 1 if side == "left" else 0
        for direction, distance in (("single_support",0.),("step_forward",.15),("step_backward",-.15)):
            offsets = np.zeros((2,3)); offsets[swing] = [distance,0,.06]
            cases.append((direction+"_"+side,offsets,np.array([.03,sign*.155]),support))
    rows, poses, feedforward = [], [], []
    for name, offsets, com, support_names in cases:
        pose, fit = fit_pose(rig,offsets,com)
        _, bounded_fit = fit_pose(rig,offsets,com,DelayedTargetControl.residual_radians)
        pose, passive = settle_passive_joints(rig, pose)
        try:
            support = solve_support(rig,pose,passive_tolerance=.05,minimum_norm=True,support_names=support_names)
        except (ValueError,RuntimeError) as error:
            support = dict(feasible=False,solver_message=str(error))
        rows.append(dict(name=name,fit=fit,rest_centered_action_window_fit=bounded_fit,
                         passive_equilibrium=passive,static_support=support,support_bodies=list(support_names)))
        poses.append(pose)
        feedforward.append(support.get("normalized_motor_torque",np.zeros(30)))
    return rows,np.array(poses),np.array(feedforward)


def settle_passive_joints(rig, pose):
    """Solve neck/wrist equilibrium at zero controls within their existing limits."""
    model,data=rig.model,mujoco.MjData(rig.model)
    controlled=set(model.actuator_trnid[rig.actuators,0])
    joints=np.array([i for i in range(1,model.njnt) if i not in controlled])
    qi,vi=model.jnt_qposadr[joints],model.jnt_dofadr[joints]
    data.qpos[:]=pose
    def residual(values):
        data.qpos[qi]=values
        mujoco.mj_forward(model,data)
        return (data.qfrc_bias-data.qfrc_passive-data.qfrc_actuator)[vi]
    fit=least_squares(residual,pose[qi],bounds=(model.jnt_range[joints,0]+1e-6,
                                             model.jnt_range[joints,1]-1e-6),max_nfev=100)
    error=float(np.abs(residual(fit.x)).max())
    return data.qpos.copy(),dict(max_generalized_force_residual_nm=error,
                                max_joint_adjustment_rad=float(np.abs(fit.x-pose[qi]).max()),
                                max_penetration_m=max([0., *[-float(c.dist) for c in data.contact[:data.ncon]]]),
                                tolerance_nm=.05,passed=bool(error<=.05))


def hold_pose(rig, pose, equilibrium_torque, seconds=10.):
    """Two independent holds using actual delayed target PD, optionally static feedforward.

    Starts at the fitted pose with zero velocity. Success does not demonstrate a
    transition, stepping, or a policy that can generate these targets.
    """
    engine = NativeDummyEngine(rig,2,"cpu",DelayedTargetControl)
    q=torch.tensor(np.tile(pose,(2,1)),dtype=torch.float32)
    v=torch.zeros((2,rig.model.nv))
    engine.reset_envs(torch.arange(2),q,v)
    c=engine.policy_control
    active=np.ones(2,dtype=bool); durations=np.zeros(2); reasons=[None,None]
    maximum_error=np.zeros(2); saturation=np.zeros(2); counts=np.zeros(2)
    allowed=[rig.model.body(n).id-1 for n in ("Foot_L","Foot_R","Toe_L","Toe_R")]
    # Add only the fixed equilibrium torque to diagnostic world 1, after the
    # production startup delay. World 0 is the unmodified controller.
    original=c.physics_control
    def control(qnow,vnow):
        result=original(qnow,vnow)
        result[1,c.indices]=(result[1,c.indices]+torch.tensor(equilibrium_torque,dtype=torch.float32)*c.scale*c.active[1,60]).clamp(-c.scale,c.scale)
        for world in np.flatnonzero(active):
            saturation[world]+=float((result[world,c.indices].abs()>=.99*c.scale).float().mean())
            counts[world]+=1
        return result
    c.physics_control=control
    for _ in range(round(seconds/engine.get_timestep())):
        c.set_reference(q,v);engine.set_cmd(0,torch.zeros(2,30));engine.step()
        forces=engine.get_ground_contact_forces(0).numpy();forces[:,allowed]=0
        for i in np.flatnonzero(active):
            data=engine.datas[i];durations[i]+=engine.get_timestep()
            maximum_error[i]=max(maximum_error[i],float(np.linalg.norm(data.qpos[:3]-pose[:3])))
            reason=("nonfinite" if not np.isfinite(data.qpos).all() else "pelvis_height" if data.qpos[2]<.65
                    else "nonfoot_contact" if np.abs(forces[i]).max()>1 else None)
            if reason:active[i]=False;reasons[i]=reason
        if not active.any():break
    return [dict(mode=mode,survived=bool(active[i]),seconds=float(durations[i]),failure=reasons[i],
                 max_root_drift_m=float(maximum_error[i]),saturated_motor_fraction=float(saturation[i]/max(1,counts[i])))
            for i,mode in enumerate(("production_pd_fixed_target","diagnostic_pd_static_feedforward"))]


def sensor_poses(rig):
    model=rig.model
    data=mujoco.MjData(model)
    poses=[]
    for pitch in (-.15,0.,.15):
        pose=rig.rest.copy();pose[3:7]=[np.cos(pitch/2),0,np.sin(pitch/2),0]
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        soles=[model.geom("g_"+n).id for n in ("Foot_L","Foot_R","Toe_L","Toe_R")]
        gap=min(mujoco.mj_geomDistance(model,data,0,g,.5,None) for g in soles)
        pose[2]-=gap+.0002
        poses.append(pose)
    return np.array(poses)


def sensor_probe(rig):
    engine=NativeDummyEngine(rig,1,"cpu",DelayedTargetControl)
    model=rig.model
    rows=[]
    for pitch,pose in zip((-.15,0.,.15),sensor_poses(rig)):
        engine.reset_envs(torch.tensor([0]),torch.tensor(pose[None],dtype=torch.float32),torch.zeros(1,model.nv))
        data=engine.datas[0];forces=engine.get_ground_contact_forces(0)[0].numpy()
        loads={n:float(forces[model.body(n).id-1,2]) for n in ("Foot_L","Foot_R","Toe_L","Toe_R")}
        full=sum(loads.values());observed=loads["Foot_L"]+loads["Foot_R"]
        rows.append(dict(root_pitch_rad=pitch,loads_n=loads,actor_observed_load_n=observed,
                         complete_sole_load_n=full,omitted_fraction=(full-observed)/full if full>0 else None,
                         force_balance_error_n=float(np.linalg.norm(forces.sum(axis=0)-data.qfrc_constraint[:3]))))
    return rows


def actuator_probe(rig):
    """Check all torque transmissions and response signs in an airborne fixture."""
    model,data=rig.model,mujoco.MjData(rig.model)
    data.qpos[:]=rig.rest;data.qpos[2]+=2
    mujoco.mj_forward(model,data)
    base_force=data.qfrc_actuator.copy();base_acceleration=data.qacc.copy()
    rows=[]
    for index,actuator in enumerate(rig.actuators):
        dof=model.jnt_dofadr[model.actuator_trnid[actuator,0]]
        data.ctrl[:]=0;data.ctrl[actuator]=rig.action_scale[index]*.1
        mujoco.mj_forward(model,data)
        expected=np.zeros(model.nv);expected[dof]=data.ctrl[actuator]
        error=np.abs(data.qfrc_actuator-base_force-expected).max()
        rows.append(dict(actuator=model.actuator(int(actuator)).name,
                         generalized_force_error_nm=float(error),
                         positive_acceleration_response=bool(data.qacc[dof]>base_acceleration[dof]),
                         full_authority_nm=float(rig.action_scale[index])))
    return rows


def audit(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    rig=Rig.load(Path(__file__).resolve().parents[1]/"dummy.xml")
    poses,states,ff=pose_cases(rig)
    for row,pose,torque in zip(poses,states,ff):
        row["holds"]=hold_pose(rig,pose,torque)
        print(json.dumps(row),flush=True)
    np.savez_compressed(out/"poses.npz",qpos=states,normalized_static_torque=ff)
    report=dict(schema="mimic_foundation_audit_v1",model_sha256=sha256(rig.path),
                controller=DelayedTargetControl(rig,1,"cpu").contract(),structure=structure(rig),
                pose_probes=poses,sensors=sensor_probe(rig),actuators=actuator_probe(rig),
                scope="Native MuJoCo diagnostics. IK and static support are necessary capacity checks, not learned/dynamic stepping. Holds start directly at each pose; optional equilibrium feedforward is diagnostic only. Passive neck/wrist equilibrium is fitted within original limits and support requires residual <=0.05 Nm.")
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    audit(parser.parse_args().out)

"""Compare exact delayed PD execution with a constant-torque local approximation."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import save, digest


def pd_torque(controller, data, target):
    control = controller.task.engine.policy_control
    q = data.qpos[controller.qi].astype(np.float32)
    v = data.qvel[controller.vi].astype(np.float32)
    value = control.gain*(target.astype(np.float32)-q)-control.gain*control.damping_time*v
    return np.clip(value,-1,1)*controller.scale32


def rollout(controller, initial, target, *, constant=False):
    """One control interval on private state; the live motor path is unchanged."""
    model = controller.model
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data,model,initial)
    torque = pd_torque(controller,data,target)
    torques=[]
    for _ in range(controller.task.rig.decimation):
        if not constant:
            torque = pd_torque(controller,data,target)
        data.ctrl[:] = 0
        data.ctrl[controller.actuators] = torque
        torques.append(torque.copy())
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    return data,np.array(torques)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from . import run_single_support_feedback as runner
    from .predictive_target_support import PredictiveTargetSupport
    original=PredictiveTargetSupport.target
    samples=[];seen=set();times=(2.,6.5,7.,7.5,8.)
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'protocol.json',dict(question='Does the private delayed target rollout match live execution, and does constant torque approximate the held-PD interval?',
        sample_issue_times_seconds=times,worlds=[0,1,8,9],native_cases=4,training_updates=0,
        qpos_tolerance=3e-6,qvel_tolerance=3e-5,
        torque_branch='Counterfactual private constant initial torque; never used by live policy',
        input_runs=['single-support-feedback-20261007-02','single-support-feedback-20261007-03'],
        source_sha256=digest(Path(__file__))))

    def audited(self,world,goal,source,radius=1.):
        target,info=original(self,world,goal,source,radius)
        now=float(self.task.steps[world])*self.task.dt
        wanted=next((t for t in times if now>=t and (world,t) not in seen),None)
        if wanted is None:
            return target,info
        seen.add((world,wanted))
        live=self.task.engine.datas[world]
        before=(live.qpos.copy(),live.qvel.copy(),self.task.engine.policy_control.pending.clone())
        predicted=self.predict(world)
        q,v=predicted.qpos.copy(),predicted.qvel.copy()
        tracking,desired=self.tracking_problem(predicted,goal)
        pd,pd_torques=rollout(self,predicted,target)
        fixed,fixed_torques=rollout(self,predicted,target,constant=True)
        acceleration_pd=(pd.qvel-v)/self.task.dt
        acceleration_fixed=(fixed.qvel-v)/self.task.dt
        samples.append(dict(world=world,issue_frame=int(self.task.steps[world]),issue_time=now,
            predicted_qpos=pd.qpos.copy(),predicted_qvel=pd.qvel.copy(),
            report=dict(world=world,issue_frame=int(self.task.steps[world]),issue_time=now,
                max_torque_change_fraction=float(np.max(abs(pd_torques-pd_torques[0])/self.scale)),
                rms_torque_change_fraction=float(np.sqrt(np.mean(((pd_torques-pd_torques[0])/self.scale)**2))),
                weighted_velocity_change_error_pd=float(np.linalg.norm(tracking@acceleration_pd-desired)),
                weighted_velocity_change_error_constant=float(np.linalg.norm(tracking@acceleration_fixed-desired)),
                constant_vs_pd_qvel_difference=float(np.max(abs(pd.qvel-fixed.qvel))),
                instantaneous_optimized_error=info['tracking_residual'])))
        assert np.array_equal(before[0],live.qpos) and np.array_equal(before[1],live.qvel)
        assert torch.equal(before[2],self.task.engine.policy_control.pending)
        return target,info

    argv=[sys.argv[0],'--release-gate','capture','--objective','pose',
        '--quiet-evidence','logs/mimickit-training/single-support-feedback-20261007-02',
        '--equilibrium-evidence','logs/mimickit-training/single-support-feedback-20261007-03',
        '--out',str(args.out/'run')]
    with patch.object(sys,'argv',argv),patch.object(PredictiveTargetSupport,'target',audited):
        runner.main()
    with np.load(args.out/'run/entry-feedback/trace.npz') as z:
        for sample in samples:
            index=[0,1,8,9].index(sample['world'])
            end=sample['issue_frame']+2
            assert end<len(z['time']) and z['alive'][end,index]
            report=sample['report']
            report['live_qpos_error']=float(np.max(abs(sample['predicted_qpos']-z['qpos'][end,index])))
            report['live_qvel_error']=float(np.max(abs(sample['predicted_qvel']-z['qvel'][end,index])))
            report['forecast_pass']=report['live_qpos_error']<3e-6 and report['live_qvel_error']<3e-5
    reports=[s['report'] for s in samples]
    assert len(reports)==20
    report=dict(completed=True,samples=reports,all_forecasts_pass=all(r['forecast_pass'] for r in reports),
        max_live_qpos_error=max(r['live_qpos_error'] for r in reports),max_live_qvel_error=max(r['live_qvel_error'] for r in reports),
        median_torque_rms_change=float(np.median([r['rms_torque_change_fraction'] for r in reports])),
        max_torque_change=max(r['max_torque_change_fraction'] for r in reports),
        median_constant_vs_pd_qvel_difference=float(np.median([r['constant_vs_pd_qvel_difference'] for r in reports])),
        extra_private_branch_substeps=len(reports)*8,live_physical_cases=4,training_updates=0,arena_changed=False)
    save(args.out/'report.json',report)
    save(args.out/'final-verification.json',dict(outputs={str(p.relative_to(args.out)):digest(p)
        for p in args.out.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print({k:v for k,v in report.items() if k!='samples'})


if __name__=='__main__':main()

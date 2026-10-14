"""Refine joint targets against their actual four-substep PD response."""
import mujoco
import numpy as np

from .predictive_target_support import PredictiveTargetSupport
from .inverse_contact_control import solve_box
from .audit_support_prediction import rollout


class IntervalTargetSupport(PredictiveTargetSupport):
    """One bounded target-space correction of the original pose controller."""

    def __init__(self, task, objective='interval'):
        objectives = {'interval': 'pose', 'interval_com': 'com', 'interval_task': 'task'}
        if objective not in objectives:
            raise ValueError('Expected interval objective')
        super().__init__(task,objective=objectives[objective])
        self.corrections = 0
        self.queries = 0
        self.maximum_cost_ratio = 0.

    def target(self, world, goal, source, radius=1.):
        initial_target,info = super().target(world,goal,source,radius)
        initial = self.predict(world)
        tracking,desired = self.tracking_problem(initial,goal)
        control = self.task.engine.policy_control
        lower = np.maximum(control.lower.numpy(),source-radius)
        upper = np.minimum(control.upper.numpy(),source+radius)
        initial_target = np.clip(initial_target,lower,upper)
        v0 = initial.qvel.copy()
        regularization = .15*control.gain

        def residual(target):
            end,_ = rollout(self,initial,target)
            self.private_steps += self.task.rig.decimation
            error = tracking@((end.qvel-v0)/self.task.dt)-desired
            return np.r_[error,regularization*(target-initial_target)]

        residual0 = residual(initial_target)
        if not np.isfinite(residual0).all():
            raise RuntimeError('Nonfinite interval objective')
        cost0 = float(residual0@residual0)
        jacobian = np.zeros((len(residual0),33))
        epsilon = 1e-4
        for index in range(33):
            step = min(epsilon,upper[index]-initial_target[index])
            if step < epsilon/2:
                step = -min(epsilon,initial_target[index]-lower[index])
            if abs(step)<1e-8:
                continue
            perturbed = initial_target.copy();perturbed[index]+=step
            jacobian[:,index]=(residual(perturbed)-residual0)/step
        hessian = jacobian.T@jacobian+np.eye(33)*1e-6
        linear = jacobian.T@residual0
        delta,rank = solve_box(hessian,linear,np.maximum(lower-initial_target,-.05),
            np.minimum(upper-initial_target,.05))
        best,best_residual,best_cost = initial_target,residual0,cost0
        if rank>=0 and np.isfinite(delta).all():
            for fraction in (1.,.5,.25,.125):
                candidate = np.clip(initial_target+fraction*delta,lower,upper)
                error = residual(candidate)
                cost = float(error@error)
                if np.isfinite(cost) and cost<best_cost-1e-9:
                    best,best_residual,best_cost = candidate,error,cost
                    self.corrections+=1
                    break
        assert best_cost<=cost0+1e-8
        self.queries+=1
        self.maximum_cost_ratio=max(self.maximum_cost_ratio,best_cost/max(cost0,1e-12))
        return best,dict(info,tracking_residual=float(np.linalg.norm(best_residual[:-33])),
            interval_initial_cost=cost0,interval_final_cost=best_cost,
            interval_target_change=float(np.max(abs(best-initial_target))))

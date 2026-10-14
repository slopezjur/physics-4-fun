"""Exact in-memory native recovery starts, including delayed control and history."""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch


ENV_FIELDS = ('previous', 'force', 'onset', 'history', 'last_latent')
TASK_FIELDS = ('steps', 'offset', 'previous_action', 'action_cost', 'previous_com')
CONTROL_FIELDS = ('ctrl', 'active', 'addition', 'reference_q', 'reference_v')
ENGINE_FIELDS = ('active', 'solver_warning', '_ground_force_cache')


@dataclass
class RecoveryStart:
    """Native data copies are process-local; no lossy qpos/qvel reconstruction."""
    model: object
    native: list
    env: dict
    task: dict
    control: dict
    engine: dict
    pending: torch.Tensor
    last_solve_contacts: bool

    @classmethod
    def capture(cls, env):
        engine, control = env.engine, env.engine.policy_control
        native = [mujoco.MjData(env.rig.model) for _ in engine.datas]
        for target, source in zip(native, engine.datas, strict=True):
            mujoco.mj_copyData(target, env.rig.model, source)
        tensors = lambda obj, keys: {key: getattr(obj, key).clone() for key in keys}
        pending = torch.stack([control.pending[(control.cursor+i) % 2].clone() for i in range(2)])
        return cls(env.rig.model, native, tensors(env, ENV_FIELDS), tensors(env.task, TASK_FIELDS),
            tensors(control, CONTROL_FIELDS), {k: getattr(engine, k).copy() for k in ENGINE_FIELDS},
            pending, engine._last_solve_contacts)

    def restore(self, env, ids=None):
        """Restore matching world indices; retain the live ring cursor and other worlds."""
        assert env.rig.model is self.model, 'Starts belong to the same native model instance'
        ids = env.task.ids if ids is None else torch.as_tensor(ids, dtype=torch.long)
        assert ids.ndim == 1 and len(ids.unique()) == len(ids)
        assert ((ids >= 0) & (ids < len(self.native))).all()
        engine, control = env.engine, env.engine.policy_control
        assert engine._last_solve_contacts == self.last_solve_contacts
        for i in ids.tolist():
            mujoco.mj_copyData(engine.datas[i], self.model, self.native[i])
        for obj, saved in ((env, self.env), (env.task, self.task), (control, self.control)):
            for key, value in saved.items():
                if not hasattr(obj, key):
                    assert len(ids) == len(self.native), 'Initialize reference arrays before a partial restore'
                    setattr(obj, key, value.clone())
                    continue
                getattr(obj, key)[ids] = value[ids]
        for i in range(2):
            control.pending[(control.cursor+i) % 2, ids] = self.pending[i, ids]
        for key, value in self.engine.items():
            getattr(engine, key)[ids.numpy()] = value[ids.numpy()]
        assert engine.native_ctrl is control.ctrl

    def arrays(self):
        """Portable inspection evidence; native exact restoration uses the copies above."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        native = np.zeros((len(self.native), mujoco.mj_stateSize(self.model, spec)))
        for row, data in zip(native, self.native, strict=True):
            mujoco.mj_getState(self.model, data, row, spec)
        result = dict(native_integration=native, pending_oldest_first=self.pending.numpy().copy())
        for prefix, values in (('env', self.env), ('task', self.task), ('control', self.control), ('engine', self.engine)):
            for key, value in values.items():
                result[f'{prefix}_{key}'] = value.numpy().copy() if isinstance(value, torch.Tensor) else value.copy()
        return result

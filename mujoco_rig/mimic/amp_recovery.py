"""Licensed motion histories for state-based recovery; no reference step clock."""
import json
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
import torch

from .amp_step import discriminator_state, load_demonstrations, reference_state
from .foot_motion import FootMotionCost

HISTORY_FRAMES = 4
DISC_PREFIXES = ("_model._disc", "_disc_obs_norm")


def actor_state_dict(state):
    """Keep the existing PPO deployment format; AMP is training-only."""
    return {key: value for key, value in state.items() if not key.startswith(DISC_PREFIXES)}


def validate_sidecar(portable, training):
    actor = actor_state_dict(training)
    if portable.keys() != actor.keys() or any(not torch.equal(value, actor[key])
                                             for key, value in portable.items()):
        raise ValueError("AMP sidecar does not contain the hash-bound portable actor")


def mirror_state(qpos, qvel, rig):
    """Reflect about the sagittal plane, swapping bilateral hinge coordinates."""
    key = (str(qpos.device), qpos.dtype)
    cache = getattr(rig, "_amp_mirror_maps", {})
    if key not in cache:
        qi, vi = np.arange(rig.model.nq), np.arange(rig.model.nv)
        qs, vs = np.ones(rig.model.nq), np.ones(rig.model.nv)
        qs[[1, 4, 6]], vs[[1, 3, 5]] = -1, -1
        for joint in range(1, rig.model.njnt):
            name = rig.model.joint(joint).name
            counterpart = name.replace("_L_", "_R_") if "_L_" in name else name.replace("_R_", "_L_")
            other = rig.model.joint(counterpart).id
            sign = 1. if name.endswith("_rx") else -1.
            axis = rig.model.jnt_axis[joint]
            if not np.allclose(axis, [0, -1, 0] if sign == 1 else
                               [0, 0, 1] if name.endswith("_ry") else [-1, 0, 0]):
                raise ValueError("Unreviewed hinge reflection axis")
            qi[rig.model.jnt_qposadr[joint]] = rig.model.jnt_qposadr[other]
            vi[rig.model.jnt_dofadr[joint]] = rig.model.jnt_dofadr[other]
            qs[rig.model.jnt_qposadr[joint]] = sign
            vs[rig.model.jnt_dofadr[joint]] = sign
        cache[key] = tuple(torch.as_tensor(value, device=qpos.device,
                            dtype=torch.long if index < 2 else qpos.dtype)
                           for index, value in enumerate((qi, vi, qs, vs)))
        rig._amp_mirror_maps = cache
    qi, vi, qs, vs = cache[key]
    return qpos[..., qi] * qs, qvel[..., vi] * vs


class MotionHistory:
    def __init__(self, state):
        self.values = state[:, None].repeat(1, HISTORY_FRAMES, 1)

    def reset(self, ids, state):
        self.values[ids] = state[ids, None]

    def append(self, state):
        self.values[:, 1:] = self.values[:, :-1].clone()
        self.values[:, 0] = state
        return self.values.flatten(1)


class AMPRecoveryMixin:
    """Add a discriminator interface without changing actor observations/rewards."""
    def __init__(self, rig, reference, num_envs, device, *, demonstrations,
                 demo_weights=None, mirror_demos=False, foot_motion_weight=0., **kwargs):
        self.demonstrations = load_demonstrations(demonstrations, rig, device)
        weights = demo_weights if demo_weights is not None else ([.5, .25, .25]
                      if len(self.demonstrations) == 3 else [1.] * len(self.demonstrations))
        self.demo_weights = torch.tensor(weights, device=device, dtype=torch.float32)
        if len(weights) != len(self.demonstrations) or not torch.isfinite(self.demo_weights).all() \
                or (self.demo_weights <= 0).any() or not 0 <= foot_motion_weight <= .1:
            raise ValueError("Require positive demonstration weights and bounded foot-motion weight")
        self.demo_weights /= self.demo_weights.sum()
        self.mirror_demos = mirror_demos
        self.foot_motion_weight = foot_motion_weight
        self._motion_cost = None
        self._demo_rng = torch.Generator(device=device).manual_seed(kwargs.get("seed", 0) + 1907)
        self._history = None
        width = rig.model.nq + rig.model.nv
        self._disc_obs_space = spaces.Box(-np.inf, np.inf, (HISTORY_FRAMES * width,), dtype=np.float32)
        super().__init__(rig, reference, num_envs, device, **kwargs)
        if any(ref.duration <= (HISTORY_FRAMES - 1) * self.dt for ref in self.demonstrations):
            raise ValueError("Demonstrations are shorter than the discriminator history")
        self._history = MotionHistory(self._live_disc_state())
        self._motion_cost = FootMotionCost(*self._foot_state(), self.dt)
        self._demo_times = {}
        for split in ("train", "heldout"):
            self._demo_times[split] = []
            lag = (HISTORY_FRAMES - 1) * self.dt
            for ref in self.demonstrations:
                time = torch.arange(lag, ref.duration, self.dt, device=device)
                # Separate complete .5 s blocks; no history straddles a split.
                block = (time / .5).long()
                inside = time - block * .5 >= lag + self.dt
                heldout = block % 5 == 4
                selected = time[inside & (heldout if split == "heldout" else ~heldout)]
                if not len(selected):
                    raise ValueError("Demonstration lacks non-overlapping calibration blocks")
                self._demo_times[split].append(selected)

    def _foot_state(self):
        loads = self.engine.get_ground_contact_forces(0)[:, self.allowed_ground, 2].abs()
        feet = torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), -1)
        return self.engine.get_body_pos(0)[:, self.feet], feet

    def _live_disc_state(self):
        engine = self.engine
        return discriminator_state(engine.native_qpos(), engine.get_root_vel(0),
                                   engine.get_root_ang_vel(0), engine.get_dof_vel(0))

    def get_disc_obs_space(self):
        return self._disc_obs_space

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        if self._history is not None:
            self._history.reset(self.ids if env_ids is None else env_ids, self._live_disc_state())
        if self._motion_cost is not None:
            self._motion_cost.reset(self.ids if env_ids is None else env_ids, *self._foot_state())
        return result

    def step(self, action):
        obs, reward, done, info = super().step(action)
        cost, acceleration, reversal = self._motion_cost.update(*self._foot_state())
        cost = cost * (self.engine.hit & self.shot_enabled)
        reward = reward - self.foot_motion_weight * cost
        self._diagnostics["foot_motion_cost"] = cost.mean()
        self._diagnostics["foot_acceleration_m_s2"] = acceleration.mean()
        self._diagnostics["rapid_contact_reversals"] = reversal.float().mean()
        history = self._history.append(self._live_disc_state())
        if history.shape[1:] != self._disc_obs_space.shape or not torch.isfinite(history).all():
            raise ValueError("Nonfinite or incompatible AMP history")
        info["disc_obs"] = history
        return obs, reward, done, info

    def fetch_disc_obs_demo(self, num_samples, split="train"):
        selection = torch.multinomial(self.demo_weights, num_samples, replacement=True, generator=self._demo_rng)
        result = torch.empty((num_samples, *self._disc_obs_space.shape), device=self.device)
        for index, ref in enumerate(self.demonstrations):
            ids = torch.nonzero(selection == index).flatten()
            if not len(ids):
                continue
            times = self._demo_times[split][index]
            picks = torch.randint(len(times), (len(ids),), generator=self._demo_rng, device=self.device)
            time = times[picks]
            mirror = torch.rand(len(ids), generator=self._demo_rng, device=self.device) < .5
            frames = []
            for frame in range(HISTORY_FRAMES):
                q, v = ref.sample(time - frame * self.dt)
                if self.mirror_demos:
                    reflected_q, reflected_v = mirror_state(q, v, self.rig)
                    q = torch.where(mirror[:, None], reflected_q, q)
                    v = torch.where(mirror[:, None], reflected_v, v)
                frames.append(reference_state(q, v))
            result[ids] = torch.cat(frames, -1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite AMP demonstration history")
        return result

    def discriminator_contract(self):
        return dict(schema="mimic_amp_recovery_history_v1", frames=HISTORY_FRAMES,
                    width=self._disc_obs_space.shape[0], control_dt=self.dt,
                    ordering="current post-step state then preceding control states; reset clears history",
                    demo_weights=self.demo_weights.tolist(), mirror_demos=self.mirror_demos,
                    calibration_split="every fifth .5s block held out, history stays inside each block",
                    foot_motion_cost=dict(weight=self.foot_motion_weight, **FootMotionCost.contract()),
                    features=["pelvis_height", "root_rotation_tangent_normal", "hinge_positions",
                              "root_linear_velocity_world", "root_angular_velocity_world", "hinge_velocities"],
                    demonstrations=[dict(path=str(ref.path), sha256=ref.sha256,
                        manifest=json.loads((ref.path.with_suffix(".json") if ref.path.with_suffix(".json").exists()
                                             else ref.path.parent / "report.json").read_text()))
                        for ref in self.demonstrations],
                    scope="Validated kinematic style/support examples; not recorded hit recovery or proof of dynamic tracking")


def amp_sidecar(path):
    path = Path(str(path).removesuffix(".tmp"))
    return path.with_suffix(".amp.pt")

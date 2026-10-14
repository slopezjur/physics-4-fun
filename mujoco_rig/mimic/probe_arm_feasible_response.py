"""Matched feasible-command interventions with unchanged physical bounds."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT
from .run_learned_handoff import PRIOR
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_hold_ppo import INITIAL, BASELINE
from .train_arm_hold_ppo_rate import DiagnosticRollout
from .target_control33 import DelayedTargetControl33
from .probe_arm_action_response import snapshot, fingerprint, reach

OUT = Path('logs/mimickit-training/arm-feasible-response-20261008-01')
PHASES = {'ramp': 90, 'hold': 210, 'return': 360}
BRANCHES = [('zero', 0., 0), ('plus-pulse', .005, 1), ('minus-pulse', -.005, 1),
            ('plus-held', .005, 12), ('minus-held', -.005, 12)]


class FeasibleCorrectionRollout(DiagnosticRollout):
    def __init__(self, env, anchor, poses):
        super().__init__(env, anchor, poses)
        self.correction = torch.zeros((16, 33))

    def reset(self, ids=None):
        super().reset(ids)
        if hasattr(self, 'correction'):
            self.correction[slice(None) if ids is None else ids] = 0

    @torch.no_grad()
    def step(self,action,active):
        from .target_control33 import DelayedTargetControl33
        from .run_single_support_feedback import flags
        x,goals,amount,quiet,frame=self.observation()
        c=self.control
        c.set_reference(*self.env.task.sample_reference(self.env.task.offset+self.env.task.steps*self.env.dt))
        source=DelayedTargetControl33.target_positions(c,self.anchor(x[:,:374]))
        raw=goals[:,self.qi]+action
        displacement=(goals[:,self.qi]-self.reference[:,self.qi]).abs().amax(-1)
        desired=(raw-source).clamp(-1,1)*amount[:,None]*(displacement/.05).clamp(0,1)[:,None]
        addition=self.previous+(desired-self.previous).clamp(-.02,.02)
        nominal = addition.clone()
        scaled_correction = self.correction * amount[:, None] * (displacement / .05).clamp(0, 1)[:, None]
        corrected = (self.previous + (nominal + scaled_correction - self.previous).clamp(-.02, .02)).clamp(-1, 1)
        addition = torch.where(self.correction != 0, corrected, nominal)
        addition[quiet]=0;addition[~active]=0;self.previous.copy_(addition)
        pre=self.env.measurements();prestand,presupport=flags(self.env,pre)
        self.env.engine.active=active.numpy().copy()
        _,_,done,_=self.env.step(addition)
        post=self.env.last;stand,support=flags(self.env,post)
        stand=torch.tensor(stand&prestand);support=torch.tensor(support&presupport)
        q=self.env.engine.native_qpos();issued=c.pending[(c.cursor-1)%2,:,:33]
        assert torch.isfinite(q[active]).all() and not self.env.engine.solver_warning[active.numpy()].any()
        assert (issued-(source+addition).clamp(c.lower,c.upper)).abs().max()<2e-6
        assert (c.ctrl[:,self.env.rig.actuators].abs()<=torch.tensor(self.env.rig.action_scale)+1e-4).all()
        target=goals[:,self.qi]
        error=(q[:,self.qi]-target).square().mean(-1)
        com_error=(post['com_m'][:,:2]-self.target_com[frame,self.ids,:2]).square().sum(-1)
        foot_error=(post['foot_xy_m']-self.initial_xy).square().sum(-1)
        reward=.45*torch.exp(-error/.05**2)+.20*torch.exp(-com_error/.02**2)+.10*torch.exp(-foot_error.mean(-1)/.02**2)+.25*support.float()
        end_time=(frame.double()+1)*self.env.dt
        holding=(end_time>3.5)&(end_time<=5.5)
        self.hold_ok &= ~holding | (support&(error<=.05**2)&(com_error<=.02**2)&(foot_error.amax(-1)<=.02**2))
        self.return_ok &= (end_time<=7.) | stand
        failed=done==1;horizon=frame+1>=self.frames
        terminal=failed|horizon
        success=horizon&~failed&self.hold_ok&self.return_ok
        reward[failed]=-2.;reward+=10.*success.float()
        corrected_success = torch.tensor(self.gate.step(frame.numpy(), q.numpy(), post['com_m'].numpy(),
            post['foot_xy_m'].numpy(), support.numpy(), stand.numpy(), failed.numpy()))
        reward = reward - 10 * success.float() + 10 * corrected_success.float()
        parts = torch.stack((.45 * torch.exp(-error / .05**2), .20 * torch.exp(-com_error / .02**2),
            .10 * torch.exp(-foot_error.mean(-1) / .02**2), .25 * support.float()), -1)
        failure = torch.where(failed, -2 - parts.sum(-1), 0)
        components = torch.cat((parts, failure[:, None], 10 * corrected_success.float()[:, None]), -1)
        broad_dense = .45 * torch.exp(-error / .10**2) + .20 * torch.exp(-com_error / .04**2) + parts[:, 2] + parts[:, 3]
        broad_reward = torch.where(failed, -2., broad_dense) + components[:, 5]
        assert torch.allclose(components.sum(-1)[active], reward[active], atol=1e-6, rtol=0)
        return reward,terminal,dict(failed=failed,success=corrected_success,issued=issued.clone(),qpos=q.clone(),
            supported=support,stand=stand,actor_active=~quiet&(amount>0),tracking_rmse=error.sqrt(),
            components=components,broad_reward=broad_reward,nominal_addition=nominal,quiet=quiet)


@torch.no_grad()
def continuation(adapter, net, active, motor, magnitude, duration):
    env, c = adapter.env, adapter.control
    rows = []
    for offset in range(60):
        x, goals, amount, quiet, frame = adapter.observation()
        model_action = net(x)
        action = model_action.clone()
        adapter.correction.zero_()
        if offset < duration:
            adapter.correction[active, motor] = magnitude
        c.set_reference(*env.task.sample_reference(env.task.offset + env.task.steps * env.dt))
        source = DelayedTargetControl33.target_positions(c, adapter.anchor(x[:, :374]))
        raw = goals[:, c.q_indices] + action
        fade = ((goals[:, c.q_indices] - adapter.reference[:, c.q_indices]).abs().amax(-1) / .05).clamp(0, 1)
        desired = (raw - source).clamp(-1, 1) * amount[:, None] * fade[:, None]
        previous = adapter.previous.clone()
        before = active.clone()
        reward, terminal, info = adapter.step(action, active)
        addition = previous + (desired - previous).clamp(-.02, .02)
        nominal = addition.clone()
        requested = nominal + adapter.correction * amount[:, None] * fade[:, None]
        projected = (previous + (requested - previous).clamp(-.02, .02)).clamp(-1, 1)
        addition = torch.where(adapter.correction != 0, projected, nominal)
        addition[quiet | ~active] = 0
        assert torch.equal(addition, adapter.previous)
        expected = (source + addition).clamp(c.lower, c.upper)
        assert torch.equal(expected, info['issued'])
        assert (env.engine.substep_torque.abs() <= c.scale + 1e-4).all()
        parts = info['components']
        broad_reward = info['broad_reward']
        values = dict(alive=before, qpos=info['qpos'], failed=info['failed'], supported=info['supported'],
            stand=info['stand'], goal=goals[:, c.q_indices], model_action=model_action, injected_action=action,
            raw_target=raw, source=source, authority=amount, fade=fade, desired_addition=desired,
            correction=adapter.correction.clone(),nominal_addition=nominal,quiet=quiet,
            previous_addition=previous, addition=addition, issued=info['issued'], applied=c.active[:, :33].clone(),
            applied_velocity=c.active[:, 33:66].clone(), reward=reward, broad_reward=broad_reward,
            components=parts, tracking_rmse=info['tracking_rmse'])
        row = {k: v[:8].numpy().copy() for k, v in values.items()}
        for key in ('q', 'v', 'torque'):
            row['substep_' + key] = getattr(env.engine, 'substep_' + key)[:, :8].numpy().copy()
        rows.append(row)
        active &= ~terminal
    return {k: np.stack([row[k] for row in rows]) for k in rows[0]}


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    (OUT / 'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('ARM_FEASIBLE_RESPONSE_PLAN.md')]
    inputs = {str(p): digest(p) for p in paths + [PARENT, PARENT.with_suffix('.amp.pt'), INITIAL, BASELINE,
        PRIOR / 'goals.npz', FOUNDATION / 'dummy.xml', FOUNDATION / 'stationary_reference.npz']}
    for p in paths:
        shutil.copy2(p, OUT / 'sources' / p.name)
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION / 'stationary_reference.npz', json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)
    env = make_environment(rig, anchor, 40.)
    engine = env.engine
    engine.record_substeps = True
    shape = (rig.decimation, 16, 33)
    engine.substep_q, engine.substep_v, engine.substep_torque = (torch.empty(shape) for _ in range(3))
    state = torch.load(INITIAL, weights_only=True)
    net = QueueBlindLearner(state['mean'], state['scale'], state['output_scale'])
    net.load_state_dict(state)
    net.eval()
    poses = np.load(PRIOR / 'goals.npz')['poses']
    adapter = FeasibleCorrectionRollout(env, anchor, poses)
    joint = rig.model.joint('UpperArm_L_ry').id
    motor = int(np.flatnonzero(adapter.qi.numpy() == rig.model.jnt_qposadr[joint])[0])
    save(OUT / 'protocol.json', dict(inputs=inputs, phases=PHASES, branches=BRANCHES, frames=60,
        dt=env.dt, physics_dt=rig.model.opt.timestep, motor=motor, qpos_index=int(adapter.qi[motor]),
        torque_limit=float(adapter.control.scale[motor]), controlled_joint='UpperArm_L_ry',
        action_placement='After nominal slew, before same slew/envelope projection; scaled by authority and fade', branching='Identical deterministic prefixes; no live state overwrites', training_updates=0, arena_changed=False))
    archived = np.load(BASELINE)
    results = []
    for phase, frame in PHASES.items():
        common = None
        for name, magnitude, duration in BRANCHES:
            active = reach(adapter, net, frame, archived)
            assert not adapter.correction.any()
            start = snapshot(adapter, active)
            if common is None:
                common = start
                prior = dict(np.load(Path('logs/mimickit-training/arm-action-response-20261008-01') / f'{phase}-state.npz'))
                assert all(np.array_equal(v, prior[k]) for k, v in common.items())
                np.savez_compressed(OUT / f'{phase}-state.npz', **common)
            else:
                assert start.keys() == common.keys()
                assert all(np.array_equal(v, common[k]) for k, v in start.items()), phase
            trace = continuation(adapter, net, active.clone(), motor, magnitude, duration)
            if magnitude == 0:
                for key in ('qpos', 'issued', 'alive'):
                    assert np.array_equal(trace[key], archived[key][frame:frame + 60]), (phase, key)
            np.savez_compressed(OUT / f'{phase}-{name}.npz', **trace)
            row = dict(phase=phase, branch=name, magnitude=magnitude, duration_frames=duration,
                start_sha256=fingerprint(start), eligible_poses=np.flatnonzero(active[:8]).tolist(),
                survived_poses=np.flatnonzero(trace['alive'][-1] & ~trace['failed'][-1]).tolist())
            results.append(row)
            save(OUT / 'progress.json', results)
            print('BRANCH', row, flush=True)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT / 'report.json', dict(results=results, inputs_unchanged=True, no_training=True, arena_changed=False,
        output_hashes={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__ == '__main__':
    main()

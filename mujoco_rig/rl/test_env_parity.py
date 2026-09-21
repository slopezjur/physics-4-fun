"""Do the CPU and GPU environments define the SAME task?

There are two implementations of one environment - `perturb_env.py` (NumPy, per-env loop, the engine
Godot ships and the one every checkpoint is scored on) and `perturb_env_warp.py` (batched CUDA, the
one that trains). Two implementations drift apart. When they do here, the failure is silent and
expensive: training optimises one reward and scoring reports another, and the gap looks like a
transfer problem rather than a bug.

Reward arithmetic is shared in recovery_reward.py; state extraction and history updates still
need this integration check across MuJoCo C and mujoco_warp.

This puts BOTH environments in an identical state and checks observations and rewards.
Standing fixtures compare native measurements end to end. Severely penetrating fixtures
also report independent contact-solver differences, then check reward arithmetic with
matched slip measurements. test_foot_contacts separately checks native load/slip parity.

    python mujoco_rig/rl/test_env_parity.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from perturb_env import PerturbEnv  # noqa: E402
from perturb_env_warp import PerturbEnvWarp  # noqa: E402
from recovery_reward import reward_terms  # noqa: E402

N = 8
# float32 vs float64 on quantities of order 1-10. Anything larger is a real disagreement.
OBS_TOL = 2.0e-3
REWARD_TOL = 2.0e-3


def main() -> int:
    cpu = PerturbEnv(num_envs=N, episode_seconds=20.0, seed=3)
    gpu = PerturbEnvWarp(num_envs=N, episode_seconds=20.0, seed=3)

    # **Interface first, numbers second.** Matching observations prove nothing if the trainer
    # reaches for an attribute only one env has: `env.model` was called `mjm` on the GPU side, and
    # a 55-minute run died at its FIRST curriculum promotion, three minutes in, having passed every
    # numeric parity check. train.py is backend-agnostic, so anything it touches must exist on both.
    used_by_trainer = ("model", "ball", "num_obs", "num_actions", "max_episode_length",
                       "authority", "ball_speed", "ball_every", "decimation", "dt",
                       "reset_all", "get_observations", "step", "randomize_episode_phase")
    # And what the scorers and preflight reach for - public, so it is part of the contract too.
    used_by_tools = ("reward", "com", "com_velocity", "fire_ball")
    missing = [(name, side) for name in used_by_trainer + used_by_tools
               for env, side in ((cpu, "cpu"), (gpu, "warp")) if not hasattr(env, name)]
    assert not missing, f"attributes train.py uses are missing: {missing}"
    print(f"[parity] interface: all {len(used_by_trainer) + len(used_by_tools)} trainer- and "
          f"tool-facing attributes on both envs")

    assert cpu.num_obs == gpu.num_obs, f"obs width {cpu.num_obs} vs {gpu.num_obs}"
    assert cpu.num_actions == gpu.num_actions, f"actions {cpu.num_actions} vs {gpu.num_actions}"
    assert cpu.max_episode_length == gpu.max_episode_length
    assert cpu.decimation == gpu.decimation and cpu.dt == gpu.dt
    assert abs(cpu.authority - gpu.authority) < 1e-9
    assert abs(cpu.total_mass - gpu.total_mass) < 1e-6, "COM mass sets differ (ball exclusion?)"
    print(f"[parity] {cpu.num_obs} obs, {cpu.num_actions} actions, "
          f"episode {cpu.max_episode_length}, total_mass {cpu.total_mass:.3f} kg - shapes agree")

    # **Check the width against num_obs, not just CPU against GPU.** Both envs appending the
    # command slots twice produced 126 values while declaring 123, and this test passed because the
    # two agreed with each other. Training then died instantly on a shape mismatch.
    for env, side in ((cpu, "cpu"), (gpu, "warp")):
        got = env.get_observations().shape[-1]
        assert got == env.num_obs, f"{side} emits {got} values but declares num_obs={env.num_obs}"
    print(f"[parity] width: both envs emit exactly num_obs={cpu.num_obs} values")

    rng = np.random.default_rng(7)
    worst_obs = worst_rew = 0.0

    # Retain the original severe pose fixtures for observation/reward arithmetic.
    # They bury feet in the floor (loads up to 22 kN); independent CPU/GPU solvers
    # distribute their contact forces differently. Compare arithmetic with the
    # same measured slip input there, and report the raw difference separately.
    # The additional standing fixtures compare end-to-end measurements/rewards
    # without substitution. test_foot_contacts also checks native load/slip parity
    # in flat, heel, toe, moving and airborne states, including zero contact buffers.
    for trial in range(8):
        severe = trial < 4
        for i, d in enumerate(cpu.datas):
            d.qpos[cpu.qadr] = (rng.uniform(-0.35, 0.35, size=cpu.num_actions) if severe else
                               cpu.rest_qpos[cpu.qadr] + rng.uniform(-.05, .05, size=cpu.num_actions))
            d.qvel[cpu.vadr] = rng.uniform(-1.5, 1.5, size=cpu.num_actions)
            d.qpos[2] = (0.70 + 0.2 * rng.random() if severe else
                         cpu.rest_pelvis_z + rng.uniform(-.01, .01))
            d.qvel[0:3] = rng.uniform(-0.5, 0.5, size=3)
            import mujoco
            mujoco.mj_forward(cpu.model, d)
            gpu.qpos[i] = torch.tensor(d.qpos, dtype=torch.float32, device=gpu.device)
            gpu.qvel[i] = torch.tensor(d.qvel, dtype=torch.float32, device=gpu.device)
        gpu._mjw.forward(gpu._m, gpu._d)
        for i, d in enumerate(cpu.datas):
            cpu._after_physics(i, d)
        gpu._after_physics()
        np.testing.assert_array_equal(cpu.grounded, gpu.grounded.cpu().numpy())
        np.testing.assert_allclose(cpu.since_landing, gpu.since_landing.cpu().numpy(), atol=1e-6)
        np.testing.assert_allclose(cpu.rapid_replants, gpu.rapid_replants.cpu().numpy(), atol=1e-6)
        np.testing.assert_allclose(cpu.settle_hold, gpu.settle_hold.cpu().numpy(), atol=1e-6)

        action = rng.uniform(-1.0, 1.0, size=(N, cpu.num_actions))
        prev = rng.uniform(-1.0, 1.0, size=(N, cpu.num_actions))
        cpu.prev_action = prev.copy()
        gpu.prev_action = torch.tensor(prev, dtype=torch.float32, device=gpu.device)

        obs_cpu = cpu.get_observations().numpy()
        obs_gpu = gpu.get_observations().cpu().numpy()
        rew_cpu = np.array([cpu.reward(i, action[i]) for i in range(N)])
        rew_gpu = gpu.reward(torch.tensor(action, dtype=torch.float32,
                                           device=gpu.device)).cpu().numpy()

        if severe:
            raw = float(np.abs(rew_cpu - rew_gpu).max())
            f = gpu.recovery_features()
            f['slip_speed_sq'] = torch.tensor(np.stack([
                cpu.recovery_features(i)['slip_speed_sq'] for i in range(N)]),
                dtype=torch.float32, device=gpu.device)
            rew_gpu = sum(reward_terms(torch, **f,
                action=torch.tensor(action, dtype=torch.float32, device=gpu.device),
                previous_action=gpu.prev_action).values()).cpu().numpy()
            print(f"  buried fixture {trial}: raw solver-dependent reward difference {raw:.2e}; "
                  "checking arithmetic with matched slip inputs")

        d_obs = float(np.abs(obs_cpu - obs_gpu).max())
        d_rew = float(np.abs(rew_cpu - rew_gpu).max())
        worst_obs, worst_rew = max(worst_obs, d_obs), max(worst_rew, d_rew)
        print(f"  trial {trial}: max |d obs| {d_obs:.2e}   max |d reward| {d_rew:.2e}")

        if d_obs > OBS_TOL:
            # Name the offending slice, because "observations differ" over 120 channels is not a
            # debuggable statement.
            per = np.abs(obs_cpu - obs_gpu).max(axis=0)
            at, layout = 0, [("gravity", 3), ("lin_vel", 3), ("ang_vel", 3), ("height", 1),
                             ("joint_pos", cpu.num_actions), ("joint_vel", cpu.num_actions),
                             ("contacts", 2), ("prev_action", cpu.num_actions)]
            for name, width in layout:
                print(f"      {name:12s} max {per[at:at + width].max():.2e}")
                at += width

    cpu.reset_idx([0, 3])
    gpu.reset_idx(torch.tensor([0, 3], device=gpu.device))
    np.testing.assert_array_equal(cpu.grounded, gpu.grounded.cpu().numpy())
    np.testing.assert_allclose(cpu.since_landing, gpu.since_landing.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(cpu.rapid_replants, gpu.rapid_replants.cpu().numpy(), atol=1e-6)
    np.testing.assert_allclose(cpu.settle_hold, gpu.settle_hold.cpu().numpy(), atol=1e-6)
    # Exercise actual step/reset boundaries: timeouts bootstrap from the final
    # state, a simultaneous fall does not, and reset observations stay separate.
    cpu.reset_all()
    gpu.reset_all()
    cpu.episode_length_buf[:] = cpu.max_episode_length - 1
    gpu.episode_length_buf[:] = gpu.max_episode_length - 1
    cpu.datas[1].qpos[2] = 0.1
    mujoco.mj_forward(cpu.model, cpu.datas[1])
    gpu.qpos[1, 2] = 0.1
    gpu._mjw.forward(gpu._m, gpu._d)
    for env in (cpu, gpu):
        device = getattr(env, 'device', 'cpu')
        obs, _, done, extras = env.step(torch.full((N, env.num_actions), .1, device=device))
        assert done.all()
        assert extras['time_outs'][0] and not extras['time_outs'][1]
        torch.testing.assert_close(extras['terminal_observation'][0, -33:-3],
                                   torch.full((30,), .1, device=device))
        torch.testing.assert_close(obs[0, -33:-3], torch.zeros(30, device=device))
    print("[parity] terminal observations, timeout bootstrapping masks and reset isolation agree")
    ok = worst_obs <= OBS_TOL and worst_rew <= REWARD_TOL
    print(f"\n  worst observation {worst_obs:.2e} (tol {OBS_TOL:.0e}), "
          f"worst reward {worst_rew:.2e} (tol {REWARD_TOL:.0e})")
    print("  PASS - standing rewards and arithmetic with matched contact inputs agree" if ok else
          "  FAIL - the environments have drifted apart")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

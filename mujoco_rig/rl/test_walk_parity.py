"""Do the CPU and GPU WALK environments define the same task?

Same contract as test_env_parity.py, for the walk task: two implementations of one environment
drift silently, and when they do, training optimises one reward while scoring reports another.
Commands are pinned identically on both sides, because a reward that depends on a randomly sampled
command cannot be compared at all unless the command matches.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from walk_env import WalkEnv  # noqa: E402
from walk_env_warp import WalkEnvWarp  # noqa: E402

N = 8
OBS_TOL = 2.0e-3
REWARD_TOL = 2.0e-3


def main() -> int:
    import mujoco
    cpu = WalkEnv(num_envs=N, episode_seconds=20.0, seed=3)
    gpu = WalkEnvWarp(num_envs=N, episode_seconds=20.0, seed=3)

    assert cpu.num_obs == gpu.num_obs, f"{cpu.num_obs} vs {gpu.num_obs}"
    # Derived from the rig, never hard-coded: the joint count changes when the body does.
    expected = 3 + 6 + 1 + 2 * cpu.num_actions + 2 + cpu.num_actions + 3
    assert cpu.num_obs == expected, f"{cpu.num_obs} != derived {expected}"
    assert cpu.num_actions == gpu.num_actions
    print(f"[walk parity] {cpu.num_obs} obs (shared layout, last 3 are the command slots), {cpu.num_actions} actions")

    # **Check the width against num_obs, not just CPU against GPU.** Both envs appending the
    # command slots twice produced 126 values while declaring 123, and this test passed because the
    # two agreed with each other. Training then died instantly on a shape mismatch.
    for env, side in ((cpu, "cpu"), (gpu, "warp")):
        got = env.get_observations().shape[-1]
        assert got == env.num_obs, f"{side} emits {got} values but declares num_obs={env.num_obs}"
    print(f"[parity] width: both envs emit exactly num_obs={cpu.num_obs} values")

    rng = np.random.default_rng(11)
    worst_obs = worst_rew = 0.0

    for trial in range(4):
        cmd = rng.uniform(-1.0, 1.0, size=3)
        cpu.set_command(*cmd)
        gpu.set_command(*cmd)

        for i, d in enumerate(cpu.datas):
            d.qpos[cpu.qadr] = rng.uniform(-0.35, 0.35, size=cpu.num_actions)
            d.qvel[cpu.vadr] = rng.uniform(-1.5, 1.5, size=cpu.num_actions)
            d.qpos[2] = 0.70 + 0.2 * rng.random()
            d.qvel[0:3] = rng.uniform(-0.5, 0.5, size=3)
            mujoco.mj_forward(cpu.model, d)
            gpu.qpos[i] = torch.tensor(d.qpos, dtype=torch.float32, device=gpu.device)
            gpu.qvel[i] = torch.tensor(d.qvel, dtype=torch.float32, device=gpu.device)
        gpu._mjw.forward(gpu._m, gpu._d)

        action = rng.uniform(-1.0, 1.0, size=(N, cpu.num_actions))
        prev = rng.uniform(-1.0, 1.0, size=(N, cpu.num_actions))
        cpu.prev_action = prev.copy()
        gpu.prev_action = torch.tensor(prev, dtype=torch.float32, device=gpu.device)
        # Air-time state is integrated, so it has to start equal or the gait term cannot match.
        cpu.air_time[:] = 0.0
        cpu.was_airborne[:] = False
        gpu.air_time.zero_()
        gpu.was_airborne.zero_()

        obs_cpu = cpu.get_observations().numpy()
        obs_gpu = gpu.get_observations().cpu().numpy()
        rew_cpu = np.array([cpu.reward(i, action[i]) for i in range(N)])
        cpu.air_time[:] = 0.0
        cpu.was_airborne[:] = False
        rew_gpu = gpu.reward(torch.tensor(action, dtype=torch.float32,
                                           device=gpu.device)).cpu().numpy()

        d_obs = float(np.abs(obs_cpu - obs_gpu).max())
        d_rew = float(np.abs(rew_cpu - rew_gpu).max())
        worst_obs, worst_rew = max(worst_obs, d_obs), max(worst_rew, d_rew)
        print(f"  trial {trial}: cmd ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})  "
              f"max |d obs| {d_obs:.2e}   max |d reward| {d_rew:.2e}")

    # A shared arithmetic bug can pass parity: both implementations used to clear air time before
    # paying for a landing. Pin the expected payment, as well as agreement between backends.
    for env in (cpu, gpu):
        env.set_command(0.25)
        env.air_time[:] = 0.0
        env.was_airborne[:] = False
        env.last_landed[:] = 0
    for i, d in enumerate(cpu.datas):
        d.xpos[cpu.foot_l, 2] = cpu.rest_foot_z[0]
        d.xpos[cpu.foot_r, 2] = cpu.rest_foot_z[1]
    gpu.xpos[:, gpu.foot_l, 2] = gpu.rest_foot_z[0]
    gpu.xpos[:, gpu.foot_r, 2] = gpu.rest_foot_z[1]
    actions_cpu = np.zeros((N, cpu.num_actions))
    actions_gpu = torch.zeros((N, gpu.num_actions), device=gpu.device)
    baseline_cpu = np.array([cpu.reward(i, actions_cpu[i]) for i in range(N)])
    baseline_gpu = gpu.reward(actions_gpu).cpu().numpy()
    for env in (cpu, gpu):
        env.air_time[:, 0] = 0.2
        env.was_airborne[:, 0] = True
    landing_cpu = np.array([cpu.reward(i, actions_cpu[i]) for i in range(N)])
    landing_gpu = gpu.reward(actions_gpu).cpu().numpy()
    np.testing.assert_allclose(landing_cpu - baseline_cpu, 0.3, atol=REWARD_TOL)
    np.testing.assert_allclose(landing_gpu - baseline_gpu, 0.3, atol=REWARD_TOL)
    print("  landing: both backends pay 0.3 for a 0.2-second swing, then clear its history")

    # **The heading hold must agree too.** It lives in step(), which the comparisons above never
    # call: pin a zero-yaw command, give both envs the same headings to hold, and compare the yaw
    # command each derives from its own pelvis orientation. The states are synced from the last trial.
    cpu.set_command(0.3, 0.0, 0.0)
    gpu.set_command(0.3, 0.0, 0.0)
    targets = rng.uniform(-np.pi, np.pi, size=N)
    cpu.heading_target[:] = targets
    gpu.heading_target = torch.tensor(targets, dtype=torch.float32, device=gpu.device)
    cpu.apply_heading_hold()
    gpu.apply_heading_hold()
    d_yaw = float(np.abs(cpu.commands[:, 2] - gpu.commands[:, 2].cpu().numpy()).max())
    print(f"  heading hold: yaw commands agree to {d_yaw:.2e}")
    worst_obs = max(worst_obs, d_yaw)

    ok = worst_obs <= OBS_TOL and worst_rew <= REWARD_TOL
    print(f"\n  worst observation {worst_obs:.2e}, worst reward {worst_rew:.2e}")
    print("  PASS - both walk environments define the same task" if ok else
          "  FAIL - the walk environments have drifted apart")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

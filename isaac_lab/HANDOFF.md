# Handoff — Isaac Lab ⇄ Godot transfer

Written 2026-08-26. Read this before touching the Isaac side.

**Goal:** train the Godot active-ragdoll in Isaac Lab (GPU-parallel, ~50× faster) and bring policies
back into Godot as ONNX.

**Status:** the pipeline is complete and verified end to end. Policies **do not transfer**. The
cause is understood, and one untried approach remains — see [Next step](#next-step).

---

## 1. The core finding

Godot/Jolt is a **maximal-coordinate constraint solver**. Isaac/PhysX is a **reduced-coordinate
articulation solver**. Policies do not cross that boundary.

In reduced coordinates the joint angles *are* the state variables, and a joint cannot come apart
because "apart" is not representable. In maximal coordinates each body carries a full 6-DOF pose and
joints are constraints satisfied *approximately*, by iteration — so they sag under load.

The decisive measurement, repeated many times:

| | zero action (hold rest pose), 8 s |
|---|---|
| Isaac (URDF rig) | 98.4% still standing |
| Isaac (D6 rig) | ~84% still standing |
| **Godot** | **0% — on the floor in under 2 s** |

That is a property of the actuator/solver, not of any policy, observation or joint ordering.

**The positive controls confirm the mechanism:**

* NVIDIA transfer policies PhysX ↔ Newton with **joint reordering alone** — same solver class.
* Isaac → real robot works because a real robot **is** an articulation: rigid links, real hinges,
  joint servos. Reality is inside the family.
* A Euphoria-style game ragdoll is not. It is deliberately soft — that is what makes it look good.

Isaac was being used correctly throughout. The mismatch is that the dummy is not the kind of object
Isaac models.

---

## 2. Do not repeat these

Seven attempts, all measured, all negative. Each entry is here so nobody spends the time again.

| # | Attempt | Outcome |
|---|---|---|
| 1 | Actuator model / stiffness scaling | Isaac's rest pose stayed stable from 1.0× down to 0.2× — could not reproduce Godot's instability |
| 2 | Assist mode (policy composed on Godot's balance layer) | Needed a new `ActiveBone.PolicyTargetOffset` seam; with real authority the policy **actively knocked over** a body that otherwise stood indefinitely |
| 3 | Observation randomisation (joint-vel noise + micro-push) | Policy relearned to stand in Isaac (474.92/480) and its Godot first action improved 0.794 → 0.458 — still fell. The saturation being targeted was a *consequence* of falling, not its cause |
| 4 | D6 rig rebuild (match the mechanism) | Rig succeeded and is **2.4× faster**; transfer still failed |
| 5 | Match the control law, both directions | Explicit PD diverges at these gains in **both** engines — Isaac's `IdealPDActuator` hit velocities ~1e10 and crashed PPO on a NaN; the same law in Godot saturated all three axes at 693 N·m |
| 6 | Jolt solver iterations 2/10 → 16/30 | Genuinely better physics (zero-action head 0.608 vs 0.211 at t=6) and still collapses |
| 7 | **Physics** domain randomisation (stiffness 0.25–1.5×, damping, mass, friction) | Policy genuinely learned the family (456/480 in Isaac) and still fell in Godot in 2 s |

**Why #7 could never have worked, and why that is the final word.** Domain randomisation explores a
range of *parameters within one dynamics model*. Godot is not a point in that space; it is a
different model. Joints that can drift apart are not a sloppy version of joints that mathematically
cannot, so no parameter range reaches them.

---

## 3. What was built and is worth keeping

### Isaac side

* **`scripts/build_d6_usd.py`** — the D6 rig. Reads the same `ActiveRagdoll.tscn` and emits
  **16 bodies, 15 D6 joints, 45 DOF, 80.60 kg**: Godot's real mechanism. Enable with `P4F_RIG=d6`.
  * **~2.4× faster than the URDF rig** — 204k vs 86k steps/s at 8192 envs. The URDF path needed 46
    bodies because URDF has no 3-DOF joint, so every `Generic6DOFJoint3D` became three stacked
    hinges plus two massless links.
  * DOF count is unchanged (15 × 3 = 45), so the **143-obs / 36-action contract carries over**.
  * No mirror: the URDF converter mapped positions `(x,y,z) → (−z,+x,y)`, determinant **−1**, a
    reflection. The D6 one uses `(−z,−x,y)`, determinant +1, asserted at build.
  * DOF order equals declaration order (0 of 45 differ, vs **32 of 45** for URDF).
* **`p4f_isaac/actuators.py`** — `StablePDActuator`, reproducing Godot's Tan-Liu-Turk SPD. Diverges;
  kept as documentation of attempt #5.
* **Env knobs, all defaulting off** so every prior result reproduces: `P4F_RIG`, `P4F_ACTUATOR`,
  `P4F_STIFFNESS_SCALE`, `P4F_OBS_NOISE_*`, `P4F_MICRO_PUSH`, `P4F_RAND_*`.
* **Probes**: `probe_d6.py`, `probe_dof_convention.py`, `zero_action_probe.py`, `inspect_rig.py`,
  `dump_reference.py`.

### Godot side (`Source/RL/Isaac/`, all additive — the native track is untouched)

* `IsaacRigContract.cs` — reads `dummy_rig.json` at runtime; nothing transcribed.
* `IsaacObservation.cs` — the 143-float layout.
* `IsaacActionSpace.cs` — clamp, `ACTION_SCALE = 0.4`, piecewise about rest.
* `IsaacPolicyDriver.cs` — owns an `InferenceSession` directly, steps every 2nd physics tick.
* **`IsaacParityTest.cs`** — cross-engine parity harness. **The most valuable artefact here.**
  Verifies joint ordering by *name*, ONNX inference, and observation slices. It caught four silent
  contract defects.
* `RagdollRLBridge.UseIsaacContract` — trains the Godot-native track against the Isaac contract.

### Verified numbers

* Parity: ordering **45/45** and **36/36** exact; ONNX inference to **1.5e-07**.
* Stand on D6: **475.87/480**. With physics randomisation: **456.57/480** across a 6× stiffness range.
* Godot native throughput: **3,683 steps/s**; a mature policy is ~295M steps ≈ **22 hours**.

---

## 4. Traps that cost real time

1. **`obs_action_contract.md` was wrong four ways** — scrambled action order, inverted roll sign,
   undocumented DOF ordering, wrong frame convention. All silent; none raised an error. **Read
   `dummy_rig.json`, never the prose.**
2. **Frame rotation belongs on JOINT frames, not body frames.** Putting it on bodies makes
   `projected_gravity_b` read `(0,−1,0)`, so the `upright` reward computes as exactly **0.0 for a
   perfectly standing body**.
3. **Isaac derives body placement by forward kinematics from `init_state`**, ignoring authored body
   positions — bad joint anchors give a correctly-shaped skeleton lying in the wrong plane.
4. **Jolt solver settings are ignored under `godot --headless --path .`** — a hand-edited
   `project.godot` value never reaches the physics server and `GetSetting` reports the default. They
   apply from the editor (F5/F6) and in exported builds. A headless A/B of them compares nothing.
   See invalidator #3 in `docs/RL-SESSION-INVARIANTS.md`.
5. **Isaac Sim hangs on Windows teardown.** Work completes, output is lost to buffering. Scripts use
   `os._exit(0)`; orphans hold several GB and starve later runs — check `tasklist` and kill.
6. **`tasklist` truncates process names** to `Godot_v4.7.1-stable_mono_` — grepping for the scene
   name silently matches nothing and a running job looks finished.

---

## 5. Next step

**Isaac Lab 3.0 + Newton's XPBD solver.** The first approach that changes the *solver class* rather
than working around it.

XPBD is position-based dynamics — the **same maximal-coordinate family Jolt is in**. If the trainer
and the game engine are in the same family, NVIDIA's own result says transfer needs nothing but
joint reordering, which is already done.

Verified as available, not assumed:

* Worktree already created: **`D:\Proyectos\Juegos\Tools\IsaacLab3`** on `release/3.0.0-beta2`
  (the v2.3.2 tree is untouched, sharing the same `.git`).
* **`XPBDSolverCfg` exists** at
  `source/isaaclab_newton/isaaclab_newton/physics/xpbd_manager_cfg.py`, citing the Macklin/Müller
  XPBD papers. It is *not* in the docs' solver list — those docs say they are incomplete.
* Isaac Lab's own Newton docs state they "validated Newton simulation against PhysX by transferring
  learned policies **in both directions**" and deployed a Newton-trained policy to a G1 robot.
* Runs **kit-less** — no Isaac Sim, no Kit, no RTX, no Vulkan. Removes the GUI crash, the teardown
  deadlock and the ~18.7 GB fixed RAM baseline.

### Install (needs Python ≥ 3.12; the existing `env_isaaclab` is 3.11 and must not be disturbed)

```bash
conda create -n env_isaaclab3 python=3.12 -y
conda activate env_isaaclab3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e D:\Proyectos\Juegos\Tools\IsaacLab3\source\isaaclab \
            -e D:\Proyectos\Juegos\Tools\IsaacLab3\source\isaaclab_newton[all] \
            -e D:\Proyectos\Juegos\Tools\IsaacLab3\source\isaaclab_rl
python -c "import newton, warp; from isaaclab_newton.physics import XPBDSolverCfg; print('OK')"
```

### Plan, cheapest first

1. **Feasibility probe** — load a minimal articulation under `XPBDSolverCfg`; confirm joints exist,
   drive, and report DOF. Mirror `scripts/probe_d6.py`, which was the right pattern. Kills the idea
   in ten minutes if XPBD lacks the drives we need.
2. **Port the D6 rig** — already Godot's real mechanism, so it should be the right model for a
   maximal-coordinate solver. Note the task registration was written against 2.3.2 and 3.0 moved APIs.
3. **Retrain Stand**, export, test in Godot with `Scenes/RL/Isaac2/Stand/IsaacStandCheckD6.tscn` and the
   existing parity harness. **Always check the zero-action baseline first** — if the body cannot hold
   its rest pose, nothing downstream is worth measuring.

**Odds: ~45%.** The remaining risk is that XPBD is not Jolt's sequential-impulse solver — same
family, different algorithm — plus beta API churn.

---

## 6. Outstanding housekeeping

* `rl/scripts/config.ps1` is on `$Task = "isaacstand"` / `$MaxMinutes = 30` — revert to
  `"perturbation"` / `120`.
* Jolt is at **16/30** position/velocity steps. Every checkpoint through `perturbation_v7` was
  trained at the **2/10** defaults — revert unless retraining. See invalidator #3.
* `perturb_policy.onnx` is still the **light-ball** policy; the heavy-ball retrain
  (99.6% at 18 N·s, 75.4% at 50) was never promoted.
* Three diagnostic columns in `evaluate_perturb.py` are dead (`max tilt` always 0, `recovery` always
  nan, per-target table empty). The survival curve itself is correct.
* Nothing is committed. Many untracked files.

---

## 7. If it fails

Stop and ship with the Godot-native track. It works, produces policies that run in the game, and
costs ~22 hours per task. Closing the gap properly means giving Godot an articulation solver —
implementing Featherstone in Jolt or binding a different `PhysicsServer3D` — which is a research
project that would change how the whole game simulates.

**What survives either way:** the D6 rig (Isaac stays a sandbox where reward shaping and curricula
can be tried in 24 minutes instead of 22 hours), the four fixed contract defects, the parity harness,
and a documented answer to a question that is not well covered publicly — *transfer works across
engines of the same solver class, and does not across classes.*

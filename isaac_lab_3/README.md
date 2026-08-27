# Isaac Lab 3 / Newton track

Separate from [`../isaac_lab/`](../isaac_lab/) on purpose. That directory is Isaac Lab **2.3.2 /
Isaac Sim 5.1 / PhysX**, and it is finished work — seven measured attempts, a rig, a parity harness
and a documented negative result, all reproducible. Nothing here edits it.

This directory is Isaac Lab **3.0.0-beta2 / Newton**, testing the one hypothesis the 2.3.2 track
never got to try. Read [`../isaac_lab/HANDOFF.md`](../isaac_lab/HANDOFF.md) first; this page assumes it.

**Status: the feasibility gate PASSES.** XPBD reproduces Godot's zero-action collapse — 0% standing,
on the floor in under 2 s — where PhysX holds the same rig at ~84% for 8 s. See §3.

---

## 1. The hypothesis

Godot/Jolt is a maximal-coordinate constraint solver; Isaac/PhysX is a reduced-coordinate
articulation solver. Policies do not cross that boundary, and no parameter range bridges it, because
Godot is not a point in the space PhysX parameterises — it is a different model.

Newton's **XPBD** solver is position-based dynamics: the same maximal-coordinate family Jolt is in.
If the trainer and the game engine are in the same family, NVIDIA's own result says transfer needs
nothing but joint reordering, which is already done and verified exact on the D6 rig.

The gate is the **zero-action hold** — command the rest pose and see whether the body stays up:

| | zero action, 8 s |
|---|---|
| Isaac 2.3.2 / PhysX, URDF rig | 98.4% still standing |
| Isaac 2.3.2 / PhysX, D6 rig | ~84% still standing |
| **Godot / Jolt** | **0% — on the floor in under 2 s** |
| **Newton / XPBD** | **0% — on the floor in under 2 s** |

Godot is the number XPBD had to reproduce, and it does.

## 2. Environment

Runs **kit-less** — no Isaac Sim, no Kit, no RTX, no Vulkan. `AppLauncher` is the Kit entry point and
fails here with a bare `KeyError: 'EXP_PATH'`; `SimulationContext` on a fresh stage is the whole
runtime. This also removes the Windows teardown deadlock that forced `os._exit(0)` on 2.3.2.

Conda env **`env_isaaclab3`** (Python 3.12.13), against the worktree at
`D:\Proyectos\Juegos\Tools\IsaacLab3` on `release/3.0.0-beta2`. The 2.3.2 tree and its `env_isaaclab`
(Python 3.11) are untouched and must stay that way.

```
torch 2.11.0+cu128   newton 1.2.1   warp 1.13.0   tensordict 0.14.0
isaaclab 6.1.17   isaaclab_newton 0.13.6   isaaclab_rl 0.5.7   rsl-rl-lib 5.0.1
isaaclab_physx 1.1.3   isaaclab_contrib 0.4.4   isaaclab_assets 0.3.5   isaaclab_tasks 1.10.9
```

The last four are not optional even for a Newton-only run: `SimulationCfg` imports
`RigidBodyMaterialCfg` through a forwarding shim that raises unless `isaaclab_physx` is installed,
and `InteractiveScene` imports a sensor config from `isaaclab_contrib`. Both fail at import with a
message that names the module but not the reason.

```bash
D:\Programas\anaconda3\envs\env_isaaclab3\python.exe isaac_lab_3/scripts/probe_xpbd.py
```

## 3. Where the gate actually stands

**Passed — the rig loads correctly under XPBD.**

| check | result |
|---|---|
| bodies / DOF / mass | 16 / 45 / 80.60 kg — exact |
| joint drives reach the solver | `joint_target_ke` 600 / 350, `joint_target_kd` 18 / 20 |
| effort limits reach the solver | `joint_effort_limit` 350 — the contract's real ceilings |
| joint limits | imported per axis, correct values |
| rest pose | pelvis 0.820, head 1.540, `projected_gravity_b` (0, 0, −1) |

So XPBD accepts the D6 rig, keeps it as one articulation, and takes the Godot gains. That was the
part most likely to kill the idea outright, and it did not.

**Passed — the zero-action hold collapses exactly like Godot's.**

64 bodies, 8 s, all-zero actions, `--iterations 2`:

| t (s) | head (m) | pelvis (m) | span (m) | mean\|q\| | standing |
|---|---|---|---|---|---|
| 0.00 | 1.539 | 0.819 | 0.720 | 0.0000 | 100.0% |
| 1.00 | 1.430 | 0.742 | 0.710 | 0.0513 | 100.0% |
| 1.50 | 0.857 | 0.557 | 0.626 | 0.1601 | 0.0% |
| 2.00 | 0.133 | 0.171 | 0.712 | 0.0754 | 0.0% |
| 7.99 | 0.140 | 0.101 | 0.718 | 0.0600 | 0.0% |

The joints genuinely articulate — mean deviation reaches 0.16 rad and the pelvis→head span folds to
0.626 before the body flattens out — so this is a ragdoll collapse, not a rigid topple. And the
drives have real authority: the same run with the gains zeroed sits at 0.55 rad mean deviation
against 0.07, roughly 8x looser. **Correct gains, joints held near rest, body on the floor in under
two seconds.** That is Godot's failure mode, and PhysX never once produced it on this rig.

**The solver-effort sweep matches Jolt's signature too.**

| XPBD iterations | mean\|q\| at t=1.0 | head at t=1.5 | standing at 8 s |
|---|---|---|---|
| 2 | 0.0513 | 0.857 | 0% |
| 8 | 0.0049 | 1.459 | 0% |
| 16 | 0.0021 | 1.486 | 0% |

More iterations hold the joints ~25x tighter and the body degrades far more gracefully — and the
rest pose is still not stable at any setting. That is verbatim the Jolt 2/10 -> 16/30 result in
[`../docs/RL-SESSION-INVARIANTS.md`](../docs/RL-SESSION-INVARIANTS.md) §3, which measured the
zero-action head at t=6 moving 0.211 -> 0.608 and concluded the same thing. Two engines, same
parameter, same qualitative response, same conclusion.

## 3b. Training: ported, fast, and blocked on episode resets

The Stand task is ported (`p4f_newton/tasks/stand/`), registered as `P4F-Dummy-Stand-Newton-v0`, and
trains — PPO runs, every reward term is live and non-degenerate. Throughput is excellent.

### Environment count, measured

RTX 4080 SUPER 16 GB, Stand under XPBD, 20-25 iterations per row, fresh process each:

| `num_envs` | steps/s | vs 4096 | proc RAM | sys RAM | VRAM |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 252,120 | 1.00x | 2.3 GB | 11.8 GB | 2.7 GB |
| 8,192 | 428,552 | 1.70x | 2.8 GB | 12.4 GB | 3.8 GB |
| **16,384** | **572,219** | **2.27x** | 3.8 GB | 13.3 GB | 5.9 GB |
| 32,768 | 645,849 | 2.56x | 6.0 GB | 15.4 GB | 10.2 GB |
| 65,536 | 86,312 | 0.34x | 12.8 GB | 22.7 GB | 15.5 GB |

**65,536 falls off a cliff — 7.5x slower than 32,768** at 15.5 GB of 16.0 VRAM. That is memory
pressure, not compute. **16,384 is the recommended default**: 89% of peak throughput at 5.9 GB.
32,768 buys 13% more for 1.7x the VRAM *and* doubles samples per gradient update, which the 2.3.2
config notes usually costs more in PPO convergence than the rate returns.

For scale, the 2.3.2 PhysX track measured 110k steps/s at 4,096 and needed 14.6 GB at 16,384. This
track is roughly 2x the throughput at a third of the VRAM, because kit-less removes Isaac Sim's
~18.7 GB fixed footprint. **Two caveats before trusting these numbers**: the rows were taken with
episode resets broken (below), so reset cost is absent from all of them; and this env has **no
`ContactSensor`** — Newton's contact reporting does not surface through Isaac Lab's sensor here, so
the four contact flags are derived from body height (`CONTACT_HEIGHT`). That is a documented
approximation, good for Stand where the feet are planted, and it needs revisiting for get-up.

### The blocker: `write_root_*_pose_to_sim_index` does not round-trip

`_reset_idx` cannot put the dummy back on its feet. Measured on a fresh env:

```
after env.reset()      pelvis = [0.836, 0.820, 0.803, 0.801]   correct
after 2nd _reset_idx   pelvis = [1.640, 1.640, 1.640, 1.640]   = 0.82 + 0.82
after 3rd _reset_idx   pelvis = [1.640, 1.640, 1.640, 1.640]   saturates, not cumulative
```

Writing `z = 0.0` instead of `0.82` gives 1.640 as well, and so does
`write_root_link_pose_to_sim_index`. **The written height is ignored entirely after the first
reset**, and the pelvis is pinned at 1.640.

**What this looks like if you do not check.** Every episode after the first respawns the dummy 0.82 m
in the air. It falls from there, resets, falls again — and the aggregate reads as a body that
*gains* height: pelvis 0.820 -> 1.582, head to 2.302, climbing rather than falling. It therefore
never trips the fall threshold, `Episode_Reward/termination` stays exactly **0.0000**, and mean
episode length pins at **479.0 of 480 at every environment count** — three independent runs agreeing
to the digit. Read casually that is a policy that has learned to stand perfectly, in 25 iterations.

It was caught by invalidator #6 in [`../docs/RL-SESSION-INVARIANTS.md`](../docs/RL-SESSION-INVARIANTS.md)
— *a metric pinned at a constant is broken, not informative* — and confirmed by tracing per step,
where the pelvis teleports 0.432 -> 1.439 in a single step with its velocity zeroed. That is a
respawn, not physics.

**Ruled out, each measured:** the ground plane's spawn method and material; commanding all 45 joints
instead of the 36 actuated ones; `decimation` 1 vs 2; `max_depenetration_velocity` 10 / 1 / 0.1;
spawning 5 cm higher; and `newton.eval_fk` after the joint write. The joint-position noise in
`_reset_idx` amplifies the symptom — it is what makes the first fall happen early enough to expose
the bad respawn — but removing it only hides the problem, and it would silently drop the start-pose
randomisation that stops the policy learning one open-loop trajectory.

The probe (`probe_xpbd.py`) is unaffected and still collapses correctly, because it never resets.

### Next step

Port the Stand task to the 3.0 API and train under XPBD, then export and test in Godot against the
existing parity harness. Two things have to be built into the environment first, both consequences
of §4.1:

* **Joint angles must come from `newton.eval_ik`,** not `robot.data.joint_pos`. That is 90 of the
  143 observation floats.
* **Uprightness must come from the pelvis body quaternion,** not `projected_gravity_b`.

Record a Newton DOF order into the rig contract before any policy trained here meets a
2.3.2-ordered observation (§4.4).

## 4. Traps found here

Each of these is silent, and each would have produced a confident wrong answer.

0. **`robot.data.joint_pos` reads exactly 0.0000 under XPBD, forever, and this is not a bug — it is
   the finding.** A maximal-coordinate solver integrates body poses directly and treats joints as
   constraints between them, so generalized joint coordinates are never part of its state. A
   reduced-coordinate solver carries joint angle AS the state variable. That a solver cannot report
   its own joint angles without an inverse-kinematics pass is precisely the property this track was
   testing for, and Godot derives joint angles the same way.

   The trap is that zero reads as a perfectly-held rest pose. Every joint reports 0.0000 while the
   body folds up and hits the floor, and `applied_torque` reads a matching clean zero, so the run
   describes a rigid body that never moved a joint — which would have been recorded as XPBD failing
   to articulate the rig at all. `newton.eval_ik(model, state, joint_q, joint_qd)` recovers the true
   coordinates from `State.body_q`, which does update. Same for `projected_gravity_b`, which is
   derived from a root quaternion Isaac Lab never refreshes and reports a steady 1.000 for a body
   lying on the floor; read the pelvis BODY quaternion instead.

   **`joint_q` is one flat array over all environments**, in blocks of `7 + 45` — the leading 7 are
   the root free joint's position and quaternion. Stripping only the first 7 leaves every *other*
   environment's root in the result, which puts world positions into a joint-angle array: `max|q|`
   reads 10.5 rad against joints limited to ±0.5, and the mean drifts with `--num_envs` rather than
   with the physics. Reshape to `(num_envs, 52)` and slice.

1. **Isaac Lab 3 changed the root quaternion convention from wxyz to xyzw.**
   `InitialStateCfg().rot` now defaults to `(0,0,0,1)`; 2.3.2 used `(1,0,0,0)`. Both spell
   "identity" in their own order, so the 2.3.2 value copies across looking correct and is read as
   **x = 1** — a 180° rotation about X. The whole articulation spawns mirrored in y and z about the
   pelvis: head at 0.100 m instead of 1.540, foot at 1.600 instead of 0.040. `root_quat_w` still
   reports identity, because the read path applies the same swap and the error round-trips.
   Nothing raises, and the body falls — which is the answer this track is looking for, so it would
   have been recorded as a passing gate.

2. **`root_pos_w` double-counts the articulation's authored root offset.** It reports 1.640 for a
   pelvis that the physics, and every other reading, puts at 0.820. Use the Pelvis body's
   `body_com_pos_w`. The wrong value is plausible, stable, and exactly double.

3. **`init_state.pos` must stay at the rest pelvis height (0.82), not zero.** The free joint's
   translation *is* the pelvis position — the authored USD transform is not added to it. Setting it
   to 0 buries the body in the floor and the run NaNs within half a second.

4. **DOF order differs from the 2.3.2 contract in 42 of 45 slots.** Newton assigns its own ordering,
   and at the rest pose every joint sits near zero, so a permutation is invisible to any numeric
   check. A Newton-specific order has to be recorded into the rig contract before any policy trained
   here is fed a 2.3.2-ordered observation. Never reorder the contract to match.

5. **`AppLauncher` fails with `KeyError: 'EXP_PATH'`**, which names nothing. It is the Kit entry
   point and this track is kit-less.

6. **Read the right slice of Newton's gain arrays.** `joint_target_ke[:6]` is the root *free* joint
   and is legitimately all zeros; the real joints start at index 6. Reading the first six is an easy
   way to conclude the actuators are dead when they are fine — it cost time here.

## 5. Layout

```
isaac_lab_3/
├── README.md              this page
├── p4f_newton/
│   └── assets.py          ArticulationCfg for the D6 rig, read from the 2.3.2 rig contract
└── scripts/
    └── probe_xpbd.py      the feasibility gate
```

The rig itself is **not** regenerated here. `dummy_d6.usd` and `dummy_d6_rig.json` are read from
`../isaac_lab/assets/`, where `build_d6_usd.py` writes them from `Scenes/ActiveRagdoll.tscn`. The
Godot scene stays the single source of truth; a second copy of a generated artefact would drift the
first time the scene is retuned.

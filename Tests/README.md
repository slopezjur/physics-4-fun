# Tests

```bash
dotnet test Physics4Fun.sln
```

145 managed tests. No Godot install, no scene tree, no display required — they run anywhere the .NET
SDK does, including CI.

One additional native foundation test is opt-in through `P4F_FOUNDATION_FIXTURES`.
`MjMimicReferenceTests` covers non-looping reference interpolation and hash rejection.
`MjMimicTargetControlTests` covers delayed targets, fresh physics feedback, queue
observations, reset clearing and rejection of incompatible actuation contracts.
`MjMimicContactContractTests` rejects unknown or inconsistent contact semantics and
preserves legacy contracts. Python's `mujoco_rig.mimic.test_contact_contract` covers
toe-only observations, last-solve snapshots, reset isolation and rejection of implicit
weight/normalization migration. The GPU contact and ball probes additionally exercise
toe-off, launch preservation and non-reset peer contact preservation.
`MjMimicPushTests` covers force-window boundaries, coordinate conversion, invalid
pulse rejection and the uninterrupted post-push settling window. Python's
`mujoco_rig.mimic.test_perturb` additionally checks native force expiry and partial
reset isolation. `python -m mujoco_rig.mimic.godot_perturb --run <benchmark-directory>`
validates the actual Godot force schedule, survival and recovery measurements.
The experimental Stand actor's full observation/control boundary is checked in the
actual Godot scene by `python -m mujoco_rig.mimic.godot_replay`; see
`mujoco_rig/mimic/README.md` for setup and commands.

`MjCapturePointTests` checks contact-dependent support, airborne states and yaw
invariance of the diagnostic. `python -m unittest mujoco_rig.mimic.test_ball_validation`
checks the 96-case Cartesian target coverage, reset observations, checkpoint suite
consistency, final settled-recovery semantics and rejection of corrupt replay traces.
Set `MIMICKIT_PATH` to the pinned checkout to include its native task integration.
`python -m mujoco_rig.mimic.godot_ball` runs the same physical-ball validation suite
through Python and the actual Godot scene, including launch/impact action timing.

`MjPolicyTests` exercises the MuJoCo deployment loop through managed simulation/inference fakes:
contract validation, reordered observation channels, zero perturb commands, action clipping and
latency, position scaling, full reset, rejected non-finite inference and deterministic disposal.
The native bridge and actual ONNX policies are additionally checked by headless Godot scene runs.
Python environment and scoring regressions live in `mujoco_rig/rl/test_architecture.py`; see
`mujoco_rig/README.md` for that command and the CPU/GPU parity checks.

## Why this can exist at all

Godot's math types — `Vector3`, `Quaternion`, `Basis`, `Transform3D`, `Mathf` — are plain managed
structs in `GodotSharp.dll`. They need no engine, no native library and no initialised
`ProjectSettings`. So any class that touches only those can be exercised in a normal test runner.

The test project is therefore a plain `Microsoft.NET.Sdk` project, not a `Godot.NET.Sdk` one:
nothing here constructs a `Node`, and the Godot SDK exists to make an assembly loadable *by* the
engine, which is the opposite of what a test host wants.

Two notes on the plumbing, both of which will look wrong until you know why:

- `Physics4Fun.csproj` carries `<Compile Remove="Tests/**/*.cs" />`. Godot's SDK globs `**/*.cs`, so
  without it the test sources compile into the *game* assembly, where xunit is not referenced.
  `Tests/.gdignore` stops the editor scanning the folder as content for the same reason.
- The test project sets `<RollForward>Major</RollForward>`. It targets `net8.0` to match the game
  exactly, but the machine may ship only a newer runtime — and the game never needed an 8.0 install
  because Godot supplies its own.

## What is covered

| area | file | what it pins |
|---|---|---|
| SPD actuator core | `PidController3DTests` | Tan-Liu-Turk stability under extreme gains, the `MaxTorque` ceiling, the D-term's 50% authority split, `Ki = 0` non-accumulation, anti-windup bounds, the SO(3) log map past the ±π/2 branch |
| Actuator maths | `ActuatorMathTests` | Hill force-velocity falloff, eccentric contractions keeping full authority, the `τ₀·ω/4` power bound, the closed-form damping solve round-trip, swing-twist decomposition |
| Behavioural FSM | `RagdollStateMachineTests` | Settle grace, ordered escalation (stumble vs. straight to flailing), dwell timers, `PushUpDrill`/`ReinforcementLearning` being absorbing, recovery give-up horizon |
| Capture point & level frame | `BiomechanicalKinematicsTests` | ICP leading the CoM, horizontal-only displacement, the height clamp, yaw-level basis orthonormality and its degenerate fallback |
| Get-up phase machine | `GetUpPhaseControllerTests` | Every phase boundary, the two-point support rule, foot-under-mass requirement, timeout flagging, and the documented ~7.03 s worst case fitting `RecoveryDuration` |

Each test pins a claim the source makes **in prose**. That is deliberate: this project justifies its
physics in long comments, and until now those claims were checked only by running a scene and
watching whether the dummy shook.

## What cannot be tested here, and why

Anything deriving from `Node` or `RigidBody3D`. Constructing one requires a live engine, so:

- `ActiveBone` itself — `_Ready`, `_IntegrateForces`, joint-limit reads (they query a
  `Generic6DofJoint3D`'s properties by string), `Teleport`, `ResetBone`
- `BalanceController`, `HumanoidRagdoll`, `RagdollRLBridge`, `BallGun`, `RagdollSpawner`
- `BiomechanicalKinematics.TryComputeCenterOfMass` — takes `IReadOnlyList<ActiveBone>`

**The simulation itself is also out of reach, permanently.** Jolt plus floating point means there is
no exact-value assertion to make about a 15 s run. That is not a gap to close; it is why the maths
had to be separable in the first place. Simulation-level checking lives in
`Scenes/AutomatedTestRunner.tscn`, which asserts a tolerance ("pelvis stayed above 0.3 m through a
12 N·s push") rather than a value:

```bash
godot --headless --path . res://Scenes/AutomatedTestRunner.tscn
```

## Extending the reach: `IBoneState`

Most decision-making classes here are already pure logic. What kept them untestable was not their
own design — it was that the context structs they receive carried `ActiveBone`, and
`ActiveBone : RigidBody3D`. One engine type in a parameter list makes every consumer downstream
require a scene tree.

`Physics4Fun.Ragdoll.Interfaces.IBoneState` is the narrow read-only slice that breaks that chain,
and `FakeBone` is the twenty lines that replace the engine. `RecoveryContext` was converted first —
which is what made `GetUpPhaseControllerTests` possible — and it needed **no call-site changes and
no scene edits**, because `ActiveBone` converts implicitly.

The same move is available for `BalanceContext` (28 members, 14 of them bones) and `RlContext`,
which would bring every `IBalanceStrategy`, every reflex module and the RL reward/termination
components into reach. Convert one struct at a time; anything still needing the concrete type keeps
working.

A write-side counterpart (`IBoneActuator`: target rotation, feed-forward offset, muscle strength)
will be needed for the modules that *drive* bones rather than only observing them.

# Dynamic Motion Synthesis (DMS) Architecture & Biomechanical Model

This document outlines the complete biomechanical foundation, physics formulation, and software architecture of the active ragdoll locomotion and dynamic balance system — a tribute to the legendary character physics of GTA IV, rebuilt from first principles.

---

## 1. Architectural Layout

```
Physics4Fun.Ragdoll/
├── Interfaces/
│   ├── IBalanceTelemetryProvider.cs    # Read-only contract for UI & diagnostics
│   ├── IBiomechanicalReflex.cs         # Autonomous reflex strategy contract
│   ├── IBalanceStrategy.cs             # Balance strategy contract (BalanceContext + Apply)
│   ├── IBoneState.cs                   # Narrow read-only bone view; what makes consumers testable
│   └── ISupportLoadDistribution.cs     # How planted limbs share body weight
├── Modules/
│   ├── DynamicSteppingModule.cs         # Instantaneous Capture Point (ICP) & 2-bone IK
│   ├── WeightTransferModule.cs          # Asymmetric weight shifting & impedance scaling
│   ├── AnkleBalanceModule.cs            # Ankle ground reaction strategy (PI) & planar foot alignment
│   ├── HipStrategyModule.cs             # Medium-tier hip strategy: posture, CoM arrest, CoM height
│   ├── PelvisStabilizationModule.cs     # Protected Balance Region: pelvis attitude PD stabilizer
│   ├── GroundContactModule.cs           # Contact-driven foot ground sensing
│   └── Reflexes/
│       ├── ArmReflexModule.cs           # Counter-momentum torques & parachute fall bracing
│       ├── VestibularGazeModule.cs      # VOR horizon leveling & gaze tracking
│       ├── HitReactionReflexModule.cs   # Localized motor collapse & wound clutching
│       └── ObstacleBracingReflexModule.cs # 3D environmental probes & wall push-off bracing
├── Trajectories/
│   ├── IMotionTrajectory.cs            # Strategy contract for procedural motion synthesis
│   ├── StandingBalanceTrajectory.cs     # Default grounded standing posture
│   ├── SupineRecoveryTrajectory.cs      # 4-phase get-up trajectory from back
│   ├── ProneRecoveryTrajectory.cs       # 4-phase get-up trajectory from front
│   ├── PushUpDrillTrajectory.cs        # Scripted push-up drill for actuator validation
│   ├── IPhasedRecoveryTrajectory.cs    # Contract for trajectories driven by GetUpPhaseController
│   └── ReactiveMotionTrajectories.cs   # Impact stumbling and flailing trajectories
├── Support/
│   └── PlantedLimbLoadDistribution.cs  # Even split across planted end-effectors
├── Diagnostics/
│   └── RagdollTelemetryRecorder.cs     # Time-series telemetry logger & CSV export
├── ActiveBone.cs                        # Biomechanical PD actuator observing Newton's 3rd Law
├── BalanceController.cs                 # High-level coordinator composing balance modules
├── BiomechanicalKinematics.cs           # Shared ICP, level-frame & CoM math (DRY)
├── BiomechanicalMotionSynthesizer.cs    # Trajectory registry & orientation-keyed dispatch
├── HumanoidRagdoll.cs                   # Root skeletal actor & state transition driver
├── OrientationClassifier.cs             # Spatial orientation classifier (Upright/Prone/Supine/Side)
├── RagdollDebugInput.cs                 # Debug interaction (impulses, freeze toggles)
├── RagdollOrientation.cs                # Orientation enum
├── RagdollState.cs                      # Behavioral state enum
├── RagdollStateMachine.cs               # Balanced/Stumbling/Flailing/KnockedOut/Recovering FSM
└── StepPhase.cs                         # DoubleSupport, LeftSwing, RightSwing
```

### Design Conventions:
* Each physical capability (ICP stepping, asymmetric weight shifting, ankle reactions, hip strategy, pelvis stabilization, arm reflexes, vestibular gaze, hit reactions, environmental bracing, and get-up synthesis) is encapsulated in a dedicated module without cross-contamination.
* Autonomous reflexes implement `IBiomechanicalReflex` and register into a pipeline dynamically without modifying core balance code. Balance strategies (stepping, weight transfer, hip, ankle, pelvis stabilization) implement `IBalanceStrategy` and register into `BalanceController`'s balance pipeline the same way — each strategy receives a single per-tick `BalanceContext` and self-gates on state/step-phase/ground-contact, so a new strategy can be added by registering it, no orchestrator edits required. Motion trajectories implement `IMotionTrajectory` and register into `BiomechanicalMotionSynthesizer`; any trajectory can be substituted for another at runtime.
* Telemetry consumers (`RagdollTelemetryRecorder`, HUDs) bind to `IBalanceTelemetryProvider`, isolating diagnostics and UI from mutable physics controllers.
* High-level controllers depend on abstractions (`IBiomechanicalReflex`, `IBalanceStrategy`, `IMotionTrajectory`, `IBalanceTelemetryProvider`) rather than concrete monolithic physics blocks.
* `BalanceController` composes the two pipelines and owns bone wiring, telemetry, and per-tick sensing; the state machine (`RagdollStateMachine`), ground sensing (`GroundContactModule`), and shared kinematics math (`BiomechanicalKinematics`) are each extracted into their own single-purpose classes rather than living inline in the coordinator.

---

## 1b. Reinforcement learning layer

A second, independent control track. It shares the body (`HumanoidRagdoll`, `ActiveBone`) and
nothing else: `HumanoidRagdoll.UpdateBoneTargetRotations` returns early in the RL state, so a policy
owns the pose completely rather than nudging a procedural one.

```
Physics4Fun.RL/
├── Interfaces/
│   └── IRlComponents.cs                # Every contract below, plus the optional diagnostics pairs
├── Actions/
│   └── JointLimitedActionSpace.cs      # [-1,1] per axis mapped into that joint's measured limits
├── Observations/
│   └── BodyStateObservation.cs         # 113 floats: root-relative quaternions, CoM, contacts, goal block
├── Rewards/
│   ├── UprightProgressReward.cs        # Potential-based head height + upright + effort
│   └── WalkForwardReward.cs            # Product-form velocity x uprightness (NOT additive)
├── Termination/
│   ├── UprightTermination.cs           # Held-standing success, fall failure, per-episode flags
│   └── WalkTermination.cs              # Fall failure only - walking has no goal state
├── Curriculum/
│   └── ReverseStartPoseCurriculum.cs   # Start-pose floor that walks down as the policy clears it
├── Perturbation/
│   └── BallGun.cs                      # Analytic-impulse projectile (see RL-DESIGN-NOTES)
├── CmdlineArgs.cs                      # --name=value parsing for the kwargs train.py forwards
├── RlAgent.cs                          # Typed root of one *Agent.tscn: its ragdoll and its bridge
├── RagdollSpawner.cs                   # Instantiates N agents per process; owns --dummies=N
├── PolicyAutoLoader.cs                 # Arena playback; resolves the newest policy for its brain
└── RagdollRLBridge.cs                  # Godot/godot_rl_agents seam and episode lifecycle
```

### Design conventions

* Five strategy contracts — `IRlActionSpace`, `IRlObservationBuilder`, `IRlRewardFunction`,
  `IRlTerminationCondition`, `IRlStartPoseCurriculum` — mirror the procedural track's
  `IBalanceStrategy` / `IBiomechanicalReflex` / `IMotionTrajectory`. A new task is new classes, not
  edits to the bridge.
* Diagnostics are **separate optional interfaces** (`IRlRewardDiagnostics`,
  `IRlTerminationDiagnostics`, `IRlActionDiagnostics`, `IRlPerturbationDiagnostics`,
  `IRlWalkDiagnostics`) so a component works without them. Each exists because a scalar alone could
  not answer "why did it produce that", and every one was added after a specific measurement failed.
* Every component implements `Describe()`, and the bridge writes those strings into the run
  manifest. Provenance therefore cannot drift from the code that produced the run.
* Scenes are grouped by **shared-weight compatibility**, not by conceptual function — see
  `Scenes/RL/README.md`. Upright (stand, get-up, perturbation) and Locomotion (walk) are separate
  brains because their objectives conflict numerically.
* One agent is **one scene, instantiated N times**, not N subtrees in a scene file. Each task owns a
  `*Agent.tscn` whose root is an `RlAgent` (body + bridge + AIController, plus a ball gun where the
  task has one); every `*Training.tscn` and `*Arena.tscn` holds a single `RagdollSpawner` that
  instantiates it `--dummies=N` times and lays the row out along X. Consumers bind to `RlAgent`
  rather than reaching into an instantiated scene by node-path string, so an agent's internal layout
  is declared once, in its own scene file.

  This is what made per-process body counts possible, and the throughput case for them is that
  Godot's per-**process** costs — socket, render server, startup — were the ceiling all along, while
  the useful work is per-**body**: 16 processes × 64 bodies measures ~4,310 steps/s against ~2,258
  for 40 processes × 1. It replaced two scenes that carried forty hand-copied agent blocks each,
  regenerated by a script whenever the count changed.

---

## 1c. Known architectural debt

Recorded rather than silently carried. Ordered by how likely each is to cause a real defect.

Three earlier entries are retired. The telemetry column list is defined once and a row-width check
fails loudly on the first frame if the header and row paths ever drift; ragdoll keyboard input is
owned solely by `RagdollDebugInput`; and the support-load policy moved out of `HumanoidRagdoll`
behind `ISupportLoadDistribution`, which also gave its known mass over-estimate a documented home
instead of an anonymous line in the body class.

One more is retired outright: the project now has a **test project**. `Tests/Physics4Fun.Tests`
covers the SPD actuator, the Hill law, swing-twist, the behavioural FSM, the capture-point maths and
the get-up phase machine in 91 tests that need no Godot install and run in ~30 ms — see
`Tests/README.md` for what is reachable and what is permanently not. The reach was widened by
`IBoneState`, a narrow read-only bone contract that lets a context struct stop carrying
`ActiveBone`; `RecoveryContext` moved first, with no call-site and no scene changes.

Two more are retired by the move to many bodies per process. `OrientationClassifier` was a static
class holding a static hysteresis latch — correct while a process simulated one ragdoll, and a
cross-body defect the moment `RagdollSpawner` put 64 in one, since `EvaluateState` runs for every
body every tick (including in the RL state, where it is not gated off) and each one overwrote the
latch for the next. It is now per-`BalanceController`, cleared from `Reset()` so no respawn path can
forget it. And `ActiveBone.IsTargetWithinJointLimits` is no longer known-broken: it uses swing-twist
decomposition instead of a `Basis.GetEuler()` round-trip, so the four wide-range axes it used to
false-positive on (Thigh, Shin, Forearm, UpperArm — the joints a get-up depends on) now read back
the angle that was actually commanded.

| # | Where | Issue | Why it matters |
|---|---|---|---|
| 1 | `ActiveBone` | ~5 responsibilities in ~840 lines: rigid body, SPD actuator, gravity/support feed-forward, force-velocity limit, joint-limit querying. | Less urgent than it reads. The SPD core was never actually in here — it lives in `PidController3D`, which has zero scene-tree references and is now covered by 17 tests. What remains engine-bound is orchestration, the load feed-forward (which walks the distal chain reading transforms), and joint-limit reads that query a `Generic6DofJoint3D` by string. The Hill law and swing-twist are pure statics and covered. A split is still worth doing, but the arithmetic it would move is now pinned, which is what makes it safe. |
| 2 | `BalanceContext` | 28 members, 14 of them `ActiveBone`; every module receives the whole body. | Two separate issues, and only one is worth fixing. The **width** is deliberately kept: passed `in`, no per-tick copy, negligible at 64 bodies, no defect ever traced to it. The **bone type** is not — carrying `ActiveBone` (a `RigidBody3D`) is what keeps every `IBalanceStrategy` and reflex module untestable, exactly as it did for `RecoveryContext` before that moved to `IBoneState`. Converting it is the next step, and needs a write-side `IBoneActuator` companion for the modules that drive bones rather than only observing them. |
| 3 | `RagdollRLBridge` | Still ~7 responsibilities after the curriculum extraction; `GetStepInfo` is 130 lines. | Cohesive dictionary assembly, so splitting it adds indirection without clarity. The `TaskKind` switch is closed for extension — the right moment to add a registry is when a third brain appears, not before. |
| 4 | `BodyStateObservation` | The 7-float goal block emits constants and has no contract of its own — no `IRlGoalProvider` alongside the other five. | Deliberately **not** fixed. The slots are inert (six emit exactly `0.0`, so their first-layer weights receive exactly zero gradient), and the interface earns nothing until something actually writes a command. The moment one does — heading for walking is the near-certain first — it should arrive as a contract, not as an edit to `Build()`. |
| 5 | `RagdollSpawner._EnterTree` | Spawn timing is load-bearing and enforced only by a comment. | `sync.gd` builds its agent list from the `AGENT` group in `_ready`, and `PolicyAutoLoader` reads `Agents` in the root's `_Ready`. Moving the spawn to `_Ready` makes Sync handshake with an empty observation space. Godot offers no way to declare this ordering, so the comment is the mechanism. |
| 6 | `BiomechanicalMotionSynthesizer` | Static registry sharing one trajectory instance across every body in the process; statelessness is an unenforced invariant. | Safe today — every trajectory is a pure function of its arguments and holds only immutable `PhaseBoundaries` — and now documented on the class. A trajectory that caches per-body state would silently share it across 64 ragdolls, producing targets that depend on which *other* body was evaluated first: nondeterministic, load-dependent, and invisible in a single-body scene. The fix if it happens is the one `OrientationClassifier` just took: move the registry off `static` and give each `BalanceController` its own. |
| 7 | `HumanoidRagdoll` | ~8 responsibilities in 693 lines: state driving, bone registry, muscle stiffness, target writing, reset/teleport/spawn, the push-up drill, telemetry ownership, recovery-context assembly. | The push-up drill is the cleanly separable one — a self-contained scripted diagnostic with its own timer, constants and progress property, sitting in the body class. Not extracted here because it is inert in every scene that matters and the churn would touch the same file as item 1; worth doing alongside that, not before it. |
| 8 | `PhysicsGrabber` | Grab physics and tether *rendering* (`ImmediateMesh`, `StandardMaterial3D`) in one class. | Debug tooling used only by `TestChamber.tscn`, so the blast radius is a debug scene. Recorded rather than fixed. |

Two notes that are not table rows:

`ActiveBone.IsTargetWithinJointLimits` is known-broken (Euler YXZ cannot represent the ±2.6 rad
limits four axes on this rig have) and its own doc comment says so. It is diagnostic-only; the RL
action path scales into the limits by construction and does not depend on it.

**The observation width is frozen at 113.** Not debt so much as a standing constraint: it has been
106, 108, 115 and 113 across the archived lineages, every change orphans every checkpoint, and one
such change cost three commits of ONNX shape-crash debugging. `Describe()` must account for every
float in `Size` — it silently did not for the goal block, and every manifest written in that window
reports a component breakdown summing to 106 beside a total of 113.

---

## 1d. MuJoCo track

A third control track, and the one where learned control has worked: the body's physics runs in
MuJoCo, driven from Godot through P/Invoke, and a policy trained on MuJoCo is scored and shipped on
the same engine. How to use it is in `mujoco_rig/README.md`; where it stands, in
`mujoco_rig/STATUS.md`.

```
mujoco_rig/
├── build_mjcf.py            # Godot's rig dump -> dummy*.xml, in named stages
├── validate.py              # The generated body is the body we meant
├── rl/
│   ├── env_config.py        # Constants that define the tasks
│   ├── body_env*.py         # CPU / GPU base environment: plant, observation, reset, step, guards
│   ├── perturb_env*.py      # + projectile, capture-step reward
│   ├── walk_env*.py         # + commands, heading hold
│   ├── train.py             # PPO loop: Task table, Curriculum, WorldSchedule, EpisodeStats
│   ├── eval.py, eval_walk.py  # CPU scorers; --json
│   └── export_onnx.py       # ONNX + the generated contract
└── scripts/
    ├── scoring.py           # One Scorer per task - the only reader of scorer output
    ├── promote.py           # Paired, measured promotion into the scenes
    ├── overnight.py         # Chained scored sessions; one SessionPolicy per task
    └── preflight.py         # Task-level checks; one check list per task

Source/RL/MuJoCo/
├── MjBridge.cs              # P/Invoke model/data handle; the one Godot <-> MuJoCo frame map
├── MujocoDummy.cs           # Scene node: load, drive (nothing / policy / scripted), step, render
├── MjPolicyDriver.cs        # Policy clock, delayed actions, complete episode reset
├── MjPolicyContract.cs      # Validated channels, action mapping, timing and command source
├── MjPolicyObservation.cs   # Observation channels written at the declared offsets
├── MjOnnxPolicy.cs          # ONNX session ownership and tensor shape checks
├── IMjPolicyState.cs        # Read-only observation view and separate actuator capability
├── MjModelDefinition.cs     # Generated MJCF configuration, independent of rendering
├── MjHeadingHold.cs         # The contract's command-filling rule for walk
└── MjBallGun, MjGaitMetrics, MjProxyBuilder, MjScriptedController, MjLayout, MjInterop
```

### Design conventions

* **The contract is the interface.** Everything Godot needs to feed a policy - observation layout,
  joint order, action mapping, and how the command channel is filled - is generated from the
  training environment by `export_onnx.py`. `MjPolicyDriver` hard-codes none of it.
* **Train on GPU, score and ship on CPU.** `mujoco_warp` is float32 and only trains; MuJoCo's C
  engine is float64 and is what Godot runs, so every decision is made on it.
* **Task behaviour lives in tables, not branches.** `train.TASKS`, `scoring.TASKS`,
  `overnight.POLICIES` and `preflight.CHECKS` hold one entry per task; a new task adds entries and
  edits no tool.
* **Tools read data, not prose.** Scorers write JSON and the contract is JSON; no decision parses
  printed text.
* **Parity is tested, not assumed.** Each task's CPU and GPU implementations are checked for the
  same numbers and the same interface.

### Boundaries enforced by the unstaged-code audit

* The policy loop depends on `IMjPolicyPlant` and `IMjPolicyInference`; the observation builder
  receives only `IMjPolicyState`. Tests supply managed fakes, with no scene tree or native library.
  Contract parsing, observation construction, inference ownership and actuation timing have separate
  reasons to change. Unsupported channels and inconsistent shapes fail at load.
* The contract declares observation offsets, command source, decimation and action latency. Perturb
  receives zero command slots, while walk receives the scene command and optional heading hold.
  Reset clears the previous action, delayed actions, inference clock and held heading together.
* `MjBridge` checks the native version before reading generated offsets, frees partially constructed
  handles and rejects use after disposal. Whole-body velocity shifts spatial velocities to each
  body's COM. Policy observations deliberately retain their trained subtree-reference semantics.
* CPU and GPU walk environments depend on immutable `WalkCommandConfig` instances. Constructing a
  different training stage cannot change an existing environment's command distribution. Stage 2
  samples reverse and lateral commands through the general sampler rather than the forward-only
  straight-line branch. Episode reset clears landing history and heading corrections.
* Scoring configures `auto_reset` and `max_shots_per_episode`; it no longer replaces live environment
  methods to suppress resets or count shots. Explicit reset remains available during evaluation.
* Promotion requires successful finite scores for both candidates when an incumbent exists. Export
  validates the network and contract in a temporary directory before replacement, restoring the old
  pair on a replacement error. This is exception recovery, not a cross-process atomic transaction.
  Overnight sessions score their initial seed, carry difficulty only from accepted sessions, and
  distinguish the latest accepted seed from the best checkpoint offered for promotion.

The audit also corrected an airtime reward that both backends always evaluated as zero: landing
cleared the counter before reading it. Tests now pin the expected landing payment, not only backend
agreement. This changes future training rewards; existing policy weights and authored physics are
unchanged. Do not compare old and new training returns as evidence of policy improvement.

Validation commands and coverage are in `Tests/README.md` and `mujoco_rig/README.md`. The existing
architectural debt in section 1c belongs to the other control tracks and is outside this change.

---

## 2. Mathematical & Biomechanical Formulation (The 7 Pillars)

### Pillar 1: Actuators & Tan-Liu-Turk Stable PD Formulation
Joints utilize the **Tan-Liu-Turk Stable Proportional-Derivative (SPD)** implicit formulation to eliminate discrete limit-cycle chatter and guarantee unconditional numerical stability across all timesteps $\Delta t$:
$$\tau = \frac{k_p \vec{\theta}_{\text{error}} + k_d (\vec{\omega}_{\text{des}} - \vec{\omega}_{\text{rel}})}{1 + \frac{k_d \Delta t}{I_{\text{eff}}} + \frac{k_p \Delta t^2}{I_{\text{eff}}}}$$
Torques are applied directly to child bones and equal-and-opposite counter-torques to parent bones, strictly observing Newton's Third Law ($\tau_{\text{parent}} = -\tau_{\text{child}}$). The measured relative angular velocity feeding the derivative term is low-pass filtered, and the derivative contribution is clamped so the proportional term always retains authority — this suppresses high-frequency chatter at the torque clamp without compromising stability.

The stability guarantee holds only if $I_{\text{eff}}$ is the bone's **real** inertia: torques act about the bone's center of mass, so on the first integration step each `ActiveBone` captures its true inverse inertia tensor from the physics state and feeds the most sensitive principal axis into the SPD denominator (`EffectiveInertia` remains only a fallback). Because real small-bone inertias shrink the denominator's authority budget, actuator gains are tuned per chain against the load they carry: weight-bearing legs run high stiffness (thigh 1600 / shin 1800 / foot 1200 with low damping), and the torso chain is sized above its inverted-pendulum gravity gradient (spine 600, chest 350).

### Pillar 2: 100% Grounded Biomechanics & Ankle Strategy
All standing forces and vertical support operate purely through internal knee and hip PD motors reacting against physical ground collision normal forces (zero artificial world forces / zero floating springs):
* **Floor Planar Alignment:** Foot orientations dynamically align with terrain normal via local joint feed-forward offsets.
* **Ankle Strategy (small perturbations):** Foot PD motors generate restorative ground reaction moments proportional to the horizontal Center of Mass (CoM) error relative to the base of support, with a clamped integral channel for DC position hold and damping matched for a near-critically-damped response.
* **Hip Strategy (medium perturbations):** When the ankle channel saturates but the ICP has not escaped, hip/torso feed-forward offsets pull the body mass back over the support polygon (sagittal CoM arrest), right the pelvis/torso attitude against the planted feet, and regulate pelvis height through symmetric knee extension.
* **Double Support:** $50/50$ nominal weight sharing with dual-foot ground reaction coupling, plus a lateral weight-shift loop that presses harder on the leg the CoM leans toward, tracking the CoP under the CoM and stiffening the loaded leg.
* **Single Support:** Stance leg carries ~95% of the target weight share with stiffened impedance ($120\%$) and an active lateral lean shifts the CoM over the stance foot so the swing leg truly unweights; swings commit — only real foot contact late in the arc counts as touchdown.

### Pillar 3: Protected Balance Region (Pelvis Stabilization)
The pelvis is the skeletal root and has no parent actuator, yet it absorbs every reaction torque from the spine and thigh motors. `PelvisStabilizationModule` (an `IBalanceStrategy`) computes the pelvis attitude error ($\text{pelvisUp} \times \text{worldUp}$) and applies a corrective PD torque directly to the pelvis, distributing the equal-and-opposite reaction across the grounded feet. The stabilizer scales with the global balance strength fades, fully disengaging during flailing, knockout, and deep-tilt regimes so the body falls naturally once balance is lost.

### Pillar 4: Instantaneous Capture Point (ICP) & Analytical 2-Bone IK
The capture-point projection and yaw-level reference frame are centralized in `BiomechanicalKinematics` (a single implementation shared by `DynamicSteppingModule`, `HipStrategyModule`, and `BalanceController`'s ICP-escape telemetry — previously duplicated with a latent inconsistency between two slightly different height formulas). When horizontal velocity displaces the CoM beyond the support polygon, orbital ICP determines the swing foot landing location:
$$\vec{x}_{\text{cp}} = \vec{x}_{\text{CoM}} + \frac{\vec{v}_{\text{CoM}}}{\omega_0}, \quad \omega_0 = \sqrt{\frac{g}{h_{\text{CoM}}}}$$
where $h_{\text{CoM}}$ is measured above the actual ground contact height. The ICP escape test is evaluated in a yaw-only level frame so pelvis pitch cannot mask horizontal divergence. Analytical 2-bone IK resolves anatomical hip pitch/roll and knee flexion along a cycloid swing trajectory directed toward the ICP landing target, with leg segment lengths measured from the skeleton at runtime.

### Pillar 5: Autonomous Upper Body Reflex Pipeline
* **Vestibulo-Ocular Reflex (VOR):** Cervical spine counter-rotates against torso pitch/roll with a $2.5^\circ$ deadband.
* **Arm Counter-Torques:** Upper limbs swing to counteract pelvis yaw/roll angular velocity.
* **Parachute Fall Bracing:** Extends arms forward ($+65^\circ$) and tucks head during falling impacts.
* **Localized Hit Reactions & Wound Clutching:** Struck bone loses motor strength ($\text{strength} = 0.20$ for $0.40\text{s}$) while the opposite hand reaches to clutch the impact location.

### Pillar 6: 3D Spatial Awareness & Grounded Obstacle Bracing
Lateral sensory raycasts ($1.3\text{m}$ radius) detect nearby walls and obstacles. The nearest arm dynamically extends toward the contact normal to establish physical contact collision constraints, stabilizing the torso without artificial world-space forces.

### Pillar 7: Behavioral State Machine & Biological Get-Up AI
A 5-state behavioral machine (`Balanced` $\to$ `Stumbling` $\to$ `Flailing` $\to$ `KnockedOut` $\to$ `Recovering`) autonomously manages state transitions. Escalation is ordered: slow collapses pass through `Stumbling` (tilt beyond $30^\circ$ or pelvis speed beyond $3\,\text{m/s}$), while only extreme tilt ($>80^\circ$) or sustained airborne time routes directly to `Flailing`:
* **Prone Recovery:** 4-phase push-up $\to$ quadruped crawl $\to$ deep squat $\to$ upright leg extension.
* **Supine Recovery:** 4-phase roll-to-prone $\to$ quadruped push-up $\to$ squat transition $\to$ upright extension.
* **Assisted Recovery:** Balance modules remain partially active ($40\%$ strength) during get-up so ankle, posture, and stepping corrections assist the keyframed trajectory.
* **Dynamic Muscle Ramping:** Muscle stiffness scales dynamically during recovery ($0.60 \to 1.0$) to maintain physical stability while rising from the floor.

# Dynamic Motion Synthesis (DMS) Architecture & Biomechanical Model

This document outlines the complete biomechanical foundation, physics formulation, and software architecture of the active ragdoll locomotion and dynamic balance system — a tribute to the legendary character physics of GTA IV, rebuilt from first principles.

---

## 1. Architectural Layout

```
Physics4Fun.Ragdoll/
├── Interfaces/
│   ├── IBalanceTelemetryProvider.cs    # Read-only contract for UI & diagnostics
│   └── IBiomechanicalReflex.cs         # Autonomous reflex strategy contract
├── Modules/
│   ├── DynamicSteppingModule.cs         # Instantaneous Capture Point (ICP) & 2-bone IK
│   ├── WeightTransferModule.cs          # Asymmetric weight shifting & impedance scaling
│   ├── AnkleBalanceModule.cs            # Ankle ground reaction strategy (PI) & planar foot alignment
│   ├── HipStrategyModule.cs             # Medium-tier hip strategy: posture, CoM arrest, CoM height
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
│   └── ReactiveMotionTrajectories.cs   # Impact stumbling and flailing trajectories
├── Diagnostics/
│   └── RagdollTelemetryRecorder.cs     # Time-series telemetry logger & CSV export
├── ActiveBone.cs                        # Biomechanical PD actuator observing Newton's 3rd Law
├── BalanceController.cs                 # High-level coordinator composing balance modules
├── BiomechanicalMotionSynthesizer.cs    # Trajectory registry & orientation-keyed dispatch
├── HumanoidRagdoll.cs                   # Root skeletal actor & state machine driver
├── OrientationClassifier.cs             # Spatial orientation classifier (Upright/Prone/Supine/Side)
├── PoseLibrary.cs                       # Anatomical rest pose definitions
├── RagdollDebugInput.cs                 # Debug interaction (impulses, freeze toggles)
├── RagdollOrientation.cs                # Orientation enum
├── RagdollState.cs                      # Behavioral state enum
└── StepPhase.cs                         # DoubleSupport, LeftSwing, RightSwing
```

### Design Conventions:
* Each physical capability (ICP stepping, asymmetric weight shifting, ankle reactions, arm reflexes, vestibular gaze, hit reactions, environmental bracing, and get-up synthesis) is encapsulated in a dedicated module without cross-contamination.
* Autonomous reflexes implement `IBiomechanicalReflex` and register into the pipeline dynamically without modifying core balance code. Motion trajectories implement `IMotionTrajectory` and register into `BiomechanicalMotionSynthesizer`; any trajectory can be substituted for another at runtime.
* Telemetry consumers (`RagdollTelemetryRecorder`, HUDs) bind to `IBalanceTelemetryProvider`, isolating diagnostics and UI from mutable physics controllers.
* High-level controllers depend on abstractions (`IBiomechanicalReflex`, `IMotionTrajectory`, `IBalanceTelemetryProvider`) rather than concrete monolithic physics blocks.

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
The pelvis is the skeletal root and has no parent actuator, yet it absorbs every reaction torque from the spine and thigh motors. A dedicated stabilizer computes the pelvis attitude error ($\text{pelvisUp} \times \text{worldUp}$) and applies a corrective PD torque directly to the pelvis, distributing the equal-and-opposite reaction across the grounded feet. The stabilizer scales with the global balance strength fades, fully disengaging during flailing, knockout, and deep-tilt regimes so the body falls naturally once balance is lost.

### Pillar 4: Instantaneous Capture Point (ICP) & Analytical 2-Bone IK
When horizontal velocity displaces the CoM beyond the support polygon, orbital ICP determines the swing foot landing location:
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

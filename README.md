# Physics4Fun: Euphoria-Style Active Ragdoll

A purely physics-driven, active ragdoll locomotion and dynamic balance system for Godot 4 (.NET). This project is a tribute to the legendary character physics of GTA IV (Dynamic Motion Synthesis), rebuilt from first principles without relying on animation state machines or artificial world forces.

## Core Philosophy: 100% Grounded Biomechanics
Unlike traditional game ragdolls that use invisible springs or "floating" central forces to stay upright, this engine enforces strict biomechanical realism. Every movement, step, and balance correction is achieved purely through internal joint torques pushing against physical ground collisions. 

## Key Features

* **Tan-Liu-Turk Stable PD Formulation:** Joints use an implicit SPD formulation that observes Newton's Third Law. This eliminates discrete limit-cycle chatter and guarantees numerical stability under high gravity loads.
* **Multi-Tier Balance Strategy:** 
  * **Ankle Strategy:** PI-controlled ground reaction moments for small perturbations and DC position hold.
  * **Hip Strategy:** Medium-tier defense using sagittal CoM arrest, posture righting, and symmetric knee extension.
  * **Stepping Strategy:** Orbital Instantaneous Capture Point (ICP) math and analytical 2-bone IK calculate dynamic swing foot landings.
* **Single Support Weight Shifting:** Active lateral leaning shifts ~95% of the body load over the stance foot, allowing the swing leg to cleanly unweight and commit to the step.
* **Protected Balance Region:** The unactuated pelvis is stabilized by distributing equal-and-opposite reaction torques across the grounded feet.
* **Autonomous Upper Body Reflexes:**
  * **Vestibulo-Ocular Reflex (VOR):** Dynamic horizon leveling and gaze tracking.
  * **Parachute Fall Bracing:** Arms dynamically reach and brace against impacts.
  * **Hit Reactions:** Localized motor collapse and wound clutching upon collision.
* **3D Spatial Awareness (Pillar 5):** Raycast-driven environmental probing allows the dummy to push off walls and obstacles to stabilize the torso without artificial aids.
* **Biological Get-Up AI (Pillar 6):** A behavioral state machine autonomously manages falls, seamlessly transitioning into a 5-phase physical recovery (plant hands $\to$ push-up torso $\to$ drive lead knee $\to$ half-kneel rise $\to$ upright extension).

## Status

The project has two independent tracks.

**Procedural (Euphoria DMS) — complete.** All foundational pillars are implemented and verified, and strictly adhere to biomechanical principles: no animation state machines, 100% active ragdoll torque physics. This is the track the feature list above describes, and it works.

**Reinforcement learning — experimental, and openly unsolved.** A parallel attempt to *learn* the get-up from scratch rather than script it. It is included because the negative results are documented and reproducible, not because it works. As of ~24M training steps:

* From a **standing** start, the policy holds a fully stabilized pose (upright, grounded, settled, balanced over its feet) for 1.5 s on roughly **36%** of attempts, and is still improving.
* From a **prone** start — the actual goal — it has **never once stood up.** Not in 24M steps here, nor in a separate 63M-step run.

Treat it as a lab notebook rather than a feature. If you want a ragdoll that gets up, use the procedural track.

## Reinforcement learning track

Pure RL over the same rig: a 106-dim observation, 36 continuous joint targets, PPO via [godot_rl_agents](https://github.com/edbeeching/godot_rl_agents) with 32 parallel Godot instances. `Source/RL/` splits the environment into swappable observation / action / reward / termination components behind `IRl*` interfaces, so an experiment changes one class instead of the bridge.

* **[docs/RL-TRAINING.md](docs/RL-TRAINING.md)** — how to run it: setup, the PowerShell scripts, what every TensorBoard metric means and how to read it.
* **[docs/RL-DESIGN-NOTES.md](docs/RL-DESIGN-NOTES.md)** — why it is built this way: measured findings, the reward and termination design, and the framework-level bugs found and worked around.

Both are written up honestly, including the things that did not work and the measurements that proved it.

## Requirements

* **Godot Engine 4.x** (Mono / .NET version)
* **.NET SDK** (Compatible with your Godot 4.x build)
* *For the RL track only:* Python 3.12 and a CUDA-capable GPU — see [docs/RL-TRAINING.md](docs/RL-TRAINING.md) for the exact install, which needs a specific torch index and cannot be a plain `pip install -r`.

## Setup & Testing

1. Open the project in Godot.
2. Build the C# solution.
3. Open `Scenes/ActiveRagdoll.tscn` or run `AutomatedTest.cs` to watch the active ragdoll balance, react to impulses, and attempt physical recoveries.

For the RL side, start with [docs/RL-TRAINING.md](docs/RL-TRAINING.md). `Scenes/RL/Upright/RagdollStandArena.tscn` plays back a trained policy continuously with no episode resets, which is the honest way to see what a policy actually does.

## Third-party code

`addons/godot_rl_agents/` is vendored from the [godot_rl_agents](https://github.com/edbeeching/godot_rl_agents) project by Edward Beeching, MIT licensed. It is checked in so a fresh clone can open the RL scenes without a separate plugin install. See [addons/godot_rl_agents/VENDORED.md](addons/godot_rl_agents/VENDORED.md) for provenance and [addons/godot_rl_agents/LICENSE](addons/godot_rl_agents/LICENSE) for the license. Everything outside that directory is this project's own code.

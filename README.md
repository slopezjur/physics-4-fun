# Physics4Fun: Euphoria-Style Active Ragdoll

A purely physics-driven, active ragdoll locomotion and dynamic balance system for Godot 4 (.NET). This project is a tribute to the legendary character physics of GTA IV (Dynamic Motion Synthesis), rebuilt from first principles without relying on animation state machines or artificial world forces.

## Core Philosophy: 100% Grounded Biomechanics
Unlike traditional game ragdolls that use invisible springs or "floating" central forces to stay upright, this engine enforces strict biomechanical realism. Every movement, step, and balance correction is achieved purely through internal joint torques pushing against physical ground collisions. 

## Key Features (The 7 Pillars)

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
* **3D Spatial Awareness:** Raycast-driven environmental probing allows the dummy to push off walls to stabilize the torso without artificial aids.
* **Biological Get-Up AI:** A 5-state behavioral machine autonomously manages falls, seamlessly transitioning into 4-phase physical recoveries (e.g., prone push-up $\to$ quadruped crawl $\to$ deep squat $\to$ upright extension).

## Documentation

For a deep dive into the mathematics and architecture of the 7 Pillars, please review the full technical documentation:
* [Architecture & Biomechanical Model](docs/ARCHITECTURE.md)

## Requirements

* **Godot Engine 4.x** (Mono / .NET version)
* **.NET SDK** (Compatible with your Godot 4.x build)

## Setup & Testing

1. Open the project in Godot.
2. Build the C# solution.
3. Open `Scenes/ActiveRagdoll.tscn` or run `AutomatedTest.cs` to watch the active ragdoll balance, react to impulses, and attempt physical recoveries.

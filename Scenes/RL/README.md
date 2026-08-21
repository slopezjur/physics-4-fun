# RL scenes, grouped by brain

The folders are not tidying. **Everything inside one folder shares an observation and action space
*and* a compatible objective, so its checkpoints are interchangeable. Across folders they are not.**

The grouping axis is deliberately **shared-weight compatibility**, not conceptual function: two
scenes belong together if and only if one policy can serve both without destroying the other. That
is an empirical property, and it has been measured here — see below.

| folder | components (`TaskKind`) | scenes |
|---|---|---|
| `Upright/` | `UprightProgressReward` / `UprightTermination` (0) | Stand, GetUp, Perturbation |
| `Locomotion/` | `WalkForwardReward` / `WalkTermination` (1) | Walk |
| `Shared/` | — | `RagdollAIController.gd` |

Folder names match the component names on purpose. A scene in `Upright/` is exactly a scene that
builds the Upright pair, so the invariant is readable off the directory tree.

## Why walking is alone

The conflict is numeric, not philosophical:

- `UprightTermination.StandingMaxSpeed = 0.6 m/s` — standing **fails** above this
- `WalkForwardReward.TargetSpeed = 0.4 m/s` — walking is **rewarded** for reaching this

A walking dummy at target speed is two-thirds of the way to disqualifying itself from "standing".
Same state space, opposing gradients. Measured cost of ignoring that:

| | `standing/all` |
|---|---|
| stand policy | 0.478 |
| after 33M steps of walk training | **0.170** |

Get-up, by contrast, *preserves* standing — a get-up run measured 96% standing success — because
get-up **is** standing, from progressively worse start poses. That is why it lives in `Upright/`.

An earlier version of this layout grouped Stand + Perturbation + Walk as "Locomote" and split GetUp
off as "Recover". That cut straight across the component boundary: it separated GetUp from the two
scenes it shares components with, and united Walk with two it shares nothing with.

## How Upright's three tasks relate

They are the same objective from different starting conditions, which is why they share weights:

- **Stand** — always upright (t = 1.0). The reference task; train it first.
- **GetUp** — reverse curriculum, floor starts at `CurriculumPoseT = 0.99` (almost upright, where a
  stand policy already succeeds) and walks back toward flat. **Resume it from a stand checkpoint.**
- **Perturbation** — upright with a ball gun pushing back. Uses the same pair with
  `EndEpisodeOnStandingSuccess = false`, because success must not be absorbing when the whole point
  is to survive an impact that arrives *after* the success criterion would have fired.

## Training vs Arena

Each task has both, and they are not interchangeable:

- **`*Training.tscn`** — episodes with terminations and resets, driven by Python over a socket. This
  is what the exported build boots, selected by the preset's feature tag.
- **`*Arena.tscn`** — `PolicyAutoLoader` sets `PlaybackMode = true`, so nothing terminates or resets
  and the policy runs continuously. The honest test of a policy, because training metrics stop
  measuring a behaviour the moment an episode ends. Press **R** to reset manually — it calls
  `Bridge.ResetEpisode()`, so the start pose is re-sampled properly.

Every arena pins its own `PromotedModelPath`. Without that, `PolicyAutoLoader` falls back to the
newest `.onnx` anywhere under `rl/runs`, and a night of walk training silently repointed the
perturbation arena at a walking policy — the dummy fell before the ball ever reached it.

Arena start poses are set to match how each policy was trained. The perturbation arena inherited the
0.2 default once and spawned the dummy at a 1–2° tilt 80% of the time, inside the exact brittleness
band that makes it fall unaided.

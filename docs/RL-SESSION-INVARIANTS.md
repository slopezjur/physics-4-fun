# RL session invariants

Rules that decide whether a training session produced knowledge or produced nothing. Every one of
them is here because it was violated on this project and cost real time — the dates and numbers are
kept so the entries stay arguments rather than opinions.

Read the **pre-flight** list before starting a run. Read **invalidators** when a result looks wrong.

---

## Pre-flight (before every run)

| # | Check | Why |
|---|---|---|
| 1 | `$Task` in `config.ps1` matches the scene you mean to train | It selects the export preset and therefore which `.exe` is built. |
| 2 | `$ExperimentName` is empty (auto-derived) | Now automatic. If you pin it, the console prints `(pinned)` in yellow — make sure you meant it. |
| 3 | **Re-export after any C# or scene change** | Training runs from `build/*.exe`, never from the editor. An unexported change silently trains the old build. |
| 4 | Export output has **zero ERROR lines** | Don't grep for "Export complete" — it prints even when files fail to pack. Count errors explicitly. |
| 5 | Resume checkpoint belongs to the same brain | Upright ↔ Walk are not weight-compatible. See `Scenes/RL/README.md`. |
| 6 | Godot editor closed, or assembly rebuilt | The open editor rewrites deleted scenes from `.godot/editor/` caches and can hold a stale DLL. |
| 7 | No orphaned processes holding ports 11008+ | A hung session blocks the next one and looks like a training failure. |

---

## Invalidators — a session that broke one of these proves nothing

### 1. One variable at a time

Two simultaneous changes make the result unattributable, and you will not get that time back.

> **Cost:** the actuator retune and the ballistics fix went out in one run. `standing/all` fell
> 0.435 → 0.281 and neither of us could say which change caused it, or whether one had helped while
> the other hurt. The run had to be discarded as evidence.

The exception is when the variables are **separately measured**. Mixed-task training changes three
tasks at once but `task_stand/*`, `task_getup/*` and `task_perturbation/*` are distinct tags, so
each stays attributable.

### 2. Verify the physics before spending hours on the policy

A policy trained against broken dynamics learns to exploit the break.

> **Cost:** ~6 hours of perturbation training against a ball that delivered 0.081 N·s on some hits
> and 81 N·s on others, at random, depending on which bone it struck. Every reward number from those
> runs describes a body that no longer exists.

Before a long run, take one 10 s arena dump (**T**) and check the [physics sanity](#physics-sanity)
list below.

### 3. `ep_rew_mean` is not comparable across changes to reward, episode length, or termination

It is a sum over a window. Change the window or what ends it and the number moves for reasons that
have nothing to do with policy quality.

Judge instead on **per-tick fractions** (`standing/*`) and **per-task rates**
(`task_<name>/success`), which are length-invariant.

### 4. Per-tick fractions are not comparable across termination changes either

They are diluted by whatever the body does after failing.

> **Cost:** adding fall termination raised `standing/all` immediately, with no policy change at
> all — it simply stopped counting the ticks spent lying on the floor. Anyone comparing across that
> change would have recorded a large improvement that did not happen.

After any termination change, the first run is a **new baseline**, not a comparison.

### 5. A metric pinned at a constant is broken, not informative

Both directions are failures: always 0 and always alarming.

> **Cost (always-zero):** `start_prone/success` read 0.000 for 1179 rollouts. Get-up had never been
> trained; nobody noticed because zero looks like a valid measurement.
>
> **Cost (always-alarming):** `JointLimitViolations` read nonzero on 98.4% of ticks because the
> check decomposes quaternions in Euler YXZ, whose principal branch cannot represent the ±2.6 rad
> limits four axes on this rig actually have. A smoke alarm that is always on cannot report a fire.

### 6. Constant telemetry columns hide dead machinery

28 columns in a dump were identical on every row — `BalanceStrength`, `WeightShareL/R`,
`PelvisStabTorque`, all get-up phase columns. Under RL the balance controller is off by design, but
the columns still write, so a reader cannot distinguish *off by design* from *broken*.

If a column is constant for a whole run, find out which of the two it is before trusting anything
near it.

### 7. Names must be derived, never remembered

> **Cost:** `$ExperimentName` read `perturbation_v3` while three later experiments ran, so v4, v5 and
> v6 all wrote into `perturbation_v3_0` — two lineages, one directory, no way to separate them
> afterwards. Separately, an arena's `PromotedModelPath` was pinned to `perturbation_v1_0` and spent
> a full day showing a 61M-step policy while v7 sat unused on disk.

Both are now derived: experiment name from `$Task`, arena policy from `BrainRunPrefixes`. Do not
reintroduce a hand-maintained string.

---

## Physics sanity — check on one 10 s arena dump

Press **T** in any arena. All of these come straight from the CSV.

| Check | Healthy | Meaning if violated |
|---|---|---|
| `SystemEnergy` rise per tick | ≤ `ActuatorPowerInW × Delta` | **Energy is being created.** Nothing downstream is trustworthy. This is the master check — it catches injection from any source. |
| `<bone>_AngVelMax` | well under **47.12 rad/s** | That value is Jolt's own default angular-velocity clamp (`0.25·π·60`). A bone pinned there is being limited by the engine's safety net, not by your tuning. |
| `PeakJointPowerInW` | ≤ `MaxTorque × MaxShorteningVelocity / 4` | The Hill limit is not engaging. ~1500 W for a 400 N·m joint at 15 rad/s. |
| `BallImpactImpulse` | ≈ `mass × speed × (1 + restitution)` | Contact resolution is inventing or swallowing momentum. |
| `TorqueMax` vs `MaxTorque` | below ceiling most ticks | Saturation is bang-bang control wearing a PD costume. |
| `LoadTorqueMag` vs ceiling | under `LoadCompensationTorqueFraction × MaxTorque` | Feed-forward is crowding the PD term out of the budget. |
| `Chatter` column | no `CHATTER` | Joint reversing tens of times per second. |

### Two numbers that predict solver trouble

Both were firing at once when the ball bug was found:

- **Mass ratio above ~100:1.** The ball was 0.0025 kg against a 16 kg chest — 6,400:1.
- **Per-tick travel exceeding the target's shape thickness.** The ball crossed 0.052 m per tick
  against 0.05 m forearms.

Sequential-impulse solvers are built for persistent contacts between comparable masses. Outside that
envelope, resolve impacts analytically instead — see `BallGun.AgeBalls`.

---

## Debugging discipline

Ordered by how much time each would have saved on the day they were learned.

### 1. Remove the suspected cause before fixing anything

If the symptom survives without it, you were about to fix the wrong thing. If it disappears, you have
halved the search space for the cost of one run.

> **Cost:** four consecutive changes — ball mass, ball size, actuator ceilings, actuator damping —
> were made on the assumption that the ball impact was too strong. Unchecking `Enabled` on the
> BallGun took ten seconds and settled more than all four combined.

### 2. Instrument before hypothesising

> **Cost:** the ball bug was invisible for a day because nothing recorded system energy, per-bone
> linear velocity, signed actuator power, or ball impulse. Once those existed the fault was obvious
> in one dump. The columns took twenty minutes to add.

### 3. A parameter that does nothing across a large range is not the mechanism

> **Cost:** ball mass was halved four times — 0.2 → 0.0025 kg, an 80× reduction — with the symptom
> unchanged at every step. That is not how a mass parameter behaves, and it should have redirected
> the search immediately. It did not, and later measurement showed the ball's momentum was never
> the energy source at all.

### 4. Distinguish measuring the thing from measuring its consequence

`ActuatorPowerW` was originally `|τ·ω|`, which counts braking as power. It read 26 kW *because* the
limbs were already moving fast, then got used as evidence that the actuators were driving them.
Signed power (`ActuatorPowerInW` / `OutW`) separates cause from effect.

### 5. A comment explaining why something is safe is not evidence that it is

The action space carried a comment stating "the RL action path does not depend on this". It was
correct. The neighbouring comment claiming "aim is not the problem" was written with equal
confidence and was wrong — the shots were landing 0.54 m low. Re-derive load-bearing claims.

---

## Task-specific traps

### Perturbation

- `EndEpisodeOnStandingSuccess = false` is correct — success must not be absorbing, or the episode
  ends before the ball lands. **Failure must still terminate**, which is a separate flag
  (`EndEpisodeOnFall`). Conflating them left every episode running its full window face-down.
- `IntervalSeconds` must exceed `MaxEpisodeSeconds` so exactly one impact lands per episode.
  Otherwise a fall cannot be attributed to a specific hit.
- The ball is **absent from the observation** by design — the policy learns recovery from
  proprioception, not dodging. That makes every miss exogenous return variance the critic cannot
  explain, which is why `ball/hit_rate` near 1.0 matters as much as the hit itself.

### Get-up

- Starts prone, so it is below every fall threshold from tick one. Any unconditional failure check
  ends it immediately.
- The curriculum floor must survive a resume, or the reverse curriculum silently restarts each
  session. Verify `standing/curriculum_t` has `min == max == last` across a chained resume.

### Walk

- No success condition, deliberately: walking is sustained behaviour with no goal state, and a
  distance threshold would end the episode at the moment the agent is doing the thing being trained.
- Rewards must be **product-form**, not additive. An additive walk reward produced a policy that
  stood still, because standing scored better than risking a step.

---

## Two-minute pre-run script

```
1. config.ps1     Task correct?  ExperimentName auto?
2. Export         re-run it, count ERROR lines
3. Arena dump     press T, check the physics sanity table
4. 5-minute run   check hit_rate, approx_kl < 0.1, and the task's headline metric
5. Long run       only now
```

Step 4 has caught something on most occasions it was run. Skipping it has never once saved time.

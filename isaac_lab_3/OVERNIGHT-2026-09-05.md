# Overnight plan — get the dummy walking in Godot

Written 2026-09-05 ~00:40. Supersedes nothing; `SIM-TO-SIM.md` remains the ledger of what is known.

## RULE ZERO — NEVER BLOCK ON THE USER

**Do not stop at a phase boundary to ask anything. Do not end a turn with a question and wait.**
The user is asleep and will read the status in the morning. Every hour spent idle is an hour lost.

If a decision point arrives — a phase gate, an ambiguous result, a choice between two next steps —
**pick the option best supported by the measurements, record the reasoning, and keep moving.** A
recorded judgement call the user can overturn in the morning is worth infinitely more than a
question they cannot answer until then.

If a phase completes early, move to the next. If ALL phases complete, do not stop: return to Phase A
with wider randomisation, sweep more checkpoints, or attack the next item in `SIM-TO-SIM.md`'s
open list. There is always more measurement available. The goal is the best sim-to-sim transfer
achievable by morning, not the completion of a checklist.

The only acceptable reasons to stop working are: the goal is achieved and hardened, or the machine
is genuinely unable to continue (and then say exactly why).

## Status at handover, all measured tonight

Three 15-minute fragments from `walk_noair/model_21122`, each resumed with
`cmd_curriculum=False cmd_speed_min=0.35 drive_overspeed_sigma=0.5 rew_feet_air_time=0.0`
(the last one MUST be re-passed every fragment — reward weights are not in `TRAINED_CONDITIONS`
and a resume silently restores the 10.0 default, which undoes the gait fix):

| | Isaac single support | Isaac flight | Godot upright @0.15 | Godot @0.10 |
|---|---|---|---|---|
| frag 1 | 68.3% | 1.2% | 6.2% | statue, 100% |
| frag 2 | 71.1% | 0.8% | 48.2% | statue, 100% |
| **frag 3** | **78.5%** | **1.1%** | **68.6%** | **96.6% upright, 3 strikes, -0.99 m** |

**This is the first thing all session that has moved the Godot number.** It is also the first
checkpoint that both stays upright AND takes steps. Displacement is still negative (walking
backwards) and step count is low, so it is not walking yet.

Live checkpoint: `logs/rsl_rl/p4f_newton_walk_rand/2026-09-04_23-51-36/model_23180.pt`

### GPU CONTENTION invalidated several fragments

A training run launched at 04:05 had, at 06:20 wall clock, completed only **3 minutes 39 seconds of
training**. Godot scoring runs and Isaac observation dumps were running concurrently the whole time
and starved it. Several "15-minute" fragments in the log above therefore had far less training than
their label claims, and their flat results should not be read as evidence that training plateaued.

**Do not run scoring concurrently with training.** The measurement work uses the same GPU. Sequence
them: train, then score, then train.

## Corrected-body fragment log (Phase A on the fixed geometry)

Warm-started from `walk_rand/model_23600`. Best checkpoint per fragment, Godot upright %:

| frag | @0.15 | @0.10 | best displacement | note |
|---|---|---|---|---|
| 1 | 7.2% | 24.4% | +0.38 m | body just changed under the policy |
| 2 | 9.2% | 15.8% | **+0.73 m** | travels furthest, falls |
| 3 | **21.2%** | 64.1% | +0.15 m | |
| 4 | 18.2% | **69.5%** | +0.01 m | |
| 5 | 16.7% | 49.0% | +0.46 m | |

**Kill gate met after fragment 5:** three consecutive fragments flat-to-declining on both authorities
(0.15: 21 -> 18 -> 17; 0.10: 64 -> 70 -> 49) with no displacement trend. Nothing walks - step counts
stay at 1-4 over 20 s and displacement is as often negative as positive.

Moving to Phase B (widened randomisation) after fragment 6.

### Fragment 6 lost — untracked launch

Fragment 6 was launched with PowerShell `Start-Process` instead of a harness-tracked background
call, to save a turn. It created its log directory, failed, and wrote no checkpoints - and because
nothing was tracking it, the failure surfaced only 15 minutes later when the directory turned out to
be empty. **Launch training through the tracked mechanism; the notification is the point.**

## PLAN SUPERSEDED 02:40 — the body itself was wrong

Every box collider in the Isaac USD was exactly HALF its authored size, including the FEET
(0.11 x 0.06 x 0.04 instead of 0.22 x 0.12 x 0.08). See `SIM-TO-SIM.md` for the full table. Isaac has
been training every policy in this project's history on a dummy with half Godot's fore-aft support
base - the single most important parameter a biped balances on.

Capsules and spheres were correct, so all four limbs matched and total mass agreed at 80.6 kg;
only the four boxes were wrong, which is why no rig check ever caught it.

**This invalidates every checkpoint and it invalidates the fragment log below.** Phases A and B were
optimising a policy on the wrong body. `CONTACT_HEIGHT` also moved 0.034 -> 0.05, because Isaac's
planted foot went from 0.0081 to 0.0286 m and what must match is the 0.020 m CLEARANCE Godot has.

**New Phase A:** identical fragment protocol, on the corrected body, warm-started from
`walk_rand/model_23600` (the best gait so far - the weights are still a reasonable prior even though
the body changed). Same recipe, same per-fragment sweep, same gates. Experiment `p4f_newton_walk_geom`.

Everything below this line predates the geometry fix and is kept for the record.

## Phase A — keep going, 15-minute fragments (first ~4 h)

Repeat the exact recipe above, resuming each fragment from the previous checkpoint. **After every
fragment, without exception:**

1. `dump_obs.py --command 0.30 --no_reset_noise` -> single support / flight / speed
2. `godot_walk_score.py --authorities 0.15 0.10` -> displacement / steps / upright
3. Restore `Models/` from the backup afterwards (the scorer promotes as a side effect)

Record every fragment in a table, including the ones that do not move.

**Success gate (stop Phase A, go to Phase C):** Godot upright >= 90% at authority 0.15 AND
displacement >= +1.0 m with >= 10 foot strikes over 20 s. That is walking, not surviving.

**Kill gate (go to Phase B):** three consecutive fragments with no improvement in BOTH Godot upright
at 0.15 and displacement. Not two - the frag1->frag2 jump was 6% -> 48% after a flat fragment.

**Watch for:** displacement is NEGATIVE right now. If uprightness keeps climbing while displacement
stays negative, the policy is learning to back away from the command; check `linVel_x` sign in the
Isaac dump too, and if Isaac is positive while Godot is negative, that is a heading/frame bug worth
its own investigation before more training.

### Heading check — CLEARED 2026-09-05 00:50

The plan flagged the negative Godot displacement as a possible heading/frame bug. It is not.
Open-loop replay, same actions into both engines: Godot's `linVel_x` is POSITIVE for the first
0.33 s (peak +0.43) and only goes negative as `grav_z` collapses -1.00 -> -0.06. It walks forward,
then topples backwards. The negative displacement is the fall, not a sign error. `cmd_x` also
confirmed sticking at 0.300 in the Isaac dumps from row 10 on, so the reference is fair.

Do not re-investigate this.

### PROTOCOL CHANGE 2026-09-05 01:15 — score THREE checkpoints per fragment

Fragment 4 looked like a collapse (68.6% -> 7.2% upright at 0.15) and was not. Scoring a MID-fragment
checkpoint from the same run:

| checkpoint | position in fragment 4 | Godot upright @0.15 |
|---|---|---|
| `model_23180` | start (= frag 3 end) | 68.6% |
| **`model_23600`** | **middle** | **78.6%** |
| `model_23866` | end | **7.2%** |

**The policy peaks INSIDE a fragment and then destroys itself within ~270 iterations.** Scoring only
the final checkpoint hides the best one and, worse, chains the next fragment off a wreck.

**So: score early / middle / late of every fragment, and resume the next fragment from the BEST of
the three by Godot upright, not from the last.** Keep fragments at 15 min rather than shortening
them - startup costs ~3-4 min, so shorter fragments spend proportionally more time booting than
training. More resolution comes from more checkpoints, not shorter runs.

Running best so far: `walk_rand/2026-09-05_00-17-05/model_23600` at 78.6% upright @0.15.

### Fragment log

| frag | resumed from | Isaac single / flight | Godot upright @0.15 | @0.10 | note |
|---|---|---|---|---|---|
| 1 | walk_noair/21122 | 68.3% / 1.2% | 6.2% | statue 100% | |
| 2 | rand/21808 | 71.1% / 0.8% | 48.2% | statue 100% | first movement |
| 3 | rand/22494 | 78.5% / 1.1% | 68.6% | 96.6%, 3 strikes | first upright AND stepping |
| 4 | rand/23180 | - | **78.6% (mid)** / 7.2% (end) | 74.6% (mid) | peak is mid-fragment |

### Why the transfer stalls — diagnosed 2026-09-05 ~02:00

The Godot uprightness gains are the policy learning to STAND STILL, not to walk. `model_23600` holds
0.78-0.81 m for 15 s with BOTH contact flags at 1 in every single sample - it never lifts a foot -
then falls. In Isaac the same checkpoint has 78.5% single support at 0.67 m/s.

**Same checkpoint, action statistics on the gait DOFs:**

| dof | Godot sd | Isaac sd | Godot \|mean\| | Isaac \|mean\| |
|---|---|---|---|---|
| `Thigh_L:0` | 0.313 | **0.670** | **0.718** | 0.070 |
| `Shin_L:0` | 0.205 | **0.875** | **0.855** | 0.075 |
| `Shin_R:0` | 0.155 | **0.819** | **0.929** | 0.288 |

Isaac oscillates around zero - a rhythm. Godot barely oscillates and sits at a large constant offset
- a held posture. **The policy is memoryless (MLP), so its gait phase lives entirely in the
observation.** Godot's feet never lift, so contacts stay (1,1) forever, so the policy never sees the
swing-phase signal, so it holds a posture, so the feet never lift. A fixed point of the closed loop.

**The span mapping makes the dead zone worse.** `Shin:0` limits are [-2.6, +0.1], and the action map
uses `span = upper` for a POSITIVE action. So a positive knee action produces
`0.15 * 0.855 * 0.1 = 0.013 rad` - nothing. Godot's policy parks on the positive side and therefore
has almost no knee authority; Isaac's swings negative half the time and gets real flexion.

**Observation slices, same checkpoint, only one is out of distribution:**
height (Godot 0.796 vs Isaac 0.752, z = 1.56). That is a CONSEQUENCE - Isaac walks with flexed knees
and a lower pelvis. At rest the two engines agree exactly (0.8293 vs 0.8294), so it is not a
calibration error.

**Ruled out as ways to break the fixed point:**

| attempt | result |
|---|---|
| push 6 N.s at t=3 | single support 10.8%, no sustained gait |
| push 12 N.s at t=3 | topples (2.8% upright) |
| command 0.70 | single support 1.0%, upright 28.7% |
| command 1.20 | single support 2.9%, upright 8.3% |

### Symmetry-breaking at spawn — TESTED, does not start the gait

Hypothesis: Isaac resets with `reset_joint_noise = 0.1` while Godot spawns at the exact rest pose, so
a memoryless policy on a perfectly symmetric body issues symmetric commands, both legs answer
together, and the contact flags never separate. Implemented as `SpawnActionNoise` (uniform noise on
every action for the first 0.5 s; kept, default 0).

| spawn noise | single support | steps | upright | displacement |
|---|---|---|---|---|
| 0.0 | 6.2% | 3 | 78.6% | +0.09 m |
| **0.15** | 5.9% | 5 | 42.2% | **+0.40 m** |
| 0.35 | 2.3% | 3 | 6.9% | +0.05 m |

**Falsified.** Single support does not move; the fixed point is not held in place by symmetry. Worth
noting 0.15 produced the largest forward displacement measured so far (+0.40 m), but at the cost of
half the uprightness, so it is a perturbation not a gait.

### Tooling trap fixed: `--set` could not take a tuple

`--set action_scale_range="(0.7,1.4)"` stored the literal STRING. `apply_overrides` parses by the
existing field's type and tuples fell through to the string branch, so it exploded at the first
reset - "too many values to unpack" - **fifteen minutes in, after 24576 environments had booted**.
`run_conditions.apply_overrides` now parses tuples and checks the arity up front.

## Phase B — widen the randomisation (if Phase A plateaus, ~2 h)

Robustness is what is driving the gain, so give it more. In order, one 15-minute fragment each,
keeping whichever helps:

1. `action_scale_range` 0.8-1.25 -> 0.7-1.4
2. `effort_scale_range` 0.65-1.0 -> 0.5-1.05
3. Both together

**Only the action pipeline works on this backend.** Solver-side plant randomisation is INERT here -
`write_joint_stiffness_to_sim`, `_damping_`, `_effort_limit_`, `_armature_` all accept a per-env
write, read back correctly, and change nothing (rank correlation -0.014 to -0.257). Do not spend
time on it again; see `SIM-TO-SIM.md`.

## Phase C — harden, if walking is achieved (~2 h)

1. Push test at authority 0.15 (`PushImpulse` in the scene) — a gait that cannot take a shove is not
   done.
2. Authority ladder upward: 0.20, 0.25. The trained scale is 0.15; anything above is margin.
3. Stand regression: `Stand/IsaacStandCheckNewton.tscn` at 0.10 must still stand.
4. Promote the winning checkpoint to `Models/` and record its provenance in the contract.

## Phase D — consolidate (last 30 min regardless of outcome)

Update `SIM-TO-SIM.md` with the fragment table and the outcome. Update memory. Leave `Models/`
holding the best checkpoint by GODOT score, not by Isaac reward. Report honestly, including any
fragment that regressed.

## Standing rules for the night

- **15-minute fragments only.** No long runs. The 132-minute run tonight peaked at 35 minutes and
  spent the next 97 getting worse.
- **Verify every flag actually applied** before believing a result. Two Jolt sweeps and one gain
  sweep were run tonight on settings that were never read. Check the run's own `params/env.yaml`.
- **Godot is bit-deterministic** — five identical runs gave identical numbers. A difference between
  runs is real, not noise.
- **Do not chase static-pose agreement.** Measured tonight: a change that made Godot's settled pose
  closer to Isaac's destroyed transfer (100% -> 15.4% upright). Score the ladder.
- Kill leftover `python`/`Godot*` processes after every run; report the count including zero.
- No commits.

# Overnight session — honest Stand to convergence, then Perturb

## Step one, before anything runs

Copy this plan into the repo as `isaac_lab_3/NIGHT-PLAN-2026-08-28.md`, beside the existing
`NIGHT-PLAN.md` from the previous session. The session is about to be compacted, so a plan that
lives only in the assistant's context or in a scratch path is a plan that will be lost. The repo
copy is also what the morning report gets appended to, so the night has one durable record.

## Context

Today established the first policy on this project that learns to balance with **no cheat**: full
300 N·m stabiliser, reaction conserved into the planted thigh, at 8 XPBD iterations. It reached 833
iterations, holds ~14 s headless in Godot, and you confirmed it looks right visually at both Jolt
2/10 and 16/30 — so the transfer is not solver-sensitive.

night06 (3,087 iterations) still stands longer, but it is built on a free torque. The honest lineage
is a quarter of the way there and nearly matching it. Tonight is about riding that curve, then using
the result to seed Perturb.

## Phase 0 — enable the chain (~10 min)

`night.py` already does the right thing: chained resume segments with a scored checkpoint after each,
so a plateau or divergence is visible within minutes rather than at the end. Two gaps:

- It has `--xpbd_iterations` but no way to pass `balance_reaction` / `balance_max_torque`. Add a
  `--set` passthrough to `train.py` (the flag already exists there).
- Default `--xpbd_iterations 2`; the honest line needs 8.

Resume conditions now carry automatically via `run_conditions.py`, so segments after the first
inherit the plant without being told.

## Phase 1 — Stand, honest, ~4 h

```
night.py --task P4F-Dummy-Stand-Newton-v0 --experiment p4f_newton_honest_s8
         --resume <honest_s8 newest> --xpbd_iterations 8 --minutes 15 --num_envs 24576
         --set balance_reaction=True --set balance_max_torque=300
```

15-minute segments, each scored by `evaluate_stand.py` against the strict criterion, appended to
`SUMMARY.md`. The zero-action baseline is measured once at the start — a policy is only meaningful
relative to it.

**Stop early if** three consecutive segments fail to improve the strict score. That is a plateau, and
more of the same will not fix it.

## Phase 2 — Perturb curriculum, built while Stand trains (~1 h)

The existing `push_impulse_range = (0, P4F_PUSH_MAX)` is a fixed range, and it is what produced two
dead checkpoints today: at a ceiling of 25 the mean shot was 12.3 N·s against a policy that survived
8, so most episodes were unwinnable and the policy correctly learned nothing mattered.

Replace with a **survival-gated ramp** in `perturb_env.py`: track the fraction of episodes surviving
the delivered impulse over a window; raise the ceiling only when that fraction exceeds a target
(~0.7), lower it when it falls below a floor (~0.4). Start at the measured threshold, not at the
ball's 18 N·s. This is the thing that stops Perturb producing dead policies.

## Phase 3 — Perturb, ~3 h

Seeded from the best Stand checkpoint by **Godot** score (`compare_godot.py`), not Isaac score — the
Isaac metric has repeatedly failed to predict transfer here. Same chained-segment approach.

## Phase 4 — the Walk gate

**Do not start Walk tonight.** Walk deletes five of Stand's reward terms and adds four; it is a
different problem and a bad Stand foundation poisons it. Record whether the gate is met, for you to
decide in the morning:

| gate | threshold |
|---|---|
| Godot stand check | `=> STANDING` for the full 20 s |
| Push survived | ≥ 8 N·s at authority 0.10 (night06's benchmark) |
| Isaac strict score | ≥ 80% sustained over 3 segments |

## What I will NOT do unattended

**Change rewards.** You raised extending them for longer stands or bigger hits. Reward changes break
comparability with every prior run (invalidator #4) and cannot be evaluated against a baseline
measured under the old reward. If Stand plateaus I will record it and stop, not start tuning shapes
overnight.

**Promote to Godot.** `Models/` currently holds the honest brain (falls at ~14 s). I will leave it
alone and report what the best checkpoint scores, so the promote decision stays yours.

**Delete anything.** Scratch lineages from today (`carrytest*`, `cap*`, `honest_cap150`,
`honest_solver8`) stay until you say otherwise.

## Process hygiene — after EVERY run, not just at the end

Check for and kill leftover `python`, `Godot*` and any Isaac processes after each segment, each
Godot check, and each phase boundary. A GPU training process nobody is watching costs the whole
machine, and a stale one will also contaminate the next measurement.

This is not hypothetical. Today: a `watch.ps1` render defaulted to `Enter = 1` on a null stdin and
started an unattended 24576-env training run that wrote 8 checkpoints into a real lineage before it
was spotted; a killed sweep left its child `python` alive; and the first benchmark sweep read ~50%
low because background load was never checked. Every one of those was a process nobody meant to be
running.

    Get-Process python,Godot* -ErrorAction SilentlyContinue

Kill anything not part of the current step, and report the count in the morning summary — including
zero, so "I checked" is distinguishable from "I forgot".

## Verification

1. First segment's `params/env.yaml` shows `balance_reaction: true`, `iterations: 8`,
   `action_scale: 0.15` — checked before trusting any number, as it has silently reverted 4× today.
2. `SUMMARY.md` gains one scored row per segment, with the zero-action baseline at the top.
3. Godot check on the best checkpoint at the end, reported not promoted.
4. No stray processes at any phase boundary, and none left at the end.
5. A morning report: strict-score curve, best checkpoint, whether the Walk gate is met, the
   curriculum's final impulse ceiling, and the process check.

---

# Running log — 2026-08-28

## Phase 0 (02:08) — done. Two edits, not one.

`night.py` gained a `--set` passthrough, forwarded to **both** the training segments and the
evaluations.

The second gap was not in the plan and mattered more. **`evaluate_stand.py` never restored the
trained conditions** — it built `env_cfg` straight from the task registry, so every segment tonight
would have been scored at `action_scale 0.4 / balance_assist 0.0 / balance_reaction False` against a
lineage trained at `0.15 / 1.0 / True`. Measured cost of that substitution on `honest_s8/model_833`:

| measured on | strict standing | mean head |
|---|---|---|
| the plant it was trained on | **68.0 %** | 1.181 m |
| task defaults (the pre-fix path) | 5.5 % | 0.449 m |

A 68% policy read as 5.5%. The plateau rule ("stop after 3 non-improving segments") would have
fired on that noise within the hour. Fixed by restoring from the checkpoint's own `params/env.yaml`,
plus a `--conditions_from` flag so the **zero-action baseline** — which has no checkpoint of its own
— is measured on the same body as the rows it is the baseline for.

## Phase 1 (02:08 →) — honest Stand chain, running

`p4f_newton_honest_s8`, resumed from `model_833`, 24576 envs, 15-min segments, until 06:15.

Verification 1 passed on `2026-08-28_02-08-44_night01/params/env.yaml`:
`iterations: 8`, `action_scale: 0.15`, `balance_assist: 1.0`, `balance_reaction: true`,
`balance_max_torque: 300.0`.

Zero-action baseline **0.4 % standing, 0.411 m** — measured on the honest plant. Worth keeping:
with `balance_assist 1.0` and the reaction **on**, doing nothing scores 0.4%. The stabiliser does
not stand the dummy up by itself, which is the thing "honest balance" was supposed to establish.

## Phase 2 (02:12–02:20) — survival-gated curriculum, built and behaviourally tested

`perturb_env.py` samples from a live `_impulse_ceiling` instead of the fixed config ceiling. Every
window of 8192 **hit** episodes (episodes that fell before the first shot carry no information about
the impulse and are not counted) it re-decides: raise ×1.10 above 70% survival, lower ×0.85 below
40%, hold between, clamped to [2.0, 18.0] N·s — 18.0 being the real Godot shot.

Two defects found while building it:

* `_was_hit` cannot answer "was this episode hit". With `push_interval_s > 0` each landed shot
  re-arms the next through `_draw_shot`, which clears the flag — so it reads False on an env that
  was just hit. Left alone (it feeds no reward and nothing tonight reads it); the curriculum counts
  with its own `_hits_this_episode`, cleared only on reset.
* The curriculum had to be made inert under `playback`. `evaluate_stand.py` patches `_get_dones` so
  nothing ever falls, so survival would read 100%, the ceiling would ramp *during* the measurement,
  and late episodes would be scored against a harder task than early ones.

Behavioural test, both branches (not just "it parses" — that reasoning failed twice yesterday):

```
[curriculum] survival 20% over 257 hit episodes; ceiling up   10.00 -> 11.00 N.s   (clamped at max)
[curriculum] survival  0% over 257 hit episodes; ceiling DOWN 11.00 ->  9.35 N.s
[curriculum] survival  4% over 256 hit episodes; ceiling DOWN  8.50 ->  7.22 N.s
```

## Phase 3 prerequisite — already satisfied

The scalar/log std mismatch I flagged before starting is **not** a blocker for seeding Perturb from
Stand: `train.py --init_from` already loads with `strict=False` and re-inflates the noise to the
config's `init_std` on purpose. The mismatch only ever bit the playback entry points.

## Process hygiene

| time | check | result |
|---|---|---|
| 02:05 | pre-start | 0 python / 0 Godot — clean start |
| 02:14 | after curriculum smoke | 2 python: night orchestrator + its training child. Smoke exited clean |
| 02:20 | after raise-branch smoke | 2 python, same two. Nothing stray |

Two scratch artefacts from my own smoke tests:

* `isaac_lab_3/scripts/logs/` — 23 MB written to the wrong place (I ran from `scripts/`, so the
  relative log root resolved there). **Removed** — it was mine, minutes old, not gitignored, and
  would have landed in the next commit.
* `isaac_lab_3/logs/rsl_rl/p4f_smoke_curriculum/` — **left in place**, per "delete nothing
  unattended". Remove with `rm -rf isaac_lab_3/logs/rsl_rl/p4f_smoke_curriculum` when you want it gone.

## Phase 1 result (02:08–04:21) — 8 segments, then the plateau rule fired

| segment | checkpoint | strict standing | head | fell | mean \|a\| |
|---|---|---|---|---|---|
| — | ZERO-ACTION | 0.4 % | 0.411 | 100.0 % | 0.000 |
| — | model_833 (start) | 68.0 % | 1.181 | — | — |
| 1 | model_1519 | 73.0 % | 1.249 | 28.1 % | 0.591 |
| 2 | model_2254 | 76.2 % | 1.280 | 25.0 % | 0.565 |
| 3 | model_2989 | 77.7 % | 1.280 | 22.3 % | 0.550 |
| 4 | model_3724 | 78.9 % | 1.294 | 21.5 % | 0.559 |
| 5 | **model_4459** | **83.6 %** | 1.334 | 17.2 % | 0.540 |
| 6 | model_5194 | 81.6 % | 1.315 | 19.1 % | 0.543 |
| 7 | model_5929 | 83.2 % | 1.334 | 17.2 % | 0.559 |
| 8 | model_6664 | 82.4 % | 1.324 | 18.0 % | 0.538 |

Segments 6–8 all failed to beat segment 5, so the plan's stop rule fired and I stopped the chain at
04:21. Head height flattened at the same time (1.334 / 1.315 / 1.334 / 1.324), so this is a real
plateau and not just the eval's ~2.4 % sampling error at 256 envs.

## The result that matters, and it is not the Isaac curve

Godot, measured on three separate checkpoints spanning the plateau:

| checkpoint | Isaac strict | Godot authority | Godot push survived |
|---|---|---|---|
| model_833 (before tonight) | 68.0 % | 0.05 | **8 N·s** |
| model_4459 | 83.6 % | 0.05 | 0 N·s |
| model_5929 | 83.2 % | 0.05 | 0 N·s |
| model_6664 | 82.4 % | 0.05 | 0 N·s |

**+15.6 points of Isaac strict score bought no Godot authority and cost all of the push
robustness.** Three independent checkpoints, same answer, so this is not a bad draw.

The likely mechanism is visible in the table above: `mean |a|` falls 0.591 → 0.538 across the night
while the strict score rises. The strict criterion contains `speed <= 0.6`, so a policy that moves
less scores better on it — and a policy that moves less has less authority in reserve to reject a
real shove. That is the statue failure mode this project has already recorded once, arriving through
a different door: not a solver ignoring effort limits this time, but a metric quietly paying for
stillness.

This is the eighth time the Isaac score has failed to predict Godot transfer, and it is the
strongest argument yet that Stand alone cannot be trained to a transferable policy - **nothing in
Stand's reward pays for push rejection**, so nothing stops the optimiser trading it away.

## Phase 3 (04:40 →) — Perturb, seeded from model_833

Seeded by **Godot** score as the plan requires, which means the pre-night checkpoint, not the best
Isaac one. `model_833` is the only checkpoint of the four that rejects an 8 N·s push at all, and the
survival-gated curriculum starts from what the policy can already survive - seeding from a 0 N·s
policy would just ramp the ceiling down to its 2.0 floor and train nothing useful.

Perturb is also the direct corrective for the finding above: its reward *does* pay for surviving the
shove, so the trade the Stand optimiser kept making is no longer free.

Running until 08:30 with the honest plant passed explicitly - `--init_from` seeds weights only and
does **not** trigger the conditions restore (that fires on `--resume`), which is precisely what the
new `--set` passthrough is for. Later segments resume and inherit it automatically.

## Phase 3 attempt 1 (04:35–04:53) — the curriculum destroyed the policy. My defect.

The ramp found the policy's edge correctly (~3.2 N·s), reversed, and climbed. Then it kept climbing:

```
3.21 -> 3.53 -> 3.88 -> 4.27 -> 4.69 -> 5.16 -> 5.68 -> 6.25 -> 6.87 -> 7.56
     -> 8.31 -> 9.15 -> 10.06 -> 11.07 -> 12.17 -> 13.39 -> 14.73 -> 16.20 -> 17.82 -> 18.00
```

3.2 to 18.0 N·s in **four minutes**. At the cap survival collapsed and did not come back:

| ceiling | 18.00 | 18.00 | 15.30 | 13.00 | 11.05 | 9.40 | 7.99 | 6.79 | 5.77 |
|---|---|---|---|---|---|---|---|---|---|
| survival | 76 % | 37 % | 22 % | 12 % | 6 % | 2 % | 1 % | 0 % | 0 % |

Survival falling *while the task gets easier* is the tell: the policy was degrading, not being
out-matched. Same collapse that killed the earlier Perturb runs, reached by a new route.

**The cause was my window size.** 8192 hit episodes at 24576 envs is one decision every ~13
seconds, so the ceiling tracked episode throughput rather than policy learning — 18 raises before
PPO had consolidated any of them. Then at 18 N·s most episodes were unwinnable, and the ×0.85
retreat was far too slow to give the behaviour back once it was gone.

I read the early descent as a mis-specified gate (episode survival being a conjunction over 2–3
hits per episode) and said I would restart on those grounds. That was wrong — the ramp reversed and
climbed on its own, so the gate was reachable. The real defect was the rate, not the threshold.

Fix, in `perturb_env_cfg.py`: window **8192 → 65536** (≈ one decision every two minutes at this env
count, the same timescale the policy learns on) and raise factor **1.10 → 1.05**. Lower factor stays
0.85 — retreating faster than advancing is still right. Plus the hold-logging added earlier, so a
ceiling that has found its equilibrium keeps reporting instead of going silent.

## Phase 3 attempt 2 (04:55 →) — `p4f_newton_perturb_honest2`

Re-seeded from `model_833` (the collapsed attempt is not a foundation), same honest plant, until
08:15. Attempt 1's tree is left in place under `p4f_newton_perturb_honest` — it is the evidence for
the table above.

---

# MORNING REPORT — 08:30

## Phase 3 attempt 2 — the curriculum works; PPO does not

13 segments, 04:42 → 08:16. The slowed ramp fixed the runaway completely: survival held in the
81–88% band and the ceiling oscillated sanely, ending at **10.17 N·s** (from a 3.2 N·s start).

The policy, however, alternates between two modes:

| segment | checkpoint | strict | head | fell | mean \|a\| |
|---|---|---|---|---|---|
| — | ZERO-ACTION | 1.2 % | 0.471 | 100 % | 0.000 |
| 1 | model_735 | 100.0 % | 1.327 | 38.7 % | 0.622 |
| 2 | model_1470 | **0.0 %** | 0.308 | 100 % | **1.105** |
| 3 | model_2205 | 98.4 % | 1.449 | 6.6 % | 0.640 |
| 4 | model_2940 | 99.2 % | 1.471 | 3.9 % | 0.649 |
| 5 | model_3675 | **0.0 %** | 0.319 | 100 % | **1.103** |
| 6 | model_4410 | 100.0 % | 1.465 | 5.1 % | 0.595 |
| 7 | model_5145 | 64.1 % | 0.567 | 100 % | **1.848** |
| 8 | model_5880 | 99.2 % | 1.452 | 8.2 % | 0.682 |
| 9 | model_6615 | **0.0 %** | 0.312 | 100 % | **2.373** |
| 10 | model_7350 | 98.8 % | 1.455 | 8.2 % | 0.609 |
| 11 | model_8085 | **0.0 %** | 0.324 | 100 % | **2.923** |
| 12 | model_8820 | 68.4 % | 0.963 | 61.7 % | 1.021 |
| 13 | model_9310 | 96.5 % | 1.336 | 21.9 % | 0.736 |

Every dead segment has `|a| > 1.1`, and it grows across the night (1.10 → 1.85 → 2.37 → 2.92). The
policy is periodically blowing its actions past the `[-1, 1]` the barrier is supposed to hold, which
slams every joint target to its limit and puts the dummy on the floor. Good segments sit at ~0.6.
**This is a training-stability problem, not a curriculum problem** - and it is the same exploding-
action term that was misdiagnosed yesterday from truncated output.

## Godot — the only measurement that has ever predicted anything

| checkpoint | Isaac strict | Godot authority | push survived |
|---|---|---|---|
| **model_833** (last night's starting point) | 68.0 % | 0.05 | **8 N·s** |
| Stand model_4459 / 5929 / 6664 | 82–84 % | 0.05 | 0 N·s |
| Perturb model_4410 | 100.0 % | 0.05 | 0 N·s |
| **Perturb model_7350** | 98.8 % | 0.05 | **8 N·s** |
| Perturb model_9310 | 96.5 % | 0.05 | 0 N·s |

**Nothing trained tonight beats `model_833` in Godot.** `model_7350` matches it. Seven hours of
training on two tasks moved the Isaac score from 68% to ~99% and moved the Godot result not at all.

An Isaac score of 100% and 0 N·s in Godot (model_4410) is the cleanest statement of the gap this
project has produced.

## Walk gate — NOT met. Do not start Walk.

| gate | threshold | actual |
|---|---|---|
| Isaac strict | ≥ 80 % over 3 segments | **met** (Stand segments 5–7) |
| Godot push | ≥ 8 N·s at authority **0.10** | **not met** - every checkpoint tops out at authority 0.05 |
| Godot stand | STANDING for the full 20 s | not verified |

The authority ceiling of 0.05 is the thing to look at. Every checkpoint measured tonight - honest
Stand, collapsed Perturb, good Perturb - lands on exactly 0.05. That is not a policy property, it is
a wall, and no amount of further training has moved it.

## What I did NOT do, as agreed

Changed no rewards. Promoted nothing - `Models/balance_policy.onnx` still holds `model_833`
(verified by its contract's `source_checkpoint`). Deleted no lineage.

## Process check

| time | result |
|---|---|
| 02:05 pre-start | 0 / 0 clean |
| 02:14, 02:20 | 2 python, both expected |
| 04:21 Phase 1 stop | orchestrator killed left its training child alive - killed it, verified 0 / 0 |
| 04:53 Phase 3a stop | 2 killed, verified 0 / 0 |
| 08:26 final | **0 python / 0 Godot** |

## What I would do next, for you to decide

1. **Fix the action explosion before any more training.** Nothing else is worth doing while half the
   checkpoints are garbage. The barrier penalises `|a| > 1` but nothing bounds it; the honest options
   are a hard `tanh` squash on the action head, or a KL / gradient-norm limit on the PPO update.
2. **Find out what the 0.05 authority wall IS.** It is the same number for every policy, which
   suggests it is a property of the Godot side - the arena, the actuator, or the contract - and not
   of any brain. Until it moves, better Isaac policies cannot show up in Godot.
3. `model_7350` is the only new checkpoint worth keeping. It equals `model_833` in Godot while
   also surviving a 10 N·s curriculum in Isaac, so it is at worst a sidegrade with more headroom.

---

# CORRECTION — 08:45. The Godot table above was too coarse, and one headline was wrong.

`measure()` reports the push as **binary**: `held = push if stands() else 0.0`, a single shove at
the default 8 N·s. So every "0 N·s" in the report above means *"did not survive 8"*, **not** "falls
to anything". Presenting it as 0 made several policies look dead that are not.

Re-measured on a proper ladder:

| checkpoint | Isaac strict | Godot authority | push survived |
|---|---|---|---|
| zero-action baseline | 1.2 % | — | 4 N·s (per `godot_check` docstring) |
| **model_833** (starting point) | 68.0 % | 0.05 | **8**, falls at 10 |
| Stand model_4459 | 83.6 % | 0.05 | **6**, falls at 8 |
| Stand model_6664 | 82.4 % | 0.05 | **6**, falls at 8 |
| **Perturb model_7350** | 98.8 % | 0.05 | **8**, falls at 10 |
| Perturb model_9310 | 96.5 % | 0.05 | **6**, falls at 8 |

**What was wrong:** "Stand actively traded away push robustness / cost all of the push survival."
It cost **one rung**, 8 → 6, not everything. That is still the Goodhart direction - Isaac up 15
points, Godot robustness down - but it is a modest regression, not a collapse, and the report
above dramatised it.

**What survives the correction:**

* Nothing trained tonight **beats** `model_833` in Godot. `model_7350` matches it exactly (8, falls
  at 10). Everything else is one rung below. Seven hours moved Isaac 68% → ~99% and moved the Godot
  ceiling not at all.
* The **0.05 authority wall is real and unmoved** - every single checkpoint, on both tasks, at every
  push level, stands only at the bottom rung. This is the finding to chase.
* `model_9310` is a **good policy**, not a dead one: stands in Godot, survives 6 N·s, beats the
  zero-action baseline's 4. The original table misrepresented it.

**Method note for next time:** the binary push test cannot rank policies. Every comparison from here
should sweep the ladder (4 / 6 / 8 / 10) rather than take the single default, or it will keep
compressing real differences into a shared "0".

---

# THE ACTION EXPLOSION — diagnosed, fixed, verified (12:30–13:15)

## Cause: the exploration std runs away, and nothing bounds it

`rsl_rl.modules.GaussianDistribution` keeps `log_std_param` as a free `nn.Parameter` with no
ceiling. When the curriculum pushes reward negative, the entropy term is the only one left with a
clean gradient, so std inflates -> sampled actions get wilder -> reward gets worse -> it inflates
further. `entropy_coef=0.005`, `desired_kl=0.01` and `max_grad_norm=1.0` are all set and none of
them bounds std.

Across last night's 13 chained segments the correlation is total - std at the end of a segment
predicts its score with no exceptions:

| segments | std at end | eval |
|---|---|---|
| 2, 5, 9, 11 | 0.94 / 1.49 / 0.91 / **3.87** | **0.0 %** |
| 7, 12 | 0.57 / 0.37 | 64 % / 68 % |
| 1, 3, 4, 6, 8, 10, 13 | 0.25–0.32 | 96–100 % |

Peak std reached **6.25**. And `deterministic_output` returns the MEAN, so evaluation never samples
- the oversized updates had dragged the mean itself outside [-1, 1], which is exactly the
`|a| = 1.1–2.9` seen in every dead checkpoint.

## Fix

`train.py --max_action_std` (default **0.4**), clamping the parameter after every `alg.update`.
The parameter, not the sampled action: capping actions downstream leaves the runaway inside the
policy and does nothing for the drifted mean at evaluation time.

## Verified — one 15-minute run, resumed from model_7350

| metric | before (13 uncapped segments) | after |
|---|---|---|
| std over bound | peaked 6.25 | **0 of 700 iterations**, max 0.320, ended 0.200 |
| mean reward | ~22 on good segments, negative on dead ones | **29.71** |
| mean episode length | 105–470, bimodal | **467.9** of 480 |
| Isaac strict | 0 % or 96–100 %, coin-flip | **99.6 %**, head 1.465 |

## And the 0.05 authority wall moved

| checkpoint | Godot authority | push |
|---|---|---|
| everything measured before today | 0.05 | 8 N·s at best |
| **stdcap model_8036** | **0.07** | 8 N·s, falls at 10 |

Reproduced across two independent ladder runs. **This corrects the morning report's claim that the
0.05 ceiling "is not a policy property, it is a wall".** It was a policy property, and capping the
exploration std moved it. The first checkpoint on this project to stand above the bottom rung.

Curriculum ceiling ended at 8.50 N·s and was still climbing smoothly at 97-98 % survival, with no
retreat after it turned - the ×1.05 ramp behaves.

`Models/` still holds `model_7350`; nothing promoted. Processes 0 / 0.

## Known gap

The evolved impulse ceiling is not persisted with the checkpoint, so every resume restarts the ramp
at the config's 10.0 and re-climbs. Costs ~4 minutes per resume today; it matters more for a long
run to 18 N·s.

---

# THE TARGET WAS WRONG, AND SO WAS THE METRIC (15:00–16:00)

## Godot's ball is 22.5 N·s, not 18

`BallGun` transfers `(velocity - reflected) * ballMass` and `reflected` carries
`BallRestitution = 0.25`, so a head-on hit delivers `m*v*(1+e)` = 3.0 * 6.0 * 1.25 = **22.5 N·s**.
BallGun's own docs state the `(1+e)` factor; `IsaacArena` printed "3.0 kg at 6 m/s = 18 N.s" and
every target on this track inherited that error - including `push_curriculum_max = 18.0`, which
capped the training disturbance *below the shot the deployment actually throws*. Corrected to 23.0,
and the arena log now states 22.5.

## The ramp was gating on a number that overstated readiness

Magnitudes are `uniform[0, ceiling]`, so the survival rate averages over shots from zero upward.
Added a per-episode peak-impulse tracker and a second rate measured only on episodes whose hardest
shot reached 90% of the ceiling. The gap is large and consistent:

| | headline rate | at the ceiling |
|---|---|---|
| leg C's ramp, raising at 14.55 | 81–86 % | **~51 %** |
| leg D after recovery, 9.69 | 93 % | 85 % |

**Leg C's 14.55 N·s was an illusion.** The ramp raised because the average said 81–86%, while the
policy was winning only half its shots at the top of its own range. The ramp gates on the average;
it should gate on the ceiling rate.

## Leg D — the policy's honest limit

Resumed at 14.55 with 51% ceiling survival, degraded to 7.59, then recovered and climbed cleanly:

```
14.55 HOLD (51%) -> HOLD (51%) -> HOLD (41%) -> 12.37 (27%) -> 10.51 (24%) -> 8.93 (31%)
 -> HOLD (48%) -> 7.59 (29%) -> HOLD (56%) -> 7.97 (85%) -> 8.37 (87%) -> 8.79 (87%)
 -> 9.23 (86%) -> 9.69 (85%)
```

The stable operating point is **~9.7 N·s at 85% ceiling survival**, not the 14.55 leg C claimed.

## Godot — the authority wall moved a second time

| checkpoint | Isaac ceiling | Godot authority | push |
|---|---|---|---|
| model_7350 (promoted) | — | 0.05 | 8 |
| model_8036 | 8.50 | 0.07 | **8** |
| **model_10829 (leg D)** | 9.69 @ 85% | **0.10** | 6 |

0.05 -> 0.07 -> **0.10**. Every checkpoint before today failed the 0.10 rung; this one stands there,
which is the value both check scenes ship with. It trades one rung of push tolerance for it.

## Next, and clearly indicated by the data

Gate the ramp on the ceiling rate rather than the average. Leg C would then have HELD at 51% instead
of climbing to a level the policy could not actually hold, and the ceiling would track the policy
instead of running ahead of it.

## Leg E — the gate works, and Godot has stopped responding

Ramp gated on the ceiling rate (fallback to the average only under 512 band samples):

```
9.69 -> 14.32   climbing, ceiling survival 71-84%
14.32 HOLD      61%   <- old gate would have raised: the average read 79%
14.32 HOLD      51%   <- and again: average 66%
14.32 -> 12.17  35%   one retreat
12.17 HOLD      50%, 59%
12.17 -> 12.78  74%   climbing again
```

Textbook behaviour: climb while winning, hold while uncertain, retreat only while losing. It ended
at **12.78 N·s with 74% ceiling survival** - a better policy than leg C's 14.55 at 51%, at a lower
nominal number.

**But Godot has flattened:**

| checkpoint | Isaac ceiling (at-ceiling survival) | Godot authority | push |
|---|---|---|---|
| model_8036 | 8.50 | 0.07 | 8 |
| model_10829 (leg D, promoted) | 9.69 @ 85% | **0.10** | 6 |
| model_11515 (leg E) | **12.78 @ 74%** | 0.10 | 6 |

Isaac's honest ceiling rose 32% from leg D to leg E and Godot did not move at all. Three
checkpoints, Isaac 8.5 -> 12.78, Godot 0.07/8 -> 0.10/6 -> 0.10/6.

**The curriculum is no longer the bottleneck.** It is measured correctly, gated correctly, ratchets
across restarts, and climbs on earned progress. Something else is holding Godot at 0.10 / 6.

## The one known mismatch still unfixed

Isaac samples `uniform[0, ceiling]` - at 12.78 that is a mean shot of 6.4 N·s. Godot delivers a
fixed 22.5 N·s on every ball. The measurement was corrected; the *distribution* was not. Matching it
means sampling in a band near the ceiling instead of from zero, which is a deliberate trade against
the "keep rehearsing easy recoveries" argument the original `low = 0` was chosen for.

---

# 2026-09-02 — the joint-velocity channel, closed out

Four interventions on the one measured observation discrepancy (Godot 20-68 rad/s on tightly limited
axes, Isaac 7.9 peak). All negative, in both directions:

| intervention | direction | Godot ladder |
|---|---|---|
| `JointVelocityFilter` 0.50 / 0.30 / 0.15 | clean the deployment | worse: brain drops 0.10 -> 0.05 |
| noise std 1.5 across all 45 DOF | harden the policy | worse (previously measured, monotonic) |
| mask 45/45, then narrow 12/45 | delete the disagreement | fails EVERY rung |
| chatter 20 rad/s on the narrow 12 in Isaac | match Godot | fails EVERY rung |

Isaac-side scores for the record: full mask plateaus 77-81%, narrow mask 83-84%, chatter 69.5% at
1470 iterations. The narrow mask scored 82.6% and failed rungs a 68% unmasked policy passes, so the
failures are not merely weakness.

**Conclusion: the channel discrepancy is real and is NOT the transfer blocker.** Cleaning it hurts,
matching it does nothing. Stop working on it.

## Also eliminated today

* **Stance mismatch.** The driver docstring says Godot's standing joints sit "0.82 rad from the rest
  pose", implying the policy cannot command Godot's stance. Dumped and mapped it: the 0.87 rad is
  `joint_Hand_R:0`, a WRIST. Every leg joint - Thigh, Shin, Foot - is within 0.1 rad. The two bodies
  agree on posture everywhere locomotion happens.
* **Action magnitude.** The working balance brain outputs maxAction 1.55-2.12 in Godot; the failing
  walk brain outputs 1.10-3.64 at the rung where it falls. Comparable - the walk policy is not
  commanding wilder actions.

## Machinery added (keep)

`obs_joint_vel_enabled`, `obs_joint_vel_min_range`, `obs_joint_vel_mask_narrow`,
`obs_joint_vel_narrow_noise` - all four in TRAINED_CONDITIONS, all mirrored in Godot through
scene -> arena -> driver -> IsaacObservation. Two of them caught restore traps before they produced
false numbers.

**One self-inflicted bug worth remembering:** `obs_joint_vel_min_range` originally both SELECTED the
narrow joints and MASKED them, so the first chatter run injected 20 rad/s and then multiplied it by
zero. It looked like a valid experiment and would have retired the last hypothesis on a run that
never happened. Caught by reading the log line, not the score - which is why that line now states
selection, treatment AND dose.

## What is left

The contact solver, untested. It is also the only difference that explains the standing/walking
asymmetry: standing is sustained double-support contact, walking is repeated make-and-break.

## The contact-flag finding (2026-09-02, evening)

Measured the foot, in both engines, under the same policy:

| | resting foot height | behaviour |
|---|---|---|
| Godot | **0.0398 m** (range 3 mm) | planted, no bounce |
| Isaac | spawns 0.0442, **sinks to 0.014** in 5 steps | ~3 cm of contact penetration |

Raising solver iterations only halves it and is non-monotonic - 0.0141 at 8, 0.0285 at 16, 0.0204
at 24, 0.0229 at 48 - so the penetration is an XPBD property, not a setting.

**The damage is the contact-FLAG timing, not the penetration.** `CONTACT_HEIGHT = 0.06` over a foot
resting at 0.040 gives Godot 2.0 cm of clearance before the flag drops; over a foot at 0.014 it
gives Isaac 4.6 cm. The policy learns "my foot is off the ground" at more than twice the true
height. A planted foot never approaches either threshold, which is exactly why standing transfers
and walking does not.

Fixed: `CONTACT_HEIGHT` 0.06 -> **0.034** (Isaac's real resting height plus Godot's 2 cm margin).

**Result: a large Isaac improvement, no transfer improvement.**

| Walk, at a 0.3 m/s command | before | after |
|---|---|---|
| steps per 12 s | 17.3 | **67.8** |
| distance | 5.9 m | **9.57 m** |
| upright | 90-92 % | **98.2 %** |
| training falls | 15-19 % | **9-11 %** |
| Godot ladder | stands only at 0.03 | stands only at 0.03 |

Four times the step count: the old policy took long sliding strides because its flag said "airborne"
while the foot was 2 cm up. The gait is now genuinely a gait. It still does not transfer.

Keep the change regardless - the simulator is more faithful and the gait is better - but it is not
the transfer blocker either.

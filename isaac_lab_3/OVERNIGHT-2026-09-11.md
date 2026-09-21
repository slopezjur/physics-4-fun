# Overnight 2026-09-11 — Perturb, then Walk

Mandate, 00:20: four hours of 15-minute Perturb sessions, then four hours of Walk; after every
session, review the result and change whatever improves the dummy. Every session is one
`overnight.py --sessions 1` call: train on the GPU, score on CPU MuJoCo, promote automatically if the
checkpoint beats what the scenes load (perturb: paired McNemar z >= 2 on the same 512 shots). The
session stopped on a usage limit from 01:55 to 04:52 and nothing ran in between, so Perturb got six
sessions and Walk eleven.

## Result

| | start of the night | end | ships as |
|---|---|---|---|
| **Perturb** - survival of ONE 30 N·s hit, 512 envs | 68.75% | **87.50%** | `balance_policy.onnx` = `p0911e_s1/model_256` |
| **Walk** - % of the joystick, five manoeuvres | 18.7% (straight only) | **46.3%** | `locomotion_policy.onnx` = `w0911f_s1/model_253` |

Perturb improved in three measured jumps, each from a change aimed at a measured failure. Walk
learned to turn in arcs and to stop when told to; it cannot turn on the spot, and measured honestly
its gait is a shuffle.

## Perturb

| # | change | one hit at 6 m/s | promotion |
|---|---|---|---|
| 1 | none - control continuation of the shipped `perturb_z_s4/model_253` | 71.5% | z +1.51, not shipped |
| 2 | **shots weighted toward the bones it fails on** | 74.2% | z +2.60, **shipped** |
| 3 | **`CAPTURE_V` 0.5 -> 1.0** | 80.9% | z +3.08, **shipped** |
| 4 | none - continuation | 80.3% | z -0.28 |
| 5 | **correct COM velocity + a capture-point placement term** | 87.5% | z +3.18, **shipped** |
| 6 | none - continuation | 85.7% | z -0.90 |

Each change gave one clear jump and a continuation after it did not - the next gain needs another
change, not more minutes.

**Why each change** - `probe_hits.py` and `probe_steps.py` (scratchpad) replay the gate's own 512
shots and break survival down:

* **Where it fell.** On the shipped brain, head hits survived 10.5%, chest 26%, spine 38%, pelvis and
  legs 86-100%: high hits were a quarter of the shots and over half of the falls. They do not push
  the COM harder (0.48-0.61 m/s vs 0.55 for a pelvis hit); they rotate the body. Training now aims
  more shots at the failing bones (`train.py --target_weights`, weights `(1 - survival) + 0.2`).
  Head+chest+spine went 25% -> 46% of the training shots; the scorer still shoots uniformly, on the
  exact random stream it always used.
* **How it fell.** The body stepped - 73-94% of fallers - but the first step was 6-12 cm long and
  landed 10-22 cm SHORT of the capture point. The step reward stopped growing at 0.5 m/s of swing,
  which a short brisk swing already reached; reaching the capture point takes about 1 m/s.
* **Where the foot lands.** Even at `CAPTURE_V` 1.0 the steps landed ~0.2 m short, so the reward now
  also pays the swing foot's closeness to the capture point (`com + v / omega0`) while off balance.
  That needed a correct COM velocity - see below.

| bone hit | shipped at dusk | final (`p0911e_s1/model_256`) |
|---|---|---|
| Head | 10.5% | 63.2% |
| Chest | 26.1% | 65.2% |
| Spine | 38.1% | 83.3% |
| Pelvis | 91.2% | 94.1% |
| Upper arms L / R | 57% / 46% | 88% / 73% |

Every push direction is ~90% except pushed-right (79%); with the right upper arm lagging too, that
is a learned handedness - the body is symmetric. The first step is still no longer and still lands
~0.17 m short: the last jump came from something the step probe does not see, most likely follow-up
steps or the trunk.

## Walk

Walk started from `walk_g_s1/model_254`, which walks straight at 0.25 m/s and cannot turn or stop.

| # | change | joystick | promotion |
|---|---|---|---|
| 1 | **stage 4**: the amble (0.15-0.40 m/s) plus turning, arcs, turning on the spot, stop | 32.4% | **shipped** (32.1 vs 17.6) |
| 2 | none - continuation | 36.9% | shipped on a clipped score; re-decided below |
| 3 | stage 5: twice the stand and spin commands | - | aborted as a statue at 7 min |
| 4 | **`STAND_PLANTED`**: pay both feet down while the command is a stand | 46.2% | **shipped** (45.4 vs 32.1) |
| 5 | none - continuation | 40.8% | not shipped |
| 6 | **foot lift measured from rest**, as in perturb (6 cm) | 45.6% | **shipped** (46.3 vs 45.4) |
| 7 | none - continuation | - | aborted as a statue at 10 min |
| 8 | walk clearance 3 cm above rest | 29.1% | not shipped - stopped walking |
| 9 | walk clearance 2 cm (first rung of a rising clearance) | 33.1% | not shipped |
| 10 | walk clearance 2.5 cm (second rung) | - | aborted as a statue at 7 min |
| 11 | walk clearance 2 cm, continuation of session 9 | - | aborted as a statue at 8 min |

What the shipped brain does, 40 s per manoeuvre:

| manoeuvre | at dusk (`walk_g`) | final (`w0911f`) |
|---|---|---|
| straight line, 0.25 m/s | +8.1 m | +6.0 m |
| turn left, 0.4 m/s + 0.6 rad/s | -7 deg (wrong way) | +367 deg (falls in some runs) |
| turn right | -158 deg | -807 deg |
| turn on the spot, 0.8 rad/s | +26 deg | falls |
| stand still | drifts 3.7 m | drifts 0.05 m |

* **The joystick score.** Walk used to be scored on the straight line alone, which cannot see a turn.
  It is now scored on all five of `eval_walk`'s manoeuvres, each 0-1 for doing what it was told times
  its uprightness. The first version clipped each part at 1.0, so walking 43% too fast read as a
  perfect straight line and session 2 shipped on it; each part is now `1 - |achieved / commanded - 1|`,
  and the gate, re-run, shipped session 1 back.
* **Standing still.** Every walk brain walked away from a stand command, 4-10 m in 40 s, although
  the tracking term charged it ~2.5 per step and the reward saw the drift exactly (`check_stand_drift`:
  0.114 m/s read, 0.114 real). Nothing paid for both feet down; `STAND_PLANTED` does, and the drift
  went to 0.05-0.15 m.
* **The gait is a shuffle.** The walk env counted a foot as lifted above an ABSOLUTE 6 cm while the
  foot origin rests at 4.4 cm - a 1.6 cm shuffle read as a step, and every gait term paid it. Measured
  from rest, the shipped brain takes 0 steps on the straight line; its higher foot rises a median
  2.4 cm and never 6. Every clearance that stopped paying the shuffle - 6 cm, 3 cm, 2.5 cm, and 2 cm
  once training continued - turned the walk into a statue within a session instead of making it lift
  higher. The shuffle's payment is what keeps walking ahead of standing: preflight's +3.6 per step
  for walking is measured on a frictionless glide, and a real gait also pays effort, jerk, rocking and
  the risk of a fall. `WALK_FOOT_CLEAR` is left at 1.6 cm above rest - exactly what the walk env
  always used - so walk trains as it did; the relative formulation stays.

## Measurement defects found and fixed

* **MuJoCo `cvel` again.** The perturb reward's COM velocity summed raw `cvel`, which is referenced to
  the subtree COM: off by a median 0.32 m/s whenever the body moved faster than 0.2 m/s - as large as
  the signal. `com_velocity` (public, both backends) shifts it to each body's own COM and matches a
  finite difference to 0.017 m/s. Preflight's "the ball delivers 137% of what it carries" was the
  same error: it delivers 93%.
* **Walk's foot clearance** was absolute - above.
* **The joystick score let overshoot through** - above.

## Still open

1. **Look at the scenes.** Both brains changed tonight, and the Godot side has not been seen since
   the heading hold was added.
2. **Walk promotion has no significance test.** Session 6 shipped on +0.9 points while falling more
   than session 4 on the arcs; perturb's paired gate would not have shipped it.
3. **Walk steps.** Give walking a stronger reason than a shuffle's payment before raising the walk
   clearance - every raise tried stopped the walk. Turning on the spot needs real steps and falls
   without them.
4. **The walk reward and the observation read the pelvis's linear velocity from raw `cvel`.** Not
   changed tonight: it is part of the contract Godot builds, so it has to change on both sides at once.
5. **Perturb next:** probe every step, not the first; the pushed-right handedness; a new change
   rather than more minutes.

The session-by-session ledger, with every probe, is `night/LEDGER-2026-09-11.md` in the session
scratchpad.

## The morning after

An hour of Perturb, then a 30-minute session with a new reward term, asked for after the night.

| # | change | one hit at 6 m/s | promotion |
|---|---|---|---|
| m1 | train on harder hits, 6.6 m/s (33 N·s) | 83.0% | z -2.27 - a regression |
| m2 | shot weights refreshed from the shipped brain's own failures | 83.4% | z -2.20 - a regression (stopped at 14 min) |
| m3 | **rest stance**, 30 minutes | **92.4%** | z +2.93, **shipped** |

**The rest stance.** After hits the dummy stood oddly, and the stance term could not see it: it
measured `|y_left - y_right|` in world axes, so `abs()` read crossed feet as a normal width, a body
that had turned was measured along the wrong axis, and a fore-aft split was ignored; the `pose`
term averages all 30 joints, so crossed legs cost it about 2%. The stance is now the feet in the
pelvis's own heading frame, signed, times the leg joints against the rest pose, paid only while
balanced (a protective step stays free), weight 2.0. The first version assumed which side the left
foot lies on, read the rest stance as crossed, failed its check and reverted itself; the sign is now
read off the rest pose. `eval.py` reports the end stance of survivors.

| | before (`p0911e_s1/model_256`) | after (`p0911i_s1/model_503`) |
|---|---|---|
| one hit: survived | 87.5% | **92.4%** |
| end stance (1 = the rest stance) | 0.29 | **0.37** |
| foot width, rest 0.31 m | 0.34 m | 0.31 m |
| split / leg error | 0.06 m / 0.27 rad | 0.04 m / 0.24 rad |
| a hit every 3 s for 20 s: survived / end stance | 41.4% / 0.24 | **53.1% / 0.29** |

Survivors did not END with crossed feet even before (0%, and 0-1% under repeated hits): what stayed
off was the legs, bent ~0.27 rad from rest, and a stagger. Crossing mid-recovery is not seen by an
end-of-episode snapshot. `balance_policy.onnx` is now `p0911i_s1/model_503`.

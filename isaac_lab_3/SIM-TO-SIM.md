# Sim-to-sim: matching Godot and Isaac

## 2026-09-06 17:43 — base randomisation keeps the Isaac gait; and a METHOD flaw that invalidates
## the last several fragment experiments

### The experiment

`reset_pitch_noise = 0.035` (2 degrees of tilt about a random horizontal axis at reset) and
`reset_ang_vel_noise = 0.25` rad/s, 15 minutes from `gl151`. Both verified applied and both added to
`TRAINED_CONDITIONS`.

**Isaac survives it** - cmd 0.30: WALK, 4.43 m, 44 strikes, 100% alternation, 100% upright; cmd 0.50:
WALK, 6.97 m, 51 strikes. That matters: this is the FIRST randomisation tried here that does not
destroy the gait. Action latency (twice) and wider action/effort ranges all produced a flamingo.

**Godot transfer is worse** - statue with 0 strikes at every explicit-path cell, falls on the spring
path. Against `gl151`'s 12 strikes at 100% upright.

### The flaw: a single fragment cannot evaluate a training change for transfer

Every 15-minute fragment started from a stepping checkpoint today:

    hz960b (more of the same)             lost stepping
    shortstride cmd 0.10-0.25             lost stepping, FLAMINGO in Isaac
    action_latency (0,2)                  lost stepping, FLAMINGO in Isaac
    action_latency (2,2) fixed            lost stepping, FLAMINGO in Isaac
    target_damping 1.0                    lost stepping, FALLEN everywhere
    wider action/effort randomisation     lost stepping
    base-dynamics randomisation           lost stepping, Isaac gait survived

**Seven fragments, zero retained Godot stepping.** And the search that produced `gl151` was 20
five-minute fragments yielding exactly one stepper.

So the property being measured survives fine-tuning roughly **one time in twenty regardless of what
is changed**. A single fragment coming back without stepping is therefore the EXPECTED outcome under
the null hypothesis, and carries almost no information about whether the change helped or hurt.

**Every "tested and failed" verdict on a training change today rests on n = 1 against a 1-in-20 base
rate.** They are not falsifications; they are underpowered. The eliminations of *deployment-side*
knobs - damping, solver steps, pelvis gains, physics rate, spring calibration - are unaffected,
because those are swept across many cells on a fixed checkpoint.

**What a valid test would need:** the same change run over roughly 10-20 seeds and scored on the
FRACTION that step in Godot, against the ~1-in-20 baseline. At 15 minutes a fragment that is 3-5
hours per training variant, which is why it has not been done - but it is the only way any of these
training-side conclusions become real.

`gl151` restored as the promoted checkpoint.


## 2026-09-06 17:21 — Featherstone is UNUSABLE, MJWarp is viable, and the scoring trap is real

### Throughput settles the solver question before transfer does

| solver | iteration time | vs XPBD | divergence from Godot |
|---|---|---|---|
| XPBD | ~0.5 s | 1x | 0.0341 /s |
| **MJWarp** | **2.5 s** | 5x | 0.0277 /s |
| **Featherstone** | **174.7 s** | **350x** | **0.0184 /s** (closest) |

**The solver closest to Godot is the one that cannot be trained on.** Featherstone managed about nine
iterations in twenty-six minutes and reported a two-hour ETA for what XPBD does in fifteen; it never
even reached the `--max_minutes` check, which only fires on iteration boundaries. Killed.

MJWarp at 2.5 s is usable but not cheap: a 15-minute fragment buys ~350 iterations against XPBD's
~1700, so evaluating it properly is a multi-hour retrain rather than a fragment. 150 iterations left
the policy FALLEN on its own plant.

### The scoring trap, demonstrated rather than warned about

The same MJWarp checkpoint, scored two ways:

    under MJWarp (correct)   FALLEN   83.5% upright   33.1% flight   16 strikes
    under XPBD   (the trap)  WALK    100.0% upright    5.8% flight   51 strikes

**A confident WALK for a policy that falls on its own training plant.** The solver is selected at
import time from an environment variable and is not replayed by `restore()`, so this is one forgotten
`P4F_SOLVER=` away at all times. The guard added an hour earlier - printing the backend and stating
that it is not restored - is what caught it here.

### Switching approach: stop matching plants, remove the dependency

Plant matching has now hit a practical wall from both directions: the closest solver is
computationally infeasible, and every parameter that CAN be matched already is. So the remaining move
is the one the external review said to run regardless, and which had been deferred in favour of more
interesting work: **make the policy stop depending on solver-specific base dynamics.**

New `reset_pitch_noise` (base tilt about a random horizontal axis at reset) and `reset_ang_vel_noise`
(base angular velocity at reset), both added to `TRAINED_CONDITIONS` so they travel with a checkpoint.
Deliberately small - 0.035 rad (2 degrees) and 0.25 rad/s in the first run. The goal is not to walk
while tilted; it is to stop the gait being keyed to one exact floating-base trajectory.


## 2026-09-06 16:34 — THE SOLVER BACKEND WAS INHERITED, AND IT IS THE WRONG ONE

Isaac Lab 3 exposes four Newton solver backends. Every checkpoint this project has ever trained used
XPBD, because it was the default in the config that was copied at the start. `P4F_SOLVER` now selects
between them (`xpbd` / `featherstone` / `mjwarp` / `kamino`); all four cfg classes import and both
alternatives tested here run the rig without modification.

Open-loop replay, the SAME recorded target trajectory into every engine, Godot on the angular-spring
path. Divergence measured as the growth rate of the torso-pitch error against Godot - the metric the
external review proposed, on the grounds that base dynamics rather than stride is the thing the
solver formulation actually determines:

| Isaac solver | d\|pitch err\|/dt vs Godot | mean \|pitch err\| | its own gait |
|---|---|---|---|
| **XPBD** (what we train on) | **0.0341 /s** | 0.2855 | WALK, 38 strikes, 100% upright |
| **Featherstone** | **0.0184 /s** | 0.2101 | 26 strikes, 81.4% upright |
| **MJWarp** | 0.0277 /s | 0.1989 | 33 strikes, 88.8% upright |

**Featherstone diverges from Godot 46% more slowly than XPBD does, and MJWarp 19% more slowly.** Both
also hold a smaller mean pitch error. The policy walks worse under them - but it was trained on XPBD,
so that is expected and is not the criterion: what matters is which plant Godot can reproduce.

These are genuinely different plants, not cosmetic variants. Under identical targets, XPBD and
Featherstone differ by **0.19 rad at the hip** and 0.14 in torso pitch.

### The trap this creates, and the guard

The solver is chosen at import time from an environment variable, so it is **not** restored by
`run_conditions.restore()`, which replays cfg fields from `params/env.yaml`. A checkpoint trained
under Featherstone and scored without `P4F_SOLVER` set is silently scored under XPBD - the same class
of silent plant invalidation already suffered with `sim.dt` and the XPBD iteration count. Until the
selection moves into the cfg proper, `_solver_cfg()` prints the backend on every run and says that it
is not restored.

**Switching backend invalidates every existing checkpoint.** Treat it as a full retrain.


## 2026-09-06 16:07 — Godot does NOT need more attitude authority. More makes it worse.

The open design question after the base-attitude finding was whether Godot needs a deliberately
stronger pelvis controller than Isaac, since its base is resolved differently. `BalanceAssist` could
not answer it (clamped to [0,1], already saturated), so `PelvisGainScale` was added, multiplying the
module's gain, damping and torque ceiling together. Verified applied on every run.

| drive | pelvis gain scale | authority | upright | strikes | travel |
|---|---|---|---|---|---|
| explicit | **1.0** | 0.125 | **100%** | **12** | 0.153 |
| explicit | 2.0 | 0.125 | 11.2% | 5 | 1.286 |
| explicit | 4.0 | 0.125 | 16.8% | 5 | 1.287 |
| explicit | 8.0 | 0.150 | 3.1% | 6 | 0.351 |
| spring | 2.0 | 0.125 | 10.2% | 4 | 0.761 |
| spring | 4.0 | 0.150 | 7.5% | 3 | 1.043 |
| spring | 8.0 | 0.150 | 8.3% | 4 | 0.886 |

**Every increase falls.** Ten cells across both drive paths, 2x to 8x. The authored gains - which are
already Isaac's, verbatim - are also already the useful maximum, and the attitude controller is not
short of authority.

That closes the "Godot needs a stronger attitude controller" hypothesis. Combined with the earlier
result that removing it entirely also falls (100% -> 28% upright), the pelvis stabiliser has a narrow
band around the authored values in which it works at all, and it is already sitting in it.

### Where the sim-to-sim work stands

Best END-TO-END: explicit path, `StanceDampingScale = 2.0`, authority 0.125, command 0.30 - **12
alternating strikes, 1.4% flight, 100% upright, sustained over 40 s, travel 0.153 m.** It marches in
place.

Best OPEN-LOOP: the angular spring, which produces Isaac-like stride (0.16-0.29 m against 0.2529)
but cannot hold the body up beyond about eight seconds at any calibration tried, including the
transfer-function match.

Eliminated today, each with a measurement: solver steps, pelvis gains in both directions, gravity
feed-forward on the spring path, spring gain calibration, phase-lead compensation, delay-aware
training, and a stronger attitude controller.


## 2026-09-06 15:41 — the spring CAN be transfer-function matched, and it still does not transfer

Calibrating the angular spring against the measured transfer function rather than against open-loop
walking - which is what the earlier k x2 d x5 tuning optimised, and the wrong objective.

    spring gains        phase difference from Isaac at 1 / 2 / 3 Hz
    k x2  d x5          3.7  15.5  19.3 deg
    k x1  d x2          0.7   2.0   7.0 deg     <- transfer-function match
    k x0.5 d x5        29.8  36.5  38.1 deg
    k x1  d x5         14.9  22.0  25.2 deg

**Damping is what drives the lag.** `d x5` is five times Isaac's damping and adds most of the phase
error; dropping to `d x2` brings Godot's phase within 0.7-7 degrees of Isaac's across 1-3 Hz, which is
as close as this project has got to a matched actuator. Gain ratio at that setting is 1.18-1.48 with
Godot high, and Isaac's own gain reading varies 0.23-0.35 run to run, so gain is the noisier half of
the comparison and phase is the reliable one.

### And it changes nothing about transfer

| | closed loop | open loop (Isaac's own targets) |
|---|---|---|
| k x1 d x2, authority 0.10-0.20 | FALLEN, 7-14% upright | FALLEN, 9-16% upright |
| k x2 d x5 (the old tuning) | FALLEN, 8.1% | FALLEN, 10.6% |

Nine cells each. **Every spring configuration falls over a 20 second run**, matched or not. The
earlier "100% upright, 14 strikes" spring result was an EIGHT second run; extended to 20 s it does not
hold, which is the same 8-to-12-second decay already noted and now confirmed at the calibrated
setting.

Note the stride column reaches 0.25-0.29 m in several open-loop rows - Isaac's own value - but every
one of those rows is at 9-16% uprightness, so they are falling bodies and the numbers are void under
the rule established earlier today.

### What this means

Matching the actuator transfer function is **necessary but not sufficient**. With phase within a few
degrees and gain within 1.2-1.5x, the body still parts company inside twenty seconds. That is
consistent with the earlier finding that the residual divergence is in BASE ATTITUDE rather than in
the joints: an actuator match cannot fix a floating-base discrepancy, because the joints were already
tracking to 0.05 rad before any of this calibration.

The explicit path with stance damping remains the best END-TO-END configuration - 12 alternating
strikes at 100% uprightness, sustained over 40 s, with no travel - and the spring remains the best
OPEN-LOOP one. Neither is a walk.


## 2026-09-06 15:12 — the lead cell does not survive scrutiny, and delay-aware training fails

**Delay-aware training.** `action_latency_steps = (2, 2)` - a FIXED two-step latency matching what the
frequency response then implied - asks the policy to learn the compensation rather than having it
hand-tuned in the driver. Verified applied (`override action_latency_steps (0, 0) -> (2.0, 2.0)`).
Result: **FLAMINGO in Isaac**, 0 strikes at cmd 0.30, 94.4% single support. The randomised (0,2)
version failed the same way last night. Fifteen minutes of finetuning cannot absorb two steps of
latency, and the motivating number has since been halved anyway. Discarded.

**The lead cell, re-tested properly:**

| condition | upright | strikes | flight | travel |
|---|---|---|---|---|
| baseline, 20 s | **100%** | **12** | 1.4% | 0.153 m |
| lead 2.0 / smoothing 0.35, 20 s | 97.5% | 8 | 3.1% | **0.682 m** |
| lead 2.0 / smoothing 0.35, **40 s** | **48.8%** | 8 | 51.5% | 0.990 m |
| lead 2.0 + spawn noise 0.05 | 78.0% | 1 | 26.6% | 1.376 m |
| lead 2.0 + 20 N.s push | 35.6% | 2 | 63.0% | 0.602 m |

**It survives 20 s and collapses by 40 s**, and it is less robust to both perturbations than the
baseline. So it is not an improvement - it trades stability for travel. The baseline remains the best
end-to-end configuration: 12 alternating strikes at 100% uprightness, sustained bit-identically over
40 s, going nowhere.

### Durable fix: the trace offset is now handled in one place

`scripts/trace_align.py` (new) provides `load_pair` and `verify_alignment`, and `probe_divergence.py`
now uses them and warns loudly if the residual action difference exceeds 1e-3 - which would mean the
lag changed or the traces are not from the same actions. Aligned residual on a test pair is
**0.000000**.

That offset has now distorted three separate comparisons and forced two retractions. It should not be
left to each script to remember.


## 2026-09-06 15:08 — CORRECTION: half the measured phase lag was my own trace offset

The 14:05 and 14:45 entries report a phase lag of 2.0-2.6x and "a pure transport delay of 32.1 ms /
1.93 policy steps". **Both numbers are wrong.** `IsaacPolicyDriver.DofTracePath` logs one policy step
late - already documented here from the action cross-correlation, `godot[n+1] == isaac[n]` exactly -
and the frequency-response analysis did not remove it, so 16.7 ms of logging offset was charged to
the plant.

### Caught by a step test that contradicted itself

A step input measured "1 policy step of dead time in Godot, 0 in Isaac". Re-measuring with each
trace's own action-column offset established first (Godot shift 1, Isaac shift 0, both matching at
mean |diff| exactly 0.000000):

    TRUE dead time   ISAAC 0 steps   GODOT 0 steps

**There is no transport delay at all.** The apparent one was the logging offset, and the same offset
was inflating the frequency response.

### The corrected transfer function

| f (Hz) | Isaac phase | Godot phase | difference | implied delay |
|---|---|---|---|---|
| 0.5 | -6.0 | -9.0 | 3.0 deg | 16.7 ms |
| 1.0 | -10.8 | -15.1 | 4.3 | 11.9 ms |
| 2.0 | -20.2 | -35.7 | 15.5 | 21.5 ms |
| 3.0 | -26.2 | -46.0 | 19.8 | 18.3 ms |
| 4.0 | -31.6 | -50.5 | 18.9 | 13.1 ms |
| 5.0 | -32.3 | -50.4 | 18.1 | 10.1 ms |

**Phase ratio 1.5-1.8x, not 2.0-2.6x. Residual lag about 15 ms, not 32 ms.** And the implied delay is
NOT constant - it scatters 10-21 ms and falls at the top of the range - so it is **actuator dynamics,
a first-order-ish lag, not a transport delay.** Gain ratio is 1.19-1.74 with Godot higher.

### What survives and what does not

**Survives:** Godot does lag Isaac, the gap is real, and it is in phase rather than gain. The
qualitative story - open loop is indifferent to phase and works, the 60 Hz feedback loop is not and
fails, damping adds lag so every damping fix spent the margin it bought - is unchanged.

**Does not survive:** the size (halved), the character (dynamics, not dead time), and the rationale
for the phase-lead compensator, which was tuned against a number that was half artifact. The lead's
empirical result stands on its own - `TargetLeadSteps 2.0` with `LeadSmoothing 0.35` still gave 97.5%
uprightness with 4.5x the travel of the baseline - but it was compensating partly for a measurement
error, and the right lead for a 15 ms first-order lag is not the right lead for a 32 ms dead time.

**Method note.** This is the third time today a Godot-vs-Isaac comparison has been distorted by that
one-step trace offset, and the second time it produced a headline result that had to be retracted.
The offset is now removed inside `freqresp.py`; it should be removed at the source instead, either by
logging after the solve or by shipping the shift in a shared loader that every comparison script uses.


## 2026-09-06 14:45 — the lag is a PURE TRANSPORT DELAY of 1.93 policy steps, and a lead compensator only half-works

### It is a delay, not a filter

Converting the measured phase difference to time. If the extra lag were a transport delay,
`phase_diff / (360 f)` is constant; if it were a low-pass filter it would collapse as frequency rises.

    f (Hz)   0.5     1.0     2.0     3.0     4.0     5.0
    delay   33.3ms  27.8ms  37.9ms  34.8ms  31.8ms  27.0ms

**Mean 32.1 ms, stdev 3.8 ms, over a tenfold frequency range = 1.93 policy steps at 60 Hz.** Godot
carries very close to two policy steps of transport delay that Isaac does not.

### A phase lead fixes the open-loop phase

`TargetLeadSteps` extrapolates the action forward by N policy steps before decoding, so every drive
path inherits it. Open loop:

    f (Hz)      lead 0     lead 2     Isaac
    0.5        -12.2 deg   -7.1 deg   -6.2 deg
    1.0        -21.0       -12.9      -10.9
    3.0        -64.0       -38.0      -26.9
    5.0        -80.4       -47.2      -32.7

Residual delay falls from 32 ms to about 6-8 ms. **And it inflates the gain ratio to 1.35-2.71x**,
which is textbook lead-compensator behaviour.

### Closed loop: raw lead is WORSE, smoothed lead is partial

Raw extrapolation of a noisy action sequence hands that noise to the joints: lead 1 takes the
baseline from 100% upright to 55.1%, lead 2 to 3.5%. Adding an EMA on the derivative
(`LeadSmoothing`) fixes that, and the best cell is genuinely interesting:

| config | upright | strikes | stride | **travel** |
|---|---|---|---|---|
| baseline, no lead | **100%** | **12** | 0.0386 | 0.153 m |
| lead 2.0, smoothing 0.35 | 97.5% | 8 | 0.0172 | **0.682 m** |

**4.5x the travel at essentially unchanged uprightness** - the first time forward progress and
staying upright have moved in the same direction. It is still an isolated island: 1.5 and 2.5 lead,
0.30 and 0.45 smoothing, and authority 0.115 or 0.135 are all worse, and no cell clears the WALK bar
(>=12 strikes with travel).

**Implementation note worth keeping.** The first version wired the lead into the angular-spring path
only, and the explicit path came back byte-identical across lead 0/1/2/3 - which reads as "no effect"
rather than "not applied". Moving it onto the ACTION vector before `Decode` fixed that and made every
drive path inherit it. Same class of error as the four inert `--set` falsifications on record: verify
a knob CHANGED something before believing it did nothing.

### Next: let the policy learn the delay instead of hand-compensating it

`action_latency_steps` exists from last night and was tested randomised (0,2) before the delay was
measured; it broke the Isaac gait. Now that the delay is known to be 1.93 steps, training at a FIXED
2-step latency asks the policy to learn exactly the compensation Godot needs, which is a better place
for it than a hand-tuned lead in the driver.


## 2026-09-06 14:05 — THE PLANT DIFFERENCE IS PHASE LAG, 2-2.6x, and it explains almost everything

Two measurements, both suggested by the external review, both decisive.

### The divergence is NOT at contact

Per-step growth of the pitch error, split by whether the step is near a foot-contact transition:

    window +/-1 step    near 0.01824   away 0.01615   ratio 1.13x
    window +/-2 steps   near 0.01786   away 0.01600   ratio 1.12x

    29.9% of total pitch-error growth falls in windows covering 27.6% of samples

**The error accumulates continuously, in proportion to time, not at touchdown.** By the review's own
framing that rules out a contact-impulse mismatch and points at an articulated inertial/reaction
difference. Contact mechanics is exonerated.

### The actuator transfer function: gain matches, PHASE does not

Sinusoidal target on `joint_Thigh_L:0`, amplitude 0.158 rad, same actions into both engines, gain and
phase by single-bin DFT:

| f (Hz) | Isaac gain | Isaac phase | Godot gain | Godot phase | phase ratio |
|---|---|---|---|---|---|
| 0.5 | 0.294 | -6.2 deg | 0.391 | -12.2 deg | 2.0x |
| 1.0 | 0.273 | -11.0 | 0.367 | -21.0 | 1.9x |
| 2.0 | 0.329 | -20.4 | 0.414 | -47.7 | 2.3x |
| 3.0 | 0.318 | -26.4 | 0.338 | -64.0 | 2.4x |
| 4.0 | 0.289 | -28.7 | 0.268 | -74.5 | 2.6x |
| 5.0 | 0.229 | -31.8 | 0.227 | -80.4 | 2.5x |

**Gain agrees within 0.93-1.34x. Phase lag is 2-2.6x Isaac's and grows with frequency.**

This one number retro-explains most of the week:

- **Gain calibration was flat** because gain was never the mismatch.
- **Open loop works and closed loop fails** because open loop does not care about phase and a
  feedback controller does. The policy is a feedback controller running at 60 Hz.
- **The pitch divergence accumulates continuously** rather than at events, which is what a lag does
  and what the contact analysis above independently found.
- **Every damping increase destabilised.** Damping ADDS phase lag. Each "fix" bought local stability
  and spent phase margin, which is why the stride-versus-stability trade-off never had a good corner.
- At 3 Hz - the vertical pogo frequency measured earlier - Godot lags 64 degrees against Isaac's 26.

Note also that BOTH engines deliver only 0.23-0.41 of the commanded amplitude at these frequencies, so
the policy is operating against a plant that under-delivers in both - but only Godot adds the lag.

### What to attack

Lag sources on the Godot side, in the order they are worth testing:
1. The target is written after the solve and takes effect on the following tick; Isaac's XPBD applies
   it inside the solve. This is at least one physics tick, and `DofTracePath` already shows a
   one-policy-step lag in the trace.
2. The angular spring's own second-order response, which the stiffness/damping pair sets.
3. Any filtering left in the observation path feeding the policy.

A phase-lead compensator - commanding the target extrapolated forward by the measured lag - is the
standard remedy and is cheap to test, because the lag is now measured per frequency rather than
guessed.


## 2026-09-06 13:55 — the flexion offset is a SYMPTOM, not a cause. The rigs agree at rest.

The previous entry proposed that the ~0.065 rad flexion offset is a constant kinematic discrepancy in
where the two engines place the joint zero. Tested directly: zero actions into both engines, no gait,
no policy, settled over the last second.

| joint | GODOT | ISAAC | offset |
|---|---|---|---|
| Thigh_L.x | 0.0055 | 0.0009 | +0.0046 |
| Shin_R.x | -0.0007 | -0.0011 | +0.0004 |
| Foot_L.x | 0.0183 | -0.0040 | +0.0223 |
| Spine.x | 0.0021 | 0.0007 | +0.0014 |

    mean |offset| at rest   0.0072 rad     (under gait: 0.065 rad)
    pelvis height           0.8196 / 0.8197  -> 0.1 mm apart
    torso pitch             +0.021 / +0.004

**The rigs agree at rest to 0.4 degrees and 0.1 mm.** There is no constant kinematic offset to find.
The 0.065 rad appears only under gait, so it is an OUTPUT of the divergent motion rather than an input
to it - the previous entry had cause and effect the wrong way round.

Combined with the stiffness sweep (the offset is flat across a sixteen-fold stiffness range, so it is
not compliance either), what is left is that under motion the two bodies simply go different places,
and the joint-angle difference is one of the several ways that shows up. There is no separable
constant to remove.

**This closes the "find the offset" line.** The remaining honest statement of the gap on the
constraint-drive path is: at rest the plants are identical; under Isaac's own recorded targets the
joints track to 0.05 rad and the stride reaches 0.16 m against Isaac's 0.25 m; and the bodies
nonetheless diverge in attitude over a few seconds. That divergence is dynamic and coupled, not a
static discrepancy waiting to be cancelled.


## 2026-09-06 13:51 — the lean is a CONSTANT ~0.065 rad flexion offset, and it is not compliance

Chasing the torso lean left after the constraint drive fixed the stride.

### Not the gravity feed-forward

`ActiveBone.ApplyGravityFeedForwardOnly` (new) applies the load compensation as an equal-and-opposite
pair ignoring `MuscleStrength`, because the spring path zeroes that and so silences the one term that
exists to cancel a constant gravitational torque. Open loop, spring at k x2 d x5:

    feed-forward OFF   75.1% upright   15 strikes   pitch 0.247
    feed-forward ON     8.6% upright    1 strike    pitch 0.495

**Worse, and it saturates** - `LoadCompensation` 1.0, 1.5 and 2.0 give byte-identical results, so the
term is pinned at `MaxTorque * LoadCompensationTorqueFraction`. Rejected.

### Not a sign error either

Per-axis correlation between Godot's and Isaac's achieved angles under the same targets, 24 axes:
**no axis is inverted.** The main flexion axes correlate +0.71 to +0.97. The seven weak ones are the
tightly-limited axes (`Shin.y/z` at +/-0.1 rad, `Foot.y/z` at +/-0.2) that chatter against their
stops, which is already on record and unrelated.

### What it is: a constant flexion offset

Same targets, the main flexion axes:

    Thigh_L.x   godot +0.303   isaac +0.187   godot is 0.116 MORE flexed
    Shin_R.x    godot -0.383   isaac -0.252   godot is 0.131 MORE flexed
    Foot_R.x    godot +0.024   isaac +0.107   godot is 0.083 LESS dorsiflexed

Godot's legs sit systematically more flexed - a crouch, which pitches the torso forward. The obvious
reading is spring deflection under body weight (`deflection = load / stiffness`), the classic
proportional offset that a hard positional constraint does not have.

**That reading is wrong.** Sweeping stiffness over a sixteen-fold range:

| kScale | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| flexion offset | 0.0720 | 0.0655 | 0.0658 | 0.0656 | 0.0658 |
| upright | **100%** | 15.2% | 11.2% | 11.7% | 7.3% |

**The offset is flat.** Deflection under load would fall as 1/k and shrink sixteen-fold; it does not
move. So the offset is **kinematic rather than compliant** - a constant difference in where the two
engines think the joint's zero is, not a spring sagging. Stiffness still governs stability (100%
upright at k x2, 7-19% above it), so the two effects are independent.

That is a different and much more tractable target than "the plant is too soft": a constant ~0.065 rad
frame or rest-pose discrepancy on the flexion axes, which should be findable by comparing the two
engines' joint zeros directly rather than through a gait.


## 2026-09-06 13:43 — the pelvis controller matches EXACTLY and is essential; the residual is a steady lean

Checked whether the torso-pitch divergence is a missing or mismatched attitude controller. It is
neither.

**The gains are already identical.** `PelvisStabilizationModule` in Godot: `Gain 600`,
`Damping 20`, `MaxTorque 300`. Isaac's `StandEnv._apply_balance_assist`: `balance_gain 600`,
`balance_damping 20`, `balance_max_torque 300`. Isaac's is a verbatim port and the numbers agree.

**And it is doing the work.** In angular-spring mode, open loop, 8 s:

| BalanceAssist | upright | strikes | pitch error |
|---|---|---|---|
| **1.0** | **100%** | **14** | **0.2914** |
| 0.5 | 24.2% | 4 | 0.7718 |
| 0.0 | 28.3% | 4 | 0.7443 |
| 2.0 | 100% | 14 | 0.2914 (identical - the driver clamps to [0,1]) |
| 1.0 + `PelvisReactIntoThighs` | 20.8% | 2 | 0.8171 |

Without it the spring drive falls; with it the body stays up for the whole run. **The pelvis
stabiliser is the only reason the constraint drive stands at all**, and it is already at full
strength - `BalanceAssist` is clamped to 1.0 in `IsaacPolicyDriver`, so 2.0 changes nothing.

### What the residual actually looks like

Godot's torso is not oscillating around upright, it is holding a **persistent forward lean of 0.2 to
0.4 rad** while Isaac stays within +/-0.09. A steady offset rather than a wobble is the signature of
a **steady-state error** - a constant disturbance torque that a finite-gain PD cannot fully reject,
not a stability or tuning failure.

That is a different thing to chase than everything before it. Candidates, none yet separated:

- **The gravity feed-forward is dead on this path.** `ActiveBone`'s load compensation acts through
  the PD, and the spring path sets `MuscleStrength = 0` to silence that PD - so the term that exists
  precisely to cancel a constant gravitational torque is switched off. Isaac's plant has no
  equivalent, but Isaac's XPBD drive holds position against gravity without needing one, which is the
  asymmetry recorded in `loadcompensation-zero-was-wrong`.
- The spring's equilibrium is the commanded joint angle, so any constant offset between the two
  engines' rest poses becomes a constant posture offset.
- The counter-torque distribution: Isaac splits the pelvis reaction across both feet explicitly;
  Godot distributes it across grounded feet, which differ once the gait is asymmetric.

The first is the most likely and the cheapest to test: reinstate a gravity feed-forward on the spring
path as a direct body torque rather than through the silenced PD.


## 2026-09-06 13:37 — with the constraint drive, JOINTS match and the TORSO does not

Calibrating the angular spring on delivered response rather than on walking: same recorded targets
into both engines, comparing achieved leg-joint angles, 20 cells of stiffness x damping.

    mean |delivered joint error|   0.0424 - 0.0509 rad across k x0.5-3.0 and d x1-5

**The landscape is flat.** A six-fold change in stiffness and five-fold in damping moves the joint
tracking error by 20%. So the spring reproduces Isaac's joint trajectory to about **2.7 degrees**
almost regardless of tuning - which is a much better joint-level match than the explicit path ever
achieved, and it means gain calibration is not where the remaining error lives.

### Where it does live: base attitude

Same targets, spring at k x2 d x5, over 5.5 s:

    t      joint err   ISAAC h  GODOT h   ISAAC pitch  GODOT pitch
    0.50     0.0486      0.809    0.824      -0.011      +0.057
    1.00     0.0465      0.816    0.848      -0.041      +0.398
    3.00     0.0531      0.824    0.841      -0.008      +0.237
    5.00     0.0397      0.827    0.829      -0.088      +0.353

**Joints track to 0.05 rad and pelvis height matches within 2 cm, while torso pitch diverges
completely** - Isaac holds within +/-0.09, Godot sits at +0.2 to +0.4, i.e. 12-23 degrees of lean.
The residual is the FLOATING BASE, not the joints.

That is a different class of difference from everything measured before. Isaac solves the humanoid as
one articulated body with a proper floating base; Godot solves sixteen independent rigid bodies
coupled by constraints. Identical joint angles do not imply identical base attitude when the base is
resolved differently.

### Jolt solver steps: swept, and MORE IS WORSE

`physics/jolt_physics_3d/simulation/velocity_steps` and `position_steps` are the constraint-solver
iterations - Jolt calls them "steps", which is why an earlier search for "iterations" found only the
motion-query setting. Both were at their defaults and `project.godot` set neither. Written
section-relative under `[physics]` and verified present in the file each run:

| velocity / position steps | upright | strikes | pitch error |
|---|---|---|---|
| **10 / 2 (default)** | **100%** | **14** | **0.2914** |
| 20 / 4 | 21.5% | 3 | 0.7952 |
| 40 / 8 | 32.3% | 4 | 0.6906 |
| 60 / 12 | 32.3% | 6 | 0.6820 |
| 40 / 16 | 31.7% | 3 | 0.6962 |

**The default is the best cell and every increase is worse**, with the pitch error more than doubling
while the joint error does not move (0.043-0.047 throughout). Eliminated, and it reinforces that the
problem is not constraint convergence at the joints.

### Status of the angular-spring path

- open loop, 8 s, k x2 d x5, default solver steps: **100% upright, 14 strikes, stride 0.16 m**
- open loop, 12 s: 75.1% upright, 15 strikes - it survives most of a run but not all of one
- closed loop with the policy: still fails, best 31.7% upright
- the explicit path's closed-loop baseline (12 strikes at 100% upright, stride 0.039) is still the
  best END-TO-END result; the spring is better open-loop and worse closed-loop


## 2026-09-06 13:28 — THE ACTUATOR FORMULATION WAS LIMITING THE STRIDE. Jolt angular spring, open loop.

`IsaacPolicyDriver.UseAngularSpring` drives the joints with Jolt's `angular_spring` - a position
constraint resolved INSIDE the solver - instead of `ActiveBone`'s explicit torque, which is silenced
by setting `MuscleStrength` to zero. No biomechanics on this path yet, deliberately: Hill, the effort
clamp and gravity feed-forward stay off so there is one unknown at a time.

**The springs were already in the scene and had been disabled the whole time**
(`angular_spring_x/enabled = false`, stiffness 350, damping 35).

### A mirrored-sign bug found first, and it invalidated the first run

The first sweep fell at every authority. Checking tracking rather than the outcome:

    commanded hip +0.315 rad  ->  achieved -0.319 rad

**Jolt's `equilibrium_point` is mirrored relative to the angles this rig reports** - the same
inversion already recorded for the joint limits, where the scene stores `[-upper, -lower]` so Jolt
enforces the anatomical range. The body was being driven backwards. After negating, tracking is good:

    mean |tracking error| = 0.047 rad over 3 s, body upright throughout

### Open loop, Isaac's recorded targets, no network

| drive | authority | upright | strikes | **stride** | travel |
|---|---|---|---|---|---|
| explicit torque | 0.125 | 40.4% | 6 | **0.0117 m** | 0.399 |
| **angular spring** | 0.150 | 41.8% | 7 | **0.1828 m** | 1.717 |
| angular spring | 0.200 | 20.1% | 2 | 0.2651 m | 1.837 |
| **angular spring, k×2 d×5** | 0.150 | **75.1%** | **15** | **0.1610 m** | 0.578 |
| Isaac source | - | 100% | 38 | 0.2529 m | 4.790 |

**Stride rises from 0.0117 m to 0.16-0.25 m - up to 21x - at equal or better uprightness.** The
best cell holds **75.1% upright with 15 foot strikes and a stride 64% of Isaac's**, against the
explicit path's 0.0117 m. This is the first evidence that the missing stride was an actuator
FORMULATION limit rather than a gain or tuning limit, and it is exactly the outcome an implicit
constraint drive predicts.

Still short of the bar: 75.1% is below the 90% uprightness every gait metric must clear, and the
75.1% cell is another isolated island (k×2 d×4, d×6, k×1.5 and k×2.5 all fall to 13-26%).

### Closed loop: the policy does NOT survive on the spring plant

| authority | k | d | upright | strikes | stride |
|---|---|---|---|---|---|
| 0.100 | 1 | 1 | **100%** | 1 | 0.0000 (statue) |
| 0.125 | 1 | 1 | 8.1% | 6 | 0.0835 |
| 0.150 | 2 | 5 | 9.5% | 0 | 0.0000 |
| 0.100 | 2 | 5 | 31.7% | 18 | 0.0325 |

Worse than the explicit path's closed-loop baseline (12 strikes at 100% upright). Expected in
hindsight: the policy was trained against Isaac's plant, and the spring is a different plant again -
matching its FORM is not the same as matching its response.

### What this changes

The open-loop result is the important one, because it removes the constraint that blocked everything
else: **Godot's actuator can produce an Isaac-like stride.** The remaining work is calibration rather
than architecture - tune the spring so Godot's DELIVERED joint response matches Isaac's (the
step-response and tracking-error tooling already exists), instead of tuning it to walk open-loop, and
only then expect the existing policy to transfer.


## 2026-09-06 12:50 — SPD compensation: it only destabilises. (An earlier 'stride x6.7' claim here is RETRACTED.)

`PidController3D.SpdCompensation` (new, 0-1) pre-multiplies both gains by `denominator^c`, so at 1.0
the joint receives the gains the rig contract authors instead of the reduced ones. This is the direct
falsification test for the Stable-PD ceiling hypothesis.

**The algebra first, because it reframes the damping work.** At 240 Hz the denominator is about 2.8
and is dominated by the `Kd·dt/I` term, not the `Kp·dt²/I` term. So **raising damping to stop the
vertical pogo also shrinks the position gain** - the stride-versus-stability trade-off measured all
night is visible directly in the denominator. The asymptote `I/dt²` is about 10100 for the thigh at
240 Hz against an authored 1600, so the authored gains are well inside the ceiling: the reduction is
avoidable loss rather than a hard physical limit.

### Open loop, Isaac's own targets, no network in either engine

    Isaac source stride                             0.2529 m

    compensation 0.00  stance 2.0   stride 0.0117   FALLEN
    compensation 0.25  stance 2.0   stride 0.0406   FALLEN
    compensation 0.75  stance 2.0   stride 0.0787   FALLEN
    compensation 1.00  stance 2.0   stride 0.0619   FALLEN

**RETRACTED — these stride numbers are invalid.** Every compensated row above has uprightness of
5-9%, meaning the body had already fallen, and stride is computed from touchdown positions, so a
collapsing body manufactures large apparent "strides". This is the same confound already recorded for
`maxFootZ` and it was walked into again. The correct reading of the table is that compensation
destabilises; it says nothing about stride.

Closed loop is the same shape - stride 0.0386 -> 0.0971 at compensation 0.75, and the body falls.

### What this means, after the retraction

Cross-checking **every** stride measured today against uprightness:

    configuration                  stride   upright   valid?
    closed comp1.0 st5 eff8        0.2833     7.8%    no - fallen
    open   comp1.0 st3 eff8        0.2622     8.6%    no - fallen
    closed comp1.0 st3 eff8        0.2126     7.2%    no - fallen
    open   comp0.75 st2 eff1       0.0787     5.6%    no - fallen
    closed BASELINE                0.0386   100.0%    YES
    Isaac source                   0.2529   100.0%    YES

**Not one configuration produces a stride above 0.04 m with the body still upright.** Every number
that looked like progress was a falling body. `SpdCompensation` and `EffortScale`, alone and
together, across 27 cells open- and closed-loop, produce exactly one thing: instability.

So the ceiling hypothesis is **not confirmed** by this. Undoing the SPD reduction does not buy stride
at constant uprightness - it buys a fall. The honest position is that the SPD reduction remains the
best explanation for the short stride on algebraic grounds (the `Kd·dt/I` term dominates the
denominator, so damping and position gain are the same budget) and has **no experimental support**
from the compensation test, because the compensated plant cannot stand up long enough to measure.

**Rule reinforced, twice in one day:** any gait metric - stride, `maxFootZ`, foot correlation,
excursion - is meaningless unless uprightness is reported beside it and is >= 90%. Both times the
error produced a result that looked like a breakthrough.


## 2026-09-06 12:28 — PLANT, definitively. The stride is impossible for Godot with the network removed.

The decisive experiment, prompted by an external review: drive BOTH engines from the same recorded
target trajectory with no network in either loop, and measure STRIDE rather than survival. The
open-loop replay had been run before but only ever scored on uprightness; stride instrumentation only
existed from this morning.

    engine / config             upright   strikes   stride    travel
    ISAAC (source)               100.0%      38     0.2529    4.790 m
    GODOT auth 0.125 damp 2.0     40.4%       6     0.0117    0.399 m
    GODOT auth 0.125 damp 1.0     11.0%       1     0.0000    1.258 m
    GODOT auth 0.15  damp 2.0      9.0%       5     0.0321    0.207 m
    GODOT auth 0.10  damp 2.0    100.0%       0     0.0000    0.015 m

**Godot cannot reproduce Isaac's stride from Isaac's own target trajectory.** 0.012-0.032 m against
0.253 m, with no policy involved anywhere. Every configuration either stands perfectly still or
falls. The missing stride is **already impossible for the Godot actuator/constraint stack**, so it is
not a closed-loop decision the policy is making badly.

That collapses the search space: **plant problem, not policy problem.** No amount of retraining,
domain randomisation, reward shaping or checkpoint selection addresses this, and the 1-in-20
transfer rate is a symptom rather than the disease.

### The alternative low-level controller, tested

`IsaacPolicyDriver.JointSpacePd` already exists: a per-axis Stable PD driving torque directly from
the RIG CONTRACT gains, bypassing `ActiveBone` entirely - no Hill force-velocity, no effort clamp, no
muscle strength, no load compensation. Same 36 targets, different actuator realisation. Never tested
with any current policy. Eight cells:

    StablePD  auth 0.125 damp 2.0   WALK    100% upright   stride 0.0386
    JointSpacePD  every cell        FALLEN  2.8-4.9% upright   excursion 1.65 m   travel up to 8.1 m

**Worse, not better.** The 1.65 m "excursion" and 8 m of travel are the legs flailing and the body
being flung - removing the biomechanical layer removes the torque limiting that keeps the body
together. So a different servo formulation does not rescue it either, at least not this one.

### What this rules in

The remaining hypothesis is the one the Stable-PD algebra predicts: Godot's effective joint gain is
capped at `I/dt^2` by the SPD denominator, measured as authored `kp 882 -> effective 109`, and that
ceiling is below what the gait needs. Fixing it means changing how Godot realises the actuator, not
how Isaac trains the policy.


## 2026-09-06 11:26 — FOUR targeted fixes for the short stride, all failed. The trade-off is the plant.

Knowing the proximate cause did not produce a fix. Each of these attacks the measured deficit
directly (swing excursion 0.087-0.105 m against Isaac's 0.152-0.211 m, stride 28-55 mm against
250 mm) and each fails the same way.

**1. Smoothed damping transition** (`DampingBlendSeconds`). Rationale was good: both feet are down
~97% of the time, so `SwingDampingScale` acts on a few percent of ticks yet flips WALK to FALLEN -
which looks like a discontinuity artefact, `Kd` stepping by 3x between ticks. Blending over 30-250 ms
never helps; at every setting the run either statues or falls.

*Implementation bug found and fixed along the way:* `_blendedScale` initialised to 1.0 and ramped UP
to the stance value, leaving the body under-damped for the first tens of milliseconds - enough for
the pogo to start. That alone turned the 100%-upright baseline into a fall. Seeded to
`StanceDampingScale` in `ResolveLegChains`; baseline restored exactly (12 strikes, 100% upright).

**2. Swing-leg action amplification** (`SwingActionScale`, new). Asks the swing leg for more angle so
it covers Isaac's excursion in the shorter time Godot's swing allows:

| swing amp | upright | swing excursion | stride | travel |
|---|---|---|---|---|
| **1.0** | **100%** | 0.105 | 0.0386 | 0.153 |
| 1.5 | 45.9% | 0.651 | 0.0523 | 0.712 |
| 2.0 | 9.5% | 0.618 | 0.0134 | 0.901 |
| 3.0 | 14.0% | 0.850 | 0.0636 | 1.349 |

**It fixes the excursion** - 0.105 -> 0.85, past Isaac's 0.21 - and the body falls. Stride barely
moves (0.039 -> 0.064), so the extra travel is toppling.

**3. Physics rate, re-tested with THIS policy and damping.** Higher rate shrinks the SPD denominator
and raises effective gain, which is the authority the leg lacks. 240/360/480 Hz x stance 1.0/1.5/2.0,
nine cells: only 240 Hz at stance 2.0 stands. Everything else falls.

**4. Short-stride training** (`cmd_lin_vel_x = (0.10, 0.25)`), to train the gait Godot can execute
rather than one it cannot. At its own band the policy becomes a FLAMINGO (cmd 0.15: 0 strikes, 98.9%
single support) and falls in Godot.

### What the pattern means

    every intervention that increases stride    ->  destabilises
    every intervention that stabilises          ->  shortens stride

Four different mechanisms - damping magnitude, damping timing, commanded angle, solver rate - all
move along the same trade-off rather than off it. **Godot's legs cannot simultaneously support the
body and swing with the authority available**, and no deployment-side knob changes that, because they
all draw on the same `Kd`/`Kp` budget through the SPD denominator.

That is a statement about the plant, not about tuning, and it means the remaining paths are:
raise the authority at its source (the SPD denominator caps effective `Kp` at `I/dt^2`, so the
authored gains cannot be reached - this may require a different low-level controller in Godot), or
train a gait whose stance phase never needs the leg to be stiff and its swing never needs it to be
free, which is a reward-shaping problem and not a transfer problem.


## 2026-09-06 10:42 — WHY it does not travel: the stride is 5-9x too short, and it is NOT slipping

Both traces now carry world horizontal foot and pelvis position (`footX_*`, `footZfwd_*`, `pelvisX`,
`pelvisZfwd`), added to `IsaacDriverDiagnostics` and `dump_obs.py`. Height alone could not tell a slip
from a missing push-off.

**Method note:** the first version measured the wrong axis. Godot's forward is -Z, which the rig frame
map `to_usd(p) = (-p[2], -p[0], p[1])` sends to USD index 0 - I used index 1, i.e. lateral drift, in
both engines. Both axes are now recorded, and displacement is reported as the planar magnitude
because the root yaws and the command is body-frame.

    total horizontal travel   ISAAC 4.75 m in 12 s (0.40 m/s)   GODOT 0.15 m in 20 s (0.008 m/s)

### The three numbers that localise it

| | touchdowns | mean stride | stance slip |
|---|---|---|---|
| GODOT L | 7 | **0.0275 m** | 0.11 mm/step |
| GODOT R | 5 | **0.0553 m** | 0.22 mm/step |
| ISAAC L | 19 | **0.2465 m** | 2.65 mm/step |
| ISAAC R | 19 | **0.2544 m** | 2.39 mm/step |

**Godot's stride is 28-55 mm against Isaac's 250 mm.** And Godot's stance slip is LOWER than Isaac's,
so the planted foot is not sliding - this is not a friction or push-off-into-slip failure. The feet
are placed firmly and simply do not go anywhere.

### The leg does not reach

Swing-phase foot excursion relative to the pelvis - how far the leg actually reaches fore-aft:

    GODOT  L 0.0865 m  (48 swing samples)      R 0.1052 m  (18 samples)
    ISAAC  L 0.1522 m  (113 swing samples)     R 0.2106 m  (580 samples)

**Half the excursion, and a swing phase an order of magnitude shorter in duration.** Foot clearance is
comparable in both (~23 mm), so the dummy lifts its foot correctly, holds it up for a fraction of the
time Isaac does, and puts it back down almost where it started.

That is the damping trade-off, now measured in the gait rather than inferred: the `Kd` that stops the
pogo is also what stops the leg reaching.

### Freeing the swing leg now that the gate can engage: still fails

`SwingDampingScale` had never actually run - the contact gate could not engage because no foot ever
cleared the 0.05 threshold. It can now (`maxFootZ` 0.063). Tested at authority 0.125, stance 2.0:

| swing damping | gate | verdict | strikes | upright | stride | travel |
|---|---|---|---|---|---|---|
| **1.00** | 0 | **WALK** | 12 | **100%** | 0.0386 | 0.153 m |
| 0.70 | 0 | FALLEN | 8 | 11.1% | 0.0335 | 0.751 m |
| 0.50 | 0 | FALLEN | 8 | 9.4% | 0.0424 | 1.044 m |
| 0.15 | 0 | FALLEN | 5 | 7.1% | 0.0509 | 1.865 m |

Travel rises monotonically as the swing leg is freed - to 1.87 m - and uprightness collapses with it.
**The travel is the body falling forward, not walking**; note the stride never improves (0.013-0.051 m
throughout), so the extra distance is toppling, not stepping.


## 2026-09-06 06:50 (overnight E/F) — 16-fragment search, and the stepping is ONE checkpoint

A sequential population search ran 02:10-06:28: 16 fragments from `walk_hz960/model_36536`, seeds
varied first, then narrower band, then push back to default. **Every one produces a genuine WALK in
Isaac** - 3.2-5.0 m, 38-46 strikes, 100% alternation, 100% upright. The Isaac gait is now
reproducible across seeds and settings rather than being one lucky checkpoint.

**None of them transfers.** Top six plus the base, four Godot cells each (28 runs):

    every search checkpoint      STATUE at maxFootZ 0.050 (no lift), or FALLEN
    hz960/model_36536  0.125/2.0 STATUE, 7 strikes, 83% alternation, maxFootZ 0.069, 100% upright
    hz960/model_36536  0.125/1.8 STATUE, 5 strikes, 75% alternation, maxFootZ 0.068, 100% upright

**The stepping belongs to `model_36536` specifically**, not to the recipe that produced it. Sixteen
siblings that walk just as well in Isaac do not step in Godot at all.

### Knob sweep at the island: nothing improves on the baseline

    baseline                        STATUE  7 strikes  83% alt  100% upright  maxFZ 0.069
    BalanceJointModules = false     FALLEN                        6.9% upright
    BalanceAssist = 0               FALLEN                       10.0% upright
    PelvisReactIntoThighs           STATUE  0 strikes            100% upright  maxFZ 0.050
    LoadCompensation = 1.0          FALLEN                        3.1% upright
    HillVelocityFilter = 1.0        STATUE  6 strikes  80% alt   100% upright  maxFZ 0.069
    LockArmsAtRest = true           STATUE  0 strikes            100% upright  maxFZ 0.050
    SwingDampingScale 0.5 + gate 1  FALLEN  8 strikes  86% alt     4.3% upright

Two things changed since the same sweep was run on the old policy. **Godot's balance layer now
HELPS** - `BalanceAssist = 0` falls, where previously it made no difference - so the earlier
"eliminated" verdict on the balance layer applies only to the statue-policy regime. And the
height-gated split gives the most strikes ever seen in Godot (8, 86% alternation) while falling; ten
further cells of stance/swing/margin around it all fall, confirming that unilateral damping always
surrenders the bilateral stability.

### Validation of the best configuration: reproducible, and very fragile

    nominal          STATUE  7 strikes  83% alt  100% upright  maxFZ 0.069
    40 s duration    STATUE  7 strikes  83% alt  100% upright  maxFZ 0.069   <- identical
    SpawnActionNoise 0.05   FALLEN  10.4% upright
    SpawnActionNoise 0.10   FALLEN   8.4% upright
    SpawnActionNoise 0.20   FALLEN   4.4% upright

Bit-identical over twice the duration, and knocked down by 0.05 of action noise for half a second.
Real, reproducible, and with no disturbance tolerance whatsoever.

### Stand regression: fails, and it is NOT from tonight

`IsaacStandCheckNewton` uses `balance_policy.onnx`, a different model, and scores 10.8% upright. Since
`ContactHeight` changed tonight, that could not be assumed pre-existing, so it was tested directly by
rebuilding with the old 0.06 and re-running:

    ContactHeight 0.05 (new)   10.8% upright
    ContactHeight 0.06 (old)    5.9% upright

**Stand fails either way, and the new threshold is slightly better.** The breakage is the known
`balance_policy` invalidation from the collider and joint-limit fixes, not tonight's work.


## 2026-09-06 02:10 (overnight E) — two more training variants tried, both discarded

**Chained fragment at 1/960 (`hz960b`): WORSE in both engines.** Isaac cmd 0.30 fell 6.90 -> 4.58 m,
and Godot lost the stepping entirely - `maxFootZ` back to 0.050, zero strikes, at every cell of the
island. Discarded by the plan's gate; `model_36536` restored.

**Command sweep at the island: no help.** Only cmd 0.30 steps. 0.50 / 0.70 / 1.00 are all statues at
`maxFootZ` 0.050, despite Isaac walking BETTER at 0.50 (9.39 m) than at 0.30. So the Godot island is
specific to the command as well as to the damping.

**`target_damping = 1.0`: WORSE.** Motivated by a real asymmetry - the policy trains on an undamped
Isaac plant and deploys onto a Godot plant damped 2x to keep it upright, so its gait assumes free legs
and meets stiff ones. Isaac still walked (3.53 m, 37 strikes) but Godot went to **FALLEN in all ten
cells**, where the undamped-trained policy had 100%-upright statues and one stepping cell. The
target-space approximation of damping is not the same object as Godot's `Kd`, and adding it costs the
stability the statues had.

### Method change for the rest of the night

Three consecutive single fragments have gone backwards, and fragment-to-fragment variance has already
flipped a verdict on this project. A chain is the wrong search: it compounds one noisy draw. Switched
to a **sequential population search around `model_36536`** - 16 fragments, seeds varied first, then
one setting at a time (narrower command band, push back to default), each scored in Isaac with
`--playback` pinned, results written to `search_results.json` with a leaderboard. Runs strictly
sequentially so the GPU is never shared, and stops itself at 07:30 to leave time for the Godot grid,
validation and the report.


## 2026-09-06 01:20 (overnight E) — best Isaac gait yet, and Godot now STEPS IN PLACE

`walk_hz960/model_36536` - 15 min at `sim.dt = 1/960, decimation = 16` on top of the narrow-band,
contact-stuck, push-hardened lineage. Isaac, `--playback` pinned so the numbers are reproducible:

| cmd | verdict | dist (12 s) | strikes | alternation | flight |
|---|---|---|---|---|---|
| 0.30 | **WALK** | **6.90 m** | **48** | 100% | 6.1% |
| 0.50 | **WALK** | **9.39 m** | **56** | 100% | 6.5% |

Best gait this project has produced, on the plant whose contact regime is closest to Godot's.

### Godot: a real, alternating step in place

An 18-cell grid plus a 10-cell refinement. The transfer is still not a walk, but the failure has
moved:

| authority | stance damping | verdict | strikes | alternation | maxFootZ | upright |
|---|---|---|---|---|---|---|
| 0.125 | 1.8 | STATUE | 5 | 75% | **0.070** | 100% |
| **0.125** | **2.0** | STATUE | **7** | **83%** | **0.069** | **100%** |
| 0.125 | 2.2 | STATUE | 0 | - | 0.050 | 100% |
| 0.135 | 2.0 | FALLEN | 2 | 100% | 0.296 | 3.1% |
| 0.115 | 2.0 | STATUE | 0 | - | 0.050 | 100% |

**Seven alternating foot strikes with 29 mm of clearance at 100% uprightness.** Compare every stable
configuration before tonight, which sat at `maxFootZ` 0.050 - the spawn value - with zero strikes.
The dummy is now picking its feet up and putting them down alternately in Godot.

It does not translate (-0.08 m in 20 s), and the island is narrow: 1.8-2.0 stance damping at 0.125
authority, with 2.2 giving no lift and 0.135 falling. But this is the first configuration in the
project where Godot exhibits gait STRUCTURE rather than a statue or a collapse.

The 1/960 fragment managed only 650 iterations in 15 minutes - eight times the physics work per
step - so this policy is undertrained on its own plant. Chaining more fragments.


## 2026-09-06 00:55 (overnight D/E) — the scripted-lift control, and penetration ELIMINATED as the blocker

### Godot's foot does not lift under a deliberate, policy-free command

A hand-written action sequence (weight shift, then left hip +0.9, knee -0.9, ankle +0.5, held),
driven open-loop into both engines. Godot at authority 0.15:

    joints REACH their commands   hip +0.321 (cmd +0.283)   knee -0.453 (cmd -0.351)
    footZ_L                        0.0398 -> 0.0398          ZERO movement
    pelvisZ                        0.830  -> 0.811           -19 mm

Above authority 0.15 the body falls before the foot rises. **First reading was that Godot's plant
cannot lift a foot at all** - but that was premature: with body weight still on the left leg, hip and
knee flexion simply reshapes the leg against the ground. The abduction "weight shift" in the script
was only 0.06 rad, nowhere near enough to unload it. The test needed a control.

### The control: identical actions in Isaac

| | ISAAC | GODOT |
|---|---|---|
| footZ_L | 0.0365 -> **0.0626** (+26 mm) | 0.0398 -> 0.0398 (**0 mm**) |
| pelvisZ | 0.819 -> 0.776 (-43 mm) | 0.830 -> 0.811 (-19 mm) |
| hip | 0.259 | **0.321** |
| knee | -0.311 | **-0.453** |

**Godot's joints rotate 1.2-1.5x MORE and its foot rises 0 mm where Isaac's rises 26 mm.** So the
plant is not short of authority - it is short of foot CLEARANCE, and Isaac gets clearance partly by
the pelvis dropping 43 mm onto a stance foot that can sink into the ground.

`replay_isaac.py` now writes `footZ_L`, `footZ_R`, `pelvisZ`, so this comparison is repeatable.

**A forward-kinematics cross-check was inconclusive and is recorded as such:** a 2-link sagittal
model (thigh 0.32, shin 0.34, anchors verified identical - hip 0.74, knee 0.42, ankle 0.08 in both
engines) carries a constant ~80 mm bias in BOTH engines, so it is miscalibrated (it ignores foot
rotation and out-of-plane motion) and cannot support a claim that either engine's reported angles
are wrong. The hypothesis that Godot's joint-angle observation is inflated is NOT supported.

### Penetration is ELIMINATED as the blocker

If the gait depended on ground compliance, removing it should break the gait. Swept `sim.dt` on
`walk_robust2/model_35556` (policy rate held at 60 Hz by matching `decimation`):

| sim.dt | penetration frac | median | max | Isaac verdict |
|---|---|---|---|---|
| 1/240 (dec 4) | 27.0% | 5.05 mm | 12.50 mm | WALK, 15 strikes |
| 1/480 (dec 8) | 16.8% | 1.50 mm | 4.20 mm | WALK, 13 strikes |
| **1/960 (dec 16)** | **1.3%** | 1.25 mm | **1.90 mm** | **WALK, 13 strikes** |
| Godot | 0.0% | - | - | - |

**At 1/960 the penetration is essentially gone and the policy still walks.** So the gait does not
depend on it, and the round-6 reading - that Isaac's gait is "built on pushing into a compliant
floor" - is wrong. What was true in round 6 is narrower: a policy trained at 1/120 breaks when moved
to 1/240. A policy trained at 1/240 tolerates 1/960 fine.

The useful consequence is that **1/960 gives Isaac a contact regime nearly identical to Godot's**, so
training there removes ground compliance as a difference between the engines rather than merely
reducing it.


## 2026-09-06 00:10 (overnight D) — MEASUREMENT BUG: every single-run gait score carried a verdict-flipping spread

Three consecutive scores of ONE checkpoint (`walk_lowscale/model_33449`, cmd 0.30), unpinned:

    WALK  3.62 m  30 strikes  100% alternation   4.2% flight
    HOP   0.28 m  40 strikes    0% alternation  24.7% flight
    WALK  3.12 m  30 strikes  100% alternation   1.9% flight

`action_scale_range` and `effort_scale_range` are drawn PER EPISODE, and `dump_obs.py` never set
`playback` - while `evaluate_walk.py` always has, with the comment "measure the policy, not the
observation noise". **The verdict itself flips between runs.** Every single-run number taken through
`gait_score` before this point carries that spread, mine included.

Fixed: `dump_obs.py --playback` pins both ranges to (1.0, 1.0) and `gait_score` passes it by default.
Re-measured three times: **4.21 / 4.06 / 4.16 m**, 32-33 strikes, 97% alternation. Reproducible.

**Second fix in the same area.** `gait_score.analyse` read the contact FLAGS, which are an observation
channel - so anything that rewrites that channel falsifies the score. A policy trained with
`obs_contact_stuck_prob` scored "0 strikes, 99.9% double support" purely because the trace carried
the forced flags. It now derives contacts from `footZ_*`, which is ground truth in both engines.
Verified the two agree 100% of samples when nothing is masking them, so this changes no earlier
conclusion - it only makes the metric robust to the masking.

## 2026-09-06 00:12 (overnight D) — the BISECTION: the plant is the blocker, not the observation loop

`obs_contact_stuck_prob = 0.5` trains the policy with the contact flags frozen at double support for
half of all episodes, so it cannot sequence a gait from flag alternation - which is exactly Godot's
situation ([[Godot's flags are a constant]]). The policy **still walks in Isaac**: cmd 0.30 gives
4.13 m / 32 strikes / 100% alternation, cmd 0.50 gives 6.60 m / 43 strikes. So the crutch was
removable.

**It does not transfer.** An 18-cell grid (authority 0.10/0.125/0.15 x stance damping 1/2/3 x arms
locked/free) gives STATUE or FALLEN in every cell, with `maxFootZ` at exactly 0.050 - the spawn value
- in every stable one.

**And the open-loop replay settles which side is at fault.** Driving Godot from this policy's exact
recorded actions, under the best damping configuration, at six settings:

    authority 0.10-0.20 x stance 1/2/3 x arms locked/free   ->  FALLEN in all six

Godot's BODY cannot execute a gait that Isaac's body executes from identical actions. **The blocker
is the plant, not the observation.** Every observation-side treatment - and there have been many -
was addressing the wrong half.

Consistent with that, Godot has **no disturbance rejection at all** under this policy: a pelvis
impulse of 20 N.s at t=2 s drops uprightness from 100% to 13.2%, and 40 or 80 N.s is no worse
because it has already fallen. Isaac trains against `push_velocity = 0.25` and survives it. Godot is
stable only when perfectly undisturbed, which is the same statement as "stable only as a statue".


## 2026-09-05 23:42 (overnight C, second measurement) — Godot's contact flags are a CONSTANT

Same policy, 240 steps, cmd 0.30:

    ISAAC   contact-flag transitions  15/239   distinct states 4   (0,0) (0,1) (1,0) (1,1)
    GODOT   contact-flag transitions   0/239   distinct states 1   (1,1) only

    ISAAC   double 14.6%   single 66.7%   flight 18.8%
    GODOT   double  100%   single   0.0%  flight   0.0%

**Two of the 143 inputs are frozen.** In Godot the policy is permanently told "both feet down" - a
state it occupied only 14.6% of the time in training - and never receives the phase signal it uses to
sequence a gait. In Isaac those two floats change 15 times in four seconds and visit all four support
states.

This is the sharpest statement yet of the transfer failure, and it is not a plant mismatch: the
plants agree on joint positions, mass, geometry and single-joint step response. **The policy is being
run open-loop with respect to gait phase**, because the one channel that carries phase is a constant.

Whether the constancy is a cause or a symptom is the open question - the flags are derived from foot
HEIGHT, and the foot does not lift, so the two are coupled. But it makes the training-side fix
concrete and falsifiable: if the policy is taught to sequence its gait WITHOUT the flags
(`obs_contact_stuck_prob`, new), it should no longer need them in Godot.


## 2026-09-05 23:35 (overnight C) — 18 of 143 channels are out of distribution, and the ARMS are the worst

Per channel, the fraction of Godot samples falling outside the 1st-99th percentile range Isaac
trained on (same policy, cmd 0.30, 240 steps):

| channel | out of range | godot mean | isaac p1..p99 |
|---|---|---|---|
| `pos_UpperArm_R.z` | **99.6%** | +0.111 | -0.205 .. -0.017 (**opposite sign**) |
| `pos_Forearm_R.x` | 98.3% | +0.340 | -0.102 .. 0.110 |
| `pos_Forearm_L.x` | 98.3% | +0.515 | -0.010 .. 0.216 |
| `pos_Thigh_R.y` | 91.7% | +0.057 | -0.011 .. 0.046 |
| `pos_Shin_L.y` | 90.8% | +0.060 | -0.019 .. 0.020 |

**18 of 143 channels are >50% out of range; 8 are >90%.** The top three are ARM joints, which carry
none of the gait but do feed the policy. Godot's arms are soft (kp 100-120) and sag under gravity
where XPBD's positional drive holds them, so the elbow sits three to five times more flexed than
anything the policy ever saw. After the arms come the tight lateral axes (`Shin.y/.z`, `Thigh.y`,
`Foot.y`), which is the limit chatter already on record.

### `LoadCompensation` is not a reliable fix for the arm sag

    authority 0.150   LC 0 -> 1   Forearm_L.x 0.372 -> 0.237   (into range, but FALLS: 7.7% upright)
    authority 0.125   LC 0 -> 1   Forearm_L.x       -> 0.495   (further OUT of range)

### `LockArmsAtRest` — and the first foot lift with the body still standing

New export: hold UpperArm/Forearm/Hand at the rest pose instead of applying the policy's offset.
Swept against stance damping at authority 0.15:

| authority | stance | arms locked | verdict | upright | height std | **maxFootZ** |
|---|---|---|---|---|---|---|
| 0.150 | 3.0 | no | STATUE | 100% | 0.0037 | 0.050 |
| 0.150 | 3.0 | yes | FALLEN | 20.6% | 0.0049 | 0.439 |
| **0.150** | **2.0** | **yes** | STATUE | **100%** | 0.0065 | **0.064** |
| 0.150 | 1.0 | yes | FALLEN | 4.3% | 0.2399 | 0.383 |

**`maxFootZ = 0.064` at 100% upright is the first time in this project a foot has risen above its
0.040 resting height while the body stayed standing** - 24 mm of clearance, with 4 strikes. Every
previous stable configuration sat at exactly 0.050, the spawn value.

It is a razor-thin island and not yet a gait: stance 1.8 falls, stance 2.2 gives no lift at all, and
the run covers -0.03 m in 20 s, so those 4 strikes are scuffing rather than stepping. Recorded as the
best Godot configuration so far, not as a solution.

**Note the arm angles barely moved** (`Forearm_L.x` 0.372 -> 0.355): locking the TARGET at rest does
not stop a soft arm sagging under gravity, so the behaviour change comes from the arms no longer
being *driven* by the policy, not from the channel returning to distribution. Making the arms
actually hold their pose needs stiffness, which has no export yet.


## 2026-09-05 23:25 (overnight A+B) — damping cannot be split ANY way that permits a step

Two mechanisms implemented and swept, both aimed at the round-8 finding that one damping value
cannot serve stance and swing. Both fail, and together they close the door on splitting a single
gain.

### A — frequency-selective damping DESTABILISES

`PidController3D.HighPassDerivativeGain` damps only the fast component of measured angular velocity
(`kd * (filtered - slowEMA)`), divided by the same SPD denominator as the other terms. The idea was
sound - the pogo is ~3 Hz, a swing is slower - and it needs no phase detection, so it sidesteps the
bootstrap problem. Measured at authority 0.15:

| stance | high-pass | upright | height std |
|---|---|---|---|
| 3.0 | 0 | **100%** | 0.0037 |
| 3.0 | 6.0 | 5.1% | 0.2659 |
| 2.0 | 6.0 | 8.0% | 0.1687 |
| 1.5 | 6.0 | 12.6% | 0.0240 |

**Adding the high-pass term to a configuration that was 100% upright drops it to 5%.** The EMA
difference lags the true velocity, and a damping torque computed from a lagged signal injects energy
at some frequency instead of removing it. Sound idea, wrong signal - a differentiator needs phase
lead, not lag.

### B — height-gated stance/swing gives up the bilateral damping that was holding the body up

`StanceGateMode = 1` makes the LOWER foot the stance foot with hysteresis, so exactly one leg is
stance even in double support - which fixes the bootstrap flaw (mode 0 can never engage because no
foot lifts). With one leg damped and the other free:

    stance 3.0 / swing 1.0     upright  3.3%   height std 0.2842
    stance 3.0 / swing 0.5     upright  5.8%   height std 0.2650
    stance 4.0 / swing 0.25    upright  4.7%   height std 0.2806

All fall: the stability came from damping BOTH legs, and freeing one brings the pogo straight back.
Keeping both damped but asymmetric restores stability and nothing else:

    stance 4.0 / swing 2.0     upright 100%   height std 0.0057   maxFootZ 0.050
    stance 8.0 / swing 3.0     upright 100%   height std 0.0054   maxFootZ 0.050

`maxFootZ` is 0.050 in every stable row, which is the SPAWN height - the foot never once rises above
its 0.040 resting height.

### The result that matters

    both legs damped >= 2x authored   ->  100% upright, ZERO foot lift
    any leg damped   <  2x authored   ->  pogo returns, falls

**Godot needs >=2x authored damping to stand and cannot swing at >=2x.** The two requirements do not
overlap at any split - bilateral, unilateral, or frequency-based - so no scaling of `Kd` alone can
produce a step with this policy. The next question is therefore not how to divide the damping but why
Godot needs so much of it, and the measured lead is that Godot's policy SATURATES far more than
Isaac's on the same task: **45.7% of action components above 0.95 against Isaac's 30.7%**. A policy
commanding extremes drives violent motion that then needs damping to contain. That points at the
observation, which is Phase C.

**Method note:** the Phase B gate as written in the overnight plan (`foot correlation < 0.7 AND
maxFootZ > 0.055`) was underspecified - a tumbling body passes both trivially, because a falling
dummy has decorrelated feet high off the ground. Every gate involving foot height needs
`upright >= 90%` beside it.


## 2026-09-05 PM (round 8) — STANCE-selective damping removes the fall; the legs move in LOCKSTEP

`IsaacPolicyDriver.StanceDampingScale` / `SwingDampingScale` split leg damping by gait phase, gated
on the same foot-height contact rule the observation uses. Authority sweep, `walk_lowscale/model_33449`:

| authority | stance | swing | verdict | upright | height std |
|---|---|---|---|---|---|
| 0.10 | 1.0 | 1.0 | FALLEN | 3.2% | 0.2513 |
| 0.125 | 3.0 | 1.0 | STATUE | **100%** | 0.0034 |
| **0.150** | **3.0** | **1.0** | **STATUE** | **100%** | 0.0037 |
| **0.150** | **4.0** | **0.5** | **STATUE** | **100%** | 0.0042 |

**The fall is solved.** Authority 0.15 - the level the gait needs, and a guaranteed collapse in every
previous configuration - now holds 100% upright. The stable region reaches the gait authority for the
first time on this project.

### What blocks the gait now: the two legs move in lockstep

Same policy, authority 0.10, 4 s:

    left-vs-right correlation      GODOT      ISAAC
    hip                           +0.868     +0.560
    knee                          +0.636     +0.553
    footZ                         +0.993     +0.568

**Godot's feet move together almost perfectly.** The body SQUATS rather than steps, which is why the
hip and knee excursions are actually LARGER than Isaac's (1.15x and 1.16x) while producing no foot
lift at all. Measured foot ranges: Godot hip 0.000..0.215 against Isaac 0.000..0.187; Godot knee
-0.353..0.000 against Isaac -0.233..+0.071.

The loop is self-sustaining and runs through the OBSERVATION: contact flags never differentiate ->
the policy drives both legs symmetrically -> no foot lifts -> the flags never differentiate. Isaac
escapes it because its flags alternate (84-96% single support).

Spawn noise does break the symmetry - `SpawnActionNoise = 1.0` takes the foot correlation from
+0.993 to **+0.220** - and the foot still never lifts. In every stable configuration `maxFootZ` is
exactly 0.050, which is the SPAWN height; after settling to 0.040 the foot never rises again.

### A flaw in the phase split, stated plainly

The split is gated on contact, and no foot ever leaves the ground, so **both legs are permanently
classified as stance** and both get the slow damping. `SwingDampingScale` never applies. The
mechanism cannot bootstrap itself: it needs a swing to detect a swing.

**Next, and it follows directly:** gate the split on something that is asymmetric even when both feet
are down - relative foot LOAD or relative foot height (the lower foot is the stance foot), or the
policy's own commanded direction - rather than on a binary contact flag that reads the same for both
legs. Alternatively make the damping frequency-selective rather than phase-selective: the pogo is a
~3 Hz oscillation while a swing is a slower excursion, so a high-pass on joint velocity would damp
the bounce without slowing the step. The second is the more principled fix and neither has been tried.


## 2026-09-05 PM (round 7) — the fall is a GROWING VERTICAL OSCILLATION, and damping kills it

Foundations reviewed first, and they are CLEAN. Per-body against `dummy_d6.usd`:

    masses          16 bodies, 80.6 kg, every body matches
    shape classes   Cube / Capsule / Sphere handled per body, no substitutions
    capsules        Godot `height` is TOTAL (with caps), USD `height` is CYLINDER only;
                    `build_d6_usd.py` subtracts 2r correctly on all six
    foot box        Godot (0.12,0.08,0.22) -> USD (0.22,0.12,0.08), permutation correct,
                    0.08 m tall, half-height 0.040 = the measured resting COM in both engines

**The geometry is not the problem.** That is a positive result: the collider fix did its job and the
rigs are the same body.

### How Godot actually fails

At authority 0.15, reading the height and foot traces rather than the verdict:

    t      height   footZ_L  footZ_R
    0.50   0.800     0.044    0.050
    1.00   0.844     0.076    0.082
    1.50   0.831     0.053    0.065
    1.67   0.869     0.118    0.100    <- both feet off the ground
    2.00   0.852     0.124    0.121
    2.33   0.526     0.115    0.114    <- collapse

Pelvis height oscillates at **~3 Hz with GROWING amplitude** - +/-0.015 m at t=0.5 s, +/-0.04 m at
t=1.8 s - until both feet leave together and the body drops. **This is not a balance failure; it is
an under-damped vertical mode being pumped**, the compliant-leg pogo recorded earlier on this project,
now caught in the act with numbers.

### The gait trained at `action_scale = 0.10`: walks in Isaac, still will not transfer

`walk_lowscale/model_33449`, 15 min from the 240 Hz checkpoint with `action_scale=0.10` so the
DEPLOYED authority sits inside Godot's damped stable region. Isaac: **WALK at both commands** -
cmd 0.30 3.89 m / 33 strikes / 100% alternation, cmd 0.50 7.02 m / 40 strikes. Godot:

| authority | DampingScale | verdict | strikes | upright | height std |
|---|---|---|---|---|---|
| 0.10 | 1.0 | FALLEN | 4 | 3.2% | 0.2513 |
| 0.10 | 3.0 | STATUE | 0 | **100%** | 0.0058 |
| 0.125 | 3.0 | FALLEN | 7 | 3.6% | **0.0047** |
| 0.15 | 3.0 | FALLEN | 3 | 14.3% | 0.0054 |

### And this is the trade-off that closes the round

At authority 0.125 with damping the height standard deviation is 0.0047 - **the pogo is gone** - and
the first 2.2 s are genuinely stable: height flat at 0.805-0.815, tilt under 0.05, creeping forward
at 0.14 m/s. But:

    footZ_L / footZ_R = 0.040 / 0.040, unchanged, for the whole 2.2 s

**The feet never leave the ground.** The same `Kd` that stops the vertical bounce also slows the
hip/knee flexion that lifts a foot, and the same policy takes 33 strikes in Isaac. So Godot sits
between two failures:

    DampingScale 1     under-damped vertical mode -> pogo -> fall
    DampingScale 3-4   swing too slow to lift a foot -> statue

Both are the SAME quantity - Godot's effective `Kd` - pulled in opposite directions by the support
phase and the swing phase. A single body-wide scalar cannot satisfy both, which is why every
body-wide treatment in this file has failed.

**That names the next fix precisely: the damping has to be phase- or axis-selective**, not global.
Options in rough order of cost: damp the STANCE leg only (contact state is already computed);
damp the vertical/support axes (hip x, knee x) differently from the swing axes; or lift Godot's
effective `Kd` at the source by reducing the SPD denominator rather than multiplying the authored
gain. All three are reachable from `ActiveBone` and none has been tried.

### Damping: the first intervention with a clean dose-response

New verified export `IsaacPolicyDriver.DampingScale`, multiplying every controlled bone's
`ActiveBone.DerivativeGain` (logged as applied, like `EffortScale`). Authority 0.15:

| DampingScale | height std | height range | upright |
|---|---|---|---|
| 1.0 | 0.0845 | 0.2422 | 9.9% |
| **2.0** | **0.0045** | **0.0301** | 17.8% |
| **4.0** | 0.0047 | 0.0293 | **24.2%** |
| 8.0 | 0.0610 | 0.2525 | 9.8% (over-damped, unstable again) |

**The oscillation drops 19x and uprightness nearly triples.** The mechanism is confirmed. And the
stable region widens - with `DampingScale = 3.0` the body holds **100% upright at 0.05 / 0.075 /
0.10 / 0.125**, where it previously fell from 0.13 up.

Why Godot needs this and Isaac does not: Godot's Stable-PD divides `Kd` by the same denominator it
divides `Kp` by, so its EFFECTIVE damping is a fraction of the authored value, while XPBD's drive is
a positional constraint that is implicitly well damped. `DampingScale` restores what the SPD
reduction removes - a correction of the solver, not a switching-off of the biomechanics.

**Still a statue, though.** Damping bought stability, not a gait: 0 strikes everywhere it stands. The
bistability is now stated exactly: **the policy needs about 0.15 authority to produce a gait, and
Godot is stable only to about 0.125.** Those two numbers do not overlap, which is the whole transfer
failure in one line - and it is a testable target rather than a mystery.


## 2026-09-05 PM (round 6) — ISAAC'S FOOT SINKS INTO THE GROUND. Godot's does not.

The foot contact had never been measured on this project. `dump_obs.py` now writes `footZ_L`,
`footZ_R` and `pelvisZ`, matching the columns `IsaacPolicyDriver.DofTracePath` already wrote, because
the 143-float observation carries only contact FLAGS - a height threshold - which cannot show how a
foot approaches the ground.

The foot box is 0.08 m tall, so a foot resting flat has its COM at 0.040 m. Same policy
(`walk_band/model_30362`), same command:

| | foot below resting height | median penetration | max |
|---|---|---|---|
| **Isaac** | **49.7% of samples** | **14.55 mm** | **39.80 mm** |
| **Godot** | **0.0%** | none | none |

    ISAAC footZ_L   0.040 0.034 0.022 0.017 0.019 0.035 0.040 0.034 0.023 0.016 ...
    GODOT footZ_L   0.050 0.044 0.040 0.040 0.052 0.055 0.045 0.041 0.055 0.051 ...

**Isaac's foot spends half the gait cycle sunk into the floor**, down to 0.017 m - nearly the full
half-depth of the foot box - while Godot's never penetrates at all. The gait the policy learned is
substantially a matter of pushing into a compliant floor, and Godot's floor is rigid.

### Retrained at 240 Hz: Isaac still walks, penetration nearly cut in half, transfer still fails

22 min from `model_30362` at `sim.dt=1/240, decimation=4` -> `walk_hz/model_32175`:

| | cmd 0.30 | cmd 0.50 | penetration |
|---|---|---|---|
| `walk_band/model_30362` (120 Hz) | WALK 5.14 m, 29 strikes | WALK 7.64 m | median 14.55 mm, max 39.80 mm |
| `walk_hz/model_32175` (240 Hz) | **WALK 3.77 m, 35 strikes** | **WALK 4.60 m, 36 strikes** | **median 5.05 mm, max 12.50 mm** |
| Godot | - | - | **0.00 mm** |

**The plant is 2.9x closer and the gait survived.** Godot transfer nonetheless fails, and differently:
FALLEN at 0.10 / 0.125 / 0.15 / 0.175 / 0.20 - the statue region is gone and there is no stable
authority left at all. Closing the ground-compliance gap did not close the transfer gap.

### `restore()` was not replaying the timestep — the first score of this checkpoint was fiction

`TRAINED_CONDITIONS` listed neither `sim.dt` nor `decimation`, so a checkpoint trained at 1/240 was
re-scored at the task default 1/120: **on a floor four times softer than the one it learned on.**
That produced a confident "HOP, 48 strikes, 22.5% flight" verdict which reversed to "WALK, 35
strikes, 5.1% flight" once the fields were added and the same checkpoint ran at its own timestep.
The tell was the penetration reading back at 12.10 mm when 240 Hz should give ~3.

Both fields are now in `TRAINED_CONDITIONS`. They travel together because `dt` and `decimation`
jointly set the POLICY rate, which the frozen contract fixes at 60 Hz. This is the same silent-plant-
invalidation the file warns about, one rung lower than anywhere it had been checked.

### It is a TIMESTEP artefact, and the two engines were never running at the same rate

    Isaac  sim.dt = 1/120 s   (120 Hz)      <- never changed
    Godot  physics_ticks_per_second = 240   <- chosen by scoring, 2026-09-05

Re-measured with `sim.dt = 1/240`, `decimation = 4` (policy still 60 Hz):

    Isaac 120 Hz   penetration median 14.55 mm   max 39.80 mm    verdict WALK
    Isaac 240 Hz   penetration median  3.30 mm   max  8.30 mm    verdict FLAMINGO, 0 strikes

**Penetration falls 4.4x**, and the policy stops walking immediately - because its gait was built on
ground compliance that the harder floor no longer provides. That is exactly the failure it shows in
Godot. `dump_obs.py` gained a `--set` passthrough so plant overrides can be measured without
retraining.

### Contact threshold: a stale constant from the collider fix, now corrected

`IsaacObservation.ContactHeight` was 0.06 against Isaac's `CONTACT_HEIGHT` of 0.05, justified by the
rigs resting their feet at different heights - measured 2026-09-04 as 0.040 m in Godot against
0.017 m in Isaac. **The collider half-size fix killed that justification**: doubling the foot to
0.08 m moved Isaac's planted foot to 0.0391 m, the same as Godot's 0.0395 m, while the thresholds
still differed:

    clearance to read airborne    Isaac 0.05 - 0.0391 = 0.0109 m
                                  Godot 0.06 - 0.0395 = 0.0205 m    <- 1.9x

**Godot needed nearly twice the foot lift to register a swing phase.** At authority 0.10 its foot
lifts to 0.0499 m - under Godot's threshold and over Isaac's - so the flags never flip, the
observation never changes, and the policy freezes. Set to 0.05 to match. Measured effect: the cliff
moves down one step (0.125 STATUE -> FALLEN), confirming the freeze was partly this, but it does not
produce a walk on its own.


## 2026-09-05 PM (round 5) — Godot is BISTABLE: frozen or divergent, never a gait

All measured with `walk_band/model_30362`, the first checkpoint that genuinely walks in Isaac at
cmd 0.30 (29 alternating strikes, 0% flight), so for the first time these are transfer measurements
of a real gait rather than of a flamingo.

### The failure is a cliff, not a gradient

    authority   0.100   0.125   0.130   0.140   0.150
    verdict    STATUE  STATUE  FALLEN  FALLEN  FALLEN

**A 4% change in authority flips a perfectly still body into a falling one**, with nothing between.
Measured action variability explains it:

    GODOT auth 0.10 (statue)   mean |d action / step| 0.0246   per-dof std 0.195
    ISAAC            (walks)                          0.1366                0.479
    GODOT auth 0.15 (falls)                           0.1967                0.603

At 0.10 the policy is effectively FROZEN — 5.6x less action variation than Isaac. A static body
gives a static observation gives a static action: a self-consistent fixed point. At 0.15 it is
WILDER than Isaac. Godot has no operating point in between, which is exactly the signature of a
loop gain that is too low or too high with no stable limit cycle available — and a walk IS a stable
limit cycle.

### What the plant actually gets wrong: joint VELOCITY, not position

Same policy, same command, first 120 steps before Godot's collapse:

    step 1    Godot mean|jv| 1.64  max 9.37   |  Isaac 0.90  max 4.07
    step 40   Godot mean|jv| 1.59  max 11.60  |  Isaac 0.58  max 4.55

Per joint, worst-first, Godot/Isaac ratio of peak velocity:

    TIGHT joints (range <= 0.4 rad)   godot max 6.46   isaac max 1.69   3.83x
    WIDE  joints (range >  0.4 rad)   godot max 5.19   isaac max 2.50   2.08x

Limit bounce is real - tight joints are ~2x worse than wide - but the FEET dominate and
`joint_Foot_R:0` has a wide 1.40 rad range while still pinning the 15.0 clip. **This is ground
contact chatter**, and it is a real motion of the body, not an observation artefact. Joint POSITION
distributions match Isaac's; only velocity diverges.

### Four treatments, all failed

| treatment | result |
|---|---|
| `JointVelocityFilter` 0.50 / 0.25 / 0.15 / 0.08 | FALLEN at 0.15 and STATUE at 0.10 at **every** alpha |
| `SpawnActionNoise` 0.3 / 0.6 | breaks the statue at 0.125 into a FALL; never a walk |
| `LoadCompensation` 1.0 | no change at any authority (round 4) |
| physics rate 120 / 180 / 240 / 360 / 480 Hz | no rate walks; see below |

The filter result matters: this is the **fifth** treatment of the joint-velocity observation channel
and the first run with a matched plant and a real gait, which is the condition under which the
earlier four were said to deserve a re-run. It does not help. **The joint-velocity observation is
eliminated as the blocker.** Filtering the observation cannot help because the excess velocity is
real body motion, not a bad reading.

### Physics rate re-scored against the new policy

The rate was last chosen against the flamingo, and this file's own rule is to score the POLICY:

    120 Hz   FALLEN at 0.10, 0.125, 0.15   (sluggish topple, upright ~30%)
    180 Hz   STATUE 0.10/0.125  -> FALLEN 0.15
    240 Hz   STATUE 0.10/0.125  -> FALLEN 0.15     <- widest stable region
    360 Hz   STATUE 0.10        -> FALLEN 0.125
    480 Hz   FALLEN at 0.10, 0.125, 0.15

**240 Hz is re-confirmed**, now on a real gait. The fall threshold falls monotonically as rate rises
(0.15 -> 0.125 -> below 0.10), which is what a shrinking SPD denominator and rising loop gain
predict. No rate produces a walk.


## 2026-09-05 PM (round 4) — THE GAIT IS FIXED. 15 minutes of finetuning on a narrow command band

The flamingo was not a reward-shape problem. `cmd_curriculum` was already `false` in every walk
lineage, so commands were drawn uniformly from `cmd_lin_vel_x = (-0.3, 1.0)` - including backward -
and the policy simply **chose** which commands to satisfy. It solved the top of the range and stood
still everywhere else, which is a perfectly good local optimum when one network must serve a 4x
speed spread plus reverse.

Fix: narrow the band and make it solve the speed Godot actually uses. Resumed from
`walk_clean/model_28941` with `--set "cmd_lin_vel_x=(0.4,0.7)"`, **15 minutes**, no reward change:

| cmd | verdict | dist (12 s) | vx | strikes | alternation | flight |
|---|---|---|---|---|---|---|
| **0.30** | **WALK** | **5.14 m** | 0.429 | **29** | **100%** | **0.0%** |
| **0.50** | **WALK** | **7.64 m** | 0.637 | **32** | **100%** | **0.3%** |
| 0.60 | FLAMINGO | 0.07 m | 0.006 | 3 | 50% | 0.0% |
| 1.00 | HOP | 6.60 m | 0.550 | 48 | 91% | 21.9% |

Against the same checkpoint before finetuning: cmd 0.30 gave 4 strikes and 0.73 m, cmd 0.60 gave
**one strike and -0.03 m**. **This is the first genuine low-speed gait in the archive** - 29-32
alternating strikes, zero flight, at the exact command the Godot scene sends. The hop survives only
at cmd 1.00, and cmd 0.60 remains a flamingo, so the band is narrow and needs widening.

### It still does not transfer

`model_30362` exported and run in Godot, 20 s:

| cmd | authority | verdict | dist | strikes | upright |
|---|---|---|---|---|---|
| 0.30 | 0.05 | STATUE | -0.01 | 0 | 100% |
| 0.30 | 0.10 | STATUE | -0.01 | 0 | 100% |
| 0.30 | 0.15 | FALLEN | 0.49 | 7 | 11.6% |
| 0.30 | 0.20 | FALLEN | 0.74 | 1 | 7.3% |
| 0.50 | 0.10 | STATUE | -0.01 | 0 | 100% |
| 0.50 | 0.15 | FALLEN | -0.61 | 7 | 10.2% |

`LoadCompensation = 1.0` does not change the outcome at either authority.

**So fixing the gait was necessary and is not sufficient.** The boundary is sharp and sits exactly at
the trained authority: 0.15 IS `action_scale`, and that is where it falls; below it the body does not
move at all. Godot at 0.10 against a contract of 0.15 is a command gain of 0.667, which is **below
the randomised `action_scale_range` the policy trained on** - so the statue and the fall are two
different failures, one out-of-distribution and one at nominal.

### New tool: `scripts/gait_score.py`

Support fraction cannot tell a gait from a statue - 95.3% single support with **one strike in 12 s**
is a flamingo, and that reading is what sent this project after a plant bug for a day. This scores
STRIDE COUNT and distance first and uses support fractions only to classify the failure, adding
`alternation` (fraction of strikes landing on the opposite foot: ~1.0 for a walk, ~0.0 for a
two-footed hop) - which is the quantity `rew_single_support` should have been shaped against.
Verdicts: FALLEN (checked first; a corpse still reports contact fractions), WALK, HOP, FLAMINGO,
STATUE.


## 2026-09-05 PM (round 3) — THE POLICY DOES NOT WALK AT THE COMMAND GODOT SENDS

Godot's walk scene sets `Command = Vector3(0.30, 0, 0)`. Measured in ISAAC, same checkpoint
(`walk_clean/model_28941`), 720 steps, `--no_reset_noise`, forced command:

| cmd | dist | vx | steps | single | double | flight |
|---|---|---|---|---|---|---|
| 0.25 | 0.73 m | 0.061 | 4 | 96.5% | 3.5% | 0% |
| **0.30 (what Godot sends)** | — | — | — | — | — | — |
| **0.60** | **-0.03 m** | **-0.003** | **1** | 95.3% | 4.7% | 0% |
| 1.00 | 9.77 m | 0.814 | 67 | 28.3% | 49.4% | **22.2%** |

**Below ~1.0 the policy stands on ONE LEG.** Near-100% single support with 1-4 foot strikes in 12 s
is a flamingo, not a gait. The recorded "94.3% single support, 0% flight — Isaac genuinely walks"
was reading the statue signature in a new disguise: `rew_single_support` was added to kill the hop
and is maximised perfectly by never putting the second foot down. At the one command where the
policy does locomote (1.00) it has **22.2% flight** — hopping again.

**So Godot has been asked to reproduce a gait that does not exist at that command.** One-legged
balance is also the hardest thing to transfer, which is why it collapses. This is a reward/training
defect, not a plant defect, and it invalidates the premise of the plant work that preceded it.

Godot for comparison, closed loop, authority 0.10: **100% double support, 0 steps, 100% upright at
every command tested (0.30 / 0.60 / 1.00)** - the command barely reaches the behaviour. At authority
0.15 it falls at cmd 0.30 and 0.60, and is a statue at 1.00.

### Open-loop replay, redone properly

Driving Godot from Isaac's genuinely-walking action sequence (cmd 1.00, 67 strikes): **Godot falls**,
height 0.83 -> 0.12, upright 15% at authority 0.10 and 5% at 0.15.

**The replay trace is off by ONE policy step.** `godot[n+1] == isaac[n]` gives max |action diff| of
exactly `0.000000` over 240 steps x 36 DOF, while the naive alignment gives 1.95. The replay is
faithful; the trace logs one step late. Every divergence number taken without that shift is an
artefact — the first version of this analysis reported `Shin_R` diverging at 0.02 s, which was the
lag and nothing else.

Aligned, the ordering inverts and is unambiguous:

    ARMS   Forearm_R:0 0.07s   UpperArm_R:2 0.08s   UpperArm_L:2 0.13s
    LEGS   Shin_R:0    0.23s   Thigh_R:0    0.28s   Shin_L:0     0.57s
    first foot-contact transition: 0.92s

**The arms go first, 3x earlier than the legs, and everything diverges before the first contact
event.** Contact is not the trigger.

### The elbow carries a persistent flexion bias, and `LoadCompensation = 0` caused part of it

    Forearm_R.x     t=0.15  godot +0.358  isaac +0.112
                    t=1.00  godot +0.467  isaac +0.119

Godot's elbow sits 3-4x more flexed than Isaac's from spawn onward — a static bias, not a transient.
Shoulder and legs track. `LoadCompensation` was set to 0.0 to "match Isaac, which has none", but
**Isaac does not need one**: its XPBD drive holds position without sag, so removing Godot's
compensator did not remove a mismatch, it created one. That is the "never switch Godot's
biomechanics off" rule being broken.

Restoring it (`LoadCompensation = 1.0`) cuts open-loop tracking error:

    Forearm_R.x    0.9917 -> 0.6333       UpperArm_L.z   0.5955 -> 0.2530
    Thigh_L.x      0.7556 -> 0.6753

**but does NOT rescue closed-loop transfer** — statue at authority 0.10, falls at 0.15, either way.
Real improvement, not the blocker.

### Two more measurements

- **Arm step response:** Godot `UpperArm_L:0` rises in 266.7 ms vs Isaac's 150.0 ms (1.78x slower),
  against 1.25x for the thigh — the per-bone SPD spread is real and worse on the arms. Isaac's arm
  probe carried NOT-SETTLED and env-reset warnings, so treat as directional.
- **The "joint velocities PINNED at 15.0" entry is RETIRED.** Godot's joint velocity now reads
  median 0.000, p95 0.083, **0% saturation** — `JointVelocityFromDifference` fixed that channel.
  Godot's joint POSITION distribution matches Isaac's (median 0.039, p95 0.300 vs 0.277); only
  velocity differs (0.000 vs 0.452 median), and that is because Godot is standing still.


## 2026-09-05 12:10 — CORRECTION: Godot is NOT sublinear. The gap is a LINEAR gain of 2.47x

The "Godot's actuator saturates" section below is **wrong** and is kept only as a record of how the
error was made. Fitting the sweep instead of dividing by the target:

```
Isaac:  delivered = 2.6424 * target + 0.0031     R2 = 0.996155
Godot:  delivered = 1.0681 * target + 0.0455     R2 = 0.999998
```

**Godot is linear to six digits.** The apparent 1.65x -> 1.35x -> 1.21x "falloff" is a division
artifact: a CONSTANT offset of +0.0455 rad divided by a growing target produces a decaying ratio all
by itself. No nonlinearity was ever present.

That retro-explains the two falsifications above it. The effort ceiling (x3) and the Hill law
(vmax x100) both returned byte-identical numbers because they are nonlinearities, and there was no
nonlinearity to find. Those were correct answers to a malformed question. The DAMPING hypothesis
dies the same way: damping-as-a-nonlinearity was invented to explain a curve that is a straight line.

**What is actually true, and it is simpler:**

| | slope (delivered per unit target) | offset |
|---|---|---|
| Isaac | **2.64** - overshoots its command | ~0 |
| Godot | **1.07** - tracks its command | +0.0455 rad |

Ratio **2.474**, which independently reproduces the 2.4x lift deficit measured from the gait. Two
unrelated measurements now agree on one number, and Godot is the CORRECT one - a position drive that
delivers the angle it was asked for. Isaac is the broken side: it overshoots every command 2.6x.

### THE ACTUATOR IS NOT THE MECHANISM — step response measured on both, 2026-09-05 PM

The contradiction: `assets.py` records the two plants matching exactly on a single-joint step, while
the whole-body replay reports 2.64x vs 1.07x. Both cannot be true. Re-measured on
`joint_Thigh_L:1`, 0.15 rad step, `P4F_XPBD_ITERATIONS=8`, free root, gravity off:

| | rise to 63.2% | overshoot | settled fraction |
|---|---|---|---|
| Isaac (`probe_plant.py`) | 33.3 ms | **+2.5%** | 1.014 |
| Godot (`Probe/IsaacPlantProbe.tscn`) | 41.7 ms | **+0.0%** | 1.000 |

**A 2.5% joint overshoot cannot produce a 264% whole-body overshoot.** The joint drives agree to a
few percent, so the 2.64x is NOT a property of the actuator, and every actuator-side fix - SPD gain
matching, effort clamp, Hill law, target damping, the per-bone `godot_plant.py` table - is aimed at
a subsystem that is already matched. Redirect to support state, contact and the multi-body chain.

Godot is also insensitive to load on this DOF - pinned pelvis, gravity on, feed-forward on or off,
all three give 41.7 ms / +0.0% / 1.000, identical to five decimals.

**Two harness defects found while doing it, both of which have silently weakened earlier results:**

1. **`--gravity` on either probe does not load the joint.** Neither probe scene has a floor, so with
   gravity on the body is in FREE FALL, and a free-falling body has no gravitational load on its
   joints. Godot gravity-ON vs gravity-OFF agreed to five decimals for exactly this reason. Use
   `FreezePelvis` (Godot) or pin the root (Isaac) to get a genuinely loaded limb - and note that
   `joint_Thigh_L:1` is the ABDUCTION axis, about which a hanging leg exerts almost no gravity
   torque, so it is the wrong DOF for a load test regardless. Use axis 0 (flexion).
2. **`assets.py`'s calibration entry is STALE.** Its "Godot 33.3 ms, rise matches EXACTLY" was taken
   at 120 Hz; the project has run at 240 Hz since, where Godot reads 41.7 ms. The conclusion that
   both calibration factors are 1.0 rests on a measurement the current configuration no longer
   reproduces.

Mass and inertia are NOT the divergence either: `build_d6_usd.py` writes explicit `CreateMassAttr`
per body, and both engines total 80.6 kg over 16 bodies, with inertia derived from colliders that
now match after the half-size fix.

**The open-loop replay is now doubly suspect** and should not be quoted again until re-run: its
Godot side was in a different support state (see below), and its Isaac side moved the pelvis 0.375 m
in 2.5 s with the feet off the ground ~45% of the time, which is as consistent with a body FALLING
as with one striding. Gate it on support state and check the pelvis height trend before drawing
anything from it.

### Action -> target mapping: VERIFIED IDENTICAL

Re-checked because the joint-limit inversion changed 32 limit values, and any range-based mapping
would have moved with them.

    Godot  IsaacActionSpace.cs:264   span = a >= 0 ? Upper : -Lower;  target = ActionScale * a * span
    Isaac  stand_env.py:291          span = where(a >= 0, upper - default, default - lower)
                                     target = default + action_scale * a * span

Isaac's `init_state.joint_pos` is `{".*": 0.0}`, so `default = 0` on every joint and the two
formulas collapse to the same expression. **The mapping matches.** Worth noting anyway:
`IsaacActionSpace`'s own class doc states Isaac's rest-pose-relative form while the code implements
the zero-relative one, and `IsaacRigContract.JointSpec` carries no `Default` field at all. Harmless
today, wrong the moment a non-zero rest pose is introduced.

### Godot's balance layer: NOT the blocker either

`BalanceJointModules = false` had been tested before, but it deliberately KEEPS pelvis stabilisation
("the one module Isaac reproduces"), and the walk scorer never set it at all - so every score in
this file, the authority ladder above included, ran with `BalanceAssist = 1.0` and the full
procedural stationary-balance layer live. That layer exists to hold a stationary stance, which is
the opposite of what a gait needs, so it was a strong candidate. `walk_clean/model_28941`,
authority 0.15, 20 s:

| condition | upright dist (m) | steps | upright |
|---|---|---|---|
| A baseline, full balance layer | 0.47 | 6 | 11.8% |
| B `BalanceJointModules = false` (pelvis only) | -0.92 | 7 | 8.2% |
| C **`BalanceAssist = 0` - no balance layer at all** | 0.52 | 3 | **11.5%** |
| D B + `PelvisReactIntoThighs` | 0.51 | 0 | 7.7% |

**Flat across all four.** Removing Godot's balance layer entirely changes nothing, so the procedural
controller is not what is fighting the policy. Eliminated.

### The replay's "Godot does not reach its target" is CONFOUNDED

The 06:35 section reads `Godot hip 0.382 vs a 0.315 target -> tracks; Isaac 0.833 -> overshoots 2.6x`
and concludes Isaac's overshoot is what lifts the feet. The two columns beside it were not used:

    Godot   contacts 100% / 100%    pelvis range 0.019 m
    Isaac   contacts  58% /  51%    pelvis range 0.375 m

Godot's feet never left the floor and its pelvis moved 19 mm in 2.5 s. A hip whose foot is planted
and whose pelvis is fixed is a closed kinematic chain against the ground - it **cannot** swing to
0.83 rad no matter what torque is applied, so "Godot only reached 0.382" measures the constraint,
not the actuator. The causal claim runs at least as well the other way: Godot's feet stay down,
therefore the hip cannot swing. **A hip-angle comparison is only meaningful between two bodies in
the same support state**, which these were not. Re-run the replay gating on swing phase before
quoting either number.

### Raising Godot's authority to compensate: TESTED, FAILED

If the mismatch is a linear gain of 2.474, then Godot at `action_scale = 0.15 x 2.474 = 0.371`
delivers exactly the motion the policy trained on, with no retraining. The ladder had never been run
above 0.15. It has now, on `walk_clean/model_28941`:

| authority | 0.15 | 0.20 | 0.25 | 0.30 | 0.37 | 0.45 |
|---|---|---|---|---|---|---|
| upright | **11.8%** | 7.3% | 7.5% | 6.3% | 5.4% | 5.9% |
| upright dist (m) | 0.47 | 0.06 | 0.02 | 0.03 | 0.01 | -0.02 |

Monotonically worse. **The reason is that 2.474 is an OPEN-LOOP gain match and the policy is a
closed loop.** Scaling the action vector scales the policy's feedback gain by the same 2.474, and a
balance controller at 2.5x loop gain is unstable by construction. Open-loop equivalence and
closed-loop stability are different requirements and this number only satisfies the first.

**Therefore the correction belongs in Isaac's plant, not in Godot's action scale**, which is the
"match the muscle, never switch Godot's biomechanics off" rule. Isaac must be made to TRACK its
targets (slope -> ~1.07) and the gait retrained on that plant. Then the policy's loop gain is right
in both engines because both plants have the same gain.

## 2026-09-05 11:00 — the gap is NOT one number: Godot's actuator SATURATES, Isaac's is linear

Open-loop replay, identical scripted actions into both engines, hip target = `amp x 0.15 x 2.1`:

| action amp | hip target | **Isaac delivered** | **Godot delivered** |
|---|---|---|---|
| 0.25 | 0.079 | 0.197 (**2.50x**) | 0.130 (**1.65x**) |
| 0.50 | 0.158 | 0.443 (**2.81x**) | 0.214 (**1.36x**) |
| 1.00 | 0.315 | 0.828 (**2.63x**) | 0.382 (**1.21x**) |

**Isaac overshoots by a constant ~2.6x at every amplitude. Godot's response is SUBLINEAR** - 1.65x
at a quarter command falling to 1.21x at full. The Isaac/Godot ratio therefore grows with amplitude,
1.5x -> 2.2x, and **no single `action_scale` can match delivered motion across the range.**

**Tested and it failed for exactly this reason.** Training Isaac at `action_scale = 0.0679`
(the 2.21x mean ratio) to make its delivered motion match Godot's: after 25 minutes Isaac itself
stopped walking (0 foot strikes, hip max 0.126 rad against 0.515 at scale 0.15) and Godot scored
3.7% upright at authority 0.15. A scalar calibration cannot work against a nonlinear plant, and the
25-minute budget was also too short to re-learn a gait at a new scale - both are true and the first
one is the reason to stop pursuing it.

### The sublinearity is NOT the effort ceiling and NOT the Hill law — both falsified

| intervention | amp 0.25 | amp 1.00 | verified applied? |
|---|---|---|---|
| baseline | 1.65x | 1.21x | - |
| **effort ceiling x3** (`EffortScale = 3.0`) | **1.65x** | **1.21x** | yes — Spine maxTorque 350 -> 1050, 12 bones |
| **Hill vmax x100** (`HillVmaxScale = 100`) | **1.65x** | **1.21x** | yes — logged, 12 bones |

Byte-identical in both cases. Tripling the torque ceiling and removing the force-velocity derating
change nothing about how far the joint travels.

**Note the trap avoided:** `--set DisableHillLimit=true` also produced identical numbers, and that
one was inert - `DisableHillLimit` does not exist as an export, only as a stale reference in
`IsaacPolicyDriver`'s own comments, so `--set` appended a property Godot ignores. Two knobs were
added (`EffortScale`, `HillVmaxScale`) precisely so the test could be verified rather than assumed.

**What is left, and it is the original hypothesis:** DAMPING. Godot's Stable-PD applies a real `kd`
term, Isaac's XPBD drive effectively applies none. Damping torque grows with joint velocity, and
velocity grows with commanded amplitude - which produces exactly the observed shape: Godot's
overshoot shrinks from 1.65x to 1.21x as the command grows, while Isaac's stays flat at ~2.6x
because nothing is resisting it. `target_damping` tested at 1.0 moved Isaac 2.6x -> 2.0x and at 3.0
made it worse, so the target-space approximation is directionally right and quantitatively wrong.

**Superseded guess (kept as a record):** the effort ceiling. Godot clamps `totalTorque.Length()`
per bone, so a larger commanded deflection asks for more torque than the ceiling allows and the
delivered angle falls behind proportionally. That is testable: sweep the effort limit in Godot and
see whether the 1.65 -> 1.21 falloff flattens. If it does, the fix is to make Isaac's
`_effort_limited` bind the same way rather than to rescale actions.

## 2026-09-05 06:35 — THE GAP IN ONE NUMBER: Isaac OVERSHOOTS its joint targets 3x; Godot tracks them

Measured with `scripts/replay_isaac.py` (new — Isaac's mirror of Godot's `ReplayActionsPath`), so
for the first time the SAME recorded action sequence drives both engines with no policy in either
loop. 150 steps of a scripted gait, authority 0.15:

| | hip range | knee range | contacts L/R | pelvis range |
|---|---|---|---|---|
| **Godot** | [-0.043, **+0.382**] | [**-0.471**, +0.043] | **100% / 100%** | 0.019 m |
| **Isaac** | [-0.100, **+0.927**] | [**-1.160**, +0.122] | 58% / 51% | 0.375 m |

The commanded target for the hip is `action 1.0 x action_scale 0.15 x span 2.1 = 0.315 rad`, and for
the knee `-0.39 rad`.

    GODOT  hip 0.382 against a 0.315 target   -> tracks it (1.2x)
    ISAAC  hip 0.833 against a 0.315 target   -> OVERSHOOTS it (2.6x)
    GODOT  knee -0.471 against -0.39          -> 1.2x
    ISAAC  knee -1.054 against -0.39          -> 2.7x

(Isaac's figures are with `action_scale_range` and `effort_scale_range` PINNED at 1.0. The first
run of this A/B left them at their randomised defaults, which inflated Isaac's excursions by about
12% - `replay_isaac.py` now pins both, because a plant comparison must not carry randomisation on
one side only.)

**Isaac's joints fly past their commanded angles by a factor of three, and that overshoot is what
lifts the feet.** Godot's Stable-PD tracks its targets, so its feet stay down (contacts 100%/100%
through the whole sequence) and the gait never happens.

**This inverts the assumption behind most of this file.** The sag measurements are real - Godot is
compliant under STATIC load - but in the DYNAMIC regime that matters for walking the deficit runs
the other way: Godot is the well-behaved tracker and Isaac is wildly under-damped. Every attempt to
close the gap by stiffening Godot was pushing the wrong side.

### Neither damping nor solver iterations closes it

| intervention | Isaac hip max | vs Godot 0.382 |
|---|---|---|
| baseline (8 iterations) | 0.833 | 2.2x |
| `target_damping = 1.0` | 0.771 | 2.0x |
| `target_damping = 3.0` | 1.024 | WORSE |
| 16 XPBD iterations | 0.845 | 2.2x |
| 32 XPBD iterations | 0.929 | WORSE |

Both are non-monotonic and neither converges on Godot. **The overshoot is momentum the XPBD
projection does not remove, and neither damping the target nor spending more solver work takes it
back out.** That is the open question to start from.

### Target-space damping: tried, only partial. Left at 0.0.

`target_damping` subtracts `kd*qd/kp` from the commanded target, which is algebraically the damping
term of a PD law and is the only route available (solver damping is inert). Swept on the same
open-loop replay:

| `target_damping` | hip range | knee range | foot contact |
|---|---|---|---|
| 0.0 | [-0.090, **+0.946**] | [-1.164, +0.073] | 61.3% |
| **1.0** | [-0.061, **+0.771**] | [-0.967, +0.071] | 66.0% |
| 3.0 | [-0.043, **+1.024**] | [-1.199, +0.039] | 63.3% |
| **GODOT** | [-0.043, **+0.382**] | [-0.471, +0.043] | **100%** |

Godot's damping (1.0) cuts the overshoot from 2.9x to 2.4x of target — real but nowhere near
Godot's 1.2x — and 3.0 makes it WORSE, which says the response is not simply under-damped. The
overshoot is ballistic: the limb carries momentum that XPBD's position projection does not remove
within its iteration budget. Damping the target cannot take that back out.

Left at 0.0 (default) since it does not reach the goal. The knob is kept and registered in
`TRAINED_CONDITIONS` because the axis is right even if this implementation is not sufficient.

**What this suggests trying first:** the overshoot is momentum XPBD does not absorb, so the lever is
the SOLVER's ability to hold a joint, not the drive's damping. Raise `P4F_XPBD_ITERATIONS` above 8
and re-run the open-loop table above - iterations are the one XPBD knob measured to change delivered
dynamics (overshoot +10.5% at 2 iterations against +0.4% at 8), and this is the first question that
has pointed squarely at them.


## 2026-09-05: EVERY BOX COLLIDER IN ISAAC WAS HALF SIZE

`isaac_lab/scripts/build_d6_usd.py` defined each box collider as a `UsdGeom.Cube` with
`size = 1.0` and then applied `scale = (sz/2, sx/2, sy/2)`. A unit cube spans -0.5..+0.5, so a scale
of `k` gives a side of `k`, not `2k`. Every box was therefore exactly **half** its authored size:

| body | Godot (authored) | expected in USD | **was in USD** |
|---|---|---|---|
| Pelvis | 0.36 x 0.20 x 0.24 | (0.24, 0.36, 0.20) | **(0.12, 0.18, 0.10)** |
| Chest | 0.42 x 0.30 x 0.28 | (0.28, 0.42, 0.30) | **(0.14, 0.21, 0.15)** |
| **Foot** | 0.12 x 0.08 x 0.22 | (0.22, 0.12, 0.08) | **(0.11, 0.06, 0.04)** |
| Hand | 0.08 x 0.08 x 0.12 | (0.12, 0.08, 0.08) | **(0.06, 0.04, 0.04)** |

**Isaac trained every policy this project has produced on feet half as long as Godot's** - half the
fore-aft support base, which is the single most important parameter a biped balances on.

Capsules and the sphere were CORRECT (they take radius and height directly), which is why this
survived every rig check: the limbs matched, and only the four boxes were wrong. Masses were also
correct, being set separately - so total mass agreed at 80.6 kg and nothing looked amiss.

**Fixed** by scaling to `(sz, sx, sy)`. USD rebuilt and verified: foot now `(0.22, 0.12, 0.08)`.

Two things had to be protected while rebuilding, and both were:
* The builder regenerates `dummy_d6_rig.json` FROM the Godot scene, and the scene's limit numbers are
  now Jolt-inverted (see the joint-limit section below). Building from the current scene would have
  written mirrored limits into Isaac and undone that fix. The USD was rebuilt from the PRE-inversion
  scene, so the rig keeps anatomical limits - verified `joint_Shin_L:0 = [-2.6, +0.1]` after.
* `build_d6_usd.py` needed Kit only for `PhysxSchema`; it now imports `pxr` standalone and skips the
  PhysX-specific articulation API (the Newton backend does not consume it), so it runs from the
  Newton environment.

**Consequence: every existing checkpoint was trained on a different body and is invalid.**
`CONTACT_HEIGHT` also had to be recalibrated - Isaac's planted foot moved from 0.0081 to 0.0286 m,
and the threshold is now 0.05 to give the same 0.020 m clearance Godot has.

### Geometry fix: correct on first principles, not yet shown to help transfer

Open-loop divergence measured after the rebuild looks WORSE (t=1.0s RMS 0.153 -> 0.451, Godot
falling by t=2s), **but the comparison is confounded**: the "before" trace used `model_18050`'s
actions and the "after" used a policy that has had 30 minutes to adapt to a body that changed
underneath it. It is not an A/B of the geometry.

A clean A/B would need the same action sequence driven open-loop into Isaac with the old and new
USD, and there is no Isaac-side open-loop replay tool - only Godot has `ReplayActionsPath`. Building
one is the honest way to settle it if the fragment loop does not.

The fix stands regardless of that number: the collider genuinely should match Godot's authored size,
and it did not. First two fragments on the corrected body score 7.2% upright at authority 0.15 with
displacement rising to +0.73 m - it travels further and falls, where the old body's best stood still.

### Re-measured on the corrected geometry (2026-09-05 06:15)

Zero action, settled, total |joint deviation| across 45 DOFs:

| | assist 0.0 | assist 1.0 |
|---|---|---|
| Isaac, half-size feet | 0.512 | 0.390 |
| **Isaac, corrected feet** | **0.282** | **0.535** |
| Godot (unchanged) | 0.640 | 1.326 |

Isaac still barely sags on the corrected body (`Forearm_L:0` -0.001 against Godot's +0.353), so the
**sag mismatch is the DRIVE, not the collider** - consistent with `write_joint_stiffness_to_sim`
being mechanically inert here. The geometry fix does not close it and was never going to.

**Every plant measurement recorded above the geometry section was taken with half-size feet** and
should be re-taken before being relied on again: the step response, the sag table, the divergence
onsets and the 240 Hz comparison.

### `rew_single_support` was gameable — it paid for standing on ONE LEG

Measured 2026-09-05 on the corrected body, the lineage had converged to:

    ISAAC   double 6.2%   SINGLE 93.8%   flight 0.0%   vx +0.00 m/s

93.8% single support with **zero forward velocity**. The term was gated on a command being PRESENT,
not on progress being made, so holding one foot in the air forever collects it maximally - the
cheapest possible way to satisfy "exactly one foot down".

Fixed by multiplying by `drive` (achieved-over-commanded speed, in [0,1]) instead of the command
gate, so single support only pays as part of actual locomotion.

**This is the same class of mistake as `feet_air_time` paying for a hop.** Both terms described the
SHAPE of walking without requiring the walking, and a policy will always find the cheapest way to
make the shape. Check the achieved velocity alongside any gait-shape reward.

## THE MECHANISM: Godot's compliant legs make a 1 Hz POGO that lifts both feet

The same policy that walks in Isaac **hops** in Godot. Measured at authority 0.10, upright window:

| | double | SINGLE | flight | feet in same state | corr(footZ_L, footZ_R) |
|---|---|---|---|---|---|
| **Godot** | 72.8% | **5.5%** | 21.7% | **94.5%** | **+0.942** |
| **Isaac** | 5.7% | **94.3%** | 0.0% | - | - |

Both Godot feet rise and fall TOGETHER (correlation +0.94). The cause is a whole-body vertical mode:

| | pelvis vertical range | bounce frequency |
|---|---|---|
| **Godot 120 Hz** | **0.227 m** | 1.04 Hz |
| Isaac | 0.075 m | 1.60 Hz |

**Compliant legs plus an 80 kg body are a spring-mass oscillator at about 1 Hz.** Isaac's joints are
rigid (see the sag table) and have no such mode. The policy's gait excites Godot's mode, the bounce
lifts both feet at once, and a walk becomes a hop. This is the same defect the POLICY had before the
reward fix - now it is the BODY doing it.

**Confirmed by stiffening.** Raising Godot's physics rate raises the effective gains and should kill
the mode. It does, completely:

| Godot rate | pelvis range | bounce | single support | flight | upright samples (15 s run) |
|---|---|---|---|---|---|
| 120 Hz | 0.227 m | 1.04 Hz | 5.5% | 21.7% | 290 |
| **240 Hz** | **0.002 m** | 0.14 Hz | 0.0% | 0.0% | **840 (full run)** |
| 480 Hz | 0.001 m | 0.07 Hz | 0.0% | 0.0% | 840 |

227 mm of bounce to 2 mm, and the dummy survives the whole run instead of a third of it.

### Balance assist trades stability for stepping — no setting gives both

At 180 Hz, where the body is otherwise stable, sweeping Godot's `BalanceAssist` (its effect on Godot
is much larger than Isaac's: 0.640 -> 1.326 total deviation when enabled, against Isaac's
0.512 -> 0.390):

| assist | authority | upright | single support | steps | upright displacement |
|---|---|---|---|---|---|
| 1.0 | 0.10 | **100%** | 0.0% | **0** | -0.01 m |
| 1.0 | 0.15 | 10.3% | 0.0% | 0 | -0.24 m |
| 0.6 | 0.15 | 15.7% | 3.7% | 1 | +0.47 m |
| **0.3** | **0.15** | 17.1% | 0.9% | **5** | **+0.57 m** |
| 0.0 | 0.15 | 8.7% | 0.4% | 0 | -0.51 m |

**Every configuration is either upright and motionless or moving and falling.** Assist 0.3 at
authority 0.15 is the closest to walking this project has produced - 5 foot strikes and 0.57 m
covered while still standing - and it only holds its feet for 17% of the run.

The pattern holds across every knob tried tonight: physics rate, authority, assist, load
compensation, randomisation width. There is no setting that produces motion AND stability, which is
what a gait is.

### Physics rate: 240 Hz. Settled by scoring the POLICY, not the statics.

The rate was changed three times tonight. Each move was on better evidence than the last, and the
sequence is worth keeping because it shows which measurements mislead:

1. **120 -> 240** on halved static sag and the trained authority surviving. Both real, both static.
2. **240 -> 120** on the open-loop replay tracking longer at 120 Hz. Real, but measured while both
   policies HOPPED - at 120 Hz Godot's pogo follows a hopping action sequence better.
3. **120 -> 180** on 180 killing the pogo as completely as 240 (0.004 m either way) for 1.5x cost.
   Real, but again a static/uprightness measurement.
4. **180 -> 240, final.** Direct A/B on the same checkpoint, scored on locomotion:

| rate | authority | upright displacement | steps | upright |
|---|---|---|---|---|
| 180 Hz | 0.15 | **-0.29 m** | 3 | 9.2% |
| **240 Hz** | 0.15 | **+0.47 m** | **6** | 11.8% |
| either | 0.10 | -0.04 m | 0 | 100% |

**Score the policy, not the body.** Pogo amplitude, uprightness and sag all said 180 ≡ 240; the
locomotion score says otherwise, and it is the one that matters.

**150 Hz is invalid** regardless: `PhysicsTicksPerPolicyStep` is derived as rate/60, and 150/60 = 2.5
is not an integer, so the policy stops running at its contracted 60 Hz. It falls instantly, 0%
upright. Valid rates are 120, 180, 240, 300, 360, 480.

### Physics rate must be an INTEGER MULTIPLE OF 60

| Godot rate | upright | pelvis range | single support | flight | steps |
|---|---|---|---|---|---|
| 120 Hz | 63.8% | 0.235 m | 3.5% | 10.3% | 4 |
| **150 Hz** | **0%** | - | - | - | - |
| **180 Hz** | **93.3%** | **0.004 m** | 0.0% | 0.0% | 0 |
| 240 Hz | 93.3% | 0.004 m | 0.0% | 0.0% | 0 |

**150 Hz falls instantly** because `PhysicsTicksPerPolicyStep` is derived as rate/60 and 150/60 = 2.5
is not an integer - the policy stops running at the 60 Hz its contract requires. Only 120, 180, 240,
300, 360, 480 are valid.

**180 Hz kills the pogo as completely as 240** (0.004 m either way, identical uprightness) for 1.5x
the physics cost instead of 2x. `project.godot` set to 180.

## 240 Hz reinstated — the earlier reversal was reasoning from a hopping policy

`physics_ticks_per_second` is back to 240. The earlier revert to 120 was based on the open-loop
replay tracking longer at 120 Hz; that measurement was taken while BOTH engines' policies hopped,
and at 120 Hz Godot's pogo happens to follow a hopping action sequence better. It was a real number
about the wrong regime.

What 240 Hz buys, measured: the spurious 1 Hz vertical mode that Isaac does not have is gone, rest
chatter matches Isaac (0.004 against 0.006 rad/s), and the body stays upright for a full run. What
it costs: 2x Godot physics compute, and the static sag is worse - which is a static measure, and
static similarity has already been shown here not to predict transfer.

**It still does not walk.** At 240 Hz the policy's true commanded behaviour in Godot is visible
without the bounce masking it, and that behaviour is a static posture: 0.0% single support, 0.0%
flight, no displacement. The remaining problem is the observation fixed point, not the body.

## THE BARRIER, MEASURED: a 2.4x SWING-FOOT LIFT DEFICIT

Foot world height added to the Godot trace (`footZ_L`, `footZ_R`, `pelvisZ`), because the contact
flag is a THRESHOLD on it and a flag that never fires cannot distinguish "the foot is not rising"
from "the foot rises but not far enough". Policy in both engines, Godot's upright window only:

| | planted footZ | peak | **swing lift** | single support |
|---|---|---|---|---|
| **Godot** | 0.0400 | 0.0650 | **0.025 m** | 2.3% |
| **Isaac** | 0.0318 | 0.0911 | **0.059 m** | 94.3% |

**Godot lifts its swing foot 2.5 cm where Isaac lifts 5.9 cm**, and Godot's contact threshold needs
2.0 cm. It sits right on the edge, which is why single support flickers at 2.3% rather than being
flat zero. The thresholds themselves are fair - 2.0 cm of clearance in Godot against 1.8 cm in
Isaac - so this is a real lift deficit, not a measurement artefact.

**The stance leg is NOT the cause.** Pelvis height is stable through the scripted gait (range
0.023 m, correlation with foot height -0.016), so the lift is not being cancelled by the supporting
leg sinking.

**An earlier version of this section claimed a 6x deficit and zero lift.** That came from comparing
Isaac's full policy against a SCRIPTED Godot gait that drove only hip and knee and left the ankle
undriven - not the same experiment. With the policy in both engines the deficit is 2.4x.

## Superseded: the scripted-gait comparison

After the reward fix Isaac genuinely walks: **94.3% single support, 0% flight, 36 foot strikes,
+0.59 m/s** under a 0.30 m/s command. The same checkpoint in Godot produces **ZERO foot strikes** at
authority 0.10, 0.15 and 0.20.

Driving Godot with a SCRIPTED gait at maximum command removes the policy from the question entirely:

| | hip max | hip sd | knee max | knee sd | foot strikes |
|---|---|---|---|---|---|
| **Isaac, walking** | 0.515 | 0.100 | 0.570 | 0.202 | **36** |
| **Godot, max scripted cmd @0.15** | 0.385 | 0.129 | 0.535 | 0.199 | **0** |
| Godot, max scripted cmd @0.30 | 0.728 | 0.253 | 1.320 | 0.365 | 2 |

**Godot swings its knee as far as Isaac does (0.535 against 0.570) and the foot barely leaves the
ground.** The leg moves; the contact does not break. Godot needs roughly TWICE the joint excursion
before a foot lifts at all.

That rules out, by measurement, everything on the actuator side: gains, effort ceilings, the Hill
law, action scale and command amplitude all produce the motion. What is left is how the body's
weight resolves onto the feet - weight transfer, the stance leg, or the contact itself.

**This is the question to start from next:** with comparable leg motion in both engines, why does
Godot's swing foot stay loaded? Instrument the FOOT HEIGHT and the per-foot contact force in Godot
(the trace currently carries only the height-derived contact flag and joint angles), and compare
against Isaac's foot height over a stride. That measurement does not exist yet and every remaining
hypothesis needs it.

---

## 2026-09-04: Godot's joint limits were MIRRORED. The dummy could not bend its knees.

**This is the defect the whole "sim-to-sim gap" has been.** Godot runs Jolt Physics, and Jolt
enforces `Generic6DOFJoint3D` angular limits with the opposite sign to the scene declaration. The
rule, measured on every asymmetric axis in the rig:

    physical range = [-declared_upper, -declared_lower]

Measured in `Scenes/RL/Isaac3/Probe`, gravity off, pelvis frozen, driving `TargetLocalRotation`
directly so nothing in the RL path is involved:

| joint | declared | Godot ENFORCED (before) |
|---|---|---|
| `joint_Shin_L:0` (knee) | [-2.60, +0.10] | [-0.10, **+2.60**] |
| `joint_Thigh_L:0` (hip) | [-0.50, +2.10] | [-2.10, **+0.50**] |
| `joint_Foot_L:0` (ankle) | [-0.80, +0.60] | [-0.60, **+0.80**] |
| `joint_UpperArm_L:0` | [-1.00, +3.00] | [-3.00, **+1.00**] |

Commanding the knee to -2.0 rad parked it at **-0.100005**, and -0.5 parked it at **-0.100008** -
the same stop, which is the mirror of the declared `upper` of +0.1.

The policy flexes the knee by commanding **-0.39 rad** (`action_scale 0.15 x action -1 x span 2.6`,
and Isaac's own mapping is identical). Godot blocked it at -0.1. **The dummy physically could not
bend its knees in the direction a gait requires**, and its hip mainly extended backwards - a bird's
leg. Anatomically the DECLARED limits are the correct ones (hip flexes forward 2.1, knee folds back
2.6), and Isaac walks with them at 0.83 m/s, so Godot's mechanism was the wrong one.

**Why it hid for so long, and why it looked like a physics-fidelity problem:** at a fixed point the
knee sits near 0 and never reaches a limit, so Stand transferred perfectly and every probe taken at
rest agreed between the engines. Only a gait visits the limit. That is the same asymmetry that made
the Hill filter, the SPD gain and the joint-velocity channel all look like candidates - every one of
them is also inert while standing.

**The fix** rewrites all 16 asymmetric axis pairs in `Scenes/ActiveRagdoll.tscn` as
`new_lower = -old_upper`, `new_upper = -old_lower`, so the PHYSICAL range equals the declared one.
Verified afterwards on six axes across three joints: knee stops at +0.100006, hip at -0.500001,
ankle at -0.800005/+0.600004, `Thigh_L:2` at [-0.200002, +0.800010], `UpperArm_R:2` at
[-2.50035, +0.50001]. Symmetric axes are untouched.

**Measured effect.** Open-loop replay of Isaac's own recorded action sequence, no policy in the
loop, so this is the plant and nothing else:

| | joint-trajectory RMS vs Isaac at t=1.5s | Godot pelvis height at t=2.0s |
|---|---|---|
| mirrored limits | 0.859 rad | 0.174 m (collapsed) |
| corrected limits | **0.315 rad** | **0.803 m (upright)** |

Closed loop, walk checkpoint `walk_spd/model_18050`: authority 0.05 went from **falling at 16 s to
standing the full 20 s**. Authority 0.10 and 0.15 still fall, and now fall FASTER - with knees that
actually flex, the same commands finally move the legs, and the body cannot yet control what it has
been given. That is progress with an unfinished second half, not a fix that failed.

---

## 2026-09-04: Godot's gravity feed-forward is 46-68% of its joint torque; Isaac has none

Godot's `LoadCompensation` supplies **46-68% of delivered joint torque** (`ffShare` in the driver
log). Isaac's `gravity_feedforward` defaults to 0.0 and every checkpoint trained without it, so the
policy learned to produce that holding torque itself and then met a body that was already producing
it - an over-actuated dummy that gets thrown.

Turning it OFF in Godot, which is the direction that matches the reference the policy learned from:

| authority | LoadCompensation 1.0 | LoadCompensation 0.0 |
|---|---|---|
| 0.05 | stands | stands |
| 0.10 | **FELL** | **STANDS** |
| 0.15 | fell | fell |

Combined with the limit fix the ladder went from "stands only at 0.03" to **standing at 0.10**.

The alternative - implementing the feed-forward faithfully in Isaac so Godot can keep its
biomechanics - is still the better long-term answer, and still gated on `probe_ff.py` leg ratios
near 1.0. Enabling the current unfaithful version costs 52.7% -> 0.0% standing.

## The ladder was measuring the wrong thing

**A statue passes an authority ladder.** Scoring the promoted walk checkpoint on the corrected rig:

| | displacement, 20 s | foot strikes | flight | upright |
|---|---|---|---|---|
| Godot, authority 0.10 | **-0.05 m** | 2 | 0.2% | 100% |
| Isaac, same policy, command 0.30 | **+10.31 m** | 38 | 45.8% | - |

It survives and does not walk. Every checkpoint this project has promoted was selected on survival,
and `scripts/godot_walk_score.py` now reports displacement, strikes and flight fraction instead so
the number cannot be satisfied by standing still.

Re-ranking two lineages on the CORRECTED rig (`walk_spd/model_18050`, `walk_hill/model_19600`) put
both at ~0 displacement and 0-2 strikes at authority 0.10, and falling at 0.15. So the remaining gap
is no longer "the body cannot do it" - it is that the policy's Isaac gait is a **bounding run with a
45.8% flight phase**, produced because the command curriculum ran its ceiling to the configured
1.0 m/s while Godot asks for 0.30. A ballistic gait is the least transferable kind.

## The corrected rig can step - confirmed directly

Driving a scripted alternating hip/knee flexion open-loop, no policy, no balance:

    Thigh_L.x reached +2.274 rad  (declared upper +2.10)
    Shin_L.x  reached -2.195 rad  (declared lower -2.60)
    4 clean left-foot lift-offs

On the mirrored rig the knee could not pass -0.1. The mechanism is now capable of a gait; it falls
in this test only because a scripted pattern carries no balance.

## The plant is now MATCHED. Measured, open loop.

Driving Godot from Isaac's own recorded action sequence - no policy, no feedback, so this is the
mechanism and nothing else. Joint-trajectory RMS against Isaac, and Godot's pelvis height:

| t | mirrored limits, LC=1 | limits fixed, LC=1 | **both fixes (limits fixed, LC=0)** |
|---|---|---|---|
| 0.50 s | 0.114 (h 0.82) | 0.210 (h 0.79) | **0.122 (h 0.80)** |
| 1.00 s | 0.195 (h 0.81) | 0.287 (h 0.84) | **0.163 (h 0.81)** |
| 1.50 s | 0.859 (h 0.31) | 0.315 (h 0.82) | **0.159 (h 0.76)** |
| 2.00 s | 0.893 (h 0.17) | 0.334 (h 0.80) | **0.120 (h 0.81)** |
| 3.00 s | 0.382 (h 0.33) | 0.364 (h 0.79) | **0.121 (h 0.80)** |

**7.4x less divergence at t=2.0, and Godot stays upright for the whole window instead of collapsing.**
The two bodies now respond to the same commands the same way, which is the thing this file has been
trying to establish since it was created.

The corrected limits also hold under real dynamics, not just in the isolated probe. Checked against
the declared limits over a whole run, mapping columns through `newton_dof_order`:

| condition | DOFs past a limit by >0.05 rad | worst excess |
|---|---|---|
| passive collapse, zero action, no balance | 6 of 45 | 0.152 rad |
| policy at authority 0.10, upright throughout | 3 of 45 | 0.071 rad |

Ordinary soft-constraint give under impact. (A first pass at this reported the knee 1.2 rad past its
stop; that was a column-to-name mapping error in the ANALYSIS - the trace is in `newton_dof_order`
and the rig JSON is not. Always map through the contract.)

**So the remaining failure is not the plant.** It is what the policy does with a matched body, which
is the reward - see below.

## The remaining problem is ROBUSTNESS, and it is now measured

With the plant matched, three separate retrains from `walk_spd/model_18050` all made Godot transfer
WORSE while Isaac reward went UP:

| run | Isaac mean reward | Godot upright @ 0.10 |
|---|---|---|
| baseline `model_18050` | ~68 | **100%** |
| +700 it, ceiling pinned 0.35 | - | 30% |
| +2100 it, double-flight penalty | **72.8** | **9.5%** |

The double-flight penalty itself did NOT work: `Episode_Reward/double_flight` sat at -20.7 to -21.3
for all 2100 iterations and never fell. The policy absorbed the cost rather than avoiding it -
airtime still pays +75.7 against a -21.3 penalty. Raising the weight is possible, but see below for
why that is probably not the lever.

**The decisive measurement.** Godot is bit-for-bit DETERMINISTIC - five identical runs of the same
config produced identical upright %, displacement, step count and final height. So differences
between checkpoints are real, not noise. And yet:

    walk_spd/model_18050            -> 100.0% upright at authority 0.10
    walk_grounded/model_18050       ->  24.7% upright at authority 0.10
    max relative weight difference  ->   1.7%  (mlp.4.bias; a handful of gradient steps)

**A 1.5% weight perturbation destroys the transfer.** The policy is not "tuned to Isaac"; it is
balanced on a marginally stable operating point that a few gradient steps walk off. That explains
every result in this file's history at once: why checkpoint selection has behaved like a lottery,
why the Isaac score has never predicted Godot transfer across ~9 attempts, and why "stands at 0.10,
falls at 0.11" is a knife edge rather than a margin.

**So the next lever is robustness, not reward shaping and not checkpoint selection.** Training
currently randomises almost nothing about the BODY: `effort_scale_range` (0.65-1.0), reset joint and
height noise, an observation noise term, and a spawn shove. Physics randomisation (`rand_*`) was
dropped from this config as "a measured dead end" - but that verdict came from the 2.3.2 PhysX path,
through `robot.root_physx_view`, which does not exist on Newton, and it was reached BEFORE the plant
was matched and while the dummy could not bend its knees. It should be re-tried.

Concretely: randomise per episode the joint stiffness and damping, the effort ceiling, body masses
and foot friction, so that no single exact operating point is exploitable. A policy that survives a
+/-20% plant cannot be balanced on a 1.5% ledge, and the plant is now close enough (0.12 rad
open-loop over 3 s) that Godot sits comfortably inside such a distribution.

## Plant randomisation CANNOT go through the solver on this backend - measured

Before building randomisation on it, every candidate knob was written with a different value per
environment, read back, and then tested for whether it changes anything mechanically. The test holds
the body against a constant target and asks whether steady joint deviation varies MONOTONICALLY with
the scale written to each environment:

| knob | write accepted? | reads back per-env? | rank correlation with its own scale | verdict |
|---|---|---|---|---|
| `write_joint_stiffness_to_sim` | yes | yes (spread 281.1) | **r = -0.014** | **INERT** |
| `write_joint_damping_to_sim` | yes | yes (spread 10.8) | **r = -0.061** | **INERT** |
| `write_joint_effort_limit_to_sim` | yes | yes (spread 400.0) | **r = +0.162** | **INERT** |
| `write_joint_armature_to_sim` | yes | yes | **r = -0.257** | **INERT** |
| body mass | **no write method** on the articulation | - | - | unavailable |
| friction | no `root_physx_view` on Newton | - | - | unavailable |

**Every one of them accepts the write and reads back correctly while changing nothing.** That is the
exact failure mode of the inert `P4F_GAIN_SCALE`, and it is why this was measured before being built
on rather than after. It also independently confirms `assets.py`'s note: under XPBD the effective
stiffness is dominated by iteration count, and the effort limit is ignored by the solver.

**A first attempt at this test was itself wrong** and is recorded so it is not repeated: measuring
per-env DIVERGENCE under zero actions reported all four as "BITES". With reset noise off and nothing
randomised at all the baseline spread was already 0.238 rad - a falling body is chaotic and amplifies
non-deterministic GPU reductions until any real effect is invisible. Hold the body, do not drop it.

**The way through:** `effort_scale` bites because it is applied in PYTHON inside `_effort_limited`,
which inverts the PD law to clamp the TARGET - it never goes near the solver. So on this backend
randomisation has to live in the action pipeline, not in the articulation. That is a real constraint
on how robustness can be trained here, and it is now measured rather than assumed.

## THE RESIDUAL GAP, FOUND: Isaac's joints do not sag and Godot's do

Zero action, gravity on, both bodies allowed to settle. This is a static test - no gait, no policy,
no timing, nothing to argue about:

| joint | **Godot settled** | **Isaac settled** |
|---|---|---|
| `joint_Forearm_L:0` | **+0.368 rad** | +0.000 |
| `joint_UpperArm_L:0` | **+0.159** | +0.002 |
| `joint_Thigh_L:0` | **+0.121** | +0.005 |
| `joint_Chest:0` | **-0.125** | -0.001 |
| `joint_Hand_L:0` | -0.033 | +0.001 |

**Godot sags up to 0.37 rad under its own weight; Isaac is rigid.** 21 degrees on the forearm.

**This invalidates the premise of the SPD gain-matching work.** That effort scaled Isaac's per-joint
`stiffness` onto Godot's measured Stable-PD values and called the discrepancy "the largest single
mismatch between the two engines" - correctly. But `write_joint_stiffness_to_sim` is mechanically
INERT on this backend (measured above, r = -0.014), and `assets.py` says the same thing in its own
words: under XPBD effective stiffness is dominated by iteration count, not by kp. So the gains were
rescaled, the YAML recorded new numbers, every checkpoint since was declared incomparable with the
ones before - and the bodies were never actually made to match. Godot is compliant; Isaac is rigid;
they still are.

**How to close it, given the solver ignores stiffness.** The one lever that bites is the TARGET,
because `_effort_limited` already rewrites it in Python. Godot's sag is a compliance deflection:
a joint carrying gravity torque `tau` under effective gain `kEff` sits at `theta = tau / kEff`. Isaac
can reproduce that exactly by displacing its commanded target by the same deflection:

    target' = target - tau_gravity / kEff

and `p4f_newton/gravity_ff.py` ALREADY computes `tau_gravity` per joint - it was written to add the
term as a feed-forward, which measured harmful. Used instead to displace the target, the same
quantity produces the compliance Godot has. Gate before trusting it: Isaac's settled zero-action
pose must reproduce the Godot column above, per joint, not just on average.

### Compliance: implemented, sign correct, frame WRONG. Left at 0.0.

`joint_compliance` displaces the commanded target by `L/kEff` using `gravity_ff.torques(clamp=False)`.
Enabling it at 1.0 moves every joint the RIGHT WAY - which confirms the sign convention and the
mechanism - and total error across the five reference joints falls only 0.798 -> 0.683 rad.

It does not pass its gate, and the full 45-DOF profile says why:

| dof | Godot sag | Isaac, compliance 1.0 |
|---|---|---|
| `joint_Forearm_L:0` / `_R:0` | **+0.368 / +0.368** | +0.215 / **+0.072** |
| `joint_UpperArm_L:0` / `_R:0` | **+0.159 / +0.159** | +0.436 / **-0.232** |

**Godot is exactly left/right symmetric and this implementation is not** - it flips sign between
mirrored limbs. The load is resolved into the PARENT BODY's frame, but a D6 joint's axes live in the
joint's own rest frame, and left and right bones have mirrored rest orientations. So component `k`
of the parent-frame torque is not DOF `k` on both sides. Total |sag| also overshoots (3.524 against
Godot's 2.599) while individual joints are wrong in both directions.

Calibrating a per-DOF scale on top of this would fit the bug. **The fix is to resolve the load in
the joint's own frame** - `parent_quat * rest_local_rotation`, the same correction that
`IsaacObservation.AppendJointVelocities` needed on the Godot side for exactly the same reason - and
then re-run this table. Symmetry is the cheap check: any correct implementation must produce
identical sag on `_L` and `_R`.

`joint_compliance` defaults to 0.0, so none of this is live.

### Compliance modelling FAILED. Both frames. Left at 0.0.

Total error against Godot's measured sag, all 45 DOFs:

| | rigid (compliance off) | parent-frame | joint-rest-frame |
|---|---|---|---|
| total \|error\| | **2.599** | 3.886 | **5.343** |

**Both versions are WORSE than leaving Isaac rigid.** The rest-frame correction fixed leg symmetry
(`Shin` asymmetry 0.003, `Thigh` 0.035, against 0.364 on `UpperArm` before) and made total accuracy
worse. An earlier note here claimed the parent-frame version improved things 0.798 -> 0.683; that was
five hand-picked joints, and across the body it overshoots. `gravity_ff`'s load model is not accurate
enough to drive a deflection, which its own recorded 0.21-0.68x leg error already said.

`joint_compliance` stays 0.0. Do not re-enable without beating 2.599 on this table.

## Godot's PHYSICS RATE closes half the sag gap, and the trained authority finally survives

The Stable-PD denominator is `1 + Kd*dt/I + Kp*dt^2/I`, so it collapses toward 1 as `dt` shrinks:
raising Godot's rate raises its effective gains and reduces sag. Measured, zero action, settled:

| Godot rate | Forearm_L | UpperArm_L | Thigh_L | Chest | **total \|sag\|** |
|---|---|---|---|---|---|
| 120 Hz | +0.368 | +0.159 | +0.121 | -0.125 | **2.599** |
| **240 Hz** | +0.353 | **-0.010** | **+0.033** | **-0.021** | **1.326** |
| 480 Hz | +0.368 | -0.010 | +0.032 | -0.018 | 1.176 |

**Half the static plant mismatch, for one line of `project.godot`.** Isaac is rigid (sag ~0.00), so
lower is closer. 480 Hz buys little over 240 for twice the cost again. The forearm does not improve
at any rate - that sag is not SPD-denominator-limited and is a separate mechanism, and it is now 27%
of the entire remaining error.

**This is Phase 1 of the original plan, which was never validly tested**: the 2026-08-26 `.pck`
hijack made the earlier 480 Hz attempt a silent no-op, so it read as "no effect".

**Transfer result** - the walk checkpoint, `LoadCompensation = 0`:

| | authority 0.10 | authority 0.15 (TRAINED) | 0.20 |
|---|---|---|---|
| 120 Hz | 100% upright | **5.1%** | - |
| **240 Hz** | 100% upright | **100% upright** | 3.2% |

**The trained authority survives in Godot for the first time.** It is still a statue - 0 foot strikes,
+0.02 m in 20 s - so this is a precondition for walking, not walking. Re-scoring the retrained
lineages at 240 Hz did NOT rescue them (`walk_robust` and `walk_grounded` both 3.2% upright at 0.15):
the higher rate helps the body, not those policies.

`project.godot` is now at 240. Cost is 2x Godot physics compute for the whole project; revert by
setting `common/physics_ticks_per_second=120`.

## The forearm thread: three hypotheses, all wrong, and one important rule

The forearm sag (0.35 rad, unchanged at 120/240/480 Hz, 27% of remaining static error) was chased
and is still unexplained. Recorded so nobody repeats it:

| hypothesis | test | result |
|---|---|---|
| Godot's gravity feed-forward holds it | `LoadCompensation` 0 vs 1 at 240 Hz | **worse** with it on: forearm 0.353 -> 0.435, total sag 1.326 -> 2.232. `LoadCompensation = 0` re-confirmed at 240 Hz |
| Reaction path (Godot reacts into FEET, Isaac into THIGHS) | added `PelvisReactIntoThighs`, matched Isaac | **no effect**: total 1.326 -> 1.357 |
| `ArmLoadBearingGain` (10x) silently not applied | assist parks the body in `Balanced`, not `ReinforcementLearning`, so the gain is skipped in the deployment configuration | **reduced static deviation 1.326 -> 1.119 and BROKE TRANSFER** - authority 0.15 fell from 100% upright to 15.4%. Reverted. |

**The rule that came out of it, and it is the most useful thing here: STATIC POSE SIMILARITY DOES
NOT PREDICT TRANSFER.** The arm-gain change made Godot's settled pose measurably closer to Isaac's
and destroyed the only authority that works. Every plant change must be scored on the ladder, not on
the settled pose. That also retires "make the sag match" as an objective in itself.

A real confound found on the way: `BalanceAssist = 0` does not merely zero a torque - the driver
puts the ragdoll in `ReinforcementLearning` instead of `Balanced`, which changes muscle stiffness and
the arm gain. So "assist off" comparisons are comparing two different bodies, not one body with and
without a torque. That invalidates reading the 0.640-vs-1.326 gap as the assist's own doing.

`PelvisReactIntoThighs` is kept (default false, no behaviour change) because it makes the reaction
path switchable and the measurement repeatable.

## The whole walk archive, re-scored on the fixed plant: only statues stand

Every checkpoint this project ever scored was scored on a broken body - mirrored knees, 120 Hz, load
compensation on. Re-scored at 240 Hz with the limits fixed and `LoadCompensation = 0`, authority 0.15:

| lineage | displacement | steps | upright |
|---|---|---|---|
| `walk_spd/night01` | -0.46 m | 3 | 4.6% |
| `walk_spd/night04` | -0.35 m | 4 | 4.2% |
| `walk_hill/night02` | +0.01 m | 4 | 4.1% |
| `walk_plant/night02` | -0.29 m | 4 | 3.2% |
| **`walk_contact/contactA`** | **-0.08 m** | **0** | **100%** |
| `walk_full/night02` | -0.91 m | 3 | 4.6% |
| `walk_cal/night01` | -0.55 m | 5 | 5.2% |

With the four retrains and the promoted checkpoint that is **~12 checkpoints across 8 lineages, and
the split is perfect: every policy that MOVES falls, every policy that does not move stands.** This
is not a lottery and it is not checkpoint selection. It is a boundary.

Tracing a mobile run at authority 0.30: the body holds 0.78-0.83 m for ~1 s while both feet report
no contact, then tips (gravity Z -0.99 -> +0.25) and is down by t=1.7 s.

**Contact threshold, checked and NOT the cause but previously mis-documented.** Godot uses 0.06 m
and the Newton task uses 0.034, while `IsaacObservation` claimed they matched. They should not match:
a planted foot rests at 0.040 m in Godot and 0.017 in Isaac, so what has to agree is the CLEARANCE
before a foot reads airborne - 0.020 against 0.017, which is close. Copying 0.034 into Godot would
make a planted foot report no contact. Comment corrected.

## Penalising the flight phase: measured twice, does not work

| weight | run | result |
|---|---|---|
| -5.0 | 2100 iterations | term pinned at -20.7, never fell; Godot 9.5% upright |
| **-20.0** | ~20 min, killed early | term went the **WRONG WAY**, -62.9 -> -71.9, while `track` rose 22.9 -> 25.1 and mean reward fell to -8.99 |

Sized so a bounding gait could not pay for itself (-85 against `feet_air_time` +73), and a true walk
would pay ZERO because it always keeps a foot down. The policy instead bought more speed and paid the
penalty out of it. **A penalty makes the current optimum cheaper without building a path to a
different one**, and `feet_air_time` is simultaneously paying for airtime. Killed at 20 minutes
rather than burning the 3-hour budget on a trend that was already going the wrong way.

**This closes reward shaping as an approach.** A grounded gait needs a formulation where walking is
the REACHABLE optimum - a contact schedule, a phase reference, a gait prior - not a scalar penalty
bolted onto a reward that pays for flight.

## THE POLICY IS NOT WALKING. IT IS HOPPING.

Contact-state occupancy of `walk_spd/model_18050` over 12 s of its own gait in Isaac:

| state | occupancy |
|---|---|
| both feet down | **52.9%** |
| only left down | 0.3% |
| only right down | 1.0% |
| **neither down (flight)** | **45.8%** |
| **feet in the SAME state** | **98.8%** |

Single support - the phase that defines walking - is **1.3%**. The feet leave and land together:

    L ..################......##############..
    R ...################......##############.

Every double-flight burst lasts exactly 0.167 s and so does every single-foot swing, which is only
possible in phase. **This is a two-footed hop at about 2.9 Hz**, and it is the only mobile behaviour
in the entire checkpoint archive.

That is why nothing transfers. A synchronised hop lives on simultaneous impulsive landings, which is
the one place Jolt and XPBD differ most - and it explains the perfect split (every mobile policy
falls, every statue stands) without needing any of the plant explanations tried before it.

The reward allowed it because `feet_air_time` pays per touchdown and a hop collects on BOTH feet
every cycle, while nothing paid for alternation.

## `rew_single_support`: pay for exactly one foot down

Zero for double support, zero for flight, so it cannot be collected by standing still OR by hopping.
Gated on commanded motion and on posture, like `feet_air_time`.

**Why a reward where the penalty failed.** `rew_double_flight` was measured at -5.0 and -20.0 and
made things worse both times - a penalty makes the current optimum cheaper without building a path
to a different one. This pays for the target behaviour, and the path is short: the feet are already
offset by one policy step, so sliding that offset toward half a cycle raises the term monotonically.

**Measured after 12 minutes / 100 iterations**, resumed from `model_18050`:

| | double | **single support** | flight | speed |
|---|---|---|---|---|
| before | 52.9% | **1.2%** | 45.8% | 0.86 m/s |
| after | 53.2% | **10.7%** | 36.1% | 0.65 m/s |

Single support **9x**, flight down, speed down - which is what a hop turning into a walk looks like.
The run stayed healthy throughout (mean reward 73-79, episode length 626-681), unlike the penalty
runs which drove reward negative. Long run in progress.

### `rew_single_support` result: it FIXES the gait in Isaac and does not fix transfer

132 min, 6000 iterations, resumed from `model_18050`. Contact occupancy in Isaac, command 0.30:

| | double | **single support** | flight | speed |
|---|---|---|---|---|
| baseline hop | 52.9% | **1.2%** | 45.8% | 0.86 m/s |
| 12 min | 53.2% | 10.7% | 36.1% | 0.65 |
| **35 min (best)** | 25.7% | **53.8%** | **20.6%** | 1.49 |
| 70 min | 29.9% | 49.0% | 21.1% | 1.35 |
| 132 min (final) | 33.8% | 45.4% | 20.8% | 1.16 |

**The term works: it turned a two-footed hop into an alternating gait.** Single support went from
1.2% to 53.8%, which is the first real gait this project has produced. Keep the term.

**Two things it did not do.** Flight plateaued at ~21% and never fell further - the whole gain
happened in the first 35 minutes and the next 97 made the gait slightly WORSE (53.8% -> 45.4% single
support). And Godot transfer did not move: 11.6% upright at authority 0.15 at 35 min, **9.1% at the
final checkpoint**, statue at 0.10 throughout.

**So an alternating gait is necessary and not sufficient.** 21% flight still means the dummy leaves
the ground a fifth of the time and lands on both feet, which is the part Jolt and XPBD disagree
about. Driving flight to zero needs more than paying for single support - the obvious next lever is
a duty-factor target (reward stance fraction per foot directly, or a phase clock in the observation),
but the observation is frozen at 143 floats and adding a phase would break the Godot contract.

**Process note:** the best checkpoint was at 35 minutes and the run went to 132. A checkpoint sweep
every ~30 min - `dump_obs.py` for occupancy plus `godot_walk_score.py` - costs about 4 minutes and
would have caught the plateau. Do that instead of trusting a long run.

### Two-sided speed tracking: fixes the SPEED, does not touch the flight

`_drive` divided achieved speed by the command and clamped to 1, so running 4x too fast earned
nothing and cost nothing. `drive_overspeed_sigma = 0.5` replaces that with
`exp(-((v - v_cmd)/sigma)^2)`, which peaks AT the command. Measured after 15 min, resumed from the
BEST single-support checkpoint (`model_19750`, 53.8% single support - not the degraded final one):

| | double | single support | **flight** | **speed** (cmd 0.30) |
|---|---|---|---|---|
| baseline hop | 52.9% | 1.2% | 45.8% | 0.86 |
| single-support, 35 min | 25.7% | 53.8% | 20.6% | 1.49 |
| single-support, final | 33.8% | 45.4% | 20.8% | 1.16 |
| **+ two-sided track** | 39.4% | 38.2% | **22.4%** | **0.56** |

**The kernel works and the hypothesis was wrong.** Speed fell 1.16 -> 0.56 m/s, close to the 0.30
command - so overspeed WAS free and now is not. But flight did not move (20.8% -> 22.4%), and Godot
transfer got slightly worse (9.1% -> 6.2% upright at 0.15).

**Flight is not instrumental to speed.** The dummy leaves the ground a fifth of the time even when
tracking the command at 0.56 m/s. That kills the "it flies in order to go fast" explanation, and it
means the remaining flight is something the gait does for its own reasons - most likely because
nothing in the reward distinguishes a step that lifts off from one that rolls through stance.

Keep `drive_overspeed_sigma`: tracking the command is correct on its own terms and it removed a real
free lunch. It is not the transfer fix.

### Removing `feet_air_time` halves the flight - and transfer STILL does not move

`feet_air_time` pays per touchdown for how long that foot was airborne, and it was the largest term
in the reward (+60 to +75 per episode). Every previous attempt added something ALONGSIDE it. Setting
it to 0.0, with `rew_single_support` present to keep the feet alternating:

| | double | single support | **flight** | speed (cmd 0.30) | **Godot @0.15** |
|---|---|---|---|---|---|
| baseline hop | 52.9% | 1.2% | 45.8% | 0.86 | - |
| + single support | 25.7% | 53.8% | 20.6% | 1.49 | 11.6% |
| + two-sided track | 39.4% | 38.2% | 22.4% | 0.56 | 6.2% |
| **+ airtime removed** | 42.8% | **46.8%** | **10.4%** | 0.60 | **3.7%** |

**The gait is now genuinely fixed in Isaac.** From a two-footed hop with 1.2% single support and
45.8% flight, to an alternating gait with 46.8% single support, 10.4% flight, tracking the commanded
speed. Removing the airtime reward was the piece that finally moved flight - it had been paying for
the lift-off the whole time. Also note it did NOT cause the sliding the term was introduced to
prevent: `rew_single_support` does that job better, because a slide cannot alternate contacts.

**And Godot transfer did not improve at all.** 11.6% -> 6.2% -> 3.7% upright at authority 0.15, all
of them "falls". Statue at 0.10 throughout.

**So the gait hypothesis is FALSIFIED as an explanation for the transfer failure.** The hop was real,
was worth finding, and is worth keeping fixed - but a mobile policy with a proper alternating
low-flight gait at the commanded speed falls in Godot exactly like the hop did. The honest statement
is the simple one and it has survived every test today:

> **Anything that moves falls in Godot. Gait quality does not change that.**

That sends the question back to the plant under MOTION - which is where the open-loop replay already
pointed: driven by Isaac's own actions Godot tracks to ~0.12 rad and stays upright for about 3 s,
then goes. The two bodies agree statically and diverge within a few strides. That divergence, not
the gait, is what is left.

## REVERSED: 240 Hz is WORSE. Reverted to 120.

Earlier today 240 Hz was adopted project-wide on two measurements: it halved the static sag
(2.599 -> 1.326) and it took the trained authority 0.15 from 5.1% upright to 100%. Both were real.
Both were the wrong test.

The open-loop replay - the same recorded action sequence driven into both engines, no policy in
either loop, which is the only measurement of the plant UNDER MOTION - says the opposite. Joint RMS
against Isaac, and Godot's pelvis height:

| action source | rate | t=0.5 | t=1.0 | t=3.0 | height @1s / @3s |
|---|---|---|---|---|---|
| `model_18050` | **120 Hz** | 0.099 | **0.153** | **0.126** | **0.81 / 0.81** |
| `model_18050` | 240 Hz | 0.331 | **1.012** | 0.207 | 0.30 / 0.12 |
| `walk_noair` | **120 Hz** | 0.119 | 0.145 | 0.882 | 0.83 / 0.28 |
| `walk_noair` | 240 Hz | 0.416 | 0.727 | 0.275 | 0.15 / 0.13 |

**At 240 Hz Godot falls inside one second under Isaac's own actions; at 120 Hz it tracks for three.**
Two independent action sequences agree.

**How both results can be true.** 240 Hz makes Godot stiffer, which helps a nearly motionless policy
hold a pose - and the thing that "stood at 0.15" was a statue covering 0.02 m in 20 s. It does not
help, and actively hurts, the dynamic response the policy has to live inside. This is the rule that
came out of the arm-gain experiment, applied to my own earlier decision:

> **Static pose similarity does not predict transfer. Score the dynamics, not the settled pose.**

Reverted to `physics_ticks_per_second=120`. The static sag is worse at 120 and that is the correct
trade: sag is a static measure and we are chasing a gait.

## The divergence probe: distal-led, immediate, and NOT contact-driven

`scripts/probe_divergence.py` reads an open-loop replay and reports per joint when its position error
starts to GROW past a threshold (subtracting the t=0 gap, so "started apart" is not reported as
"diverged"), whether velocity or position led, and how the onset sits relative to foot strikes.

Measured at 120 Hz, clean start (t=0 gaps 0.000-0.026 rad, so the bodies genuinely start together):

* First to go are `Foot_L:0` and `Foot_R:0`, symmetric, within one policy step.
* Then forearms and hands, then shins, then spine - **distal-led, propagating inward**.
* **Velocity leads position everywhere** (`Shin_L:0` velocity at 0.02 s, position at 0.13 s).
* All of it **before the first foot-contact transition at 0.55 s**, so it is not contact-event driven.
* Foot friction 1.2 vs Isaac's 1.0 changes the peaks slightly and **not the onsets at all** (0.02 /
  0.07 / 0.08 either way), so it is not friction.

### Godot micro-vibrates at rest and Isaac does not - and it predicts the divergence order

Zero action, settled, joint velocity magnitude (rad/s):

| | median | p95 | max |
|---|---|---|---|
| **Godot ankle** | **0.086** | 0.174 | 0.519 |
| Isaac ankle | 0.006 | 0.766 | 1.984 |
| **Godot shin** | **0.083** | 0.262 | 0.403 |
| Isaac shin | 0.002 | 0.028 | 0.101 |

**Godot's median is 14x higher at the ankle and 40x at the shin.** Godot chatters continuously at low
amplitude; Isaac sits still and occasionally spikes. Isaac's foot also rests 7 mm INTO the ground
(median height 0.0081, min -0.0067), which is XPBD's soft contact - Jolt resolves the same standing
contact by buzzing instead.

**It predicts which joints go first.** Rank correlation between a joint's rest chatter and its
divergence onset in the open-loop replay: **Spearman -0.536** over 45 DOFs. The quietest joints
(`Chest:1`, `Head:1`, `Spine:1`, chatter 0.0001-0.0003) never diverge inside 4 s; the noisiest go
within 0.1-0.3 s.

**Stated honestly, this is a strong lead and not proof.** Chatter magnitude is confounded with joint
range and carried mass - big joints chatter more AND move more - so the correlation is consistent
with the chatter seeding the divergence and also with both being driven by load. What it does
establish is that the two engines resolve a STANDING CONTACT differently, continuously, before any
gait event, and that friction is not the parameter (1.2 vs 1.0 moved no onset at all).

### Jolt solver settings move the chatter around; they do not remove it

| config | ankle | shin | worst-on-body |
|---|---|---|---|
| baseline (velocity 10 / position 2 / slop 0.02) | 0.0856 | 0.0828 | 0.264 |
| velocity 30 / position 8 | 0.0637 | **0.1512** | 0.270 |
| velocity 60 / position 16 | 0.0590 | 0.1484 | 0.276 |
| penetration_slop 0.002 | **0.0544** | 0.1459 | 0.267 |
| speculative_contact 0.002 | 0.0862 | 0.0800 | 0.268 |
| **ISAAC** | **0.0060** | **0.0020** | - |

More solver work cuts ankle chatter by up to 36% and nearly DOUBLES the shin's; worst-on-body sits at
0.264-0.276 in every configuration. Nothing gets within an order of magnitude of Isaac. The standing
chatter is not a tunable property of Jolt's contact solver - it looks intrinsic to driving joints
with explicit PD torques against constraints, which is the solver-class difference this project
identified long ago and cannot configure away.

**Setting-name trap, cost two full sweeps.** Godot project settings are written SECTION-RELATIVE.
Under `[physics]` the key is `jolt_physics_3d/simulation/velocity_steps`, NOT
`physics/jolt_physics_3d/simulation/velocity_steps` - the full path inside the section becomes
`physics/physics/...` and is silently ignored. The prefix is also `simulation/`, not `solver/`.
Symptom: byte-identical results across every value, which reads as "no effect" and is actually
"never applied". `ProjectSettings.HasSetting` plus reading the value back is the check; two sweeps
were run and believed before that check was made.

### 240 Hz DOES eliminate the chatter - and the chatter is not the blocker either

Controlled A/B, same code, back to back, zero action, settled (median |joint velocity|, rad/s):

| | ankle | shin | thigh | worst-on-body |
|---|---|---|---|---|
| 120 Hz | 0.0856 | 0.0828 | 0.0998 | 0.264 |
| **240 Hz** | **0.0040** | **0.0037** | **0.0073** | **0.010** |
| **ISAAC** | **0.0060** | **0.0020** | - | - |

**At 240 Hz Godot's standing chatter MATCHES Isaac's**; at 120 Hz it is ~20x worse. So the chatter is
not intrinsic to Jolt after all - the earlier conclusion ("solver settings only redistribute it") was
right about the solver settings and wrong to generalise, because the integration RATE fixes what the
solver iteration counts could not.

**And it still does not transfer.** Scoring WALKING at both rates:

| rate | policy | authority | displacement | steps | upright |
|---|---|---|---|---|---|
| **120 Hz** | `walk_noair` | 0.15 | **+0.34 m** | 3 | 30.1% |
| 120 Hz | `model_18050` | 0.15 | -0.23 | 4 | 5.1% |
| 240 Hz | `walk_noair` | 0.15 | -0.25 | 5 | 3.7% |
| 240 Hz | `model_18050` | 0.15 | +0.02 | **0** | 100% |

**Every 240 Hz result is a statue** - 0 steps, no displacement, upright. 120 Hz keeps the largest
forward displacement this project has recorded (+0.34 m, from the new alternating-gait policy).

So the Spearman -0.536 between rest chatter and divergence order was a correlation and not a cause:
removing the chatter entirely removes the motion with it. **120 Hz stays.** The revert was right, for
a reason different from the one recorded at the time.

## Four retrains, four regressions - and the reason is simpler than Goodhart

| run | intervention | Isaac mean reward | Godot upright @0.10 | Godot displacement |
|---|---|---|---|---|
| `walk_spd/model_18050` | baseline | ~68 | **100%** | -0.05 m |
| `walk_ground2` | ceiling pinned 0.35, 700 it | - | 30% | -1.09 m |
| `walk_grounded` | double-flight penalty, 2100 it | 72.8 | 9.5% | -0.14 m |
| `walk_robust` | action-scale randomisation, 2100 it | **75.1** | 34% | -1.05 m |

Isaac reward rose monotonically while Godot uprightness fell. The tempting reading is Goodhart, and
that is how this file described it earlier today. **The simpler reading is better supported: the
baseline is winning the ladder by not moving.** It covers -0.05 m in 20 s with 2 foot strikes. Every
retrain makes the policy MORE mobile - which is what training for locomotion does - and in Godot a
policy that actually moves falls over. The ladder scores survival, so the least mobile policy wins it.

**There is therefore no transfer success to protect.** `model_18050` stands; everything else falls;
none of them walks. The four regressions are not four failures to preserve a working gait, they are
four confirmations that Godot cannot yet carry a gait at all.

That is consistent with the open-loop measurement: under Isaac's own actions Godot now tracks to
0.12 rad and stays upright for ~3 s, then goes. The plant is close - 7.4x closer than yesterday -
but the residual still compounds within the length of a few strides, and a marginally stable policy
has no margin to absorb it.

**What this rules out.** More training from this checkpoint, in any of the three flavours tried.
Reward shaping toward a grounded gait (measured: the penalty was absorbed, not avoided). Checkpoint
selection (a 1.5% weight change flips the result, so the lineage is a lottery). Solver-side plant
randomisation (inert on this backend, measured above).

**What is left, in the order I would try it.** (1) Close the residual plant gap further - 0.12 rad
over 3 s is good but evidently not good enough, and the open-loop replay now localises exactly which
joints drift first, which is a measurement nobody has had before. (2) Train much longer with the
action-pipeline randomisation that does bite - 2100 iterations is far too short for robustness to
appear, and this run at least did not diverge. (3) Score by displacement DURING training rather than
after, so the lineage is selected on locomotion instead of survival.

## Tried on the corrected rig and did NOT work

Recorded so the next session does not spend the time again.

| attempt | result |
|---|---|
| `BalanceJointModules = false` at a=0.15 | still falls (peak linVel 2.01) |
| `HillVelocityFilter = 1.0` at a=0.15 | still falls |
| `BalanceAssist = 0.0` at a=0.15 | still falls (2.28) |
| Command 0.60 and 1.00 instead of 0.30 | uprightness DROPS to 75% / 61% at a=0.10; still 0 net displacement. The policy over-achieves speed in Isaac (0.86 m/s under a 0.30 command), so asking for more looked promising. It is not the missing input. |
| 15 min retrain, `cmd_curriculum=False`, `cmd_speed_min=0.35`, `-InitFrom model_18050` | **worse**: upright 19% at both authorities. `-InitFrom` resets the optimiser and the iteration count, and 735 iterations at a changed command distribution degrades a policy rather than adapting it. A grounded-gait run needs a proper resume and far more than 15 minutes. |

## Where this leaves the gap

Two real defects were found and fixed, and they moved the ladder from "stands only at 0.03" to
"stands at 0.10". Neither produces a GAIT, and the reason is now specific rather than mysterious:

**Isaac's walk policy has learned a bounding run with a 45.8% flight phase**, at 0.86 m/s under a
0.30 m/s command, because the command curriculum ran its ceiling to the configured 1.0 m/s. In Godot
the same policy stands still. A ballistic gait is the least transferable kind of gait, and no amount
of plant matching will make a Jolt body reproduce an XPBD flight phase.

**The next thing to do is train a policy that walks rather than bounds**, on the corrected rig, with
a proper resume rather than `-InitFrom`, and scored by `godot_walk_score.py` (displacement and foot
strikes) rather than by survival. That is a training-side problem now, not a configuration-matching
one - which is a different and much better place to be than this file described yesterday.

## Verified matched, so not worth re-testing

Gravity `-9.81` both. Total mass `80.6 kg` both, and all 16 body masses agree. `dt = 1/120`,
decimation 2, policy 60 Hz both. Actuator plant re-derived after the limit fix and unchanged (worst
bone 1.4%, pose noise), so the limit correction does NOT invalidate existing checkpoints.

## 2026-09-04: Godot's joint-velocity observation measured the wrong quantity

Isaac's `joint_qd` is the derivative of its own `joint_q`, in the same generalised coordinates.
Godot built slice [55:100] from BODY angular velocities instead - `omega_bone - omega_parent`
projected on an axis. Those are the same number only while a joint is near its rest pose; they are
related by the Euler kinematic matrix, which departs sharply from identity exactly where a joint is
far from rest, which is the knee during a gait.

Measured under the walk policy at the trained authority 0.15, both bodies still upright, command
matched at 0.30 m/s:

| channel | worst-on-body jointVel, median |
|---|---|
| Godot, omega (old) | **15.00 rad/s - pinned at the observation clip** |
| Godot, theta-dot (`JointVelocityFromDifference`) | **7.02** |
| Isaac `joint_qd` | **3.76** |

Per joint the old channel reported the knee at a median **12.09 rad/s** where the true coordinate
rate is **1.32** - a 9x overstatement that held 45 of the 143 observation floats saturated. Fixed by
differencing the reported angle, which gives theta-dot by construction. Worth ~2 s of extra survival
at authority 0.05 on its own; a fidelity fix, not the blocker.

## 2026-09-04: the ONNX Godot runs IS the policy Isaac trained (first time checked)

`export.verify()` only ever checked the graph's shapes and its answer to a synthetic rest pose.
`scripts/probe_parity.py` now feeds 960 REAL observation vectors through both the torch actor and
the ONNX session: **max |onnx - torch| = 1.9e-6**. The two engines run the same controller, so every
remaining difference is the body. This had never been tested and would have invalidated everything
downstream had it failed.

## Two claims in the previous version of this file were wrong

* **"Godot pins jointVel at 15.0, Isaac sits at 7.28."** The Godot half was measured AFTER the dummy
  had already fallen. Pre-fall at authority 0.05 only 6 samples in 39,105 saturate. Saturation is
  real, but only at the trained authority 0.15 and only because of the omega/theta-dot defect above.
  The Isaac half was also measured while the eval env commanded ~0.17 m/s rather than Godot's 0.30,
  because `_cmd_speed_ceiling` is runtime state that the EVAL path does not restore - an eval of a
  policy trained to a 1.0 m/s ceiling resamples from the starting 0.25. `dump_obs.py --command`
  forces it. **Correction to an earlier version of this note:** training resumes DO carry the
  ceiling - `train.py` has `CURRICULUM_CARRY` and writes `curriculum.json` next to the checkpoints
  ("curriculum ceiling 0.350 m/s recorded for the next run"). Only measurement was affected.
* **"Joint velocity anti-correlates with joint angle (-0.87)."** That was an alignment error in the
  analysis, not in the engine: the reported rate is a BACKWARD difference and it was compared
  against a FORWARD one. Correctly aligned the channel is +0.73 overall. The leg `.y` axes are still
  genuinely negative (-0.52 to -0.37), which is the Euler-rate effect above.

---

## The ledger

| subsystem | status | evidence |
|---|---|---|
| Joint stiffness | **MATCHED** | Step response, `joint_Thigh_L:1`, 0.15 rad: rise to 63.2% is 33.3 ms in BOTH engines |
| Joint damping | **MATCHED** | Same test: overshoot +0.4% Isaac, +0.9% Godot, at 8 solver iterations |
| Solver iterations | matched | 8 both sides (`P4F_XPBD_ITERATIONS=8`, Jolt 30/16 velocity/position) |
| Physics rate | matched | 120 Hz both; policy 60 Hz |
| Balance assist | matched | 1.0 both (Isaac restores it from the checkpoint, Godot scene sets it) |
| Contact flag height | matched | `CONTACT_HEIGHT = 0.034`, calibrated to Godot's planted foot at 0.0398 |
| Observation contract | matched | 143 obs / 36 actions, DOF order from the policy contract |
| Action -> joint target | matched | Same affine map about the rest pose; limits agree |
| Effort clamp | matched | `StandEnv._effort_limited` inverts the PD law to clamp the TARGET, per bone as a 3-axis norm, exactly as Godot bounds `totalTorque.Length()`. `enforce_effort_limit: true` in every run |
| Hill force-velocity law | matched | `StandEnv._hill_scale` is a faithful port of `ActiveBone.ComputeForceVelocityScale` - same shortening projection, same linear falloff, same eccentric exemption, same `vmax = 15.0` |
| **Hill velocity FILTER** | **fixed 2026-09-03** | Godot smooths the velocity feeding the Hill law (`HillVelocityFilterAlpha`, scenes set 0.15); Isaac used the RAW value. Measured effect below |
| **Gravity feed-forward** | **NOT matched** | Godot ~11 N.m per leg bone, ~5.6 per upper arm. Isaac: none (implemented, off - see below) |

An earlier version of this table claimed the effort clamp and the Hill law were missing from Isaac.
**That was wrong** - both were already implemented and enabled. Reading the code before writing the
ledger would have caught it. What was actually missing was one parameter inside the Hill law.

---

## What matching the joint dynamics cost, and bought

The largest single discrepancy, now closed: Godot drives every joint through `PidController3D`,
which uses the Tan-Liu-Turk Stable PD form and divides both gains by `1 + Kd*dt/I + Kp*dt^2/I`.
Measured live, Godot applied **a seventh** of the gain the rig contract authors (`kEff = 0.15`,
`kp 882 -> 109`) while Isaac's XPBD applied the whole of it.

The correction was already documented in `assets.py` as "the largest single mismatch between the two
engines" - and had never been applied to a single training run, because `GAIN_SCALE` defaulted to
1.0 and `P4F_GAIN_SCALE` was set nowhere in the repository. An env-gated correction that defaults to
inert is indistinguishable from no correction at all.

It is per BONE, not global: the denominator divides by each bone's inertia, so the measured spread is
**34x** (hand 0.0157, foot 0.0641, thigh 0.1757, chest 0.5404). The old single "0.176" figure is the
THIGH's factor and nothing else's.

**Result of closing it:** Isaac trains better than any previous lineage (mean reward 80.2 -> 81.9,
episode length 703, standing plateau 49-57%). Godot transfer is **unchanged** - the walk brain stands
at authority 0.03 and falls at 0.05, 0.10 and 0.15, exactly as before.

So the gain gap was real, large, and not the blocker.

---

## Tooling this produced (all reusable)

| tool | what it answers |
|---|---|
| `scripts/probe_plant.py` + `Scenes/RL/Isaac3/Probe` | Step response of one joint in each engine. The ONLY test that compares delivered dynamics rather than parameter names. |
| `scripts/derive_plant.py` -> `p4f_newton/godot_plant.py` | Generates the per-bone gain table, skeleton, masses, pivots and ceilings from a Godot run log. One source of truth: the Godot scene. |
| `scripts/probe_ff.py` | Per-bone diff of the gravity feed-forward against Godot's. |
| `p4f_newton/gravity_ff.py` | Godot's feed-forward reproduced in Isaac. Off by default; see below. |
| `run_conditions._check_plant` | Warns when a checkpoint is scored on different actuator gains than it trained on. |

`[PLANT]` lines in any Isaac3 check scene carry the whole Godot-side plant, emitted once per run.

---

## The Hill velocity filter: the first mechanism that explains the ladder

Godot filters the joint velocity before the Hill law with a per-tick EMA
(`filtered += (raw - filtered) * alpha`), and the check scenes set `alpha = 0.15`. Isaac fed the law
the raw velocity. Measured under the trained walk policy, 720 policy steps, 64 envs, 8 iterations:

| | median fvScale | p05 | min |
|---|---|---|---|
| Isaac, raw velocity | **0.583** | 0.461 | 0.369 |
| Isaac, filtered 0.15 | **0.857** | 0.781 | 0.754 |
| Godot, same policy (standing) | ~0.98 | - | - |

**Isaac's actuator was held at 58% of its ceiling through an entire gait** while Godot's runs near
full authority. A policy trained against a permanently derated actuator commands targets calibrated
for it, then meets Godot's nearly underated one and over-drives by roughly 1.7x.

That is the first mechanism that explains the SHAPE of the authority ladder rather than just its
failure: the brain survives at 0.03 because every command is scaled down enough to hide the excess,
and falls at 0.05 and above because it is not. It also explains why standing transferred - a
regulator at a fixed point barely moves, so the Hill law is inert in both engines and the discrepancy
never appears.

Fixed: `hill_velocity_filter = 0.15` on the Isaac side, applied per physics substep at the same
120 Hz, registered in `TRAINED_CONDITIONS`. Training `p4f_newton_walk_hill` on it now.

Caveat kept honest: Godot's ~0.98 was measured while STANDING at authority 0.03, and Isaac's 0.857
during an actual gait. Godot walking would derate too. The comparison establishes that the filter
was missing and that it matters, not that the two now agree to two decimals.

## What is left after this

**The gravity feed-forward**, and nothing else that has been measured. Godot applies ~11 N.m per leg
bone and ~5.6 per upper arm; Isaac applies none. `gravity_ff.py` implements it and `probe_ff.py`
diffs it per bone: faithful on the arms (+/-12%), only 0.21-0.68x on the LEGS. Switching it on as-is
cost 52.7% -> 0.0% standing, because an unfaithful feed-forward is an extra torque field biased in
the wrong places, not a weaker version of the right one. **Gate: leg ratios near 1.0 before
re-enabling.** The residual error is geometric - the term rides on a ~3 cm lever and the two rigs'
rest poses differ by ~2 cm.

## Eliminated by DIRECT test in Godot, no retraining needed

The fastest way to test "is X the blocker" turned out not to be matching Isaac up to Godot and
retraining, but pushing GODOT down to Isaac's value and re-running the existing brain. Each of these
is one scene property and about four minutes:

| candidate | test | result |
|---|---|---|
| Hill velocity filter | Godot `HillVelocityFilter` 0.15 -> 1.0 (raw, as Isaac trained) | `fvScale` fell 0.97-1.00 -> 0.64-0.92, confirming it took. **Still falls** at 0.05 and 0.10 |
| Foot/body friction | Godot foot 1.2 -> 0.5, body 0.9 -> 0.5 (Isaac's scene default) | **Still falls** at 0.05, 0.10, 0.15 |
| Joint-level balance modules | `BalanceJointModules = false` - stepping, arm reflex, gaze, ankle gains all off, pelvis stabilisation kept | log confirms it fired. **Still falls** at 0.05, 0.10, 0.15 |
| Joint limit damping | `angular_limit_*/damping = 20`, `restitution = 0` on all 45 axes | chatter unchanged (still pinned at 15.0). **Still falls** |

A 1.67x coincidence nearly sold the Hill hypothesis: the brain survives at 0.03 and fails at 0.05,
and the filter mismatch predicted a ~1.7x over-drive. Testing it directly killed it. Ratios that
match a prediction are not evidence; the intervention is.

---

## The largest remaining measured difference is in the OBSERVATION, not the plant

| | worst joint velocity, same policy |
|---|---|
| Isaac | median **7.28** rad/s, p95 8.73, max 16.8 |
| Godot | **PINNED at 15.0**, the observation clip - Shin_L.y, Hand_L.x/y, UpperArm_R.y |

Godot saturates observation slice [55:100] constantly; Isaac sits at about half. 45 of the 143
floats the policy reads arrive out of distribution in Godot.

The cause is documented and is a SOLVER-CLASS difference, not a setting: Jolt fights a joint limit
with an explicit torque, so a tightly-limited axis (`Shin_L.y` is +/-0.10 rad at kp 1800) slams into
its stop and bounces, while XPBD enforces the same limit as a rigid position constraint and does not.
Adding limit damping and zero restitution to all 45 axes changed nothing, so it is not reachable
from the joint's own limit configuration.

**This is the next thing to attack, and it is the last large measured gap.** Note that the four
previous treatments of this channel (filter, noise, mask-full, mask-narrow) all failed - but every
one of them was tried BEFORE the plant was matched, on a body whose joints were 6.5x too stiff. They
are worth re-running now, and the honest options are: reduce the chatter at its source in Godot
(nothing found yet), or make Isaac's training see the same saturated channel.

## Method rules earned the hard way

- **Measure the delivered response, not the parameter.** Every conclusion drawn from comparing
  config values has been wrong here.
- **Set `P4F_XPBD_ITERATIONS` when probing.** Iterations are part of the plant: overshoot is +10.5%
  at 2 and +0.4% at 8. Probing at the module default while training ran at 8 produced a "structural
  damping mismatch" that does not exist.
- **Check both rigs are in the same POSE before comparing per-bone anything.** A 20-step settle let
  Isaac sag off the rest pose; two rigs in different poses disagree about every lever arm, which is
  indistinguishable from a broken model.
- **Calibrate on a DOF with generous headroom in both engines, and verify on a second.** A calibration
  taken on a joint at half its travel made the thigh 50% too slow.
- **Check the sample count on a Godot run.** A run that crashes early prints the SPAWN head of 1.550,
  which reads exactly like a clean stand. 21 lines vs 69 for a full 20 s run.
- **Never use `Godot_..._console.exe`.** An export written into the Godot install folder on
  2026-08-26 left a name-matching `.pck` beside it, so that binary boots as a self-contained game and
  ignores `--path` for project settings. `IsaacPolicyDriver` prints `root ...`; empty means hijacked.
- Buffers: everything per-BODY is live under XPBD, everything joint-level or root-derived is frozen
  and reads a plausible zero. Use `p4f_newton/state.py`, never `robot.data.joint_pos`.

## 2026-09-06 18:50 — Jolt constraint stabilisation: defaults are already optimal

Deployment-side sweep, fixed checkpoint `gl151`, authority 0.125, `StanceDampingScale 2.0`, 20 s.
Properly powered (7 cells, one checkpoint), unlike the training-side fragment tests.

**First run was invalid and was discarded.** The seeded A/B killed earlier promotes each candidate to
`Models/` in order to score it, so killing it mid-run left a foreign checkpoint promoted. The sweep's
"default" row read FALLEN / 13.3% upright against a known baseline of WALK / 100%. Re-ran after
restoring `gl151`; the default row then reproduced the baseline exactly, which is the check that
validates the rest of the table.

```
setting                                  verdict  strk  upright   travel  pitch sd
default (baumgarte 0.2, slop 0.02)          WALK    12  100.0%    0.153    0.0495   <- best
baumgarte 0.05                            STATUE     6  100.0%    0.010    0.0470
baumgarte 0.5                             STATUE     7  100.0%    0.045    0.0495
baumgarte 0.9                             FALLEN     4    8.1%    0.168    0.4163
penetration_slop 0.005                      WALK    12  100.0%    0.153    0.0495   <- inert
slop 0.005 + baumgarte 0.5                STATUE     7  100.0%    0.045    0.0495   <- inert
speculative 0.005                         FALLEN     7   10.0%    0.530    0.2486
```

**Godot's Jolt defaults are already at the optimum.** Baumgarte degrades in *both* directions — lower
under-corrects into a statue, higher destabilises into a fall — which is the same one-dimensional
stride-versus-stability trade-off every other parameter has shown.

`penetration_slop` is applied and mechanically inert. Verified rather than assumed, because "applied
but inert" and "silently ignored" produce identical bit-identical rows:

- The name is real: `physics/jolt_physics_3d/simulation/penetration_slop`, enumerated from
  `ProjectSettings.get_property_list()` inside Godot (the full Jolt setting list is 33 entries).
- The write lands: with the sweep's edit in place Godot reads the setting back as `0.5`.
- An absurd 0.5 m slop is still bit-identical to default, minimum pelvis height unchanged at 0.8045 m.

The reason is `speculative_contact_distance`: at 0.02 contacts are resolved before penetration ever
develops, so the slop allowance is never reached — which is also why shrinking speculative distance to
0.005 was catastrophic (FALLEN, 10% upright) while slop changes do nothing.

Side benefit: identical configurations reproduce bit-identically, so every row that *did* differ
differs because of the setting.

## 2026-09-06 19:05 — Momentum diagnostic: the limbs move, the body does not

The external reviewer's angular-momentum diagnostic, built and run. Both engines now log the system
centre of mass, its velocity, and the whole-body **orbital** angular momentum about it,
`sum m_i (r_i - r_com) x (v_i - v_com)`, in the Isaac frame.

Orbital term only, deliberately. The spin term needs each body's inertia tensor in a common
convention — a second unverified mapping — whereas the masses are already verified identical. The
Isaac dump asserts the remaining assumption instead of trusting it: `max |body_com_pos_w -
body_link_pos_w| = 0.00 mm`, so Godot's use of body origin is exactly equivalent, and total mass
reads 80.60 kg over 16 bodies on both sides. `body_com_pos_w` and `body_com_lin_vel_w` are both on
the LIVE side of the XPBD buffer split.

### Two confounds killed before the result was read

1. **Closed-loop comparison is meaningless here.** The first run had each engine on its own policy;
   action alignment came out at 0.704, i.e. the two engines were executing different trajectories, and
   the resulting "5.3x momentum ratio" merely restated that Isaac walks and Godot does not. Redone
   open-loop from Isaac's own recorded actions: **alignment 0.000000**.
2. **A 25% action-scale mismatch.** The deployment baseline runs `ActionScaleOverride 0.125` while the
   checkpoint's restored training scale is 0.1. At 0.125 the open-loop replay collapses — pelvis
   0.828 -> 0.150 m, knee folding to -2.64 rad against a commanded ~0.4 — which reads exactly like a
   stance leg that cannot bear weight. **It is not.** At matched 0.10 the body stands the full 20 s
   and the knee range is 0.38 vs Isaac's 0.25 rad. The "buckling" was the confound.

### The result, matched scale, identical actions, both upright for the full 20 s

```
                        GODOT     ISAAC    ratio
orbital |L| (kg m^2/s)   2.08      5.74     2.76x
|dL/dt| (N m)           31.97     92.18     2.88x
COM speed (m/s)         0.0354    0.4252   12.03x
COM travel              0.009 m   3.19 m
```

And yet the joints move **as much as Isaac's**:

```
joint            godot sd   isaac sd   amplitude ratio
Thigh_L.x         0.0331     0.0333        0.99x
Thigh_R.x         0.0354     0.0546        0.65x
Shin_L.x          0.0617     0.0683        0.90x
Shin_R.x          0.0451     0.0522        0.86x
```

**Under identical joint commands, with matched amplitude of joint motion, Isaac's body travels 3.19 m
and Godot's travels 9 mm.** The limbs move; the body they are attached to does not respond. `dL/dt` is
the net external torque about the COM, so Godot's ground is delivering **one third** the torque.

### The mechanical link: the foot never leaves the floor

```
            godot min/max/sd          isaac min/max/sd        above 0.05 m
footZ_L   0.0394 0.0478 0.0003     0.0387 0.0727 0.0075     godot 0.0%  isaac 15.8%
footZ_R   0.0395 0.0478 0.0004     0.0387 0.1033 0.0171     godot 0.0%  isaac 80.2%
```

A foot pinned inside 8 mm with **sd 0.0004** over 20 s is not swinging. Both feet stay planted, so leg
motion only rearranges the body internally — no swing, no step, no net momentum, no travel. That is
the mechanism behind the 12x COM-speed gap, and it is measured here with the policy removed from the
loop, so it is not a policy artefact.

### Where this points

The hip/knee **amplitudes** match but their **correlation with Isaac does not**, and not because of
accumulated phase drift — `Thigh_L.x` correlates at +0.017 over the first 2 s, before drift can
accumulate. Identical targets in, matched amplitude out, scrambled relative phase.

That is exactly what a per-joint lag predicts: the Stable-PD denominator `1 + Kd*dt/I + Kp*dt^2/I`
divides by **that joint's** inertia, so hip and knee lag by different amounts. Foot lift depends on
hip and knee flexing *together*; each joint can track its own target with the right amplitude while
the coordination between them is destroyed. The earlier frequency sweep measured **one hip joint** and
so could not see this.

**Next test:** per-joint phase lag across the leg chain (hip / knee / ankle) rather than one joint —
if the lags differ substantially between joints in Godot and not in Isaac, that is the mechanism.

## 2026-09-06 19:10 — Per-joint transfer function: the HIP carries the lag, the knee does not

Follow-up to the momentum diagnostic. The earlier frequency sweep drove one hip joint, so it could
not see a hip-versus-knee difference. Same method — sinusoid on one joint, single-bin DFT, Godot's
one-step trace offset removed — now run across the leg chain on the **explicit torque path** (the one
that ships), matched action scale, 1/2/3 Hz.

```
                 ISAAC gain  phase | GODOT gain  phase | d(phase)
hip   (Thigh_L)
  1 Hz              0.211   -12.2  |    0.626   -23.5  |   -11.4
  2 Hz              0.299   -20.6  |    0.491   -35.3  |   -14.7
  3 Hz              0.272   -26.1  |    0.372   -45.3  |   -19.2
knee  (Shin_L)
  1 Hz              0.342    -6.3  |    0.434   -13.9  |    -7.6
  2 Hz              0.364   -10.4  |    0.441    -8.8  |    +1.6
  3 Hz              0.357   -16.2  |    0.448   -14.0  |    +2.2
ankle (Foot_L)
  1 Hz              0.068   -33.8  |    0.057   -39.3  |    -5.5
  2 Hz              0.082   -43.0  |    0.035   -40.0  |    +2.9
  3 Hz              0.048   -67.2  |    0.029   -24.5  |   +42.6
```

**The lag is not spread evenly across the chain — the hip carries essentially all of it.** The hip
lags Isaac by 11-19 deg and grows with frequency; the knee is within 2-8 deg and does not. Phase
error spread across the chain: **5.9 deg at 1 Hz, 17.7 deg at 2 Hz, 61.8 deg at 3 Hz.**

That is the signature the momentum result predicted. Each joint can track its own target with a
plausible amplitude while the *relative* timing between hip and knee is distorted — and foot lift is
precisely a hip-knee coordination, which is why the amplitudes matched (sd ratios 0.65-0.99) and the
foot still never left the floor.

The ankle row at 3 Hz (+42.6 deg) is **not** trustworthy: the gain there is 0.029, i.e. the joint is
barely moving, so its phase is noise. The ankle contributes little either way.

Second observation, weighted lower because gain is the noisy half of this measurement (Isaac's own
gain reading varies 0.23-0.35 run to run at fixed frequency): the **gain ratio also differs per
joint** — hip 2.97x, knee 1.27x, ankle 0.84x at 1 Hz. Godot's hip moves nearly three times as far as
Isaac's for the same command while its ankle moves slightly less. Note the direction: at the hip
Godot is *more* responsive, not less, which does not fit the "Godot is uniformly softer" reading of
the Stable-PD ceiling.

**This is per-joint, and the fix has to be per-joint.** A global gain, damping or lead term cannot
correct a hip that lags 15 deg while the knee lags 2 — which is why every global treatment tried so
far moved along the same stride-versus-stability trade-off instead of off it.

## 2026-09-06 19:25 — Per-joint hip correction: works at the plant, fails end-to-end

`IsaacPolicyDriver.HipLeadSteps` and `HipActionScale` apply a lead and a gain multiplier to the hip
DOFs only, resolved from the rig contract's own bone names rather than index arithmetic.

**Plant-level check first, because a treatment that does not move the number it was designed to move
should not be gait-scored at all.** It moves it, cleanly and monotonically:

```
hip phase error (Godot - Isaac)    1 Hz     2 Hz     3 Hz
  HipLeadSteps 0                  -12.5    -15.5    -18.2
  HipLeadSteps 1                   -7.0     -6.6     -7.9
  HipLeadSteps 2                   -1.8     +0.7     -0.8   <- nulled, flat across frequency
  HipLeadSteps 3                   +3.0     +6.7     +4.8
```

**End-to-end it is strictly harmful**, on the fixed `gl151` checkpoint:

```
 auth  lead  hipK |  verdict  strk  upright   travel  maxFootZ
0.125   0.0  1.00 |     WALK    12  100.0%    0.153    0.0629   <- untouched baseline, still best
0.125   1.0  1.00 |   STATUE     0  100.0%    0.006    0.0499
0.125   2.0  1.00 |   FALLEN     5   33.9%    0.709    0.6226
0.125   3.0  1.00 |   FALLEN     3   23.9%    0.537    0.3461
0.125   0.0  0.80 |   STATUE     0  100.0%    0.027    0.0499
0.125   0.0  0.60 |   FALLEN     3   43.3%    0.519    0.2900
0.125   0.0  0.40 |   STATUE     0  100.0%    0.008    0.0499
0.150   0.0  0.60 |   FALLEN     7   28.2%    0.959    0.5929
```

Also worth recording: at the **trained** authority 0.100 every lead is a STATUE, baseline included.
The working configuration needs 0.125, i.e. 25% *above* what the policy trained at.

### The pattern, now three for three

Every treatment that provably made Godot's plant more like Isaac's has made transfer worse:

1. scaling Godot's actions by the measured open-loop factor 2.474 — monotonically worse
2. hip phase lead, verified to null 18.2 deg to 0.8 deg — statue, then fall
3. hip gain scale, correcting a verified 2.97x — statue

**Plant fidelity is anti-correlated with transfer in this system.** The configuration that works is not
the Isaac-like one; it sits 25% above the trained authority with 2x stance damping. That is a result
about strategy, not about a parameter.

## 2026-09-06 19:28 — Within-engine leg coordination

Cross-engine correlations were confounded by phase drift, so this is measured entirely WITHIN each
engine, open-loop, matched scale:

```
                    GODOT L / R        ISAAC L / R
hip-knee corr      -0.300 / -0.141   +0.075 / +0.196
knee -> foot lift  -0.663 / -0.807   +0.209 / -0.751
LEG SHORTENING     0.0116 / 0.0119 m  0.0490 / 0.0527 m
```

**Godot's leg shortens by 12 mm where Isaac's shortens by 49-53 mm** — 4.3x less — even though the hip
and knee joint amplitudes match to 0.65-0.99. The hip-knee correlation changes SIGN between the
engines: Isaac's hip and knee flex slightly together, Godot's oppose. The joints move; the leg does
not fold; the foot does not clear.

This is a clean statement of the defect, and it is NOT a phase or gain error at either joint on its
own, because correcting each of those in isolation is what the sweep above just eliminated.

## 2026-09-06 19:45 — Forward kinematics: nothing is holding the foot down

Two very different explanations for a foot that never clears the floor: either the joint angles never
form a folded leg (coordination), or they do and something holds the foot (contact, friction).
Planar sagittal chain, segment lengths from the rest-pose anchors (thigh 0.32 m, shin 0.34 m),
comparing each engine's measured foot height against what ITS OWN angles predict:

```
                                   GODOT L / R        ISAAC L / R
predicted shortening (own angles)  0.0135 / 0.0171   0.0269 / 0.0296
MEASURED foot-below-pelvis range   0.0116 / 0.0119   0.0490 / 0.0527
corr(predicted, measured)          +0.664 / +0.891   +0.710 / +0.674
```

**Godot's foot sits exactly where its own hip and knee angles put it.** Predicted 13-17 mm, measured
12 mm. Nothing is holding it down — contact and friction are exonerated as the reason the foot does
not clear, which closes a line that the contact-flag observation kept suggesting.

Two further readings:

* **Isaac clears nearly twice what its planar hip+knee chain predicts** (49 mm measured vs 27 mm
  predicted), so Isaac recruits DOFs this two-link model ignores — ankle, hip abduction, pelvis
  attitude. Godot does not: its measured value sits slightly *below* its prediction.
* **Isaac's hip+knee combination alone folds the leg twice as far as Godot's** (27 vs 14 mm) even
  though the per-joint amplitudes match to 0.65-0.99. Same amplitudes, half the fold: the defect is
  in how the two joints combine, not in either one's size.

Which is consistent with the coordination measurement (hip-knee correlation flips sign, Godot -0.30 /
-0.14 against Isaac +0.08 / +0.20) — and awkward for the obvious fix, since correcting the hip's
phase and gain individually is exactly what was eliminated an hour earlier.

## 2026-09-06 19:50 — Hip-knee relative phase (suggestive, n=1)

Measured within each engine at its own dominant gait frequency, so it is immune to any common-mode
lag. Only the RIGHT leg is comparable — both engines land on 1.60 Hz there; the left picked 1.60 Hz
in Godot against 3.15 Hz in Isaac, so those two are not the same quantity and are not compared.

```
leg R, both at 1.60 Hz     hip amp   knee amp   knee-minus-hip phase
  godot                     0.0211    0.0254         -12.7 deg
  isaac                     0.0578    0.0462         +63.5 deg
```

**A 76 deg difference in the hip-knee relationship**, against a per-joint absolute lag of only 11-19
deg at the hip. If it holds up, the coordination error is several times larger than any single joint's
lag — which fits the FK result (same amplitudes, half the fold) and explains why correcting one joint
at a time did nothing useful.

Treated as suggestive, not established: n=1 leg, amplitudes are small (0.02-0.06 rad), and the
left-leg spectra were not clean enough to give a second reading. Note also that 76 deg at 1.6 Hz is
132 ms, i.e. ~8 policy steps — and a 2-step hip lead already destabilised the closed loop, so this is
not something a lead term can simply absorb.

## 2026-09-06 20:10 — Every training run's final checkpoint, scored in Godot: 1 of 62

The broadest deployment-side search this project has run. Final checkpoint of all 62 training runs,
each promoted and scored at the known-good configuration (authority 0.125, `StanceDampingScale 2.0`,
`LoadCompensation 0`, 20 s). One policy per cell, one fixed runtime, so it is properly powered by
construction. `gl151` was re-measured inside the sweep as an internal control and reproduced exactly
(WALK, 12 strikes, 100% upright, 0.153 m).

```
WALK    1
STATUE 34
FALLEN 25
```

**Nothing in the entire training history beats the incumbent, and only one checkpoint out of 62 walks
at all.** The transfer rate is 1.6%, lower than the 1-in-20 previously assumed - that earlier figure
came from `gl151`'s own siblings, which are the most favourable sample available.

Runners-up, both classified STATUE because they barely travel, but both worth noting because they
clear the foot *higher* than the incumbent does:

```
2026-09-06_00-53-56_hz960   7 strikes  100% upright  travel 0.076  maxFootZ 0.0694  <- best clearance
2026-09-06_07-07-38_as015   5 strikes  100% upright  travel 0.102  maxFootZ 0.0651
2026-09-06_09-26-16_gl151  12 strikes  100% upright  travel 0.153  maxFootZ 0.0629  <- incumbent
```

The two runners-up come from the 960 Hz physics and the action-scale 0.15 lineages. Foot clearance
and travel are not maximised by the same checkpoint, which is worth remembering: the incumbent wins on
travel while lifting its feet least of the three.

The practical consequence: **checkpoint selection is exhausted as a lever.** It was the only axis that
had ever improved the Godot result, and a 62-cell sweep across every lineage now returns the
incumbent. Further gains have to come from somewhere other than picking a better existing policy.

## 2026-09-06 20:20 — Hip lead does not restore the leg fold: the hip-lag hypothesis is dead

The hip-lead sweep failed end-to-end, which left two readings open: the coordination diagnosis is
right and the lead merely destabilises the closed loop, or the diagnosis is wrong. Open-loop replay
settles it — closed-loop stability is not in play there:

```
 lead |    foldL    foldR |  maxFootZ |   predL   predR
  0.0 |   0.0118   0.0119 |    0.0499 |  0.0135  0.0171
  1.0 |   0.0120   0.0121 |    0.0499 |  0.0135  0.0170
  2.0 |   0.0122   0.0123 |    0.0499 |  0.0135  0.0172
  3.0 |   0.0122   0.0123 |    0.0499 |  0.0134  0.0176
ISAAC reference: fold 0.0490 / 0.0527
```

**A lead that provably nulls the hip phase error moves the leg fold by 4% and the foot clearance not
at all.** `maxFootZ` is pinned at 0.0499 in every cell. The hip phase lag is real, correctable, and
**not the cause of the missing foot clearance** - it is dead as a mechanism, not merely unstable in
closed loop.

Note the *predicted* fold is flat too (0.0135 throughout), which is the reason: a lead shifts phase,
and fold range depends on the angles' magnitude and combination, not their timing.

### Where the fold actually goes: operating pose

Fold is `L1*cos(h) + L2*cos(h+k)`, quadratic in the angles near zero, so the mean pose matters more
than the oscillation - the same AC amplitude gives far more vertical travel about a flexed leg than a
straight one.

```
        hip mean  hip sd  knee mean  knee sd   mean drop
leg L
 godot   0.1509   0.0331    -0.1751   0.0617     0.6555
 isaac   0.1985   0.0333    -0.0427   0.0683     0.6484
leg R
 godot   0.0764   0.0354    -0.2281   0.0451     0.6545
 isaac  -0.0450   0.0546    -0.2548   0.0522     0.6429

counterfactual - godot's OWN oscillation re-centred on isaac's mean pose:
  leg L  0.0135 -> 0.0204   (isaac 0.0269)
  leg R  0.0171 -> 0.0361   (isaac 0.0296)
```

Suggestive, and **explicitly not acted on**: the two legs disagree - one lands below Isaac and one
above - and Isaac's own left and right knee means differ by 0.21 rad, so the means are not well
estimated over a 20 s window. Fitting a per-leg DC offset to these numbers would be fitting noise.
Recorded because "the pose, not the oscillation" is a different class of explanation from everything
tried so far, and is worth a properly designed test rather than a quick offset.

## 2026-09-06 20:48 — Per-checkpoint deployment tuning: one working cell, shared by all three

The 62-run sweep held the runtime fixed at the incumbent's tuning, which is the right control for
comparing policies but cannot say how good each policy could be at its own optimum. Top three
checkpoints x authority {0.100, 0.125, 0.150} x `StanceDampingScale` {1.5, 2.0, 3.0}, 27 cells.

```
             auth 0.125 / damp 2.0        every other cell
gl151        WALK   12 strikes  0.153 m   statue or fall
hz960        STATUE  7 strikes  0.076 m   statue or fall
as015        STATUE  5 strikes  0.102 m   statue or fall
```

**All three checkpoints have the same optimum, and it is a single cell.** Authority 0.100 is a statue
for every policy; 0.150 falls for every policy; damping 1.5 falls and 3.0 freezes. The one near-miss
is `gl151` at 0.150/3.0 - FALLEN with 12 strikes at 75.2% upright.

That reframes the operating point: it is a property of the **runtime configuration**, not of the
policy. The configuration defines a narrow window, and the question a checkpoint has to answer is
whether it happens to be compatible with that window - which 1 of 62 is. It also matches the
bistability already on record: frozen below, falling above, with almost no gait in between.

Combined with the 62-run result, **both deployment-side levers are now exhausted**: no better
checkpoint exists in the training history, and no better configuration exists around the working
point. Remaining gains have to come from training.

## 2026-09-08 12:05 — Per-joint plant randomisation: implemented, verified, marginal

The strategy change the measurements pointed to: stop correcting Godot toward Isaac (three verified
attempts, all harmful) and instead make no single per-joint response learnable.

`per_joint_action_scale_range` and `per_joint_latency_steps` on `StandEnv`, both in the **Python
action pipeline** because solver-side writes are inert on Newton/XPBD. Per-joint gain multiplies the
whole-body draw; per-joint latency reads the existing ring buffer with a per-(env, joint) gather.
Both added to `TRAINED_CONDITIONS` so an eval cannot silently score them at nominal.

**Verified operative before any GPU time was spent** - the discipline that made this session's
eliminations strong:

```
TRAINING  joint scale spread ACROSS JOINTS within env 0: 0.630..1.504 (sd 0.2418)
          joint delay spread ACROSS JOINTS within env 0: 0..3
PLAYBACK  joint scale 1.000..1.000 (sd 0.0000), joint delay None
```

Fine-tuned from `gl151` (the one transferring policy), 30 min, ranges (0.6, 1.6) and (0, 3), giving
**52 checkpoints**, all scored in Godot at the working cell.

### Result: marginal, and one retraction

```
stepping (upright >= 90%, strikes > 0):  4/52 = 7.7%
```

Against the recorded ~5% rate at which fine-tuning fragments retain stepping regardless, **this is not
a significant lift** (expected 2.6, observed 4, p ~ 0.25). Reported as such rather than as a 5x
improvement over the 1.6% cross-run base rate, which would be the wrong comparison - that base rate
is across unrelated lineages, and this is a fine-tune of a known-good checkpoint.

**`model_37026` looked like a breakthrough and is not.** At 20 s: HOP, 12 strikes, 97.9% upright,
**1.295 m** - 8.5x the incumbent. At 40 s:

```
FALLEN  12 strikes  49.0% upright  travel 1.293 m  pelvis ends at 0.133 m
quarter travel: 0.136 / 1.242 / 0.099 / 0.064 m
```

All the travel happens between 10 and 20 s and the body then topples. **The 20 s uprightness gate
caught a body accelerating into a fall while still nominally upright.** Third instance of this
confound class; the gate has to be the full run, not a window.

**`model_36977` is a genuine but small improvement.** At 40 s: STATUE, 9 strikes, **100% upright**,
pelvis min 0.800, travel **0.190 m** and maxFootZ **0.0764** - beating the incumbent's 0.153 m and
0.0629 on both. But the travel is 0.187 / 0.013 / 0.002 / 0.000 m by quarter: it takes a few steps in
the first ten seconds and then stops. Same qualitative failure as everything before it.

Both candidates re-confirm the single working cell: authority 0.125 / damping 2.0, statue at 0.100,
fall at 0.150.

**Verdict: per-joint randomisation is not the fix.** It buys a marginally better checkpoint and no
change in kind. The dummy still does not walk in Godot.

## 2026-09-08 12:18 — Where the foot clearance goes: cancellation, not a missing DOF

Chasing the one unexplained measurement — Isaac clears nearly twice what its planar hip+knee chain
predicts, Godot slightly less than its own. Standardised regression of foot-below-pelvis height on
every leg DOF plus pelvis attitude, open-loop matched-scale pair, R^2 0.94-0.98 so the model is
adequate:

```
leg L            GODOT range   beta | ISAAC range   beta
Thigh_x               0.2265   1.81 |      0.2870   0.57
Shin_x                0.3745   1.95 |      0.2813   0.86
Foot_x                0.2334   1.79 |      0.2911   0.51
Thigh_z               0.0819   0.01 |      0.1801  -0.18
pelvis_pitch          0.1382  -1.52 |      0.1317   0.81
```

**Godot's standardised betas are 2-4x Isaac's while its net clearance is 4x smaller.** Large opposing
contributions that cancel; Isaac's are modest and combine. That is the quantitative form of "same
amplitudes, half the fold" — it is cancellation, not a missing degree of freedom.

`pelvis_pitch` enters with **opposite sign** (+0.81 Isaac, -1.52 Godot) at near-identical pitch range.

### Three follow-ups, all negative

**The balance layer is not suppressing clearance.** It was eliminated on 2026-09-05 on UPRIGHTNESS,
which is not the failing quantity, so it was retested on clearance:

```
BalanceAssist |   foldL   foldR | maxFootZ | pelvisZ mean
         1.00 |  0.0118  0.0119 |   0.0499 |       0.8149
         0.50 |  0.0113  0.0113 |   0.0499 |       0.8152
         0.00 |  0.8884  0.7985 |   0.2320 |       0.1770   <- FALLEN, artifact
```

Halving it does not help; removing it collapses the body. The 0.888 fold at 0.0 is a falling body and
is not clearance. Elimination now holds on the right metric.

**No joint is limit-saturated.** Worst is `Shin_L.z` at 76.6% of its span; nothing binds. One small
oddity: `Shin_R.y` reaches 0.114 against a stated limit of 0.100.

**No sign bug.** `Foot_L.z` looked inverted in the gait trace (corr −0.727 between engines, action to
achieved −0.236 Godot vs +0.378 Isaac, and left/right asymmetric — the shape of the two sign bugs
already found in this rig). Single-DOF sinusoid drives say otherwise:

```
dof             ISAAC gain   GODOT gain   verdict
Shin_L_rx          +0.1050      +0.0911   OK
Thigh_L_rz         +0.0388      +0.0229   OK
Thigh_R_rz         +0.0230      +0.0224   OK
Foot_L_rz          -0.0079      +0.0020   inconclusive - near-zero in BOTH
Foot_R_rz          -0.0087      +0.0020   inconclusive - near-zero in BOTH
```

**Ankle roll is inert in both engines** (|gain| <= 0.009 against 0.105 at the knee), so the -0.727 was
a correlation between two noise signals. Dismissed rather than reported.

### What survives

One quantified difference: **Godot's hip rotation responds ~60% as strongly as Isaac's** (signed gain
+0.0229 vs +0.0388) and uses less than half the range in gait (8.2% of span vs 18.0%). That is the
only DOF-level asymmetry left standing, and it is a magnitude difference, not a missing capability.

## 2026-09-08 12:40 — Pelvis-pitch sign flip is emergent, not structural; and Godot stands SPLIT

External review made a fair methodological hit: the clearance regression is observational, R^2 of
0.94-0.98 can encode correlation rather than causality, and the global fit pools all gait phases.

Settled without a perturbation run. For an instantaneous KINEMATIC Jacobian, holding joint angles and
pitching the body by d(theta) moves a point at forward offset dx by ~ dx*d(theta), so
`sign(dz_foot/d_pitch) == sign(foot's forward offset from the pelvis)`. Tested WITHIN each engine
(forward axis recovered from each body's own travel, so no cross-engine frame mapping is assumed),
phase-conditioned into quartiles of the foot-height cycle:

```
signs agree in 4 of 8 Godot quartiles and 1 of 8 Isaac quartiles - i.e. chance
```

**The pitch-clearance relationship is not an instantaneous kinematic effect in either engine.** It is
an emergent correlation, so the sign flip is dropped as a structural lead - which is the reviewer's
own stated criterion for abandoning it.

### The same measurement found something real: a split stance

```
mean fore-aft foot offset from the pelvis
  GODOT   leg L  +0.1184 m   leg R  -0.0873 m   -> 0.2057 m fore-aft split
  ISAAC   leg L  -0.0145 m   leg R  +0.0190 m   -> 0.0335 m, both feet under the pelvis
```

**Godot holds its feet straddled fore-aft, 6x wider than Isaac.** That is a posture difference, not a
timing one, and it is mechanically sufficient on its own to prevent a swing: a leg carrying load in a
split stance cannot leave the ground until the body unloads it, and the body only unloads it by
shifting weight over the other foot - which is the travel that never happens.

It also matches the earlier operating-pose result (leg fold is quadratic in joint angle, so mean pose
dominates clearance) and the "takes a few steps then stops" signature of every checkpoint that
survives 40 s. Recorded as the most promising remaining observation; NOT yet acted on, because
posture-offset corrections are a class this project has not tested and the last three correction
classes all failed.

## 2026-09-08 12:55 — CORRECTION: the open-loop comparison was asymmetric

Found while setting up the solver-backend replay the external reviewer asked for. `replay_isaac.py`
never set `cfg.playback`, so it pinned `action_scale_range` by hand and left `effort_scale_range`
live. Fixed (playback pins every per-episode draw). Re-running then exposed a bigger problem with the
comparisons built on top of it.

**Isaac cannot follow its own recorded actions open-loop either.**

```
                    pelvis mean   min    above 0.7 m   maxFootZ
ISAAC closed-loop      0.820     0.801     100.0%       0.1033
ISAAC open-loop        0.769     0.411      89.9%       0.5640
GODOT open-loop        0.815     0.808     100.0%       0.0499
```

Isaac's open-loop replay repeatedly half-collapses and its feet flail to 0.56 m. So every
"identical actions" comparison since 2026-09-06 put Isaac's **closed-loop** trajectory against Godot's
**open-loop** replay. The action sequences were identical - alignment 0.000000 was real - but the two
sides were not doing the same thing, and Isaac's 3.19 m of travel had feedback producing it.

### Redone symmetrically, open-loop vs open-loop

The headline conclusion is **unchanged and larger**:

```
                   GODOT     ISAAC open-loop   ratio
orbital |L|         2.08          10.64        5.12x   (was 2.76x vs closed-loop)
|dL/dt| (N m)      31.97         109.12        3.41x
COM speed           0.035          0.616       17.42x
COM travel          0.011 m        0.325 m     28.74x
leg fold L          0.0116         0.6536      56x
maxFootZ            0.0478         0.5640
```

### But one specific claim was WRONG and is retracted

**"Same joint amplitudes, half the fold" is not correct.** Amplitudes were measured against Isaac
closed-loop:

```
joint         GODOT   ISAAC closed  ISAAC open   ratio vs closed   ratio vs open
Thigh_L.x    0.0331     0.0333       0.0725          0.99             0.46
Shin_L.x     0.0617     0.0683       0.0793          0.90             0.78
Thigh_R.x    0.0354     0.0546       0.0828          0.65             0.43
Shin_R.x     0.0451     0.0522       0.0803          0.86             0.56
```

Symmetrically, **Godot's joints move roughly HALF as much** (0.43-0.78x), not the same amount.

The gap does not close, though. Leg fold is quadratic in joint angle, so half the amplitude predicts
about a QUARTER of the fold - and Godot delivers **1/56th**. A super-linear factor of roughly 14x
remains unexplained, and the clearance regression redone symmetrically still shows Godot's per-DOF
betas (Thigh_x 2.94, Shin_x 4.13, Foot_x 3.44, pelvis_pitch -2.41) an order of magnitude above
Isaac's (0.00, 0.25, 0.33, 0.65).

So the *cancellation* reading survives; the *"same amplitudes"* premise does not. Everything the
external reviewer wrote in response to update #6 was built on that premise and needs re-basing.

### Featherstone is numerically unstable on this rig

The reviewer's primary recommendation, run on the same actions: **5 NaN rows**, `pelvisZ` mean NaN.
Featherstone does not currently produce a usable trajectory here, so "does Featherstone move the plant
toward Jolt?" cannot be answered until that is diagnosed. The two non-NaN metrics both moved toward
Godot (COM travel 0.40x the XPBD gap, pitch sd 0.52x), which is suggestive but is two numbers from a
run that also produced NaNs, and is not worth acting on.

## 2026-09-08 13:08 — CAPABILITY: Godot cannot lift a foot and stay upright, at any setting

The most basic question in the investigation, and it had never been asked directly: **can Godot's body
raise a foot when explicitly told to?** No policy, no gait, no feedback - a sustained maximal one-leg
flexion (`Thigh_L_rx` +1.0, `Shin_L_rx` -1.0, 1 s ramp then hold), replayed open-loop.

```
setting                       maxFootZ_L   rise above rest   pelvisZ min
auth 0.125 damp 1.0               0.0499        0.0000          0.8127
auth 0.125 damp 1.5               0.0499        0.0000          0.8122
auth 0.125 damp 2.0               0.0499        0.0000          0.8117
auth 0.125 damp 3.0               0.0499        0.0000          0.8102
auth 0.125 balance 0.5            0.0499        0.0000          0.8122
auth 0.125 SpdCompensation 0.25   0.0499        0.0000          0.8118
--- everything below LIFTS THE FOOT AND FALLS ---
auth 0.125 balance 0.0            0.1636        0.1137          0.1200
auth 0.175                        0.1563        0.1064          0.2064
auth 0.250                        0.2029        0.1530          0.1299
auth 0.400                        0.2108        0.1609          0.1292
SpdCompensation 0.50              0.2002        0.1503          0.1348
SpdCompensation 0.75              0.4458        0.3959          0.2079
SpdCompensation 1.00              0.2513        0.2014          0.1184

ISAAC at its TRAINED scale 0.1    0.0613        0.0216          0.7813   <- lifts AND stays upright
```

**The foot rise is exactly 0.0000, bit-identical, across a 3x damping range, at half balance assist,
and at SpdCompensation 0.25.** The only thing that ever moves the foot is the body falling over.
Isaac, at *lower* authority than Godot's, lifts the foot 21-26 mm with the pelvis at 0.78.

This is the project's central wall stated as a capability rather than an outcome:

```
Godot can keep the body upright   OR   lift a foot.   Never both.
Isaac does both at once, with less commanded authority.
```

It subsumes a long list of earlier observations: why every treatment slid along one
stride-versus-stability trade-off, why the best configuration marches in place at 100% upright with no
clearance, why 1 checkpoint in 62 "walks" (they are all marching), and why matching the plant never
helped - the plant genuinely cannot perform the motion.

It also puts the Stable-PD ceiling back on the table, but with a precise shape. `SpdCompensation`
restores the authored gains and **does** buy the lift (0.15-0.40 m of rise, so the authority argument
is correct) - and destabilises the body every time, exactly as recorded on 2026-09-05, now confirmed
against a clean capability metric instead of a gait outcome.

**Next, and directly implied:** compensation is applied to EVERY joint, including the stance leg and
the joints the balance controller drives - and an explicit-torque PD at restored gain is precisely the
formulation that goes unstable. The untested variant is compensation on the **swing leg only**, which
is unloaded and far less prone to that instability. The driver already discriminates swing from stance
(`IsSwingLegBone`, `SwingActionScale`, stance/swing damping split), so this is a small change.

### Swing-only SPD compensation: one real gain, but the test is confounded

`IsaacPolicyDriver.SwingSpdCompensation` applies the compensation to the swing leg's chain only,
reusing the stance side `ApplyPhaseDamping` already resolves.

```
lifting the LEFT leg          maxFootZ   rise     pelvisZ min
  SwingSpdComp 0.25            0.0499   0.0000     0.8126
  SwingSpdComp 0.50            0.0499   0.0000     0.8088   <- whole-body 0.5 FELL here
  SwingSpdComp 0.75            0.5604   0.5105     0.1191
  SwingSpdComp 1.00            0.1576   0.1077     0.1283

lifting the RIGHT leg
  SwingSpdComp 0.25            0.0499   0.0000     0.8124
  SwingSpdComp 0.50            0.2023   0.1524     0.2168
  SwingSpdComp 0.75            0.1796   0.1297     0.1197
  SwingSpdComp 1.00            0.0499   0.0000     0.8086
```

**One genuine gain:** swing-only compensation at 0.5 keeps the body upright where whole-body
compensation at 0.5 fell, so confining the restored gain to the unloaded leg does buy stability.

**But no cell delivers lift AND upright**, and the right-leg sweep is **non-monotonic** - 0.5 lifts and
falls while 1.0 gives zero lift and stays upright. That is the stance gate flipping between runs:
with neither foot leaving the ground, "swing" is ambiguous and the compensation lands on whichever leg
the height test happened to pick. The experiment cannot distinguish the treatment from the gating, so
no conclusion is drawn from it beyond the stability observation.

To do this properly the swing leg has to be named explicitly rather than inferred, which the current
export cannot express.

## 2026-09-08 13:32 — THE MECHANISM: the STANCE knee buckles under single-leg load

`ForcedSwingSide` was added so a capability probe can NAME the leg it is lifting instead of inferring
it from foot heights - the gate is undefined when neither foot leaves the ground, which is exactly the
condition under test, and it had made the previous sweep non-monotonic.

With the swing leg named, the picture is unambiguous. Commanded one-leg lift, swept over amplitude:

```
lift amp   max rise   rise while upright   pelvis min   % upright   stance knee
   0.1      0.0000        0.0000             0.8174      100.0%       -0.047
   0.2      0.0000        0.0000             0.8174      100.0%       -0.048
   0.3      0.0000        0.0000             0.8174      100.0%       -0.051
   0.5      0.1266        0.0000             0.1406       37.9%       -2.631   <- BUCKLES
   0.75     0.4652        0.0093             0.1975       33.3%       -2.623   <- BUCKLES
   1.0      0.1017        0.0213             0.1225       40.5%       -0.871   (topples, pitch 1.00)
ISAAC 1.0    0.0258        0.0258             0.7778      100.0%
```

**When Godot lifts one foot, the STANCE knee collapses to -2.63 rad (-151 deg).** The supporting leg
folds. This is a quasi-static SUPPORT failure, not a coordination, timing or observation failure, and
it is the first mechanism found that is upstream of everything else measured.

The arithmetic is consistent with it. Holding 80.6 kg on one leg puts roughly 790 N through a shin of
~0.34 m, so the knee needs on the order of **270 N.m/rad** to be stable against buckling - below that
the gravitational moment grows faster with flexion than the restoring moment and the joint runs away.
The driver's own live readout is `kp 882 -> 148` after the Stable-PD denominator. Isaac drives the
authored 1200-1600. **Godot's knee is under the critical stiffness for single-leg support and Isaac's
is far above it.** (Order-of-magnitude estimate, and the 148 is the body-wide worst-case figure the
driver prints rather than a knee-specific measurement - worth pinning down before it is built on.)

This subsumes the capability result above: the reason a maximal lift command produces 0.0000 m of rise
at every upright setting is that any real weight transfer collapses the other leg, so the only stable
configuration is both feet planted.

### Stance-only compensation gets closest, and still does not hold

`SwingSpdCompensation 0` with `SpdCompensation` on the stance chain:

```
stance comp   max rise   rise while upright   pelvis min   % upright
    0.00       0.0000        0.0000            0.8117      100.0%
    0.40       0.0000        0.0000            0.8102      100.0%
    0.55       0.3121        0.0000            0.2085       75.2%
    0.70       0.1587        0.0840            0.1195       18.1%
    1.00       0.4597        0.0242            0.1987       71.9%
ISAAC          0.0206        0.0206            0.7822      100.0%
```

At 1.0 it reaches **0.0242 m of clearance while upright, matching Isaac's 0.0206**, and at 0.7 it
briefly reaches 0.084 m - the first time Godot has cleared its foot at Isaac-comparable height without
already having fallen. Neither holds; the body goes down later in the run. Note also that this sweep
conflates the stance leg with the torso, because the startup pass applies `SpdCompensation` to every
controlled bone and only the leg chains are overridden per step.

### What this implies

The fix is not more compensation tuning. It is a stance leg that can HOLD, and an explicit-torque PD
cannot be driven to the required stiffness without going unstable - that is what the Stable-PD
denominator exists to prevent. The untried configuration that fits the diagnosis is a **hybrid drive**:
Jolt's `angular_spring`, which is resolved INSIDE the constraint solver and is therefore stable at
stiffnesses the explicit path cannot reach, on the STANCE leg, with the explicit path kept for swing.
The spring path already measured 21x stride open-loop with its residual in torso pitch rather than in
the joints.

## 2026-09-08 13:55 — CORRECTION to the buckling arithmetic, and gravity feed-forward is the cause

**The stiffness estimate in the entry above was wrong.** The `[PLANT]` table the driver already emits
on the first step gives the exact per-bone figures:

```
bone      kp     kd    inertia    denom    kpEff     scale
Thigh   1600     32    0.0805     3.00     533.2     0.333
Shin    1800     36    0.0481     4.77     377.3     0.210
Foot    1200      6    0.0091     6.02     199.4     0.166
```

Godot's knee is **kpEff 377.3**, not the 148 I quoted - 148 is the body-wide worst case the driver
prints, which belongs to a different joint. Against a single-leg buckling load of roughly
`648 N x 0.34 m ~ 220 N.m/rad`, **377 is above critical and the knee should be statically stable.**
So the static-stiffness explanation is wrong and is withdrawn.

### The actual cause: the gravity feed-forward was switched off

Same capability probe, sweeping `LoadCompensation` and the Hill law:

```
treatment                 rise upright   pelvis min   % upright   stance knee
loadComp 0.0  hill x1        0.0213        0.1225      40.5%       -0.871   buckles
loadComp 0.5  hill x1        0.0163        0.1945      31.9%       -0.919   buckles
loadComp 1.0  hill x1        0.0000        0.7828     100.0%       -0.181   HOLDS
loadComp 0.0  hill x100      0.0213        0.1295      40.5%       -0.769   Hill irrelevant
loadComp 1.0  hill x100      0.0000        0.7828     100.0%       -0.181
```

**`LoadCompensation` is what holds the stance leg up.** With it at 1.0 the knee sits at -0.18 rad and
the body stays upright through the whole probe; at 0.0 and 0.5 it collapses. The Hill force-velocity
law is not involved - x100 is byte-identical.

And **`LoadCompensation = 0.0` is in the working configuration and in every sweep this project has
run**, which is a long-standing note in these logs ("LoadCompensation=0 was wrong") that was never
followed through. It is the setting that makes single-leg support impossible, which is why the body
can only ever stand on two feet and march.

### And restoring it still does not produce a gait

Because the working cell was tuned with the feed-forward off, the operating point was re-searched
rather than inherited: `LoadCompensation 1.0` x authority {0.075, 0.10, 0.125, 0.15} x
`StanceDampingScale` {1.0, 1.5, 2.0, 3.0}, 40 s, travel per quarter.

**16 of 16 cells fail.** Every cell is either a STATUE at 100% upright with ~0.03 m of travel, or
FALLEN at 1.4-6.2% upright. Nothing walks. For comparison the incumbent, with the feed-forward OFF,
is WALK / 12 strikes / 100% upright / 0.154 m.

So the fourth mechanistically-justified plant fix in this project behaves like the first three:
it corrects the defect it targets, at the plant level, verifiably - and the closed loop gets worse.
The pattern in [[plant-fidelity-anticorrelates]] now has a fourth instance, and this one is not a
fidelity argument but a plain mechanical one, which makes it the most surprising of the four.

## 2026-09-08 15:41 — Training WITH the gravity feed-forward: 0 of 40

The one configuration the buckling result pointed at: train on a plant whose stance leg can hold, and
deploy on that same plant. Isaac already reproduces Godot's feed-forward (`p4f_newton/gravity_ff.py`)
and `gravity_feedforward` is in `TRAINED_CONDITIONS`. The incumbent's `params/env.yaml` confirms it
trained at **0.0** - so Isaac and Godot have always been consistent, but consistent in the
configuration where single-leg support is impossible.

Verified operative before spending GPU time (same checkpoint, `gravity_feedforward` 0.0 vs 1.0):
leg joints differ by up to **0.21 rad**, pelvis height by 0.031 m.

Fine-tuned from `gl151` with `gravity_feedforward = 1.0`, 30 min, 1000 iterations, **40 checkpoints**,
all scored in Godot at `LoadCompensation = 1.0` to match.

```
sustained (upright >= 90%, strikes > 0, travel in every quarter):   0 / 40
upright >= 90% at all:                                             1 / 40   (a statue, 0 strikes)
incumbent, feed-forward OFF:      WALK  12 strikes  100% upright  0.154 m
```

**39 of 40 fall.** The gravity feed-forward path is closed on the evidence available: it demonstrably
fixes the stance-leg collapse at the plant level, and produces no gait either with the existing policy
(16 of 16 configuration cells) or with one fine-tuned for it (0 of 40 checkpoints).

**Caveat worth stating:** this was a 30-minute fine-tune from a policy trained on a materially
different plant, not a from-scratch run. Adapting to a changed plant may simply need more than 1000
iterations, so this is evidence against the cheap version of the fix rather than proof against the
fix. A from-scratch training run is hours, and is the only way to settle it.

That makes **five** mechanistically-justified plant corrections that behave the same way: each fixes
the defect it targets, verifiably, and the closed loop gets worse or stays broken.

### Method note

Scoring switched to **two-stage**: screen every checkpoint at 20 s, and re-run only those still
upright with strikes at 40 s. Most checkpoints are unambiguous statues or falls well inside 20 s,
while the 40 s gate is only needed to catch a topple in progress. Roughly halves a sweep at no cost to
the verdict.

## 2026-09-08 15:45 — MJWarp plant comparison: closer than XPBD on 6 of 10 metrics

Featherstone NaNs, so the reviewer's solver comparison was run on MJWarp instead - which is also the
backend current Isaac Lab documentation calls the primary validated Newton path. Same recorded
actions, same restored conditions, open-loop into all three, compared against Godot's open-loop replay
(the symmetric pair, after the `playback` fix).

```
metric              GODOT   ISAAC xpbd   ISAAC mjwarp   closer to Godot?
|L|                2.0782      10.6357         7.7123   YES  (0.66x the XPBD gap)
|dL/dt|           31.9447     109.1771        87.3046   YES  (0.72x)
COM speed          0.0354       0.6158         0.5931   YES  (0.96x)
COM travel         0.0113       0.3271         0.4426   no   (1.37x)
maxFootZ           0.0499       0.5640         0.1309   YES  (0.16x)
fold L             0.0118       0.6536         0.5390   YES  (0.82x)
fold R             0.0119       0.7013         0.5322   YES  (0.75x)
split stance       0.2057       0.1960         0.1514   no   (5.61x)
hip-knee corr L   -0.3067       0.0657         0.3771   no   (1.84x)
pitch sd           0.0252       0.2443         0.1396   YES  (0.52x)

solver             rows   pelvisZ mean   min     NaN rows
xpbd               1200      0.769      0.411       0
featherstone       1200        nan        nan       5
mjwarp             1200      0.729      0.367       0
```

The sharpest difference is **foot clearance**: MJWarp's open-loop replay reaches 0.131 m against
XPBD's 0.564 m, i.e. it does not flail. Isaac's own closed-loop gait peaks at 0.103 m, so MJWarp
open-loop stays near the intended motion where XPBD diverges into thrashing. It is a better
CONDITIONED integrator for this rig regardless of what it does for transfer.

**Caution on the framework.** "Closer to Godot" has been anticorrelated with transfer five times on
this project, so solver proximity is weak evidence for picking a training backend - which is the
reviewer's stated rationale. The load-bearing argument for MJWarp is that it is numerically
well-behaved and validated, not that its plant resembles Jolt's. Transfer is being tested directly
rather than inferred.

## 2026-09-08 16:16 — MJWarp training: 18 of 18 statues, 0 falls

Fine-tuned from `gl151` on the MJWarp backend, 20 min, 450 iterations (MJWarp is ~2.2x slower per
iteration than XPBD), **18 checkpoints**, scored in Godot at the working cell with two-stage screening.

```
sustained (upright >= 90%, strikes, travel in every quarter):   0 / 18
upright >= 90%:                                                18 / 18
verdicts:                                    18 STATUE, 0 FALLEN, 0 WALK
maxFootZ:                    0.0499 on 17 of 18 (the resting height - the foot never leaves the floor)
travel:                                             0.010 - 0.043 m
incumbent (XPBD):                     WALK, 12 strikes, 100% upright, 0.154 m
```

**Every MJWarp checkpoint is a statue.** That is a distinct signature from the XPBD lineages, which
produce a mix of statues and falls: MJWarp policies are uniformly conservative, holding the body
perfectly upright and never lifting a foot. None falls, and none walks.

So the solver-proximity argument does not survive its own test. MJWarp's plant is measurably closer to
Godot's on 6 of 10 metrics - notably foot clearance, where its open-loop replay reaches 0.131 m against
XPBD's 0.564 m - and its policies transfer WORSE than XPBD's, not better. That is a sixth instance of
plant proximity failing to predict transfer, and this time proximity was the explicit selection
criterion.

**Same caveat as the gravity-feed-forward run:** 450 iterations of fine-tuning onto a changed plant is
a small adaptation budget, so this is evidence against the cheap version rather than proof against
MJWarp. It is also a smaller sample (18 checkpoints against a ~5% base rate expects ~1 hit), so it can
detect a strong effect and not a subtle one. What it does rule out is MJWarp being an easy win.

## 2026-09-08 17:00 — PHASE 0, first result: the leg lifts fine. The BODY never transfers weight.

Phase 0 asks whether ANY authored Godot/Jolt configuration can lift a foot and stay upright - the
precondition for every architecture except replacing the physics. 13 configurations (explicit torque
x LoadCompensation x SpdCompensation, and the solver-resolved angular spring at k x1/2/4/8), scripted
one-leg lift ramped over 1 s and held for 10 s, gate = 0.02 m clearance with pelvis >= 0.78 held 3 s.

**13 of 13 fail, and 11 report clearance of exactly 0.0000** - an identical bit-exact zero across very
different drives, which is the signature of a measurement artifact rather than a physical result. It
was checked rather than recorded, and the check overturned the interpretation.

```
spring k4 d2, commanded hip target ~0.26 rad
  t     Thigh_L.x   Shin_L.x   footZ_L   pelvisZ
0.00      -0.000     -0.000     0.0498    0.8298
0.50       0.126     -0.157     0.0410    0.8198
1.00       0.255     -0.323     0.0446    0.8160
6.00       0.262     -0.330     0.0449    0.8156
9.98       0.261     -0.330     0.0449    0.8157
```

**The leg flexion is near-perfect.** The angular spring tracks the hip to 0.2625 rad against a 0.26
target - 0.002 rad of error - and folds the knee to -0.33. The foot nonetheless stays on the floor,
and the **pelvis drops 1.4 cm instead**: the body squats by the amount the leg shortens.

**So the missing capability is not lifting a leg. It is WEIGHT TRANSFER.** Nothing moves the body's
mass over the other foot, both legs stay loaded, and flexing one simply lowers the whole body.

That unifies several previously separate findings into one mechanism:

* the COM is 17x slower and travels 9 mm against Isaac's 3.19 m
* the feet sit in a 0.21 m fore-aft split with both permanently loaded
* double support 95.9% of the time against Isaac's 6.2%
* per-DOF influences on clearance that are large and cancel

All of these are what a body that cannot move its centre of mass over one foot looks like.

**And it means the probe was under-specified.** It commanded a leg flexion with no lateral weight
shift, which is not how anything lifts a foot - a human shifts weight sideways first. The Phase 0
gate cannot be evaluated until the scripted primitive includes a weight shift, so **no configuration
has actually been ruled out yet.** The next step is to establish which DOFs move the COM laterally in
Godot and how far they can move it, then re-run Phase 0 with a shift-then-lift primitive.

## 2026-09-08 17:10 — PHASE 0 status: weight shift SOLVED, single-leg support still not

Continuing Phase 0 after the discovery that the probe was under-specified (a leg flexion with no
weight shift). Three further rounds, 28 more configurations, 41 in total.

### New capability found: the body CAN shift its weight

Hip roll (`Thigh_*_rz`) is the lateral-shift degree of freedom; every other candidate is at noise
level:

```
dof            sign   d comY    |shift|      (needed ~0.14 m toward the stance foot)
Thigh_L_rz      +1   +0.0516    0.0516   <- and footZ_L rises to 0.0523
Thigh_R_rz      -1   -0.0521    0.0521
Spine_rz        +1   +0.0077    0.0082
Thigh_L_ry      +1   +0.0012    0.0049
```

At the deployment authority of 0.125 the commanded roll is only 0.025 rad, because the joint limit is
[-0.8, +0.2] and the action maps onto that span. **That cap is a policy-transfer constraint, not a
plant one, and a scripted controller is not bound by it.** Raising it:

```
authority   d comY   hip roll   pelvis min   % upright
   0.125   +0.0516     0.1019      0.8173      100.0%
   0.250   +0.1048     0.2104      0.8173      100.0%   <- best
   0.500   +0.0490     0.4164      0.8091      100.0%
   1.000   +0.0868     0.8020      0.7717       17.3%
```

**The COM shifts 0.105 m toward the stance foot at 100% uprightness with the pelvis steady at 0.817.**
That is a real weight transfer and it is new - the capability the body appeared to lack is available,
it was simply never commanded and was throttled by the deployment authority.

### But single-leg support still fails, in a strictly binary way

Shift + gentle lift, sweeping lift amplitude, knee ratio, shift-to-lift delay, `LoadCompensation`, and
an explicit stance-leg extension - 24 cells:

```
 lift  stanceExt  loadComp |   peak    final  pelvis min  % upright  HELD
 0.25       0.00      0.00 | 0.0262   0.0017      0.8173    100.0%   0.17
 0.40       0.30      0.00 | 0.2220   0.1552      0.1240     47.0%   0.42
 0.25       0.00      1.00 | 0.0203  -0.0017      0.8170    100.0%   0.02
 0.40       0.30      1.00 | 0.2264   0.1592      0.1239     45.3%   0.40
```

Every cell is one of two outcomes: **peak clearance 0.020-0.026 m at 100% upright with the foot back
on the floor by the end**, or **peak 0.16-0.22 m with the pelvis at 0.124** - fallen. Nothing sustains
clearance while upright, and the gate (0.02 m held 3 s) is failed by all 41 configurations.

Note the peak clearance is bit-identical at 0.0262 across lift amplitudes 0.15/0.25/0.40, so it is
produced by the SHIFT tipping the body, not by the lift - consistent with the leg-lift-squats-the-body
mechanism found at the start of Phase 0.

### Phase 0 is not yet decided

41 configurations fail, but the authored-configuration search is not exhausted. Still untried, and all
of them legitimate product parameters:

* **hip roll joint LIMITS** - roll reached its authored 0.2 rad limit and the COM shift saturated at
  0.105 m against the ~0.14 m the stance geometry needs. This is the most direct lead.
* **stance width** - the hips sit at +/-0.14 m; adducting both legs to narrow the stance would reduce
  the shift required, rather than increasing the shift available.
* physics rate, `JointSpacePd` (bypasses the biomechanical layer), authored scene gains.

The plan's rule stands: only if all of these fail is option B (replace the character physics) forced.

## 2026-09-08 17:45 — PHASE 0 VERDICT: FAIL. Godot's ragdoll cannot hold single-leg support.

~100 configurations, no policy, no RL, no gait - a scripted weight shift and leg lift. Axes swept:

```
drive             explicit torque; solver-resolved angular spring at k x1/2/4/8, d x1/2
authority         0.125 / 0.25 / 0.40 / 0.50 / 1.0
stance damping    1.0 / 1.5 / 2.0 / 3.0
LoadCompensation  0.0 / 0.25 / 0.5 / 0.75 / 1.0
SpdCompensation   0 - 1.0, whole-body / swing-only / stance-only (with ForcedSwingSide)
BalanceAssist     0.0 / 0.25 / 0.5 / 1.0      PelvisGainScale 0.25 / 0.5 / 1.0
weight shift      hip roll, spine, ankles, all combinations; stance narrowing
lift              amplitude 0.15 - 1.0, knee ratio 0.3 - 1.0, delay 2 s / 3 s
physics rate      240 / 480 / 960 Hz
biomechanics      JointSpacePd (Hill law, effort clamp and muscle layer bypassed)
```

**Every configuration fails the gate** (0.02 m clearance with pelvis >= 0.78, held 3 s). The outcome
is binary everywhere:

```
small lift  ->  foot returns to the floor, body 100% upright, HELD <= 0.75 s
large lift  ->  foot clears 0.17-0.67 m, pelvis 0.125, body FALLEN
JointSpacePd -> collapses immediately, 5.3% upright
```

Best cell in the entire search: peak clearance **0.0305 m at 100% upright, sustained 0.32 s** (hip roll
+ both ankles, lift 0.25). Isaac, on the SAME rig, holds 0.021-0.026 m indefinitely with the pelvis at
0.78.

### What Phase 0 did establish

* **Weight transfer works.** Hip roll moves the COM +0.109 m onto the stance foot at 100% uprightness.
  The foot spans +0.08..+0.20 m, so the body genuinely IS over the support foot - the support polygon
  is not the constraint, and this capability was simply never commanded before.
* **The leg tracks its target.** The angular spring follows a 0.26 rad hip command to within 0.002 rad.
* **The failure is holding body height on one leg.** Once weight transfers, nothing sustains the
  stance leg, and the body either settles back onto both feet or collapses.

### What this means for the architecture

The rig DESIGN is not the problem: **Isaac balances this same rig - same masses, same geometry, same
joint limits - on one leg.** What cannot do it is the Jolt realisation of the actuator and solver.

So the conclusion is narrow and specific rather than "Godot is unsuitable":

* **Options A and C are blocked at their lowest layer.** Neither a Jolt-trained policy nor a
  hand-written deterministic controller can produce a gait on a body that cannot stand on one leg. A
  Jolt-trained policy would simply discover marching in place as optimal, which is what every
  Isaac-trained one already does.
* **Option D survives, in modified form.** Motion matching plus active ragdoll does NOT require the
  physics to solve balance if the root is kinematically driven and the limbs follow physically - the
  usual game active-ragdoll architecture. Physical balance is then a reaction system, not a locomotion
  system, which is what the existing 100%-upright body is already good at.
* **Option B directly addresses the finding**, at the cost of replacing the character physics and
  discarding the Hill/effort/feed-forward layer.

Phase 0 therefore does not force B outright, as the original plan supposed. It forces a choice between
**B** (physical locomotion, new physics) and **D'** (kinematic root, physical reactions, keep Jolt).

## 2026-09-08 18:00 — The working configuration now lives in the scene

`Scenes/RL/Isaac3/Walk/IsaacWalkCheckNewton.tscn` set `ActionScaleOverride = 0.10` and left
`StanceDampingScale` at its default of 1.0. The 27-cell sweep measured that exact cell as a **STATUE**
with zero foot strikes; the project's best result needs **authority 0.125 with `StanceDampingScale
2.0`**, and those values existed nowhere in the repository - only in command lines. Opening the scene
produced a statue with no way to know why.

The scene now carries them, verified with **no command-line overrides**:

```
scene defaults: WALK  strikes 12  upright 100.0%  travel 0.154 m
```

**Do not "tidy" these back to 0.10 / 1.0.** They are the measured operating point, they are a single
working cell in a 3x3 grid, and the same cell is optimal for all three of the best checkpoints - see
the 27-cell sweep above and [[one-working-cell]].

Also fixed: `logs/` added to `.gitignore`. Runs land in `logs/rsl_rl/` at the repository root, which
the existing `isaac_lab/logs/` and `isaac_lab_3/logs/` patterns do not reach - 8.3 GB across 64 runs,
of which 189 yaml/json files were not covered by the `*.pt` rule and would have been committed. The
files must stay on disk (`run_conditions.restore()` reads `params/env.yaml`), they simply are not
source.

## 2026-09-08 18:20 — Removed the eliminated-treatment surface from the driver

`IsaacPolicyDriver` had accumulated one exported knob per experiment. Thirteen of them corresponded to
treatments this ledger records as measured and **eliminated**, each still a live branch: the lead
compensator (`TargetLeadSteps`, `LeadSmoothing`, `HipLeadSteps`, `HipActionScale`), SPD compensation
in all three variants (`SpdCompensation`, `SwingSpdCompensation`, `ForcedSwingSide`), `PelvisGainScale`,
`SwingActionScale`, the high-pass damping pair (`HighPassDampingScale`, `HighPassAlpha`),
`DampingBlendSeconds`, and `SpringGravityFeedForward`.

Removed, along with the code they reached: `LeadActions`, `HipSlots`, `ApplySwingCompensation`,
`IsSwingLegBone`, `ActiveBone.ApplyGravityFeedForwardOnly`, the `PidController3D` compensation factor
and high-pass branch, and six backing arrays.

Also removed **`UpdateLedTargets`, which was already dead** - never called, and its `_ledTargetEuler`
output never read. It was orphaned when the lead moved from target space onto the action vector.

```
IsaacPolicyDriver   1967 -> 1575 lines,  47 -> 34 exports
PidController3D     10968 -> ~7000 chars (classic Tan-Liu-Turk form restored)
```

**Verified behaviour-neutral rather than assumed.** Every removal was a branch inert at its default
(`SpdCompensation 0` made the compensation factor exactly 1.0; `DampingBlendSeconds 0` made the blend
alpha exactly 1.0), so the walk scene must produce the identical trajectory:

```
before   WALK  12 strikes  100.0% upright  0.154 m
after    WALK  12 strikes  100.0% upright  0.154 m
bit-comparison over 2400 steps:  max |d pelvisZ| 0.000000,  max |d Thigh_L.x| 0.000000
```

The findings these knobs produced are preserved in this ledger, which is where they belong; the
product does not need to carry a configuration surface for hypotheses that were disproved.

## 2026-09-08 19:00 — OPTION B: MuJoCo holds single-leg support on the same rig. Godot cannot.

Phase 0 established that Godot's ragdoll cannot stand on one leg in ~100 configurations, which blocks
every architecture needing physical locomotion in Jolt. Option B - move the character physics to
MuJoCo - rests on a premise that had never been tested: **can MuJoCo do it with OUR rig?**

### The model is generated from Godot, not hand-copied

`mujoco_rig/build_mjcf.py` reads the `[PLANT]` dump the Godot driver already emits (mass, parent,
world position and joint-anchor offset per body), the collider shapes from `ActiveRagdoll.tscn`, and
the per-DOF limits from `dummy_rig.json`. Godot remains the single source of truth for the rig; there
is no second hand-maintained copy to drift.

Two conversion bugs were caught by checking rather than assuming:

* **Joint axes were not passed through the frame map.** Positions were mapped by `to_mj` but axes were
  written raw, so `x` became `(1,0,0)` instead of `(0,-1,0)`. Zero targets still produce the correct
  rest pose, so `validate.py` passed and only commanded motion would have revealed it.
* **The actuator model was wrong.** Godot drives `tau = kp(target - q) - kd*qd`; the first version put
  `kd` on the JOINT and left the position actuator as a `kp`-only spring at up to 1800, which
  oscillates the limb apart. MuJoCo expresses Godot's form directly as `kp` **and** `kv` on the
  actuator.

Validated against Godot before any capability claim:

```
                spawn pelvis   settled pelvis   feet      mass
MuJoCo             0.8298          0.8193       0.0394   80.600 kg
Godot              0.8298          ~0.817       0.0394   80.60  kg
```

### The gate

Same primitive, same gate as Phase 0 (clearance >= 0.02 m, pelvis >= 0.78, held >= 3 s), and the same
pelvis attitude controller both engines use (gain 600, damping 20, max torque 300 — without it the
Godot body collapses outright, so a probe lacking one is not comparable).

```
engine            best clearance   held        outcome
GODOT (~100 cfg)     0.0305 m      0.32 s      FAIL
ISAAC (XPBD)         0.021-0.026 m sustained   pass
MUJOCO               0.0575 m      8.35 s      PASS
```

Confirmed in detail over a 20 s run, because a large `HELD` number is exactly the trap that made a
Godot topple score 97.9% upright:

```
over the held phase (17.5 s):
  lifted foot clearance    min 0.0482   mean 0.0539
  OTHER foot displacement  max 0.0150   (planted)
  pelvis height            min 0.7875   mean 0.8036
```

**The dummy stands on one leg for 17.5 s in MuJoCo, on the rig converted from Godot's own dump.**
Option B's physics premise holds.

### One honest caveat, and the real remaining risk

The stance foot **drifts laterally ~0.7 m over 18 s** (~0.04 m/s) while balanced. That is a slow slide,
not a fall, and it does not affect the single-support result — but it is unexplained and should be
understood before it is built on (friction, or a net lateral bias in the attitude controller).

More importantly: **this validates the physics only.** B's actual risk was never whether MuJoCo can
simulate a humanoid; it is the Godot<->MuJoCo integration - a GDExtension, transform sync, fixed
timestep, and cost for ~50 NPCs. None of that is tested yet. What this result does is remove the one
way B could have failed cheaply.

## 2026-09-08 19:30 — THE DUMMY WALKS IN MUJOCO. Scripted, no RL.

Phase 0 in MuJoCo gave one step's worth of capability (17.5 s of single-leg support). A gait is that
alternating, so the next question needed no RL at all: a plain oscillator, lateral weight shift and
leg swing in antiphase, with the same pelvis attitude controller both engines use.

**First result was retracted.** At 20 s a cell read 47 strikes, 100% upright, 2.195 m with travel in
every quarter. At 40 s the same cell is 70.9% upright, pelvis min 0.127, flight 33.9%: it survives the
window and collapses after it. **Third time this exact confound has appeared on this project**, and
the reason the 40 s gate exists. The sweep was re-run at 40 s with flight and single-support gates
added, so hopping cannot pass as walking.

### The result, 40 s, gates enforced

`freq 0.6, shift 0.25, lift -0.4, knee 0.4`:

```
                        MUJOCO (scripted)      GODOT (best ever)
upright                   100.0%  min 0.797     100%
strikes                     96                   12
single support              51.4%                 1.3%
flight (both aloft)          0.0%                  -
travel                    4.835 m (0.121 m/s)   0.154 m
per quarter    0.962  1.492  1.366  1.759       0.117  0.044  0.001  0.001
max clearance             0.1384 m              0.0629 m
```

**A sustained alternating walk: 4.835 m in 40 s, 100% upright, zero flight phase, travel in every
quarter.** Godot's best is 0.154 m of marching in place with 1.3% single support. That is ~31x the
travel with genuine single-leg support, on the SAME rig, converted from Godot's own dump, with no
policy of any kind.

### Honest limit: it is open loop and eventually falls

At 90 s: 73.1% upright, per-quarter travel 3.040 / 2.871 / 1.685 / 0.098 - it walks for roughly the
first 60 s and then falls. Expected for a fixed oscillator whose only feedback is the pelvis attitude
controller; nothing corrects gait phase against accumulated drift. That is precisely the job of a
learned or state-machine controller, and it is a much easier problem than the one Jolt posed, which
was that no controller could exist because the body could not stand on one leg.

### Where this leaves option B

The physics premise is now settled twice over: MuJoCo holds single-leg support, and it walks. What
remains untested is the part that always carried B's real risk - the Godot<->MuJoCo integration
(GDExtension, transform sync, fixed timestep, cost for ~50 NPCs). Nothing about the bridge is
validated yet.

## 2026-09-08 19:50 — Option B de-risked on all three axes

B's risk was never the physics. Three things had to hold, and all three now do.

**1. Physics.** MuJoCo holds single-leg support on this rig (17.5 s, 5 cm clearance) and walks it
(4.835 m in 40 s, 100% upright, 51.4% single support, 0% flight). Godot cannot do either.

**2. Performance - the number that could have killed B.**

```
28.2 us per physics step, 51 dofs
real-time factor 148x on ONE core
at 240 Hz: 0.028 ms/character/frame -> 592 characters inside a 16.7 ms frame, single-threaded
```

A game needing ~50 NPCs spends roughly **1.4 ms/frame on one core**, and characters are independent
so they parallelise. This is not a constraint.

**3. Interop - and no GDExtension is needed.** The project is Godot Mono on net8.0 and `mujoco.dll`
ships a C API, so P/Invoke reaches it from the existing C# assembly; no C++ toolchain, no godot-cpp.
Proven end to end in `mujoco_rig/csharp_spike`:

```
model loaded from C#, mj_name2id("Pelvis") = 1
20000 steps -> 29.1 us/step, 143x real time
```

**29.1 us from C# against 28.2 us from Python - P/Invoke adds no meaningful overhead.**

One trap worth recording: reading `mjModel` fields at offsets 0/4/8 returned `nq=52, nv=0, nu=51`
against the true `52/51/36` - plausible, and wrong. The head is **8-byte strided** (`nq` at 0, `nv` at
8, `nu` at 16, `nbody` at 32). A production bridge must use generated bindings - the `mujoco` package
ships an `introspect` module that emits struct layouts - rather than hand-computed offsets.

### What remains for B

Implementation, not risk: generated `mjModel`/`mjData` bindings, a Godot node that steps MuJoCo and
syncs body transforms onto the existing skeleton, and a decision about which layer keeps the
biomechanics (MuJoCo has native muscle actuators, so the Hill law and effort limits need not be lost).

## 2026-09-08 20:15 — THE DUMMY WALKS IN GODOT, driven by MuJoCo

The bridge is built and running end to end inside the engine.

```
[MujocoDummy] MuJoCo 3.8.1: 16 bodies, 36 actuators, 16 visual proxies
[MujocoDummy] t=50.0s  strikes=42  upright=100.0%  single=20.8%  flight=0.0%  travel=6.062m
[MujocoDummy] Godot/Jolt best ever: 12 strikes, 1.3% single support, 0.154 m
```

**6.062 m in 50 s at 100% uprightness with zero flight phase, rendered in Godot.** Against the Jolt
ragdoll's best-ever 0.154 m of marching in place - 39x the travel and 16x the single support. The rate
is 0.121 m/s, identical to the standalone Python harness, so the bridge reproduces the simulation
rather than approximating it.

### What was built

```
Source/RL/MuJoCo/MjInterop.cs    P/Invoke surface; no GDExtension, no C++ toolchain
Source/RL/MuJoCo/MjLayout.cs     GENERATED struct offsets (mujoco_rig/gen_offsets.py)
Source/RL/MuJoCo/MjBridge.cs     model + state, body poses converted into Godot's frame
Source/RL/MuJoCo/MujocoDummy.cs  Godot node: steps MuJoCo, drives the gait, syncs transforms
Scenes/RL/Isaac3/MuJoCo/MujocoWalk.tscn
```

The rig is generated from Godot's own `[PLANT]` dump and the visual proxies are built by reading that
same generated model, so the skeleton and the shapes are each defined exactly once.

### Three silent-failure bugs, all caught by measuring rather than assuming

1. **Joint axes were not passed through the frame map** - written raw, so `x` became `(1,0,0)` instead
   of `(0,-1,0)`. Zero targets still gave the correct rest pose, so the validation gate passed.
2. **The actuator model was a kp-only spring.** Godot drives `tau = kp(target-q) - kd*qd`; `kd` had
   been put on the JOINT, leaving a spring at up to 1800 that shakes the limb apart. MuJoCo expresses
   Godot's form directly as `kp` **and** `kv` on the actuator.
3. **`mjOBJ_ACTUATOR` was guessed as 8.** It is 19; 8 is `mjOBJ_LIGHT`. Every actuator lookup returned
   -1, `SetControl` ignored it, and the first in-engine run reported a perfectly upright body with
   **0 strikes and 0.002 m of travel** - a gait that appeared to run and did nothing. A failed lookup
   now logs an error instead of failing silently.

Struct offsets are generated for the same reason: reading `mjModel` at hand-computed 0/4/8 returned
`nq=52, nv=0, nu=51` against the true `52/51/36`. `MjBridge` re-checks them at load and refuses to run
if they disagree.

### Status of option B

Physics, performance and integration are all now demonstrated:

```
physics       17.5 s single-leg support; walks 4.8 m in 40 s standalone
performance   28.2 us/step, 148x real time on ONE core; ~50 NPCs ~= 1.4 ms/frame
integration   P/Invoke from the existing C# assembly, 29.1 us/step in-engine
end to end    6.062 m walked in Godot at 100% upright
```

The gait is still an open-loop oscillator and falls at roughly 60 s, because nothing corrects phase
drift. That is a controller's job, and unlike the Jolt situation a controller can now exist.

## 2026-09-08 21:30 — STAND, PERTURB and WALK proven in MuJoCo, and running in Godot

### In MuJoCo (Python reference, `mujoco_rig/prove2.py`)

```
STAND     120 s   100.0% upright, pelvis 0.819, drift 0.005 m                        PASS
PERTURB   recovers 75 N.s (0.93 m/s of COM velocity); falls at 150 N.s               PASS
WALK      180 s    99.2% upright, 413 strikes, 50.9% single support, 2.4% flight,
                   21.389 m travelled                                                PASS
```

### In Godot, driven over the bridge

```
STAND     120 s   100.0% upright, 0.002 m drift                                      PASS
PERTURB   120 s   100.0% upright; the 500 N / 150 ms shove moves it 0.045 m
                  against 0.002 m standing, and it recovers                          PASS
WALK       60 s    99.7% upright, 55 strikes, 23.1% single support, 0.0% flight,
                   5.994 m                                                           PASS
```

Godot/Jolt, for comparison: 12 strikes, 1.3% single support, 0.154 m, and no configuration of ~100
can stand on one leg at all.

### The controller: damping, not servoing

The open-loop oscillator falls at ~60 s, and the diagnosis was specific - a GROWING LATERAL
OSCILLATION, with COM-minus-stance-midpoint staying inside +/-0.08 m for most of the run and then
blowing out to -0.37/+0.45 before the body goes down.

Two things had to be measured rather than assumed:

* **Only the SYMMETRIC hip pair moves the centre of mass.** The antisymmetric pair that produces the
  step moves it 0.001-0.006 m; both hips together move it 0.89 m at 0.25 rad. So the step is
  antisymmetric and the balance term symmetric, superposed on the same two joints.
* **Damping works, position-servoing does not.** Gain swept 0 / 0.05 / 0.10 / 0.30 rad per metre of
  position error: every non-zero proportional gain destabilises the gait (0.05 fails at 60 s, 0.30 at
  40 s) because it fights the lateral excursion the gait needs. Bleeding off lateral MOMENTUM instead
  carries the walk from ~60 s to 180 s.

### Two silent failures caught, and one open discrepancy

**`godot_run.py --set` appends unknown keys to the END of the scene file**, which lands them inside
the last node block rather than the root. `LateralDamping` was therefore being set on a
DirectionalLight3D and ignored, and an A/B that returned bit-identical results looked like "the term
is inert" when it actually meant "the flag never arrived". Declaring the property in the scene makes
the in-place replace work. **Any sweep of a property the scene does not already declare is suspect.**

**A one-step "push" is not a push.** 900 N for a single 4 ms step is 3.75 N.s, about 0.05 m/s on an
80 kg body, and PERTURB initially reported numbers identical to STAND. Held for 150 ms it is a real
shove.

**Open:** the C# walk degrades between 60 s and 120 s (49.9% upright at 120 s) where the Python
reference sustains 180 s at 99.2%. Both run the same gait, the same gains and the same balance torque,
and the timestep is now read from the model rather than hardcoded, which did not account for it. Not
yet explained; STAND and PERTURB are unaffected.

## 2026-09-08 22:00 — PERTURB, properly characterised (and an earlier claim corrected)

"It stayed upright" is not a perturbation metric. The Godot node now records **peak displacement,
peak tilt and recovery time** around the shove, measured relative to the pose captured immediately
before it.

First, the force path was verified rather than assumed - if a 4000 N shove had left the body at 100%
upright, the force would not have been reaching it at all:

```
PushForce   upright   travel     outcome
     0 N    100.0%    0.002 m    baseline
   500 N    100.0%    0.045 m    survives
  1000 N     42.3%    1.183 m    knocked down
  4000 N     41.5%    4.893 m    thrown
```

**A correction.** The first recovery test required the body to return within 2 cm of where it started,
and reported `recovered=NEVER` for a 75 N.s shove that plainly survived upright - it simply ends up
0.159 m displaced, which is what a shoved human does. Recovery is regaining POSTURE (tilt < 0.05 rad,
pelvis > 0.78, COM velocity < 0.05 m/s), not returning to the original spot.

With the corrected criterion the response is monotone, with a sharp cliff:

```
push          impulse    dV        peakOffset   peakTilt   recovered
200 N x 0.15s   30 N.s   0.37 m/s    0.029 m      0.9 deg    0.65 s
400 N           60 N.s   0.74 m/s    0.100 m      3.2 deg    0.98 s
500 N           75 N.s   0.93 m/s    0.159 m      5.7 deg    1.31 s
600 N           90 N.s   1.12 m/s    0.263 m     10.9 deg    1.79 s
--- cliff ---
700 N          105 N.s   1.30 m/s    1.154 m     81.3 deg    NEVER
800 N          120 N.s   1.49 m/s    1.091 m     77.4 deg    NEVER
```

**PERTURB recovers up to 90 N.s (1.12 m/s of induced COM velocity) and fails at 105 N.s.** Displacement,
tilt and settling time all grow smoothly up to that point, which is what a working balance response
looks like; beyond it the body goes over at ~80 degrees of tilt.

For scale, 90 N.s on an 80.6 kg body is a firm two-handed shove. The recovery is ankle/hip strategy
only - the controller never takes a protective step, so this threshold is what balance alone buys and
a stepping reflex would raise it.

## 2026-09-08 22:40 — PERTURB, corrected: the real test is a BALL, not a force on the pelvis

The project's PERTURB task is `BallGun`, not an abstract impulse. Godot fires the SMALL ball on every
shot (`SmallBallProbability = 1.0`): **3.0 kg, radius 0.06 m, 6.0 m/s +/- 0.25, restitution 0.25,
spawned 2.0 m out, aimed at one of twelve target bones**. Impulse is `m*(v - v_reflected)`, so
1.25 * 3.0 * 6.0 = **22.5 N.s**, not the 18 N.s momentum alone suggests.

**The earlier PERTURB result was measuring a different experiment.** A force applied at the pelvis is
a force through the centre of mass; a ball hits a LIMB, and the torque `r x F` it adds is what
`perturb_env_cfg` calls "most of what makes a shove hard to reject". The pelvis push survived 90 N.s;
the ball topples the body at 16 N.s.

A real ball was added to the model (free body, `solref="-8000 -30"`) and a `BallGun` written against
Godot's parameters.

### A physical bug the numbers exposed

The first per-bone run reported **0 of 12 survived, with peak tilt 87.9 deg and pelvis 0.127 for every
single bone** - identical to three significant figures across a 7.7 to 30.0 N.s range of delivered
impulse. A 4x range of input cannot produce an identical output, and the tell was that identity.

Cause: the ball was PARKED at `z = -5`, beneath an infinite floor plane, i.e. penetrating the ground
by five metres. MuJoCo ejects it violently, and that - not the shot - was knocking the body over.
Parked at `(40, 40, 2)` instead. The projectile is also excluded from the body's own centre of mass;
including a 3 kg free body dragged the COM the balance term regulates, and while falling it moved the
reported COM to z = 53.

### With that fixed

```
target bone   delivered   peak tilt   verdict        target bone   delivered  peak tilt  verdict
       Head     25.0 N.s      3.5 deg  SURVIVES       Forearm_L     11.7 N.s    0.8 deg  SURVIVES
      Chest     24.7          2.2      SURVIVES      UpperArm_R     15.3        1.2      SURVIVES
      Spine     26.4          1.4      SURVIVES       Forearm_R     11.7        0.8      SURVIVES
     Pelvis     28.0          1.0      SURVIVES         Thigh_L     32.5        1.8      SURVIVES
 UpperArm_L     20.0          1.0      SURVIVES          Shin_L     26.2        0.8      SURVIVES
                                                        Thigh_R     31.7        1.9      SURVIVES
                                                         Shin_R     25.3        0.8      SURVIVES
```

**12 of 12 target bones survive**, and the response is now physically coherent: tilt scales with the
height of the impact (head 3.5 deg, chest 2.2, shins 0.8), which is what a moment arm does. Delivered
impulse spans 11.7-32.5 N.s against Godot's 22.5 N.s head-on - glancing hits deliver less, square ones
more.

Sustained fire, one ball every 6 s:

```
STANDING under 6 balls   mean 15.3 N.s   peak tilt  8.7 deg   pelvis min 0.815   SURVIVED
WALKING  under 6 balls   mean 16.3 N.s   peak tilt 85.9 deg   pelvis min 0.122   FELL
```

Standing absorbs repeated ball strikes. **Walking under fire does not**, which is expected and worth
stating plainly: the gait is a fixed oscillator with no reactive stepping, so a hit that displaces it
has nothing to catch it. Ankle and hip strategy alone is what is being measured; a protective step is
the missing behaviour.

## 2026-09-08 23:10 — The real BallGun runs in Godot

`MjBallGun` fires Godot's own perturbation test inside MuJoCo, from the Godot side, with the ball's
qpos/qvel addresses read from `jnt_qposadr` / `jnt_dofadr` rather than assuming the projectile is the
last joint - true today, and silently wrong the first time the rig gains a body.

```
ball 1 -> Shin_L      24.3 N.s        ball 4 -> Shin_L      25.8 N.s
ball 2 -> Forearm_L   11.8 N.s        ball 5 -> Shin_L      24.0 N.s
ball 3 -> Chest       30.4 N.s        ball 6 -> Forearm_L   11.8 N.s

45 s, 100.0% upright, peak tilt 3.5 deg, pelvis 0.819, drift 0.012 m   SURVIVED
```

Delivered impulses bracket Godot's 22.5 N.s head-on figure, and the body absorbs six of them without
losing posture.

Two things the projectile forced, both worth keeping: it is excluded from the body's centre of mass
(3 kg of free body otherwise drags the quantity the balance term regulates) and from the reported
total mass, which had been printing 83.6 kg against the rig's 80.6.

### Status of the three behaviours, in Godot

```
STAND     120 s   100.0% upright, 0.002 m drift                                   PASS
PERTURB    45 s   100.0% upright, 6 balls, 11.8-30.4 N.s, peak tilt 3.5 deg       PASS
WALK       60 s    99.7% upright, 55 strikes, 0.0% flight, 5.994 m                PASS
```

Known limits, stated rather than buried: the walk degrades between 60 s and 120 s in C# where the
Python reference sustains 180 s (unexplained), and walking UNDER ball fire falls - the gait is a fixed
oscillator with no reactive stepping, so a displacing hit has nothing to catch it.

## 2026-09-08 23:40 — Three loadable scenes, each with its own environment

The scenes were unusable by hand: no ground at all (the dummy rendered in a void), a fixed camera the
walk left within seconds, and an auto-quit.

```
Scenes/RL/Isaac3/MuJoCo/MujocoStand.tscn     10 m pad,   dummy.xml
Scenes/RL/Isaac3/MuJoCo/MujocoPerturb.tscn   16 m pad,   dummy_ball.xml   BallGun on
Scenes/RL/Isaac3/MuJoCo/MujocoWalk.tscn     120 m field, dummy.xml
```

Each carries a ground mesh, a camera that follows the pelvis, and `AutoQuitSeconds = 0` so it runs
until stopped.

### The projectile is not free, so there are two models

`build_mjcf.py` now emits `dummy.xml` and `dummy_ball.xml`. A 3 kg free body in the model shortens the
walk's stable horizon **even when parked and excluded from the centre of mass**: measured, the Python
walk fell from 180 s to about 60 s, and the C# walk from 99.7% upright at 60 s to 65.7% at 45 s,
purely from the ball existing. STAND and WALK load the clean rig; only PERTURB pays for the ball.

That regression is worth remembering as a class: adding an unused body to a model is not free, and it
showed up as a gait failure rather than as anything pointing at the projectile.

### Verified, 60 s each, in Godot

```
STAND     100.0% upright, 0.002 m drift
PERTURB   100.0% upright, 8 balls, peak tilt 3.5 deg, pelvis 0.819        SURVIVED
WALK       99.7% upright, 55 strikes, 0.0% flight, 5.994 m
```

## 2026-09-09 00:10 — The rendered dummy looked wrong. The rig was right; the DRAWING was wrong.

Loading the scenes by hand showed a jumble of scattered parts rather than a humanoid, which is exactly
the kind of thing that should stop a result being trusted.

**Checked before reassuring.** Godot's colliders all sit at identity relative to their body, so geom
placement was not the issue, and every body position matches Godot to **0.0000 m** across all 16
bones at the spawn pose. The physics rig is correct and the measured results stand.

The fault was in the visual proxies only: every capsule was rotated 90 degrees. A MuJoCo capsule runs
along its local Z, and under the basis change `R_godot = M^T R_mj M` a local vector maps as `M^T v`,
so `M^T(0,0,1) = (0,1,0)` - Godot's Y, which is already the axis `CapsuleMesh` uses. The rotation was
unnecessary and laid every limb on its side. Removed; the limb axes now check out against the bone
directions:

```
Thigh_L  -> Shin_L      (+0.00,-1.00,+0.00)   vertical
Shin_L   -> Foot_L      (+0.00,-0.99,-0.14)   vertical
UpperArm_L -> Forearm_L (+0.00,-1.00,-0.00)   vertical
Spine    -> Chest       (+0.00,+1.00,-0.00)   vertical
```

Worth stating plainly: a wrong visual is not a harmless cosmetic bug. It is indistinguishable from a
wrong rig until someone checks, and it correctly destroyed confidence in results that were in fact
sound.

## 2026-09-09 01:00 — Passive joints were free, and the balance assist is doing the perturb recovery

### The head sank into the chest: passive joints had no gains

`dummy_rig.json` marks Head and Hand joints `actuated=False` but still gives them stiffness 120 /
damping 12, and Godot holds them at rest through `NeutraliseUncommandedBones`. The generator emitted
them as FREE hinges: a 5 kg head on a frictionless pivot, which gravity rotated down until it sank
into the chest, since MuJoCo disables parent-child contact. Passive joints now carry their authored
spring, and the head-chest gap holds at 0.2900 m indefinitely.

**It was not cosmetic.** Flopping head and hands were disturbing the gait: with them held, the walk
went from 5.994 m to **8.412 m at 100% upright over 60 s**, and PERTURB from 8 to 15.8 deg peak tilt
with the heavier ball while still surviving.

### The balance assist, measured

The pelvis attitude controller writes an EXTERNAL torque onto the pelvis via `xfrc_applied`. Nothing
in the body produces it. Both engines carry it (`balance_assist`, gain 600 / damping 20 / max 300),
but it is worth knowing how much of each result it is responsible for:

```
condition                     upright   pelvisMin   assist mean   peak
STANDING, with assist          100.0%     0.816       0.5 N.m     0.8 N.m
STANDING, assist removed       100.0%     0.816          -           -
BALL IMPACTS, with assist      100.0%     0.815       8.7 N.m    90.7 N.m
BALL IMPACTS, assist removed    33.4%     0.108          -           -
```

**Standing is genuine** - the legs hold the body and the assist contributes essentially nothing.
**The perturbation recovery is not**: it is carried by up to 90.7 N.m of external torque, and without
it the body falls. The legs never step to catch themselves, because nothing asks them to.

That is the honest reading of the PERTURB pass, and it is the clearest possible argument for training:
a policy on this plant with the assist removed or penalised has to recover using ankles, hips and a
protective step, which is the behaviour the assist is currently standing in for.

## 2026-09-09 01:40 — RL on the MuJoCo plant: training and runtime are finally the same engine

The first training run on this project where the plant the policy learns on IS the plant it ships on.
Every previous attempt lost the policy crossing an engine boundary; there is no boundary here.

**The task is defined by what it removes.** The pelvis balance assist is absent. It was measured
carrying the perturbation recovery - up to 90.7 N.m of external torque written onto the pelvis, with
the body falling to 33.4% upright without it - so a policy here has to keep its feet under its centre
of mass instead.

```
mujoco_rig/rl/perturb_env.py   vectorised MuJoCo env, ball firing, NO assist
mujoco_rig/rl/ppo.py           self-contained clipped PPO with GAE
mujoco_rig/rl/train.py         training loop
```

Observation is 120 floats: projected gravity, pelvis linear and angular velocity, height, 36 joint
positions and velocities, two foot contacts, and the previous action. Reward is upright + height +
low COM speed + **COM over the feet** (which is what a protective step achieves) minus effort and
jerk, with a fall penalty.

### Two things that had to be fixed before it would run

**The action authority was too high for the random phase.** An untrained Gaussian policy commands
every joint to a random extreme at once, and at the authored gains (kp up to 1800) that drove MuJoCo
to `Nan, Inf or huge value in QACC` before any learning happened. A quarter of the joint range is
ample for balance and keeps the random phase numerically sane.

**rsl_rl was abandoned for this task.** The project's own stack was tried first and cost several
rounds on plumbing rather than physics: this build takes `actor`/`critic` blocks instead of `policy`,
rejects resolved defaults passed back as constructor arguments, has a NaN guard that cannot read a
TensorDict, and a normaliser calling `.var(unbiased=...)` on one. The replacement is ~150 lines of
standard clipped PPO, which is worth the trade because a surprising result can now be attributed to
the environment rather than to an unfamiliar framework.

Sanity checks before training: observations finite, and a zero-action rollout puts 8 of 8 envs on the
floor - a task nothing can fail teaches nothing.

### The first two runs were teaching the policy to survive PPO

Two runs failed before one worked, and both failed for the same reason in different disguises.

**Run 1 destroyed a working controller.** After 475 iterations the policy stood 27.9% of a QUIET
40 s. Doing nothing scores 100% there - an all-zero action holds every joint at its rest pose
through position actuators at the authored kp of up to 1800, and that alone stands indefinitely.
The trained policy was worse than the absence of a policy, in both conditions:

| condition | upright, 40 s | fell | median fall |
|---|---|---|---|
| zero action, quiet | **100.0 %** | 0/8 | never |
| zero action, under fire | **32.3 %** | 8/8 | 11.7 s |
| run-1 policy it-475, quiet | 27.9 % | 7/8 | 3.7 s |
| run-1 policy it-475, under fire | 17.7 % | 8/8 | 3.7 s |

The 32.3% independently reproduces the 33.4% measured when the pelvis assist was removed, so the
task is well posed: the ball topples the unaided body every time. The starting point was the bug.
A randomly initialised actor throws away a controller that already works, so the last actor layer is
now zeroed - the untrained policy IS the rest-pose hold, verified at 100% quiet - and improvement is
measured against something real.

**Run 2 fell before the ball arrived.** Starting from the baseline lifted the opening return from
184 to 446, and the body still hit the floor 1.7 s into every episode. The first ball fires at
2-4 s. Nothing was hitting it: the exploration noise was.

So the tolerated noise was measured instead of guessed a third time (`rl/probe_noise.py`). What
governs survival is the PRODUCT of policy std and action authority - the size of the random jump in
the joint target, which a stiff position actuator answers immediately:

| std x authority | 0.0375 | 0.0250 | 0.0125 | 0.0100 | 0.0075 | 0.0050 |
|---|---|---|---|---|---|---|
| time to fall | 1.57 s | 2.33 s | 6.96 s | 10.81 s | never | never |
| envs fallen (of 6) | 6 | 6 | 6 | 5 | 0 | 0 |

The threshold is sharp at about 0.008. Authority stays at 0.25 because a protective step needs the
range, so std drops 0.5 -> 0.03. The textbook 0.5 was 47x past what this plant tolerates.

**Result.** Run 3 reached episode length 491 of 1199 steps in 65 iterations; run 1 needed 490
iterations to reach 122. Same reward, same environment, same network - the entire difference is
starting at the working controller and exploring inside what the body can absorb.

*Next lever if this plateaus:* std per joint rather than one scalar. The destabilising joints are
the stiff ones (kp 1800), while the arms are harmless at any noise level, so a single scalar is
throttling arm exploration to protect the legs.

### Run 3 flatlined because the task was impossible, not because PPO was broken

Run 3 held the baseline and stopped: return 2506 at iteration 65, 2667 at 375, and a 40 s score of
32.9% upright against the unaided plant's 32.3%. It had learned not to break the rest-pose hold and
nothing else. A flat return is what a WORKING algorithm does on an unwinnable task, so the task got
audited before the algorithm did.

**Two hypotheses died on measurement.**

*Slowing the policy down does nothing.* The plant tolerates the trained policy using its full 0.25
authority, so the intolerance looked like it might be to high-FREQUENCY target jumps. It is not - at
std 0.10 the time to fall moves only 2.33 s (60 Hz) -> 2.76 (30 Hz) -> 3.03 (20 Hz). What the body
rejects is perturbation magnitude spread across 36 joints at once, essentially regardless of rate.

*The ball is not too heavy.* `build_mjcf.py` records that the projectile was raised from Godot's real
3.0 kg to 10 kg so the hit would read on screen, and that display decision had become the training
perturbation. **The first mass sweep is RETRACTED**: it wrote `model.body_mass` on a live `MjModel`,
which leaves MuJoCo's precomputed `body_invweight0` / `dof_invweight0` constants stale, and those
scale the contact solver. It reported a 0.5 kg ball as more destructive than a 10 kg one - the same
inverted-nonsense signature as the ball parked under the floor. Recompiling the MJCF per mass gives
a monotonic answer:

| single impact, no policy | 0.5 kg | 1.0 | 2.0 | 3.0 | 6.0 | 10.0 |
|---|---|---|---|---|---|---|
| survived | 100 % | 100 % | 100 % | 100 % | 75 % | 75 % |
| max tilt | 0.3d | 0.5d | 0.7d | 1.0d | 23.5d | 24.3d |

Godot's real 3 kg barely registers at 1.0 degrees, which independently confirms `build_mjcf`'s own
note of 3.5 degrees.

**What was actually wrong was the firing rate.** One 10 kg impact is survivable 75% of the time with
no policy at all, yet the same ball every 2-4 s topples 8/8 in 6.8 s: the next ball lands before the
recovery finishes, so no controller can complete one. Training was scoring policies on a task where
every policy loses.

The rate moved to 4-7 s. Note what this does and does not change: the unaided 40 s score barely
moves (32.3% -> 30.1%, still 0/12 surviving), because ~7 hits at 75% each compound to 13%. What
changes is the CEILING. Each hit is now individually recoverable, so raising per-hit survival is
worth something, and 0.99^7 is 93% where 0.75^7 is 13%. That headroom is the whole training signal.

**Reference to beat, unaided, 40 s, balls every 4-7 s: 30.1% upright, 0/12 survive, median fall
12.8 s.** Quiet-room control: 100% upright, 12/12, over the full 40 s.

### Runs 3 and 4 both converged to the rest-pose hold, and a curriculum is why run 5 did not

Fixing the firing rate did not, on its own, produce a recovery. Run 4 scored **29.9% upright against
the unaided plant's 30.1%** after 550 iterations - the same non-result as run 3, at a difficulty
that is now winnable. Both runs found the same local optimum: hold the rest pose, correct gently,
never step.

The reason is arithmetic, and both halves of it are measured:

- A protective step is a large coordinated leg motion.
- Exploration is std 0.03 at 0.25 authority, which is 0.75% of each joint's range. Random
  perturbations that small will not assemble one in any amount of wall-clock time.
- Larger uniform noise cannot be used, because `probe_noise.py` shows it topples the unaided body
  by itself - at std 0.10 the body is down in 2.3 s, before any ball arrives.

A behaviour too large for the exploration to find has to be **grown** rather than discovered, which
is what a curriculum is for: keep every increment inside the small noise ball around the policy you
already have.

**Impulse is m*dv, so ball SPEED buys the same difficulty ramp as ball mass** - and with no model
recompile, which matters here because writing `body_mass` on a live model is exactly the mistake
that produced the retracted mass sweep. Speed starts at 3.0 m/s (30 N.s, just past where the
unaided body begins to fail) and is promoted 15% each time the policy survives 75% of an episode,
up to the 6.0 m/s the Godot scene fires. The stage is saved beside the weights, so a checkpoint can
never be scored at a difficulty it never trained at.

First effect, immediately: the first promotion landed at iteration 50, and episode length went to
932 of 1199 steps - past the ~780 ceiling that runs 3 and 4 both stalled against.

**Correction: the first curriculum was decorative.** Promoting on "75% of an episode survived" with
no dwell time went 3.0 -> 6.0 m/s in 150 iterations (2.2 minutes), and episode length then fell
straight back to 733 - exactly the plateau the curriculum existed to escape. Promotion was trivial
at low speed because the rest-pose hold already survives a gentle ball, so the policy was handed
every stage before it had learned anything at that stage.

The gate now needs near-perfect survival (93% of the episode, over 100 episodes) AND a minimum
dwell of 200 iterations at each difficulty. At 3.0 m/s the policy reached 1102 of 1199 steps within
105 iterations, so the dwell - not the score - is what holds it there long enough to consolidate.

*Next lever if this plateaus:* the reward's `still` term, `exp(-2 * |com_velocity|)` at weight 1.0,
pays the body for NOT moving its centre of mass - which is precisely what a protective step has to
do. It is a headwind on the target behaviour and was left alone here only so that this run's result
is attributable to the curriculum alone.

## 2026-09-09 - Training moves to the GPU, and the honest accounting of what that bought

The MuJoCo track had ended up on **one CPU core of eight, with the 4080 Super at 0%** - which is the
opposite of why this project went to Isaac in the first place. `perturb_env.py` steps every `MjData`
sequentially in a Python loop, so core count never mattered.

**`mujoco_warp` 3.8.1 was already installed** in `env_isaaclab3`, having arrived as a Newton
dependency, and it matches `mujoco` 3.8.1 exactly. It is MuJoCo's CUDA backend - same MJCF, same
model, `nworld` as a batch dimension. `put_model` accepted our rig unchanged: `implicitfast`,
negative `solref`, box/capsule/sphere/plane all supported. **No new dependency, no MJCF edit.**

(MJX was not used. `jax[cuda]` has no Windows wheels, and MuJoCo Warp is the successor GPU backend.)

### Throughput, measured on the real rig

| backend | envs | policy SPS | physics steps/s | realtime |
|---|---|---|---|---|
| cpu | 96 | 2,693 | 10,773 | 45x |
| warp | 1,024 | 20,917 | 83,668 | 349x |
| warp | 4,096 | 56,786 | 227,146 | 946x |
| **warp** | **8,192** | **71,522** | **286,089** | **1,192x** |

26.6x, at 1.9 GiB of 16 GB VRAM. The network follows the sim to the GPU because at the resulting
batch a PPO pass is **346 ms on CPU against 7.9 ms on CUDA**.

### The part that is NOT a 26x win

**26x throughput is not 26x learning, and the first GPU runs proved it.** At 4,096 envs x 24 steps
the policy reached episode length 897 in 1.6 min, where the 96-env CPU run reached 1,102 in 1.5 min
- more samples, no faster. The reason is that PPO moves the policy only as far as its KL cap allows,
so a 42x larger batch bought a cleaner gradient and the *same number of updates*. Cutting the
rollout to 8 steps (65,536 samples per update, still 28x the CPU batch) roughly tripled the update
rate and closed most of the gap: 1,084 at 3.0 min.

So the GPU's value is not early-stage speed, where 96 envs already give enough signal on an easy
task. It is total experience, which is what the CPU track ran out of - run v6 spent 55 minutes and
stalled at ball speed 3.63 m/s of a 6.0 target. That is the comparison that decides this, and it is
the one being run.

### Two silent traps found on the way in

**Constraint buffers: two are per-world and one is aggregate.** Both mistakes are quiet.
`njmax` and `nconmax` are per world; `naconmax` is the total across all worlds. Sizing `njmax` as a
total tried to allocate a 15 GiB dense Jacobian at 1,024 envs and failed loudly - the lucky case.
Sizing `naconmax` per-world instead did not fail at all: at 256 worlds the engine wanted 1,020 while
512 was set, so contacts across the batch were being dropped with nothing but a line on stderr from
inside a CUDA kernel to say so. `assert_buffers_ok()` now checks the real peaks.

**float32 vs float64, and the control run that makes it interpretable.** `mujoco_warp` is float32
throughout, with no float64 option; MuJoCo's C engine is float64. `parity_gpu.py` replays one
identical control sequence into both:

| t (s) | CPU vs CPU, 1e-9 nudge | CPU vs GPU |
|---|---|---|
| 0.50 | 1.0e-09, 0.0000 deg | 2.0e-06, 0.0069 deg |
| 1.49 | 1.0e-09, 0.0000 deg | 2.3e-05, 0.0000 deg |
| 2.48 | 1.0e-09, 0.0000 deg | 1.3e-02, 0.1384 deg |

The control is the point. Two CPU trajectories started a nanometre apart do **not** diverge at all
over 2.5 s, so this system is not chaotic here and the GPU departure cannot be excused as chaos.
It is the engine, it starts at float32 epsilon, and it amplifies through contact.

**Why that is nevertheless safe, unlike Isaac.** Isaac trained on PhysX/XPBD and shipped on Jolt -
different solver classes, with no cheap way to score on the deployment engine. Here the deployment
engine is MuJoCo CPU, Godot drives it by P/Invoke, and scoring on it is a two-minute run that
already exists. So the rule is structural rather than hopeful:

> **Train on GPU. Score and ship on CPU MuJoCo. A checkpoint is never judged by the engine that
> trained it.**

`eval.py` is unchanged and CPU-only, which is the whole point. `test_env_parity.py` additionally
asserts the two environment implementations define the same task - they agree on observations to
**1.49e-08** and rewards to **6.0e-07**, so training and scoring are optimising the same thing.

Also corrected while measuring: the rig is **80.60 kg**, not the 41 kg written in `probe_ball.py`.

## 2026-09-09 (overnight) - closing the loop: a policy that runs IN Godot

### The Walk scene was never asked a question it could answer

`MujocoDummy.DriveGait` is a lateral weight shift plus a half-cycle leg lift. It has no push-off, no
ankle control, and **no heading input at all**, so "walk in a straight line" and "turn left" are not
requests it can accept - it travels by shifting weight and dragging. That is the whole reason it
looks wrong, and no tuning of the oscillator fixes it.

Checked the feet while there, since they were the reported suspect, and they are **correct**:
`to_mj((0.12, 0.08, 0.22))` gives half-extents (0.11, 0.06, 0.04), so the foot's long axis is MuJoCo
+X - which is also forward, `to_mj` of Godot's -Z. The ankle sits 3 cm behind the foot centre, so
14 cm of toe and 8 cm of heel. The proxy mesh mapping is right too. What is wrong is the controller,
not the geometry.

Joint ranges settle which axis is which, and are worth writing down: `Thigh_rx` spans **2.60 rad**
(hip flexion, the walking axis), `Shin_rx` **2.70** (knee), while `Thigh_rz` spans 1.00 (abduction).

### A commanded walk, trained

`walk_env_warp.py` (GPU) and `walk_env.py` (CPU scorer), subclassing the perturb envs so the model,
buffers, stepping and resets are not written a third time. The observation gains **three channels -
forward speed, lateral speed, turn rate, in the pelvis frame** - which is what makes straight-line
walking and turning the same policy instead of two scripts. A tenth of the worlds are commanded to
stand, because a walk that cannot stop is not controllable from Godot.

**First attempt produced a textbook statue**: 100% upright, 0/6 falls, and vx **+0.002** against a
commanded 0.60, with zero steps and zero single support. The reward was at fault, not the algorithm.
At the original `exp(-err/0.25)` simply standing still already collects 24% of the tracking reward,
and with a flat survival bonus on top there was more value in never risking the -10 for falling than
in walking. Sharpening to `/0.15` drops standing to 9%, the tracking weights now dominate posture
(4.0 and 2.0 against 0.5 and 0.3), and the flat bonus is gone - it paid the body for existing.

### Perturb: the reward was fighting the behaviour it asked for

The `still` term, `exp(-2*|com_velocity|)` at weight 1.0, pays the body for NOT moving its centre of
mass - which is exactly what a protective step must do. It is demoted to 0.2 (jitter damping only)
and `support` (COM over the feet, which is what a step achieves) is promoted from 1.5 to **3.0**.

Ball mass raised 10 kg -> **15 kg** as asked. The curriculum start drops 3.0 -> 2.0 m/s to keep the
opening impulse at the same 30 N.s, which is where the unaided body first begins to fail. New
reference, no policy, 40 s, balls every 4-7 s at 6 m/s (90 N.s): **29.1% upright, 0/12 surviving,
median fall 11.8 s** - barely worse than the 10 kg ball's 30.1%, because the body already fails
there.

### The loop closes in Godot

`Source/RL/MuJoCo/MjPolicyDriver.cs` runs a trained ONNX policy against the MuJoCo body through the
existing P/Invoke bridge, replacing both the scripted gait and the pelvis balance assist.

**Nothing about the contract is hard-coded.** Observation layout, joint order, action scale and
joint limits are read from the `*.contract.json` that `export_onnx.py` generates from the training
environment itself - the Isaac track's hand-written contract was wrong in four separate ways and
every one was silent. Two new bridge readers were needed, `JointPosition` and `JointVelocity`, and
they index `qpos`/`qvel` through `jnt_qposadr`/`jnt_dofadr` respectively - those differ by one slot
because a free joint takes seven qpos entries but only six DOFs.

`PolicyPath` and `WalkCommand` are **declared in all three scene files**, not passed as unknown keys:
`godot_run.py --set` appends undeclared properties to the end of the file, where they land on
whatever node is last, and that once made an entire A/B return bit-identical results.

The proxy head now carries the same two eyes and mouth as the authored ragdoll. Cosmetic, but it
gives the dummy a visible front - which is what "the feet point the wrong way" needs to be checked
against.

## 2026-09-09 17:00 — The plant had no limp state: position actuators replaced with torque

The dummy was stiff with a brain and a rag without one, with nothing between. That was never a
training problem. Measured over one 6 s backward fall, zero command throughout:

| | mean joint bend | max bend | pelvis z |
|---|---|---|---|
| actuators on, commanding **zero** | **1.8°** | 17.9° | 0.139 m |
| no actuators (`dummy_limp.xml`) | **38.9°** | 172.2° | 0.166 m |

The body flexed by under two degrees on average while it toppled, because a `position` actuator is a
spring to a target pose: its zero command means *hold the rest pose*, and `kp` 533 N·m/rad at the
hip answers 10° of bend with **93 N·m** — half a human hip's entire capacity — spent on holding
still. Peak actuator force during that passive fall was 118.7 N·m. **There is no setting of a
position actuator that means "no muscle."** That single fact explains "no brain means collapse, brain
means stiff", and no amount of training could have fixed it.

### The two-layer model, generalised from the neck

| layer | what it is | value |
|---|---|---|
| ligament | passive joint spring + damping, always present | `LIGAMENT` = 5% of that joint's peak torque per rad, damping 5% of that |
| muscle | `motor` actuator; the policy's output IS newton-metres | `forcerange = ±HUMAN_TORQUE` |

Zero muscle is now zero torque, so the unpowered body matches the ragdoll it should be: mean joint
bend **24.3° against `dummy_limp`'s 23.3°**, pelvis settling at 0.165 vs 0.164 m. "Dead", "stunned"
and "alive" stop being separate models and become a scale on the policy's output.

5% was measured, not guessed: at 10% the plank starts returning, at 0% the joints are loose.
Standing costs almost nothing to hold — **1.5 N·m at the hip, 3.9 at the knee** — so the muscle layer
is for corrections, not for fighting gravity, which is what makes torque control viable here.

### Three defects found while measuring, all independent

**1. The rest pose was self-intersecting.** At `qpos0` the forearms sat **5.5 cm inside the thighs**,
the hands 4.4 cm, the forearms 1.1 cm inside the pelvis — 11 contacts, costing **331.8 N·m at each
shoulder** (human peak: 90) just to hold. Every episode of every run so far started there. 8° of
abduction clears it by 3–5 cm. It cannot be done with the joint's `ref`, which relabels the joint
value without moving anything, so it is a `rest` keyframe that both environments and
`MjBridge.ResetData` now reset to, with `springref` on the same joints so the ligament rests in the
cleared pose.

**2. The height reward paid the policy to crouch.** `_reward` targeted pelvis z = 0.82 m from a body
that no longer exists; this one rests at **0.917 m**. Standing correctly collected 0.685 of the
height reward instead of 1.0, and the gradient pointed 10 cm downward. Now read from the model, in
all four envs, along with the fall threshold and `eval`'s upright height.

**3. The torque plant is numerically unstable without armature.** The first torque run spent its
whole 15 minutes printing `nefc overflow - please increase njmax to 1011` and then went NaN. Two
causes: a body that genuinely crumples generates far more constraints than one that falls as a plank
(peak nefc 117 and 27 contacts, against buffers sized 128/32 for the old plant), and the joints had
**zero armature**. Measured over 10 random-torque rollouts of 20 s:

| armature | divergences | peak nefc |
|---|---|---|
| 0.0 | **10 of 10** | 622 |
| 0.01 | 0 | 105 |
| **0.02** | **0** | 117 |

MuJoCo silently resets the state when `qacc` goes bad, so a trainer would have been learning from
garbage without any error. `JOINT_ARMATURE = 0.02` on every hinge but the head, whose neck was tuned
without it; `NJMAX` 128 → 256, contacts 32 → 64.

### What this cost and what it did not

`authority` becomes a fraction of peak torque rather than of joint range, and it has to be far
higher — **0.6, not 0.25** — because the policy now supplies all of the muscle. The observation
layout, the 105/30 widths, `POLICY_EXCLUDE`, the neck (still a position servo, deliberately: it is
outside the action space and its job is muscle tone) and the body's geometry are all unchanged, and
both parity tests still pass at 1.49e-08.

The unpowered baseline the next run has to beat: **57.4 steps, 0.96 s**.

## 2026-09-09 19:30 — Housekeeping pass, and two more defects it turned up

A review of the uncommitted MuJoCo work: dead artefacts removed, docs re-synced, and the two largest
single-responsibility violations split up. Two of the findings were real defects, not tidiness.

### The body spawned 1.09 cm in the air

`build_mjcf`'s final normalisation scales the rig about the origin to make standing height exact.
A sole that is not already at y = 0 is therefore scaled AWAY from it, and this one sat 1.09 cm up -
so the dummy dropped a centimetre at the start of every episode, adding an impact the policy had to
absorb before it could do anything, in every run ever recorded. Fixed by translating the finished
body so the soles rest on the floor, and `validate.py` now asserts it.

Re-scored on the corrected model, the current checkpoint moved from 5.8% upright / 2.30 s to
**5.0% / 1.99 s**, against an unaided 2.0% / 0.80 s. Slightly worse, and honestly so: the policy was
trained on the body that hovered, and the free centimetre of drop was part of what it had learned.

### `validate.py` had been asserting the opposite of the design

Its gate was "the model, with every actuator at zero, still stands" - printed as `COLLAPSED - model
is wrong, do not proceed`. That belonged to the position-actuated plant where standing was free.
Against the torque plant a correct model MUST collapse unpowered, so the gate would have failed
forever on a correct body. It now checks the four things that have each been silently wrong:
the `rest` keyframe exists and does not self-intersect, stature and mass are on target, the soles
start on the floor, and an unpowered body CRUMPLES (mean joint bend > 10°) rather than falling as a
plank.

### Structure

`MujocoDummy.cs` was 880 lines doing five jobs. Split, with no behaviour change (build clean, both
parity tests still at 1.49e-08):

```
MujocoDummy.cs          880 -> 498   the loop, input, ball schedule, policy dispatch
MjProxyBuilder.cs           180      Godot meshes read from the model's own MJCF
MjGaitMetrics.cs            213      strikes, uprightness, travel, perturbation response
MjScriptedController.cs     115      the pre-RL oscillator and balance assist
```

On the Python side the CPU environment - the one Godot ships against - was importing its fall
threshold, action latency and authority constants **from the CUDA backend it never uses**, and so
was the scorer. Those constants now live in `rl/env_config.py` and both backends depend on it
rather than on each other.

### Removed

The full anthropometric rebuild (`ANTHROPOMETRIC`, `anthropometry.py`) - measured, rejected, and
kept only as a dead branch that would have rebuilt the body it took several rounds to reject. Ten
pre-RL spike scripts and their scripted-controller dependency chain. Three gain-sweep model
variants and the broken external-data ONNX export. Two policies trained on the position plant,
whose contracts have no `action_to_control.mode` and would have been driven as joint targets
against `motor` actuators - silently wrong, which is the worst kind. Run logs and `MUJOCO_LOG.TXT`,
now gitignored alongside `logs/`.

## 2026-09-09 21:30 — Four throughput levers measured, four eliminated

The GPU shows 65% utilisation at 95 W of a 320 W card, and nothing else is saturated either, so the
obvious question is why the trainer will not go faster. Four candidates, all measured, none of them
real.

### 1. Gradient steps per batch — free, and worthless

Collecting a 262,144-sample batch takes ~12 s; the PPO update on it takes ~10 ms. So `epochs x
minibatches` is genuinely free, and the prediction was that raising it buys learning per hour at no
cost. Three arms, 15 min each, identical seed (`stand_1h/model_300`), on PERTURB because stand is
now solved and its episode length sits at the ceiling:

| arm | grad steps/iter | it/min | return | ep_len | mean KL | mean LR |
|---|---|---|---|---|---|---|
| **5 x 4 (baseline)** | 20 | 5.22 | **5,268** | **906** | +0.0247 | 2.02e-04 |
| 8 x 4 | 32 | 5.26 | 4,901 (-7%) | 857 (-5%) | +0.0339 | 1.57e-04 |
| 5 x 8 | 40 | 5.19 | 5,210 (-1%) | 892 (-2%) | +0.0306 | 1.23e-04 |

The wall-cost prediction held exactly - 60% more gradient steps for 0% more time. The benefit did
not. PPO is on-policy: every extra pass moves the policy further from the one that collected the
batch, KL overshoots its 0.01 target, and the adaptive controller cuts the learning rate to
compensate. The clipped surrogate and the KL-adaptive LR **already regulate the step size**; extra
passes convert useful update into drift that then gets penalised. Keep 5 x 4.

### 2. Halving the physics rate — buys nothing we need, costs stability

240 Hz x decimation 4 -> 120 Hz x decimation 2 keeps the policy at 60 Hz and halves the integration
work. It cannot be a training-only change: a 240 Hz policy scored at 120 Hz once reversed its own
verdict, so timestep is a plant parameter and training and deployment must share it.

| | 240 Hz (shipped) | 120 Hz |
|---|---|---|
| random torque, 6 x 20 s | **0 divergences** | **6 divergences** |
| single-core realtime | 113x | 222x |

`JOINT_ARMATURE = 0.02` was measured at 4.17 ms and does not survive doubling the step; recovering
stability means more armature, which is artificial limb inertia - the opposite of what a body whose
whole selling point is impact response needs. The joint stops would soften too (`solreflimit` is
2x the timestep by construction). And the throughput is not needed: **113 fully-simulated characters
per CPU core** already exceeds what the game puts on screen. If NPC count ever binds, the answer is
level-of-detail, not a coarser timestep for the character being shot in front of the camera.

### 3. Capping the constraint solver — it already exits early

The model runs MuJoCo's defaults: Newton, `iterations=100`, `ls_iterations=50`, `tolerance=1e-8`.
Those looked absurdly generous for a 45-DOF body. Measured over 14,396 steps of real motion:

```
Newton iterations actually used: mean 3.36, median 3, p99 6, max 9   (the cap is 100)
```

The solver converges in three iterations and stops. Capping it at 8/12 changed GPU throughput from
22,635 to 22,671 policy steps/s - 0.2%, noise. `mujoco_warp` exits early exactly as the C engine
does; the cap was never being paid.

### 4. CG instead of Newton — 12% slower

`solver="CG" iterations="20"`: 20,024 steps/s against Newton's 22,855. Newton's quadratic
convergence on three iterations beats CG's cheaper-but-more iterations at this scale.

### Where the time actually goes

| envs | ms per policy step | policy steps/s | us per world |
|---|---|---|---|
| 2,048 | 120.1 | 17,052 | 58.6 |
| 4,096 | 206.1 | 19,873 | 50.3 |
| 8,192 | 370.8 | 22,093 | 45.3 |
| 16,384 | 716.9 | 22,855 | **43.8** |

Per-world cost falls 58.6 -> 43.8 us and is flattening; 8x the worlds buys 34% more throughput. If
the GPU were launch-bound the ms/step column would be flat and more worlds would be nearly free. It
is not - this is real work, converging on a work-bound asymptote around 43 us per world per policy
step, and 16,384 envs is already ~95% of the way there.

**So the appearance of an idle GPU is an artefact of how utilisation is reported.**
`nvidia-smi utilization.gpu` is the fraction of time at least one kernel is resident, not the
fraction of SMs busy; one kernel using 3% of the card reads as 100%. A MuJoCo step is a long chain
of small, mutually dependent kernels - broadphase, narrowphase, constraint assembly, three Newton
iterations, integrate - and each one occupies a sliver of the card while the rest waits on it. More
VRAM, more cores and more RAM do nothing for latency of that shape.

**Conclusion: stop optimising throughput.** ~22,000 policy steps/s is what this plant costs on this
hardware, and it was enough to solve standing in one hour. What is left is sample efficiency -
reward shaping, curriculum, or an off-policy learner that reuses data - not configuration.

## 2026-09-09 23:15 — The perturbation ball never hit anything

The balance policy scores 100% upright under "ball speed 6.0 m/s" in `eval.py` and falls over in the
Godot scene at the first shot. The gap was not sim-to-sim. **The training projectile misses.**

It is spawned 2 m from the target with its velocity pointed straight at the bone and no
compensation for gravity. A projectile's range is `v^2/g`:

| ball speed | max range | reaches a target 2 m away? | impulse delivered |
|---|---|---|---|
| 2.00 m/s | 0.41 m | **no, at any angle** | 16.8 N.s |
| **2.42 m/s** (what the curriculum reached) | 0.60 m | **no, at any angle** | 18.0 N.s |
| 4.00 m/s | 1.63 m | no | 41.6 N.s |
| 6.00 m/s | 3.67 m | yes, but arrives 0.55 m low | 50.8 N.s |

**The minimum speed that can reach 2 m is 4.43 m/s.** The curriculum ran 2.00 -> 2.42, so every
shot of every perturb run was physically incapable of reaching the dummy - it hit the floor about
60 cm after launch. The 241 and 351 "contacts" measured at those speeds are the ball ROLLING INTO
THE FEET. Perturb has been stand training with a ball on the ground, which is exactly why the policy
has no recovery behaviour: it was never given anything to recover from.

### The fix: the projectile does not fall

Gravity is cancelled on the ball alone (`xfrc_applied = m*g` upward, in both environments and in
`MjBallGun`). The flight is a straight line, so the standoff and the reaction time are constant
across the whole curriculum and speed maps linearly onto the impulse the ball carries. Measured as
the DUMMY's own momentum change at contact:

| ball speed | carries | delivered |
|---|---|---|
| 2.0 m/s | 30.0 N.s | 22.8 N.s |
| 3.0 m/s | 45.0 N.s | 37.3 N.s |
| 4.0 m/s | 60.0 N.s | 54.8 N.s |
| 6.0 m/s | 90.0 N.s | 74.3 N.s |

76-91% transfer, monotonic. That is a difficulty axis. The trainer now logs it as newton-seconds
beside the speed. Both parity tests still pass at 1.49e-08.

### `MjBallGun.DeliveredImpulse` measured the wrong body, twice

It returned the BALL's velocity change since launch, which gravity dominates: a 15 kg ball in free
flight for 0.33 s reports ~49 N.s before touching anything, so every shot cleared the 1 N.s hit
threshold and the reported 74-104 N.s impacts were mostly gravity.

Switching to the dummy's momentum change since launch was no better - a dummy that falls over
accumulates the whole fall, and one shot reported 168.6 N.s for knocking it down. **One physics step
is what separates them**: free fall contributes `g*dt*mass = 2.8 N.s` per step, an impact spikes far
above it. Now per-step, with the hit threshold raised 1 -> 8 N.s. The same scene reads:

```
ball 1 -> Shin_L    42,1 N.s delivered        ball 5 -> Shin_L     9,2 N.s delivered
ball 2 -> Forearm_L  8,1 N.s delivered        ball 7 -> Thigh_L    5,5 N.s MISS (no contact)
ball 4 -> Shin_L    28,6 N.s delivered        ball 8 -> Pelvis    35,0 N.s delivered
```

Real hits, real misses, and magnitudes that match what the training environment delivers.

**Nothing here is a training failure, and no amount of training minutes would have fixed it.** The
perturb policy has to be retrained from the balance brain against a gun that actually connects.

### Correction, same evening: gravity comes back after the approach

Cancelling the ball's gravity for the whole episode was wrong in a way the first measurements did
not show. A shot that MISSES then flies straight on for ever - it never lands, it re-enters, and it
strikes the dummy from behind. A projectile that cannot fall is its own uncontrolled perturbation.

Gravity is now cancelled only for the approach - `BALL_SPAWN_DISTANCE / speed * 1.2`, the nominal
flight plus 20% - after which the ball is an ordinary object that drops, lands and rolls. Both
environments and `MjBallGun` share the rule. Measured, the hit is unchanged and the ball ends up on
the floor where it belongs:

| ball speed | peak delivered | share of in-play time resting on the floor |
|---|---|---|
| 2.0 m/s | 19.9 N.s | 56% |
| 4.0 m/s | 42.1 N.s | 60% |
| 6.0 m/s | 65.2 N.s | 72% |

## 2026-09-10 00:10 — RETRACTION: there was never a sim-to-sim gap. I was reading the wrong row.

Several claims today said the balance policy scores "100% upright at 6 m/s, and 8 m/s" in the CPU
scorer while falling in the Godot scene, and concluded there was a sim-to-sim discrepancy. **All of
those numbers were the quiet-room row.** `eval.py` prints `POLICY, no ball` before `POLICY, under
fire`, and the greps used to read those results took the first block. The same command, showing
both sections:

```
=== POLICY, no ball ===        upright 100.0 %   never fell 100.0 %   time to fall 39.99 s
=== POLICY, under fire ===     upright  10.2 %   never fell   0.0 %   time to fall  4.08 s
```

This is the second time on this project that a partial read of a multi-section report produced a
confident wrong conclusion, and it wasted an evening chasing a gap that does not exist.

### What the policy actually does

Under fire, every 4 s, 16 envs x 40 s - reading the correct section:

| ball | carried | upright | never fell |
|---|---|---|---|
| 2.0 m/s | 30 N.s | 19.1% | **0 of 16** |
| 3.0 m/s | 45 N.s | 15.0% | 0 of 16 |
| 4.0 m/s | 60 N.s | 14.1% | 0 of 16 |
| 5.0 m/s | 75 N.s | 13.1% | 0 of 16 |

And the time-to-fall tracks the FIRST shot exactly - 4.08 s at a 3 s cadence, 5.07 s at 4 s, 6.11 s
at 5 s, 8.06 s at 7 s. **It falls about one second after the first ball, whatever the ball is.** It
cannot take a single real impact of any size.

### Godot agrees, and always did

Same scene, same brain, gun switched off: **100% upright over 30 s, 6.5 cm of travel** - identical
to the scorer's quiet room. Gun on: it falls. The two engines were never in disagreement; the C#
observation build, the frame maps, the action mapping and the ONNX are all correct, and the
gun-off run is the evidence.

### So what is actually missing

Nothing structural. The balance policy has only ever been trained against a projectile that
**missed** - the curriculum reached 2.42 m/s from a 2 m standoff, where a ballistic ball cannot
travel further than 0.60 m. It learned to stand, which it does perfectly, and it has never once
been hit. The perturbation task starts now.

## 2026-09-11 — two lessons from shipping the walk: the contract must carry what `step()` computes, and `cvel` is not a body velocity

**The heading hold existed in training and not in Godot.** From 2026-09-10 22:00 every walk
checkpoint was trained with a heading hold: while the commanded yaw is zero, the env writes a
correction toward the heading the command began on into the yaw slot of the observation, inside
`step()`. `MjPolicyDriver` wrote the raw zero. The observation LAYOUT was identical, every parity
check passed, and the scene would still have run a different policy from the one that was scored -
the Isaac track's contract failure, in a new place. The contract now carries a `command` block
(`yaw_mode: heading_hold`, gain 0.5, the formula and the latch rule), `MjHeadingHold` applies it with
a heading formula identical to the env's over 10,000 random rotations, and a reset forgets the held
heading. **Anything the env computes into the observation is part of the contract, not only the
layout.**

**MuJoCo's `cvel` is referenced to the subtree centre of mass, not to the body.** Read raw,
`cvel[foot, 3:6]` reported -0.14 m/s for a foot moving at +0.70 - the wrong sign - and a capture-step
reward built on it paid for stepping AWAY from the fall. A body's own linear velocity is
`lin + ang x (xpos - subtree_com[root])`, verified by finite difference. Any velocity read from
`cvel` needs that shift, and a finite-difference check rather than trust in the field's name.

`mujoco_rig/csharp_spike/` and `prove2.py`, referenced in earlier entries, have since been removed;
`MjBridge` and the environments are what they became.

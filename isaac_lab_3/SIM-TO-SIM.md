# Sim-to-sim: matching Godot and Isaac

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

# Perturb recovery diagnosis — 2026-09-20

The correction below has since been implemented as `loaded_contact_recovery_v3`.
See [STATUS.md](STATUS.md) for its training experiment and [README.md](README.md)
for current measurement and scoring semantics. This report preserves the original
v2 diagnosis; its saved traces describe the old measurement, not the corrected code.

## Decision

Correct the reward's foot-contact measurement before another training experiment.
The current foot-origin-height proxy frequently calls a weight-bearing, tilted foot
airborne. This removes support reward during the recovery being learned. It is a
demonstrated measurement defect, not proof that it alone caused the rejected
checkpoints or that correcting it will improve survival.

Keep the incumbent exported policy. Do not resume the rejected candidate or increase
movement penalties on the basis of these results.

## Reproducible comparison

- Incumbent: `logs/mujoco/2026-09-11_10-08-42_p0911i_s1/model_503.pt`.
- Candidate: `logs/mujoco/2026-09-20_17-38-23_recovery_reference_dense/model_25.pt`.
- CPU MuJoCo; seed 17; 512 worlds; six seconds; one 8 kg ball at 6 m/s,
  launched at 1.5–1.8 seconds. Deterministic mean actions; no automatic resets.
- The diagnostic observer calls the original physics step exactly once, then reads
  contacts at every physics substep. It does not change actions, physics, RNG state,
  reward weights, policy weights, or promotion criteria.
- Intended target, launch time, and heading match between policies. All 512 shots
  make confirmed body contact in each run. Thus missed projectiles do not explain
  the successful trials in this batch.
- All **1,024 per-world survival and recovery outcomes** match the saved, uninstrumented
  evaluations exactly. Sliding, replants, and action-change aggregates also match.
- Reconstructing returned rewards from recorded terms and the fall penalty has
  maximum absolute error below `9.54e-7` for both policies.

Incumbent: 302 survivors and 277 settled recoveries. Candidate: 295 survivors and
267 recoveries. There are 62 lost recoveries and 52 gained recoveries, rather than
one uniform failure pattern. The existing candidate rejection remains valid.

Raw evidence and executable diagnostic scripts are in `logs/perturb-diagnosis/`:
`trace.py`, `analyze.py`, `render.py`, `incumbent.{json,npz}`,
`candidate.{json,npz}`, and `analysis.txt`. JSON files include checkpoint and plant
hashes. These local experimental artifacts live under the repository's ignored
logs directory; this report records the conclusions in the project documentation.

Original investigation commands (before the v3 implementation; use the current
versioned evaluator for new comparisons rather than overwriting these v2 traces):

```powershell
python logs/perturb-diagnosis/trace.py --checkpoint logs/mujoco/2026-09-11_10-08-42_p0911i_s1/model_503.pt --out logs/perturb-diagnosis/incumbent
python logs/perturb-diagnosis/trace.py --checkpoint logs/mujoco/2026-09-20_17-38-23_recovery_reference_dense/model_25.pt --out logs/perturb-diagnosis/candidate
python logs/perturb-diagnosis/analyze.py
python logs/perturb-diagnosis/render.py
```

## Confirmed defect: load-bearing feet disappear from the reward

Both `rl/perturb_env.py` and `rl/perturb_env_warp.py` use a foot origin's height
above its rest height, with 1.2/2.5 cm hysteresis, to construct `grounded`.
`rl/recovery_reward.py` then excludes an airborne foot from capture-point support.
If both proxy flags are false, support reward is zero.

A foot can rotate onto its heel, toe, or edge and raise its origin by several
centimetres while still carrying substantial body weight. Hysteresis cannot make
origin height a reliable measurement of contact in that configuration.

The following measurements use only control samples before the first fall. Floor
load sums contact normal forces for each foot **and its toe**, with a 5 N threshold:

- During the first 0.5 seconds after impact, **47.9%** of incumbent foot samples
  labelled airborne actually carry floor load. Over 0.5–1.5 seconds, this is **61.1%**.
  The misclassified samples average approximately **558 N** and **504 N**, respectively.
- Both proxy feet are airborne despite at least one loaded foot in **40.3%** of early
  response samples and **46.5%** of the following second's samples. Support reward
  is therefore exactly zero throughout those samples.
- The candidate has the same defect: 48.6%/61.7% false-air classifications and
  40.9%/46.7% falsely unsupported body samples in those windows.
- This is not a threshold artifact: changing the diagnostic load threshold from
  1 N to 20 N changes the incumbent false-air percentages from 47.9% to 47.5%,
  and from 61.2% to 60.7%.

As an **offline reward-only counterfactual**, substitute loaded-foot flags into the
existing support capsule, keeping the recorded motion and every weight fixed.
Incumbent mean support reward rises from **0.706 to 1.227** in the first half-second,
and from **1.024 to 1.677** in the next second. Candidate values change similarly.
This demonstrates missing reward credit, not a trained improvement. Foot-centre
capsules are still an approximation to the true support region.

Changing grounding also exposes motion previously excluded from sliding penalties:
using the same recorded foot-origin velocities with measured grounding changes the
incumbent's mean sliding term from `-0.199` to `-0.392` in the first half-second,
and from `-0.016` to `-0.282` in the following second. A correction must distinguish
actual contact-point slip from foot-origin motion during a heel/toe pivot. The
support-only counterfactual above is not the change in total reward.

The diagnostic reads forces using MuJoCo's documented
[`mj_contactForce`](https://mujoco.readthedocs.io/en/3.4.0/APIreference/APIfunctions.html#mj-contactforce),
whose first contact-frame component is the normal force. The installed GPU backend
also exposes `mujoco_warp.contact_force`, along with contact geometry and world IDs;
there is a supported route to equivalent CPU/GPU measurements.

## What the failures show

The incumbent falls in 210 trials. Of 126 shots aimed at head, chest, or spine,
99 cause falls (78.6%); other targets account for 111 falls in 386 trials (28.8%).
Target labels describe intended aim; the recorded `hit_parts` array also stores
which bodies actually received contact force. Median time from first body contact
to a fall is **1.16 seconds** for incumbent failures and **1.28 seconds** for candidate
failures.

Failed trials generally contain foot repositioning rather than no response. A
diagnostic step requires over 83 ms without appreciable floor load, then a landing
with at least 5 cm foot-origin displacement before the first fall. Under that
definition, 78.6% of incumbent failures include a step. Median first qualifying
landing is 0.375 seconds after impact for failures versus 0.354 seconds for survivors.
These observational statistics do not establish the optimal response timing;
landings can include ballistic motion from the hit. They do not support a blanket
diagnosis of delayed stepping.

Representative replays show tilted-foot support followed by loss of balance,
and cases where similar initial reactions diverge later. They were selected to
show both failure and improvement, not only the candidate's worst outcomes:

- [Both fall, chest-targeted world 4](../logs/perturb-diagnosis/both-fall-high-hit-world-4.mp4).
  At approximately 2.50 seconds, both controllers carry hundreds of newtons through
  the right foot while both reward contact flags say airborne and support reward is zero.
- [Lost recovery, thigh-targeted world 18](../logs/perturb-diagnosis/lost-recovery-world-18.mp4).
  Incumbent recovers; candidate initially stands again but falls at 4.62 seconds.
- [Gained recovery, upper-arm-targeted world 5](../logs/perturb-diagnosis/gained-recovery-world-5.mp4).
  Incumbent falls; candidate recovers.
- [Both recover, pelvis-targeted world 1](../logs/perturb-diagnosis/both-recover-world-1.mp4).

The videos render saved MuJoCo poses with display-only colors. They are not recordings
of the Godot scene. The scene's repeated three-second shots remain a separate validation
requirement after an improved single-hit controller exists.

## Reward balance and secondary observations

Across the incumbent's first 0.5 seconds after impact, mean per-control-step terms are
`limits=-3.391`, `sliding=-0.199`, `action_rate=-0.128`, and
`rapid_replants=-0.0002`. The corresponding support contribution is only `+0.706`.
Joint-limit penalties dominate these movement penalties. Their weight already
existed in the old objective: this is not a newly introduced coefficient bug.

The quiet incumbent also rests multiple joints near or beyond nominal stops, with
mean limit cost about `-2.249`. It is not yet a natural reference pose. Simply
weakening that penalty would conceal this issue, and strengthening movement costs
would not repair the missing support signal. Neither change is recommended as the
first experiment.

A separate six-second quiet-room pilot (64 worlds, seed 117) survived 64/64 with
mean actions, 60/64 with saved Gaussian exploration, and 64/64 with one-quarter
of that exploration scale. The saved mean standard deviation is about 0.177.
Exploration therefore contributes instability, but this short pilot does not
establish it as the cause of the earlier training collapse.

A one-world quiet probe also showed dominant 30 Hz command alternation while the
feet moved only millimetres. In that probe, the incumbent's maximum foot-origin
range over seconds 2–5 was about 3 mm, versus 4 mm for the candidate. Reduced action
oscillation is not equivalent to better foot placement. These are small diagnostic
probes, not new policy acceptance scores.

## Proposed correction and acceptance checks

Make **contact measurement the only training-variable change** in the next experiment:

1. Extract floor-contact load separately in the CPU and GPU adapters, aggregating
   foot and toe geometries by world and side. Do not count ball/self contacts as support.
   Convert both backends to the same force convention and sample at the same physics
   stage. Use a shared load-based hysteresis function with documented units.
2. Feed that state consistently into support reward, landing/replant history, and
   the training recovery indicator. Measure tangential slip at loaded contact points
   rather than treating all foot-origin translation during rotation as sliding.
   Include a stationary-contact heel/toe pivot fixture to verify that distinction.
   Preserve the trained observation layout and
   torque/latency mapping. Do not also change smoothing, reference strength, reward
   weights, authority, or exploration in this experiment.
3. Version the changed reward and reset/warm the critic from the incumbent actor.
   Use direct fixtures for flat feet, heel/toe pivot, unloaded raised feet, actual
   flight, ball-only contact, and partial resets. Require CPU/GPU load and reward
   parity, not merely identical height-proxy arithmetic.
4. Preserve the existing scorer and promotion gates for this first controlled
   comparison. Add the measured-contact results alongside them; do not silently
   replace the historical recovery metric and call the changed score an improvement.
   If the scorer is later migrated, rebaseline both policies under the new version.
5. Compare one short training chunk against the incumbent on the fixed hard-hit
   batch, the 40-second quiet gate, and a separate seed. Stop on regression. Only a
   passing candidate proceeds to repeated-shot scene validation or longer training.

This investigation made no production reward/configuration changes and ran no new
training. No policy was exported, committed, or pushed.

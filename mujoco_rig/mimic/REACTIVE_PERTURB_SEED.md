# Stationary reactive Perturb seed

Latest reward audit, corrections, training results and 600-second standing
retention are recorded in [RECOVERY_LEARNING_AUDIT.md](RECOVERY_LEARNING_AUDIT.md).

Arena override: the user explicitly requested the experimental final actor from
`reactive-ball-arena-continue-5min-03/model.pt`. It is loaded for manual inspection;
the validated 2 kg checkpoint described below remains the admitted seed.

The selected checkpoint is `logs/mimickit-training/reactive-ball-training-2kg-01/best.pt`,
update 32, SHA-256
`01d99760efd142b591aa98d893d563a4413f49ae4eab208db063a2f2f93ced71`.
`seed_manifest.json` binds its Stand parent, stationary reference, character
model, 2 kg projectile, quiet hold, and unseen hit suite. Use this selected
checkpoint for the next mass stage; `model.pt` at update 64 regressed and is
rejected.

## Contract and training

`ReactiveBallTask` starts at one of the eight fixed Stand postures. Its actor
retains the admitted 362-input, 30-output observation/control contract, a
constant reference, zero phase, two-step target delay, and the original rig and
motor limits. The ball is a real free body with contact forces. A saved copy of
`dummy_ball.xml` changes only projectile mass from 8 to 2 kg; the loader verifies
that the character MJCF is identical. The ball position, shot angle, and target
body are not actor inputs. Body state and ground loads provide feedback.

Training draws horizontal launch angles uniformly around the dummy, torso
targets from Chest/Spine/Pelvis, speeds 1–2 m/s, and launch times from steps
60–120. One quarter of episodes are quiet. Before impact, the symmetric
foot-anchor reward preserves two-foot standing. After confirmed ball contact,
the objective rewards upright height and supported settling without a fixed
foot position or chosen stepping leg. No motion clip or left-step clock runs.

The PPO run used 512 GPU worlds, 64 updates, 1,048,576 transitions, a frozen
Stand observation transform, and 16 critic-only warmup updates. Every 16
updates it checked native quiet standing and 32 physical torso hits across
all eight postures and four cardinal directions. The Stand seed recovered
30/32 at 2 kg and 2 m/s; postures 0 and 6 fell on forward chest hits around
four seconds. Update 32 recovers **32/32** while preserving **8/8** native
30-second quiet cases. Updates 48 and 64 lose quiet standing and are rejected.

## Independent validation and limits

- The selected actor passes native and GPU eight-posture 30-second quiet
  checks. Native quiet holds pass **8/8 uninterrupted 600-second** cases;
  minimum per-foot load is 237.66 N and maximum foot error is 7.65 mm.
- A fresh native suite with seed 270935 passes **64/64 confirmed 2 kg hits**
  and **64/64 recoveries** through 33 seconds. Angles vary within each
  quadrant, torso targets vary, and speeds remain 1–2 m/s. Neither foot is
  prescribed; both must be loaded and the root settled for the final two seconds.
- The Newton/MuJoCo-Warp training backend confirms and recovers **8/8**
  forward torso hits at 2 kg with the same selected checkpoint.
- The selected actor recovers **18/32** cardinal 5 kg stress hits, equal to the
  Stand seed's 18/32 at that mass. The next curriculum stage must improve this.
- Quiet cases 0 and 5 creep slowly after many minutes. The worst final-minute
  foot motion is 0.733 mm, within the declared 1 mm gate but above the original
  Stand actor's 0.014 mm. Retain the Stand checkpoint as the quiet-standing
  authority and tighten long-hold retention as the mass curriculum advances.

The first 12-second unseen report incorrectly recorded zero survivors because
its Python-float timeout slightly exceeded the last float32 control timestamp.
It is retained as `report.pre-timeout-fix.json` and
`unseen_report.pre-timeout-fix.json`. The corrected score is 64/64; a separate
33-second unseen suite with a new seed supplies the final admission evidence.

The seed is selected in `MimicPerturbArena.tscn` as the current actor. The scene
now runs an unlimited experimental 5 kg stress test with 1 m/s physical balls
every 2.5 seconds, random angles and torso targets. A headless 20-second run
confirmed eight contacts and no fall. The 5 kg training gate has not admitted
a replacement actor. The scene bundle contains `stand.onnx`, a constant reference,
and no recovery step controller. ONNX action parity is 2.98e-7; a headless Godot
forward chest trial confirmed contact and completed 33 seconds with 6.3 mm
final root error. The 5 kg continuation failed 600-second foot stability and
unseen-hit gates, so heavier impacts remain outside the scene's validated range.
The subsequent 3 kg curriculum reached 29/32 cardinal hits and 64/64 unseen
12-second hits, but only 3/8 long quiet-standing cases passed. That actor also
remains outside the scene.

## Sustained-recovery and repeated-hit pilot

`reactive-ball-5kg-recovery-v2-5min-01` trained for 301.42 synchronized GPU
iteration seconds with 2,048 worlds: 53 PPO updates and 3,473,408 transitions.
Its 12-second episodes retained 25% quiet starts; half of hit episodes scheduled
a second physical 5 kg torso shot at steps 360–420 from an independently sampled
angle. After contact, the objective added a bonus that rises over two continuous
seconds of upright, slow, two-foot-loaded balance. Either foot remains free to
step; no step clock or foot-placement target was introduced.

The 2 kg starting actor passed 8/8 short quiet holds and recovered 23/32
single 5 kg selection hits at 1.5 m/s. The final actor recovered 22/32 and
failed the quiet hold. Direction 2 fell from 8/8 to 5/8 recoveries, so the
directional checkpoint gate rejected it and retained iteration 0 as `best.pt`.
The retained actor recovered 54/64 unseen single hits. In a separate two-shot
native suite, 23/32 second impacts were confirmed and 17/32 trials recovered;
the other trials include falls before the second launch. Selection, unseen,
and repeated-hit gates all remain incomplete. The Arena actor is unchanged.

This pilot establishes that the added settling bonus and second-hit exposure
alone do not prevent directional and quiet-standing regression. The next actor
update needs explicit retention against the admitted quiet actor and balanced
per-direction recovery evidence before any longer continuation or scene export.

`reactive-ball-5kg-recovery-anchor-5min-01` adds a 0.02 maximum Gaussian KL
guard on 128 recorded quiet-standing observations from the admitted starting
actor. It uses the same 5 kg objective and second-shot curriculum, 2,048 GPU
worlds, a 306.66-second training-iteration budget, and a native gate every 16
updates. Over 55 updates it collected 3,604,480 transitions. Update 16 is the
selected `best.pt`: short quiet standing passes, all previously passing hit
directions are retained, and the cardinal 5 kg gate rises from 23/32 to 24/32.
The only added recovery is direction 3 (7/8 to 8/8); direction 0 stays 0/8.
The final actor is identical for measured gate purposes at 24/32. Once the
quiet-state KL reached 0.02, no actor steps were accepted after update 16.

The selected actor recovers 54/64 unseen single hits and 17/32 two-shot trials;
24/32 second impacts are confirmed. GPU short quiet retention passes. This is
measurable retention with a small gain, not a robust 5 kg recovery controller.
The incomplete directional, unseen, and repeated-hit gates block Arena export.
Further GPU time with this saturated anchor does not teach the missing forward
recovery. The next experiment needs targeted forward recovery states and a
curriculum that can improve them while keeping the quiet and directional gates.

```powershell
& $python -B -m mujoco_rig.mimic.train_reactive_ball `
  --mimickit D:/Proyectos/Juegos/Tools/MimicKit `
  --initialize-from logs/mimickit-training/reactive-ball-training-2kg-01/best.pt `
  --reference logs/mimickit-steps/continuous-stand-02/stationary_reference.npz `
  --ball-model logs/mimickit-training/reactive-ball-seed-01/ball_5.xml `
  --out logs/mimickit-training/<new-5kg-run>
```

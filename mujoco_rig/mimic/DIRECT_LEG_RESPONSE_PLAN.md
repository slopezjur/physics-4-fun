# Direct leg-response screen - 2026-10-09

This diagnostic follows the rejected direct-recovery PPO pilot. No new policy
training, Arena replacement or physics-parameter changes are included.

1. Cross both hit onsets (1.0 and 1.7 seconds) with both axes, both signs, both
   forces (150/300 N), and all eight reset poses: 128 frozen-source cases.
   Reproduce the 64 previous matching cases exactly.
2. On 150 N X pushes, test six fixed motor combinations at signed amplitudes
   -0.075, -0.025, +0.025 and +0.075 rad, at both onsets: 768 cases. Combinations
   are bilateral ankle pitch, bilateral hip pitch, left flexion, right flexion,
   lateral transfer and split hip pitch. Authored axes are rx for sagittal pitch
   and rz for lateral rotation. Flexion coefficients are hip +0.5, knee -1,
   ankle +0.5; transfer is hips +1 and ankles -1; split is left hip +1/right -1.
3. Every correction starts one control interval after the pulse begins, stays
   requested for 12 intervals, then returns to zero. Keep the existing 0.1-rad
   envelope, 0.02-rad/interval slew and two-interval motor delay. No state resets,
   root forces other than the prescribed chest push, or trajectory injection.
   Exact replay must verify identical qpos/qvel before issue and during the two
   delay intervals. These are scripted diagnostic additions, not learned steps.
4. Rank profiles using only poses 0-5 across both timings: net settled-recovery
   gain, newly rescued cases, survival duration, then smaller amplitude. The
   other poses are reported separately and cannot influence selection.
5. For that one selected profile, test both timings with 300 N pushes, a quiet
   six-second episode, and a short source-feedback ablation. During the correction
   plus return/delay window, replay the original baseline source actions on all
   16 leg motors while the remaining motors retain live feedback. Run an identical
   replay with zero addition first; it must reproduce the whole baseline exactly.
   This separates the frozen source's leg response from the added intervention.

Maximum budget: 63 rollouts / 1,008 six-second physical cases; no optimizer steps.
Keep compact traces and provenance, without model checkpoints. Earlier sealed
source files remain unchanged.

Primary outcome: final two seconds continuously meeting the existing balance and
posture gates, with both feet carrying at least 30 N and no physical failure.
Report gained/lost recoveries separately, not just total survival. Independently
audit foot unloading, collision-box clearance, relative placement and landing/
settling using the existing corrective-step audit. Source opposition is measured
as its nominal target change projected onto the actual added correction, compared
with the paired baseline; it is a trajectory response, not proof of causality.
The replay ablation supplies a separate intervention on leg feedback.

A rescue can establish a useful diagnostic condition for subsequent learning.
Only audited placement plus settling establishes a mechanical recovery step.
Failure of this finite probe set does not prove recovery is impossible. Neither
outcome establishes learned or human-natural recovery, nor projectile admission.

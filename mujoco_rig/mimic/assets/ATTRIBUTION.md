# Motion reference attribution

The 100STYLE Dataset - Ian Mason.

Source: <https://www.ianxmason.com/100style/>

Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0):
<https://creativecommons.org/licenses/by/4.0/>.

This reference is adapted from `Neutral_ID.bvh`, frames 700 through 1059
(every second frame). Changes: cropped and resampled; axes and skeleton retargeted
to the project dummy using fixed-foot, collision-aware inverse kinematics;
temporally smoothed; uncontrolled neck/wrist joints held at zero.
The excerpt is not looped. Exact input hashes and validation are recorded in
`stand_reference.json`.

The experimental `steps/forward_reference.npz` and `steps/backward_reference.npz`
adapt `Neutral_FW.bvh` frames [480, 720) and `Neutral_BW.bvh` frames [520, 760),
respectively, at stride 2. These are walking priors, not recorded impact recoveries.
Changes: crop/resample, semantic-axis remapping, scale 0.9, inferred stance locks,
smoothed foot pitch and moving-foot collision-aware IK with joint continuity bounds.
Walking root height is projected onto the lowest sole to preserve floor support;
inferred stance targets are grounded. These are reference-processing changes.
The rig, physical joint limits and motor strengths are unchanged. Each accompanying
JSON manifest records the source hashes, license, adaptation and kinematic checks;
passing those checks does not establish dynamic tracking or impact recovery.

Retain this attribution, source link, license link and indication of changes when
redistributing the reference or an adapted version. No endorsement is implied.

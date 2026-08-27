using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Hip strategy: the medium-perturbation balance tier between the ankle strategy and
/// capture-point stepping (SRP). Active only in double support, where it owns the
/// thigh/shin/spine feed-forward offsets. Three channels:
///  1. Posture: rights the unactuated pelvis/torso by counter-rotating hips and spine;
///     the reaction torque against the planted feet brakes the root fall.
///  2. CoM arrest: drives hip/torso targets from the yaw-level sagittal CoM error so the
///     body mass is pulled back over the support polygon in the band where the ankle
///     strategy is saturated but the ICP has not escaped yet.
///  3. CoM height: symmetric knee extension/flexion holds the pelvis at its rest height
///     above the ground instead of free-riding on the closed-chain equilibrium.
/// </summary>
public class HipStrategyModule : IBalanceStrategy
{
    // Posture channel (tilt righting)
    public float HipPitchGain { get; set; } = 0.8f;
    public float HipRollGain { get; set; } = 0.8f;
    public float MaxHipPitchOffset { get; set; } = 0.40f;
    public float MaxHipRollOffset { get; set; } = 0.30f;
    public float SpinePostureScale { get; set; } = 0.5f;
    public float MaxSpinePitchOffset { get; set; } = 0.30f;
    public float MaxSpineRollOffset { get; set; } = 0.20f;

    // CoM arrest channel: rad of hip offset per meter of yaw-level sagittal CoM error
    public float ComArrestGain { get; set; } = 2.0f;
    public float ComArrestDamping { get; set; } = 1.0f;
    public float MaxComArrestOffset { get; set; } = 0.20f;
    public float SpineArrestScale { get; set; } = 0.5f;

    // CoM height channel: rad of knee offset per meter of pelvis height error
    public float HeightGain { get; set; } = 0.8f;
    public float MaxHeightOffset { get; set; } = 0.12f;

    private float _restPelvisHeight = -1.0f; // measured lazily from the spawn pose

    public void Reset()
    {
        _restPelvisHeight = -1.0f;
    }

    public void Apply(in BalanceContext context)
    {
        bool hasGroundContact = context.IsGroundedL || context.IsGroundedR;
        bool isBalancedOrStumbling = context.State == RagdollState.Balanced || context.State == RagdollState.Stumbling;
        if (!(isBalancedOrStumbling && context.Strength > 0.01f && hasGroundContact && context.CurrentStepPhase == StepPhase.DoubleSupport))
        {
            return;
        }

        IBoneState pelvis = context.Pelvis;
        IBoneState? spine = context.Spine;
        IBoneState? thighL = context.ThighL;
        IBoneState? thighR = context.ThighR;
        IBoneState? shinL = context.ShinL;
        IBoneState? shinR = context.ShinR;
        IBoneState? footL = context.FootL;
        IBoneState? footR = context.FootR;
        Vector3 centerOfMass = context.CenterOfMass;
        Vector3 centerOfMassVelocity = context.CenterOfMassVelocity;
        float groundY = (context.GroundPointL.Y + context.GroundPointR.Y) * 0.5f;
        float strength = context.Strength;

        if (footL == null || footR == null || !footL.IsValid || !footR.IsValid)
        {
            return;
        }

        // 1. Posture righting (pelvis attitude error -> hip and spine counter-rotation)
        Vector3 localUp = pelvis.GlobalTransform.Basis.Inverse() * Vector3.Up;
        float pitchError = Mathf.Clamp(Mathf.Atan2(localUp.Z, localUp.Y) * HipPitchGain, -MaxHipPitchOffset, MaxHipPitchOffset);
        float rollError = Mathf.Clamp(Mathf.Atan2(-localUp.X, localUp.Y) * HipRollGain, -MaxHipRollOffset, MaxHipRollOffset);

        // 2. Sagittal CoM arrest in a yaw-level frame (pelvis pitch cannot mask divergence).
        // Local +Z is forward; a forward error demands the +pitch braking offset that
        // rotates the thighs backward against the planted feet (established convention).
        Basis levelBasis = BiomechanicalKinematics.ComputeLevelBasis(pelvis.GlobalTransform.Basis);
        Vector3 supportCenter = (footL.GlobalPosition + footR.GlobalPosition) * 0.5f;
        Vector3 comLocal = levelBasis.Inverse() * (centerOfMass - supportCenter);
        Vector3 velLocal = levelBasis.Inverse() * centerOfMassVelocity;
        float arrestPitch = Mathf.Clamp(
            (comLocal.Z * ComArrestGain) + (velLocal.Z * ComArrestDamping),
            -MaxComArrestOffset, MaxComArrestOffset);

        // 3. CoM height: flex knees (-X) when the pelvis rides low.
        // We clamp this to be strictly <= 0.0f to prevent knee hyperextension when riding high.
        if (_restPelvisHeight < 0.0f)
        {
            _restPelvisHeight = pelvis.GlobalPosition.Y - groundY;
        }
        float heightError = _restPelvisHeight - (pelvis.GlobalPosition.Y - groundY);
        float heightOffset = Mathf.Clamp(heightError * HeightGain, -MaxHeightOffset, 0.0f) * strength;
        
        // 4. Hip Pitch Calculation: Apply balance correction + squat compensation
        // If the knee bends by `heightOffset` (negative), the hip must bend by `-heightOffset` (positive) to keep the torso upright.
        float finalHipPitch = (pitchError + arrestPitch) * strength - heightOffset;
        Quaternion hipOffset = Quaternion.FromEuler(new Vector3(finalHipPitch, 0.0f, rollError * strength));

        if (thighL != null && thighL.IsValid)
        {
            thighL.FeedForwardTargetOffset = hipOffset;
        }
        if (thighR != null && thighR.IsValid)
        {
            thighR.FeedForwardTargetOffset = hipOffset;
        }

        if (spine != null && spine.IsValid)
        {
            Vector3 spineLocalUp = spine.GlobalTransform.Basis.Inverse() * Vector3.Up;
            float spinePitch = Mathf.Clamp(Mathf.Atan2(spineLocalUp.Z, spineLocalUp.Y) * SpinePostureScale, -MaxSpinePitchOffset, MaxSpinePitchOffset);
            float spineRoll = Mathf.Clamp(Mathf.Atan2(-spineLocalUp.X, spineLocalUp.Y) * SpinePostureScale, -MaxSpineRollOffset, MaxSpineRollOffset);
            // Same corrective convention as the hip: +error tips the target backward, righting the torso.
            // The arrest term counter-leans the torso mass against the CoM error.
            spine.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3((spinePitch + arrestPitch * SpineArrestScale) * strength, 0.0f, spineRoll * strength));
        }

        Quaternion kneeOffset = Quaternion.FromEuler(new Vector3(heightOffset, 0.0f, 0.0f));
        if (shinL != null && shinL.IsValid)
        {
            shinL.FeedForwardTargetOffset = kneeOffset;
        }
        if (shinR != null && shinR.IsValid)
        {
            shinR.FeedForwardTargetOffset = kneeOffset;
        }
    }
}

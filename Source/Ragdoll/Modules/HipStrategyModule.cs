using Godot;

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
public class HipStrategyModule
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

    public void Apply(
        ActiveBone pelvis,
        ActiveBone? spine,
        ActiveBone? thighL,
        ActiveBone? thighR,
        ActiveBone? shinL,
        ActiveBone? shinR,
        ActiveBone? footL,
        ActiveBone? footR,
        Vector3 centerOfMass,
        Vector3 centerOfMassVelocity,
        float groundY,
        float strength)
    {
        if (footL == null || footR == null || !GodotObject.IsInstanceValid(footL) || !GodotObject.IsInstanceValid(footR))
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
        Basis levelBasis = ComputeLevelBasis(pelvis.GlobalTransform.Basis);
        Vector3 supportCenter = (footL.GlobalPosition + footR.GlobalPosition) * 0.5f;
        Vector3 comLocal = levelBasis.Inverse() * (centerOfMass - supportCenter);
        Vector3 velLocal = levelBasis.Inverse() * centerOfMassVelocity;
        float arrestPitch = Mathf.Clamp(
            (comLocal.Z * ComArrestGain) + (velLocal.Z * ComArrestDamping),
            -MaxComArrestOffset, MaxComArrestOffset);

        Quaternion hipOffset = Quaternion.FromEuler(new Vector3((pitchError + arrestPitch) * strength, 0.0f, rollError * strength));
        if (thighL != null && GodotObject.IsInstanceValid(thighL))
        {
            thighL.FeedForwardTargetOffset = hipOffset;
        }
        if (thighR != null && GodotObject.IsInstanceValid(thighR))
        {
            thighR.FeedForwardTargetOffset = hipOffset;
        }

        if (spine != null && GodotObject.IsInstanceValid(spine))
        {
            Vector3 spineLocalUp = spine.GlobalTransform.Basis.Inverse() * Vector3.Up;
            float spinePitch = Mathf.Clamp(Mathf.Atan2(spineLocalUp.Z, spineLocalUp.Y) * SpinePostureScale, -MaxSpinePitchOffset, MaxSpinePitchOffset);
            float spineRoll = Mathf.Clamp(Mathf.Atan2(-spineLocalUp.X, spineLocalUp.Y) * SpinePostureScale, -MaxSpineRollOffset, MaxSpineRollOffset);
            // Same corrective convention as the hip: +error tips the target backward, righting the torso.
            // The arrest term counter-leans the torso mass against the CoM error.
            spine.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3((spinePitch + arrestPitch * SpineArrestScale) * strength, 0.0f, spineRoll * strength));
        }

        // 3. CoM height: extend (+X, anatomical knee extension) when the pelvis rides low,
        // flex when it rides high. Symmetric on both knees; owns the shin offsets in
        // double support, where no other module writes them.
        if (_restPelvisHeight < 0.0f)
        {
            _restPelvisHeight = pelvis.GlobalPosition.Y - groundY;
        }
        float heightError = _restPelvisHeight - (pelvis.GlobalPosition.Y - groundY);
        float heightOffset = Mathf.Clamp(heightError * HeightGain, -MaxHeightOffset, MaxHeightOffset) * strength;
        Quaternion kneeOffset = Quaternion.FromEuler(new Vector3(heightOffset, 0.0f, 0.0f));
        if (shinL != null && GodotObject.IsInstanceValid(shinL))
        {
            shinL.FeedForwardTargetOffset = kneeOffset;
        }
        if (shinR != null && GodotObject.IsInstanceValid(shinR))
        {
            shinR.FeedForwardTargetOffset = kneeOffset;
        }
    }

    /// <summary>
    /// Builds a level (yaw-only) basis from the pelvis orientation, flattening its
    /// forward axis onto the horizontal plane so pitch/roll do not leak into local Y.
    /// </summary>
    private static Basis ComputeLevelBasis(Basis pelvisBasis)
    {
        Vector3 forward = -pelvisBasis.Z;
        forward.Y = 0.0f;
        if (forward.LengthSquared() < 1e-6f)
        {
            forward = new Vector3(0.0f, 0.0f, -1.0f);
        }

        forward = forward.Normalized();
        Vector3 zAxis = -forward;
        Vector3 xAxis = Vector3.Up.Cross(zAxis).Normalized();
        return new Basis(xAxis, Vector3.Up, zAxis);
    }
}

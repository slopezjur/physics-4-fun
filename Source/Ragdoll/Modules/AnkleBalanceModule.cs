using Godot;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Encapsulates ankle ground reaction strategy and ground torque coupling in double support (SRP).
/// </summary>
public class AnkleBalanceModule
{
    // Effective gains in consistent units: radians of ankle offset per meter of CoM error.
    // DC position hold comes from the integral term, NOT from high proportional gain: pushing G
    // above ~5 holds position but rings against the foot actuator's phase lag (growing sway that
    // topples the body after a few seconds). Moderate P for dynamics, clamped I for position.
    public float AnklePitchGain { get; set; } = 3.0f;
    public float AnklePitchDamping { get; set; } = 3.5f; // kd = 2 * zeta * sqrt(kp), zeta ~ 1.0
    public float AnkleRollGain { get; set; } = 2.5f;
    public float AnkleRollDamping { get; set; } = 3.0f;

    // Integral channel: rad of offset per (m·s) of sustained CoM error. Charges to the DC-hold
    // offset (~5.4 * error, from the foot actuator's real ~76 Nm/rad stiffness) in ~1 s.
    public float AnkleIntegralGain { get; set; } = 5.0f;
    public float MaxIntegralOffset { get; set; } = 0.20f; // rad, anti-windup clamp per axis

    private Vector2 _comErrorIntegral = Vector2.Zero; // (x: lateral, y: sagittal) in m·s

    /// <summary>Discharges the integrator on state changes (falls, recoveries) to avoid stale offsets.</summary>
    public void Reset()
    {
        _comErrorIntegral = Vector2.Zero;
    }

    // thighL/thighR are kept for signature compatibility with BalanceController;
    // hip posture correction lives in BalanceController, so they are unused here.
    public void ApplyBalance(
        ActiveBone pelvis,
        ActiveBone? footL,
        ActiveBone? footR,
        ActiveBone? thighL,
        ActiveBone? thighR,
        Vector3 centerOfMass,
        float strength,
        float delta)
    {
        if (footL == null || footR == null || !GodotObject.IsInstanceValid(footL) || !GodotObject.IsInstanceValid(footR))
        {
            return;
        }

        Vector3 supportCenter = (footL.GlobalPosition + footR.GlobalPosition) * 0.5f;
        Vector3 comError = centerOfMass - supportCenter;
        Vector3 comVelocity = pelvis.LinearVelocity;

        // Charge the anti-windup-clamped integrator; opposite-sign errors discharge it naturally.
        // The raw integral is clamped so its offset contribution never exceeds MaxIntegralOffset radians.
        if (delta > 0.0f)
        {
            float integralLimit = MaxIntegralOffset / Mathf.Max(0.001f, AnkleIntegralGain);
            _comErrorIntegral.X = Mathf.Clamp(_comErrorIntegral.X + comError.X * delta, -integralLimit, integralLimit);
            _comErrorIntegral.Y = Mathf.Clamp(_comErrorIntegral.Y + comError.Z * delta, -integralLimit, integralLimit);
        }

        // Rig faces -Z: a forward CoM error (comError.Z < 0) requires a plantarflexion
        // moment (negative X rotation) to push the CoP forward and brake the fall.
        float pitchAngle = (AnklePitchGain * comError.Z) + (AnklePitchDamping * comVelocity.Z) + (AnkleIntegralGain * _comErrorIntegral.Y);
        // Lateral strategy: roll the foot to shift the CoP toward the CoM error side.
        float rollAngle = (-AnkleRollGain * comError.X) - (AnkleRollDamping * comVelocity.X) - (AnkleIntegralGain * _comErrorIntegral.X);

        pitchAngle = Mathf.Clamp(pitchAngle * strength, -0.60f, 0.60f);
        rollAngle = Mathf.Clamp(rollAngle * strength, -0.35f, 0.35f);

        Quaternion ankleOffset = Quaternion.FromEuler(new Vector3(pitchAngle, 0.0f, rollAngle));

        // Planar floor alignment through local feed-forward offset
        Vector3 footUpL = footL.GlobalTransform.Basis.Y.Normalized();
        Vector3 tiltAxisL = footUpL.Cross(Vector3.Up);
        float tiltAngleL = Mathf.Clamp(tiltAxisL.Length(), 0.0f, 0.35f);
        Quaternion planarOffsetL = Quaternion.Identity;
        if (tiltAngleL > 0.001f)
        {
            Vector3 localAxisL = (footL.GlobalTransform.Basis.Inverse() * tiltAxisL).Normalized();
            planarOffsetL = new Quaternion(localAxisL, tiltAngleL * strength);
        }

        Vector3 footUpR = footR.GlobalTransform.Basis.Y.Normalized();
        Vector3 tiltAxisR = footUpR.Cross(Vector3.Up);
        float tiltAngleR = Mathf.Clamp(tiltAxisR.Length(), 0.0f, 0.35f);
        Quaternion planarOffsetR = Quaternion.Identity;
        if (tiltAngleR > 0.001f)
        {
            Vector3 localAxisR = (footR.GlobalTransform.Basis.Inverse() * tiltAxisR).Normalized();
            planarOffsetR = new Quaternion(localAxisR, tiltAngleR * strength);
        }

        footL.FeedForwardTargetOffset = (ankleOffset * planarOffsetL).Normalized();
        footR.FeedForwardTargetOffset = (ankleOffset * planarOffsetR).Normalized();
    }
}

using System;
using Godot;

namespace Physics4Fun.Core.Math;

/// <summary>
/// Proportional-Integral-Derivative (PID) controller operating on 3D rotations (SO(3)).
/// Computes corrective torque from rotational displacement and angular velocity.
/// </summary>
public sealed class PidController3D
{
    public float ProportionalGain { get; set; }
    public float IntegralGain { get; set; }
    public float DerivativeGain { get; set; }
    public float MaxTorque { get; set; }
    public float MaxIntegral { get; set; }

    /// <summary>
    /// Effective rotational inertia (kg·m²) used by the SPD implicit denominator.
    /// Should approximate the loaded inertia seen by the joint, not the raw body inertia.
    /// </summary>
    public float EffectiveInertia { get; set; } = 0.12f;

    // EMA smoothing factor for the measured angular velocity feeding the D term
    private const float AngularVelocityFilterAlpha = 0.4f;

    private Vector3 _integralError = Vector3.Zero;
    private Vector3 _filteredAngularVelocity = Vector3.Zero;

    public PidController3D(
        float proportionalGain = 500.0f,
        float derivativeGain = 50.0f,
        float integralGain = 0.0f,
        float maxTorque = 10000.0f,
        float maxIntegral = 100.0f)
    {
        ProportionalGain = proportionalGain;
        DerivativeGain = derivativeGain;
        IntegralGain = integralGain;
        MaxTorque = maxTorque;
        MaxIntegral = maxIntegral;
    }

    /// <summary>
    /// Computes the required torque to align current rotation to target rotation.
    /// </summary>
    /// <param name="currentRotation">Current orientation in global or reference frame.</param>
    /// <param name="targetRotation">Desired target orientation in the same reference frame.</param>
    /// <param name="currentAngularVelocity">Current angular velocity (rad/s) in the reference frame.</param>
    /// <param name="delta">Physics delta time in seconds.</param>
    /// <param name="targetAngularVelocity">Optional desired angular velocity (rad/s).</param>
    /// <returns>Torque vector to be applied to the rigid body.</returns>
    public Vector3 Update(
        Quaternion currentRotation,
        Quaternion targetRotation,
        Vector3 currentAngularVelocity,
        float delta,
        Vector3? targetAngularVelocity = null)
    {
        if (delta <= 0.0f)
        {
            return Vector3.Zero;
        }

        // Relative rotation from current to target: target = qError * current  =>  qError = target * current^-1
        Quaternion qError = targetRotation * currentRotation.Inverse();

        // Enforce shortest arc path (quaternion double cover)
        if (qError.W < 0.0f)
        {
            qError = new Quaternion(-qError.X, -qError.Y, -qError.Z, -qError.W);
        }

        Vector3 angularError = QuaternionToRotationVector(qError);

        // Tan-Liu-Turk Stable Proportional-Derivative (SPD) formulation
        // Ensures unconditional discrete numerical stability (prevents discrete limit-cycle chatter and physics explosion)
        float inertia = Mathf.Max(0.01f, EffectiveInertia);
        float denominator = 1.0f + (DerivativeGain * delta / inertia) + (ProportionalGain * delta * delta / inertia);

        // Implicitly stabilized proportional and derivative terms
        Vector3 pTerm = (ProportionalGain / denominator) * angularError;

        // Integral term with anti-windup clamping
        if (IntegralGain > 0.0f)
        {
            _integralError += angularError * delta;
            if (MaxIntegral > 0.0f)
            {
                _integralError = _integralError.LimitLength(MaxIntegral);
            }
        }
        else
        {
            _integralError = Vector3.Zero;
        }
        Vector3 iTerm = (IntegralGain / denominator) * _integralError;

        // Low-pass filter (EMA) on the measured relative angular velocity to suppress D-term chatter
        _filteredAngularVelocity += (currentAngularVelocity - _filteredAngularVelocity) * AngularVelocityFilterAlpha;

        Vector3 targetAngVel = targetAngularVelocity ?? Vector3.Zero;
        Vector3 angVelError = targetAngVel - _filteredAngularVelocity;
        Vector3 dTerm = (DerivativeGain / denominator) * angVelError;

        // Clamp the D-term contribution separately so the P term always retains authority
        if (MaxTorque > 0.0f)
        {
            float maxDerivativeTorque = MaxTorque * 0.5f;
            if (dTerm.LengthSquared() > maxDerivativeTorque * maxDerivativeTorque)
            {
                dTerm = dTerm.Normalized() * maxDerivativeTorque;
            }
        }

        Vector3 totalTorque = pTerm + iTerm + dTerm;

        // Clamp torque output to prevent physics explosion
        if (MaxTorque > 0.0f && totalTorque.LengthSquared() > MaxTorque * MaxTorque)
        {
            totalTorque = totalTorque.Normalized() * MaxTorque;
        }

        return totalTorque;
    }

    /// <summary>
    /// Converts a unit quaternion into an angle-axis rotation vector (Lie algebra log map).
    /// </summary>
    public static Vector3 QuaternionToRotationVector(Quaternion q)
    {
        float sinHalfAngle = Mathf.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z);
        if (sinHalfAngle < 1e-6f)
        {
            // First-order Taylor approximation for small angles
            return 2.0f * new Vector3(q.X, q.Y, q.Z);
        }

        float halfAngle = Mathf.Atan2(sinHalfAngle, Mathf.Clamp(q.W, -1.0f, 1.0f));
        float angle = 2.0f * halfAngle;
        Vector3 axis = new Vector3(q.X, q.Y, q.Z) / sinHalfAngle;

        return axis * angle;
    }

    /// <summary>
    /// Resets accumulated integral error.
    /// </summary>
    public void Reset()
    {
        _integralError = Vector3.Zero;
        _filteredAngularVelocity = Vector3.Zero;
    }
}

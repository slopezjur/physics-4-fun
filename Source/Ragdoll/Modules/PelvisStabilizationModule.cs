using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Pelvis Protected Balance Region (Euphoria DMS): the pelvis is the unactuated skeletal root, so
/// it is stabilized directly with an attitude PD torque; the counter-torque is distributed across
/// the grounded feet as reaction against the ground, keeping the interaction internal (SRP).
/// </summary>
public class PelvisStabilizationModule : IBalanceStrategy
{
    public float Gain { get; set; } = 600.0f;
    public float Damping { get; set; } = 20.0f;
    public float MaxTorque { get; set; } = 300.0f;

    public Vector3 LastTorque { get; private set; } = Vector3.Zero;

    // EMA smoothing factor for the pelvis stabilizer's angular velocity (per 120 Hz tick)
    private const float AngularVelocityFilterAlpha = 0.25f;
    private Vector3 _filteredAngularVelocity = Vector3.Zero;

    public void Reset()
    {
        _filteredAngularVelocity = Vector3.Zero;
    }

    /// <summary>Zeroes only the reported torque telemetry, leaving the EMA filter state untouched.</summary>
    public void ClearTorque()
    {
        LastTorque = Vector3.Zero;
    }

    public void Apply(in BalanceContext context)
    {
        // Only active during upright balance phases. Disabled during recovery to prevent artificial
        // floating/dragging on the floor. Unlike a grounding/strength gate failure below, this
        // outer gate does NOT zero LastTorque — it leaves the previous tick's value untouched.
        if (context.State != RagdollState.Balanced && context.State != RagdollState.Stumbling)
        {
            return;
        }

        LastTorque = Vector3.Zero;

        bool groundedL = context.IsGroundedL && context.FootL != null && GodotObject.IsInstanceValid(context.FootL);
        bool groundedR = context.IsGroundedR && context.FootR != null && GodotObject.IsInstanceValid(context.FootR);
        if ((!groundedL && !groundedR) || context.Strength <= 0.01f)
        {
            return;
        }

        ActiveBone pelvis = context.Pelvis;

        // Axis-angle attitude error between pelvis up axis and world up (magnitude ~ sin(angle))
        Vector3 pelvisUp = pelvis.GlobalTransform.Basis.Y.Normalized();
        Vector3 attitudeError = pelvisUp.Cross(Vector3.Up);

        // Low-pass filter the pelvis angular velocity: joint reaction chatter (~20 rad/s at 120 Hz)
        // would otherwise dominate the damping term and pump energy through the grounded feet
        _filteredAngularVelocity += (pelvis.AngularVelocity - _filteredAngularVelocity) * AngularVelocityFilterAlpha;

        Vector3 torque = (Gain * attitudeError) - (Damping * _filteredAngularVelocity);
        float maxTorque = MaxTorque * context.Strength;
        if (torque.LengthSquared() > maxTorque * maxTorque)
        {
            torque = torque.Normalized() * maxTorque;
        }

        LastTorque = torque;
        pelvis.ApplyTorque(torque);

        int groundedCount = (groundedL ? 1 : 0) + (groundedR ? 1 : 0);
        Vector3 reaction = -torque / groundedCount;
        if (groundedL)
        {
            context.FootL!.ApplyTorque(reaction);
        }
        if (groundedR)
        {
            context.FootR!.ApplyTorque(reaction);
        }
    }
}

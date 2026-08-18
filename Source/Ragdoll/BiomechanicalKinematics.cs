using System.Collections.Generic;
using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Shared biomechanical kinematics math used across balance strategy modules (DRY): yaw-level
/// reference frames, Instantaneous Capture Point projection, and mass-weighted center of mass.
/// </summary>
public static class BiomechanicalKinematics
{
    /// <summary>
    /// Builds a level (yaw-only) basis from a pelvis orientation, flattening its forward axis
    /// onto the horizontal plane so pitch/roll do not leak into local Y.
    /// </summary>
    public static Basis ComputeLevelBasis(Basis pelvisBasis)
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

    /// <summary>
    /// Projects the horizontal Instantaneous Capture Point from CoM position/velocity and CoM
    /// height above the ground (orbital inverted-pendulum model, omega0 = sqrt(g / height)).
    /// </summary>
    public static Vector3 ComputeInstantaneousCapturePoint(Vector3 centerOfMass, Vector3 comLinearVelocity, float comHeightAboveGround)
    {
        float safeHeight = Mathf.Max(0.2f, comHeightAboveGround);
        float omega0 = Mathf.Sqrt(9.81f / safeHeight);
        Vector3 flatVelocity = new Vector3(comLinearVelocity.X, 0.0f, comLinearVelocity.Z);
        return centerOfMass + (flatVelocity / omega0);
    }

    /// <summary>
    /// Mass-weighted center of mass and CoM velocity across all active bones.
    /// Returns false (leaving the out params at Vector3.Zero) when there is nothing to weight.
    /// </summary>
    public static bool TryComputeCenterOfMass(IReadOnlyList<ActiveBone> bones, float totalMass, out Vector3 centerOfMass, out Vector3 centerOfMassVelocity)
    {
        centerOfMass = Vector3.Zero;
        centerOfMassVelocity = Vector3.Zero;

        if (bones.Count == 0 || totalMass <= 0.0f)
        {
            return false;
        }

        Vector3 weightedSum = Vector3.Zero;
        Vector3 weightedVelocitySum = Vector3.Zero;
        foreach (var bone in bones)
        {
            if (GodotObject.IsInstanceValid(bone))
            {
                weightedSum += bone.GlobalPosition * bone.Mass;
                weightedVelocitySum += bone.LinearVelocity * bone.Mass;
            }
        }

        centerOfMass = weightedSum / totalMass;
        centerOfMassVelocity = weightedVelocitySum / totalMass;
        return true;
    }
}

using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Evaluates pelvis/body coordinate frames relative to world gravity to determine anatomical orientation.
/// Adheres to Single Responsibility Principle (SRP) by separating orientation math from balance physics.
/// </summary>
public static class OrientationClassifier
{
    // Hysteresis state: enter Upright when upDot > 0.70, remain Upright until upDot < 0.45
    private static RagdollOrientation _previous = RagdollOrientation.Upright;

    public static RagdollOrientation Classify(Basis pelvisBasis)
    {
        Vector3 pelvisUp = pelvisBasis.Y.Normalized();
        Vector3 pelvisForward = -pelvisBasis.Z.Normalized();

        float upDot = pelvisUp.Dot(Vector3.Up);
        float forwardDot = pelvisForward.Dot(Vector3.Up);

        if (upDot > 0.70f || (_previous == RagdollOrientation.Upright && upDot > 0.45f))
        {
            _previous = RagdollOrientation.Upright;
            return RagdollOrientation.Upright;
        }

        if (forwardDot > 0.20f)
        {
            _previous = RagdollOrientation.Supine;
            return RagdollOrientation.Supine; // Chest facing sky (on back)
        }

        if (forwardDot < -0.20f)
        {
            _previous = RagdollOrientation.Prone;
            return RagdollOrientation.Prone;  // Chest facing ground (on belly)
        }

        _previous = RagdollOrientation.Side;
        return RagdollOrientation.Side;
    }
}

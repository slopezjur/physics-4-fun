using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// Strategy interface for evaluating procedural bone target rotations.
/// Follows the Open/Closed Principle (OCP) allowing new motion patterns to be added without modifying existing code.
/// </summary>
public interface IMotionTrajectory
{
    Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized);
}

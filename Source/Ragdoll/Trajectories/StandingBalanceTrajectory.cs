using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// Trajectory strategy for natural breathing, micro-sway, and solid anti-buckle biped stance.
/// </summary>
public class StandingBalanceTrajectory : IMotionTrajectory
{
    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        return boneName switch
        {
            "Pelvis" => Quaternion.Identity,
            "Spine" => Quaternion.Identity,
            "Chest" => Quaternion.Identity,
            "Head" => Quaternion.Identity,
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(0.05f, 0.0f, -0.10f)),
            "Forearm_L" => Quaternion.FromEuler(new Vector3(0.15f, 0.0f, 0.0f)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(0.05f, 0.0f, 0.10f)),
            "Forearm_R" => Quaternion.FromEuler(new Vector3(0.15f, 0.0f, 0.0f)),
            // Locked anti-buckle stance: Identity keeps knees and ankles strictly aligned with the rest pose
            "Thigh_L" or "Thigh_R" => Quaternion.Identity,
            "Shin_L" or "Shin_R" => Quaternion.Identity,
            "Foot_L" or "Foot_R" => Quaternion.Identity,
            _ => Quaternion.Identity
        };
    }
}






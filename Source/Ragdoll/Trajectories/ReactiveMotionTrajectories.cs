using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// Strategy for stumbling reaction with dynamic arm windmilling and leg recovery stepping.
/// </summary>
public class StumbleTrajectory : IMotionTrajectory
{
    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        float windL = Mathf.Sin(globalTime * 7.5f) * 0.85f;
        float windR = Mathf.Cos(globalTime * 7.0f) * 0.85f;
        float wobble = Mathf.Sin(globalTime * 4.5f) * 0.30f;

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(-0.18f + wobble, 0, wobble * 0.5f)),
            "Chest" => Quaternion.FromEuler(new Vector3(-0.08f, 0, wobble)),
            "Head" => Quaternion.FromEuler(new Vector3(0.22f, 0, -wobble * 0.6f)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(windL, 0.35f, -0.65f)),
            "Forearm_L" => Quaternion.FromEuler(new Vector3(0.75f + windL * 0.3f, 0, 0)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(windR, -0.35f, 0.65f)),
            "Forearm_R" => Quaternion.FromEuler(new Vector3(0.75f + windR * 0.3f, 0, 0)),
            "Thigh_L" => Quaternion.FromEuler(new Vector3(-0.20f + Mathf.Sin(globalTime * 5.0f) * 0.3f, 0, 0)),
            "Shin_L" => Quaternion.FromEuler(new Vector3(0.35f, 0, 0)),
            "Thigh_R" => Quaternion.FromEuler(new Vector3(-0.20f - Mathf.Sin(globalTime * 5.0f) * 0.3f, 0, 0)),
            "Shin_R" => Quaternion.FromEuler(new Vector3(0.35f, 0, 0)),
            _ => Quaternion.Identity
        };
    }
}

/// <summary>
/// Strategy for airborne falling / flailing panic reaction.
/// </summary>
public class FlailTrajectory : IMotionTrajectory
{
    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        float panic = Mathf.Sin(globalTime * 9.0f) * 0.45f;

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(-0.35f, 0, 0)),
            "Chest" => Quaternion.FromEuler(new Vector3(-0.30f, 0, 0)),
            "Head" => Quaternion.FromEuler(new Vector3(-0.45f, 0, 0)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(1.2f + panic, 0.4f, -0.4f)),
            "Forearm_L" => Quaternion.FromEuler(new Vector3(1.7f, 0.2f, 0.0f)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(1.2f - panic, -0.4f, 0.4f)),
            "Forearm_R" => Quaternion.FromEuler(new Vector3(1.7f, -0.2f, 0.0f)),
            "Thigh_L" => Quaternion.FromEuler(new Vector3(0.6f, 0, 0.12f)),
            "Shin_L" => Quaternion.FromEuler(new Vector3(0.85f, 0, 0)),
            "Thigh_R" => Quaternion.FromEuler(new Vector3(0.6f, 0, -0.12f)),
            "Shin_R" => Quaternion.FromEuler(new Vector3(0.85f, 0, 0)),
            _ => Quaternion.Identity
        };
    }
}

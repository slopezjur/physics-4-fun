using System;
using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Library of procedural joint poses and dynamic Euphoria-style reaction animations.
/// </summary>
public static class PoseLibrary
{
    public static Quaternion ComputeBoneTarget(string boneName, RagdollState state, float time, float stateTime)
    {
        return state switch
        {
            RagdollState.Balanced => GetBalancedPose(boneName, time),
            RagdollState.Stumbling => GetStumblePose(boneName, time),
            RagdollState.Flailing => GetFlailPose(boneName, time),
            RagdollState.Recovering => GetRecoveryPose(boneName, stateTime),
            RagdollState.KnockedOut => Quaternion.Identity,
            _ => Quaternion.Identity
        };
    }

    private static Quaternion GetBalancedPose(string boneName, float time)
    {
        float breathe = Mathf.Sin(time * 2.0f) * 0.02f;
        float sway = Mathf.Sin(time * 1.2f) * 0.012f;

        return boneName switch
        {
            "Pelvis" => Quaternion.Identity,
            "Spine" => Quaternion.FromEuler(new Vector3(-0.03f + breathe, sway, 0)),
            "Chest" => Quaternion.FromEuler(new Vector3(-0.04f + breathe * 0.5f, -sway * 0.5f, 0)),
            "Head" => Quaternion.FromEuler(new Vector3(0.05f - breathe * 0.3f, -sway, 0)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(0.05f + breathe, 0.0f, -0.12f)),
            "Forearm_L" => Quaternion.FromEuler(new Vector3(0.18f, 0.0f, 0.0f)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(0.05f + breathe, 0.0f, 0.12f)),
            "Forearm_R" => Quaternion.FromEuler(new Vector3(0.18f, 0.0f, 0.0f)),
            "Thigh_L" => Quaternion.FromEuler(new Vector3(-0.02f, 0.0f, 0.0f)),
            "Shin_L" => Quaternion.FromEuler(new Vector3(0.04f, 0.0f, 0.0f)),
            "Foot_L" => Quaternion.FromEuler(new Vector3(-0.02f, 0.0f, 0.0f)),
            "Thigh_R" => Quaternion.FromEuler(new Vector3(-0.02f, 0.0f, 0.0f)),
            "Shin_R" => Quaternion.FromEuler(new Vector3(0.04f, 0.0f, 0.0f)),
            "Foot_R" => Quaternion.FromEuler(new Vector3(-0.02f, 0.0f, 0.0f)),
            _ => Quaternion.Identity
        };
    }

    private static Quaternion GetStumblePose(string boneName, float time)
    {
        float flailL = Mathf.Sin(time * 7.0f) * 0.9f;
        float flailR = Mathf.Cos(time * 6.5f) * 0.9f;
        float wobble = Mathf.Sin(time * 4.5f) * 0.25f;

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(-0.15f + wobble, 0, wobble * 0.4f)),
            "Chest" => Quaternion.FromEuler(new Vector3(-0.05f, 0, wobble)),
            "Head" => Quaternion.FromEuler(new Vector3(0.2f, 0, -wobble * 0.5f)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(flailL, 0.3f, -0.6f)),
            "Forearm_L" => Quaternion.FromEuler(new Vector3(0.7f + flailL * 0.3f, 0, 0)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(flailR, -0.3f, 0.6f)),
            "Forearm_R" => Quaternion.FromEuler(new Vector3(0.7f + flailR * 0.3f, 0, 0)),
            "Thigh_L" => Quaternion.FromEuler(new Vector3(-0.15f + Mathf.Sin(time * 5.0f) * 0.25f, 0, 0)),
            "Shin_L" => Quaternion.FromEuler(new Vector3(0.3f, 0, 0)),
            "Thigh_R" => Quaternion.FromEuler(new Vector3(-0.15f - Mathf.Sin(time * 5.0f) * 0.25f, 0, 0)),
            "Shin_R" => Quaternion.FromEuler(new Vector3(0.3f, 0, 0)),
            _ => Quaternion.Identity
        };
    }

    private static Quaternion GetFlailPose(string boneName, float time)
    {
        float panic = Mathf.Sin(time * 9.0f) * 0.4f;

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(-0.30f, 0, 0)),
            "Chest" => Quaternion.FromEuler(new Vector3(-0.25f, 0, 0)),
            "Head" => Quaternion.FromEuler(new Vector3(-0.40f, 0, 0)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(1.1f + panic, 0.35f, -0.35f)),
            "Forearm_L" => Quaternion.FromEuler(new Vector3(1.6f, 0.2f, 0.0f)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(1.1f - panic, -0.35f, 0.35f)),
            "Forearm_R" => Quaternion.FromEuler(new Vector3(1.6f, -0.2f, 0.0f)),
            "Thigh_L" => Quaternion.FromEuler(new Vector3(0.5f, 0, 0.1f)),
            "Shin_L" => Quaternion.FromEuler(new Vector3(0.8f, 0, 0)),
            "Thigh_R" => Quaternion.FromEuler(new Vector3(0.5f, 0, -0.1f)),
            "Shin_R" => Quaternion.FromEuler(new Vector3(0.8f, 0, 0)),
            _ => Quaternion.Identity
        };
    }

    private static Quaternion GetRecoveryPose(string boneName, float stateTime)
    {
        // 0.0s - 1.0s: Tuck knees and plant arms
        if (stateTime < 1.0f)
        {
            float t = Mathf.Clamp(stateTime / 1.0f, 0.0f, 1.0f);
            return boneName switch
            {
                "Spine" => Quaternion.FromEuler(new Vector3(0.5f * t, 0, 0)),
                "Chest" => Quaternion.FromEuler(new Vector3(0.4f * t, 0, 0)),
                "Thigh_L" or "Thigh_R" => Quaternion.FromEuler(new Vector3(-1.2f * t, 0, 0)), // Knees to chest
                "Shin_L" or "Shin_R" => Quaternion.FromEuler(new Vector3(1.2f * t, 0, 0)),  // Folded knees
                "UpperArm_L" or "UpperArm_R" => Quaternion.FromEuler(new Vector3(0.8f * t, 0, 0)), // Reach down
                "Forearm_L" or "Forearm_R" => Quaternion.FromEuler(new Vector3(0.4f * t, 0, 0)),
                _ => Quaternion.Identity
            };
        }
        else // 1.0s - 2.5s: Push off the ground
        {
            float t = Mathf.Clamp((stateTime - 1.0f) / 1.5f, 0.0f, 1.0f);
            float spine = Mathf.Lerp(0.5f, 0.0f, t);
            float chest = Mathf.Lerp(0.4f, 0.0f, t);
            float thigh = Mathf.Lerp(-1.2f, 0.0f, t);
            float shin = Mathf.Lerp(1.2f, 0.0f, t);
            float arm = Mathf.Lerp(0.8f, 0.0f, t);
            float forearm = Mathf.Lerp(0.4f, 0.0f, t);

            return boneName switch
            {
                "Spine" => Quaternion.FromEuler(new Vector3(spine, 0, 0)),
                "Chest" => Quaternion.FromEuler(new Vector3(chest, 0, 0)),
                "Thigh_L" or "Thigh_R" => Quaternion.FromEuler(new Vector3(thigh, 0, 0)),
                "Shin_L" or "Shin_R" => Quaternion.FromEuler(new Vector3(shin, 0, 0)),
                "UpperArm_L" or "UpperArm_R" => Quaternion.FromEuler(new Vector3(arm, 0, 0)),
                "Forearm_L" or "Forearm_R" => Quaternion.FromEuler(new Vector3(forearm, 0, 0)),
                _ => Quaternion.Identity
            };
        }
    }
}

using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// 4-Phase Prone (lying on belly) physical recovery trajectory strategy:
/// Phase 1 (0.00-0.25): Hands under chest, knees slide forward under hips.
/// Phase 2 (0.25-0.55): Push-up into quadruped (all-fours) stance.
/// Phase 3 (0.55-0.80): Bear-crawl to deep squat, shifting CoM over feet.
/// Phase 4 (0.80-1.00): Quad drive to full vertical extension.
/// </summary>
public class ProneRecoveryTrajectory : IMotionTrajectory
{
    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        float t = Mathf.Clamp(phaseNormalized, 0.0f, 1.0f);
        float spinePitch, chestPitch, thighPitch, shinPitch, footPitch, armPitch, armRoll, forearmPitch;

        if (t < 0.25f)
        {
            // Phase 1: Hands planted, knees tucking under
            float p = t / 0.25f;
            spinePitch = Mathf.Lerp(0.0f, -0.20f, p);
            chestPitch = Mathf.Lerp(0.0f, -0.30f, p);
            thighPitch = Mathf.Lerp(0.0f, 0.75f, p);
            shinPitch = Mathf.Lerp(0.0f, -1.20f, p);
            footPitch = Mathf.Lerp(0.0f, -0.40f, p);
            armPitch = Mathf.Lerp(0.0f, 0.70f, p);
            armRoll = Mathf.Lerp(0.0f, 0.25f, p);
            forearmPitch = Mathf.Lerp(0.0f, 1.40f, p);
        }
        else if (t < 0.55f)
        {
            // Phase 2: Push up into Quadruped Stance
            float p = (t - 0.25f) / 0.30f;
            spinePitch = Mathf.Lerp(-0.20f, 0.10f, p);
            chestPitch = Mathf.Lerp(-0.30f, 0.05f, p);
            thighPitch = Mathf.Lerp(0.75f, 1.35f, p);
            shinPitch = Mathf.Lerp(-1.20f, -1.45f, p);
            footPitch = Mathf.Lerp(-0.40f, -0.20f, p);
            armPitch = Mathf.Lerp(0.70f, 0.45f, p);
            armRoll = Mathf.Lerp(0.25f, 0.15f, p);
            forearmPitch = Mathf.Lerp(1.40f, 0.25f, p);
        }
        else if (t < 0.80f)
        {
            // Phase 3: Feet plant, Bear Crawl to Squat
            float p = (t - 0.55f) / 0.25f;
            spinePitch = Mathf.Lerp(0.10f, 0.45f, p);
            chestPitch = Mathf.Lerp(0.05f, 0.30f, p);
            thighPitch = Mathf.Lerp(1.35f, 0.95f, p);
            shinPitch = Mathf.Lerp(-1.45f, -1.10f, p);
            footPitch = Mathf.Lerp(-0.20f, 0.0f, p);
            armPitch = Mathf.Lerp(0.45f, 0.80f, p);
            armRoll = Mathf.Lerp(0.15f, 0.10f, p);
            forearmPitch = Mathf.Lerp(0.25f, 0.20f, p);
        }
        else
        {
            // Phase 4: Full Leg Extension
            float p = (t - 0.80f) / 0.20f;
            spinePitch = Mathf.Lerp(0.45f, 0.0f, p);
            chestPitch = Mathf.Lerp(0.30f, 0.0f, p);
            thighPitch = Mathf.Lerp(0.95f, 0.0f, p);
            shinPitch = Mathf.Lerp(-1.10f, 0.0f, p);
            footPitch = Mathf.Lerp(0.0f, 0.0f, p);
            armPitch = Mathf.Lerp(0.80f, 0.05f, p);
            armRoll = Mathf.Lerp(0.10f, 0.0f, p);
            forearmPitch = Mathf.Lerp(0.20f, 0.20f, p);
        }

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(spinePitch, 0, 0)),
            "Chest" => Quaternion.FromEuler(new Vector3(chestPitch, 0, 0)),
            "Head" => Quaternion.FromEuler(new Vector3(-0.1f, 0, 0)),
            "Thigh_L" or "Thigh_R" => Quaternion.FromEuler(new Vector3(thighPitch, 0, 0)),
            "Shin_L" or "Shin_R" => Quaternion.FromEuler(new Vector3(shinPitch, 0, 0)),
            "Foot_L" or "Foot_R" => Quaternion.FromEuler(new Vector3(footPitch, 0, 0)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(armPitch, 0, -armRoll)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(armPitch, 0, armRoll)),
            "Forearm_L" or "Forearm_R" => Quaternion.FromEuler(new Vector3(forearmPitch, 0, 0)),
            _ => Quaternion.Identity
        };
    }
}

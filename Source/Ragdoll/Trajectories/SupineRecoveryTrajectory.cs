using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// 4-Phase Supine (lying on back) physical recovery trajectory strategy:
/// Phase 1 (0.00-0.30): Roll over to Prone (reach left arm across, flex & cross right leg, roll spine).
/// Phase 2 (0.30-0.60): Hands and knees press into floor, pushing up into Quadruped stance.
/// Phase 3 (0.60-0.85): Bear-crawl feet flat under pelvis, establishing ground contact.
/// Phase 4 (0.85-1.00): Quad drive from squat to full upright balance.
/// </summary>
public class SupineRecoveryTrajectory : IMotionTrajectory
{
    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        float t = Mathf.Clamp(phaseNormalized, 0.0f, 1.0f);
        float spinePitch, spineRoll, chestPitch, chestRoll;
        float thighPitchL, thighPitchR, shinPitch, footPitch;
        float armPitchL, armPitchR, armRollL, armRollR, forearmPitch;

        if (t < 0.30f)
        {
            // Phase 1: Roll over to Prone
            float p = t / 0.30f;
            spinePitch = Mathf.Lerp(0.0f, -0.25f, p);
            spineRoll = Mathf.Lerp(0.0f, 0.85f, p);
            chestPitch = Mathf.Lerp(0.0f, -0.30f, p);
            chestRoll = Mathf.Lerp(0.0f, 0.85f, p);

            thighPitchL = Mathf.Lerp(0.0f, 0.70f, p);
            thighPitchR = Mathf.Lerp(0.0f, 0.50f, p);
            shinPitch = Mathf.Lerp(0.0f, -1.20f, p);
            footPitch = Mathf.Lerp(0.0f, -0.30f, p);

            armPitchL = Mathf.Lerp(0.0f, 0.80f, p);
            armRollL = Mathf.Lerp(0.0f, -0.40f, p);
            armPitchR = Mathf.Lerp(0.0f, 0.40f, p);
            armRollR = Mathf.Lerp(0.0f, 0.20f, p);
            forearmPitch = Mathf.Lerp(0.0f, 1.20f, p);
        }
        else if (t < 0.60f)
        {
            // Phase 2: Push up into Quadruped (all-fours) Stance
            float p = (t - 0.30f) / 0.30f;
            spinePitch = Mathf.Lerp(-0.25f, 0.10f, p);
            spineRoll = Mathf.Lerp(0.85f, 0.0f, p);
            chestPitch = Mathf.Lerp(-0.30f, 0.05f, p);
            chestRoll = Mathf.Lerp(0.85f, 0.0f, p);

            thighPitchL = Mathf.Lerp(0.70f, 1.35f, p);
            thighPitchR = Mathf.Lerp(0.50f, 1.35f, p);
            shinPitch = Mathf.Lerp(-1.20f, -1.45f, p);
            footPitch = Mathf.Lerp(-0.30f, -0.20f, p);

            armPitchL = Mathf.Lerp(0.80f, 0.45f, p);
            armRollL = Mathf.Lerp(-0.40f, 0.15f, p);
            armPitchR = Mathf.Lerp(0.40f, 0.45f, p);
            armRollR = Mathf.Lerp(0.20f, 0.15f, p);
            forearmPitch = Mathf.Lerp(1.20f, 0.25f, p);
        }
        else if (t < 0.85f)
        {
            // Phase 3: Bear-Crawl to Squat Transition
            float p = (t - 0.60f) / 0.25f;
            spinePitch = Mathf.Lerp(0.10f, 0.45f, p);
            spineRoll = 0.0f;
            chestPitch = Mathf.Lerp(0.05f, 0.30f, p);
            chestRoll = 0.0f;

            thighPitchL = Mathf.Lerp(1.35f, 0.95f, p);
            thighPitchR = Mathf.Lerp(1.35f, 0.95f, p);
            shinPitch = Mathf.Lerp(-1.45f, -1.10f, p);
            footPitch = Mathf.Lerp(-0.20f, 0.0f, p);

            armPitchL = Mathf.Lerp(0.45f, 0.80f, p);
            armRollL = Mathf.Lerp(0.15f, 0.10f, p);
            armPitchR = Mathf.Lerp(0.45f, 0.80f, p);
            armRollR = Mathf.Lerp(0.15f, 0.10f, p);
            forearmPitch = Mathf.Lerp(0.25f, 0.20f, p);
        }
        else
        {
            // Phase 4: Full Upright Leg Extension
            float p = (t - 0.85f) / 0.15f;
            spinePitch = Mathf.Lerp(0.45f, 0.0f, p);
            spineRoll = 0.0f;
            chestPitch = Mathf.Lerp(0.30f, 0.0f, p);
            chestRoll = 0.0f;

            thighPitchL = Mathf.Lerp(0.95f, 0.0f, p);
            thighPitchR = Mathf.Lerp(0.95f, 0.0f, p);
            shinPitch = Mathf.Lerp(-1.10f, 0.0f, p);
            footPitch = 0.0f;

            armPitchL = Mathf.Lerp(0.80f, 0.06f, p);
            armRollL = Mathf.Lerp(0.10f, 0.0f, p);
            armPitchR = Mathf.Lerp(0.80f, 0.06f, p);
            armRollR = Mathf.Lerp(0.10f, 0.0f, p);
            forearmPitch = Mathf.Lerp(0.20f, 0.20f, p);
        }

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(spinePitch, 0, spineRoll)),
            "Chest" => Quaternion.FromEuler(new Vector3(chestPitch, 0, chestRoll)),
            "Head" => Quaternion.FromEuler(new Vector3(0.1f, 0, 0)),
            "Thigh_L" => Quaternion.FromEuler(new Vector3(thighPitchL, 0, 0)),
            "Thigh_R" => Quaternion.FromEuler(new Vector3(thighPitchR, 0, 0)),
            "Shin_L" or "Shin_R" => Quaternion.FromEuler(new Vector3(shinPitch, 0, 0)),
            "Foot_L" or "Foot_R" => Quaternion.FromEuler(new Vector3(footPitch, 0, 0)),
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(armPitchL, 0, -armRollL)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(armPitchR, 0, armRollR)),
            "Forearm_L" or "Forearm_R" => Quaternion.FromEuler(new Vector3(forearmPitch, 0, 0)),
            _ => Quaternion.Identity
        };
    }
}

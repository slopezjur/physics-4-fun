using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

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
            // Phase 1: Roll over to Prone. Arm L reaches across, Arm R tucks.
            float p = t / 0.30f;
            spinePitch = Mathf.Lerp(0.0f, -0.20f, p);
            spineRoll = Mathf.Lerp(0.0f, 1.50f, p); // Roll body strongly
            chestPitch = Mathf.Lerp(0.0f, -0.30f, p);
            chestRoll = Mathf.Lerp(0.0f, 1.50f, p);

            thighPitchL = Mathf.Lerp(0.0f, 1.20f, p);
            thighPitchR = Mathf.Lerp(0.0f, 0.50f, p);
            shinPitch = Mathf.Lerp(0.0f, -1.50f, p);
            footPitch = 0.0f;

            armPitchL = Mathf.Lerp(0.0f, 1.00f, p);
            armRollL = Mathf.Lerp(0.0f, -0.50f, p); // Reach across body
            
            // Arm R goes directly to the planted push-up position
            armPitchR = Mathf.Lerp(0.0f, -0.50f, p);
            armRollR = Mathf.Lerp(0.0f, 1.00f, p); 
            forearmPitch = Mathf.Lerp(0.0f, 2.00f, p);
        }
        else if (t < 0.60f)
        {
            // Phase 2: Explosive Push Up. Same as Prone Phase 2.
            float p = (t - 0.30f) / 0.30f;
            spinePitch = Mathf.Lerp(-0.20f, -0.10f, p);
            spineRoll = Mathf.Lerp(1.50f, 0.0f, p);
            chestPitch = Mathf.Lerp(-0.30f, -0.10f, p);
            chestRoll = Mathf.Lerp(1.50f, 0.0f, p);

            thighPitchL = Mathf.Lerp(1.20f, 1.40f, p);
            thighPitchR = Mathf.Lerp(0.50f, 1.40f, p);
            shinPitch = Mathf.Lerp(-1.50f, -1.50f, p);
            footPitch = Mathf.Lerp(0.0f, 0.50f, p);

            // Now both arms mirror the push-up extension
            armPitchL = Mathf.Lerp(1.00f, 1.00f, p); 
            armRollL = Mathf.Lerp(-0.50f, 0.50f, p);
            
            armPitchR = Mathf.Lerp(-0.50f, 1.00f, p);
            armRollR = Mathf.Lerp(1.00f, 0.50f, p);
            
            forearmPitch = Mathf.Lerp(2.00f, 0.20f, p);
        }
        else if (t < 0.85f)
        {
            // Phase 3: Rock back onto heels. Same as Prone Phase 3.
            float p = (t - 0.60f) / 0.25f;
            spinePitch = Mathf.Lerp(-0.10f, -0.50f, p);
            spineRoll = 0.0f;
            chestPitch = Mathf.Lerp(-0.10f, -0.50f, p);
            chestRoll = 0.0f;

            thighPitchL = Mathf.Lerp(1.40f, 1.00f, p);
            thighPitchR = Mathf.Lerp(1.40f, 1.00f, p);
            shinPitch = Mathf.Lerp(-1.50f, -1.20f, p);
            footPitch = Mathf.Lerp(0.50f, -0.20f, p);

            armPitchL = Mathf.Lerp(1.00f, 1.50f, p);
            armRollL = Mathf.Lerp(0.50f, 0.20f, p);
            armPitchR = Mathf.Lerp(1.00f, 1.50f, p);
            armRollR = Mathf.Lerp(0.50f, 0.20f, p);
            forearmPitch = Mathf.Lerp(0.20f, 0.10f, p);
        }
        else
        {
            // Phase 4: Stand up. Same as Prone Phase 4.
            float p = (t - 0.85f) / 0.15f;
            spinePitch = Mathf.Lerp(-0.50f, 0.0f, p);
            spineRoll = 0.0f;
            chestPitch = Mathf.Lerp(-0.50f, 0.0f, p);
            chestRoll = 0.0f;

            thighPitchL = Mathf.Lerp(1.00f, 0.0f, p);
            thighPitchR = Mathf.Lerp(1.00f, 0.0f, p);
            shinPitch = Mathf.Lerp(-1.20f, 0.0f, p);
            footPitch = Mathf.Lerp(-0.20f, 0.0f, p);

            armPitchL = Mathf.Lerp(1.50f, 0.0f, p);
            armRollL = Mathf.Lerp(0.20f, 0.0f, p);
            armPitchR = Mathf.Lerp(1.50f, 0.0f, p);
            armRollR = Mathf.Lerp(0.20f, 0.0f, p);
            forearmPitch = Mathf.Lerp(0.10f, 0.0f, p);
        }

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(spinePitch, 0, spineRoll)),
            "Chest" => Quaternion.FromEuler(new Vector3(chestPitch, 0, chestRoll)),
            "Head" => Quaternion.FromEuler(new Vector3(-0.2f, 0, 0)),
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

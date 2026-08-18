using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

public class ProneRecoveryTrajectory : IMotionTrajectory
{
    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        float t = Mathf.Clamp(phaseNormalized, 0.0f, 1.0f);
        float spinePitch, chestPitch, thighPitch, shinPitch, footPitch, armPitch, armRoll, forearmPitch;

        if (t < 0.25f)
        {
            // Phase 1: Biomechanically accurate hand plant.
            // Arms pulled back (-0.5 pitch), flared out (1.0 roll), elbows deeply bent (2.0 pitch).
            float p = t / 0.25f;
            spinePitch = Mathf.Lerp(0.0f, -0.20f, p); 
            chestPitch = Mathf.Lerp(0.0f, -0.30f, p); 
            thighPitch = Mathf.Lerp(0.0f, 1.20f, p); 
            shinPitch = Mathf.Lerp(0.0f, -1.50f, p); 
            footPitch = 0.0f;
            
            armPitch = Mathf.Lerp(0.0f, -0.50f, p);
            armRoll = Mathf.Lerp(0.0f, 1.00f, p); 
            forearmPitch = Mathf.Lerp(0.0f, 2.00f, p); 
        }
        else if (t < 0.55f)
        {
            // Phase 2: Explosive Push Up.
            // Shoulders rotate forward to push chest up (pitch goes to 1.0).
            // Elbows straighten to push (pitch goes to 0.2).
            float p = (t - 0.25f) / 0.30f;
            spinePitch = Mathf.Lerp(-0.20f, -0.10f, p);
            chestPitch = Mathf.Lerp(-0.30f, -0.10f, p);
            thighPitch = Mathf.Lerp(1.20f, 1.40f, p);
            shinPitch = Mathf.Lerp(-1.50f, -1.50f, p);
            footPitch = Mathf.Lerp(0.0f, 0.50f, p); // Plant toes
            
            armPitch = Mathf.Lerp(-0.50f, 1.00f, p); 
            armRoll = Mathf.Lerp(1.00f, 0.50f, p); 
            forearmPitch = Mathf.Lerp(2.00f, 0.20f, p); 
        }
        else if (t < 0.80f)
        {
            // Phase 3: Rock back onto heels.
            float p = (t - 0.55f) / 0.25f;
            spinePitch = Mathf.Lerp(-0.10f, -0.50f, p);
            chestPitch = Mathf.Lerp(-0.10f, -0.50f, p); 
            thighPitch = Mathf.Lerp(1.40f, 1.00f, p); 
            shinPitch = Mathf.Lerp(-1.50f, -1.20f, p); 
            footPitch = Mathf.Lerp(0.50f, -0.20f, p); // Feet flat
            
            armPitch = Mathf.Lerp(1.00f, 1.50f, p); 
            armRoll = Mathf.Lerp(0.50f, 0.20f, p);
            forearmPitch = Mathf.Lerp(0.20f, 0.10f, p);
        }
        else
        {
            // Phase 4: Stand up
            float p = (t - 0.80f) / 0.20f;
            spinePitch = Mathf.Lerp(-0.50f, 0.0f, p);
            chestPitch = Mathf.Lerp(-0.50f, 0.0f, p);
            thighPitch = Mathf.Lerp(1.00f, 0.0f, p);
            shinPitch = Mathf.Lerp(-1.20f, 0.0f, p);
            footPitch = Mathf.Lerp(-0.20f, 0.0f, p);
            
            armPitch = Mathf.Lerp(1.50f, 0.0f, p);
            armRoll = Mathf.Lerp(0.20f, 0.0f, p);
            forearmPitch = Mathf.Lerp(0.10f, 0.0f, p);
        }

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(spinePitch, 0, 0)),
            "Chest" => Quaternion.FromEuler(new Vector3(chestPitch, 0, 0)),
            "Head" => Quaternion.FromEuler(new Vector3(-0.2f, 0, 0)),
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

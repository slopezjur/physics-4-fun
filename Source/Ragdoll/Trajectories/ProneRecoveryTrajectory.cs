using System.Collections.Generic;
using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// Asymmetric prone get-up: plant hands -> push the torso up -> drive ONE knee under the chest and
/// plant that foot -> rise over the planted leg while the trail leg swings through -> stand.
///
/// The asymmetry is the whole point. An earlier version drove both legs with a single shared angle,
/// which cannot work: with both legs mirrored, neither is ever free to move under the body, so the
/// centre of mass never gets over a support point. Telemetry showed CoM-to-support stuck at
/// 0.70-0.80 m against a 0.28 m tolerance for the entire attempt.
///
/// Every keyframe here is inside the rig's real joint envelope (see ActiveRagdoll.tscn):
///   hip X [-0.50, +2.10]   knee X [-2.60, +0.10]   ankle X [-0.80, +0.60]
///   hip Z  L [-0.20, +0.80] / R [-0.80, +0.20]
///   shoulder X [-1.00, +3.00], shoulder Z  L [-0.50, +2.50] / R [-2.50, +0.50]
///   elbow X [-0.10, +2.60]   spine X [-0.40, +1.00]   chest X [-0.50, +0.50]
/// Commanding past a stop does not bend the joint further, it just pins the actuator against the
/// limit at full torque - which is exactly what the previous arm roll did.
/// </summary>
public class ProneRecoveryTrajectory : IPhasedRecoveryTrajectory
{
    /// <summary>PlantHands | PushUpTorso | DriveLeadKnee | HalfKneelRise | StandUp.</summary>
    public IReadOnlyList<float> PhaseBoundaries { get; } = new[] { 0.15f, 0.35f, 0.60f, 0.85f };

    /// <summary>
    /// Shoulder abduction magnitude. Applied as +Z on the left and -Z on the right: the shoulder Z
    /// limits are mirrored ([-0.50,+2.50] left, [-2.50,+0.50] right), so this sign convention is
    /// the one with 2.5 rad of travel. The opposite sign hits a 0.5 rad stop almost immediately.
    /// </summary>
    private const float ArmFlare = 1.00f;

    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        return EvaluateBoneTarget(boneName, new RecoveryPose(globalTime, phaseNormalized, BodySide.Right));
    }

    public Quaternion EvaluateBoneTarget(string boneName, in RecoveryPose pose)
    {
        float t = Mathf.Clamp(pose.PhaseNormalized, 0.0f, 1.0f);

        float spinePitch, chestPitch, headPitch;
        float armPitch, armFlare, forearmPitch;
        float leadHip, leadKnee, leadAnkle, leadAbduct;
        float trailHip, trailKnee, trailAnkle;

        if (t < 0.15f)
        {
            // PlantHands: set the push-up base. Elbows fold so the palms come under the shoulders,
            // upper arms flare outward. Legs stay long - nothing is asked of them yet.
            float p = t / 0.15f;
            spinePitch = Mathf.Lerp(0.0f, -0.20f, p);
            chestPitch = Mathf.Lerp(0.0f, -0.25f, p);
            headPitch = Mathf.Lerp(0.0f, -0.30f, p);

            armPitch = Mathf.Lerp(0.0f, -0.40f, p);
            armFlare = Mathf.Lerp(0.0f, ArmFlare, p);
            forearmPitch = Mathf.Lerp(0.0f, 2.00f, p);

            leadHip = 0.0f; leadKnee = 0.0f; leadAnkle = 0.0f; leadAbduct = 0.0f;
            trailHip = 0.0f; trailKnee = 0.0f; trailAnkle = 0.0f;
        }
        else if (t < 0.35f)
        {
            // PushUpTorso: elbows extend and the shoulders drive the chest clear of the floor.
            // Spine and chest extend to their real stops (-0.38 / -0.45), not past them.
            float p = (t - 0.15f) / 0.20f;
            spinePitch = Mathf.Lerp(-0.20f, -0.38f, p);
            chestPitch = Mathf.Lerp(-0.25f, -0.45f, p);
            headPitch = Mathf.Lerp(-0.30f, -0.50f, p);

            armPitch = Mathf.Lerp(-0.40f, 0.30f, p);
            armFlare = Mathf.Lerp(ArmFlare, 0.60f, p);
            forearmPitch = Mathf.Lerp(2.00f, 0.20f, p);

            leadHip = 0.0f; leadKnee = 0.0f; leadAnkle = 0.0f; leadAbduct = 0.0f;
            trailHip = 0.0f; trailKnee = Mathf.Lerp(0.0f, -0.30f, p); trailAnkle = Mathf.Lerp(0.0f, 0.40f, p);
        }
        else if (t < 0.60f)
        {
            // DriveLeadKnee: the lead hip and knee fold hard so the knee travels under the chest and
            // the sole can find the floor. Abduction swings the knee slightly wide of the torso
            // instead of into it. The trail leg stays long and its toes stay planted, giving the
            // drive something to push against.
            float p = (t - 0.35f) / 0.25f;
            spinePitch = Mathf.Lerp(-0.38f, -0.30f, p);
            chestPitch = Mathf.Lerp(-0.45f, -0.35f, p);
            headPitch = Mathf.Lerp(-0.50f, -0.30f, p);

            armPitch = Mathf.Lerp(0.30f, 0.60f, p);
            armFlare = Mathf.Lerp(0.60f, 0.40f, p);
            forearmPitch = Mathf.Lerp(0.20f, 0.30f, p);

            leadHip = Mathf.Lerp(0.0f, 1.90f, p);
            leadKnee = Mathf.Lerp(0.0f, -2.20f, p);
            leadAnkle = Mathf.Lerp(0.0f, 0.40f, p);
            leadAbduct = Mathf.Lerp(0.0f, 0.45f, p);

            trailHip = Mathf.Lerp(0.0f, -0.20f, p);
            trailKnee = Mathf.Lerp(-0.30f, -0.45f, p);
            trailAnkle = 0.40f;
        }
        else if (t < 0.85f)
        {
            // HalfKneelRise: the classic half-kneel. The torso comes upright while the lead leg
            // extends to push the pelvis up over the planted foot, and the trail knee folds so the
            // trail leg can swing through. Arms come off the floor and reach forward as a
            // counterweight, which is what keeps the rise from toppling backwards.
            float p = (t - 0.60f) / 0.25f;
            spinePitch = Mathf.Lerp(-0.30f, 0.10f, p);
            chestPitch = Mathf.Lerp(-0.35f, 0.05f, p);
            headPitch = Mathf.Lerp(-0.30f, 0.0f, p);

            armPitch = Mathf.Lerp(0.60f, 1.20f, p);
            armFlare = Mathf.Lerp(0.40f, 0.15f, p);
            forearmPitch = Mathf.Lerp(0.30f, 0.50f, p);

            leadHip = Mathf.Lerp(1.90f, 0.80f, p);
            leadKnee = Mathf.Lerp(-2.20f, -0.90f, p);
            leadAnkle = Mathf.Lerp(0.40f, 0.15f, p);
            leadAbduct = Mathf.Lerp(0.45f, 0.10f, p);

            trailHip = Mathf.Lerp(-0.20f, 1.10f, p);
            trailKnee = Mathf.Lerp(-0.45f, -1.40f, p);
            trailAnkle = Mathf.Lerp(0.40f, 0.20f, p);
        }
        else
        {
            // StandUp: both legs converge on neutral and the arms drop to rest.
            float p = (t - 0.85f) / 0.15f;
            spinePitch = Mathf.Lerp(0.10f, 0.0f, p);
            chestPitch = Mathf.Lerp(0.05f, 0.0f, p);
            headPitch = Mathf.Lerp(0.0f, 0.0f, p);

            armPitch = Mathf.Lerp(1.20f, 0.05f, p);
            armFlare = Mathf.Lerp(0.15f, 0.10f, p);
            forearmPitch = Mathf.Lerp(0.50f, 0.15f, p);

            leadHip = Mathf.Lerp(0.80f, 0.0f, p);
            leadKnee = Mathf.Lerp(-0.90f, 0.0f, p);
            leadAnkle = Mathf.Lerp(0.15f, 0.0f, p);
            leadAbduct = Mathf.Lerp(0.10f, 0.0f, p);

            trailHip = Mathf.Lerp(1.10f, 0.0f, p);
            trailKnee = Mathf.Lerp(-1.40f, 0.0f, p);
            trailAnkle = Mathf.Lerp(0.20f, 0.0f, p);
        }

        bool leadIsLeft = pose.LeadSide == BodySide.Left;

        // Hip abduction Z is mirrored between sides: positive swings the left leg outward, negative
        // the right. Same magnitude, opposite sign, so the lead knee always clears the torso.
        float leftHip = leadIsLeft ? leadHip : trailHip;
        float leftKnee = leadIsLeft ? leadKnee : trailKnee;
        float leftAnkle = leadIsLeft ? leadAnkle : trailAnkle;
        float leftAbduct = leadIsLeft ? leadAbduct : 0.0f;

        float rightHip = leadIsLeft ? trailHip : leadHip;
        float rightKnee = leadIsLeft ? trailKnee : leadKnee;
        float rightAnkle = leadIsLeft ? trailAnkle : leadAnkle;
        float rightAbduct = leadIsLeft ? 0.0f : -leadAbduct;

        return boneName switch
        {
            "Spine" => Quaternion.FromEuler(new Vector3(spinePitch, 0, 0)),
            "Chest" => Quaternion.FromEuler(new Vector3(chestPitch, 0, 0)),
            "Head" => Quaternion.FromEuler(new Vector3(headPitch, 0, 0)),

            "Thigh_L" => Quaternion.FromEuler(new Vector3(leftHip, 0, leftAbduct)),
            "Shin_L" => Quaternion.FromEuler(new Vector3(leftKnee, 0, 0)),
            "Foot_L" => Quaternion.FromEuler(new Vector3(leftAnkle, 0, 0)),

            "Thigh_R" => Quaternion.FromEuler(new Vector3(rightHip, 0, rightAbduct)),
            "Shin_R" => Quaternion.FromEuler(new Vector3(rightKnee, 0, 0)),
            "Foot_R" => Quaternion.FromEuler(new Vector3(rightAnkle, 0, 0)),

            // Shoulder abduction: +Z left, -Z right. See ArmFlare - this is the sign with travel.
            "UpperArm_L" => Quaternion.FromEuler(new Vector3(armPitch, 0, armFlare)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(armPitch, 0, -armFlare)),
            "Forearm_L" or "Forearm_R" => Quaternion.FromEuler(new Vector3(forearmPitch, 0, 0)),

            _ => Quaternion.Identity
        };
    }
}

using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// Isolated knee push-up ("semi push-up"), the beginner variant done with the knees planted.
///
/// This is a development drill, not part of the get-up. It exists because the real push-up phase
/// was passing its chest-height criterion for the wrong reason: the arms could only manage ~14
/// N.m/rad of effective stiffness with 89 degrees of tracking error and 0.10 torque coherence, so
/// what actually lifted the chest was the spine arching into a cobra. Height alone cannot tell an
/// arm press apart from a back arch.
///
/// Two design choices make this drill self-verifying:
///
///   1. The torso is commanded RIGID AT NEUTRAL (spine and chest both zero) for the whole cycle.
///      With the back unable to arch, the only thing that can raise the chest is the arms.
///   2. The knees stay flexed so the shins lift clear of the floor and the body pivots about the
///      knees rather than the toes, roughly halving the moment the arms must overcome.
///
/// Phase 0 is the bottom of the rep (elbows folded, chest low), 1 is the top (elbows extended).
/// Every value below sits inside the rig's joint envelope - see ProneRecoveryTrajectory for the
/// full table. Commanding past a stop pins the actuator at full torque and produces no motion.
/// </summary>
public class PushUpDrillTrajectory : IMotionTrajectory
{
    /// <summary>Elbow flexion at the bottom of the rep (rad). Stop is +2.60.</summary>
    private const float ElbowBottom = 2.00f;

    /// <summary>Elbow flexion at lockout (rad). Slightly bent rather than hyperextended.</summary>
    private const float ElbowTop = 0.15f;

    /// <summary>Shoulder abduction, held through the rep. Applied +Z left / -Z right: the shoulder
    /// Z limits are mirrored, and this is the sign with 2.5 rad of travel rather than 0.5.</summary>
    private const float ShoulderFlare = 0.55f;

    /// <summary>Knee flexion holding the shins up off the floor, making this a KNEE push-up.</summary>
    private const float KneeFlex = -1.60f;

    /// <summary>Slight hip extension so the thigh stays down and the trunk holds a plank line.</summary>
    private const float HipExtend = -0.15f;

    /// <summary>Toes pointed so the feet stay clear of the ground and cannot take load.</summary>
    private const float AnklePoint = -0.40f;

    public Quaternion EvaluateBoneTarget(string boneName, float globalTime, float phaseNormalized)
    {
        float t = Mathf.Clamp(phaseNormalized, 0.0f, 1.0f);

        float elbow = Mathf.Lerp(ElbowBottom, ElbowTop, t);
        float shoulderPitch = Mathf.Lerp(-0.35f, 0.25f, t);

        return boneName switch
        {
            // Held at neutral on purpose: a plank, not a cobra. See the class summary.
            "Spine" or "Chest" => Quaternion.Identity,
            "Head" => Quaternion.FromEuler(new Vector3(-0.20f, 0, 0)),

            "UpperArm_L" => Quaternion.FromEuler(new Vector3(shoulderPitch, 0, ShoulderFlare)),
            "UpperArm_R" => Quaternion.FromEuler(new Vector3(shoulderPitch, 0, -ShoulderFlare)),
            "Forearm_L" or "Forearm_R" => Quaternion.FromEuler(new Vector3(elbow, 0, 0)),

            "Thigh_L" or "Thigh_R" => Quaternion.FromEuler(new Vector3(HipExtend, 0, 0)),
            "Shin_L" or "Shin_R" => Quaternion.FromEuler(new Vector3(KneeFlex, 0, 0)),
            "Foot_L" or "Foot_R" => Quaternion.FromEuler(new Vector3(AnklePoint, 0, 0)),

            _ => Quaternion.Identity
        };
    }
}

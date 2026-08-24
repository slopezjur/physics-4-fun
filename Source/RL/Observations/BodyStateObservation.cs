using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Observations;

/// <summary>
/// Whole-body proprioceptive observation, shared by EVERY task, replacing the minimal
/// BasicPoseObservation used to prove the pipeline in M2.
///
/// Task-neutral on purpose, and load-bearing because of it: the width is published to Python at
/// handshake and baked into every checkpoint, so a policy trained on one task can only be restored
/// onto another while this stays identical. That is the sole reason walking can be bootstrapped
/// from a standing policy rather than trained from noise.
///
/// It carries no task indicator and no goal, which is the current limit of the "one brain" idea:
/// stand, get-up and perturbation share an objective so they need none, but a task that wants the
/// body to go somewhere would need a goal vector added here - and adding one invalidates every
/// existing checkpoint.
///
/// Three deliberate changes from the M2 placeholder:
///
/// 1. **Quaternions, not Euler angles.** Euler wraps discontinuously, so two nearly identical poses
///    can produce wildly different vectors near a boundary - noise the network must waste capacity
///    undoing. Sign is canonicalised (W >= 0) because q and -q are the same rotation but different
///    input vectors.
/// 2. **Root-relative, not world.** Bone rotations are expressed relative to the pelvis and
///    positions relative to the pelvis, so the policy sees body configuration rather than where the
///    dummy happens to be in the arena. Absolute world position is not predictive of anything here.
/// 3. **Contact flags included.** Whether the hands and feet actually bear load is the single most
///    decisive fact when pushing off the floor, and the M2 set omitted it entirely - the policy
///    literally could not tell whether its foot was planted.
/// </summary>
public sealed class BodyStateObservation : IRlObservationBuilder
{
    /// <summary>Bones whose contact state is reported, in order.</summary>
    private static readonly string[] ContactBoneNames =
    {
        "Hand_L", "Hand_R", "Foot_L", "Foot_R"
    };

    /// <summary>
    /// Root block, 18 floats: pelvis height (1), tilt (1), up-vector (3), linear velocity (3),
    /// angular velocity (3), head height (1), CoM offset from pelvis (3), CoM velocity (3).
    /// Must match exactly what Build() emits - Size is published to Python at handshake, and the
    /// not-ready path returns a zero vector of this width, so a mismatch desyncs the transport.
    /// </summary>
    /// <summary>
    /// Root block, 18 floats: pelvis height (1), tilt (1), up-vector (3), linear velocity (3),
    /// angular velocity (3), head height (1), CoM offset from pelvis (3), CoM velocity (3).
    /// </summary>
    private const int RootComponents = 18;

    /// <summary>
    /// Universal Joystick, 7 floats: Target Velocity (X, Z), Target Jump, Target Turn,
    /// Target Posture (Stand/Crouch/Prone), and Terrain Slope (X, Z).
    /// </summary>
    private const int JoystickComponents = 7;

    /// <summary>Per controlled bone: root-relative quaternion (4) + angular velocity (3).</summary>
    private const int ComponentsPerBone = 7;

    private readonly int _boneCount;

    public BodyStateObservation(int boneCount) => _boneCount = boneCount;

    public int Size => RootComponents + JoystickComponents + (_boneCount * ComponentsPerBone) + ContactBoneNames.Length;

    public string Describe() =>
        $"BodyStateObservation(root={RootComponents}, perBone={ComponentsPerBone} [quaternion+angvel], "
        + $"bones={_boneCount}, contacts={ContactBoneNames.Length}, total={Size})";

    public float[] Build(in RlContext context)
    {
        var obs = new List<float>(Size);

        ActiveBone pelvis = context.Pelvis;
        Quaternion pelvisRot = pelvis.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
        Quaternion pelvisInv = pelvisRot.Inverse();
        Vector3 pelvisPos = pelvis.GlobalPosition;

        obs.Add(pelvisPos.Y);
        obs.Add(context.Balance.CurrentTiltAngleDeg / 180.0f);

        // The pelvis up-axis in world space: an unambiguous, wrap-free encoding of which way the
        // body is facing, which a single tilt scalar cannot express (it loses the direction of lean).
        Vector3 up = pelvis.GlobalTransform.Basis.Y;
        obs.Add(up.X);
        obs.Add(up.Y);
        obs.Add(up.Z);

        Vector3 vel = pelvis.LinearVelocity;
        obs.Add(vel.X);
        obs.Add(vel.Y);
        obs.Add(vel.Z);

        Vector3 angVel = pelvis.AngularVelocity;
        obs.Add(angVel.X);
        obs.Add(angVel.Y);
        obs.Add(angVel.Z);

        ActiveBone? head = context.Ragdoll.Head;
        obs.Add(head != null && GodotObject.IsInstanceValid(head) ? head.GlobalPosition.Y : 0.0f);

        Vector3 comOffset = pelvisInv * (context.Balance.CenterOfMass - pelvisPos);
        obs.Add(comOffset.X);
        obs.Add(comOffset.Y);
        obs.Add(comOffset.Z);

        Vector3 comVel = context.Balance.CenterOfMassVelocity;
        obs.Add(comVel.X);
        obs.Add(comVel.Y);
        obs.Add(comVel.Z);

        // Universal Joystick Placeholder (Zeroes for now, preparing the observation space)
        obs.Add(0.0f); // Target Velocity X
        obs.Add(0.0f); // Target Velocity Z
        obs.Add(0.0f); // Target Jump
        obs.Add(0.0f); // Target Turn
        obs.Add(1.0f); // Target Posture (1.0 = Stand)
        obs.Add(0.0f); // Terrain Slope X
        obs.Add(0.0f); // Terrain Slope Z

        foreach (ActiveBone? bone in context.ControlledBones)
        {
            if (bone == null || !GodotObject.IsInstanceValid(bone))
            {
                obs.AddRange(new float[ComponentsPerBone]);
                continue;
            }

            Quaternion boneRot = bone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
            Quaternion relative = (pelvisInv * boneRot).Normalized();
            if (relative.W < 0.0f)
            {
                // q and -q represent the same rotation; canonicalising stops the network seeing an
                // arbitrary sign flip as a large change in input.
                relative = new Quaternion(-relative.X, -relative.Y, -relative.Z, -relative.W);
            }

            obs.Add(relative.X);
            obs.Add(relative.Y);
            obs.Add(relative.Z);
            obs.Add(relative.W);

            Vector3 boneAngVel = bone.AngularVelocity;
            obs.Add(boneAngVel.X);
            obs.Add(boneAngVel.Y);
            obs.Add(boneAngVel.Z);
        }

        AppendContactFlags(context, obs);
        return obs.ToArray();
    }

    private static void AppendContactFlags(in RlContext context, List<float> obs)
    {
        foreach (string name in ContactBoneNames)
        {
            ActiveBone? found = null;
            foreach (ActiveBone candidate in context.Ragdoll.GetBones())
            {
                if (candidate.BoneName == name)
                {
                    found = candidate;
                    break;
                }
            }

            bool contacting = found != null
                              && GodotObject.IsInstanceValid(found)
                              && found.IsInContactWithWorld();
            obs.Add(contacting ? 1.0f : 0.0f);
        }
    }
}

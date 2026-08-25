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
/// It carries no task indicator, which is the current limit of the "one brain" idea: stand, get-up
/// and perturbation share an objective so they need none.
///
/// It DOES carry a goal block, reserved and constant - see <see cref="JoystickComponents"/>. That
/// block exists ahead of any task using it precisely because adding one later would invalidate
/// every existing checkpoint, and Walk (the task that will need it) is bootstrapped from a Stand
/// checkpoint that has to match width. Paying that cost once, up front, is the only way the
/// bootstrap survives the day walking gains a heading command.
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
    private const int RootComponents = 18;

    /// <summary>
    /// Goal block ("Universal Joystick"), 7 floats. Constant today - see the slot map in Build().
    ///
    /// Present in EVERY task's observation, including the three Upright ones that need no goal at
    /// all, and that is the point. Walk is a separate brain but is bootstrapped from a Stand
    /// checkpoint, which only works while the widths match (see the class summary). Walk is also
    /// the task that will need a command: WalkForwardReward.HeadingWeight is pinned to 0.0 purely
    /// because the observation carries no heading. A command has to reach the input layer - there
    /// is no way to express "go that way" to a trained network except through this vector - so the
    /// slots have to exist in the shared observation BEFORE walking uses them, or the bootstrap
    /// breaks on the day it does.
    ///
    /// The width is therefore load-bearing and frozen. It has already churned 106 -> 108 -> 115 ->
    /// 113 across the archived lineages, and commits 31573c2 -> 94402ad -> ebbaa1b were three
    /// commits spent on an ONNX shape crash caused by exactly that. Every change orphans every
    /// checkpoint. Do not change it without a decision record.
    ///
    /// KNOWN DEBT, deliberately not acted on: terrain slope has no slot here. It was in this block
    /// and was removed - it is exteroception, not a command, so it belongs in the root sensing
    /// block, and no scene can currently produce a non-zero value (every RL floor is a flat
    /// axis-aligned BoxShape3D and there is no slope code anywhere in the project). Adding it later
    /// is its own deliberate width decision, not a free slot to reclaim.
    /// </summary>
    private const int JoystickComponents = 7;

    /// <summary>Per controlled bone: root-relative quaternion (4) + angular velocity (3).</summary>
    private const int ComponentsPerBone = 7;

    private readonly int _boneCount;

    public BodyStateObservation(int boneCount) => _boneCount = boneCount;

    public int Size => RootComponents + JoystickComponents + (_boneCount * ComponentsPerBone) + ContactBoneNames.Length;

    /// <summary>
    /// Written verbatim into the run manifest, so the components listed here MUST account for
    /// every float in Size. They did not: the goal block was added without touching this string,
    /// and every manifest written since reported root=18 + 12x7 + 4 = 106 next to total=113, with
    /// seven floats unaccounted for. That is precisely the drift ARCHITECTURE.md claims cannot
    /// happen. Adding a component and not adding it here reintroduces it.
    /// </summary>
    public string Describe() =>
        $"BodyStateObservation(root={RootComponents}, goal={JoystickComponents} [targetVel+jump+turn+posture1hot], "
        + $"perBone={ComponentsPerBone} [quaternion+angvel], "
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

        // Goal block, JoystickComponents floats. Constant until a task actually issues a command;
        // see the JoystickComponents summary for why it is here before anything drives it.
        //
        // Posture is a THREE-SLOT ONE-HOT, not one scalar. It was a single ordinal
        // (0=Prone/0.5=Crouch/1=Stand) sharing the block with two terrain-slope slots, which
        // asserted Crouch lies numerically between Prone and Stand - the network would have been
        // handed that ordering as fact. Splitting it costs nothing because the two slope slots it
        // replaces were dead: exteroception in a command block, unreachable on flat floors.
        //
        // The emitted vector is UNCHANGED by that split - [0,0,0,0,1,0,0] before and after, with
        // the 1.0 still at index 4 - so every checkpoint trained on the old layout stays valid.
        // Slots 5 and 6 have emitted exactly 0.0 for every step ever trained, so their first-layer
        // weights received exactly zero gradient and are still at initialization. There is nothing
        // in the policy that learned the old meaning and would have to unlearn it.
        obs.Add(0.0f); // 0: Target Velocity X
        obs.Add(0.0f); // 1: Target Velocity Z
        obs.Add(0.0f); // 2: Target Jump
        obs.Add(0.0f); // 3: Target Turn
        obs.Add(1.0f); // 4: Target Posture - Stand
        obs.Add(0.0f); // 5: Target Posture - Crouch
        obs.Add(0.0f); // 6: Target Posture - Prone

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

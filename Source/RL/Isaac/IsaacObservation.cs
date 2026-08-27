using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// The 143-float observation an Isaac Lab policy was trained on. See `isaac_lab/obs_action_contract.md` §3.
///
/// This is NOT <see cref="Observations.BodyStateObservation"/> resized. That layout carries
/// root-relative bone quaternions; this one carries joint angles, which is the state a
/// reduced-coordinate PhysX articulation actually has. Different physical quantities, so the two
/// cannot share an implementation and a policy trained on one is meaningless against the other.
///
/// <para><b>Two conversions are load-bearing, and both are silent when wrong</b> - a mismatched
/// observation produces a dummy that flails, never an error:</para>
///
/// <list type="number">
/// <item><description><b>Frame remap.</b> Isaac's pelvis frame is X-forward / Y-left / Z-up; Godot's
/// is X-left / Y-up / Z-back. Same body, different bases, so every pelvis-frame vector needs its
/// components permuted by <see cref="ToIsaacFrame"/>. The rest-pose check: Godot gravity
/// <c>(0,-1,0)</c> remaps to <c>(0,0,-1)</c>, the upright value the contract specifies.</description></item>
/// <item><description><b>Roll sign.</b> Isaac's <c>_rz</c> joints rotate about URDF +X, which is
/// Godot -Z, so <c>isaac = -godot</c> on every roll axis. Handled by
/// <see cref="IsaacRigContract.ToIsaac"/>.</description></item>
/// </list>
///
/// <para>Joint ordering for the two 45-wide slices comes from <c>physx_dof_order</c> - what PhysX
/// assigned at USD import, which the environments index <c>data.joint_pos</c> with directly. It is
/// not the URDF declaration order, and it is read from the rig contract rather than restated.</para>
/// </summary>
public sealed class IsaacObservation : IRlObservationBuilder
{
    /// <summary>Slice widths, in emission order. Must sum to <see cref="Size"/>.</summary>
    private const int GravityComponents = 3;
    private const int LinearVelocityComponents = 3;
    private const int AngularVelocityComponents = 3;
    private const int PelvisHeightComponents = 1;
    private const int CommandComponents = 3;

    /// <summary>Contact flags, in the contract's order: Hand_L, Hand_R, Foot_L, Foot_R.</summary>
    private static readonly string[] ContactBoneNames = { "Hand_L", "Hand_R", "Foot_L", "Foot_R" };

    private readonly IsaacRigContract _rig;

    /// <summary>Bone per DOF slot, resolved once. Null entries emit zeros rather than throwing mid-step.</summary>
    private readonly ActiveBone?[] _dofBones;

    /// <summary>Previous action as SENT - pre-scaling, already clamped to [-1,1]. Contract slice [104:140].</summary>
    private readonly float[] _previousAction;

    /// <summary>Velocity command (vx, vy, yaw_rate). Zero for Stand; Walk and Run drive it.</summary>
    private Vector3 _command;

    public IsaacObservation(IsaacRigContract rig, HumanoidRagdoll ragdoll)
    {
        _rig = rig;
        _previousAction = new float[rig.ActuatedJoints.Count];
        _dofBones = new ActiveBone?[rig.DofOrder.Count];

        var byName = new Dictionary<string, ActiveBone>();
        foreach (ActiveBone bone in ragdoll.GetBones())
        {
            byName[bone.BoneName] = bone;
        }

        for (int i = 0; i < _dofBones.Length; i++)
        {
            string boneName = rig.DofOrder[i].Bone;
            if (byName.TryGetValue(boneName, out ActiveBone? bone))
            {
                _dofBones[i] = bone;
            }
            else
            {
                GD.PushError($"[IsaacObservation] Rig contract names bone '{boneName}', absent from the ragdoll.");
            }
        }
    }

    public int Size =>
        GravityComponents + LinearVelocityComponents + AngularVelocityComponents + PelvisHeightComponents
        + _rig.DofOrder.Count * 2
        + ContactBoneNames.Length
        + _rig.ActuatedJoints.Count
        + CommandComponents;

    public string Describe() =>
        $"IsaacObservation(gravity=3, linVel=3, angVel=3, pelvisHeight=1, "
        + $"jointPos={_rig.DofOrder.Count}, jointVel={_rig.DofOrder.Count}, "
        + $"contacts={ContactBoneNames.Length}, prevAction={_rig.ActuatedJoints.Count}, "
        + $"command={CommandComponents}, total={Size})";

    /// <summary>
    /// Records what was sent to the actuators this step, for slice [104:140] of the NEXT
    /// observation. The contract is explicit that this is the action as sent - clamped, but before
    /// <c>ACTION_SCALE</c> and before the joint-range mapping.
    /// </summary>
    public void RecordAction(float[] clampedAction)
    {
        int count = Mathf.Min(clampedAction.Length, _previousAction.Length);
        System.Array.Copy(clampedAction, _previousAction, count);
    }

    /// <summary>Sets the velocity command for Walk and Run. Stand leaves it at zero.</summary>
    public void SetCommand(Vector3 command) => _command = command;

    /// <summary>Clears the previous-action history. Call when the body is teleported or reset.</summary>
    public void Reset()
    {
        System.Array.Clear(_previousAction);
        _command = Vector3.Zero;
    }

    /// <summary>
    /// Godot pelvis-frame components -> Isaac pelvis-frame components. Must be the same map the
    /// rig converter applies to positions, or the two engines disagree about what each slot means.
    ///
    /// <para>This matches `isaac_lab/scripts/build_d6_usd.py:to_usd`, which is a proper rotation
    /// (determinant +1). The older URDF converter used <c>(-z, +x, y)</c> instead - determinant -1,
    /// a REFLECTION - which made the Isaac dummy a mirror image of the Godot one and inverted every
    /// rotation about a mapped axis. The `reversed` flags and negated roll limits in the old rig
    /// contract were compensating for that mirror; the D6 rig needs none of it.</para>
    /// </summary>
    public static Vector3 ToIsaacFrame(in Vector3 godot) => new(-godot.Z, -godot.X, godot.Y);

    public float[] Build(in RlContext context)
    {
        var obs = new List<float>(Size);

        ActiveBone pelvis = context.Pelvis;
        Basis pelvisBasis = pelvis.GlobalTransform.Basis;
        Basis pelvisInv = pelvisBasis.Inverse();

        // Isaac's `projected_gravity_b` is the world gravity DIRECTION in the body frame, unit
        // length. Godot's gravity runs along -Y, so this is (0,-1,0) at rest and remaps to the
        // (0,0,-1) the contract specifies for upright.
        Append(obs, ToIsaacFrame(pelvisInv * Vector3.Down));

        // Linear velocity is heading-relative (yaw only) on the Isaac side: `quat_rotate_inverse(
        // yaw_quat(root_quat_w), root_lin_vel_w)`. Using the full pelvis basis instead would fold
        // pitch and roll into the velocity the policy reads, which is a different quantity.
        Append(obs, ToIsaacFrame(YawOnly(pelvisBasis).Inverse() * pelvis.LinearVelocity));

        // Angular velocity IS full-basis on the Isaac side (`root_ang_vel_b`), unlike linear.
        Append(obs, ToIsaacFrame(pelvisInv * pelvis.AngularVelocity));

        obs.Add(pelvis.GlobalPosition.Y);

        AppendJointPositions(obs);
        AppendJointVelocities(obs);
        AppendContactFlags(context, obs);

        foreach (float value in _previousAction)
        {
            obs.Add(value);
        }

        obs.Add(_command.X);
        obs.Add(_command.Y);
        obs.Add(_command.Z);

        return obs.ToArray();
    }

    /// <summary>
    /// Joint angle minus rest, per DOF. Isaac's default joint position is 0 for every joint - the
    /// Godot rest pose - so the deviation from rest IS `joint_pos - default`.
    /// </summary>
    private void AppendJointPositions(List<float> obs)
    {
        for (int i = 0; i < _dofBones.Length; i++)
        {
            ActiveBone? bone = _dofBones[i];
            if (bone == null || !GodotObject.IsInstanceValid(bone))
            {
                obs.Add(0.0f);
                continue;
            }

            Vector3 euler = DeviationFromRest(bone);
            IsaacRigContract.JointSpec spec = _rig.DofOrder[i];
            obs.Add(IsaacRigContract.ToIsaac(spec, Component(euler, spec.GodotAxis)));
        }
    }

    /// <summary>
    /// Joint velocity per DOF: the child's angular velocity relative to its parent, resolved in the
    /// parent's frame and projected onto the joint's own axis.
    ///
    /// This is the relative rate about each axis rather than the exact rate of the corresponding
    /// stacked revolute in the URDF chain. The two differ once the joint is far from rest, because
    /// the middle and distal joints of the chain rotate in frames the proximal one has already
    /// turned. Close enough at the small deflections the policy operates at, and it avoids
    /// differentiating the angles, which would inject a step of quantisation noise into 45 slots.
    /// </summary>
    private void AppendJointVelocities(List<float> obs)
    {
        for (int i = 0; i < _dofBones.Length; i++)
        {
            ActiveBone? bone = _dofBones[i];
            if (bone == null || !GodotObject.IsInstanceValid(bone))
            {
                obs.Add(0.0f);
                continue;
            }

            ActiveBone? parent = bone.ParentBone;
            Vector3 relative = bone.AngularVelocity - (parent != null && GodotObject.IsInstanceValid(parent)
                ? parent.AngularVelocity
                : Vector3.Zero);

            Basis frame = parent != null && GodotObject.IsInstanceValid(parent)
                ? parent.GlobalTransform.Basis
                : bone.GlobalTransform.Basis;

            Vector3 local = frame.Inverse() * relative;
            IsaacRigContract.JointSpec spec = _rig.DofOrder[i];
            float rate = IsaacRigContract.ToIsaac(spec, Component(local, spec.GodotAxis));

            if (JointVelocityFilter < 1.0f)
            {
                _filteredJointVelocity ??= new float[_dofBones.Length];
                float alpha = Mathf.Clamp(JointVelocityFilter, 0.001f, 1.0f);
                _filteredJointVelocity[i] += (rate - _filteredJointVelocity[i]) * alpha;
                rate = _filteredJointVelocity[i];
            }

            obs.Add(JointVelocityClip > 0.0f ? Mathf.Clamp(rate, -JointVelocityClip, JointVelocityClip) : rate);
        }
    }

    /// <summary>
    /// Bound on observation slice [55:100], rad/s. **Must match `obs_joint_vel_clip` in the Isaac
    /// task**, and is read from the policy contract's <c>joint_velocity_clip</c> field so the two
    /// cannot drift apart. Zero disables it, which is what a 2.3.2 policy wants - it was trained
    /// without one.
    ///
    /// <para>Godot's tightly-limited twist axes chatter against their stops in a way Isaac's solver
    /// does not produce. <c>Shin_L.y</c> is limited to +/-0.10 rad and driven at kp=1800: XPBD
    /// enforces that limit as a rigid position constraint, while Jolt has ActiveBone's explicit
    /// torque fighting the constraint, so the joint slams into its stop and bounces. Measured on a
    /// body still standing at 0.81 m - <c>Forearm_R.y</c> at 33.9 rad/s and <c>Shin_R.y</c> at
    /// 68.1, against a peak of 7.5 across all 45 DOF for the same policy in Isaac.</para>
    ///
    /// <para>That is 45 of the 143 observation floats arriving an order of magnitude outside
    /// anything training produced. The normaliser is baked into the exported graph, so
    /// out-of-range input is amplified straight into a saturated action - the raw output reached
    /// 3.21 against a clamp of 1. The clip bounds it; the matching training-time noise on the Isaac
    /// side is what actually makes the policy tolerate it.</para>
    /// </summary>
    public float JointVelocityClip { get; set; }

    /// <summary>
    /// EMA smoothing on observation slice [55:100], in (0,1]. 1 reports the raw per-tick rate.
    ///
    /// <para><b>Godot's raw joint velocity does not report the limb, it reports the solver.</b>
    /// Measured while the dummy is standing perfectly still under Godot's own balance layer - head
    /// steady at 1.533 m for twenty seconds - the joint-velocity slice still reads about **5 rad/s**
    /// continuously, because tightly-limited twist axes chatter against their own constraints.
    /// Isaac reads about 0.2 rad/s in the same situation.</para>
    ///
    /// <para>So 45 of the 143 observation floats tell the policy the body is thrashing at a moment
    /// when it is motionless, and it acts on that. Clipping bounds the extreme; it does nothing
    /// about a persistent 25x offset in the normal case. Filtering is what makes the channel report
    /// the same physical quantity in both engines.</para>
    /// </summary>
    public float JointVelocityFilter { get; set; } = 1.0f;

    /// <summary>Smoothed joint velocities, one per DOF. See <see cref="JointVelocityFilter"/>.</summary>
    private float[]? _filteredJointVelocity;

    /// <summary>
    /// Height below which a contact bone counts as "down", metres, when <see cref="UseHeightContacts"/>
    /// is set. Matches <c>CONTACT_HEIGHT</c> in the Newton task.
    /// </summary>
    public const float ContactHeight = 0.06f;

    /// <summary>
    /// Derive the four contact flags from bone HEIGHT rather than real contact.
    ///
    /// <para>Off by default, because real contact is the better signal and it is what the Isaac Lab
    /// 2.3.2 policies were trained against. Set it for a policy trained under Isaac Lab 3 / Newton:
    /// Newton's contact reporting does not surface through Isaac Lab's <c>ContactSensor</c> on that
    /// backend, so the task derives these four floats from height instead
    /// (<c>CONTACT_HEIGHT</c> in <c>stand_env.py</c>).</para>
    ///
    /// <para>The two disagree in both directions - a foot 5 cm above the floor reads "down" here and
    /// "up" from a contact test, and a foot resting on a raised surface reads the opposite. Four
    /// floats of 143, and for a planted-feet stand they mostly agree; it matters more once hands
    /// bear load, i.e. for get-up.</para>
    /// </summary>
    public bool UseHeightContacts { get; set; }

    private void AppendContactFlags(in RlContext context, List<float> obs)
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

            bool valid = found != null && GodotObject.IsInstanceValid(found);
            bool down = valid
                && (UseHeightContacts
                    ? found!.GlobalPosition.Y < ContactHeight
                    : found!.IsInContactWithWorld());
            obs.Add(down ? 1.0f : 0.0f);
        }
    }

    /// <summary>
    /// The bone's rotation away from its rest pose, as Euler angles about its own local axes.
    ///
    /// Decomposed <see cref="EulerOrder.Xyz"/> to match the URDF chain, which stacks the three
    /// revolutes as <c>rx -> ry -> rz</c> and therefore composes <c>Rx * Ry * Rz</c>. Godot's
    /// default Euler order is YXZ, a different rotation from the same three numbers - which is why
    /// this cannot use the bare <c>Quaternion.GetEuler()</c> overload.
    /// </summary>
    public static Vector3 DeviationFromRest(ActiveBone bone)
    {
        ActiveBone? parent = bone.ParentBone;
        Quaternion boneRot = bone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
        Quaternion parentRot = parent != null && GodotObject.IsInstanceValid(parent)
            ? parent.GlobalTransform.Basis.GetRotationQuaternion().Normalized()
            : Quaternion.Identity;

        Quaternion local = (parentRot.Inverse() * boneRot).Normalized();
        Quaternion deviation = (bone.GetRestLocalRotation().Inverse() * local).Normalized();
        return new Basis(deviation).GetEuler(EulerOrder.Xyz);
    }

    /// <summary>
    /// The pelvis basis with pitch and roll removed, matching Isaac's `yaw_quat`. Built from the
    /// projection of the forward axis onto the ground plane rather than from an Euler angle, which
    /// would be ill-conditioned when the body is near-vertical.
    /// </summary>
    private static Basis YawOnly(in Basis basis)
    {
        Vector3 forward = new Vector3(basis.Z.X, 0.0f, basis.Z.Z);
        if (forward.LengthSquared() < 1e-8f)
        {
            // Body is pitched fully forward or back; its Z has no ground-plane component. Fall
            // back to the X axis, which cannot be degenerate at the same time.
            forward = new Vector3(basis.X.X, 0.0f, basis.X.Z);
            if (forward.LengthSquared() < 1e-8f)
            {
                return Basis.Identity;
            }
            forward = forward.Normalized();
            return new Basis(forward.Cross(Vector3.Up), Vector3.Up, forward).Orthonormalized();
        }

        forward = forward.Normalized();
        return new Basis(Vector3.Up.Cross(forward), Vector3.Up, forward).Orthonormalized();
    }

    private static float Component(in Vector3 v, char axis) => axis switch
    {
        'x' => v.X,
        'y' => v.Y,
        'z' => v.Z,
        _ => 0.0f,
    };

    private static void Append(List<float> obs, in Vector3 v)
    {
        obs.Add(v.X);
        obs.Add(v.Y);
        obs.Add(v.Z);
    }
}

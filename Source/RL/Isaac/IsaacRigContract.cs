using System.Collections.Generic;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// The Isaac Lab rig contract, read from `assets/dummy_rig.json` at runtime.
///
/// **Nothing here is transcribed.** The joint names, their order, their limits and their axis
/// conventions are read out of the same file `tools/tscn_to_urdf.py` generates and
/// `p4f_isaac/assets.py` consumes on the training side. That is the whole point: a policy is
/// defined by the joint ordering it was trained against, and a C# array restating those 36 names
/// is an array that will disagree with the rig the first time the scene is retuned - silently,
/// because a permuted action vector produces a flailing dummy rather than an error.
///
/// This project has already been bitten by exactly that. `obs_action_contract.md` published the
/// action order as `RagdollRLBridge.ControlledBoneNames` x (x,y,z); the generator actually appends
/// to `actuated_joints` inside its TREE-order loop, giving Spine, Thigh_L, Thigh_R, Chest, ...
/// Only index 0 coincided. Reading the list makes that class of drift unrepresentable.
///
/// <para><b>Limits are in Isaac's convention, not Godot's.</b> `Lower`/`Upper` are the URDF values,
/// which for a roll (z) axis are the negated-and-swapped Godot pair - see <see cref="JointSpec"/>.
/// Consumers should do their arithmetic in Isaac space and convert once, at the boundary.</para>
/// </summary>
public sealed class IsaacRigContract
{
    /// <summary>Default location; the rig lives inside the Godot project, so `res://` reaches it.</summary>
    public const string DefaultPath = "res://isaac_lab/assets/dummy_rig.json";

    /// <summary>Joints per decomposed Godot joint: one revolute per axis (see the URDF chain).</summary>
    public const int AxesPerBone = 3;

    /// <summary>
    /// One revolute DOF - a third of one Godot <c>Generic6DOFJoint3D</c>.
    /// </summary>
    /// <param name="Name">URDF joint name, e.g. <c>UpperArm_L_rz</c>.</param>
    /// <param name="Bone">Owning <see cref="Ragdoll.ActiveBone.BoneName"/>.</param>
    /// <param name="GodotAxis">Which Godot local axis this DOF rotates about: 'x', 'y' or 'z'.</param>
    /// <param name="Reversed">
    /// True when Isaac's sense is opposite to Godot's, i.e. <c>isaac = -godot</c>. Set for every
    /// roll (z) axis, because `AXIS_MAP["z"]` emits URDF +X - which is Godot -Z - while negating
    /// only the limit pair, not the axis vector.
    /// </param>
    /// <param name="Lower">Lower limit in ISAAC's convention, radians.</param>
    /// <param name="Upper">Upper limit in ISAAC's convention, radians.</param>
    /// <param name="Stiffness">Drive proportional gain, N.m/rad. Same value the Godot scene authors.</param>
    /// <param name="Damping">Drive derivative gain, N.m.s/rad.</param>
    /// <param name="Effort">Torque ceiling, N.m.</param>
    public sealed record JointSpec(
        string Name,
        string Bone,
        char GodotAxis,
        bool Reversed,
        float Lower,
        float Upper,
        float Stiffness,
        float Damping,
        float Effort);

    /// <summary>
    /// The 36 policy-actuated DOF, in ACTION-VECTOR order: index <c>3*i+{0,1,2}</c> is bone i's
    /// x/y/z. This is `actuated_joints`, which is what `find_joints(..., preserve_order=True)`
    /// resolves the action tensor through on the Isaac side.
    /// </summary>
    public IReadOnlyList<JointSpec> ActuatedJoints { get; }

    /// <summary>
    /// All 45 DOF in OBSERVATION order - `physx_dof_order`, the ordering PhysX assigned at USD
    /// import. Observation slices [10:55] and [55:100] come from raw `data.joint_pos` /
    /// `data.joint_vel`, so they follow this and NOT the URDF's declaration order.
    /// </summary>
    public IReadOnlyList<JointSpec> DofOrder { get; }

    private IsaacRigContract(IReadOnlyList<JointSpec> actuated, IReadOnlyList<JointSpec> dofOrder)
    {
        ActuatedJoints = actuated;
        DofOrder = dofOrder;
    }

    /// <summary>
    /// Reads and validates the contract. Throws rather than returning a partial rig: every failure
    /// here means the policy would be driven against joints it was not trained on, which is worse
    /// than not starting.
    /// </summary>
    /// <summary>
    /// Reads the observation DOF order out of a policy contract written by
    /// <c>isaac_lab_3/scripts/export.py</c>, or null when no path is given.
    ///
    /// <para><b>Why a policy can override the rig's own ordering.</b> `physx_dof_order` records
    /// what PhysX assigned at USD import. A policy trained under Isaac Lab 3 / Newton is indexed by
    /// what NEWTON assigned, and the two differ in <b>42 of 45 slots</b> on this rig. Feeding a
    /// Newton policy the PhysX order permutes 90 of the 143 observation floats.</para>
    ///
    /// <para>That failure is silent in the worst way: at the rest pose every joint angle sits near
    /// zero, so a permutation changes almost nothing numerically and no value check can see it. The
    /// dummy simply flails. This is the same defect class that made `obs_action_contract.md` wrong
    /// four separate ways, which is why the ordering travels WITH the policy instead of being
    /// assumed from the rig.</para>
    /// </summary>
    public static IReadOnlyList<string>? LoadDofOrderOverride(string resPath)
    {
        if (string.IsNullOrEmpty(resPath))
        {
            return null;
        }

        using Godot.FileAccess? file = Godot.FileAccess.Open(resPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            throw new System.IO.FileNotFoundException(
                $"Policy contract not found at '{resPath}' (Godot error {Godot.FileAccess.GetOpenError()}). "
                + "Export a policy with isaac_lab_3/scripts/export.py, which writes it beside the .onnx.");
        }

        using JsonDocument doc = JsonDocument.Parse(file.GetAsText());
        if (!doc.RootElement.TryGetProperty("newton_dof_order", out JsonElement order))
        {
            return null;
        }

        var names = new List<string>(order.GetArrayLength());
        foreach (JsonElement element in order.EnumerateArray())
        {
            names.Add(element.GetString() ?? string.Empty);
        }
        return names;
    }

    /// <summary>
    /// Reads <c>joint_velocity_clip</c> from a policy contract, or 0 when the file has no such
    /// field. See <see cref="IsaacObservation.JointVelocityClip"/> for why it matters; the point of
    /// reading it from the contract rather than hard-coding it is that the bound is part of what
    /// the policy was trained against, and a Godot side using a different one is feeding the
    /// network a channel it never saw.
    /// </summary>
    /// <summary>
    /// Reads <c>action_rate_limit</c> from a policy contract, or 0 when absent. See
    /// <see cref="IsaacActionSpace.ActionRateLimit"/>. It travels with the policy for the same
    /// reason the DOF order does: a policy trained with a cap is a different controller from one
    /// trained without, and neither failure mode announces itself.
    /// </summary>
    public static float LoadActionRateLimit(string resPath) => LoadContractFloat(resPath, "action_rate_limit");

    public static float LoadJointVelocityClip(string resPath) => LoadContractFloat(resPath, "joint_velocity_clip");

    /// <summary>
    /// Reads <c>action_scale</c> from a policy contract, or 0 when absent (use the default then).
    /// See <see cref="IsaacActionSpace.ActionScale"/> - it must come from the policy, not a const.
    /// </summary>
    public static float LoadActionScale(string resPath) => LoadContractFloat(resPath, "action_scale");

    /// <summary>
    /// One-line description of WHICH brain a contract belongs to: brain stem, run directory,
    /// checkpoint and action scale, e.g.
    /// <c>balance &lt;- stand_assist/2026-08-27_14-12-16_night06 model_3087 (scale 0.15)</c>.
    ///
    /// Deliberately the same facts in the same order as the <c>[watch]</c> line the Isaac side
    /// prints, so the two can be compared at a glance. That matters more since the export was
    /// renamed after the BRAIN rather than the task: one artifact now serves Stand and Perturb, so
    /// the scene name no longer tells you what is loaded - which is precisely how a measured-dead
    /// perturb export ran unnoticed while the working brain sat beside it.
    ///
    /// Returns an empty string when the contract or the field is missing; the caller should stay
    /// silent rather than print a half-line.
    /// </summary>
    public static string DescribeBrain(string contractResPath, string policyResPath)
    {
        string checkpoint = LoadContractString(contractResPath, "source_checkpoint");
        if (string.IsNullOrEmpty(checkpoint))
        {
            return string.Empty;
        }

        // "…/logs/rsl_rl/p4f_newton_stand_assist/<run>/model_3087.pt" -> run + checkpoint. Split on
        // both separators: the contract records a Windows path, but it is only ever text here.
        string[] parts = checkpoint.Split('/', '\\');
        string model = System.IO.Path.GetFileNameWithoutExtension(parts[^1]);
        string run = parts.Length >= 2 ? parts[^2] : string.Empty;
        string lineage = parts.Length >= 3 ? parts[^3].Replace("p4f_newton_", string.Empty) : string.Empty;

        // The brain is the artifact stem - "balance" out of "balance_policy.onnx".
        string brain = System.IO.Path.GetFileNameWithoutExtension(policyResPath).Replace("_policy", string.Empty);
        float scale = LoadActionScale(contractResPath);

        // Invariant, so the scale reads "0.15" and not the locale's "0,15". The whole point of this
        // line is to be comparable at a glance with the `[watch]` line the Isaac side prints, and a
        // decimal comma in a column of numbers reads as a thousands separator.
        return string.Format(
            System.Globalization.CultureInfo.InvariantCulture,
            "{0} <- {1}/{2} {3} (scale {4:0.###})", brain, lineage, run, model, scale);
    }

    private static string LoadContractString(string resPath, string field)
    {
        if (string.IsNullOrEmpty(resPath))
        {
            return string.Empty;
        }

        using Godot.FileAccess? file = Godot.FileAccess.Open(resPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            return string.Empty;
        }

        using JsonDocument doc = JsonDocument.Parse(file.GetAsText());
        return doc.RootElement.TryGetProperty(field, out JsonElement value)
            ? value.GetString() ?? string.Empty
            : string.Empty;
    }

    private static float LoadContractFloat(string resPath, string field)
    {
        if (string.IsNullOrEmpty(resPath))
        {
            return 0.0f;
        }

        using Godot.FileAccess? file = Godot.FileAccess.Open(resPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            return 0.0f;
        }

        using JsonDocument doc = JsonDocument.Parse(file.GetAsText());
        return doc.RootElement.TryGetProperty(field, out JsonElement value) ? value.GetSingle() : 0.0f;
    }

    /// <param name="dofOrderOverride">
    /// Observation DOF order to use instead of the rig's own `physx_dof_order`. Supply the policy's
    /// `newton_dof_order` when driving a policy trained under Isaac Lab 3 / Newton.
    /// </param>
    public static IsaacRigContract Load(
        string resPath = DefaultPath,
        IReadOnlyList<string>? dofOrderOverride = null)
    {
        using Godot.FileAccess? file = Godot.FileAccess.Open(resPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            throw new System.IO.FileNotFoundException(
                $"Isaac rig contract not found at '{resPath}' (Godot error {Godot.FileAccess.GetOpenError()}). "
                + "Run isaac_lab/scripts/convert.ps1 to generate it.");
        }

        using JsonDocument doc = JsonDocument.Parse(file.GetAsText());
        JsonElement root = doc.RootElement;

        JsonElement joints = root.GetProperty("joints");
        var byName = new Dictionary<string, JointSpec>(joints.EnumerateObject().TryGetNonEnumeratedCount(out int n) ? n : 45);
        foreach (JsonProperty entry in joints.EnumerateObject())
        {
            string godotAxis = entry.Value.GetProperty("godot_axis").GetString() ?? string.Empty;
            if (godotAxis.Length != 1)
            {
                throw new System.InvalidOperationException(
                    $"Joint '{entry.Name}' has godot_axis '{godotAxis}'; expected a single character.");
            }

            byName[entry.Name] = new JointSpec(
                entry.Name,
                entry.Value.GetProperty("bone").GetString() ?? string.Empty,
                godotAxis[0],
                entry.Value.GetProperty("reversed").GetBoolean(),
                entry.Value.GetProperty("lower").GetSingle(),
                entry.Value.GetProperty("upper").GetSingle(),
                entry.Value.GetProperty("stiffness").GetSingle(),
                entry.Value.GetProperty("damping").GetSingle(),
                entry.Value.GetProperty("effort").GetSingle());
        }

        List<JointSpec> actuated = Resolve(root, "actuated_joints", byName, resPath);

        // `physx_dof_order` is written by convert_asset.py from the LIVE articulation, because the
        // import - not the URDF - decides it. An older rig JSON predates that step and has no such
        // key; failing loudly is right, since silently substituting declaration order would put
        // 90 observation slots in the wrong places and look like a policy that does not transfer.
        List<JointSpec> dofOrder;
        if (dofOrderOverride != null)
        {
            // The policy carries its own ordering. Resolved through the same name->spec dictionary,
            // so an unknown joint still fails loudly rather than silently shifting the slices.
            dofOrder = new List<JointSpec>(dofOrderOverride.Count);
            foreach (string name in dofOrderOverride)
            {
                if (!byName.TryGetValue(name, out JointSpec? spec))
                {
                    throw new System.InvalidOperationException(
                        $"The policy contract lists '{name}' in newton_dof_order, but '{resPath}' "
                        + "has no matching entry under 'joints'. The policy and the rig disagree "
                        + "about what joints exist; check they were generated from the same scene.");
                }
                dofOrder.Add(spec);
            }
            GD.Print($"[IsaacRigContract] observation DOF order from the policy contract "
                     + $"({dofOrder.Count} entries), not the rig's physx_dof_order.");
        }
        else if (!root.TryGetProperty("physx_dof_order", out _))
        {
            throw new System.InvalidOperationException(
                $"'{resPath}' has no 'physx_dof_order'. It predates the step in convert_asset.py "
                + "that records it. Re-run isaac_lab/scripts/convert.ps1.");
        }
        else
        {
            dofOrder = Resolve(root, "physx_dof_order", byName, resPath);
        }

        if (actuated.Count % AxesPerBone != 0)
        {
            throw new System.InvalidOperationException(
                $"'actuated_joints' has {actuated.Count} entries, not a multiple of {AxesPerBone}.");
        }

        return new IsaacRigContract(actuated, dofOrder);
    }

    private static List<JointSpec> Resolve(
        JsonElement root,
        string key,
        Dictionary<string, JointSpec> byName,
        string resPath)
    {
        JsonElement array = root.GetProperty(key);
        var resolved = new List<JointSpec>(array.GetArrayLength());
        foreach (JsonElement element in array.EnumerateArray())
        {
            string name = element.GetString() ?? string.Empty;
            if (!byName.TryGetValue(name, out JointSpec? spec))
            {
                throw new System.InvalidOperationException(
                    $"'{resPath}' lists '{name}' in {key} but has no matching entry under 'joints'.");
            }
            resolved.Add(spec);
        }
        return resolved;
    }

    /// <summary>
    /// EXPERIMENT: treat the Godot-to-URDF map as the reflection it measurably is.
    ///
    /// `tools/tscn_to_urdf.py:to_urdf` sends godot (x,y,z) to (-z, x, y), whose matrix has
    /// determinant -1 - a reflection, not a rotation. The URDF dummy is therefore a MIRROR image of
    /// the Godot one (UpperArm_L sits at godot x=+0.36, which maps to urdf y=+0.24, i.e. the
    /// opposite side once ROS's +Y-is-left convention is applied).
    ///
    /// Under a reflection M, a rotation of angle theta about axis n becomes angle MINUS theta about
    /// axis Mn - so every axis should invert, not just roll. `AXIS_MAP` inverts only z, via the
    /// limit swap. If that is the whole story, the correct conversion is a plain negation on all
    /// three axes, and this flag makes that testable in one place rather than in two.
    ///
    /// <para>Roll is the axis that must NOT gain a second flip. `AXIS_MAP` declares godot z as urdf
    /// +X while `to_urdf` actually sends godot +Z to urdf -X; the `reversed` flag and the swapped
    /// limit pair already absorb that direction reversal. Adding the reflection's sense reversal on
    /// top would flip it twice and land back where it started. So the two hypotheses differ by
    /// exactly an XOR: roll-only (as documented) versus pitch-and-yaw-only (reflection).</para>
    ///
    /// Set false to fall back to the contract-as-documented behaviour.
    /// </summary>
    public static bool MirrorCompensation { get; set; } = false;

    /// <summary>Whether this DOF's sense differs between the engines, under the active hypothesis.</summary>
    private static bool Inverts(in JointSpec spec) => spec.Reversed ^ MirrorCompensation;

    /// <summary>
    /// Converts a joint angle from Godot's sense to Isaac's. The inverse is the same operation -
    /// see <see cref="ToGodot"/>.
    /// </summary>
    public static float ToIsaac(in JointSpec spec, float godotAngle) =>
        Inverts(spec) ? -godotAngle : godotAngle;

    /// <summary>Converts a joint angle from Isaac's sense to Godot's. Self-inverse with <see cref="ToIsaac"/>.</summary>
    public static float ToGodot(in JointSpec spec, float isaacAngle) =>
        Inverts(spec) ? -isaacAngle : isaacAngle;

    /// <summary>Bone names in action order, one per three DOF. Order matches <see cref="ActuatedJoints"/>.</summary>
    public string[] ActuatedBoneNames()
    {
        var names = new string[ActuatedJoints.Count / AxesPerBone];
        for (int i = 0; i < names.Length; i++)
        {
            names[i] = ActuatedJoints[i * AxesPerBone].Bone;
        }
        return names;
    }
}

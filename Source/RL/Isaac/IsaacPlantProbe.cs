using Godot;
using Physics4Fun.Ragdoll;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// Godot half of the two-engine step-response probe. The Isaac half is
/// `isaac_lab_3/scripts/probe_plant.py`; run both with the same bone, axis, amplitude and gravity
/// setting and the traces are directly comparable.
///
/// <para><b>Why a probe rather than a comparison of gains.</b> `p4f_newton/godot_plant.py` scales
/// Isaac's <c>stiffness</c>/<c>damping</c> by the factor Godot's Stable PD applies to its own
/// authored gains. That corrects a real, measured discrepancy - but it assumes <c>ke</c> in Newton's
/// XPBD drive and <c>Kp</c> in Godot's SPD buy the same torque, and they plainly might not: XPBD
/// applies the drive as a compliant positional constraint INSIDE the solve, while Godot computes an
/// explicit torque and applies it as a force. No amount of reading parameter names settles that.
/// Commanding one joint a step and recording where the angle actually goes does.</para>
///
/// <para>Gravity is disabled by default so the drive is the only thing moving the limb, and the
/// pelvis is frozen so the reaction torque cannot simply push the whole body around. What is left
/// is a second-order response whose rise time gives the effective natural frequency and whose
/// overshoot gives the damping ratio.</para>
///
/// <para>Output is machine-parseable `[GPROBE]` lines, matching the Isaac probe's trace format.</para>
/// </summary>
public partial class IsaacPlantProbe : Node3D
{
    /// <summary>Bone to step. Godot bone names, e.g. Forearm_L - the Isaac joint `joint_Forearm_L:0`.</summary>
    [Export] public string BoneName { get; set; } = "Forearm_L";

    /// <summary>Local axis to rotate about: 0 = X, 1 = Y, 2 = Z. Matches the `:0/:1/:2` joint suffix.</summary>
    [Export(PropertyHint.Range, "0,2,1")] public int Axis { get; set; }

    /// <summary>Commanded step amplitude (rad), about <see cref="Axis"/> from the bone's rest pose.</summary>
    [Export] public float TargetRad { get; set; } = 0.30f;

    /// <summary>Seconds held at rest before the step, so the body is settled when it lands.</summary>
    [Export] public float SettleSeconds { get; set; } = 0.5f;

    /// <summary>Seconds of trace recorded after the step.</summary>
    [Export] public float DurationSeconds { get; set; } = 1.5f;

    /// <summary>Zero every bone's gravity. The Isaac probe's default is likewise gravity OFF.</summary>
    [Export] public bool DisableGravity { get; set; } = true;

    /// <summary>
    /// Zero the gravity feed-forward. Only meaningful with <see cref="DisableGravity"/> off - with no
    /// gravity there is no load to compensate, so the feed-forward is already zero.
    /// </summary>
    [Export] public bool DisableLoadCompensation { get; set; }

    /// <summary>
    /// Freeze the pelvis so the reaction torque moves the limb rather than the whole body.
    ///
    /// <para><b>Must MATCH the Isaac probe, which does not pin its root.</b> Godot applies an
    /// equal-and-opposite torque to the parent by design; with a frozen pelvis that reaction is
    /// absorbed by the world, and with a free one it counter-rotates the body. Those are different
    /// mechanical systems and their joint responses are not comparable. Freezing here while Isaac
    /// ran free produced exactly that mismatch: "overshoot" peaks arriving 2.5 s after the step -
    /// slow drift of a tumbling body, not a second-order transient - and more damping reading as
    /// MORE overshoot. Default false, matching Isaac.</para>
    /// </summary>
    [Export] public bool FreezePelvis { get; set; }

    private HumanoidRagdoll? _ragdoll;
    private ActiveBone? _bone;
    private Quaternion _rest = Quaternion.Identity;
    private float _elapsed;
    private bool _stepped;
    private bool _done;

    public override void _Ready()
    {
        _ragdoll = FindRagdoll(this);
        if (_ragdoll == null)
        {
            GD.PrintErr("[GPROBE] no HumanoidRagdoll under this node - nothing to probe.");
            SetPhysicsProcess(false);
            return;
        }

        // RL state, so the procedural balance controller does not drive the bones underneath the
        // probe. Without this the measurement is of the controller, not of the actuator.
        _ragdoll.SetState(RagdollState.ReinforcementLearning);

        foreach (ActiveBone bone in _ragdoll.GetBones())
        {
            if (!IsInstanceValid(bone))
            {
                continue;
            }
            if (DisableGravity)
            {
                bone.GravityScale = 0.0f;
            }
            if (DisableLoadCompensation)
            {
                bone.LoadCompensationScale = 0.0f;
            }
            bone.TargetLocalRotation = bone.GetRestLocalRotation();
            if (bone.BoneName == BoneName)
            {
                _bone = bone;
            }
        }

        if (FreezePelvis && _ragdoll.Pelvis != null && IsInstanceValid(_ragdoll.Pelvis))
        {
            _ragdoll.Pelvis.Freeze = true;
        }

        if (_bone == null)
        {
            GD.PrintErr($"[GPROBE] no bone named '{BoneName}'.");
            SetPhysicsProcess(false);
            return;
        }

        _rest = _bone.GetRestLocalRotation();
        GD.Print($"[GPROBE] bone={BoneName} axis={Axis} target={TargetRad:G6} "
                 + $"gravity={(DisableGravity ? "OFF" : "ON")} "
                 + $"loadComp={(DisableLoadCompensation ? "OFF" : "ON")} "
                 + $"kp={_bone.ProportionalGain:G6} kd={_bone.DerivativeGain:G6} "
                 + $"physicsHz={Engine.PhysicsTicksPerSecond}");
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_done || _bone == null || !IsInstanceValid(_bone))
        {
            return;
        }

        _elapsed += (float)delta;

        if (!_stepped)
        {
            if (_elapsed < SettleSeconds)
            {
                return;
            }
            _bone.TargetLocalRotation = (_rest * AxisQuaternion(TargetRad)).Normalized();
            _stepped = true;
            _elapsed = 0.0f;
            GD.Print($"[GPROBE] start angle={CurrentAngle():G6}");
            return;
        }

        GD.Print($"[GPROBE] t={_elapsed * 1000.0f:F1} angle={CurrentAngle():G6}");

        if (_elapsed >= DurationSeconds)
        {
            _done = true;
            GD.Print($"[GPROBE] final angle={CurrentAngle():G6}");
            GetTree().Quit();
        }
    }

    private Quaternion AxisQuaternion(float radians)
    {
        Vector3 axis = Axis switch
        {
            0 => Vector3.Right,
            1 => Vector3.Up,
            _ => Vector3.Back,
        };
        return new Quaternion(axis, radians);
    }

    /// <summary>Angle about <see cref="Axis"/> between the bone's current local rotation and rest.</summary>
    private float CurrentAngle()
    {
        if (_bone == null || !IsInstanceValid(_bone))
        {
            return 0.0f;
        }

        Quaternion self = _bone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
        Quaternion parent = _bone.ParentBone != null && IsInstanceValid(_bone.ParentBone)
            ? _bone.ParentBone.GlobalTransform.Basis.GetRotationQuaternion().Normalized()
            : Quaternion.Identity;

        Quaternion local = (parent.Inverse() * self).Normalized();
        Vector3 euler = (_rest.Inverse() * local).Normalized().GetEuler();
        return Axis switch
        {
            0 => euler.X,
            1 => euler.Y,
            _ => euler.Z,
        };
    }

    private static HumanoidRagdoll? FindRagdoll(Node node)
    {
        foreach (Node child in node.GetChildren())
        {
            if (child is HumanoidRagdoll ragdoll)
            {
                return ragdoll;
            }
            HumanoidRagdoll? found = FindRagdoll(child);
            if (found != null)
            {
                return found;
            }
        }
        return null;
    }
}

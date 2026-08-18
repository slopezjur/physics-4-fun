using Godot;
using Physics4Fun.Core.Math;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Biomechanical active bone with impedance-controlled 3D rotational actuation.
/// Uses pure PD control (Ki = 0) to eliminate contact limit-cycle chatter and shaking.
/// Enforces Newton's Third Law by applying reaction torques to the parent bone.
/// </summary>
public partial class ActiveBone : RigidBody3D
{
    [Export] public string BoneName { get; set; } = string.Empty;
    [Export] public ActiveBone? ParentBone { get; set; }

    [ExportGroup("Biomechanical Actuator Gains")]
    [Export] public float ProportionalGain { get; set; } = 450.0f;
    [Export] public float DerivativeGain { get; set; } = 40.0f;
    [Export] public float IntegralGain { get; set; } = 0.0f; // Pure PD control prevents contact windup & jitter
    [Export] public float MaxTorque { get; set; } = 1200.0f;

    /// <summary>
    /// Anti-windup clamp for the integral error (rad·s). Bounds the static-load feed-forward
    /// contribution of the I term so swing/impact transients cannot wind it up.
    /// </summary>
    [Export] public float MaxIntegralError { get; set; } = 0.5f;

    /// <summary>
    /// Fallback rotational inertia (kg·m²) fed to the SPD denominator until the bone's real
    /// inertia tensor is captured from the physics state on the first integration step
    /// (see _IntegrateForces). Torques act about the bone's CoM, so the real free inertia
    /// is what the implicit formulation requires — a fictitious larger value voids stability.
    /// </summary>
    [Export] public float EffectiveInertia { get; set; } = 0.35f;

    /// <summary>Smallest inertia (kg·m²) accepted from the physics state; guards the SPD denominator against degenerate/zero tensors.</summary>
    private const float MinCapturedInertia = 0.01f;

    /// <summary>Smallest inverse-inertia tensor entry treated as valid; below this the body is effectively static.</summary>
    private const float MinValidInverseInertia = 1e-6f;

    public float ApparentInertiaMultiplier { get; set; } = 1.0f;

    public float MuscleStrength { get; set; } = 1.0f;
    public Quaternion TargetLocalRotation { get; set; } = Quaternion.Identity;
    
    /// <summary>
    /// Kinematic angular offset injected by the motor cortex (BalanceController).
    /// Used for dynamic closed-loop balance without fighting the local PID.
    /// </summary>
    public Quaternion FeedForwardTargetOffset { get; set; } = Quaternion.Identity;
    public Vector3 LastAppliedTorque { get; private set; } = Vector3.Zero;
    public float LastTrackingErrorDeg { get; private set; } = 0.0f;

    [Signal]
    public delegate void HitReceivedEventHandler(ActiveBone bone, Vector3 hitPoint, Vector3 impulse);

    private PidController3D _pid = null!;
    private Transform3D _initialTransform;
    private Quaternion _restLocalRotation = Quaternion.Identity;
    private Quaternion _smoothedFeedForwardOffset = Quaternion.Identity;

    public override void _Ready()
    {
        _initialTransform = GlobalTransform;
        _pid = new PidController3D(ProportionalGain, DerivativeGain, IntegralGain, MaxTorque);
        _pid.EffectiveInertia = EffectiveInertia;
        _pid.MaxIntegral = MaxIntegralError;

        ContactMonitor = true;
        MaxContactsReported = 3;
        BodyEntered += OnBodyEntered;

        // Auto-resolve ParentBone from joint connections if not assigned in inspector
        if (ParentBone == null)
        {
            ResolveParentBoneFromJoints();
        }

        if (ParentBone != null && IsInstanceValid(ParentBone))
        {
            Quaternion parentRot = ParentBone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
            Quaternion selfRot = GlobalTransform.Basis.GetRotationQuaternion().Normalized();
            _restLocalRotation = (parentRot.Inverse() * selfRot).Normalized();
        }
        else
        {
            _restLocalRotation = GlobalTransform.Basis.GetRotationQuaternion().Normalized();
        }

        TargetLocalRotation = _restLocalRotation;
    }

    private void OnBodyEntered(Node body)
    {
        if (body is RigidBody3D otherRigidBody && !(body is ActiveBone))
        {
            Vector3 relativeVel = otherRigidBody.LinearVelocity - LinearVelocity;
            if (relativeVel.Length() > 4.0f)
            {
                Vector3 impulse = relativeVel * otherRigidBody.Mass;
                EmitSignal(SignalName.HitReceived, this, GlobalPosition, impulse);
            }
        }
    }

    public void ApplyBiomechanicalTorque(float delta)
    {
        if (ParentBone == null || MuscleStrength <= 0.001f || !IsInstanceValid(ParentBone))
        {
            return;
        }

        // Dynamically scale actuator impedance with muscle strength
        _pid.ProportionalGain = ProportionalGain * MuscleStrength;
        _pid.DerivativeGain = DerivativeGain * Mathf.Sqrt(MuscleStrength);
        _pid.IntegralGain = IntegralGain * MuscleStrength;
        _pid.MaxTorque = MaxTorque * MuscleStrength;

        // Smooth reflex and motor offsets to eliminate high-frequency chatter
        float smoothAlpha = 1.0f - Mathf.Exp(-25.0f * delta);
        _smoothedFeedForwardOffset = _smoothedFeedForwardOffset.Slerp(FeedForwardTargetOffset, smoothAlpha).Normalized();

        Quaternion parentGlobalRot = ParentBone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
        // Incorporate motor cortex feed-forward offset
        Quaternion targetGlobalRot = (parentGlobalRot * TargetLocalRotation * _smoothedFeedForwardOffset).Normalized();
        Quaternion currentGlobalRot = GlobalTransform.Basis.GetRotationQuaternion().Normalized();

        // Biomechanical relative angular velocity damping
        Vector3 relativeAngVel = AngularVelocity - ParentBone.AngularVelocity;

        Vector3 torque = _pid.Update(currentGlobalRot, targetGlobalRot, relativeAngVel, delta);
        LastAppliedTorque = torque;

        // Calculate angular tracking error magnitude for diagnostics
        Quaternion errorQuat = (targetGlobalRot * currentGlobalRot.Inverse()).Normalized();
        float errorAngleRad = 2.0f * Mathf.Acos(Mathf.Clamp(Mathf.Abs(errorQuat.W), 0.0f, 1.0f));
        LastTrackingErrorDeg = Mathf.RadToDeg(errorAngleRad);
        
        // Apply equal and opposite torque synchronously before physics step
        ApplyTorque(torque);
        ParentBone.ApplyTorque(-torque);
    }

    private bool _pendingReset = false;
    private float _baseCapturedInertia = -1.0f;

    public override void _IntegrateForces(PhysicsDirectBodyState3D state)
    {
        base._IntegrateForces(state);

        // Feed the SPD denominator the bone's REAL free inertia about its CoM (torques are applied
        // about the CoM, not the joint pivot).
        if (_baseCapturedInertia < 0.0f)
        {
            Basis invInertia = state.GetInverseInertiaTensor();
            float maxInv = Mathf.Max(Mathf.Abs(invInertia.X.X), Mathf.Max(Mathf.Abs(invInertia.Y.Y), Mathf.Abs(invInertia.Z.Z)));
            if (maxInv > MinValidInverseInertia)
            {
                _baseCapturedInertia = Mathf.Max(MinCapturedInertia, 1.0f / maxInv);
            }
            else
            {
                _baseCapturedInertia = EffectiveInertia;
            }
        }

        // Dynamically scale apparent inertia when grounded. Light limbs (like arms) coupled 
        // to the ground bear the weight of the entire body. Scaling the inertia prevents the 
        // Tan-Liu-Turk SPD denominator from crippling the torque output during push-ups.
        if (_baseCapturedInertia > 0.0f)
        {
            _pid.EffectiveInertia = _baseCapturedInertia * ApparentInertiaMultiplier;
        }

        if (_pendingReset)
        {
            state.Transform = _initialTransform;
            state.LinearVelocity = Vector3.Zero;
            state.AngularVelocity = Vector3.Zero;
            _pendingReset = false;
        }
    }

    public void ResetBone()
    {
        _pendingReset = true;
        Rid rid = GetRid();
        PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.Transform, _initialTransform);
        PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.LinearVelocity, Vector3.Zero);
        PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.AngularVelocity, Vector3.Zero);

        GlobalTransform = _initialTransform;
        LinearVelocity = Vector3.Zero;
        AngularVelocity = Vector3.Zero;
        TargetLocalRotation = _restLocalRotation;
        FeedForwardTargetOffset = Quaternion.Identity;
        _smoothedFeedForwardOffset = Quaternion.Identity;
        _pid.Reset();
    }

    public Quaternion GetRestLocalRotation() => _restLocalRotation;

    /// <summary>
    /// True when this bone is in physical contact with a non-ragdoll body (floor, walls, props).
    /// Uses the contact monitor enabled in _Ready — real collision contacts, not ray proximity.
    /// </summary>
    public bool IsInContactWithWorld()
    {
        foreach (Node3D body in GetCollidingBodies())
        {
            if (body is not ActiveBone)
            {
                return true;
            }
        }
        return false;
    }

    private void ResolveParentBoneFromJoints()
    {
        Node? parentNode = GetParent();
        if (parentNode == null) return;

        foreach (Node child in parentNode.GetChildren())
        {
            if (child is Generic6DofJoint3D joint)
            {
                Node? nodeB = joint.GetNodeOrNull(joint.NodeB);
                if (nodeB == this)
                {
                    Node? nodeA = joint.GetNodeOrNull(joint.NodeA);
                    if (nodeA is ActiveBone parentActiveBone)
                    {
                        ParentBone = parentActiveBone;
                        return;
                    }
                }
            }
        }
    }
}

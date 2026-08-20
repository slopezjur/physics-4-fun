using System.Collections.Generic;
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

    /// <summary>
    /// Target closed-loop damping ratio used only when <see cref="AutoTuneDamping"/> is enabled
    /// (1.0 = critically damped).
    /// </summary>
    [Export] public float DampingRatio { get; set; } = 1.0f;

    /// <summary>
    /// Derive the derivative gain from <see cref="DampingRatio"/> and the captured inertia instead
    /// of using the authored <see cref="DerivativeGain"/>.
    ///
    /// OFF by default: the authored per-bone gains are hand-tuned to sensible achieved damping
    /// (zeta ~0.63-0.83 on torso and legs, ~1.4-1.7 on the arms) and standing balance depends on
    /// them. Enabling this re-derives every joint at a uniform zeta, which is a substantial retune
    /// - the ankle in particular moves from Kd 6 to ~49 - so treat it as a tuning experiment and
    /// verify with a fresh telemetry dump, not as a drop-in default.
    /// </summary>
    [Export] public bool AutoTuneDamping { get; set; } = false;

    /// <summary>
    /// Numerical floor (kg·m²) on the inertia read from the physics state. This is a divide-by-zero
    /// guard ONLY and must stay far below every real bone inertia (the lightest, the hand, is
    /// ~5.3e-4 kg·m² about its minor axis).
    ///
    /// The SPD denominator is unconditionally stable only when it sees the body's TRUE inertia:
    /// the per-tick velocity factor is (1 + Kp·dt²/I) / (1 + Kd·dt/I + Kp·dt²/I), which lies in (0,1]
    /// when I is correct. Feeding a value LARGER than the real inertia shrinks the denominator
    /// while the applied torque still divides by the real inertia, so the factor becomes
    /// 1 - (Kd·dt/I_real)/denominator and goes negative - the joint reverses and amplifies its own
    /// angular velocity every tick. A 0.01 floor did exactly that to the four lightest bone types
    /// (hand, forearm, upper arm, foot), buzzing the arms at the Nyquist frequency.
    /// </summary>
    private const float MinCapturedInertia = 1e-5f;

    /// <summary>Smallest inverse-inertia tensor entry treated as valid; below this the body is effectively static.</summary>
    private const float MinValidInverseInertia = 1e-6f;

    /// <summary>Scales the gravity / support-load feed-forward torque (1.0 = full physical compensation).</summary>
    [Export] public float LoadCompensationScale { get; set; } = 1.0f;

    public float MuscleStrength { get; set; } = 1.0f;

    /// <summary>
    /// Extra impedance scale for phases where this bone is a load-bearing strut rather than a
    /// free limb, applied on top of <see cref="MuscleStrength"/> with the same convention (Kp
    /// linear, Kd by sqrt, so the achieved damping ratio is preserved).
    ///
    /// Deliberately does NOT scale <see cref="MaxTorque"/>: the actuator's torque ceiling is a
    /// physical limit, and raising it would hide whether the limb can really do the job. Gains that
    /// let an arm hang relaxed are far too soft to press a body off the floor, and that difference
    /// is a property of the phase, not of the bone.
    /// </summary>
    public float LoadBearingGainScale { get; set; } = 1.0f;
    public Quaternion TargetLocalRotation { get; set; } = Quaternion.Identity;
    
    /// <summary>
    /// Kinematic angular offset injected by the motor cortex (BalanceController).
    /// Used for dynamic closed-loop balance without fighting the local PID.
    /// </summary>
    public Quaternion FeedForwardTargetOffset { get; set; } = Quaternion.Identity;
    public Vector3 LastAppliedTorque { get; private set; } = Vector3.Zero;

    /// <summary>
    /// Closed-loop PD contribution to the last applied torque, before load compensation and the
    /// output clamp. Recorded separately from <see cref="LastLoadCompensationTorque"/> so a limb
    /// thrashing under a flickering pose target can be told apart from one being kicked by the
    /// support-load feed-forward switching on and off with ground contact.
    /// </summary>
    public Vector3 LastPdTorque { get; private set; } = Vector3.Zero;

    /// <summary>Gravity / support-load feed-forward contribution to the last applied torque, before the output clamp.</summary>
    public Vector3 LastLoadCompensationTorque { get; private set; } = Vector3.Zero;

    public float LastTrackingErrorDeg { get; private set; } = 0.0f;

    [Signal]
    public delegate void HitReceivedEventHandler(ActiveBone bone, Vector3 hitPoint, Vector3 impulse);
    private PidController3D _pid = null!;
    private Transform3D _initialTransform;
    private Quaternion _restLocalRotation = Quaternion.Identity;
    private Quaternion _smoothedFeedForwardOffset = Quaternion.Identity;
    private Generic6DofJoint3D? _joint;
    private readonly List<ActiveBone> _distalChain = new();

    /// <summary>Gravity magnitude (m/s^2) used by the load-compensation feed-forward.</summary>
    private const float GravityMagnitude = 9.81f;

    /// <summary>Guard against cycles when walking the parent chain.</summary>
    private const int MaxChainDepth = 32;

    /// <summary>Share of total body mass this limb carries when planted; set by HumanoidRagdoll.</summary>
    public float SupportedMassShare { get; set; } = 0.0f;

    public override void _Ready()
    {
        _initialTransform = GlobalTransform;
        _pid = new PidController3D(ProportionalGain, DerivativeGain, IntegralGain, MaxTorque);
        _pid.EffectiveInertia = EffectiveInertia;
        _pid.MaxIntegral = MaxIntegralError;

        ContactMonitor = true;
        MaxContactsReported = 3;
        BodyEntered += OnBodyEntered;

        // Always resolve the joint: it is both the parent link and the pivot that
        // load-compensation torques act about.
        ResolveJointAndParent();

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

        // Dynamically scale actuator impedance with muscle strength. Kd scales with sqrt(strength)
        // so the damping ratio Kd/(2·√(Kp·I)) is preserved as Kp scales linearly.
        float impedanceScale = MuscleStrength * Mathf.Max(0.0f, LoadBearingGainScale);
        _pid.ProportionalGain = ProportionalGain * impedanceScale;
        _pid.DerivativeGain = ResolvedDerivativeGain * Mathf.Sqrt(impedanceScale);
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

        // Size the SPD denominator to the axis this actuator is correcting on, not to the bone's
        // smallest principal axis. Both give a stable (0,1] velocity factor, but the minimum
        // needlessly throws away authority on every other axis - by 2.6x at the ankle and 5.4x at
        // the knee, whose flexion axes are transverse to the limb's slender principal axis.
        Quaternion errorQuat = (targetGlobalRot * currentGlobalRot.Inverse()).Normalized();
        if (errorQuat.W < 0.0f)
        {
            errorQuat = new Quaternion(-errorQuat.X, -errorQuat.Y, -errorQuat.Z, -errorQuat.W);
        }
        Vector3 errorRotationVector = PidController3D.QuaternionToRotationVector(errorQuat);
        LastTrackingErrorDeg = Mathf.RadToDeg(errorRotationVector.Length());
        LastEffectiveInertia = GetEffectiveInertiaAboutAxis(errorRotationVector);
        _pid.EffectiveInertia = LastEffectiveInertia;

        Vector3 torque = _pid.Update(currentGlobalRot, targetGlobalRot, relativeAngVel, delta);
        
        // Feed-forward load compensation: supplies the steady-state torque needed to hold this
        // joint against gravity (and against body weight when the limb is planted), so the PD
        // term only has to correct the residual instead of developing large error to generate
        // support force. This is the physical replacement for the removed 10x/100x hack.
        Vector3 loadCompensation = ComputeLoadCompensationTorque(SupportedMassShare) * MuscleStrength;
        Vector3 totalTorque = torque + loadCompensation;

        LastPdTorque = torque;
        LastLoadCompensationTorque = loadCompensation;

        // Clamp the combined output so the feed-forward cannot exceed the actuator's limit
        float maxTorque = MaxTorque * MuscleStrength;
        if (maxTorque > 0.0f && totalTorque.LengthSquared() > maxTorque * maxTorque)
        {
            totalTorque = totalTorque.Normalized() * maxTorque;
        }

        LastAppliedTorque = totalTorque;

        // Apply equal and opposite torque synchronously before physics step
        ApplyTorque(totalTorque);
        ParentBone.ApplyTorque(-totalTorque);
    }

    private bool _pendingReset = false;
    private float _baseCapturedInertia = -1.0f;
    private Basis _inverseInertiaTensor = Basis.Identity;
    private bool _hasInverseInertiaTensor;

    public override void _IntegrateForces(PhysicsDirectBodyState3D state)
    {
        base._IntegrateForces(state);

        // Feed the SPD denominator the bone's REAL free inertia about its CoM (torques are applied
        // about the CoM, not the joint pivot). The full world-space tensor is kept, not just its
        // smallest principal value, so the denominator can be evaluated about the axis the
        // actuator is actually correcting on (see GetEffectiveInertiaAboutAxis).
        _inverseInertiaTensor = state.GetInverseInertiaTensor();
        _hasInverseInertiaTensor = true;

        if (_baseCapturedInertia < 0.0f)
        {
            Basis invInertia = _inverseInertiaTensor;
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

        if (_pendingReset)
        {
            state.Transform = _initialTransform;
            state.LinearVelocity = Vector3.Zero;
            state.AngularVelocity = Vector3.Zero;
            _pendingReset = false;
        }
        if (_pendingTeleportTransform.HasValue)
        {
            state.Transform = _pendingTeleportTransform.Value;
            state.LinearVelocity = Vector3.Zero;
            state.AngularVelocity = Vector3.Zero;
            _pendingTeleportTransform = null;
        }
    }

    /// <summary>Inertia (kg·m²) the SPD denominator was last evaluated with; diagnostic mirror of the per-axis projection.</summary>
    public float LastEffectiveInertia { get; private set; }

    /// <summary>
    /// Rotational inertia (kg·m²) resisting a torque applied about <paramref name="axis"/>.
    ///
    /// For a torque T·â the angular acceleration along â is T·(â·I⁻¹·â), so the scalar inertia the
    /// SPD denominator needs is 1/(â·I⁻¹·â). This is always ≥ the smallest principal inertia, with
    /// equality only when â IS the minor axis, so it recovers real authority rather than inflating
    /// anything: the stability guarantee still holds because the denominator is being matched to
    /// the inertia that actually governs the motion it is damping.
    /// </summary>
    public float GetEffectiveInertiaAboutAxis(Vector3 axis)
    {
        if (!_hasInverseInertiaTensor || axis.LengthSquared() < 1e-12f)
        {
            return CapturedInertia;
        }

        Vector3 unitAxis = axis.Normalized();
        float inverseAlongAxis = unitAxis.Dot(_inverseInertiaTensor * unitAxis);

        // Below this the body is effectively static about the axis; fall back rather than divide.
        if (inverseAlongAxis <= MinValidInverseInertia)
        {
            return CapturedInertia;
        }

        return Mathf.Max(MinCapturedInertia, 1.0f / inverseAlongAxis);
    }

    /// <summary>
    /// Rotational inertia (kg·m²) about the bone's minor principal axis - the conservative lower
    /// bound, used as the fallback before the tensor is captured.
    /// </summary>
    public float CapturedInertia => _baseCapturedInertia > 0.0f ? _baseCapturedInertia : EffectiveInertia;

    /// <summary>
    /// Derivative gain the actuator actually uses at full muscle strength - auto-tuned from
    /// <see cref="DampingRatio"/> and <see cref="CapturedInertia"/>, or the authored value when
    /// <see cref="AutoTuneDamping"/> is off.
    /// </summary>
    public float ResolvedDerivativeGain => AutoTuneDamping
        ? ComputeDerivativeGainForDamping(ProportionalGain, CapturedInertia, DampingRatio, PhysicsDelta)
        : DerivativeGain;

    /// <summary>Physics timestep (s) the SPD denominator is evaluated at.</summary>
    private static float PhysicsDelta => 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);

    /// <summary>Slack (rad) allowed past a joint stop before a target counts as out of range.</summary>
    private const float JointLimitTolerance = 0.02f;

    /// <summary>
    /// True when the commanded <see cref="TargetLocalRotation"/> is reachable within this bone's
    /// joint limits.
    ///
    /// Asking for an angle past a hard stop does not bend the joint further - it pins the actuator
    /// against the limit at full torque, producing a large permanent tracking error and no motion.
    /// A get-up trajectory commanding 1.0 rad of shoulder roll into a 0.5 rad stop looked exactly
    /// like a controller failure until this was measured, so it is worth reporting directly.
    ///
    /// Returns true when there is no joint to check against, so unconstrained bones never register.
    /// </summary>
    /// <summary>
    /// Per-axis angular limits of this bone's joint, in radians about its rest frame.
    ///
    /// Exposed so an external controller can scale its commands to what the joint can actually
    /// reach. Anything outside these bounds is not a smaller motion than requested - it is the
    /// actuator pinned against a hard stop at full torque with a permanent tracking error (see
    /// IsTargetWithinJointLimits below), which is indistinguishable from any other out-of-range
    /// command. A controller that cannot see these numbers cannot avoid that.
    ///
    /// Axes with no joint, or with their limit disabled, report the supplied fallback so callers
    /// do not have to special-case free rotation.
    /// </summary>
    public void GetJointAngularLimits(float fallback, out Vector3 lower, out Vector3 upper)
    {
        lower = new Vector3(-fallback, -fallback, -fallback);
        upper = new Vector3(fallback, fallback, fallback);

        if (_joint == null || !IsInstanceValid(_joint))
        {
            return;
        }

        ReadAxisLimit("angular_limit_x", fallback, out float lx, out float ux);
        ReadAxisLimit("angular_limit_y", fallback, out float ly, out float uy);
        ReadAxisLimit("angular_limit_z", fallback, out float lz, out float uz);
        lower = new Vector3(lx, ly, lz);
        upper = new Vector3(ux, uy, uz);
    }

    private void ReadAxisLimit(string axisPrefix, float fallback, out float lower, out float upper)
    {
        if (_joint == null || !(bool)_joint.Get($"{axisPrefix}/enabled"))
        {
            lower = -fallback;
            upper = fallback;
            return;
        }

        lower = (float)_joint.Get($"{axisPrefix}/lower_angle");
        upper = (float)_joint.Get($"{axisPrefix}/upper_angle");
    }

    /// <remarks>
    /// KNOWN LIMITATION - unreliable for axes whose X limit exceeds +/-pi/2 (1.571 rad).
    ///
    /// Godot decomposes a Basis with Euler order YXZ, whose principal branch can only represent
    /// |x| &lt;= pi/2. Four axes on this rig exceed that - Thigh x(+2.10), Shin x(-2.60),
    /// Forearm x(+2.60), UpperArm x(+3.00) - and for those GetEuler() returns a different triple
    /// than the one commanded, so the comparison below is against the wrong numbers. Measured: a
    /// commanded (2.6, 0.05, 0.05) comes back as (0.54, -3.09, -3.09) and reports a violation that
    /// did not occur.
    ///
    /// A correct check needs swing-twist decomposition about each joint axis rather than Euler.
    /// Until then, treat a reported violation on those axes as unverified. The RL action path does
    /// not depend on this: JointLimitedActionSpace scales into the limits by construction.
    /// </remarks>
    public bool IsTargetWithinJointLimits()
    {
        if (_joint == null || !IsInstanceValid(_joint))
        {
            return true;
        }

        // The joint's limits are expressed about its rest frame, which is what _restLocalRotation
        // captures, so compare the commanded offset from rest rather than the absolute local pose.
        Vector3 commanded = (_restLocalRotation.Inverse() * TargetLocalRotation).Normalized().GetEuler();

        return IsAxisWithinLimit(commanded.X, "angular_limit_x")
            && IsAxisWithinLimit(commanded.Y, "angular_limit_y")
            && IsAxisWithinLimit(commanded.Z, "angular_limit_z");
    }

    private bool IsAxisWithinLimit(float angle, string axisPrefix)
    {
        if (_joint == null || !(bool)_joint.Get($"{axisPrefix}/enabled"))
        {
            return true;
        }

        float lower = (float)_joint.Get($"{axisPrefix}/lower_angle");
        float upper = (float)_joint.Get($"{axisPrefix}/upper_angle");
        return angle >= lower - JointLimitTolerance && angle <= upper + JointLimitTolerance;
    }

    /// <summary>
    /// Derivative gain achieving a target closed-loop damping ratio under the SPD law.
    ///
    /// SPD divides the proportional AND derivative terms by the same denominator
    /// den = 1 + Kd·dt/I + Kp·dt²/I, so the achieved natural frequency is √(Kp/(I·den)) while the
    /// achieved damping term is Kd/(I·den). The damping ratio is therefore
    ///
    ///     ζ = Kd / (2·√(Kp·I·den))
    ///
    /// - the naive free-body form ζ = Kd/(2·√(Kp·I)) overstates it by √den, which is a factor of
    /// 4-6 on the leg joints. Because den itself depends on Kd, substituting and solving the
    /// resulting quadratic gives the closed form below.
    /// </summary>
    public static float ComputeDerivativeGainForDamping(float proportionalGain, float inertia, float dampingRatio, float delta)
    {
        float kp = Mathf.Max(0.0f, proportionalGain);
        float i = Mathf.Max(1e-9f, inertia);
        float z = Mathf.Max(0.0f, dampingRatio);
        float h = Mathf.Max(0.0f, delta);

        // Kd² - 4ζ²·Kp·dt·Kd - 4ζ²·Kp·(I + Kp·dt²) = 0
        float discriminant = (z * z * kp * kp * h * h) + (kp * (i + kp * h * h));
        return (2.0f * z * z * kp * h) + (2.0f * z * Mathf.Sqrt(Mathf.Max(0.0f, discriminant)));
    }

    /// <summary>
    /// Closed-loop damping ratio this actuator actually achieves under the SPD law, at full muscle
    /// strength. ~1.0 is critically damped; below ~0.3 the joint rings and sheds load.
    /// </summary>
    public float AchievedDampingRatio
    {
        get
        {
            float dt = PhysicsDelta;
            float i = Mathf.Max(1e-9f, CapturedInertia);
            float kd = ResolvedDerivativeGain;
            float denominator = 1.0f + (kd * dt / i) + (ProportionalGain * dt * dt / i);
            float scale = ProportionalGain * i * denominator;
            return scale > 0.0f ? kd / (2.0f * Mathf.Sqrt(scale)) : 0.0f;
        }
    }

    public Transform3D InitialTransform => _initialTransform;

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

    private Transform3D? _pendingTeleportTransform;

    public void Teleport(Transform3D targetTransform)
    {
        _pendingTeleportTransform = targetTransform;
        Rid rid = GetRid();
        PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.Transform, targetTransform);
        PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.LinearVelocity, Vector3.Zero);
        PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.AngularVelocity, Vector3.Zero);

        GlobalTransform = targetTransform;
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

    /// <summary>
    /// Locates the Generic6DofJoint3D connecting this bone to its parent and caches it.
    /// The joint's global position is the pivot that load-compensation torques act about.
    /// Also fills in ParentBone when it was not assigned in the inspector.
    /// </summary>
    private void ResolveJointAndParent()
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
                    _joint = joint;
                    Node? nodeA = joint.GetNodeOrNull(joint.NodeA);
                    if (ParentBone == null && nodeA is ActiveBone parentActiveBone)
                    {
                        ParentBone = parentActiveBone;
                    }
                    return;
                }
            }
        }
    }

    /// <summary>
    /// Caches every bone distal to this one (itself included) so load compensation knows which
    /// masses hang off this joint. Called once by HumanoidRagdoll after all bones are registered.
    /// </summary>
    public void BuildDistalChain(IReadOnlyList<ActiveBone> allBones)
    {
        _distalChain.Clear();
        foreach (var candidate in allBones)
        {
            ActiveBone? cursor = candidate;
            int guard = 0;
            while (cursor != null && guard++ < MaxChainDepth)
            {
                if (cursor == this)
                {
                    _distalChain.Add(candidate);
                    break;
                }
                cursor = cursor.ParentBone;
            }
        }
    }

    /// <summary>
    /// Feed-forward torque that holds this joint against gravity, replacing the former
    /// 10x-gain / 100x-inertia hack with actual physics.
    ///
    /// Open chain: the joint must react the weight of everything distal to it,
    /// tau = -sum_i (r_i x m_i g), with r_i measured from the joint pivot.
    ///
    /// Closed chain (a distal bone is touching the world - hand planted, foot planted): the limb
    /// is instead a support strut carrying a share of total body weight up from the contact.
    /// That share is applied at the contact point as a Jacobian-transpose virtual force,
    /// tau = r_contact x F_support, which is what lets light arms actually push an 80 kg torso up.
    /// </summary>
    private Vector3 ComputeLoadCompensationTorque(float supportedMassShare)
    {
        if (_distalChain.Count == 0 || LoadCompensationScale <= 0.0f)
        {
            return Vector3.Zero;
        }

        Vector3 pivot = GetJointPivot();
        Vector3 gravityTorque = Vector3.Zero;
        Vector3 lowestContact = Vector3.Zero;
        bool hasContact = false;
        float contactY = float.MaxValue;

        foreach (var bone in _distalChain)
        {
            if (!IsInstanceValid(bone)) continue;

            Vector3 r = bone.GlobalPosition - pivot;
            Vector3 weight = new Vector3(0.0f, -GravityMagnitude * bone.GravityScale * bone.Mass, 0.0f);
            gravityTorque += r.Cross(weight);

            if (bone.IsInContactWithWorld() && bone.GlobalPosition.Y < contactY)
            {
                contactY = bone.GlobalPosition.Y;
                lowestContact = bone.GlobalPosition;
                hasContact = true;
            }
        }

        // Open-chain term: hold up the distal limb's own weight.
        Vector3 compensation = -gravityTorque;

        // Closed-chain term: this limb is planted, so it also carries body weight from the contact.
        if (hasContact && supportedMassShare > 0.0f)
        {
            // The limb presses DOWN into the ground; the equal-and-opposite normal force is what
            // lifts the body. tau = J^T F with F applied at the contact (J^T reduces to r x F here).
            Vector3 supportForce = new Vector3(0.0f, -GravityMagnitude * supportedMassShare, 0.0f);
            Vector3 rContact = lowestContact - pivot;
            compensation += rContact.Cross(supportForce);
        }

        return compensation * LoadCompensationScale;
    }

    private Vector3 GetJointPivot()
    {
        if (_joint != null && IsInstanceValid(_joint))
        {
            return _joint.GlobalPosition;
        }
        // Fallback: approximate the pivot as the midpoint between this bone and its parent.
        if (ParentBone != null && IsInstanceValid(ParentBone))
        {
            return (GlobalPosition + ParentBone.GlobalPosition) * 0.5f;
        }
        return GlobalPosition;
    }
}

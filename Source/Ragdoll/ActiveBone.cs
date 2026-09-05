using System.Collections.Generic;
using Godot;
using Physics4Fun.Core.Math;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Biomechanical active bone with impedance-controlled 3D rotational actuation.
/// Uses pure PD control (Ki = 0) to eliminate contact limit-cycle chatter and shaking.
/// Enforces Newton's Third Law by applying reaction torques to the parent bone.
/// </summary>
public partial class ActiveBone : RigidBody3D, Interfaces.IBoneState
{
    /// <summary>
    /// <see cref="Interfaces.IBoneState.IsValid"/>. Every other member of that contract is already
    /// satisfied by RigidBody3D/Node3D (GlobalPosition, GlobalTransform, LinearVelocity,
    /// AngularVelocity, Mass) or by this class (BoneName, IsInContactWithWorld), so implementing it
    /// costs exactly this one property.
    /// </summary>
    public bool IsValid => IsInstanceValid(this);

    [Export] public string BoneName { get; set; } = string.Empty;
    [Export] public ActiveBone? ParentBone { get; set; }

    [ExportGroup("Biomechanical Actuator Gains")]
    [Export] public float ProportionalGain { get; set; } = 450.0f;
    [Export] public float DerivativeGain { get; set; } = 40.0f;
    [Export] public float IntegralGain { get; set; } = 0.0f; // Pure PD control prevents contact windup & jitter
    /// <summary>
    /// Torque ceiling (N.m) for a bone that does not override it.
    ///
    /// 300 rather than the 1200 this used to be, matching the 1.5x-human-peak rule the rig's
    /// authored ceilings now follow. Every actuated bone in ActiveRagdoll.tscn sets its own value,
    /// so the only bone left on the default is the pelvis - which is the root, has no ParentBone,
    /// and therefore never actuates at all (its telemetry TorqueMag is 0.000 for 100% of ticks).
    /// The old default was inert but it was a loaded gun: any bone added without an explicit
    /// ceiling would have inherited roughly six times a human hip's peak torque.
    /// </summary>
    [Export] public float MaxTorque { get; set; } = 300.0f;

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

    /// <summary>
    /// Largest share of <see cref="MaxTorque"/> the gravity feed-forward may claim, leaving the
    /// rest for the PD term.
    ///
    /// Without this the feed-forward was unbounded and the only clamp was on the SUM, which
    /// preserves the feed-forward's direction and discards the PD contribution whenever load
    /// compensation alone exceeds the ceiling. Measured on a 10 s RL arena dump, that happened on
    /// 23.3% of ticks at the spine and 20.0% at the chest: one tick in five, the torso stopped
    /// tracking its target and became a pure gravity strut. Because the feed-forward grows with the
    /// lever arm between the joint and its load, it peaks exactly when the pose is worst - so the
    /// actuator lost authority precisely when it was needed, which is a positive feedback loop into
    /// the fall it was trying to arrest.
    ///
    /// 0.5 matches the split PidController3D already applies to its own D-term for the same reason
    /// ("so the P term always retains authority"). Median demand is around 30 N.m against ceilings
    /// in the hundreds, so this bounds the tail without touching normal operation.
    /// </summary>
    [Export] public float LoadCompensationTorqueFraction { get; set; } = 0.5f;

    /// <summary>
    /// Joint angular speed (rad/s) at which this actuator can no longer produce ANY torque in the
    /// direction it is already turning - the Hill force-velocity limit.
    ///
    /// A torque ceiling bounds force; it does not bound POWER, and power is what was wrong here.
    /// Nothing stopped a joint from holding its full ceiling while spinning at Jolt's 47.12 rad/s
    /// angular-velocity clamp, and 400 N.m x 47.12 rad/s is 18.8 kW. Measured on two arena dumps:
    /// 24.0 kW at one thigh and 71 kW across the body, against roughly 2.6 kW for a world-class
    /// sprinter at full output. That is the "shotgun" - a ball carrying 0.048 J triggers a joint
    /// that then delivers 200 J in a single tick.
    ///
    /// Real muscle cannot do this: force falls with shortening velocity and reaches zero at the
    /// maximum shortening velocity. Modelling that caps peak mechanical power at tau0*wMax/4
    /// automatically, so at 15 rad/s every joint on this rig lands at human power (thigh 1.5 kW,
    /// knee 1.5 kW, ankle 0.94 kW) without any ceiling changing.
    ///
    /// 15 rad/s is the low end of human peak joint angular velocity (15-20 rad/s in a fast kick).
    /// Raising it raises peak power linearly; setting it to 0 disables the limit entirely.
    ///
    /// Crucially this only applies to torque that DRIVES the joint. Braking a limb that is already
    /// moving keeps full authority - see ComputeForceVelocityScale - because that is exactly when
    /// the actuator needs it, and because eccentric muscle is stronger than isometric, not weaker.
    /// </summary>
    [Export] public float MaxShorteningVelocity { get; set; } = 15.0f;

    /// <summary>
    /// EMA smoothing on the joint velocity that feeds the Hill force-velocity law, in (0,1].
    /// 1 uses the raw per-tick velocity, which is the original behaviour.
    ///
    /// <para><b>The Hill law should see the muscle's shortening rate, not solver noise.</b> A joint
    /// whose axis is tightly limited chatters against its own constraint - measured at 15 to 68
    /// rad/s on twist axes with the body still upright, against 7.5 as the peak across all 45 DOF
    /// in Isaac. That is not motion the muscle is producing, but it reads as maximal shortening, so
    /// the derating drives the ceiling to zero and `ActiveBone` then applies EXACTLY no torque. The
    /// muscle switches off because of a numerical artifact, the body goes limp, and it topples -
    /// measured as `fvScale` reaching 0.00 by t=1 s while the actuators were being asked for 2.23x
    /// their ceiling.</para>
    ///
    /// <para>Filtering is the honest fix rather than raising
    /// <see cref="MaxShorteningVelocity"/>: the limit is real biomechanics and worth keeping, it
    /// simply needs a velocity signal that reflects the limb rather than the solver. The D term
    /// already gets this treatment inside <see cref="PidController3D"/>.</para>
    /// </summary>
    [Export] public float HillVelocityFilterAlpha { get; set; } = 1.0f;

    /// <summary>Smoothed relative joint velocity feeding the Hill law. See HillVelocityFilterAlpha.</summary>
    private Vector3 _hillFilteredVelocity = Vector3.Zero;

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

    /// <summary>
    /// An additional target offset owned exclusively by an external policy driver. Identity by
    /// default, so every existing code path is unaffected.
    ///
    /// <para>Separate from <see cref="FeedForwardTargetOffset"/> because that one is NOT free: the
    /// balance reflexes write it every tick (AnkleBalanceModule, HipStrategyModule,
    /// DynamicSteppingModule, ArmReflexModule), as does HumanoidRagdoll when it clears them. So does
    /// the trajectory layer with <see cref="TargetLocalRotation"/>. An outside writer targeting
    /// either is overwritten within the same frame - measured as a policy whose output changed the
    /// resulting motion by nothing at all, identical to three decimals against a control that
    /// ignored the policy entirely.</para>
    ///
    /// <para>This exists so an Isaac-trained policy can act as a CORRECTION on top of the
    /// procedural controller rather than replacing it. Godot's rest pose is not an equilibrium once
    /// the balance layer is switched off - the body is on the floor in under 2 seconds - while
    /// Isaac's is, so a policy trained there has no balance behaviour of its own to fall back on.
    /// See IsaacPolicyDriver.AssistMode.</para>
    /// </summary>
    public Quaternion PolicyTargetOffset { get; set; } = Quaternion.Identity;
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

    /// <summary>
    /// Hill force-velocity scale applied to the torque ceiling on the last tick, in [0,1].
    ///
    /// 1 means the joint was still or being braked and had full authority; 0 means it was already
    /// turning at MaxShorteningVelocity in the direction the actuator was pushing and was denied
    /// any further drive. Recorded so a run can be read for whether the limit is engaging at all,
    /// rather than the limit being a change nobody can see the effect of.
    /// </summary>
    public float LastForceVelocityScale { get; private set; } = 1.0f;

    public float LastTrackingErrorDeg { get; private set; } = 0.0f;

    [Signal]
    public delegate void HitReceivedEventHandler(ActiveBone bone, Vector3 hitPoint, Vector3 impulse);
    private PidController3D _pid = null!;
    private Transform3D _initialTransform;
    private Quaternion _restLocalRotation = Quaternion.Identity;
    private Quaternion _smoothedFeedForwardOffset = Quaternion.Identity;
    private Generic6DofJoint3D? _joint;

    /// <summary>
    /// The Generic6DofJoint3D connecting this bone to its parent, once resolved. Exposed so a
    /// controller can drive the joint's own motors - which Jolt solves INSIDE the constraint solve -
    /// instead of adding external torque. See IsaacPolicyDriver.JointMotorDrive.
    /// </summary>
    public Generic6DofJoint3D? Joint => _joint;
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
        Quaternion targetGlobalRot =
            (parentGlobalRot * TargetLocalRotation * _smoothedFeedForwardOffset * PolicyTargetOffset).Normalized();
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
        float maxTorque = MaxTorque * MuscleStrength;

        // Record the DEMAND rather than what survives the bound below, so telemetry keeps showing
        // how far the feed-forward overshoots. Recording the clamped value instead would hide the
        // overflow that motivated LoadCompensationTorqueFraction in the first place.
        LastPdTorque = torque;
        LastLoadCompensationTorque = loadCompensation;

        // Bound the feed-forward BEFORE summing. Clamping only the sum lets a large feed-forward
        // crowd the PD term out of the budget entirely - see LoadCompensationTorqueFraction.
        float maxLoadCompensation = maxTorque * LoadCompensationTorqueFraction;
        if (maxLoadCompensation > 0.0f
            && loadCompensation.LengthSquared() > maxLoadCompensation * maxLoadCompensation)
        {
            loadCompensation = loadCompensation.Normalized() * maxLoadCompensation;
        }

        Vector3 totalTorque = torque + loadCompensation;

        // Hill force-velocity limit, applied here rather than to _pid.MaxTorque because the scale
        // depends on the direction of the torque actually being commanded, which is not known until
        // the PD and feed-forward terms have been summed.
        float hillAlpha = Mathf.Clamp(HillVelocityFilterAlpha, 0.001f, 1.0f);
        _hillFilteredVelocity += (relativeAngVel - _hillFilteredVelocity) * hillAlpha;
        LastForceVelocityScale =
            ComputeForceVelocityScale(totalTorque, _hillFilteredVelocity, MaxShorteningVelocity);
        float velocityLimitedMax = maxTorque * LastForceVelocityScale;

        // Clamp the combined output so the pair together cannot exceed what the actuator can
        // deliver at this joint's current speed.
        if (velocityLimitedMax > 0.0f && totalTorque.LengthSquared() > velocityLimitedMax * velocityLimitedMax)
        {
            totalTorque = totalTorque.Normalized() * velocityLimitedMax;
        }
        else if (velocityLimitedMax <= 0.0f)
        {
            totalTorque = Vector3.Zero;
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
    /// Uses SWING-TWIST decomposition, not Euler angles, and that is the whole correctness story.
    ///
    /// This previously read the commanded offset through <c>Basis.GetEuler()</c>, which Godot
    /// decomposes in YXZ order - a parameterisation whose principal branch can only represent
    /// |x| &lt;= pi/2. Four axes on this rig exceed that: Thigh x(+2.10), Shin x(-2.60),
    /// Forearm x(+2.60), UpperArm x(+3.00). For those, GetEuler() returns a DIFFERENT triple than
    /// the one commanded, and the comparison ran against the wrong numbers. Measured: a commanded
    /// (2.6, 0.05, 0.05) came back as (0.54, -3.09, -3.09) and reported a violation that did not
    /// occur.
    ///
    /// That made the JointLimitViolations telemetry column actively misleading rather than merely
    /// absent, because it false-positived on precisely the four joints a get-up depends on - and
    /// the column's stated meaning is "an actuator is pinned against a hard stop achieving
    /// nothing", which is exactly the wrong conclusion to hand someone debugging a failed rise.
    ///
    /// The twist angle about an axis has no such branch problem: it is recovered directly from the
    /// quaternion and is single-valued across the full (-pi, pi], which covers every limit this rig
    /// declares. See <see cref="TwistAngleAbout"/>.
    ///
    /// Still an approximation, and worth being precise about which one. A Generic6DofJoint3D
    /// constrains three axes jointly, so per-axis twist is exact only when the commanded rotation
    /// is dominated by one axis. That is the case for the joints this reports on - shin and forearm
    /// are hinges with +/-0.10 rad of slop on their other two axes - and for a coupled rotation it
    /// is a close bound rather than a guarantee. Diagnostic-only either way: the RL action path
    /// does not depend on it, since JointLimitedActionSpace scales into the limits by construction.
    /// </remarks>
    public bool IsTargetWithinJointLimits()
    {
        if (_joint == null || !IsInstanceValid(_joint))
        {
            return true;
        }

        // The joint's limits are expressed about its rest frame, which is what _restLocalRotation
        // captures, so compare the commanded offset from rest rather than the absolute local pose.
        Quaternion commanded = (_restLocalRotation.Inverse() * TargetLocalRotation).Normalized();

        // X is the primary axis on every joint of this rig: the four wide ranges are all about X
        // (Thigh +2.10, Shin -2.60, Forearm +2.60, UpperArm +3.00) while Y and Z carry the small
        // off-axis budget. So twist about X is the meaningful per-axis angle, and the swing
        // residual is the off-axis deviation.
        DecomposeSwingTwist(commanded, Vector3.Right, out float twistX, out float swingAngle);

        if (!IsAxisWithinLimit(twistX, "angular_limit_x"))
        {
            return false;
        }

        // The swing residual is checked against a CONE, not per-axis, and the reason is that after
        // removing the twist there is no basis-aligned Y/Z pair left to compare - the residual is a
        // single rotation about some axis in the YZ plane. Bounding it by the widest of the four
        // Y/Z limits is therefore a necessary condition rather than an exact test: it will not
        // false-positive (the property that made the old Euler version harmful), but a rotation
        // that lands in the narrow direction of an asymmetric envelope can slip through.
        //
        // Verified against the rig's real limits for the four wide-range joints: Shin (-2.50,0,0)
        // -> swing 0.0000 of 0.10; Forearm (2.60,0.05,0.05) -> 0.0492 of 0.15; Thigh
        // (2.00,0.20,0.30) -> 0.1438 of 0.80; UpperArm (3.00,0.30,0.40) -> 0.4598 of 2.50.
        float swingBudget = MaxOffAxisLimit();
        return swingAngle <= swingBudget + JointLimitTolerance;
    }

    /// <summary>
    /// Splits <paramref name="q"/> into a rotation about <paramref name="axis"/> (the twist) and
    /// whatever remains (the swing), reporting the twist as a signed angle and the swing as an
    /// unsigned magnitude.
    ///
    /// This replaced a <c>Basis.GetEuler()</c> round-trip, and the difference is not cosmetic.
    /// Godot decomposes YXZ, whose principal branch only reaches |x| &lt;= pi/2, so a commanded
    /// (2.6, 0.05, 0.05) came back as (0.54, -3.09, -3.09) - the same ROTATION expressed as a
    /// different triple, which then compared against the per-axis limits as a violation. Twist is
    /// recovered straight from the quaternion and is single-valued on (-pi, pi], covering every
    /// limit this rig declares.
    ///
    /// Sign is canonicalised (w &gt;= 0) first: q and -q are the same rotation but yield twist
    /// angles differing by 2*pi, which compare against the limits differently.
    /// </summary>
    internal static void DecomposeSwingTwist(Quaternion q, Vector3 axis, out float twistAngle, out float swingAngle)
    {
        if (q.W < 0.0f)
        {
            q = new Quaternion(-q.X, -q.Y, -q.Z, -q.W);
        }

        Vector3 projected = axis * new Vector3(q.X, q.Y, q.Z).Dot(axis);
        var twist = new Quaternion(projected.X, projected.Y, projected.Z, q.W);

        // Degenerate at a 180-degree swing, where the twist component vanishes and its axis is
        // undefined. Reporting zero twist and the full rotation as swing is the honest reading:
        // the pose is nowhere near any of this rig's envelopes, so it fails the cone test anyway.
        if (twist.LengthSquared() < 1e-9f)
        {
            twistAngle = 0.0f;
            swingAngle = 2.0f * Mathf.Atan2(new Vector3(q.X, q.Y, q.Z).Length(), q.W);
            return;
        }

        twist = twist.Normalized();
        twistAngle = 2.0f * Mathf.Atan2(projected.Dot(axis), twist.W);

        Quaternion swing = q * twist.Inverse();
        if (swing.W < 0.0f)
        {
            swing = new Quaternion(-swing.X, -swing.Y, -swing.Z, -swing.W);
        }
        swingAngle = 2.0f * Mathf.Atan2(new Vector3(swing.X, swing.Y, swing.Z).Length(), swing.W);
    }

    /// <summary>Widest enabled Y/Z limit magnitude - the cone budget for the swing residual.</summary>
    private float MaxOffAxisLimit()
    {
        float budget = 0.0f;
        foreach (string axisPrefix in new[] { "angular_limit_y", "angular_limit_z" })
        {
            if (_joint == null || !(bool)_joint.Get($"{axisPrefix}/enabled"))
            {
                return Mathf.Pi;
            }

            budget = Mathf.Max(budget, Mathf.Abs((float)_joint.Get($"{axisPrefix}/lower_angle")));
            budget = Mathf.Max(budget, Mathf.Abs((float)_joint.Get($"{axisPrefix}/upper_angle")));
        }

        return budget;
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
        PolicyTargetOffset = Quaternion.Identity;
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
        PolicyTargetOffset = Quaternion.Identity;
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

    /// <summary>
    /// Hill-type force-velocity scale on the torque ceiling, in [0,1].
    ///
    /// Only the component of joint velocity ALONG the commanded torque counts as shortening. A
    /// joint being braked - velocity opposing the torque - is an eccentric contraction, where real
    /// muscle is stronger than isometric, so it keeps full authority here rather than being
    /// penalised. That asymmetry is what makes this safe for balance: arresting a limb is braking,
    /// and arresting a limb is most of what standing up consists of. What it removes is the ability
    /// to keep driving a joint that is already spinning, which is the term that was injecting
    /// kilowatts.
    ///
    /// Linear falloff rather than Hill's hyperbola. The hyperbola is the better fit to real muscle,
    /// but the quantity being bounded here is peak power, and both forms cap it at the same order;
    /// the linear form has no extra parameter to tune and cannot go negative.
    /// </summary>
    /// <param name="maxShorteningVelocity">
    /// Passed rather than read from <see cref="MaxShorteningVelocity"/> so this is a pure function
    /// of its arguments. The Hill law is the one piece of actuator maths with no dependence on the
    /// scene tree, and taking the gain as a parameter is what lets it be exercised - or simply
    /// reasoned about - without constructing a RigidBody3D. Same argument the architecture notes
    /// make for extracting the actuator wholesale; this is the part of it that costs nothing.
    /// </param>
    internal static float ComputeForceVelocityScale(
        Vector3 torque, Vector3 relativeAngularVelocity, float maxShorteningVelocity)
    {
        if (maxShorteningVelocity <= 0.0f)
        {
            return 1.0f;
        }

        float torqueLengthSquared = torque.LengthSquared();
        if (torqueLengthSquared < 1e-12f)
        {
            return 1.0f;
        }

        float shortening = relativeAngularVelocity.Dot(torque) / Mathf.Sqrt(torqueLengthSquared);
        if (shortening <= 0.0f)
        {
            return 1.0f;
        }

        return Mathf.Max(0.0f, 1.0f - (shortening / maxShorteningVelocity));
    }

    /// <summary>
    /// World position of this bone's joint anchor - the point the gravity feed-forward takes its
    /// lever arms about. Exposed read-only so the Isaac side can reproduce the SAME pivot instead of
    /// approximating it as the bone/parent midpoint, which would misplace every lever arm and so
    /// misestimate a feed-forward that carries 45-68% of the body's holding torque.
    /// </summary>
    public Vector3 JointPivotWorld => GetJointPivot();

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

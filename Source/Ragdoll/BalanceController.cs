using System;
using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll.Interfaces;
using Physics4Fun.Ragdoll.Modules;
using Physics4Fun.Ragdoll.Modules.Reflexes;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// High-level coordinator for closed-loop biomechanical balance, stepping, and autonomous reflexes (Euphoria DMS / SOLID).
/// Composes dedicated, single-responsibility modules and exposes read-only telemetry.
/// </summary>
public partial class BalanceController : Node, IBalanceTelemetryProvider
{
    [ExportGroup("Dynamic Stepping (Capture Point)")]
    [Export] public bool EnableDynamicStepping { get => _stepping.EnableDynamicStepping; set => _stepping.EnableDynamicStepping = value; }
    [Export] public float StepDuration { get => _stepping.StepDuration; set => _stepping.StepDuration = value; }
    [Export] public float StepHeight { get => _stepping.StepHeight; set => _stepping.StepHeight = value; }
    [Export] public float StanceWidth { get => _stepping.StanceWidth; set => _stepping.StanceWidth = value; }

    [ExportGroup("Gaze & Vestibular Reflex (VOR Horizon Leveling)")]
    [Export] public bool EnableVestibularGaze { get => _gazeReflex.IsEnabled; set => _gazeReflex.IsEnabled = value; }
    [Export] public float HeadHorizonGain { get => _gazeReflex.HeadHorizonGain; set => _gazeReflex.HeadHorizonGain = value; }
    [Export] public float HeadLookAheadGain { get => _gazeReflex.HeadLookAheadGain; set => _gazeReflex.HeadLookAheadGain = value; }

    [ExportGroup("Arm Reflexes & Protective Falling")]
    [Export] public bool EnableArmReflexes { get => _armReflex.IsEnabled; set => _armReflex.IsEnabled = value; }
    [Export] public float ArmCounterBalanceGain { get => _armReflex.ArmCounterBalanceGain; set => _armReflex.ArmCounterBalanceGain = value; }
    [Export] public float ImpactBraceDistance { get => _armReflex.ImpactBraceDistance; set => _armReflex.ImpactBraceDistance = value; }

    [ExportGroup("Ankle Ground Reaction Strategy")]
    [Export] public float AnklePitchGain { get => _ankleBalance.AnklePitchGain; set => _ankleBalance.AnklePitchGain = value; }
    [Export] public float AnklePitchDamping { get => _ankleBalance.AnklePitchDamping; set => _ankleBalance.AnklePitchDamping = value; }
    [Export] public float AnkleRollGain { get => _ankleBalance.AnkleRollGain; set => _ankleBalance.AnkleRollGain = value; }
    [Export] public float AnkleRollDamping { get => _ankleBalance.AnkleRollDamping; set => _ankleBalance.AnkleRollDamping = value; }

    [ExportGroup("Pelvis Stabilization (Protected Balance Region)")]
    [Export] public float PelvisStabilizerGain { get; set; } = 600.0f;
    [Export] public float PelvisStabilizerDamping { get; set; } = 20.0f;
    [Export] public float PelvisStabilizerMaxTorque { get; set; } = 300.0f;

    [ExportGroup("Biomechanical Thresholds")]
    [Export] public float DecoupleVelocityThreshold { get; set; } = 3.0f;
    [Export] public float MaxStumbleTiltAngleDeg { get; set; } = 30.0f;
    [Export] public float KnockoutTiltAngleDeg { get; set; } = 80.0f;
    [Export] public float AutoRecoveryDelay { get; set; } = 0.8f;
    [Export] public float RecoveryDuration { get; set; } = 2.4f;

    [ExportGroup("Configuration")]
    [Export] public Godot.Collections.Dictionary<int, float> StateBalanceStrengthMap { get; set; } = new()
    {
        { (int)RagdollState.Balanced, 1.0f },
        { (int)RagdollState.Stumbling, 0.85f },
        { (int)RagdollState.Flailing, 0.0f },
        { (int)RagdollState.KnockedOut, 0.0f },
        { (int)RagdollState.Recovering, 0.4f }
    };

    // Subsystem Modules (SRP)
    private readonly DynamicSteppingModule _stepping = new();
    private readonly WeightTransferModule _weightTransfer = new();
    private readonly AnkleBalanceModule _ankleBalance = new();
    private readonly HipStrategyModule _hipStrategy = new();
    private readonly ArmReflexModule _armReflex = new();
    private readonly VestibularGazeModule _gazeReflex = new();
    private readonly HitReactionReflexModule _hitReaction = new();
    private readonly ObstacleBracingReflexModule _obstacleBracing = new();
    private readonly List<IBiomechanicalReflex> _reflexPipeline = new();

    // Live Telemetry Provider (ISP)
    public Vector3 CenterOfMass { get; private set; } = Vector3.Zero;
    public float TotalMass { get; private set; } = 0.0f;
    public RagdollOrientation CurrentOrientation { get; private set; } = RagdollOrientation.Upright;
    public float RecoveryProgressNormalized { get; private set; } = 0.0f;
    public float CurrentTiltAngleDeg { get; private set; } = 0.0f;
    public float LastSuspensionForce { get; private set; } = 0.0f;
    public float BalanceStrengthNow { get; private set; } = 0.0f;
    public float IcpEscapeDistance { get; private set; } = 0.0f;
    public Vector3 PelvisStabilizerTorque { get; private set; } = Vector3.Zero;
    public Vector3 CenterOfMassVelocity { get; private set; } = Vector3.Zero;

    public float StateTimerValue
    {
        get
        {
            if (_settleGraceTimer > 0.0f)
            {
                return _settleGraceTimer;
            }
            return _lastEvaluatedState switch
            {
                RagdollState.Stumbling => Mathf.Max(0.0f, _stumbleTimer),
                RagdollState.KnockedOut => Mathf.Max(0.0f, AutoRecoveryDelay - _groundRestTimer),
                RagdollState.Recovering => Mathf.Max(0.0f, _recoveryTimer),
                _ => 0.0f
            };
        }
    }

    public float SwingProgressNormalized =>
        CurrentStepPhase == StepPhase.DoubleSupport ? 0.0f : Mathf.Clamp(_stepping.StepProgress, 0.0f, 1.0f);
    public bool IsGroundedL { get; private set; } = false;
    public bool IsGroundedR { get; private set; } = false;
    public StepPhase CurrentStepPhase => _stepping.CurrentStepPhase;
    public float CurrentWeightShareL => _weightTransfer.CurrentWeightShareL;
    public float CurrentWeightShareR => _weightTransfer.CurrentWeightShareR;
    public Vector3 GroundPointL => _groundPointL;
    public Vector3 GroundPointR => _groundPointR;
    public ActiveBone? FootL => _footL;
    public ActiveBone? FootR => _footR;
    public ActiveBone? ThighL => _thighL;
    public ActiveBone? ThighR => _thighR;
    public ActiveBone? ShinL => _shinL;
    public ActiveBone? ShinR => _shinR;

    // Bone Actuator References
    private ActiveBone _pelvis = null!;
    private ActiveBone? _spine;
    private ActiveBone? _chest;
    private ActiveBone? _head;
    private ActiveBone? _upperArmL;
    private ActiveBone? _upperArmR;
    private ActiveBone? _forearmL;
    private ActiveBone? _forearmR;
    private ActiveBone? _thighL;
    private ActiveBone? _thighR;
    private ActiveBone? _shinL;
    private ActiveBone? _shinR;
    private ActiveBone? _footL;
    private ActiveBone? _footR;
    private readonly List<ActiveBone> _bones = new();
    private readonly Godot.Collections.Array<Rid> _ragdollRids = new();

    private Vector3 _groundPointL = Vector3.Zero;
    private Vector3 _groundPointR = Vector3.Zero;
    private float _stumbleTimer;
    private float _groundRestTimer;
    private float _recoveryTimer;
    private float _flailGroundTimer;
    private float _airborneTimer;
    private float _settleGraceTimer = 0.15f;
    private RagdollState _lastEvaluatedState = RagdollState.Balanced;

    // EMA smoothing factor for the pelvis stabilizer's angular velocity (per 120 Hz tick)
    private const float PelvisStabAngVelFilterAlpha = 0.25f;
    private Vector3 _pelvisStabFilteredAngVel = Vector3.Zero;

    public void Initialize(ActiveBone pelvis, ActiveBone? chest, IEnumerable<ActiveBone> allBones)
    {
        _pelvis = pelvis;
        _chest = chest;
        _bones.Clear();
        _bones.AddRange(allBones);
        _ragdollRids.Clear();

        foreach (var bone in _bones)
        {
            _ragdollRids.Add(bone.GetRid());
            switch (bone.BoneName)
            {
                case "Spine": _spine = bone; break;
                case "Foot_L": _footL = bone; break;
                case "Foot_R": _footR = bone; break;
                case "Shin_L": _shinL = bone; break;
                case "Shin_R": _shinR = bone; break;
                case "Thigh_L": _thighL = bone; break;
                case "Thigh_R": _thighR = bone; break;
                case "UpperArm_L": _upperArmL = bone; break;
                case "UpperArm_R": _upperArmR = bone; break;
                case "Forearm_L": _forearmL = bone; break;
                case "Forearm_R": _forearmR = bone; break;
                case "Head": _head = bone; break;
            }
        }

        TotalMass = 0.0f;
        foreach (var bone in _bones)
        {
            TotalMass += bone.Mass;
        }

        // Initialize Reflex Modules
        _armReflex.Initialize(_pelvis, _chest, _upperArmL, _upperArmR, _forearmL, _forearmR, _ragdollRids);
        _gazeReflex.Initialize(_pelvis, _chest, _head);
        _hitReaction.Initialize(_chest, _head, _upperArmL, _upperArmR, _forearmL, _forearmR);
        _obstacleBracing.Initialize(_chest, _upperArmL, _upperArmR, _forearmL, _forearmR, _ragdollRids);

        // Register in OCP Reflex Pipeline
        _reflexPipeline.Clear();
        _reflexPipeline.Add(_armReflex);
        _reflexPipeline.Add(_gazeReflex);
        _reflexPipeline.Add(_hitReaction);
        _reflexPipeline.Add(_obstacleBracing);
    }

    public void RegisterHit(ActiveBone hitBone, Vector3 hitPoint, Vector3 impulse)
    {
        _hitReaction.RegisterHit(hitBone, hitPoint, impulse);
        _stumbleTimer = 1.2f;
    }

    public void Reset()
    {
        _stumbleTimer = 0.0f;
        _groundRestTimer = 0.0f;
        _recoveryTimer = 0.0f;
        _flailGroundTimer = 0.0f;
        _airborneTimer = 0.0f;
        _settleGraceTimer = 0.15f;
        _lastEvaluatedState = RagdollState.Balanced;
        LastSuspensionForce = 0.0f;
        CurrentTiltAngleDeg = 0.0f;
        BalanceStrengthNow = 0.0f;
        IcpEscapeDistance = 0.0f;
        PelvisStabilizerTorque = Vector3.Zero;
        CenterOfMassVelocity = Vector3.Zero;

        _stepping.Reset();
        _weightTransfer.Reset();
        foreach (var reflex in _reflexPipeline)
        {
            reflex.Reset();
        }
    }

    public void RegisterReflex(IBiomechanicalReflex reflex)
    {
        if (!_reflexPipeline.Contains(reflex))
        {
            _reflexPipeline.Add(reflex);
        }
    }

    public RagdollState EvaluateState(RagdollState currentState, float delta)
    {
        if (_pelvis == null || !GodotObject.IsInstanceValid(_pelvis))
        {
            return currentState;
        }

        _lastEvaluatedState = currentState;
        UpdateCenterOfMass();
        ClassifyOrientation();
        UpdateGroundSensors();
        UpdateIcpEscapeDistance();

        Vector3 pelvisUp = _pelvis.GlobalTransform.Basis.Y.Normalized();
        float dot = Mathf.Clamp(pelvisUp.Dot(Vector3.Up), -1.0f, 1.0f);
        CurrentTiltAngleDeg = Mathf.RadToDeg(Mathf.Acos(dot));
        float currentHeight = _pelvis.GlobalPosition.Y;
        float speed = _pelvis.LinearVelocity.Length();

        if (!IsGroundedL && !IsGroundedR)
        {
            _airborneTimer += delta;
        }
        else
        {
            _airborneTimer = 0.0f;
        }

        if (_settleGraceTimer > 0.0f)
        {
            _settleGraceTimer -= delta;
            return RagdollState.Balanced;
        }

        switch (currentState)
        {
            case RagdollState.Balanced:
                if (CurrentTiltAngleDeg > KnockoutTiltAngleDeg || _airborneTimer > 0.6f)
                {
                    _groundRestTimer = 0.0f;
                    _flailGroundTimer = 0.0f;
                    return RagdollState.Flailing;
                }
                if (CurrentTiltAngleDeg > MaxStumbleTiltAngleDeg || speed > DecoupleVelocityThreshold)
                {
                    _stumbleTimer = 1.4f;
                    return RagdollState.Stumbling;
                }
                return RagdollState.Balanced;

            case RagdollState.Stumbling:
                _stumbleTimer -= delta;
                if (CurrentTiltAngleDeg > KnockoutTiltAngleDeg || (currentHeight < 0.25f && speed < 1.0f) || _airborneTimer > 0.6f)
                {
                    _groundRestTimer = 0.0f;
                    _flailGroundTimer = 0.0f;
                    return RagdollState.Flailing;
                }
                if (_stumbleTimer <= 0.0f && CurrentTiltAngleDeg < 30.0f && (IsGroundedL || IsGroundedR))
                {
                    return RagdollState.Balanced;
                }
                return RagdollState.Stumbling;

            case RagdollState.Flailing:
                if (currentHeight < 0.35f && speed < 2.5f)
                {
                    _flailGroundTimer += delta;
                    if (_flailGroundTimer >= 0.25f)
                    {
                        _flailGroundTimer = 0.0f;
                        _groundRestTimer = 0.0f;
                        return RagdollState.KnockedOut;
                    }
                }
                else
                {
                    _flailGroundTimer = 0.0f;
                }
                return RagdollState.Flailing;

            case RagdollState.KnockedOut:
                if (currentHeight < 0.45f && speed < 1.0f)
                {
                    _groundRestTimer += delta;
                    if (_groundRestTimer >= AutoRecoveryDelay)
                    {
                        _groundRestTimer = 0.0f;
                        _recoveryTimer = RecoveryDuration;
                        RecoveryProgressNormalized = 0.0f;
                        return RagdollState.Recovering;
                    }
                }
                else
                {
                    _groundRestTimer = 0.0f;
                }
                return RagdollState.KnockedOut;

            case RagdollState.Recovering:
                _recoveryTimer -= delta;
                RecoveryProgressNormalized = Mathf.Clamp(1.0f - (_recoveryTimer / RecoveryDuration), 0.0f, 1.0f);

                if (RecoveryProgressNormalized >= 0.85f && CurrentTiltAngleDeg < 30.0f && currentHeight > 0.60f && (IsGroundedL || IsGroundedR))
                {
                    RecoveryProgressNormalized = 1.0f;
                    return RagdollState.Balanced;
                }

                if (_recoveryTimer <= -1.5f)
                {
                    _groundRestTimer = 0.0f;
                    return RagdollState.KnockedOut;
                }
                return RagdollState.Recovering;

            default:
                return currentState;
        }
    }

    public void TriggerStumble()
    {
        _stumbleTimer = 1.4f;
    }

    public void ApplyBalanceForces(RagdollState state, float delta)
    {
        if (_pelvis == null || !GodotObject.IsInstanceValid(_pelvis))
        {
            return;
        }

        int stateKey = (int)state;
        float strength = StateBalanceStrengthMap.TryGetValue(stateKey, out float value) ? value : 1.0f;

        // 1. Autonomous Reflex Pipeline (OCP)
        foreach (var reflex in _reflexPipeline)
        {
            reflex.Update(state, CurrentStepPhase, delta);
        }

        if (strength <= 0.01f)
        {
            LastSuspensionForce = 0.0f;
            BalanceStrengthNow = 0.0f;
            PelvisStabilizerTorque = Vector3.Zero;
            _stepping.Reset();
            _weightTransfer.Reset();
            _ankleBalance.Reset();
            _hipStrategy.Reset();
            return;
        }

        Vector3 pelvisUp = _pelvis.GlobalTransform.Basis.Y.Normalized();
        float dot = Mathf.Clamp(pelvisUp.Dot(Vector3.Up), -1.0f, 1.0f);
        float tiltAngle = Mathf.RadToDeg(Mathf.Acos(dot));
        float speed = _pelvis.LinearVelocity.Length();

        if (tiltAngle > 80.0f)
        {
            LastSuspensionForce = 0.0f;
            BalanceStrengthNow = 0.0f;
            PelvisStabilizerTorque = Vector3.Zero;
            _stepping.Reset();
            _ankleBalance.Reset();
            _hipStrategy.Reset();
            return;
        }

        float tiltFade = Mathf.Clamp(1.0f - ((tiltAngle - 60.0f) / 20.0f), 0.0f, 1.0f);
        float speedFade = Mathf.Clamp(1.0f - (speed / DecoupleVelocityThreshold), 0.0f, 1.0f);
        float activeStrength = strength * tiltFade * speedFade;
        BalanceStrengthNow = activeStrength;

        // Pelvis Protected Balance Region: direct attitude stabilization of the unactuated root
        ApplyPelvisStabilization(activeStrength);

        // 2. Dynamic Stepping Module (Capture Point)
        if (state == RagdollState.Balanced || state == RagdollState.Stumbling)
        {
            _stepping.Update(
                _pelvis,
                _thighL,
                _thighR,
                _shinL,
                _shinR,
                _footL,
                _footR,
                CenterOfMass,
                IsGroundedL,
                IsGroundedR,
                _groundPointL,
                _groundPointR,
                _settleGraceTimer > 0.0f,
                activeStrength,
                delta);
        }
        else
        {
            _stepping.Reset();
        }

        // 3. Continuous Asymmetric Weight Transfer Module
        // Lateral CoM error in the yaw-level frame drives double-support weight shifting
        float lateralComError = 0.0f;
        if (_footL != null && _footR != null && GodotObject.IsInstanceValid(_footL) && GodotObject.IsInstanceValid(_footR))
        {
            Vector3 supportCenter = (_footL.GlobalPosition + _footR.GlobalPosition) * 0.5f;
            Vector3 flatForward = -_pelvis.GlobalTransform.Basis.Z;
            flatForward.Y = 0.0f;
            flatForward = flatForward.Normalized();
            Vector3 levelRight = flatForward.Cross(Vector3.Up);
            lateralComError = (CenterOfMass - supportCenter).Dot(levelRight);
        }

        _weightTransfer.Update(
            CurrentStepPhase,
            _stepping.StepProgress,
            lateralComError,
            _thighL,
            _thighR,
            _shinL,
            _shinR,
            delta);

        // 4. Pure Internal Joint Biomechanics (Zero Floating Forces)
        LastSuspensionForce = 0.0f;

        // 5. Hip Strategy (posture righting, sagittal CoM arrest, CoM height) via internal joint PD offsets
        bool hasGroundContact = IsGroundedL || IsGroundedR;
        if (activeStrength > 0.01f && hasGroundContact && CurrentStepPhase == StepPhase.DoubleSupport)
        {
            float groundY = (_groundPointL.Y + _groundPointR.Y) * 0.5f;
            _hipStrategy.Apply(
                _pelvis,
                _spine,
                _thighL,
                _thighR,
                _shinL,
                _shinR,
                _footL,
                _footR,
                CenterOfMass,
                CenterOfMassVelocity,
                groundY,
                activeStrength);
        }

        // 6. Ankle Ground Reaction Strategy (Double Support Only)
        if (CurrentStepPhase == StepPhase.DoubleSupport && (state == RagdollState.Balanced || state == RagdollState.Stumbling) && hasGroundContact)
        {
            _ankleBalance.ApplyBalance(_pelvis, _footL, _footR, _thighL, _thighR, CenterOfMass, CenterOfMassVelocity, activeStrength, delta);
        }
    }

    /// <summary>
    /// Pelvis Protected Balance Region (Euphoria DMS): the pelvis is the unactuated root, so it is
    /// stabilized directly with an attitude PD torque; the counter-torque is distributed across the
    /// grounded feet as reaction against the ground, keeping the interaction internal.
    /// </summary>
    private void ApplyPelvisStabilization(float activeStrength)
    {
        PelvisStabilizerTorque = Vector3.Zero;

        bool groundedL = IsGroundedL && _footL != null && GodotObject.IsInstanceValid(_footL);
        bool groundedR = IsGroundedR && _footR != null && GodotObject.IsInstanceValid(_footR);
        if ((!groundedL && !groundedR) || activeStrength <= 0.01f)
        {
            return;
        }

        // Axis-angle attitude error between pelvis up axis and world up (magnitude ~ sin(angle))
        Vector3 pelvisUp = _pelvis.GlobalTransform.Basis.Y.Normalized();
        Vector3 attitudeError = pelvisUp.Cross(Vector3.Up);

        // Low-pass filter the pelvis angular velocity: joint reaction chatter (~20 rad/s at 120 Hz)
        // would otherwise dominate the damping term and pump energy through the grounded feet
        _pelvisStabFilteredAngVel += (_pelvis.AngularVelocity - _pelvisStabFilteredAngVel) * PelvisStabAngVelFilterAlpha;

        Vector3 torque = (PelvisStabilizerGain * attitudeError) - (PelvisStabilizerDamping * _pelvisStabFilteredAngVel);
        float maxTorque = PelvisStabilizerMaxTorque * activeStrength;
        if (torque.LengthSquared() > maxTorque * maxTorque)
        {
            torque = torque.Normalized() * maxTorque;
        }

        PelvisStabilizerTorque = torque;
        _pelvis.ApplyTorque(torque);

        int groundedCount = (groundedL ? 1 : 0) + (groundedR ? 1 : 0);
        Vector3 reaction = -torque / groundedCount;
        if (groundedL)
        {
            _footL!.ApplyTorque(reaction);
        }
        if (groundedR)
        {
            _footR!.ApplyTorque(reaction);
        }
    }

    /// <summary>
    /// Horizontal distance from the Instantaneous Capture Point to the support center,
    /// expressed in a yaw-level pelvis frame (mirrors the stepping module's ICP computation).
    /// </summary>
    private void UpdateIcpEscapeDistance()
    {
        if (_footL == null || _footR == null || !GodotObject.IsInstanceValid(_footL) || !GodotObject.IsInstanceValid(_footR))
        {
            IcpEscapeDistance = 0.0f;
            return;
        }

        float omega0 = Mathf.Sqrt(9.81f / Mathf.Max(0.35f, _pelvis.GlobalPosition.Y));
        Vector3 comVel = _pelvis.LinearVelocity;
        Vector3 icp = CenterOfMass + (new Vector3(comVel.X, 0.0f, comVel.Z) / omega0);
        Vector3 supportCenter = (_footL.GlobalPosition + _footR.GlobalPosition) * 0.5f;

        // Yaw-level frame: flatten pelvis axes onto the ground plane
        Vector3 forward = -_pelvis.GlobalTransform.Basis.Z;
        forward.Y = 0.0f;
        forward = forward.Normalized();
        Vector3 right = forward.Cross(Vector3.Up);

        Vector3 offset = icp - supportCenter;
        float localX = offset.Dot(right);
        float localZ = offset.Dot(forward);
        IcpEscapeDistance = new Vector2(localX, localZ).Length();
    }

    private void UpdateGroundSensors()
    {
        var spaceState = _pelvis.GetWorld3D().DirectSpaceState;

        IsGroundedL = UpdateFootGroundSensor(_footL, ref _groundPointL, spaceState);
        IsGroundedR = UpdateFootGroundSensor(_footR, ref _groundPointR, spaceState);
    }

    /// <summary>
    /// Contact-driven foot grounding: grounded = real physical contact with the world (contact monitor)
    /// AND sole facing downward. The raycast is only used to query ground-point height (step targets,
    /// foot elevation), never to decide contact state.
    /// </summary>
    private bool UpdateFootGroundSensor(ActiveBone? foot, ref Vector3 groundPoint, PhysicsDirectSpaceState3D spaceState)
    {
        if (foot == null || !GodotObject.IsInstanceValid(foot))
        {
            return false;
        }

        // Ground-point height query (works up to 0.25 m below the foot; fallback to foot position)
        Vector3 rayStart = foot.GlobalPosition + new Vector3(0, 0.05f, 0);
        Vector3 rayEnd = foot.GlobalPosition - new Vector3(0, 0.25f, 0);
        var query = PhysicsRayQueryParameters3D.Create(rayStart, rayEnd, 1);
        query.Exclude = _ragdollRids;
        var result = spaceState.IntersectRay(query);
        groundPoint = result.Count > 0 ? (Vector3)result["position"] : foot.GlobalPosition;

        // Foot local -Y is the sole axis (identity-oriented at rest); require it within ~60 deg of world down
        bool soleDown = (-foot.GlobalTransform.Basis.Y).Normalized().Dot(Vector3.Down) > 0.5f;

        return soleDown && foot.IsInContactWithWorld();
    }

    private void ClassifyOrientation()
    {
        CurrentOrientation = OrientationClassifier.Classify(_pelvis.GlobalTransform.Basis);
    }

    private void UpdateCenterOfMass()
    {
        if (_bones.Count == 0 || TotalMass <= 0.0f)
        {
            return;
        }

        Vector3 weightedSum = Vector3.Zero;
        Vector3 weightedVelocitySum = Vector3.Zero;
        foreach (var bone in _bones)
        {
            if (GodotObject.IsInstanceValid(bone))
            {
                weightedSum += bone.GlobalPosition * bone.Mass;
                weightedVelocitySum += bone.LinearVelocity * bone.Mass;
            }
        }

        CenterOfMass = weightedSum / TotalMass;
        CenterOfMassVelocity = weightedVelocitySum / TotalMass;
    }
}

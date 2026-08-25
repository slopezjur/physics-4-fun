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
    [Export] public float PelvisStabilizerGain { get => _pelvisStabilization.Gain; set => _pelvisStabilization.Gain = value; }
    [Export] public float PelvisStabilizerDamping { get => _pelvisStabilization.Damping; set => _pelvisStabilization.Damping = value; }
    [Export] public float PelvisStabilizerMaxTorque { get => _pelvisStabilization.MaxTorque; set => _pelvisStabilization.MaxTorque = value; }

    [ExportGroup("Biomechanical Thresholds")]
    [Export] public float DecoupleVelocityThreshold { get => _stateMachine.DecoupleVelocityThreshold; set => _stateMachine.DecoupleVelocityThreshold = value; }
    [Export] public float MaxStumbleTiltAngleDeg { get => _stateMachine.MaxStumbleTiltAngleDeg; set => _stateMachine.MaxStumbleTiltAngleDeg = value; }
    [Export] public float KnockoutTiltAngleDeg { get => _stateMachine.KnockoutTiltAngleDeg; set => _stateMachine.KnockoutTiltAngleDeg = value; }
    [Export] public float AutoRecoveryDelay { get => _stateMachine.AutoRecoveryDelay; set => _stateMachine.AutoRecoveryDelay = value; }
    [Export] public float RecoveryDuration { get => _stateMachine.RecoveryDuration; set => _stateMachine.RecoveryDuration = value; }

    [ExportGroup("Configuration")]
    [Export] public Godot.Collections.Dictionary<int, float> StateBalanceStrengthMap { get; set; } = new()
    {
        { (int)RagdollState.Balanced, 1.0f },
        { (int)RagdollState.Stumbling, 0.85f },
        { (int)RagdollState.Flailing, 0.0f },
        { (int)RagdollState.KnockedOut, 0.0f },
        { (int)RagdollState.Recovering, 1.0f },
        { (int)RagdollState.PushUpDrill, 0.0f },
        { (int)RagdollState.ReinforcementLearning, 0.0f }
    };

    // Subsystem Modules (SRP)
    private readonly RagdollStateMachine _stateMachine = new();
    private readonly GroundContactModule _groundContact = new();

    /// <summary>
    /// Per-ragdoll, because it latches hysteresis - see <see cref="OrientationClassifier"/>. It was
    /// a static class, which silently shared one latch across every body in the process.
    /// </summary>
    private readonly OrientationClassifier _orientationClassifier = new();
    private readonly DynamicSteppingModule _stepping = new();
    private readonly WeightTransferModule _weightTransfer;
    private readonly AnkleBalanceModule _ankleBalance = new();
    private readonly HipStrategyModule _hipStrategy = new();
    private readonly PelvisStabilizationModule _pelvisStabilization = new();
    private readonly ArmReflexModule _armReflex = new();
    private readonly VestibularGazeModule _gazeReflex = new();
    private readonly HitReactionReflexModule _hitReaction = new();
    private readonly ObstacleBracingReflexModule _obstacleBracing = new();
    private readonly List<IBiomechanicalReflex> _reflexPipeline = new();
    private readonly List<IBalanceStrategy> _balancePipeline = new();

    // Live Telemetry Provider (ISP)
    public Vector3 CenterOfMass { get; private set; } = Vector3.Zero;
    public float TotalMass { get; private set; } = 0.0f;
    public RagdollOrientation CurrentOrientation { get; private set; } = RagdollOrientation.Upright;
    public float RecoveryProgressNormalized => _stateMachine.RecoveryProgressNormalized;
    public float CurrentTiltAngleDeg { get; private set; } = 0.0f;
    public float BalanceStrengthNow { get; private set; } = 0.0f;
    public float IcpEscapeDistance { get; private set; } = 0.0f;

    /// <summary>
    /// Whether <see cref="IcpEscapeDistance"/> was computable this tick - i.e. at least one foot is
    /// bearing load, so there is a base of support to measure escape FROM.
    ///
    /// Exists because the distance alone is ambiguous at 0.0: that is both "the capture point sits
    /// exactly over the support centre" and "there was nothing to compare it against". Consumers
    /// that treat the two alike will read an airborne body as perfectly balanced, which is why
    /// UprightTermination previously had to demand BOTH feet grounded before trusting the number -
    /// a requirement that also outlawed the protective step it was trying to measure.
    /// </summary>
    public bool IsIcpValid { get; private set; }
    public Vector3 PelvisStabilizerTorque => _pelvisStabilization.LastTorque;
    public Vector3 CenterOfMassVelocity { get; private set; } = Vector3.Zero;

    public float StateTimerValue => _stateMachine.StateTimerValue;

    public float SwingProgressNormalized =>
        CurrentStepPhase == StepPhase.DoubleSupport ? 0.0f : Mathf.Clamp(_stepping.StepProgress, 0.0f, 1.0f);
    public bool IsGroundedL => _groundContact.IsGroundedL;
    public bool IsGroundedR => _groundContact.IsGroundedR;
    public StepPhase CurrentStepPhase => _stepping.CurrentStepPhase;
    public float CurrentWeightShareL => _weightTransfer.CurrentWeightShareL;
    public float CurrentWeightShareR => _weightTransfer.CurrentWeightShareR;
    public Vector3 GroundPointL => _groundContact.GroundPointL;
    public Vector3 GroundPointR => _groundContact.GroundPointR;
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
    private ActiveBone? _handL;
    private ActiveBone? _handR;
    private ActiveBone? _thighL;
    private ActiveBone? _thighR;
    private ActiveBone? _shinL;
    private ActiveBone? _shinR;
    private ActiveBone? _footL;
    private ActiveBone? _footR;
    private readonly List<ActiveBone> _bones = new();
    private readonly Godot.Collections.Array<Rid> _ragdollRids = new();

    public BalanceController()
    {
        _weightTransfer = new WeightTransferModule(_stepping);
    }

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
                case "Hand_L": _handL = bone; break;
                case "Hand_R": _handR = bone; break;
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

        // Register in OCP Balance Strategy Pipeline. Order preserves the exact execution
        // sequence ApplyBalanceForces used before this pipeline existed: pelvis stabilization ->
        // stepping -> weight transfer -> hip strategy -> ankle balance.
        _balancePipeline.Clear();
        _balancePipeline.Add(_pelvisStabilization);
        _balancePipeline.Add(_stepping);
        _balancePipeline.Add(_weightTransfer);
        _balancePipeline.Add(_hipStrategy);
        _balancePipeline.Add(_ankleBalance);
    }

    public void RegisterHit(ActiveBone hitBone, Vector3 hitPoint, Vector3 impulse)
    {
        _hitReaction.RegisterHit(hitBone, hitPoint, impulse);
        _stateMachine.TriggerHitStumble();
    }

    public void Reset()
    {
        _stateMachine.Reset();
        CurrentTiltAngleDeg = 0.0f;
        BalanceStrengthNow = 0.0f;
        IcpEscapeDistance = 0.0f;
        IsIcpValid = false;
        _pelvisStabilization.ClearTorque();
        CenterOfMassVelocity = Vector3.Zero;

        // Orientation hysteresis describes continuous motion, so it must not survive a teleport.
        // Reset() is already called on every respawn path (ResetRagdoll, DropToProne,
        // TeleportToGetUpPose, StartReinforcementLearning), so folding it in here is what keeps a
        // future respawn path from silently forgetting it.
        _orientationClassifier.Reset();

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

        UpdateCenterOfMass();
        ClassifyOrientation();
        _groundContact.Update(_pelvis, _footL, _footR, _ragdollRids);
        UpdateIcpEscapeDistance();

        Vector3 pelvisUp = _pelvis.GlobalTransform.Basis.Y.Normalized();
        float dot = Mathf.Clamp(pelvisUp.Dot(Vector3.Up), -1.0f, 1.0f);
        CurrentTiltAngleDeg = Mathf.RadToDeg(Mathf.Acos(dot));
        float currentHeight = _pelvis.GlobalPosition.Y;
        float speed = _pelvis.LinearVelocity.Length();

        return _stateMachine.Evaluate(currentState, delta, CurrentTiltAngleDeg, currentHeight, speed, IsGroundedL, IsGroundedR);
    }

    public void TriggerStumble()
    {
        _stateMachine.TriggerStumble();
    }

    public void ApplyBalanceForces(RagdollState state, Recovery.GetUpPhase getUpPhase, float delta)
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
            BalanceStrengthNow = 0.0f;
            _pelvisStabilization.ClearTorque();
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

        // The tilt gate exists so balance strategies do not fight a fall in progress. It must not
        // apply while recovering: a get-up spends its whole duration far past 80 degrees, and
        // blanket-zeroing the pipeline there left the pelvis with no attitude control at all.
        // The upright-biped strategies stay off regardless - they self-gate to Balanced/Stumbling.
        bool isRecovering = state == RagdollState.Recovering;

        if (!isRecovering && tiltAngle > 80.0f)
        {
            BalanceStrengthNow = 0.0f;
            _pelvisStabilization.ClearTorque();
            _stepping.Reset();
            _ankleBalance.Reset();
            _hipStrategy.Reset();
            return;
        }

        // Tilt fade is meaningless when the body is deliberately not upright; speed fade still
        // applies, since a fast-moving body should not be receiving stabilizer torque either way.
        float tiltFade = isRecovering ? 1.0f : Mathf.Clamp(1.0f - ((tiltAngle - 60.0f) / 20.0f), 0.0f, 1.0f);
        float speedFade = Mathf.Clamp(1.0f - (speed / DecoupleVelocityThreshold), 0.0f, 1.0f);
        float activeStrength = strength * tiltFade * speedFade;
        BalanceStrengthNow = activeStrength;

        // 2. Balance Strategy Pipeline (OCP): pelvis stabilization -> stepping -> weight transfer ->
        // hip strategy -> ankle balance. Each strategy self-gates on state/step-phase/ground-contact.
        BalanceContext context = BuildBalanceContext(state, getUpPhase, activeStrength, delta);
        foreach (var strategy in _balancePipeline)
        {
            strategy.Apply(in context);
        }
    }

    private BalanceContext BuildBalanceContext(RagdollState state, Recovery.GetUpPhase getUpPhase, float activeStrength, float delta)
    {
        Vector3 icp = Vector3.Zero;
        Vector3 baseOfSupportCenter = Vector3.Zero;
        if (_footL != null && _footR != null && GodotObject.IsInstanceValid(_footL) && GodotObject.IsInstanceValid(_footR))
        {
            float groundY = (GroundPointL.Y + GroundPointR.Y) * 0.5f;
            icp = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(CenterOfMass, _pelvis.LinearVelocity, CenterOfMass.Y - groundY);
            baseOfSupportCenter = (_footL.GlobalPosition + _footR.GlobalPosition) * 0.5f;
        }

        return new BalanceContext(
            _pelvis, _spine, _chest, _head,
            _thighL, _thighR, _shinL, _shinR,
            _footL, _footR,
            _forearmL, _forearmR, _handL, _handR,
            state, getUpPhase, CurrentStepPhase, activeStrength, delta,
            CenterOfMass, CenterOfMassVelocity, icp, baseOfSupportCenter,
            IsGroundedL, IsGroundedR, GroundPointL, GroundPointR,
            _stateMachine.IsSettleGraceActive);
    }

    /// <summary>
    /// Horizontal distance from the Instantaneous Capture Point to the support center,
    /// expressed in a yaw-level pelvis frame (mirrors the stepping module's ICP computation).
    ///
    /// The support centre comes from the feet that are actually GROUNDED, not from both feet
    /// unconditionally. Averaging a foot that bears no load puts the reference point halfway
    /// toward open air, so the escape distance was wrong precisely during single support - the one
    /// phase where a protective step is happening and the answer matters most.
    ///
    /// KNOWN DEBT: BuildBalanceContext computes baseOfSupportCenter with the same both-feet
    /// average and still has this bug. It is left alone deliberately, because that value feeds
    /// DynamicSteppingModule and changing it alters tuned procedural stepping behaviour that
    /// nothing here can re-verify. The two ICP computations were deliberately aligned once before
    /// (they disagreed on CoM height); they are now knowingly divergent on support centre, and
    /// that should be closed by fixing the context rather than by reverting this.
    /// </summary>
    private void UpdateIcpEscapeDistance()
    {
        bool footLValid = _footL != null && GodotObject.IsInstanceValid(_footL);
        bool footRValid = _footR != null && GodotObject.IsInstanceValid(_footR);
        bool useL = footLValid && IsGroundedL;
        bool useR = footRValid && IsGroundedR;

        if (!useL && !useR)
        {
            // No base of support. Reported as 0.0 for continuity with existing telemetry columns,
            // but IsIcpValid is what callers must branch on - see that property.
            IcpEscapeDistance = 0.0f;
            IsIcpValid = false;
            return;
        }

        // Ground-relative CoM height, matching DynamicSteppingModule's ICP computation exactly
        // (previously this used absolute pelvis height, a latent inconsistency between the two).
        // Averaged only over grounded feet for the same reason as the support centre: a raised
        // foot's probe reports the floor beneath wherever it currently hangs.
        float groundY = useL && useR ? (GroundPointL.Y + GroundPointR.Y) * 0.5f
                      : useL ? GroundPointL.Y
                      : GroundPointR.Y;
        Vector3 icp = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(CenterOfMass, _pelvis.LinearVelocity, CenterOfMass.Y - groundY);
        Vector3 supportCenter = useL && useR ? (_footL!.GlobalPosition + _footR!.GlobalPosition) * 0.5f
                              : useL ? _footL!.GlobalPosition
                              : _footR!.GlobalPosition;

        // Yaw-level frame: flatten pelvis axes onto the ground plane
        Basis levelBasis = BiomechanicalKinematics.ComputeLevelBasis(_pelvis.GlobalTransform.Basis);
        Vector3 localOffset = levelBasis.Inverse() * (icp - supportCenter);
        IcpEscapeDistance = new Vector2(localOffset.X, localOffset.Z).Length();
        IsIcpValid = true;
    }

    private void ClassifyOrientation()
    {
        CurrentOrientation = _orientationClassifier.Classify(_pelvis.GlobalTransform.Basis);
    }

    private void UpdateCenterOfMass()
    {
        if (BiomechanicalKinematics.TryComputeCenterOfMass(_bones, TotalMass, out Vector3 com, out Vector3 comVelocity))
        {
            CenterOfMass = com;
            CenterOfMassVelocity = comVelocity;
        }
    }
}

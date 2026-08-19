using System;
using System.Collections.Generic;
using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Root coordinator for the active humanoid ragdoll.
/// Streams Dynamic Motion Synthesis (DMS) trajectories to the biomechanical bone actuators.
/// </summary>
public partial class HumanoidRagdoll : Node3D
{
    [Export] public RagdollState CurrentState { get; set; } = RagdollState.Balanced;

    [ExportGroup("Node References")]
    [Export] public ActiveBone? Pelvis { get; set; }
    [Export] public ActiveBone? Spine { get; set; }
    [Export] public ActiveBone? Chest { get; set; }
    [Export] public ActiveBone? Head { get; set; }
    [Export] public BalanceController? Balance { get; set; }

    [ExportGroup("Configuration")]
    [Export] public Godot.Collections.Dictionary<int, float> StateStiffnessMap { get; set; } = new()
    {
        { (int)RagdollState.Balanced, 1.0f },
        { (int)RagdollState.Stumbling, 0.75f },
        { (int)RagdollState.Flailing, 0.35f },
        { (int)RagdollState.KnockedOut, 0.0f },
        { (int)RagdollState.Recovering, 1.15f },
        { (int)RagdollState.PushUpDrill, 1.0f }
    };

    /// <summary>
    /// Proportional-gain multiplier applied to the arm chain while it is load-bearing (push-up
    /// drill or get-up). The authored arm gains are tuned for a relaxed hanging arm - measured at
    /// only ~14 N.m/rad of effective stiffness, 8-15x weaker than the torso - which is far too soft
    /// to control a press. Raising it here is safe in a way the old removed 10x hack was not: under
    /// correct per-axis inertia a higher Kp also raises the SPD denominator, moving the per-tick
    /// velocity factor toward 1 (more stable, not less). MaxTorque is deliberately untouched.
    /// </summary>
    [Export] public float ArmLoadBearingGain { get; set; } = 10.0f;

    private readonly List<ActiveBone> _allBones = new();
    private readonly List<ActiveBone> _plantedLimbs = new();
    private readonly Recovery.GetUpPhaseController _getUpPhases = new();
    private readonly Diagnostics.RagdollTelemetryRecorder _recorder = new();
    private float _time;
    private float _stateTime;
    private float _drillTime;

    /// <summary>
    /// Recovery trajectory locked in for the current get-up attempt.
    ///
    /// Selecting per frame from the live orientation made the plan flip mid-rise: a wobbling
    /// pelvis reclassifies Prone/Side/Supine 8-11 times per attempt, and Prone and Supine recovery
    /// command opposing poses (Supine phase 1 is "roll over to prone"). Each flip teleported every
    /// bone target, leaving the limbs with high-magnitude torque pointing in a near-random
    /// direction each tick. One attempt commits to one plan.
    /// </summary>
    private Trajectories.IPhasedRecoveryTrajectory? _activeRecoveryTrajectory;

    /// <summary>
    /// Which leg drives the current get-up. Latched with the trajectory for the same reason: a
    /// per-frame choice would oscillate, and swapping lead legs mid-rise would undo the knee drive.
    /// </summary>
    private BodySide _recoveryLeadSide = BodySide.Right;

    /// <summary>Current get-up lead leg; meaningful only while Recovering.</summary>
    public BodySide RecoveryLeadSide => _recoveryLeadSide;

    public Diagnostics.RagdollTelemetryRecorder Recorder => _recorder;

    /// <summary>Current get-up stage; Complete when not recovering.</summary>
    public Recovery.GetUpPhase CurrentGetUpPhase => _getUpPhases.CurrentPhase;

    public override void _UnhandledInput(InputEvent @event)
    {
        if (@event is InputEventKey keyEvent && keyEvent.Pressed && !keyEvent.Echo)
        {
            if (keyEvent.Keycode == Key.T)
            {
                StartTelemetryRecording();
            }
        }
    }

    public void StartTelemetryRecording(float duration = Diagnostics.RagdollTelemetryRecorder.DefaultDurationSeconds)
    {
        ResetRagdoll();
        _recorder.StartRecording(duration);
    }

    public override void _Ready()
    {
        FindAndRegisterBones(this);

        // Cache each bone's distal subtree once, so load compensation knows which masses
        // hang off every joint. Requires ParentBone to be resolved, which happens in ActiveBone._Ready.
        foreach (var bone in _allBones)
        {
            bone.BuildDistalChain(_allBones);
        }

        GD.Print($"[HumanoidRagdoll] Registered {_allBones.Count} active bones:");
        foreach (var bone in _allBones)
        {
            string parentName = bone.ParentBone != null ? bone.ParentBone.BoneName : "None (Root Pelvis)";
            GD.Print($"  - Bone '{bone.BoneName}' connected to Parent '{parentName}'");
        }

        if (Balance == null)
        {
            GD.PushError("[HumanoidRagdoll] BalanceController dependency is missing! Please assign it in the inspector.");
            return;
        }

        if (Pelvis != null)
        {
            Balance.Initialize(Pelvis, Chest, _allBones);
        }

        _recorder.RecordingFinished += ResetRagdoll;
    }

    public override void _PhysicsProcess(double delta)
    {
        float dt = (float)delta;
        _time += dt;
        _stateTime += dt;

        if (CurrentState == RagdollState.PushUpDrill)
        {
            _drillTime += dt;
        }

        if (Balance != null)
        {
            RagdollState newState = Balance.EvaluateState(CurrentState, dt);
            if (newState != CurrentState)
            {
                SetState(newState);
            }
        }


        // Advance the get-up phase machine from real contact/CoM state before targets are computed.
        if (CurrentState == RagdollState.Recovering)
        {
            _getUpPhases.Update(BuildRecoveryContext(), dt);
        }

        UpdateBoneMuscleStiffness();
        UpdateBoneTargetRotations();

        Balance?.ApplyBalanceForces(CurrentState, _getUpPhases.CurrentPhase, dt);

        // Recompute which limbs are load-bearing before torques are generated, so the
        // load-compensation feed-forward uses this tick's contact state.
        UpdateSupportLoadDistribution();

        foreach (var bone in _allBones)
        {
            if (IsInstanceValid(bone))
            {
                bone.ApplyBiomechanicalTorque(dt);
            }
        }

        if (_recorder.IsRecording && Balance != null && Pelvis != null)
        {
            _recorder.RecordFrame(
                dt, CurrentState, Balance, Pelvis, _allBones, _getUpPhases, _recoveryLeadSide, DrillCycleNormalized);
        }
    }

    public void SetState(RagdollState newState)
    {
        CurrentState = newState;
        _stateTime = 0.0f;

        if (newState == RagdollState.Stumbling)
        {
            Balance?.TriggerStumble();
        }

        // Each entry into Recovering starts a fresh get-up attempt from phase one, and commits to
        // the trajectory matching the orientation the body is actually in at that moment.
        if (newState == RagdollState.Recovering)
        {
            _getUpPhases.Reset();
            _activeRecoveryTrajectory = BiomechanicalMotionSynthesizer.GetRecoveryTrajectory(
                Balance?.CurrentOrientation ?? RagdollOrientation.Prone);
            _recoveryLeadSide = ChooseRecoveryLeadSide();
        }
        else
        {
            _activeRecoveryTrajectory = null;
        }
        GD.Print($"[HumanoidRagdoll] Transitioned to state: {newState}");
    }


    public void ApplyImpulseToChest(Vector3 impulse)
    {
        if (Chest != null && IsInstanceValid(Chest))
        {
            Chest.ApplyCentralImpulse(impulse);
            if (CurrentState == RagdollState.Balanced)
            {
                SetState(RagdollState.Stumbling);
            }
        }
    }

    public void ResetRagdoll()
    {
        _time = 0.0f;
        _stateTime = 0.0f;
        CurrentState = RagdollState.Balanced;
        _activeRecoveryTrajectory = null;
        Balance?.Reset();
        foreach (var bone in _allBones)
        {
            if (IsInstanceValid(bone))
            {
                bone.ResetBone();
            }
        }
        UpdateBoneMuscleStiffness();
        UpdateBoneTargetRotations();
        GD.Print("[HumanoidRagdoll] Ragdoll reset to initial standing state.");
    }

    public void DropToProne()
    {
        _time = 0.0f;
        _stateTime = 0.0f;
        CurrentState = RagdollState.KnockedOut;
        _activeRecoveryTrajectory = null;
        Balance?.Reset();
        
        // Rotate -90 degrees around X (pitch) and elevate slightly to avoid floor clipping
        Transform3D proneOffset = new Transform3D(new Basis(Vector3.Right, -Mathf.Pi / 2), new Vector3(0, 0.5f, 0));

        foreach (var bone in _allBones)
        {
            if (IsInstanceValid(bone))
            {
                bone.Teleport(proneOffset * bone.InitialTransform);
            }
        }
        UpdateBoneMuscleStiffness();
        UpdateBoneTargetRotations();
        GD.Print("[HumanoidRagdoll] Dropped to prone.");
    }

    /// <summary>Seconds spent settling into the bottom pose before reps begin.</summary>
    private const float DrillSetupSeconds = 1.0f;

    /// <summary>Seconds per full down-up repetition.</summary>
    private const float DrillRepSeconds = 2.4f;

    /// <summary>
    /// Starts the isolated knee push-up drill: teleports the body prone and hands the pose over to
    /// <see cref="Trajectories.PushUpDrillTrajectory"/>, which cycles reps on a clock. No phase
    /// machine, no exit criteria, no legs - a loop to watch while the arm press is developed.
    /// </summary>
    public void StartPushUpDrill()
    {
        DropToProne();
        _drillTime = 0.0f;
        CurrentState = RagdollState.PushUpDrill;
        UpdateBoneMuscleStiffness();
        UpdateBoneTargetRotations();
        GD.Print("[HumanoidRagdoll] Push-up drill started.");
    }

    /// <summary>
    /// Drill rep clock in [0,1]: 0 is the bottom of the rep, 1 is lockout. Holds at the bottom for
    /// the setup window so the body can settle onto its knees and hands before the first press,
    /// then runs a raised cosine so the reversals at both ends are smooth rather than stepped.
    /// </summary>
    public float DrillCycleNormalized
    {
        get
        {
            if (CurrentState != RagdollState.PushUpDrill || _drillTime < DrillSetupSeconds)
            {
                return 0.0f;
            }

            float phase = (_drillTime - DrillSetupSeconds) / DrillRepSeconds;
            return 0.5f * (1.0f - Mathf.Cos(phase * Mathf.Tau));
        }
    }

    public IReadOnlyList<ActiveBone> GetBones() => _allBones;

    public float CurrentMuscleStiffness
    {
        get
        {
            int stateKey = (int)CurrentState;
            return StateStiffnessMap.TryGetValue(stateKey, out float value) ? value : 1.0f;
        }
    }

    private void UpdateBoneTargetRotations()
    {
        RagdollOrientation orientation = Balance?.CurrentOrientation ?? RagdollOrientation.Upright;

        // During recovery the trajectory is driven by achieved physical state (contacts made,
        // torso lifted, CoM over the feet) rather than the state machine's clock. The timer
        // remains only as the state machine's give-up safety net.
        bool isRecovering = CurrentState == RagdollState.Recovering;
        Trajectories.IPhasedRecoveryTrajectory? recovery = isRecovering ? _activeRecoveryTrajectory : null;

        float progress;
        if (isRecovering)
        {
            progress = recovery != null
                ? _getUpPhases.ToNormalizedProgress(recovery.PhaseBoundaries)
                : (Balance?.RecoveryProgressNormalized ?? 0.0f);
        }
        else if (CurrentState == RagdollState.PushUpDrill)
        {
            // The drill runs on its own rep clock rather than any recovery progress.
            progress = DrillCycleNormalized;
        }
        else
        {
            progress = Balance?.RecoveryProgressNormalized ?? 0.0f;
        }

        foreach (var bone in _allBones)
        {
            if (string.IsNullOrEmpty(bone.BoneName))
            {
                continue;
            }

            // Evaluate the latched trajectory directly while recovering: routing through
            // ComputeTargetRotation would re-dispatch on the live orientation and reintroduce the
            // mid-attempt plan swap the latch exists to prevent.
            Quaternion targetLocal = recovery != null
                ? recovery.EvaluateBoneTarget(
                    bone.BoneName,
                    new Trajectories.RecoveryPose(_time, progress, _recoveryLeadSide))
                : BiomechanicalMotionSynthesizer.ComputeTargetRotation(
                    bone.BoneName,
                    CurrentState,
                    orientation,
                    _time,
                    progress);

            bone.TargetLocalRotation = (bone.GetRestLocalRotation() * targetLocal).Normalized();
            bone.FeedForwardTargetOffset = Quaternion.Identity;
        }
    }

    private void UpdateBoneMuscleStiffness()
    {
        int stateKey = (int)CurrentState;
        float stiffness = StateStiffnessMap.TryGetValue(stateKey, out float value) ? value : 1.0f;

        if (CurrentState == RagdollState.Recovering && Balance != null)
        {
            // Smoothly ramp up muscle stiffness from 0.60 (assisted get-up) to 1.0 (squat/stand extension).
            // NOTE: the former 10x arm/torso stiffness boost and 100x ApparentInertiaMultiplier were removed.
            // They defeated the Tan-Liu-Turk SPD denominator (which requires the bone's REAL inertia to
            // guarantee stability), producing ~1000 N-m torques on 2.5 kg limbs and exploding the solver.
            // The load-bearing problem they were trying to solve is now handled physically by the
            // gravity- and support-load compensation feed-forward in ActiveBone.
            stiffness = Mathf.Lerp(0.60f, 1.0f, Balance.RecoveryProgressNormalized);
        }

        // The arms stop being free limbs and become support struts during a press. Their authored
        // gains are tuned for a relaxed hang (~14 N.m/rad effective, against a torso at 121-201),
        // which cannot control a push-up, so scale their impedance for the phases that need it.
        bool armsLoadBearing = CurrentState == RagdollState.PushUpDrill
                               || CurrentState == RagdollState.Recovering;
        float armGain = armsLoadBearing ? Mathf.Max(1.0f, ArmLoadBearingGain) : 1.0f;

        foreach (var bone in _allBones)
        {
            bone.MuscleStrength = stiffness;
            bone.LoadBearingGainScale = IsArmBone(bone.BoneName) ? armGain : 1.0f;
        }
    }

    private static bool IsArmBone(string boneName)
    {
        return boneName.StartsWith("UpperArm", System.StringComparison.Ordinal)
            || boneName.StartsWith("Forearm", System.StringComparison.Ordinal)
            || boneName.StartsWith("Hand", System.StringComparison.Ordinal);
    }


    /// <summary>
    /// Limb end-effectors that can act as ground struts. When one is in contact, its whole limb
    /// chain carries a share of body weight and needs support-load compensation, not just
    /// gravity compensation for its own (light) distal mass.
    /// </summary>
    private static readonly string[] SupportEndEffectors = { "Forearm_L", "Forearm_R", "Foot_L", "Foot_R" };

    /// <summary>
    /// Distributes total body mass across whichever limbs are currently planted, so each support
    /// chain knows how much weight it must hold up. This is what lets light arms push an 80 kg
    /// torso off the floor without faking gains or inertia.
    /// </summary>
    private void UpdateSupportLoadDistribution()
    {
        foreach (var bone in _allBones)
        {
            bone.SupportedMassShare = 0.0f;
        }

        _plantedLimbs.Clear();
        foreach (var bone in _allBones)
        {
            if (!IsInstanceValid(bone) || System.Array.IndexOf(SupportEndEffectors, bone.BoneName) < 0)
            {
                continue;
            }
            if (bone.IsInContactWithWorld())
            {
                _plantedLimbs.Add(bone);
            }
        }

        if (_plantedLimbs.Count == 0)
        {
            return;
        }

        float totalMass = 0.0f;
        foreach (var bone in _allBones)
        {
            totalMass += bone.Mass;
        }

        float share = totalMass / _plantedLimbs.Count;

        // Assign the share up each planted limb chain, stopping at the torso: the torso is the
        // load being carried, not a strut carrying it.
        foreach (var endEffector in _plantedLimbs)
        {
            ActiveBone? cursor = endEffector;
            int guard = 0;
            while (cursor != null && guard++ < 32)
            {
                if (cursor.BoneName == "Chest" || cursor.BoneName == "Spine" || cursor.BoneName == "Pelvis")
                {
                    break;
                }
                cursor.SupportedMassShare = share;
                cursor = cursor.ParentBone;
            }
        }
    }

    /// <summary>Snapshot of the sensors the get-up phase machine reasons about.</summary>
    private Recovery.RecoveryContext BuildRecoveryContext()
    {
        ActiveBone? forearmL = FindBone("Forearm_L");
        ActiveBone? forearmR = FindBone("Forearm_R");
        ActiveBone? handL = FindBone("Hand_L");
        ActiveBone? handR = FindBone("Hand_R");
        ActiveBone? footL = Balance?.FootL;
        ActiveBone? footR = Balance?.FootR;

        float groundY = 0.0f;
        if (Balance != null)
        {
            groundY = (Balance.GroundPointL.Y + Balance.GroundPointR.Y) * 0.5f;
        }

        bool leadIsLeft = _recoveryLeadSide == BodySide.Left;

        return new Recovery.RecoveryContext(
            Pelvis!,
            Chest,
            forearmL,
            forearmR,
            handL,
            handR,
            LeadFoot: leadIsLeft ? footL : footR,
            TrailFoot: leadIsLeft ? footR : footL,
            Balance?.CenterOfMass ?? Vector3.Zero,
            groundY,
            Balance?.CurrentTiltAngleDeg ?? 0.0f);
    }

    /// <summary>
    /// Picks the leg that will drive the get-up: whichever foot is already horizontally nearer the
    /// centre of mass, since that one has the least distance to travel to get under the body.
    /// Defaults to the right when the feet are unavailable or equidistant.
    /// </summary>
    private BodySide ChooseRecoveryLeadSide()
    {
        ActiveBone? footL = Balance?.FootL;
        ActiveBone? footR = Balance?.FootR;
        if (footL == null || footR == null || !IsInstanceValid(footL) || !IsInstanceValid(footR))
        {
            return BodySide.Right;
        }

        Vector3 com = Balance?.CenterOfMass ?? Vector3.Zero;
        float distL = new Vector2(com.X - footL.GlobalPosition.X, com.Z - footL.GlobalPosition.Z).Length();
        float distR = new Vector2(com.X - footR.GlobalPosition.X, com.Z - footR.GlobalPosition.Z).Length();
        return distL < distR ? BodySide.Left : BodySide.Right;
    }

    private ActiveBone? FindBone(string boneName)
    {
        foreach (var bone in _allBones)
        {
            if (bone.BoneName == boneName)
            {
                return bone;
            }
        }
        return null;
    }
    private void FindAndRegisterBones(Node node)
    {
        foreach (var child in node.GetChildren())
        {
            if (child is ActiveBone bone)
            {
                _allBones.Add(bone);
                bone.HitReceived += OnBoneHitReceived;
            }
            FindAndRegisterBones(child);
        }
    }

    private void OnBoneHitReceived(ActiveBone bone, Vector3 hitPoint, Vector3 impulse)
    {
        if (Balance != null && IsInstanceValid(Balance))
        {
            Balance.RegisterHit(bone, hitPoint, impulse);
            if (CurrentState == RagdollState.Balanced)
            {
                SetState(RagdollState.Stumbling);
            }
        }
    }
}







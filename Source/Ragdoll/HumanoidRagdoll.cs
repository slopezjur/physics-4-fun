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

    /// <summary>
    /// Skip the procedural joint-target layer while leaving everything else running.
    ///
    /// <para>Distinct from entering <see cref="RagdollState.ReinforcementLearning"/>, which also
    /// switches off the balance strategies - they self-gate to Balanced/Stumbling, so the RL state
    /// leaves the body with no root stabilisation at all. That is fine for a policy trained to be
    /// the sole controller of a passively stable body, and wrong for this one: Godot's dummy cannot
    /// hold the rest pose without an external stabilising wrench, while Isaac's nearly can.</para>
    ///
    /// <para>Setting this in the Balanced state gives the split that actually matches: the POLICY
    /// owns the joint targets, and Godot's balance layer owns pelvis stabilisation, ankle and hip
    /// strategies. See IsaacPolicyDriver.BalanceAssist.</para>
    /// </summary>
    public bool SuppressProceduralPose { get; set; }

    [ExportGroup("Configuration")]
    [Export] public Godot.Collections.Dictionary<int, float> StateStiffnessMap { get; set; } = new()
    {
        { (int)RagdollState.Balanced, 1.0f },
        { (int)RagdollState.Stumbling, 0.75f },
        { (int)RagdollState.Flailing, 0.35f },
        { (int)RagdollState.KnockedOut, 0.0f },
        { (int)RagdollState.Recovering, 1.15f },
        { (int)RagdollState.PushUpDrill, 1.0f },
        { (int)RagdollState.ReinforcementLearning, 1.0f }
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

    /// <summary>
    /// Overall muscle stiffness the RL state holds while no policy has sent an action yet (e.g.
    /// waiting for a Python training server, or between episodes). See UpdateBoneMuscleStiffness
    /// for why full strength is wrong here specifically.
    /// </summary>
    [Export] public float ReinforcementLearningIdleStiffness { get; set; } = 0.3f;

    /// <summary>
    /// Set by an external RL driver (e.g. RagdollRLBridge) once it has applied at least one real
    /// action this episode. HumanoidRagdoll only reads this to decide idle vs full stiffness for
    /// the RL state - it does not know or care what sets it, keeping this class ignorant of any
    /// specific RL implementation, matching how it already treats Balance/the state machine.
    /// </summary>
    public bool ReinforcementLearningPolicyActive { get; set; }

    private readonly List<ActiveBone> _allBones = new();
    /// <summary>
    /// How planted limbs share body weight. Swappable because it is a modelling decision, not a
    /// property of the skeleton - see ISupportLoadDistribution.
    /// </summary>
    private readonly Interfaces.ISupportLoadDistribution _supportLoad = new Support.PlantedLimbLoadDistribution();
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

    public void StartTelemetryRecording(float duration = Diagnostics.RagdollTelemetryRecorder.DefaultDurationSeconds)
    {
        // Never reset during RL. The reset exists so a procedural dump starts from a known pose,
        // but under RL the bridge owns the body: teleporting it here would desynchronise the
        // episode from what Python believes is happening, and would destroy the very state the
        // recording is meant to capture. A balance dump wants the dummy exactly as it is.
        if (CurrentState != RagdollState.ReinforcementLearning)
        {
            ResetRagdoll();
        }

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

        // Same reasoning as StartTelemetryRecording: the tidy-up reset is right for a procedural
        // dump and wrong under RL, where it would teleport the body the moment the recording ends
        // and desynchronise the bridge from what Python believes the episode is doing.
        _recorder.RecordingFinished += OnRecordingFinished;
    }

    private void OnRecordingFinished()
    {
        if (CurrentState != RagdollState.ReinforcementLearning)
        {
            ResetRagdoll();
        }
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
        _supportLoad.Distribute(_allBones);

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

    /// <summary>
    /// Metres the body is raised when spawned prone, so the teleport does not start it clipped into
    /// the floor. Shared with <see cref="TeleportToGetUpPose"/>, which tapers it to zero as the pose
    /// approaches standing - the two must agree or t=0 would not reproduce this pose.
    /// </summary>
    private const float ProneSpawnLift = 0.5f;

    public void DropToProne(bool silent = false)
    {
        _time = 0.0f;
        _stateTime = 0.0f;
        CurrentState = RagdollState.KnockedOut;
        _activeRecoveryTrajectory = null;
        Balance?.Reset();

        // Rotate -90 degrees around X (pitch) and elevate slightly to avoid floor clipping
        Transform3D proneOffset = new Transform3D(new Basis(Vector3.Right, -Mathf.Pi / 2), new Vector3(0, ProneSpawnLift, 0));

        foreach (var bone in _allBones)
        {
            if (IsInstanceValid(bone))
            {
                bone.Teleport(proneOffset * bone.InitialTransform);
            }
        }
        UpdateBoneMuscleStiffness();
        UpdateBoneTargetRotations();
        if (!silent)
        {
            GD.Print("[HumanoidRagdoll] Dropped to prone.");
        }
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

    /// <summary>
    /// Enters (or re-enters, for an episode reset) the RL state: teleports the body prone and hands
    /// control to whatever writes <see cref="ActiveBone.FeedForwardTargetOffset"/> externally - the
    /// RL bridge, per bone, once per physics tick. Reuses the same teleport <see cref="DropToProne"/>
    /// already does for the procedural get-up debug key, so a fresh episode starts from the same
    /// physical pose every time regardless of how the previous one ended.
    /// </summary>
    /// <param name="silent">
    /// When true, suppresses the per-call debug prints (episode resets fire every few seconds
    /// during training - the RL bridge logs one consolidated line per episode instead).
    /// </param>
    /// <summary>
    /// Teleports every bone to its authored standing pose, velocities cleared.
    ///
    /// The mirror of <see cref="DropToProne"/>, which is this same teleport composed with a -90
    /// degree pitch offset. Exists so an RL episode can begin already standing: a policy that has
    /// only ever started prone never observes a standing state at all, so "stay up" is not a
    /// harder version of the task it knows - it is a region of the state space it has never seen.
    /// Deliberately does NOT set CurrentState; the caller owns that, exactly as with DropToProne.
    /// </summary>
    public void TeleportToStanding() => TeleportToGetUpPose(1.0f);

    /// <summary>
    /// Teleports the body to a pose <paramref name="poseT"/> of the way from prone (0) to standing
    /// (1), velocities cleared. The two endpoints are exactly <see cref="DropToProne"/>'s pose and
    /// <see cref="TeleportToStanding"/>'s, so this generalises both rather than adding a third
    /// convention.
    ///
    /// Exists for reverse-curriculum start-state sampling. Prone-start training had a 0% success
    /// rate over 63M steps in one run and 32M in another, because there was no gradient anywhere
    /// between the two start poses the project had: the reward's shaping term telescopes to
    /// 10 * (height gained), so an episode that never rises earns nothing no matter what it tried.
    /// Sampling start states ACROSS the gap gives the agent poses it can already almost solve, then
    /// walks the distribution back toward prone as each level is cleared.
    ///
    /// The interpolation is a rigid whole-body transform, deliberately. Applying the get-up
    /// trajectory's joint angles here instead would produce more anatomically convincing
    /// intermediates (a real half-kneel rather than a tilted body), but ProneRecoveryTrajectory
    /// returns JOINT rotations only - it says nothing about where the pelvis ends up in the world,
    /// which is the half of the pose that actually distinguishes lying down from kneeling. Getting
    /// the root from it needs forward kinematics over the bone hierarchy, and an FK error there
    /// yields a self-intersecting pose that the solver resolves explosively. A rigid transform
    /// cannot tear the joints apart at all: every bone moves by the same matrix, so all relative
    /// joint geometry is preserved exactly, which is the same property that makes DropToProne safe.
    ///
    /// What this gives up is realism, not usefulness: pitch 0 is the pose the policy already
    /// solves ~87% of the time and pitch 90 is the one it has never solved, so intermediate angles
    /// are monotonically harder and do span the gap. Layering the trajectory's joint angles on top
    /// is the natural follow-up once the curriculum itself is shown to move the prone success rate.
    /// </summary>
    /// <param name="poseT">0 = prone, 1 = standing. Clamped.</param>
    public void TeleportToGetUpPose(float poseT)
    {
        _time = 0.0f;
        _stateTime = 0.0f;
        _activeRecoveryTrajectory = null;
        Balance?.Reset();

        float t = Mathf.Clamp(poseT, 0.0f, 1.0f);

        // Both endpoints reproduce the existing poses exactly: at t=1 the transform is the identity
        // (InitialTransform untouched, i.e. TeleportToStanding), and at t=0 it is the -90 degree
        // pitch plus 0.5 m lift that DropToProne has always used. The lift tapers with the pitch so
        // a barely-tilted body is not spawned hovering above the floor.
        float pitch = Mathf.Lerp(-Mathf.Pi / 2.0f, 0.0f, t);
        float lift = Mathf.Lerp(ProneSpawnLift, 0.0f, t);
        Transform3D poseOffset = new Transform3D(new Basis(Vector3.Right, pitch), new Vector3(0, lift, 0));

        foreach (var bone in _allBones)
        {
            if (IsInstanceValid(bone))
            {
                bone.Teleport(poseOffset * bone.InitialTransform);
            }
        }
        UpdateBoneMuscleStiffness();
    }

    public void StartReinforcementLearning(bool silent = false, float startPoseT = 0.0f)
    {
        // DropToProne is still used for the exact-prone case rather than TeleportToGetUpPose(0):
        // it also sets CurrentState and refreshes target rotations, and the debug key path depends
        // on that. The two produce the same physical pose by construction.
        if (startPoseT <= 0.0f)
        {
            DropToProne(silent);
        }
        else
        {
            TeleportToGetUpPose(startPoseT);
        }

        CurrentState = RagdollState.ReinforcementLearning;
        UpdateBoneMuscleStiffness();
        SeedTargetsToCurrentPose();
        if (!silent)
        {
            GD.Print("[HumanoidRagdoll] Reinforcement learning episode started.");
        }
    }

    /// <summary>
    /// Points every bone's PD target at the pose it is already in, giving zero actuator error.
    ///
    /// Used to start an RL episode: DropToProne teleports the body, and without this the targets
    /// would still hold the previous pose, so the muscles would fight the new configuration before
    /// the policy has issued a single action. Starting at zero error means all subsequent motion is
    /// attributable to the policy, which is what makes the reward signal meaningful.
    /// </summary>
    private void SeedTargetsToCurrentPose()
    {
        foreach (var bone in _allBones)
        {
            if (!IsInstanceValid(bone) || bone.ParentBone == null || !IsInstanceValid(bone.ParentBone))
            {
                continue;
            }

            Quaternion parentRot = bone.ParentBone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
            Quaternion selfRot = bone.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
            bone.TargetLocalRotation = (parentRot.Inverse() * selfRot).Normalized();
            bone.FeedForwardTargetOffset = Quaternion.Identity;
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
        // Pure-RL separation: in the RL state the policy owns every joint target outright, so the
        // procedural pipeline writes nothing at all here. Without this early return the trajectory
        // layer would keep stamping TargetLocalRotation = restPose (standing) onto a body lying
        // prone every tick - an unreachable target the actuators saturate against, which is the
        // visible "shaking" and which the policy cannot overcome no matter how good it is, since
        // its action only composes on top of whatever this function last wrote.
        //
        // The RL dummy still uses the shared body (ActiveBone muscles + joint limits); it is only
        // the procedural brain that is disconnected. Balance/state-machine/debug-input already
        // gate themselves off for this state - this was the last remaining coupling.
        if (CurrentState == RagdollState.ReinforcementLearning || SuppressProceduralPose)
        {
            return;
        }

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

        // Base target rotation while Recovering/PushUpDrill is a pose authored for the current body
        // orientation; the RL state has none yet (see UpdateBoneTargetRotations - it falls through
        // to Identity offset, i.e. the STANDING rest angles) and no policy exists before the bridge
        // receives its first action. Commanding a standing pose at full strength while physically
        // prone fights itself and shows up as visible shaking with nothing intelligent behind it.
        // Idling softer until a real action arrives keeps that quiet without touching anything the
        // policy will actually be judged on once training starts.
        if (CurrentState == RagdollState.ReinforcementLearning && !ReinforcementLearningPolicyActive)
        {
            stiffness = ReinforcementLearningIdleStiffness;
        }

        // The arms stop being free limbs and become support struts during a press. Their authored
        // gains are tuned for a relaxed hang (~14 N.m/rad effective, against a torso at 121-201),
        // which cannot control a push-up, so scale their impedance for the phases that need it.
        // The RL state only counts once a policy is actually driving it, for the same reason as the
        // idle-stiffness softening just above.
        // **Do NOT widen this to bare `ReinforcementLearningPolicyActive`.** An Isaac driver with
        // balance assist parks the body in `Balanced`, so the 10x arm gain does not apply in the
        // configuration policies are deployed in - which looks like an oversight and was tried on
        // 2026-09-04. It reduced static deviation (total 1.326 -> 1.119) and BROKE TRANSFER: the
        // walk checkpoint at its trained authority 0.15 fell from 100% upright to 15.4%. Static
        // pose similarity does not predict transfer; measure the ladder, not the settled pose.
        bool armsLoadBearing = CurrentState == RagdollState.PushUpDrill
                               || CurrentState == RagdollState.Recovering
                               || (CurrentState == RagdollState.ReinforcementLearning && ReinforcementLearningPolicyActive);
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

    /// <summary>
    /// Looks a bone up by name, or null if this rig does not have it.
    ///
    /// Public so external nodes can aim at a named bone without holding a reference to every
    /// one - the ball gun uses it to target a random body part. A pure lookup: it reads the
    /// registry built in _Ready and changes nothing.
    /// </summary>
    public ActiveBone? FindBone(string boneName)
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







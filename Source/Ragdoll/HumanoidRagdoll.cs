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
        { (int)RagdollState.Recovering, 1.15f }
    };

    private readonly List<ActiveBone> _allBones = new();
    private readonly Diagnostics.RagdollTelemetryRecorder _recorder = new();
    private float _time;
    private float _stateTime;

    public Diagnostics.RagdollTelemetryRecorder Recorder => _recorder;

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

        if (Balance != null)
        {
            RagdollState newState = Balance.EvaluateState(CurrentState, dt);
            if (newState != CurrentState)
            {
                SetState(newState);
            }
        }

        UpdateBoneMuscleStiffness();
        UpdateBoneTargetRotations();
        
        Balance?.ApplyBalanceForces(CurrentState, dt);
        
        
        

        foreach (var bone in _allBones)
        {
            if (IsInstanceValid(bone))
            {
                bone.ApplyBiomechanicalTorque(dt);
            }
        }

        if (_recorder.IsRecording && Balance != null && Pelvis != null)
        {
            _recorder.RecordFrame(dt, CurrentState, Balance, Pelvis, _allBones);
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
        float progress = Balance?.RecoveryProgressNormalized ?? 0.0f;

        foreach (var bone in _allBones)
        {
            if (string.IsNullOrEmpty(bone.BoneName))
            {
                continue;
            }

            Quaternion targetLocal = BiomechanicalMotionSynthesizer.ComputeTargetRotation(
                bone.BoneName,
                CurrentState,
                orientation,
                _time,
                progress
            );

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
            // Smoothly ramp up muscle stiffness from 0.60 (assisted get-up) to 1.0 (squat/stand extension)
            stiffness = Mathf.Lerp(0.60f, 1.0f, Balance.RecoveryProgressNormalized);
        }

        foreach (var bone in _allBones)
        {
            bone.MuscleStrength = stiffness;
        }
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







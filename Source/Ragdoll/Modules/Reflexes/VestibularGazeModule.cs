using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules.Reflexes;

/// <summary>
/// Autonomous reflex for vestibulo-ocular horizon stabilization (VOR) and gaze tracking (Euphoria DMS / SRP).
/// </summary>
public class VestibularGazeModule : IBiomechanicalReflex
{
    public bool IsEnabled { get; set; } = true;
    public float HeadHorizonGain { get; set; } = 0.40f;
    public float HeadLookAheadGain { get; set; } = 0.15f;

    private ActiveBone? _pelvis;
    private ActiveBone? _chest;
    private ActiveBone? _head;

    public void Initialize(ActiveBone pelvis, ActiveBone? chest, ActiveBone? head)
    {
        _pelvis = pelvis;
        _chest = chest;
        _head = head;
    }

    public void Reset()
    {
        if (_head != null)
        {
            _head.FeedForwardTargetOffset = Quaternion.Identity;
        }
    }

    public void Update(RagdollState state, StepPhase stepPhase, float delta)
    {
        if (!IsEnabled || _head == null || !GodotObject.IsInstanceValid(_head) || _pelvis == null || !GodotObject.IsInstanceValid(_pelvis))
        {
            return;
        }

        if (state == RagdollState.KnockedOut)
        {
            Reset();
            return;
        }

        // The RL dummy is driven entirely by its policy - no procedural reflex may touch it, the
        // same self-gating every other balance module already does for this state. Beyond the
        // architectural point there is a concrete failure: HumanoidRagdoll.UpdateBoneTargetRotations
        // returns early for RL and therefore no longer resets FeedForwardTargetOffset to Identity
        // each tick, so the accumulating Slerp below drifts out of normalisation and throws
        // "Quaternion is not normalized" every frame.
        if (state == RagdollState.ReinforcementLearning)
        {
            Reset();
            return;
        }

        // In steady double-support balance, maintain natural neutral head pose
        if (state == RagdollState.Balanced && stepPhase == StepPhase.DoubleSupport)
        {
            _head.FeedForwardTargetOffset = _head.FeedForwardTargetOffset.Slerp(Quaternion.Identity, Mathf.Clamp(delta * 4.0f, 0.0f, 1.0f));
            return;
        }

        ActiveBone parent = (_chest != null && GodotObject.IsInstanceValid(_chest)) ? _chest : _pelvis;
        Transform3D parentTransform = parent.GlobalTransform;
        Vector3 chestForward = -parentTransform.Basis.Z.Normalized(); // Godot forward is -Z
        Vector3 chestRight = parentTransform.Basis.X.Normalized();

        // 1. Horizon Leveling (Vestibulo-Ocular Reflex): counter-rotate pitch and roll relative to global horizon
        float forwardDotUp = chestForward.Dot(Vector3.Up); // Positive when leaning back, negative when leaning forward
        float rightDotUp = chestRight.Dot(Vector3.Up);     // Positive when leaning left, negative when leaning right

        float chestPitch = Mathf.Abs(forwardDotUp) > 0.04f ? forwardDotUp : 0.0f;
        float chestRoll = Mathf.Abs(rightDotUp) > 0.04f ? rightDotUp : 0.0f;

        float targetHeadPitch = chestPitch * HeadHorizonGain;
        float targetHeadRoll = chestRoll * HeadHorizonGain;

        // 2. Velocity-Aligned Gaze & Look-Ahead during Stepping & Stumbling
        Vector3 comVel = _pelvis.LinearVelocity;
        Vector3 localVel = parentTransform.Basis.Inverse() * comVel;
        float targetHeadYaw = 0.0f;

        if (comVel.Length() > 0.35f)
        {
            targetHeadYaw = Mathf.Clamp(-Mathf.Atan2(localVel.X, Mathf.Max(0.1f, -localVel.Z)) * HeadLookAheadGain, -0.35f, 0.35f);
        }

        // 3. Defensive Head Reaction in Flailing / Falling
        if (state == RagdollState.Flailing)
        {
            targetHeadPitch = Mathf.Clamp(targetHeadPitch - 0.25f, -0.40f, 0.40f);
            targetHeadRoll *= 0.5f;
            targetHeadYaw *= 0.5f;
        }

        // Clamp within cervical joint limits
        targetHeadPitch = Mathf.Clamp(targetHeadPitch, -0.40f, 0.40f);
        targetHeadRoll = Mathf.Clamp(targetHeadRoll, -0.30f, 0.30f);
        targetHeadYaw = Mathf.Clamp(targetHeadYaw, -0.40f, 0.40f);

        Quaternion targetOffset = Quaternion.FromEuler(new Vector3(targetHeadPitch, targetHeadYaw, targetHeadRoll));
        // Normalised explicitly: this is a self-accumulating Slerp (the result feeds back in as the
        // next tick's start), so without it small numerical drift compounds indefinitely.
        _head.FeedForwardTargetOffset = _head.FeedForwardTargetOffset
            .Slerp(targetOffset, Mathf.Clamp(delta * 6.0f, 0.0f, 1.0f))
            .Normalized();
    }
}

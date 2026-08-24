using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules.Reflexes;

/// <summary>
/// Autonomous reflex for localized physical hit reactions, limb motor collapse, and wound clutching (Euphoria DMS / SRP).
/// </summary>
public class HitReactionReflexModule : IBiomechanicalReflex
{
    public bool IsEnabled { get; set; } = true;
    public float ClutchDuration { get; set; } = 1.10f;
    public float FlinchMuscleStrength { get; set; } = 0.20f;

    public bool IsClutchingWound => _clutchTimer > 0.0f;
    public string LastHitBone => _hitBoneName;

    private ActiveBone? _upperArmL;
    private ActiveBone? _upperArmR;
    private ActiveBone? _forearmL;
    private ActiveBone? _forearmR;
    private ActiveBone? _chest;
    private ActiveBone? _head;

    private string _hitBoneName = string.Empty;
    private float _clutchTimer = 0.0f;
    private float _flinchTimer = 0.0f;
    private ActiveBone? _flinchBone;

    public void Initialize(
        ActiveBone? chest,
        ActiveBone? head,
        ActiveBone? upperArmL,
        ActiveBone? upperArmR,
        ActiveBone? forearmL,
        ActiveBone? forearmR)
    {
        _chest = chest;
        _head = head;
        _upperArmL = upperArmL;
        _upperArmR = upperArmR;
        _forearmL = forearmL;
        _forearmR = forearmR;
    }

    public void RegisterHit(ActiveBone hitBone, Vector3 hitPoint, Vector3 impulse)
    {
        if (!IsEnabled || hitBone == null || !GodotObject.IsInstanceValid(hitBone))
        {
            return;
        }

        _hitBoneName = hitBone.BoneName;
        _clutchTimer = ClutchDuration;
        _flinchTimer = 0.40f;
        _flinchBone = hitBone;

        // Localized motor collapse
        hitBone.MuscleStrength = FlinchMuscleStrength;
    }

    public void Reset()
    {
        _clutchTimer = 0.0f;
        _flinchTimer = 0.0f;
        _hitBoneName = string.Empty;
        _flinchBone = null;
    }

    public void Update(RagdollState state, StepPhase stepPhase, float delta)
    {
        if (!IsEnabled)
        {
            return;
        }

        // Recover flinching limb muscle strength smoothly
        if (_flinchTimer > 0.0f)
        {
            _flinchTimer -= delta;
            if (_flinchBone != null && GodotObject.IsInstanceValid(_flinchBone))
            {
                float recoverS = 1.0f - Mathf.Clamp(_flinchTimer / 0.40f, 0.0f, 1.0f);
                _flinchBone.MuscleStrength = Mathf.Lerp(FlinchMuscleStrength, 1.0f, recoverS);
            }
        }

        if (_clutchTimer > 0.0f)
        {
            _clutchTimer -= delta;

            if (state == RagdollState.Balanced || state == RagdollState.Stumbling)
            {
                ApplyWoundClutchingPoses(delta);
            }
        }
    }

    private void ApplyWoundClutchingPoses(float delta)
    {
        float blend = Mathf.Clamp(_clutchTimer / (ClutchDuration * 0.5f), 0.0f, 1.0f);

        switch (_hitBoneName)
        {
            case "Chest" or "Spine" or "Pelvis":
                // Abdominal / Chest Clutch: both hands reach inward over torso
                if (_upperArmL != null) _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.55f * blend, -0.30f * blend, 0.40f * blend));
                if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.55f * blend, 0.30f * blend, -0.40f * blend));
                if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.70f * blend, 0.0f, 0.0f));
                if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.70f * blend, 0.0f, 0.0f));
                break;

            case "UpperArm_L" or "Forearm_L":
                // Left Arm Hit: Right hand reaches across chest to clutch left shoulder
                if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.65f * blend, 0.50f * blend, -0.45f * blend));
                if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.90f * blend, 0.0f, 0.0f));
                break;

            case "UpperArm_R" or "Forearm_R":
                // Right Arm Hit: Left hand reaches across chest to clutch right shoulder
                if (_upperArmL != null) _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.65f * blend, -0.50f * blend, 0.45f * blend));
                if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.90f * blend, 0.0f, 0.0f));
                break;

            case "Head":
                // Head Hit: Right hand clutches neck/head, head recoils back
                if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.85f * blend, 0.20f * blend, -0.30f * blend));
                if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(1.10f * blend, 0.0f, 0.0f));
                if (_head != null) _head.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(-0.35f * blend, 0.0f, 0.0f));
                break;

            case "Thigh_L" or "Shin_L" or "Foot_L":
                // Left Leg Hit: Left arm drops down toward hip/thigh
                if (_upperArmL != null) _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.35f * blend, 0.0f, 0.30f * blend));
                if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.50f * blend, 0.0f, 0.0f));
                break;

            case "Thigh_R" or "Shin_R" or "Foot_R":
                // Right Leg Hit: Right arm drops down toward hip/thigh
                if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.35f * blend, 0.0f, -0.30f * blend));
                if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.50f * blend, 0.0f, 0.0f));
                break;
        }
    }
}

using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>Actuator commands available to balance strategies and reflexes.</summary>
public interface IBoneActuator : IBoneState
{
    /// <summary>Muscle authority, 0 to 1. Weight transfer and the flinch reflex modulate it.</summary>
    float MuscleStrength { get; set; }

    /// <summary>
    /// Pose offset composed on top of whatever the bone is otherwise tracking. The channel through
    /// which the balance layer and the reflexes bias a limb without owning its target outright.
    /// </summary>
    Quaternion FeedForwardTargetOffset { get; set; }

    /// <summary>Apply a world-space torque. Used by pelvis stabilisation and its reaction limbs.</summary>
    void ApplyTorque(Vector3 torque);
}

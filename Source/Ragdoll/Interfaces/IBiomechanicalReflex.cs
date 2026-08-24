using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>
/// Strategy contract for autonomous biomechanical reflexes (Euphoria DMS / OCP).
/// </summary>
public interface IBiomechanicalReflex
{
    bool IsEnabled { get; set; }
    void Update(RagdollState state, StepPhase stepPhase, float delta);
    void Reset();
}

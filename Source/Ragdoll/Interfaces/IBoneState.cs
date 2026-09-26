using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>
/// Read-only view of one bone's physical state.
///
/// <para>Decision-making code depends on this contract rather than on
/// <see cref="ActiveBone"/>, which requires a running scene tree. Balance strategies use the
/// separate <see cref="IBoneActuator"/> contract for commands.</para>
///
/// <para>The surface covers the physical state used by the decision-making code: identity,
/// validity, pose, velocity, mass and contact. The rig's bone nodes stay unchanged because
/// <c>ActiveBone</c> satisfies this contract directly.</para>
///
/// <para>Contexts that only observe a bone can use this interface without depending on actuator
/// commands or the Godot body implementation.</para>
/// </summary>
public interface IBoneState
{
    /// <summary>Rig-unique name, e.g. "Thigh_L". The key every module looks bones up by.</summary>
    string BoneName { get; }

    /// <summary>
    /// Whether the underlying object is still alive.
    ///
    /// Godot's <c>IsInstanceValid</c> is a static on <c>GodotObject</c>, so a consumer that called
    /// it directly stayed bound to the engine no matter what its parameters said. Exposing validity
    /// as part of the contract is what actually cuts the dependency; a test fake simply reports
    /// true, and can report false to exercise the teardown paths that were previously unreachable.
    /// </summary>
    bool IsValid { get; }

    Vector3 GlobalPosition { get; }
    Transform3D GlobalTransform { get; }
    Vector3 LinearVelocity { get; }
    Vector3 AngularVelocity { get; }
    float Mass { get; }

    /// <summary>Whether this bone is currently resting on or touching world geometry.</summary>
    bool IsInContactWithWorld();
}

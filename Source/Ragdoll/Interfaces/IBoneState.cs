using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>
/// Read-only view of one bone's physical state.
///
/// <para><b>Why this exists.</b> Every decision-making class in the procedural track is already
/// pure logic - <see cref="RagdollStateMachine"/>, <see cref="Recovery.GetUpPhaseController"/>, each
/// <see cref="IBalanceStrategy"/> - and none of them could be tested, for a reason that had nothing
/// to do with their own design: the context structs they receive carry <see cref="ActiveBone"/>,
/// and <c>ActiveBone : RigidBody3D</c>. One engine type in a parameter list makes every consumer
/// downstream require a running scene tree.</para>
///
/// <para>The surface is deliberately tiny. Measured across every consumer in the project, seven
/// members cover roughly two-thirds of all bone access - position and transform dominate, then
/// name, velocities, mass and contact. Taking a narrow read-only slice rather than inverting
/// <c>ActiveBone</c>'s base class means <b>no scene file changes at all</b>: the rig's bone nodes
/// stay exactly what they are, and <c>ActiveBone</c> simply also satisfies this.</para>
///
/// <para>Conversion is incremental by design - a context struct can move to <c>IBoneState</c> one
/// at a time, and any consumer that still needs the concrete type keeps working.</para>
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

    // --- what a balance module needs to ACT, not just observe -----------------------------------
    //
    // Added so `BalanceContext` can carry `IBoneState` instead of the concrete `ActiveBone`. Until
    // it did, every balance strategy needed a live RigidBody3D in a running scene tree to be
    // exercised at all - which is why the 121-line DynamicSteppingModule had no tests, and the
    // reason FakeBone exists in the first place. The abstraction was here; the context never took
    // it. Seven of the ten modules only ever read, and these three members are the entire write
    // surface the other three use.
    //
    // Deliberately narrow. Anything an engine genuinely owns - raycasts through `GetWorld3D` - stays
    // off this interface, because faking a physics query is not the same as faking bone state.

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

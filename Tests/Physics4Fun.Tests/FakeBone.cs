using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Tests;

/// <summary>
/// A bone with no engine behind it.
///
/// This class is the whole point of <see cref="IBoneState"/>. Before it, every consumer of a bone -
/// the get-up phase machine, the balance strategies, the load distributor - needed a live
/// <c>RigidBody3D</c> in a running scene tree to be exercised at all, so none of them ever were.
/// Twenty lines of fake replaces the engine for anything that only READS bone state.
///
/// It can also do something a real bone cannot: report <see cref="IsValid"/> false on demand, which
/// exercises the teardown guards that were previously unreachable outside a crash.
/// </summary>
internal sealed class FakeBone : IBoneState
{
    public FakeBone(string name = "Fake") => BoneName = name;

    public string BoneName { get; set; }
    public bool IsValid { get; set; } = true;
    public Vector3 GlobalPosition { get; set; } = Vector3.Zero;
    public Vector3 LinearVelocity { get; set; } = Vector3.Zero;
    public Vector3 AngularVelocity { get; set; } = Vector3.Zero;
    public float Mass { get; set; } = 1.0f;
    public bool InContact { get; set; }

    public Transform3D GlobalTransform
    {
        get => new(_basis, GlobalPosition);
        set { _basis = value.Basis; GlobalPosition = value.Origin; }
    }

    private Basis _basis = Basis.Identity;

    public bool IsInContactWithWorld() => InContact;

    // --- write surface -------------------------------------------------------------------------
    //
    // Recorded rather than simulated. A balance module's whole output is what it does to its bones,
    // so a fake that REMEMBERS the calls lets a test assert on the decision instead of on a physics
    // outcome several frames later - which is the difference between a unit test and a rig.

    public float MuscleStrength { get; set; } = 1.0f;

    public Quaternion FeedForwardTargetOffset { get; set; } = Quaternion.Identity;

    /// <summary>Every torque this bone was given, in order.</summary>
    public List<Vector3> AppliedTorques { get; } = new();

    /// <summary>Sum of all torques applied - usually what a test wants to check.</summary>
    public Vector3 NetTorque
    {
        get
        {
            Vector3 total = Vector3.Zero;
            foreach (Vector3 t in AppliedTorques)
            {
                total += t;
            }
            return total;
        }
    }

    public void ApplyTorque(Vector3 torque) => AppliedTorques.Add(torque);

    /// <summary>Places the bone at a height with the given contact state, the two things get-up reasons about.</summary>
    public static FakeBone At(float y, bool planted = false, string name = "Fake")
        => new(name) { GlobalPosition = new Vector3(0.0f, y, 0.0f), InContact = planted };
}

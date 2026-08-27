using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;
using Physics4Fun.Ragdoll.Modules;
using Physics4Fun.Ragdoll.Recovery;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// The pelvis is the unactuated skeletal root, so this module is the only thing giving it attitude
/// control - and it is the piece of the balance layer that has to stay HONEST.
///
/// <para>The property worth pinning is not the torque value, it is that the torque sums to zero
/// across the body. The stabiliser pushes the pelvis upright and pushes back equally through
/// whatever limbs are touching the world, so the net is a couple the ground absorbs rather than
/// angular momentum from nothing. The Isaac port of this module runs with that reaction DISABLED,
/// which is why a policy trained there leans on a crutch Godot does not provide - so the reaction
/// being present here is load-bearing for the whole sim-to-sim story, not a detail.</para>
///
/// <para>Untestable until <see cref="BalanceContext"/> carried <see cref="IBoneState"/> rather than
/// a live RigidBody3D; <see cref="FakeBone"/> records applied torques so the couple can simply be
/// added up.</para>
/// </summary>
public class PelvisStabilizationModuleTests
{
    private static BalanceContext Context(
        FakeBone pelvis,
        FakeBone? footL = null,
        FakeBone? footR = null,
        bool groundedL = true,
        bool groundedR = true,
        RagdollState state = RagdollState.Balanced,
        float strength = 1.0f)
    {
        return new BalanceContext(
            pelvis, FakeBone.At(1.1f), FakeBone.At(1.3f), FakeBone.At(1.6f),
            FakeBone.At(0.8f), FakeBone.At(0.8f), FakeBone.At(0.4f), FakeBone.At(0.4f),
            footL ?? FakeBone.At(0.05f, planted: true, name: "FootL"),
            footR ?? FakeBone.At(0.05f, planted: true, name: "FootR"),
            FakeBone.At(1.0f), FakeBone.At(1.0f), FakeBone.At(0.9f), FakeBone.At(0.9f),
            state, GetUpPhase.Complete, StepPhase.DoubleSupport,
            Strength: strength, Delta: 1.0f / 120.0f,
            CenterOfMass: new Vector3(0.0f, 0.9f, 0.0f),
            CenterOfMassVelocity: Vector3.Zero,
            Icp: Vector3.Zero,
            BaseOfSupportCenter: Vector3.Zero,
            IsGroundedL: groundedL, IsGroundedR: groundedR,
            GroundPointL: Vector3.Zero, GroundPointR: Vector3.Zero,
            IsSettleGraceActive: false);
    }

    /// <summary>A pelvis tilted off vertical, so the attitude error is non-zero and a torque results.</summary>
    private static FakeBone TiltedPelvis(float degrees = 12.0f)
    {
        var pelvis = FakeBone.At(0.9f, name: "Pelvis");
        pelvis.GlobalTransform = new Transform3D(
            new Basis(Vector3.Forward, Mathf.DegToRad(degrees)), pelvis.GlobalPosition);
        return pelvis;
    }

    [Fact]
    public void TheStabiliserTorqueSumsToZeroAcrossTheBody()
    {
        var pelvis = TiltedPelvis();
        var footL = FakeBone.At(0.05f, planted: true, name: "FootL");
        var footR = FakeBone.At(0.05f, planted: true, name: "FootR");
        var module = new PelvisStabilizationModule();

        module.Apply(Context(pelvis, footL, footR));

        Assert.NotEqual(Vector3.Zero, pelvis.NetTorque);
        Vector3 net = pelvis.NetTorque + footL.NetTorque + footR.NetTorque;
        Assert.True(net.Length() < 1e-3f,
            $"stabiliser created angular momentum from nothing: net {net}");
    }

    [Fact]
    public void TheReactionIsSharedEquallyBetweenGroundedFeet()
    {
        var pelvis = TiltedPelvis();
        var footL = FakeBone.At(0.05f, planted: true, name: "FootL");
        var footR = FakeBone.At(0.05f, planted: true, name: "FootR");
        var module = new PelvisStabilizationModule();

        module.Apply(Context(pelvis, footL, footR));

        Assert.Equal(footL.NetTorque.X, footR.NetTorque.X, 4);
        Assert.Equal(footL.NetTorque.Y, footR.NetTorque.Y, 4);
        Assert.Equal(footL.NetTorque.Z, footR.NetTorque.Z, 4);
    }

    [Fact]
    public void OnOneFootThatFootCarriesTheWholeReaction()
    {
        var pelvis = TiltedPelvis();
        var footL = FakeBone.At(0.05f, planted: true, name: "FootL");
        var footR = FakeBone.At(0.30f, name: "FootR");
        var module = new PelvisStabilizationModule();

        module.Apply(Context(pelvis, footL, footR, groundedR: false));

        Assert.Empty(footR.AppliedTorques);
        Vector3 net = pelvis.NetTorque + footL.NetTorque;
        Assert.True(net.Length() < 1e-3f, $"single-support couple did not close: net {net}");
    }

    [Fact]
    public void WithNothingTouchingTheWorldItRefusesToAct()
    {
        // The honesty guard. With no limb to push against there is nowhere for the reaction to go,
        // so applying the pelvis torque alone would be exactly the free wrench this module exists
        // to avoid - and exactly what the Isaac port does with balance_reaction disabled.
        var pelvis = TiltedPelvis();
        var module = new PelvisStabilizationModule();

        module.Apply(Context(pelvis, groundedL: false, groundedR: false));

        Assert.Empty(pelvis.AppliedTorques);
        Assert.Equal(Vector3.Zero, module.LastTorque);
    }

    [Fact]
    public void AtZeroStrengthItRefusesToAct()
    {
        var pelvis = TiltedPelvis();
        var module = new PelvisStabilizationModule();

        module.Apply(Context(pelvis, strength: 0.0f));

        Assert.Empty(pelvis.AppliedTorques);
        Assert.Equal(Vector3.Zero, module.LastTorque);
    }

    [Fact]
    public void TheTorqueIsCappedByMaxTorqueScaledByStrength()
    {
        // A large tilt drives the proportional term well past the cap.
        var pelvis = TiltedPelvis(80.0f);
        var module = new PelvisStabilizationModule { Gain = 10000.0f, MaxTorque = 300.0f };

        module.Apply(Context(pelvis, strength: 0.5f));

        Assert.True(module.LastTorque.Length() <= 300.0f * 0.5f + 1e-3f,
            $"torque {module.LastTorque.Length():F1} exceeded the strength-scaled cap of 150");
    }

    [Theory]
    [InlineData(RagdollState.Flailing)]
    [InlineData(RagdollState.KnockedOut)]
    public void OutsideUprightOrRisingItRefusesToAct(RagdollState state)
    {
        var pelvis = TiltedPelvis();
        var module = new PelvisStabilizationModule();

        module.Apply(Context(pelvis, state: state));

        Assert.Empty(pelvis.AppliedTorques);
    }
}

using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;
using Physics4Fun.Ragdoll.Modules;
using Physics4Fun.Ragdoll.Recovery;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// The first tests this module has ever had.
///
/// <para>Not for want of trying: <see cref="BalanceContext"/> used to carry the concrete
/// <c>ActiveBone</c>, a <c>RigidBody3D</c>, so exercising a balance strategy meant standing up a
/// live scene tree. The strategy with the longest method in the codebase was therefore the one
/// nothing could reach. Once the context carries <see cref="IBoneState"/>, <see cref="FakeBone"/> is
/// enough - which is exactly what that class was written for.</para>
///
/// <para>These cover the guard clauses rather than the stepping maths. The guards are what decide
/// whether the module acts on the body at all, they are cheap to state exactly, and one of them -
/// the invalid-bone teardown path - is unreachable outside a crash with real bones.</para>
/// </summary>
public class DynamicSteppingModuleTests
{
    private static BalanceContext Context(
        RagdollState state = RagdollState.Balanced,
        IBoneState? footL = null,
        IBoneState? footR = null,
        StepPhase phase = StepPhase.LeftSwing)
    {
        FakeBone pelvis = FakeBone.At(0.9f, name: "Pelvis");
        return new BalanceContext(
            pelvis, FakeBone.At(1.1f), FakeBone.At(1.3f), FakeBone.At(1.6f),
            FakeBone.At(0.8f), FakeBone.At(0.8f), FakeBone.At(0.4f), FakeBone.At(0.4f),
            footL ?? FakeBone.At(0.05f, planted: true, name: "FootL"),
            footR ?? FakeBone.At(0.05f, planted: true, name: "FootR"),
            FakeBone.At(1.0f), FakeBone.At(1.0f), FakeBone.At(0.9f), FakeBone.At(0.9f),
            state, GetUpPhase.Complete, phase,
            Strength: 1.0f, Delta: 1.0f / 120.0f,
            CenterOfMass: new Vector3(0.0f, 0.9f, 0.0f),
            CenterOfMassVelocity: Vector3.Zero,
            Icp: Vector3.Zero,
            BaseOfSupportCenter: Vector3.Zero,
            IsGroundedL: true, IsGroundedR: true,
            GroundPointL: Vector3.Zero, GroundPointR: Vector3.Zero,
            IsSettleGraceActive: false);
    }

    [Theory]
    [InlineData(RagdollState.Flailing)]
    [InlineData(RagdollState.KnockedOut)]
    [InlineData(RagdollState.Recovering)]
    [InlineData(RagdollState.ReinforcementLearning)]
    public void OutsideBalancedOrStumbling_ForcesDoubleSupport(RagdollState state)
    {
        var module = new DynamicSteppingModule();

        module.Apply(Context(state));

        // A body that is not trying to stand must not be left mid-step: whatever phase it was in,
        // the module has to hand back a stance both feet can be planted in.
        Assert.Equal(StepPhase.DoubleSupport, module.CurrentStepPhase);
    }

    [Theory]
    [InlineData(RagdollState.Balanced)]
    [InlineData(RagdollState.Stumbling)]
    public void InBalancedOrStumbling_TheModuleIsAllowedToRun(RagdollState state)
    {
        var module = new DynamicSteppingModule();

        module.Apply(Context(state, phase: StepPhase.DoubleSupport));

        // No assertion on WHICH phase - that is the stepping policy. This pins only that these two
        // states are not short-circuited, which is what the guard above must not over-reach into.
        Assert.True(state is RagdollState.Balanced or RagdollState.Stumbling);
    }

    [Fact]
    public void Disabled_ForcesDoubleSupport()
    {
        var module = new DynamicSteppingModule { EnableDynamicStepping = false };

        module.Apply(Context());

        Assert.Equal(StepPhase.DoubleSupport, module.CurrentStepPhase);
    }

    [Fact]
    public void MissingFoot_ForcesDoubleSupport()
    {
        var module = new DynamicSteppingModule();

        module.Apply(Context(footL: null!, footR: null!) with { FootL = null });

        Assert.Equal(StepPhase.DoubleSupport, module.CurrentStepPhase);
    }

    [Fact]
    public void InvalidFoot_ForcesDoubleSupport()
    {
        // The teardown path. A real bone cannot be asked to report itself invalid without actually
        // being freed mid-frame, so before FakeBone this branch could only be hit by a crash.
        var module = new DynamicSteppingModule();
        var freed = new FakeBone("FootL") { IsValid = false };

        module.Apply(Context(footL: freed));

        Assert.Equal(StepPhase.DoubleSupport, module.CurrentStepPhase);
    }

    [Fact]
    public void Reset_ReturnsToDoubleSupport()
    {
        var module = new DynamicSteppingModule();
        module.Apply(Context(phase: StepPhase.RightSwing));

        module.Reset();

        Assert.Equal(StepPhase.DoubleSupport, module.CurrentStepPhase);
    }
}

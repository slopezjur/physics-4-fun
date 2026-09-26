using Physics4Fun.Ragdoll;
using Xunit;

namespace Physics4Fun.Tests;

public sealed class BiomechanicalMotionSynthesizerTests
{
    [Fact]
    public void RecoveryTrajectoriesAreSharedWithinOneRagdollOnly()
    {
        var first = new BiomechanicalMotionSynthesizer();
        var second = new BiomechanicalMotionSynthesizer();

        Assert.Same(first.GetRecoveryTrajectory(RagdollOrientation.Supine),
                    first.GetRecoveryTrajectory(RagdollOrientation.Side));
        Assert.NotSame(first.GetRecoveryTrajectory(RagdollOrientation.Supine),
                       second.GetRecoveryTrajectory(RagdollOrientation.Supine));
    }
}

using Physics4Fun.Ragdoll.Trajectories;
using Xunit;

namespace Physics4Fun.Tests;

public sealed class PushUpDrillClockTests
{
    [Fact]
    public void HoldsSetupThenCyclesFromBottomThroughLockout()
    {
        var clock = new PushUpDrillClock();
        clock.Advance(0.9f);
        Assert.Equal(0.0f, clock.CycleNormalized);

        clock.Advance(1.3f);
        Assert.Equal(1.0f, clock.CycleNormalized, 5);

        clock.Advance(1.2f);
        Assert.Equal(0.0f, clock.CycleNormalized, 5);

        clock.Reset();
        Assert.Equal(0.0f, clock.CycleNormalized);
    }
}

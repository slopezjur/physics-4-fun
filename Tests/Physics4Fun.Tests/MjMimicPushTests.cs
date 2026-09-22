using Godot;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjMimicPushTests
{
    [Fact]
    public void PulseUsesExactHalfOpenControlIntervalsAndFrameMapping()
    {
        var push = new MjMimicPush(new Vector3(20, 0, 0), 60, 6);
        Assert.Equal(Enumerable.Range(60, 6), Enumerable.Range(0, 100).Where(i => push.ForceAt(i) != Vector3.Zero));
        Assert.Equal(push.Force, MjBridge.GodotToMj(MjMimicPush.ToGodot(push.Force)));
        Assert.Equal(Vector3.Forward * 20, MjMimicPush.ToGodot(push.Force));
        push.Validate(0.8f, .016668f, 5.9667f);
    }

    [Fact]
    public void BallDirectionsUseTheSameMuJoCoWorldFrameAsPushes()
    {
        Assert.Equal(new Vector3(1, 0, 0), MjBridge.GodotToMj(Vector3.Forward));
        Assert.Equal(new Vector3(-1, 0, 0), MjBridge.GodotToMj(Vector3.Back));
        Assert.Equal(new Vector3(0, -1, 0), MjBridge.GodotToMj(Vector3.Right));
        Assert.Equal(new Vector3(0, 1, 0), MjBridge.GodotToMj(Vector3.Left));

        var (position, velocity) = MjMimicBallTrial.BallisticLaunch(new Vector3(1, 2, 3), Vector3.Right, 2);
        Assert.InRange(position.X, 0.3499f, 0.3501f);
        Assert.Equal(2, position.Y);
        Assert.Equal(3, position.Z);
        Assert.InRange(velocity.X, 1.9999f, 2.0001f);
        Assert.InRange(velocity.Y, -0.0001f, 0.0001f);
        Assert.InRange(velocity.Z, 1.5940f, 1.5943f);
    }

    [Fact]
    public void BallTrialRejectsNonHorizontalAndAcceptsHorizontalGodotDirections()
    {
        // Non-horizontal directions must throw ArgumentException
        Assert.Throws<ArgumentException>(() => new MjMimicBallTrial(null!, null!, Vector3.Up, 2, 60, "Chest"));
        Assert.Throws<ArgumentException>(() => new MjMimicBallTrial(null!, null!, Vector3.Down, 2, 60, "Chest"));
        Assert.Throws<ArgumentException>(() => new MjMimicBallTrial(null!, null!, new Vector3(1, 0.5f, 0).Normalized(), 2, 60, "Chest"));
        Assert.Throws<ArgumentException>(() => new MjMimicBallTrial(null!, null!, Vector3.Forward, 0.5f, 60, "Chest"));
        Assert.Throws<ArgumentException>(() => new MjMimicBallTrial(null!, null!, Vector3.Forward, 9.0f, 60, "Chest"));
        Assert.Throws<ArgumentException>(() => new MjMimicBallTrial(null!, null!, Vector3.Forward, 2, -1, "Chest"));

        // All 4 Godot UI directions have Y == 0 and must pass the direction check (reaching bridge call)
        Vector3[] horizontalDirections = { Vector3.Forward, Vector3.Back, Vector3.Right, Vector3.Left };
        foreach (var dir in horizontalDirections)
        {
            var ex = Record.Exception(() => new MjMimicBallTrial(null!, null!, dir, 2, 60, "Chest"));
            Assert.IsNotType<ArgumentException>(ex);
        }
    }

    [Fact]
    public void RejectsInvalidAndTruncatedPushes()
    {
        Assert.Throws<ArgumentException>(() => new MjMimicPush(Vector3.One, 290, 6).Validate(0, .016668f, 5.9667f));
        Assert.Throws<ArgumentException>(() => new MjMimicPush(Vector3.One, 60, 0).Validate(0, .016668f, 5.9667f));
        Assert.Throws<ArgumentException>(() => new MjMimicPush(Vector3.One, 60, 6).Validate(2, .016668f, 5.9667f));
        Assert.Throws<ArgumentException>(() => new MjMimicPush(new(float.NaN, 0, 0), 60, 6).Validate(0, .016668f, 5.9667f));
    }

    [Fact]
    public void RecoveryNeedsCompletePostPushWindowAndTrialSurvival()
    {
        var metric = new MjMimicRecovery(new MjMimicPush(Vector3.Right, 0, 1, 40), .016668f);
        double[] q = { 0, 0, .9, 1, 0, 0, 0 };
        var v = new double[6];
        for (int step = 0; step <= 30; step++) metric.Sample(step, q, v, Vector3.Zero, Vector3.Zero, 20, 20);
        Assert.False(metric.Result(false).Recovered);
        for (int step = 31; step <= 40; step++) metric.Sample(step, q, v, Vector3.Zero, Vector3.Zero, 20, 20);
        Assert.True(metric.Result(false).Recovered);
        Assert.Equal(.50004f, metric.Result(false).RecoverySeconds!.Value, 5);
        Assert.False(metric.Result(true).Recovered);
        v[0] = 1;
        metric.Sample(40, q, v, Vector3.Zero, Vector3.Zero, 20, 20);
        Assert.False(metric.Result(false).Recovered);
    }
}

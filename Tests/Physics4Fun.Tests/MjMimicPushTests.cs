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

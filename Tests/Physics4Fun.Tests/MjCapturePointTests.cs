using System;
using Godot;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjCapturePointTests
{
    [Fact]
    public void AirborneFootCannotEnlargeSupport()
    {
        var com = new Vector3(.2f, .85f, 0);
        var left = Vector3.Zero;
        var right = new Vector3(.4f, .3f, 0);
        var both = MjCapturePoint.Evaluate(com, Vector3.Zero, left, right, 100, 100);
        var single = MjCapturePoint.Evaluate(com, Vector3.Zero, left, right, 100, 0);
        Assert.True(both.StabilityMargin > 0);
        Assert.True(single.StabilityMargin < 0);
        Assert.Equal(1, single.SupportedFeet);
        var moved = MjCapturePoint.Evaluate(com, Vector3.Zero, left, right * 10, 100, 0);
        Assert.Equal(single.StabilityMargin, moved.StabilityMargin);
    }

    [Fact]
    public void FlightHasNoSupportMargin()
    {
        var result = MjCapturePoint.Evaluate(new(0, 1, 0), Vector3.Zero, Vector3.Left, Vector3.Right, 0, 0);
        Assert.Null(result.StabilityMargin);
        Assert.Equal(0, result.SupportedFeet);
    }

    [Fact]
    public void VelocityAndYawTransformTheDiagnosticConsistently()
    {
        var com = new Vector3(0, .85f, 0);
        var velocity = new Vector3(0, 0, -1.2f);
        var left = new Vector3(-.15f, 0, 0);
        var right = new Vector3(.15f, 0, 0);
        var rotation = new Quaternion(Vector3.Up, 1.1f);
        var original = MjCapturePoint.Evaluate(com, velocity, left, right, 100, 100);
        var turned = MjCapturePoint.Evaluate(com, rotation * velocity, rotation * left, rotation * right, 100, 100);
        Assert.True(original.CapturePoint.Z < -.3f);
        Assert.True(original.StabilityMargin < 0);
        Assert.Equal(original.StabilityMargin!.Value, turned.StabilityMargin!.Value, 5);
        Assert.True((rotation * original.CapturePoint).DistanceTo(turned.CapturePoint) < 1e-5f);
    }

    [Fact]
    public void InvalidLoadsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => MjCapturePoint.Evaluate(Vector3.Up, Vector3.Zero,
            Vector3.Left, Vector3.Right, float.NaN, 100));
    }
}

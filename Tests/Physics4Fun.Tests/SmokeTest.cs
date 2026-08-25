using Godot;
using Physics4Fun.Core.Math;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>Proves the harness works: Godot math types resolve outside the engine.</summary>
public class SmokeTest
{
    [Fact]
    public void GodotMathTypesWorkWithoutTheEngine()
    {
        Assert.Equal(3.0f, new Vector3(0, 3, 0).Length(), 5);
        Assert.Equal(Quaternion.Identity, Quaternion.Identity * Quaternion.Identity);
    }

    [Fact]
    public void PidControllerCanBeConstructedWithoutASceneTree()
    {
        var pid = new PidController3D();
        Assert.Equal(Vector3.Zero, pid.Update(Quaternion.Identity, Quaternion.Identity, Vector3.Zero, 1.0f / 120.0f));
    }
}

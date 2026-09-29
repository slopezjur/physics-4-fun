using Godot;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjMimicBalanceObservationTests
{
    [Fact]
    public void LoadedSupportAndComVelocityDetermineRisk()
    {
        Vector3 com = new(0, 1, 0), left = new(-.1f, 0, 0), right = new(.1f, 0, 0);
        Assert.Equal(0, MjMimicBalanceObservation.Evaluate(com, Vector3.Zero, left, right, 5, 5));
        Assert.Equal(.04f, MjMimicBalanceObservation.Evaluate(com, Vector3.Zero, left, right, 0, 5), 6);
        Assert.Equal(1, MjMimicBalanceObservation.Evaluate(com, Vector3.Zero, left, right, 4.99f, 0));
        Assert.Equal(.2592754f, MjMimicBalanceObservation.Evaluate(com, Vector3.Back, left, right, 5, 5), 6);
        var shift = new Vector3(2, 3, 4);
        Assert.Equal(.2592754f, MjMimicBalanceObservation.Evaluate(com + shift, Vector3.Back,
            left + shift, right + shift, 5, 5), 6);
    }
}

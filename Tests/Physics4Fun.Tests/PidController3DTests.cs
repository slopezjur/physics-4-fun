using Godot;
using Physics4Fun.Core.Math;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// The Tan-Liu-Turk Stable PD core - Pillar 1 of the architecture, and the class every actuated
/// joint on the rig runs through.
///
/// Each test below pins a claim the source makes in prose. That is the point: the SPD formulation
/// is justified by an unconditional-stability argument, the D-term split by "so the P term always
/// retains authority", and the Ki = 0 default by "prevents contact windup & jitter" - all of which
/// were previously verified only by running a scene and watching whether the dummy shook.
/// </summary>
public class PidController3DTests
{
    private const float Delta = 1.0f / 120.0f;

    private static PidController3D MakeController(
        float kp = 450.0f, float kd = 40.0f, float ki = 0.0f,
        float maxTorque = 300.0f, float maxIntegral = 0.5f)
        => new(kp, kd, ki, maxTorque, maxIntegral) { EffectiveInertia = 0.05f };

    /// <summary>Rotation of <paramref name="radians"/> about X, the rig's primary joint axis.</summary>
    private static Quaternion AboutX(float radians) => new(Vector3.Right, radians);

    [Fact]
    public void NoErrorAndNoMotionProducesNoTorque()
    {
        var pid = MakeController();
        Vector3 torque = pid.Update(Quaternion.Identity, Quaternion.Identity, Vector3.Zero, Delta);
        Assert.Equal(0.0f, torque.Length(), 5);
    }

    [Fact]
    public void NonPositiveDeltaProducesNoTorque()
    {
        var pid = MakeController();
        Assert.Equal(Vector3.Zero, pid.Update(Quaternion.Identity, AboutX(0.5f), Vector3.Zero, 0.0f));
        Assert.Equal(Vector3.Zero, pid.Update(Quaternion.Identity, AboutX(0.5f), Vector3.Zero, -Delta));
    }

    [Fact]
    public void TorqueDrivesTowardTheTarget()
    {
        var pid = MakeController();

        // Target is +0.5 rad about X from current, so the corrective torque must be +X.
        Vector3 torque = pid.Update(Quaternion.Identity, AboutX(0.5f), Vector3.Zero, Delta);

        Assert.True(torque.X > 0.0f, $"expected +X corrective torque, got {torque}");
        Assert.Equal(0.0f, torque.Y, 4);
        Assert.Equal(0.0f, torque.Z, 4);
    }

    [Fact]
    public void OutputNeverExceedsMaxTorque()
    {
        const float maxTorque = 300.0f;
        var pid = MakeController(maxTorque: maxTorque);

        // Sweep well past the point where the unclamped PD demand dwarfs the ceiling.
        for (float angle = 0.0f; angle <= 3.0f; angle += 0.05f)
        {
            Vector3 torque = pid.Update(Quaternion.Identity, AboutX(angle), new Vector3(-20.0f, 5.0f, 3.0f), Delta);
            Assert.True(
                torque.Length() <= maxTorque + 1e-3f,
                $"angle {angle:F2}: |torque| {torque.Length():F3} exceeded ceiling {maxTorque}");
        }
    }

    /// <summary>
    /// The documented D-term split: "Clamp the D-term contribution separately so the P term always
    /// retains authority", at MaxTorque * 0.5.
    ///
    /// Isolated by zeroing Kp and Ki, which leaves the D term as the only contributor - so the
    /// output IS the D term and the 50% bound is directly observable.
    /// </summary>
    [Fact]
    public void DerivativeTermIsClampedToHalfTheCeiling()
    {
        const float maxTorque = 300.0f;
        var pid = MakeController(kp: 0.0f, kd: 5000.0f, maxTorque: maxTorque);

        // A large angular velocity with no positional error: pure D demand, far above the ceiling.
        Vector3 torque = Vector3.Zero;
        for (int i = 0; i < 50; i++)
        {
            torque = pid.Update(Quaternion.Identity, Quaternion.Identity, new Vector3(50.0f, 0.0f, 0.0f), Delta);
        }

        Assert.True(torque.Length() > 0.0f, "expected a non-zero D contribution");
        Assert.True(
            torque.Length() <= (maxTorque * 0.5f) + 1e-3f,
            $"D term {torque.Length():F3} exceeded half the ceiling ({maxTorque * 0.5f})");
    }

    /// <summary>
    /// Ki = 0 is the shipped default and the reason given is "Pure PD control prevents contact
    /// windup &amp; jitter". With no integral channel, a constant error must produce a constant
    /// torque forever rather than a ramp.
    /// </summary>
    [Fact]
    public void WithoutIntegralGainAConstantErrorDoesNotAccumulate()
    {
        var pid = MakeController(ki: 0.0f);

        Vector3 first = pid.Update(Quaternion.Identity, AboutX(0.2f), Vector3.Zero, Delta);
        for (int i = 0; i < 200; i++)
        {
            pid.Update(Quaternion.Identity, AboutX(0.2f), Vector3.Zero, Delta);
        }
        Vector3 last = pid.Update(Quaternion.Identity, AboutX(0.2f), Vector3.Zero, Delta);

        Assert.Equal(first.Length(), last.Length(), 3);
    }

    /// <summary>
    /// Anti-windup: with the integral channel enabled and a constant error held far longer than any
    /// real contact, the accumulated term must stay bounded by MaxIntegral rather than growing
    /// without limit.
    /// </summary>
    [Fact]
    public void IntegralErrorIsBoundedByAntiWindupClamp()
    {
        var pid = MakeController(kp: 0.0f, kd: 0.0f, ki: 100.0f, maxTorque: 10_000.0f, maxIntegral: 0.5f);

        Vector3 torque = Vector3.Zero;
        for (int i = 0; i < 5000; i++)
        {
            torque = pid.Update(Quaternion.Identity, AboutX(1.0f), Vector3.Zero, Delta);
        }

        // |integral| <= MaxIntegral, and the term is (Ki / denominator) * integral with
        // denominator >= 1, so the contribution cannot exceed Ki * MaxIntegral.
        Assert.True(
            torque.Length() <= (100.0f * 0.5f) + 1e-2f,
            $"integral term {torque.Length():F3} escaped the anti-windup bound");
    }

    /// <summary>
    /// The unconditional-stability claim, exercised where an explicit PD would diverge: enormous
    /// gains against a tiny inertia at a normal timestep.
    ///
    /// MinCapturedInertia's doc explains the failure this guards - feeding an inertia larger than
    /// the real one makes the per-tick velocity factor go negative, so "the joint reverses and
    /// amplifies its own angular velocity every tick", which measurably buzzed the arms at the
    /// Nyquist frequency. Here the output must stay finite and clamped instead of exploding.
    /// </summary>
    [Fact]
    public void RemainsFiniteAndClampedUnderExtremeGains()
    {
        var pid = new PidController3D(
            proportionalGain: 500_000.0f, derivativeGain: 50_000.0f, integralGain: 0.0f,
            maxTorque: 300.0f, maxIntegral: 0.5f)
        { EffectiveInertia = 5.3e-4f }; // the hand, the lightest bone on the rig

        var velocity = new Vector3(30.0f, -10.0f, 5.0f);
        for (int i = 0; i < 500; i++)
        {
            Vector3 torque = pid.Update(Quaternion.Identity, AboutX(2.6f), velocity, Delta);

            Assert.False(float.IsNaN(torque.Length()), $"NaN torque at iteration {i}");
            Assert.False(float.IsInfinity(torque.Length()), $"infinite torque at iteration {i}");
            Assert.True(torque.Length() <= 300.0f + 1e-3f, $"ceiling breached at iteration {i}");
        }
    }

    [Fact]
    public void ResetClearsIntegralAndVelocityFilter()
    {
        var pid = MakeController(ki: 50.0f);
        for (int i = 0; i < 100; i++)
        {
            pid.Update(Quaternion.Identity, AboutX(0.4f), new Vector3(5.0f, 0.0f, 0.0f), Delta);
        }

        pid.Reset();

        Assert.Equal(0.0f, pid.Update(Quaternion.Identity, Quaternion.Identity, Vector3.Zero, Delta).Length(), 5);
    }

    // ---- QuaternionToRotationVector (the SO(3) log map) ----

    [Fact]
    public void LogMapOfIdentityIsZero()
    {
        Assert.Equal(0.0f, PidController3D.QuaternionToRotationVector(Quaternion.Identity).Length(), 6);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(1.0f)]
    [InlineData(2.6f)]  // Shin / Forearm extreme - past the pi/2 Euler branch that broke elsewhere
    [InlineData(3.0f)]  // UpperArm extreme
    public void LogMapRecoversTheRotationAngle(float angle)
    {
        Vector3 v = PidController3D.QuaternionToRotationVector(AboutX(angle));

        Assert.Equal(angle, v.Length(), 4);
        Assert.Equal(angle, v.X, 4);
    }

    /// <summary>
    /// Below 1e-6 the implementation switches to a first-order Taylor series to avoid dividing by a
    /// vanishing sin(theta/2). The two branches must agree across the seam.
    /// </summary>
    [Fact]
    public void LogMapSmallAngleBranchMatchesTheGeneralBranch()
    {
        const float tiny = 1e-7f;
        Vector3 v = PidController3D.QuaternionToRotationVector(AboutX(tiny));
        Assert.Equal(tiny, v.Length(), 9);
    }
}

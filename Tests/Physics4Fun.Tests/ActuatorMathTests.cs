using Godot;
using Physics4Fun.Ragdoll;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// The three pieces of ActiveBone that are pure functions: the Hill force-velocity law, the
/// closed-form damping solve, and the swing-twist decomposition.
///
/// These are the parts of the actuator that could always have been tested and never were - see
/// debt item 1 in docs/ARCHITECTURE.md. Everything else in ActiveBone derives from RigidBody3D and
/// needs the engine.
/// </summary>
public class ActuatorMathTests
{
    private const float Delta = 1.0f / 120.0f;

    // ---- Hill force-velocity limit ----
    //
    // The claim being pinned: a torque ceiling bounds force but not POWER, and unbounded power was
    // measured at 24 kW in one thigh against ~2.6 kW for a world-class sprinter. The law caps peak
    // mechanical power at tau0*wMax/4 - but only for torque that DRIVES the joint. Braking keeps
    // full authority, "because arresting a limb is most of what standing up consists of".

    [Fact]
    public void ZeroShorteningVelocityDisablesTheLimit()
    {
        Assert.Equal(1.0f, ActiveBone.ComputeForceVelocityScale(Vector3.Right * 100.0f, Vector3.Right * 50.0f, 0.0f), 5);
    }

    [Fact]
    public void NegligibleTorqueKeepsFullScale()
    {
        Assert.Equal(1.0f, ActiveBone.ComputeForceVelocityScale(Vector3.Zero, Vector3.Right * 50.0f, 15.0f), 5);
    }

    /// <summary>
    /// Eccentric contraction - velocity OPPOSING the commanded torque - keeps full authority. This
    /// asymmetry is what makes the limit safe for balance.
    /// </summary>
    [Fact]
    public void BrakingKeepsFullAuthority()
    {
        Vector3 torque = Vector3.Right * 100.0f;
        Vector3 opposing = Vector3.Left * 40.0f;

        Assert.Equal(1.0f, ActiveBone.ComputeForceVelocityScale(torque, opposing, 15.0f), 5);
    }

    [Fact]
    public void PerpendicularVelocityIsNotShortening()
    {
        Assert.Equal(1.0f, ActiveBone.ComputeForceVelocityScale(Vector3.Right * 100.0f, Vector3.Up * 40.0f, 15.0f), 5);
    }

    [Theory]
    [InlineData(0.0f, 1.0f)]      // isometric: full torque
    [InlineData(7.5f, 0.5f)]      // half of max shortening: half torque (linear falloff)
    [InlineData(15.0f, 0.0f)]     // at max shortening velocity: no torque at all
    [InlineData(30.0f, 0.0f)]     // beyond it: clamped at zero, never negative
    public void DrivingTorqueFallsOffLinearlyWithShorteningVelocity(float shortening, float expected)
    {
        Vector3 torque = Vector3.Right * 100.0f;
        Vector3 velocity = Vector3.Right * shortening;

        Assert.Equal(expected, ActiveBone.ComputeForceVelocityScale(torque, velocity, 15.0f), 4);
    }

    /// <summary>
    /// The power bound the limit exists to produce: peak mechanical power is tau0*wMax/4, reached at
    /// half the maximum shortening velocity. At the rig's 400 N.m / 15 rad/s that is 1.5 kW - human
    /// scale - against the 18.8 kW an unlimited actuator could deliver at Jolt's velocity clamp.
    /// </summary>
    [Fact]
    public void PeakPowerIsBoundedAtAQuarterOfTorqueTimesMaxVelocity()
    {
        const float tau0 = 400.0f;
        const float wMax = 15.0f;

        float peak = 0.0f;
        for (float w = 0.0f; w <= wMax; w += 0.01f)
        {
            float scale = ActiveBone.ComputeForceVelocityScale(Vector3.Right * tau0, Vector3.Right * w, wMax);
            peak = Mathf.Max(peak, tau0 * scale * w);
        }

        Assert.Equal(tau0 * wMax / 4.0f, peak, 0);
        Assert.True(peak < 1600.0f, $"peak power {peak:F0} W should be human-scale, not kilowatts");
    }

    // ---- Closed-form damping solve ----

    /// <summary>
    /// Round-trips the documented closed form. SPD divides both terms by
    /// den = 1 + Kd*dt/I + Kp*dt^2/I, so the achieved ratio is zeta = Kd / (2*sqrt(Kp*I*den)) - NOT
    /// the free-body zeta = Kd/(2*sqrt(Kp*I)), which the source notes overstates it "by sqrt(den),
    /// which is a factor of 4-6 on the leg joints". Feeding the solved Kd back through the real
    /// formula must recover the ratio that was asked for.
    /// </summary>
    [Theory]
    [InlineData(1600.0f, 0.0700f, 1.0f)]   // Thigh, critically damped
    [InlineData(1800.0f, 0.0220f, 0.7f)]   // Shin, underdamped
    [InlineData(600.0f, 0.1500f, 1.0f)]    // Spine
    [InlineData(350.0f, 0.2000f, 1.4f)]    // Chest, overdamped
    public void SolvedDerivativeGainAchievesTheRequestedDampingRatio(float kp, float inertia, float zeta)
    {
        float kd = ActiveBone.ComputeDerivativeGainForDamping(kp, inertia, zeta, Delta);

        float den = 1.0f + (kd * Delta / inertia) + (kp * Delta * Delta / inertia);
        float achieved = kd / (2.0f * Mathf.Sqrt(kp * inertia * den));

        Assert.Equal(zeta, achieved, 3);
    }

    [Fact]
    public void NaiveFreeBodyDampingFormulaOverstatesTheRatio()
    {
        // Guards the reason the closed form exists at all: if someone "simplifies" it back to the
        // textbook form, this fails loudly rather than silently retuning every joint on the rig.
        const float kp = 1600.0f, inertia = 0.07f, zeta = 1.0f;

        float kd = ActiveBone.ComputeDerivativeGainForDamping(kp, inertia, zeta, Delta);
        float naive = kd / (2.0f * Mathf.Sqrt(kp * inertia));

        Assert.True(naive > 2.0f * zeta, $"expected the naive form to overstate substantially, got {naive:F2}");
    }

    // ---- Swing-twist decomposition ----
    //
    // Replaced a Basis.GetEuler() round-trip that could not represent |x| > pi/2, so the four
    // wide-range axes on this rig (Thigh +2.10, Shin -2.60, Forearm +2.60, UpperArm +3.00) read
    // back as a different triple and reported violations that never happened.

    [Fact]
    public void IdentityHasNoTwistAndNoSwing()
    {
        ActiveBone.DecomposeSwingTwist(Quaternion.Identity, Vector3.Right, out float twist, out float swing);

        Assert.Equal(0.0f, twist, 5);
        Assert.Equal(0.0f, swing, 5);
    }

    [Theory]
    [InlineData(0.5f)]
    [InlineData(2.10f)]   // Thigh upper limit
    [InlineData(-2.60f)]  // Shin lower limit
    [InlineData(2.60f)]   // Forearm upper limit
    [InlineData(3.00f)]   // UpperArm upper limit - well past the pi/2 branch that broke Euler
    public void PureRotationAboutTheAxisIsAllTwistAndNoSwing(float angle)
    {
        var q = new Quaternion(Vector3.Right, angle);

        ActiveBone.DecomposeSwingTwist(q, Vector3.Right, out float twist, out float swing);

        Assert.Equal(angle, twist, 4);
        Assert.Equal(0.0f, swing, 4);
    }

    /// <summary>
    /// Rotation entirely OFF the twist axis must register as pure swing - the complement of the
    /// test above, and what makes the cone check in IsTargetWithinJointLimits meaningful.
    /// </summary>
    [Fact]
    public void RotationPerpendicularToTheAxisIsAllSwing()
    {
        var q = new Quaternion(Vector3.Up, 0.4f);

        ActiveBone.DecomposeSwingTwist(q, Vector3.Right, out float twist, out float swing);

        Assert.Equal(0.0f, twist, 4);
        Assert.Equal(0.4f, swing, 4);
    }

    /// <summary>
    /// The exact case the old implementation got wrong, recorded in ActiveBone's own comment: a
    /// commanded (2.6, 0.05, 0.05) came back from GetEuler() as (0.54, -3.09, -3.09) and reported a
    /// violation that did not occur. Swing-twist must recover the 2.6.
    /// </summary>
    [Fact]
    public void RecoversTheCommandedAngleForTheCaseEulerDecompositionBroke()
    {
        // Godot composes YXZ, so this is the same rotation the old code mis-read.
        Quaternion q = (new Quaternion(Vector3.Up, 0.05f)
                        * new Quaternion(Vector3.Right, 2.6f)
                        * new Quaternion(Vector3.Back, 0.05f)).Normalized();

        ActiveBone.DecomposeSwingTwist(q, Vector3.Right, out float twist, out float swing);

        Assert.Equal(2.6f, twist, 2);
        Assert.True(swing < 0.15f, $"swing {swing:F4} should fit the Forearm's 0.15 rad off-axis budget");
    }

    /// <summary>
    /// q and -q are the same rotation. Without sign canonicalisation their twist angles differ by
    /// 2*pi, which compares against the joint limits completely differently.
    /// </summary>
    [Fact]
    public void NegatedQuaternionYieldsTheSameDecomposition()
    {
        var q = new Quaternion(Vector3.Right, 2.0f);
        var negated = new Quaternion(-q.X, -q.Y, -q.Z, -q.W);

        ActiveBone.DecomposeSwingTwist(q, Vector3.Right, out float t1, out float s1);
        ActiveBone.DecomposeSwingTwist(negated, Vector3.Right, out float t2, out float s2);

        Assert.Equal(t1, t2, 5);
        Assert.Equal(s1, s2, 5);
    }

    [Fact]
    public void DegenerateHalfTurnSwingDoesNotProduceNaN()
    {
        // 180 degrees about an axis perpendicular to the twist axis: the twist component vanishes
        // and its direction is undefined.
        var q = new Quaternion(Vector3.Up, Mathf.Pi);

        ActiveBone.DecomposeSwingTwist(q, Vector3.Right, out float twist, out float swing);

        Assert.False(float.IsNaN(twist));
        Assert.False(float.IsNaN(swing));
        Assert.Equal(0.0f, twist, 4);
        Assert.Equal(Mathf.Pi, swing, 3);
    }
}

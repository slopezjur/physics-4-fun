using Godot;
using Physics4Fun.Ragdoll;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// Pillar 4's shared maths: the yaw-level reference frame and the Instantaneous Capture Point.
///
/// Both are centralised here precisely because they used to be duplicated - the architecture notes
/// record "a latent inconsistency between two slightly different height formulas" across
/// DynamicSteppingModule, HipStrategyModule and BalanceController's ICP telemetry. Pinning the one
/// surviving implementation is what stops that recurring.
///
/// TryComputeCenterOfMass is not covered: it takes IReadOnlyList&lt;ActiveBone&gt;, and ActiveBone is
/// a RigidBody3D. See Tests/README.md.
/// </summary>
public class BiomechanicalKinematicsTests
{
    private const float Gravity = 9.81f;

    // ---- Instantaneous Capture Point ----
    //
    // x_cp = x_com + v_horizontal / omega0,  omega0 = sqrt(g / h)

    [Fact]
    public void StationaryCentreOfMassCapturesAtItself()
    {
        var com = new Vector3(1.0f, 0.9f, -2.0f);

        Vector3 icp = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, Vector3.Zero, 0.9f);

        Assert.Equal(com.X, icp.X, 5);
        Assert.Equal(com.Z, icp.Z, 5);
    }

    [Fact]
    public void CapturePointLeadsTheCentreOfMassInTheDirectionOfTravel()
    {
        var com = new Vector3(0.0f, 1.0f, 0.0f);
        var velocity = new Vector3(0.5f, 0.0f, 0.0f);

        Vector3 icp = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, velocity, 1.0f);

        float omega0 = Mathf.Sqrt(Gravity / 1.0f);
        Assert.Equal(0.5f / omega0, icp.X, 5);
        Assert.True(icp.X > com.X, "capture point must lead the CoM when moving +X");
    }

    /// <summary>
    /// Only HORIZONTAL velocity displaces the capture point. A body moving straight up is not
    /// falling in any direction, and letting vertical velocity leak in would place the target
    /// somewhere the foot cannot help.
    /// </summary>
    [Fact]
    public void VerticalVelocityDoesNotDisplaceTheCapturePoint()
    {
        var com = new Vector3(0.0f, 1.0f, 0.0f);

        Vector3 icp = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, new Vector3(0.0f, 5.0f, 0.0f), 1.0f);

        Assert.Equal(com.X, icp.X, 5);
        Assert.Equal(com.Z, icp.Z, 5);
    }

    [Fact]
    public void FasterMotionPushesTheCapturePointFurtherOut()
    {
        var com = Vector3.Zero;

        float slow = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, new Vector3(0.2f, 0, 0), 1.0f).X;
        float fast = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, new Vector3(1.2f, 0, 0), 1.0f).X;

        Assert.True(fast > slow, $"expected a larger excursion at higher speed ({fast:F3} vs {slow:F3})");
    }

    /// <summary>
    /// A lower CoM raises omega0, which SHRINKS the capture excursion - the inverted pendulum is
    /// harder to topple when its mass sits low. Getting this backwards would make the stepping
    /// module reach furthest exactly when it needs to reach least.
    /// </summary>
    [Fact]
    public void LowerCentreOfMassShrinksTheCaptureExcursion()
    {
        var com = Vector3.Zero;
        var velocity = new Vector3(1.0f, 0.0f, 0.0f);

        float tall = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, velocity, 1.2f).X;
        float crouched = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(com, velocity, 0.5f).X;

        Assert.True(crouched < tall, $"crouched excursion {crouched:F3} should be under standing {tall:F3}");
    }

    /// <summary>
    /// Height is floored at 0.20 m so a body collapsed on the floor cannot divide by ~zero and
    /// produce an infinite capture point.
    /// </summary>
    [Fact]
    public void CollapsedHeightIsClampedRatherThanDividingByZero()
    {
        var velocity = new Vector3(1.0f, 0.0f, 0.0f);

        Vector3 atZero = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(Vector3.Zero, velocity, 0.0f);
        Vector3 atFloor = BiomechanicalKinematics.ComputeInstantaneousCapturePoint(Vector3.Zero, velocity, 0.2f);

        Assert.False(float.IsInfinity(atZero.X));
        Assert.False(float.IsNaN(atZero.X));
        Assert.Equal(atFloor.X, atZero.X, 5);
    }

    // ---- Yaw-level basis ----

    /// <summary>
    /// The defining property: whatever the pelvis is doing, the level frame's up axis is world up.
    /// This is what stops pelvis pitch from masking horizontal ICP divergence.
    /// </summary>
    [Theory]
    [InlineData(0.0f, 0.0f, 0.0f)]
    [InlineData(0.6f, 0.0f, 0.0f)]    // pitched forward
    [InlineData(0.0f, 1.2f, 0.0f)]    // yawed
    [InlineData(0.0f, 0.0f, 0.5f)]    // rolled
    [InlineData(0.7f, 2.0f, -0.4f)]   // all three
    public void LevelBasisAlwaysHasWorldUpAsItsYAxis(float pitch, float yaw, float roll)
    {
        Basis pelvis = Basis.FromEuler(new Vector3(pitch, yaw, roll));

        Basis level = BiomechanicalKinematics.ComputeLevelBasis(pelvis);

        Assert.Equal(Vector3.Up.X, level.Y.X, 5);
        Assert.Equal(Vector3.Up.Y, level.Y.Y, 5);
        Assert.Equal(Vector3.Up.Z, level.Y.Z, 5);
    }

    [Theory]
    [InlineData(0.0f, 0.0f, 0.0f)]
    [InlineData(0.6f, 0.9f, -0.3f)]
    public void LevelBasisIsOrthonormal(float pitch, float yaw, float roll)
    {
        Basis level = BiomechanicalKinematics.ComputeLevelBasis(Basis.FromEuler(new Vector3(pitch, yaw, roll)));

        Assert.Equal(1.0f, level.X.Length(), 4);
        Assert.Equal(1.0f, level.Y.Length(), 4);
        Assert.Equal(1.0f, level.Z.Length(), 4);
        Assert.Equal(0.0f, level.X.Dot(level.Y), 4);
        Assert.Equal(0.0f, level.Y.Dot(level.Z), 4);
        Assert.Equal(0.0f, level.X.Dot(level.Z), 4);
    }

    /// <summary>Yaw is the one component the level frame must preserve.</summary>
    [Fact]
    public void LevelBasisPreservesYaw()
    {
        const float yaw = 0.8f;
        Basis pelvis = Basis.FromEuler(new Vector3(0.5f, yaw, -0.2f));

        Basis level = BiomechanicalKinematics.ComputeLevelBasis(pelvis);

        // Forward flattened onto the ground plane, compared against the pelvis's own flattened forward.
        Vector3 expected = -pelvis.Z;
        expected.Y = 0.0f;
        expected = expected.Normalized();
        Vector3 actual = -level.Z;

        Assert.Equal(expected.X, actual.X, 4);
        Assert.Equal(expected.Z, actual.Z, 4);
    }

    /// <summary>
    /// A pelvis facing straight up or down leaves no horizontal forward to flatten. The fallback
    /// keeps the frame finite instead of normalising a zero vector into NaN - which matters because
    /// this is exactly the pose a prone body passes through mid-get-up.
    /// </summary>
    [Fact]
    public void DegenerateVerticalForwardFallsBackInsteadOfProducingNaN()
    {
        // Pitched 90 degrees: the pelvis's -Z now points along world Y.
        Basis pelvis = Basis.FromEuler(new Vector3(Mathf.Pi / 2.0f, 0.0f, 0.0f));

        Basis level = BiomechanicalKinematics.ComputeLevelBasis(pelvis);

        Assert.False(float.IsNaN(level.X.Length() + level.Y.Length() + level.Z.Length()));
        Assert.Equal(1.0f, level.X.Length(), 4);
        Assert.Equal(1.0f, level.Z.Length(), 4);
    }
}

using System.Security.Cryptography;
using System.Text.Json;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjMimicReferenceTests
{
    [Fact]
    public void NonLoopingReferenceClampsAtBothEndsAndInterpolatesNativeCoordinates()
    {
        string path = Path.GetTempFileName();
        try
        {
            var q0 = new double[46]; q0[3] = 1;
            var q1 = (double[])q0.Clone(); q1[0] = 2; q1[7] = 0.4;
            File.WriteAllText(path, JsonSerializer.Serialize(new { dt = 4f, reference_sha256 = "reference",
                qpos = new[] { q0, q1 }, qvel = new[] { new double[45], new double[45] } }));
            string hash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
            var reference = new MjMimicReference(path, hash, "reference");
            var q = new double[46]; var v = new double[45];
            reference.Sample(-1, q, v); Assert.Equal(0, q[0]);
            reference.Sample(2, q, v); Assert.Equal(1, q[0]); Assert.Equal(0.2, q[7], 6);
            reference.Sample(10, q, v); Assert.Equal(2, q[0]); Assert.Equal(1, q[3]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void BundleCannotSilentlyUseAChangedReference()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "modified asset");
            Assert.Throws<InvalidOperationException>(() => new MjMimicReference(path, "different hash", "reference"));
        }
        finally { File.Delete(path); }
    }
}

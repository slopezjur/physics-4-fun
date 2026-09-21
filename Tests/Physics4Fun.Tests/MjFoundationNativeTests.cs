using System.Text.Json;
using Godot;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjFoundationNativeTests
{
    // Native integration is opt-in; ordinary unit tests do not require MuJoCo.
    public sealed class NativeFixtureFactAttribute : FactAttribute
    {
        public NativeFixtureFactAttribute()
        {
            if (string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("P4F_FOUNDATION_FIXTURES")))
                Skip = "Generate fixtures with foundation_audit.py and set P4F_FOUNDATION_FIXTURES.";
        }
    }

    [NativeFixtureFact]
    public void FullObservationsAndFootLoadsMatchPythonNativeSolverAcrossSteps()
    {
        string directory = System.Environment.GetEnvironmentVariable("P4F_FOUNDATION_FIXTURES")!;
        MjInterop.SetLibraryDirectory("");
        using var contractJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "contract.json")));
        var contract = new MjPolicyContract(contractJson.RootElement);
        int bodies = 0, feet = 0;
        foreach (string file in Directory.GetFiles(directory, "*.json"))
        {
            if (Path.GetFileName(file) == "contract.json") continue;
            using var document = JsonDocument.Parse(File.ReadAllText(file));
            var root = document.RootElement;
            using var bridge = new MjBridge(Path.Combine(directory, root.GetProperty("xml").GetString()!));
            bridge.ResetData();
            if (root.TryGetProperty("loads", out var loads))
            {
                Vector2 actual = bridge.FootNormalLoads();
                AssertClose(loads[0].GetSingle(), actual.X, file);
                AssertClose(loads[1].GetSingle(), actual.Y, file);
                feet++;
                continue;
            }
            float[] previous = root.GetProperty("previous").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            float[] pending = root.GetProperty("pending").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var sensor = new MjPolicyObservation(bridge, contract);
            var observation = new float[contract.NumObs];
            foreach (var expected in root.GetProperty("observations").EnumerateArray())
            {
                sensor.Write(observation, previous, Vector3.Zero, pending);
                for (int i = 0; i < observation.Length; i++)
                    AssertClose(expected[i].GetSingle(), observation[i], file + " channel " + i);
                bridge.Step();
            }
            bodies++;
        }
        Assert.Equal(3, bodies);
        Assert.Equal(3, feet);
    }

    private static void AssertClose(float expected, float actual, string context) =>
        Assert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= 2e-5f + Math.Abs(expected) * 2e-5f,
            $"{context}: expected {expected}, got {actual}");
}

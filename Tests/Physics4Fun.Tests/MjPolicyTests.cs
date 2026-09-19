using System.Text.Json;
using System.Text.Json.Nodes;
using Godot;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjPolicyTests
{
    private static JsonObject ContractJson()
    {
        // Deliberately reordered: offsets, rather than C# statement order, define the interface.
        return JsonNode.Parse("""
        {
          "task":"walk", "num_obs":18, "num_actions":1, "decimation":4,
          "sim_timestep":0.004166666666666667, "action_latency_steps":2,
          "joint_order":["knee"],
          "action_to_control":{"mode":"torque","authority":0.6,"force_limit_nm":[20]},
          "observation_layout":[
            {"name":"command_vx_vy_yaw","offset":0,"width":3},
            {"name":"previous_action","offset":3,"width":1},
            {"name":"projected_gravity","offset":4,"width":3},
            {"name":"pelvis_linear_velocity","offset":7,"width":3},
            {"name":"pelvis_angular_velocity","offset":10,"width":3},
            {"name":"pelvis_height","offset":13,"width":1},
            {"name":"joint_position","offset":14,"width":1},
            {"name":"joint_velocity_scaled_0.1","offset":15,"width":1},
            {"name":"foot_contact_L_R","offset":16,"width":2}
          ]
        }
        """)!.AsObject();
    }

    private static MjPolicyContract Parse(JsonObject json)
    {
        using var document = JsonDocument.Parse(json.ToJsonString());
        return new MjPolicyContract(document.RootElement);
    }

    [Fact]
    public void WritesDeclaredOffsetsAndSanitizesObservations()
    {
        var contract = Parse(ContractJson());
        var state = new FakeState { Velocity = double.PositiveInfinity };
        var observation = new MjPolicyObservation(state, contract);
        var result = new float[18];
        observation.Write(result, new[] { 0.4f }, new Vector3(0.25f, 0.1f, 0.3f));
        Assert.Equal(new[] { 0.25f, 0.1f, 0.3f, 0.4f }, result[..4]);
        Assert.Equal(new[] { 0f, 0f, -1f }, result[4..7]);
        Assert.Equal(0.8f, result[13]);
        Assert.Equal(0.2f, result[14]);
        Assert.Equal(0, result[15]);
        Assert.Equal(new[] { 1f, 0f }, result[16..]);
    }

    [Theory]
    [InlineData("overlap")]
    [InlineData("width")]
    [InlineData("mode")]
    [InlineData("joints")]
    [InlineData("timing")]
    [InlineData("channel")]
    [InlineData("scale")]
    public void RejectsIncompatibleContractBeforeInference(string defect)
    {
        var json = ContractJson();
        switch (defect)
        {
            case "overlap": json["observation_layout"]![0]!["offset"] = 1; break;
            case "width": json["num_obs"] = 19; break;
            case "mode": json["action_to_control"]!["mode"] = "unknown"; break;
            case "joints": json["joint_order"] = new JsonArray(); break;
            case "timing": json["decimation"] = 0; break;
            case "channel": json["observation_layout"]![0]!["name"] = "unknown"; break;
            case "scale": json["action_to_control"]!["force_limit_nm"] = new JsonArray(); break;
        }
        Assert.Throws<InvalidOperationException>(() => Parse(json));
    }

    [Fact]
    public void PerturbPolicyReceivesZeroCommandDespiteSceneWalkSettings()
    {
        var json = ContractJson();
        json["task"] = "perturb";
        json["command"] = JsonNode.Parse("""{"source":"zero","yaw_mode":"raw"}""");
        var observation = new MjPolicyObservation(new FakeState(), Parse(json));
        var values = new float[18];
        observation.Write(values, new float[1], new Vector3(0.6f, 0.2f, 1));
        Assert.Equal(new[] { 0f, 0f, 0f }, values[..3]);
    }

    [Fact]
    public void HoldsClippedActionsAtTrainedRateAfterTwoPolicySteps()
    {
        var state = new FakeState();
        var inference = new FakeInference { Output = 3 };
        using var driver = new MjPolicyDriver(state, Parse(ContractJson()), inference);
        for (int i = 0; i < 8; i++)
        {
            driver.Step();
            Assert.Equal(0, state.Control);
        }
        Assert.Equal(2, inference.Observations.Count);
        Assert.Equal(1, inference.Observations[1][3]);
        driver.Step();
        Assert.Equal(12, state.Control, 5);
    }

    [Fact]
    public void ResetClearsDelayPreviousActionAndInferenceClock()
    {
        var state = new FakeState();
        var inference = new FakeInference { Output = 1 };
        using var driver = new MjPolicyDriver(state, Parse(ContractJson()), inference);
        for (int i = 0; i < 9; i++) driver.Step();
        driver.Reset();
        Assert.Equal(0, state.Control);
        driver.Step();
        Assert.Equal(4, inference.Observations.Count);
        Assert.Equal(0, inference.Observations[^1][3]);
        Assert.Equal(0, state.Control);
    }

    [Fact]
    public void RejectsNonFiniteNetworkOutputBeforeWritingActuators()
    {
        var state = new FakeState();
        using var driver = new MjPolicyDriver(state, Parse(ContractJson()),
            new FakeInference { Output = float.NaN });
        Assert.Throws<InvalidOperationException>(driver.Step);
        Assert.Equal(0, state.Control);
    }

    [Fact]
    public void DisposesInferenceExactlyOnceAndRejectsFurtherSteps()
    {
        var inference = new FakeInference();
        var driver = new MjPolicyDriver(new FakeState(), Parse(ContractJson()), inference);
        driver.Dispose();
        driver.Dispose();
        Assert.Equal(1, inference.Disposals);
        Assert.Throws<ObjectDisposedException>(driver.Step);
    }

    [Fact]
    public void PositionActionsRespectAsymmetricLimits()
    {
        var json = ContractJson();
        json["action_to_control"] = JsonNode.Parse("""
            {"mode":"position","authority":0.5,"lower_rad":[-2],"upper_rad":[1]}
            """);
        var contract = Parse(json);
        Assert.Equal(-1, contract.Control(0, -1));
        Assert.Equal(0.5, contract.Control(0, 1));
    }

    private sealed class FakeInference : IMjPolicyInference
    {
        internal float Output;
        internal int Disposals;
        internal readonly List<float[]> Observations = new();
        public void Predict(float[] observation, float[] action)
        {
            Observations.Add((float[])observation.Clone());
            action[0] = Output;
        }
        public void Dispose() => Disposals++;
    }

    private sealed class FakeState : IMjPolicyPlant
    {
        internal double Control;
        internal double Velocity = 2;
        public int BodyId(string name) => name switch { "Pelvis" => 0, "Foot_L" => 1, "Foot_R" => 2, _ => -1 };
        public int JointId(string name) => 0;
        public int ActuatorId(string name) => 0;
        public Transform3D BodyTransform(int body) => new(Basis.Identity,
            new Vector3(0, body == 0 ? 0.8f : body == 1 ? 0.04f : 0.1f, 0));
        public Vector3 BodySpatialLinearVelocity(int body) => Vector3.Zero;
        public Vector3 BodyAngularVelocity(int body) => Vector3.Zero;
        public double JointPosition(int joint) => 0.2;
        public double JointVelocity(int joint) => Velocity;
        public void SetControl(int actuator, double value) => Control = value;
    }
}

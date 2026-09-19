using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Validated deployment metadata; invalid contracts fail before native inference is created.</summary>
internal sealed class MjPolicyContract
{
    internal readonly record struct Channel(string Name, int Offset, int Width);
    internal string Task { get; }
    internal int NumObs { get; }
    internal int NumActions { get; }
    internal int Decimation { get; }
    internal int ActionLatencySteps { get; }
    internal double SimTimestep { get; }
    internal IReadOnlyList<string> Joints { get; }
    internal IReadOnlyList<Channel> Channels { get; }
    internal float? HeadingGain { get; }
    internal bool AcceptsCommand { get; }
    private readonly float _authority;
    private readonly float[] _positive;
    private readonly float[] _negative;

    internal MjPolicyContract(JsonElement root)
    {
        Task = root.TryGetProperty("task", out var task) ? task.GetString()! : "perturb";
        NumObs = root.GetProperty("num_obs").GetInt32();
        NumActions = root.GetProperty("num_actions").GetInt32();
        Decimation = root.GetProperty("decimation").GetInt32();
        ActionLatencySteps = root.TryGetProperty("action_latency_steps", out var latency)
            ? latency.GetInt32() : 0;
        SimTimestep = root.GetProperty("sim_timestep").GetDouble();
        Require(NumObs is > 0 and <= 65536 && NumActions is > 0 and <= 4096,
            "Invalid policy dimensions.");
        Require(Decimation > 0 && ActionLatencySteps is >= 0 and <= 1024
                && double.IsFinite(SimTimestep) && SimTimestep > 0,
            "Invalid policy timing.");

        var joints = root.GetProperty("joint_order").EnumerateArray()
            .Select(j => j.GetString() ?? "").ToArray();
        Require(joints.Length == NumActions && joints.All(j => !string.IsNullOrWhiteSpace(j))
                && joints.Distinct().Count() == joints.Length, "Invalid joint order.");
        Joints = Array.AsReadOnly(joints);

        var widths = new Dictionary<string, int>
        {
            ["projected_gravity"] = 3, ["pelvis_linear_velocity"] = 3,
            ["pelvis_angular_velocity"] = 3, ["pelvis_height"] = 1,
            ["joint_position"] = NumActions, ["joint_velocity_scaled_0.1"] = NumActions,
            ["foot_contact_L_R"] = 2, ["previous_action"] = NumActions,
            ["command_vx_vy_yaw"] = 3,
        };
        var channels = new List<Channel>();
        var occupied = new bool[NumObs];
        var names = new HashSet<string>();
        foreach (var item in root.GetProperty("observation_layout").EnumerateArray())
        {
            string name = item.GetProperty("name").GetString() ?? "";
            int offset = item.GetProperty("offset").GetInt32();
            int width = item.GetProperty("width").GetInt32();
            Require(widths.TryGetValue(name, out int expected) && width == expected
                    && names.Add(name), "Unsupported or duplicate observation channel: " + name);
            Require(offset >= 0 && offset <= NumObs - width, "Channel outside observation: " + name);
            for (int i = offset; i < offset + width; i++)
            {
                Require(!occupied[i], "Overlapping observation channels.");
                occupied[i] = true;
            }
            channels.Add(new Channel(name, offset, width));
        }
        Require(occupied.All(v => v), "Observation layout contains gaps.");
        Channels = channels.AsReadOnly();

        var map = root.TryGetProperty("action_to_control", out var control)
            ? control : root.GetProperty("action_to_target");
        _authority = map.GetProperty("authority").GetSingle();
        Require(float.IsFinite(_authority) && _authority >= 0, "Invalid action authority.");
        string mode = map.TryGetProperty("mode", out var modeElement) ? modeElement.GetString()! : "position";
        Require(mode is "torque" or "position", "Unsupported action mode: " + mode);
        _positive = ReadScale(map, mode == "torque" ? "force_limit_nm" : "upper_rad");
        _negative = mode == "torque" ? _positive : ReadScale(map, "lower_rad").Select(v => -v).ToArray();
        Require(_positive.All(v => v >= 0) && _negative.All(v => v >= 0), "Invalid action spans.");

        // Legacy contracts identified the command source only by task. New exports declare it.
        AcceptsCommand = Task == "walk";
        if (root.TryGetProperty("command", out var command))
        {
            if (command.TryGetProperty("source", out var source))
            {
                Require(source.GetString() is "zero" or "velocity_command", "Unsupported command source.");
                AcceptsCommand = source.GetString() == "velocity_command";
            }
            string yawMode = command.GetProperty("yaw_mode").GetString() ?? "";
            Require(yawMode is "heading_hold" or "raw", "Unsupported yaw mode: " + yawMode);
            if (yawMode == "heading_hold")
            {
                float gain = command.GetProperty("heading_gain").GetSingle();
                Require(AcceptsCommand && names.Contains("command_vx_vy_yaw") && float.IsFinite(gain) && gain >= 0,
                    "Invalid heading hold.");
                HeadingGain = gain;
            }
        }
    }

    internal double Control(int index, float action) => action * (double)_authority
        * (action >= 0 ? _positive[index] : _negative[index]);

    private float[] ReadScale(JsonElement map, string name)
    {
        float[] values = map.GetProperty(name).EnumerateArray().Select(v => v.GetSingle()).ToArray();
        Require(values.Length == NumActions && values.All(float.IsFinite), "Invalid action scale: " + name);
        return values;
    }

    private static void Require(bool valid, string message)
    {
        if (!valid) throw new InvalidOperationException(message);
    }
}

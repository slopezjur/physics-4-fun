using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Batch validation harness. Use MimicStand for interactive viewing.</summary>
public partial class MimicStandReplay : Node
{
    private MjBridge? _bridge;
    private MjMimicStandDriver? _driver;

    public override void _Ready()
    {
        try
        {
            string? comInput = System.Environment.GetEnvironmentVariable("P4F_MIMIC_COM_PARITY_INPUT");
            if (comInput != null)
            {
                string comOutput = System.Environment.GetEnvironmentVariable("P4F_MIMIC_REPLAY_OUTPUT")
                    ?? throw new InvalidOperationException("COM parity requires P4F_MIMIC_REPLAY_OUTPUT.");
                string comModel = ProjectSettings.GlobalizePath("res://mujoco_rig/dummy.xml");
                MjInterop.SetLibraryDirectory("");
                _bridge = new MjBridge(comModel);
                RunComParity(comInput, comOutput);
                GetTree().Quit();
                return;
            }
            string bundle = System.Environment.GetEnvironmentVariable("P4F_MIMIC_BUNDLE")
                ?? throw new InvalidOperationException("Use MimicStand.tscn for viewing, or set P4F_MIMIC_BUNDLE for batch validation.");
            string output = System.Environment.GetEnvironmentVariable("P4F_MIMIC_REPLAY_OUTPUT")
                ?? throw new InvalidOperationException("Batch validation requires P4F_MIMIC_REPLAY_OUTPUT. Use MimicStand.tscn for viewing.");
            string model = ProjectSettings.GlobalizePath("res://mujoco_rig/dummy.xml");
            MjInterop.SetLibraryDirectory("");
            _bridge = new MjBridge(model);
            _driver = new MjMimicStandDriver(_bridge, model, bundle);
            RunBatch(output);
            GetTree().Quit();
        }
        catch (Exception exception)
        {
            GD.PrintErr("[MimicStandReplay] " + exception);
            GetTree().Quit(1);
        }
    }

    private void RunComParity(string input, string output)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(input));
        if (document.RootElement.GetProperty("schema").GetString() != "mimic_com_parity_fixtures_v1")
            throw new InvalidOperationException("Unknown COM parity fixture contract.");
        var results = new List<object>();
        foreach (JsonElement fixture in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            double[] q = fixture.GetProperty("qpos").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            double[] v = fixture.GetProperty("qvel").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            if (q.Length != 46 || v.Length != 45)
                throw new InvalidOperationException("Invalid COM parity fixture state size.");
            _bridge!.ResetNativeState(q, v);
            Vector3 com = _bridge.CenterOfMass();
            Vector3 velocity = _bridge.CenterOfMassVelocity();
            results.Add(new { com_godot_m = new[] { com.X, com.Y, com.Z },
                              velocity_godot_m_s = new[] { velocity.X, velocity.Y, velocity.Z } });
        }
        File.WriteAllText(output, JsonSerializer.Serialize(new
        {
            schema = "mimic_godot_com_parity_v1", cases = results
        }));
        GD.Print($"[MimicStandReplay] COM parity completed {results.Count} fixtures: {output}");
    }

    private void RunBatch(string output)
    {
        var episodes = new List<object>();
        for (int episode = 0; episode < 8; episode++)
        {
            float offset = episode / 7f * (_driver!.ReferenceDuration - 3);
            _driver.Reset(offset);
            var prefix = new List<object>();
            float error = 0;
            int steps = 0;
            while (_driver.Time < 3 && !_driver.Fallen)
            {
                _driver.Step();
                error += _driver.TrackingError;
                if (steps < 16)
                    prefix.Add(new { observation = (float[])_driver.Observation.Clone(),
                        action = (float[])_driver.Action.Clone(), qpos_after = _driver.Pose });
                steps++;
            }
            episodes.Add(new { phase = offset, success = !_driver.Fallen, survival_seconds = _driver.Time,
                mean_root_tracking_error_m = error / steps, prefix });
        }
        File.WriteAllText(output, JsonSerializer.Serialize(new { episodes }));
        GD.Print("[MimicStandReplay] completed eight episodes: " + output);
    }

    public override void _ExitTree()
    {
        _driver?.Dispose(); _driver = null;
        _bridge?.Dispose(); _bridge = null;
    }
}

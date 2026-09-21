using System;
using System.Collections.Generic;
using System.IO;
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

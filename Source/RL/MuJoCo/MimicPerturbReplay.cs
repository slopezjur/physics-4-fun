using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Batch replay of the exact versioned Python push protocol.</summary>
public partial class MimicPerturbReplay : Node
{
    public override void _Ready()
    {
        try
        {
            string input = System.Environment.GetEnvironmentVariable("P4F_MIMIC_PUSH_PROTOCOL")
                ?? throw new InvalidOperationException("Set P4F_MIMIC_PUSH_PROTOCOL for batch validation.");
            string output = System.Environment.GetEnvironmentVariable("P4F_MIMIC_REPLAY_OUTPUT")
                ?? throw new InvalidOperationException("Set P4F_MIMIC_REPLAY_OUTPUT for batch validation.");
            using var document = JsonDocument.Parse(File.ReadAllText(input));
            var protocol = document.RootElement;
            if (protocol.GetProperty("schema").GetString() == "mimic_ball_validation_v2")
            {
                MjMimicBallReplay.Run(protocol, output);
                GD.Print("[MimicPerturbReplay] physical-ball validation complete.");
                GetTree().Quit();
                return;
            }
            if (protocol.GetProperty("schema").GetString() != "mimic_push_v1"
                || protocol.GetProperty("force_frame").GetString() != "MuJoCo_world"
                || protocol.GetProperty("application_point").GetString() != "body_COM")
                throw new InvalidOperationException("Unsupported push protocol.");
            var recovery = protocol.GetProperty("recovery");
            var limits = new Dictionary<string, double> {
                ["height_m"] = MjMimicRecovery.MinimumHeight, ["tilt_degrees"] = MjMimicRecovery.MaximumTiltDegrees,
                ["horizontal_speed_m_s"] = MjMimicRecovery.MaximumHorizontalSpeed,
                ["angular_speed_rad_s"] = MjMimicRecovery.MaximumAngularSpeed,
                ["each_foot_load_n"] = MjMimicRecovery.MinimumFootLoad, ["settle_seconds"] = MjMimicRecovery.SettleSeconds };
            if (limits.Any(pair => recovery.GetProperty(pair.Key).GetDouble() != pair.Value))
                throw new InvalidOperationException("Recovery measurement contract mismatch.");
            string model = ProjectSettings.GlobalizePath("res://mujoco_rig/dummy.xml");
            string bundle = protocol.GetProperty("bundle").GetString()!;
            MjMimicReference.VerifyHash(model, protocol.GetProperty("model_sha256").GetString()!);
            MjMimicReference.VerifyHash(Path.Combine(bundle, "stand.onnx"), protocol.GetProperty("actor_sha256").GetString()!);
            MjMimicReference.VerifyHash(Path.Combine(bundle, "stand_reference.npz"), protocol.GetProperty("reference_sha256").GetString()!);
            MjInterop.SetLibraryDirectory("");
            using var bridge = new MjBridge(model);
            using var driver = new MjMimicStandDriver(bridge, model, bundle);
            if (Math.Abs(driver.ControlTime - protocol.GetProperty("control_timestep").GetDouble()) > 1e-8)
                throw new InvalidOperationException("Push timestep mismatch.");
            var episodes = new List<object>();
            foreach (var spec in protocol.GetProperty("cases").EnumerateArray())
            {
                if (spec.GetProperty("body").GetString() != "Chest") throw new InvalidOperationException("Unsupported push body.");
                var force = spec.GetProperty("force").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                if (force.Length != 3) throw new InvalidOperationException("Expected a three-axis force.");
                var push = new MjMimicPush(new Vector3(force[0], force[1], force[2]),
                    spec.GetProperty("start_step").GetInt32(), spec.GetProperty("duration_steps").GetInt32(),
                    spec.GetProperty("trial_steps").GetInt32());
                var trial = new MjMimicPerturbTrial(bridge, driver, push, spec.GetProperty("phase").GetSingle());
                var trace = new List<object>();
                while (driver.StepCount < push.TrialSteps && !driver.Fallen)
                {
                    int step = driver.StepCount;
                    trial.Step();
                    if (step < 2 || step >= push.StartStep - 2 && step < push.StartStep + push.DurationSteps + 12)
                        trace.Add(new { step, force = new[] { trial.AppliedForce.X, trial.AppliedForce.Y, trial.AppliedForce.Z },
                            observation = (float[])driver.Observation.Clone(), action = (float[])driver.Action.Clone(), qpos_after = driver.Pose });
                }
                episodes.Add(new { name = spec.GetProperty("name").GetString(), metrics = trial.Result, trace });
            }
            File.WriteAllText(output, JsonSerializer.Serialize(episodes, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower }));
            GD.Print($"[MimicPerturbReplay] completed {episodes.Count} trials.");
            GetTree().Quit();
        }
        catch (Exception exception) { GD.PrintErr("[MimicPerturbReplay] " + exception); GetTree().Quit(1); }
    }
}

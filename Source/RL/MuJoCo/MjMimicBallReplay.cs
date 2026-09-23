using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Physical-ball replay against native Python's versioned validation cases.</summary>
internal static class MjMimicBallReplay
{
    internal static void Run(JsonElement protocol, string output)
    {
        string model = ProjectSettings.GlobalizePath("res://mujoco_rig/dummy_ball.xml");
        string bundle = protocol.GetProperty("bundle").GetString()!;
        MjMimicReference.VerifyHash(model, protocol.GetProperty("world_model_sha256").GetString()!);
        MjMimicReference.VerifyHash(Path.Combine(bundle, "stand.onnx"), protocol.GetProperty("actor_sha256").GetString()!);
        MjMimicReference.VerifyHash(Path.Combine(bundle, "stand_reference.npz"), protocol.GetProperty("reference_sha256").GetString()!);
        MjInterop.SetLibraryDirectory("");
        using var bridge = new MjBridge(model);
        using var driver = new MjMimicStandDriver(bridge, model, bundle);
        if (Math.Abs(driver.ControlTime - protocol.GetProperty("control_timestep").GetDouble()) > 1e-8)
            throw new InvalidOperationException("Ball replay timestep mismatch.");
        int ball = bridge.BodyId("ball");
        var results = new List<object>();
        foreach (var spec in protocol.GetProperty("cases").EnumerateArray())
        {
            driver.Reset(spec.GetProperty("phase").GetSingle());
            int direction = spec.GetProperty("direction").GetInt32();
            if (direction < 0 || direction > 3) throw new InvalidOperationException("Invalid ball direction.");
            // Match torch's float32 angular construction before converting frames.
            float angle = direction * (float)Math.PI / 2;
            Vector3 heading = MjMimicPush.ToGodot(new(MathF.Cos(angle), MathF.Sin(angle), 0));
            int launch = spec.GetProperty("launch_step").GetInt32();
            int duration = spec.GetProperty("trial_steps").GetInt32();
            var trial = new MjMimicBallTrial(bridge, driver, heading, spec.GetProperty("speed").GetSingle(),
                launch, spec.GetProperty("target_body").GetString()!);
            int firstHit = -1;
            var trace = new List<object>();
            while (!driver.Fallen && driver.StepCount < duration)
            {
                int step = driver.StepCount;
                trial.Step();
                if (firstHit < 0 && driver.BallHit) firstHit = driver.StepCount;
                if (step < 2 || step >= launch - 2 && step < launch + 6
                    || firstHit >= 0 && step >= firstHit - 1 && step < firstHit + 8)
                {
                    Vector3 position = MjBridge.GodotToMj(bridge.BodyTransform(ball).Origin);
                    trace.Add(new { step, observation = (float[])driver.Observation.Clone(),
                        action = (float[])driver.Action.Clone(), qpos_after = driver.Pose,
                        ball_position = new[] { position.X, position.Y, position.Z } });
                }
            }
            results.Add(new { name = spec.GetProperty("name").GetString(), first_hit_step = firstHit,
                survived = !driver.Fallen && driver.StepCount == duration,
                survival_seconds = driver.Time, trace });
        }
        File.WriteAllText(output, JsonSerializer.Serialize(results));
    }
}

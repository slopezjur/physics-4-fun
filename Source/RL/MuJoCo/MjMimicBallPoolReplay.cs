using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Native regression checks for independent Arena projectile launches.</summary>
internal static class MjMimicBallPoolReplay
{
    internal static void Run(JsonElement protocol, string output)
    {
        string model = protocol.GetProperty("model").GetString()!;
        string bundle = protocol.GetProperty("bundle").GetString()!;
        MjInterop.SetLibraryDirectory("");
        using var bridge = new MjBridge(model);
        using var driver = new MjMimicStandDriver(bridge, model, bundle);
        driver.Reset(0);
        using var character = new MjBridge(Path.Combine(bundle, "dummy.xml"));
        character.ResetNativeState(driver.Pose, driver.Velocity);
        double massError = Math.Abs(character.TotalMass() - bridge.TotalMass());
        float comError = character.CenterOfMass().DistanceTo(bridge.CenterOfMass());
        Require(massError < 1e-9 && comError < 1e-6, "Projectiles changed character balance sensors");

        var trial = NewTrial();
        Require(trial.Capacity == 64, "Expected the Arena's 64-ball model");
        Require(trial.Launch(Vector3.Forward), "First launch failed");
        var first = State();
        Require(trial.Launch(Vector3.Right) && trial.Launch(Vector3.Back), "Concurrent launches failed");
        var concurrent = State();
        double preservedError = first.Q.Skip(46).Take(7).Zip(concurrent.Q.Skip(46), (a, b) => Math.Abs(a - b)).Max();
        preservedError = Math.Max(preservedError,
            first.V.Skip(45).Take(6).Zip(concurrent.V.Skip(45), (a, b) => Math.Abs(a - b)).Max());
        Require(preservedError == 0, "A new launch changed an earlier projectile");
        for (int i = 0; i < 3; i++)
        {
            Vector3 velocity = bridge.FreeJointVelocity(bridge.JointId(i == 0 ? "ball_free" : $"ball_free_{i}"));
            Require(Math.Abs(new Vector2(velocity.X, velocity.Z).Length() - 3) < 1e-6,
                "Arena horizontal launch speed must be 3 m/s");
        }
        Require(character.CenterOfMass().DistanceTo(bridge.CenterOfMass()) < 1e-6,
            "Launched projectiles changed character COM");
        for (int i = 0; i < 30 && !driver.Fallen; i++) trial.Step();
        int confirmedHits = trial.ConfirmedHits;
        Require(confirmedHits >= 2, "Concurrent projectile hits were not accounted for independently");

        driver.Reset(0);
        trial = NewTrial();
        for (int i = 0; i < trial.Capacity; i++) Require(trial.Launch(), "Pool filled too early");
        var full = State();
        Require(!trial.Launch() && trial.ShotCount == 64, "Exhausted pool recycled a projectile");
        var rejected = State();
        Require(full.Q.SequenceEqual(rejected.Q) && full.V.SequenceEqual(rejected.V), "Rejected launch mutated physics");
        driver.Reset(0);
        var reset = State();
        for (int i = 0; i < 64; i++)
            Require(reset.Q[46 + 7 * i] >= 40 && reset.Q[48 + 7 * i] == 2
                && reset.V.Skip(45 + 6 * i).Take(6).All(x => x == 0), "Restart did not clear a ball");
        trial = NewTrial();
        Require(trial.Launch() && trial.ShotCount == 1, "Restart did not restore launch capacity");

        File.WriteAllText(output, JsonSerializer.Serialize(new {
            passed = true, capacity = 64, horizontal_speed_m_s = 3, confirmed_concurrent_hits = confirmedHits,
            previous_ball_state_error = preservedError, character_mass_error = massError,
            character_com_error = comError, exhausted_pool_preserves_balls = true, restart_clears_balls = true
        }, new JsonSerializerOptions { WriteIndented = true }));

        MjMimicBallTrial NewTrial() => new(bridge, driver, Vector3.Forward, 3, int.MaxValue, "Chest",
            independentProjectiles: true);
        (double[] Q, double[] V) State()
        {
            var q = new double[bridge.PositionCount]; var v = new double[bridge.VelocityCount];
            bridge.ReadNativeState(q, v);
            return (q, v);
        }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}

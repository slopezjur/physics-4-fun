using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Runs the hash-bound native recovery controller beside the local MuJoCo scene.</summary>
internal sealed class MjRecoveryPolicy : IMjPolicyInference
{
    private readonly Process _process;

    internal MjRecoveryPolicy(string bundle)
    {
        using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(bundle, "recovery.json")));
        var manifest = json.RootElement;
        float sceneMass = manifest.GetProperty("ball_mass_kg").GetSingle();
        float trainedMass = manifest.TryGetProperty("trained_ball_mass_kg", out var training)
            ? training.GetSingle() : 0.25f;
        if (manifest.GetProperty("schema").GetString() != "mimic_recovery_scene_bundle_v1"
            || !float.IsFinite(sceneMass) || sceneMass < 0.1f || sceneMass > 8f
            || trainedMass != 0.25f)
            throw new InvalidOperationException("Unsupported recovery scene bundle");
        foreach (var (file, hash) in new[]
        {
            ("training_report.json", "training_report_sha256"), ("recovery.pt", "checkpoint_sha256"),
            ("motion.npz", "motion_sha256"), ("source.json", "source_sha256"),
            ("search.json", "search_sha256"), ("contract.json", "contract_sha256"),
            ("dummy.xml", "character_sha256"), ("dummy_ball.xml", "ball_world_sha256")
        }) MjMimicReference.VerifyHash(Path.Combine(bundle, file), manifest.GetProperty(hash).GetString()!);
        if (manifest.TryGetProperty("support_schedule_sha256", out var scheduleHash))
            MjMimicReference.VerifyHash(Path.Combine(bundle, "support_schedule.npz"), scheduleHash.GetString()!);

        string root = Godot.ProjectSettings.GlobalizePath("res://");
        string executable = Path.Combine(root, "mujoco_rig", "mimic", ".venv", "Scripts", "python.exe");
        if (!File.Exists(executable))
            throw new FileNotFoundException("Recovery scene requires the local MuJoCo Python environment", executable);
        var start = new ProcessStartInfo(executable)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };
        start.ArgumentList.Add("-m");
        start.ArgumentList.Add("mujoco_rig.mimic.recovery_policy_server");
        start.ArgumentList.Add("--bundle");
        start.ArgumentList.Add(bundle);
        _process = Process.Start(start) ?? throw new InvalidOperationException("Recovery controller did not start");
        try
        {
            if (_process.StandardOutput.ReadLine() != "READY")
                throw new InvalidOperationException("Recovery controller did not become ready");
        }
        catch { Dispose(); throw; }
    }

    internal void Reset()
    {
        _process.StandardInput.WriteLine("RESET");
        if (_process.StandardOutput.ReadLine() != "OK")
            throw new InvalidOperationException("Recovery controller reset failed");
    }

    public void Predict(float[] observation, float[] action)
    {
        if (observation.Length != 362 || action.Length != 30 || observation.Any(x => !float.IsFinite(x)))
            throw new InvalidOperationException("Recovery observation/action contract mismatch");
        _process.StandardInput.WriteLine(JsonSerializer.Serialize(observation));
        string reply = _process.StandardOutput.ReadLine()
            ?? throw new InvalidOperationException("Recovery controller exited during prediction");
        float[] values = JsonSerializer.Deserialize<float[]>(reply)
            ?? throw new InvalidOperationException("Recovery controller returned no action");
        if (values.Length != action.Length || values.Any(x => !float.IsFinite(x)))
            throw new InvalidOperationException("Recovery controller returned an invalid action");
        values.CopyTo(action, 0);
    }

    public void Dispose()
    {
        if (!_process.HasExited) _process.Kill(entireProcessTree: true);
        _process.Dispose();
    }
}

using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Hash-checked, non-looping native-hinge reference for the experimental Stand actor.</summary>
internal sealed class MjMimicReference
{
    private readonly float[][] _positions, _velocities;
    internal float FrameTime { get; }
    internal float Duration => (_positions.Length - 1) * FrameTime;

    internal MjMimicReference(string path, string expectedHash, string referenceHash)
    {
        VerifyHash(path, expectedHash);
        using var json = JsonDocument.Parse(File.ReadAllText(path));
        var root = json.RootElement;
        if (root.GetProperty("reference_sha256").GetString() != referenceHash)
            throw new InvalidOperationException("Reference identity does not match actor");
        FrameTime = root.GetProperty("dt").GetSingle();
        _positions = ReadFrames(root.GetProperty("qpos"), 46);
        _velocities = ReadFrames(root.GetProperty("qvel"), 45);
        if (_positions.Length != _velocities.Length || _positions.Length < 2
            || !float.IsFinite(FrameTime) || FrameTime <= 0 || Duration < 3)
            throw new InvalidOperationException("Invalid standing reference");
    }

    private static float[][] ReadFrames(JsonElement element, int width)
    {
        var frames = element.EnumerateArray().Select(frame =>
            frame.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();
        if (frames.Any(frame => frame.Length != width || frame.Any(x => !float.IsFinite(x))))
            throw new InvalidOperationException("Invalid reference state");
        return frames;
    }

    internal void Sample(float time, double[] qpos, double[] qvel)
    {
        float frame = Math.Clamp(time / FrameTime, 0, _positions.Length - 1);
        int index = Math.Min((int)frame, _positions.Length - 2);
        float alpha = frame - index;
        for (int j = 0; j < qpos.Length; j++)
            qpos[j] = _positions[index][j] + alpha * (_positions[index + 1][j] - _positions[index][j]);
        for (int j = 0; j < qvel.Length; j++)
            qvel[j] = _velocities[index][j] + alpha * (_velocities[index + 1][j] - _velocities[index][j]);
        float norm = MathF.Sqrt((float)(qpos[3] * qpos[3] + qpos[4] * qpos[4]
            + qpos[5] * qpos[5] + qpos[6] * qpos[6]));
        if (norm < 0.5f) throw new InvalidOperationException("Invalid reference quaternion");
        for (int j = 3; j < 7; j++) qpos[j] /= norm;
    }

    internal static Quaternion Rotation(double[] q) => new((float)q[4], (float)q[5], (float)q[6], (float)q[3]);

    internal static void VerifyHash(string path, string expected)
    {
        string actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
        if (!actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Content hash mismatch: " + path);
    }
}

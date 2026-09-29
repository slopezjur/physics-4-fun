using System;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Versioned policy input in Godot coordinates; matches Python capture_distance.</summary>
internal static class MjMimicBalanceObservation
{
    internal const string Contract = "mimic_stationary_capture_target_pd_v2";

    internal static void Validate(JsonElement spec)
    {
        foreach (var (key, value) in new[] {
            ("schema", "instantaneous_loaded_capture_v1"),
            ("velocity", "instantaneous_character_com_world"),
            ("projectile", "excluded"), ("ground_height", "minimum_foot_origin_height") })
            if (spec.GetProperty(key).GetString() != value)
                throw new InvalidOperationException("Unsupported balance observation: " + key);
        foreach (var (key, value) in new[] {
            ("sole_radius_m", .06), ("minimum_foot_load_n", 5.0),
            ("minimum_com_height_above_feet_m", .05), ("unsupported_error_m", 1.0),
            ("gravity_m_s2", 9.81) })
            if (Math.Abs(spec.GetProperty(key).GetDouble() - value) > 1e-9)
                throw new InvalidOperationException("Unsupported balance observation: " + key);
    }

    internal static float Evaluate(Vector3 com, Vector3 velocity, Vector3 left, Vector3 right,
        float leftLoad, float rightLoad)
    {
        bool lc = leftLoad >= 5, rc = rightLoad >= 5;
        if (!lc && !rc) return 1;
        float inverseOmega = MathF.Sqrt(Math.Max(com.Y - Math.Min(left.Y, right.Y), .05f) / 9.81f);
        Vector2 p = new(com.X + velocity.X * inverseOmega, com.Z + velocity.Z * inverseOmega);
        Vector2 a = new(left.X, left.Z), b = new(right.X, right.Z);
        if (!lc) a = b;
        if (!rc) b = a;
        Vector2 segment = b - a;
        float along = Math.Clamp((p - a).Dot(segment) / Math.Max(segment.LengthSquared(), 1e-8f), 0, 1);
        return Math.Max(p.DistanceTo(a + along * segment) - .06f, 0);
    }
}

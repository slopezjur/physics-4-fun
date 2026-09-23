using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

public readonly record struct CapturePointResult(Vector3 CapturePoint, float? StabilityMargin, int SupportedFeet);

/// <summary>
/// Flat-ground inverted-pendulum diagnostic, using a capsule around loaded feet.
/// This approximation does not prove stability or command a recovery step.
/// </summary>
public static class MjCapturePoint
{
    public const float MinimumFootLoad = 5f; // Newtons, including toe contacts.

    public static CapturePointResult Evaluate(Vector3 com, Vector3 velocity, Vector3 left, Vector3 right,
        float leftLoad, float rightLoad, float footRadius = .08f)
    {
        if (!com.IsFinite() || !velocity.IsFinite() || !left.IsFinite() || !right.IsFinite()
            || !float.IsFinite(leftLoad) || !float.IsFinite(rightLoad)
            || !float.IsFinite(footRadius) || footRadius <= 0)
            throw new ArgumentException("Expected finite state, loads in newtons and a positive support radius.");
        float omega = MathF.Sqrt(9.81f / Math.Max(com.Y, .30f));
        Vector3 capture = new(com.X + velocity.X / omega, 0, com.Z + velocity.Z / omega);
        bool lc = leftLoad > MinimumFootLoad, rc = rightLoad > MinimumFootLoad;
        int supported = (lc ? 1 : 0) + (rc ? 1 : 0);
        if (supported == 0) return new(capture, null, 0);
        Vector2 l = new(left.X, left.Z), r = new(right.X, right.Z), p = new(capture.X, capture.Z);
        if (!lc) l = r;
        if (!rc) r = l;
        Vector2 segment = r - l;
        float t = segment.LengthSquared() > 1e-6f
            ? Math.Clamp((p - l).Dot(segment) / segment.LengthSquared(), 0, 1) : 0;
        return new(capture, footRadius - p.DistanceTo(l + t * segment), supported);
    }
}

using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// Fills a walk policy's yaw command the way it was trained: under a heading-hold contract, a zero
/// yaw command becomes a correction toward the heading the command began on.
/// </summary>
/// <remarks>
/// <para>A walk brain trained with the hold expects its yaw slot to carry this correction whenever
/// the commanded yaw is zero; writing the raw 0 runs a different policy from the one that was
/// scored. Contracts written before the hold existed carry no <c>command</c> block, and get the raw
/// command.</para>
/// <para>Mirrors <c>WalkEnvWarp.apply_heading_hold</c>. The heading is MuJoCo's
/// <c>atan2(R[1,0], R[0,0])</c> of the pelvis: under <see cref="MjBridge"/>'s <c>m^T R m</c>
/// conversion the pelvis's MuJoCo x-axis is Godot's -Z, so <c>GodotToMj(basis * (0, 0, -1))</c> is
/// that axis in MuJoCo's world frame - identical over 10,000 random rotations. <c>Mathf.Wrap</c>
/// returns [-pi, pi), the same range as the env's <c>remainder(x + pi, 2 pi) - pi</c>.</para>
/// </remarks>
internal sealed class MjHeadingHold
{
    private bool _latched;
    private float _held;
    private Vector3 _latchedCommand;

    internal MjHeadingHold(float gain) => Gain = gain;

    /// <summary>Commanded turn rate, rad/s, per radian of heading error.</summary>
    internal float Gain { get; }

    /// <summary>Forgets the held heading. Call after the body is reset.</summary>
    internal void Reset() => _latched = false;

    /// <summary>The command the policy was trained to see, for this raw command and pelvis pose.</summary>
    internal Vector3 Apply(Vector3 command, Transform3D pelvis)
    {
        if (command.Z != 0.0f)
        {
            _latched = false;
            return command;
        }

        Vector3 forward = MjBridge.GodotToMj(pelvis.Basis * new Vector3(0.0f, 0.0f, -1.0f));
        float heading = Mathf.Atan2(forward.Y, forward.X);
        if (!_latched || command != _latchedCommand)
        {
            _held = heading;
            _latchedCommand = command;
            _latched = true;
        }

        float error = Mathf.Wrap(_held - heading, -Mathf.Pi, Mathf.Pi);
        return new Vector3(command.X, command.Y, Mathf.Clamp(Gain * error, -1.0f, 1.0f));
    }
}

using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

internal sealed class MjMimicBallTrial
{
    internal static readonly string[] TargetBones =
    {
        "Head", "Chest", "Spine", "Pelvis",
        "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
        "Thigh_L", "Shin_L", "Thigh_R", "Shin_R",
    };

    private readonly MjBridge _bridge;
    private readonly MjMimicStandDriver _driver;
    private readonly int _joint;
    private readonly string _configuredTarget;
    private readonly float _speed;
    private readonly int _launchStep;
    private readonly Random _rng;
    private Vector3 _direction;

    internal bool Launched { get; private set; }
    internal string ActiveTargetName { get; private set; } = string.Empty;

    internal MjMimicBallTrial(MjBridge bridge, MjMimicStandDriver driver, Vector3 direction,
        float speed, int launchStep, string target, int seed = 1)
    {
        if (!direction.IsFinite() || direction.LengthSquared() < .5f || Math.Abs(direction.Y) > 1e-6
            || !float.IsFinite(speed) || speed < 1 || speed > 8 || launchStep < 0)
            throw new ArgumentException("Ball speed must be 1–8 m/s with a horizontal direction and valid launch step.");
        _bridge = bridge; _driver = driver; _direction = direction.Normalized();
        _speed = speed; _launchStep = launchStep;
        _configuredTarget = target;
        _rng = new Random(seed);
        if (bridge != null)
        {
            _joint = MjPolicyObservation.Required(bridge.JointId("ball_free"), "ball_free");
            if (!string.Equals(target, "Random", StringComparison.OrdinalIgnoreCase))
                MjPolicyObservation.Required(bridge.BodyId(target), target);
        }
    }

    internal void Launch(Vector3? overrideDirection = null, string? overrideTarget = null)
    {
        if (_bridge == null) return;
        if (overrideDirection.HasValue)
        {
            Vector3 dir = overrideDirection.Value;
            if (dir.IsFinite() && dir.LengthSquared() >= .5f && Math.Abs(dir.Y) <= 1e-6)
                _direction = dir.Normalized();
        }

        string targetName = overrideTarget ?? (string.Equals(_configuredTarget, "Random", StringComparison.OrdinalIgnoreCase)
            ? TargetBones[_rng.Next(0, TargetBones.Length)]
            : _configuredTarget);

        int targetBodyId = _bridge.BodyId(targetName);
        if (targetBodyId < 0) targetBodyId = _bridge.BodyId("Chest");

        ActiveTargetName = targetName;
        Vector3 target = MjBridge.GodotToMj(_bridge.BodyTransform(targetBodyId).Origin);
        Vector3 direction = MjBridge.GodotToMj(_direction).Normalized();
        var (position, velocity) = BallisticLaunch(target, direction, _speed);
        _bridge.SetFreeJoint(_joint, MjMimicPush.ToGodot(position), MjMimicPush.ToGodot(velocity));
        _bridge.Forward();
        Launched = true;
    }

    internal static (Vector3 Position, Vector3 Velocity) BallisticLaunch(Vector3 target, Vector3 direction, float speed)
    {
        const float distance = .65f;
        Vector3 position = target - direction * distance;
        Vector3 velocity = direction * speed + new Vector3(0, 0, 9.81f * .5f * (distance / speed));
        return (position, velocity);
    }

    internal void Step()
    {
        _driver.Step(_driver.StepCount == _launchStep ? () => Launch() : null);
    }
}

using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

internal sealed class MjMimicBallTrial
{
    private readonly MjBridge _bridge;
    private readonly MjMimicStandDriver _driver;
    private readonly int _joint, _target;
    private readonly Vector3 _direction;
    private readonly float _speed;
    private readonly int _launchStep;
    internal bool Launched { get; private set; }

    internal MjMimicBallTrial(MjBridge bridge, MjMimicStandDriver driver, Vector3 direction,
        float speed, int launchStep, string target)
    {
        if (!direction.IsFinite() || direction.LengthSquared() < .5f || Math.Abs(direction.Z) > 1e-6
            || !float.IsFinite(speed) || speed < 1 || speed > 8 || launchStep < 0)
            throw new ArgumentException("Ball speed must be 1–8 m/s with a horizontal direction and valid launch step.");
        _bridge = bridge; _driver = driver; _direction = direction.Normalized();
        _speed = speed; _launchStep = launchStep;
        _joint = MjPolicyObservation.Required(bridge.JointId("ball_free"), "ball_free");
        _target = MjPolicyObservation.Required(bridge.BodyId(target), target);
    }

    internal void Launch()
    {
        if (Launched) return;
        Vector3 target = MjBridge.GodotToMj(_bridge.BodyTransform(_target).Origin);
        var (position, velocity) = BallisticLaunch(target, _direction, _speed);
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
        if (_driver.StepCount == _launchStep) Launch();
        _driver.Step();
    }
}

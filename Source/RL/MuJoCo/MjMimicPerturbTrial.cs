using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Shared push application and measurement for interactive and batch trials.</summary>
internal sealed class MjMimicPerturbTrial
{
    private readonly MjBridge _bridge;
    private readonly MjMimicStandDriver _driver;
    private readonly int _chest, _left, _right;
    private readonly Vector3[] _loads;
    private readonly MjMimicRecovery _metrics;
    internal MjMimicPush Push { get; }
    internal Vector3 AppliedForce { get; private set; }
    internal MjMimicRecoveryResult Result => _metrics.Result(_driver.Fallen);

    internal MjMimicPerturbTrial(MjBridge bridge, MjMimicStandDriver driver, MjMimicPush push, float phase)
    {
        push.Validate(phase, driver.ControlTime, driver.ReferenceDuration);
        _bridge = bridge; _driver = driver; Push = push;
        _chest = MjPolicyObservation.Required(bridge.BodyId("Chest"), "Chest");
        _left = MjPolicyObservation.Required(bridge.BodyId("Foot_L"), "Foot_L");
        _right = MjPolicyObservation.Required(bridge.BodyId("Foot_R"), "Foot_R");
        _loads = new Vector3[bridge.BodyCount];
        _metrics = new MjMimicRecovery(push, driver.ControlTime);
        driver.Reset(phase);
        Sample();
    }

    internal void Step()
    {
        if (_driver.Fallen || _driver.StepCount >= Push.TrialSteps) return;
        AppliedForce = Push.ForceAt(_driver.StepCount);
        _bridge.SetBodyForce(_chest, MjMimicPush.ToGodot(AppliedForce));
        try { _driver.Step(); }
        finally { _bridge.SetBodyForce(_chest, Vector3.Zero); }
        Sample();
    }

    private void Sample()
    {
        _bridge.ReadNativeGroundForces(_loads);
        _metrics.Sample(_driver.StepCount, _driver.Pose, _driver.Velocity,
            MjBridge.GodotToMj(_bridge.BodyTransform(_left).Origin),
            MjBridge.GodotToMj(_bridge.BodyTransform(_right).Origin),
            Math.Abs(_loads[_left].Z), Math.Abs(_loads[_right].Z));
    }
}

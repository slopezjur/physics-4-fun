using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>World-frame COM force held over integer control intervals, without actuator latency.</summary>
internal sealed record MjMimicPush(Vector3 Force, int StartStep, int DurationSteps, int TrialSteps = 300)
{
    internal Vector3 ForceAt(int step) => step >= StartStep && step < StartStep + DurationSteps ? Force : Vector3.Zero;
    internal void Validate(float phase, float dt, float referenceDuration)
    {
        if (!float.IsFinite(dt) || dt <= 0 || !float.IsFinite(referenceDuration)
            || !Force.IsFinite() || !float.IsFinite(phase) || phase < 0 || phase + TrialSteps * dt > referenceDuration
            || StartStep < 0 || DurationSteps < 1 || (long)StartStep + DurationSteps + Math.Ceiling(0.5 / dt) > TrialSteps)
            throw new ArgumentException("Push must fit in the reference and leave at least half a second for recovery.");
    }
    internal static Vector3 ToGodot(Vector3 native) => new(-native.Y, native.Z, -native.X);
}

internal sealed record MjMimicRecoveryResult(bool Survived, bool Recovered, float? RecoverySeconds,
    float SurvivalSeconds, float MaxHorizontalDisplacementM, float FootTravelM, int ContactSwitches);

/// <summary>Diagnostic settled recovery; this does not modify policy rewards.</summary>
internal sealed class MjMimicRecovery
{
    internal const double MinimumHeight = .75, MaximumTiltDegrees = 15, MaximumHorizontalSpeed = .2,
        MaximumAngularSpeed = 1, MinimumFootLoad = 5, SettleSeconds = .5;
    private readonly MjMimicPush _push;
    private readonly float _dt;
    private Vector2 _origin;
    private Vector3 _previousLeft, _previousRight;
    private bool _leftContact, _rightContact, _initialized;
    private int _lastStep, _stableSteps, _switches;
    private float _displacement, _travel;
    private float? _recovery;

    internal MjMimicRecovery(MjMimicPush push, float dt) { _push = push; _dt = dt; }

    internal void Sample(int step, double[] q, double[] v, Vector3 left, Vector3 right, float leftLoad, float rightLoad)
    {
        _lastStep = step;
        bool lc = leftLoad > MinimumFootLoad, rc = rightLoad > MinimumFootLoad;
        var position = new Vector2((float)q[0], (float)q[1]);
        if (step <= _push.StartStep)
        {
            _origin = position; _previousLeft = left; _previousRight = right;
            _leftContact = lc; _rightContact = rc; _initialized = true;
            return;
        }
        if (!_initialized) throw new InvalidOperationException("Sample before the push first.");
        _displacement = Math.Max(_displacement, position.DistanceTo(_origin));
        _travel += left.DistanceTo(_previousLeft) + right.DistanceTo(_previousRight);
        _switches += (lc != _leftContact ? 1 : 0) + (rc != _rightContact ? 1 : 0);
        _previousLeft = left; _previousRight = right; _leftContact = lc; _rightContact = rc;
        if (step <= _push.StartStep + _push.DurationSteps) return;
        double upZ = 1 - 2 * (q[4] * q[4] + q[5] * q[5]);
        bool settled = q[2] >= MinimumHeight && upZ >= Math.Cos(MaximumTiltDegrees * Math.PI / 180)
            && Math.Sqrt(v[0] * v[0] + v[1] * v[1]) <= MaximumHorizontalSpeed
            && Math.Sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]) <= MaximumAngularSpeed && lc && rc;
        _stableSteps = settled ? _stableSteps + 1 : 0;
        if (!settled) _recovery = null;
        else if (_stableSteps == (int)Math.Ceiling(SettleSeconds / _dt))
            _recovery = (step - _push.StartStep - _push.DurationSteps) * _dt;
    }

    internal MjMimicRecoveryResult Result(bool fallen)
    {
        bool survived = !fallen && _lastStep >= _push.TrialSteps;
        return new(survived, survived && _recovery.HasValue, survived ? _recovery : null,
            _lastStep * _dt, _displacement, _travel, _switches);
    }
}

using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// Gait and perturbation telemetry for a MuJoCo-driven body: strikes, uprightness, single support,
/// flight, travel, and the peak displacement / tilt / recovery time around a disturbance.
/// </summary>
/// <remarks>
/// <para>Separated from <see cref="MujocoDummy"/> because measuring a run and driving one change for
/// different reasons, and seventeen counters interleaved with the physics loop made both harder to
/// read. Every number here is reported in the same terms as the Jolt runs so results stay
/// comparable across engines.</para>
/// <para><b>Uprightness is height, not tilt.</b> The 0.78 m threshold and the 2 cm foot-lift
/// threshold are the ones the Godot/Jolt gait numbers were always measured with; changing them
/// would silently make old and new runs incomparable, which has cost this project a retraction
/// before.</para>
/// </remarks>
internal sealed class MjGaitMetrics
{
    private const float UprightHeight = 0.78f;
    private const float FootLift = 0.02f;

    private readonly MjBridge _bridge;
    private readonly int _pelvis;
    private readonly int _footL;
    private readonly int _footR;
    private readonly float _pushAtSeconds;
    private readonly float _pushSeconds;
    private readonly bool _trackTilt;

    private int _samples;
    private int _uprightSamples;
    private int _singleSupport;
    private int _flight;
    private int _strikes;
    private bool _wasUpL;
    private bool _wasUpR;
    private float _restL;
    private float _restR;
    private Vector2 _startXz;
    private Vector2 _lastXz;

    private Vector2 _prePushXz;
    private bool _pushCaptured;
    private float _peakPushOffset;
    private float _peakTiltRad;
    private double _recoveredAt = -1.0;
    private float _worstTiltDeg;

    internal MjGaitMetrics(MjBridge bridge, int pelvis, int footL, int footR,
                           float pushAtSeconds, float pushSeconds, bool trackTilt)
    {
        _bridge = bridge;
        _pelvis = pelvis;
        _footL = footL;
        _footR = footR;
        _pushAtSeconds = pushAtSeconds;
        _pushSeconds = pushSeconds;
        _trackTilt = trackTilt;
    }

    /// <summary>Worst pelvis tilt seen so far, in degrees.</summary>
    internal float WorstTiltDegrees => _worstTiltDeg;

    /// <summary>Clears every counter, for a scene reset.</summary>
    internal void Reset()
    {
        _samples = _uprightSamples = _singleSupport = _flight = _strikes = 0;
        _wasUpL = _wasUpR = false;
        _pushCaptured = false;
        _peakPushOffset = 0.0f;
        _peakTiltRad = 0.0f;
        _worstTiltDeg = 0.0f;
        _recoveredAt = -1.0;
    }

    /// <summary>One sample of the run. Call once per physics step.</summary>
    internal void Sample(double elapsed)
    {
        if (_footL < 0 || _footR < 0)
        {
            return;
        }

        Vector3 pelvis = _bridge.BodyTransform(_pelvis).Origin;
        if (_samples == 0)
        {
            _startXz = new Vector2(pelvis.X, pelvis.Z);
            _restL = _bridge.BodyTransform(_footL).Origin.Y;
            _restR = _bridge.BodyTransform(_footR).Origin.Y;
        }

        _samples++;
        _lastXz = new Vector2(pelvis.X, pelvis.Z);
        if (pelvis.Y >= UprightHeight)
        {
            _uprightSamples++;
        }

        SamplePerturbation(pelvis, elapsed);

        bool upL = _bridge.BodyTransform(_footL).Origin.Y - _restL > FootLift;
        bool upR = _bridge.BodyTransform(_footR).Origin.Y - _restR > FootLift;
        if (upL ^ upR)
        {
            _singleSupport++;
        }

        if (upL && upR)
        {
            _flight++;
        }

        if (_wasUpL && !upL)
        {
            _strikes++;
        }

        if (_wasUpR && !upR)
        {
            _strikes++;
        }

        _wasUpL = upL;
        _wasUpR = upR;
    }

    /// <summary>
    /// Peak displacement, peak tilt and recovery time around a scripted shove.
    /// </summary>
    /// <remarks>
    /// Measured relative to the pose captured just before the push, so the numbers describe the
    /// RESPONSE rather than the run.
    /// </remarks>
    private void SamplePerturbation(Vector3 pelvis, double elapsed)
    {
        if (_pushAtSeconds <= 0.0f && !_trackTilt)
        {
            return;
        }

        // Tilt is tracked under BALL fire too, not only around a scripted shove.
        float tiltNow = Mathf.Acos(Mathf.Clamp(
            _bridge.BodyTransform(_pelvis).Basis.Y.Dot(Vector3.Up), -1.0f, 1.0f));
        _worstTiltDeg = Math.Max(_worstTiltDeg, Mathf.RadToDeg(tiltNow));

        if (_pushAtSeconds <= 0.0f)
        {
            return;
        }

        if (!_pushCaptured && elapsed >= _pushAtSeconds)
        {
            _prePushXz = new Vector2(pelvis.X, pelvis.Z);
            _pushCaptured = true;
        }

        if (!_pushCaptured)
        {
            return;
        }

        _peakPushOffset = Math.Max(_peakPushOffset,
                                   new Vector2(pelvis.X, pelvis.Z).DistanceTo(_prePushXz));
        _peakTiltRad = Math.Max(_peakTiltRad, tiltNow);

        // **Recovery is regaining POSTURE, not the original spot.** Requiring a return to within
        // 2 cm of where it started reported NEVER for a 75 N.s shove that the body plainly survived
        // upright - it simply ends up 0.159 m displaced, which is what a shoved human does.
        bool settled = tiltNow < 0.05f && pelvis.Y > UprightHeight
                       && _bridge.CenterOfMassVelocity().Length() < 0.05f;
        if (_recoveredAt < 0.0 && settled && elapsed > _pushAtSeconds + _pushSeconds + 0.5)
        {
            _recoveredAt = elapsed - _pushAtSeconds;
        }
    }

    /// <summary>Prints the gait summary in the same terms as the Godot/Jolt runs.</summary>
    internal void Report(double elapsed, float pushForce, MjBallGun? gun, int shotsFired)
    {
        float travel = _lastXz.DistanceTo(_startXz);
        GD.Print($"[MujocoDummy] t={elapsed:F1}s  strikes={_strikes}  "
                 + $"upright={(float)_uprightSamples / Math.Max(1, _samples):P1}  "
                 + $"single={(float)_singleSupport / Math.Max(1, _samples):P1}  "
                 + $"flight={(float)_flight / Math.Max(1, _samples):P1}  "
                 + $"travel={travel:F3}m");

        if (gun != null)
        {
            float pelvisY = _bridge.BodyTransform(_pelvis).Origin.Y;
            int hits = gun.Shots - gun.Misses;
            GD.Print($"[MujocoDummy] BALLGUN {gun.Shots} shots, {hits} hits, {gun.Misses} misses "
                     + $"({(gun.Shots > 0 ? hits * 100.0 / gun.Shots : 0.0):F0}% hit rate), "
                     + $"ball {gun.Mass:F1} kg");
            GD.Print($"[MujocoDummy] BALLGUN {shotsFired} shots  peakTilt="
                     + $"{_worstTiltDeg:F1}deg  pelvisEnd={pelvisY:F3}  "
                     + (pelvisY > UprightHeight ? "SURVIVED" : "FELL"));
        }

        if (_pushAtSeconds > 0.0f)
        {
            float impulse = pushForce * _pushSeconds;
            string rec = _recoveredAt >= 0.0 ? $"{_recoveredAt:F2}s" : "NEVER";
            GD.Print($"[MujocoDummy] PERTURB {pushForce:F0}N x {_pushSeconds:F2}s = {impulse:F0} N.s "
                     + $"| peakOffset={_peakPushOffset:F3}m  "
                     + $"peakTilt={Mathf.RadToDeg(_peakTiltRad):F1}deg  recovered={rec}");
        }
    }
}

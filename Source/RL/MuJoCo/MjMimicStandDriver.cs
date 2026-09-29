using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Versioned experimental Stand inference, separate from shipped policy contracts.</summary>
internal sealed class MjMimicStandDriver : IDisposable
{
    private readonly MjBridge _bridge;
    private readonly MjMimicWorld _world;
    private readonly MjMimicReference _reference;
    private readonly IMjPolicyInference _policy;
    private readonly MjMimicTargetControl? _targetControl;
    private readonly int[] _actuators, _allowedGround;
    private readonly float[] _scales;
    private readonly float[][] _pending = { new float[30], new float[30] };
    private readonly double[] _q = new double[46], _v = new double[45];
    private readonly double[] _targetQ = new double[46], _targetV = new double[45];
    private readonly Vector3[] _forces;
    private readonly int _left, _right;
    private readonly bool _correctedContacts;
    private readonly bool _waitForBallHit;
    private readonly bool _stationaryReference;
    private readonly bool _captureObservation;
    private int _cursor, _steps;
    private int? _recoveryStartStep;
    private float _offset;
    internal float ControlTime { get; }
    internal float Time => _steps * ControlTime;
    internal int StepCount => _steps;
    internal float ReferenceDuration => _reference.Duration;
    internal bool StationaryReference => _stationaryReference;
    internal float[] Observation { get; }
    internal float[] Action { get; } = new float[30];
    internal double[] Pose => (double[])_q.Clone();
    internal double[] Velocity => (double[])_v.Clone();
    internal bool Fallen { get; private set; }
    internal float TrackingError { get; private set; }
    internal bool BallHit { get; private set; }
    internal bool BallContactThisStep { get; private set; }
    internal bool RecoveryStarted => !_waitForBallHit || _recoveryStartStep.HasValue;
    internal float RecoveryTime => _waitForBallHit
        ? _recoveryStartStep.HasValue ? Math.Max(0, _steps - _recoveryStartStep.Value) * ControlTime : 0
        : _offset + Time;
    internal float GroundNormalLoad(int body) => Math.Abs(_forces[body].Z);
    // Observational only: never modifies the exported policy's targets or actions.
    internal CapturePointResult Balance => MjCapturePoint.Evaluate(
        _bridge.CenterOfMass(), _bridge.CenterOfMassVelocity(),
        _bridge.BodyTransform(_left).Origin, _bridge.BodyTransform(_right).Origin,
        Math.Abs(_forces[_left].Z) + Math.Abs(_forces[_allowedGround[2]].Z),
        Math.Abs(_forces[_right].Z) + Math.Abs(_forces[_allowedGround[3]].Z));

    internal MjMimicStandDriver(MjBridge bridge, string modelPath, string bundle, bool waitForBallHit = false)
    {
        _bridge = bridge;
        _waitForBallHit = waitForBallHit;
        int ballId = bridge.BodyId("ball");
        if (ballId >= 0) bridge.ExcludeFromCom = ballId;
        _world = new MjMimicWorld(bridge, modelPath);
        using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(bundle, "contract.json")));
        var c = json.RootElement;
        _correctedContacts = MjMimicContactContract.IsCorrected(c);
        string observationContract = c.GetProperty("observation_contract").GetString()!;
        _captureObservation = observationContract == MjMimicBalanceObservation.Contract;
        _stationaryReference = _captureObservation || observationContract == "mimic_stationary_stand_target_pd_v1";
        if (_captureObservation) MjMimicBalanceObservation.Validate(c.GetProperty("balance_observation"));
        if (_stationaryReference && (c.GetProperty("reference_semantics").GetProperty("phase").GetString() != "constant_zero"
            || c.GetProperty("reference_semantics").GetProperty("target").GetString() != "constant_pose_per_case"))
            throw new InvalidOperationException("Unsupported stationary reference semantics");
        bool targetPd = _correctedContacts || observationContract == "mimic_stand_target_pd_v1" || _stationaryReference;
        int numObs = _captureObservation ? 363 : targetPd ? 362 : 300;
        if ((!targetPd && observationContract != "mimic_stand_v1")
            || c.GetProperty("num_obs").GetInt32() != numObs || c.GetProperty("latency_steps").GetInt32() != 2
            || c.GetProperty("decimation").GetInt32() != 4
            || c.GetProperty("action_mode").GetString() != (targetPd ? "reference_relative_target_pd" : "normalized_motor_torque"))
            throw new InvalidOperationException("Unsupported Mimic Stand contract");
        Observation = new float[numObs];
        _targetControl = targetPd ? new MjMimicTargetControl(c) : null;
        var names = new[] { "root_position_world", "root_rotation_tangent_normal", "root_linear_velocity_world",
            "root_angular_velocity_world", "all_hinge_positions", "all_hinge_velocities_scaled_0.1",
            "foot_normal_load_kN_L_R", "pending_actions_oldest_first", "future_reference_poses_1_2_3", "reference_phase" };
        var widths = new[] { 3, 6, 3, 3, 39, 39, 2, 60, 144, 1 };
        if (targetPd) { names[7] = "pending_targets_q_v_valid_oldest_first"; widths[7] = 122; }
        if (_correctedContacts) names[6] = "foot_and_toe_normal_load_kN_L_R";
        if (_stationaryReference) names[9] = "stationary_reference_phase_zero";
        if (_captureObservation)
        {
            names = names.Append("loaded_capture_error_m").ToArray();
            widths = widths.Append(1).ToArray();
        }
        var layout = c.GetProperty("observation_layout");
        if (layout.GetArrayLength() != names.Length) throw new InvalidOperationException("Unknown observation layout");
        int offset = 0;
        for (int i = 0; i < names.Length; i++)
        {
            if (layout[i].GetProperty("name").GetString() != names[i]
                || layout[i].GetProperty("offset").GetInt32() != offset
                || layout[i].GetProperty("width").GetInt32() != widths[i])
                throw new InvalidOperationException("Unknown observation layout");
            offset += widths[i];
        }
        MjMimicReference.VerifyHash(_world.CharacterPath, c.GetProperty("model_sha256").GetString()!);
        double timestep = new MjModelDefinition(modelPath).Timestep;
        ControlTime = c.GetProperty("control_timestep").GetSingle();
        if (Math.Abs(c.GetProperty("physics_timestep").GetDouble() - timestep) > 1e-9
            || Math.Abs(ControlTime - timestep * 4) > 1e-8)
            throw new InvalidOperationException("Mimic timestep mismatch");
        _scales = c.GetProperty("torque_scale_nm").EnumerateArray().Select(x => x.GetSingle()).ToArray();
        _actuators = c.GetProperty("actuator_names").EnumerateArray()
            .Select(x => MjPolicyObservation.Required(bridge.ActuatorId(x.GetString()!), x.GetString()!)).ToArray();
        if (_actuators.Length != 30 || _scales.Length != 30 || _scales.Any(x => !float.IsFinite(x) || x <= 0))
            throw new InvalidOperationException("Invalid Mimic actuator contract");
        if (!_actuators.SequenceEqual(c.GetProperty("actuator_indices").EnumerateArray().Select(x => x.GetInt32())))
            throw new InvalidOperationException("Actuator ordering mismatch");
        int joint = 1;
        foreach (var name in c.GetProperty("joint_names").EnumerateArray())
            if (bridge.JointId(name.GetString()!) != joint++) throw new InvalidOperationException("Hinge ordering mismatch");
        if (joint != 40) throw new InvalidOperationException("Unexpected hinge count");
        _targetControl?.ValidateJointMapping(c.GetProperty("actuator_names").EnumerateArray()
            .Select(x => bridge.JointId(x.GetString()!)).ToArray());
        _reference = new MjMimicReference(Path.Combine(bundle, "reference_frames.json"),
            c.GetProperty("reference_frames_sha256").GetString()!, c.GetProperty("reference_sha256").GetString()!);
        if (Math.Abs(_reference.Duration - c.GetProperty("reference_duration").GetSingle()) > 1e-5)
            throw new InvalidOperationException("Reference duration mismatch");
        _forces = new Vector3[bridge.BodyCount];
        _left = bridge.BodyId("Foot_L"); _right = bridge.BodyId("Foot_R");
        _allowedGround = new[] { _left, _right, bridge.BodyId("Toe_L"), bridge.BodyId("Toe_R") };
        string recovery = Path.Combine(bundle, "recovery.json");
        if (_waitForBallHit && !File.Exists(recovery))
            throw new InvalidOperationException("Impact-triggered stepping requires a recovery controller");
        if (File.Exists(recovery)) _policy = new MjRecoveryPolicy(bundle);
        else
        {
            string actor = Path.Combine(bundle, "stand.onnx");
            MjMimicReference.VerifyHash(actor, c.GetProperty("actor_sha256").GetString()!);
            _policy = new MjOnnxPolicy(File.ReadAllBytes(actor), numObs, 30);
        }
    }

    internal void Reset(float offset)
    {
        if (!float.IsFinite(offset) || offset < 0 || offset > _reference.Duration - 3 + 1e-5
            || (_stationaryReference && offset != 0))
            throw new ArgumentOutOfRangeException(nameof(offset));
        if (_waitForBallHit && offset != 0)
            throw new ArgumentException("Impact-triggered recovery starts at reference phase zero", nameof(offset));
        _offset = offset; _steps = _cursor = 0; _recoveryStartStep = null;
        Fallen = false; TrackingError = 0; BallHit = false; BallContactThisStep = false;
        foreach (var pending in _pending) Array.Clear(pending);
        _targetControl?.Reset();
        if (_policy is MjRecoveryPolicy recovery) recovery.Reset();
        Array.Clear(Action);
        _reference.Sample(offset, _q, _v);
        _world.Reset(_q, _v);
        _bridge.ReadNativeGroundForces(_forces);
    }

    internal void Observe()
    {
        _world.Read(_q, _v);
        if (!_correctedContacts) _bridge.ReadNativeGroundForces(_forces);
        int index = 0;
        Put(_q, 0, 3, ref index);
        PutRotation(_q, ref index);
        Put(_v, 0, 3, ref index);
        Put(MjMimicReference.Rotation(_q) * new Vector3((float)_v[3], (float)_v[4], (float)_v[5]), ref index);
        Put(_q, 7, 39, ref index);
        Put(_v, 6, 39, ref index, 0.1f);
        Observation[index++] = Math.Abs(_forces[_left].Z) * 0.001f
            + (_correctedContacts ? Math.Abs(_forces[_allowedGround[2]].Z) * 0.001f : 0);
        Observation[index++] = Math.Abs(_forces[_right].Z) * 0.001f
            + (_correctedContacts ? Math.Abs(_forces[_allowedGround[3]].Z) * 0.001f : 0);
        if (_targetControl != null) _targetControl.Observe(Observation, ref index);
        else
            for (int slot = 0; slot < 2; slot++)
                foreach (float value in _pending[(_cursor + slot) % 2]) Observation[index++] = value;
        float phase = _stationaryReference ? 0 : RecoveryTime;
        for (int future = 1; future <= 3; future++)
        {
            _reference.Sample(phase + future * ControlTime, _targetQ, _targetV);
            Put(_targetQ, 0, 3, ref index); PutRotation(_targetQ, ref index);
            Put(_targetQ, 7, 39, ref index);
        }
        Observation[index++] = _stationaryReference ? 0 : phase / _reference.Duration;
        if (_captureObservation)
            Observation[index++] = MjMimicBalanceObservation.Evaluate(
                _bridge.CenterOfMass(), _bridge.CenterOfMassVelocity(),
                _bridge.BodyTransform(_left).Origin, _bridge.BodyTransform(_right).Origin,
                Observation[93] * 1000, Observation[94] * 1000);
        if (index != Observation.Length || Observation.Any(x => !float.IsFinite(x)))
            throw new InvalidOperationException("Invalid Mimic observation");
    }

    internal void Step(System.Action? beforePhysics = null)
    {
        BallContactThisStep = false;
        Observe();
        _policy.Predict(Observation, Action);
        if (Action.Any(x => !float.IsFinite(x))) throw new InvalidOperationException("Nonfinite Mimic action");
        _bridge.ClearControls();
        for (int i = 0; i < 30; i++)
        {
            // Float multiplication matches the training control tensor before native-double storage.
            if (_targetControl == null) _bridge.SetControl(_actuators[i], _pending[_cursor][i] * _scales[i]);
            Action[i] = Math.Clamp(Action[i], -1, 1);
            _pending[_cursor][i] = Action[i];
        }
        if (_targetControl != null)
        {
            _reference.Sample(RecoveryTime, _targetQ, _targetV);
            _targetControl.Enqueue(_targetQ, _targetV, Action);
        }
        _cursor = (_cursor + 1) % 2;
        // Match Python: observe/predict first, then apply a scheduled disturbance.
        // mj_forward during a ball launch can already change contact-load sensors.
        beforePhysics?.Invoke();
        for (int substep = 0; substep < 4; substep++)
        {
            if (_targetControl != null)
            {
                _world.Read(_q, _v);
                for (int i = 0; i < 30; i++) _bridge.SetControl(_actuators[i], _targetControl.Torque(i, _q, _v, _scales[i]));
            }
            _bridge.Step();
            int ball = _bridge.BodyId("ball");
            if (ball >= 0 && _bridge.HasCharacterContact(ball, _world.CharacterBodyCount))
            {
                BallContactThisStep = true;
                BallHit = true;
            }
        }
        if (_correctedContacts) _bridge.ReadNativeGroundForces(_forces);
        _bridge.Forward();
        _steps++;
        if (_waitForBallHit && BallHit && !_recoveryStartStep.HasValue)
        {
            _recoveryStartStep = _steps;
            if (_policy is MjRecoveryPolicy recovery) recovery.Reset();
        }
        _world.Read(_q, _v);
        if (!_correctedContacts) _bridge.ReadNativeGroundForces(_forces);
        Fallen = _q[2] < 0.65;
        for (int body = 1; body < _world.CharacterBodyCount; body++)
        {
            Vector3 force = _forces[body].Abs();
            if (!_allowedGround.Contains(body) && Math.Max(force.X, Math.Max(force.Y, force.Z)) > 1)
                Fallen = true;
        }
        _reference.Sample(RecoveryTime, _targetQ, _targetV);
        TrackingError = new Vector3((float)(_q[0] - _targetQ[0]), (float)(_q[1] - _targetQ[1]),
            (float)(_q[2] - _targetQ[2])).Length();
    }

    private void Put(double[] values, int offset, int count, ref int index, float scale = 1)
    {
        for (int i = 0; i < count; i++) Observation[index++] = (float)values[offset + i] * scale;
    }
    private void Put(Vector3 value, ref int index)
    {
        Observation[index++] = value.X; Observation[index++] = value.Y; Observation[index++] = value.Z;
    }
    private void PutRotation(double[] q, ref int index)
    {
        Quaternion rotation = MjMimicReference.Rotation(q);
        Put(rotation * Vector3.Right, ref index); Put(rotation * Vector3.Back, ref index);
    }
    public void Dispose() => _policy.Dispose();
}

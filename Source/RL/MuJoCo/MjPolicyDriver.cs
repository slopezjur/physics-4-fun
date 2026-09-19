using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Coordinates observation, inference and delayed actuation at the trained policy rate.</summary>
internal sealed class MjPolicyDriver : IDisposable
{
    private readonly IMjPolicyPlant _state;
    private readonly MjPolicyContract _contract;
    private readonly IMjPolicyInference _inference;
    private readonly MjPolicyObservation _observation;
    private readonly int[] _actuators;
    private readonly float[] _obs;
    private readonly float[] _previous;
    private readonly float[] _prediction;
    private readonly float[][] _pending;
    private float[] _applied;
    private int _queueIndex;
    private int _sinceInference;
    private bool _disposed;

    // Ownership of inference transfers only after construction succeeds.
    internal MjPolicyDriver(IMjPolicyPlant state, MjPolicyContract contract, IMjPolicyInference inference)
    {
        _state = state;
        _contract = contract;
        _observation = new MjPolicyObservation(state, contract);
        _actuators = contract.Joints.Select(n =>
            MjPolicyObservation.Required(state.ActuatorId(n), n)).ToArray();
        _obs = new float[contract.NumObs];
        _previous = new float[contract.NumActions];
        _prediction = new float[contract.NumActions];
        _applied = new float[contract.NumActions];
        _pending = Enumerable.Range(0, contract.ActionLatencySteps)
            .Select(_ => new float[contract.NumActions]).ToArray();
        _inference = inference;
    }

    internal static MjPolicyDriver Load(IMjPolicyPlant state, string path, double timestep)
    {
        using var file = Godot.FileAccess.Open(Path.ChangeExtension(path, ".contract.json"),
            Godot.FileAccess.ModeFlags.Read)
            ?? throw new InvalidOperationException("Missing policy contract for " + path);
        using var document = JsonDocument.Parse(file.GetAsText());
        var contract = new MjPolicyContract(document.RootElement);
        if (Math.Abs(contract.SimTimestep - timestep) > 1e-9)
            throw new InvalidOperationException("Model timestep differs from the policy contract.");
        using var model = Godot.FileAccess.Open(path, Godot.FileAccess.ModeFlags.Read)
            ?? throw new InvalidOperationException("Could not open policy " + path);
        var inference = new MjOnnxPolicy(model.GetBuffer((long)model.GetLength()), contract);
        try { return new MjPolicyDriver(state, contract, inference); }
        catch { inference.Dispose(); throw; }
    }

    internal Vector3 Command { get; set; }
    internal string Task => _contract.Task;

    internal void Step()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_sinceInference == 0)
        {
            _observation.Write(_obs, _previous, Command);
            _inference.Predict(_obs, _prediction);
            if (_prediction.Any(v => !float.IsFinite(v)))
                throw new InvalidOperationException("Policy produced a non-finite action.");
            for (int i = 0; i < _prediction.Length; i++)
                _previous[i] = Math.Clamp(_prediction[i], -1, 1);

            if (_pending.Length == 0)
                _previous.CopyTo(_applied, 0);
            else
            {
                // A fixed ring stores commands waiting through the neuromuscular delay.
                (_applied, _pending[_queueIndex]) = (_pending[_queueIndex], _applied);
                _previous.CopyTo(_pending[_queueIndex], 0);
                _queueIndex = (_queueIndex + 1) % _pending.Length;
            }
            _sinceInference = _contract.Decimation;
        }
        _sinceInference--;
        for (int i = 0; i < _actuators.Length; i++)
            _state.SetControl(_actuators[i], _contract.Control(i, _applied[i]));
    }

    /// <summary>Clears the complete episode state, including actions waiting in the delay queue.</summary>
    internal void Reset()
    {
        _observation.Reset();
        Array.Clear(_previous);
        Array.Clear(_prediction);
        Array.Clear(_applied);
        foreach (float[] pending in _pending) Array.Clear(pending);
        _queueIndex = _sinceInference = 0;
        foreach (int actuator in _actuators) _state.SetControl(actuator, 0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _inference.Dispose();
    }
}

using System;
using System.Linq;
using System.Text.Json;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Delayed reference targets with fresh, bounded PD feedback at every physics step.</summary>
internal sealed class MjMimicTargetControl
{
    private readonly int[] _qIndices, _vIndices;
    private readonly float[] _lower, _upper;
    private readonly float _residual, _gain, _damping;
    private readonly float[][] _pending = { new float[61], new float[61] };
    private readonly float[] _active = new float[61];
    private int _cursor;

    internal MjMimicTargetControl(JsonElement contract)
    {
        _qIndices = contract.GetProperty("target_qpos_indices").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        _vIndices = contract.GetProperty("target_qvel_indices").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        _lower = Floats(contract, "target_lower_radians");
        _upper = Floats(contract, "target_upper_radians");
        _residual = contract.GetProperty("residual_radians").GetSingle();
        _gain = contract.GetProperty("pd_gain_per_torque_limit").GetSingle();
        _damping = contract.GetProperty("pd_damping_time").GetSingle();
        if (_qIndices.Length != 30 || _vIndices.Length != 30 || _lower.Length != 30 || _upper.Length != 30
            || _qIndices.Distinct().Count() != 30 || _vIndices.Distinct().Count() != 30
            || _qIndices.Any(x => x < 7 || x >= 46) || _vIndices.Any(x => x < 6 || x >= 45)
            || !float.IsFinite(_residual) || _residual <= 0 || !float.IsFinite(_gain) || _gain <= 0
            || !float.IsFinite(_damping) || _damping < 0
            || Enumerable.Range(0, 30).Any(i => !float.IsFinite(_lower[i]) || !float.IsFinite(_upper[i])
                || _lower[i] >= _upper[i] || _qIndices[i] != _vIndices[i] + 1)
            || contract.GetProperty("target_timing").GetString() != "sample_at_issue_hold_after_delay"
            || contract.GetProperty("initial_motor_control").GetString() != "zero_until_valid_target"
            || contract.GetProperty("pd_update").GetString() != "every_physics_step"
            || contract.GetProperty("feedforward").GetString() != "none")
            throw new InvalidOperationException("Unsupported target-PD control contract");
    }

    internal void ValidateJointMapping(int[] jointIds)
    {
        if (!jointIds.Select(id => id + 6).SequenceEqual(_qIndices))
            throw new InvalidOperationException("Target-PD actuator/joint mapping mismatch");
    }

    internal void Reset()
    {
        foreach (var slot in _pending) Array.Clear(slot);
        Array.Clear(_active);
        _cursor = 0;
    }

    internal void Observe(float[] observation, ref int index)
    {
        for (int slot = 0; slot < 2; slot++)
            foreach (float value in _pending[(_cursor + slot) % 2]) observation[index++] = value;
    }

    internal void Enqueue(double[] referenceQ, double[] referenceV, float[] action)
    {
        Array.Copy(_pending[_cursor], _active, 61);
        var target = _pending[_cursor];
        for (int i = 0; i < 30; i++)
        {
            target[i] = Math.Clamp((float)referenceQ[_qIndices[i]] + action[i] * _residual, _lower[i], _upper[i]);
            target[i + 30] = (float)referenceV[_vIndices[i]];
        }
        target[60] = 1;
        _cursor = (_cursor + 1) % 2;
    }

    internal float Torque(int motor, double[] q, double[] v, float limit)
    {
        float error = _active[motor] - (float)q[_qIndices[motor]];
        float velocityError = _active[motor + 30] - (float)v[_vIndices[motor]];
        return Math.Clamp(_gain * error + (_gain * _damping) * velocityError, -1, 1) * limit * _active[60];
    }

    private static float[] Floats(JsonElement c, string name) =>
        c.GetProperty(name).EnumerateArray().Select(x => x.GetSingle()).ToArray();
}

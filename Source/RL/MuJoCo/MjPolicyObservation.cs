using System;
using System.Linq;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Writes supported observation channels at the offsets declared by the contract.</summary>
internal sealed class MjPolicyObservation
{
    private readonly IMjPolicyState _state;
    private readonly MjPolicyContract _contract;
    private readonly int _pelvis, _footL, _footR;
    private readonly int[] _joints;
    private readonly MjHeadingHold? _heading;

    internal MjPolicyObservation(IMjPolicyState state, MjPolicyContract contract)
    {
        _state = state;
        _contract = contract;
        _pelvis = Required(state.BodyId("Pelvis"), "Pelvis");
        _footL = Required(state.BodyId("Foot_L"), "Foot_L");
        _footR = Required(state.BodyId("Foot_R"), "Foot_R");
        _joints = contract.Joints.Select(n => Required(state.JointId(n), n)).ToArray();
        _heading = contract.HeadingGain is float gain ? new MjHeadingHold(gain) : null;
    }

    internal void Reset() => _heading?.Reset();

    internal void Write(float[] destination, float[] previousAction, Vector3 command)
    {
        Transform3D pelvis = _state.BodyTransform(_pelvis);
        Basis inverse = pelvis.Basis.Inverse();
        foreach (var channel in _contract.Channels)
        {
            Span<float> output = destination.AsSpan(channel.Offset, channel.Width);
            switch (channel.Name)
            {
                case "projected_gravity": Put(output, MjBridge.GodotToMj(inverse * Vector3.Down)); break;
                // Keep cvel semantics: the shipped policies were trained on this reference point.
                case "pelvis_linear_velocity": Put(output, MjBridge.GodotToMj(inverse * _state.BodySpatialLinearVelocity(_pelvis))); break;
                case "pelvis_angular_velocity": Put(output, MjBridge.GodotToMj(inverse * _state.BodyAngularVelocity(_pelvis))); break;
                case "pelvis_height": output[0] = pelvis.Origin.Y; break;
                case "joint_position":
                    for (int i = 0; i < _joints.Length; i++) output[i] = (float)_state.JointPosition(_joints[i]);
                    break;
                case "joint_velocity_scaled_0.1":
                    for (int i = 0; i < _joints.Length; i++) output[i] = (float)_state.JointVelocity(_joints[i]) * 0.1f;
                    break;
                case "foot_contact_L_R":
                    output[0] = _state.BodyTransform(_footL).Origin.Y < 0.05f ? 1 : 0;
                    output[1] = _state.BodyTransform(_footR).Origin.Y < 0.05f ? 1 : 0;
                    break;
                case "previous_action": previousAction.AsSpan().CopyTo(output); break;
                case "command_vx_vy_yaw":
                    Put(output, _contract.AcceptsCommand
                        ? _heading?.Apply(command, pelvis) ?? command : Vector3.Zero);
                    break;
            }
        }
        // Match both training backends' observation sanitation and clipping.
        for (int i = 0; i < destination.Length; i++)
            destination[i] = float.IsFinite(destination[i]) ? Math.Clamp(destination[i], -100, 100) : 0;
    }

    internal static int Required(int id, string name) => id >= 0 ? id
        : throw new InvalidOperationException("Policy model is missing '" + name + "'.");

    private static void Put(Span<float> output, Vector3 v)
    {
        output[0] = v.X; output[1] = v.Y; output[2] = v.Z;
    }
}

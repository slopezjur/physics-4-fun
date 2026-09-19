using System;
using System.Collections.Generic;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// The pre-RL controller: an open-loop gait oscillator, a lateral balance term, and an external
/// attitude torque on the pelvis.
/// </summary>
/// <remarks>
/// <para><b>This is scaffolding, not a controller.</b> It demonstrates that the plant can walk - it
/// did, 4.835 m in 40 s at 100% uprightness - and it is what a trained policy REPLACES. Nothing
/// here corrects gait phase against accumulated drift, so it walks for about 60 s and then falls.
/// The balance torque is worth naming separately: it writes up to 300 N.m of EXTERNAL torque onto
/// the pelvis, which no character actually has, and every RL result is measured with it off.</para>
/// <para>It only applies to a POSITION-actuated model. Every control it writes is a joint target in
/// radians, and against the `motor` actuators the current plant uses those numbers would be read as
/// newton-metres - <see cref="MujocoDummy"/> disables it rather than letting that happen quietly.
/// It is kept because `build_mjcf.ACTUATOR_MODE = "position"` can still produce that plant, and
/// because it is the only thing that moves the body with no policy at all.</para>
/// <para>Extracted from <see cref="MujocoDummy"/>: a legacy controller and the node that hosts the
/// simulation change for entirely different reasons, and interleaving them meant reading past
/// oscillator constants to follow the physics loop.</para>
/// </remarks>
internal sealed class MjScriptedController
{
    private const double BalanceGain = 600.0;
    private const double BalanceDamping = 20.0;
    private const double BalanceMaxTorque = 300.0;

    private readonly MjBridge _bridge;
    private readonly IReadOnlyDictionary<string, int> _actuators;
    private readonly int _pelvis;

    internal MjScriptedController(MjBridge bridge, IReadOnlyDictionary<string, int> actuators,
                                  int pelvis)
    {
        _bridge = bridge;
        _actuators = actuators;
        _pelvis = pelvis;
    }

    /// <summary>Hip-roll gain against lateral centre-of-mass velocity.</summary>
    internal float LateralDamping { get; set; } = 0.05f;

    /// <summary>Ceiling on the lateral correction, in radians of hip roll.</summary>
    internal float MaxLateralCorrection { get; set; } = 0.12f;

    internal float GaitFrequency { get; set; } = 0.6f;
    internal float ShiftAmplitude { get; set; } = 0.25f;
    internal float LiftAmplitude { get; set; } = -0.4f;
    internal float KneeAmplitude { get; set; } = 0.4f;

    /// <summary>The measured oscillator: weight shift leading the swing by a quarter cycle.</summary>
    internal void DriveGait(double t, bool walk)
    {
        _bridge.ClearControls();
        if (!walk)
        {
            // Standing still is the same controller with the gait switched off: the balance term
            // alone holds the body, which is what STAND measures.
            _bridge.SetControl(_actuators["Thigh_L_rz"], Balance());
            _bridge.SetControl(_actuators["Thigh_R_rz"], Balance());
            return;
        }

        double w = 2.0 * Math.PI * GaitFrequency * t;
        double warm = Math.Min(1.0, t / 1.0);
        double shift = ShiftAmplitude * Math.Sin(w) * warm;
        double swing = -Math.Cos(w) * warm;

        _bridge.SetControl(_actuators["Thigh_L_rz"], shift + Balance());
        _bridge.SetControl(_actuators["Thigh_R_rz"], -shift + Balance());
        _bridge.SetControl(_actuators["Thigh_L_rx"], LiftAmplitude * Math.Max(0.0, swing));
        _bridge.SetControl(_actuators["Shin_L_rx"], KneeAmplitude * Math.Max(0.0, swing));
        _bridge.SetControl(_actuators["Thigh_R_rx"], LiftAmplitude * Math.Max(0.0, -swing));
        _bridge.SetControl(_actuators["Shin_R_rx"], KneeAmplitude * Math.Max(0.0, -swing));
    }

    /// <summary>
    /// Symmetric hip-roll term that bleeds off lateral centre-of-mass momentum.
    /// </summary>
    /// <remarks>
    /// The Python controller damps <c>cvel[1]</c>, MuJoCo's lateral axis. Under the frame map that
    /// axis is Godot's X negated (<c>godot.x = -mj.y</c>), so damping MuJoCo's +y is damping Godot's
    /// -X, and the sign flips once here rather than being carried implicitly.
    /// </remarks>
    private double Balance()
    {
        double extra = LateralDamping * _bridge.CenterOfMassVelocity().X;
        return Math.Clamp(extra, -MaxLateralCorrection, MaxLateralCorrection);
    }

    /// <summary>Restores the pelvis toward upright, matching both engines' attitude controller.</summary>
    internal void ApplyBalanceTorque()
    {
        if (_pelvis < 0)
        {
            return;
        }

        Basis basis = _bridge.BodyTransform(_pelvis).Basis;
        Vector3 up = basis.Y;                       // the pelvis's own up axis, in Godot's frame
        Vector3 axis = up.Cross(Vector3.Up);        // rotation that would restore it
        Vector3 omega = _bridge.BodyAngularVelocity(_pelvis);
        Vector3 tau = ((float)BalanceGain * axis) - ((float)BalanceDamping * omega);
        if (tau.Length() > BalanceMaxTorque)
        {
            tau = tau.Normalized() * (float)BalanceMaxTorque;
        }

        _bridge.SetBodyTorque(_pelvis, tau);
    }
}

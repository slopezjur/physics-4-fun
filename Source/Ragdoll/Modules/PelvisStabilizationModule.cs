using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Pelvis Protected Balance Region (Euphoria DMS): the pelvis is the unactuated skeletal root, so
/// it is stabilized directly with an attitude PD torque; the counter-torque is distributed across
/// the grounded feet as reaction against the ground, keeping the interaction internal (SRP).
/// </summary>
public class PelvisStabilizationModule : IBalanceStrategy
{
    public float Gain { get; set; } = 600.0f;
    public float Damping { get; set; } = 20.0f;
    public float MaxTorque { get; set; } = 300.0f;

    public Vector3 LastTorque { get; private set; } = Vector3.Zero;

    // EMA smoothing factor for the pelvis stabilizer's angular velocity (per 120 Hz tick)
    private const float AngularVelocityFilterAlpha = 0.25f;
    private Vector3 _filteredAngularVelocity = Vector3.Zero;

    public void Reset()
    {
        _filteredAngularVelocity = Vector3.Zero;
    }

    /// <summary>Zeroes only the reported torque telemetry, leaving the EMA filter state untouched.</summary>
    public void ClearTorque()
    {
        LastTorque = Vector3.Zero;
    }

    /// <summary>
    /// Attitude torque cap (N·m) while rising from the ground. Deliberately well under
    /// <see cref="MaxTorque"/>: during a get-up this assists the limbs in bringing the pelvis
    /// upright, it does not drive the motion. If the rise only works with this raised, the leg
    /// drive is not actually working.
    /// </summary>
    public float RecoveryMaxTorque { get; set; } = 120.0f;

    /// <summary>
    /// Send the stabiliser's reaction into the grounded THIGHS instead of the grounded feet.
    ///
    /// <para><b>For sim-to-sim only; the game keeps the feet.</b> Isaac reproduces this module but
    /// reacts into the thighs, because a Newton foot cannot hold a raw body torque - 0.0009 kg.m^2
    /// against the pelvis's 0.0285, so the 300 N.m cap spins it hundreds of degrees per tick. The
    /// two engines therefore run different reaction paths, and it shows: measured 2026-09-04 with
    /// zero action, turning the assist on moves Godot's total joint deviation 0.640 -> 1.326 while
    /// Isaac's moves 0.512 -> 0.390. Same gains (600/20/300/0.25), opposite effect - Godot's assist
    /// disturbs the body and Isaac's settles it, and a policy trained against the calm one meets
    /// the noisy one in Godot.
    /// </para>
    /// </summary>
    public bool ReactIntoThighs { get; set; }

    /// <summary>Limbs that can carry the stabilizer's reaction into the ground, in preference order.</summary>
    private readonly List<IBoneState> _reactionLimbs = new();

    public void Apply(in BalanceContext context)
    {
        // Active during upright balance, and during the two get-up stages where the body is
        // genuinely rising over a support leg. The pelvis is the unactuated skeletal root, so
        // without this it has no attitude control at all - measured as exactly 0 N·m of pelvis
        // torque on every frame of every recovery dump.
        bool upright = context.State == RagdollState.Balanced || context.State == RagdollState.Stumbling;
        bool rising = context.State == RagdollState.Recovering
                      && (context.GetUpPhase == Recovery.GetUpPhase.HalfKneelRise
                          || context.GetUpPhase == Recovery.GetUpPhase.StandUp);

        // Unlike a grounding/strength gate failure below, this outer gate does NOT zero LastTorque —
        // it leaves the previous tick's value untouched.
        if (!upright && !rising)
        {
            return;
        }

        LastTorque = Vector3.Zero;

        if (context.Strength <= 0.01f)
        {
            return;
        }

        // The reaction has to go somewhere real, or this becomes angular momentum from nothing.
        // While upright that means the grounded feet; while rising it means whichever limbs are
        // actually touching the world, which prone is the forearms rather than any sole-down foot.
        CollectReactionLimbs(context, rising);
        if (_reactionLimbs.Count == 0)
        {
            return;
        }

        IBoneState pelvis = context.Pelvis;

        // Axis-angle attitude error between pelvis up axis and world up (magnitude ~ sin(angle))
        Vector3 pelvisUp = pelvis.GlobalTransform.Basis.Y.Normalized();
        Vector3 attitudeError = pelvisUp.Cross(Vector3.Up);

        // Low-pass filter the pelvis angular velocity: joint reaction chatter (~20 rad/s at 120 Hz)
        // would otherwise dominate the damping term and pump energy through the grounded feet
        _filteredAngularVelocity += (pelvis.AngularVelocity - _filteredAngularVelocity) * AngularVelocityFilterAlpha;

        Vector3 torque = (Gain * attitudeError) - (Damping * _filteredAngularVelocity);
        float maxTorque = (rising ? RecoveryMaxTorque : MaxTorque) * context.Strength;
        if (torque.LengthSquared() > maxTorque * maxTorque)
        {
            torque = torque.Normalized() * maxTorque;
        }

        LastTorque = torque;
        pelvis.ApplyTorque(torque);

        Vector3 reaction = -torque / _reactionLimbs.Count;
        foreach (IBoneState limb in _reactionLimbs)
        {
            limb.ApplyTorque(reaction);
        }
    }

    private void CollectReactionLimbs(in BalanceContext context, bool rising)
    {
        _reactionLimbs.Clear();

        if (!rising)
        {
            IBoneState? left = ReactIntoThighs ? context.ThighL : context.FootL;
            IBoneState? right = ReactIntoThighs ? context.ThighR : context.FootR;
            if (context.IsGroundedL && left != null && left.IsValid)
            {
                _reactionLimbs.Add(left);
            }
            if (context.IsGroundedR && right != null && right.IsValid)
            {
                _reactionLimbs.Add(right);
            }
            return;
        }

        // Sole-down grounding never qualifies while the body is still folded on the floor, so use
        // real world contact instead — the same test GetUpPhaseController plants its phases on.
        AddIfContacting(context.FootL);
        AddIfContacting(context.FootR);
        AddIfContacting(context.ForearmL);
        AddIfContacting(context.ForearmR);
        AddIfContacting(context.HandL);
        AddIfContacting(context.HandR);
    }

    private void AddIfContacting(IBoneState? limb)
    {
        if (limb != null && limb.IsValid && limb.IsInContactWithWorld())
        {
            _reactionLimbs.Add(limb);
        }
    }
}

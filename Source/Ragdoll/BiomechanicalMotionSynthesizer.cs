using System;
using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll.Trajectories;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Dynamic Motion Synthesis (DMS) registry and coordinator.
/// Uses the Strategy Pattern (OCP) to dynamically evaluate bone targets from registered
/// IMotionTrajectory strategies.
///
/// <para><b>Every registered trajectory MUST be stateless.</b> This registry is static, so one
/// instance of each trajectory is shared by every ragdoll in the process - and
/// <see cref="RL.RagdollSpawner"/> now puts up to 64 of them in one process, all ticking the same
/// objects every physics frame.</para>
///
/// <para>That is safe today only because <see cref="ComputeTargetRotation"/> passes the entire
/// per-body situation as arguments - bone, state, orientation, time, phase - and every
/// implementation is a pure function of them, holding nothing but immutable
/// <c>PhaseBoundaries</c>. It is an invariant, not an accident: a trajectory that caches so much as
/// a <c>private float _lastPhase</c> would silently share it across all 64 bodies, and the symptom
/// would be bone targets that depend on which OTHER ragdoll was evaluated immediately before -
/// nondeterministic, load-dependent, and invisible in a single-body test scene.</para>
///
/// <para>This is not hypothetical: <see cref="OrientationClassifier"/> was a static class with
/// exactly that shape and had to be made per-ragdoll for exactly that reason. If a trajectory ever
/// needs per-body memory, move the registry off <c>static</c> and give each
/// <see cref="BalanceController"/> its own, the way it now owns its own classifier.</para>
/// </summary>
public static class BiomechanicalMotionSynthesizer
{
    private static readonly IMotionTrajectory _standingTrajectory = new StandingBalanceTrajectory();
    private static readonly IMotionTrajectory _stumbleTrajectory = new StumbleTrajectory();
    private static readonly IMotionTrajectory _flailTrajectory = new FlailTrajectory();
    private static readonly IMotionTrajectory _supineRecovery = new SupineRecoveryTrajectory();
    private static readonly IMotionTrajectory _proneRecovery = new ProneRecoveryTrajectory();
    private static readonly IMotionTrajectory _pushUpDrill = new PushUpDrillTrajectory();

    private static readonly Dictionary<RagdollState, IMotionTrajectory> _stateTrajectories = new()
    {
        { RagdollState.Balanced, _standingTrajectory },
        { RagdollState.Stumbling, _stumbleTrajectory },
        { RagdollState.Flailing, _flailTrajectory },
        { RagdollState.PushUpDrill, _pushUpDrill }
    };

    private static readonly Dictionary<RagdollOrientation, IMotionTrajectory> _recoveryTrajectories = new()
    {
        { RagdollOrientation.Supine, _supineRecovery },
        { RagdollOrientation.Prone, _proneRecovery },
        { RagdollOrientation.Side, _supineRecovery } // Side rolls into supine recovery
    };


    /// <summary>
    /// The recovery trajectory that would be selected for the given orientation, exposed so
    /// GetUpPhaseController can read its phase boundaries and drive it from physical state.
    /// </summary>
    public static IPhasedRecoveryTrajectory? GetRecoveryTrajectory(RagdollOrientation orientation)
    {
        if (_recoveryTrajectories.TryGetValue(orientation, out IMotionTrajectory? trajectory))
        {
            return trajectory as IPhasedRecoveryTrajectory;
        }
        return _supineRecovery as IPhasedRecoveryTrajectory;
    }

    public static Quaternion ComputeTargetRotation(
        string boneName,
        RagdollState state,
        RagdollOrientation orientation,
        float globalTime,
        float phaseNormalized)
    {
        if (state == RagdollState.KnockedOut)
        {
            return Quaternion.Identity;
        }

        if (state == RagdollState.Recovering)
        {
            if (_recoveryTrajectories.TryGetValue(orientation, out IMotionTrajectory? recoveryStrategy))
            {
                return recoveryStrategy.EvaluateBoneTarget(boneName, globalTime, phaseNormalized);
            }
            return _supineRecovery.EvaluateBoneTarget(boneName, globalTime, phaseNormalized);
        }

        if (_stateTrajectories.TryGetValue(state, out IMotionTrajectory? strategy))
        {
            return strategy.EvaluateBoneTarget(boneName, globalTime, phaseNormalized);
        }

        return Quaternion.Identity;
    }
}

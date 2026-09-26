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
/// Each ragdoll owns its trajectory registry, so a trajectory can retain per-body state without
/// sharing it with other bodies in the same training process.
/// </summary>
public sealed class BiomechanicalMotionSynthesizer
{
    private readonly IMotionTrajectory _standingTrajectory = new StandingBalanceTrajectory();
    private readonly IMotionTrajectory _stumbleTrajectory = new StumbleTrajectory();
    private readonly IMotionTrajectory _flailTrajectory = new FlailTrajectory();
    private readonly IMotionTrajectory _supineRecovery = new SupineRecoveryTrajectory();
    private readonly IMotionTrajectory _proneRecovery = new ProneRecoveryTrajectory();
    private readonly IMotionTrajectory _pushUpDrill = new PushUpDrillTrajectory();

    private readonly Dictionary<RagdollState, IMotionTrajectory> _stateTrajectories;
    private readonly Dictionary<RagdollOrientation, IMotionTrajectory> _recoveryTrajectories;

    public BiomechanicalMotionSynthesizer()
    {
        _stateTrajectories = new()
        {
            { RagdollState.Balanced, _standingTrajectory },
            { RagdollState.Stumbling, _stumbleTrajectory },
            { RagdollState.Flailing, _flailTrajectory },
            { RagdollState.PushUpDrill, _pushUpDrill }
        };
        _recoveryTrajectories = new()
        {
            { RagdollOrientation.Supine, _supineRecovery },
            { RagdollOrientation.Prone, _proneRecovery },
            { RagdollOrientation.Side, _supineRecovery }
        };
    }
    /// <summary>
    /// The recovery trajectory that would be selected for the given orientation, exposed so
    /// GetUpPhaseController can read its phase boundaries and drive it from physical state.
    /// </summary>
    public IPhasedRecoveryTrajectory? GetRecoveryTrajectory(RagdollOrientation orientation)
    {
        if (_recoveryTrajectories.TryGetValue(orientation, out IMotionTrajectory? trajectory))
        {
            return trajectory as IPhasedRecoveryTrajectory;
        }
        return _supineRecovery as IPhasedRecoveryTrajectory;
    }

    public Quaternion ComputeTargetRotation(
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

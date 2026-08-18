using System;
using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll.Trajectories;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Dynamic Motion Synthesis (DMS) registry and coordinator.
/// Uses the Strategy Pattern (OCP) to dynamically evaluate bone targets from registered IMotionTrajectory strategies.
/// </summary>
public static class BiomechanicalMotionSynthesizer
{
    private static readonly IMotionTrajectory _standingTrajectory = new StandingBalanceTrajectory();
    private static readonly IMotionTrajectory _stumbleTrajectory = new StumbleTrajectory();
    private static readonly IMotionTrajectory _flailTrajectory = new FlailTrajectory();
    private static readonly IMotionTrajectory _supineRecovery = new SupineRecoveryTrajectory();
    private static readonly IMotionTrajectory _proneRecovery = new ProneRecoveryTrajectory();

    private static readonly Dictionary<RagdollState, IMotionTrajectory> _stateTrajectories = new()
    {
        { RagdollState.Balanced, _standingTrajectory },
        { RagdollState.Stumbling, _stumbleTrajectory },
        { RagdollState.Flailing, _flailTrajectory }
    };

    private static readonly Dictionary<RagdollOrientation, IMotionTrajectory> _recoveryTrajectories = new()
    {
        { RagdollOrientation.Supine, _supineRecovery },
        { RagdollOrientation.Prone, _proneRecovery },
        { RagdollOrientation.Side, _supineRecovery } // Side rolls into supine recovery
    };

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

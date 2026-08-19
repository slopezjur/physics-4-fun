using System.Collections.Generic;
using Godot;

namespace Physics4Fun.Ragdoll.Trajectories;

/// <summary>
/// Everything a recovery trajectory needs to evaluate a pose, passed by value so trajectories can
/// stay stateless singletons (<see cref="BiomechanicalMotionSynthesizer"/> holds them as
/// <c>static readonly</c>). The lead side in particular must NOT become a mutable field on the
/// trajectory, or two ragdolls would fight over it.
/// </summary>
public readonly record struct RecoveryPose(
    float GlobalTime,
    float PhaseNormalized,
    BodySide LeadSide
);

/// <summary>
/// A recovery trajectory whose timeline is divided into discrete biomechanical phases
/// (plant hands -> push up torso -> drive lead knee -> half-kneel rise -> stand).
///
/// Exposing the boundaries lets <see cref="Recovery.GetUpPhaseController"/> drive the trajectory
/// from achieved physical state (contacts established, CoM over support) instead of from a clock,
/// which is what makes the get-up quasi-static: each pose is held until the body actually reaches
/// it, rather than being replaced on a fixed schedule regardless of whether the push succeeded.
/// </summary>
public interface IPhasedRecoveryTrajectory : IMotionTrajectory
{
    /// <summary>
    /// Normalized timeline positions separating consecutive phases, ascending, exclusive of 0 and 1.
    /// A 5-phase trajectory returns 4 boundaries.
    /// </summary>
    IReadOnlyList<float> PhaseBoundaries { get; }

    /// <summary>
    /// Evaluates the bone's target with full pose context. This is the entry point the get-up uses;
    /// the inherited <see cref="IMotionTrajectory.EvaluateBoneTarget"/> exists only so recovery
    /// trajectories remain usable wherever a plain trajectory is expected, and assumes a right lead.
    /// </summary>
    Quaternion EvaluateBoneTarget(string boneName, in RecoveryPose pose);
}

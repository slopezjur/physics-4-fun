using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>
/// Read-only contract providing live biomechanical telemetry for HUD and diagnostic systems (ISP).
/// </summary>
public interface IBalanceTelemetryProvider
{
    StepPhase CurrentStepPhase { get; }
    float CurrentWeightShareL { get; }
    float CurrentWeightShareR { get; }
    bool IsGroundedL { get; }
    bool IsGroundedR { get; }
    float CurrentTiltAngleDeg { get; }
    Vector3 CenterOfMass { get; }
    float TotalMass { get; }

    /// <summary>Active balance strength after tilt/speed fades.</summary>
    float BalanceStrengthNow { get; }

    /// <summary>Timer relevant to the current state (grace/stumble/ground-rest/recovery), seconds remaining.</summary>
    float StateTimerValue { get; }

    /// <summary>Horizontal distance from the ICP to the support center (yaw-level frame), as computed for stepping.</summary>
    float IcpEscapeDistance { get; }

    /// <summary>
    /// Whether <see cref="IcpEscapeDistance"/> means anything this tick - false when no foot is
    /// grounded, because there is then no base of support to measure escape from. Callers MUST
    /// check this before treating a small distance as good balance; 0.0 is also what an airborne
    /// body reports.
    /// </summary>
    bool IsIcpValid { get; }

    /// <summary>Current swing phase progress [0..1], 0 when in double support.</summary>
    float SwingProgressNormalized { get; }

    /// <summary>Last corrective torque applied by the pelvis stabilizer (Vector3.Zero when inactive).</summary>
    Vector3 PelvisStabilizerTorque { get; }

    /// <summary>True mass-weighted center-of-mass velocity across all active bones.</summary>
    Vector3 CenterOfMassVelocity { get; }
}

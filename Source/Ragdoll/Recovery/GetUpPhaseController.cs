using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll.Trajectories;

namespace Physics4Fun.Ragdoll.Recovery;

/// <summary>
/// Discrete biomechanical stages of rising from the ground, following the way a person actually
/// stands up from face-down: set the hands, push the torso clear, bring ONE knee under the chest
/// and plant that foot, rise over that leg while the other swings through, then stand.
/// </summary>
public enum GetUpPhase
{
    PlantHands = 0,
    PushUpTorso = 1,
    DriveLeadKnee = 2,
    HalfKneelRise = 3,
    StandUp = 4,
    Complete = 5
}

/// <summary>
/// Sensor snapshot the phase controller reasons about. Kept separate from BalanceContext because
/// get-up cares about limbs (forearms) that the balance strategies never consult.
///
/// Feet are supplied already resolved into lead/trail so the controller stays side-agnostic: which
/// leg leads is decided once per attempt by HumanoidRagdoll, not re-derived here.
/// </summary>
public readonly record struct RecoveryContext(
    ActiveBone Pelvis,
    ActiveBone? Chest,
    ActiveBone? ForearmL,
    ActiveBone? ForearmR,
    ActiveBone? HandL,
    ActiveBone? HandR,
    ActiveBone? LeadFoot,
    ActiveBone? TrailFoot,
    Vector3 CenterOfMass,
    float GroundY,
    float PelvisTiltDeg
);

/// <summary>
/// The measured quantities every phase exit criterion is judged on, captured together so telemetry
/// can report exactly what the controller saw. Clearances are relative to the ground datum in
/// <see cref="RecoveryContext.GroundY"/>.
/// </summary>
public readonly record struct RecoveryCriteria(
    float ChestClearance,
    float PelvisClearance,
    float LeadFootComDistance,
    bool HandsPlanted,
    bool LeadFootPlanted,
    bool TrailFootPlanted,
    float PelvisTiltDeg
);

/// <summary>
/// Drives get-up progress from achieved physical state rather than a timer (Euphoria-style
/// reactivity instead of canned playback).
///
/// Within a phase, progress ramps to 1 over a nominal duration and then HOLDS - the pose is
/// maintained until the phase's physical exit criteria are actually satisfied. This is what makes
/// the motion quasi-static: the body is never asked to move on to the next support configuration
/// before it has established the current one. A per-phase timeout breaks deadlocks so a failed
/// phase degrades into the next attempt instead of hanging forever.
/// </summary>
public class GetUpPhaseController
{
    /// <summary>Nominal seconds to blend into each phase's pose before holding for its criteria.</summary>
    public float PhaseBlendDuration { get; set; } = 0.55f;

    /// <summary>Seconds a phase may hold before advancing regardless (deadlock breaker).</summary>
    public float PhaseTimeout { get; set; } = 1.2f;

    /// <summary>Chest clearance above ground (m) that counts as "torso pushed up".</summary>
    public float ChestClearance { get; set; } = 0.35f;

    /// <summary>Pelvis height above ground (m) that counts as "risen".</summary>
    public float RisenPelvisHeight { get; set; } = 0.60f;

    /// <summary>Pelvis height above ground (m) that counts as having reached the half-kneel.</summary>
    public float HalfKneelPelvisHeight { get; set; } = 0.45f;

    /// <summary>
    /// Max horizontal distance (m) from the CoM to the LEAD foot before the body may rise over it.
    /// Measured to the lead foot rather than the midpoint between both feet: in an asymmetric
    /// get-up the trail foot is still far behind, so a midpoint test can never be satisfied.
    /// </summary>
    public float ComOverSupportTolerance { get; set; } = 0.35f;

    /// <summary>Torso tilt (deg) below which the body counts as upright.</summary>
    public float UprightTiltDeg { get; set; } = 35.0f;

    public GetUpPhase CurrentPhase { get; private set; } = GetUpPhase.PlantHands;
    public bool IsComplete => CurrentPhase == GetUpPhase.Complete;

    /// <summary>Blend progress within the current phase [0..1]; holds at 1 while awaiting criteria.</summary>
    public float WithinPhaseProgress { get; private set; }

    /// <summary>Seconds spent in the current phase, including hold time.</summary>
    public float PhaseElapsed { get; private set; }

    /// <summary>True when the last advance was forced by timeout rather than met criteria.</summary>
    public bool LastAdvanceWasTimeout { get; private set; }

    /// <summary>Criteria measured on the most recent <see cref="Update"/>; the values the phase decision was made from.</summary>
    public RecoveryCriteria LastCriteria { get; private set; }

    public void Reset()
    {
        CurrentPhase = GetUpPhase.PlantHands;
        WithinPhaseProgress = 0.0f;
        PhaseElapsed = 0.0f;
        LastAdvanceWasTimeout = false;
        LastCriteria = default;
    }

    public void Update(in RecoveryContext context, float delta)
    {
        if (IsComplete)
        {
            return;
        }

        PhaseElapsed += delta;
        WithinPhaseProgress = PhaseBlendDuration > 0.0f
            ? Mathf.Clamp(PhaseElapsed / PhaseBlendDuration, 0.0f, 1.0f)
            : 1.0f;

        // Measured every tick, not only at the decision point, so telemetry shows how close each
        // criterion came during the hold rather than just its value on the frame the phase ended.
        LastCriteria = MeasureCriteria(context);

        // Only consider leaving once the pose has actually been commanded in full.
        if (WithinPhaseProgress < 1.0f)
        {
            return;
        }

        bool criteriaMet = EvaluateExitCriteria(LastCriteria);
        bool timedOut = PhaseElapsed >= PhaseBlendDuration + PhaseTimeout;

        if (criteriaMet || timedOut)
        {
            LastAdvanceWasTimeout = !criteriaMet;
            CurrentPhase = (GetUpPhase)((int)CurrentPhase + 1);
            WithinPhaseProgress = 0.0f;
            PhaseElapsed = 0.0f;
        }
    }

    private bool EvaluateExitCriteria(in RecoveryCriteria snapshot)
    {
        switch (CurrentPhase)
        {
            case GetUpPhase.PlantHands:
                // Both hands on the floor: a push-up needs two points of support, and pushing off
                // one arm just rolls the body onto its side.
                return snapshot.HandsPlanted;

            case GetUpPhase.PushUpTorso:
                // Torso genuinely lifted off the ground, not merely commanded to lift.
                return snapshot.ChestClearance > ChestClearance;

            case GetUpPhase.DriveLeadKnee:
                // The lead sole is down AND the foot has actually travelled under the mass. Without
                // the second half the leg can plant far behind the CoM, which supports nothing.
                return snapshot.LeadFootPlanted
                       && snapshot.LeadFootComDistance < ComOverSupportTolerance;

            case GetUpPhase.HalfKneelRise:
                // Pelvis has climbed to half-kneel height while the CoM stays over the lead foot.
                return snapshot.PelvisClearance > HalfKneelPelvisHeight
                       && snapshot.LeadFootComDistance < ComOverSupportTolerance;

            case GetUpPhase.StandUp:
                return snapshot.PelvisClearance > RisenPelvisHeight
                       && snapshot.PelvisTiltDeg < UprightTiltDeg;

            default:
                return true;
        }
    }

    /// <summary>
    /// Measures every quantity the exit criteria are judged on, in one place. Exposed so telemetry
    /// records the values the controller actually decided from rather than recomputing them, which
    /// is how a corrupted ground datum stayed invisible while silently failing every height test.
    /// </summary>
    public RecoveryCriteria MeasureCriteria(in RecoveryContext context)
    {
        bool hasChest = context.Chest != null && GodotObject.IsInstanceValid(context.Chest);

        return new RecoveryCriteria(
            ChestClearance: hasChest ? context.Chest!.GlobalPosition.Y - context.GroundY : float.NaN,
            PelvisClearance: context.Pelvis.GlobalPosition.Y - context.GroundY,
            LeadFootComDistance: ComputeLeadFootComDistance(context),
            // Both sides must be supporting, but either the hand or the forearm counts: the elbows
            // are folded flat at the start of the plant and extend as the push develops, so which
            // of the two is actually touching changes during the phase.
            HandsPlanted: (IsPlanted(context.ForearmL) || IsPlanted(context.HandL))
                          && (IsPlanted(context.ForearmR) || IsPlanted(context.HandR)),
            LeadFootPlanted: IsPlanted(context.LeadFoot),
            TrailFootPlanted: IsPlanted(context.TrailFoot),
            PelvisTiltDeg: context.PelvisTiltDeg);
    }

    /// <summary>
    /// Horizontal distance from the CoM to the lead foot; infinite when that foot is missing.
    /// This is the quantity that decides whether the body has a support point to rise over, and it
    /// deliberately ignores the trail foot, which is still far behind during an asymmetric get-up.
    /// </summary>
    private static float ComputeLeadFootComDistance(in RecoveryContext context)
    {
        ActiveBone? leadFoot = context.LeadFoot;
        if (leadFoot == null || !GodotObject.IsInstanceValid(leadFoot))
        {
            return float.PositiveInfinity;
        }

        return new Vector2(
            context.CenterOfMass.X - leadFoot.GlobalPosition.X,
            context.CenterOfMass.Z - leadFoot.GlobalPosition.Z).Length();
    }

    private static bool IsPlanted(ActiveBone? bone)
    {
        return bone != null && GodotObject.IsInstanceValid(bone) && bone.IsInContactWithWorld();
    }

    /// <summary>
    /// Maps the discrete phase plus within-phase blend onto the trajectory's own normalized
    /// timeline, so existing keyframed trajectories work unchanged - only the clock driving
    /// them is replaced.
    /// </summary>
    public float ToNormalizedProgress(IReadOnlyList<float> phaseBoundaries)
    {
        if (IsComplete)
        {
            return 1.0f;
        }

        int phase = (int)CurrentPhase;
        float lower = phase == 0 ? 0.0f : phaseBoundaries[Mathf.Min(phase - 1, phaseBoundaries.Count - 1)];
        float upper = phase < phaseBoundaries.Count ? phaseBoundaries[phase] : 1.0f;
        return Mathf.Lerp(lower, upper, WithinPhaseProgress);
    }
}

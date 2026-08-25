using Godot;
using Physics4Fun.Ragdoll.Recovery;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// The five-phase get-up machine: PlantHands -> PushUpTorso -> DriveLeadKnee -> HalfKneelRise ->
/// StandUp -> Complete.
///
/// These are the first tests that exist for it. It was always pure decision logic, but
/// <see cref="RecoveryContext"/> carried <c>ActiveBone</c> - a RigidBody3D - so exercising it meant
/// standing up a scene tree and hoping the physics reproduced the case you wanted. Behind
/// <c>IBoneState</c> the same logic takes a <see cref="FakeBone"/> and every phase boundary becomes
/// a direct assertion.
///
/// This matters more than the coverage number: get-up from prone is the project's stated unsolved
/// goal, and "the trajectory is advanced by achieved physical state, not by a clock" is the central
/// claim of the design. It is now checkable.
/// </summary>
public class GetUpPhaseControllerTests
{
    private const float Delta = 1.0f / 120.0f;

    /// <summary>A prone body: nothing planted, nothing lifted.</summary>
    private static RecoveryContext Prone(
        FakeBone? pelvis = null, FakeBone? chest = null,
        FakeBone? handL = null, FakeBone? handR = null,
        FakeBone? forearmL = null, FakeBone? forearmR = null,
        FakeBone? leadFoot = null, FakeBone? trailFoot = null,
        Vector3? com = null, float groundY = 0.0f, float tiltDeg = 90.0f)
        => new(
            Pelvis: pelvis ?? FakeBone.At(0.15f),
            Chest: chest ?? FakeBone.At(0.15f),
            ForearmL: forearmL ?? FakeBone.At(0.05f),
            ForearmR: forearmR ?? FakeBone.At(0.05f),
            HandL: handL ?? FakeBone.At(0.05f),
            HandR: handR ?? FakeBone.At(0.05f),
            LeadFoot: leadFoot ?? FakeBone.At(0.05f),
            TrailFoot: trailFoot ?? FakeBone.At(0.05f),
            CenterOfMass: com ?? Vector3.Zero,
            GroundY: groundY,
            PelvisTiltDeg: tiltDeg);

    /// <summary>Ticks past the blend window so the controller is allowed to consider exiting.</summary>
    private static void TickPastBlend(GetUpPhaseController c, in RecoveryContext ctx)
    {
        int ticks = Mathf.CeilToInt(c.PhaseBlendDuration / Delta) + 1;
        for (int i = 0; i < ticks; i++)
        {
            c.Update(ctx, Delta);
        }
    }

    [Fact]
    public void StartsAtPlantHands()
    {
        Assert.Equal(GetUpPhase.PlantHands, new GetUpPhaseController().CurrentPhase);
    }

    /// <summary>
    /// The blend window is a floor on phase duration: "only consider leaving once the pose has
    /// actually been commanded in full". Even with every exit criterion already satisfied, the
    /// controller must not skip ahead on tick one.
    /// </summary>
    [Fact]
    public void WillNotLeaveAPhaseBeforeThePoseIsFullyCommanded()
    {
        var controller = new GetUpPhaseController();
        RecoveryContext ready = Prone(
            handL: FakeBone.At(0.02f, planted: true),
            handR: FakeBone.At(0.02f, planted: true));

        controller.Update(ready, Delta);

        Assert.Equal(GetUpPhase.PlantHands, controller.CurrentPhase);
        Assert.True(controller.WithinPhaseProgress < 1.0f);
    }

    // ---- Phase 1: PlantHands ----

    /// <summary>
    /// "A push-up needs two points of support, and pushing off one arm just rolls the body onto its
    /// side." One planted hand must not advance the phase.
    /// </summary>
    [Fact]
    public void OneHandDownIsNotEnoughToLeavePlantHands()
    {
        var controller = new GetUpPhaseController { PhaseTimeout = 999.0f };
        RecoveryContext oneHand = Prone(handL: FakeBone.At(0.02f, planted: true));

        TickPastBlend(controller, oneHand);

        Assert.Equal(GetUpPhase.PlantHands, controller.CurrentPhase);
    }

    [Fact]
    public void BothHandsDownAdvancesToPushUpTorso()
    {
        var controller = new GetUpPhaseController { PhaseTimeout = 999.0f };
        RecoveryContext bothHands = Prone(
            handL: FakeBone.At(0.02f, planted: true),
            handR: FakeBone.At(0.02f, planted: true));

        TickPastBlend(controller, bothHands);

        Assert.Equal(GetUpPhase.PushUpTorso, controller.CurrentPhase);
        Assert.False(controller.LastAdvanceWasTimeout);
    }

    /// <summary>
    /// Either the hand or the forearm counts as support on each side, "because the elbows are
    /// folded flat at the start of the plant and extend as the push develops".
    /// </summary>
    [Fact]
    public void ForearmContactSubstitutesForHandContact()
    {
        var controller = new GetUpPhaseController { PhaseTimeout = 999.0f };
        RecoveryContext onElbows = Prone(
            forearmL: FakeBone.At(0.02f, planted: true),
            forearmR: FakeBone.At(0.02f, planted: true));

        TickPastBlend(controller, onElbows);

        Assert.Equal(GetUpPhase.PushUpTorso, controller.CurrentPhase);
    }

    // ---- Phase 2: PushUpTorso ----

    /// <summary>
    /// "Torso genuinely lifted off the ground, not merely commanded to lift." The phase must hold
    /// while the chest is still down, however long the trajectory has been playing.
    /// </summary>
    [Fact]
    public void PushUpHoldsWhileTheChestIsStillOnTheFloor()
    {
        var controller = new GetUpPhaseController { PhaseTimeout = 999.0f };
        RecoveryContext handsDown = Prone(
            handL: FakeBone.At(0.02f, planted: true),
            handR: FakeBone.At(0.02f, planted: true));
        TickPastBlend(controller, handsDown);
        Assert.Equal(GetUpPhase.PushUpTorso, controller.CurrentPhase);

        // Chest at 0.10 m against a 0.35 m clearance requirement.
        RecoveryContext chestDown = Prone(
            chest: FakeBone.At(0.10f),
            handL: FakeBone.At(0.02f, planted: true),
            handR: FakeBone.At(0.02f, planted: true));
        TickPastBlend(controller, chestDown);

        Assert.Equal(GetUpPhase.PushUpTorso, controller.CurrentPhase);
    }

    [Fact]
    public void PushUpAdvancesOnceTheChestClears()
    {
        var controller = new GetUpPhaseController { PhaseTimeout = 999.0f };
        RecoveryContext handsDown = Prone(
            handL: FakeBone.At(0.02f, planted: true),
            handR: FakeBone.At(0.02f, planted: true));
        TickPastBlend(controller, handsDown);

        RecoveryContext chestUp = Prone(
            chest: FakeBone.At(0.50f),
            handL: FakeBone.At(0.02f, planted: true),
            handR: FakeBone.At(0.02f, planted: true));
        TickPastBlend(controller, chestUp);

        Assert.Equal(GetUpPhase.DriveLeadKnee, controller.CurrentPhase);
    }

    // ---- Phase 3: DriveLeadKnee ----

    /// <summary>
    /// The half of the criterion that is easy to forget: "Without the second half the leg can plant
    /// far behind the CoM, which supports nothing." A planted foot two metres away must not count.
    /// </summary>
    [Fact]
    public void PlantedFootFarFromTheCentreOfMassDoesNotCountAsSupport()
    {
        var controller = new GetUpPhaseController { PhaseTimeout = 999.0f };
        var farFoot = new FakeBone("LeadFoot") { GlobalPosition = new Vector3(2.0f, 0.02f, 0.0f), InContact = true };

        RecoveryCriteria criteria = controller.MeasureCriteria(Prone(leadFoot: farFoot, com: Vector3.Zero));

        Assert.True(criteria.LeadFootPlanted);
        Assert.True(criteria.LeadFootComDistance > controller.ComOverSupportTolerance);
    }

    [Fact]
    public void MissingLeadFootReportsInfiniteDistanceRatherThanZero()
    {
        var controller = new GetUpPhaseController();

        RecoveryCriteria criteria = controller.MeasureCriteria(
            Prone(leadFoot: new FakeBone("LeadFoot") { IsValid = false }));

        Assert.True(float.IsPositiveInfinity(criteria.LeadFootComDistance));
        Assert.False(criteria.LeadFootPlanted);
    }

    // ---- Timeout path ----

    /// <summary>
    /// The give-up valve. A body that never satisfies a criterion must still advance, and must be
    /// FLAGGED as having timed out - the distinction telemetry needs to tell a real rise from the
    /// machine forcing itself forward.
    /// </summary>
    [Fact]
    public void PhaseAdvancesOnTimeoutAndFlagsIt()
    {
        var controller = new GetUpPhaseController { PhaseBlendDuration = 0.1f, PhaseTimeout = 0.2f };
        RecoveryContext hopeless = Prone(); // nothing planted, nothing lifted, forever

        int ticks = Mathf.CeilToInt((0.1f + 0.2f) / Delta) + 2;
        for (int i = 0; i < ticks; i++)
        {
            controller.Update(hopeless, Delta);
        }

        Assert.Equal(GetUpPhase.PushUpTorso, controller.CurrentPhase);
        Assert.True(controller.LastAdvanceWasTimeout, "a forced advance must be reported as a timeout");
    }

    /// <summary>
    /// The worst-case horizon the state machine's RecoveryDuration is sized against: 4 phases x
    /// (PhaseBlendDuration 0.55 + PhaseTimeout 1.2) = 7.03 s. RagdollStateMachine.RecoveryDuration
    /// is 10 s specifically to exceed this, after attempts were observed dying at exactly +7.5 s.
    /// </summary>
    [Fact]
    public void WorstCaseRunReachesCompleteWithinTheDocumentedHorizon()
    {
        var controller = new GetUpPhaseController();
        RecoveryContext hopeless = Prone();

        float elapsed = 0.0f;
        while (!controller.IsComplete && elapsed < 20.0f)
        {
            controller.Update(hopeless, Delta);
            elapsed += Delta;
        }

        Assert.True(controller.IsComplete, "the timeout chain must always terminate");
        Assert.True(elapsed < 10.0f, $"worst case {elapsed:F2}s must fit RecoveryDuration's 10 s horizon");
        Assert.True(elapsed > 7.0f, $"worst case {elapsed:F2}s should be the documented ~7.03 s, not shorter");
    }

    [Fact]
    public void CompleteIsAbsorbing()
    {
        var controller = new GetUpPhaseController();
        RecoveryContext hopeless = Prone();

        for (int i = 0; i < 5000; i++)
        {
            controller.Update(hopeless, Delta);
        }

        Assert.Equal(GetUpPhase.Complete, controller.CurrentPhase);
        Assert.True(controller.IsComplete);
    }

    [Fact]
    public void ResetReturnsToTheFirstPhase()
    {
        var controller = new GetUpPhaseController();
        for (int i = 0; i < 5000; i++)
        {
            controller.Update(Prone(), Delta);
        }
        Assert.True(controller.IsComplete);

        controller.Reset();

        Assert.Equal(GetUpPhase.PlantHands, controller.CurrentPhase);
        Assert.False(controller.IsComplete);
        Assert.Equal(0.0f, controller.PhaseElapsed, 5);
    }

    // ---- Criteria measurement ----

    /// <summary>
    /// Clearances are measured against the supplied ground datum, not against world zero - which is
    /// the bug the source calls out: "a corrupted ground datum stayed invisible while silently
    /// failing every height test".
    /// </summary>
    [Fact]
    public void ClearancesAreRelativeToTheGroundDatum()
    {
        var controller = new GetUpPhaseController();

        RecoveryCriteria onSlab = controller.MeasureCriteria(
            Prone(pelvis: FakeBone.At(1.60f), chest: FakeBone.At(1.90f), groundY: 1.0f));

        Assert.Equal(0.60f, onSlab.PelvisClearance, 4);
        Assert.Equal(0.90f, onSlab.ChestClearance, 4);
    }

    [Fact]
    public void MissingChestReportsNaNClearanceRatherThanAFalseZero()
    {
        var controller = new GetUpPhaseController();

        RecoveryCriteria criteria = controller.MeasureCriteria(
            Prone(chest: new FakeBone("Chest") { IsValid = false }));

        Assert.True(float.IsNaN(criteria.ChestClearance));
    }
}

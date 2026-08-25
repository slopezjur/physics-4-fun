using Godot;
using Physics4Fun.Ragdoll;
using Xunit;

namespace Physics4Fun.Tests;

/// <summary>
/// Pillar 7's behavioural FSM: Balanced -> Stumbling -> Flailing -> KnockedOut -> Recovering.
///
/// Pure decision logic driven by sensor values the caller supplies, with no sensing of its own -
/// which is exactly why it can be tested here while the modules that feed it cannot.
/// </summary>
public class RagdollStateMachineTests
{
    private const float Delta = 1.0f / 120.0f;
    private const float SettleGraceSeconds = 0.15f;

    /// <summary>
    /// A machine past its settle grace. Fresh instances spend the first 0.15 s forcing Balanced,
    /// which would otherwise mask every transition under test.
    /// </summary>
    private static RagdollStateMachine Settled()
    {
        var sm = new RagdollStateMachine();
        for (float t = 0.0f; t <= SettleGraceSeconds + Delta; t += Delta)
        {
            sm.Evaluate(RagdollState.Balanced, Delta, 0.0f, 1.0f, 0.0f, true, true);
        }
        return sm;
    }

    private static RagdollState Step(
        RagdollStateMachine sm, RagdollState state,
        float tilt = 0.0f, float height = 1.0f, float speed = 0.0f,
        bool groundedL = true, bool groundedR = true)
        => sm.Evaluate(state, Delta, tilt, height, speed, groundedL, groundedR);

    /// <summary>
    /// Ticks for <paramref name="seconds"/>, FEEDING THE RETURNED STATE BACK each time - which is
    /// what HumanoidRagdoll._PhysicsProcess does (`CurrentState = Balance.EvaluateState(CurrentState, dt)`).
    ///
    /// Re-passing the original state instead is a subtly wrong harness: several transitions zero
    /// their dwell timer on the way out, so a caller that keeps asserting the old state re-arms the
    /// timer every tick and the machine oscillates instead of latching.
    /// </summary>
    private static RagdollState Run(
        RagdollStateMachine sm, RagdollState state, float seconds,
        float tilt = 0.0f, float height = 1.0f, float speed = 0.0f,
        bool groundedL = true, bool groundedR = true)
    {
        int ticks = Mathf.RoundToInt(seconds / Delta);
        for (int i = 0; i < ticks; i++)
        {
            state = sm.Evaluate(state, Delta, tilt, height, speed, groundedL, groundedR);
        }
        return state;
    }

    // ---- Settle grace ----

    /// <summary>
    /// The grace window exists because every entry point teleports the body first. Without it a
    /// spawn pose momentarily reads as a fall.
    /// </summary>
    [Fact]
    public void SettleGraceForcesBalancedEvenAtKnockoutTilt()
    {
        var sm = new RagdollStateMachine();

        Assert.Equal(RagdollState.Balanced, Step(sm, RagdollState.Balanced, tilt: 179.0f, height: 0.05f));
        Assert.True(sm.IsSettleGraceActive);
    }

    [Fact]
    public void SettleGraceExpiresAndTransitionsResume()
    {
        var sm = Settled();

        Assert.False(sm.IsSettleGraceActive);
        Assert.Equal(RagdollState.Flailing, Step(sm, RagdollState.Balanced, tilt: 179.0f));
    }

    // ---- Absorbing states ----
    //
    // PushUpDrill and ReinforcementLearning are checked AHEAD of the settle grace, because both are
    // entered by hand right after a teleport - and the grace branch would otherwise force Balanced
    // on the very next tick, so the drill would never run and the RL policy would never get control.

    [Theory]
    [InlineData(RagdollState.PushUpDrill)]
    [InlineData(RagdollState.ReinforcementLearning)]
    public void ScriptedStatesAreAbsorbingEvenDuringSettleGrace(RagdollState state)
    {
        var sm = new RagdollStateMachine();

        // Sensor values that would otherwise force Flailing immediately.
        Assert.Equal(state, Step(sm, state, tilt: 179.0f, height: 0.02f, speed: 9.0f, groundedL: false, groundedR: false));
    }

    [Theory]
    [InlineData(RagdollState.PushUpDrill)]
    [InlineData(RagdollState.ReinforcementLearning)]
    public void ScriptedStatesNeverSelfExit(RagdollState state)
    {
        var sm = Settled();

        for (int i = 0; i < 2000; i++)
        {
            Assert.Equal(state, Step(sm, state, tilt: 175.0f, height: 0.01f, groundedL: false, groundedR: false));
        }
    }

    // ---- Balanced ----

    [Fact]
    public void StaysBalancedWithinAllThresholds()
    {
        var sm = Settled();
        Assert.Equal(RagdollState.Balanced, Step(sm, RagdollState.Balanced, tilt: 10.0f, speed: 1.0f));
    }

    [Fact]
    public void ModerateTiltEscalatesToStumbling()
    {
        var sm = Settled();
        Assert.Equal(RagdollState.Stumbling, Step(sm, RagdollState.Balanced, tilt: 45.0f));
    }

    [Fact]
    public void HighSpeedEscalatesToStumblingEvenWhileUpright()
    {
        var sm = Settled();
        Assert.Equal(RagdollState.Stumbling, Step(sm, RagdollState.Balanced, tilt: 5.0f, speed: 4.0f));
    }

    /// <summary>
    /// Escalation is ordered: only extreme tilt skips Stumbling. A slow collapse must pass through
    /// it so the stumble reflexes get a chance to run.
    /// </summary>
    [Fact]
    public void ExtremeTiltSkipsStumblingAndGoesStraightToFlailing()
    {
        var sm = Settled();
        Assert.Equal(RagdollState.Flailing, Step(sm, RagdollState.Balanced, tilt: 95.0f));
    }

    [Fact]
    public void SustainedAirborneTimeRoutesDirectlyToFlailing()
    {
        var sm = Settled();

        RagdollState state = Run(sm, RagdollState.Balanced, 0.6f + Delta, tilt: 5.0f, groundedL: false, groundedR: false);

        Assert.Equal(RagdollState.Flailing, state);
    }

    // ---- Stumbling ----

    [Fact]
    public void StumbleRecoversToBalancedOnceTheTimerExpiresAndTiltIsSafe()
    {
        var sm = Settled();
        Step(sm, RagdollState.Balanced, tilt: 45.0f); // arms the 1.4 s stumble timer

        RagdollState state = Run(sm, RagdollState.Stumbling, 1.4f + Delta, tilt: 10.0f);

        Assert.Equal(RagdollState.Balanced, state);
    }

    [Fact]
    public void StumbleCollapsesToFlailingWhenLowAndSlow()
    {
        var sm = Settled();
        Step(sm, RagdollState.Balanced, tilt: 45.0f);

        Assert.Equal(RagdollState.Flailing, Step(sm, RagdollState.Stumbling, tilt: 20.0f, height: 0.2f, speed: 0.5f));
    }

    // ---- Flailing -> KnockedOut ----

    [Fact]
    public void FlailingRequiresSustainedGroundContactBeforeKnockout()
    {
        var sm = Settled();

        // Below the 0.25 s dwell it must still be Flailing.
        RagdollState state = Run(sm, RagdollState.Flailing, 0.20f, height: 0.2f, speed: 0.5f);
        Assert.Equal(RagdollState.Flailing, state);

        state = Run(sm, state, 0.10f, height: 0.2f, speed: 0.5f);
        Assert.Equal(RagdollState.KnockedOut, state);
    }

    [Fact]
    public void FlailingDwellResetsIfTheBodyIsStillMoving()
    {
        var sm = Settled();

        // Fast enough to keep resetting the dwell timer, so knockout never latches.
        Assert.Equal(RagdollState.Flailing, Run(sm, RagdollState.Flailing, 5.0f, height: 0.2f, speed: 5.0f));
    }

    // ---- KnockedOut -> Recovering ----

    [Fact]
    public void KnockedOutRisesAfterTheAutoRecoveryDelay()
    {
        var sm = Settled();

        RagdollState state = Run(sm, RagdollState.KnockedOut, 0.8f + Delta, height: 0.2f, speed: 0.2f);

        Assert.Equal(RagdollState.Recovering, state);
    }

    // ---- Recovering ----

    /// <summary>
    /// Success is purely physical - upright, risen, in contact - not clock-based. The old
    /// "progress >= 0.85" gate tied it to a timer that no longer tracks the phase-driven
    /// trajectory, and would have blocked an early clean rise.
    /// </summary>
    [Fact]
    public void RecoverySucceedsOnPhysicalStateNotElapsedTime()
    {
        var sm = Settled();

        RagdollState state = Step(sm, RagdollState.Recovering, tilt: 10.0f, height: 0.9f);

        Assert.Equal(RagdollState.Balanced, state);
        Assert.Equal(1.0f, sm.RecoveryProgressNormalized, 5);
    }

    [Fact]
    public void RecoveryGivesUpBackToKnockedOutAfterItsHorizon()
    {
        var sm = Settled();
        sm.RecoveryDuration = 1.0f;

        // Enter Recovering so _recoveryTimer is armed from RecoveryDuration.
        RagdollState state = Run(sm, RagdollState.KnockedOut, 0.8f + Delta, height: 0.2f, speed: 0.2f);
        Assert.Equal(RagdollState.Recovering, state);

        // Give-up horizon is RecoveryDuration plus the 1.5 s of overrun the machine allows.
        state = Run(sm, state, 1.0f + 1.5f + Delta, tilt: 120.0f, height: 0.1f);

        Assert.Equal(RagdollState.KnockedOut, state);
    }

    [Fact]
    public void RecoveryProgressIsNormalisedAndMonotonic()
    {
        var sm = Settled();
        sm.RecoveryDuration = 2.0f;

        RagdollState state = Run(sm, RagdollState.KnockedOut, 0.8f + Delta, height: 0.2f, speed: 0.2f);
        Assert.Equal(RagdollState.Recovering, state);

        float previous = -1.0f;
        for (float t = 0.0f; t < 1.5f; t += Delta)
        {
            state = Step(sm, state, tilt: 120.0f, height: 0.1f);

            Assert.InRange(sm.RecoveryProgressNormalized, 0.0f, 1.0f);
            Assert.True(sm.RecoveryProgressNormalized >= previous, "recovery progress went backwards");
            previous = sm.RecoveryProgressNormalized;
        }
    }

    // ---- Reset ----

    [Fact]
    public void ResetRestoresTheSettleGrace()
    {
        var sm = Settled();
        Assert.False(sm.IsSettleGraceActive);

        sm.Reset();

        Assert.True(sm.IsSettleGraceActive);
        Assert.Equal(RagdollState.Balanced, Step(sm, RagdollState.Balanced, tilt: 179.0f));
    }
}

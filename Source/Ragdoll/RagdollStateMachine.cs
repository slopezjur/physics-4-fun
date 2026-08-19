using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Behavioral state machine (Euphoria DMS Pillar 7): Balanced -> Stumbling -> Flailing ->
/// KnockedOut -> Recovering. Owns state-transition timers and thresholds; pure decision logic
/// driven by sensor values the caller supplies each tick (SRP — no sensing of its own).
/// </summary>
public class RagdollStateMachine
{
    public float DecoupleVelocityThreshold { get; set; } = 3.0f;
    public float MaxStumbleTiltAngleDeg { get; set; } = 30.0f;
    public float KnockoutTiltAngleDeg { get; set; } = 80.0f;
    public float AutoRecoveryDelay { get; set; } = 0.8f;
    /// <summary>
    /// Nominal get-up window. The trajectory itself is now advanced by GetUpPhaseController from
    /// physical state, so this only sets the give-up horizon: a quasi-static rise dwells on each
    /// phase until its criteria are met, which takes considerably longer than keyframe playback.
    ///
    /// Must comfortably exceed the phase machine's worst case, 4 phases x (PhaseBlendDuration 0.55
    /// + PhaseTimeout 1.2) = 7.03 s. At the previous 6.0 s the abandon point (6.0 + 1.5 = 7.5 s)
    /// left only 0.47 s of margin, and attempts were observed being killed at exactly +7.5 s.
    /// </summary>
    public float RecoveryDuration { get; set; } = 10.0f;

    public float RecoveryProgressNormalized { get; private set; } = 0.0f;
    public bool IsSettleGraceActive => _settleGraceTimer > 0.0f;

    private float _stumbleTimer;
    private float _groundRestTimer;
    private float _recoveryTimer;
    private float _flailGroundTimer;
    private float _airborneTimer;
    private float _settleGraceTimer = 0.15f;
    private RagdollState _lastEvaluatedState = RagdollState.Balanced;

    /// <summary>Timer relevant to the current state (grace/stumble/ground-rest/recovery), seconds remaining.</summary>
    public float StateTimerValue
    {
        get
        {
            if (_settleGraceTimer > 0.0f)
            {
                return _settleGraceTimer;
            }
            return _lastEvaluatedState switch
            {
                RagdollState.Stumbling => Mathf.Max(0.0f, _stumbleTimer),
                RagdollState.KnockedOut => Mathf.Max(0.0f, AutoRecoveryDelay - _groundRestTimer),
                RagdollState.Recovering => Mathf.Max(0.0f, _recoveryTimer),
                _ => 0.0f
            };
        }
    }

    public void Reset()
    {
        _stumbleTimer = 0.0f;
        _groundRestTimer = 0.0f;
        _recoveryTimer = 0.0f;
        _flailGroundTimer = 0.0f;
        _airborneTimer = 0.0f;
        _settleGraceTimer = 0.15f;
        _lastEvaluatedState = RagdollState.Balanced;
    }

    public void TriggerStumble() => _stumbleTimer = 1.4f;

    /// <summary>Distinct (shorter) stumble timer used when a stumble is triggered by a registered hit.</summary>
    public void TriggerHitStumble() => _stumbleTimer = 1.2f;

    /// <summary>
    /// Evaluates one state-machine tick from caller-supplied sensor values (tilt, height, speed,
    /// foot contact). Pure decision logic; the caller owns sensing (CoM, ground contact, tilt).
    /// </summary>
    public RagdollState Evaluate(RagdollState currentState, float delta, float tiltAngleDeg, float currentHeight, float speed, bool isGroundedL, bool isGroundedR)
    {
        _lastEvaluatedState = currentState;

        if (!isGroundedL && !isGroundedR)
        {
            _airborneTimer += delta;
        }
        else
        {
            _airborneTimer = 0.0f;
        }

        // Checked ahead of the settle grace: the drill is entered by hand right after a teleport,
        // which resets the grace timer, and the grace branch would otherwise force the state to
        // Balanced on the very next tick and the drill would never run.
        if (currentState == RagdollState.PushUpDrill)
        {
            return RagdollState.PushUpDrill;
        }

        if (_settleGraceTimer > 0.0f)
        {
            _settleGraceTimer -= delta;
            return RagdollState.Balanced;
        }

        switch (currentState)
        {
            case RagdollState.Balanced:
                if (tiltAngleDeg > KnockoutTiltAngleDeg || _airborneTimer > 0.6f)
                {
                    _groundRestTimer = 0.0f;
                    _flailGroundTimer = 0.0f;
                    return RagdollState.Flailing;
                }
                if (tiltAngleDeg > MaxStumbleTiltAngleDeg || speed > DecoupleVelocityThreshold)
                {
                    _stumbleTimer = 1.4f;
                    return RagdollState.Stumbling;
                }
                return RagdollState.Balanced;

            case RagdollState.Stumbling:
                _stumbleTimer -= delta;
                if (tiltAngleDeg > KnockoutTiltAngleDeg || (currentHeight < 0.25f && speed < 1.0f) || _airborneTimer > 0.6f)
                {
                    _groundRestTimer = 0.0f;
                    _flailGroundTimer = 0.0f;
                    return RagdollState.Flailing;
                }
                if (_stumbleTimer <= 0.0f && tiltAngleDeg < 30.0f && (isGroundedL || isGroundedR))
                {
                    return RagdollState.Balanced;
                }
                return RagdollState.Stumbling;

            case RagdollState.Flailing:
                if (currentHeight < 0.35f && speed < 2.5f)
                {
                    _flailGroundTimer += delta;
                    if (_flailGroundTimer >= 0.25f)
                    {
                        _flailGroundTimer = 0.0f;
                        _groundRestTimer = 0.0f;
                        return RagdollState.KnockedOut;
                    }
                }
                else
                {
                    _flailGroundTimer = 0.0f;
                }
                return RagdollState.Flailing;

            case RagdollState.KnockedOut:
                if (currentHeight < 0.45f && speed < 1.0f)
                {
                    _groundRestTimer += delta;
                    if (_groundRestTimer >= AutoRecoveryDelay)
                    {
                        _groundRestTimer = 0.0f;
                        _recoveryTimer = RecoveryDuration;
                        RecoveryProgressNormalized = 0.0f;
                        return RagdollState.Recovering;
                    }
                }
                else
                {
                    _groundRestTimer = 0.0f;
                }
                return RagdollState.KnockedOut;

            case RagdollState.Recovering:
                _recoveryTimer -= delta;
                RecoveryProgressNormalized = Mathf.Clamp(1.0f - (_recoveryTimer / RecoveryDuration), 0.0f, 1.0f);

                // Success is now purely physical: upright, risen, and in contact. The old
                // "progress >= 0.85" gate tied success to the clock, which no longer tracks the
                // phase-driven trajectory and would block an early, clean rise.
                if (tiltAngleDeg < 30.0f && currentHeight > 0.60f && (isGroundedL || isGroundedR))
                {
                    RecoveryProgressNormalized = 1.0f;
                    return RagdollState.Balanced;
                }

                if (_recoveryTimer <= -1.5f)
                {
                    _groundRestTimer = 0.0f;
                    return RagdollState.KnockedOut;
                }
                return RagdollState.Recovering;

            default:
                return currentState;
        }
    }
}

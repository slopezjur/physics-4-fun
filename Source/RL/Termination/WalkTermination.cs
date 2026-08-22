using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Termination;

/// <summary>
/// Termination for the walking task: fail on falling, otherwise run the full window.
///
/// There is deliberately NO success condition, which is the structural difference from
/// UprightTermination. Getting up is a transition to a goal state, so making that state absorbing is
/// what separates "stood up" from "stood up then fell over". Walking is a sustained periodic
/// behaviour with no goal state to be absorbed into - the only meaningful success is "was still
/// walking when the window ended", and that is already what running to TimeLimit means. Inventing
/// a success threshold (say, 5 m travelled) would end the episode at the moment the agent is doing
/// the thing being trained, and pay it to stop.
///
/// Falling DOES end the episode, and that asymmetry carries most of the learning signal. With a
/// 6 s window, going down at t = 1 s forfeits roughly 40 reward - far more than any explicit
/// penalty would sensibly apply - so early termination is what makes staying upright valuable
/// without needing a large FallPenalty that would make the gait timid.
/// </summary>
public sealed class WalkTermination : IRlTerminationCondition, IRlTerminationDiagnostics
{
    /// <summary>
    /// Head height (m) below which the attempt counts as fallen.
    ///
    /// 1.00 m against a standing head height of about 1.40 m on this rig, so it sits roughly 0.40 m
    /// down. That margin is set by the gait, not by the geometry of falling: a walking stride
    /// oscillates the head by a few centimetres and a deep crouch reaches perhaps 1.15 m, so a
    /// threshold at 1.20 m would terminate mid-stride and teach the policy to walk stiff-legged to
    /// avoid it. Lower would be safer still, but every extra centimetre is time spent collecting
    /// reward while already on the way down, which is what pays for a controlled topple.
    ///
    /// Head rather than pelvis for the same reason UprightProgressReward uses head height as its
    /// potential: pelvis height barely moves when the body folds at the waist, so a pelvis
    /// threshold misses the most common way this rig goes down.
    /// </summary>
    private const float FallenHeadHeight = 1.00f;

    /// <summary>
    /// Torso tilt (deg) beyond which the attempt counts as fallen, independent of height.
    ///
    /// Needed because height alone cannot catch a fall that is still in progress. A body pitched
    /// 60 deg forward with the head still at 1.05 m is unrecoverable on this rig but passes the
    /// height check, and every tick it survives there pays alive plus a large forward-velocity term
    /// as it accelerates downward. That is the controlled-topple gait, and it scores well right up
    /// until it lands. Cutting on tilt removes the payout.
    ///
    /// 50 deg leaves room above the 30 deg that UprightTermination calls upright, so ordinary
    /// walking lean does not trip it.
    /// </summary>
    private const float FallenTiltDeg = 50.0f;

    /// <summary>Forward speed (m/s) above which a tick counts toward the "moving" diagnostic.</summary>
    private const float MovingSpeedThreshold = 0.3f;

    /// <summary>
    /// Grace period (s) at the start of an episode during which a fall cannot be declared.
    ///
    /// The reset teleports every bone into the standing pose with zero velocity, and the body needs
    /// a few physics ticks to settle onto its feet. Without a grace window a transient on tick one
    /// can end the episode before the policy has acted at all, which shows up as a mass of
    /// near-zero-length episodes and makes every per-episode average meaningless.
    /// </summary>
    private const float SettleSeconds = 0.25f;

    /// <summary>
    /// Forward axis for this episode, captured on the first tick exactly as WalkForwardReward does.
    ///
    /// Duplicated rather than shared because the two components are independent by design - a
    /// termination must not depend on a particular reward being installed - and the capture is
    /// three lines. Both read -Basis.Z projected flat, so they cannot disagree about which way is
    /// forward.
    /// </summary>
    private Vector3 _forward = Vector3.Forward;
    private bool _hasForward;

    private readonly Dictionary<string, int> _conditionTicks = new()
    {
        ["upright"] = 0,
        ["head"] = 0,
        ["grounded"] = 0,
        ["moving"] = 0,
    };

    private int _totalTicks;

    /// <summary>
    /// How this episode ended, latched on the terminal tick. All zero if it timed out.
    ///
    /// Exists because "Fallen" was one undifferentiated reason covering two very different
    /// failures, and the fix for each is the opposite of the fix for the other. Measured state
    /// before this was added: standing/upright 0.9986 and standing/head 0.9977, i.e. the body sits
    /// inside both limits for 99.8% of ticks and then tips over in the last one or two - so the
    /// interesting question is not "how healthy was it on average" but "which way did it go".
    ///
    /// Sagittal failure points at gait continuation (it never completes a stride cycle); lateral
    /// failure points at frontal-plane balance, where bipeds actually fall and where the sagittal
    /// plane's partial self-correction does not help.
    /// </summary>
    private bool _fellHead;
    private bool _fellTilt;
    private bool _fellLateral;
    private bool _fellForward;

    /// <summary>Reused across episodes; Reset zeroes it rather than reallocating.</summary>
    private readonly Dictionary<string, float> _conditionRates = new();

    public IReadOnlyDictionary<string, float> EpisodeConditionRates
    {
        get
        {
            float denominator = Mathf.Max(1, _totalTicks);
            foreach (var entry in _conditionTicks)
            {
                _conditionRates[entry.Key] = entry.Value / denominator;
            }

            // Per-episode FLAGS, not tick rates - deliberately not divided by the tick count. They
            // ride this dictionary because the bridge forwards every entry as stand_<key> and the
            // trainer averages those across the rollout's episodes, turning a 0/1 flag into the
            // rate wanted: "what fraction of episodes ended this way". Same trick as
            // UprightTermination's disturbed/recovered.
            _conditionRates["fall_head"] = _fellHead ? 1.0f : 0.0f;
            _conditionRates["fall_tilt"] = _fellTilt ? 1.0f : 0.0f;
            _conditionRates["fall_lateral"] = _fellLateral ? 1.0f : 0.0f;
            _conditionRates["fall_forward"] = _fellForward ? 1.0f : 0.0f;
            return _conditionRates;
        }
    }

    public string Describe() =>
        $"WalkTermination(fallenHead={FallenHeadHeight}m, fallenTilt={FallenTiltDeg}deg, "
        + $"settle={SettleSeconds}s, movingSpeed={MovingSpeedThreshold}m/s, noSuccessCondition)";

    /// <summary>
    /// Always false: walking has no terminal success criterion, as Describe already advertises with
    /// noSuccessCondition. Distance covered is scored continuously by WalkForwardReward rather than
    /// being a goal the episode can reach, so there is no moment at which the task is "done".
    /// </summary>
    public bool SucceededThisEpisode => false;

    public void Reset()
    {
        _totalTicks = 0;
        _hasForward = false;
        _forward = Vector3.Forward;
        _fellHead = false;
        _fellTilt = false;
        _fellLateral = false;
        _fellForward = false;

        // Iterates the dictionary's own keys rather than a hand-written list, so adding a condition
        // above cannot leave stale counts leaking across episodes. Keys are buffered because the
        // indexer assignment mutates the collection being enumerated.
        foreach (string key in new List<string>(_conditionTicks.Keys))
        {
            _conditionTicks[key] = 0;
        }
    }

    /// <summary>
    /// Splits the pelvis lean into "along the direction of travel" and "across it" at the moment of
    /// failure, and latches which dominated.
    ///
    /// The lean vector is the pelvis up-axis flattened onto the ground plane - the direction the
    /// body is tipping. Projected onto the episode's own forward axis rather than a world axis, so
    /// the answer stays correct if the spawn pose ever gains a random yaw. Same yaw-level treatment
    /// BalanceController.UpdateIcpEscapeDistance uses on the capture point.
    /// </summary>
    private void RecordFallDirection(in RlContext context)
    {
        Vector3 lean = context.Pelvis.GlobalTransform.Basis.Y;
        lean.Y = 0.0f;
        if (lean.LengthSquared() <= 1e-6f)
        {
            return;
        }

        float along = Mathf.Abs(lean.Dot(_forward));
        float across = (lean - _forward * lean.Dot(_forward)).Length();
        if (across >= along)
        {
            _fellLateral = true;
        }
        else
        {
            _fellForward = true;
        }
    }

    public bool IsTerminal(in RlContext context, out string reason)
    {
        if (!_hasForward)
        {
            Vector3 facing = -context.Pelvis.GlobalTransform.Basis.Z;
            facing.Y = 0.0f;
            _forward = facing.LengthSquared() > 1e-6f ? facing.Normalized() : Vector3.Forward;
            _hasForward = true;
        }

        // Diagnostics are accumulated before any early return, so a fallen episode still reports
        // which conditions held while it lasted. Skipping them on the terminal tick would bias
        // every rate toward the healthy ticks that preceded it.
        bool fallen = RecordConditions(context, out bool headOk, out bool tiltOk);

        if (fallen && context.EpisodeElapsedSeconds >= SettleSeconds)
        {
            // Latched here rather than inside RecordConditions: that runs every tick, and what is
            // wanted is the state on the tick the episode actually ended, not a running tally.
            // Both can be true at once - a body folded forward and low trips each check - which is
            // itself worth seeing, so they are separate flags rather than one enum.
            _fellHead = !headOk;
            _fellTilt = !tiltOk;
            RecordFallDirection(context);

            reason = "Fallen";
            return true;
        }

        if (context.EpisodeElapsedSeconds >= context.MaxEpisodeSeconds)
        {
            reason = "TimeLimit";
            return true;
        }

        reason = "None";
        return false;
    }

    /// <summary>
    /// Updates the per-condition tick counts and returns whether the body counts as fallen.
    ///
    /// Every condition is evaluated rather than short-circuited, for the reason given in
    /// UprightTermination.IsStanding: the rates are how "why is it failing" stays answerable, and a
    /// short-circuit silently stops counting the conditions after the first one that fails.
    /// </summary>
    private bool RecordConditions(in RlContext context, out bool headOk, out bool tiltOk)
    {
        ActiveBone? head = context.Ragdoll.Head;
        float headHeight = head != null && GodotObject.IsInstanceValid(head)
            ? head.GlobalPosition.Y
            : 0.0f;

        headOk = headHeight >= FallenHeadHeight;
        tiltOk = context.Balance.CurrentTiltAngleDeg <= FallenTiltDeg;

        // One foot is enough: a walking gait spends most of its time in single support, and
        // requiring both would report a correct stride as ungrounded for half of every cycle.
        bool grounded = context.Balance.IsGroundedL || context.Balance.IsGroundedR;
        bool moving = context.Balance.CenterOfMassVelocity.Dot(_forward) >= MovingSpeedThreshold;

        _totalTicks++;
        if (tiltOk) { _conditionTicks["upright"]++; }
        if (headOk) { _conditionTicks["head"]++; }
        if (grounded) { _conditionTicks["grounded"]++; }
        if (moving) { _conditionTicks["moving"]++; }

        return !headOk || !tiltOk;
    }
}

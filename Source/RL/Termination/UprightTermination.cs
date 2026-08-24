using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Termination;

/// <summary>
/// Termination for the upright tasks: succeed on standing stably, fail on inversion, time out as a
/// backstop.
///
/// Shared by stand, get-up and perturbation. The perturbation task passes endEpisodeOnSuccess:false
/// so that success stops being absorbing - see that flag for why it must.
///
/// The success condition matters as much as the reward. Without one, an episode always runs the
/// full timer, so the agent has no way to distinguish "stood up" from "stood up then fell over at
/// t=7.9s" - both collect the same total. Ending the episode the moment standing is achieved and
/// held makes success an absorbing, unambiguous outcome.
///
/// Standing must be *held* rather than merely touched, or the agent learns to lunge upward through
/// the threshold and collapse - that momentarily satisfies a height check while being the opposite
/// of the intended behaviour.
/// </summary>
public sealed class UprightTermination : IRlTerminationCondition, IRlTerminationDiagnostics
{
    /// <summary>
    /// Whether reaching the standing criterion ENDS the episode (and pays StandingBonus).
    ///
    /// True for the get-up: success there is a one-off achievement, and making it absorbing is what
    /// distinguishes "stood up" from "stood up then fell over at t=7.9 s".
    ///
    /// False for the balance/perturbation task, where it is actively wrong. That task fires a ball
    /// at the dummy and asks it to stay upright; if success absorbed, the episode would end at
    /// roughly 0.6 s of settling plus StandingHoldSeconds - i.e. BEFORE the first ball lands at
    /// 1.0 s - and the perturbation would never be experienced at all. Worse, with a ball arriving
    /// every N seconds each hit resets the hold counter, so "hold 1.5 s continuously" and "get hit
    /// every 2 s" fight each other by construction and success becomes unreachable no matter how
    /// good the policy is.
    ///
    /// With this false the episode always runs its full window and the reward's per-tick upright
    /// term does the scoring, which is the honest expression of "stayed up through the hit".
    ///
    /// It does NOT stop StandingBonus being paid. It used to: EvaluateTerminal keyed the bonus off
    /// reason "Standing", which only this flag can produce, so turning absorption off silently
    /// removed the largest term in the reward and left the agent with no gradient at all toward the
    /// criterion it was being scored on. Measured over 5.3M steps of perturbation_v10:
    /// reward/terminal 0.000 at every one of 470 logged points, standing/all +0.009, and 82% of the
    /// reward gain coming from the shaping potential while standing/tilt DEGRADED. The bonus is now
    /// keyed off <see cref="SucceededThisEpisode"/>, which is independent of this flag.
    ///
    /// Settable per EPISODE rather than fixed per scene, because mixed-task training varies it: a
    /// perturbation episode and a stand episode run in the same process minutes apart and need
    /// opposite answers. The bridge assigns it in ResetEpisode.
    /// </summary>
    public bool EndEpisodeOnSuccess { get; set; } = true;

    /// <summary>
    /// Whether a fall ends the episode. Set per-episode by the bridge to whether this episode
    /// started standing, because the same component also serves the prone get-up task.
    ///
    /// Separate from <see cref="EndEpisodeOnSuccess"/> on purpose: the perturbation task turns
    /// success OFF - a hit must be able to arrive after the success criterion is met - while
    /// needing failure very much ON. Conflating the two is what left every perturbation episode
    /// running its full window face-down on the floor.
    /// </summary>
    public bool EndEpisodeOnFall { get; set; }

    /// <param name="endEpisodeOnSuccess">
    /// False for perturbation/balance training. See <see cref="EndEpisodeOnSuccess"/>.
    /// </param>
    public UprightTermination(bool endEpisodeOnSuccess = true)
    {
        EndEpisodeOnSuccess = endEpisodeOnSuccess;
    }

    /// <summary>Head height (m) above which the body counts as standing.</summary>
    private const float StandingHeadHeight = 1.35f;

    /// <summary>Max torso tilt (deg) still considered upright.</summary>
    private const float StandingTiltDeg = 30.0f;

    /// <summary>Max CoM speed (m/s) for standing to count as settled rather than mid-topple.</summary>
    private const float StandingMaxSpeed = 0.6f;

    /// <summary>
    /// Seconds the standing condition must hold continuously before success is declared.
    ///
    /// 1.5 s, up from 0.75 s. The shorter window was satisfiable by a policy that falls over
    /// immediately afterwards, and it was: at 22.7M steps start_standing/success read 0.96 while
    /// the same policy watched in RagdollStandArena - which runs PlaybackMode, so UpdateDone returns
    /// early and nothing ever terminates or resets - stayed up for 1-2 s and then went down. The
    /// 0.96 was real but measured nothing beyond t = 0.75 s, because success ends the episode
    /// there. reward/terminal was 9.38 of an ep_rew_mean of 9.02, so this bonus is effectively the
    /// entire return and the policy optimised exactly the quantity that stops being observed.
    ///
    /// This is a curriculum step, not a cure: the same gaming recurs at whatever threshold is set,
    /// and the eventual answer for standing starts is a far longer criterion, or dropping
    /// success-absorption entirely so the episode runs its full window and the reward's upright
    /// term has to earn it tick by tick.
    /// </summary>
    private const float StandingHoldSeconds = 1.5f;

    /// <summary>
    /// Max horizontal distance (m) from the Instantaneous Capture Point to the support centre for
    /// standing to count as balanced - i.e. "the mass is over the feet".
    ///
    /// Taken from DynamicSteppingModule's own escape trigger (|x| > 0.12, |z| > 0.15): the point
    /// at which the procedural controller decides balance is lost and a step is required. Reusing
    /// that figure rather than inventing one keeps "standing" meaning the same thing in the RL and
    /// Euphoria tracks, on the same rig, with the same leg geometry.
    ///
    /// Without this, height + tilt + speed are all satisfied at the apex of a topple - upright,
    /// high, and momentarily slow - so the success bonus could be collected for falling over
    /// slowly. The condition could not distinguish standing from mid-fall.
    /// </summary>
    private const float StandingMaxIcpEscape = 0.15f;

    /// <summary>Pelvis height (m) below which, combined with inversion, the attempt has failed.</summary>
    private const float FallenPelvisHeight = 0.12f;

    /// <summary>Pelvis tilt (deg) beyond which the body is inverted.</summary>
    private const float FallenTiltDeg = 170.0f;

    /// <summary>
    /// Head height (m) below which an episode that STARTED STANDING counts as fallen.
    ///
    /// The inversion test above requires the pelvis under 0.12 m AND tilt past 170 deg - upside
    /// down with its hips on the floor - and a normal topple ends at roughly 0.25 m and 90 deg, so
    /// it never fired. Every standing episode therefore ran the full window: measured ep_len_mean
    /// sat at 75.6 steps against a 75.6-step cap, dead flat, with the back half of each failure
    /// spent lying on the ground producing samples that teach nothing.
    ///
    /// 1.00 m against a standing head height near 1.40 m, matching WalkTermination for the same
    /// reason it chose that value: far enough below a deep crouch that a legitimate low posture is
    /// not a failure, high enough that the body is not paid for the second half of its own fall.
    /// </summary>
    private const float FallenHeadHeight = 1.00f;

    /// <summary>
    /// Torso tilt (deg) beyond which a standing-start episode counts as fallen, independent of
    /// height - a body pitched past this is not coming back on this rig.
    ///
    /// 60 rather than WalkTermination's 50, because this task is specifically about surviving an
    /// impact: a ball can put the torso past 45 deg and still be caught. It leaves a 30 deg band
    /// between StandingTiltDeg and here in which the episode is neither succeeding nor failed,
    /// which is exactly the region a recovery has to pass through.
    /// </summary>
    private const float FallenStandingTiltDeg = 60.0f;

    /// <summary>
    /// Grace period (s) at episode start during which a fall cannot be declared.
    ///
    /// The reset teleports every bone into the start pose with zero velocity and the body needs a
    /// few ticks to settle onto its feet. Without this a transient on tick one ends the episode
    /// before the policy has acted at all, which shows up as a mass of near-zero-length episodes
    /// and makes every per-episode average meaningless.
    /// </summary>
    private const float SettleSeconds = 0.25f;

    private float _standingHeldSeconds;

    /// <summary>
    /// Latched once <see cref="StandingHoldSeconds"/> of continuous standing is reached, whether or
    /// not <see cref="EndEpisodeOnSuccess"/> lets that end the episode.
    ///
    /// This is the honest success signal for the balance task. Note the timing that makes it
    /// meaningful there rather than trivially true: the hold is 1.5 s and the first ball lands at
    /// 1.0 s, so the criterion CANNOT be satisfied before the impact. On a perturbation episode a
    /// latched true therefore means "was knocked, recovered, and then stood stably for 1.5 s
    /// continuous" - which is the behaviour the task exists to train.
    /// </summary>
    private bool _standingAchieved;

    /// <inheritdoc />
    public bool SucceededThisEpisode => _standingAchieved;

    /// <summary>
    /// Set once the body leaves the standing envelope on tilt or ICP, after the settle grace.
    ///
    /// "Got knocked out of shape at some point", as distinct from "ended up on the floor". It is
    /// the missing half of the recovery question: v11 could report that 58.6% of episodes held the
    /// criterion, but not whether those episodes were ever actually troubled.
    /// </summary>
    private bool _wasDisturbed;

    /// <summary>Ticks this episode on which each success sub-condition held, plus the tick count.</summary>
    private readonly Dictionary<string, int> _conditionTicks = new()
    {
        ["grounded"] = 0,
        ["both_grounded"] = 0,
        ["head"] = 0,
        ["tilt"] = 0,
        ["speed"] = 0,
        ["icp"] = 0,
        ["all"] = 0,
    };

    private int _totalTicks;

    /// <summary>Reused across episodes; Reset zeroes it rather than reallocating (see the reward's note).</summary>
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
            // ride this dictionary because the bridge already forwards every entry as stand_<key>
            // and the trainer averages those across the rollout's episodes, which turns a 0/1 flag
            // into exactly the rate wanted: "what fraction of episodes were disturbed / recovered".
            // Adding a parallel channel for two booleans would duplicate that plumbing for nothing.
            _conditionRates["disturbed"] = _wasDisturbed ? 1.0f : 0.0f;
            _conditionRates["recovered"] = _wasDisturbed && _standingAchieved ? 1.0f : 0.0f;
            return _conditionRates;
        }
    }

    public string Describe() =>
        $"UprightTermination(standHead={StandingHeadHeight}m, standTilt={StandingTiltDeg}deg, "
        + $"standSpeed={StandingMaxSpeed}m/s, standIcpEscape={StandingMaxIcpEscape}m, "
        + $"eitherFootGrounded, hold={StandingHoldSeconds}s, endOnSuccess={EndEpisodeOnSuccess}, "
        + $"endOnFall={EndEpisodeOnFall}, fallenHead={FallenHeadHeight}m, "
        + $"fallenStandingTilt={FallenStandingTiltDeg}deg, settle={SettleSeconds}s, "
        + $"fallenHeight={FallenPelvisHeight}m, fallenTilt={FallenTiltDeg}deg)";

    public void Reset()
    {
        _standingHeldSeconds = 0.0f;
        _standingAchieved = false;
        _wasDisturbed = false;
        _totalTicks = 0;

        // Iterates the dictionary's own keys rather than a hand-written list, which is what let
        // "both_grounded" be added above without a second edit here - and what stops the next
        // addition from silently carrying counts across episodes. Keys are buffered because the
        // indexer assignment mutates the collection being enumerated.
        foreach (string key in new List<string>(_conditionTicks.Keys))
        {
            _conditionTicks[key] = 0;
        }
    }

    public bool IsTerminal(in RlContext context, out string reason)
    {
        // IsStanding is evaluated either way, never short-circuited: the standing/* diagnostics are
        // how "which sub-condition is blocking" stays answerable, and they are just as useful for
        // the balance task as for the get-up even when success no longer ends anything.
        if (IsStanding(context))
        {
            _standingHeldSeconds += context.Delta;
            if (_standingHeldSeconds >= StandingHoldSeconds)
            {
                // Latched BEFORE the absorption check, so the criterion is recorded identically
                // whether or not it also ends the episode. Reversing these two - which is what the
                // code did while the flag lived inside the EndEpisodeOnSuccess branch - makes
                // success unobservable on exactly the task that needs it measured.
                _standingAchieved = true;
                if (EndEpisodeOnSuccess)
                {
                    reason = "Standing";
                    return true;
                }
            }
        }
        else
        {
            _standingHeldSeconds = 0.0f;
        }

        // Inversion on the floor. Height alone is deliberately not a failure - episodes start
        // prone, at low pelvis height, by design.
        bool inverted = context.Pelvis.GlobalPosition.Y < FallenPelvisHeight
                        && context.Balance.CurrentTiltAngleDeg > FallenTiltDeg;
        if (inverted)
        {
            reason = "Inverted";
            return true;
        }

        // Falling, for an episode that started upright. Gated on EndEpisodeOnFall rather than
        // applied unconditionally because the get-up task starts prone: it begins below both
        // thresholds by design, and an ungated check would end every get-up episode on tick one.
        if (EndEpisodeOnFall && context.EpisodeElapsedSeconds >= SettleSeconds)
        {
            ActiveBone? fallenHead = context.Ragdoll.Head;
            bool headDown = fallenHead != null
                            && GodotObject.IsInstanceValid(fallenHead)
                            && fallenHead.GlobalPosition.Y < FallenHeadHeight;
            bool pitchedOver = context.Balance.CurrentTiltAngleDeg > FallenStandingTiltDeg;

            if (headDown || pitchedOver)
            {
                reason = "Fallen";
                return true;
            }
        }

        if (context.EpisodeElapsedSeconds >= context.MaxEpisodeSeconds)
        {
            reason = "TimeLimit";
            return true;
        }

        reason = "None";
        return false;
    }

    private bool IsStanding(in RlContext context)
    {
        ActiveBone? head = context.Ragdoll.Head;
        if (head == null || !GodotObject.IsInstanceValid(head))
        {
            _totalTicks++;
            return false;
        }

        // Every sub-condition is evaluated and counted, rather than short-circuited, so the rates
        // reported through IRlTerminationDiagnostics say WHICH check is blocking success.
        //
        // ONE foot is enough, matching WalkTermination for the reason given there: requiring both
        // reports single support as failure, and single support is half of every stride - and all
        // of every protective step. Demanding both feet made a step break the criterion and reset
        // the hold timer, so the only reachable success was standing rigid through the impact.
        // Measured consequence over v10+v11: standing/grounded 0.966 and DECLINING, a policy
        // optimising toward never moving its feet.
        //
        // The airborne exploit that the both-feet rule used to guard against is now closed at the
        // source instead. IcpEscapeDistance reads 0.0 both for "perfectly balanced" and for "no
        // base of support"; BalanceController.IsIcpValid separates them, so the ICP test can be
        // trusted during single support rather than disabled by it.
        bool bothGrounded = context.Balance.IsGroundedL && context.Balance.IsGroundedR;
        bool grounded = context.Balance.IsGroundedL || context.Balance.IsGroundedR;
        bool headOk = head.GlobalPosition.Y >= StandingHeadHeight;
        bool tiltOk = context.Balance.CurrentTiltAngleDeg <= StandingTiltDeg;
        bool speedOk = context.Balance.CenterOfMassVelocity.Length() <= StandingMaxSpeed;
        bool icpOk = context.Balance.IsIcpValid
                     && context.Balance.IcpEscapeDistance <= StandingMaxIcpEscape;

        _totalTicks++;
        if (grounded) { _conditionTicks["grounded"]++; }
        // Reported alongside, not instead: every run before this one measured "grounded" as BOTH
        // feet, and without this column the 0.96-0.97 history becomes silently incomparable to the
        // near-1.0 the relaxed test will produce. It is also the direct read on whether the agent
        // has started stepping at all.
        if (bothGrounded) { _conditionTicks["both_grounded"]++; }
        if (headOk) { _conditionTicks["head"]++; }
        if (tiltOk) { _conditionTicks["tilt"]++; }
        if (speedOk) { _conditionTicks["speed"]++; }
        if (icpOk) { _conditionTicks["icp"]++; }

        bool standing = grounded && headOk && tiltOk && speedOk && icpOk;
        if (standing) { _conditionTicks["all"]++; }

        // Disturbance latch: the body left the standing envelope on an axis a shove acts through,
        // after the reset transient has settled. This is what separates "recovered" from "was
        // never troubled" - without it, success and survival are the same number, which is exactly
        // what v11 measured (conditional success given survival = 1.03 across every sextile).
        if (context.EpisodeElapsedSeconds >= SettleSeconds && (!tiltOk || !icpOk))
        {
            _wasDisturbed = true;
        }

        return standing;
    }
}

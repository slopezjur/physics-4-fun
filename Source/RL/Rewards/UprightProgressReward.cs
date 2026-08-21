using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Rewards;

/// <summary>
/// Reward for getting upright and staying there, under pure RL (no procedural reference to imitate).
///
/// Shared by THREE tasks - stand, get-up and perturbation - which is why it is named for the
/// objective rather than for any one of them. It was called GetUpProgressReward, and that was
/// wrong in the same way the scene names were: standing and getting up are not different problems
/// here, they are the same objective from different start poses, and balance-under-impact is that
/// objective with something pushing back. Only walking needs a different reward, because only
/// walking wants the body to go somewhere.
///
/// Replaces the original UprightAliveReward, which was measured flat over 500k steps. The two
/// structural fixes:
///
/// 1. **A gradient that exists from the very first pose.** The dominant term is potential-based
///    shaping on head height: r = gamma*PHI(s') - PHI(s), with PHI = head height. Height changes
///    continuously as the body pushes off the floor, so every small improvement is rewarded -
///    unlike cos(tilt), which barely moves until the body is already most of the way up.
///    Potential-based shaping is used specifically because it provably leaves the optimal policy
///    unchanged (Ng, Harada & Russell 1999): it can accelerate learning but cannot invent a
///    degenerate optimum, which a naive "+height each tick" bonus absolutely can (it pays for
///    hovering rather than standing). Applied here in its undiscounted episodic form - see the
///    comment in Evaluate for why the discounted variant is wrong at this call site.
///
/// 2. **An effort penalty**, so the policy is not free to slam every actuator to its limit. This
///    is what keeps the motion from becoming the rigid full-torque flailing that a pure progress
///    reward converges to, and is the main term shaping whether the result looks human.
/// </summary>
public sealed class UprightProgressReward : IRlRewardFunction, IRlRewardDiagnostics
{
    /// <summary>Scales the height-progress shaping term.</summary>
    private const float ProgressWeight = 10.0f;

    /// <summary>Scales the sustained-uprightness term (per second).</summary>
    private const float UprightWeight = 1.0f;

    /// <summary>
    /// Scales the effort penalty (per second; effort itself is a 0-1 capacity fraction).
    ///
    /// Constructor-injected rather than constant because it must differ between discovery and
    /// refinement. HumanUP reports that applying deployment-strength control regularization from
    /// the start "fails entirely" - the penalty suppresses exactly the vigorous, wasteful motion
    /// that finding a get-up requires. Measured here too: over 63M steps at 0.25, reducing effort
    /// accounted for 6% of all reward gained while the agent never once left the floor.
    /// </summary>
    private readonly float _effortWeight;

    /// <param name="effortWeight">
    /// Near-zero (the default) during discovery; raise toward 0.25 once a get-up exists and the
    /// goal shifts to making it smooth and economical.
    /// </param>
    public UprightProgressReward(float effortWeight = 0.02f)
    {
        _effortWeight = effortWeight;
    }

    /// <summary>
    /// Head height (m) that normalises the shaping potential to 1.0.
    ///
    /// Deliberately not called "standing head height": UprightTermination has a constant of that
    /// name set to 1.35, and the two are different quantities that were easy to confuse. This one
    /// is a scale factor for the potential function; that one is the success threshold. Keeping
    /// this slightly above the success height means the potential is still rising through the last
    /// of the motion rather than saturating just before the body gets there.
    /// </summary>
    private const float PotentialReferenceHeadHeight = 1.5f;

    /// <summary>
    /// One-off payout for reaching the Standing termination.
    ///
    /// Must exceed the per-tick reward the agent forfeits by ending the episode early, or success
    /// is literally punished. Worst case that forfeit is UprightWeight * MaxEpisodeSeconds - i.e.
    /// standing at t=0 and giving up the entire episode's upright reward - so that product is the
    /// break-even point, and anything below it makes "stand up, then deliberately wobble to avoid
    /// tripping the success check" the higher-scoring strategy.
    ///
    /// There are now two windows: standing starts run 4.0 s and prone starts 8.0 s
    /// (RagdollRLBridge.MaxEpisodeSeconds / ProneMaxEpisodeSeconds). The binding case is the longer
    /// one, so break-even is 8.0 and this sits at 2.5x it - the same margin the original single 8 s
    /// design had. Deliberately kept well above break-even so succeeding is decisively better
    /// rather than marginally so.
    ///
    /// The "standing at t=0" worst case is now unreachable in any event: UprightTermination requires
    /// StandingHoldSeconds = 1.5 s of continuous success before it fires, so the most an agent can
    /// forfeit is the remaining window after that, and the real margin is wider than the figure
    /// above. Raising the hold moves this in the safe direction, which is why it needed no
    /// corresponding change here.
    ///
    /// The same bonus is paid whether the episode began standing or prone, despite those being
    /// vastly different achievements. That needs no correction: PPO does not rank episodes against
    /// one another, it fits a value function, so the critic simply learns V(prone) < V(standing)
    /// and advantages - which are what the gradient actually uses - are already relative to that.
    /// </summary>
    private const float StandingBonus = 20.0f;

    private float _previousPotential;
    private bool _hasPreviousPotential;

    /// <summary>
    /// Per-term episode totals, signed so shaping + upright + effort + terminal == episode reward.
    /// Reused across episodes rather than reallocated - Evaluate runs every physics tick on every
    /// one of 40 parallel environments, so a per-tick allocation here is not free.
    /// </summary>
    private readonly Dictionary<string, float> _componentTotals = new()
    {
        ["shaping"] = 0.0f,
        ["upright"] = 0.0f,
        ["effort"] = 0.0f,
        ["terminal"] = 0.0f,
    };

    public IReadOnlyDictionary<string, float> EpisodeComponentTotals => _componentTotals;

    public void Reset()
    {
        // Cleared rather than zeroed: the first tick of an episode has no previous state to
        // difference against, and pretending it was zero would hand out a large spurious reward
        // for simply existing at whatever height the reset pose happens to be.
        _hasPreviousPotential = false;
        _previousPotential = 0.0f;

        _componentTotals["shaping"] = 0.0f;
        _componentTotals["upright"] = 0.0f;
        _componentTotals["effort"] = 0.0f;
        _componentTotals["terminal"] = 0.0f;
    }

    /// <summary>
    /// Pays the standing bonus, and nothing for the failure modes. Timing out is deliberately
    /// neutral rather than penalised: the per-tick terms already reflect how far the attempt got,
    /// and an extra timeout penalty would push the agent toward ending episodes early by throwing
    /// itself into the Inverted state.
    /// </summary>
    public string Describe() =>
        $"UprightProgressReward(progress={ProgressWeight}, upright={UprightWeight}, "
        + $"effort={_effortWeight}, standingBonus={StandingBonus}, potentialHeadHeight={PotentialReferenceHeadHeight})";

    public float EvaluateTerminal(in RlContext context, string reason, bool succeeded)
    {
        // Keyed on the success FLAG, not on the reason string. Reason "Standing" only exists when
        // the termination is configured to make success absorbing, so the old test paid nothing on
        // the balance task - where success deliberately does not end the episode - and the agent
        // trained for 5.3M steps against a reward whose largest term could not fire.
        //
        // Ending on a fall still forfeits it, even if the criterion was met earlier in the episode.
        // Without that guard "hold 1.5 s, bank 20, then collapse" scores identically to staying up,
        // and this reward has already been gamed once in exactly that shape - see the note on
        // UprightTermination.StandingHoldSeconds, where a 0.75 s hold produced start_standing/success
        // 0.96 from a policy that fell over immediately afterwards. A balance episode that survives
        // reaches the time limit; one that does not ends as "Fallen".
        bool payable = succeeded && reason != "Fallen";
        float bonus = payable ? StandingBonus : 0.0f;
        _componentTotals["terminal"] += bonus;
        return bonus;
    }

    public float Evaluate(in RlContext context)
    {
        float potential = ComputePotential(context);

        // Undiscounted difference, NOT gamma*PHI' - PHI.
        //
        // The discounted form is only correct when applied once per agent decision with the
        // trainer's own gamma. This runs once per PHYSICS tick - 8 ticks per decision - so a
        // gamma factor injected a constant (gamma-1)*PHI drift every tick: with PHI ~ 0.24 held
        // roughly steady, that is -0.1*0.24 per tick, ~ -23 over a 968-tick episode. Measured
        // exactly that: ep_rew_mean pinned at -23 and flat, a floor no policy could escape.
        //
        // The plain difference telescopes exactly to ProgressWeight * (PHI_end - PHI_start)
        // regardless of tick rate or action_repeat - i.e. "reward for height actually gained" -
        // which is both drift-free and directly interpretable.
        float shaping = 0.0f;
        if (_hasPreviousPotential)
        {
            shaping = ProgressWeight * (potential - _previousPotential);
        }
        _previousPotential = potential;
        _hasPreviousPotential = true;

        float uprightDot = Mathf.Clamp(Mathf.Cos(Mathf.DegToRad(context.Balance.CurrentTiltAngleDeg)), 0.0f, 1.0f);
        float upright = UprightWeight * uprightDot * context.Delta;

        float effort = _effortWeight * ComputeNormalisedEffort(context) * context.Delta;

        // Recorded signed (effort negative) so the four terms reconstruct the episode reward
        // exactly - see IRlRewardDiagnostics.EpisodeComponentTotals.
        _componentTotals["shaping"] += shaping;
        _componentTotals["upright"] += upright;
        _componentTotals["effort"] -= effort;

        return shaping + upright - effort;
    }

    /// <summary>
    /// Potential = normalised head height. Head rather than pelvis because it is the last thing to
    /// come up in a real get-up, so it keeps producing gradient through the final rise instead of
    /// saturating early the way pelvis height does once kneeling.
    /// </summary>
    private static float ComputePotential(in RlContext context)
    {
        ActiveBone? head = context.Ragdoll.Head;
        if (head == null || !GodotObject.IsInstanceValid(head))
        {
            return 0.0f;
        }

        return Mathf.Clamp(head.GlobalPosition.Y / PotentialReferenceHeadHeight, 0.0f, 1.0f);
    }

    /// <summary>
    /// Mean fraction of each actuator's own torque capacity currently in use, in [0,1].
    ///
    /// Normalising per bone rather than against one global constant matters: MaxTorque spans
    /// 50 N.m at the wrist to 1800 N.m at the hip on this rig, so a single reference figure both
    /// mis-scales the penalty (it dwarfed the progress term) and mis-distributes it (a saturated
    /// wrist would look "cheap" while a lightly-loaded hip looked expensive). A capacity fraction
    /// is scale-free and penalises straining any muscle equally.
    /// </summary>
    private static float ComputeNormalisedEffort(in RlContext context)
    {
        float total = 0.0f;
        int counted = 0;
        foreach (ActiveBone? bone in context.ControlledBones)
        {
            if (bone == null || !GodotObject.IsInstanceValid(bone))
            {
                continue;
            }

            float capacity = Mathf.Max(1.0f, bone.MaxTorque);
            total += Mathf.Clamp(bone.LastPdTorque.Length() / capacity, 0.0f, 1.0f);
            counted++;
        }

        return counted > 0 ? total / counted : 0.0f;
    }
}

using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;

namespace Physics4Fun.RL.Interfaces;

/// <summary>
/// Everything an RL observation/reward/termination strategy is allowed to see, passed by
/// reference each tick. Mirrors the BalanceContext pattern in Physics4Fun.Ragdoll.Interfaces:
/// strategies read a context they cannot mutate, rather than reaching into the ragdoll or the
/// bridge themselves. That keeps a reward function from quietly acquiring the ability to drive
/// the body, which is the bridge's job alone.
/// </summary>
public readonly record struct RlContext(
    HumanoidRagdoll Ragdoll,
    ActiveBone Pelvis,
    BalanceController Balance,
    IReadOnlyList<ActiveBone?> ControlledBones,
    float EpisodeElapsedSeconds,
    float MaxEpisodeSeconds,
    float Delta
);

/// <summary>
/// Builds the observation vector handed to the policy. Implementations must return exactly
/// <see cref="Size"/> floats every call - the size is published to Python once at handshake, and a
/// mismatch mid-episode desynchronises the whole transport.
/// </summary>
public interface IRlObservationBuilder
{
    /// <summary>Fixed observation width, published to Python during the env_info handshake.</summary>
    int Size { get; }

    float[] Build(in RlContext context);

    /// <summary>
    /// Human-readable description including the tuning constants, recorded into each training
    /// run's manifest. Implemented here rather than copied into the Python side so the record can
    /// never drift from the values actually compiled into the build.
    /// </summary>
    string Describe();
}

/// <summary>
/// Translates a policy's raw output vector into per-bone joint target offsets.
///
/// Extracted for the same reason observation/reward/termination were: it is a policy of its own
/// that gets rewritten as the setup evolves, and it had already been rewritten twice inside the
/// bridge - once when the action range was widened from 0.6 to 2.6 rad, and again when uniform
/// scaling was replaced by per-axis joint limits. The second of those was a genuine defect (71.8%
/// of the commanded range lay past a hard stop) that lived unnoticed inside 683 lines of transport
/// and episode bookkeeping. Encoding is a separable concern; giving it a seam makes the next
/// variant - an action-bound curriculum, say - a new class rather than another edit here.
/// </summary>
public interface IRlActionSpace
{
    /// <summary>Total floats the policy emits per decision. Published to Python at handshake.</summary>
    int Size { get; }

    /// <summary>
    /// Called once the controlled bones are known, so limits can be read from the live joints
    /// rather than restated as constants that drift from the rig.
    /// </summary>
    void Bind(IReadOnlyList<ActiveBone?> controlledBones);

    /// <summary>Clears per-episode statistics. Does not re-read the rig.</summary>
    void Reset();

    /// <summary>Decodes <paramref name="action"/> into <paramref name="offsets"/>, one per bone.</summary>
    void Decode(float[] action, Quaternion[] offsets);

    /// <summary>Description including the resolved ranges, for the run manifest.</summary>
    string Describe();
}

/// <summary>
/// Optional add-on reporting how hard the policy is pushing against its own bounds.
///
/// Separate from <see cref="IRlActionSpace"/> (ISP) for the same reason as the other diagnostics
/// interfaces: decoding works without it. Worth measuring because saturated commands do not bend
/// a joint further - they pin the actuator at a hard stop under full torque, and every
/// out-of-range action then produces an identical physical result, flattening the policy gradient.
/// </summary>
public interface IRlActionDiagnostics
{
    /// <summary>Per-episode action statistics, keyed by name (e.g. saturation rate in [0,1]).</summary>
    IReadOnlyDictionary<string, float> EpisodeActionStats { get; }
}

/// <summary>
/// Per-tick reward. Kept deliberately separate from the bridge so reward shaping - the part of an
/// RL setup that gets rewritten most often - can be swapped without touching transport, episode
/// bookkeeping, or the actuator seam.
/// </summary>
public interface IRlRewardFunction
{
    /// <summary>Called once per episode start; clears any per-episode accumulators.</summary>
    void Reset();

    /// <summary>Reward earned during this tick alone. Callers accumulate; implementations do not.</summary>
    float Evaluate(in RlContext context);

    /// <summary>
    /// One-off reward applied when the episode ends, given the termination reason.
    ///
    /// Separate from <see cref="Evaluate"/> because a per-tick reward cannot see WHY an episode
    /// stopped, and "reached the goal" versus "ran out of time" must be paid differently. Without
    /// this, ending early on success silently forfeits the remaining per-tick reward, which pays
    /// the agent to avoid succeeding.
    /// </summary>
    float EvaluateTerminal(in RlContext context, string reason);

    /// <summary>Description including weights, for the run manifest. See IRlObservationBuilder.Describe.</summary>
    string Describe();
}

/// <summary>
/// Optional add-on for reward functions that can break their output down by term.
///
/// Deliberately separate from <see cref="IRlRewardFunction"/> (ISP): a reward is completely usable
/// without it, and the bridge simply skips the extra reporting for any reward that does not
/// implement it.
///
/// Exists because a single scalar reward is unfalsifiable in TensorBoard. A curve sitting near
/// zero cannot distinguish "no term is producing signal" from "two terms are cancelling", and
/// guessing between those from the total alone produced a confidently wrong diagnosis once
/// already. Per-term totals make that distinction directly observable instead of inferred.
/// </summary>
public interface IRlRewardDiagnostics
{
    /// <summary>
    /// Per-term totals accumulated since the last <see cref="IRlRewardFunction.Reset"/>, keyed by
    /// term name. Signed so that the terms sum to the episode reward the trainer already logs -
    /// penalties are stored negative. That invariant is what makes the decomposition checkable
    /// rather than merely suggestive.
    /// </summary>
    IReadOnlyDictionary<string, float> EpisodeComponentTotals { get; }
}

/// <summary>
/// Optional add-on for termination conditions that can report how often each part of their
/// SUCCESS criterion held.
///
/// Exists because "the success condition never fired" has two completely different causes that
/// look identical from outside: the agent never reached the goal, or the goal is unreachable on
/// this rig because a threshold is mis-set. Per-condition rates separate them. Without this the
/// only way to tell them apart is to guess, and a wrong guess sends an entire training run after
/// a target that cannot be hit.
/// </summary>
public interface IRlTerminationDiagnostics
{
    /// <summary>Fraction of this episode's ticks on which each named sub-condition held, in [0,1].</summary>
    IReadOnlyDictionary<string, float> EpisodeConditionRates { get; }
}

/// <summary>Decides when an episode ends, and why (the reason surfaces in logs and the HUD).</summary>
public interface IRlTerminationCondition
{
    /// <summary>Called once per episode start; clears any per-episode accumulators (e.g. hold timers).</summary>
    void Reset();

    bool IsTerminal(in RlContext context, out string reason);

    /// <summary>Description including thresholds, for the run manifest. See IRlObservationBuilder.Describe.</summary>
    string Describe();
}

using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Rewards;

/// <summary>
/// Reward for learning to walk forward, resumed from a policy that can already stand.
///
/// Structurally different from UprightProgressReward, and deliberately so. That reward is dominated
/// by potential-based shaping on head height, because getting up is a one-off transition to a goal
/// state. Walking has no goal state - it is a periodic gait that must be sustained - so the
/// dominant term here is a RATE (forward speed) rather than a difference in a potential. A
/// potential on distance travelled would telescope to "total displacement" and pay identically for
/// walking 6 m and for falling forward 6 m, which is precisely the failure to avoid.
///
/// The reward is a PRODUCT of forward speed and uprightness, not a sum of weighted terms. That is
/// the single most important thing about it, and it was arrived at by measurement rather than by
/// taste.
///
/// The first version was additive: velocity + alive + upright - heading - effort, weighted so that
/// walking at target speed scored 48 over a 6 s episode against 12 for standing still. Resumed from
/// a policy that could already stand, it converged inside five minutes to standing perfectly still -
/// walk/fell 0.164 -> 0.011 while walk/forward_speed 0.0186 -> 0.0069. It did not fail to learn; it
/// learned the wrong thing, fast.
///
/// The 4x ratio was not the flaw. The flaw is that an additive alive term pays for existing, so
/// standing still had a positive score worth protecting, and the path to walking descends before it
/// climbs: a real step risks a fall while creeping forward pays almost nothing. Seeding forward
/// momentum at episode start (RagdollRLBridge.InitialForwardSpeed) was tried next and was absorbed
/// just as cleanly - the agent learned to plant and kill 0.65 m/s in about 0.18 s, roughly 289 N,
/// well inside foot friction. Any fix gets absorbed while standing still still pays.
///
/// In product form standing still scores exactly ZERO, because the speed factor is zero. There is
/// no longer a comfortable state to protect, and the only way to score at all is to move forward
/// while upright. This is the formulation the DeepMind control suite uses for its locomotion tasks,
/// for the same reason.
/// </summary>
public sealed class WalkForwardReward : IRlRewardFunction, IRlRewardDiagnostics, IRlWalkDiagnostics
{
    /// <summary>
    /// Scales the whole product term, in reward per second at full speed and fully upright.
    ///
    /// Now the ONLY positive term in the function, so it sets the entire scale rather than
    /// competing with anything: a 6 s episode walking at target speed while upright is worth 36,
    /// and standing still is worth 0 regardless of this value. Under the additive form the number
    /// mattered because it had to out-argue the alive bonus; under the product form it does not,
    /// which is why it was left at 6.0 when that form was removed.
    /// </summary>
    private const float VelocityWeight = 6.0f;

    /// <summary>
    /// Forward speed (m/s) at which the speed factor saturates. Above it, going faster buys nothing
    /// and the only remaining way to score higher is to keep going for LONGER.
    ///
    /// 0.4 m/s, down from 1.0. The saturation was always the mechanism meant to stop the reward
    /// paying for a dive - but at 1.0 m/s it never engaged, because the policy never got near it.
    /// A 2-hour run settled at 0.43 m/s with a speed factor of only 0.43, so faster still paid
    /// linearly, and it optimised the wrong axis: over the last six segments walk/forward_speed
    /// crept 0.391 -> 0.428 while walk/alive_fraction sat flat at 0.33 and walk/fell was pinned at
    /// 1.000. standing/grounded fell 0.983 -> 0.834 across the run, so it was becoming more
    /// ballistic, not more stable - a 2-second sprint ending in a guaranteed fall.
    ///
    /// That is a local optimum with a real barrier: escaping it means going SLOWER for a while, to
    /// trade peak speed for balance, and every step in that direction loses reward immediately.
    /// Setting the target below the speed already achieved removes the barrier by construction -
    /// the speed factor is pinned at 1.0, extra speed is worth exactly nothing, and the only
    /// gradient left points at surviving the window.
    ///
    /// Deliberately a CURRICULUM value, not a final one. 0.4 m/s is a slow walk. Once the dummy can
    /// sustain it for a full window, raising this is the natural next rung - and the frontier
    /// machinery in RagdollRLBridge could drive it the same way it drives start pose.
    /// </summary>
    private const float TargetSpeed = 0.4f;

    // There is deliberately no alive term and no standalone upright term. Both existed in the first
    // version of this reward and both were removed after measurement - see the class summary.

    /// <summary>
    /// Penalty per second per metre of lateral drift from the line the episode started on.
    ///
    /// This is what makes "walk forward" mean a straight line rather than any locomotion at all.
    /// It is needed because the observation carries no heading: BodyStateObservation is root-relative
    /// for bone rotations, so the policy cannot see which way it is pointing in world terms and
    /// nothing else in the reward would object to walking in a circle.
    ///
    /// Kept small relative to the velocity term. Penalising drift hard enough to dominate would
    /// teach standing still - the drift-optimal behaviour - which is the optimum this whole
    /// function is shaped to avoid.
    /// </summary>
    private const float HeadingWeight = 0.5f;

    /// <summary>
    /// One-off penalty for ending the episode by falling. Zero, deliberately.
    ///
    /// It was -5. Under the product reward that becomes actively harmful: early in training the
    /// agent cannot walk, so every outcome scores about zero, and a negative fall penalty makes
    /// standing still (0) strictly better than attempting anything (risking -5). That is precisely
    /// the risk aversion this rewrite exists to remove, and it would reintroduce it through the back
    /// door.
    ///
    /// The cost of falling is already the forfeited remainder of the episode. Once the agent can
    /// earn 6/s, going down at t = 2 s of 6 s gives up 24 - a real penalty that scales with how much
    /// the agent has to lose, and one that is correctly near zero while it has nothing to lose yet.
    /// </summary>
    private const float FallPenalty = 0.0f;

    private readonly float _effortWeight;

    public WalkForwardReward(float effortWeight = 0.02f)
    {
        _effortWeight = effortWeight;
    }

    /// <summary>
    /// Forward axis and origin for this episode, captured on the first tick.
    ///
    /// Captured per episode rather than fixed to world -Z so the reward means "the way the body was
    /// facing when it started", which stays correct if the spawn pose ever gains a random yaw. It
    /// is then held CONSTANT for the rest of the episode: recomputing it each tick would let the
    /// agent turn to face whatever direction it happens to be moving and collect the velocity term
    /// for walking in a circle, which is exactly what HeadingWeight exists to prevent.
    ///
    /// -Basis.Z is the rig convention, matching BiomechanicalKinematics and OrientationClassifier.
    /// </summary>
    private Vector3 _forward = Vector3.Forward;
    private Vector3 _origin;
    private bool _hasStart;

    private float _speedSum;
    private int _speedTicks;
    private float _peakDistance;
    private float _lastLateral;
    private float _lastElapsed;
    private float _windowSeconds;
    private bool _fell;

    private readonly Dictionary<string, float> _componentTotals = new()
    {
        ["progress"] = 0.0f,
        ["heading"] = 0.0f,
        ["effort"] = 0.0f,
        ["terminal"] = 0.0f,
    };

    /// <summary>Reused across episodes rather than reallocated - see UprightProgressReward.</summary>
    private readonly Dictionary<string, float> _walkStats = new();

    public IReadOnlyDictionary<string, float> EpisodeComponentTotals => _componentTotals;

    /// <summary>
    /// Distance and speed in METRES, not rates - which is why these are reported here rather than
    /// through the termination's condition rates. "Walked 0.4 m" and "walked 4 m" are the whole
    /// question for this task, and a fraction in [0,1] cannot express either.
    /// </summary>
    public IReadOnlyDictionary<string, float> EpisodeWalkStats
    {
        get
        {
            _walkStats["forward_speed"] = _speedTicks > 0 ? _speedSum / _speedTicks : 0.0f;
            _walkStats["distance"] = _peakDistance;
            _walkStats["lateral_drift"] = _lastLateral;
            _walkStats["fell"] = _fell ? 1.0f : 0.0f;

            // Fraction of the episode window actually survived. Deliberately NOT a condition rate:
            // rates divide by the ticks that HAPPENED, so an episode cut short at 10% would report
            // 1.0 for anything that held throughout - the exact opposite of what is being asked.
            _walkStats["alive_fraction"] = _windowSeconds > 0.0f
                ? Mathf.Clamp(_lastElapsed / _windowSeconds, 0.0f, 1.0f)
                : 0.0f;
            return _walkStats;
        }
    }

    public void Reset()
    {
        _hasStart = false;
        _forward = Vector3.Forward;
        _origin = Vector3.Zero;
        _speedSum = 0.0f;
        _speedTicks = 0;
        _peakDistance = 0.0f;
        _lastLateral = 0.0f;
        _lastElapsed = 0.0f;
        _windowSeconds = 0.0f;
        _fell = false;

        _componentTotals["progress"] = 0.0f;
        _componentTotals["heading"] = 0.0f;
        _componentTotals["effort"] = 0.0f;
        _componentTotals["terminal"] = 0.0f;
    }

    public string Describe() =>
        $"WalkForwardReward(velocity={VelocityWeight}, targetSpeed={TargetSpeed}m/s, "
        + $"form=velocity*upright(product), heading={HeadingWeight}, "
        + $"effort={_effortWeight}, fallPenalty={FallPenalty})";

    /// <param name="succeeded">
    /// Unused: walking has no success criterion, so WalkTermination reports false unconditionally.
    /// This reward pays only a failure penalty, which the reason string alone identifies.
    /// </param>
    public float EvaluateTerminal(in RlContext context, string reason, bool succeeded)
    {
        if (reason != "Fallen")
        {
            return 0.0f;
        }

        _fell = true;
        _componentTotals["terminal"] += FallPenalty;
        return FallPenalty;
    }

    public float Evaluate(in RlContext context)
    {
        Vector3 pelvisPos = context.Pelvis.GlobalPosition;

        if (!_hasStart)
        {
            // Projected onto the horizontal plane: the pelvis pitches through the gait cycle, and
            // an unprojected forward axis would tilt with it and start crediting vertical motion as
            // progress. Falls back to world -Z if the horizontal component is degenerate, which
            // cannot happen from a standing start but would otherwise give a zero-length axis and
            // silently zero the entire velocity term.
            Vector3 facing = -context.Pelvis.GlobalTransform.Basis.Z;
            facing.Y = 0.0f;
            _forward = facing.LengthSquared() > 1e-6f ? facing.Normalized() : Vector3.Forward;
            _origin = pelvisPos;
            _hasStart = true;
        }

        Vector3 travel = pelvisPos - _origin;
        travel.Y = 0.0f;

        float along = travel.Dot(_forward);
        _peakDistance = Mathf.Max(_peakDistance, along);

        // Magnitude of the component of travel perpendicular to the start axis.
        _lastLateral = (travel - (_forward * along)).Length();

        float forwardSpeed = context.Balance.CenterOfMassVelocity.Dot(_forward);
        _speedSum += forwardSpeed;
        _speedTicks++;

        _lastElapsed = context.EpisodeElapsedSeconds;
        _windowSeconds = context.MaxEpisodeSeconds;

        // Clamped to [0,1], not [-1,1]. Under the product form a negative speed factor would flip
        // the sign of the uprightness factor, so a policy walking BACKWARDS while upright would be
        // penalised more than one walking backwards while toppling - which is incoherent. Backward
        // motion earns nothing; it does not earn negative.
        float speedFactor = Mathf.Clamp(forwardSpeed / TargetSpeed, 0.0f, 1.0f);

        float uprightFactor = Mathf.Clamp(Mathf.Cos(Mathf.DegToRad(context.Balance.CurrentTiltAngleDeg)), 0.0f, 1.0f);

        // The product. Either factor at zero scores zero, which is the whole point: standing still
        // upright pays nothing, and sprinting while face-down pays nothing.
        float progress = VelocityWeight * speedFactor * uprightFactor * context.Delta;

        float heading = HeadingWeight * _lastLateral * context.Delta;

        float effort = _effortWeight * ComputeNormalisedEffort(context) * context.Delta;

        // Signed so the terms reconstruct the episode reward exactly - see IRlRewardDiagnostics.
        _componentTotals["progress"] += progress;
        _componentTotals["heading"] -= heading;
        _componentTotals["effort"] -= effort;

        return progress - heading - effort;
    }

    /// <summary>
    /// Mean fraction of each actuator's torque capacity in use, in [0,1]. Identical in form to
    /// UprightProgressReward.ComputeNormalisedEffort - see the reasoning there for why the
    /// normalisation is per bone rather than against one global torque constant.
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

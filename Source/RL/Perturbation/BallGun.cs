using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Perturbation;

/// <summary>
/// Fires physical balls at the ragdoll on a fixed interval, to train balance recovery from
/// external impacts.
///
/// Separate from <see cref="Physics4Fun.Physics.ProjectileShooter"/> rather than a mode added to
/// it: that one is mouse-driven and fires from the active camera, which is exactly wrong for
/// training. Forty headless processes have no camera and no mouse, and a human-timed shot is not
/// reproducible between runs. This one is time-driven, camera-independent, and seeded per episode.
/// The two share only the idea of a ball.
///
/// **Deliberately not told to the policy.** No observation carries the gun's state or the incoming
/// ball. The agent has to react to what it feels through its own proprioception - the standard
/// setup for perturbation training, and the difference between learning to *recover* and learning
/// to *anticipate a scripted event*. Dodging is a different task and would need the ball in the
/// observation vector.
/// </summary>
public partial class BallGun : Node3D, IRlPerturbationDiagnostics, IRlPerturbationSchedule
{
    /// <summary>The ragdoll to aim at. Balls target its chest, falling back to the pelvis.</summary>
    [Export] public HumanoidRagdoll? Target { get; set; }

    /// <summary>
    /// Optional. When set, the gun resets itself whenever the bridge starts a new episode.
    ///
    /// The gun watches the bridge rather than the bridge driving the gun, so perturbation stays a
    /// purely additive node: no scene without it changes, and RagdollRLBridge needs no knowledge
    /// that perturbation exists.
    /// </summary>
    [Export] public RagdollRLBridge? Bridge { get; set; }

    /// <summary>
    /// Simulation seconds between shots.
    ///
    /// 10 s is deliberately LONGER than the training episode window (5 s), which is how "exactly one
    /// ball per episode" is expressed without a shot counter: the second shot would fall past the
    /// end of the episode. One hit per episode keeps credit assignment clean - one perturbation, one
    /// recovery, one outcome. With several hits per episode there is no way to tell which one caused
    /// the fall.
    ///
    /// Not set to exactly the window length: the second shot would then land on the episode
    /// boundary, and whether it fires at all would depend on tick ordering.
    ///
    /// The ARENA overrides this at runtime, because there the point is to watch repeated recoveries
    /// rather than to train, and playback has no episode boundary for a long interval to hide
    /// behind. That override used to live in the arena .tscn as an inline BallGun node; it does
    /// not any more, and looking for it there will find nothing. Agents are now instantiated from
    /// a shared `*Agent.tscn` by RagdollSpawner, so no editor-time node exists to override. The
    /// arena sets it through PolicyAutoLoader.PerturbationIntervalOverride via
    /// <see cref="Physics4Fun.RL.Interfaces.IRlPerturbationSchedule"/> instead.
    /// </summary>
    [Export] public float IntervalSeconds { get; set; } = 10.0f;

    /// <summary>
    /// Grace period at episode start before the first shot, so the dummy is not hit mid-teleport
    /// while the actuators are still settling into the reset pose.
    /// </summary>
    [Export] public float FirstShotDelaySeconds { get; set; } = 1.0f;

    /// <summary>
    /// Launch speed (m/s). Impulse is scaled by mass, so this is the actual ball speed.
    ///
    /// 1.5 kg at 6 m/s, down from the 3 kg at 9 m/s this was first written with. That original shot
    /// was not a perturbation, it was a knockdown - computed against the rig's real numbers
    /// (80.6 kg summed from the bone masses, CoM 0.840 m, I about the ground line 69.5 kg.m^2,
    /// chest impact at 1.250 m):
    ///
    ///     energy to tip a PASSIVE body over the toe edge          6.74 J
    ///     3 kg @ 9 m/s delivers                                  11.79 J   = 1.75x tipping
    ///     2 kg @ 6 m/s delivers                                   2.33 J   = 0.35x tipping
    ///     1.5 kg @ 6 m/s delivers                                  1.31 J   = 0.19x tipping
    ///     0.75 kg @ 6 m/s delivers                                 0.33 J   = 0.049x tipping  <- now
    ///
    /// At 1.75x no balance policy can absorb the hit without stepping, so training on it would have
    /// produced a flat zero success rate with no gradient - the same failure mode as a curriculum
    /// rung set past the cliff.
    ///
    /// 2 kg was measured too: against a policy with no perturbation training it drove reward/shaping
    /// to -7.99 (final head height ~0.30 m, i.e. on the floor) and fired episode_end/Inverted at
    /// 1.7% - a condition that had never once triggered in this project. Fully flat on its back is a
    /// knockdown, not a shove, so 1.5 kg it is.
    ///
    /// Cross-checked against the criterion that actually decides success: a 1.5 kg @ 6 m/s hit
    /// shifts the instantaneous capture point by 0.039 m against UprightTermination's 0.15 m
    /// allowance, about a quarter of the balance budget. A real disturbance the policy must actively
    /// null, well short of a free topple.
    ///
    /// This is also the natural axis for a difficulty curriculum - the frontier machinery in
    /// RagdollRLBridge would drive impulse instead of start pose with no change to its logic.
    ///
    /// **Measured against a trained standing policy, and left alone deliberately.** Over a 5-minute
    /// run (perturbation_v1_0, resumed from the 46.7M-step get-up policy) reward/shaping sat at
    /// -7.07 while ball/hits was 0.922 - and 0.922 * -7.7 + 0.078 * 0 = -7.1 closes the arithmetic
    /// exactly. Every hit put the body on the floor; every miss left it standing; the recovery rate
    /// was zero.
    ///
    /// (That analysis is about SPEED, which has not moved. BallMass was later halved for a reason
    /// impulse arithmetic does not capture - see BallMass.)
    ///
    /// The instinct is to read that as "the ball is still too strong" and reduce it again. The
    /// numbers in the table above say otherwise, and they are the reason this value did not move.
    /// Half of those shots were the SMALL ball, which by the same transferred-energy calculation
    /// delivers about 0.023 J - roughly 0.3% of the 6.74 J needed to tip this rig - and it flattened
    /// the body just as reliably. A disturbance three orders of magnitude below the passive tipping
    /// threshold is not what is knocking the dummy over.
    ///
    /// What is, is the policy: it has never in 46.7M steps experienced a disturbance, so it holds a
    /// knife-edge balance that any contact ends. This is the same brittleness that walled the get-up
    /// curriculum at 1.8-2.7 deg of start tilt. Perturbation strength is not the lever - training
    /// time is, and eventually a stepping reflex, since ankle and hip strategy alone cannot recover
    /// a capture point that has left the support polygon.
    ///
    /// Note the units trap this sits on: comparing the ball's own kinetic energy (27 J at these
    /// settings) against the body's tipping energy is meaningless, because almost none of a 1.5 kg
    /// ball's energy transfers to an 80.6 kg body. The table above is transferred energy. Mixing the
    /// two makes an already-gentle shot look like a 4x overshoot.
    /// </summary>
    [Export] public float LaunchSpeed { get; set; } = 6.0f;

    /// <summary>
    /// Ball mass (kg). See <see cref="LaunchSpeed"/> for how the pair was chosen.
    ///
    /// 0.75 kg, halved from 1.5 after watching it in the arena with the perturbation policy
    /// correctly loaded for the first time - earlier viewings had the arena silently running a
    /// WALKING policy, so "the ball knocks it down" was not a fair observation of this task.
    ///
    /// Transferred energy scales with mass SQUARED against a much heavier body, so halving the mass
    /// quarters the disturbance: 1.31 J -> 0.33 J, from 0.19x the tipping threshold down to 0.049x.
    /// The impulse arithmetic says that is far below what should topple an 80.6 kg body either way -
    /// but the heavy ball is a 0.44 m sphere that does not bounce off, so it stays in contact and
    /// keeps pushing, and a sustained shove is not what an impulse calculation models. Mass matters
    /// more here than the table suggests, which is why this is worth changing and the small ball's
    /// figure was not.
    /// </summary>
    [Export] public float BallMass { get; set; } = 0.75f;
    [Export] public float BallRadius { get; set; } = 0.22f;

    /// <summary>Horizontal distance from the target the heavy ball spawns at.</summary>
    [Export] public float SpawnDistance { get; set; } = 5.0f;

    /// <summary>
    /// Horizontal spawn distance for the SMALL ball, which is fired from much closer in.
    ///
    /// 2 m against the heavy ball's 5 m. At 6 m/s that is 0.33 s of flight rather than 0.83 s, which
    /// does two things: it leaves far less time to react, and it cuts the chance of missing - a
    /// small ball crossing 5 m of a moving target is the main reason ball/hit_rate sat below 1.0,
    /// and a shot that misses is indistinguishable from a recovery in every metric except that one.
    /// </summary>
    [Export] public float SmallBallSpawnDistance { get; set; } = 2.0f;

    /// <summary>Vertical spread (m) applied to the HEAVY ball's aim point, so hits are not always dead centre.</summary>
    [Export] public float AimHeightJitter { get; set; } = 0.25f;

    /// <summary>
    /// Vertical spread (m) for the SMALL ball. Zero, deliberately.
    ///
    /// This used to reuse AimHeightJitter clamped to the ball radius, which sounds tight until it
    /// is measured against the tolerance it has to fit inside. A hit needs the ball centre within
    /// (ball radius + limb radius) of the limb axis; on a 0.05 m forearm with a 0.06 m ball that is
    /// 0.11 m, and the clamped jitter spent 0.06 m of it - over half the budget - before the limb
    /// had moved at all. Variety is already supplied by the twelve target bones and the random
    /// azimuth, so the jitter was buying spread the shot could not afford.
    /// </summary>
    [Export] public float SmallBallAimJitter { get; set; } = 0.0f;

    /// <summary>
    /// Cap (m) on how far ahead of a bone the gun may aim when leading a moving target.
    ///
    /// Leading is exact for a bone moving at constant velocity, and a flailing limb is not - it can
    /// reverse inside a 0.33 s flight. Uncapped, a forearm snapping through 4 m/s would send the
    /// shot 1.3 m into empty space, which is a worse error than not leading at all. 0.5 m is about
    /// a limb's reach from its socket, so past it the extrapolation has left the body.
    /// </summary>
    [Export] public float MaxAimLead { get; set; } = 0.5f;

    /// <summary>
    /// Continuous collision detection on the ball. OFF, which reverses the original setting.
    ///
    /// It was turned on because a small fast ball "tunnels straight through the ragdoll at these
    /// speeds and the hit silently never happens". That reasoning was sound for the ball it was
    /// written for - 0.03 m radius at 12 m/s - and no longer applies: at 0.06 m radius and 6.22 m/s
    /// the ball advances 0.052 m per 120 Hz tick against its own 0.12 m diameter, so it cannot pass
    /// through anything without a discrete test seeing it.
    ///
    /// It is off because CCD is the prime suspect for the one-way contact this gun produces.
    /// Measured over 36 consecutive contact ticks, the ball's speed rose 5.98 -> 7.11 m/s, a change
    /// entirely accounted for by gravity, while the struck chest gained 74 N.s in a single tick -
    /// twelve times the ball's whole momentum, from a ball that lost none of it. Momentum was being
    /// created rather than exchanged. Godot-Jolt implements CCD as a linear cast that REPOSITIONS
    /// the body instead of resolving an ordinary contact, which is the right shape for a push that
    /// never pushes back.
    ///
    /// Kept as an export rather than deleted so the comparison can be rerun from the scene without
    /// a recompile. If BallImpactImpulse now scales with the momentum the ball loses, this was it.
    /// </summary>
    [Export] public bool BallContinuousCd { get; set; } = false;

    /// <summary>
    /// How much of its speed the ball keeps when it bounces off the ragdoll, 0-1.
    ///
    /// Purely so the ball visibly rebounds instead of vanishing. The first version of the analytic
    /// impact freed the ball on contact, which was correct arithmetic and looked absurd - the dummy
    /// appeared to swallow every shot. The bounce is cosmetic in intent but is done properly rather
    /// than faked: the bone receives exactly the momentum the ball LOST, m*(v - v_reflected), so the
    /// visual and the physics cannot disagree.
    ///
    /// 0.25 reads as a solid body rather than a rubber ball. Raising it raises the transferred
    /// impulse too, by (1 + restitution) for a head-on hit - which is real physics, not a bug: a
    /// ball that bounces back pushes harder than one that stops dead. If the hit needs to be
    /// softened, use SmallBallSpeed, which scales momentum directly.
    /// </summary>
    [Export] public float BallRestitution { get; set; } = 0.25f;

    /// <summary>
    /// Fire from a random compass direction each shot rather than always the same side.
    ///
    /// On by default: a fixed direction is learnable as a single scripted counter-lean, which
    /// looks like balance recovery until the first shot arrives from anywhere else.
    /// </summary>
    [Export] public bool RandomizeDirection { get; set; } = true;

    /// <summary>Simulation seconds a ball survives before being freed.</summary>
    [Export] public float BallLifetimeSeconds { get; set; } = 4.0f;

    /// <summary>
    /// Chance a given shot is a SMALL ball aimed at a random body part rather than the heavy
    /// chest shot above. 0 disables the small ball entirely.
    ///
    /// The two are different disturbances, not two sizes of one. The heavy ball is 0.44 m across
    /// and does not bounce off, so it stays in contact and keeps pushing - a sustained shove whose
    /// scale is set by kinetic energy. The small ball is brief contact at speed, so momentum
    /// transfer dominates.
    ///
    /// The figures this paragraph used to quote - "0.2 kg at 6 m/s ... under 4% of the balance
    /// budget" - were left behind by the mass-ratio work recorded on SmallBallMass, which raised
    /// the mass by more than an order of magnitude to stop the solver amplifying light contacts.
    /// They described a ball that has not been fired since. At the mass that is ACTUALLY fired,
    /// the same linear model reads: 3.0 kg at 6 m/s carries 18 N.s, moves an 80.6 kg body by
    /// 0.223 m/s and shifts the capture point roughly 0.062 m against the 0.15 m allowance in
    /// UprightTermination - about 41% of the balance budget, not 4%. This is a shove, not a poke,
    /// and it should be read as one when tuning.
    ///
    /// Note also that the ball does not appear in BodyStateObservation. The policy cannot see the
    /// shot coming and can only react to contact, so the disturbance has to stay inside what a
    /// purely reactive recovery can absorb.
    ///
    /// 6 m/s rather than the 12 it was first written at, and the SPEED was cut rather than the mass
    /// because the complaint was that it looked too fast - halving mass would have left it just as
    /// fast on screen. It now travels at the same speed as the heavy ball, so the two read as
    /// different sizes rather than as different weapons. Speed is also the only safe difficulty
    /// knob here: momentum is linear in it, while mass is pinned from below by the solver's
    /// contact-ratio limit.
    ///
    /// Fired from ONE gun rather than a second gun node, deliberately. Two guns on independent
    /// clocks would put several balls in the air at once, and the reason the training interval
    /// exceeds the episode window is to keep exactly one impact per episode so a fall can be
    /// attributed to a specific hit. Randomising the profile per shot gives variety without
    /// giving that up.
    /// </summary>
    [Export] public float SmallBallProbability { get; set; } = 1.0f;

    /// <summary>
    /// 1.0 kg. Raised from 0.0025 kg - a 400x INCREASE - because the mass was being driven the
    /// wrong way for five iterations straight.
    ///
    /// The sequence was 0.2 -> 0.1 -> 0.05 -> 0.025 -> 0.0025 kg, each halving made in the belief
    /// that a lighter ball would hit more gently, and the impact was reported as unchanged every
    /// single time. It was: the violence never came from the ball's momentum. Measured on a 10 s
    /// arena dump, a routine impact injected 45-112 J and one chest impact injected 380 J in a
    /// single tick, against a ball carrying 0.048 J - amplification of roughly 1000x to 8000x. The
    /// actuators were loafing at a fifth of their ceiling on the tick it happened, so the energy was
    /// not theirs either.
    ///
    /// What it was is the mass ratio. At 0.0025 kg against a 16 kg chest the solver was being asked
    /// to resolve a 6400:1 contact, and sequential-impulse solvers lose accuracy long before that -
    /// aggravated here by ContinuousCd on a 0.06 m sphere travelling 0.052 m per tick, about its own
    /// radius. Every halving of the mass made that ratio worse, which is exactly why the symptom
    /// tracked the intervention in the wrong direction.
    ///
    /// 1.0 kg puts the chest ratio at 16:1 and the lightest targeted bone (a 0.5 kg hand) at 1:2,
    /// well inside what the solver handles cleanly. If per-impact energy injection now lands near
    /// the physical value for this ball (about 19 J at 6.22 m/s) the diagnosis holds; if it stays
    /// in the hundreds, the mass ratio is not the mechanism and BallImpactImpulse below will say so
    /// directly rather than by inference.
    ///
    /// The value settled at 3.0 rather than the 1.0 this note argues for, and the note was not
    /// updated. 3.0 kg is further from the solver's bad regime, not closer, so it is safe in the
    /// direction this investigation cared about - but it is also 3x the disturbance the paragraph
    /// above sizes, which matters when reading the balance budget.
    ///
    /// Do NOT tune difficulty downward through this field. The whole point of the sequence above is
    /// that lowering mass drove the mass ratio toward the regime where the solver injected 1000x to
    /// 8000x the ball's own energy, and the symptom tracked the intervention backwards for five
    /// iterations. SmallBallSpeed is the knob: momentum is linear in it and it has no such floor.
    /// </summary>
    [Export] public float SmallBallMass { get; set; } = 3.0f;
    /// <summary>
    /// 0.06 m radius - a 12 cm ball. Halved to 0.03 twice and restored twice; leave it here.
    ///
    /// Size is a MEASUREMENT parameter on this gun, not a difficulty one. Mass sets how hard the
    /// hit is and is tuned independently below, so shrinking the ball buys nothing except a smaller
    /// cross-section to hit a limb with - and a shot that misses is indistinguishable from a
    /// recovery in every metric except ball/hit_rate, which is the exact ambiguity
    /// IRlPerturbationDiagnostics exists to remove.
    ///
    /// The first revert (ball/hit_rate 0.93 -> 0.773) was the right call on the wrong diagnosis.
    /// The comment written at the time concluded "aim is not the problem ... a forearm moves a long
    /// way in 0.33 s". Half right: it does, which is why Fire now leads the target. But the
    /// dominant error was that the shot was launched perfectly horizontally with gravity enabled,
    /// so it arrived 0.54 m low; a smaller ball simply had less chance of clipping something on the
    /// way down, which made a ballistics bug look like an aim-spread problem. That is fixed, so if
    /// there is ever a real reason to want 0.03 it should now be viable - but there is no such
    /// reason, because mass is the knob that matters.
    /// </summary>
    [Export] public float SmallBallRadius { get; set; } = 0.06f;

    /// <summary>
    /// 6 m/s, down from 8. The documented intent for this ball has always been "same speed as the
    /// heavy ball" (see SmallBallProbability, and LaunchSpeed = 6.0); 8 was never reconciled with
    /// that and made the small ball the harder of the two disturbances by a third.
    ///
    /// Momentum drops 24 -> 18 N.s, taking the capture-point shift from roughly 55% of the
    /// UprightTermination balance allowance to 41%. Chosen over cutting mass because mass has a
    /// solver floor and speed does not - see SmallBallMass.
    /// </summary>
    [Export] public float SmallBallSpeed { get; set; } = 6.0f;

    /// <summary>
    /// Fractional spread on the small ball's launch speed, applied per shot. 0.25 means +/-25%.
    ///
    /// The perturbation was deterministic: every shot delivered the same impulse, so a policy could
    /// learn exactly one recovery and never be asked for another. Momentum is linear in speed, so
    /// this spreads the disturbance over roughly 22-37 N.s instead of pinning it at 30, which is the
    /// difference between learning "the" recovery and learning to recover.
    ///
    /// Speed rather than mass because it varies the disturbance and its visible approach together -
    /// a ball thrown harder looks thrown harder. Mass would change the hit invisibly.
    ///
    /// The result is clamped in Fire against the overlap probe's ceiling; see MaxProbeSafeSpeed.
    /// </summary>
    [Export] public float SmallBallSpeedJitter { get; set; } = 0.25f;

    /// <summary>
    /// Bones the SMALL ball may target, chosen uniformly per shot. The heavy ball always aims at
    /// the chest, because a 0.44 m sphere aimed at a forearm mostly hits the torso anyway.
    ///
    /// Resolved through HumanoidRagdoll.FindBone; a name this rig does not have is skipped rather
    /// than fatal, so the gun degrades to the bones that do exist instead of refusing to fire.
    /// </summary>
    [Export] public string[] SmallBallTargetBones { get; set; } =
    {
        "Head", "Chest", "Spine", "Pelvis",
        "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
        "Thigh_L", "Shin_L", "Thigh_R", "Shin_R"
    };

    /// <summary>Master switch, so one scene can carry the gun and disable it per experiment.</summary>
    [Export] public bool Enabled { get; set; } = true;

    private readonly List<Ball> _live = new();
    private float _timeUntilNextShot;
    private int _lastSeenEpisode = -1;
    private readonly RandomNumberGenerator _rng = new();

    /// <summary>
    /// Downward acceleration (m/s^2) balls experience, read from the project rather than hardcoded.
    ///
    /// Balls are spawned with the default GravityScale of 1, so this is exactly what they fall at,
    /// and the drop compensation in Fire is only correct while the two agree.
    /// </summary>
    private float _gravity = 9.8f;

    private readonly struct Ball
    {
        public Ball(RigidBody3D body, float remaining, ActiveBone? aimBone, bool hasHit = false)
        {
            Body = body;
            Remaining = remaining;
            AimBone = aimBone;
            HasHit = hasHit;
        }

        public RigidBody3D Body { get; }
        public float Remaining { get; }

        /// <summary>
        /// The bone this ball was actually aimed at, kept so a hit can be scored against its
        /// intended target rather than against "touched the ragdoll anywhere".
        ///
        /// Without this, ball/hit_rate cannot tell a forearm shot that connected from one that
        /// sailed past the arm and grazed a shin - and for a gun that picks a target bone per shot,
        /// those are different events. It was the absence of this distinction that let a 0.54 m
        /// ballistic drop hide behind a hit rate of 0.84.
        /// </summary>
        public ActiveBone? AimBone { get; }

        /// <summary>
        /// Set once this ball has delivered its impulse, so it can strike at most once.
        ///
        /// Without it a ball still overlapping on the following tick would deliver its momentum
        /// again, which is the same repeated-impulse runaway the contact solver used to produce -
        /// only written by hand, which would be worse.
        /// </summary>
        public bool HasHit { get; }
    }

    private int _shotsThisEpisode;
    private int _hitsThisEpisode;
    private int _smallShotsThisEpisode;
    private int _aimHitsThisEpisode;

    private int _liveBalls;
    private float _fastestBallSpeed;
    private float _tickImpactImpulse;
    private string _tickImpactBone = "-";
    private float _tickImpactMassRatio;

    /// <summary>
    /// Mass reported BY the physics body, not the value we asked for.
    ///
    /// Exists because SmallBallMass was changed 400x - 0.0025 kg to 1.0 kg - with no measurable
    /// change in behaviour, and a parameter that inert is either irrelevant to the mechanism or
    /// never arriving. Mass is assigned in a RigidBody3D object initializer, before the node enters
    /// the tree; if that assignment is not reaching Jolt then every mass change made against this
    /// gun has been a no-op in both directions, which would explain a great deal of history.
    /// </summary>
    private float _tickBallMass;

    /// <summary>
    /// Physics layer the ragdoll's bones occupy (ActiveRagdoll.tscn sets collision_layer = 2 on
    /// every bone). The overlap probe queries exactly this and nothing else.
    /// </summary>
    private const uint RagdollLayerMask = 2;

    /// <summary>
    /// Layer the ball is placed on, chosen so that nothing on the ragdoll's mask can see it.
    ///
    /// Bones are layer 2 / mask 1 and the ball used to be the engine default layer 1 / mask 1,
    /// which is precisely why they collided: the bones' mask included the ball's layer. Moving the
    /// ball to layer 4 while leaving its mask at 1 keeps it colliding with the ground - a missed
    /// shot still lands and rolls - while removing it entirely from the ragdoll's solver.
    /// </summary>
    private const uint BallLayer = 4;
    private const uint WorldMask = 1;

    /// <summary>Reused sphere for the overlap probe; allocating one per ball per tick would churn.</summary>
    private readonly SphereShape3D _probeShape = new();
    private readonly PhysicsShapeQueryParameters3D _probeQuery = new();

    /// <summary>Reused across episodes rather than reallocated - see the reward's note on per-tick allocation.</summary>
    private readonly Dictionary<string, float> _stats = new()
    {
        ["shots"] = 0.0f, ["hits"] = 0.0f, ["small_shots"] = 0.0f,
    };

    public IReadOnlyDictionary<string, float> EpisodePerturbationStats
    {
        get
        {
            _stats["shots"] = _shotsThisEpisode;
            _stats["hits"] = _hitsThisEpisode;
            _stats["small_shots"] = _smallShotsThisEpisode;
            _stats["aim_hits"] = _aimHitsThisEpisode;
            return _stats;
        }
    }

    public override void _Ready()
    {
        _rng.Randomize();
        _timeUntilNextShot = FirstShotDelaySeconds;
        _gravity = (float)ProjectSettings.GetSetting("physics/3d/default_gravity", 9.8f);

        Target ??= FindTargetInOwnAgent();

        // Printed once at startup, not per shot. A perturbation run and a plain run produce
        // otherwise identical logs, and mistaking one for the other silently invalidates any
        // comparison between them - the same confusion the playback banner exists to prevent.
        GD.Print(
            $"[BallGun] {(Enabled ? "ARMED" : "disabled")} - every {IntervalSeconds:F1}s "
            + $"(first at {FirstShotDelaySeconds:F1}s), {BallMass:F1}kg @ {LaunchSpeed:F1}m/s "
            + $"from {SpawnDistance:F1}m, {(RandomizeDirection ? "random" : "fixed")} direction, "
            + $"target={(Target != null ? Target.Name.ToString() : "NONE")}");
    }

    /// <summary>
    /// Last-resort target lookup, scoped to THIS gun's own agent.
    ///
    /// It used to be <c>GetTree().Root.FindChild("ActiveRagdoll", true, false)</c> - a whole-tree
    /// search returning the first match anywhere. That was harmless while a scene held exactly one
    /// body, and became silently wrong the moment RagdollSpawner started putting 64 in one process:
    /// every gun that fell through to it acquired Agent_1, so 63 dummies were shelled by guns aimed
    /// at a body metres away while their own stood unperturbed. The perturbation metrics would have
    /// looked like a policy that had learned to ignore impacts.
    ///
    /// Resolved through the <see cref="RlAgent"/> ancestor rather than by name, so it cannot cross
    /// an agent boundary by construction rather than by a filter applied afterwards. Returns null
    /// when there is no such ancestor, which is the honest answer - a gun outside any agent has no
    /// body it can be said to belong to, and Update() already treats a null Target as disarmed.
    /// </summary>
    private HumanoidRagdoll? FindTargetInOwnAgent()
    {
        for (Node? node = GetParent(); node != null; node = node.GetParent())
        {
            if (node is RlAgent agent)
            {
                return agent.Ragdoll;
            }
        }

        return null;
    }

    /// <summary>
    /// Driven by physics ticks, not frames.
    ///
    /// Training runs headless at a speedup, where _Process rate is neither stable nor meaningful.
    /// The physics delta already carries Engine.TimeScale, so accumulating it counts SIMULATION
    /// seconds - the same clock RagdollRLBridge measures episode length on. A 3 s interval is
    /// therefore 3 s of simulated time whatever --speedup is set to, which is what makes runs at
    /// different speedups comparable.
    /// </summary>
    public override void _PhysicsProcess(double delta)
    {
        float dt = (float)delta;

        if (Bridge != null && IsInstanceValid(Bridge) && Bridge.EpisodeCount != _lastSeenEpisode)
        {
            _lastSeenEpisode = Bridge.EpisodeCount;
            ResetForNewEpisode();
        }

        _liveBalls = 0;
        _fastestBallSpeed = 0.0f;
        _tickImpactImpulse = 0.0f;
        _tickImpactBone = "-";
        _tickImpactMassRatio = 0.0f;
        _tickBallMass = 0.0f;

        AgeBalls(dt);

        // Pushed rather than pulled: the recorder lives on the ragdoll and the ragdoll knows
        // nothing about perturbation sources, so reversing this would invert the dependency.
        if (Target != null && IsInstanceValid(Target))
        {
            Target.Recorder.ReportProjectile(
                _liveBalls, _fastestBallSpeed, _tickImpactBone, _tickImpactImpulse,
                _tickImpactMassRatio, _tickBallMass);
        }

        if (!Enabled || Target == null || !IsInstanceValid(Target))
        {
            return;
        }

        _timeUntilNextShot -= dt;
        if (_timeUntilNextShot <= 0.0f)
        {
            Fire();
            _timeUntilNextShot = Mathf.Max(0.1f, IntervalSeconds);
        }
    }

    /// <summary>Clears every ball in flight and restarts the firing clock at the grace period.</summary>
    public void ResetForNewEpisode()
    {
        foreach (Ball ball in _live)
        {
            if (IsInstanceValid(ball.Body))
            {
                ball.Body.QueueFree();
            }
        }
        _live.Clear();
        _timeUntilNextShot = FirstShotDelaySeconds;
        _shotsThisEpisode = 0;
        _hitsThisEpisode = 0;
        _smallShotsThisEpisode = 0;
        _aimHitsThisEpisode = 0;
    }

    /// <summary>
    /// Whether this ball is currently touching the ragdoll, counted at most once per ball.
    ///
    /// Contacts are read from the body rather than via a signal so the check stays on the physics
    /// clock with everything else here, and so a ball that grazes and separates inside one tick is
    /// still caught by the next AgeBalls pass while it remains in contact.
    /// </summary>
    /// <summary>
    /// Fastest a ball of this radius may travel before the overlap probe can miss it entirely.
    ///
    /// AgeBalls probes a sphere at the ball's position once per tick, so consecutive probes cover
    /// the flight path continuously only while the ball advances less than its own diameter per
    /// tick. Past that the probes leave gaps and a shot can pass straight through untouched. At
    /// 0.06 m radius and 120 Hz the ceiling is 14.4 m/s; the clamp exists so speed jitter cannot
    /// wander over it, since the failure is silent - a miss looks like a miss, not like a bug.
    /// </summary>
    private static float MaxProbeSafeSpeed(float radius)
    {
        return 2.0f * radius * Mathf.Max(1, Engine.PhysicsTicksPerSecond);
    }

    /// <summary>Radius the overlap probe uses for a given ball, taken from its own sphere shape.</summary>
    private float BallProbeRadius(Ball ball)
    {
        foreach (Node child in ball.Body.GetChildren())
        {
            if (child is CollisionShape3D collision && collision.Shape is SphereShape3D sphere)
            {
                return sphere.Radius;
            }
        }
        return SmallBallRadius;
    }

    /// <summary>
    /// The ragdoll bone overlapping a sphere at <paramref name="position"/>, or null.
    ///
    /// Queried against the real bone colliders rather than against bone origins and nominal radii,
    /// so a box chest and a capsule forearm are both tested as the shapes they actually are. This
    /// replaces GetCollidingBodies, which could only report contacts the solver had already decided
    /// to generate - and the solver's decisions were the thing under suspicion.
    /// </summary>
    private ActiveBone? FindOverlappingBone(Vector3 position, float radius)
    {
        World3D? world = GetWorld3D();
        if (world == null)
        {
            return null;
        }

        _probeShape.Radius = radius;
        _probeQuery.Shape = _probeShape;
        _probeQuery.Transform = new Transform3D(Basis.Identity, position);
        _probeQuery.CollisionMask = RagdollLayerMask;
        _probeQuery.CollideWithBodies = true;
        _probeQuery.CollideWithAreas = false;

        // Four is ample: the question is which bone to credit, and overlapping more than a handful
        // of bones at once would mean the ball is already deep inside the torso.
        var hits = world.DirectSpaceState.IntersectShape(_probeQuery, 4);
        foreach (var hit in hits)
        {
            // The ancestor test is load-bearing with more than one agent per process. Agents are
            // spaced along X and the probe is a world-space shape query, so a ball passing between
            // two rows can overlap a NEIGHBOUR's bone and credit the impact to a body this gun
            // never fired at - poisoning both dummies' ball_* diagnostics at once.
            if (hit.TryGetValue("collider", out Variant collider)
                && collider.As<GodotObject>() is ActiveBone bone
                && IsInstanceValid(bone)
                && Target != null
                && Target.IsAncestorOf(bone))
            {
                return bone;
            }
        }

        return null;
    }


    /// <summary>
    /// Ages balls on the simulation clock instead of SceneTree timers.
    ///
    /// SceneTree timers are themselves scaled by Engine.TimeScale and allocate a signal connection
    /// per ball; across a thousand-odd bodies (16 processes x 64) each firing for hours that is a
    /// steady churn for something a float subtraction does exactly as well. The argument got
    /// stronger when RagdollSpawner made the gun per-agent rather than per-scene: one gun per body
    /// means the allocation count now scales with --dummies too, not just with --n_parallel.
    /// </summary>
    private void AgeBalls(float dt)
    {
        for (int i = _live.Count - 1; i >= 0; i--)
        {
            float remaining = _live[i].Remaining - dt;
            if (remaining <= 0.0f || !IsInstanceValid(_live[i].Body))
            {
                if (IsInstanceValid(_live[i].Body))
                {
                    _live[i].Body.QueueFree();
                }
                _live.RemoveAt(i);
                continue;
            }
            Ball ball = _live[i];
            Vector3 velocity = ball.Body.LinearVelocity;
            float ballMass = ball.Body.Mass;
            Vector3 ballPosition = ball.Body.GlobalPosition;

            _liveBalls++;
            _tickBallMass = ballMass;
            _fastestBallSpeed = Mathf.Max(_fastestBallSpeed, velocity.Length());

            // HasHit gates this: a ball delivers its impulse exactly once. Without that gate a ball
            // still overlapping on the following tick would deliver it again, which is the same
            // repeated-impulse runaway the contact solver was producing, just written by hand.
            ActiveBone? struck = ball.HasHit
                ? null
                : FindOverlappingBone(ballPosition, BallProbeRadius(ball));

            if (struck != null)
            {
                Vector3 offset = ballPosition - struck.GlobalPosition;

                // Outward normal approximated from the bone's centre to the ball. Exact for the
                // sphere head and close enough on capsules and boxes for a rebound direction; the
                // transferred impulse below does not depend on it being exact, only the bounce does.
                Vector3 normal = offset.LengthSquared() > 1e-6f
                    ? offset.Normalized()
                    : -velocity.Normalized();

                Vector3 reflected = (velocity - (2.0f * velocity.Dot(normal) * normal))
                                    * Mathf.Clamp(BallRestitution, 0.0f, 1.0f);

                // The bone receives exactly the momentum the ball lost. Applied at the ball's offset
                // so an off-centre hit produces the torque it should. This is the whole impact: no
                // contact manifold, no penetration recovery, no mass ratio, and no dependence on
                // which tick the overlap happens to be noticed.
                Vector3 impulse = (velocity - reflected) * ballMass;
                struck.ApplyImpulse(impulse, offset);

                // Kept in flight rather than freed, so the shot visibly rebounds and then falls to
                // the ground under its own gravity. It cannot strike again - HasHit is set below.
                ball.Body.LinearVelocity = reflected;

                _hitsThisEpisode++;
                if (struck == ball.AimBone) { _aimHitsThisEpisode++; }

                _tickImpactImpulse = impulse.Length();
                _tickImpactBone = struck.BoneName;
                _tickImpactMassRatio = struck.Mass / Mathf.Max(1e-6f, ballMass);

                _live[i] = new Ball(ball.Body, remaining, ball.AimBone, true);
                continue;
            }

            _live[i] = new Ball(ball.Body, remaining, ball.AimBone, ball.HasHit);
        }
    }

    private void Fire()
    {
        bool small = SmallBallProbability > 0.0f && _rng.Randf() < SmallBallProbability;

        ActiveBone? aimBone = small ? PickSmallBallTarget() : Target!.Chest;
        if (aimBone == null || !IsInstanceValid(aimBone))
        {
            aimBone = Target!.Chest;
        }
        if (aimBone == null || !IsInstanceValid(aimBone))
        {
            aimBone = Target!.Pelvis;
        }
        if (aimBone == null || !IsInstanceValid(aimBone))
        {
            return;
        }

        float mass = small ? SmallBallMass : BallMass;
        float radius = small ? SmallBallRadius : BallRadius;
        float speed = small ? SmallBallSpeed : LaunchSpeed;
        if (small && SmallBallSpeedJitter > 0.0f)
        {
            speed *= 1.0f + _rng.RandfRange(-SmallBallSpeedJitter, SmallBallSpeedJitter);
        }
        speed = Mathf.Clamp(speed, 0.1f, MaxProbeSafeSpeed(radius));

        float spawnDistance = small ? SmallBallSpawnDistance : SpawnDistance;

        // Flight time is exact rather than an estimate: there is no drag, the spawn offset below is
        // purely horizontal, and the launch keeps horizontal speed at exactly `speed`. It is
        // computed first because both corrections that follow are functions of it.
        float flightTime = spawnDistance / Mathf.Max(speed, 0.01f);

        // Lead the target. The aim point used to be the bone's position at the instant of firing,
        // which is where it no longer was by the time the ball got there - over a 0.33 s small-ball
        // flight, a limb moving faster than about 0.33 m/s escaped a shot aimed dead at it. A
        // stumbling forearm clears that easily, so the shots that mattered most were the ones most
        // likely to miss.
        Vector3 lead = aimBone.LinearVelocity * flightTime;
        if (lead.LengthSquared() > MaxAimLead * MaxAimLead)
        {
            lead = lead.Normalized() * MaxAimLead;
        }

        float jitter = small ? SmallBallAimJitter : AimHeightJitter;

        Vector3 aimPoint = aimBone.GlobalPosition + lead
                           + new Vector3(0.0f, _rng.RandfRange(-jitter, jitter), 0.0f);

        float azimuth = RandomizeDirection ? _rng.RandfRange(0.0f, Mathf.Tau) : 0.0f;
        Vector3 offset = new Vector3(Mathf.Cos(azimuth), 0.0f, Mathf.Sin(azimuth)) * spawnDistance;
        Vector3 spawn = aimPoint + offset;

        var body = new RigidBody3D
        {
            Mass = mass,
            // See BallContinuousCd. Irrelevant to the ragdoll now that the ball is off its layer,
            // but still governs how the ball meets the ground.
            ContinuousCd = BallContinuousCd,

            // The ball is deliberately NOT on a layer the ragdoll can see - see BallLayer. The
            // impact is applied analytically in AgeBalls instead of being resolved by the contact
            // solver, because the solver could not do it correctly at any setting: with CCD on it
            // repositioned the ball and pushed the bone without the ball ever losing momentum, and
            // with CCD off the same one-way push appeared on every box-shaped bone (pelvis, chest,
            // hands: 7 of 7 recorded hits) and on capsules thinner than about 0.07 m. Measured, a
            // ball carrying 6.22 N.s handed a 14 kg pelvis 81.5 N.s in one tick while its own speed
            // rose from 5.98 to 6.56 m/s. That is not a hard contact to tune, it is momentum being
            // created, and a 0.06 m sphere crossing 0.052 m per tick is simply outside what a
            // discrete solver resolves against a 16-body jointed articulation.
            CollisionLayer = BallLayer,
            CollisionMask = WorldMask,
            // NO GlobalPosition here. It used to be set in this initializer as well as after
            // AddChild below, and the initializer runs while the body is still outside the scene
            // tree - where Node3D.GlobalPosition has no parent transform to resolve against. Godot
            // logged "Condition !is_inside_tree() is true" with a full C# backtrace on EVERY shot,
            // from all 32 training processes at once, which buried genuine errors in the console.
            // The assignment after AddChild is the one that actually places the ball.
        };

        var shape = new CollisionShape3D { Shape = new SphereShape3D { Radius = radius } };
        body.AddChild(shape);

        var mesh = new MeshInstance3D
        {
            Mesh = new SphereMesh
            {
                Radius = radius,
                Height = radius * 2.0f,
                Material = new StandardMaterial3D
                {
                    // Distinct colours so the two profiles are tellable apart on sight.
                    AlbedoColor = small
                        ? new Color(0.95f, 0.85f, 0.15f)
                        : new Color(0.9f, 0.25f, 0.1f),
                    Roughness = 0.3f,
                    Metallic = 0.6f
                }
            }
        };
        body.AddChild(mesh);

        AddChild(body);
        body.GlobalPosition = spawn;

        // Launch on an arc that ARRIVES at aimPoint, rather than pointing straight at it.
        //
        // This is the single largest correction in this file. The shot direction was
        // (aimPoint - spawn), and since the spawn offset is purely horizontal that vector is purely
        // horizontal too - so every ball was launched flat, with gravity left on, and fell
        // 0.5*g*t^2 during the flight. For the small ball that is 0.54 m over 2 m of travel; for
        // the heavy ball 3.4 m over 5 m, which puts it on the floor before it arrives. Against a
        // hit tolerance of roughly 0.08 m the drop was five times the entire error budget, which
        // made the target bone the gun so carefully selects decoration: a head shot landed at the
        // pelvis, a pelvis shot at the shins, and a shin shot in the dirt. The surviving 0.84 hit
        // rate was balls clipping the body on the way past.
        //
        // Adding 0.5*g*t of upward launch velocity cancels exactly that drop. Horizontal speed is
        // untouched, so flightTime above stays exact; total launch speed rises 6.0 -> 6.22 m/s for
        // the small ball, a 4% change that leaves the disturbance the policy feels effectively as
        // it was.
        Vector3 launchVelocity = (aimPoint - spawn).Normalized() * speed
                                 + Vector3.Up * (0.5f * _gravity * flightTime);

        // Impulse scales with mass so LaunchSpeed is the ball's actual speed in m/s, and changing
        // BallMass changes how hard it hits without also changing how fast it arrives.
        body.ApplyCentralImpulse(launchVelocity * mass);

        // Printed once per episode's first shot rather than per shot: if the requested mass and
        // the mass the physics body reports disagree, that is the single most important line in the
        // log and it should not be competing with hundreds of copies of itself.
        if (_shotsThisEpisode == 0)
        {
            GD.Print($"[BallGun] shot 1: requested mass {mass:F4} kg, body reports {body.Mass:F4} kg, "
                     + $"radius {radius:F3} m, ccd={BallContinuousCd}");
        }

        _live.Add(new Ball(body, BallLifetimeSeconds, aimBone));
        _shotsThisEpisode++;
        if (small) { _smallShotsThisEpisode++; }
    }

    /// <summary>Uniform pick from SmallBallTargetBones, skipping any name this rig lacks.</summary>
    private ActiveBone? PickSmallBallTarget()
    {
        if (Target == null || SmallBallTargetBones == null || SmallBallTargetBones.Length == 0)
        {
            return null;
        }

        int start = _rng.RandiRange(0, SmallBallTargetBones.Length - 1);
        for (int i = 0; i < SmallBallTargetBones.Length; i++)
        {
            string name = SmallBallTargetBones[(start + i) % SmallBallTargetBones.Length];
            // No ancestor test here, unlike the probe above: FindBone searches Target's own
            // skeleton, so the result cannot belong to another agent. The guard a patch script
            // added here was unreachable anyway - it tested Target for null on the line after
            // dereferencing it, inside a method that already returns early when it is null.
            ActiveBone? bone = Target.FindBone(name);
            if (bone != null && IsInstanceValid(bone))
            {
                return bone;
            }
        }

        return null;
    }
}

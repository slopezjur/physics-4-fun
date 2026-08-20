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
public partial class BallGun : Node3D, IRlPerturbationDiagnostics
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
    /// The arena scene overrides this to 3 s, because there the point is to watch repeated
    /// recoveries rather than to train. See its .tscn.
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
    /// shifts the instantaneous capture point by 0.039 m against GetUpTermination's 0.15 m
    /// allowance, about a quarter of the balance budget. A real disturbance the policy must actively
    /// null, well short of a free topple.
    ///
    /// This is also the natural axis for a difficulty curriculum - the frontier machinery in
    /// RagdollRLBridge would drive impulse instead of start pose with no change to its logic.
    /// </summary>
    [Export] public float LaunchSpeed { get; set; } = 6.0f;

    /// <summary>Ball mass (kg). See <see cref="LaunchSpeed"/> for how the pair was chosen.</summary>
    [Export] public float BallMass { get; set; } = 1.5f;
    [Export] public float BallRadius { get; set; } = 0.22f;

    /// <summary>Horizontal distance from the target the ball spawns at.</summary>
    [Export] public float SpawnDistance { get; set; } = 5.0f;

    /// <summary>Vertical spread (m) applied to the aim point, so hits are not always dead centre.</summary>
    [Export] public float AimHeightJitter { get; set; } = 0.25f;

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
    /// transfer dominates: 0.2 kg at 6 m/s moves an 80.6 kg body by 0.018 m/s, shifting the capture
    /// point about 0.005 m against the 0.15 m allowance in GetUpTermination - under 4% of the
    /// balance budget. Globally that is a poke; locally it snaps a limb, which is the point.
    ///
    /// 6 m/s rather than the 12 it was first written at, and the SPEED was cut rather than the mass
    /// because the complaint was that it looked too fast - halving mass would have left it just as
    /// fast on screen. It now travels at the same speed as the heavy ball, so the two read as
    /// different sizes rather than as different weapons.
    ///
    /// Fired from ONE gun rather than a second gun node, deliberately. Two guns on independent
    /// clocks would put several balls in the air at once, and the reason the training interval
    /// exceeds the episode window is to keep exactly one impact per episode so a fall can be
    /// attributed to a specific hit. Randomising the profile per shot gives variety without
    /// giving that up.
    /// </summary>
    [Export] public float SmallBallProbability { get; set; } = 0.5f;

    [Export] public float SmallBallMass { get; set; } = 0.2f;
    [Export] public float SmallBallRadius { get; set; } = 0.06f;
    [Export] public float SmallBallSpeed { get; set; } = 6.0f;

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

    private readonly struct Ball
    {
        public Ball(RigidBody3D body, float remaining, bool hasHit = false)
        {
            Body = body;
            Remaining = remaining;
            HasHit = hasHit;
        }

        public RigidBody3D Body { get; }
        public float Remaining { get; }

        /// <summary>Set once the ball has touched the ragdoll, so one ball counts at most one hit.</summary>
        public bool HasHit { get; }
    }

    private int _shotsThisEpisode;
    private int _hitsThisEpisode;
    private int _smallShotsThisEpisode;

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
            return _stats;
        }
    }

    public override void _Ready()
    {
        _rng.Randomize();
        _timeUntilNextShot = FirstShotDelaySeconds;

        if (Target == null)
        {
            Target = GetTree().Root.FindChild("ActiveRagdoll", true, false) as HumanoidRagdoll;
        }

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

        AgeBalls(dt);

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
    }

    /// <summary>
    /// Whether this ball is currently touching the ragdoll, counted at most once per ball.
    ///
    /// Contacts are read from the body rather than via a signal so the check stays on the physics
    /// clock with everything else here, and so a ball that grazes and separates inside one tick is
    /// still caught by the next AgeBalls pass while it remains in contact.
    /// </summary>
    private bool DetectHit(Ball ball)
    {
        if (ball.HasHit || !IsInstanceValid(ball.Body))
        {
            return false;
        }

        foreach (Node3D other in ball.Body.GetCollidingBodies())
        {
            if (other is ActiveBone)
            {
                _hitsThisEpisode++;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Ages balls on the simulation clock instead of SceneTree timers.
    ///
    /// SceneTree timers are themselves scaled by Engine.TimeScale and allocate a signal connection
    /// per ball; at 40 processes firing every 3 s for hours that is a steady churn for something a
    /// float subtraction does exactly as well.
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
            _live[i] = new Ball(_live[i].Body, remaining, _live[i].HasHit || DetectHit(_live[i]));
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
            aimBone = Target.Pelvis;
        }
        if (aimBone == null || !IsInstanceValid(aimBone))
        {
            return;
        }

        float mass = small ? SmallBallMass : BallMass;
        float radius = small ? SmallBallRadius : BallRadius;
        float speed = small ? SmallBallSpeed : LaunchSpeed;

        // A small ball aimed at a limb needs a tighter spread, or the aim jitter alone can make
        // it miss what it was aimed at - which reads as a hit-rate hole rather than as a miss.
        float jitter = small ? Mathf.Min(AimHeightJitter, radius) : AimHeightJitter;

        Vector3 aimPoint = aimBone.GlobalPosition
                           + new Vector3(0.0f, _rng.RandfRange(-jitter, jitter), 0.0f);

        float azimuth = RandomizeDirection ? _rng.RandfRange(0.0f, Mathf.Tau) : 0.0f;
        Vector3 offset = new Vector3(Mathf.Cos(azimuth), 0.0f, Mathf.Sin(azimuth)) * SpawnDistance;
        Vector3 spawn = aimPoint + offset;

        var body = new RigidBody3D
        {
            Mass = mass,
            // The ball is small, fast and aimed at thin limbs; without continuous detection it
            // tunnels straight through the ragdoll at these speeds and the hit silently never
            // happens - which would look like a policy that learned to ignore impacts.
            ContinuousCd = true,
            // Required for GetCollidingBodies to return anything - see DetectHit. Four is ample:
            // the question is "did this ball touch the ragdoll at all", not how many bones it brushed.
            ContactMonitor = true,
            MaxContactsReported = 4,
            GlobalPosition = spawn
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

        // Impulse scales with mass so LaunchSpeed is the ball's actual speed in m/s, and changing
        // BallMass changes how hard it hits without also changing how fast it arrives.
        body.ApplyCentralImpulse((aimPoint - spawn).Normalized() * speed * mass);

        _live.Add(new Ball(body, BallLifetimeSeconds));
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
            ActiveBone? bone = Target.FindBone(name);
            if (bone != null && IsInstanceValid(bone))
            {
                return bone;
            }
        }

        return null;
    }
}

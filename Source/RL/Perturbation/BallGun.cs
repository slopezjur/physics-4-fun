using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll;

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
public partial class BallGun : Node3D
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

    /// <summary>Simulation seconds between shots.</summary>
    [Export] public float IntervalSeconds { get; set; } = 3.0f;

    /// <summary>
    /// Grace period at episode start before the first shot, so the dummy is not hit mid-teleport
    /// while the actuators are still settling into the reset pose.
    /// </summary>
    [Export] public float FirstShotDelaySeconds { get; set; } = 1.5f;

    /// <summary>Launch speed (m/s). Impulse is scaled by mass, so this is the actual ball speed.</summary>
    [Export] public float LaunchSpeed { get; set; } = 9.0f;

    [Export] public float BallMass { get; set; } = 3.0f;
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

    /// <summary>Master switch, so one scene can carry the gun and disable it per experiment.</summary>
    [Export] public bool Enabled { get; set; } = true;

    private readonly List<Ball> _live = new();
    private float _timeUntilNextShot;
    private int _lastSeenEpisode = -1;
    private readonly RandomNumberGenerator _rng = new();

    private readonly struct Ball
    {
        public Ball(RigidBody3D body, float remaining)
        {
            Body = body;
            Remaining = remaining;
        }

        public RigidBody3D Body { get; }
        public float Remaining { get; }
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
            _live[i] = new Ball(_live[i].Body, remaining);
        }
    }

    private void Fire()
    {
        ActiveBone? aimBone = Target!.Chest;
        if (aimBone == null || !IsInstanceValid(aimBone))
        {
            aimBone = Target.Pelvis;
        }
        if (aimBone == null || !IsInstanceValid(aimBone))
        {
            return;
        }

        Vector3 aimPoint = aimBone.GlobalPosition
                           + new Vector3(0.0f, _rng.RandfRange(-AimHeightJitter, AimHeightJitter), 0.0f);

        float azimuth = RandomizeDirection ? _rng.RandfRange(0.0f, Mathf.Tau) : 0.0f;
        Vector3 offset = new Vector3(Mathf.Cos(azimuth), 0.0f, Mathf.Sin(azimuth)) * SpawnDistance;
        Vector3 spawn = aimPoint + offset;

        var body = new RigidBody3D
        {
            Mass = BallMass,
            // The ball is small, fast and aimed at thin limbs; without continuous detection it
            // tunnels straight through the ragdoll at these speeds and the hit silently never
            // happens - which would look like a policy that learned to ignore impacts.
            ContinuousCd = true,
            GlobalPosition = spawn
        };

        var shape = new CollisionShape3D { Shape = new SphereShape3D { Radius = BallRadius } };
        body.AddChild(shape);

        var mesh = new MeshInstance3D
        {
            Mesh = new SphereMesh
            {
                Radius = BallRadius,
                Height = BallRadius * 2.0f,
                Material = new StandardMaterial3D
                {
                    AlbedoColor = new Color(0.9f, 0.25f, 0.1f),
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
        body.ApplyCentralImpulse((aimPoint - spawn).Normalized() * LaunchSpeed * BallMass);

        _live.Add(new Ball(body, BallLifetimeSeconds));
    }
}

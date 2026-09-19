using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// Godot's <c>BallGun</c>, firing inside the MuJoCo simulation.
/// </summary>
/// <remarks>
/// <para><b>The PERTURB task is a thrown ball, not a force on the pelvis.</b> A force applied at the
/// pelvis acts through the centre of mass; a ball hits a LIMB, and the torque <c>r x F</c> it adds is
/// what the Isaac task config calls "most of what makes a shove hard to reject". Measured: the pelvis
/// push was survivable to 90 N.s, while the ball topples the body at 16 N.s. Testing with the force
/// was measuring a different experiment.</para>
/// <para>Parameters mirror <c>BallGun</c>'s small ball, which is what it fires on every shot because
/// <c>SmallBallProbability</c> is 1.0: 3.0 kg, radius 0.06 m, 6.0 m/s +/- 0.25, spawned 2.0 m out,
/// aimed at one of twelve target bones. Godot's delivered impulse is <c>m*(v - v_reflected)</c> which
/// at restitution 0.25 is 22.5 N.s, not the 18 N.s momentum alone suggests.</para>
/// </remarks>
internal sealed class MjBallGun : IDisposable
{
    /// <summary>The twelve bones <c>BallGun.SmallBallTargetBones</c> aims at.</summary>
    internal static readonly string[] TargetBones =
    {
        "Head", "Chest", "Spine", "Pelvis",
        "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
        "Thigh_L", "Shin_L", "Thigh_R", "Shin_R",
    };

    /// <summary>
    /// Ball mass in kg, read from the MODEL rather than assumed.
    /// </summary>
    /// <remarks>
    /// The impulse arithmetic must use the mass MuJoCo is actually simulating. Carrying a separate
    /// constant here would silently misreport every delivered impulse the moment the model changed.
    /// </remarks>
    internal float Mass { get; private set; } = 3.0f;

    /// <summary>Launch speed in m/s. Godot's is 6.0.</summary>
    internal float Speed { get; set; } = 6.0f;

    private const float SpeedJitter = 0.25f;
    private const float SpawnDistance = 2.0f;

    /// <summary>
    /// Cap on how far ahead of a moving bone the gun may aim, matching <c>BallGun.MaxAimLead</c>.
    /// </summary>
    /// <remarks>
    /// Without lead the gun aims where the bone WAS at trigger time. Over the ~0.33 s flight from
    /// 2 m a swinging limb has moved well clear, and the shot misses - which wastes the interval and,
    /// in training, an entire perturbation the episode was supposed to contain.
    /// </remarks>
    private const float MaxAimLead = 0.5f;

    /// <summary>
    /// Where the ball waits between shots.
    /// </summary>
    /// <remarks>
    /// **Above the floor, not below it.** The world floor is an infinite plane, so parking the ball
    /// beneath it means penetrating the ground and MuJoCo ejects it violently - that produced an
    /// identical 87.9 deg tilt on every target bone regardless of impulse, which is what exposed it.
    /// </remarks>
    private static readonly Vector3 Park = new(40.0f, 2.0f, 40.0f);

    private readonly MjBridge _bridge;
    private readonly RandomNumberGenerator _rng = new();
    private readonly int _ballBody;
    private readonly int _ballJoint;
    private readonly (string Name, int Body)[] _targets;

    private Vector3 _launchVelocity;
    private Vector3 _comPrev;
    private double _levelUntil;
    private float _bodyMass;
    private Vector3 _aimPoint;

    /// <summary>Shots fired and shots that actually transferred momentum.</summary>
    internal int Shots { get; private set; }

    /// <summary>Shots that delivered less than <see cref="HitThreshold"/>; a wasted interval.</summary>
    internal int Misses { get; private set; }

    /// <summary>Impulse below which a shot counts as a miss rather than a hit, N.s.</summary>
    // Free fall alone moves the whole body by g*dt*mass = 2.8 N.s in one step, so a
    // threshold below that would call every shot a hit. 8 N.s is a clear impact.
    internal const float HitThreshold = 8.0f;

    internal MjBallGun(MjBridge bridge, ulong seed = 1)
    {
        _bridge = bridge;
        _rng.Seed = seed;
        _ballBody = bridge.BodyId("ball");
        _ballJoint = bridge.JointId("ball_free");

        var ids = new System.Collections.Generic.List<(string Name, int Body)>();
        foreach (string bone in TargetBones)
        {
            int id = bridge.BodyId(bone);
            if (id >= 0)
            {
                ids.Add((bone, id));
            }
        }

        _targets = ids.ToArray();
        if (_ballBody >= 0)
        {
            Mass = (float)bridge.BodyMass(_ballBody);
        }

        _bodyMass = (float)bridge.TotalMass() - Mass;
        Park_();
    }

    /// <summary>Upward force that exactly cancels the projectile's weight, in Godot's frame.</summary>
    /// <remarks>
    /// **The projectile must not fall.** A ballistic ball aimed straight at a bone from
    /// <see cref="SpawnDistance"/> cannot reach it below 4.43 m/s - a projectile's range is
    /// v^2/g, which is 0.60 m at the 2.42 m/s the training curriculum reached, so every shot
    /// landed on the floor short of the dummy and "ball speed" varied the miss distance rather
    /// than the difficulty. Cancelling gravity makes the flight a straight line, so speed maps
    /// linearly onto delivered impulse and the standoff is the same at every difficulty. The
    /// training environment does exactly this; see perturb_env._park_ball.
    /// </remarks>
    private Vector3 AntiGravity => new(0.0f, Mass * 9.81f, 0.0f);

    /// <summary>True when the model actually carries a projectile.</summary>
    internal bool Available => _ballBody >= 0 && _ballJoint >= 0 && _targets.Length > 0;

    /// <summary>Body index of the projectile, so it can be excluded from centre-of-mass sums.</summary>
    internal int BallBody => _ballBody;

    /// <summary>True while a shot is resolving.</summary>
    internal bool InFlight { get; private set; }

    /// <summary>Name of the bone the current shot was aimed at.</summary>
    internal string LastTarget { get; private set; } = "-";

    internal void Reset()
    {
        Shots = Misses = 0;
        LastTarget = "-";
        Park_();
    }

    public void Dispose() => _rng.Dispose();

    /// <summary>Holds the ball out of the way; a free body left alone falls forever.</summary>
    internal void Park_()
    {
        if (!Available)
        {
            return;
        }

        _bridge.SetFreeJoint(_ballJoint, Park, Vector3.Zero);
        _bridge.SetBodyForce(_ballBody, Vector3.Zero);
        _levelUntil = 0.0;
        InFlight = false;
    }

    /// <summary>Launches at a uniformly chosen target bone from a random heading.</summary>
    internal void Fire(double now)
    {
        if (!Available)
        {
            return;
        }

        int index = _rng.RandiRange(0, _targets.Length - 1);
        int body = _targets[index].Body;
        Vector3 aim = _bridge.BodyTransform(body).Origin;
        LastTarget = _targets[index].Name;

        float theta = _rng.RandfRange(0.0f, Mathf.Tau);
        var direction = new Vector3(Mathf.Cos(theta), 0.0f, Mathf.Sin(theta));
        Vector3 start = aim + (direction * SpawnDistance);
        start.Y = Mathf.Max(start.Y, 0.1f);

        float speed = Speed + _rng.RandfRange(-SpeedJitter, SpeedJitter);

        // Lead the target. Exact for a bone at constant velocity, and a flailing limb is not - hence
        // the cap, which is BallGun's own reasoning for MaxAimLead.
        Vector3 bodyVelocity = _bridge.BodyOriginVelocity(body);
        float flight = start.DistanceTo(aim) / Mathf.Max(speed, 0.01f);
        Vector3 lead = bodyVelocity * flight;
        if (lead.Length() > MaxAimLead)
        {
            lead = lead.Normalized() * MaxAimLead;
        }

        _aimPoint = aim + lead;
        Vector3 velocity = (_aimPoint - start).Normalized() * speed;

        _bridge.SetFreeJoint(_ballJoint, start, velocity);
        _bridge.SetBodyForce(_ballBody, AntiGravity);
        // 20% past the nominal flight, then the ball is on its own.
        _levelUntil = now + (SpawnDistance / Mathf.Max(speed, 0.01f) * 1.2f);
        _launchVelocity = velocity;
        _comPrev = _bridge.CenterOfMassVelocity();
        InFlight = true;
        Shots++;
    }

    /// <summary>Restores normal physics once the shot has had time to arrive.</summary>
    /// <remarks>
    /// The approach is held level so that speed means impulse rather than miss distance, but a ball
    /// that never falls is its own perturbation: a shot that misses would fly straight on for ever
    /// and come back through the dummy. Past the flight window it is an ordinary object again -
    /// it drops, lands and rolls. Call once per physics step.
    /// </remarks>
    internal void Update(double now)
    {
        if (!Available || !InFlight || _levelUntil <= 0.0 || now < _levelUntil)
        {
            return;
        }

        _bridge.SetBodyForce(_ballBody, Vector3.Zero);
        _levelUntil = 0.0;
    }

    /// <summary>Records a resolved shot, counting it as a hit or a miss.</summary>
    internal void Retire(float deliveredImpulse)
    {
        if (deliveredImpulse < HitThreshold)
        {
            Misses++;
        }

        Park_();
    }

    /// <summary>
    /// Impulse delivered to the DUMMY in the last physics step, in newton-seconds. Call once per
    /// step while a shot is in flight; the caller keeps the peak.
    /// </summary>
    /// <remarks>
    /// <para>**Measured on the body, per step, not on the ball since launch.** The original version
    /// returned the ball's own velocity change since launch, which gravity dominates - a 15 kg ball
    /// in free flight for 0.33 s reported ~49 N.s before touching anything, so every shot counted
    /// as a hit against the 1 N.s threshold.</para>
    /// <para>Integrating the BODY's change since launch is no better: a dummy that falls over
    /// accumulates the whole fall and reported 168 N.s for a shot that knocked it down. One step is
    /// what separates the two - free fall contributes g*dt*mass = 2.8 N.s per step, an impact
    /// spikes far above that.</para>
    /// </remarks>
    internal float DeliveredImpulse()
    {
        if (!InFlight)
        {
            return 0.0f;
        }

        Vector3 now = _bridge.CenterOfMassVelocity();
        float delivered = (now - _comPrev).Length() * _bodyMass;
        _comPrev = now;
        return delivered;
    }
}

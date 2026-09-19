using System;
using System.Collections.Generic;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// Steps the dummy's physics in MuJoCo and renders the result in Godot.
/// </summary>
/// <remarks>
/// <para><b>Why this exists.</b> Phase 0 measured ~100 authored configurations of the Jolt ragdoll and
/// none could lift a foot and stay upright: the outcome was binary everywhere, either the foot stayed
/// on the floor or the body fell. MuJoCo holds single-leg support for 17.5 s on the same rig and walks
/// it 4.835 m in 40 s at 100% uprightness with 0% flight phase. So the humanoid's physics moves to
/// MuJoCo and Godot renders it.</para>
/// <para>The model is generated from the Godot scene by <c>mujoco_rig/build_mjcf.py</c>. Visual
/// proxies are built by reading that same file (<see cref="MjProxyBuilder"/>), so the shapes are
/// never declared twice.</para>
/// <para>What drives the body, in order of precedence: nothing at all (<see cref="Passive"/>); a
/// trained policy (<see cref="PolicyPath"/>, run by <see cref="MjPolicyDriver"/>); or, on a
/// position-actuated model only, the Jolt-era scripted gait and balance assist
/// (<see cref="MjScriptedController"/>). The current plant is torque-actuated, where only the first
/// two apply.</para>
/// </remarks>
public partial class MujocoDummy : Node3D
{
    /// <summary>Generated MuJoCo model. Regenerate with <c>mujoco_rig/build_mjcf.py</c>.</summary>
    [Export] public string ModelPath { get; set; } = "res://mujoco_rig/dummy.xml";

    /// <summary>
    /// Directory holding <c>mujoco.dll</c>. It ships with the Python environment rather than the
    /// game, so the loader has to be pointed at it explicitly.
    /// </summary>
    [Export] public string MujocoLibraryDirectory { get; set; } = string.Empty;

    /// <summary>
    /// A trained ONNX policy to drive the body with, instead of the scripted gait and the balance
    /// assist. Empty keeps the scripted controller.
    /// </summary>
    /// <remarks>
    /// **This is the deliverable of the whole MuJoCo track.** The policy trains on mujoco_warp,
    /// is scored on MuJoCo's C engine, and runs here against that same C engine - no boundary in
    /// between. Setting it switches OFF <see cref="MjScriptedController.ApplyBalanceTorque"/>, which is the external
    /// pelvis torque the policy exists to replace; leaving that on would measure the assist.
    /// </remarks>
    [Export] public string PolicyPath { get; set; } = string.Empty;

    /// <summary>Commanded velocity for a walk policy: forward, lateral, turn rate (m/s, m/s, rad/s).</summary>
    /// <remarks>
    /// Ignored by a perturb policy, which leaves its command channel at zero. The Walk scene sets
    /// 0.25 m/s, the stage-1 speed the shipped walk brain was trained and scored at; a command it
    /// never saw is, in effect, a different policy.
    /// </remarks>
    [Export] public Vector3 WalkCommand { get; set; } = new(0.6f, 0.0f, 0.0f);

    /// <summary>
    /// Drive NOTHING - no scripted gait, no balance assist, no policy. The body as authored.
    /// </summary>
    /// <remarks>
    /// <para>This is the baseline every result is read against, and it is worth being able to see
    /// rather than only measure. Clearing <see cref="PolicyPath"/> is NOT the same thing: the scene
    /// then still runs the scripted controller and <see cref="MjScriptedController.ApplyBalanceTorque"/>, which writes up
    /// to 90.7 N.m of external torque onto the pelvis and holds the body up on its own.</para>
    /// <para>Measured on the compliant plant: with this on, the dummy collapses in about 2.5 s and
    /// scores 6.4% upright over 40 s. That is the character as designed - it is meant to need active
    /// balance, and it is why "do nothing" is not a viable policy any more.</para>
    /// </remarks>
    [Export] public bool Passive { get; set; }

    /// <summary>Drive the scripted gait. Off leaves the body standing under the balance controller.</summary>
    /// <remarks>Ignored when <see cref="PolicyPath"/> is set - a policy drives every joint itself.</remarks>
    [Export] public bool Walk { get; set; } = true;

    /// <summary>Gait frequency in Hz. 0.6 is the measured best.</summary>
    [Export] public float GaitFrequency { get; set; } = 0.6f;

    /// <summary>Lateral weight-shift amplitude, radians of hip roll.</summary>
    [Export] public float ShiftAmplitude { get; set; } = 0.25f;

    /// <summary>Swing-leg hip amplitude, radians.</summary>
    [Export] public float LiftAmplitude { get; set; } = -0.4f;

    /// <summary>Swing-leg knee amplitude, radians.</summary>
    [Export] public float KneeAmplitude { get; set; } = 0.4f;

    /// <summary>
    /// Symmetric hip-roll gain per m/s of lateral centre-of-mass velocity.
    /// </summary>
    /// <remarks>
    /// <para><b>Damping only - there is deliberately no proportional term.</b> Swept 0 / 0.05 / 0.10 /
    /// 0.30 rad per metre of position error: every non-zero proportional gain destabilises the gait
    /// (0.05 already fails at 60 s, 0.30 at 40 s) because it fights the lateral excursion the gait
    /// needs. Bleeding off lateral MOMENTUM instead walks 180 s at 99.2% uprightness with 2.4% flight,
    /// where the undamped oscillator falls at about 60 s.</para>
    /// <para>The two hips act on the centre of mass only when driven with the SAME sign: the
    /// antisymmetric pair that produces the step moves it 0.001-0.006 m, while both together move it
    /// 0.89 m at 0.25 rad. So the step is antisymmetric and the balance term symmetric, superposed on
    /// the same two joints.</para>
    /// </remarks>
    [Export] public float LateralDamping { get; set; } = 0.05f;

    /// <summary>Clamp on the symmetric balance term, radians, so it cannot swamp the gait.</summary>
    [Export] public float MaxLateralCorrection { get; set; } = 0.12f;

    /// <summary>
    /// Fire Godot's BallGun inside MuJoCo - the project's real PERTURB test.
    /// </summary>
    /// <remarks>
    /// Prefer this to <see cref="PushAtSeconds"/>: a force at the pelvis acts through the centre of
    /// mass, while a ball hits a limb and adds the torque that actually makes a shove hard to reject.
    /// The pelvis push survived 90 N.s; the ball topples at 16 N.s.
    /// </remarks>
    [Export] public bool BallGunEnabled { get; set; }

    /// <summary>Seconds between shots in the SCENE.</summary>
    /// <remarks>
    /// <para>Deliberately shorter than the 4-7 s the policy trains against: the scene is there to
    /// be watched, and a hit every three seconds keeps something happening on screen.</para>
    /// <para><b>It is therefore harder than the trained task, and the score here is not the
    /// score.</b> Measured with no policy at all, a single 10 kg impact is survivable 75% of the
    /// time, but the same ball every 2-4 s topples 8/8 in 6.8 s - because the next one lands
    /// mid-recovery and no recovery ever finishes. Judge a policy with <c>eval.py</c>, which uses
    /// the trained cadence; treat this scene as a demo, not a measurement.</para>
    /// </remarks>
    [Export] public float BallInterval { get; set; } = 3.0f;

    /// <summary>Delay before the first shot.</summary>
    [Export] public float BallFirstShot { get; set; } = 2.0f;

    /// <summary>Seconds at which to shove the pelvis, for perturbation tests. Zero disables.</summary>
    [Export] public float PushAtSeconds { get; set; }

    /// <summary>Shove force in newtons, applied along Godot's X for <see cref="PushSeconds"/>.</summary>
    [Export] public float PushForce { get; set; } = 500.0f;

    /// <summary>How long the shove is held. 150 ms at 500 N is 75 N.s, the measured recovery limit.</summary>
    [Export] public float PushSeconds { get; set; } = 0.15f;

    /// <summary>
    /// Keep a child <c>Camera3D</c> pointed at the dummy. Without it the walk leaves frame in
    /// seconds, since it covers about 6 m per minute.
    /// </summary>
    [Export] public bool FollowCamera { get; set; } = true;

    /// <summary>Camera offset from the pelvis, in Godot metres.</summary>
    [Export] public Vector3 CameraOffset { get; set; } = new(2.5f, 1.0f, 3.0f);

    /// <summary>Seconds before the arena reports and quits. Zero runs indefinitely.</summary>
    [Export] public float AutoQuitSeconds { get; set; }

    private MjBridge? _bridge;
    private List<(int Body, Node3D Node)> _proxies = new();
    private readonly Dictionary<string, int> _actuators = new();
    private int _pelvis = -1;
    private int _footL = -1;
    private int _footR = -1;
    private double _elapsed;
    private double _accumulator;

    private Camera3D? _camera;
    private MjBallGun? _gun;
    private MjPolicyDriver? _policy;
    private double _nextShot;
    private double _shotAt = -1.0;
    private float _shotPeakImpulse;
    private int _shotsFired;
    private MjGaitMetrics? _metrics;
    private MjScriptedController? _scripted;

    /// <summary>True when the model drives its joints with <c>motor</c> (torque) actuators.</summary>
    /// <remarks>
    /// Detected from the MJCF text rather than from mjModel, because the P/Invoke layout does not
    /// expose actuator_gaintype and adding an offset for one boolean is not worth regenerating
    /// <see cref="MjLayout"/>. The generator only ever emits <c>&lt;motor</c> for the policy's own
    /// joints, so the substring is exact.
    /// </remarks>
    private bool _torqueActuators;
    private bool _warnedScripted;
    private bool _wasPassive;
    private double _manualPushUntil;
    private MjModelDefinition? _definition;

    // Read from the model rather than hardcoded: a mismatch against the real timestep makes the
    // accumulator drift, skipping or doubling steps over long runs.
    private double _mjTimestep = 0.004167;

    public override void _Ready()
    {
        try
        {
            MjInterop.SetLibraryDirectory(MujocoLibraryDirectory);
            string path = ProjectSettings.GlobalizePath(ModelPath);
            _definition = new MjModelDefinition(path);
            _mjTimestep = _definition.Timestep;
            _torqueActuators = _definition.HasTorqueActuators;
            _bridge = LoadModel(path);
            if (_bridge == null)
            {
                return;
            }

            _pelvis = MjPolicyObservation.Required(_bridge.BodyId("Pelvis"), "Pelvis");
            _footL = MjPolicyObservation.Required(_bridge.BodyId("Foot_L"), "Foot_L");
            _footR = MjPolicyObservation.Required(_bridge.BodyId("Foot_R"), "Foot_R");
            _metrics = new MjGaitMetrics(_bridge, _pelvis, _footL, _footR,
                                         PushAtSeconds, PushSeconds, BallGunEnabled);
            CreateControllers(_bridge);
            CreateBallGun(_bridge);

            _camera = GetNodeOrNull<Camera3D>("Camera3D");
            _nextShot = BallFirstShot;
            _proxies = MjProxyBuilder.Build(_bridge, this, _definition.Document);
            LogSummary(_bridge);
        }
        catch (Exception e)
        {
            _ExitTree();
            SetPhysicsProcess(false);
            GD.PrintErr("[MujocoDummy] initialization failed: " + e.Message);
        }
    }

    /// <summary>
    /// Scene keys, matching <c>RagdollDebugInput</c> so the MuJoCo scenes behave like the others.
    /// </summary>
    /// <remarks>
    /// <para>R resets, which is the one that actually matters here: a fallen dummy is otherwise a
    /// dead scene that has to be relaunched, and relaunching costs a MuJoCo model load and a Warp
    /// kernel warm-up.</para>
    /// <para>B fires the ball on demand rather than waiting out <see cref="BallInterval"/>, and
    /// Space is the same shove the Jolt scenes bind. Handled in <c>_UnhandledInput</c> so the
    /// camera controller keeps first claim on WASD.</para>
    /// </remarks>
    public override void _UnhandledInput(InputEvent @event)
    {
        if (_bridge == null || @event is not InputEventKey { Pressed: true, Echo: false } key)
        {
            return;
        }

        switch (key.Keycode)
        {
            case Key.R:
                ResetDummy();
                break;
            case Key.B:
                if (_gun is not { Available: true })
                {
                    // Silence here reads as a broken key. The Stand and Walk scenes load
                    // `dummy.xml`, which carries no projectile at all - only `dummy_ball.xml` does.
                    GD.Print("[MujocoDummy] no ball in this model (" + ModelPath + "); "
                             + "use the Perturb scene, which loads dummy_ball.xml");
                }
                else
                {
                    FireShot(_gun);
                    GD.Print($"[MujocoDummy] manual shot -> {_gun.LastTarget}");
                }

                break;
            case Key.Space:
                _manualPushUntil = _elapsed + PushSeconds;
                GD.Print($"[MujocoDummy] manual shove {PushForce:F0} N");
                break;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_bridge == null)
        {
            return;
        }

        try
        {
            // MuJoCo runs on its own fixed timestep; step whole substeps to track wall time without
            // letting a long frame spiral.
            _accumulator += Math.Min(delta, 0.1);
            while (_accumulator >= _mjTimestep)
            {
                _elapsed += _mjTimestep;
                DriveBody(_bridge);
                ApplyPush(_bridge);
                ServiceBallGun();
                _bridge.Step();
                _accumulator -= _mjTimestep;
            }

            SyncProxies(_bridge);
            FollowWithCamera(_bridge);
            _metrics?.Sample(_elapsed);

            if (AutoQuitSeconds > 0.0f && _elapsed >= AutoQuitSeconds)
            {
                _metrics?.Report(_elapsed, PushForce, BallGunEnabled ? _gun : null, _shotsFired);
                GetTree().Quit();
            }
        }
        catch (Exception e)
        {
            SetPhysicsProcess(false);
            _ExitTree();
            GD.PrintErr("[MujocoDummy] simulation stopped: " + e.Message);
        }
    }

    public override void _ExitTree()
    {
        _policy?.Dispose();
        _policy = null;
        _gun?.Dispose();
        _gun = null;
        _bridge?.Dispose();
        _bridge = null;
    }

    /// <summary>Loads the model at its rest pose, and reads which kind of actuator it carries.</summary>
    private MjBridge? LoadModel(string path)
    {
        MjBridge bridge;
        try
        {
            bridge = new MjBridge(path);
        }
        catch (Exception e)
        {
            GD.PrintErr("[MujocoDummy] " + e.Message);
            return null;
        }

        // Start from the model's own rest pose. MuJoCo's fresh mjData is qpos0, where the arms
        // hang inside the legs; MjBridge.ResetData prefers the `rest` keyframe.
        bridge.ResetData();

        return bridge;
    }

    /// <summary>The scripted gait, for a position-actuated model, and the policy when one is set.</summary>
    private void CreateControllers(MjBridge bridge)
    {
        if (!string.IsNullOrWhiteSpace(PolicyPath))
        {
            // Fail loudly. A policy that silently declines to load leaves the scripted gait running,
            // and the scene then looks like the policy is simply bad - which is a far more expensive
            // thing to debug than a missing file.
            _policy = MjPolicyDriver.Load(bridge, PolicyPath, _mjTimestep);
            GD.Print($"[MujocoDummy] driving with {PolicyPath} ({_policy.Task}); "
                     + "scripted gait and balance assist are OFF");
            return;
        }
        if (_torqueActuators || bridge.ActuatorCount == 0) return;

        foreach (string joint in new[]
                 {
                     "Thigh_L_rz", "Thigh_R_rz", "Thigh_L_rx", "Thigh_R_rx", "Shin_L_rx", "Shin_R_rx",
                 })
        {
            int id = bridge.ActuatorId(joint);
            if (id < 0)
            {
                throw new InvalidOperationException($"Scripted actuator '{joint}' not found in the model");
            }

            _actuators[joint] = id;
        }

        _scripted = new MjScriptedController(bridge, _actuators, _pelvis)
        {
            LateralDamping = LateralDamping,
            MaxLateralCorrection = MaxLateralCorrection,
            GaitFrequency = GaitFrequency,
            ShiftAmplitude = ShiftAmplitude,
            LiftAmplitude = LiftAmplitude,
            KneeAmplitude = KneeAmplitude,
        };


    }

    /// <summary>The projectile carried by <c>dummy_ball.xml</c>, parked until it is fired.</summary>
    private void CreateBallGun(MjBridge bridge)
    {
        _gun = new MjBallGun(bridge);
        if (BallGunEnabled && !_gun.Available)
        {
            GD.PrintErr("[MujocoDummy] BallGunEnabled but the model carries no 'ball' body - "
                        + "regenerate it with mujoco_rig/build_mjcf.py");
        }

        if (_gun.Available)
        {
            // Keep the projectile out of the body's own centre of mass; including it drags the
            // quantity the balance term regulates.
            bridge.ExcludeFromCom = _gun.BallBody;
            _gun.Park_();
        }
    }

    private void LogSummary(MjBridge bridge)
    {
        GD.Print($"[MujocoDummy] MuJoCo {MjLayout.Version}: {bridge.BodyCount - 1} bodies, "
                 + $"{bridge.ActuatorCount} actuators, {_proxies.Count} visual proxies");
        // Sanity-check the derived quantities rather than trusting them: a zero mass total silently
        // makes CenterOfMass* return the zero vector, which would disable the balance term without
        // any error at all.
        GD.Print($"[MujocoDummy] totalMass={bridge.TotalMass():F3} kg  "
                 + $"COM={bridge.CenterOfMass()}  COMvel={bridge.CenterOfMassVelocity()}");
    }

    /// <summary>Returns the body to its rest pose and clears every accumulated metric.</summary>
    /// <remarks>
    /// The gait clock is reset too. Leaving <c>_elapsed</c> running would restart the body at rest
    /// in the middle of a gait cycle, which looks like the reset failed.
    /// </remarks>
    private void ResetDummy()
    {
        if (_bridge == null)
        {
            return;
        }

        _bridge.ResetData();
        _bridge.ClearControls();
        // A held heading must not outlive the body it was held for: without this, after a reset the
        // dummy is told to turn back toward wherever it was facing before.
        _policy?.Reset();
        _gun?.Reset();
        _manualPushUntil = 0;
        _elapsed = 0.0;
        _accumulator = 0.0;
        _nextShot = BallFirstShot;
        _shotAt = -1.0;
        _shotsFired = 0;
        _metrics?.Reset();
        SyncProxies(_bridge);
        GD.Print("[MujocoDummy] reset");
    }

    /// <summary>Writes this physics step's controls: none, the policy's, or the scripted gait's.</summary>
    private void DriveBody(MjBridge bridge)
    {
        bridge.SetBodyTorque(_pelvis, Vector3.Zero);
        if (_wasPassive != Passive) _policy?.Reset();
        _wasPassive = Passive;
        if (Passive)
        {
            // Nothing at all: no gait, no assist, no policy. On the torque plant zero control is
            // zero muscle and the body crumples; a position-actuated model instead holds its
            // authored rest pose, which is what "no controller" means there.
            bridge.ClearControls();
        }
        else if (_policy != null)
        {
            // The policy replaces BOTH the scripted gait and the balance assist. It decides its
            // own inference rate from the contract, so it is called every physics step.
            _policy.Command = WalkCommand;
            _policy.Step();
        }
        else if (_torqueActuators)
        {
            // The scripted gait and the balance assist both write JOINT TARGETS, which a
            // `motor` actuator reads as newton-metres. Driving them against a torque plant
            // means commanding hundreds of N.m of nonsense, so they are disabled rather than
            // ported: they are Jolt-era scaffolding that a policy replaces outright.
            if (!_warnedScripted)
            {
                _warnedScripted = true;
                GD.Print("[MujocoDummy] this model has torque actuators; the scripted gait and "
                         + "balance assist are disabled. Set PolicyPath to drive it, or "
                         + "Passive = true to watch the body unpowered.");
            }

            bridge.ClearControls();
        }
        else if (_scripted != null)
        {
            _scripted.DriveGait(_elapsed, Walk);
            _scripted.ApplyBalanceTorque();
        }
    }

    /// <summary>The scheduled pelvis shove, held for <see cref="PushSeconds"/>.</summary>
    private void ApplyPush(MjBridge bridge)
    {
        bool pushing = _elapsed < _manualPushUntil || (PushAtSeconds > 0.0f
            && _elapsed >= PushAtSeconds && _elapsed < PushAtSeconds + PushSeconds);
        bridge.SetBodyForce(_pelvis, pushing ? new Vector3(PushForce, 0.0f, 0.0f) : Vector3.Zero);
    }

    /// <summary>Fires and retires shots, and records the impulse each one delivered.</summary>
    private void ServiceBallGun()
    {
        if (_gun == null || !_gun.Available)
        {
            return;
        }

        // **Park it every step, not once.** A free body left alone falls forever: after 45 s it is
        // 9.9 km down doing 441 m/s, and carrying those magnitudes in the state degrades the solve
        // for everything else. Measured: the walk dropped from 99.7% upright to 65.7% purely from
        // the projectile existing in the model.
        if (!_gun.InFlight && !BallGunEnabled)
        {
            _gun.Park_();
            return;
        }

        _gun.Update(_elapsed);

        if (!_gun.InFlight && _elapsed >= _nextShot)
        {
            FireShot(_gun);
        }
        else if (_gun.InFlight)
        {
            _shotPeakImpulse = Math.Max(_shotPeakImpulse, _gun.DeliveredImpulse());
            if (_elapsed - _shotAt > 2.0)
            {
                bool hit = _shotPeakImpulse >= MjBallGun.HitThreshold;
                GD.Print($"[MujocoDummy] ball {_shotsFired} -> {_gun.LastTarget}: "
                         + $"{_shotPeakImpulse:F1} N.s "
                         + (hit ? "delivered" : "MISS (no contact)"));
                _gun.Retire(_shotPeakImpulse);
                // A miss wastes the whole interval, so re-fire promptly rather than idling.
                _nextShot = _elapsed + (hit ? BallInterval : 0.5);
            }
        }
    }

    /// <summary>Fires the ball now, and starts measuring what it delivers.</summary>
    private void FireShot(MjBallGun gun)
    {
        gun.Fire(_elapsed);
        _shotAt = _elapsed;
        _shotPeakImpulse = 0.0f;
        _shotsFired++;
    }

    private void SyncProxies(MjBridge bridge)
    {
        foreach ((int body, Node3D node) in _proxies)
        {
            node.Transform = bridge.BodyTransform(body);
        }
    }

    private void FollowWithCamera(MjBridge bridge)
    {
        if (!FollowCamera || _camera == null)
        {
            return;
        }

        Vector3 pelvis = bridge.BodyTransform(_pelvis).Origin;
        _camera.GlobalPosition = pelvis + CameraOffset;
        _camera.LookAt(pelvis, Vector3.Up);
    }
}

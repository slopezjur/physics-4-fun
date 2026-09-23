using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Physical projectile trials, with the original force-pulse diagnostic available.</summary>
public partial class MimicPerturb : MimicTrial
{
    [ExportGroup("Experiment")]
    [Export(PropertyHint.Enum, "Ball collision,Force pulse")] public int Mode { get; set; }
    [Export(PropertyHint.Range, "1,8,0.1")] public float BallSpeed { get; set; } = 2;
    [Export(PropertyHint.Enum, "Random,Head,Chest,Spine,Pelvis,UpperArm_L,Forearm_L,UpperArm_R,Forearm_R,Thigh_L,Shin_L,Thigh_R,Shin_R")]
    public string BallTarget { get; set; } = "Random";
    [Export] public bool AutoFire { get; set; }
    [Export(PropertyHint.Range, "1,10,0.5")] public float FireInterval { get; set; } = 2.5f;
    [ExportGroup("Push at chest center of mass")]
    [Export(PropertyHint.Range, "0,100,1")] public float ForceNewtons { get; set; } = 20;
    [Export(PropertyHint.Range, "0,240,1")] public int PushStartStep { get; set; } = 60;
    [Export(PropertyHint.Range, "1,60,1")] public int PushDurationSteps { get; set; } = 6;
    [Export(PropertyHint.Enum, "Forward,Backward,Left,Right,Random")] public int Direction { get; set; } = 4;
    private MjMimicPerturbTrial? _trial;
    private MjMimicBallTrial? _ballTrial;
    private string _directionName = string.Empty;
    private Vector3 _activeDirection;
    private double _nextFireTime;
    private readonly Random _rng = new();
    // UI directions are authored in Godot's Y-up frame. Convert them to MuJoCo's
    // world frame at the force/ball boundary so labels and batch protocols agree.
    private static readonly Vector3[] Directions = { Vector3.Forward, Vector3.Back, Vector3.Right, Vector3.Left };
    private static readonly string[] DirectionNames = { "Forward (+X)", "Backward (-X)", "Left (-Y)", "Right (+Y)" };
    protected override string TrialName => "MimicPerturb";
    protected override string ViewerTask => "ball";
    protected override void ConfigureLaunch()
    {
        base.ConfigureLaunch();
        string? speed = LaunchOption("--ball-speed");
        if (speed != null) BallSpeed = float.Parse(speed, System.Globalization.CultureInfo.InvariantCulture);
    }

    public override void _Ready()
    {
        for (int i = 0; i < 5; i++)
        {
            var btn = GetNodeOrNull<Button>($"Hud/Margin/Controls/Directions/Direction{i}");
            if (btn != null)
            {
                int selected = i;
                btn.Pressed += () => { Direction = selected; Restart(); };
            }
        }
        GetNode<Button>("Hud/Margin/Controls/Launch").Pressed += FireBall;
        base._Ready();
    }

    public override void _UnhandledKeyInput(InputEvent @event)
    {
        if (@event is InputEventKey { Pressed: true, Echo: false } key)
        {
            if (key.PhysicalKeycode == Key.B) { FireBall(); GetViewport().SetInputAsHandled(); }
            else base._UnhandledKeyInput(@event);
        }
        else base._UnhandledKeyInput(@event);
    }

    private void FireBall()
    {
        if (CanAdvance && _ballTrial != null) _ballTrial.Launch();
    }

    private protected override void ResetTrial()
    {
        if (Direction < 0 || Direction > 4 || !float.IsFinite(ForceNewtons) || ForceNewtons < 0)
            throw new InvalidOperationException("Choose a valid direction and nonnegative finite push force.");

        if (Direction == 4)
        {
            float angle = (float)(_rng.NextDouble() * 2.0 * Math.PI);
            _activeDirection = new Vector3((float)Math.Cos(angle), 0, (float)Math.Sin(angle));
            _directionName = $"Random ({(int)(angle * 180 / Math.PI)}°)";
        }
        else
        {
            _activeDirection = Directions[Direction];
            _directionName = DirectionNames[Direction];
        }

        _trial = null; _ballTrial = null;
        if (Mode == 0)
        {
            if (PushStartStep < 0 || (PushStartStep * Driver.ControlTime + .65f / BallSpeed + .5f) >= ConfiguredDuration)
                throw new InvalidOperationException("Ball launch must leave time for flight and recovery.");
            Driver.Reset(ConfiguredPhase);
            _ballTrial = new MjMimicBallTrial(Bridge, Driver, _activeDirection, BallSpeed, PushStartStep, BallTarget, _rng.Next());
            _nextFireTime = PushStartStep * Driver.ControlTime + FireInterval;
            return;
        }
        if (Mode != 1) throw new InvalidOperationException("Unknown perturb mode.");
        var push = new MjMimicPush(MjBridge.GodotToMj(_activeDirection) * ForceNewtons,
            PushStartStep, PushDurationSteps,
            (int)Math.Ceiling(ConfiguredDuration / Driver.ControlTime));
        _trial = new MjMimicPerturbTrial(Bridge, Driver, push, ConfiguredPhase);
    }

    private protected override void StepTrial()
    {
        if (_ballTrial != null)
        {
            _ballTrial.Step();
            if (AutoFire && !Driver.Fallen && Driver.Time >= _nextFireTime)
            {
                _nextFireTime += FireInterval;
                float angle = (float)(_rng.NextDouble() * 2.0 * Math.PI);
                Vector3 randomDir = new((float)Math.Cos(angle), 0, (float)Math.Sin(angle));
                _directionName = $"AutoFire ({(int)(angle * 180 / Math.PI)}°)";
                _ballTrial.Launch(randomDir);
            }
        }
        else
        {
            _trial!.Step();
        }
    }

    protected override string TrialDetails
    {
        get
        {
            if (_ballTrial != null)
            {
                string targetDesc = string.IsNullOrEmpty(_ballTrial.ActiveTargetName) ? BallTarget : _ballTrial.ActiveTargetName;
                var cp = Driver.Balance;
                string marginText = cp.StabilityMargin is float margin ? $"{margin * 100:F1} cm" : "no ground support";
                return $"\nBall · {_directionName} · {BallSpeed:F1} m/s horizontal · target {targetDesc}"
                    + $"\n{(_ballTrial.Launched ? Driver.BallHit ? "Impact confirmed" : "Launched — in flight" : "Waiting to launch · B: launch now")}"
                    + $"\nApprox. capture-point margin: {marginText} (diagnostic only)"
                    + "\n8 kg · radius 9 cm · ballistic flight, gravity enabled";
            }
            if (_trial == null) return string.Empty;
            var result = _trial.Result;
            var push = _trial.Push;
            string recovery = Driver.Fallen ? "fallen" : Driver.StepCount < push.TrialSteps ? "measuring"
                : result.Recovered ? $"settled in {result.RecoverySeconds:F2}s" : "not settled at trial end";
            string pulse = Driver.StepCount < push.StartStep ? "Push pending"
                : Driver.StepCount < push.StartStep + push.DurationSteps ? "PUSH ACTIVE" : "Push finished";
            return $"\n{pulse} · {_directionName} · {push.Force.Length():F0} N for {push.DurationSteps * Driver.ControlTime:F3}s"
                + $" at {push.StartStep * Driver.ControlTime:F2}s · {push.Force.Length() * push.DurationSteps * Driver.ControlTime:F2} N·s"
                + $"\nRecovery: {recovery} · Displacement: {result.MaxHorizontalDisplacementM * 100:F1} cm"
                + $"\nFoot travel: {result.FootTravelM * 100:F1} cm · Contact switches: {result.ContactSwitches}";
        }
    }
}

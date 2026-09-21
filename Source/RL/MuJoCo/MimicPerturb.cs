using System;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Physical projectile trials, with the original force-pulse diagnostic available.</summary>
public partial class MimicPerturb : MimicTrial
{
    [ExportGroup("Experiment")]
    [Export(PropertyHint.Enum, "Ball collision,Force pulse")] public int Mode { get; set; }
    [Export(PropertyHint.Range, "1,8,0.1")] public float BallSpeed { get; set; } = 2;
    [Export(PropertyHint.Enum, "Chest,Spine")] public string BallTarget { get; set; } = "Chest";
    [ExportGroup("Push at chest center of mass")]
    [Export(PropertyHint.Range, "0,100,1")] public float ForceNewtons { get; set; } = 20;
    [Export(PropertyHint.Range, "0,240,1")] public int PushStartStep { get; set; } = 60;
    [Export(PropertyHint.Range, "1,60,1")] public int PushDurationSteps { get; set; } = 6;
    [Export(PropertyHint.Enum, "Forward,Backward,Left,Right")] public int Direction { get; set; }
    private MjMimicPerturbTrial? _trial;
    private MjMimicBallTrial? _ballTrial;
    private string _directionName = string.Empty;
    private static readonly Vector3[] Directions = { Vector3.Right, Vector3.Left, Vector3.Down, Vector3.Up };
    private static readonly string[] DirectionNames = { "Forward (+X)", "Backward (-X)", "Left (-Y)", "Right (+Y)" };
    protected override string TrialName => "MimicPerturb";
    protected override void ConfigureLaunch()
    {
        base.ConfigureLaunch();
        string? speed = LaunchOption("--ball-speed");
        if (speed != null) BallSpeed = float.Parse(speed, System.Globalization.CultureInfo.InvariantCulture);
    }

    public override void _Ready()
    {
        for (int i = 0; i < 4; i++)
        {
            int selected = i;
            GetNode<Button>($"Hud/Margin/Controls/Directions/Direction{i}").Pressed += () => { Direction = selected; Restart(); };
        }
        GetNode<Button>("Hud/Margin/Controls/Launch").Pressed += FireBall;
        base._Ready();
    }

    public override void _UnhandledKeyInput(InputEvent @event)
    {
        if (@event is InputEventKey { Pressed: true, Echo: false, PhysicalKeycode: Key.B })
        { FireBall(); GetViewport().SetInputAsHandled(); }
        else base._UnhandledKeyInput(@event);
    }

    private void FireBall()
    {
        if (CanAdvance && _ballTrial != null) _ballTrial.Launch();
    }

    private protected override void ResetTrial()
    {
        if (Direction < 0 || Direction >= Directions.Length || !float.IsFinite(ForceNewtons) || ForceNewtons < 0)
            throw new InvalidOperationException("Choose a valid direction and nonnegative finite push force.");
        _directionName = DirectionNames[Direction];
        _trial = null; _ballTrial = null;
        if (Mode == 0)
        {
            if (PushStartStep < 0 || (PushStartStep * Driver.ControlTime + .65f / BallSpeed + .5f) >= ConfiguredDuration)
                throw new InvalidOperationException("Ball launch must leave time for flight and recovery.");
            Driver.Reset(ConfiguredPhase);
            _ballTrial = new MjMimicBallTrial(Bridge, Driver, Directions[Direction], BallSpeed, PushStartStep, BallTarget);
            return;
        }
        if (Mode != 1) throw new InvalidOperationException("Unknown perturb mode.");
        var push = new MjMimicPush(Directions[Direction] * ForceNewtons, PushStartStep, PushDurationSteps,
            (int)Math.Ceiling(ConfiguredDuration / Driver.ControlTime));
        _trial = new MjMimicPerturbTrial(Bridge, Driver, push, ConfiguredPhase);
        _directionName = DirectionNames[Direction];
    }

    private protected override void StepTrial()
    {
        if (_ballTrial != null) _ballTrial.Step(); else _trial!.Step();
    }

    protected override string TrialDetails
    {
        get
        {
            if (_ballTrial != null)
                return $"\nBall · {_directionName} · {BallSpeed:F1} m/s horizontal · target {BallTarget}"
                    + $"\n{(_ballTrial.Launched ? Driver.BallHit ? "Impact confirmed" : "Launched — no impact detected yet" : "Waiting to launch · B: launch now")}"
                    + "\n8 kg · radius 9 cm · ballistic flight, gravity enabled";
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

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Shared scene lifecycle for finite motion-guided physics trials.</summary>
public abstract partial class MimicTrial : Node3D
{
    [ExportGroup("Simulation")]
    [Export(PropertyHint.File, "*.xml")] public string ModelPath { get; set; } = "res://mujoco_rig/dummy.xml";
    [Export(PropertyHint.Dir)] public string BundleDirectory { get; set; } = string.Empty;
    [Export] public bool UseLatestExport { get; set; } = true;
    [Export(PropertyHint.Dir)] public string MujocoLibraryDirectory { get; set; } = string.Empty;
    [Export(PropertyHint.Range, "3,6,0.1")] public float TrialSeconds { get; set; } = 5;
    [Export(PropertyHint.Range, "0,3,0.01")] public float StartPhaseSeconds { get; set; }

    private MjBridge? _bridge;
    private MjMimicStandDriver? _driver;
    private List<(int Body, Node3D Node)> _proxies = new();
    private Label _status = null!;
    private Button _pauseButton = null!, _restartButton = null!;
    private double _accumulator;
    private float _trialSeconds, _startPhase;
    private bool _paused, _finished;
    private protected MjBridge Bridge => _bridge!;
    private protected MjMimicStandDriver Driver => _driver!;
    protected float ConfiguredPhase => _startPhase;
    protected float ConfiguredDuration => _trialSeconds;
    protected bool CanAdvance => _driver != null && !_paused && !_finished;
    protected virtual string TrialName => "MimicStand";
    protected virtual string ViewerTask => "stand";
    private protected virtual void ResetTrial() => Driver.Reset(ConfiguredPhase);
    private protected virtual void StepTrial() => Driver.Step();
    protected virtual string TrialDetails => string.Empty;
    protected static string? LaunchOption(string name)
    {
        string[] args = OS.GetCmdlineUserArgs();
        int index = Array.IndexOf(args, name);
        if (index < 0) return null;
        if (index + 1 >= args.Length || args[index + 1].StartsWith("--"))
            throw new ArgumentException($"Missing value for {name}");
        return args[index + 1];
    }
    protected virtual void ConfigureLaunch()
    {
        string? requested = LaunchOption("--mimic-bundle");
        if (requested != null) { BundleDirectory = requested; return; }
        string selection = ProjectSettings.GlobalizePath($"res://logs/mimickit-viewer/{ViewerTask}.json");
        if (!UseLatestExport || !File.Exists(selection)) return;
        using var document = JsonDocument.Parse(File.ReadAllText(selection));
        var state = document.RootElement;
        if (state.GetProperty("schema").GetString() != "mimic_viewer_selection_v1"
            || state.GetProperty("task").GetString() != ViewerTask)
            throw new InvalidOperationException($"Invalid {ViewerTask} viewer selection: {selection}");
        BundleDirectory = state.GetProperty("bundle").GetString()
            ?? throw new InvalidOperationException("Viewer selection has no bundle.");
    }

    public override void _Ready()
    {
        _status = GetNode<Label>("Hud/Margin/Controls/Status");
        _pauseButton = GetNode<Button>("Hud/Margin/Controls/Buttons/Pause");
        _restartButton = GetNode<Button>("Hud/Margin/Controls/Buttons/Restart");
        _pauseButton.Pressed += TogglePause;
        _restartButton.Pressed += Restart;
        try
        {
            ConfigureLaunch();
            string bundle = ProjectSettings.GlobalizePath(BundleDirectory);
            if (string.IsNullOrWhiteSpace(BundleDirectory) || !File.Exists(Path.Combine(bundle, "contract.json")))
                throw new InvalidOperationException("Set Bundle Directory in the Inspector to a complete exported policy bundle.");
            string model = ProjectSettings.GlobalizePath(ModelPath);
            string library = string.IsNullOrWhiteSpace(MujocoLibraryDirectory)
                ? string.Empty : ProjectSettings.GlobalizePath(MujocoLibraryDirectory);
            MjInterop.SetLibraryDirectory(library);
            _bridge = new MjBridge(model);
            _driver = new MjMimicStandDriver(_bridge, model, bundle);
            _trialSeconds = TrialSeconds;
            _startPhase = StartPhaseSeconds;
            if (!float.IsFinite(_trialSeconds) || _trialSeconds < 3
                || !float.IsFinite(_startPhase) || _startPhase < 0
                || _startPhase + _trialSeconds > _driver.ReferenceDuration)
                throw new InvalidOperationException("Trial must last at least three seconds and fit within the reference clip.");
            _proxies = MjProxyBuilder.Build(_bridge, this, new MjModelDefinition(model).Document);
            Restart();
            if (_driver != null)
                GD.Print($"[{TrialName}] Loaded {bundle}; live {_trialSeconds:F1}s trial. R: restart, P: pause.");
        }
        catch (Exception exception)
        {
            Fail(exception);
        }
    }

    public override void _UnhandledKeyInput(InputEvent @event)
    {
        if (@event is not InputEventKey { Pressed: true, Echo: false } key) return;
        if (key.PhysicalKeycode == Key.R) Restart();
        else if (key.PhysicalKeycode == Key.P) TogglePause();
        else return;
        GetViewport().SetInputAsHandled();
    }

    protected void Restart()
    {
        if (_driver == null) return;
        try
        {
            ResetTrial();
            _accumulator = 0;
            _paused = _finished = false;
            UpdateProxies();
            UpdateStatus();
        }
        catch (Exception exception) { Fail(exception); }
    }

    private void TogglePause()
    {
        if (_driver == null || _finished) return;
        _paused = !_paused;
        UpdateStatus();
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_driver == null || _paused || _finished) return;
        try
        {
            _accumulator += delta;
            while (_accumulator >= _driver.ControlTime && !_finished)
            {
                StepTrial();
                _accumulator -= _driver.ControlTime;
                _finished = _driver.Fallen || _driver.Time >= _trialSeconds;
                if (_finished)
                    GD.Print($"[{TrialName}] {(_driver.Fallen ? "Fallen" : "Completed")}: {_driver.Time:F3}s; root error {_driver.TrackingError:F4}m. {TrialDetails}");
            }
            UpdateProxies();
            UpdateStatus();
        }
        catch (Exception exception)
        {
            Fail(exception);
        }
    }

    private void UpdateProxies()
    {
        foreach (var (body, node) in _proxies) node.Transform = _bridge!.BodyTransform(body);
    }

    private void UpdateStatus()
    {
        string state = _driver!.Fallen ? "Fallen — press R to restart"
            : _finished ? "Trial complete — press R to restart" : _paused ? "Paused" : "Standing";
        _status.Text = $"{state}\n{Math.Min(_driver.Time, _trialSeconds):F1} / {_trialSeconds:F1} s · Root error: {_driver.TrackingError * 100:F1} cm" + TrialDetails;
        _pauseButton.Text = _paused ? "Resume (P)" : "Pause (P)";
        _pauseButton.Disabled = _finished;
    }

    private void Fail(Exception exception)
    {
        ReleaseSimulation();
        _status.Text = $"Unable to start or continue {TrialName}.\n" + exception.Message;
        _pauseButton.Disabled = _restartButton.Disabled = true;
        GD.PrintErr($"[{TrialName}] " + exception);
    }

    private void ReleaseSimulation()
    {
        _driver?.Dispose(); _driver = null;
        _bridge?.Dispose(); _bridge = null;
    }

    public override void _ExitTree() => ReleaseSimulation();
}

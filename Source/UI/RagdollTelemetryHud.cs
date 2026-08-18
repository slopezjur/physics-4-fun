using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.UI;

/// <summary>
/// Real-time on-screen telemetry dashboard for active ragdoll physical diagnostics.
/// Displays state, stability metrics, forces, angles, and interactive debug shortcuts.
/// Anchored cleanly to the top-right corner with mouse pass-through enabled (ISP).
/// </summary>
public partial class RagdollTelemetryHud : PanelContainer
{
    [Export] public HumanoidRagdoll? Ragdoll { get; set; }
    [Export] public BalanceController? Balance { get; set; }

    public IBalanceTelemetryProvider? TelemetryProvider => Balance;

    private Label _telemetryLabel = null!;
    private Label _recStatusLabel = null!;
    private ProgressBar _progressBar = null!;

    public override void _Ready()
    {
        MouseFilter = MouseFilterEnum.Ignore;
        SetAnchorsPreset(LayoutPreset.TopRight);
        AnchorLeft = 1.0f;
        AnchorTop = 0.0f;
        AnchorRight = 1.0f;
        AnchorBottom = 0.0f;
        OffsetLeft = -390.0f;
        OffsetTop = 20.0f;
        OffsetRight = -20.0f;
        OffsetBottom = 360.0f;
        GrowHorizontal = GrowDirection.Begin;
        GrowVertical = GrowDirection.End;

        var margin = new MarginContainer();
        margin.MouseFilter = MouseFilterEnum.Ignore;
        margin.AddThemeConstantOverride("margin_left", 14);
        margin.AddThemeConstantOverride("margin_top", 12);
        margin.AddThemeConstantOverride("margin_right", 14);
        margin.AddThemeConstantOverride("margin_bottom", 12);
        AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.MouseFilter = MouseFilterEnum.Ignore;
        margin.AddChild(vbox);

        _telemetryLabel = new Label();
        _telemetryLabel.MouseFilter = MouseFilterEnum.Ignore;
        _telemetryLabel.AddThemeFontSizeOverride("font_size", 13);
        _telemetryLabel.AddThemeColorOverride("font_color", new Color(0.9f, 0.95f, 1.0f));
        vbox.AddChild(_telemetryLabel);

        _recStatusLabel = new Label();
        _recStatusLabel.MouseFilter = MouseFilterEnum.Ignore;
        _recStatusLabel.AddThemeFontSizeOverride("font_size", 13);
        _recStatusLabel.AddThemeColorOverride("font_color", new Color(1.0f, 0.35f, 0.35f));
        _recStatusLabel.Visible = false;
        vbox.AddChild(_recStatusLabel);

        _progressBar = new ProgressBar();
        _progressBar.MouseFilter = MouseFilterEnum.Ignore;
        _progressBar.CustomMinimumSize = new Vector2(0, 16);
        _progressBar.ShowPercentage = true;
        _progressBar.Visible = false;
        vbox.AddChild(_progressBar);

        if (Ragdoll == null)
        {
            Ragdoll = GetTree().Root.FindChild("ActiveRagdoll", true, false) as HumanoidRagdoll;
        }
        if (Balance == null && Ragdoll != null)
        {
            Balance = Ragdoll.Balance;
        }
    }

    public override void _Process(double delta)
    {
        if (Ragdoll == null || !IsInstanceValid(Ragdoll) || _telemetryLabel == null)
        {
            return;
        }

        var pelvis = Ragdoll.Pelvis;
        float pelvisY = pelvis != null && IsInstanceValid(pelvis) ? pelvis.GlobalPosition.Y : 0.0f;
        Vector3 pelvisVel = pelvis != null && IsInstanceValid(pelvis) ? pelvis.LinearVelocity : Vector3.Zero;
        float tiltAngle = TelemetryProvider != null ? TelemetryProvider.CurrentTiltAngleDeg : 0.0f;
        float balanceStrength = (TelemetryProvider?.BalanceStrengthNow ?? 0.0f) * 100.0f;
        float icpEscape = TelemetryProvider?.IcpEscapeDistance ?? 0.0f;
        float pelvisStabTorque = TelemetryProvider?.PelvisStabilizerTorque.Length() ?? 0.0f;
        float muscleStrength = Ragdoll.CurrentMuscleStiffness * 100.0f;

        int fps = (int)Engine.GetFramesPerSecond();
        int physicsFps = Engine.PhysicsTicksPerSecond;

        bool groundedL = TelemetryProvider?.IsGroundedL ?? false;
        bool groundedR = TelemetryProvider?.IsGroundedR ?? false;
        string groundStatus = $"L: {(groundedL ? "ON" : "AIR")} | R: {(groundedR ? "ON" : "AIR")}";
        string stepPhase = TelemetryProvider != null ? TelemetryProvider.CurrentStepPhase.ToString() : "N/A";
        float weightL = (TelemetryProvider?.CurrentWeightShareL ?? 0.5f) * 100.0f;
        float weightR = (TelemetryProvider?.CurrentWeightShareR ?? 0.5f) * 100.0f;

        bool isRecording = Ragdoll.Recorder.IsRecording;
        _recStatusLabel.Visible = isRecording;
        _progressBar.Visible = isRecording;
        if (isRecording)
        {
            float progress = Ragdoll.Recorder.ProgressNormalized;
            _progressBar.Value = progress * 100.0f;
            _recStatusLabel.Text = $"● RECORDING: {Ragdoll.Recorder.ElapsedRecordingTime:F1}s / {Ragdoll.Recorder.MaxDuration:F1}s ({(int)(progress * 100)}%)";
        }

        _telemetryLabel.Text =
            $"--- TELEMETRY METRICS ---\n" +
            $"Engine: Jolt @ {physicsFps} Hz (FPS: {fps})\n" +
            $"State: {Ragdoll.CurrentState.ToString().ToUpper()}\n" +
            $"Step Phase: {stepPhase}\n" +
            $"Weight Transfer: L: {weightL:F0}% | R: {weightR:F0}%\n" +
            $"Grounded Feet: {groundStatus}\n" +
            $"Muscle Strength: {muscleStrength:F0}%\n" +
            $"Pelvis Height: {pelvisY:F3} m (Target: 0.82m)\n" +
            $"Pelvis Speed: {pelvisVel.Length():F2} m/s (Y: {pelvisVel.Y:F2})\n" +
            $"Torso Tilt: {tiltAngle:F1}° (KO Limit: 80°)\n" +
            $"Balance Strength: {balanceStrength:F0}% | ICP Escape: {icpEscape:F2} m\n" +
            $"Pelvis Stabilizer: {pelvisStabTorque:F0} N·m\n\n" +
            $"--- DEBUG CONTROLS ---\n" +
            $"[T] Reset & Record 5s Telemetry\n" +
            $"[1-5] Force State (1:Bal, 2:Stumble, 3:Flail, 4:KO, 5:Rec)\n" +
            $"[G] Toggle Zero-Gravity\n" +
            $"[F] Freeze / Unfreeze Pelvis\n" +
            $"[Space] Chest Impact Kick\n" +
            $"[R] Reset Simulation\n";
    }
}

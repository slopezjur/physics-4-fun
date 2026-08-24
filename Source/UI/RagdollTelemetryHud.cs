using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;
using Physics4Fun.RL;

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

    /// <summary>
    /// Optional: only set in the RL arena scene. Left null (and inert) in TestChamber and any
    /// other scene without an RL bridge, so this HUD stays usable everywhere.
    /// </summary>
    [Export] public RagdollRLBridge? RLBridge { get; set; }

    public IBalanceTelemetryProvider? TelemetryProvider => Balance;

    private Label _telemetryLabel = null!;
    private Label _recStatusLabel = null!;
    private ProgressBar _progressBar = null!;

    /// <summary>
    /// Point size for the telemetry text, and the basis for the panel padding.
    ///
    /// 11 rather than the original 13: the RL block grew several lines (episode, start pose,
    /// curriculum floor, per-episode reward) and the panel started running off the bottom of the
    /// screen during recording, when the extra status line appears. Exported so a scene that shows
    /// more can go smaller still without changing the others - the perturbation arena uses 10.
    /// </summary>
    [Export] public int FontSize { get; set; } = 11;

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
        int pad = Mathf.RoundToInt(FontSize * 0.9f);
        margin.AddThemeConstantOverride("margin_left", pad);
        margin.AddThemeConstantOverride("margin_top", pad);
        margin.AddThemeConstantOverride("margin_right", pad);
        margin.AddThemeConstantOverride("margin_bottom", pad);
        AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.MouseFilter = MouseFilterEnum.Ignore;
        margin.AddChild(vbox);

        _telemetryLabel = new Label();
        _telemetryLabel.MouseFilter = MouseFilterEnum.Ignore;
        _telemetryLabel.AddThemeFontSizeOverride("font_size", FontSize);
        _telemetryLabel.AddThemeColorOverride("font_color", new Color(0.9f, 0.95f, 1.0f));
        vbox.AddChild(_telemetryLabel);

        _recStatusLabel = new Label();
        _recStatusLabel.MouseFilter = MouseFilterEnum.Ignore;
        _recStatusLabel.AddThemeFontSizeOverride("font_size", FontSize);
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
            (Ragdoll.CurrentState == RagdollState.Recovering ? $"Get-Up Phase: {Ragdoll.CurrentGetUpPhase}\n" : string.Empty) +
            (Ragdoll.CurrentState == RagdollState.PushUpDrill ? $"Push-Up Rep: {Ragdoll.DrillCycleNormalized:F2} (0=bottom, 1=lockout)\n" : string.Empty) +
            (Ragdoll.CurrentState == RagdollState.ReinforcementLearning && RLBridge != null
                ? $"--- RL EPISODE ---\n" +
                  $"Episode: {RLBridge.EpisodeCount} | " +
                  $"Elapsed: {RLBridge.EpisodeElapsedSeconds:F1}s / {RLBridge.EffectiveMaxEpisodeSeconds:F1}s\n" +
                  $"Start Pose: {RLBridge.StartPoseT:F2} ({DescribeStartPose(RLBridge.StartPoseT)}) | " +
                  $"Curriculum Floor: {RLBridge.ActiveCurriculumT:F2}\n" +
                  $"Reward (live): {RLBridge.CurrentAccumulatedReward:F2}\n" +
                  $"Last Episode: {RLBridge.LastEpisodeEndReason} | reward={RLBridge.LastEpisodeReward:F2} | {RLBridge.LastEpisodeDurationSeconds:F1}s\n" +
                  $"Policy Active: {(Ragdoll.ReinforcementLearningPolicyActive ? "YES" : "NO (idling)")}\n"
                : string.Empty) +
            $"Step Phase: {stepPhase}\n" +
            $"Weight Transfer: L: {weightL:F0}% | R: {weightR:F0}%\n" +
            $"Grounded Feet: {groundStatus}\n" +
            $"Muscle Strength: {muscleStrength:F0}%\n" +
            $"Pelvis Height: {pelvisY:F3} m (Target: 0.82m)\n" +
            $"Pelvis Speed: {pelvisVel.Length():F2} m/s (Y: {pelvisVel.Y:F2})\n" +
            $"Torso Tilt: {tiltAngle:F1}° (KO Limit: 80°)\n" +
            $"Balance Strength: {balanceStrength:F0}% | ICP Escape: {icpEscape:F2} m\n" +
            $"Pelvis Stabilizer: {pelvisStabTorque:F0} N·m\n\n" +
            (RLBridge == null
                ? $"--- DEBUG CONTROLS ---\n" +
                  $"[T] Reset & Record 5s Telemetry\n" +
                  $"[1-5] Force State (1:Bal, 2:Stumble, 3:Flail, 4:KO, 5:Rec)\n" +
                  $"[G] Toggle Zero-Gravity\n" +
                  $"[F] Freeze / Unfreeze Pelvis\n" +
                  $"[Space] Chest Impact Kick\n" +
                  $"[R] Reset Simulation\n"
                : $"--- CAMERA ---\n" +
                  $"[Right-Drag] Look around\n" +
                  $"[WASD / Q,E] Move / Up,Down\n" +
                  $"[Shift] Move faster\n" +
                  $"[Wheel] Adjust speed\n\n" +
                  $"--- DEBUG CONTROLS ---\n" +
                  $"Ragdoll keys disabled during RL episodes\n");
    }

    /// <summary>
    /// Words for the 0 = prone, 1 = standing start-pose scale, so the HUD reads as a pose rather
    /// than a bare number. Boundaries are descriptive only - nothing branches on them.
    /// </summary>
    private static string DescribeStartPose(float poseT) => poseT switch
    {
        >= 1.0f => "STANDING",
        >= 0.75f => "leaning",
        >= 0.5f => "half-risen",
        >= 0.25f => "low",
        > 0.0f => "near-prone",
        _ => "PRONE",
    };
}

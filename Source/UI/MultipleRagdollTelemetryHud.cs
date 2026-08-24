using Godot;

namespace Physics4Fun.UI;

public partial class MultipleRagdollTelemetryHud : PanelContainer
{
    private Label _telemetryLabel = null!;

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

        var panelStyle = new StyleBoxFlat
        {
            BgColor = new Color(0, 0, 0, 0.75f),
            CornerRadiusTopLeft = 8,
            CornerRadiusTopRight = 8,
            CornerRadiusBottomLeft = 8,
            CornerRadiusBottomRight = 8,
            ContentMarginLeft = FontSize,
            ContentMarginTop = FontSize,
            ContentMarginRight = FontSize,
            ContentMarginBottom = FontSize,
        };
        AddThemeStyleboxOverride("panel", panelStyle);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 8);
        AddChild(vbox);

        _telemetryLabel = new Label
        {
            Text = "Multiple Agents Loaded...",
            LabelSettings = new LabelSettings
            {
                FontSize = FontSize,
                FontColor = Colors.White,
                Font = ThemeDB.FallbackFont,
                LineSpacing = -2
            }
        };
        vbox.AddChild(_telemetryLabel);
    }

    public override void _Process(double delta)
    {
        _telemetryLabel.Text = $"Multi-Agent Scenario\nFPS: {Engine.GetFramesPerSecond()}";
    }
}

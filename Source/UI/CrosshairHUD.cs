using Godot;
using Physics4Fun.Physics;

namespace Physics4Fun.UI;

/// <summary>
/// Dynamic screen-center crosshair & target inspection pointer.
/// Visualizes hovered physics limbs and active grab statuses.
/// </summary>
public partial class CrosshairHUD : Control
{
    [Export] public PhysicsGrabber? Grabber { get; set; }

    private Label _targetLabel = null!;
    private Color _reticleColor = new Color(1, 1, 1, 0.7f);

    public override void _Ready()
    {
        MouseFilter = MouseFilterEnum.Ignore;
        SetAnchorsPreset(LayoutPreset.FullRect);

        _targetLabel = new Label
        {
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            Position = new Vector2(0, 24),
            GrowHorizontal = GrowDirection.Both
        };
        _targetLabel.AddThemeConstantOverride("outline_size", 4);
        _targetLabel.AddThemeColorOverride("font_outline_color", Colors.Black);
        _targetLabel.AddThemeFontSizeOverride("font_size", 14);
        AddChild(_targetLabel);

        if (Grabber == null)
        {
            Grabber = GetTree().CurrentScene.GetNodeOrNull<PhysicsGrabber>("PhysicsGrabber");
        }
    }

    public override void _Process(double delta)
    {
        QueueRedraw();
        UpdateLabelText();
    }

    public override void _Draw()
    {
        Vector2 center = GetViewportRect().Size / 2.0f;
        _targetLabel.Position = new Vector2(center.X - 150, center.Y + 16);
        _targetLabel.Size = new Vector2(300, 30);

        if (Grabber != null && Grabber.IsGrabbing)
        {
            _reticleColor = new Color(1.0f, 0.3f, 0.2f, 0.95f);
            // Draw active grab circle
            DrawArc(center, 12.0f, 0, Mathf.Tau, 32, _reticleColor, 2.5f);
            DrawCircle(center, 3.5f, _reticleColor);
        }
        else if (Grabber != null && Grabber.HoveredBody != null)
        {
            _reticleColor = new Color(0.2f, 1.0f, 0.5f, 0.95f);
            // Draw hover diamond
            DrawRect(new Rect2(center - new Vector2(6, 6), new Vector2(12, 12)), _reticleColor, false, 2.0f);
            DrawCircle(center, 2.5f, _reticleColor);
        }
        else
        {
            _reticleColor = new Color(1.0f, 1.0f, 1.0f, 0.6f);
            // Draw default subtle crosshair
            float size = 8.0f;
            float gap = 3.0f;
            DrawLine(center + new Vector2(gap, 0), center + new Vector2(size + gap, 0), _reticleColor, 1.5f);
            DrawLine(center - new Vector2(gap, 0), center - new Vector2(size + gap, 0), _reticleColor, 1.5f);
            DrawLine(center + new Vector2(0, gap), center + new Vector2(0, size + gap), _reticleColor, 1.5f);
            DrawLine(center - new Vector2(0, gap), center - new Vector2(0, size + gap), _reticleColor, 1.5f);
        }
    }

    private void UpdateLabelText()
    {
        if (Grabber == null)
        {
            _targetLabel.Text = string.Empty;
            return;
        }

        if (Grabber.IsGrabbing && Grabber.GrabbedBody != null)
        {
            _targetLabel.Text = $"[HOLDING: {Grabber.GrabbedBody.Name} ({Grabber.GrabbedBody.Mass:F1}kg)]\n[Scroll: Adjust Distance | Release: Toss]";
            _targetLabel.Modulate = new Color(1.0f, 0.85f, 0.3f);
        }
        else if (Grabber.HoveredBody != null)
        {
            _targetLabel.Text = $"[Target: {Grabber.HoveredBody.Name} ({Grabber.HoveredBody.Mass:F1}kg) - {Grabber.HoveredDistance:F1}m]\n[Middle-Click / G: Grab]";
            _targetLabel.Modulate = new Color(0.3f, 1.0f, 0.6f);
        }
        else
        {
            _targetLabel.Text = string.Empty;
        }
    }
}

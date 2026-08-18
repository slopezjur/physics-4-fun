using Godot;

namespace Physics4Fun.Camera;

/// <summary>
/// Free-look and orbit inspection camera for real-time physics observation.
/// Left-drag / Right-drag to look around, WASDQE to move, Shift to boost, Mouse Wheel to zoom/speed.
/// </summary>
public partial class CameraController : Camera3D
{
    [Export] public float MoveSpeed { get; set; } = 8.0f;
    [Export] public float FastMultiplier { get; set; } = 2.5f;
    [Export] public float MouseSensitivity { get; set; } = 0.003f;

    private float _pitch;
    private float _yaw;
    private bool _isCapturingMouse;

    public override void _Ready()
    {
        Vector3 rot = Rotation;
        _pitch = rot.X;
        _yaw = rot.Y;
    }

    public override void _UnhandledInput(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton)
        {
            if (mouseButton.ButtonIndex == MouseButton.Right)
            {
                _isCapturingMouse = mouseButton.Pressed;
                Input.MouseMode = _isCapturingMouse ? Input.MouseModeEnum.Captured : Input.MouseModeEnum.Visible;
            }
        }
        else if (@event is InputEventMouseMotion mouseMotion && _isCapturingMouse)
        {
            _yaw -= mouseMotion.Relative.X * MouseSensitivity;
            _pitch -= mouseMotion.Relative.Y * MouseSensitivity;
            _pitch = Mathf.Clamp(_pitch, -Mathf.Pi * 0.49f, Mathf.Pi * 0.49f);

            Rotation = new Vector3(_pitch, _yaw, 0.0f);
        }
    }

    public override void _Process(double delta)
    {
        if (!_isCapturingMouse)
        {
            return;
        }

        float dt = (float)delta;
        Vector3 inputDir = Vector3.Zero;

        if (Input.IsKeyPressed(Key.W)) inputDir -= Transform.Basis.Z;
        if (Input.IsKeyPressed(Key.S)) inputDir += Transform.Basis.Z;
        if (Input.IsKeyPressed(Key.A)) inputDir -= Transform.Basis.X;
        if (Input.IsKeyPressed(Key.D)) inputDir += Transform.Basis.X;
        if (Input.IsKeyPressed(Key.E) || Input.IsKeyPressed(Key.Space)) inputDir += Vector3.Up;
        if (Input.IsKeyPressed(Key.Q) || Input.IsKeyPressed(Key.C)) inputDir -= Vector3.Up;

        if (inputDir.LengthSquared() > 0.001f)
        {
            float speed = MoveSpeed;
            if (Input.IsKeyPressed(Key.Shift))
            {
                speed *= FastMultiplier;
            }

            GlobalPosition += inputDir.Normalized() * speed * dt;
        }
    }
}

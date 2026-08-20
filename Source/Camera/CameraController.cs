using Godot;

namespace Physics4Fun.Camera;

/// <summary>
/// Free-look inspection camera for real-time physics observation.
/// Right-drag to look around, WASD + Q/E to move, Shift to boost, mouse wheel to change speed.
/// </summary>
public partial class CameraController : Camera3D
{
    /// <summary>Starting movement speed (m/s of real time). The wheel adjusts it from here.</summary>
    [Export] public float MoveSpeed { get; set; } = 8.0f;

    [Export] public float FastMultiplier { get; set; } = 2.5f;
    [Export] public float MouseSensitivity { get; set; } = 0.003f;

    /// <summary>Multiplier applied per wheel notch. Geometric so the same notch feels equal at any speed.</summary>
    private const float SpeedStepFactor = 1.15f;

    private const float MinMoveSpeed = 0.25f;
    private const float MaxMoveSpeed = 200.0f;

    private float _pitch;
    private float _yaw;
    private bool _isCapturingMouse;

    /// <summary>Live speed, seeded from MoveSpeed and then owned by the wheel.</summary>
    private float _currentMoveSpeed;

    public override void _Ready()
    {
        Vector3 rot = Rotation;
        _pitch = rot.X;
        _yaw = rot.Y;
        _currentMoveSpeed = MoveSpeed;
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
            else if (_isCapturingMouse
                     && mouseButton.Pressed
                     && (mouseButton.ButtonIndex == MouseButton.WheelUp
                         || mouseButton.ButtonIndex == MouseButton.WheelDown))
            {
                // Gated on _isCapturingMouse so this cannot collide with PhysicsGrabber, which
                // binds the same wheel to grab distance while IsGrabbing and does not mark the
                // event handled. "Right button held" means the user is flying, not grabbing.
                //
                // Gated on Pressed because Godot emits a press AND a release per wheel notch;
                // acting on both would double every adjustment.
                float step = mouseButton.ButtonIndex == MouseButton.WheelUp
                    ? SpeedStepFactor
                    : 1.0f / SpeedStepFactor;
                _currentMoveSpeed = Mathf.Clamp(_currentMoveSpeed * step, MinMoveSpeed, MaxMoveSpeed);
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

        // Divide out Engine.TimeScale, which godot_rl_agents' sync.gd sets to --speedup (8 during
        // training). Godot scales the delta handed to _Process by it, so without this the camera
        // flew 8x too fast in the --viz window while behaving normally in the standalone arena -
        // same scene, same MoveSpeed, entirely down to the simulation clock. Camera navigation is
        // UI, not simulation, so it belongs on real time regardless of how fast physics is running.
        float dt = (float)delta / Mathf.Max(0.0001f, (float)Engine.TimeScale);
        Vector3 inputDir = Vector3.Zero;

        if (Input.IsKeyPressed(Key.W)) inputDir -= Transform.Basis.Z;
        if (Input.IsKeyPressed(Key.S)) inputDir += Transform.Basis.Z;
        if (Input.IsKeyPressed(Key.A)) inputDir -= Transform.Basis.X;
        if (Input.IsKeyPressed(Key.D)) inputDir += Transform.Basis.X;
        if (Input.IsKeyPressed(Key.E) || Input.IsKeyPressed(Key.Space)) inputDir += Vector3.Up;
        if (Input.IsKeyPressed(Key.Q) || Input.IsKeyPressed(Key.C)) inputDir -= Vector3.Up;

        if (inputDir.LengthSquared() > 0.001f)
        {
            float speed = _currentMoveSpeed;
            if (Input.IsKeyPressed(Key.Shift))
            {
                speed *= FastMultiplier;
            }

            GlobalPosition += inputDir.Normalized() * speed * dt;
        }
    }
}

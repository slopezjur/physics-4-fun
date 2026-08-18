using Godot;

namespace Physics4Fun.Physics;

/// <summary>
/// Motorized kinematic rotating obstacle (sweeper / turntable / spinner).
/// Uses AnimatableBody3D to impart accurate physics contact velocities in Jolt.
/// </summary>
public partial class RotatingObstacle : AnimatableBody3D
{
    [Export] public Vector3 RotationAxis { get; set; } = Vector3.Up;
    [Export] public float RotationSpeedRadPerSec { get; set; } = 2.0f;

    public override void _PhysicsProcess(double delta)
    {
        float angle = RotationSpeedRadPerSec * (float)delta;
        Rotate(RotationAxis.Normalized(), angle);
    }
}

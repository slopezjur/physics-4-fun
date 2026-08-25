using Godot;

namespace Physics4Fun.Physics;

/// <summary>
/// Shoots physical cannonballs into the scene. Supports continuous rapid-fire while holding Left Mouse Button.
/// </summary>
public partial class ProjectileShooter : Node
{
    [Export] public float FireRate { get; set; } = 7.0f; // Shots per second
    [Export] public float LaunchForce { get; set; } = 35.0f;
    [Export] public float BallMass { get; set; } = 5.0f;
    [Export] public float BallRadius { get; set; } = 0.25f;

    private Camera3D _camera = null!;
    private bool _isFiring;
    private float _timeUntilNextShot;

    public override void _Ready()
    {
        _camera = GetViewport().GetCamera3D();
    }

    public override void _UnhandledInput(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton && mouseButton.ButtonIndex == MouseButton.Left)
        {
            _isFiring = mouseButton.Pressed;

            // Fire on the press itself rather than waiting for the next _Process, so a single
            // click feels immediate instead of picking up up to one frame of latency.
            TryFire();
        }
    }

    public override void _Process(double delta)
    {
        if (_timeUntilNextShot > 0.0f)
        {
            _timeUntilNextShot -= (float)delta;
        }

        TryFire();
    }

    /// <summary>
    /// Fires if the button is held and the cooldown has expired.
    ///
    /// Both call sites ran an identical copy of this: press-to-fire in _UnhandledInput and
    /// hold-to-repeat in _Process. Two copies of a guard plus a cooldown reset is how the two
    /// quietly drift apart - change the rate limit in one and the click path keeps the old one.
    /// </summary>
    private void TryFire()
    {
        if (!_isFiring || _timeUntilNextShot > 0.0f)
        {
            return;
        }

        SpawnAndShootProjectile();
        _timeUntilNextShot = 1.0f / Mathf.Max(0.1f, FireRate);
    }

    private void SpawnAndShootProjectile()
    {
        if (_camera == null)
        {
            _camera = GetViewport().GetCamera3D();
            if (_camera == null) return;
        }

        var body = new RigidBody3D
        {
            Mass = BallMass,
            ContinuousCd = true,
            GlobalPosition = _camera.GlobalPosition + (-_camera.GlobalTransform.Basis.Z * 0.8f)
        };

        var collisionShape = new CollisionShape3D();
        var sphereShape = new SphereShape3D { Radius = BallRadius };
        collisionShape.Shape = sphereShape;
        body.AddChild(collisionShape);

        var meshInstance = new MeshInstance3D();
        var sphereMesh = new SphereMesh { Radius = BallRadius, Height = BallRadius * 2.0f };
        var material = new StandardMaterial3D
        {
            AlbedoColor = new Color(0.9f, 0.25f, 0.1f),
            Roughness = 0.3f,
            Metallic = 0.6f
        };
        sphereMesh.Material = material;
        meshInstance.Mesh = sphereMesh;
        body.AddChild(meshInstance);

        GetTree().CurrentScene.AddChild(body);

        Vector3 impulseDir = -_camera.GlobalTransform.Basis.Z;
        body.ApplyCentralImpulse(impulseDir * LaunchForce * BallMass);

        // Auto-cleanup after 6 seconds to maintain performance
        var timer = GetTree().CreateTimer(6.0);
        timer.Timeout += () =>
        {
            if (IsInstanceValid(body))
            {
                body.QueueFree();
            }
        };
    }
}

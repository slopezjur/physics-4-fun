using Godot;
using Physics4Fun.Ragdoll;

namespace Physics4Fun.Physics;

/// <summary>
/// Interactive physics grabber (Gravity Gun / Drag tool).
/// Allows grabbing any RigidBody3D / ActiveBone with Middle-Click or 'G' key,
/// dragging them through 3D space with spring-damper dynamics, and tossing them.
/// Displays a 3D visual tether line while dragging.
/// </summary>
public partial class PhysicsGrabber : Node
{
    [Export] public float MaxGrabDistance { get; set; } = 30.0f;
    [Export] public float SpringStiffness { get; set; } = 400.0f;
    [Export] public float SpringDamping { get; set; } = 25.0f;
    [Export] public float ThrowForceMultiplier { get; set; } = 2.0f;

    private Camera3D _camera = null!;
    private RigidBody3D? _grabbedBody;
    private Vector3 _localGrabPoint;
    private float _currentGrabDistance;
    private Vector3 _lastTargetPosition;
    private Vector3 _targetVelocity;

    // Visual 3D Tether Line
    private MeshInstance3D _tetherMeshInstance = null!;
    private ImmediateMesh _immediateMesh = null!;
    private StandardMaterial3D _tetherMaterial = null!;

    public bool IsGrabbing => _grabbedBody != null && IsInstanceValid(_grabbedBody);
    public RigidBody3D? GrabbedBody => _grabbedBody;
    public RigidBody3D? HoveredBody { get; private set; }
    public Vector3 HoveredPoint { get; private set; }
    public float HoveredDistance { get; private set; }

    public override void _Ready()
    {
        _camera = GetViewport().GetCamera3D();

        // Setup 3D visual tether line
        _immediateMesh = new ImmediateMesh();
        _tetherMaterial = new StandardMaterial3D
        {
            ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded,
            AlbedoColor = new Color(0.2f, 0.95f, 1.0f, 0.9f),
            Transparency = BaseMaterial3D.TransparencyEnum.Alpha
        };

        _tetherMeshInstance = new MeshInstance3D
        {
            Mesh = _immediateMesh,
            MaterialOverride = _tetherMaterial,
            CastShadow = GeometryInstance3D.ShadowCastingSetting.Off
        };
        AddChild(_tetherMeshInstance);
    }

    public override void _UnhandledInput(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton)
        {
            if (mouseButton.ButtonIndex == MouseButton.Middle)
            {
                if (mouseButton.Pressed)
                {
                    TryGrab();
                }
                else
                {
                    ReleaseGrab();
                }
            }
            else if (mouseButton.ButtonIndex == MouseButton.WheelUp && IsGrabbing)
            {
                _currentGrabDistance = Mathf.Clamp(_currentGrabDistance + 0.5f, 1.5f, MaxGrabDistance);
            }
            else if (mouseButton.ButtonIndex == MouseButton.WheelDown && IsGrabbing)
            {
                _currentGrabDistance = Mathf.Clamp(_currentGrabDistance - 0.5f, 1.5f, MaxGrabDistance);
            }
        }
        else if (@event is InputEventKey keyEvent && keyEvent.Keycode == Key.E && !keyEvent.Echo)
        {
            if (keyEvent.Pressed)
            {
                TryGrab();
            }
            else
            {
                ReleaseGrab();
            }
        }
    }

    public override void _Process(double delta)
    {
        UpdateHoverRaycast();
        UpdateTetherVisual();
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!IsGrabbing || _grabbedBody == null)
        {
            return;
        }

        if (_camera == null)
        {
            _camera = GetViewport().GetCamera3D();
            if (_camera == null) return;
        }

        float dt = (float)delta;
        Vector3 rayDir = -_camera.GlobalTransform.Basis.Z;
        Vector3 targetPos = _camera.GlobalPosition + (rayDir * _currentGrabDistance);

        _targetVelocity = (targetPos - _lastTargetPosition) / Mathf.Max(dt, 0.001f);
        _lastTargetPosition = targetPos;

        // Calculate world position of the grabbed point on the body
        Vector3 worldGrabPoint = _grabbedBody.GlobalTransform * _localGrabPoint;
        Vector3 displacement = targetPos - worldGrabPoint;

        // Point velocity on the rigid body
        Vector3 r = worldGrabPoint - _grabbedBody.GlobalPosition;
        Vector3 pointVelocity = _grabbedBody.LinearVelocity + _grabbedBody.AngularVelocity.Cross(r);

        Vector3 force = (displacement * SpringStiffness) - (pointVelocity * SpringDamping);

        _grabbedBody.ApplyForce(force, r);
    }

    private void UpdateHoverRaycast()
    {
        if (IsGrabbing)
        {
            HoveredBody = _grabbedBody;
            return;
        }

        if (_camera == null)
        {
            _camera = GetViewport().GetCamera3D();
            if (_camera == null) return;
        }

        var spaceState = _camera.GetWorld3D().DirectSpaceState;
        Vector3 rayOrigin = _camera.GlobalPosition;
        Vector3 rayDir = -_camera.GlobalTransform.Basis.Z;
        Vector3 rayEnd = rayOrigin + (rayDir * MaxGrabDistance);

        var query = PhysicsRayQueryParameters3D.Create(rayOrigin, rayEnd);
        query.CollideWithBodies = true;
        query.CollideWithAreas = false;
        query.CollisionMask = uint.MaxValue; // Hit all layers

        var result = spaceState.IntersectRay(query);
        if (result.Count > 0 && result["collider"].Obj is RigidBody3D hitBody)
        {
            HoveredBody = hitBody;
            HoveredPoint = (Vector3)result["position"];
            HoveredDistance = rayOrigin.DistanceTo(HoveredPoint);
        }
        else
        {
            HoveredBody = null;
            HoveredDistance = 0.0f;
        }
    }

    private void UpdateTetherVisual()
    {
        _immediateMesh.ClearSurfaces();

        if (!IsGrabbing || _grabbedBody == null || _camera == null)
        {
            return;
        }

        Vector3 rayDir = -_camera.GlobalTransform.Basis.Z;
        Vector3 targetPos = _camera.GlobalPosition + (rayDir * _currentGrabDistance);
        Vector3 worldGrabPoint = _grabbedBody.GlobalTransform * _localGrabPoint;

        // Draw line between cursor target point and grabbed body point
        _immediateMesh.SurfaceBegin(Mesh.PrimitiveType.Lines);
        _immediateMesh.SurfaceAddVertex(targetPos);
        _immediateMesh.SurfaceAddVertex(worldGrabPoint);
        _immediateMesh.SurfaceEnd();
    }

    private void TryGrab()
    {
        if (_camera == null)
        {
            _camera = GetViewport().GetCamera3D();
            if (_camera == null) return;
        }

        var spaceState = _camera.GetWorld3D().DirectSpaceState;
        Vector3 rayOrigin = _camera.GlobalPosition;
        Vector3 rayDir = -_camera.GlobalTransform.Basis.Z;
        Vector3 rayEnd = rayOrigin + (rayDir * MaxGrabDistance);

        var query = PhysicsRayQueryParameters3D.Create(rayOrigin, rayEnd);
        query.CollideWithBodies = true;
        query.CollideWithAreas = false;
        query.CollisionMask = uint.MaxValue; // Hit all layers

        var result = spaceState.IntersectRay(query);
        if (result.Count > 0 && result["collider"].Obj is RigidBody3D hitBody)
        {
            _grabbedBody = hitBody;
            Vector3 hitPos = (Vector3)result["position"];
            _localGrabPoint = _grabbedBody.GlobalTransform.AffineInverse() * hitPos;
            _currentGrabDistance = rayOrigin.DistanceTo(hitPos);
            _lastTargetPosition = hitPos;

            // If grabbed body belongs to a ragdoll, trigger flailing reaction
            if (_grabbedBody.GetParent() is HumanoidRagdoll ragdoll)
            {
                ragdoll.SetState(RagdollState.Flailing);
            }

            GD.Print($"[PhysicsGrabber] Grabbed {_grabbedBody.Name} ({_grabbedBody.Mass:F1}kg)");
        }
    }

    private void ReleaseGrab()
    {
        if (!IsGrabbing || _grabbedBody == null)
        {
            return;
        }

        Vector3 throwImpulse = _targetVelocity * _grabbedBody.Mass * ThrowForceMultiplier;
        if (throwImpulse.LengthSquared() > 1.0f)
        {
            _grabbedBody.ApplyCentralImpulse(throwImpulse);
        }

        GD.Print($"[PhysicsGrabber] Released {_grabbedBody.Name}");
        _grabbedBody = null;
        _immediateMesh.ClearSurfaces();
    }
}

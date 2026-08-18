using Godot;
using Physics4Fun.Core.Math;

namespace Physics4Fun.Physics;

/// <summary>
/// Controls an active physics body / joint using 3D Angular PID torque calculations.
/// Demonstrates balance recovery, target orientation tracking, and perturbation resistance.
/// </summary>
public partial class ActiveJointController : RigidBody3D
{
    [ExportGroup("PID Gains")]
    [Export] public float ProportionalGain { get; set; } = 400.0f;
    [Export] public float DerivativeGain { get; set; } = 40.0f;
    [Export] public float IntegralGain { get; set; } = 0.0f;
    [Export] public float MaxTorque { get; set; } = 3000.0f;
    [Export] public float MaxIntegral { get; set; } = 50.0f;

    [ExportGroup("Target Orientation")]
    [Export] public Vector3 TargetEulerDegrees { get; set; } = Vector3.Zero;

    [ExportGroup("Perturbation Testing")]
    [Export] public float DisturbanceForceMagnitude { get; set; } = 40.0f;

    private PidController3D _pid = null!;
    private Transform3D _initialTransform;
    private Vector3 _appliedTorque = Vector3.Zero;
    private Quaternion _targetQuaternion = Quaternion.Identity;

    public override void _Ready()
    {
        _initialTransform = GlobalTransform;
        _pid = new PidController3D(ProportionalGain, DerivativeGain, IntegralGain, MaxTorque, MaxIntegral);
        UpdateTargetRotation();
    }

    public override void _Process(double delta)
    {
        // Keyboard controls to test disturbance and target angle variation
        if (Input.IsActionJustPressed("ui_accept"))
        {
            // Apply horizontal lateral kick at top of the body
            Vector3 randomDir = new Vector3(
                (float)GD.RandRange(-1.0, 1.0),
                0.2f,
                (float)GD.RandRange(-1.0, 1.0)
            ).Normalized();

            Vector3 offset = GlobalTransform.Basis.Y * 1.0f;
            ApplyImpulse(randomDir * DisturbanceForceMagnitude, offset);
            GD.Print($"[ActiveJoint] Applied impulse kick: {randomDir * DisturbanceForceMagnitude}");
        }

        if (Input.IsKeyPressed(Key.R))
        {
            ResetBody();
        }

        // Dynamic tilt adjustment via arrow keys
        float tiltSpeed = 45.0f * (float)delta;
        Vector3 euler = TargetEulerDegrees;
        if (Input.IsKeyPressed(Key.Up)) euler.X -= tiltSpeed;
        if (Input.IsKeyPressed(Key.Down)) euler.X += tiltSpeed;
        if (Input.IsKeyPressed(Key.Left)) euler.Z += tiltSpeed;
        if (Input.IsKeyPressed(Key.Right)) euler.Z -= tiltSpeed;

        if (euler != TargetEulerDegrees)
        {
            TargetEulerDegrees = euler;
            UpdateTargetRotation();
        }
    }

    public override void _IntegrateForces(PhysicsDirectBodyState3D state)
    {
        // Sync PID parameters with Inspector changes in real-time
        _pid.ProportionalGain = ProportionalGain;
        _pid.DerivativeGain = DerivativeGain;
        _pid.IntegralGain = IntegralGain;
        _pid.MaxTorque = MaxTorque;
        _pid.MaxIntegral = MaxIntegral;

        Quaternion currentRot = state.Transform.Basis.GetRotationQuaternion();
        Vector3 angularVelocity = state.AngularVelocity;

        _appliedTorque = _pid.Update(
            currentRot,
            _targetQuaternion,
            angularVelocity,
            state.Step
        );

        state.ApplyTorque(_appliedTorque);
    }

    private void UpdateTargetRotation()
    {
        Vector3 rad = new Vector3(
            Mathf.DegToRad(TargetEulerDegrees.X),
            Mathf.DegToRad(TargetEulerDegrees.Y),
            Mathf.DegToRad(TargetEulerDegrees.Z)
        );
        _targetQuaternion = Basis.FromEuler(rad).GetRotationQuaternion();
    }

    private void ResetBody()
    {
        GlobalTransform = _initialTransform;
        LinearVelocity = Vector3.Zero;
        AngularVelocity = Vector3.Zero;
        _pid.Reset();
        GD.Print("[ActiveJoint] Reset state.");
    }
}

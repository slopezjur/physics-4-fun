using Godot;
using System;

public partial class AutomatedTest : Node
{
    // Push scenario: forward impulse on the chest mid-run; the ragdoll must recover.
    private static readonly Vector3 PushImpulse = new(0.0f, 0.0f, -12.0f); // N·s, -Z (forward)
    private const double PushTime = 3.0;
    private const double TestDuration = 10.1;

    private Node3D? _ragdoll;
    private RigidBody3D? _pelvis;
    private RigidBody3D? _chest;
    private double _time;
    private bool _failed;
    private bool _pushed;

    public override void _Ready()
    {
        PackedScene scene = GD.Load<PackedScene>("res://Scenes/TestChamber.tscn");
        Node root = scene.Instantiate();
        AddChild(root);

        _ragdoll = root.GetNode<Node3D>("ActiveRagdoll");
        _pelvis = _ragdoll.GetNode<RigidBody3D>("Pelvis");
        _chest = _ragdoll.GetNode<RigidBody3D>("Chest");

        if (_ragdoll is Physics4Fun.Ragdoll.HumanoidRagdoll humanoid)
        {
            humanoid.StartTelemetryRecording();
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        _time += delta;

        if (!_pushed && _time >= PushTime && _chest != null)
        {
            _pushed = true;
            _chest.ApplyCentralImpulse(PushImpulse);
            GD.Print($"PUSH at {_time:F2}s: {PushImpulse.Length():F0} N·s forward on Chest");
        }

        if (_pelvis != null)
        {
            if (_pelvis.GlobalPosition.Y < 0.3f && !_failed)
            {
                GD.Print($"FAILED at {_time:F2}s: Pelvis fell below 0.3m ({_pelvis.GlobalPosition.Y:F3}m)!");
                _failed = true;
            }
        }

        if (_time >= TestDuration)
        {
            if (!_failed)
            {
                GD.Print($"SUCCESS: Character stood for {TestDuration - 0.1:F1}s including a {PushImpulse.Length():F0} N·s push!");
            }
            GetTree().Quit(_failed ? 1 : 0);
        }
    }
}

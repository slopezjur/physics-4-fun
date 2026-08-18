using Godot;
using System;

public partial class AutomatedTest : Node
{
    private Node3D? _ragdoll;
    private RigidBody3D? _pelvis;
    private double _time;
    private bool _failed;

    public override void _Ready()
    {
        PackedScene scene = GD.Load<PackedScene>("res://Scenes/TestChamber.tscn");
        Node root = scene.Instantiate();
        AddChild(root);

        _ragdoll = root.GetNode<Node3D>("ActiveRagdoll");
        _pelvis = _ragdoll.GetNode<RigidBody3D>("Pelvis");

        if (_ragdoll is Physics4Fun.Ragdoll.HumanoidRagdoll humanoid)
        {
            humanoid.StartTelemetryRecording();
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        _time += delta;
        if (_pelvis != null)
        {
            if (_pelvis.GlobalPosition.Y < 0.3f && !_failed)
            {
                GD.Print($"FAILED at {_time:F2}s: Pelvis fell below 0.3m ({_pelvis.GlobalPosition.Y:F3}m)!");
                _failed = true;
            }
        }

        if (_time >= 5.05)
        {
            if (!_failed) GD.Print("SUCCESS: Character stood for 5.0 seconds!");
            GetTree().Quit(_failed ? 1 : 0);
        }
    }
}

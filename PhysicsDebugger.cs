using Godot;
using System;
public partial class PhysicsDebugger : Node
{
    public override void _Process(double delta)
    {
        GetTree().Quit();
    }
}

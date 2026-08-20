using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Handles debug keyboard input to control the ragdoll states and simulation variables.
/// Separated from core physics logic to adhere to the Single Responsibility Principle.
/// </summary>
public partial class RagdollDebugInput : Node
{
    [Export] public HumanoidRagdoll? Ragdoll { get; set; }

    private bool _zeroGravity = false;

    public override void _Input(InputEvent @event)
    {
        if (Ragdoll == null || !IsInstanceValid(Ragdoll))
        {
            return;
        }

        // The RL state is externally driven (see RagdollRLBridge) - a stray debug keypress must
        // not be able to yank CurrentState out from under an active episode.
        if (Ragdoll.CurrentState == RagdollState.ReinforcementLearning)
        {
            return;
        }

        if (@event is InputEventKey keyEvent && keyEvent.Pressed && !keyEvent.Echo)
        {
            switch (keyEvent.Keycode)
            {
                case Key.Key1:
                    Ragdoll.SetState(RagdollState.Balanced);
                    break;
                case Key.Key2:
                    Ragdoll.SetState(RagdollState.Stumbling);
                    break;
                case Key.Key3:
                    Ragdoll.SetState(RagdollState.Flailing);
                    break;
                case Key.Key4:
                    Ragdoll.SetState(RagdollState.KnockedOut);
                    break;
                case Key.Key5:
                    Ragdoll.SetState(RagdollState.Recovering);
                    break;
                case Key.Key6:
                    Ragdoll.DropToProne();
                    break;
                case Key.Key7:
                    Ragdoll.StartPushUpDrill();
                    break;
                case Key.G:
                    ToggleZeroGravity();
                    break;
                case Key.F:
                    ToggleFreezePelvis();
                    break;
                case Key.Space:
                    Ragdoll.ApplyImpulseToChest(new Vector3(0, 10, -35));
                    break;
                case Key.R:
                    Ragdoll.ResetRagdoll();
                    break;
            }
        }
    }

    private void ToggleZeroGravity()
    {
        _zeroGravity = !_zeroGravity;
        float gravityScale = _zeroGravity ? 0.0f : 1.0f;
        foreach (var bone in Ragdoll!.GetBones())
        {
            if (IsInstanceValid(bone))
            {
                bone.GravityScale = gravityScale;
            }
        }
        GD.Print($"[DebugInput] Zero-Gravity set to: {_zeroGravity}");
    }

    private void ToggleFreezePelvis()
    {
        if (Ragdoll?.Pelvis != null && IsInstanceValid(Ragdoll.Pelvis))
        {
            Ragdoll.Pelvis.Freeze = !Ragdoll.Pelvis.Freeze;
            GD.Print($"[DebugInput] Pelvis Freeze set to: {Ragdoll.Pelvis.Freeze}");
        }
    }
}

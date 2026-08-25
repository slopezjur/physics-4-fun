using Godot;
using Physics4Fun.Ragdoll;

namespace Physics4Fun.RL;

/// <summary>
/// Root of one `*Agent.tscn` - the smallest unit that can be instantiated N times into an arena or
/// training scene: one body, one bridge, one AIController.
///
/// It exists to give <see cref="RagdollSpawner"/> and <see cref="PolicyAutoLoader"/> a TYPED handle
/// on an agent's parts. Before it, the spawner reached into freshly instantiated scenes with
/// <c>GetNode&lt;HumanoidRagdoll&gt;("ActiveRagdoll")</c> and
/// <c>GetNode&lt;RagdollRLBridge&gt;("RLBridge")</c> - the agent scene's internal layout duplicated
/// as string literals in C#, throwing at runtime if a node were ever renamed, and silently wrong if
/// a task's agent scene arranged itself differently. Each agent scene now declares its own
/// internals once, in the scene file, and every consumer binds to this contract instead.
///
/// Deliberately holds no behaviour. It is a manifest, not a controller - the bridge owns the
/// episode lifecycle, and adding logic here would give the RL track a second place to look for it.
/// </summary>
public partial class RlAgent : Node3D
{
    /// <summary>The body this agent drives.</summary>
    [Export] public HumanoidRagdoll? Ragdoll { get; set; }

    /// <summary>The godot_rl_agents seam and episode lifecycle owner for this agent.</summary>
    [Export] public RagdollRLBridge? Bridge { get; set; }

    /// <summary>
    /// Both references are required, and a missing one is a scene wiring error rather than a
    /// recoverable condition - an agent with no bridge is invisible to Sync's AGENT group scan and
    /// simply contributes nothing, which looks like a throughput shortfall rather than a broken
    /// scene. Reported here so it names itself on the first frame instead.
    /// </summary>
    public override void _Ready()
    {
        if (Ragdoll == null || !IsInstanceValid(Ragdoll))
        {
            GD.PushError($"[RlAgent] '{Name}' has no Ragdoll assigned - this agent will not act.");
        }

        if (Bridge == null || !IsInstanceValid(Bridge))
        {
            GD.PushError($"[RlAgent] '{Name}' has no Bridge assigned - this agent will not train.");
        }
    }
}

using System.Collections.Generic;
using Godot;
using Physics4Fun.UI;

namespace Physics4Fun.RL;

/// <summary>
/// Instantiates N copies of one `*Agent.tscn` into a scene and lays them out along X.
///
/// This is the node that made `--dummies=N` possible, and it is worth being precise about WHY that
/// mattered, because the obvious reading is wrong. It is not about running more bodies for their
/// own sake - it is that Godot process startup, socket handshake and render/physics server
/// overhead are all per-PROCESS, while the useful work is per-BODY. Sharing one process across 64
/// bodies moved measured throughput from ~2,258 steps/s at 32 processes x 1 body to ~4,310 at 16 x
/// 64, and the resulting sample diversity per PPO update is what carried stand_v28 from
/// task_stand/success 0.0 to 0.9986. See the throughput table in docs/RL-DESIGN-NOTES.md.
///
/// Agents are built from an <see cref="RlAgent"/> scene rather than hand-copied into the scene
/// file. The two `RagdollMultipleStand*.tscn` scenes this replaced carried FORTY duplicated
/// Agent_N blocks each, ~640 lines apiece, which is why changing one agent property meant forty
/// edits or none.
///
/// <para><b>Why _EnterTree and not _Ready.</b> Two consumers read the agent set before this node's
/// own _Ready would run: godot_rl_agents' <c>sync.gd</c> builds its agent list in <c>_ready</c>
/// from <c>get_tree().get_nodes_in_group("AGENT")</c>, and <see cref="PolicyAutoLoader"/> reads
/// <see cref="Agents"/> in its own <c>_Ready</c> on the scene ROOT, which Godot runs after every
/// descendant. Spawning in <c>_EnterTree</c> puts the agents in the tree before either. This
/// ordering is load-bearing; moving the spawn to <c>_Ready</c> makes Sync see zero agents and the
/// process handshake with an empty observation space.</para>
/// </summary>
public partial class RagdollSpawner : Node3D
{
    /// <summary>The `*Agent.tscn` to instantiate. Its root must carry an <see cref="RlAgent"/>.</summary>
    [Export] public PackedScene? AgentScene { get; set; }

    /// <summary>
    /// Agent count when the process was not launched by the trainer - opening the scene in the
    /// editor, or pressing F5. The trainer overrides it with <c>--dummies=N</c>.
    /// </summary>
    [Export] public int DefaultDummies { get; set; } = 1;

    /// <summary>Metres between adjacent agents along X. Must clear the widest fall arc.</summary>
    [Export] public float Spacing { get; set; } = 2.0f;

    /// <summary>Floor width for a single agent; each additional agent adds <see cref="Spacing"/>.</summary>
    [Export] public float BaseFloorWidth { get; set; } = 6.0f;

    /// <summary>
    /// Floor depth, held constant regardless of agent count because agents spread along X only.
    ///
    /// Task-dependent and NOT cosmetic: Walk needs a runway. WalkTermination has no out-of-bounds
    /// condition, so a dummy that walks off the edge is scored as having fallen, and the resulting
    /// metric looks exactly like a policy regression.
    /// </summary>
    [Export] public float BaseFloorDepth { get; set; } = 6.0f;

    /// <summary>Optional. Resized to fit the spawned row; left alone when unset.</summary>
    [Export] public Node3D? Floor { get; set; }

    /// <summary>Optional. Bound to the first agent so a visible run has live telemetry.</summary>
    [Export] public RagdollTelemetryHud? TelemetryHud { get; set; }

    /// <summary>
    /// The spawned agents, in layout order, valid from this node's _EnterTree onward.
    ///
    /// Exposed as a read-only list because consumers PULL from it - see the _EnterTree note above.
    /// The previous version pushed instead: it did <c>GetParent() as PolicyAutoLoader</c> and wrote
    /// into that parent's field, so a child mutated its parent, the dependency pointed the wrong
    /// way, and it only worked while the spawner happened to be a direct child of the autoloader.
    /// </summary>
    public IReadOnlyList<RlAgent> Agents => _agents;

    private readonly List<RlAgent> _agents = new();

    public override void _EnterTree()
    {
        if (AgentScene == null)
        {
            GD.PushError("[RagdollSpawner] No AgentScene assigned - nothing to spawn.");
            return;
        }

        int dummies = ResolveDummyCount();
        GD.Print($"[RagdollSpawner] Spawning {dummies} agent(s) from {AgentScene.ResourcePath}.");

        SpawnAgents(dummies);
        ResizeFloor(dummies);
        BindTelemetry();
    }

    /// <summary>
    /// Agent count from <c>--dummies=N</c>, falling back to <see cref="DefaultDummies"/>.
    ///
    /// A non-positive or unparseable value is reported rather than silently clamped: it can only
    /// come from the launcher, and a run that quietly trains on one body when it was asked for 64
    /// is a throughput mystery that costs more to diagnose than the warning costs to read.
    /// </summary>
    private int ResolveDummyCount()
    {
        int? requested = CmdlineArgs.ReadInt("dummies");
        if (requested == null)
        {
            return Mathf.Max(1, DefaultDummies);
        }

        if (requested.Value < 1)
        {
            GD.PushWarning(
                $"[RagdollSpawner] --dummies={requested.Value} is not a positive count; "
                + $"falling back to {Mathf.Max(1, DefaultDummies)}.");
            return Mathf.Max(1, DefaultDummies);
        }

        return requested.Value;
    }

    /// <summary>Instantiates the row, centred on this node's origin.</summary>
    private void SpawnAgents(int dummies)
    {
        float startX = -(dummies - 1) * Spacing / 2.0f;

        for (int i = 0; i < dummies; i++)
        {
            var agent = AgentScene!.Instantiate<Node3D>();
            agent.Name = $"Agent_{i + 1}";
            agent.Position = new Vector3(startX + (i * Spacing), 0.0f, 0.0f);
            AddChild(agent);

            if (agent is RlAgent typed)
            {
                _agents.Add(typed);
            }
            else
            {
                GD.PushError(
                    $"[RagdollSpawner] {AgentScene.ResourcePath} root is not an RlAgent. "
                    + "PolicyAutoLoader and the telemetry HUD cannot bind to it.");
            }
        }
    }

    /// <summary>
    /// Widens the floor to cover the spawned row.
    ///
    /// The shape and mesh are DUPLICATED before being written to. Godot sub-resources are shared:
    /// assigning into the BoxShape3D the scene loaded would mutate the resource itself, which
    /// persists back into the .tscn on an editor save and leaks into every other scene sharing it.
    /// The previous version wrote in place, so opening a training scene with --dummies=64 and
    /// saving would have baked a 132 m floor into the file.
    /// </summary>
    private void ResizeFloor(int dummies)
    {
        if (Floor == null || !IsInstanceValid(Floor))
        {
            return;
        }

        float width = BaseFloorWidth + ((dummies - 1) * Spacing);

        var collision = Floor.GetNodeOrNull<CollisionShape3D>("CollisionShape3D");
        if (collision?.Shape is BoxShape3D box)
        {
            var resized = (BoxShape3D)box.Duplicate();
            resized.Size = new Vector3(width, box.Size.Y, BaseFloorDepth);
            collision.Shape = resized;
        }

        var mesh = Floor.GetNodeOrNull<MeshInstance3D>("MeshInstance3D");
        if (mesh?.Mesh is BoxMesh boxMesh)
        {
            var resized = (BoxMesh)boxMesh.Duplicate();
            resized.Size = new Vector3(width, boxMesh.Size.Y, BaseFloorDepth);
            mesh.Mesh = resized;
        }
    }

    /// <summary>
    /// Points the HUD at the first agent.
    ///
    /// Unconditionally, where the previous version did this only when exactly one agent was
    /// spawned and hid the HUD otherwise. That gate made the binding dead in the multi-agent case
    /// this class exists to serve: --viz renders exactly one window, and the useful thing to see in
    /// it is the telemetry of the body being watched. One agent's worth of HUD costs nothing
    /// against 64 simulated bodies, and headless runs never render it at all.
    /// </summary>
    private void BindTelemetry()
    {
        if (TelemetryHud == null || !IsInstanceValid(TelemetryHud) || _agents.Count == 0)
        {
            return;
        }

        RlAgent first = _agents[0];
        TelemetryHud.Ragdoll = first.Ragdoll;
        TelemetryHud.RLBridge = first.Bridge;
        TelemetryHud.Visible = true;
    }
}

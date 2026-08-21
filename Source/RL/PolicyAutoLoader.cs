using Godot;

namespace Physics4Fun.RL;

/// <summary>
/// Puts the inspection scene into ONNX inference mode using the most recently trained policy, so
/// pressing Play watches the current brain instead of starting anything.
///
/// Attached to the arena scene's ROOT node deliberately: godot_rl_agents' Sync node begins with
/// "await get_parent().ready", so a root script's _Ready is guaranteed to run before Sync decides
/// which mode it is in. Setting control_mode any later would be ignored.
///
/// If no policy exists yet (nothing trained, or the folder was cleared) it falls back to HUMAN
/// mode rather than letting Sync's assert fire. A missing model is the normal state of a fresh
/// clone, not an error worth crashing the scene over.
/// </summary>
public partial class PolicyAutoLoader : Node3D
{
    /// <summary>The Sync node whose control mode this sets. </summary>
    [Export] public Node? Sync { get; set; }

    /// <summary>Bridge to switch into continuous playback (no episode resets).</summary>
    [Export] public RagdollRLBridge? Bridge { get; set; }

    /// <summary>
    /// Optional hard pin. Empty by default, which is the normal case - an arena should follow its
    /// own brain's latest training rather than a path someone has to remember to update.
    ///
    /// Set it only to freeze an arena on one specific checkpoint, for a comparison against a known
    /// policy. A pinned path that exists beats BrainRunPrefixes.
    /// </summary>
    [Export] public string PromotedModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Run-directory name prefixes belonging to this arena's brain. The newest .onnx under any
    /// matching directory is loaded.
    ///
    /// This exists because "newest .onnx anywhere" is actively wrong on this project. Runs for
    /// different brains share one folder, and a night of walk training silently repointed the
    /// perturbation arena at a walking policy - the dummy fell before the ball ever arrived and it
    /// looked like a training regression rather than the wrong file. Pinning exact paths fixed that
    /// and created a slower failure: the pins go stale, and the perturbation arena was still loading
    /// a 61M-step policy from perturbation_v1_0 long after v7 existed.
    ///
    /// Filtering by prefix keeps the isolation without the staleness. Empty means "any run", which
    /// restores the old unsafe behaviour and should not be used on a rig with more than one brain.
    /// </summary>
    [Export] public string[] BrainRunPrefixes { get; set; } = System.Array.Empty<string>();

    /// <summary>Scanned for the newest matching *.onnx.</summary>
    [Export] public string RunsDirectory { get; set; } = "res://rl/runs";

    /// <summary>Sync.ControlModes.HUMAN - no policy, no server, no training.</summary>
    private const int ControlModeHuman = 0;

    /// <summary>Sync.ControlModes.ONNX_INFERENCE - runs a policy locally, never opens a socket.</summary>
    private const int ControlModeOnnxInference = 2;

    public override void _Ready()
    {
        if (Sync == null || !IsInstanceValid(Sync))
        {
            GD.PushWarning("[PolicyAutoLoader] No Sync node assigned; leaving control mode alone.");
            return;
        }

        // Set FIRST and unconditionally. PlaybackMode describes what this SCENE is for - an
        // inspection scene that never trains - so it must not depend on whether a policy happened
        // to be found. Tying it to policy discovery meant that any time the lookup came up empty
        // (nothing trained yet, or the run logs written somewhere else) the training time limit
        // silently came back to life and teleported the body to prone every 8 seconds - looking
        // exactly like the bug this was supposed to fix.
        if (Bridge != null && IsInstanceValid(Bridge))
        {
            Bridge.PlaybackMode = true;
        }
        else
        {
            GD.PushWarning(
                "[PolicyAutoLoader] No Bridge assigned - episodes will keep timing out every "
                + "MaxEpisodeSeconds. Assign the RLBridge node to keep playback continuous.");
        }

        string? model = ResolveNewestPolicy();
        if (model == null)
        {
            Sync.Set("control_mode", ControlModeHuman);
            GD.Print(
                "[PolicyAutoLoader] No trained policy found - idling (no policy, no training).\n"
                + $"  Train one, then promote it or leave it in {RunsDirectory}/<run>/.\n"
                + "  Nothing drives the body and nothing resets it - press R to drop to prone.");
            return;
        }

        Sync.Set("onnx_model_path", model);
        Sync.Set("control_mode", ControlModeOnnxInference);

        // Spelled out because the episode lines that follow ("Episode 3 ended -> starting episode
        // 4") look identical whether a policy is being trained or merely replayed, and mistaking
        // playback for training is an easy and confusing error to make.
        GD.Print(
            "\n=====================================================\n"
            + " PLAYBACK MODE - no training, no Python, no learning\n"
            + $" Policy : {model}\n"
            + $" Trained: {ExtractStepCount(model)}\n"
            + " Runs continuously - press R to retry from prone.\n"
            + "=====================================================");
    }

    /// <summary>
    /// Manual retry, since playback deliberately never resets on its own. RagdollDebugInput gates
    /// itself off during the RL state, so this key does not collide with the procedural controls.
    /// </summary>
    public override void _UnhandledInput(InputEvent @event)
    {
        if (Bridge == null || !IsInstanceValid(Bridge) || !Bridge.PlaybackMode)
        {
            return;
        }

        if (@event is InputEventKey { Pressed: true, Echo: false, Keycode: Key.R })
        {
            GD.Print("[PolicyAutoLoader] Manual reset - retrying from prone.");
            Bridge.ResetEpisode();
        }
    }

    /// <summary>
    /// Pulls the step count out of a checkpoint filename (checkpoint_000024704.onnx), so the
    /// banner says how trained the policy actually is. A policy with a few thousand steps behaves
    /// like noise, and knowing that up front explains the twitching rather than leaving it to look
    /// like a bug.
    /// </summary>
    private static string ExtractStepCount(string path)
    {
        string file = path.GetFile().GetBaseName();
        int underscore = file.LastIndexOf('_');
        if (underscore < 0 || !long.TryParse(file[(underscore + 1)..], out long steps))
        {
            return "unknown step count";
        }

        string note = steps < 1_000_000 ? " (very early - expect near-random motion)" : string.Empty;
        return $"{steps:N0} steps{note}";
    }

    /// <summary>
    /// The pinned model if one is set and exists, else the most recently modified .onnx in a run
    /// directory belonging to this arena's brain.
    /// </summary>
    private string? ResolveNewestPolicy()
    {
        if (!string.IsNullOrEmpty(PromotedModelPath) && Godot.FileAccess.FileExists(PromotedModelPath))
        {
            return PromotedModelPath;
        }

        using DirAccess? runs = DirAccess.Open(RunsDirectory);
        if (runs == null)
        {
            return null;
        }

        string? newest = null;
        ulong newestTime = 0;

        foreach (string runDir in runs.GetDirectories())
        {
            if (!MatchesBrain(runDir))
            {
                continue;
            }

            string path = $"{RunsDirectory}/{runDir}";
            using DirAccess? files = DirAccess.Open(path);
            if (files == null)
            {
                continue;
            }

            foreach (string file in files.GetFiles())
            {
                if (!file.EndsWith(".onnx", System.StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                string full = $"{path}/{file}";
                ulong modified = Godot.FileAccess.GetModifiedTime(full);
                if (modified >= newestTime)
                {
                    newestTime = modified;
                    newest = full;
                }
            }
        }

        return newest;
    }

    /// <summary>
    /// Whether a run directory belongs to this arena's brain.
    ///
    /// Prefix match on the directory name, which is exactly the experiment name plus SB3's numeric
    /// suffix (perturbation_v7_0), so "perturbation" selects the whole lineage and nothing else.
    /// </summary>
    private bool MatchesBrain(string runDirectoryName)
    {
        if (BrainRunPrefixes == null || BrainRunPrefixes.Length == 0)
        {
            return true;
        }

        foreach (string prefix in BrainRunPrefixes)
        {
            if (!string.IsNullOrEmpty(prefix)
                && runDirectoryName.StartsWith(prefix, System.StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}

using System.Collections.Generic;
using System.Text.Json;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// Checks Godot's observation and inference path against a reference captured from Isaac Lab.
/// Run headless:
///
/// <code>
/// godot --headless res://Scenes/RL/Shared/IsaacParityTest.tscn
/// </code>
///
/// <para><b>Why this exists rather than just watching the arena.</b> Every way this transfer can be
/// wrong is silent. A permuted joint order, an inverted roll sign, an un-remapped gravity vector -
/// none of them raise an error; they produce a body that moves badly, which is indistinguishable
/// from a policy that simply did not transfer. All three of those defects were in
/// `obs_action_contract.md` at the same time while every trained policy behaved perfectly, because
/// Isaac reads the rig contract and not the prose.</para>
///
/// <para>Two independent stages, because they fail differently:</para>
///
/// <list type="number">
/// <item><description><b>Inference parity.</b> Feed Isaac's exact recorded observation through
/// Godot's ONNX session. Same graph, same input, so the actions must match to float precision.
/// This isolates the runtime with the observation builder held out - if it fails, nothing about
/// the rig mapping matters yet.</description></item>
/// <item><description><b>Observation parity.</b> Build Godot's own observation at the rest pose and
/// compare it slot by slot. At rest the expected values are sharp and independently knowable, so a
/// mismatch localises to a named slice instead of being a vague behavioural difference.</description></item>
/// </list>
/// </summary>
public partial class IsaacParityTest : Node
{
    /// <summary>Reference written by `isaac_lab/scripts/dump_reference.py`.</summary>
    [Export] public string ReferencePath { get; set; } = "res://isaac_lab/assets/reference_obs.json";

    /// <summary>Policy to check inference against. Must be the same task the reference was dumped from.</summary>
    [Export(PropertyHint.File, "*.onnx")]
    public string PolicyPath { get; set; } = "res://isaac_lab/exported/stand_policy.onnx";

    /// <summary>Scene providing a ragdoll standing at rest on a floor.</summary>
    [Export] public string ChamberPath { get; set; } = "res://Scenes/TestChamber.tscn";

    /// <summary>Rig contract to check against. Must be the one the reference was dumped from.</summary>
    [Export] public string RigContractPath { get; set; } = IsaacRigContract.DefaultPath;

    /// <summary>
    /// Tolerance for the observation comparison. Generous on purpose, and sized to a measurement:
    /// Isaac's reference is captured after its reset has already taken a physics step, so its
    /// joints have drifted up to 0.097 rad off rest, while Godot's first tick is the authored pose
    /// exactly. That difference is physics, not mapping, and 0.15 clears it.
    ///
    /// <para><b>Known weakness of this stage.</b> At rest every joint is near zero, so a PERMUTED
    /// joint order is nearly invisible here - swapping two near-zero values changes almost nothing.
    /// This stage reliably catches frame and sign errors in the gravity, height and contact slices,
    /// where the expected values are large and distinct. Ordering is caught by the structural check
    /// in <see cref="CheckOrderingParity"/> instead, which compares the resolved names directly
    /// rather than inferring a permutation from numbers.</para>
    /// </summary>
    [Export] public float ObservationTolerance { get; set; } = 0.15f;

    /// <summary>Tolerance for inference parity. Tight: identical graph, identical input.</summary>
    [Export] public float InferenceTolerance { get; set; } = 1e-4f;

    private HumanoidRagdoll? _ragdoll;
    private IsaacObservation? _observation;
    private IsaacRigContract? _rig;
    private ActiveBone?[] _controlledBones = System.Array.Empty<ActiveBone?>();
    private JsonDocument? _reference;
    /// <summary>
    /// Physics ticks to let elapse before capturing Godot's observation.
    ///
    /// Not zero, and the reason is the contact slice. On the very first tick Godot's collision
    /// detection has not run yet, so the feet report no contact while Isaac's reference - captured
    /// after its reset already stepped the simulation - reports both planted. That is a sampling
    /// artefact, not a mapping error, and comparing at tick 0 reports it as a failure every time.
    ///
    /// Kept small: the body is held at its authored pose by its own controller, so a few ticks
    /// barely move it, while waiting longer would let the two solvers diverge and turn a slot
    /// check into a physics comparison.
    /// </summary>
    [Export] public int CaptureAfterTicks { get; set; } = 240;

    private int _ticks;
    private bool _done;
    private int _failures;

    public override void _Ready()
    {
        if (!LoadReference())
        {
            Finish(1);
            return;
        }

        try
        {
            _rig = IsaacRigContract.Load(RigContractPath);
        }
        catch (System.Exception e)
        {
            GD.PrintErr($"[parity] rig contract: {e.Message}");
            Finish(1);
            return;
        }

        var chamber = GD.Load<PackedScene>(ChamberPath).Instantiate();
        AddChild(chamber);
        _ragdoll = FindRagdoll(chamber);
        if (_ragdoll == null)
        {
            GD.PrintErr($"[parity] no HumanoidRagdoll under {ChamberPath}");
            Finish(1);
            return;
        }

        _observation = new IsaacObservation(_rig, _ragdoll);

        string[] boneNames = _rig.ActuatedBoneNames();
        _controlledBones = new ActiveBone?[boneNames.Length];
        var byName = new Dictionary<string, ActiveBone>();
        foreach (ActiveBone bone in _ragdoll.GetBones())
        {
            byName[bone.BoneName] = bone;
        }
        for (int i = 0; i < boneNames.Length; i++)
        {
            _controlledBones[i] = byName.TryGetValue(boneNames[i], out ActiveBone? b) ? b : null;
        }

        CheckOrderingParity();
        CheckInferenceParity();
    }

    /// <summary>
    /// Stage 0. The joint orderings Godot resolved, against the ones Isaac actually used.
    ///
    /// This is the check that catches a permutation, and it is a name comparison rather than a
    /// numeric one on purpose. At the rest pose every joint angle is near zero, so a scrambled
    /// order produces a nearly identical observation vector - the defect that matters most is the
    /// one the numbers are least able to see. Comparing the resolved names sidesteps that entirely.
    ///
    /// The reference carries both lists as Isaac reported them from the live articulation, so this
    /// also fails loudly if `dummy_rig.json` has drifted from the rig the policy was trained on.
    /// </summary>
    private void CheckOrderingParity()
    {
        CompareNames("physx_dof_order", _rig!.DofOrder);
        CompareNames("actuated_joint_names", _rig.ActuatedJoints);
    }

    private void CompareNames(string key, IReadOnlyList<IsaacRigContract.JointSpec> resolved)
    {
        if (!_reference!.RootElement.TryGetProperty(key, out JsonElement array))
        {
            GD.PrintErr($"[parity] reference has no '{key}'; re-run dump_reference.py.");
            _failures++;
            return;
        }

        var expected = new List<string>(array.GetArrayLength());
        foreach (JsonElement element in array.EnumerateArray())
        {
            expected.Add(element.GetString() ?? string.Empty);
        }

        int mismatches = 0;
        int firstBad = -1;
        for (int i = 0; i < Mathf.Min(expected.Count, resolved.Count); i++)
        {
            if (expected[i] != resolved[i].Name)
            {
                mismatches++;
                if (firstBad < 0)
                {
                    firstBad = i;
                }
            }
        }

        bool ok = expected.Count == resolved.Count && mismatches == 0;
        GD.Print($"[parity] 0. {key,-21}: {(ok ? "PASS" : "FAIL")}  "
                 + $"{resolved.Count}/{expected.Count} entries, {mismatches} mismatched");

        if (!ok)
        {
            _failures++;
            if (firstBad >= 0)
            {
                GD.PrintErr($"[parity]    first at {firstBad}: godot '{resolved[firstBad].Name}' "
                            + $"vs isaac '{expected[firstBad]}'");
            }
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_done || _observation == null || _ragdoll == null)
        {
            return;
        }

        if (_ticks++ < CaptureAfterTicks)
        {
            return;
        }
        _done = true;

        CheckObservationParity((float)delta);
        Finish(_failures == 0 ? 0 : 1);
    }

    private bool LoadReference()
    {
        using Godot.FileAccess? file = Godot.FileAccess.Open(ReferencePath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            GD.PrintErr($"[parity] no reference at {ReferencePath}. "
                        + "Run isaac_lab/scripts/dump_reference.py first.");
            return false;
        }

        _reference = JsonDocument.Parse(file.GetAsText());
        return true;
    }

    /// <summary>
    /// Stage 1. Isaac's observation through Godot's session must reproduce Isaac's action.
    /// </summary>
    private void CheckInferenceParity()
    {
        float[] refObs = ReadFloats("observation");
        float[] refAct = ReadFloats("action");

        using Godot.FileAccess? file = Godot.FileAccess.Open(PolicyPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            GD.PrintErr($"[parity] cannot open {PolicyPath}");
            _failures++;
            return;
        }

        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(file.GetBuffer((long)file.GetLength()));
        string inputName = "obs";
        foreach (string key in session.InputMetadata.Keys)
        {
            inputName = key;
            break;
        }

        var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(refObs, new[] { 1, refObs.Length });
        var inputs = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
        {
            Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(inputName, tensor),
        };

        float[] got;
        using (var results = session.Run(inputs))
        {
            got = System.Linq.Enumerable.ToArray(
                System.Linq.Enumerable.First(results).AsEnumerable<float>());
        }

        float worst = 0.0f;
        int worstIndex = -1;
        for (int i = 0; i < Mathf.Min(got.Length, refAct.Length); i++)
        {
            float d = Mathf.Abs(got[i] - refAct[i]);
            if (d > worst)
            {
                worst = d;
                worstIndex = i;
            }
        }

        bool ok = got.Length == refAct.Length && worst <= InferenceTolerance;
        GD.Print($"[parity] 1. inference  : {(ok ? "PASS" : "FAIL")}  "
                 + $"{got.Length}/{refAct.Length} actions, worst delta {worst:E3}"
                 + (worstIndex >= 0 ? $" at index {worstIndex}" : ""));

        if (!ok)
        {
            _failures++;
            GD.PrintErr("[parity]    the ONNX runtime disagrees with Isaac on the same input. "
                        + "Nothing downstream of this is meaningful until it passes.");
        }
    }

    /// <summary>
    /// Stage 2. Godot's own observation at rest, against Isaac's, slice by slice.
    /// </summary>
    private void CheckObservationParity(float delta)
    {
        float[] reference = ReadFloats("observation");
        float[] built = _observation!.Build(new RlContext(
            _ragdoll!,
            _ragdoll.Pelvis!,
            _ragdoll.Balance!,
            _controlledBones,
            0.0f,
            0.0f,
            delta));

        GD.Print($"[parity] 2. observation: built {built.Length}, reference {reference.Length}");
        if (built.Length != reference.Length)
        {
            GD.PrintErr("[parity]    width mismatch - the rig contract and the reference disagree.");
            _failures++;
            return;
        }

        foreach (JsonProperty slice in _reference!.RootElement.GetProperty("layout").EnumerateObject())
        {
            int from = slice.Value[0].GetInt32();
            int to = slice.Value[1].GetInt32();

            float worst = 0.0f;
            int worstIndex = from;
            for (int i = from; i < to && i < built.Length; i++)
            {
                float d = Mathf.Abs(built[i] - reference[i]);
                if (d > worst)
                {
                    worst = d;
                    worstIndex = i;
                }
            }

            bool ok = worst <= ObservationTolerance;
            bool binding = IsBinding(slice.Name);
            if (!ok && binding)
            {
                _failures++;
            }

            string verdict = ok ? "ok  " : binding ? "FAIL" : "info";
            GD.Print($"[parity]    {slice.Name,-20} [{from,3}:{to,3}]  {verdict}  "
                     + $"worst {worst:F4} at {worstIndex} "
                     + $"(godot {built[worstIndex]:+0.0000;-0.0000}, isaac {reference[worstIndex]:+0.0000;-0.0000})");

            if (!ok && (slice.Name == "joint_pos" || slice.Name == "joint_vel"))
            {
                ReportWorstJoints(built, reference, from, to);
            }
        }
    }

    /// <summary>
    /// Whether a slice's disagreement is a real defect or just the two engines' spawn transients.
    ///
    /// The binding slices are the ones fully determined by POSE, so a correct implementation must
    /// reproduce them regardless of how each engine settles: which way gravity points in the pelvis
    /// frame, how high the pelvis is, which feet are loaded.
    ///
    /// The rest are not. Godot's ragdoll is not in equilibrium at spawn - its arms swing, and the
    /// hands were measured at 21 rad/s two ticks in - while Isaac's reference is captured from a
    /// freshly reset articulation that is very nearly static. Comparing those numbers measures the
    /// difference between two spawn transients, not the correctness of a mapping, and asserting on
    /// them would produce a test that fails for reasons no one can fix.
    ///
    /// <para>This does leave the roll-axis SIGN unverified by this stage, since at rest every joint
    /// angle is near zero and a negation of a near-zero value is invisible. Settling it needs a
    /// commanded non-rest pose compared across both engines - see the note in the class summary.</para>
    /// </summary>
    private static bool IsBinding(string sliceName) => sliceName switch
    {
        "projected_gravity" => true,
        "pelvis_height" => true,
        "contacts" => true,
        "prev_action" => true,
        "command" => true,
        _ => false,
    };

    /// <summary>
    /// Names the joints that disagree most. A permuted DOF order shows up here as a scatter of
    /// large errors across unrelated joints, where a sign error shows up as pairs that match in
    /// magnitude and differ in sign - two different diagnoses from the same table.
    /// </summary>
    private void ReportWorstJoints(float[] built, float[] reference, int from, int to)
    {
        var rows = new List<(float Delta, string Name, float Godot, float Isaac)>();
        for (int i = from; i < to && i < built.Length; i++)
        {
            int dof = i - from;
            string name = dof < _rig!.DofOrder.Count ? _rig.DofOrder[dof].Name : $"dof{dof}";
            rows.Add((Mathf.Abs(built[i] - reference[i]), name, built[i], reference[i]));
        }

        rows.Sort((a, b) => b.Delta.CompareTo(a.Delta));
        for (int i = 0; i < Mathf.Min(8, rows.Count); i++)
        {
            (float d, string name, float g, float isaac) = rows[i];
            string hint = Mathf.Abs(g + isaac) < Mathf.Abs(g - isaac) * 0.1f ? "  <- sign?" : string.Empty;
            GD.Print($"[parity]        {name,-18} godot {g,+9:F4}  isaac {isaac,+9:F4}  delta {d:F4}{hint}");
        }
    }

    private float[] ReadFloats(string key)
    {
        JsonElement array = _reference!.RootElement.GetProperty(key);
        var values = new float[array.GetArrayLength()];
        int i = 0;
        foreach (JsonElement element in array.EnumerateArray())
        {
            values[i++] = element.GetSingle();
        }
        return values;
    }

    private static HumanoidRagdoll? FindRagdoll(Node root)
    {
        if (root is HumanoidRagdoll found)
        {
            return found;
        }
        foreach (Node child in root.GetChildren())
        {
            HumanoidRagdoll? hit = FindRagdoll(child);
            if (hit != null)
            {
                return hit;
            }
        }
        return null;
    }

    private void Finish(int code)
    {
        GD.Print(code == 0 ? "[parity] PASS" : $"[parity] FAIL ({_failures} slice(s))");
        _reference?.Dispose();
        GetTree().Quit(code);
    }
}

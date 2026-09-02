using System.Collections.Generic;
using Godot;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// Runs an Isaac Lab policy on one ragdoll, in-engine, with no training and nothing that resets it.
/// The Isaac-side equivalent of pressing F5 on an arena scene.
///
/// <para><b>Why this bypasses godot_rl_agents entirely.</b> That addon's
/// <c>GodotONNX.ONNXInference</c> is written against the Stable-Baselines3 export convention: it
/// requires an input named <c>state_ins</c>, reads outputs named <c>output</c> and
/// <c>state_outs</c>, and its <c>Initialize</c> does <c>session.OutputMetadata["output"]</c>. An
/// rsl_rl graph is <c>obs[1,143] -> actions[1,36]</c> with none of those names, so it throws at
/// load rather than misbehaving. Rather than bend the Isaac export to fit an addon convention
/// nothing else here needs, this owns an <see cref="InferenceSession"/> directly.</para>
///
/// <para>Bypassing <c>Sync</c> also removes <c>action_repeat</c> from the picture. That property is
/// scene-wide, and the Godot-native track needs it at 8 (15 Hz) while an Isaac policy needs 2
/// (60 Hz) - one scene cannot serve both. Counting physics ticks here means the two tracks coexist
/// without either being retuned.</para>
///
/// <para>The observation normaliser is baked into the graph (<c>actor_obs_normalization=True</c>),
/// so raw observations go in and nothing here may normalise them. The output is the distribution
/// MEAN, not a sample: deterministic at inference, which is what you want in-engine.</para>
/// </summary>
public partial class IsaacPolicyDriver : Node
{
    /// <summary>Physics ticks per policy step. 120 Hz physics / 2 = the 60 Hz training rate.</summary>
    public const int PhysicsTicksPerPolicyStep = 2;

    /// <summary>The body this drives. Required.</summary>
    [Export] public HumanoidRagdoll? Ragdoll { get; set; }

    /// <summary>
    /// Policy to run, e.g. <c>res://isaac_lab/exported/stand_policy.onnx</c>. Which brain this is
    /// - stand, walk, perturb or run - is decided entirely by this path; they share one
    /// observation and action contract, which is what makes the "experts" split cheap.
    /// </summary>
    [Export(PropertyHint.File, "*.onnx")] public string PolicyPath { get; set; } = string.Empty;

    /// <summary>Rig contract location. The default is right unless the assets moved.</summary>
    [Export] public string RigContractPath { get; set; } = IsaacRigContract.DefaultPath;

    /// <summary>
    /// The policy's own contract, written beside the .onnx by <c>isaac_lab_3/scripts/export.py</c>.
    /// Empty for an Isaac Lab 2.3.2 / PhysX policy, which carries no such file.
    ///
    /// <para><b>Required for any policy trained under Isaac Lab 3 / Newton.</b> It supplies
    /// `newton_dof_order`, which differs from the rig's own `physx_dof_order` in 42 of 45 slots.
    /// Without it, 90 of the 143 observation floats reach the network permuted - and because every
    /// joint angle is near zero at the rest pose, nothing numeric can detect it. The dummy just
    /// flails.</para>
    /// </summary>
    [Export(PropertyHint.File, "*.json")] public string PolicyContractPath { get; set; } = string.Empty;

    /// <summary>
    /// Derive the four contact flags from bone height instead of real contact. Set for Newton
    /// policies; see <see cref="IsaacObservation.UseHeightContacts"/>.
    /// </summary>
    [Export] public bool HeightContacts { get; set; }

    /// <summary>
    /// Velocity command (vx, vy, yaw_rate) written into observation slice [140:143]. Zero for
    /// Stand and Perturbation, which were trained with it held at zero throughout. Walk and Run
    /// read it, so an arena for either wants a non-zero forward component.
    /// </summary>
    [Export] public Vector3 Command { get; set; } = Vector3.Zero;

    /// <summary>
    /// EMA smoothing applied to the joint velocity feeding each bone's Hill force-velocity law.
    /// 1 leaves it raw (Godot's default). ~0.15 filters solver chatter out of it.
    ///
    /// <para>Preferred over <see cref="DisableHillLimit"/>: it keeps the biomechanics and removes
    /// only the artifact. See <see cref="ActiveBone.HillVelocityFilterAlpha"/>.</para>
    /// </summary>
    [Export] public float HillVelocityFilter { get; set; } = 1.0f;

    /// <summary>
    /// Low-pass alpha for the joint velocities written into observation slice [55:100]. 1.0 is off.
    ///
    /// <para><b>The filter existed and was wired to nothing.</b> `IsaacObservation.JointVelocityFilter`
    /// was implemented and documented - including the measurement that motivates it - but no arena,
    /// driver or scene ever set it, so it sat at its 1.0 default and every deployment fed the policy
    /// raw solver velocity.</para>
    ///
    /// <para>Measured with the observation clip lifted, while the dummy stood still at pelvis 0.83:
    /// joint velocities of 8, 18, 20 and 35 rad/s on Spine.x and Shin_L.y. Isaac's maximum across 64
    /// environments in the same posture is 7.9. The clip at 15 hid roughly half of it. So 45 of 143
    /// observation floats reported thrashing to a policy trained on a body that does not thrash.</para>
    ///
    /// <para><b>MEASURED, AND IT MAKES TRANSFER WORSE. Leave it at 1.0.</b> With the promoted
    /// balance brain on the Stand check at authority 0.10: filter 1.00 STANDS, 0.50 falls, 0.30
    /// falls, 0.15 falls - and at 0.15 the brain needed the authority dropped to 0.05 to stand at
    /// all. The reasoning above is sound and the chatter is real; the remedy is not. An EMA at
    /// alpha 0.15 is a ~0.11 s time constant, and the lag it adds to a balance-critical signal
    /// costs more than the noise it removes.</para>
    ///
    /// <para>Kept wired rather than deleted because it was previously implemented, documented and
    /// reachable from nothing, which is how it survived long enough to look like an answer. The
    /// knob is now settable and OFF by default, and this paragraph is the reason to leave it
    /// there.</para>
    /// </summary>
    [Export] public float JointVelocityFilter { get; set; } = 1.0f;

    /// <summary>
    /// Feed observation slice [55:100] to the policy at all. **Must match the task config's
    /// `obs_joint_vel_enabled` for the checkpoint being run.** See
    /// <see cref="IsaacObservation.JointVelocityEnabled"/>.
    /// </summary>
    [Export] public bool JointVelocityEnabled { get; set; } = true;

    /// <summary>
    /// Mask the joint-velocity observation on joints narrower than this, in radians. 0 is off.
    /// **Must match `obs_joint_vel_min_range` for the checkpoint being run.**
    /// </summary>
    [Export] public float JointVelocityMinRange { get; set; }

    /// <summary>
    /// Seconds after which to dump the FULL 45-DOF joint pose once, as a JSON array. 0 disables it.
    ///
    /// <para><b>Why this exists.</b> While Godot's balance layer holds a successful stand - head
    /// 1.533, rock steady - its joints sit about 0.82 rad away from the rest pose. Isaac's
    /// `default_joint_pos` is all zeros, and the policy commands offsets from there bounded by
    /// `action_scale * span`, which for a +/-0.5 rad axis is 0.2 rad. **Godot's standing stance is
    /// therefore outside the policy's reachable action space**, and "zero action" means a pose that
    /// stands in Isaac and falls over in Godot.</para>
    ///
    /// <para>Dumping the stance lets it become Isaac's operating point, so action zero means the
    /// same body configuration in both engines.</para>
    /// </summary>
    [Export] public float DumpPoseAfterSeconds { get; set; }

    /// <summary>Logs the first step's observation and action. Cheap, and the fastest way to spot a mapping error.</summary>
    [Export] public bool LogFirstStep { get; set; } = true;

    /// <summary>
    /// Seconds between per-slice observation diagnostics. Zero disables them.
    ///
    /// Reports the largest magnitude in each slice, which is what localises an out-of-distribution
    /// input. The policy's observation normaliser is baked into the graph, so a slice whose values
    /// are far outside what training saw is amplified into a saturated action - and the action
    /// magnitude alone cannot say WHICH slice did it.
    /// </summary>
    [Export] public float DiagnosticInterval { get; set; } = 0.5f;

    /// <summary>
    /// Seconds spent holding EVERY bone at its rest pose before inference starts.
    ///
    /// This exists to reproduce Isaac's initial condition rather than to be cautious. An Isaac
    /// episode begins from `joint_pos = 0` on all 45 DOF with the drives already tracking that
    /// target, so the policy's first observation is a body genuinely at rest. Godot's ragdoll
    /// arrives from its procedural controller in a posed stance - measured at 0.94 rad on the worst
    /// joint against Isaac's 0.10 - and commanding rest does not teleport it there; the PD has to
    /// drive it, which takes time the first observation does not give it.
    ///
    /// Without this the policy's first action came out at 1.9, nearly twice its clamp, from a body
    /// that was standing perfectly well.
    /// </summary>
    [Export] public float WarmupSeconds { get; set; } = 1.0f;

    /// <summary>
    /// Ignore the policy and command all-zero actions, i.e. hold the rest pose.
    ///
    /// A control, not a feature. Zero action is the rest pose by construction, and in Isaac that
    /// pose is statically stable - the feet carry 790 N against an 80.6 kg body, and a zero-action
    /// policy survives 168 steps. If Godot's body collapses under the same command, then the gap is
    /// in the body and its actuators rather than in the observation or action mapping, and no
    /// amount of fixing the wiring will make a trained policy transfer.
    /// </summary>
    [Export] public bool ZeroActionBaseline { get; set; }

    /// <summary>
    /// Balance-controller strength during the RL state, 0 to 1. 0 is Godot's default (off).
    ///
    /// <para><b>This supplies the one thing measurement says is missing.</b> Godot's dummy stands at
    /// essentially the REST POSE - dumping the stance while its balance layer holds a steady 1.533 m
    /// head shows every actuated joint within the policy's reach (foot 0.11 rad against a 0.24
    /// reach, upper arm 0.10 against 0.20); only the passive HAND joints sit far out, and they
    /// dangle. So the pose was never the problem.</para>
    ///
    /// <para>What differs is that Godot reaches that pose with an external stabilising wrench -
    /// pelvis stabilisation, ankle and hip strategies - which `StateBalanceStrengthMap` sets to
    /// **0.0** for `ReinforcementLearning`. Isaac has no equivalent and does not need one, because
    /// its body is passively far more stable: under zero action Isaac's joints deviate 0.12 rad and
    /// its centre of mass drifts 0.05 m, against Godot's 1.01 rad and 0.30 m. The policy therefore
    /// learned to stand on a body that mostly holds itself, and in Godot it is asked to replace a
    /// balance controller it never knew existed.</para>
    ///
    /// <para>Unlike <see cref="AssistMode"/> this keeps the ragdoll in the RL state, so the
    /// procedural layer still does NOT write joint targets - the policy owns the pose, and the
    /// balance layer owns root stabilisation. That division is a real architecture, not a
    /// workaround.</para>
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float BalanceAssist { get; set; }

    /// <summary>
    /// Overrides the action scale the policy was trained with. 0 uses the contract's value.
    ///
    /// <para>A diagnostic dial, not a fix: it tells you how much of the policy's commanded
    /// deflection a body can absorb before the stand breaks. With balance assist on, zero action
    /// stands indefinitely and the trained policy falls in four seconds, so somewhere between them
    /// is the authority at which this policy stops being actively harmful. Driving a policy at a
    /// scale it was not trained at is a mismatch by definition - the honest fix is retraining at
    /// whatever scale this identifies.</para>
    /// </summary>
    [Export] public float ActionScaleOverride { get; set; }

    [Export] public bool AssistMode { get; set; }

    /// <summary>
    /// How much of the policy's commanded offset is applied in <see cref="AssistMode"/>, 0 to 1.
    /// 1 is the full offset; 0 is Godot's balance layer alone.
    ///
    /// <para><b>Why this needs a dial at all.</b> Godot's balance layer holds the dummy indefinitely
    /// on its own - measured at head 1.533 and pelvis 0.816, dead steady for 20 s. The Isaac policy
    /// was trained as the ONLY controller, in a simulator whose body is passively far more stable
    /// (under zero action Isaac's joints deviate 0.12 rad and its centre of mass drifts 0.05 m,
    /// against Godot's 1.01 rad and 0.30 m). Dropped in at full authority it does not assist that
    /// layer, it fights it - and the pair together fall faster than either alone.</para>
    ///
    /// <para>At reduced authority the policy becomes a corrective term on a body that is already
    /// balanced, which is a legitimate architecture and an honest one to report: the brain is
    /// modulating a stable stance rather than producing it.</para>
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float AssistAuthority { get; set; } = 1.0f;

    /// <summary>
    /// Drive the joints the way ISAAC does - per-DOF PD in joint space - instead of the way Godot
    /// does, and disable <see cref="ActiveBone"/>'s own actuator on the bones this owns.
    ///
    /// <para><b>The last structural difference between the engines.</b> Everything else now matches:
    /// same 16-body mechanism with 15 D6 joints, geometry to a centimetre, identical gains on every
    /// bone, joint ordering exact, ONNX inference to 1.5e-07, and a frame map that is a proper
    /// rotation. What does not match is the control law. `ActiveBone` runs Stable PD on the QUATERNION
    /// error of a bone's orientation, with a gravity feed-forward, an integral term, an SPD velocity
    /// factor and a hard torque clamp; Isaac runs a plain per-DOF PD folded into the articulation
    /// solve.</para>
    ///
    /// <para>The evidence that this is what matters: Godot's body collapses in under two seconds
    /// under ZERO action - commanding nothing but the rest pose - while Isaac's holds 84% standing
    /// for eight seconds under the identical command. That is a property of the actuator, not of any
    /// policy, observation or joint ordering.</para>
    ///
    /// <para>Check this with <see cref="ZeroActionBaseline"/> before involving the policy at all. If
    /// joint-space PD cannot hold the rest pose either, the gap is deeper than the control law and
    /// nothing downstream is worth testing.</para>
    /// </summary>
    [Export] public bool JointSpacePd { get; set; }

    private InferenceSession? _session;
    private IsaacRigContract? _rig;
    private IsaacObservation? _observation;
    private IsaacActionSpace? _actions;
    private ActiveBone?[] _controlledBones = System.Array.Empty<ActiveBone?>();
    private Quaternion[] _offsets = System.Array.Empty<Quaternion>();
    private bool _dumpedPose;
    private float _elapsed;
    private string _inputName = "obs";
    private int _tick;
    private bool _ready;
    private bool _loggedFirstStep;
    private float _sinceDiagnostic;
    private float _warmupRemaining;
    /// <summary>Reporting only; see <see cref="IsaacDriverDiagnostics"/>. Null until _Ready finishes.</summary>
    private IsaacDriverDiagnostics? _diagnostics;

    private float _trackingError;
    private float _appliedTorque;

    /// <summary>Slice bounds from obs_action_contract.md §3, for the per-slice diagnostic.</summary>

    public override void _Ready()
    {
        if (Ragdoll == null || !IsInstanceValid(Ragdoll))
        {
            GD.PushError("[IsaacPolicyDriver] No Ragdoll assigned; this node will do nothing.");
            return;
        }

        try
        {
            _rig = IsaacRigContract.Load(
                RigContractPath,
                IsaacRigContract.LoadDofOrderOverride(PolicyContractPath));
        }
        catch (System.Exception e)
        {
            GD.PushError($"[IsaacPolicyDriver] Could not load the rig contract: {e.Message}");
            return;
        }

        _observation = new IsaacObservation(_rig, Ragdoll)
        {
            UseHeightContacts = HeightContacts,
            JointVelocityClip = IsaacRigContract.LoadJointVelocityClip(PolicyContractPath),
            JointVelocityFilter = JointVelocityFilter,
            JointVelocityEnabled = JointVelocityEnabled,
            JointVelocityMinRange = JointVelocityMinRange,
        };
        if (!JointVelocityEnabled)
        {
            GD.Print("[IsaacPolicyDriver] joint velocity channel MASKED - obs[55:100] is zero, "
                     + "matching a policy trained with obs_joint_vel_enabled=False");
        }
        if (JointVelocityFilter < 1.0f)
        {
            GD.Print($"[IsaacPolicyDriver] joint velocity filter {JointVelocityFilter:F2} on the "
                     + "observation - the policy reads the limb, not the solver");
        }
        _actions = new IsaacActionSpace(_rig)
        {
            ActionRateLimit = IsaacRigContract.LoadActionRateLimit(PolicyContractPath),
            ActionScale = ActionScaleOverride > 0.0f
                ? ActionScaleOverride
                : (IsaacRigContract.LoadActionScale(PolicyContractPath) is var scale && scale > 0.0f
                    ? scale
                    : IsaacActionSpace.DefaultActionScale),
        };

        ResolveControlledBones();

        if (!LoadPolicy())
        {
            return;
        }

        // After ResolveControlledBones, so the bone array it holds is the final one.
        _diagnostics = new IsaacDriverDiagnostics(
            Ragdoll!, _rig!, _actions!, _controlledBones, JointSpacePd);

        // The body must be in RL state or its own procedural controller keeps driving the bones and
        // fights every command this issues.
        //
        // Order matters. `UpdateBoneMuscleStiffness` reads BOTH of these flags - it softens to
        // ReinforcementLearningIdleStiffness while no policy is active, and applies
        // ArmLoadBearingGain only once one is - so they have to be set before anything triggers a
        // refresh, or the bones keep the previous state's gains.
        if (!AssistMode)
        {
            Ragdoll.CurrentState = RagdollState.ReinforcementLearning;
            Ragdoll.ReinforcementLearningPolicyActive = true;

            if (BalanceAssist > 0.0f && Ragdoll.Balance != null && IsInstanceValid(Ragdoll.Balance))
            {
                // **Balanced, not ReinforcementLearning.** Raising the RL entry in
                // `StateBalanceStrengthMap` is not enough on its own: every balance strategy
                // self-gates to Balanced/Stumbling, so in the RL state they stay off whatever the
                // strength says - measured as a trace identical to no assist at all, at both 1.0
                // and 0.6. Staying in Balanced wakes them, and `SuppressProceduralPose` is what
                // stops the procedural layer stamping its own joint targets over the policy's.
                Ragdoll.CurrentState = RagdollState.Balanced;
                Ragdoll.SuppressProceduralPose = true;
                Ragdoll.Balance.StateBalanceStrengthMap[(int)RagdollState.Balanced] =
                    Mathf.Clamp(BalanceAssist, 0.0f, 1.0f);
                GD.Print($"[IsaacPolicyDriver] balance assist {BalanceAssist:F2} - policy owns the "
                         + "pose, Godot's balance layer owns root stabilisation");
            }
        }
        else
        {
            // Stay in the procedural state so UpdateBoneTargetRotations keeps writing a balanced
            // base pose every tick. Run after the ragdoll has written it, so the offset composes
            // on top rather than being overwritten the same frame.
            ProcessPriority = 100;
            ProcessPhysicsPriority = 100;
        }

        // NOT TeleportToStanding(), though it is tempting: it would also refresh the muscle
        // stiffness, which is otherwise private and only reachable through a state transition. But
        // it was measured to make things worse - the teleported pose is not statically stable in
        // Godot once the balance controller is off, so the body is already tilted 32 degrees by the
        // time warmup ends, against 0.5 degrees when left where the procedural controller had it.
        // Refreshing the gains is not worth starting the policy from a toppling body.
        //
        // (StartReinforcementLearning() is likewise wrong here - it drops the body to prone.)
        NeutraliseUncommandedBones();
        if (HillVelocityFilter < 1.0f)
        {
            int filtered = 0;
            foreach (ActiveBone bone in Ragdoll!.GetBones())
            {
                if (!IsInstanceValid(bone))
                {
                    continue;
                }
                bone.HillVelocityFilterAlpha = HillVelocityFilter;
                filtered++;
            }
            GD.Print($"[IsaacPolicyDriver] Hill velocity filter {HillVelocityFilter:F2} on "
                     + $"{filtered} bone(s) - the force-velocity law sees the limb, not the solver");
        }
        if (JointSpacePd)
        {
            SilenceGodotActuators();
        }
        _warmupRemaining = WarmupSeconds;

        _ready = true;
        GD.Print($"[IsaacPolicyDriver] {System.IO.Path.GetFileName(PolicyPath)} -> "
                 + $"{_observation.Size} obs / {_actions.Size} actions at "
                 + $"{Engine.PhysicsTicksPerSecond / PhysicsTicksPerPolicyStep} Hz");
    }

    private bool LoadPolicy()
    {
        if (string.IsNullOrWhiteSpace(PolicyPath))
        {
            GD.PushError("[IsaacPolicyDriver] PolicyPath is empty. Point it at an exported .onnx.");
            return false;
        }

        using Godot.FileAccess? file = Godot.FileAccess.Open(PolicyPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            GD.PushError($"[IsaacPolicyDriver] Cannot open '{PolicyPath}' "
                         + $"(Godot error {Godot.FileAccess.GetOpenError()}). Export a policy first with "
                         + "isaac_lab/scripts/export.ps1.");
            return false;
        }

        try
        {
            // Read through FileAccess rather than handing OnnxRuntime the path: `res://` is not a
            // filesystem path once the project is exported into a .pck.
            _session = new InferenceSession(file.GetBuffer((long)file.GetLength()));
        }
        catch (System.Exception e)
        {
            GD.PushError($"[IsaacPolicyDriver] Failed to load '{PolicyPath}': {e.Message}");
            return false;
        }

        return ValidateSignature();
    }

    /// <summary>
    /// Checks the graph against the contract before a single step runs. A width mismatch would
    /// otherwise surface as an OnnxRuntime exception once per physics tick, or - worse, if the
    /// widths happen to line up - as a body that merely moves wrongly.
    /// </summary>
    private bool ValidateSignature()
    {
        IReadOnlyDictionary<string, NodeMetadata> inputs = _session!.InputMetadata;
        if (inputs.Count != 1)
        {
            GD.PushError($"[IsaacPolicyDriver] Expected exactly one graph input, found {inputs.Count}: "
                         + string.Join(", ", inputs.Keys));
            return false;
        }

        foreach (KeyValuePair<string, NodeMetadata> input in inputs)
        {
            _inputName = input.Key;
            int width = input.Value.Dimensions[^1];
            if (width != _observation!.Size)
            {
                GD.PushError($"[IsaacPolicyDriver] Graph expects {width} observations, the contract "
                             + $"builds {_observation.Size}. The rig and the policy disagree - "
                             + "re-export, or re-run convert.ps1.");
                return false;
            }
        }

        foreach (KeyValuePair<string, NodeMetadata> output in _session.OutputMetadata)
        {
            int width = output.Value.Dimensions[^1];
            if (width != _actions!.Size)
            {
                GD.PushError($"[IsaacPolicyDriver] Graph emits {width} actions, the contract expects "
                             + $"{_actions.Size}.");
                return false;
            }
            break;
        }

        return true;
    }

    /// <summary>
    /// Commands every bone the policy does NOT actuate back to its rest pose, and holds it there.
    ///
    /// <para>Head, Hand_L and Hand_R - 9 of the 45 DOF - are passive in both engines. On the Isaac
    /// side that means a PD drive tracking a constant zero target, so they sit at rest for the
    /// entire episode and the policy only ever saw them there. On the Godot side nothing owns them
    /// once the procedural controller is switched off: they keep whatever
    /// <see cref="ActiveBone.TargetLocalRotation"/> that controller last wrote, which is a posed
    /// stance, not rest. Measured before this: both hands held 0.98 rad (56 degrees) off rest
    /// indefinitely, against 0.02-0.06 in Isaac's reference.</para>
    ///
    /// <para>That is six observation slots pinned a full radian away from anything training
    /// contained, feeding a normaliser that turns the discrepancy into a saturated action. The
    /// actuated bones need no such treatment - the policy overwrites them every step.</para>
    /// </summary>
    private void NeutraliseUncommandedBones()
    {
        var actuated = new HashSet<string>(_rig!.ActuatedBoneNames());
        int reset = 0;

        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            if (actuated.Contains(bone.BoneName) || !IsInstanceValid(bone))
            {
                continue;
            }
            bone.TargetLocalRotation = bone.GetRestLocalRotation();
            reset++;
        }

        GD.Print($"[IsaacPolicyDriver] held {reset} uncommanded bone(s) at rest, matching Isaac's "
                 + "zero-target PD on the passive DOF");
    }

    /// <summary>
    /// Holds every bone at rest during warmup, actuated ones included. Reproduces the state an
    /// Isaac episode starts from: all 45 DOF at zero with the drives already tracking it.
    /// </summary>
    private void HoldAllAtRest()
    {
        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            if (IsInstanceValid(bone))
            {
                bone.TargetLocalRotation = bone.GetRestLocalRotation();
            }
        }
    }

    /// <summary>Index into the action vector for a controlled bone, or -1.</summary>
    private int IndexOfControlledBone(ActiveBone bone)
    {
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            if (ReferenceEquals(_controlledBones[i], bone))
            {
                return i * IsaacRigContract.AxesPerBone;
            }
        }
        return -1;
    }

    private static char AxisName(int axis) => axis switch { 0 => 'x', 1 => 'y', _ => 'z' };

    /// <summary>Fallback motor force for a bone with no contract entry, N.m.</summary>
    private const float DefaultMotorForce = 100.0f;

    private void SilenceGodotActuators()
    {
        int silenced = 0;
        foreach (ActiveBone? bone in _controlledBones)
        {
            if (bone != null && IsInstanceValid(bone))
            {
                bone.MuscleStrength = 0.0f;
                silenced++;
            }
        }
        GD.Print($"[IsaacPolicyDriver] joint-space PD: silenced {silenced} Godot actuator(s)");
    }

    private void ApplyJointSpaceTorque(float delta)
    {
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone == null || !IsInstanceValid(bone))
            {
                continue;
            }

            ActiveBone? parent = bone.ParentBone;
            if (parent == null || !IsInstanceValid(parent))
            {
                continue;
            }

            Vector3 current = IsaacObservation.DeviationFromRest(bone);
            Vector3 target = _actions!.TargetEuler[i];

            // Rate about the joint's axes: the child's angular velocity relative to its parent,
            // resolved in the parent's frame - which is the frame the angles are measured in.
            Vector3 rate = parent.GlobalTransform.Basis.Inverse()
                           * (bone.AngularVelocity - parent.AngularVelocity);

            var torque = Vector3.Zero;
            int baseIndex = i * IsaacRigContract.AxesPerBone;
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                IsaacRigContract.JointSpec spec = _rig!.ActuatedJoints[baseIndex + axis];
                float error = Component(target, spec.GodotAxis) - Component(current, spec.GodotAxis);

                // Inertia about this joint's own axis, not the bone's smallest principal axis: a
                // flexion axis is usually transverse to a limb's slender axis, and using the
                // minimum throws away authority by up to 5x at the knee. Guarded well BELOW any
                // real inertia - clamping it UP to a fictitious value voids the SPD stability
                // guarantee and turns the D term into a per-tick velocity amplifier.
                float inertia = Mathf.Max(
                    MinEffectiveInertia, bone.GetEffectiveInertiaAboutAxis(AxisVector(spec.GodotAxis)));
                float denominator = 1.0f
                                    + (spec.Damping * delta / inertia)
                                    + (spec.Stiffness * delta * delta / inertia);

                float t = (spec.Stiffness * error - spec.Damping * Component(rate, spec.GodotAxis))
                          / denominator;
                SetComponent(ref torque, spec.GodotAxis, Mathf.Clamp(t, -spec.Effort, spec.Effort));
            }

            Vector3 world = parent.GlobalTransform.Basis * torque;
            bone.ApplyTorque(world);
            parent.ApplyTorque(-world);

            // Tracking error, accumulated for the diagnostic. This is the number that separates the
            // two remaining explanations: if the joints SIT at their commanded angles and the body
            // still falls, the controller is fine and something about the body is wrong. If they
            // cannot reach them under full torque, the constraint solver is sagging under load -
            // which is a property of Godot's solver that no Isaac actuator model could reproduce.
            _trackingError = Mathf.Max(_trackingError, (target - current).Length());
            _appliedTorque = Mathf.Max(_appliedTorque, torque.Length());
        }
    }

    /// <summary>
    /// Divide-by-zero guard for the SPD denominator. Must stay below every real body inertia - see
    /// <see cref="ApplyJointSpaceTorque"/>.
    /// </summary>
    private const float MinEffectiveInertia = 1e-5f;

    private static Vector3 AxisVector(char axis) => axis switch
    {
        'x' => Vector3.Right,
        'y' => Vector3.Up,
        _ => Vector3.Back,
    };

    private static float Component(in Vector3 v, char axis) => axis switch
    {
        'x' => v.X,
        'y' => v.Y,
        'z' => v.Z,
        _ => 0.0f,
    };

    private static void SetComponent(ref Vector3 v, char axis, float value)
    {
        switch (axis)
        {
            case 'x': v.X = value; break;
            case 'y': v.Y = value; break;
            case 'z': v.Z = value; break;
        }
    }

    private void ResolveControlledBones()
    {
        string[] boneNames = _rig!.ActuatedBoneNames();
        _controlledBones = new ActiveBone?[boneNames.Length];
        _offsets = new Quaternion[boneNames.Length];

        var byName = new Dictionary<string, ActiveBone>();
        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            byName[bone.BoneName] = bone;
        }

        for (int i = 0; i < boneNames.Length; i++)
        {
            _offsets[i] = Quaternion.Identity;
            if (byName.TryGetValue(boneNames[i], out ActiveBone? bone))
            {
                _controlledBones[i] = bone;
            }
            else
            {
                GD.PushError($"[IsaacPolicyDriver] Rig contract actuates '{boneNames[i]}', "
                             + "which this ragdoll does not have.");
            }
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!_ready)
        {
            return;
        }

        if (_warmupRemaining > 0.0f)
        {
            _warmupRemaining -= (float)delta;
            if (!AssistMode)
            {
                HoldAllAtRest();
            }
            if (_warmupRemaining <= 0.0f)
            {
                GD.Print("[IsaacPolicyDriver] warmup complete, policy engaging");
            }
            return;
        }

        // Decimation, matching `decimation = 2` on the Isaac side. The commanded targets persist
        // between policy steps, so the PD drives keep tracking on the tick in between - which is
        // exactly what the trainer does.
        if (_tick++ % PhysicsTicksPerPolicyStep == 0)
        {
            Step((float)delta);
        }

        // The PD itself runs at the FULL physics rate, on every tick including the ones between
        // policy steps. That is what Isaac's drives do - the target is held while the actuator keeps
        // integrating - and running it only on policy ticks would halve the control rate.
        if (JointSpacePd)
        {
            ApplyJointSpaceTorque((float)delta);
        }

        // At the FULL physics rate, like the PD above: the target persists between policy steps and
        // the motor keeps closing on it, which is what Isaac's drives do between decimated steps.
    }

    private void Step(float delta)
    {
        _elapsed += delta * PhysicsTicksPerPolicyStep;
        _observation!.SetCommand(Command);

        var context = new RlContext(
            Ragdoll!,
            Ragdoll.Pelvis!,
            Ragdoll.Balance!,
            _controlledBones,
            0.0f,
            0.0f,
            delta);

        float[] obs = _observation.Build(context);
        float[] actions = ZeroActionBaseline ? new float[_actions!.Size] : Infer(obs);
        if (actions.Length == 0)
        {
            return;
        }

        _actions!.Decode(actions, _offsets);

        // The clamped action - not the raw one - is what the next observation must carry, because
        // that is what the environment recorded during training.
        _observation.RecordAction(_actions.ClampedAction);

        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone != null && IsInstanceValid(bone))
            {
                if (AssistMode)
                {
                    // FeedForwardTargetOffset, not TargetLocalRotation. The procedural layer
                    // rewrites TargetLocalRotation every tick and ActiveBone consumes it in the
                    // same step, so an offset written from here lands after it has already been
                    // used and is gone by the next one - measured as output identical to three
                    // decimals with the policy ignored, i.e. completely inert.
                    //
                    // This property is the seam built for exactly this: ActiveBone composes it as
                    // `parentGlobalRot * TargetLocalRotation * _smoothedFeedForwardOffset`, and
                    // nothing in the procedural pipeline writes it.
                    // Slerp from identity: identity is "leave the balance layer's pose alone",
                    // so this scales the policy's correction rather than blending two poses.
                    bone.PolicyTargetOffset = AssistAuthority >= 1.0f
                        ? _offsets[i]
                        : Quaternion.Identity.Slerp(_offsets[i], Mathf.Clamp(AssistAuthority, 0.0f, 1.0f))
                            .Normalized();
                }
                else
                {
                    bone.TargetLocalRotation = (bone.GetRestLocalRotation() * _offsets[i]).Normalized();
                }
            }
        }

        if (DumpPoseAfterSeconds > 0.0f && !_dumpedPose && _elapsed >= DumpPoseAfterSeconds)
        {
            _dumpedPose = true;
            var pose = new System.Text.StringBuilder("[");
            for (int i = 0; i < _rig!.DofOrder.Count; i++)
            {
                pose.Append(i > 0 ? ", " : "").Append(obs[10 + i].ToString(
                    "0.0000", System.Globalization.CultureInfo.InvariantCulture));
            }
            GD.Print($"[IsaacPolicyDriver] STANCE (newton_dof_order, rad from rest) {pose.Append(']')}");
        }

        if (LogFirstStep && !_loggedFirstStep)
        {
            _loggedFirstStep = true;
            _diagnostics?.LogFirstStep(obs, actions, _trackingError, _appliedTorque);
        }

        if (DiagnosticInterval > 0.0f)
        {
            _sinceDiagnostic += delta * PhysicsTicksPerPolicyStep;
            if (_sinceDiagnostic >= DiagnosticInterval)
            {
                _sinceDiagnostic = 0.0f;
                _diagnostics?.LogSlices(obs, actions, _trackingError, _appliedTorque);
            }
        }
    }

    private float[] Infer(float[] obs)
    {
        var tensor = new DenseTensor<float>(obs, new[] { 1, obs.Length });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        try
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session!.Run(inputs);
            foreach (DisposableNamedOnnxValue result in results)
            {
                return result.AsEnumerable<float>().ToArray();
            }
        }
        catch (OnnxRuntimeException e)
        {
            GD.PushError($"[IsaacPolicyDriver] Inference failed: {e.Message}");
            _ready = false;
        }

        return System.Array.Empty<float>();
    }

    /// <summary>
    /// Prints the slices with sharp, independently-known rest-pose values. This is the cheapest
    /// instrument for the failure this whole path is exposed to: a permuted joint order or a
    /// flipped roll sign produces a plausible-looking vector and a body that merely moves wrongly.
    /// Gravity should read (0,0,-1) upright, pelvis height 0.82, joint positions ~0 at rest.
    /// </summary>
    public override void _ExitTree()
    {
        _session?.Dispose();
        _session = null;
    }
}

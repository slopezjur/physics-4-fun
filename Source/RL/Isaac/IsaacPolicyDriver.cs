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
    /// Remove Godot's Hill force-velocity derating for the duration of this run.
    ///
    /// <para><b>This is a plant correction, not a cheat, and it is the single largest measured
    /// difference between the two actuators.</b> `ActiveBone` scales its torque ceiling down as a
    /// joint turns in the direction it is being driven, reaching zero at
    /// <see cref="ActiveBone.MaxShorteningVelocity"/> (15 rad/s) - and when that scale hits zero
    /// `ActiveBone` sets the applied torque to exactly <c>Vector3.Zero</c>. The muscle switches
    /// off.</para>
    ///
    /// <para>Isaac has no such model. Its drives are a PD against a flat torque ceiling, which is
    /// what the policy was trained against, so leaving the derating on means driving the policy
    /// through a plant it has never seen. Measured in a Godot run of the stand check:
    /// <c>fvScale</c> falls to <b>0.00</b> by t=1.0 s while <c>demand</c> climbs to 2.23x the
    /// ceiling and <c>deliver</c> sits at 0.53 - the actuators are being asked for more than twice
    /// what they can give and are handing back nothing. Independently, sweeping the torque budget
    /// in Isaac puts Godot's effective authority at 0.4-0.5 of nominal, which is the same
    /// finding from the other side.</para>
    ///
    /// <para>The derating is a real biomechanical property and belongs in the Godot-native track.
    /// It just is not part of the contract an Isaac policy was trained against.</para>
    /// </summary>
    /// <summary>
    /// EMA smoothing applied to the joint velocity feeding each bone's Hill force-velocity law.
    /// 1 leaves it raw (Godot's default). ~0.15 filters solver chatter out of it.
    ///
    /// <para>Preferred over <see cref="DisableHillLimit"/>: it keeps the biomechanics and removes
    /// only the artifact. See <see cref="ActiveBone.HillVelocityFilterAlpha"/>.</para>
    /// </summary>
    [Export] public float HillVelocityFilter { get; set; } = 1.0f;

    /// <summary>
    /// EMA smoothing on the joint-velocity OBSERVATION, in (0,1]. 1 is raw. See
    /// <see cref="IsaacObservation.JointVelocityFilter"/> - Godot reports ~5 rad/s on a motionless
    /// standing body where Isaac reports ~0.2.
    /// </summary>
    [Export] public float JointVelocityFilter { get; set; } = 1.0f;

    [Export] public bool DisableHillLimit { get; set; }

    /// <summary>
    /// Remove Godot's gravity feed-forward for the duration of this run.
    ///
    /// <para>`ActiveBone` adds a `loadCompensation` term that supplies the steady-state torque
    /// needed to hold a joint against gravity, bounded at
    /// <see cref="ActiveBone.LoadCompensationTorqueFraction"/> (0.5) of the bone's ceiling, and then
    /// clamps `pd + loadCompensation` to that ceiling. **Isaac has no equivalent.** Its drives are a
    /// plain PD, so the policy learned to command poses whose own tracking error generates the
    /// holding torque - and in Godot that error is being cancelled by a term the policy does not
    /// know about, while up to half the budget it does control is reserved for it.</para>
    ///
    /// <para>Like <see cref="DisableHillLimit"/>, this is a plant correction rather than a tuning
    /// knob: the feed-forward is a good idea and belongs in the Godot-native track, it is simply not
    /// part of the contract an Isaac policy was trained against.</para>
    /// </summary>
    [Export] public bool DisableLoadCompensation { get; set; }

    /// <summary>
    /// Raise each bone's gains so its EFFECTIVE stiffness matches the rig contract, undoing the SPD
    /// denominator. 0 disables it; 1 targets the authored gain exactly.
    ///
    /// <para><b>The largest measured difference between the two actuators.</b> `PidController3D`
    /// uses the Tan-Liu-Turk SPD form and divides both gains by
    /// <c>1 + kd*dt/I + kp*dt^2/I</c>. Measured in the stand check, that denominator leaves the
    /// controlled bones applying <b>kEff = 0.15</b> - fifteen per cent of the authored proportional
    /// gain - while Isaac's XPBD applies the drive inside the solve at the full value. The policy
    /// therefore trained against joints several times stiffer than the ones it is driving.</para>
    ///
    /// <para>Compensating is safe in a way that raising a plain PD gain would not be: SPD's
    /// denominator grows with <c>kp</c>, so the loop stays stable as the gain rises. It is also
    /// CAPPED by the same algebra - as <c>kp'</c> tends to infinity the effective gain tends to
    /// <c>I/dt^2</c>, about 1440 N.m/rad for a 0.1 kg m^2 limb at 120 Hz, so a joint authored at
    /// 1800 cannot be fully recovered. This closes most of the gap, not all of it.</para>
    ///
    /// <para>Solving <c>kp' / (1 + kd*dt/I + kp'*dt^2/I) = target</c> for <c>kp'</c> gives
    /// <c>kp' = target * (1 + kd*dt/I) / (1 - target*dt^2/I)</c>, undefined once the target exceeds
    /// the cap - which is why the denominator is floored below.</para>
    /// </summary>
    [Export] public float GainCompensation { get; set; }

    /// <summary>
    /// Drive the policy's joint targets through the joint's OWN angular motors instead of adding
    /// external torque.
    ///
    /// <para><b>The structural difference between the two engines, not a tuning knob.</b> Isaac
    /// applies its drives inside the XPBD solve; `ActiveBone` can only add an external torque that
    /// Jolt then integrates. Every measurement pointed at that one fact: the commanded pose never
    /// reaches the body (<c>trackErr</c> 0.40 rad mean, 1.01 worst, immediately, with the dummy
    /// still upright at 0.79 m); effective stiffness is 110 N.m/rad against the 882 the rig
    /// authors, because `PidController3D`'s SPD denominator divides both gains by
    /// <c>1 + kd*dt/I + kp*dt^2/I</c>; and it cannot be fixed by raising gains, because as kp tends
    /// to infinity the effective gain tends to <c>I/dt^2</c> - measured at ~700, BELOW the authored
    /// 882.</para>
    ///
    /// <para>A Generic6DofJoint3D's angular motors are solved by Jolt as part of the constraint
    /// system, so they have no <c>I/dt^2</c> ceiling and no explicit-integration stability limit -
    /// the same class of actuator Isaac has. Implemented as a position servo on a velocity motor:
    /// <c>target_velocity = MotorServoGain * (target - current)</c>, bounded by
    /// <see cref="MotorMaxVelocity"/> and by each axis's effort limit from the rig contract.</para>
    ///
    /// <para><b>The rig's gains stop describing this plant.</b> A velocity servo is parameterised by
    /// a rate, not by N.m/rad, so `stiffness` and `damping` no longer apply - which is exactly why a
    /// policy wants retraining against it rather than being expected to transfer unchanged.</para>
    /// </summary>
    [Export] public bool JointMotorDrive { get; set; }

    /// <summary>Servo gain, rad/s of commanded joint rate per rad of angle error.</summary>
    [Export] public float MotorServoGain { get; set; } = 20.0f;

    /// <summary>Ceiling on the commanded joint rate, rad/s. Bounds the slew out of a large error.</summary>
    [Export] public float MotorMaxVelocity { get; set; } = 12.0f;

    /// <summary>
    /// Command rest on any joint axis whose half-range is below this, in radians. 0 disables it.
    /// See <see cref="IsaacActionSpace.LockAxesBelow"/> - an experiment against Godot's twist-axis
    /// chatter, not part of the contract.
    /// </summary>
    [Export] public float LockAxesBelow { get; set; }

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
    /// Leave Godot's procedural balance layer running and compose the policy's action on top of it,
    /// instead of taking the body over outright.
    ///
    /// <para><b>Why this exists.</b> `UpdateBoneTargetRotations` early-returns in the RL state, so
    /// entering it disconnects balance, trajectories and posture in one go and the policy inherits
    /// a body held up by nothing but joint targets. Isaac's body does not need that support - its
    /// rest pose is a genuine equilibrium, measured at 98.4% still standing after 8 seconds of
    /// all-zero actions, and still ~80% at a fifth of the stiffness. Godot's is not: the same
    /// command puts it on the floor in under 2 seconds. An Isaac-trained policy therefore arrives
    /// expecting passive stability that does not exist here.</para>
    ///
    /// <para>The original design worked this way. That early return's own comment notes the
    /// trajectory layer used to write a base pose which "its action only composes on top of", and
    /// it was removed because a standing target is unreachable for a PRONE get-up body. For Stand
    /// the body is already upright, so the base pose is appropriate rather than saturating.</para>
    ///
    /// <para><b>The honest caveat:</b> the policy is then driving a different plant from the one it
    /// was evaluated on. Zero action no longer means the rest pose, it means whatever the
    /// procedural layer commands. Treat a success here as "the brain contributes something useful
    /// on top of Godot's controller", not as a clean transfer.</para>
    /// </summary>
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
    private bool _gainsCompensated;
    private bool _dumpedPose;
    private float _elapsed;
    private string _inputName = "obs";
    private int _tick;
    private bool _ready;
    private bool _loggedFirstStep;
    private float _sinceDiagnostic;
    private float _warmupRemaining;
    private float _trackingError;
    private float _appliedTorque;

    /// <summary>Slice bounds from obs_action_contract.md §3, for the per-slice diagnostic.</summary>
    private static readonly (string Name, int From, int To)[] Slices =
    {
        ("gravity", 0, 3), ("linVel", 3, 6), ("angVel", 6, 9), ("height", 9, 10),
        ("jointPos", 10, 55), ("jointVel", 55, 100), ("contacts", 100, 104),
        ("prevAct", 104, 140), ("command", 140, 143),
    };

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
        };
        _actions = new IsaacActionSpace(_rig)
        {
            LockAxesBelow = LockAxesBelow,
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
        if (JointMotorDrive)
        {
            EnableJointMotors();
        }

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

        if (DisableHillLimit)
        {
            int lifted = 0;
            foreach (ActiveBone bone in Ragdoll!.GetBones())
            {
                if (!IsInstanceValid(bone))
                {
                    continue;
                }
                // Far above any rate a joint reaches, so ComputeForceVelocityScale stays at 1.
                bone.MaxShorteningVelocity = 1.0e6f;
                lifted++;
            }
            GD.Print($"[IsaacPolicyDriver] lifted the Hill force-velocity limit on {lifted} bone(s) - "
                     + "Isaac's drives have no velocity derating");
        }

        if (DisableLoadCompensation)
        {
            int cleared = 0;
            foreach (ActiveBone bone in Ragdoll!.GetBones())
            {
                if (!IsInstanceValid(bone))
                {
                    continue;
                }
                // Bounds the feed-forward at zero, which removes it without touching ActiveBone.
                bone.LoadCompensationTorqueFraction = 0.0f;
                cleared++;
            }
            GD.Print($"[IsaacPolicyDriver] removed the gravity feed-forward on {cleared} bone(s) - "
                     + "Isaac's drives are a plain PD");
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

    /// <summary>
    /// Zeroes <see cref="ActiveBone.MuscleStrength"/> on every bone the policy drives, so its own
    /// PD contributes nothing and the joint-space controller is the only thing applying torque.
    ///
    /// ActiveBone short-circuits at <c>MuscleStrength &lt;= 0.001f</c>, so this silences it outright
    /// rather than merely weakening it. Uncommanded bones keep their actuator: nothing else would
    /// hold the head and hands.
    /// </summary>
    /// <summary>
    /// Turns on the angular motors of every joint the policy drives, and silences ActiveBone on
    /// those bones so the motor is the only thing acting.
    ///
    /// <para>The force limit per axis comes from the rig contract's own <c>effort</c>, so the motor
    /// is bounded by the same ceiling the Isaac task enforces. Uncommanded bones keep their muscle:
    /// nothing else would hold the head and hands.</para>
    /// </summary>
    private void EnableJointMotors()
    {
        int enabled = 0;
        foreach (ActiveBone? bone in _controlledBones)
        {
            if (bone == null || !IsInstanceValid(bone) || bone.Joint == null || !IsInstanceValid(bone.Joint))
            {
                continue;
            }

            // The joint motor replaces the muscle rather than fighting it. ActiveBone short-circuits
            // at MuscleStrength <= 0.001, so this silences it outright.
            bone.MuscleStrength = 0.0f;

            int baseIndex = IndexOfControlledBone(bone);
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                char name = AxisName(axis);
                float effort = baseIndex >= 0
                    ? _rig!.ActuatedJoints[baseIndex + axis].Effort
                    : DefaultMotorForce;
                bone.Joint.Set($"angular_motor_{name}/enabled", true);
                bone.Joint.Set($"angular_motor_{name}/force_limit", effort);
                bone.Joint.Set($"angular_motor_{name}/target_velocity", 0.0f);
            }
            enabled++;
        }

        GD.Print($"[IsaacPolicyDriver] joint-motor drive: enabled motors on {enabled} joint(s), "
                 + "solved inside Jolt's constraint solver like Isaac's XPBD drives");
    }

    /// <summary>
    /// Position servo on each joint motor: commanded rate proportional to the remaining angle error.
    ///
    /// The error is measured the same way the observation measures joint angle - deviation from the
    /// rest pose, in Godot's sense - so the target the policy asked for and the angle it is compared
    /// against are the same quantity. Anything else reintroduces the frame confusion the DOF-order
    /// contract exists to prevent.
    /// </summary>
    private void DriveJointMotors()
    {
        if (_actions == null)
        {
            return;
        }

        for (int i = 0; i < _controlledBones.Length && i < _actions.TargetEuler.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone == null || !IsInstanceValid(bone) || bone.Joint == null || !IsInstanceValid(bone.Joint))
            {
                continue;
            }

            Vector3 current = IsaacObservation.DeviationFromRest(bone);
            Vector3 target = _actions.TargetEuler[i];

            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                char name = AxisName(axis);
                float error = Component(target, name) - Component(current, name);

                // **Negated: the motor's positive sense is opposite to `DeviationFromRest`.**
                // Measured rather than reasoned about - driving it unnegated took the tracking
                // error to 2.15 rad mean and 3.52 worst and folded the knee to its stop, while
                // negating gives 0.09/0.27 falling to 0.03/0.16. `DeviationFromRest` uses a
                // swing-twist decomposition of the child relative to its parent; Jolt's angular
                // motor drives body A relative to body B about the joint frame's axis, which is the
                // opposite convention. Nothing errors either way, and both look like "the servo is
                // running" in a log.
                float velocity = Mathf.Clamp(
                    -MotorServoGain * error, -MotorMaxVelocity, MotorMaxVelocity);
                bone.Joint.Set($"angular_motor_{name}/target_velocity", velocity);
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

    /// <summary>
    /// Per-DOF PD in joint space, the way Isaac drives the articulation.
    ///
    /// <code>tau = (kp * (target - current) - kd * rate) / (1 + kd*dt/I + kp*dt^2/I)</code>, clamped
    /// to the joint's effort limit, applied about the bone's own local axes and reacted onto the
    /// parent - the same action and reaction pair ActiveBone uses. Gains come from the rig contract,
    /// which is the same file Isaac's drives were built from.
    ///
    /// <para><b>The denominator is Tan-Liu-Turk SPD and it is mandatory, not a refinement.</b> The
    /// naive form <c>tau = kp*error - kd*rate</c> is what this method used first, and it is
    /// unconditionally unstable at this rig's gains. Isaac applies its drives INSIDE the XPBD
    /// solve - position-based, implicit, stable at any stiffness - while Godot can only add an
    /// external torque that Jolt then integrates explicitly. At the hip's kp=1800 with a limb
    /// inertia near 0.1 kg m^2, the undamped natural frequency is ~134 rad/s against a 120 Hz tick,
    /// so one step advances the oscillator by more than a radian of phase and it diverges.</para>
    ///
    /// <para>Measured with a ZERO action - every joint commanded to the rest pose it is already
    /// sitting in, which should cost no torque at all: all 36 actuators pinned at their effort
    /// limit (693 Nm = 400 x sqrt 3), relative joint rates of 91 rad/s, 1.35 rad of tracking error,
    /// and the dummy on the floor in under two seconds. That is the controller shaking the body
    /// apart on its own, with no policy involved.</para>
    ///
    /// <para>SPD is the standard explicit-integration equivalent of the implicit drive Isaac uses,
    /// and it is what <see cref="PidController3D"/> already runs for the Godot-native track, so
    /// both paths in this project now share one formulation.</para>
    /// </summary>
    /// <summary>
    /// Applies <see cref="GainCompensation"/>. Deferred to the first physics tick on purpose:
    /// `ActiveBone.LastEffectiveInertia` is only populated once the bone has run its muscle update,
    /// and reading it at load time gives zero.
    ///
    /// <para><b>Raises the proportional gain ONLY.</b> The first version scaled kp and kd together
    /// to preserve the damping ratio, which was self-defeating: the SPD denominator is
    /// <c>1 + kd*dt/I + kp*dt^2/I</c> and the kd term DOMINATES it - measured at 3.4 of a total
    /// 5.7 - so raising kd inflates the very denominator being compensated. It solved for a gain
    /// 222,229x the authored one and made the effective stiffness worse, 0.15 to 0.04.</para>
    ///
    /// <para><b>And the target is capped, because there is a hard ceiling.</b> As kp tends to
    /// infinity the effective gain tends to <c>I/dt^2</c> - measured at about 700 N.m/rad against an
    /// authored mean of 882, so the rig's stiffness is simply not reachable in Godot at 120 Hz by
    /// any gain. Asking for more than the ceiling makes the solved gain negative or infinite. This
    /// targets a fraction of the ceiling instead, which is honest about what the engine can do.</para>
    /// </summary>
    private void ApplyGainCompensation()
    {
        float dt = 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);
        int raised = 0;
        float authored = 0.0f;
        float achieved = 0.0f;

        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            if (!IsInstanceValid(bone) || bone.ProportionalGain <= 0.0f)
            {
                continue;
            }

            float inertia = Mathf.Max(1e-5f, bone.LastEffectiveInertia);
            float ceiling = inertia / (dt * dt);
            // Never chase the asymptote: at the ceiling the solved gain diverges, and the closer the
            // target the more violent the loop becomes for a vanishing return.
            float target = Mathf.Min(bone.ProportionalGain * GainCompensation, ceiling * CeilingFraction);

            float damping = 1.0f + (bone.DerivativeGain * dt / inertia);
            float headroom = 1.0f - (target * dt * dt / inertia);
            if (headroom <= 0.0f)
            {
                continue;
            }

            float solved = target * damping / headroom;
            authored += bone.ProportionalGain;
            achieved += target;
            if (solved > bone.ProportionalGain)
            {
                bone.ProportionalGain = solved;  // kd deliberately untouched - see the summary above
                raised++;
            }
        }

        if (raised > 0)
        {
            GD.Print($"[IsaacPolicyDriver] raised kp on {raised} bone(s) so effective stiffness goes "
                     + $"{achieved / raised:F0} against an authored {authored / raised:F0} "
                     + "- Isaac applies its drives at the full authored gain");
        }
    }

    /// <summary>
    /// Largest share of the <c>I/dt^2</c> ceiling <see cref="GainCompensation"/> will target.
    /// Approaching 1 sends the solved gain to infinity for no useful gain in stiffness.
    /// </summary>
    private const float CeilingFraction = 0.6f;

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

        if (GainCompensation > 0.0f && !_gainsCompensated)
        {
            _gainsCompensated = true;
            ApplyGainCompensation();
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
        if (JointMotorDrive)
        {
            DriveJointMotors();
        }
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
            LogStep(obs, actions);
        }

        if (DiagnosticInterval > 0.0f)
        {
            _sinceDiagnostic += delta * PhysicsTicksPerPolicyStep;
            if (_sinceDiagnostic >= DiagnosticInterval)
            {
                _sinceDiagnostic = 0.0f;
                LogSlices(obs, actions);
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
    private void LogStep(float[] obs, float[] actions)
    {
        float maxJoint = 0.0f;
        for (int i = 10; i < 55 && i < obs.Length; i++)
        {
            maxJoint = Mathf.Max(maxJoint, Mathf.Abs(obs[i]));
        }

        float maxAction = 0.0f;
        foreach (float a in actions)
        {
            maxAction = Mathf.Max(maxAction, Mathf.Abs(a));
        }

        GD.Print($"[IsaacPolicyDriver] first step: gravity=({obs[0]:F3},{obs[1]:F3},{obs[2]:F3}) "
                 + $"pelvisHeight={obs[9]:F3} maxJointPos={maxJoint:F3} maxAction={maxAction:F3} "
                 + $"worstJoint={WorstDof(obs, 10, 55)}");
    }

    /// <summary>
    /// Largest magnitude per observation slice, next to the resulting action magnitude. Reading the
    /// two together is the point: an action far outside [-1,1] with every slice in a plausible
    /// range means the mapping is wrong, while the same action alongside one slice reading orders
    /// of magnitude high means the policy is simply being shown something training never contained.
    /// </summary>
    /// <summary>
    /// Where the actuators' torque budget is actually going, across the controlled bones.
    ///
    /// <para>Sweeping the torque budget in Isaac locates Godot at an effective
    /// <c>effort_scale</c> of 0.4-0.5 - it delivers roughly HALF the authority its
    /// <see cref="ActiveBone.MaxTorque"/> numbers promise, even though those numbers are identical
    /// to the rig contract's <c>effort</c> values. This reports the candidates for the missing
    /// half, so the answer is measured rather than reasoned about:</para>
    ///
    /// <list type="bullet">
    /// <item><description><c>demand</c> - PD plus feed-forward, as a fraction of the bone's
    /// ceiling. Above 1.0 means the actuator is being asked for more than it can ever give.</description></item>
    /// <item><description><c>deliver</c> - what survived every clamp, same units. The gap between
    /// this and <c>demand</c> IS the missing authority.</description></item>
    /// <item><description><c>fvScale</c> - the Hill force-velocity derating. It falls as a joint
    /// moves fast, so a chattering body loses torque exactly when it needs it most.</description></item>
    /// </list>
    /// </summary>
    private string TorqueBudgetReport()
    {
        float demand = 0.0f;
        float deliver = 0.0f;
        float worstFv = 1.0f;
        int counted = 0;

        foreach (ActiveBone? bone in _controlledBones)
        {
            if (bone == null || !IsInstanceValid(bone))
            {
                continue;
            }

            float ceiling = bone.MaxTorque * bone.MuscleStrength;
            if (ceiling <= 0.0f)
            {
                continue;
            }

            demand = Mathf.Max(demand,
                (bone.LastPdTorque + bone.LastLoadCompensationTorque).Length() / ceiling);
            deliver = Mathf.Max(deliver, bone.LastAppliedTorque.Length() / ceiling);
            worstFv = Mathf.Min(worstFv, bone.LastForceVelocityScale);
            counted++;
        }

        return counted == 0
            ? string.Empty
            : $" demand={demand:F2} deliver={deliver:F2} fvScale={worstFv:F2}"
              + $" kEff={EffectiveGainFraction():F2}" + StiffnessReport() + TrackingReport() + BalanceReport();
    }

    /// <summary>
    /// What fraction of its authored proportional gain each joint actually applies, averaged.
    ///
    /// <para><b>This is the suspected home of the missing authority.</b> `ActiveBone` drives through
    /// `PidController3D`, which uses the Tan-Liu-Turk SPD form and divides BOTH gains by
    /// <c>1 + kd*dt/I + kp*dt^2/I</c>. For the knee - kp=1800, kd=36, dt=1/120 - a limb inertia near
    /// 0.1 kg m^2 gives a denominator around 5.25, so the effective stiffness is under a fifth of
    /// the authored value. Isaac's XPBD applies the drive inside the solve at the full gain, with no
    /// such division.</para>
    ///
    /// <para>It also explains a number that looked reassuring: <c>demand</c> reads only ~0.18 of the
    /// ceiling, which seemed to say the actuators were not even working hard. They are not - the
    /// denominator divided the request down before the ceiling ever came into it.</para>
    ///
    /// <para>Near 1.0 means Godot is applying what the rig contract says. Well below it means the
    /// policy is driving a much softer joint than the one it trained against, and that no amount of
    /// raising <see cref="ActiveBone.MaxTorque"/> will help, because the request never reaches the
    /// ceiling.</para>
    /// </summary>
    private float EffectiveGainFraction()
    {
        float total = 0.0f;
        int counted = 0;
        float dt = 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);

        foreach (ActiveBone? bone in _controlledBones)
        {
            if (bone == null || !IsInstanceValid(bone) || bone.ProportionalGain <= 0.0f)
            {
                continue;
            }

            float inertia = Mathf.Max(1e-5f, bone.LastEffectiveInertia);
            float denominator = 1.0f
                                + (bone.DerivativeGain * dt / inertia)
                                + (bone.ProportionalGain * dt * dt / inertia);
            total += 1.0f / denominator;
            counted++;
        }

        return counted == 0 ? 1.0f : total / counted;
    }

    /// <summary>
    /// Horizontal offset of the whole-body centre of mass from the midpoint between the feet, and
    /// the feet's height above the floor.
    ///
    /// <para><b>The balance question, which every joint-level diagnostic misses.</b> Zero-action
    /// traces show the two engines failing in different ways: Isaac SAGS - drops 8 cm, joints stay
    /// near rest, angular velocity near zero, holds for a second - while Godot TIPS, holding its
    /// height while angular velocity grows monotonically from the first sample. A topple means the
    /// centre of mass is leaving the support polygon, which is upstream of anything the actuator
    /// does.</para>
    ///
    /// <para>An offset well inside the foot span is a body that can sag but not fall over; one
    /// outside it is falling over regardless of how well the joints track.</para>
    /// </summary>
    private string BalanceReport()
    {
        if (Ragdoll == null || !IsInstanceValid(Ragdoll))
        {
            return string.Empty;
        }

        var com = Vector3.Zero;
        float mass = 0.0f;
        foreach (ActiveBone bone in Ragdoll.GetBones())
        {
            if (!IsInstanceValid(bone))
            {
                continue;
            }
            com += bone.GlobalPosition * bone.Mass;
            mass += bone.Mass;
        }

        if (mass <= 0.0f)
        {
            return string.Empty;
        }
        com /= mass;

        ActiveBone? left = Ragdoll.FindBone("Foot_L");
        ActiveBone? right = Ragdoll.FindBone("Foot_R");
        if (left == null || right == null || !IsInstanceValid(left) || !IsInstanceValid(right))
        {
            return string.Empty;
        }

        Vector3 mid = (left.GlobalPosition + right.GlobalPosition) * 0.5f;
        float offset = new Vector2(com.X - mid.X, com.Z - mid.Z).Length();
        float footHeight = Mathf.Min(left.GlobalPosition.Y, right.GlobalPosition.Y);
        float strength = 0.0f;
        int bones = 0;
        foreach (ActiveBone bone in Ragdoll.GetBones())
        {
            if (IsInstanceValid(bone))
            {
                strength += bone.MuscleStrength;
                bones++;
            }
        }

        return $" comOff={offset:F3}m feet={footHeight:F3}m mass={mass:F1}kg"
               + $" muscle={(bones > 0 ? strength / bones : 0.0f):F2}";
    }

    /// <summary>
    /// How far each joint sits from the angle the policy actually commanded, radians.
    ///
    /// <para>The question every other diagnostic dances around: <b>does the body ever adopt the pose
    /// the policy asked for?</b> Torque can be unclamped and gains can be whatever they are, but if
    /// the joints never reach their targets then the policy's intent is not reaching the body at
    /// all, and no amount of matching the actuator model will help.</para>
    ///
    /// <para>Reported as mean and worst across the controlled bones. Isaac's drives hold their
    /// targets closely once settled - joint velocity decays to 0.18 rad/s - so a large error here is
    /// a difference in kind, not degree.</para>
    /// </summary>
    private string TrackingReport()
    {
        if (_actions == null)
        {
            return string.Empty;
        }

        float total = 0.0f;
        float worst = 0.0f;
        int counted = 0;

        for (int i = 0; i < _controlledBones.Length && i < _actions.TargetEuler.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone == null || !IsInstanceValid(bone) || bone.ParentBone == null)
            {
                continue;
            }

            float error = (_actions.TargetEuler[i] - IsaacObservation.DeviationFromRest(bone)).Length();
            total += error;
            worst = Mathf.Max(worst, error);
            counted++;
        }

        return counted == 0 ? string.Empty : $" trackErr={total / counted:F2}/{worst:F2}rad";
    }

    /// <summary>
    /// Absolute effective stiffness per bone against the gain the rig contract authored, and the
    /// hard ceiling <c>I/dt^2</c> that no gain can exceed.
    ///
    /// <para>Reported because the FRACTION alone misleads once you try to compensate: raising
    /// <c>kp</c> also raises the SPD denominator, so the fraction falls while the absolute value
    /// barely moves. The ceiling is the number that matters - as <c>kp</c> tends to infinity the
    /// effective gain tends to <c>I/dt^2</c>, so a limb light enough at 120 Hz simply cannot be
    /// driven as stiffly as the contract asks, by any gain.</para>
    /// </summary>
    private string StiffnessReport()
    {
        float dt = 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);
        float authored = 0.0f;
        float effective = 0.0f;
        float ceiling = 0.0f;
        int counted = 0;

        foreach (ActiveBone? bone in _controlledBones)
        {
            if (bone == null || !IsInstanceValid(bone) || bone.ProportionalGain <= 0.0f)
            {
                continue;
            }

            float inertia = Mathf.Max(1e-5f, bone.LastEffectiveInertia);
            float denominator = 1.0f
                                + (bone.DerivativeGain * dt / inertia)
                                + (bone.ProportionalGain * dt * dt / inertia);
            authored += bone.ProportionalGain;
            effective += bone.ProportionalGain / denominator;
            ceiling += inertia / (dt * dt);
            counted++;
        }

        if (counted == 0)
        {
            return string.Empty;
        }
        return $" kp={authored / counted:F0}->{effective / counted:F0} ceiling={ceiling / counted:F0}";
    }

    /// <summary>
    /// Name and value of the largest-magnitude entry in an observation slice, as
    /// <c>Name:value</c>. The DOF order is the policy's own, so the index maps straight onto the
    /// contract's joint list and the answer is directly comparable with Isaac.
    /// </summary>
    private string WorstDof(float[] obs, int from, int to)
    {
        int worst = -1;
        float peak = -1.0f;
        for (int i = from; i < to && i < obs.Length; i++)
        {
            float magnitude = Mathf.Abs(obs[i]);
            if (magnitude > peak)
            {
                peak = magnitude;
                worst = i - from;
            }
        }

        if (worst < 0 || _rig == null || worst >= _rig.DofOrder.Count)
        {
            return "n/a";
        }

        IsaacRigContract.JointSpec spec = _rig.DofOrder[worst];
        return $"{spec.Bone}.{spec.GodotAxis}:{peak:F1}";
    }

    private void LogSlices(float[] obs, float[] actions)
    {
        var line = new System.Text.StringBuilder("[IsaacPolicyDriver] ");
        foreach ((string name, int from, int to) in Slices)
        {
            float peak = 0.0f;
            for (int i = from; i < to && i < obs.Length; i++)
            {
                peak = Mathf.Max(peak, Mathf.Abs(obs[i]));
            }
            line.Append($"{name}={peak:F2} ");
        }

        float maxAction = 0.0f;
        foreach (float a in actions)
        {
            maxAction = Mathf.Max(maxAction, Mathf.Abs(a));
        }
        line.Append($"| maxAction={maxAction:F2}");

        // Name the worst joint-velocity DOF, not just its magnitude.
        //
        // The slice peak alone cannot distinguish "the whole body is moving" from "one light distal
        // bone is chattering", and those need opposite fixes. Measured against Isaac running the
        // same policy at the same instant - Godot 33.90 rad/s against Isaac's 2.95, on a body still
        // standing at 0.81 m - the difference has to be localised before it can be explained.
        line.Append($" worstDof={WorstDof(obs, 55, 100)}");
        line.Append(TorqueBudgetReport());
        if (JointSpacePd)
        {
            line.Append($" trackErr={_trackingError:F3}rad torque={_appliedTorque:F0}Nm");
            _trackingError = 0.0f;
            _appliedTorque = 0.0f;
        }
        GD.Print(line.ToString());
    }

    public override void _ExitTree()
    {
        _session?.Dispose();
        _session = null;
    }
}

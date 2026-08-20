using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Actions;
using Physics4Fun.RL.Interfaces;
using Physics4Fun.RL.Observations;
using Physics4Fun.RL.Rewards;
using Physics4Fun.RL.Termination;

namespace Physics4Fun.RL;

/// <summary>
/// Bridges the godot_rl_agents obs/action/reward/done contract to <see cref="HumanoidRagdoll"/>,
/// without HumanoidRagdoll or ActiveBone knowing anything about RL exists.
///
/// The plugin's AIController3D is a GDScript base class, and Godot does not support a C# script
/// extending a GDScript one - so a thin GDScript adapter (RagdollAIController.gd) is required as
/// the actual node the plugin's Sync node talks to. That adapter forwards every call straight into
/// this class, which is where all the real logic lives, matching the rest of this project's
/// C#-first convention. See Scenes/RL/RagdollAIController.gd for the (intentionally minimal) other
/// half of this seam.
///
/// Control model: PURE RL. The policy owns every controlled joint's target outright - it is not
/// correcting a procedural pose. HumanoidRagdoll.UpdateBoneTargetRotations returns early for
/// RagdollState.ReinforcementLearning, and Balance/state-machine/debug-input already gate
/// themselves off, so no Euphoria behaviour reaches this dummy. What IS shared is the body: the
/// ActiveBone PD/SPD muscles and the Generic6DofJoint3D limits. The procedural track keeps running
/// untouched in TestChamber.tscn from the same scene file.
///
/// Timing: this node's PhysicsProcessPriority is set high in _Ready() so its _PhysicsProcess runs
/// AFTER HumanoidRagdoll's every tick, which is what lets the targets written here survive to be
/// read by ApplyBiomechanicalTorque on the following tick.
/// </summary>
public partial class RagdollRLBridge : Node
{
    [Export] public HumanoidRagdoll? Ragdoll { get; set; }

    /// <summary>
    /// The GDScript AIController3D-derived node (RagdollAIController.gd). Referenced only to poll
    /// its "needs_reset" field: godot_rl_agents sets that flag when Python explicitly sends a reset
    /// message (standard gymnasium semantics - the terminal (obs, reward, done=true) tuple for a
    /// fallen episode must reach Python before the physical reset happens, not before), and the
    /// framework leaves "who performs the actual reset" to the environment/player, which here is
    /// this bridge. Untyped Node, not the GDScript class, since C# cannot reference a GDScript type.
    /// </summary>
    [Export] public Node? AiController { get; set; }

    /// <summary>Bones the placeholder observation/action set reads and writes, in a fixed order.</summary>
    private static readonly string[] ControlledBoneNames =
    {
        "Spine", "Chest",
        "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
        "Thigh_L", "Shin_L", "Foot_L",
        "Thigh_R", "Shin_R", "Foot_R"
    };

    /// <summary>
    /// Episode length (s) after which it ends regardless of physical state. Without this, an
    /// episode only ends on the "fell upside down" condition below, which in practice never fires
    /// at all - so episodes (and therefore Python-visible reward, ep_rew_mean) could go arbitrarily
    /// long. This is also what makes the AIController3D's own generic reset_after safety valve
    /// irrelevant in practice: this timeout always fires first and, unlike that valve, is actually
    /// reported to Python as a real episode boundary.
    ///
    /// 8 s originally, cut to 3 s at 12M steps for sample composition: the body started standing,
    /// held the success criterion for about 0.26 s (standing/all = 0.032 of 968 ticks), fell, and
    /// then lay on the floor for the remaining ~7 s with its head at 29 cm - measured, from
    /// reward/shaping telescoping to 10 * (PHI_end - PHI_start). Only ~12 of 121 decision steps
    /// carried anything worth learning from, and shortening raised the informative share from ~10%
    /// to ~27% at unchanged throughput.
    ///
    /// Now 4 s, and the extra second buys something quite different from headroom. The cap is
    /// asymmetric: an episode that SUCCEEDS ends at roughly 0.6 s of settling plus
    /// StandingHoldSeconds, so at 1.5 s it finishes near 2.1 s and never touches the cap at all.
    /// Only failures run to the wall, which is where the whole cost of a longer window is paid.
    ///
    /// What 3 s cost was false negatives. With a 1.5 s hold the latest a hold can BEGIN and still
    /// complete inside a 3 s window is t = 1.5, against a policy that enters the standing region at
    /// ~0.6 s - 0.9 s of slack. An attempt entering at t = 1.6 gets truncated mid-hold and scored a
    /// failure despite being on track, and the scoring is invisible: the Inverted condition in
    /// GetUpTermination has never once fired in this project, so falling and timing out report the
    /// SAME terminal reason and cannot be told apart after the fact. 4 s moves that false-negative
    /// band from [1.5 s, inf) to [2.5 s, inf). Buy the margin, because there is no measurement that
    /// says whether it was needed.
    ///
    /// Not lower than 3 s, for two reasons that still hold. Truncated episodes are only worth as
    /// much as the value bootstrap that handles them, and shorter episodes truncate in states that
    /// still have real continuation value - see TruncationBootstrapWrapper in rl/train.py, which
    /// had to land with the 8 -> 3 change rather than after it. And godot_rl hands SB3 the terminal
    /// observation where it expects the reset one, so exactly one action per episode is conditioned
    /// on the previous episode's final pose; that costs 1/45 of samples at 3 s, 1/60 at 4 s, and
    /// 1/30 at 2 s.
    ///
    /// Applies to episodes that START STANDING. Prone starts use ProneMaxEpisodeSeconds instead -
    /// see there for why one constant cannot serve both.
    /// </summary>
    [Export] public float MaxEpisodeSeconds { get; set; } = 4.0f;

    /// <summary>
    /// Episode length (s) for episodes that begin prone.
    ///
    /// A get-up simply cannot happen in the standing window. The procedural reference is the scale
    /// to judge by: GetUpPhaseController runs five phases at PhaseBlendDuration = 0.55 s nominal
    /// and PhaseTimeout = 1.2 s worst case, so 2.75 s to 6.0 s before the body is even upright, and
    /// StandingHoldSeconds then needs another 1.5 s on top before success is declared. Capping a
    /// prone episode at the standing window would make the task unsolvable by construction rather
    /// than merely hard - the timer would always fire first, and every prone episode would be a
    /// guaranteed failure no matter how good the policy got.
    ///
    /// The standing window stays short for the opposite reason: those episodes finish in ~2.1 s, so
    /// anything beyond that is dead sampling time. Splitting the two is what lets each start pose
    /// get the window its task actually needs.
    ///
    /// Left at 8 s when StandingHoldSeconds went 0.75 -> 1.5, which spends half of the worst-case
    /// margin (6.0 + 1.5 = 7.5 against 8.0, down from 1.25 s of slack to 0.5 s). Deliberate: an RL
    /// get-up is under no obligation to follow the procedural phase timing, and widening this costs
    /// throughput on the half of episodes that currently always run to the wall at a 0% success
    /// rate. Revisit only once start_prone/success is nonzero AND those episodes cluster at the cap.
    /// </summary>
    [Export] public float ProneMaxEpisodeSeconds { get; set; } = 8.0f;

    /// <summary>
    /// Probability that an episode begins already standing rather than prone (Reference State
    /// Initialization).
    ///
    /// 1.0 starts every episode standing, 0.0 reproduces the original always-prone behaviour, and
    /// values between mix the two - which is the curriculum knob: begin at 1.0 so the policy first
    /// learns the goal state it will otherwise never visit, then lower it to introduce the harder
    /// starts.
    ///
    /// Rationale, from DeepMimic's own ablation: without reference-state initialization the agent
    /// "is unlikely to encounter states from a successful flip and never discovers such high
    /// reward states". Measured here as well - 63M steps and 524,000 always-prone episodes
    /// produced zero successes, because the terminal reward was never once observed.
    ///
    /// 0.5, lowered from the 1.0 that ran for the first 22.6M steps. At 1.0 the policy reached a
    /// 94% success rate - but on a 1.05 s episode of which 0.75 s is the mandatory hold, i.e. it
    /// learned "do not fall over in the first third of a second", never a get-up, because the prone
    /// start had no support at all. This reintroduces it.
    ///
    /// Half by EPISODE is far more than half by SAMPLE, which is what actually trains: a standing
    /// episode runs ~16 decision steps and a prone one ~120, so a 50/50 episode split puts ~88% of
    /// samples on the prone task while retaining the standing skill for the remaining ~12%. That
    /// retention matters because a get-up ENDS in standing - the two skills compose.
    ///
    /// SUPERSEDED by the reverse curriculum below, which samples a start pose from a continuum
    /// rather than picking one of two. Kept only as the floor of that continuum: an episode still
    /// starts exactly standing with this probability, so the goal-state skill keeps being refreshed
    /// no matter how far the curriculum has walked toward prone. Set it to 0 to hand every episode
    /// to the curriculum.
    /// </summary>
    [Export] public float StandingStartProbability { get; set; } = 0.2f;

    /// <summary>
    /// Hardest (lowest) start pose the curriculum currently samples, on the 0 = prone, 1 = standing
    /// scale of <see cref="HumanoidRagdoll.TeleportToGetUpPose"/>. Episodes draw uniformly from
    /// [CurriculumPoseT, 1].
    ///
    /// Walks DOWN toward 0 as the agent clears each level. That direction is the whole idea: the
    /// get-up's terminal state is the one thing already learned (87% success on a 1.5 s hold as of
    /// 32M steps), so the curriculum extends backwards from a solved goal instead of forwards from
    /// an unsolved start. Attempting it earlier would not have worked - at 36% the goal state was
    /// not solid enough to build on.
    ///
    /// Starts just BELOW 1.0, not at it. At exactly 1.0 the sampling range [1, 1] is a single point,
    /// so every episode starts standing, no episode counts as a curriculum episode, no outcome is
    /// ever recorded, and the level can never advance - a curriculum that is silently frozen at the
    /// one pose already solved. _Ready clamps this below 1.0 for the same reason.
    /// </summary>
    [Export] public float CurriculumPoseT { get; set; } = 0.99f;

    /// <summary>
    /// How much closer to prone each advance moves the floor.
    ///
    /// 0.01, not the 0.1 this was first written with, because the pose parameter is severely
    /// non-linear in difficulty. The start pose is a rigid tilt, so the CoM moves off the support
    /// polygon as comHeight * sin(pitch); with the CoM near the 0.82 m pelvis rest height and
    /// GetUpTermination allowing 0.15 m of ICP escape, the body is at its tipping point at
    ///
    ///     asin(0.15 / 0.90) = 9.6 degrees, i.e. t = 0.893.
    ///
    /// So every pose the balance controller can recover from lives in t = [0.893, 1.0] - the first
    /// 11% of the range - and everything below it is already falling. At 0.1 the first rung landed
    /// on t = 0.90, CoM 0.141 m against a 0.15 m limit: exactly the tipping point, dropped from a
    /// standstill. Measured over ~1,300 episodes it scored 0.000 while exact-standing scored 0.50,
    /// and because a level only advances ON SUCCESS the curriculum would have been frozen there for
    /// the entire run, looking healthy the whole time.
    ///
    /// At 0.01 the rungs are ~0.9 degrees apart, which spans the recoverable band in ~11 levels.
    /// Expect the curriculum to stall near 0.89 - that stall is meaningful, not a bug: it is where
    /// a rigid tilt stops being a balance problem and becomes a get-up, which needs a change of
    /// support (hands, knees) that no whole-body rotation can express. See TeleportToGetUpPose.
    /// </summary>
    [Export] public float CurriculumStep { get; set; } = 0.01f;

    /// <summary>
    /// Success rate over the trailing window that promotes the curriculum to the next level.
    ///
    /// 0.7 rather than something near 1.0 deliberately: waiting for near-perfection at every level
    /// spends most of training polishing poses that are not the goal, and the levels are not
    /// independent - competence at 0.5 keeps improving while the agent works on 0.4.
    /// </summary>
    [Export] public float CurriculumAdvanceRate { get; set; } = 0.7f;

    /// <summary>
    /// Episodes of history the advance decision reads. 20 at ~45 standing episodes per rollout means
    /// roughly two decisions per rollout at most, so the level cannot run away inside one update.
    /// </summary>
    [Export] public int CurriculumWindow { get; set; } = 20;

    /// <summary>
    /// Share of curriculum episodes drawn from the FRONTIER band [floor, floor + step] rather than
    /// uniformly over the whole cleared range [floor, 1].
    ///
    /// Without this the curriculum outruns the policy, which is exactly what the first run did:
    /// uniform sampling from [0.955, 1] has a mean pose of 0.978, so the advance window was
    /// dominated by easy rehearsal episodes and cleared the 70% gate while the frontier itself was
    /// at 0.5% (pose_0.96) and 0.0% (pose_0.95). The floor kept descending past levels the agent
    /// could not do at all, which looks like progress and is not.
    ///
    /// It also fixes a sampling-rate problem that would have bitten later. The frontier is one step
    /// wide out of a range of (1 - floor), so under uniform sampling its share SHRINKS as the
    /// curriculum descends: 22% of draws at floor 0.955, but 2% at floor 0.5. Voting would have
    /// slowed to a crawl exactly when the levels got interesting. A fixed share keeps the decision
    /// rate constant no matter how far down the curriculum has walked.
    /// </summary>
    [Export] public float CurriculumFrontierShare { get; set; } = 0.5f;

    /// <summary>
    /// Whether reaching the standing criterion ends the episode and pays the terminal bonus.
    ///
    /// True for the get-up task. **Set false in the perturbation scene**: there the episode would
    /// otherwise end at ~2.1 s, before the first ball even lands, and the hold requirement would
    /// fight the ball interval. See GetUpTermination for the full argument.
    /// </summary>
    [Export] public bool EndEpisodeOnStandingSuccess { get; set; } = true;

    /// <summary>
    /// Optional perturbation source (the ball gun) polled for episode telemetry only.
    ///
    /// Typed as Node, not BallGun: Godot exports must be Godot types, and more importantly this
    /// bridge is deliberately ignorant of what perturbs it. It casts to IRlPerturbationDiagnostics
    /// at report time, so a scene without a gun - or with a different one - needs no change here,
    /// and nothing about the perturbation reaches the policy through this path.
    /// </summary>
    [Export] public Node? PerturbationSource { get; set; }

    /// <summary>
    /// Effort-penalty weight handed to the reward. Near zero during discovery; see
    /// GetUpProgressReward's constructor for why deployment-strength regularization blocks it.
    /// </summary>
    [Export] public float EffortWeight { get; set; } = 0.02f;

    /// <summary>
    /// Playback: run the policy continuously instead of in episodes.
    ///
    /// Episodes exist for TRAINING - a learner needs bounded, comparable rollouts, and the time
    /// limit guarantees a boundary the optimiser can see. When simply watching a trained policy
    /// none of that applies, and the timer becomes actively harmful: it teleports the dummy back
    /// to the start pose every MaxEpisodeSeconds, including at the exact moment it might have been
    /// about to stand.
    ///
    /// With this set the episode never terminates on its own, so the body attempts to get up and
    /// then holds whatever it achieved until the scene is stopped (or reset manually with R).
    /// Set by PolicyAutoLoader when it switches Sync into ONNX inference.
    /// </summary>
    public bool PlaybackMode { get; set; }

    public int ActionSize => _actions.Size;

    /// <summary>Observation width, delegated to the active builder and published at handshake.</summary>
    public int ObservationSize => _observations.Size;

    // Composed strategies rather than inlined logic: reward shaping and termination are the parts
    // of an RL setup that get rewritten constantly, and keeping them behind interfaces means that
    // work never touches the transport contract or the actuator seam below. Swap these to
    // experiment (see docs/RL-DESIGN-NOTES.md); the rest of this class stays untouched.
    private IRlActionSpace _actions = null!;
    private IRlObservationBuilder _observations = null!;
    private IRlRewardFunction _reward = null!;
    private IRlTerminationCondition _termination = null!;

    /// <summary>Elapsed time (s) in the current episode. Exposed for HUD display.</summary>
    public float EpisodeElapsedSeconds { get; private set; }

    /// <summary>1-based count of episodes started so far this session. Exposed for HUD/log display.</summary>
    public int EpisodeCount { get; private set; }

    /// <summary>
    /// Reward accumulated so far this EPISODE.
    ///
    /// Deliberately not <c>_accumulatedReward</c>, which is the undelivered balance owed to the
    /// trainer and is zeroed by <see cref="DrainReward"/> on every decision step - reading that
    /// as an episode total silently reports one step's worth instead, which is what the HUD and
    /// the episode log line were both doing.
    /// </summary>
    public float CurrentAccumulatedReward => _episodeRewardTotal;

    /// <summary>Total reward the previous episode accumulated before it ended. Exposed for HUD/log display.</summary>
    public float LastEpisodeReward { get; private set; }

    /// <summary>Duration (s) of the previous episode. Exposed for HUD/log display.</summary>
    public float LastEpisodeDurationSeconds { get; private set; }

    /// <summary>Why the previous episode ended ("Fallen" or "TimeLimit"). Exposed for HUD/log display.</summary>
    public string LastEpisodeEndReason { get; private set; } = "None";

    /// <summary>
    /// This episode's actual time limit - <see cref="MaxEpisodeSeconds"/> for standing starts,
    /// <see cref="ProneMaxEpisodeSeconds"/> for prone ones. Exposed so the HUD shows the window
    /// the episode is really running against instead of the standing one it used to hardcode.
    /// </summary>
    public float EffectiveMaxEpisodeSeconds => _effectiveMaxEpisodeSeconds;

    /// <summary>Whether the current episode began standing (vs prone). Exposed for HUD display.</summary>
    public bool StartedStanding => _startedStanding;

    /// <summary>This episode's start pose, 0 = prone to 1 = standing. Exposed for HUD display.</summary>
    public float StartPoseT => _startPoseT;

    /// <summary>Hardest start pose the curriculum has reached so far. Exposed for HUD display.</summary>
    public float ActiveCurriculumT => _activeCurriculumT;

    private ActiveBone?[] _controlledBones = System.Array.Empty<ActiveBone?>();




    private Quaternion[] _pendingOffsets = System.Array.Empty<Quaternion>();
    /// <summary>Reward owed to the trainer since the last DrainReward - NOT an episode total.</summary>
    private float _accumulatedReward;

    /// <summary>
    /// Episode-total reward, accumulated in parallel with <see cref="_accumulatedReward"/> but
    /// never drained. Kept as an independent sum rather than derived from the reward function's
    /// own per-term totals, so that "terms sum to total" stays a real cross-check of both sides
    /// rather than an identity that cannot fail.
    /// </summary>
    private float _episodeRewardTotal;

    private bool _done;
    private string _doneReason = "None";
    private bool _resetPending;

    /// <summary>Which pose the current episode began from - reported so runs stay interpretable.</summary>
    private bool _startedStanding;

    /// <summary>This episode's start pose on the 0 = prone, 1 = standing scale.</summary>
    private float _startPoseT = 1.0f;

    /// <summary>The curriculum floor actually in use; initialised from CurriculumPoseT, then walked down.</summary>
    private float _activeCurriculumT = 1.0f;

    /// <summary>
    /// Trailing success/failure history for the advance decision, oldest first.
    ///
    /// Per-process rather than shared: the 32 training instances each run their own curriculum off
    /// their own results. They cannot see each other without a transport change, and the divergence
    /// is mild (they receive the same policy every rollout, so their success rates track closely)
    /// and arguably useful - a spread of levels in the batch is a wider start-state distribution,
    /// which is what the curriculum is for in the first place.
    /// </summary>
    private readonly System.Collections.Generic.Queue<bool> _curriculumHistory = new();
    private bool _physicsRateApplied;

    /// <summary>
    /// This episode's time limit, chosen from the start pose in ResetEpisode.
    ///
    /// Resolved once per episode rather than per tick so it cannot change underneath a
    /// running episode - the termination check compares against it every tick, and a value
    /// that shifted mid-episode would truncate or extend an attempt already in progress.
    /// </summary>
    private float _effectiveMaxEpisodeSeconds = 4.0f;
    /// <summary>Remaining times GetStepInfo reports the full config before going quiet.</summary>
    private int _configReportsRemaining = 5;

    /// <summary>
    /// Simulation rate (Hz) the ragdoll's PD/SPD gains were tuned at - see project.godot, which
    /// sets 120. Restoring it here is what keeps a learned policy transferable to the procedural
    /// track: at a different rate the actuators have measurably different dynamics.
    /// </summary>
    private const float TunedPhysicsHz = 120.0f;

    public override void _Ready()
    {
        // Godot's default physics process priority is 0 for every node; a higher number runs
        // later in the same tick. See the class summary for why "later than HumanoidRagdoll" is
        // required here, not merely convenient.
        ProcessPhysicsPriority = 100;

        // Seeded from the export here rather than at the field, so a scene or a resumed run can set
        // the starting level and the runtime value still has somewhere separate to walk down to.
        //
        // Capped strictly below 1.0: at exactly 1.0 the sample range collapses to a point, every
        // episode starts standing, nothing is ever recorded as a curriculum outcome and the level
        // can never advance. That failure is silent - the run looks healthy and simply never
        // progresses - so it is clamped here rather than left to whoever edits the scene.
        _activeCurriculumT = Mathf.Clamp(CurriculumPoseT, 0.0f, 1.0f - CurriculumStep);

        _pendingOffsets = new Quaternion[ControlledBoneNames.Length];
        for (int i = 0; i < _pendingOffsets.Length; i++)
        {
            _pendingOffsets[i] = Quaternion.Identity;
        }

        if (Ragdoll == null)
        {
            GD.PushError("[RagdollRLBridge] Ragdoll is not assigned in the inspector.");
            return;
        }

        CacheControlledBones();

        _actions = new JointLimitedActionSpace(ControlledBoneNames);
        _actions.Bind(_controlledBones);
        _observations = new GetUpObservation(ControlledBoneNames.Length);
        _reward = new GetUpProgressReward(EffortWeight);
        _termination = new GetUpTermination(EndEpisodeOnStandingSuccess);

        // Enters the RL state and takes the first episode's starting pose immediately - without
        // this the ragdoll would sit in RagdollState.Balanced (the project default) until the
        // first "needs_reset" arrives from Python, and get_obs() would be observing the wrong state.
        ResetEpisode();
    }

    private void CacheControlledBones()
    {
        _controlledBones = new ActiveBone?[ControlledBoneNames.Length];
        var byName = new System.Collections.Generic.Dictionary<string, ActiveBone>();
        foreach (var bone in Ragdoll!.GetBones())
        {
            byName[bone.BoneName] = bone;
        }

        for (int i = 0; i < ControlledBoneNames.Length; i++)
        {
            if (byName.TryGetValue(ControlledBoneNames[i], out var bone))
            {
                _controlledBones[i] = bone;
            }
            else
            {
                GD.PushError($"[RagdollRLBridge] Bone '{ControlledBoneNames[i]}' not found on Ragdoll.");
            }
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (Ragdoll == null || Ragdoll.Balance == null)
        {
            return;
        }

        EnsurePhysicsTickRate();

        // Write this tick's action out to the actuators (see class summary for the timing
        // requirement this ordering satisfies), then accumulate reward and check the done
        // condition against the state the action just produced won't be reflected in until next
        // tick - both computed off the CURRENT physical state, which is what a step-based RL loop
        // actually wants (reward for the state reached, not the state about to be commanded).
        if (CheckAndConsumeResetRequest())
        {
            return;
        }

        // A terminal step that Sync has already reported to Python (see ClearDone) is reset here,
        // one tick later. Deferring by a tick is what guarantees the TERMINAL observation is the
        // one Python receives for that step - resetting the instant the done condition fires would
        // teleport the body first and hand Python the fresh episode's pose labelled as the old
        // episode's final state.
        if (_resetPending)
        {
            _resetPending = false;
            ResetEpisode();
            return;
        }

        ApplyPendingOffsets();
        AccumulateReward((float)delta);
        EpisodeElapsedSeconds += (float)delta;
        UpdateDone((float)delta);
    }

    /// <summary>
    /// Polls the GDScript adapter's "needs_reset" (set once Python explicitly requests a reset -
    /// see the AiController property doc) and, if set, performs the actual physical reset and
    /// clears it by calling back into the adapter's reset(). Skips the rest of this tick's action/
    /// reward/done work when a reset just happened, since it would be operating on the pre-reset
    /// pose for one stray frame otherwise.
    /// </summary>
    private bool CheckAndConsumeResetRequest()
    {
        if (AiController == null || !GodotObject.IsInstanceValid(AiController))
        {
            return false;
        }

        if (!(bool)AiController.Get("needs_reset"))
        {
            return false;
        }

        ResetEpisode();
        AiController.Call("reset");
        return true;
    }

    /// <summary>
    /// Called by the GDScript adapter's set_done_false() override, which Sync invokes right after
    /// it has read done=true and sent the terminal (obs, reward, done) tuple to Python.
    ///
    /// That makes this the correct - and only - safe place to schedule the physical reset. The
    /// framework deliberately leaves resetting to the environment: godot_rl's Python side never
    /// sends a reset in response to a done (GodotEnv.reset() is an explicit call SB3 makes once at
    /// startup), and sync.gd's own _reset_agents_if_done() is commented out for training mode. So
    /// if the bridge does not reset itself here, nothing does - the done condition stays true and
    /// every subsequent step is reported terminal, collapsing ep_len_mean to 1.
    /// </summary>
    public void ClearDone()
    {
        _done = false;
        _resetPending = true;
    }

    /// <summary>
    /// Writes the policy's joint targets straight onto the actuators.
    ///
    /// Under pure RL this is the ONLY thing setting TargetLocalRotation for these bones -
    /// HumanoidRagdoll.UpdateBoneTargetRotations returns early for this state - so the policy owns
    /// the pose completely rather than nudging a procedural one. Targets compose as
    /// rest * action, matching the convention the procedural path uses for its own trajectories,
    /// so "zero action" means "rest pose" rather than some arbitrary origin.
    ///
    /// Uncontrolled bones (pelvis, head, hands) keep whatever target the episode reset seeded, so
    /// they hold their pose passively instead of going limp.
    /// </summary>
    private void ApplyPendingOffsets()
    {
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone != null && GodotObject.IsInstanceValid(bone))
            {
                bone.TargetLocalRotation = (bone.GetRestLocalRotation() * _pendingOffsets[i]).Normalized();
            }
        }
    }

    private void AccumulateReward(float delta)
    {
        float tickReward = _reward.Evaluate(BuildContext(delta));
        _accumulatedReward += tickReward;
        _episodeRewardTotal += tickReward;
    }

    private void UpdateDone(float delta)
    {
        if (_done)
        {
            return;
        }

        // Never terminate during playback - see PlaybackMode. Reward still accumulates so the HUD
        // keeps showing a live figure; it simply is not used to end anything.
        if (PlaybackMode)
        {
            return;
        }

        ActiveBone? pelvis = Ragdoll!.Pelvis;
        if (pelvis == null || !GodotObject.IsInstanceValid(pelvis))
        {
            return;
        }

        RlContext context = BuildContext(delta);
        if (_termination.IsTerminal(context, out string reason))
        {
            _done = true;
            _doneReason = reason;

            // Paid once, here, rather than per tick - UpdateDone returns early while _done is
            // already set, so this cannot double-pay within an episode.
            float terminalReward = _reward.EvaluateTerminal(context, reason);
            _accumulatedReward += terminalReward;
            _episodeRewardTotal += terminalReward;
        }
    }

    /// <summary>
    /// Restores the physics tick rate the ragdoll was tuned for.
    ///
    /// godot_rl_agents' Sync node sets Engine.physics_ticks_per_second = speedup * 60 and
    /// time_scale = speedup, so the effective simulation step is 1/60 s at ANY speedup - half the
    /// 120 Hz this project configures and tuned its SPD gains against. Left alone, an RL policy
    /// would learn against different actuator dynamics than the procedural controller experiences,
    /// and behaviour would not transfer between the two tracks.
    ///
    /// Derived from Engine.TimeScale (which Sync has already set to the speedup) rather than
    /// hardcoded, so it stays correct at any --speedup. Deferred to the first physics tick because
    /// Sync writes the rate during its own _ready, which can run after this node's.
    /// </summary>
    private void EnsurePhysicsTickRate()
    {
        if (_physicsRateApplied)
        {
            return;
        }

        _physicsRateApplied = true;
        float speedup = Mathf.Max(1.0f, (float)Engine.TimeScale);
        Engine.PhysicsTicksPerSecond = (int)(speedup * TunedPhysicsHz);
        GD.Print($"[RagdollRLBridge] Physics tick rate set to {Engine.PhysicsTicksPerSecond} Hz "
                 + $"(speedup {speedup:F0}, effective dt 1/{TunedPhysicsHz:F0}s).");
    }

    /// <summary>
    /// Snapshots the per-tick state the strategies read. Built fresh each call rather than cached:
    /// it is a small readonly struct passed by reference, and a stale context would silently feed
    /// last tick's physics into this tick's reward.
    /// </summary>
    private RlContext BuildContext(float delta)
    {
        HumanoidRagdoll ragdoll = Ragdoll!;
        return new RlContext(
            ragdoll,
            ragdoll.Pelvis!,
            ragdoll.Balance!,
            _controlledBones,
            EpisodeElapsedSeconds,
            _effectiveMaxEpisodeSeconds,
            delta);
    }

    /// <summary>
    /// Called by the GDScript adapter's get_obs(). Delegates to the active observation builder;
    /// returns a correctly-sized zero vector if the rig is not ready, since a wrong-width vector
    /// would desynchronise the transport rather than merely losing one frame of data.
    /// </summary>
    public float[] ComputeObservations()
    {
        if (Ragdoll?.Pelvis == null || Ragdoll.Balance == null)
        {
            return new float[_observations.Size];
        }

        return _observations.Build(BuildContext(0.0f));
    }

    /// <summary>Called by the GDScript adapter's set_action(). Length must equal <see cref="ActionSize"/>.</summary>
    public void ApplyAction(float[] action)
    {
        if (action.Length != ActionSize)
        {
            GD.PushError($"[RagdollRLBridge] Action length {action.Length} != expected {ActionSize}.");
            return;
        }

        _actions.Decode(action, _pendingOffsets);

        // A real action arrived: the ragdoll can stop idling soft and take on the full impedance
        // an actual policy needs to press/balance with. See HumanoidRagdoll.ReinforcementLearningPolicyActive.
        if (Ragdoll != null)
        {
            Ragdoll.ReinforcementLearningPolicyActive = true;
        }
    }

    /// <summary>Called by the GDScript adapter's get_reward(). Returns and clears the accumulated reward.</summary>
    public float DrainReward()
    {
        float value = _accumulatedReward;
        _accumulatedReward = 0.0f;
        return value;
    }

    /// <summary>
    /// Called by the GDScript adapter's get_info(). Carries two distinct payloads, both of which
    /// ride the info channel because it is the only per-transition side channel the plugin
    /// exposes: the run's provenance (once, at the start) and the reward decomposition (once per
    /// episode, on the terminal transition).
    ///
    /// This is the provenance record for a training run: the Python trainer knows its own
    /// hyperparameters but nothing about the reward, termination, action range or bone set
    /// compiled into this build, and those are exactly the parts that change between experiments.
    /// Each strategy describes itself (see IRlRewardFunction.Describe) rather than the values
    /// being duplicated on the Python side, so the manifest cannot drift from the running code.
    ///
    /// Reported for the first few calls rather than every step (it is constant, and sync sends
    /// info on every transition for every parallel env), but not exactly once either: the very
    /// first call is consumed by the initial reset(), whose info a trainer may never inspect. A
    /// handful of repeats guarantees it lands in at least one step() the trainer does look at,
    /// while staying negligible against a 106-float observation.
    /// </summary>
    public Godot.Collections.Dictionary GetStepInfo()
    {
        var info = new Godot.Collections.Dictionary();

        // Emitted only on the terminal transition: the totals are per-episode, so any earlier tick
        // would report a partial sum that means nothing. Keys are prefixed so the Python side can
        // pick them out without knowing which terms this particular reward happens to define.
        if (_done && _reward is IRlRewardDiagnostics diagnostics)
        {
            foreach (var component in diagnostics.EpisodeComponentTotals)
            {
                info[$"rew_{component.Key}"] = component.Value;
            }

            info["rew_total"] = _episodeRewardTotal;
            info["episode_end_reason"] = _doneReason;
            info["started_standing"] = _startedStanding ? 1.0f : 0.0f;
            info["start_pose_t"] = _startPoseT;
            info["curriculum_t"] = _activeCurriculumT;

            // Whether the ball actually connected. Without it a good balance score is ambiguous:
            // "recovered from the hit" and "was never hit" are indistinguishable in every other
            // metric, and shots CAN miss (randomised direction plus AimHeightJitter).
            if (PerturbationSource is IRlPerturbationDiagnostics perturbation)
            {
                foreach (var stat in perturbation.EpisodePerturbationStats)
                {
                    info[$"ball_{stat.Key}"] = stat.Value;
                }
            }
            if (_actions is IRlActionDiagnostics actionDiagnostics)
            {
                foreach (var stat in actionDiagnostics.EpisodeActionStats)
                {
                    info[$"act_{stat.Key}"] = stat.Value;
                }
            }

            // Which part of the success check held, and how often. Turns "Standing never fired"
            // from an unexplained fact into a located one.
            if (_termination is IRlTerminationDiagnostics terminationDiagnostics)
            {
                foreach (var condition in terminationDiagnostics.EpisodeConditionRates)
                {
                    info[$"stand_{condition.Key}"] = condition.Value;
                }
            }
        }

        if (_configReportsRemaining <= 0)
        {
            return info;
        }

        _configReportsRemaining--;
        foreach (var entry in new Godot.Collections.Dictionary
        {
            { "controlled_bones", string.Join(",", ControlledBoneNames) },
            { "action_size", ActionSize },
            { "observation_size", ObservationSize },
            { "action_space", _actions.Describe() },
            { "max_episode_seconds", MaxEpisodeSeconds },
            { "prone_max_episode_seconds", ProneMaxEpisodeSeconds },
            { "standing_start_probability", StandingStartProbability },
            { "curriculum_pose_t", CurriculumPoseT },
            { "curriculum_step", CurriculumStep },
            { "curriculum_advance_rate", CurriculumAdvanceRate },
            { "curriculum_window", CurriculumWindow },
            { "curriculum_frontier_share", CurriculumFrontierShare },
            { "end_episode_on_standing_success", EndEpisodeOnStandingSuccess },
            { "effort_weight", EffortWeight },
            { "physics_ticks_per_second", Engine.PhysicsTicksPerSecond },
            { "engine_time_scale", Engine.TimeScale },
            { "observation", _observations.Describe() },
            { "reward", _reward.Describe() },
            { "termination", _termination.Describe() },
            { "control_model", "pure_rl_absolute_targets" }
        })
        {
            info[entry.Key] = entry.Value;
        }

        return info;
    }

    /// <summary>
    /// True when this process has no display, i.e. it is one of the headless training workers
    /// rather than the single --viz instance (or the editor).
    /// </summary>
    /// <summary>godot_rl's base port - sync.gd DEFAULT_PORT, which rl/train.py offsets per process.</summary>
    private const int GodotRlBasePort = 11008;

    /// <summary>
    /// True on exactly one process per training batch: the instance holding godot_rl's base port.
    ///
    /// Per-episode logging has to be gated on something. Training runs 32 processes that all reach
    /// the episode boundary within a few ticks of each other, so an ungated print produces 32
    /// near-identical lines every few seconds - unreadable, and 32 processes doing synchronous
    /// stdout writes is not free either.
    ///
    /// It used to be gated on "has a display" (DisplayServer.GetName() != "headless"), which
    /// silently made the episode log a --viz-only feature: resume.ps1 and train.ps1 launch every
    /// instance headless, so the lines vanished from the normal training path entirely. Instance
    /// identity is the property actually wanted, and it is available - rl/train.py hands process p
    /// the port GodotEnv.DEFAULT_PORT + p, and sync.gd reads that same --port=N argument. So the
    /// process holding the base port is process 0, window or no window.
    ///
    /// Not a behaviour change for --viz: visible_count is 1, so show_window = (p &lt; 1) makes
    /// process 0 both the visible instance AND the base-port one. The same single process logs as
    /// before; the headless path simply stops being excluded.
    ///
    /// No --port argument at all means the trainer did not launch this - the arena scene, or F5 in
    /// the editor - which is a single instance that should always log.
    /// </summary>
    private static bool IsPrimaryInstance()
    {
        const string prefix = "--port=";
        foreach (string arg in OS.GetCmdlineArgs())
        {
            if (arg.StartsWith(prefix, System.StringComparison.Ordinal))
            {
                return int.TryParse(arg[prefix.Length..], out int port) && port == GodotRlBasePort;
            }
        }

        return true;
    }

    /// <summary>Called by the GDScript adapter's get_done().</summary>
    public bool IsDone() => _done;

    /// <summary>
    /// Draws this episode's start pose: exactly standing with probability StandingStartProbability,
    /// otherwise uniform over [_activeCurriculumT, 1].
    ///
    /// Uniform over the whole cleared range rather than only at the current floor, so earlier levels
    /// keep being rehearsed. Sampling the floor alone is the classic way to make a curriculum forget
    /// what it just learned - the policy is free to trade away competence at 0.7 while specialising
    /// on 0.5, and nothing measures the loss until the level it needs stops working.
    /// </summary>
    private float SampleStartPose()
    {
        if (GD.Randf() < StandingStartProbability)
        {
            return 1.0f;
        }

        // Frontier band: the one step the curriculum is actually trying to clear. Only these vote
        // (see IsFrontierPose), so their share has to be held fixed rather than left to shrink as
        // the range widens.
        if (GD.Randf() < CurriculumFrontierShare)
        {
            return Mathf.Lerp(_activeCurriculumT, Mathf.Min(1.0f, _activeCurriculumT + CurriculumStep), GD.Randf());
        }

        // Rehearsal over everything already cleared, so competence at earlier levels is not traded
        // away while the frontier is being learned.
        return Mathf.Lerp(_activeCurriculumT, 1.0f, GD.Randf());
    }

    /// <summary>
    /// Whether a start pose sits in the band the curriculum is currently trying to clear, and so
    /// counts toward the advance decision. Rehearsal and standing-refresher episodes are excluded:
    /// they are drawn from levels already passed, so letting them vote measures the past.
    /// </summary>
    private bool IsFrontierPose(float poseT)
        => poseT < 1.0f && poseT <= _activeCurriculumT + CurriculumStep;

    /// <summary>
    /// Feeds one curriculum episode's outcome to the trailing window and advances the level when the
    /// window clears CurriculumAdvanceRate.
    ///
    /// The window is CLEARED on advance, not slid. A slid window still holds the successes that
    /// triggered the promotion, so it would re-trigger within an episode or two and run the level
    /// down several steps before the policy has seen the new distribution at all. Clearing forces a
    /// full window of fresh evidence before the next move.
    /// </summary>
    private void RecordCurriculumOutcome(bool succeeded)
    {
        if (_activeCurriculumT <= 0.0f)
        {
            return;
        }

        _curriculumHistory.Enqueue(succeeded);
        while (_curriculumHistory.Count > CurriculumWindow)
        {
            _curriculumHistory.Dequeue();
        }

        if (_curriculumHistory.Count < CurriculumWindow)
        {
            return;
        }

        int wins = 0;
        foreach (bool outcome in _curriculumHistory)
        {
            if (outcome) { wins++; }
        }

        int considered = _curriculumHistory.Count;
        if ((float)wins / considered < CurriculumAdvanceRate)
        {
            return;
        }

        float previous = _activeCurriculumT;
        _activeCurriculumT = Mathf.Max(0.0f, _activeCurriculumT - CurriculumStep);
        _curriculumHistory.Clear();

        // Printed even on headless instances, unlike the per-episode line: this is the run's actual
        // progress signal, it fires a handful of times per session at most, and a curriculum that
        // silently stops advancing is the thing worth noticing early.
        GD.Print(
            $"[RagdollRLBridge] Curriculum advanced {previous:F2} -> {_activeCurriculumT:F2} "
            + $"({wins}/{considered} at the previous level, episode {EpisodeCount})");
    }

    /// <summary>
    /// Called by the GDScript adapter's reset() override. Physically resets the ragdoll and clears
    /// this bridge's per-episode state.
    /// </summary>
    public void ResetEpisode()
    {
        // Capture the outgoing episode's stats before anything is cleared, so the HUD and the log
        // line below can report on the episode that just ended rather than the fresh one.
        bool hadPriorEpisode = EpisodeCount > 0;
        if (hadPriorEpisode)
        {
            LastEpisodeReward = _episodeRewardTotal;
            LastEpisodeDurationSeconds = EpisodeElapsedSeconds;
            LastEpisodeEndReason = _doneReason;

            // Only FRONTIER episodes vote - see IsFrontierPose. Letting rehearsal or refresher
            // episodes count is what let the first run's floor descend to 0.955 while pose_0.955
            // itself scored 0.000.
            if (IsFrontierPose(_startPoseT))
            {
                RecordCurriculumOutcome(_doneReason == "Standing");
            }
        }
        EpisodeCount++;

        // Cleared before the teleport, not after: StartReinforcementLearning() itself recomputes
        // muscle stiffness, and a fresh episode should start idling soft (see
        // HumanoidRagdoll.ReinforcementLearningPolicyActive) until its own first action arrives,
        // not inherit "active" from whatever the previous episode left behind.
        if (Ragdoll != null)
        {
            Ragdoll.ReinforcementLearningPolicyActive = false;
        }
        // DropToProne's own per-call print would spam the console every few seconds once training
        // is running; the single consolidated line below covers both halves of a reset instead.
        _startPoseT = SampleStartPose();
        _startedStanding = _startPoseT >= 1.0f;

        // The window scales with the pose, not with a standing/prone flag. A start halfway up needs
        // more than the standing window (there is a rise to perform first) and less than the prone
        // one (most of the rise is already done), and giving every non-standing pose the full 8 s
        // would spend most of the curriculum's samples on episodes that ended long before the timer.
        _effectiveMaxEpisodeSeconds = Mathf.Lerp(ProneMaxEpisodeSeconds, MaxEpisodeSeconds, _startPoseT);

        Ragdoll?.StartReinforcementLearning(silent: true, startPoseT: _startPoseT);

        // One process logs, not all 32 - see IsPrimaryInstance for why it is keyed on the port
        // rather than on having a display.
        if (IsPrimaryInstance())
        {
            if (hadPriorEpisode)
            {
                GD.Print(
                    $"[RagdollRLBridge] Episode {EpisodeCount - 1} ended ({LastEpisodeEndReason}, "
                    + $"reward={LastEpisodeReward:F2}, duration={LastEpisodeDurationSeconds:F1}s) "
                    + $"-> starting episode {EpisodeCount}");
            }
            else
            {
                GD.Print($"[RagdollRLBridge] Starting episode {EpisodeCount}.");
            }
        }

        _actions.Reset();
        _reward.Reset();
        _termination.Reset();
        _accumulatedReward = 0.0f;
        _episodeRewardTotal = 0.0f;
        _done = false;
        _doneReason = "None";
        EpisodeElapsedSeconds = 0.0f;
        for (int i = 0; i < _pendingOffsets.Length; i++)
        {
            _pendingOffsets[i] = Quaternion.Identity;
        }
    }
}

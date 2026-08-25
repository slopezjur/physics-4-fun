using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Actions;
using Physics4Fun.RL.Interfaces;
using Physics4Fun.RL.Observations;
using Physics4Fun.RL.Rewards;
using Physics4Fun.RL.Termination;

namespace Physics4Fun.RL;

/// <summary>
/// Which task the bridge builds its reward and termination for.
///
/// Only the reward and termination vary. The observation and action space are deliberately shared
/// across every task, because that is what makes a policy trained on one task resumable on
/// another: the tensor widths are published to Python at handshake and baked into the checkpoint,
/// so changing them turns a resume into a from-scratch run. Walking is trained by restoring a
/// standing policy, which is only possible because both tasks see the same 113 floats.
///
/// The balance/perturbation task is not listed: it uses the get-up components with
/// <see cref="RagdollRLBridge.EndEpisodeOnStandingSuccess"/> set false, so it needs no separate
/// value here and adding one would imply a component pair that does not exist.
/// </summary>
public enum RlTaskKind
{
    /// <summary>Stand up from prone, or stay standing. Also used for the ball-perturbation task.</summary>
    GetUp,

    /// <summary>Walk forward in a straight line from a standing start.</summary>
    Walk,
}

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
    /// UprightTermination has never once fired in this project, so falling and timing out report the
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
    /// UprightTermination allowing 0.15 m of ICP escape, the body is at its tipping point at
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
    /// fight the ball interval. See UprightTermination for the full argument.
    /// </summary>
    [Export] public bool EndEpisodeOnStandingSuccess { get; set; } = true;

    /// <summary>
    /// Which reward and termination pair the bridge builds. See <see cref="RlTaskKind"/>.
    ///
    /// Defaults to GetUp so every existing scene keeps its current behaviour without being touched
    /// - the walk scenes are the only ones that set it.
    /// </summary>
    [Export] public RlTaskKind TaskKind { get; set; } = RlTaskKind.GetUp;

    /// <summary>
    /// Chance a STANDING-START episode also fires the ball gun, making it a perturbation episode.
    /// 0 (the default) keeps every scene single-task and unchanged.
    ///
    /// This is what makes one brain out of the three Upright tasks instead of three that overwrite
    /// each other. They already share every component - the same observation, action space, reward
    /// and termination - and differ only in start pose and whether something pushes back. What they
    /// did NOT share was a training run, and training them in sequence measurably destroys the
    /// earlier one: a walk run took standing/all from 0.478 to 0.170, and a perturbation run took it
    /// from 0.480 to 0.459 while learning nothing itself.
    ///
    /// Sampling the task per EPISODE instead removes the problem by construction. Nothing is ever
    /// left unpracticed for long enough to be forgotten, and the balance episodes stop degrading the
    /// standing skill they depend on.
    ///
    /// Only standing starts are eligible. A ball arriving while the dummy is halfway through getting
    /// up is a third, harder task nobody asked for, and it would make a fall impossible to attribute
    /// to either the get-up or the impact.
    /// </summary>
    [Export] public float PerturbationEpisodeProbability { get; set; } = 0.0f;

    /// <summary>
    /// Maximum forward speed (m/s) the whole body is given at episode start. 0 disables.
    ///
    /// This is Reference State Initialization aimed at an EXPLORATION barrier rather than at a
    /// starting pose, and it exists because the first walk run failed in a specific, measurable
    /// way. Resumed from a policy that could already stand, the agent converged within 5 minutes
    /// to standing perfectly still: walk/fell fell 0.164 -> 0.011 while walk/forward_speed fell
    /// 0.0186 -> 0.0069. It did not fail to learn - it learned the wrong thing, and learned it
    /// quickly.
    ///
    /// The reward weights were not the problem. Standing still scores 12 over a 6 s episode and
    /// walking at target speed scores 48, so the destination was already four times better. The
    /// barrier is that the PATH between them descends: taking a real step risks a fall (-5 plus the
    /// whole remaining episode), while creeping forward pays almost nothing, since the velocity
    /// term is proportional to speed. Raising the destination value cannot fix a barrier that
    /// exploration never reaches, which is why VelocityWeight was left alone.
    ///
    /// Starting the body already moving removes the local optimum by CONSTRUCTION instead of by
    /// re-weighting. Standing still is no longer available: the momentum has to go somewhere, and
    /// the only ways to resolve it are to step or to fall. Stepping is then discovered because it
    /// is necessary, not because it is worth slightly more.
    ///
    /// Applied uniformly to every bone via HumanoidRagdoll.GetBones, as a rigid translation of the
    /// whole body. Seeding only the pelvis would leave the limbs behind and have the joints fight
    /// the discrepancy for the first few ticks, which reads as a shove rather than as walking onto
    /// the scene at speed.
    ///
    /// KEEP THIS AT OR BELOW WalkForwardReward.TargetSpeed. The scenes carried 0.8 for three runs
    /// after TargetSpeed was cut 1.0 -> 0.4, so every episode launched the body at 1.2x to 2x the
    /// speed it was paid for, into a speed factor that saturates at 0.4 - excess momentum earning
    /// nothing and costing balance. walk_v3 shows the result over 19M steps: walk/fell pinned at
    /// 1.000 for all 2326 logged points, walk/forward_speed 0.375 (BELOW the 0.48 m/s floor of the
    /// launch, i.e. decelerating all episode), walk/distance 0.94 m of pure coast, and
    /// standing/grounded DECLINING 0.818 -> 0.769 as the gait became more ballistic rather than
    /// less. Posture was never the problem - standing/upright held 0.999 throughout - the body
    /// simply ran out of runway 2.2 s in, every single time.
    /// </summary>
    [Export] public float InitialForwardSpeed { get; set; } = 0.0f;

    /// <summary>
    /// Fraction of <see cref="InitialForwardSpeed"/> below which an episode never starts, so the
    /// speed is sampled from [floor, 1] * InitialForwardSpeed rather than being a single value.
    ///
    /// A spread rather than a constant because a policy trained on exactly one entry speed learns a
    /// fixed catch, and a fixed catch is not a gait. The floor stays well above zero: sampling near
    /// zero would restore the standing-still option on those episodes and reopen the optimum this
    /// setting exists to close.
    /// </summary>
    [Export] public float InitialForwardSpeedFloor { get; set; } = 0.6f;

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
    /// UprightProgressReward's constructor for why deployment-strength regularization blocks it.
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
    public float ActiveCurriculumT => _curriculum.Floor;

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

    /// <summary>
    /// Whether the episode that just ended met its success criterion, latched at termination.
    ///
    /// Distinct from _doneReason because the two diverge on every non-absorbing task: a balance
    /// episode that survives its hit succeeds AND ends as "TimeLimit". See
    /// IRlTerminationCondition.SucceededThisEpisode.
    /// </summary>
    private bool _episodeSucceeded;

    private bool _resetPending;

    /// <summary>Which pose the current episode began from - reported so runs stay interpretable.</summary>
    private bool _startedStanding;

    /// <summary>This episode's start pose on the 0 = prone, 1 = standing scale.</summary>
    private float _startPoseT = 1.0f;

    /// <summary>The curriculum floor actually in use; initialised from CurriculumPoseT, then walked down.</summary>
    /// <summary>
    /// Start-pose curriculum. Constructed in _Ready from the exports below, because Godot requires
    /// [Export] to live on the Node while the policy itself does not belong here - see
    /// IRlStartPoseCurriculum.
    /// </summary>
    private IRlStartPoseCurriculum _curriculum =
        new Curriculum.ReverseStartPoseCurriculum(1.0f, 0.01f, 0.5f, 20, 0.7f);

    /// <summary>
    /// Trailing success/failure history for the advance decision, oldest first.
    ///
    /// Per-process rather than shared: the 32 training instances each run their own curriculum off
    /// their own results. They cannot see each other without a transport change, and the divergence
    /// is mild (they receive the same policy every rollout, so their success rates track closely)
    /// and arguably useful - a spread of levels in the batch is a wider start-state distribution,
    /// which is what the curriculum is for in the first place.
    /// </summary>
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
        // The curriculum floor is RESTORED from the trainer when resuming, not reset to the scene
        // constant.
        //
        // This was the single most damaging bug in the RL track. The floor lives in the Godot
        // process while the policy lives in the SB3 checkpoint, so every resume restored the weights
        // and then handed all 32 fresh processes a floor of CurriculumPoseT again. Training happens
        // in sessions, so the curriculum descended a little, the session ended, and the progress was
        // thrown away - every time. getup_v4_2 shows it exactly: min=0.9080, max=0.9900,
        // last=0.9900. It reached 0.908 and went back to the start.
        //
        // That is why start_prone/success read 0.000 across 46.7M steps. Prone was never hard; the
        // curriculum was never allowed to REACH it. A 60-second test descends 0.99 -> 0.96 on its
        // own, so descent rate was never the constraint either.
        //
        // rl/train.py passes --curriculum_start=<t> when resuming, taken from the MINIMUM floor the
        // restored run ever reached (see _last_curriculum_floor). Minimum rather than last because
        // the floor only ever descends within a session, so the lowest value is the honest
        // high-water mark - and the last logged value is often a post-restart reset.
        float? restoredFloor = CmdlineArgs.ReadFloat("curriculum_start");
        _curriculum = new Curriculum.ReverseStartPoseCurriculum(
            restoredFloor ?? CurriculumPoseT, CurriculumStep, CurriculumFrontierShare,
            CurriculumWindow, CurriculumAdvanceRate);
        if (restoredFloor.HasValue && IsPrimaryInstance())
        {
            GD.Print($"[RagdollRLBridge] Curriculum floor restored to {_curriculum.Floor:F3} "
                     + $"(scene default was {CurriculumPoseT:F3}).");
        }

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
        _observations = new BodyStateObservation(ControlledBoneNames.Length);
        // Only these two vary by task. The observation and action space above are shared on
        // purpose - see RlTaskKind for why that is what makes cross-task resuming possible.
        (_reward, _termination) = TaskKind switch
        {
            RlTaskKind.Walk => ((IRlRewardFunction)new WalkForwardReward(EffortWeight),
                                (IRlTerminationCondition)new WalkTermination()),
            _ => (new UprightProgressReward(EffortWeight),
                  new UprightTermination(EndEpisodeOnStandingSuccess)),
        };

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

            // Captured here rather than read back later, mirroring _doneReason: the termination's
            // per-episode state is cleared by _termination.Reset() in ResetEpisode, and both the
            // info dict and the curriculum vote are consumed after that point.
            _episodeSucceeded = _termination.SucceededThisEpisode;

            // Paid once, here, rather than per tick - UpdateDone returns early while _done is
            // already set, so this cannot double-pay within an episode.
            float terminalReward = _reward.EvaluateTerminal(context, reason, _episodeSucceeded);
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
    /// while staying negligible against a 113-float observation.
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

            // Success as its own channel rather than something the trainer re-derives from the end
            // reason. That derivation is what broke: train.py counted reason == "Standing", a string
            // the perturbation task never emits, so every per-task and per-pose success rate read
            // 0.000 across whole runs regardless of what the policy actually did.
            info["episode_success"] = _episodeSucceeded ? 1.0f : 0.0f;
            // Which task this episode was. The whole point of mixing is that no task degrades while
            // another trains, and that is only checkable if success is reported PER TASK - an
            // aggregate rate cannot distinguish "all three improving" from "balance improving while
            // standing rots", which is exactly the failure mixing exists to prevent.
            info["episode_task"] = EpisodeTask;
            info["started_standing"] = _startedStanding ? 1.0f : 0.0f;
            info["start_pose_t"] = _startPoseT;
            info["curriculum_t"] = _curriculum.Floor;

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
            // How far and how fast, in metres. Its own prefix because these have UNITS: they are
            // neither reward terms (which must sum to the episode reward) nor rates in [0,1], and
            // "walked 0.4 m" versus "walked 4 m" is the entire question for that task.
            if (_reward is IRlWalkDiagnostics walkDiagnostics)
            {
                foreach (var stat in walkDiagnostics.EpisodeWalkStats)
                {
                    info[$"walk_{stat.Key}"] = stat.Value;
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
            { "task_kind", TaskKind.ToString() },
            { "end_episode_on_standing_success", EndEpisodeOnStandingSuccess },
            { "effort_weight", EffortWeight },
            { "physics_ticks_per_second", Engine.PhysicsTicksPerSecond },
            { "engine_time_scale", Engine.TimeScale },
            { "observation", _observations.Describe() },
            { "reward", _reward.Describe() },
            { "termination", _termination.Describe() },
            { "curriculum", _curriculum.Describe() },
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
    /// otherwise whatever the curriculum has unlocked.
    ///
    /// The curriculum samples over its whole cleared range rather than only at the current floor,
    /// so earlier levels keep being rehearsed. Sampling the floor alone is the classic way to make a
    /// curriculum forget what it just learned - the policy is free to trade away competence at 0.7
    /// while specialising
    /// on 0.5, and nothing measures the loss until the level it needs stops working.
    /// </summary>
    /// <summary>Which of the Upright tasks this episode is, for per-task metrics.</summary>
    public string EpisodeTask { get; private set; } = "stand";

    /// <summary>
    /// Decides whether this episode is stand, getup or perturbation, and configures the two things
    /// that differ: whether the ball fires, and whether success ends the episode.
    ///
    /// Success absorption is tied to the ball rather than exported separately, because the two are
    /// not independent. UprightTermination requires StandingHoldSeconds of CONTINUOUS success, and a
    /// ball resets that counter on every hit - so "hold 1.5 s uninterrupted" and "get hit" are
    /// mutually exclusive by construction, and leaving absorption on would make a perturbation
    /// episode unwinnable no matter how good the policy is. Worse, without the ball the episode
    /// would end at roughly 2.1 s, before the first shot at 1.0 s had any chance to matter.
    /// </summary>
    private void SelectEpisodeTask()
    {
        bool mixing = PerturbationEpisodeProbability > 0.0f;
        bool hasGun = PerturbationSource != null && IsInstanceValid(PerturbationSource);

        // With mixing OFF the gun is owned by the scene, so whether this is a perturbation episode
        // is decided by whether the gun is armed - not by a roll that never happens. Reading it
        // rather than assuming: the single-task perturbation scene fires on every episode and was
        // reporting all 128 of them as task_stand, which is a metric quietly describing the wrong
        // experiment.
        bool gunArmed = hasGun && PerturbationSource!.Get("Enabled").AsBool();

        bool perturbation = _startedStanding
                            && (mixing ? GD.Randf() < PerturbationEpisodeProbability : gunArmed);

        EpisodeTask = perturbation ? "perturbation" : (_startedStanding ? "stand" : "getup");

        if (_termination is UprightTermination upright)
        {
            upright.EndEpisodeOnSuccess = EndEpisodeOnStandingSuccess && !perturbation;

            // Failure terminates whenever the episode began upright, independently of whether
            // success does. A prone get-up start is below the fall thresholds by definition, so it
            // must not arm this or every get-up episode would end on its first tick.
            upright.EndEpisodeOnFall = _startedStanding;
        }

        // Touched ONLY when mixing is on. Otherwise the scene owns its own gun, and writing to it
        // here would silently re-arm a gun a single-task scene had deliberately disabled.
        //
        // Set() by name rather than casting to BallGun, so the bridge stays ignorant of what
        // perturbs it - the same reason PerturbationSource is typed as Node.
        if (PerturbationEpisodeProbability > 0.0f
            && PerturbationSource != null && IsInstanceValid(PerturbationSource))
        {
            PerturbationSource.Set("Enabled", perturbation);
        }
    }

    /// <summary>
    /// Gives the whole body a forward velocity at episode start. See <see cref="InitialForwardSpeed"/>.
    ///
    /// Uses the pelvis facing rather than world -Z so it agrees with WalkForwardReward and
    /// WalkTermination, which capture the same axis on their first tick. If the three disagreed the
    /// body would be launched along one axis and scored along another.
    /// </summary>
    private void ApplyInitialForwardVelocity()
    {
        if (InitialForwardSpeed <= 0.0f || Ragdoll == null)
        {
            return;
        }

        ActiveBone? pelvis = Ragdoll.Pelvis;
        if (pelvis == null || !IsInstanceValid(pelvis))
        {
            return;
        }

        Vector3 facing = -pelvis.GlobalTransform.Basis.Z;
        facing.Y = 0.0f;
        if (facing.LengthSquared() <= 1e-6f)
        {
            return;
        }

        float speed = InitialForwardSpeed
                      * Mathf.Lerp(Mathf.Clamp(InitialForwardSpeedFloor, 0.0f, 1.0f), 1.0f, GD.Randf());
        Vector3 velocity = facing.Normalized() * speed;

        foreach (ActiveBone bone in Ragdoll.GetBones())
        {
            if (IsInstanceValid(bone))
            {
                // Assigned, not added. StartReinforcementLearning has just teleported every bone
                // and zeroed its velocity, so there is nothing to preserve - and adding would let a
                // stray residual from the previous episode survive the reset.
                bone.LinearVelocity = velocity;
            }
        }
    }

    /// <summary>
    /// Draws this episode's start pose: a standing refresher with StandingStartProbability, else
    /// whatever the curriculum has unlocked.
    ///
    /// The standing draw stays here rather than moving into the curriculum because it is task
    /// composition, not curriculum policy - the perturbation and stand scenes set it to 1.0 and
    /// never touch a curriculum at all.
    /// </summary>
    private float SampleStartPose()
        => GD.Randf() < StandingStartProbability ? 1.0f : _curriculum.SampleStartPose();

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
            if (_curriculum.IsFrontierPose(_startPoseT))
            {
                // Votes on the success FLAG, not the end reason. Under the old string test the
                // perturbation task could never cast a winning vote, so the floor sat frozen at
                // 0.99 for the whole of v10 - a curriculum that cannot observe success cannot move.
                CurriculumAdvance advance = _curriculum.RecordOutcome(_episodeSucceeded);
                if (advance.Advanced)
                {
                    // Printed even on headless instances, unlike the per-episode line: this is the
                    // run's actual progress signal, it fires a handful of times per session at
                    // most, and a curriculum that silently stops advancing is the thing worth
                    // noticing early.
                    GD.Print(
                        $"[RagdollRLBridge] Curriculum advanced {advance.PreviousFloor:F2} -> "
                        + $"{advance.NewFloor:F2} ({advance.Wins}/{advance.Considered} at the "
                        + $"previous level, episode {EpisodeCount})");
                }
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
        ApplyInitialForwardVelocity();
        SelectEpisodeTask();

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
        _episodeSucceeded = false;
        EpisodeElapsedSeconds = 0.0f;
        for (int i = 0; i < _pendingOffsets.Length; i++)
        {
            _pendingOffsets[i] = Quaternion.Identity;
        }
    }
}

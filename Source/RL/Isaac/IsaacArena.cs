using Godot;
using Physics4Fun.Ragdoll;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// Watch an Isaac Lab policy drive the dummy. The Isaac-side answer to "press F5 on an arena
/// scene": no training, no server, no episode, nothing that resets the body.
///
/// <para>Builds its contents in code rather than being hand-authored as a .tscn, for the same
/// reason <see cref="IsaacParityTest"/> does: the driver needs a node reference to the ragdoll, and
/// wiring that by NodePath in a scene file is exactly the kind of pin that goes stale silently -
/// the multi-agent arena on the Godot-native track already lost thirty-nine of forty bodies to one
/// such pin. Resolving it by search cannot drift.</para>
///
/// <para>Reuses <c>TestChamber.tscn</c> for the floor, lighting and camera. That scene also carries
/// a <c>ProjectileShooter</c> and <c>PhysicsGrabber</c>, which is a feature here rather than a
/// leak: they are the manual equivalent of the Perturbation task, so a stand policy can be poked
/// by hand to see what it recovers from.</para>
/// </summary>
public partial class IsaacArena : Node3D
{
    /// <summary>Which brain to watch. Point this at any of the exported policies.</summary>
    [Export(PropertyHint.File, "*.onnx")]
    public string PolicyPath { get; set; } = "res://isaac_lab/exported/stand_policy.onnx";

    /// <summary>Rig contract to resolve joint order and limits against. Must match the policy.</summary>
    [Export] public string RigContractPath { get; set; } = IsaacRigContract.DefaultPath;

    /// <summary>
    /// The policy's own contract, written beside the .onnx by isaac_lab_3/scripts/export.py.
    /// REQUIRED for an Isaac Lab 3 / Newton policy - it carries the observation DOF ordering, which
    /// differs from the rig's physx_dof_order in 42 of 45 slots. Leave empty for a 2.3.2 policy.
    /// </summary>
    [Export(PropertyHint.File, "*.json")] public string PolicyContractPath { get; set; } = string.Empty;

    /// <summary>Derive contact flags from bone height rather than real contact. Set for Newton policies.</summary>
    [Export] public bool HeightContacts { get; set; }

    /// <summary>
    /// Seconds the driver lets the body settle before it starts acting. Forwarded to
    /// <see cref="IsaacPolicyDriver.WarmupSeconds"/>; -1 keeps the driver's own default.
    ///
    /// <para><b>Set it to 0 for a Stand policy.</b> The whole premise of this rig is that it
    /// collapses in under two seconds with nothing driving it, so any warmup hands the policy a
    /// body that has already fallen - measured at 32 degrees of tilt and a pelvis at 0.686 m by the
    /// end of a 1 s warmup, which is far outside anything the policy saw in training. A warmup is
    /// only meaningful for a body that is stable when passive.</para>
    /// </summary>
    [Export] public float WarmupSeconds { get; set; } = -1.0f;

    /// <summary>Scene supplying floor, light and camera.</summary>
    [Export] public string ChamberPath { get; set; } = "res://Scenes/TestChamber.tscn";

    /// <summary>
    /// Velocity command written into the observation's last three slots: (x forward, y lateral,
    /// z yaw rate). Leave at zero for Stand and Perturbation; Walk and Run were trained reading it,
    /// so those want a forward component.
    ///
    /// <para>Walk's sampled training ranges are x in [-0.3, 1.0] m/s, y in [-0.3, 0.3] m/s and yaw
    /// in [-0.5, 0.5] rad/s (see `WalkEnvCfg`). This comment previously said "+/-0.6 m/s", which
    /// matched no task - commanding outside the trained range asks for behaviour the policy never
    /// saw, and a wrong constant in a doc comment is exactly how the 18-vs-22.5 N.s ball error
    /// propagated through this track.</para>
    /// </summary>
    [Export] public Vector3 Command { get; set; } = Vector3.Zero;

    /// <summary>
    /// Seconds of free fall before the policy takes over. Zero starts it immediately.
    ///
    /// Useful above zero for telling two failure modes apart: a body that collapses the instant the
    /// policy engages is being driven wrongly, while one that collapses identically with and
    /// without the policy is not being driven at all.
    /// </summary>
    [Export] public float PolicyDelaySeconds { get; set; }

    /// <summary>Passed to the driver: command rest instead of the policy. See IsaacPolicyDriver.ZeroActionBaseline.</summary>
    [Export] public bool ZeroAction { get; set; }

    /// <summary>Passed to the driver. See IsaacPolicyDriver.DumpPoseAfterSeconds.</summary>
    [Export] public float DumpPoseAfterSeconds { get; set; }

    /// <summary>Passed to the driver. See IsaacPolicyDriver.BalanceAssist.</summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float BalanceAssist { get; set; }

    /// <summary>Passed to the driver. See IsaacPolicyDriver.ActionScaleOverride.</summary>
    [Export] public float ActionScaleOverride { get; set; }

    [Export] public bool AssistMode { get; set; }

    /// <summary>Passed to the driver. See IsaacPolicyDriver.AssistAuthority.</summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float AssistAuthority { get; set; } = 1.0f;

    /// <summary>Passed to the driver: drive joints Isaac-style. See IsaacPolicyDriver.JointSpacePd.</summary>
    [Export] public bool JointSpacePd { get; set; }

    /// <summary>Passed to the driver. See IsaacPolicyDriver.HillVelocityFilter.</summary>
    [Export] public float HillVelocityFilter { get; set; } = 1.0f;

    /// <summary>Passed to the driver. See IsaacPolicyDriver.JointVelocityFilter.</summary>
    [Export] public float JointVelocityFilter { get; set; } = 1.0f;

    /// <summary>Passed to the driver. See IsaacPolicyDriver.JointVelocityEnabled.</summary>
    [Export] public bool JointVelocityEnabled { get; set; } = true;

    /// <summary>Passed to the driver. See IsaacPolicyDriver.JointVelocityMinRange.</summary>
    [Export] public float JointVelocityMinRange { get; set; }

    /// <summary>
    /// Passed to the driver: seconds between per-slice observation diagnostics. Defaults to the
    /// driver's own 0.5 s. Drop it to ~0.03 to compare against `slice_stats.py` step by step -
    /// half a second is 30 policy steps, which is far too coarse to see WHERE two engines diverge.
    /// </summary>
    [Export] public float DiagnosticInterval { get; set; } = -1.0f;

    /// <summary>
    /// Spawn Godot's real <see cref="Perturbation.BallGun"/> and fire actual balls at the dummy.
    ///
    /// <para>Distinct from <see cref="PushImpulse"/>, and both are worth having. `PushImpulse` is an
    /// ANALYTIC impulse on the pelvis - it is what the Isaac task trains against, so it is the
    /// like-for-like check. A real ball is the Godot-native scenario: a rigid body with a collider
    /// that has to actually connect, delivering a contact-dependent impulse to whichever bone it
    /// hits. If the policy survives the analytic push and not the ball, the gap is contact and
    /// aiming rather than balance.</para>
    ///
    /// <para>BallGun's defaults already ARE the shot the Godot track fires: `SmallBallProbability`
    /// is 1.0, so every shot is the small ball - 3.0 kg at 6 m/s, aimed at a uniformly chosen bone
    /// from 2 m out.</para>
    ///
    /// <para><b>That transfers 22.5 N.s, not the 18 the momentum alone suggests.</b> BallGun
    /// applies `(velocity - reflected) * ballMass` and `reflected` carries `BallRestitution`, so a
    /// head-on hit delivers `m*v*(1+e)` = 3.0 * 6.0 * 1.25. This line used to read "= 18 N.s" and
    /// the Isaac-side curriculum inherited the error: its ceiling was capped at 18, so the training
    /// disturbance never once reached the magnitude this arena actually throws.</para>
    /// </summary>
    [Export] public bool SpawnBallGun { get; set; }

    /// <summary>Seconds between shots when <see cref="SpawnBallGun"/> is set. BallGun's own default
    /// is 10 s so training gets one hit per episode; an arena wants repeated recoveries.</summary>
    [Export] public float BallInterval { get; set; } = 4.0f;

    /// <summary>Delay before the first ball, so the dummy has settled first.</summary>
    [Export] public float BallFirstShotSeconds { get; set; } = 3.0f;

    [Export] public float PushImpulse { get; set; }

    /// <summary>When to deliver <see cref="PushImpulse"/>, seconds. Late enough to have settled.</summary>
    [Export] public float PushAtSeconds { get; set; } = 4.0f;

    /// <summary>Push direction in world space; normalised. Defaults to +X (forward).</summary>
    [Export] public Vector3 PushDirection { get; set; } = Vector3.Right;

    [Export] public float AutoQuitSeconds { get; set; }

    /// <summary>Seconds between head-height reports while <see cref="AutoQuitSeconds"/> is active.</summary>
    [Export] public float ReportInterval { get; set; } = 0.5f;

    private IsaacPolicyDriver? _driver;
    private UI.RagdollTelemetryHud? _telemetry;
    private HumanoidRagdoll? _ragdoll;
    private float _elapsed;
    private bool _pushed;
    private bool _ballGunReady;
    private float _headAtPush;
    private bool _started;
    private float _nextReport;
    private float _minHeadHeight = float.MaxValue;

    /// <summary>
    /// <b>R restarts the whole check</b> - scene, dummy, driver and verdict together.
    ///
    /// <para><see cref="RagdollDebugInput"/> already binds R to <c>ResetRagdoll</c>, which puts the
    /// body back but leaves this arena's clock, its warmup, its first-step logging and its
    /// min-height verdict carrying state from the previous attempt. For a check whose entire output
    /// is "did it stand", a half-reset is worse than none: the numbers describe two runs at once.
    /// Reloading the scene resets every one of them by construction.</para>
    /// </summary>
    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventKey { Pressed: true, Echo: false, Keycode: Key.R })
        {
            GD.Print("[IsaacArena] R - restarting the scene.");
            GetTree().ReloadCurrentScene();
        }
    }

    public override void _Ready()
    {
        Node chamber = GD.Load<PackedScene>(ChamberPath).Instantiate();
        AddChild(chamber);

        _ragdoll = FindRagdoll(chamber);
        if (_ragdoll == null)
        {
            GD.PushError($"[IsaacArena] no HumanoidRagdoll under {ChamberPath}");
            return;
        }

        // Say WHICH brain is about to drive this body, before it does. Since the export was named
        // after the brain rather than the task, one artifact serves several scenes and the scene
        // name no longer identifies the policy - so a scene silently loading the wrong or a stale
        // brain looks exactly like a scene loading the right one. Mirrors the `[watch]` line.
        string brain = IsaacRigContract.DescribeBrain(PolicyContractPath, PolicyPath);
        if (!string.IsNullOrEmpty(brain))
        {
            GD.Print($"[IsaacArena] brain: {brain}");

            // Same line on screen. The HUD lives inside the CHAMBER scene, not this one - the arena
            // instantiates TestChamber through ChamberPath - so it has to be found rather than
            // exported. Searching the chamber subtree only, never the whole tree: a whole-tree
            // FindChild returns the first match anywhere, which is how the telemetry HUD once got
            // pinned to Agent_1 in a 64-body scene.
            _telemetry = chamber.FindChild("RagdollTelemetryHud", true, false) as UI.RagdollTelemetryHud;
            if (_telemetry != null)
            {
                _telemetry.BrainLine = brain;
                GD.Print("[IsaacArena] telemetry HUD bound - brain shown on screen.");
            }
            else
            {
                // Not an error - a chamber may legitimately have no HUD - but say so, because the
                // alternative is an on-screen readout that silently stops appearing if the node is
                // renamed, and nobody notices a panel that is merely missing a line.
                GD.Print($"[IsaacArena] no RagdollTelemetryHud under {ChamberPath} - brain shown in console only.");
            }
        }

        _driver = new IsaacPolicyDriver
        {
            Name = "IsaacPolicyDriver",
            Ragdoll = _ragdoll,
            PolicyPath = PolicyPath,
            RigContractPath = RigContractPath,
            PolicyContractPath = PolicyContractPath,
            HeightContacts = HeightContacts,
            Command = Command,
            ZeroActionBaseline = ZeroAction,
            AssistMode = AssistMode,
            BalanceAssist = BalanceAssist,
            ActionScaleOverride = ActionScaleOverride,
            DumpPoseAfterSeconds = DumpPoseAfterSeconds,
            AssistAuthority = AssistAuthority,
            JointSpacePd = JointSpacePd,
            HillVelocityFilter = HillVelocityFilter,
            JointVelocityFilter = JointVelocityFilter,
            JointVelocityEnabled = JointVelocityEnabled,
            JointVelocityMinRange = JointVelocityMinRange,
            DiagnosticInterval = DiagnosticInterval >= 0.0f ? DiagnosticInterval : 0.5f,
        };

        if (WarmupSeconds >= 0.0f)
        {
            _driver.WarmupSeconds = WarmupSeconds;
        }

        // Deferred so the ragdoll's own _Ready has resolved its bones before the driver resolves
        // them again by name. Adding it inline would run the driver's _Ready first.
        if (PolicyDelaySeconds <= 0.0f)
        {
            CallDeferred(Node.MethodName.AddChild, _driver);
            _started = true;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        _elapsed += (float)delta;

        // Mirrored from `_started` every frame rather than set at the moment of attaching, because
        // the driver is attached from TWO places - immediately when PolicyDelaySeconds is 0 (which
        // every current check scene sets), and from the delayed branch below. Setting the flag at
        // one of them left the HUD reporting "Godot balance still holding" while the policy in fact
        // had the body, which is worse than showing nothing.
        if (_telemetry != null)
        {
            _telemetry.BrainEngaged = _started;
        }

        if (!_started && _driver != null && _elapsed >= PolicyDelaySeconds)
        {
            _started = true;
            AddChild(_driver);
            GD.Print($"[IsaacArena] policy engaged at {_elapsed:F2}s");
        }

        if (SpawnBallGun && !_ballGunReady && _ragdoll != null && IsInstanceValid(_ragdoll))
        {
            _ballGunReady = true;
            var gun = new Perturbation.BallGun
            {
                Target = _ragdoll,
                IntervalSeconds = BallInterval,
                FirstShotDelaySeconds = BallFirstShotSeconds,
            };
            AddChild(gun);
            GD.Print($"[IsaacArena] BallGun firing every {BallInterval:F1}s "
                     + $"(first at {BallFirstShotSeconds:F1}s) - 3.0 kg at 6 m/s, "
                     + $"transferring 22.5 N.s per head-on hit");
        }

        if (PushImpulse > 0.0f && !_pushed && _elapsed >= PushAtSeconds
            && _ragdoll?.Pelvis != null && IsInstanceValid(_ragdoll.Pelvis))
        {
            _pushed = true;
            Vector3 direction = PushDirection.LengthSquared() > 0.0f
                ? PushDirection.Normalized()
                : Vector3.Right;
            _ragdoll.Pelvis.ApplyImpulse(direction * PushImpulse);
            _headAtPush = _ragdoll.Head?.GlobalPosition.Y ?? 0.0f;
            GD.Print($"[IsaacArena] PUSH {PushImpulse:F1} N.s at t={_elapsed:F2}s, head={_headAtPush:F3}");
        }

        if (AutoQuitSeconds <= 0.0f)
        {
            return;
        }

        ActiveBone? head = _ragdoll?.Head;
        if (head != null && IsInstanceValid(head))
        {
            float h = head.GlobalPosition.Y;
            // Tracked from the moment the policy engages, not from scene load: before that the body
            // is in free fall by design and its height says nothing about the policy.
            if (_started)
            {
                _minHeadHeight = Mathf.Min(_minHeadHeight, h);
            }

            if (_elapsed >= _nextReport)
            {
                _nextReport += ReportInterval;
                GD.Print($"[IsaacArena] t={_elapsed,5:F2}s  head={h:F3}  pelvis={_ragdoll!.Pelvis!.GlobalPosition.Y:F3}");
            }
        }

        if (_elapsed >= AutoQuitSeconds)
        {
            // 1.35 is STANDING_HEAD_HEIGHT from stand_env_cfg.py - the same threshold the Isaac
            // evaluator scores against, so a pass here means the same thing it means there.
            float final = head != null && IsInstanceValid(head) ? head.GlobalPosition.Y : 0.0f;
            bool stood = final >= 1.35f;
            GD.Print($"[IsaacArena] final head={final:F3}  min-since-engaged={_minHeadHeight:F3}  "
                     + $"=> {(stood ? "STANDING" : "FELL")}");
            GetTree().Quit(stood ? 0 : 1);
        }
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
}

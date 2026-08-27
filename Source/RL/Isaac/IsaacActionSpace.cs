using System.Collections.Generic;
using System.Text;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// The 36-float action mapping an Isaac Lab policy was trained against. See
/// `isaac_lab/obs_action_contract.md` §2.
///
/// Three floats per bone, an absolute joint target expressed as a fraction of that axis's own
/// reachable range, piecewise-linear about the rest pose:
///
/// <code>
/// a      = clamp(policy_output, -1, +1)
/// span   = (upper - default) if a >= 0 else (default - lower)
/// target = default + ACTION_SCALE * a * span
/// </code>
///
/// <para><b>Differences from <see cref="Actions.JointLimitedActionSpace"/>,</b> which is the
/// Godot-native equivalent and stays in use for that track:</para>
///
/// <list type="bullet">
/// <item><description><see cref="ActionScale"/> is applied. Omitting it makes every commanded
/// motion 2.5x too large. It is part of the contract, not a tuning knob - mapping the full
/// [-1,1] onto a joint's whole range is too coarse to balance with, and measured at scale 1.0 the
/// policy plateaued at 48-step episodes while doing nothing at all survives 168.</description></item>
/// <item><description>Limits come from the rig contract in ISAAC's convention, and the roll axes
/// are converted back to Godot's sense at the end. The Godot-native class reads live joint limits
/// instead, which are the same numbers for pitch and yaw but negated-and-swapped for
/// roll.</description></item>
/// <item><description>Composition is <c>Rx * Ry * Rz</c> via <see cref="EulerOrder.Xyz"/>, matching
/// the URDF's stacked <c>rx -> ry -> rz</c> chain. The Godot-native class uses
/// <c>Quaternion.FromEuler</c>, which is YXZ - a different rotation from the same three
/// numbers.</description></item>
/// </list>
///
/// <para>The clamp is mandatory and not defensive politeness. The network's output is an unbounded
/// Gaussian mean; nothing in the architecture keeps it inside [-1,1]. A Stand policy trained
/// without a barrier term reached raw outputs of +/-18 with 97% of components past the boundary,
/// which fed unclamped commands roughly 15x their intended deflection. The training environment
/// clamps, so the policy is DEFINED by its clamped output.</para>
/// </summary>
public sealed class IsaacActionSpace : IRlActionSpace, IRlActionDiagnostics
{
    /// <summary>Contract constant. Must match `action_scale` in `stand_env_cfg.py`.</summary>
    public const float DefaultActionScale = 0.4f;

    /// <summary>
    /// Action scale actually in force, read from the policy contract's <c>action_scale</c>.
    ///
    /// <para><b>This was a hardcoded 0.4 and that is a latent contract bug.</b> `export.py` has
    /// always WRITTEN `action_scale` into the contract, and nothing on this side read it - so a
    /// policy trained at any other scale would have been driven at 0.4 regardless, commanding
    /// deflections several times what it intended, with nothing to indicate it. The same defect
    /// class as the DOF order: the number travels with the policy, so it must be read from the
    /// policy.</para>
    /// </summary>
    public float ActionScale { get; set; } = DefaultActionScale;

    /// <summary>
    /// Half-range below which an axis is commanded to rest instead of to the policy's output.
    /// Zero (the default) commands every axis, which is what the contract literally says.
    ///
    /// <para><b>An experiment, not a contract change - and the numbers say why it is worth
    /// trying.</b> Every limb's twist axis is limited to about +/-0.10 rad, so after
    /// <see cref="ActionScale"/> the policy's entire authority there is +/-0.04 rad: functionally
    /// nothing. The torque it generates is not nothing. Godot drives those axes with an explicit PD
    /// at up to kp=1800 while Jolt enforces the limit as a constraint, so the joint slams into its
    /// stop and bounces - measured at 33.9 rad/s on `Forearm_R.y` and 68.1 on `Shin_R.y` with the
    /// body still standing at 0.81 m, against a peak of 7.5 across all 45 DOF for the same policy
    /// in Isaac, whose position-based solver has no such fight.</para>
    ///
    /// <para>That chatter costs twice over: it saturates 45 of the 143 observation floats, and
    /// because Godot bounds a bone's three axes against ONE torque budget, it burns the authority
    /// the knee needs to hold the body up. Commanding zero at least stops the policy from driving
    /// it. Compare against the same scene with this at 0 before drawing any conclusion.</para>
    /// </summary>
    public float LockAxesBelow { get; set; }

    /// <summary>
    /// Maximum change in each action component per policy step, in the action's own [-1,1] units.
    /// Zero disables it. **Read from the policy contract's `action_rate_limit`** - a policy trained
    /// with a cap must be driven with the same cap, and one trained without must not have one
    /// imposed.
    ///
    /// <para><b>Why it exists.</b> Godot's joint velocities saturate within 33 ms - two policy
    /// steps - of the first action, with the body still perfectly upright at 0.82 m, and the worst
    /// DOF at that moment is a main hinge axis on a light arm rather than a chattering twist axis.
    /// So it is a genuine slew to the first commanded target, not chatter. Isaac shows the same
    /// transient at 7.54 rad/s and it decays; Godot's climbs past 15. The policy commands roughly
    /// 0.785 rad of deflection as a STEP, and the two engines disagree about how to execute a step:
    /// Isaac resolves it inside a position-based solve that is inherently rate-limited by the
    /// timestep, Godot slews a light limb at whatever the torque allows.</para>
    ///
    /// <para>Removing the step from the command should leave the two engines in the quasi-static
    /// regime where they already agree.</para>
    /// </summary>
    public float ActionRateLimit { get; set; }

    /// <summary>Last commanded action, the state the rate limit integrates from.</summary>
    private readonly float[] _rateLimited;

    /// <summary>|action| at or above which a command counts as pinned against its bound.</summary>
    private const float SaturationThreshold = 0.95f;

    private readonly IsaacRigContract _rig;
    private readonly string[] _boneNames;

    /// <summary>Scratch buffer for the clamped action, exposed so the observation can record it.</summary>
    private readonly float[] _clamped;

    /// <summary>Per-bone commanded joint angles in Godot's sense, radians. Valid after Decode.</summary>
    private readonly Vector3[] _targetEuler;

    private long _saturatedComponents;
    private long _totalComponents;
    private long _clippedComponents;
    private float _maxMagnitude;
    private readonly Dictionary<string, float> _stats = new();

    public IsaacActionSpace(IsaacRigContract rig)
    {
        _rig = rig;
        _boneNames = rig.ActuatedBoneNames();
        _clamped = new float[rig.ActuatedJoints.Count];
        _rateLimited = new float[rig.ActuatedJoints.Count];
        _targetEuler = new Vector3[rig.ActuatedJoints.Count / IsaacRigContract.AxesPerBone];
    }

    public int Size => _rig.ActuatedJoints.Count;

    /// <summary>
    /// The action after clamping, as the contract's "previous action" slice wants it: in [-1,1],
    /// before <see cref="ActionScale"/> and before the range mapping. Valid after
    /// <see cref="Decode"/>.
    /// </summary>
    public float[] ClampedAction => _clamped;

    /// <summary>
    /// The commanded joint angles per bone, in Godot's sense, as the quaternion offsets were built
    /// from. Exposed for the joint-space PD path, which needs the angles themselves rather than a
    /// composed rotation. Valid after <see cref="Decode"/>.
    /// </summary>
    public Vector3[] TargetEuler => _targetEuler;

    public IReadOnlyDictionary<string, float> EpisodeActionStats
    {
        get
        {
            _stats["saturation"] = _totalComponents > 0 ? (float)_saturatedComponents / _totalComponents : 0.0f;
            // Distinct from saturation, and the more diagnostic of the two: saturation says the
            // policy is asking for the end of a joint's range, which is legitimate. Clipping says
            // it asked for MORE than the range exists, which means the barrier term failed and the
            // exported policy is not the one that was evaluated.
            _stats["clipped"] = _totalComponents > 0 ? (float)_clippedComponents / _totalComponents : 0.0f;
            _stats["max_magnitude"] = _maxMagnitude;
            return _stats;
        }
    }

    public void Bind(IReadOnlyList<ActiveBone?> controlledBones)
    {
        // Nothing to resolve: limits come from the rig contract, which is the same file the policy
        // was trained against. Reading them off the live joints instead would silently diverge the
        // moment the scene is retuned without regenerating the rig - and the divergence would be a
        // policy driven against ranges it never saw.
    }

    public void Reset()
    {
        System.Array.Clear(_rateLimited);
        _saturatedComponents = 0;
        _totalComponents = 0;
        _clippedComponents = 0;
        _maxMagnitude = 0.0f;
    }

    public void Decode(float[] action, Quaternion[] offsets)
    {
        // Once per call, and Decode is called once per POLICY step - matching
        // `StandEnv._pre_physics_step`. Applying it per physics tick instead would silently double
        // the real rate against what the policy trained with.
        // `_rateLimited` is filled here and read below; the caller's `action` array is left alone
        // on purpose, so the saturation and clipping counters still see the RAW network output.
        // Overwriting it would make `clipped` read zero for a policy whose mean has drifted far
        // outside the usable range - the one statistic that detects a failed barrier term.
        for (int i = 0; i < action.Length && i < _rateLimited.Length; i++)
        {
            float clamped = Mathf.Clamp(action[i], -1.0f, 1.0f);
            if (ActionRateLimit > 0.0f)
            {
                _rateLimited[i] += Mathf.Clamp(clamped - _rateLimited[i], -ActionRateLimit, ActionRateLimit);
            }
            else
            {
                _rateLimited[i] = clamped;
            }
        }

        for (int i = 0; i < offsets.Length; i++)
        {
            int baseIndex = i * IsaacRigContract.AxesPerBone;
            offsets[i] = Quaternion.Identity;
            if (baseIndex + 2 >= action.Length)
            {
                continue;
            }

            var godotEuler = Vector3.Zero;
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                int index = baseIndex + axis;
                IsaacRigContract.JointSpec spec = _rig.ActuatedJoints[index];
                float isaacTarget = MapToRange(action[index], _rateLimited[index], spec, index);

                // One conversion, at the boundary: everything above is in Isaac's convention,
                // everything below is Godot's.
                float godotAngle = IsaacRigContract.ToGodot(spec, isaacTarget);
                SetComponent(ref godotEuler, spec.GodotAxis, godotAngle);
            }

            _targetEuler[i] = godotEuler;
            offsets[i] = Basis.FromEuler(godotEuler, EulerOrder.Xyz).GetRotationQuaternion().Normalized();
        }
    }

    /// <summary>
    /// One policy output onto one joint axis, in Isaac's convention.
    ///
    /// `default` is 0 for every joint - the Godot rest pose - so <c>a = 0</c> is the rest pose and
    /// the two halves of the range have different slopes wherever the limits are asymmetric. That
    /// asymmetry is intended, not an approximation: interpolating affinely across [lower, upper]
    /// would put zero at the range MIDPOINT, which for the knee (`[-2.6, 0.1]`) is -1.25 rad, and
    /// a freshly initialised policy would command a deep crouch from its very first step.
    /// </summary>
    private float MapToRange(
        float rawAction, float commanded, in IsaacRigContract.JointSpec spec, int index)
    {
        // `commanded` is what actually drives the joint - the rate-limited value when a limit is
        // active, the plain clamp otherwise. `rawAction` is the untouched network output and is
        // only used for the diagnostics below, which exist to detect a policy whose mean has
        // drifted outside the usable range.
        float clamped = commanded;
        _clamped[index] = clamped;

        float magnitude = Mathf.Abs(rawAction);
        _totalComponents++;
        if (magnitude > 1.0f)
        {
            _clippedComponents++;
        }
        if (Mathf.Abs(clamped) >= SaturationThreshold)
        {
            _saturatedComponents++;
        }
        if (magnitude > _maxMagnitude)
        {
            _maxMagnitude = magnitude;
        }

        float span = clamped >= 0.0f ? spec.Upper : -spec.Lower;
        if (LockAxesBelow > 0.0f && Mathf.Max(spec.Upper, -spec.Lower) < LockAxesBelow)
        {
            return 0.0f;
        }
        return ActionScale * clamped * span;
    }

    private static void SetComponent(ref Vector3 v, char axis, float value)
    {
        switch (axis)
        {
            case 'x': v.X = value; break;
            case 'y': v.Y = value; break;
            case 'z': v.Z = value; break;
        }
    }

    /// <summary>
    /// Reports the ranges actually read from the rig contract, so a manifest cannot drift from the
    /// limits the policy was really trained against.
    /// </summary>
    public string Describe()
    {
        var parts = new StringBuilder($"IsaacActionSpace(scale={ActionScale}; isaacLimits; ");
        for (int i = 0; i < _boneNames.Length; i++)
        {
            if (i > 0)
            {
                parts.Append("; ");
            }
            int b = i * IsaacRigContract.AxesPerBone;
            parts.Append($"{_boneNames[i]}:");
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                IsaacRigContract.JointSpec spec = _rig.ActuatedJoints[b + axis];
                parts.Append($"{spec.GodotAxis}[{spec.Lower:F2},{spec.Upper:F2}]{(spec.Reversed ? "-" : "")}");
            }
        }
        return parts.Append(')').ToString();
    }
}

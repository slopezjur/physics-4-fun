using System.Collections.Generic;
using System.Text;
using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Actions;

/// <summary>
/// Absolute joint-target action space, scaled per axis into each joint's own reachable range.
///
/// Three floats per bone, interpreted as an Euler offset from that bone's rest pose. Each
/// component is mapped from [-1,1] onto that specific axis's [lower, upper] limit, read from the
/// live joint at bind time.
///
/// **Why per-axis rather than one global scale.** A single MaxActionAngle applied to every axis of
/// every bone was measured to put 71.8% of the action space beyond a hard stop: the knee and elbow
/// roll axes are limited to +/-0.10 and +/-0.15 rad, against a +/-2.6 rad command range - a 26x
/// over-range. Commands past a stop do not produce a smaller motion; they pin the actuator at full
/// torque with a permanent tracking error, so every out-of-range action yields the SAME physical
/// result. That flattens the policy gradient across most of the output space, which is an
/// exploration problem no reward shaping or curriculum can fix.
/// </summary>
public sealed class JointLimitedActionSpace : IRlActionSpace, IRlActionDiagnostics
{
    /// <summary>Euler components commanded per bone.</summary>
    private const int ComponentsPerBone = 3;

    /// <summary>
    /// Range used for an axis whose joint declares no limit (free rotation, or no joint at all).
    /// Only a fallback - every constrained axis uses its own measured bounds.
    /// </summary>
    private const float UnconstrainedFallbackAngle = 2.6f;

    /// <summary>|action| at or above which a command counts as pinned against its bound.</summary>
    private const float SaturationThreshold = 0.95f;

    private readonly string[] _boneNames;
    private Vector3[] _lower = System.Array.Empty<Vector3>();
    private Vector3[] _upper = System.Array.Empty<Vector3>();

    private long _saturatedComponents;
    private long _totalComponents;
    private readonly Dictionary<string, float> _stats = new();

    public JointLimitedActionSpace(string[] boneNames)
    {
        _boneNames = boneNames;
        _lower = new Vector3[boneNames.Length];
        _upper = new Vector3[boneNames.Length];
        for (int i = 0; i < boneNames.Length; i++)
        {
            _lower[i] = new Vector3(-UnconstrainedFallbackAngle, -UnconstrainedFallbackAngle, -UnconstrainedFallbackAngle);
            _upper[i] = new Vector3(UnconstrainedFallbackAngle, UnconstrainedFallbackAngle, UnconstrainedFallbackAngle);
        }
    }

    public int Size => _boneNames.Length * ComponentsPerBone;

    public IReadOnlyDictionary<string, float> EpisodeActionStats
    {
        get
        {
            _stats["saturation"] = _totalComponents > 0
                ? (float)_saturatedComponents / _totalComponents
                : 0.0f;
            return _stats;
        }
    }

    public void Bind(IReadOnlyList<ActiveBone?> controlledBones)
    {
        for (int i = 0; i < _boneNames.Length && i < controlledBones.Count; i++)
        {
            ActiveBone? bone = controlledBones[i];
            if (bone != null && GodotObject.IsInstanceValid(bone))
            {
                bone.GetJointAngularLimits(UnconstrainedFallbackAngle, out _lower[i], out _upper[i]);
            }
        }
    }

    public void Reset()
    {
        _saturatedComponents = 0;
        _totalComponents = 0;
    }

    public void Decode(float[] action, Quaternion[] offsets)
    {
        for (int i = 0; i < offsets.Length; i++)
        {
            int baseIndex = i * ComponentsPerBone;
            Vector3 lower = _lower[i];
            Vector3 upper = _upper[i];

            offsets[i] = Quaternion.FromEuler(new Vector3(
                Scale(action[baseIndex + 0], lower.X, upper.X),
                Scale(action[baseIndex + 1], lower.Y, upper.Y),
                Scale(action[baseIndex + 2], lower.Z, upper.Z)));
        }
    }

    /// <summary>
    /// Maps one policy output in [-1,1] onto one axis of a joint's reachable range.
    ///
    /// Split at zero rather than interpolating linearly across [lower, upper], so action 0 is
    /// always the REST pose. Joint ranges here are strongly asymmetric - the knee's x limit is
    /// [-2.60, +0.10] - and a straight lerp would place the neutral action at the range midpoint,
    /// commanding a permanently bent knee for a policy that outputs zero. Splitting keeps "no
    /// action" meaning "no deflection" while still reaching the full range in both directions.
    /// </summary>
    private float Scale(float action, float lower, float upper)
    {
        float clamped = Mathf.Clamp(action, -1.0f, 1.0f);

        _totalComponents++;
        if (Mathf.Abs(clamped) >= SaturationThreshold)
        {
            _saturatedComponents++;
        }

        return clamped >= 0.0f ? clamped * upper : -clamped * lower;
    }

    /// <summary>
    /// Reports the ranges actually resolved from the rig, not the constants above, so the run
    /// manifest cannot drift from the joints the policy was really trained against.
    /// </summary>
    public string Describe()
    {
        var parts = new StringBuilder("JointLimitedActionSpace(perAxis; ");
        for (int i = 0; i < _boneNames.Length; i++)
        {
            if (i > 0)
            {
                parts.Append("; ");
            }
            parts.Append(
                $"{_boneNames[i]}:x[{_lower[i].X:F2},{_upper[i].X:F2}]"
                + $"y[{_lower[i].Y:F2},{_upper[i].Y:F2}]"
                + $"z[{_lower[i].Z:F2},{_upper[i].Z:F2}]");
        }
        return parts.Append(')').ToString();
    }
}

using Godot;
using Godot.Collections;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules.Reflexes;

/// <summary>
/// Autonomous reflex for 3D environmental spatial awareness, wall reach, and obstacle push-off bracing (Euphoria DMS / Pillar 5).
/// </summary>
public class ObstacleBracingReflexModule : IBiomechanicalReflex
{
    public bool IsEnabled { get; set; } = true;
    public float SensorRadius { get; set; } = 1.30f;
    public float BracePushForce { get; set; } = 140.0f;

    public bool IsBracing { get; private set; } = false;

    private IBoneState? _chest;
    private IBoneState? _upperArmL;
    private IBoneState? _upperArmR;
    private IBoneState? _forearmL;
    private IBoneState? _forearmR;
    private Array<Rid>? _excludeRids;

    public void Initialize(
        IBoneState? chest,
        IBoneState? upperArmL,
        IBoneState? upperArmR,
        IBoneState? forearmL,
        IBoneState? forearmR,
        Array<Rid>? excludeRids)
    {
        _chest = chest;
        _upperArmL = upperArmL;
        _upperArmR = upperArmR;
        _forearmL = forearmL;
        _forearmR = forearmR;
        _excludeRids = excludeRids;
    }

    public void Reset()
    {
        IsBracing = false;
    }

    public void Update(RagdollState state, StepPhase stepPhase, float delta)
    {
        if (!IsEnabled || _chest == null || !_chest.IsValid)
        {
            IsBracing = false;
            return;
        }

        // Only brace when in active balance, stumbling, or flailing
        if (state != RagdollState.Balanced && state != RagdollState.Stumbling && state != RagdollState.Flailing)
        {
            IsBracing = false;
            return;
        }

        // A raycast is an ENGINE capability, not bone state - so this is the one place these
        // modules still need the concrete body. Cast here rather than widening IBoneState:
        // faking a physics query is not the same as faking a bone, and pretending otherwise
        // would make the interface untestable in the way it was added to prevent.
        var world = ((Node3D)_chest).GetWorld3D();
        if (world == null) return;
        var spaceState = world.DirectSpaceState;

        Vector3 chestPos = _chest.GlobalPosition;
        Vector3 leftDir = _chest.GlobalTransform.Basis.X.Normalized();
        Vector3 rightDir = -leftDir;
        Vector3 forwardDir = -_chest.GlobalTransform.Basis.Z.Normalized();

        // 1. Probe Left Side
        bool hitLeft = ProbeDirection(spaceState, chestPos, leftDir, out Vector3 leftContact, out Vector3 leftNormal, out float leftDist);
        // 2. Probe Right Side
        bool hitRight = ProbeDirection(spaceState, chestPos, rightDir, out Vector3 rightContact, out Vector3 rightNormal, out float rightDist);
        // 3. Probe Forward
        bool hitForward = ProbeDirection(spaceState, chestPos, forwardDir, out Vector3 fwdContact, out Vector3 fwdNormal, out float fwdDist);

        IsBracing = hitLeft || hitRight || hitForward;

        if (hitLeft && leftDist < SensorRadius)
        {
            float intensity = Mathf.Clamp(1.0f - (leftDist / SensorRadius), 0.0f, 1.0f);
            if (_upperArmL != null) _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.30f * intensity, 0.0f, 1.10f * intensity));
            if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.60f * intensity, 0.0f, 0.0f));
        }

        if (hitRight && rightDist < SensorRadius)
        {
            float intensity = Mathf.Clamp(1.0f - (rightDist / SensorRadius), 0.0f, 1.0f);
            if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.30f * intensity, 0.0f, -1.10f * intensity));
            if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.60f * intensity, 0.0f, 0.0f));
        }

        if (hitForward && fwdDist < SensorRadius && !hitLeft && !hitRight)
        {
            float intensity = Mathf.Clamp(1.0f - (fwdDist / SensorRadius), 0.0f, 1.0f);
            if (_upperArmL != null) _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.85f * intensity, -0.15f * intensity, 0.20f * intensity));
            if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.85f * intensity, 0.15f * intensity, -0.20f * intensity));
            if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.50f * intensity, 0.0f, 0.0f));
            if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.50f * intensity, 0.0f, 0.0f));
        }
    }

    private bool ProbeDirection(
        PhysicsDirectSpaceState3D spaceState,
        Vector3 origin,
        Vector3 direction,
        out Vector3 contactPoint,
        out Vector3 normal,
        out float distance)
    {
        contactPoint = Vector3.Zero;
        normal = Vector3.Up;
        distance = float.MaxValue;

        Vector3 rayEnd = origin + direction * SensorRadius;
        var query = PhysicsRayQueryParameters3D.Create(origin, rayEnd, 1);
        if (_excludeRids != null)
        {
            query.Exclude = _excludeRids;
        }

        var result = spaceState.IntersectRay(query);
        if (result.Count > 0)
        {
            contactPoint = (Vector3)result["position"];
            normal = (Vector3)result["normal"];
            distance = origin.DistanceTo(contactPoint);
            return true;
        }

        return false;
    }
}

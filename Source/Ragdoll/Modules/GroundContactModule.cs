using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Contact-driven foot ground sensing (SRP): grounded = real physical contact with the world
/// (contact monitor) AND sole facing downward. The raycast is only used to query ground-point
/// height (step targets, foot elevation), never to decide contact state.
/// </summary>
public class GroundContactModule
{
    /// <summary>
    /// Downward reach (m) of the ground-height probe, measured from just above the foot. Sized to
    /// find the floor even with the leg fully raised during a get-up, not merely at stance height.
    /// </summary>
    private const float GroundProbeDistance = 3.0f;

    public bool IsGroundedL { get; private set; }
    public bool IsGroundedR { get; private set; }
    public Vector3 GroundPointL { get; private set; } = Vector3.Zero;
    public Vector3 GroundPointR { get; private set; } = Vector3.Zero;

    public void Update(IBoneState pelvis, IBoneState? footL, IBoneState? footR, Godot.Collections.Array<Rid> excludeRids)
    {
        // A raycast is an ENGINE capability, not bone state - so this is the one place these
        // modules still need the concrete body. Cast here rather than widening IBoneState:
        // faking a physics query is not the same as faking a bone, and pretending otherwise
        // would make the interface untestable in the way it was added to prevent.
        var spaceState = ((Node3D)pelvis).GetWorld3D().DirectSpaceState;

        Vector3 groundPointL = GroundPointL;
        Vector3 groundPointR = GroundPointR;
        IsGroundedL = UpdateFootGroundSensor(footL, ref groundPointL, spaceState, excludeRids);
        IsGroundedR = UpdateFootGroundSensor(footR, ref groundPointR, spaceState, excludeRids);
        GroundPointL = groundPointL;
        GroundPointR = groundPointR;
    }

    private static bool UpdateFootGroundSensor(IBoneState? foot, ref Vector3 groundPoint, PhysicsDirectSpaceState3D spaceState, Godot.Collections.Array<Rid> excludeRids)
    {
        if (foot == null || !foot.IsValid)
        {
            return false;
        }

        // Ground-point height query. The probe must reach the floor from a RAISED foot: during a
        // get-up the feet swing well clear of the ground, and a short probe fails exactly when the
        // answer matters most. A 0.25 m probe made the reported ground height climb with the foot
        // (measured up to 0.736 m mid-recovery), which silently invalidated every height-based
        // get-up criterion.
        Vector3 rayStart = foot.GlobalPosition + new Vector3(0, 0.05f, 0);
        Vector3 rayEnd = foot.GlobalPosition - new Vector3(0, GroundProbeDistance, 0);
        var query = PhysicsRayQueryParameters3D.Create(rayStart, rayEnd, 1);
        query.Exclude = excludeRids;
        var result = spaceState.IntersectRay(query);

        // On a genuine miss keep the last known floor height. Adopting the foot's own position
        // would make the sensor report "the ground is wherever the foot happens to be".
        if (result.Count > 0)
        {
            groundPoint = (Vector3)result["position"];
        }

        // Foot local -Y is the sole axis (identity-oriented at rest); require it within ~60 deg of world down
        bool soleDown = (-foot.GlobalTransform.Basis.Y).Normalized().Dot(Vector3.Down) > 0.5f;

        return soleDown && foot.IsInContactWithWorld();
    }
}

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Left/right selector for motions that are inherently asymmetric.
///
/// Rising from prone is the motivating case: a symmetric get-up, with both legs commanded
/// identically, is not a human motion and cannot bring the centre of mass over a support point.
/// One leg must lead — drive its knee under the chest and plant — while the other trails.
/// </summary>
public enum BodySide
{
    Left,
    Right
}

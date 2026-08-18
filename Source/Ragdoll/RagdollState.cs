namespace Physics4Fun.Ragdoll;

/// <summary>
/// Active ragdoll behavioral states mimicking Euphoria-style reaction tiers.
/// </summary>
public enum RagdollState
{
    /// <summary>
    /// Upright balance active, tracking standing stance with strong muscle stiffness.
    /// </summary>
    Balanced,

    /// <summary>
    /// Disturbance encountered: dynamic arm flailing, body counter-sway to restore CoM.
    /// </summary>
    Stumbling,

    /// <summary>
    /// Airborne / falling / grabbed: defensive reach and protective impact tuck.
    /// </summary>
    Flailing,

    /// <summary>
    /// Complete muscle relaxation: zero torque output, pure Newtonian ragdoll physics.
    /// </summary>
    KnockedOut,

    /// <summary>
    /// Automatic recovery sequence: pushing off ground to stand back up.
    /// </summary>
    Recovering
}

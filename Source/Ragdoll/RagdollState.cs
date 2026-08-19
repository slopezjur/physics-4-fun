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
    Recovering,

    /// <summary>
    /// Debug drill: an isolated knee push-up, entered and left by hand rather than by the state
    /// machine. Exists to develop the arm press on its own, without the legs or the get-up phase
    /// machine in the way.
    /// </summary>
    PushUpDrill
}

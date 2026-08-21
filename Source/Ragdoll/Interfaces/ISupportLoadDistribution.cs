using System.Collections.Generic;

namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>
/// Decides how much body weight each planted limb is asked to hold up, by writing
/// <see cref="ActiveBone.SupportedMassShare"/> across the rig once per physics tick.
///
/// A contract rather than a method on the body because it is a modelling decision with real
/// alternatives - total mass split evenly, mass above the support only, a measured ground-reaction
/// split - and each produces a materially different feed-forward. Keeping it swappable means the
/// alternatives can be compared instead of argued about.
/// </summary>
public interface ISupportLoadDistribution
{
    /// <summary>Assigns each bone's supported share for this tick. Called before actuator update.</summary>
    void Distribute(IReadOnlyList<ActiveBone> allBones);

    /// <summary>Self-description for diagnostics, matching the convention the RL components use.</summary>
    string Describe();
}

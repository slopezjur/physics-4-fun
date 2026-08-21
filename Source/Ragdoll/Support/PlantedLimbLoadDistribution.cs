using System.Collections.Generic;
using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Support;

/// <summary>
/// Shares body weight across whichever limbs are currently planted, so each support chain knows how
/// much load its feed-forward has to hold. This is what lets light arms push an 80 kg torso off the
/// floor without faking gains or inertia.
///
/// Extracted from HumanoidRagdoll, which was carrying it alongside bone registry, state
/// transitions, pose authoring and actuator driving. It is a physics *policy* - how much weight a
/// planted limb is asked to carry is a modelling decision with alternatives - and the body class
/// should not be the only place it can live. Behind an interface for the same reason the balance
/// strategies and the RL components are.
/// </summary>
public sealed class PlantedLimbLoadDistribution : ISupportLoadDistribution
{
    /// <summary>
    /// Limb end-effectors that can act as ground struts. When one is in contact, its whole limb
    /// chain carries a share of body weight and needs support-load compensation, not just gravity
    /// compensation for its own (light) distal mass.
    /// </summary>
    private static readonly string[] SupportEndEffectors = { "Forearm_L", "Forearm_R", "Foot_L", "Foot_R" };

    /// <summary>
    /// Bones the share stops at when walking up a limb chain. The torso is the load being carried,
    /// not a strut carrying it.
    /// </summary>
    private static readonly string[] TorsoStops = { "Chest", "Spine", "Pelvis" };

    /// <summary>Guard against a malformed parent chain looping forever.</summary>
    private const int MaxChainWalk = 32;

    private readonly List<ActiveBone> _plantedLimbs = new();

    public void Distribute(IReadOnlyList<ActiveBone> allBones)
    {
        foreach (ActiveBone bone in allBones)
        {
            bone.SupportedMassShare = 0.0f;
        }

        _plantedLimbs.Clear();
        foreach (ActiveBone bone in allBones)
        {
            if (!GodotObject.IsInstanceValid(bone)
                || System.Array.IndexOf(SupportEndEffectors, bone.BoneName) < 0)
            {
                continue;
            }

            if (bone.IsInContactWithWorld())
            {
                _plantedLimbs.Add(bone);
            }
        }

        if (_plantedLimbs.Count == 0)
        {
            return;
        }

        float totalMass = 0.0f;
        foreach (ActiveBone bone in allBones)
        {
            totalMass += bone.Mass;
        }

        // KNOWN OVER-ESTIMATE, kept deliberately for now: this is total body mass, so a planted
        // limb is asked to hold its own weight and the other limbs' as well. The physically correct
        // share is the mass ABOVE the support, which for a single planted leg is roughly 25-30%
        // less on this rig (the leg pair is 28 kg of 80.6 kg).
        //
        // Left as-is because changing it changes the feed-forward on every episode of every task,
        // and it should be one measured change rather than a side effect of moving the code. The
        // bound in ActiveBone.LoadCompensationTorqueFraction limits what the over-estimate can do
        // in the meantime: the feed-forward can never take more than half the actuator budget.
        float share = totalMass / _plantedLimbs.Count;

        foreach (ActiveBone endEffector in _plantedLimbs)
        {
            ActiveBone? cursor = endEffector;
            int guard = 0;
            while (cursor != null && guard++ < MaxChainWalk)
            {
                if (System.Array.IndexOf(TorsoStops, cursor.BoneName) >= 0)
                {
                    break;
                }

                cursor.SupportedMassShare = share;
                cursor = cursor.ParentBone;
            }
        }
    }

    public string Describe() =>
        $"PlantedLimbLoadDistribution(endEffectors=[{string.Join(",", SupportEndEffectors)}], "
        + $"stopsAt=[{string.Join(",", TorsoStops)}], share=totalMass/plantedCount)";
}

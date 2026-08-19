using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

public readonly record struct BalanceContext(
    ActiveBone Pelvis,
    ActiveBone? Spine,
    ActiveBone? Chest,
    ActiveBone? Head,
    ActiveBone? ThighL,
    ActiveBone? ThighR,
    ActiveBone? ShinL,
    ActiveBone? ShinR,
    ActiveBone? FootL,
    ActiveBone? FootR,
    ActiveBone? ForearmL,
    ActiveBone? ForearmR,
    ActiveBone? HandL,
    ActiveBone? HandR,
    RagdollState State,
    Recovery.GetUpPhase GetUpPhase,
    StepPhase CurrentStepPhase,
    float Strength,
    float Delta,
    Vector3 CenterOfMass,
    Vector3 CenterOfMassVelocity,
    Vector3 Icp,
    Vector3 BaseOfSupportCenter,
    bool IsGroundedL,
    bool IsGroundedR,
    Vector3 GroundPointL,
    Vector3 GroundPointR,
    bool IsSettleGraceActive
);

public interface IBalanceStrategy
{
    void Reset();
    void Apply(in BalanceContext context);
}

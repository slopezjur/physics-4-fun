using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

public readonly record struct BalanceContext(
    IBoneState Pelvis,
    IBoneState? Spine,
    IBoneState? Chest,
    IBoneState? Head,
    IBoneState? ThighL,
    IBoneState? ThighR,
    IBoneState? ShinL,
    IBoneState? ShinR,
    IBoneState? FootL,
    IBoneState? FootR,
    IBoneState? ForearmL,
    IBoneState? ForearmR,
    IBoneState? HandL,
    IBoneState? HandR,
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

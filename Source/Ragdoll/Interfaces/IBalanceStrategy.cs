using Godot;

namespace Physics4Fun.Ragdoll.Interfaces;

public readonly record struct BalanceContext(
    IBoneActuator Pelvis,
    IBoneActuator? Spine,
    IBoneActuator? Chest,
    IBoneActuator? Head,
    IBoneActuator? ThighL,
    IBoneActuator? ThighR,
    IBoneActuator? ShinL,
    IBoneActuator? ShinR,
    IBoneActuator? FootL,
    IBoneActuator? FootR,
    IBoneActuator? ForearmL,
    IBoneActuator? ForearmR,
    IBoneActuator? HandL,
    IBoneActuator? HandR,
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

using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Encapsulates continuous asymmetric weight transfer and dynamic joint impedance modulation (SRP).
/// Depends on DynamicSteppingModule for the current step phase/progress (must run after it in the
/// balance pipeline) — an explicit, intentional collaboration between two same-tick strategies.
/// </summary>
public class WeightTransferModule : IBalanceStrategy
{
    private readonly DynamicSteppingModule _stepping;

    public WeightTransferModule(DynamicSteppingModule stepping)
    {
        _stepping = stepping;
    }

    public float CurrentWeightShareL { get; private set; } = 0.5f;
    public float CurrentWeightShareR { get; private set; } = 0.5f;

    public void Reset()
    {
        CurrentWeightShareL = 0.5f;
        CurrentWeightShareR = 0.5f;
    }

    public void Apply(in BalanceContext context)
    {
        // Upright double-support weight shifting only. The other strategies already self-gate this
        // way; this one did not, and once the tilt gate stopped blanket-disabling the pipeline
        // during recovery it would otherwise have started shifting weight between feet that are
        // still folded underneath a body lying on the floor.
        if (context.State != RagdollState.Balanced && context.State != RagdollState.Stumbling)
        {
            Reset();
            return;
        }

        // Lateral CoM error in the yaw-level frame drives double-support weight shifting
        float lateralComError = 0.0f;
        IBoneState? footL = context.FootL;
        IBoneState? footR = context.FootR;
        if (footL != null && footR != null && footL.IsValid && footR.IsValid)
        {
            Vector3 supportCenter = (footL.GlobalPosition + footR.GlobalPosition) * 0.5f;
            Vector3 flatForward = -context.Pelvis.GlobalTransform.Basis.Z;
            flatForward.Y = 0.0f;
            flatForward = flatForward.Normalized();
            Vector3 levelRight = flatForward.Cross(Vector3.Up);
            lateralComError = (context.CenterOfMass - supportCenter).Dot(levelRight);
        }

        StepPhase stepPhase = _stepping.CurrentStepPhase;
        float stepProgress = _stepping.StepProgress;
        IBoneState? thighL = context.ThighL;
        IBoneState? thighR = context.ThighR;
        IBoneState? shinL = context.ShinL;
        IBoneState? shinR = context.ShinR;
        float delta = context.Delta;

        float targetShareL = 0.5f;
        float targetShareR = 0.5f;

        if (stepPhase == StepPhase.LeftSwing)
        {
            // Left leg in swing -> Right leg takes ~95% stance load so the swing foot truly unweights
            float s = Mathf.Clamp(stepProgress, 0.0f, 1.0f);
            float swingWeight = 0.05f + 0.05f * (1.0f - Mathf.Sin(Mathf.Pi * s));
            targetShareL = swingWeight;
            targetShareR = 1.0f - swingWeight;
        }
        else if (stepPhase == StepPhase.RightSwing)
        {
            // Right leg in swing -> Left leg takes ~95% stance load so the swing foot truly unweights
            float s = Mathf.Clamp(stepProgress, 0.0f, 1.0f);
            float swingWeight = 0.05f + 0.05f * (1.0f - Mathf.Sin(Mathf.Pi * s));
            targetShareR = swingWeight;
            targetShareL = 1.0f - swingWeight;
        }
        else
        {
            // Double support lateral balance: press harder on the leg the CoM leans toward,
            // shifting the CoP under the CoM (left leg is the +X side of the rig)
            float shift = Mathf.Clamp(lateralComError * 1.5f, -0.15f, 0.15f);
            targetShareL = 0.5f + shift;
            targetShareR = 1.0f - targetShareL;
        }

        // Smooth continuous exponential blend
        float blendSpeed = 8.0f;
        CurrentWeightShareL = Mathf.Lerp(CurrentWeightShareL, targetShareL, Mathf.Clamp(delta * blendSpeed, 0.0f, 1.0f));
        CurrentWeightShareR = Mathf.Lerp(CurrentWeightShareR, targetShareR, Mathf.Clamp(delta * blendSpeed, 0.0f, 1.0f));

        // Dynamic Joint Impedance Modulation (stance leg stiffens, swing leg relaxes)
        float limbStrengthL = 1.0f;
        float limbStrengthR = 1.0f;

        if (stepPhase == StepPhase.LeftSwing)
        {
            limbStrengthL = 0.75f;
            limbStrengthR = 1.20f;
        }
        else if (stepPhase == StepPhase.RightSwing)
        {
            limbStrengthR = 0.75f;
            limbStrengthL = 1.20f;
        }
        else
        {
            // In double support, modestly stiffen the loaded leg so it resists sinking on the lean side
            limbStrengthL = 1.0f + (CurrentWeightShareL - 0.5f) * 0.4f;
            limbStrengthR = 1.0f + (CurrentWeightShareR - 0.5f) * 0.4f;
        }

        if (thighL != null && shinL != null && thighL.IsValid && shinL.IsValid)
        {
            thighL.MuscleStrength = limbStrengthL;
            shinL.MuscleStrength = limbStrengthL;
        }

        if (thighR != null && shinR != null && thighR.IsValid && shinR.IsValid)
        {
            thighR.MuscleStrength = limbStrengthR;
            shinR.MuscleStrength = limbStrengthR;
        }
    }
}

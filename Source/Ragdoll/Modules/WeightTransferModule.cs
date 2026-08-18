using Godot;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Encapsulates continuous asymmetric weight transfer and dynamic joint impedance modulation (SRP).
/// </summary>
public class WeightTransferModule
{
    public float CurrentWeightShareL { get; private set; } = 0.5f;
    public float CurrentWeightShareR { get; private set; } = 0.5f;

    public void Reset()
    {
        CurrentWeightShareL = 0.5f;
        CurrentWeightShareR = 0.5f;
    }

    public void Update(
        StepPhase stepPhase,
        float stepProgress,
        float lateralComError,
        ActiveBone? thighL,
        ActiveBone? thighR,
        ActiveBone? shinL,
        ActiveBone? shinR,
        float delta)
    {
        float targetShareL = 0.5f;
        float targetShareR = 0.5f;

        if (stepPhase == StepPhase.LeftSwing)
        {
            // Left leg in swing -> Right leg takes 80% stance load, Left has 20% compliant swing support
            float s = Mathf.Clamp(stepProgress, 0.0f, 1.0f);
            float swingWeight = 0.20f + 0.10f * (1.0f - Mathf.Sin(Mathf.Pi * s));
            targetShareL = swingWeight;
            targetShareR = 1.0f - swingWeight;
        }
        else if (stepPhase == StepPhase.RightSwing)
        {
            // Right leg in swing -> Left leg takes 80% stance load, Right has 20% compliant swing support
            float s = Mathf.Clamp(stepProgress, 0.0f, 1.0f);
            float swingWeight = 0.20f + 0.10f * (1.0f - Mathf.Sin(Mathf.Pi * s));
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

        if (thighL != null && shinL != null && GodotObject.IsInstanceValid(thighL) && GodotObject.IsInstanceValid(shinL))
        {
            thighL.MuscleStrength = limbStrengthL;
            shinL.MuscleStrength = limbStrengthL;
        }

        if (thighR != null && shinR != null && GodotObject.IsInstanceValid(thighR) && GodotObject.IsInstanceValid(shinR))
        {
            thighR.MuscleStrength = limbStrengthR;
            shinR.MuscleStrength = limbStrengthR;
        }
    }
}

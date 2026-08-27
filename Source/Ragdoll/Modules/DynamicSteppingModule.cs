using Godot;
using Physics4Fun.Ragdoll;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules;

/// <summary>
/// Encapsulates Instantaneous Capture Point (ICP) orbital dynamics and 2-bone analytical stepping IK (SRP).
/// </summary>
public class DynamicSteppingModule : IBalanceStrategy
{
    public bool EnableDynamicStepping { get; set; } = true;
    public float StepDuration { get; set; } = 0.28f;
    public float StepHeight { get; set; } = 0.08f;
    public float StanceWidth { get; set; } = 0.28f;

    public StepPhase CurrentStepPhase { get; private set; } = StepPhase.DoubleSupport;
    public float StepProgress { get; private set; } = 0.0f;

    private Vector3 _stepStartPos = Vector3.Zero;
    private Vector3 _stepTargetPos = Vector3.Zero;
    private float _doubleSupportDuration = 0.0f;
    private bool _lastSwingWasLeft = false;

    // Leg segment lengths measured once from the actual skeleton (thigh->shin, shin->foot)
    private float _legLen1 = 0.27f;
    private float _legLen2 = 0.27f;
    private bool _legLengthsMeasured = false;

    public void Reset()
    {
        CurrentStepPhase = StepPhase.DoubleSupport;
        StepProgress = 0.0f;
        _doubleSupportDuration = 0.0f;
        _lastSwingWasLeft = false;
    }

    public void Apply(in BalanceContext context)
    {
        if (context.State != RagdollState.Balanced && context.State != RagdollState.Stumbling)
        {
            Reset();
            return;
        }

        if (!EnableDynamicStepping || context.FootL == null || context.FootR == null || !context.FootL.IsValid || !context.FootR.IsValid)
        {
            CurrentStepPhase = StepPhase.DoubleSupport;
            return;
        }

        IBoneState pelvis = context.Pelvis;
        IBoneState? thighL = context.ThighL;
        IBoneState? thighR = context.ThighR;
        IBoneState? shinL = context.ShinL;
        IBoneState? shinR = context.ShinR;
        IBoneState footL = context.FootL;
        IBoneState footR = context.FootR;
        bool isGroundedL = context.IsGroundedL;
        bool isGroundedR = context.IsGroundedR;
        Vector3 groundPointL = context.GroundPointL;
        Vector3 groundPointR = context.GroundPointR;
        bool isSettleGraceActive = context.IsSettleGraceActive;
        float strength = context.Strength;
        float delta = context.Delta;

        Vector3 icp = context.Icp;
        Vector3 supportCenter = context.BaseOfSupportCenter;

        if (CurrentStepPhase == StepPhase.DoubleSupport)
        {
            _doubleSupportDuration += delta;

            if (thighL != null) thighL.FeedForwardTargetOffset = Quaternion.Identity;
            if (thighR != null) thighR.FeedForwardTargetOffset = Quaternion.Identity;
            if (shinL != null) shinL.FeedForwardTargetOffset = Quaternion.Identity;
            if (shinR != null) shinR.FeedForwardTargetOffset = Quaternion.Identity;

            // Use a yaw-only (level) pelvis basis so pelvis pitch/roll during a fall
            // does not bleed horizontal ICP escape into the local vertical axis.
            Basis levelBasis = BiomechanicalKinematics.ComputeLevelBasis(pelvis.GlobalTransform.Basis);
            Vector3 localIcp = levelBasis.Inverse() * (icp - supportCenter);
            float pelvisSpeed = new Vector2(pelvis.LinearVelocity.X, pelvis.LinearVelocity.Z).Length();
            float tiltCos = pelvis.GlobalTransform.Basis.Y.Normalized().Dot(Vector3.Up);

            bool isSeriouslyPerturbed = pelvisSpeed > 0.35f || tiltCos < 0.95f; // Speed > 0.35m/s or tilt > 18 deg
            bool isIcpEscaped = Mathf.Abs(localIcp.X) > 0.12f || Mathf.Abs(localIcp.Z) > 0.15f;

            // Trigger a step when ICP leaves the support base
            if (!isSettleGraceActive && _doubleSupportDuration >= 0.04f && (isSeriouslyPerturbed || isIcpEscaped))
            {
                _doubleSupportDuration = 0.0f;
                // If falling predominantly forward/backward, we MUST alternate legs to run/walk without tripping.
                // Only override the alternating walk cycle if the fall is strongly lateral.
                bool isLateralFall = Mathf.Abs(localIcp.X) > 0.06f && Mathf.Abs(localIcp.X) > Mathf.Abs(localIcp.Z) * 0.4f;
                bool swingLeft = isLateralFall ? (localIcp.X > 0.0f) : !_lastSwingWasLeft;
                _lastSwingWasLeft = swingLeft;

                CurrentStepPhase = swingLeft ? StepPhase.LeftSwing : StepPhase.RightSwing;
                StepProgress = 0.0f;

                IBoneState swingFoot = swingLeft ? footL : footR;
                _stepStartPos = swingFoot.GlobalPosition;

                Vector3 hipPos = pelvis.GlobalTransform * new Vector3(swingLeft ? 0.14f : -0.14f, -0.06f, 0.0f);
                // Left side is +X (matches the hip offsets above), so the left stance offset is +X
                Vector3 sideOffset = pelvis.GlobalTransform.Basis.X * (swingLeft ? StanceWidth * 0.5f : -StanceWidth * 0.5f);
                Vector3 rawTarget = icp + sideOffset;
                Vector3 stepOffset = rawTarget - hipPos;
                stepOffset.Y = 0.0f;
                if (stepOffset.Length() > 0.52f)
                {
                    stepOffset = stepOffset.Normalized() * 0.52f;
                }

                _stepTargetPos = hipPos + stepOffset;
                // When the swing foot's ground ray misses, fall back to the stance foot's ground Y
                bool swingGroundValid = swingLeft ? isGroundedL : isGroundedR;
                _stepTargetPos.Y = swingGroundValid
                    ? (swingLeft ? groundPointL.Y : groundPointR.Y)
                    : (swingLeft ? groundPointR.Y : groundPointL.Y);
            }
        }
        else
        {
            // Dynamic fast step duration based on perturbation speed
            float pelvisSpeed = new Vector2(pelvis.LinearVelocity.X, pelvis.LinearVelocity.Z).Length();
            float dynamicDuration = Mathf.Clamp(0.16f / (1.0f + pelvisSpeed * 0.6f), 0.09f, 0.20f);

            // Execute Swing Phase Arc
            StepProgress += delta / dynamicDuration;
            float s = Mathf.Clamp(StepProgress, 0.0f, 1.0f);
            float smoothS = s * s * (3.0f - 2.0f * s);

            Vector3 currentFootTarget = _stepStartPos.Lerp(_stepTargetPos, smoothS);
            currentFootTarget.Y += 4.0f * StepHeight * s * (1.0f - s);

            bool isLeft = CurrentStepPhase == StepPhase.LeftSwing;
            IBoneState? swingThigh = isLeft ? thighL : thighR;
            IBoneState? swingShin = isLeft ? shinL : shinR;

            if (swingThigh != null && swingShin != null && swingThigh.IsValid && swingShin.IsValid)
            {
                // Measure real segment lengths once from the skeleton instead of hardcoding
                if (!_legLengthsMeasured)
                {
                    _legLen1 = Mathf.Max(0.05f, swingThigh.GlobalPosition.DistanceTo(swingShin.GlobalPosition));
                    _legLen2 = Mathf.Max(0.05f, swingShin.GlobalPosition.DistanceTo((isLeft ? footL : footR).GlobalPosition));
                    _legLengthsMeasured = true;
                }

                Vector3 hipPos = pelvis.GlobalTransform * new Vector3(isLeft ? 0.14f : -0.14f, -0.06f, 0.0f);
                Vector3 worldTargetVec = currentFootTarget - hipPos;
                Vector3 localTargetVec = pelvis.GlobalTransform.Basis.Inverse() * worldTargetVec;

                float legLen1 = _legLen1;
                float legLen2 = _legLen2;
                float dist = Mathf.Clamp(localTargetVec.Length(), 0.05f, (legLen1 + legLen2) * 0.98f);

                float cosKnee = (legLen1 * legLen1 + legLen2 * legLen2 - dist * dist) / (2.0f * legLen1 * legLen2);
                float kneeAngle = Mathf.Pi - Mathf.Acos(Mathf.Clamp(cosKnee, -1.0f, 1.0f));

                float cosHip = (dist * dist + legLen1 * legLen1 - legLen2 * legLen2) / (2.0f * dist * legLen1);
                float hipIKOffset = Mathf.Acos(Mathf.Clamp(cosHip, -1.0f, 1.0f));

                float hipPitch = Mathf.Atan2(-localTargetVec.Z, -localTargetVec.Y);
                // Because the knee bends backward (negative pitch), we must rotate the thigh forward (positive pitch)
                hipPitch += hipIKOffset;

                float hipRoll = Mathf.Atan2(localTargetVec.X, -localTargetVec.Y);

                swingThigh.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(hipPitch * strength, 0.0f, hipRoll * strength));
                // Negative X flexes knee backward anatomically
                swingShin.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(-kneeAngle * strength, 0.0f, 0.0f));
            }

            // Maintain upright stance leg posture without fighting joint limits
            IBoneState? stanceThigh = isLeft ? thighR : thighL;
            IBoneState? stanceShin = isLeft ? shinR : shinL;
            if (stanceThigh != null && stanceShin != null && stanceThigh.IsValid && stanceShin.IsValid)
            {
                Vector3 localUp = pelvis.GlobalTransform.Basis.Inverse() * Vector3.Up;
                float stancePitch = Mathf.Clamp(Mathf.Atan2(localUp.Z, localUp.Y) * 0.8f, -0.30f, 0.30f);
                // Strong lateral lean onto the stance side: actively shifts the CoM over the stance
                // foot so the swing leg unweights and can actually leave the ground.
                float stanceRoll = Mathf.Clamp(Mathf.Atan2(-localUp.X, localUp.Y) * 0.8f + (isLeft ? -0.12f : 0.12f), -0.25f, 0.25f);

                stanceShin.FeedForwardTargetOffset = Quaternion.Identity;
                // Same corrective convention as BalanceController's hip posture: +pitch brakes the fall
                // (the previous -stancePitch sign was regenerative and fought the double-support path)
                stanceThigh.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(stancePitch * strength, 0.0f, stanceRoll * strength));
            }

            IBoneState swingFoot = isLeft ? footL : footR;
            bool isGroundedSwing = isLeft ? isGroundedL : isGroundedR;
            // Commit to the swing: only real foot contact late in the arc (descending) counts as
            // touchdown. Aborting on early contact/height thrashed the step machine — the swing
            // foot was still loaded from stance, so every step aborted at s~0.45 without relocating.
            bool touchdown = isGroundedSwing && s > 0.60f;

            if (s >= 1.0f || touchdown)
            {
                CurrentStepPhase = StepPhase.DoubleSupport;
            }
        }
    }
}

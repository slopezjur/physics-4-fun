using Godot;
using Physics4Fun.Ragdoll.Interfaces;

namespace Physics4Fun.Ragdoll.Modules.Reflexes;

/// <summary>
/// Autonomous reflex for reactive arm counter-balancing and protective ground impact bracing (Euphoria DMS / SRP).
/// </summary>
public class ArmReflexModule : IBiomechanicalReflex
{
    public bool IsEnabled { get; set; } = true;
    public float ArmCounterBalanceGain { get; set; } = 0.08f;
    public float ImpactBraceDistance { get; set; } = 1.4f;

    private IBoneState? _pelvis;
    private IBoneState? _chest;
    private IBoneState? _upperArmL;
    private IBoneState? _upperArmR;
    private IBoneState? _forearmL;
    private IBoneState? _forearmR;
    private Godot.Collections.Array<Rid> _ragdollRids = new();

    public void Initialize(
        IBoneState pelvis,
        IBoneState? chest,
        IBoneState? upperArmL,
        IBoneState? upperArmR,
        IBoneState? forearmL,
        IBoneState? forearmR,
        Godot.Collections.Array<Rid> ragdollRids)
    {
        _pelvis = pelvis;
        _chest = chest;
        _upperArmL = upperArmL;
        _upperArmR = upperArmR;
        _forearmL = forearmL;
        _forearmR = forearmR;
        _ragdollRids = ragdollRids;
    }

    public void Reset()
    {
        if (_upperArmL != null) _upperArmL.FeedForwardTargetOffset = Quaternion.Identity;
        if (_upperArmR != null) _upperArmR.FeedForwardTargetOffset = Quaternion.Identity;
        if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.Identity;
        if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.Identity;
    }

    public void Update(RagdollState state, StepPhase stepPhase, float delta)
    {
        if (!IsEnabled || _pelvis == null || _upperArmL == null || _upperArmR == null ||
            !_pelvis.IsValid || !_upperArmL.IsValid || !_upperArmR.IsValid)
        {
            return;
        }

        Vector3 angVel = _pelvis.AngularVelocity;
        Vector3 comVel = _pelvis.LinearVelocity;
        float tiltDeg = Mathf.RadToDeg(Mathf.Acos(Mathf.Clamp(_pelvis.GlobalTransform.Basis.Y.Normalized().Dot(Vector3.Up), -1.0f, 1.0f)));

        // 1. Proactive Impact Bracing / Floor Reach (Flailing, falling fast, or severe tilt)
        bool isImminentFall = state == RagdollState.Flailing || comVel.Y < -0.40f || tiltDeg > 28.0f;
        if (isImminentFall)
        {
        // A raycast is an ENGINE capability, not bone state - so this is the one place these
        // modules still need the concrete body. Cast here rather than widening IBoneState:
        // faking a physics query is not the same as faking a bone, and pretending otherwise
        // would make the interface untestable in the way it was added to prevent.
            var spaceState = ((Node3D)_pelvis).GetWorld3D().DirectSpaceState;
            Vector3 rayStart = (_chest != null && _chest.IsValid) ? _chest.GlobalPosition : _pelvis.GlobalPosition;
            Vector3 rayDir = comVel.Normalized();
            if (rayDir.Y > -0.2f) rayDir = (rayDir - new Vector3(0, 0.9f, 0)).Normalized();

            var query = PhysicsRayQueryParameters3D.Create(rayStart, rayStart + rayDir * ImpactBraceDistance, 1);
            query.Exclude = _ragdollRids;
            var hit = spaceState.IntersectRay(query);

            if (hit.Count > 0 || tiltDeg > 35.0f)
            {
                float reachPitch = 0.95f;
                float elbowBend = 0.65f;

                _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(reachPitch, 0.20f, -0.25f));
                _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(reachPitch, -0.20f, 0.25f));

                if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(elbowBend, 0, 0));
                if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(elbowBend, 0, 0));

                return;
            }
        }

        // 2. Active Centroidal Momentum Counter-Balancing (Balanced & Stumbling)
        if (state == RagdollState.Balanced || state == RagdollState.Stumbling)
        {
            float k = 0.05f;

            // Arms swing opposite to torso angular velocity and velocity
            float armPitchL = Mathf.Clamp(-angVel.X * k - comVel.Z * 0.15f, -0.80f, 0.80f);
            float armRollL = Mathf.Clamp(-angVel.Z * k - comVel.X * 0.15f, -0.65f, 0.40f);

            float armPitchR = Mathf.Clamp(-angVel.X * k - comVel.Z * 0.15f, -0.80f, 0.80f);
            float armRollR = Mathf.Clamp(angVel.Z * k + comVel.X * 0.15f, -0.40f, 0.65f);

            // Natural Gait Arm Swing during Stepping
            if (stepPhase == StepPhase.LeftSwing)
            {
                armPitchR += 0.35f;
                armPitchL -= 0.25f;
            }
            else if (stepPhase == StepPhase.RightSwing)
            {
                armPitchL += 0.35f;
                armPitchR -= 0.25f;
            }

            _upperArmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(armPitchL, 0.0f, armRollL));
            _upperArmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(armPitchR, 0.0f, armRollR));

            if (_forearmL != null) _forearmL.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.40f, 0, 0));
            if (_forearmR != null) _forearmR.FeedForwardTargetOffset = Quaternion.FromEuler(new Vector3(0.40f, 0, 0));
        }
        else
        {
            Reset();
        }
    }
}

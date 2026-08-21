using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Godot;

namespace Physics4Fun.Ragdoll.Diagnostics;

/// <summary>
/// High-resolution diagnostic telemetry recorder.
/// Captures comprehensive time-series snapshot of orbital dynamics, contact geometry,
/// joint flexion, actuator torques, and PID tracking errors to CSV.
/// </summary>
public class RagdollTelemetryRecorder
{
    /// <summary>Default recording window in seconds. Single source of truth for all entry points.</summary>
    public const float DefaultDurationSeconds = 10.0f;

    public bool IsRecording { get; private set; } = false;
    public float ElapsedRecordingTime { get; private set; } = 0.0f;
    public float MaxDuration { get; private set; } = DefaultDurationSeconds;

    public event Action? RecordingFinished;

    public float ProgressNormalized => MaxDuration > 0.0f ? Mathf.Clamp(ElapsedRecordingTime / MaxDuration, 0.0f, 1.0f) : 0.0f;

    /// <summary>
    /// Bones written to the per-frame CSV, in column order. The hands were absent until the
    /// Nyquist-chatter investigation: they are the lightest bones and therefore the first to go
    /// unstable, yet the forearm was the only proxy visible in the dump.
    /// </summary>
    private static readonly string[] BoneNames = {
        "Pelvis", "Spine", "Chest", "Head",
        "UpperArm_L", "Forearm_L", "Hand_L", "UpperArm_R", "Forearm_R", "Hand_R",
        "Thigh_L", "Shin_L", "Foot_L", "Thigh_R", "Shin_R", "Foot_R"
    };

    /// <summary>
    /// Per-bone column suffixes, in emission order. The single definition the header is built from
    /// and the absent-bone filler is sized by.
    ///
    /// It used to be a hand-maintained count sitting next to a separately hand-written header
    /// string, which is a duplication that fails SILENTLY: if the two drift, every column after the
    /// mistake is misaligned and the CSV still parses, so a dump reads as valid data about the
    /// wrong quantities. Adding the energy columns meant editing three places in sync and getting
    /// all three right. Now the count is derived, and ValidateRowWidth below catches any remaining
    /// header/row disagreement on the first frame rather than in an analysis a day later.
    /// </summary>
    private static readonly string[] BoneColumnSuffixes = {
        "PosY", "Strength", "TorqueMag", "TorqueX", "TorqueY", "TorqueZ",
        "EulerX", "EulerY", "EulerZ", "AngVelX", "AngVelY", "AngVelZ",
        "TrackingErrorDeg", "PdTorqueMag", "LoadTorqueMag",
        "LinVelMag", "PowerW", "FvScale"
    };

    private static int BoneColumnCount => BoneColumnSuffixes.Length;

    /// <summary>Filler for a bone that is absent from the rig, keeping the row aligned to the header.</summary>
    private static readonly string EmptyBoneColumns =
        string.Concat(Enumerable.Repeat(",0", BoneColumnSuffixes.Length));

    /// <summary>Gravity (m/s^2) used for the potential-energy column; matches ActiveBone.</summary>
    private const float GravityForEnergy = 9.81f;

    private int _ballCount;
    private float _ballSpeed;
    private string _ballImpactBone = "-";
    private float _ballImpactImpulse;
    private float _ballMassRatio;
    private float _ballMassActual;

    /// <summary>
    /// Latest projectile state, pushed each physics tick by whatever perturbation source is armed.
    ///
    /// Pushed rather than pulled because this recorder lives on the ragdoll, and the ragdoll must
    /// not acquire a dependency on the RL layer's ball gun to be able to describe itself.
    ///
    /// BallImpactImpulse is the momentum the ball MEASURABLY lost in a tick where it was touching a
    /// bone (mass * |dv|), which by Newton's third law is what the bone received. It exists because
    /// the launch momentum only says what the ball should deliver; five rounds of chasing a
    /// "shotgun" impact went by without anyone able to state what it actually delivered.
    /// BallMassRatio is the struck bone's mass over the ball's - the quantity a sequential-impulse
    /// solver degrades on, and the one that got worse every time the ball was made lighter.
    /// </summary>
    public void ReportProjectile(int count, float speed, string impactBone, float impulse,
                                float massRatio, float massActual)
    {
        _ballMassActual = massActual;
        _ballCount = count;
        _ballSpeed = speed;
        _ballImpactBone = string.IsNullOrEmpty(impactBone) ? "-" : impactBone;
        _ballImpactImpulse = impulse;
        _ballMassRatio = massRatio;
    }

    /// <summary>Angular speed (rad/s) below which a direction reversal is treated as noise, not chatter.</summary>
    private const float ChatterAngVelFloor = 0.05f;

    /// <summary>Direction reversals per second above which a joint is reported as chattering.</summary>
    private const float ChatterReversalRate = 30.0f;

    /// <summary>Achieved damping ratio below which a joint is flagged as ringing / prone to shedding load.</summary>
    private const float UnderdampedRatio = 0.35f;

    private readonly List<string> _csvRows = new();
    private readonly Dictionary<string, BoneDiagnostics> _boneDiagnostics = new();

    /// <summary>Bones seen commanded outside their joint envelope at any point in the recording.</summary>
    private readonly HashSet<string> _jointLimitOffenders = new();

    private float _deltaSum;
    private readonly CultureInfo _inv = CultureInfo.InvariantCulture;

    /// <summary>
    /// Running per-bone actuator statistics accumulated across the recording, written to the
    /// companion actuators.csv. These are the quantities that identify a discrete PD limit cycle:
    /// sustained angular speed plus a high angular-velocity reversal rate.
    /// </summary>
    private sealed class BoneDiagnostics
    {
        public float Mass;
        public float InertiaSum;
        public float InertiaMin;
        public float ProportionalGain;
        public float DerivativeGain;
        public float MaxTorque;
        public bool AutoTuned;

        public int Samples;
        public int Reversals;
        public Vector3 PreviousAngVel;
        public float AngVelSum;
        public float AngVelMax;
        public float ErrorSum;
        public float ErrorMax;
        public float TorqueSum;
        public float TorqueMax;
        public Vector3 TorqueVectorSum;
    }

    public void StartRecording(float duration = DefaultDurationSeconds)
    {
        MaxDuration = duration;
        ElapsedRecordingTime = 0.0f;
        IsRecording = true;
        _csvRows.Clear();
        _boneDiagnostics.Clear();
        _jointLimitOffenders.Clear();
        _deltaSum = 0.0f;

        // Build CSV Header
        var sb = new StringBuilder();
        sb.Append("Time,Delta,State,Orientation,GetUpPhase,RecoveryProgress,");
        sb.Append("PelvisPosX,PelvisPosY,PelvisPosZ,PelvisVelX,PelvisVelY,PelvisVelZ,PelvisSpeed,PelvisTiltDeg,PelvisAngVelX,PelvisAngVelY,PelvisAngVelZ,");
        sb.Append("CoMPosX,CoMPosY,CoMPosZ,CoMVelX,CoMVelY,CoMVelZ,");
        sb.Append("IcpX,IcpZ,SupportCenterX,SupportCenterZ,LocalIcpX,LocalIcpZ,");
        sb.Append("KneeAngleL_Deg,KneeAngleR_Deg,LegLengthL,LegLengthR,");
        sb.Append("ElbowAngleL_Deg,ElbowAngleR_Deg,DrillCycle,");
        sb.Append("GroundPointL_Y,GroundPointR_Y,FootElevationL,FootElevationR,");
        sb.Append("IsGroundedL,IsGroundedR,WeightShareL,WeightShareR,BalanceStrength,StateTimer,IcpEscape,SwingProgress,PelvisStabTorque,StepPhase,");

        // Get-up decision inputs: the ground datum the phase controller reasons about plus every
        // quantity its exit criteria are judged on, so a phase that times out can be read directly
        // against how close it came instead of being inferred.
        sb.Append("GroundDatumY,ChestClearance,PelvisClearance,LeadFootComDist,HandsPlanted,LeadFootPlanted,TrailFootPlanted,LeadSide,PhaseElapsed,PhaseTimedOut,JointLimitViolations");

        // Whole-body energy bookkeeping. A torque ceiling bounds FORCE, not POWER: an actuator held
        // at its ceiling while the joint spins at Jolt's 47.12 rad/s angular-velocity clamp delivers
        // tau*omega watts, and nothing in this rig limits that product. Without these columns the
        // only evidence of energy injection was the body ending up faster than anything that hit it,
        // which is a conclusion three steps removed from a measurement. SystemEnergy should fall
        // during contact and rise only by roughly ActuatorPowerW * Delta; any tick where it gains
        // far more than the actuators could have supplied is the solver manufacturing energy.
        // ActuatorPowerInW sums only joints doing POSITIVE work - the actuator driving the joint,
        // which is the only term that can add mechanical energy. The previous single column summed
        // |tau.omega| and so counted braking as power too, which made a limit that only bounds
        // driving look like it had failed. Absorption is reported separately rather than discarded,
        // because a body that stops absorbing is also a body about to fall over.
        sb.Append(",SystemKE,SystemPE,SystemEnergy,ActuatorPowerInW,ActuatorPowerOutW,PeakJointPowerInW,PeakJointPowerBone");
        sb.Append(",BallCount,BallSpeed,BallImpactBone,BallImpactImpulse,BallMassRatio,BallMassActual");

        foreach (string name in BoneNames)
        {
            // LinVelMag: per-bone linear speed was absent entirely, so per-bone kinetic energy
            // could not be computed and only the CoM aggregate was visible - which cancels exactly
            // the internal flailing that matters here. PowerW: this joint's tau*omega, the term the
            // torque ceiling does not bound. FvScale: whether the Hill limit engaged.
            foreach (string suffix in BoneColumnSuffixes)
            {
                sb.Append($",{name}_{suffix}");
            }
        }

        _csvRows.Add(sb.ToString());
        GD.Print($"[RagdollTelemetryRecorder] Started {duration:F1}s comprehensive telemetry recording session...");
    }

    public void RecordFrame(
        float delta,
        RagdollState state,
        BalanceController balance,
        ActiveBone pelvis,
        IReadOnlyList<ActiveBone> bones,
        Recovery.GetUpPhaseController getUpPhases,
        BodySide leadSide,
        float drillCycle)
    {
        if (!IsRecording) return;

        ElapsedRecordingTime += delta;

        var sb = new StringBuilder();
        sb.Append($"{ElapsedRecordingTime.ToString("F4", _inv)},{delta.ToString("F4", _inv)},{state},{balance.CurrentOrientation},{getUpPhases.CurrentPhase},{balance.RecoveryProgressNormalized.ToString("F3", _inv)},");
        
        // 1. Pelvis & CoM Kinematics
        Vector3 pPos = pelvis.GlobalPosition;
        Vector3 pVel = pelvis.LinearVelocity;
        Vector3 pAngVel = pelvis.AngularVelocity;
        Vector3 com = balance.CenterOfMass;
        Vector3 comVel = balance.CenterOfMassVelocity;

        sb.Append($"{pPos.X.ToString("F3", _inv)},{pPos.Y.ToString("F3", _inv)},{pPos.Z.ToString("F3", _inv)},");
        sb.Append($"{pVel.X.ToString("F3", _inv)},{pVel.Y.ToString("F3", _inv)},{pVel.Z.ToString("F3", _inv)},{pVel.Length().ToString("F3", _inv)},");
        sb.Append($"{balance.CurrentTiltAngleDeg.ToString("F2", _inv)},");
        sb.Append($"{pAngVel.X.ToString("F3", _inv)},{pAngVel.Y.ToString("F3", _inv)},{pAngVel.Z.ToString("F3", _inv)},");
        sb.Append($"{com.X.ToString("F3", _inv)},{com.Y.ToString("F3", _inv)},{com.Z.ToString("F3", _inv)},");
        sb.Append($"{comVel.X.ToString("F3", _inv)},{comVel.Y.ToString("F3", _inv)},{comVel.Z.ToString("F3", _inv)},");

        // 2. Orbital ICP & Support Polygon Geometry
        // Omega from true CoM height above the average ground-point level (min clamp 0.2 m)
        float groundAvgY = (balance.GroundPointL.Y + balance.GroundPointR.Y) * 0.5f;
        float omega0 = Mathf.Sqrt(9.81f / Mathf.Max(0.2f, com.Y - groundAvgY));
        Vector3 icp = com + (new Vector3(comVel.X, 0, comVel.Z) / omega0);

        Vector3 footPosL = balance.FootL != null && GodotObject.IsInstanceValid(balance.FootL) ? balance.FootL.GlobalPosition : pPos;
        Vector3 footPosR = balance.FootR != null && GodotObject.IsInstanceValid(balance.FootR) ? balance.FootR.GlobalPosition : pPos;
        Vector3 supportCenter = (footPosL + footPosR) * 0.5f;

        // Yaw-only level basis: pelvis forward flattened onto the horizontal plane (Up = +Y, X = Up x Z)
        Vector3 pelvisFwd = -pelvis.GlobalTransform.Basis.Z;
        Vector3 levelZ = new Vector3(pelvisFwd.X, 0.0f, pelvisFwd.Z);
        levelZ = levelZ.LengthSquared() > 1e-6f ? levelZ.Normalized() : Vector3.Forward;
        Vector3 levelX = Vector3.Up.Cross(levelZ);
        Vector3 icpOffset = icp - supportCenter;
        float localIcpX = levelX.Dot(icpOffset);
        float localIcpZ = levelZ.Dot(icpOffset);

        sb.Append($"{icp.X.ToString("F3", _inv)},{icp.Z.ToString("F3", _inv)},");
        sb.Append($"{supportCenter.X.ToString("F3", _inv)},{supportCenter.Z.ToString("F3", _inv)},");
        sb.Append($"{localIcpX.ToString("F3", _inv)},{localIcpZ.ToString("F3", _inv)},");

        // Indexed bone lookup, built once and reused by the elbow geometry below and the per-bone
        // columns further down.
        var boneMap = new Dictionary<string, ActiveBone>();
        foreach (var b in bones)
        {
            if (GodotObject.IsInstanceValid(b))
            {
                boneMap[b.BoneName] = b;
            }
        }

        // 3. Knee Flexion & Leg Extension Geometry
        float kneeAngleL = 0.0f, kneeAngleR = 0.0f;
        float legLengthL = 0.0f, legLengthR = 0.0f;

        if (balance.ThighL != null && balance.ShinL != null && balance.FootL != null
            && GodotObject.IsInstanceValid(balance.ThighL) && GodotObject.IsInstanceValid(balance.ShinL) && GodotObject.IsInstanceValid(balance.FootL))
        {
            legLengthL = (balance.ThighL.GlobalPosition - balance.FootL.GlobalPosition).Length();
            kneeAngleL = ComputeJointFlexionDeg(balance.ThighL.GlobalPosition, balance.ShinL.GlobalPosition, balance.FootL.GlobalPosition);
        }

        if (balance.ThighR != null && balance.ShinR != null && balance.FootR != null
            && GodotObject.IsInstanceValid(balance.ThighR) && GodotObject.IsInstanceValid(balance.ShinR) && GodotObject.IsInstanceValid(balance.FootR))
        {
            legLengthR = (balance.ThighR.GlobalPosition - balance.FootR.GlobalPosition).Length();
            kneeAngleR = ComputeJointFlexionDeg(balance.ThighR.GlobalPosition, balance.ShinR.GlobalPosition, balance.FootR.GlobalPosition);
        }

        sb.Append($"{kneeAngleL.ToString("F1", _inv)},{kneeAngleR.ToString("F1", _inv)},{legLengthL.ToString("F3", _inv)},{legLengthR.ToString("F3", _inv)},");

        // 3b. Elbow extension. This is what separates a real arm press from a spine arch: chest
        // height alone was satisfied by the back curling into a cobra while the arms did nothing.
        float elbowAngleL = ComputeElbowFlexionDeg(boneMap, "UpperArm_L", "Forearm_L", "Hand_L");
        float elbowAngleR = ComputeElbowFlexionDeg(boneMap, "UpperArm_R", "Forearm_R", "Hand_R");
        sb.Append($"{elbowAngleL.ToString("F1", _inv)},{elbowAngleR.ToString("F1", _inv)},{drillCycle.ToString("F3", _inv)},");

        // 4. Ground Points & Foot Elevations
        float groundPointLY = balance.GroundPointL.Y;
        float groundPointRY = balance.GroundPointR.Y;
        float footElevationL = footPosL.Y - groundPointLY;
        float footElevationR = footPosR.Y - groundPointRY;

        sb.Append($"{groundPointLY.ToString("F3", _inv)},{groundPointRY.ToString("F3", _inv)},{footElevationL.ToString("F3", _inv)},{footElevationR.ToString("F3", _inv)},");

        // 5. Ground & Balance State
        sb.Append($"{(balance.IsGroundedL ? 1 : 0)},{(balance.IsGroundedR ? 1 : 0)},");
        sb.Append($"{balance.CurrentWeightShareL.ToString("F3", _inv)},{balance.CurrentWeightShareR.ToString("F3", _inv)},");
        sb.Append($"{balance.BalanceStrengthNow.ToString("F2", _inv)},{balance.StateTimerValue.ToString("F2", _inv)},");
        sb.Append($"{balance.IcpEscapeDistance.ToString("F3", _inv)},{balance.SwingProgressNormalized.ToString("F2", _inv)},");
        sb.Append($"{balance.PelvisStabilizerTorque.Length().ToString("F1", _inv)},{balance.CurrentStepPhase},");

        // 5b. Get-up decision inputs, as measured by the phase controller itself.
        // JointLimitViolations counts bones whose commanded target lies outside the joint's own
        // envelope. Anything above zero means an actuator is pinned against a hard stop at full
        // torque achieving nothing - the failure mode that kept the shoulders jammed through every
        // push-up attempt while the trajectory asked for twice the roll the joint allows.
        int jointLimitViolations = 0;
        foreach (var candidate in bones)
        {
            if (GodotObject.IsInstanceValid(candidate) && !candidate.IsTargetWithinJointLimits())
            {
                jointLimitViolations++;
                _jointLimitOffenders.Add(candidate.BoneName);
            }
        }

        Recovery.RecoveryCriteria criteria = getUpPhases.LastCriteria;
        sb.Append($"{groundAvgY.ToString("F3", _inv)},");
        sb.Append($"{Format(criteria.ChestClearance)},{Format(criteria.PelvisClearance)},{Format(criteria.LeadFootComDistance)},");
        sb.Append($"{(criteria.HandsPlanted ? 1 : 0)},{(criteria.LeadFootPlanted ? 1 : 0)},{(criteria.TrailFootPlanted ? 1 : 0)},{leadSide},");
        sb.Append($"{getUpPhases.PhaseElapsed.ToString("F3", _inv)},{(getUpPhases.LastAdvanceWasTimeout ? 1 : 0)},{jointLimitViolations}");

        // 5c. Energy bookkeeping - see the header comment for why torque ceilings do not bound this.
        // Rotational energy uses CapturedInertia (a scalar about the bone's stiffest axis) rather
        // than the full tensor, so it is an estimate; it is accurate enough to separate a joule from
        // a hundred joules, which is the question being asked.
        float systemKe = 0.0f;
        float systemPe = 0.0f;
        float actuatorPowerIn = 0.0f;
        float actuatorPowerOut = 0.0f;
        float peakJointPowerIn = 0.0f;
        string peakJointPowerBone = "-";
        foreach (var energyBone in bones)
        {
            if (!GodotObject.IsInstanceValid(energyBone))
            {
                continue;
            }

            systemKe += (0.5f * energyBone.Mass * energyBone.LinearVelocity.LengthSquared())
                        + (0.5f * energyBone.CapturedInertia * energyBone.AngularVelocity.LengthSquared());
            systemPe += energyBone.Mass * GravityForEnergy * energyBone.GlobalPosition.Y;

            // Power is measured against the joint's RELATIVE angular velocity, because that is the
            // rate the actuator's own degree of freedom is moving. Using absolute angular velocity
            // would credit a whole limb carried along by its parent as work this joint performed.
            Vector3 relativeAngVel = energyBone.ParentBone != null
                                     && GodotObject.IsInstanceValid(energyBone.ParentBone)
                ? energyBone.AngularVelocity - energyBone.ParentBone.AngularVelocity
                : energyBone.AngularVelocity;

            // Signed: positive means the actuator is driving the joint and adding energy, negative
            // means it is braking and removing it. Only the first can explain energy appearing.
            float jointPower = energyBone.LastAppliedTorque.Dot(relativeAngVel);
            if (jointPower > 0.0f)
            {
                actuatorPowerIn += jointPower;
                if (jointPower > peakJointPowerIn)
                {
                    peakJointPowerIn = jointPower;
                    peakJointPowerBone = energyBone.BoneName;
                }
            }
            else
            {
                actuatorPowerOut -= jointPower;
            }
        }

        sb.Append($",{systemKe.ToString("F1", _inv)},{systemPe.ToString("F1", _inv)},{(systemKe + systemPe).ToString("F1", _inv)}");
        sb.Append($",{actuatorPowerIn.ToString("F1", _inv)},{actuatorPowerOut.ToString("F1", _inv)}");
        sb.Append($",{peakJointPowerIn.ToString("F1", _inv)},{peakJointPowerBone}");
        sb.Append($",{_ballCount},{_ballSpeed.ToString("F2", _inv)},{_ballImpactBone}");
        sb.Append($",{_ballImpactImpulse.ToString("F4", _inv)},{_ballMassRatio.ToString("F1", _inv)}");
        sb.Append($",{_ballMassActual.ToString("F4", _inv)}");

        // 6. Per-bone columns
        foreach (string name in BoneNames)
        {
            if (boneMap.TryGetValue(name, out var b))
            {
                AccumulateBoneDiagnostics(name, b);

                Vector3 euler = b.GlobalTransform.Basis.GetEuler();
                Vector3 torque = b.LastAppliedTorque;
                Vector3 angVel = b.AngularVelocity;

                sb.Append($",{b.GlobalPosition.Y.ToString("F3", _inv)}");
                sb.Append($",{b.MuscleStrength.ToString("F2", _inv)}");
                sb.Append($",{torque.Length().ToString("F2", _inv)}");
                sb.Append($",{torque.X.ToString("F2", _inv)},{torque.Y.ToString("F2", _inv)},{torque.Z.ToString("F2", _inv)}");
                sb.Append($",{Mathf.RadToDeg(euler.X).ToString("F1", _inv)},{Mathf.RadToDeg(euler.Y).ToString("F1", _inv)},{Mathf.RadToDeg(euler.Z).ToString("F1", _inv)}");
                sb.Append($",{angVel.X.ToString("F2", _inv)},{angVel.Y.ToString("F2", _inv)},{angVel.Z.ToString("F2", _inv)}");
                sb.Append($",{b.LastTrackingErrorDeg.ToString("F2", _inv)}");
                sb.Append($",{b.LastPdTorque.Length().ToString("F2", _inv)},{b.LastLoadCompensationTorque.Length().ToString("F2", _inv)}");

                Vector3 boneRelativeAngVel = b.ParentBone != null && GodotObject.IsInstanceValid(b.ParentBone)
                    ? angVel - b.ParentBone.AngularVelocity
                    : angVel;
                sb.Append($",{b.LinearVelocity.Length().ToString("F2", _inv)}");
                sb.Append($",{torque.Dot(boneRelativeAngVel).ToString("F1", _inv)}");
                sb.Append($",{b.LastForceVelocityScale.ToString("F3", _inv)}");
            }
            else
            {
                sb.Append(EmptyBoneColumns);
            }
        }

        _deltaSum += delta;
        string row = sb.ToString();
        ValidateRowWidth(row);
        _csvRows.Add(row);

        if (ElapsedRecordingTime >= MaxDuration)
        {
            StopAndSave();
        }
    }

    private void AccumulateBoneDiagnostics(string name, ActiveBone bone)
    {
        if (!_boneDiagnostics.TryGetValue(name, out var d))
        {
            d = new BoneDiagnostics();
            _boneDiagnostics[name] = d;
        }

        // Gains are re-read every frame: the inertia is only captured on the bone's first
        // integration step, so the first recorded frames may still hold the authored fallback.
        d.Mass = bone.Mass;
        // The SPD denominator is sized per-axis each tick, so the manifest reports the mean of what
        // was actually used alongside the minor-axis floor it can never fall below.
        d.InertiaSum += bone.LastEffectiveInertia > 0.0f ? bone.LastEffectiveInertia : bone.CapturedInertia;
        d.InertiaMin = bone.CapturedInertia;
        d.ProportionalGain = bone.ProportionalGain;
        d.DerivativeGain = bone.ResolvedDerivativeGain;
        d.MaxTorque = bone.MaxTorque;
        d.AutoTuned = bone.AutoTuneDamping;

        Vector3 angVel = bone.AngularVelocity;
        float speed = angVel.Length();
        float torque = bone.LastAppliedTorque.Length();
        float error = bone.LastTrackingErrorDeg;

        // A discrete PD limit cycle reverses the angular velocity every physics tick. Requiring
        // both samples to clear a floor keeps low-amplitude noise around zero from registering.
        if (d.Samples > 0
            && speed > ChatterAngVelFloor
            && d.PreviousAngVel.Length() > ChatterAngVelFloor
            && d.PreviousAngVel.Dot(angVel) < 0.0f)
        {
            d.Reversals++;
        }

        d.PreviousAngVel = angVel;
        d.Samples++;
        d.AngVelSum += speed;
        d.AngVelMax = Mathf.Max(d.AngVelMax, speed);
        d.ErrorSum += error;
        d.ErrorMax = Mathf.Max(d.ErrorMax, error);
        d.TorqueSum += torque;
        d.TorqueMax = Mathf.Max(d.TorqueMax, torque);
        d.TorqueVectorSum += bone.LastAppliedTorque;
    }

    /// <summary>
    /// Fails loudly if a data row does not have the same number of columns as the header.
    ///
    /// Checked once per recording, on the first data row, because the failure it guards against is
    /// structural rather than intermittent: a header and a row built by separate code paths that
    /// have drifted. Every column after the mismatch would be attributed to the wrong name, and
    /// nothing about the resulting file looks wrong - it parses, the values are plausible, and the
    /// conclusions drawn from it are simply about different quantities than they claim.
    /// </summary>
    private void ValidateRowWidth(string row)
    {
        if (_csvRows.Count != 1)
        {
            return;
        }

        int headerColumns = CountColumns(_csvRows[0]);
        int rowColumns = CountColumns(row);
        if (headerColumns != rowColumns)
        {
            GD.PushError(
                $"[RagdollTelemetryRecorder] Header has {headerColumns} columns but rows have "
                + $"{rowColumns}. The dump is misaligned and must not be trusted - a column was "
                + "added to one of RecordFrame's two build paths and not the other.");
        }
    }

    private static int CountColumns(string line)
    {
        int columns = 1;
        foreach (char character in line)
        {
            if (character == ',') { columns++; }
        }
        return columns;
    }

    /// <summary>
    /// Builds the companion actuator manifest: the per-bone impedance the SPD law actually ran
    /// with, alongside the measured oscillation.
    ///
    /// VelocityFactor is the per-tick angular-velocity multiplier of the damping term,
    /// 1 - (Kd·dt/I) / (1 + Kd·dt/I + Kp·dt²/I). Tan-Liu-Turk SPD keeps this in (0,1] as long as
    /// the denominator sees the body's true inertia; a value at or below 0 means the joint
    /// reverses its own angular velocity every tick, and below -1 it amplifies it. That single
    /// column is what makes this class of bug visible instead of merely audible.
    /// </summary>
    private List<string> BuildActuatorManifest()
    {
        float dt = _csvRows.Count > 1 && _deltaSum > 0.0f
            ? _deltaSum / (_csvRows.Count - 1)
            : 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);

        var rows = new List<string>
        {
            "Bone,Mass,InertiaUsed,InertiaMin,Kp,Kd,AutoTuned,MaxTorque,KpEff,KdEff,VelocityFactor,Stability,ZetaAchieved,Damping,"
            + "AngVelAvg,AngVelMax,ReversalsPerSec,Chatter,TrackErrAvgDeg,TrackErrMaxDeg,TorqueAvg,TorqueMax,TorqueCoherence"
        };

        foreach (string name in BoneNames)
        {
            if (!_boneDiagnostics.TryGetValue(name, out var d) || d.Samples == 0)
            {
                continue;
            }

            float inertia = Mathf.Max(1e-9f, d.InertiaSum / d.Samples);
            float a = d.DerivativeGain * dt / inertia;
            float b = d.ProportionalGain * dt * dt / inertia;
            float denominator = 1.0f + a + b;
            float velocityFactor = 1.0f - (a / denominator);

            string stability = velocityFactor > 0.0f ? "STABLE"
                : velocityFactor > -1.0f ? "MARGINAL"
                : "UNSTABLE";

            float zetaScale = d.ProportionalGain * inertia * denominator;
            float zeta = zetaScale > 0.0f ? d.DerivativeGain / (2.0f * Mathf.Sqrt(zetaScale)) : 0.0f;
            string damping = zeta <= 0.0f ? "NONE"
                : zeta < UnderdampedRatio ? "UNDERDAMPED"
                : zeta > 2.0f ? "OVERDAMPED"
                : "OK";

            float duration = d.Samples * dt;
            float reversalsPerSec = duration > 0.0f ? d.Reversals / duration : 0.0f;
            float angVelAvg = d.AngVelSum / d.Samples;
            bool chattering = reversalsPerSec > ChatterReversalRate && angVelAvg > 1.0f;

            rows.Add(string.Join(",",
                name,
                d.Mass.ToString("F3", _inv),
                inertia.ToString("G4", _inv),
                d.InertiaMin.ToString("G4", _inv),
                d.ProportionalGain.ToString("F1", _inv),
                d.DerivativeGain.ToString("F3", _inv),
                d.AutoTuned ? "1" : "0",
                d.MaxTorque.ToString("F1", _inv),
                (d.ProportionalGain / denominator).ToString("F2", _inv),
                (d.DerivativeGain / denominator).ToString("F3", _inv),
                velocityFactor.ToString("F3", _inv),
                stability,
                zeta.ToString("F2", _inv),
                damping,
                angVelAvg.ToString("F3", _inv),
                d.AngVelMax.ToString("F2", _inv),
                reversalsPerSec.ToString("F1", _inv),
                chattering ? "CHATTER" : "-",
                (d.ErrorSum / d.Samples).ToString("F2", _inv),
                d.ErrorMax.ToString("F2", _inv),
                (d.TorqueSum / d.Samples).ToString("F2", _inv),
                d.TorqueMax.ToString("F2", _inv),
                // |mean torque vector| / mean |torque|: 1.0 is a steady directed push, ~0 means the
                // actuator spent its whole output reversing direction and achieved nothing.
                (d.TorqueSum > 0.0f ? d.TorqueVectorSum.Length() / d.TorqueSum : 0.0f).ToString("F3", _inv)));
        }

        return rows;
    }

    /// <summary>
    /// Formats a criterion value that may legitimately be non-finite: chest clearance is NaN when
    /// the chest is unavailable, and CoM-to-support distance is infinite when a foot is missing.
    /// Writing raw NaN/Inf into the CSV would break downstream numeric parsing.
    /// </summary>
    private string Format(float value)
    {
        return float.IsFinite(value) ? value.ToString("F3", _inv) : string.Empty;
    }

    /// <summary>
    /// Geometric flexion of a three-segment joint in degrees (0 = fully straight), measured at the
    /// middle body's origin between the vectors to its neighbours. No hardcoded segment lengths, so
    /// the same measurement serves the knee (thigh/shin/foot) and the elbow (upper arm/forearm/hand).
    /// </summary>
    /// <summary>Elbow flexion from the bone map, or 0 when any of the three segments is missing.</summary>
    private static float ComputeElbowFlexionDeg(
        Dictionary<string, ActiveBone> boneMap, string upperArm, string forearm, string hand)
    {
        if (!boneMap.TryGetValue(upperArm, out var upper)
            || !boneMap.TryGetValue(forearm, out var fore)
            || !boneMap.TryGetValue(hand, out var palm))
        {
            return 0.0f;
        }

        return ComputeJointFlexionDeg(upper.GlobalPosition, fore.GlobalPosition, palm.GlobalPosition);
    }

    private static float ComputeJointFlexionDeg(Vector3 proximalPos, Vector3 jointPos, Vector3 distalPos)
    {
        Vector3 toProximal = proximalPos - jointPos;
        Vector3 toDistal = distalPos - jointPos;
        if (toProximal.LengthSquared() < 1e-8f || toDistal.LengthSquared() < 1e-8f)
        {
            return 0.0f;
        }

        float cosAngle = Mathf.Clamp(toProximal.Normalized().Dot(toDistal.Normalized()), -1.0f, 1.0f);
        return 180.0f - Mathf.RadToDeg(Mathf.Acos(cosAngle));
    }

    public void StopAndSave(string relativePath = "Debug/telemetry_dump.csv")
    {
        IsRecording = false;
        try
        {
            string globalPath = ProjectSettings.GlobalizePath($"res://{relativePath}");
            string? baseDir = Path.GetDirectoryName(globalPath);
            if (!string.IsNullOrEmpty(baseDir) && !Directory.Exists(baseDir))
            {
                Directory.CreateDirectory(baseDir);
            }

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string baseName = Path.GetFileNameWithoutExtension(globalPath);
            string dumpName = $"{baseName}_{timestamp}";
            
            // Create a subfolder for this specific dump
            string dumpDir = Path.Combine(baseDir ?? "", dumpName);
            Directory.CreateDirectory(dumpDir);

            // Save the csv inside the new subfolder
            string newFileName = $"{dumpName}.csv";
            string finalPath = Path.Combine(dumpDir, newFileName);

            File.WriteAllLines(finalPath, _csvRows);
            GD.Print($"[RagdollTelemetryRecorder] SUCCESSFULLY saved {_csvRows.Count - 1} frames to: {finalPath}");

            List<string> manifest = BuildActuatorManifest();
            string manifestPath = Path.Combine(dumpDir, "actuators.csv");
            File.WriteAllLines(manifestPath, manifest);
            GD.Print($"[RagdollTelemetryRecorder] Actuator manifest ({manifest.Count - 1} bones) saved to: {manifestPath}");

            foreach (string row in manifest)
            {
                if (row.Contains("UNSTABLE") || row.Contains("MARGINAL") || row.Contains("CHATTER") || row.Contains("UNDERDAMPED"))
                {
                    GD.PushWarning($"[RagdollTelemetryRecorder] Actuator instability: {row}");
                }
            }

            if (_jointLimitOffenders.Count > 0)
            {
                GD.PushWarning(
                    "[RagdollTelemetryRecorder] Targets commanded outside joint limits (actuator pinned against a stop, "
                    + $"no motion produced): {string.Join(", ", _jointLimitOffenders)}");
            }

            // Keep only the last 3 dump subfolders
            if (!string.IsNullOrEmpty(baseDir))
            {
                var dirs = new DirectoryInfo(baseDir).GetDirectories($"{baseName}_*");
                if (dirs.Length > 3)
                {
                    Array.Sort(dirs, (a, b) => a.CreationTime.CompareTo(b.CreationTime));
                    for (int i = 0; i < dirs.Length - 3; i++)
                    {
                        dirs[i].Delete(true);
                        GD.Print($"[RagdollTelemetryRecorder] Deleted old telemetry folder: {dirs[i].Name}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            GD.PushError($"[RagdollTelemetryRecorder] Failed to save telemetry CSV: {ex.Message}");
        }
        finally
        {
            RecordingFinished?.Invoke();
        }
    }
}

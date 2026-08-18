using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
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

    private readonly List<string> _csvRows = new();
    private readonly CultureInfo _inv = CultureInfo.InvariantCulture;

    public void StartRecording(float duration = DefaultDurationSeconds)
    {
        MaxDuration = duration;
        ElapsedRecordingTime = 0.0f;
        IsRecording = true;
        _csvRows.Clear();

        // Build CSV Header
        var sb = new StringBuilder();
        sb.Append("Time,Delta,State,Orientation,RecoveryProgress,");
        sb.Append("PelvisPosX,PelvisPosY,PelvisPosZ,PelvisVelX,PelvisVelY,PelvisVelZ,PelvisSpeed,PelvisTiltDeg,PelvisAngVelX,PelvisAngVelY,PelvisAngVelZ,");
        sb.Append("CoMPosX,CoMPosY,CoMPosZ,CoMVelX,CoMVelY,CoMVelZ,");
        sb.Append("IcpX,IcpZ,SupportCenterX,SupportCenterZ,LocalIcpX,LocalIcpZ,");
        sb.Append("KneeAngleL_Deg,KneeAngleR_Deg,LegLengthL,LegLengthR,");
        sb.Append("GroundPointL_Y,GroundPointR_Y,FootElevationL,FootElevationR,");
        sb.Append("IsGroundedL,IsGroundedR,WeightShareL,WeightShareR,BalanceStrength,StateTimer,IcpEscape,SwingProgress,PelvisStabTorque,StepPhase");

        // Append 14 Bone columns with tracking error
        string[] boneNames = {
            "Pelvis", "Spine", "Chest", "Head",
            "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
            "Thigh_L", "Shin_L", "Foot_L", "Thigh_R", "Shin_R", "Foot_R"
        };

        foreach (string name in boneNames)
        {
            sb.Append($",{name}_PosY,{name}_Strength,{name}_TorqueMag,{name}_TorqueX,{name}_TorqueY,{name}_TorqueZ,{name}_EulerX,{name}_EulerY,{name}_EulerZ,{name}_AngVelX,{name}_AngVelY,{name}_AngVelZ,{name}_TrackingErrorDeg");
        }

        _csvRows.Add(sb.ToString());
        GD.Print($"[RagdollTelemetryRecorder] Started {duration:F1}s comprehensive telemetry recording session...");
    }

    public void RecordFrame(
        float delta,
        RagdollState state,
        BalanceController balance,
        ActiveBone pelvis,
        IReadOnlyList<ActiveBone> bones)
    {
        if (!IsRecording) return;

        ElapsedRecordingTime += delta;

        var sb = new StringBuilder();
        sb.Append($"{ElapsedRecordingTime.ToString("F4", _inv)},{delta.ToString("F4", _inv)},{state},{balance.CurrentOrientation},{balance.RecoveryProgressNormalized.ToString("F3", _inv)},");
        
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

        // 3. Knee Flexion & Leg Extension Geometry
        float kneeAngleL = 0.0f, kneeAngleR = 0.0f;
        float legLengthL = 0.0f, legLengthR = 0.0f;

        if (balance.ThighL != null && balance.ShinL != null && balance.FootL != null
            && GodotObject.IsInstanceValid(balance.ThighL) && GodotObject.IsInstanceValid(balance.ShinL) && GodotObject.IsInstanceValid(balance.FootL))
        {
            legLengthL = (balance.ThighL.GlobalPosition - balance.FootL.GlobalPosition).Length();
            kneeAngleL = ComputeKneeFlexionDeg(balance.ThighL.GlobalPosition, balance.ShinL.GlobalPosition, balance.FootL.GlobalPosition);
        }

        if (balance.ThighR != null && balance.ShinR != null && balance.FootR != null
            && GodotObject.IsInstanceValid(balance.ThighR) && GodotObject.IsInstanceValid(balance.ShinR) && GodotObject.IsInstanceValid(balance.FootR))
        {
            legLengthR = (balance.ThighR.GlobalPosition - balance.FootR.GlobalPosition).Length();
            kneeAngleR = ComputeKneeFlexionDeg(balance.ThighR.GlobalPosition, balance.ShinR.GlobalPosition, balance.FootR.GlobalPosition);
        }

        sb.Append($"{kneeAngleL.ToString("F1", _inv)},{kneeAngleR.ToString("F1", _inv)},{legLengthL.ToString("F3", _inv)},{legLengthR.ToString("F3", _inv)},");

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
        sb.Append($"{balance.PelvisStabilizerTorque.Length().ToString("F1", _inv)},{balance.CurrentStepPhase}");

        // 6. Build Bone Dictionary for indexed lookup
        var boneMap = new Dictionary<string, ActiveBone>();
        foreach (var b in bones)
        {
            if (GodotObject.IsInstanceValid(b))
            {
                boneMap[b.BoneName] = b;
            }
        }

        string[] boneNames = {
            "Pelvis", "Spine", "Chest", "Head",
            "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
            "Thigh_L", "Shin_L", "Foot_L", "Thigh_R", "Shin_R", "Foot_R"
        };

        foreach (string name in boneNames)
        {
            if (boneMap.TryGetValue(name, out var b))
            {
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
            }
            else
            {
                sb.Append(",0,0,0,0,0,0,0,0,0,0,0,0,0");
            }
        }

        _csvRows.Add(sb.ToString());

        if (ElapsedRecordingTime >= MaxDuration)
        {
            StopAndSave();
        }
    }

    /// <summary>
    /// Geometric knee flexion in degrees (0 = fully straight), measured at the shin joint
    /// origin between the vectors to the thigh and foot origins. No hardcoded segment lengths.
    /// </summary>
    private static float ComputeKneeFlexionDeg(Vector3 thighPos, Vector3 shinPos, Vector3 footPos)
    {
        Vector3 toThigh = thighPos - shinPos;
        Vector3 toFoot = footPos - shinPos;
        if (toThigh.LengthSquared() < 1e-8f || toFoot.LengthSquared() < 1e-8f)
        {
            return 0.0f;
        }

        float cosAngle = Mathf.Clamp(toThigh.Normalized().Dot(toFoot.Normalized()), -1.0f, 1.0f);
        return 180.0f - Mathf.RadToDeg(Mathf.Acos(cosAngle));
    }

    public void StopAndSave(string relativePath = "Debug/telemetry_dump.csv")
    {
        IsRecording = false;
        try
        {
            string globalPath = ProjectSettings.GlobalizePath($"res://{relativePath}");
            string? dir = Path.GetDirectoryName(globalPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            File.WriteAllLines(globalPath, _csvRows);
            GD.Print($"[RagdollTelemetryRecorder] SUCCESSFULLY saved {_csvRows.Count - 1} frames to: {globalPath}");
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

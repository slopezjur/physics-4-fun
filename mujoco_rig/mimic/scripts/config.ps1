# Mimic experiments use their isolated .venv, never the legacy MuJoCo training scripts.
# Paths below are relative to the project root unless absolute.
$Mimic = @{
    Python = 'mujoco_rig/mimic/.venv/Scripts/python.exe'
    MimicKit = 'tools/MimicKit'
    Godot = 'tools/Godot/Godot.exe'
    LogRoot = 'logs/mimickit-training'
    DefaultExperiment = 'perturb'
    Minutes = 30
    Envs = 2048
    Iterations = 1000000
    Seed = 210921
    # Native validation every N completed PPO updates; the suite depends on the trainer.
    # Initial and final validation always run (or reuse identical cached results).
    EvaluationInterval = 64
    ActorMaxKl = 0.03
    # Defaults preserve the current guarded target-PD fine-tuning path. Override them per run
    # with train.ps1 -Control/-Stability without changing this file.
    Control = 'target_pd'
    # Legacy checkpoints require legacy semantics. Use support_v2 with a matching
    # checkpoint, or start a fresh Stand experiment with -Stability upstream.
    ContactMode = 'legacy'
    Stability = 'guarded'
    # A warm start loads weights + normalization, not optimizer/RNG state.
    InitializeFrom = 'logs/mimickit-perturb/guarded-perturb-04/best.pt'
    SourceContract = 'logs/mimickit-perturb/guarded-perturb-04/export/contract.json'
    WatchBundle = 'logs/mimickit-perturb/guarded-perturb-04/export'
    # Stationary recovery uses train-reactive.ps1 and the same Envs/EvaluationInterval.
    ReactiveTraining = @{
        Minutes = 5
        Seed = 270933
        InitializeFrom = 'logs/mimickit-training/reactive-ball-training-2kg-01/best.pt'
        Reference = 'logs/mimickit-steps/continuous-stand-02/stationary_reference.npz'
        BallModel = 'logs/mimickit-training/reactive-ball-seed-01/ball_5.xml'
        RecoveryMode = 'capture_v3'
        RecoveryResetBank = 'logs/mimickit-training/recovery-post-hit-resets-01/recovery_resets.npz'
        RecoveryResetFraction = 0.5
        ForwardFraction = 0.5
        QuietFraction = 0.25
        SecondShotFraction = 0.5
        EpisodeSeconds = 12
        SpeedMin = 1.0
        SpeedMax = 1.5
        QuietReplayWeight = 0.1
        ActorLearningRate = 1e-5
        ActorWarmup = 4
        StepsPerIter = 64
        TdLambda = 0.98
        CheckpointInterval = 8
    }
    ReactiveStepTraining = @{
        Minutes = 20
        Seed = 280902
        InitializeFrom = 'logs/mimickit-training/reactive-ball-capture-resets-5min-01/best.pt'
        Reference = 'logs/mimickit-steps/continuous-stand-02/stationary_reference.npz'
        BallModel = 'logs/mimickit-training/reactive-ball-seed-01/ball_5.xml'
        RecoveryMode = 'step_v4'
        RecoveryResetBank = 'logs/mimickit-training/step-native-exploration-2mps-01/recovery_resets.npz'
        RecoveryResetFraction = 0.5
        ForwardFraction = 0.0
        QuietFraction = 0.25
        SecondShotFraction = 0.5
        EpisodeSeconds = 12
        SpeedMin = 1.0
        SpeedMax = 2.0
        QuietReplayWeight = 1.0
        QuietAnchorSamples = 512
        ActorLearningRate = 2.5e-5
        ActorWarmup = 8
        StepsPerIter = 64
        TdLambda = 0.98
        CheckpointInterval = 8
    }
    Experiments = @{
        stand = @{ Task = 'stand' }
        perturb = @{
            Task = 'ball'
            BallSpeedMin = 1.0     # Horizontal m/s. Vertical speed compensates gravity at launch only.
            BallSpeedMax = 2.5
            QuietFraction = 0.25 # Keep undisturbed standing episodes during training.
            BallReward = 'reference' # Opt in per run with -BallReward recovery_v1 until validated.
        }
        perturb_front_back = @{
            Task = 'ball'
            Minutes = 15
            BallSpeedMin = 1.0
            BallSpeedMax = 2.0
            QuietFraction = 0.25
            BallReward = 'recovery_v1'
            BallCurriculum = 'front_back_torso_v1'
            # Review the actual trained candidate; best.pt remains separately retained.
            ViewerPolicy = 'final'
            InitializeFrom = 'logs/mimickit-training/perturb-resume-15min-01/best.pt'
            SourceContract = 'logs/mimickit-training/perturb-resume-15min-01/export/contract.json'
        }
        perturb_mixed = @{
            Task = 'ball'
            Minutes = 30
            BallSpeedMin = 1.0
            BallSpeedMax = 2.0
            QuietFraction = 0.25
            BallReward = 'recovery_v1'
            BallCurriculum = 'mixed_recovery_v1' # 50% focused, 25% broad impacts, 25% quiet.
            ViewerPolicy = 'final'
            # Continue the reviewed final candidate, not its unchanged retained best.pt.
            InitializeFrom = 'logs/mimickit-training/perturb-front-back-15min-01/model.pt'
            SourceContract = 'logs/mimickit-training/perturb-front-back-15min-01/final-export/contract.json'
        }
    }
}
# Ball mass/radius/friction/contact stiffness come from generated dummy_ball.xml.
# Do not edit generated physics to change difficulty: vary launch speed first.

# Keep machine-specific installations out of version control. Environment overrides win.
$Mimic.ReactiveStepRiskTraining = $Mimic.ReactiveStepTraining.Clone()
$Mimic.ReactiveStepRiskTraining.Minutes = 10
$Mimic.ReactiveStepRiskTraining.Seed = 280903
$Mimic.ReactiveStepRiskTraining.InitializeFrom = 'logs/mimickit-training/reactive-step-v4-2mps-20min-01/best.pt'
$Mimic.ReactiveStepRiskTraining.RiskLegStd = 0.2
$Mimic.ReactiveStepRiskTraining.ActorMaxKl = 0.01
$Mimic.ReactiveStepRiskTraining.ActorLearningRate = 1e-4
$Mimic.ReactiveStepRiskTraining.ActorWarmup = 4

$Mimic.ReactiveGatedTraining = $Mimic.ReactiveStepRiskTraining.Clone()
$Mimic.ReactiveGatedTraining.Minutes = 20
$Mimic.ReactiveGatedTraining.Seed = 280906
$Mimic.ReactiveGatedTraining.InitializeFrom = 'logs/mimickit-training/step-gated-seed-01/best.pt'
$Mimic.ReactiveGatedTraining.ActorLearningRate = 1e-3
$Mimic.ReactiveGatedTraining.ActorMaxKl = 0.02

$Mimic.ReactiveCaptureTraining = $Mimic.ReactiveGatedTraining.Clone()
$Mimic.ReactiveCaptureTraining.Minutes = 5
$Mimic.ReactiveCaptureTraining.Seed = 280910
$Mimic.ReactiveCaptureTraining.InitializeFrom = 'logs/mimickit-training/step-capture-seed-01/best.pt'

$Mimic.ReactiveTransferTraining = $Mimic.ReactiveCaptureTraining.Clone()
$Mimic.ReactiveTransferTraining.Minutes = 10
$Mimic.ReactiveTransferTraining.RecoveryMode = 'step_v5'

# Tests the support-aware actor's larger internal correction branch at unchanged motor limits.
$Mimic.ReactiveSupportTraining = $Mimic.ReactiveTransferTraining.Clone()
$Mimic.ReactiveSupportTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-support-seed-01/best.pt'
$Mimic.ReactiveSupportTraining.PreferRecoveredSteps = $true

# Controlled optimizer-penalty ablation; identical seed, reward, shots and physical clipping.
$Mimic.ReactiveSupportNoBoundTraining = $Mimic.ReactiveSupportTraining.Clone()
$Mimic.ReactiveSupportNoBoundTraining.ActionBoundWeight = 0.0

# Rehearse successful source reactions, including its measured step, during PPO.
$Mimic.ReactiveSupportReplayTraining = $Mimic.ReactiveSupportNoBoundTraining.Clone()
$Mimic.ReactiveSupportReplayTraining.ImpactReplayTrace = 'logs/mimickit-training/recovery-loop-support-seed-trace-01/trace.npz'
$Mimic.ReactiveSupportReplayTraining.ImpactReplayWeight = 1.0
$Mimic.ReactiveSupportReplayTraining.ImpactReplaySamples = 512

# Test the observed delayed-fall return gap without changing reward or curriculum.
$Mimic.ReactiveSupportLongHorizonTraining = $Mimic.ReactiveSupportReplayTraining.Clone()
$Mimic.ReactiveSupportLongHorizonTraining.Discount = 0.997

# Isolate exploration activation from the separate discount experiment.
$Mimic.ReactiveSupportExplorationTraining = $Mimic.ReactiveSupportReplayTraining.Clone()
$Mimic.ReactiveSupportExplorationTraining.RiskExplorationGate = 'support'

# Experimental heavy-impact curriculum; admission remains separate from training eligibility.
$Mimic.ReactiveHeavyCurriculumTraining = $Mimic.ReactiveTransferTraining.Clone()
$Mimic.ReactiveHeavyCurriculumTraining.InitializeFromQuietReport = 'logs/mimickit-training/step-capture-seed-quiet600-01/report.json'
$Mimic.ReactiveHeavyCurriculumTraining.BallModel = 'logs/mimickit-training/recovery-loop-support-curriculum-01/mass-8-radius-0.09.xml'
$Mimic.ReactiveHeavyCurriculumTraining.SpeedMin = 2.0
$Mimic.ReactiveHeavyCurriculumTraining.SpeedMax = 2.0
$Mimic.ReactiveHeavyCurriculumTraining.RecoveryResetBank = $null
$Mimic.ReactiveHeavyCurriculumTraining.RecoveryResetFraction = 0.0
$Mimic.ReactiveHeavyCurriculumTraining.ActionBoundWeight = 0.0
$Mimic.ReactiveHeavyCurriculumTraining.PreferRecoveredSteps = $true

# Continue the heavy actor only after separate long-quiet and lighter-hit retention checks.
$Mimic.ReactiveHeavyContinueTraining = $Mimic.ReactiveHeavyCurriculumTraining.Clone()
$Mimic.ReactiveHeavyContinueTraining.Remove('InitializeFromQuietReport')
$Mimic.ReactiveHeavyContinueTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-heavy-ppo-10min-01/best.pt'
$Mimic.ReactiveHeavyContinueTraining.Minutes = 20

# Address measured heavy head-impact failures with the existing capture actor.
$Mimic.ReactiveHeavyHeadTraining = $Mimic.ReactiveHeavyContinueTraining.Clone()
$Mimic.ReactiveHeavyHeadTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-heavy-continue-40min-03/best.pt'
$Mimic.ReactiveHeavyHeadTraining.TargetSet = 'torso_head'

# Cover milder impulses after fixed-speed training regressed two 5 kg diagnostic cases.
$Mimic.ReactiveHeavyRangeTraining = $Mimic.ReactiveHeavyContinueTraining.Clone()
$Mimic.ReactiveHeavyRangeTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-heavy-continue-40min-03/best.pt'
$Mimic.ReactiveHeavyRangeTraining.SpeedMin = 1.25
$Mimic.ReactiveHeavyRangeTraining.SpeedMax = 2.0

# Isolate late settling with a frozen capture policy and a trainable damping layer.
$Mimic.ReactiveDampingSettlingTraining = $Mimic.ReactiveHeavyContinueTraining.Clone()
$Mimic.ReactiveDampingSettlingTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-damping-heavy40-seed-02/best.pt'
$Mimic.ReactiveDampingSettlingTraining.InitializeFromQuietReport = 'logs/mimickit-training/recovery-loop-damping-heavy40-quiet600-02/report.json'
$Mimic.ReactiveDampingSettlingTraining.BallModel = 'logs/mimickit-training/recovery-loop-heavy-continue-curriculum-02/mass-6-radius-0.09.xml'
$Mimic.ReactiveDampingSettlingTraining.EpisodeSeconds = 30
$Mimic.ReactiveDampingSettlingTraining.SecondShotFraction = 0.0
$Mimic.ReactiveDampingSettlingTraining.TargetSet = 'all'
# At 64 control steps/update, allow fresh impacts after a complete 30 s episode.
$Mimic.ReactiveDampingSettlingTraining.ActorStallIterations = 32

# One-hit-per-30-second, all-body follow-up from the retained heavy recovery actor.
# This keeps the impact schedule sparse enough to measure recovery and preserves
# quiet-standing retention while exposing every supported target body/angle.
$Mimic.ReactiveAllTarget30sTraining = $Mimic.ReactiveHeavyContinueTraining.Clone()
$Mimic.ReactiveAllTarget30sTraining.Minutes = 15
$Mimic.ReactiveAllTarget30sTraining.Seed = 281701
$Mimic.ReactiveAllTarget30sTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-heavy-continue-40min-03/best.pt'
$Mimic.ReactiveAllTarget30sTraining.BallModel = 'logs/mimickit-training/recovery-loop-support-curriculum-01/mass-8-radius-0.09.xml'
$Mimic.ReactiveAllTarget30sTraining.EpisodeSeconds = 30
$Mimic.ReactiveAllTarget30sTraining.SecondShotFraction = 0.0
$Mimic.ReactiveAllTarget30sTraining.TargetSet = 'all'
$Mimic.ReactiveAllTarget30sTraining.QuietFraction = 0.25
$Mimic.ReactiveAllTarget30sTraining.SpeedMin = 2.0
$Mimic.ReactiveAllTarget30sTraining.SpeedMax = 2.0

# Continue the explicit Arena candidate for a longer random-body perturb run.
# The source is experimental until the fresh mass/quiet gates pass.
$Mimic.ReactiveCandidateAllTarget30s30minTraining = $Mimic.ReactiveAllTarget30sTraining.Clone()
$Mimic.ReactiveCandidateAllTarget30s30minTraining.Minutes = 30
$Mimic.ReactiveCandidateAllTarget30s30minTraining.Seed = 281705
$Mimic.ReactiveCandidateAllTarget30s30minTraining.InitializeFrom = 'logs/mimickit-training/recovery-loop-all-target-30s-15min-02/model.pt'

# Isolated AMP style pilot; retain the same physical curriculum and state-based objective.
$Mimic.ReactiveAMPRecoveryPilot = $Mimic.ReactiveAllTarget30sTraining.Clone()
$Mimic.ReactiveAMPRecoveryPilot.Minutes = 5
$Mimic.ReactiveAMPRecoveryPilot.Seed = 292110
$Mimic.ReactiveAMPRecoveryPilot.InitializeFrom = 'logs/mimickit-training/recovery-loop-candidate-all-target-30min-01/best.pt'
$Mimic.ReactiveAMPRecoveryPilot.AMPDemonstrations = @(
    'mujoco_rig/mimic/assets/stand_reference.npz',
    'mujoco_rig/mimic/assets/steps/forward_reference.npz',
    'mujoco_rig/mimic/assets/steps/backward_reference.npz'
)

# Calibrated AMP follow-up from the explicitly previewed portable actor.
$Mimic.ReactiveAMPRecoverySmooth15min = $Mimic.ReactiveAMPRecoveryPilot.Clone()
$Mimic.ReactiveAMPRecoverySmooth15min.Minutes = 15
$Mimic.ReactiveAMPRecoverySmooth15min.Seed = 292140
$Mimic.ReactiveAMPRecoverySmooth15min.InitializeFrom = 'logs/mimickit-training/amp-recovery-prior-5min-01/model.pt'
$Mimic.ReactiveAMPRecoverySmooth15min.AMPDemonstrations = @(
    'mujoco_rig/mimic/assets/stand_reference.npz',
    'mujoco_rig/mimic/assets/steps/forward_reference.npz',
    'mujoco_rig/mimic/assets/steps/backward_reference.npz',
    'logs/mimickit-steps/weight-shift-reference-01/weight_shift_reference.npz',
    'logs/mimickit-steps/placed-step-forward-09',
    'logs/mimickit-steps/placed-step-backward-09'
)
$Mimic.ReactiveAMPRecoverySmooth15min.AMPDemoWeights = @(0.35, 0.15, 0.15, 0.15, 0.1, 0.1)
$Mimic.ReactiveAMPRecoverySmooth15min.AMPMirrorDemos = $true
$Mimic.ReactiveAMPRecoverySmooth15min.AMPFootMotionWeight = 0.04
$Mimic.ReactiveAMPRecoverySmooth15min.AMPDiscLearningRate = 0.001
$Mimic.ReactiveAMPRecoverySmooth15min.AMPCalibrationSteps = 256

# Test useful lift/landing and loaded-slip cost after the grounded-credit audit.
$Mimic.ReactiveAMPPlacement15min = $Mimic.ReactiveAMPRecoverySmooth15min.Clone()
$Mimic.ReactiveAMPPlacement15min.InitializeFrom = 'logs/mimickit-training/amp-recovery-smooth-15min-02/model.pt'
$Mimic.ReactiveAMPPlacement15min.Seed = 300930
$Mimic.ReactiveAMPPlacement15min.RecoveryMode = 'step_v6'
$Mimic.ReactiveAMPPlacement15min.SpeedMin = 2.0
$Mimic.ReactiveAMPPlacement15min.SpeedMax = 2.5

# User-requested 10 kg / 4 m/s continuation from the latest trained Perturb actor.
# Exact existing long-quiet evidence permits the new projectile curriculum.
$Mimic.ReactiveAMPHeavy30min = $Mimic.ReactiveAMPPlacement15min.Clone()
$Mimic.ReactiveAMPHeavy30min.Minutes = 30
$Mimic.ReactiveAMPHeavy30min.Seed = 300980
$Mimic.ReactiveAMPHeavy30min.InitializeFrom = 'logs/mimickit-training/amp-useful-placement-15min-01/model.pt'
$Mimic.ReactiveAMPHeavy30min.InitializeFromQuietReport = 'logs/mimickit-training/amp-useful-placement-final-quiet600-01/report.json'
$Mimic.ReactiveAMPHeavy30min.BallModel = 'logs/mimickit-training/amp-heavy-10kg-4mps-30min-01-plant/dummy_ball.xml'
$Mimic.ReactiveAMPHeavy30min.RecoveryMode = 'step_v7'
$Mimic.ReactiveAMPHeavy30min.SpeedMin = 4.0
$Mimic.ReactiveAMPHeavy30min.SpeedMax = 4.0
# 32 updates cover a complete 30-second episode before declaring updates stalled.
$Mimic.ReactiveAMPHeavy30min.ActorStallIterations = 32

# Continue the final heavy Perturb preview for the user's second 30-minute window.
$Mimic.ReactiveAMPHeavyContinue30min = $Mimic.ReactiveAMPHeavy30min.Clone()
$Mimic.ReactiveAMPHeavyContinue30min.Remove('InitializeFromQuietReport')
$Mimic.ReactiveAMPHeavyContinue30min.InitializeFrom = 'logs/mimickit-training/amp-heavy-10kg-4mps-30min-01/model.pt'
$Mimic.ReactiveAMPHeavyContinue30min.BallModel = 'logs/mimickit-training/amp-heavy-10kg-4mps-30min-02-continue-plant/dummy_ball.xml'
$Mimic.ReactiveAMPHeavyContinue30min.Seed = 300981
# Preserve the trained discriminator/normalization: startup recalibration worsened
# held-out classification (96% to 92%) on this unchanged physical task.
$Mimic.ReactiveAMPHeavyContinue30min.AMPCalibrationSteps = 0

# Isolated commanded-motion foundation; the stationary Arena actor remains frozen.
$Mimic.MotionTrackerFoundationTraining = @{
    Minutes = 5
    Seed = 300951
    EpisodeSeconds = 30
    RootTrackingSigmaM = 0.03
    RootTrackingWeight = 0.45
    RootRewardMode = 'inverse_huber'
    RootAccuracyScaleM = 0.005
    FixedFraction = 0.25
    InitializeFrom = 'logs/mimickit-training/amp-recovery-smooth-15min-02/model.pt'
    QuietReport = 'logs/mimickit-training/amp-recovery-smooth-final-quiet600-02/report.json'
    Stationary = 'logs/mimickit-steps/continuous-stand-02/stationary_reference.npz'
    Motions = @(
        'logs/mimickit-steps/weight-shift-small-01/weight_shift_reference.npz'
    )
}

# Fresh torque-policy experiment; source/preview physics are isolated from prior brains.
$Mimic.TorqueStandTraining = @{
    Minutes = 15
    Seed = 300990
    Reference = 'logs/mimickit-steps/continuous-stand-02/stationary_reference.npz'
}

# Continue the final spring-free actor and its complete AMP state for 45 additional minutes.
$Mimic.TorqueStandContinue45min = $Mimic.TorqueStandTraining.Clone()
$Mimic.TorqueStandContinue45min.Minutes = 45
$Mimic.TorqueStandContinue45min.Seed = 300991
$Mimic.TorqueStandContinue45min.InitializeFrom = 'logs/mimickit-training/stand-torque-spring-free-30min-01/model.pt'

# Additional 15-minute window from the completed 45-minute final actor and AMP sidecar.
$Mimic.TorqueStandContinue15min = $Mimic.TorqueStandTraining.Clone()
$Mimic.TorqueStandContinue15min.Minutes = 15
$Mimic.TorqueStandContinue15min.Seed = 300992
$Mimic.TorqueStandContinue15min.InitializeFrom = 'logs/mimickit-training/stand-torque-spring-free-45min-02/model.pt'

# Corrected whole-body objective; warm-start the latest actor, adding learned neck channels.
$Mimic.TorqueStandWholeBodyTraining = $Mimic.TorqueStandTraining.Clone()
$Mimic.TorqueStandWholeBodyTraining.Seed = 300993
$Mimic.TorqueStandWholeBodyTraining.InitializeFrom = 'logs/mimickit-training/stand-torque-spring-free-15min-03/model.pt'

# Stage posture pressure only after measured quiet-standing readiness.
$Mimic.TorqueStandStagedPostureTraining = $Mimic.TorqueStandTraining.Clone()
$Mimic.TorqueStandStagedPostureTraining.Seed = 300995
$Mimic.TorqueStandStagedPostureTraining.InitializeFrom = 'logs/mimickit-training/stand-torque-whole-body-15min-04/model.pt'
$Mimic.TorqueStandStagedPostureTraining.RewardMode = 'staged_v3'
$Mimic.TorqueStandStagedPostureTraining.AllowRewardChange = $true

# Worst-body posture objective from the measured all-eight balanced intermediate.
$Mimic.TorqueStandWorstPostureTraining = $Mimic.TorqueStandStagedPostureTraining.Clone()
$Mimic.TorqueStandWorstPostureTraining.Minutes = 30
$Mimic.TorqueStandWorstPostureTraining.Seed = 300996
$Mimic.TorqueStandWorstPostureTraining.InitializeFrom = 'logs/mimickit-training/stand-torque-staged-posture-15min-05/evaluation_000128.pt'
$Mimic.TorqueStandWorstPostureTraining.RewardMode = 'worst_v4'

# Learn upright torque control on short holds before extending their duration.
$Mimic.TorqueStandShortUprightTraining = $Mimic.TorqueStandWorstPostureTraining.Clone()
$Mimic.TorqueStandShortUprightTraining.Minutes = 15
$Mimic.TorqueStandShortUprightTraining.Seed = 300997
$Mimic.TorqueStandShortUprightTraining.RewardMode = 'short_upright_v5'

# Whole-body quiet-Stand imitation; preserves the latest balanced actor and AMP state.
$Mimic.TorqueStandWholePoseTraining = $Mimic.TorqueStandShortUprightTraining.Clone()
$Mimic.TorqueStandWholePoseTraining.Seed = 300998
$Mimic.TorqueStandWholePoseTraining.InitializeFrom = 'logs/mimickit-training/stand-torque-short-upright-15min-07/model.pt'
$Mimic.TorqueStandWholePoseTraining.RewardMode = 'whole_pose_v6'

# Short corrective pilot from the diagnosed final actor; execution needs authorization.
$Mimic.TorqueStandLimitAwarePilot = $Mimic.TorqueStandWholePoseTraining.Clone()
$Mimic.TorqueStandLimitAwarePilot.Minutes = 5
$Mimic.TorqueStandLimitAwarePilot.Seed = 300999
$Mimic.TorqueStandLimitAwarePilot.InitializeFrom = 'logs/mimickit-training/stand-torque-whole-pose-15min-08/model.pt'
$Mimic.TorqueStandLimitAwarePilot.RewardMode = 'limit_aware_v7'

# Add state occupancy cost to the measured v7 final actor; no automatic run.
$Mimic.TorqueStandLimitOccupancyPilot = $Mimic.TorqueStandLimitAwarePilot.Clone()
$Mimic.TorqueStandLimitOccupancyPilot.Seed = 301000
$Mimic.TorqueStandLimitOccupancyPilot.InitializeFrom = 'logs/mimickit-training/stand-torque-limit-aware-5min-09/model.pt'
$Mimic.TorqueStandLimitOccupancyPilot.RewardMode = 'limit_occupancy_v8'

# User-authorized reference imitation with one-minute episodes and native checks.
$Mimic.TorqueStandReferenceHour = $Mimic.TorqueStandWholePoseTraining.Clone()
$Mimic.TorqueStandReferenceHour.Minutes = 60
$Mimic.TorqueStandReferenceHour.Seed = 301001
$Mimic.TorqueStandReferenceHour.InitializeFrom = 'logs/mimickit-training/stand-torque-whole-pose-15min-08/model.pt'
$Mimic.TorqueStandReferenceHour.RewardMode = 'reference_v9'
$Mimic.TorqueStandReferenceHour.EpisodeSeconds = 60
$Mimic.TorqueStandReferenceHour.AssessmentSeconds = 60

# Fresh, untrained actor/critic/AMP/normalization; preserve the reference-hour settings.
$Mimic.TorqueStandReferenceFreshHour = $Mimic.TorqueStandReferenceHour.Clone()
$Mimic.TorqueStandReferenceFreshHour.InitializeFrom = 'logs/mimickit-training/stand-torque-reference-fresh-12/model.pt'

$localConfig = Join-Path $PSScriptRoot 'config.local.ps1'
if (Test-Path -LiteralPath $localConfig) { . $localConfig }
if ($env:MIMICKIT_PATH) { $Mimic.MimicKit = $env:MIMICKIT_PATH }
if ($env:P4F_GODOT_EXE) { $Mimic.Godot = $env:P4F_GODOT_EXE }

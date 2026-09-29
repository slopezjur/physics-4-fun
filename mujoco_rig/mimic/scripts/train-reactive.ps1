param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [Alias('Preset')][string] $PresetName = 'ReactiveTraining',
    [double] $Minutes = 0,
    [int] $Envs = 0,
    [int] $EvaluationInterval = 0,
    [string] $InitializeFrom = '',
    [string] $Out = '',
    [switch] $ResumeSelectedSameMass,
    [switch] $ResumeExperimentalFinal,
    [double] $ActorLearningRate = 0,
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
$preset = $settings[$PresetName]
if ($preset -isnot [hashtable]) { throw "Config must define preset $PresetName." }
if ($ResumeSelectedSameMass -and $ResumeExperimentalFinal) { throw 'Choose one continuation mode.' }
if ($ActorLearningRate -ne 0) {
    if ($ActorLearningRate -le 0 -or [double]::IsNaN($ActorLearningRate) -or [double]::IsInfinity($ActorLearningRate)) {
        throw 'ActorLearningRate must be finite and positive.'
    }
    $preset = $preset.Clone()
    $preset.ActorLearningRate = $ActorLearningRate
}
if ($Minutes -eq 0) { $Minutes = $preset.Minutes }
if ($Envs -eq 0) { $Envs = $settings.Envs }
if ($EvaluationInterval -eq 0) { $EvaluationInterval = $settings.EvaluationInterval }
if ($Minutes -le 0 -or [double]::IsNaN($Minutes) -or [double]::IsInfinity($Minutes)) { throw 'Minutes must be finite and positive.' }
if ($Envs -lt 32 -or $Envs % 8 -ne 0) { throw 'Envs must be at least 32 and a multiple of eight.' }
if ($EvaluationInterval -lt 1) { throw 'EvaluationInterval must be positive.' }
if (-not $InitializeFrom) { $InitializeFrom = $preset.InitializeFrom }
if (-not $Out) { $Out = Join-Path $settings.LogRoot ((Get-Date -Format 'yyyy-MM-dd_HH-mm-ss-fff') + '_reactive') }
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$python = Require-MimicPath $settings.Python
$arguments = @('-B', '-u', '-m', 'mujoco_rig.mimic.train_reactive_ball',
    '--mimickit', (Require-MimicPath $settings.MimicKit),
    '--initialize-from', (Require-MimicPath $InitializeFrom),
    '--reference', (Require-MimicPath $preset.Reference),
    '--ball-model', (Require-MimicPath $preset.BallModel),
    '--out', $destination, '--seconds', (Format-MimicNumber ($Minutes * 60)),
    '--envs', "$Envs", '--iterations', "$($settings.Iterations)",
    '--evaluation-interval', "$EvaluationInterval", '--seed', "$($preset.Seed)",
    '--recovery-mode', $preset.RecoveryMode)
if ($preset.RecoveryResetBank) { $arguments += @('--recovery-reset-bank', (Require-MimicPath $preset.RecoveryResetBank)) }
if ($preset.InitializeFromQuietReport) {
    if ($ResumeSelectedSameMass -or $ResumeExperimentalFinal) { throw 'Quiet-report initialization cannot combine continuation modes.' }
    $arguments += @('--initialize-from-quiet-report', (Require-MimicPath $preset.InitializeFromQuietReport))
}
$options = [ordered]@{
    'recovery-reset-fraction' = 'RecoveryResetFraction'
    'forward-fraction' = 'ForwardFraction'
    'quiet-fraction' = 'QuietFraction'
    'second-shot-fraction' = 'SecondShotFraction'
    'episode-seconds' = 'EpisodeSeconds'
    'speed-min' = 'SpeedMin'
    'speed-max' = 'SpeedMax'
    'quiet-replay-weight' = 'QuietReplayWeight'
    'impact-replay-weight' = 'ImpactReplayWeight'
    'impact-replay-samples' = 'ImpactReplaySamples'
    'actor-learning-rate' = 'ActorLearningRate'
    'actor-warmup' = 'ActorWarmup'
    'actor-stall-iterations' = 'ActorStallIterations'
    'steps-per-iter' = 'StepsPerIter'
    'td-lambda' = 'TdLambda'
    'discount' = 'Discount'
    'checkpoint-interval' = 'CheckpointInterval'
    'quiet-anchor-samples' = 'QuietAnchorSamples'
    'risk-leg-std' = 'RiskLegStd'
    'actor-max-kl' = 'ActorMaxKl'
    'action-bound-weight' = 'ActionBoundWeight'
}
foreach ($entry in $options.GetEnumerator()) {
    if ($preset.ContainsKey($entry.Value)) {
        $arguments += @("--$($entry.Key)", (Format-MimicNumber $preset[$entry.Value]))
    }
}
if ($ResumeSelectedSameMass) { $arguments += '--resume-selected-same-mass' }
if ($ResumeExperimentalFinal) { $arguments += '--resume-experimental-final' }
if ($preset.PreferRecoveredSteps) { $arguments += '--prefer-recovered-steps' }
if ($preset.TargetSet) { $arguments += @('--target-set', "$($preset.TargetSet)") }
if ($preset.RiskExplorationGate) { $arguments += @('--risk-exploration-gate', "$($preset.RiskExplorationGate)") }
if ($preset.ImpactReplayTrace) { $arguments += @('--impact-replay-trace', (Require-MimicPath $preset.ImpactReplayTrace)) }
if ($DryRun) { @{ executable = $python; arguments = $arguments } | ConvertTo-Json -Depth 4; return }
New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
Write-Host "[Mimic] Stationary recovery · $Minutes training minutes · $destination"
Push-Location $MimicProjectRoot
try {
    & $python @arguments 2>&1 | Tee-Object -FilePath "$destination.console.log"
    if ($LASTEXITCODE -ne 0) { throw "Reactive training failed ($LASTEXITCODE). See $destination.console.log" }
} finally { Pop-Location }

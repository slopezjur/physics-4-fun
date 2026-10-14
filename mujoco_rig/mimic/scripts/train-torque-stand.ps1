param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [string] $Preset = 'TorqueStandTraining',
    [string] $Out = 'logs/mimickit-training/stand-torque-spring-free-30min-01'
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
$training = $settings[$Preset]
if ($training -isnot [hashtable]) { throw "Config must define $Preset." }
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match '(mujoco_rig[./\\]mimic[./\\]train_|mimickit-training.*runner\.py)'
}
if ($active) { throw "Another Mimic trainer is active: $($active.ProcessId -join ', ')" }
$arguments = @('-B', '-u', '-m', 'mujoco_rig.mimic.train_torque_stand',
    '--mimickit', (Require-MimicPath $settings.MimicKit),
    '--reference', (Require-MimicPath $training.Reference),
    '--out', $destination, '--seconds', (Format-MimicNumber ($training.Minutes * 60)),
    '--envs', "$($settings.Envs)", '--evaluation-interval', "$($settings.EvaluationInterval)",
    '--seed', "$($training.Seed)")
if ($training.InitializeFrom) {
    $arguments += @('--initialize-from', (Require-MimicPath $training.InitializeFrom))
}
if ($training.RewardMode) { $arguments += @('--reward-mode', $training.RewardMode) }
if ($training.AllowRewardChange) { $arguments += '--allow-reward-change' }
if ($training.EpisodeSeconds) { $arguments += @('--episode-seconds', (Format-MimicNumber $training.EpisodeSeconds)) }
if ($training.AssessmentSeconds) { $arguments += @('--assessment-seconds', (Format-MimicNumber $training.AssessmentSeconds)) }
Push-Location $MimicProjectRoot
try {
    & (Require-MimicPath $settings.Python) @arguments
    if ($LASTEXITCODE -ne 0) { throw "Torque Stand trainer exited with code $LASTEXITCODE." }
} finally { Pop-Location }

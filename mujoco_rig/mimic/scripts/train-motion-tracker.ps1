param(
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$Config = 'mujoco_rig/mimic/scripts/config.ps1',
    [double]$Minutes = 0,
    [string]$Resume,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$settings = Read-MimicConfig $Config
$pilot = $settings.MotionTrackerFoundationTraining
if ($Minutes -le 0) { $Minutes = $pilot.Minutes }
$arguments = @('-m', 'mujoco_rig.mimic.train_motion_tracker',
    '--mimickit', (Require-MimicPath $settings.MimicKit),
    '--checkpoint', (Require-MimicPath $pilot.InitializeFrom),
    '--quiet-report', (Require-MimicPath $pilot.QuietReport),
    '--stationary', (Require-MimicPath $pilot.Stationary),
    '--out', (Resolve-MimicPath $Out),
    '--seconds', (Format-MimicNumber ($Minutes * 60)),
    '--episode-seconds', (Format-MimicNumber $pilot.EpisodeSeconds),
    '--root-tracking-sigma-m', (Format-MimicNumber $pilot.RootTrackingSigmaM),
    '--root-tracking-weight', (Format-MimicNumber $pilot.RootTrackingWeight),
    '--root-reward-mode', $pilot.RootRewardMode,
    '--root-accuracy-scale-m', (Format-MimicNumber $pilot.RootAccuracyScaleM),
    '--fixed-fraction', (Format-MimicNumber $pilot.FixedFraction),
    '--envs', [string]$settings.Envs,
    '--evaluation-interval', [string]$settings.EvaluationInterval,
    '--seed', [string]$pilot.Seed)
foreach ($motion in $pilot.Motions) { $arguments += @('--motion', (Require-MimicPath $motion)) }
if ($Resume) { $arguments += @('--resume', (Require-MimicPath $Resume)) }
$python = Require-MimicPath $settings.Python
if ($DryRun) {
    [pscustomobject]@{ Python=$python; Arguments=$arguments } | ConvertTo-Json -Depth 4
    return
}
Push-Location $MimicProjectRoot
try { & $python @arguments; if ($LASTEXITCODE -ne 0) { throw "Motion tracker exited with code $LASTEXITCODE" } }
finally { Pop-Location }

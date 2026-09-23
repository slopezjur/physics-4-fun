param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [string] $Experiment = '',
    [double] $Minutes = 0,
    [int] $Envs = 0,
    [int] $Iterations = 0,
    [int] $EvaluationInterval = 0,
    [ValidateSet('reference', 'recovery_v1')]
    [string] $BallReward = '',
    [ValidateSet('torque', 'target_pd')]
    [string] $Control = '',
    [ValidateSet('legacy', 'support_v2')]
    [string] $ContactMode = '',
    [ValidateSet('upstream', 'guarded')]
    [string] $Stability = '',
    [string] $InitFrom = '',
    [string] $SourceContract = '',
    [string] $Out = '',
    [switch] $NoSelectViewer,
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
# Capture command-line overrides before config.ps1 assigns its defaults.
$ControlOverride = $Control
$ContactModeOverride = $ContactMode
$StabilityOverride = $Stability
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
$Control = if ($ControlOverride) { $ControlOverride } else { $settings.Control }
$ContactMode = if ($ContactModeOverride) { $ContactModeOverride } elseif ($settings.ContactMode) { $settings.ContactMode } else { 'legacy' }
$Stability = if ($StabilityOverride) { $StabilityOverride } else { $settings.Stability }
if (-not $Control) { $Control = 'target_pd' }
if (-not $Stability) { $Stability = 'guarded' }
if ($Control -notin @('torque', 'target_pd')) { throw "Unsupported control mode: $Control" }
if ($ContactMode -notin @('legacy', 'support_v2')) { throw "Unsupported contact mode: $ContactMode" }
if ($ContactMode -eq 'support_v2' -and $Control -ne 'target_pd') {
    throw 'support_v2 requires target_pd control.'
}
if ($Stability -notin @('upstream', 'guarded')) { throw "Unsupported stability mode: $Stability" }
if (-not $Experiment) { $Experiment = $settings.DefaultExperiment }
if (-not $settings.Experiments.ContainsKey($Experiment)) { throw "Unknown experiment: $Experiment" }
$preset = $settings.Experiments[$Experiment]
if ($preset.Task -eq 'ball' -and ($Control -ne 'target_pd' -or $Stability -ne 'guarded')) {
    throw 'The ball Perturb experiment requires -Control target_pd and -Stability guarded.'
}
if ($Minutes -eq 0) { $Minutes = $settings.Minutes }
if ($Envs -eq 0) { $Envs = $settings.Envs }
if ($Iterations -eq 0) { $Iterations = $settings.Iterations }
if ($EvaluationInterval -eq 0) { $EvaluationInterval = $settings.EvaluationInterval }
if ($EvaluationInterval -lt 8 -or $EvaluationInterval % 8 -ne 0) { throw 'EvaluationInterval must be a positive multiple of eight.' }
if ($Minutes -le 0 -or [double]::IsNaN($Minutes) -or [double]::IsInfinity($Minutes)) { throw 'Minutes must be finite and positive.' }
if ([bool]$InitFrom -ne [bool]$SourceContract) { throw 'Specify both -InitFrom and -SourceContract.' }
# Guarded fine-tuning needs a source checkpoint. Preserve the configured warm start by default;
# upstream mode may intentionally start fresh when no explicit source is supplied.
if (-not $InitFrom -and $Stability -eq 'guarded') {
    $InitFrom = $settings.InitializeFrom; $SourceContract = $settings.SourceContract
}
$python = Require-MimicPath $settings.Python
$checkout = Require-MimicPath $settings.MimicKit
if (-not $Out) {
    $Out = Join-Path $settings.LogRoot ((Get-Date -Format 'yyyy-MM-dd_HH-mm-ss-fff') + "_$Experiment")
}
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$arguments = @('-u', '-X', 'faulthandler', '-m', 'mujoco_rig.mimic.train_stand',
    '--mimickit', $checkout, '--control', $Control, '--contact-mode', $ContactMode, '--stability', $Stability, '--task', $preset.Task,
    '--out', $destination, '--seconds', (Format-MimicNumber ($Minutes * 60)), '--envs', "$Envs",
    '--iterations', "$Iterations", '--seed', "$($settings.Seed)",
    '--evaluation-interval', "$EvaluationInterval", '--actor-max-kl', (Format-MimicNumber $settings.ActorMaxKl))
if ($InitFrom) {
    $arguments += @('--initialize-from', (Require-MimicPath $InitFrom),
        '--source-contract', (Require-MimicPath $SourceContract))
}
if ($NoSelectViewer) { $arguments += '--no-select-viewer' }
if ($preset.Task -eq 'ball') {
    if (-not $BallReward) { $BallReward = $preset.BallReward }
    if (-not $BallReward) { $BallReward = 'reference' }
    $arguments += @('--ball-speed-min', (Format-MimicNumber $preset.BallSpeedMin),
        '--ball-speed-max', (Format-MimicNumber $preset.BallSpeedMax), '--quiet-fraction', (Format-MimicNumber $preset.QuietFraction),
        '--ball-reward', $BallReward)
} elseif ($BallReward) {
    throw '-BallReward only applies to ball Perturb experiments.'
}
if ($DryRun) { @{ executable = $python; arguments = $arguments } | ConvertTo-Json -Depth 4; return }
New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
Write-Host "[Mimic] $Experiment · $Minutes minutes · $destination"
Push-Location $MimicProjectRoot
try {
    & $python @arguments 2>&1 | Tee-Object -FilePath "$destination.console.log"
    if ($LASTEXITCODE -ne 0) { throw "Mimic training failed ($LASTEXITCODE). See $destination.console.log" }
} finally { Pop-Location }

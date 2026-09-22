param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [string] $Experiment = '',
    [double] $Minutes = 0,
    [int] $Envs = 0,
    [int] $Iterations = 0,
    [ValidateSet('torque', 'target_pd')]
    [string] $Control = '',
    [ValidateSet('upstream', 'guarded')]
    [string] $Stability = '',
    [string] $InitFrom = '',
    [string] $SourceContract = '',
    [string] $Out = '',
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
# Capture command-line overrides before config.ps1 assigns its defaults.
$ControlOverride = $Control
$StabilityOverride = $Stability
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
$Control = if ($ControlOverride) { $ControlOverride } else { $settings.Control }
$Stability = if ($StabilityOverride) { $StabilityOverride } else { $settings.Stability }
if (-not $Control) { $Control = 'target_pd' }
if (-not $Stability) { $Stability = 'guarded' }
if ($Control -notin @('torque', 'target_pd')) { throw "Unsupported control mode: $Control" }
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
    '--mimickit', $checkout, '--control', $Control, '--stability', $Stability, '--task', $preset.Task,
    '--out', $destination, '--seconds', (Format-MimicNumber ($Minutes * 60)), '--envs', "$Envs",
    '--iterations', "$Iterations", '--seed', "$($settings.Seed)",
    '--evaluation-interval', "$($settings.EvaluationInterval)", '--actor-max-kl', (Format-MimicNumber $settings.ActorMaxKl))
if ($InitFrom) {
    $arguments += @('--initialize-from', (Require-MimicPath $InitFrom),
        '--source-contract', (Require-MimicPath $SourceContract))
}
if ($preset.Task -eq 'ball') {
    $arguments += @('--ball-speed-min', (Format-MimicNumber $preset.BallSpeedMin),
        '--ball-speed-max', (Format-MimicNumber $preset.BallSpeedMax), '--quiet-fraction', (Format-MimicNumber $preset.QuietFraction))
}
if ($DryRun) { @{ executable = $python; arguments = $arguments } | ConvertTo-Json -Depth 4; return }
New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
Write-Host "[Mimic] $Experiment · $Minutes minutes · $destination"
Push-Location $MimicProjectRoot
try {
    & $python @arguments 2>&1 | Tee-Object -FilePath "$destination.console.log"
    if ($LASTEXITCODE -ne 0) { throw "Mimic training failed ($LASTEXITCODE). See $destination.console.log" }
} finally { Pop-Location }

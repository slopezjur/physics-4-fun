param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [string] $Experiment = '',
    [double] $Minutes = 0,
    [int] $Envs = 0,
    [int] $Iterations = 0,
    [string] $InitFrom = '',
    [string] $SourceContract = '',
    [string] $Out = '',
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
if (-not $Experiment) { $Experiment = $settings.DefaultExperiment }
if (-not $settings.Experiments.ContainsKey($Experiment)) { throw "Unknown experiment: $Experiment" }
$preset = $settings.Experiments[$Experiment]
if ($Minutes -eq 0) { $Minutes = $settings.Minutes }
if ($Envs -eq 0) { $Envs = $settings.Envs }
if ($Iterations -eq 0) { $Iterations = $settings.Iterations }
if ($Minutes -le 0 -or [double]::IsNaN($Minutes) -or [double]::IsInfinity($Minutes)) { throw 'Minutes must be finite and positive.' }
if ([bool]$InitFrom -ne [bool]$SourceContract) { throw 'Specify both -InitFrom and -SourceContract.' }
if (-not $InitFrom) { $InitFrom = $settings.InitializeFrom; $SourceContract = $settings.SourceContract }
$python = Require-MimicPath $settings.Python
$checkout = Require-MimicPath $settings.MimicKit
if (-not $Out) {
    $Out = Join-Path $settings.LogRoot ((Get-Date -Format 'yyyy-MM-dd_HH-mm-ss-fff') + "_$Experiment")
}
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$arguments = @('-u', '-X', 'faulthandler', '-m', 'mujoco_rig.mimic.train_stand',
    '--mimickit', $checkout, '--control', 'target_pd', '--stability', 'guarded', '--task', $preset.Task,
    '--initialize-from', (Require-MimicPath $InitFrom), '--source-contract', (Require-MimicPath $SourceContract),
    '--out', $destination, '--seconds', (Format-MimicNumber ($Minutes * 60)), '--envs', "$Envs",
    '--iterations', "$Iterations", '--seed', "$($settings.Seed)",
    '--evaluation-interval', "$($settings.EvaluationInterval)", '--actor-max-kl', (Format-MimicNumber $settings.ActorMaxKl))
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

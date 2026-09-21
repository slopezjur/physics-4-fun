param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [string] $Experiment = '',
    [string] $Run = '',
    [double] $BallSpeed = 0,
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
if (-not $Experiment) { $Experiment = $settings.DefaultExperiment }
if (-not $settings.Experiments.ContainsKey($Experiment)) { throw "Unknown experiment: $Experiment" }
$preset = $settings.Experiments[$Experiment]
$task = $preset.Task
$bundle = Require-MimicPath $settings.WatchBundle
if ($Run) {
    $bundle = Require-MimicPath (Join-Path (Resolve-MimicPath $Run) 'export')
    $metadata = Get-Content -LiteralPath (Join-Path $bundle 'experiment.json') -Raw | ConvertFrom-Json
    $task = $metadata.task
    if ($task -eq 'ball' -and $BallSpeed -eq 0) { $BallSpeed = $metadata.horizontal_speed_max }
}
if ($task -notin @('stand', 'ball')) { throw "Unsupported exported task: $task" }
$scene = if ($task -eq 'ball') { 'MimicPerturb' } else { 'MimicStand' }
if (-not (Test-Path -LiteralPath (Join-Path $bundle 'contract.json'))) { throw 'Bundle has no contract.json.' }
$godot = Require-MimicPath $settings.Godot
$arguments = @('--path', $MimicProjectRoot, "res://Scenes/RL/Isaac3/MuJoCo/$scene.tscn", '--', '--mimic-bundle', $bundle)
if ($task -eq 'ball') {
    if ($BallSpeed -eq 0) { $BallSpeed = $settings.Experiments.perturb.BallSpeedMax }
    if ($BallSpeed -lt 1 -or $BallSpeed -gt 8 -or [double]::IsNaN($BallSpeed)) { throw 'BallSpeed must be 1–8 m/s.' }
    $arguments += @('--ball-speed', (Format-MimicNumber $BallSpeed))
}
if ($DryRun) { @{ executable = $godot; arguments = $arguments } | ConvertTo-Json -Depth 4; return }
Write-Host "[Mimic] $scene · $bundle"
& $godot @arguments
if ($LASTEXITCODE -ne 0) { throw "Godot exited with $LASTEXITCODE" }

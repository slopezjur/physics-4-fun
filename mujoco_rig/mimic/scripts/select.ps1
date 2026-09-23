param(
    [string] $Config = "$PSScriptRoot/config.ps1",
    [string] $Run = '',
    [switch] $Previous,
    [string] $Experiment = ''
)
$ErrorActionPreference = 'Stop'
if ([bool]$Run -eq [bool]$Previous) { throw 'Specify either -Run or -Previous.' }
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
$python = Require-MimicPath $settings.Python
$arguments = @('-B', '-m', 'mujoco_rig.mimic.viewer_selection')
if ($Run) {
    $arguments += @('--run', (Require-MimicPath $Run))
} else {
    if (-not $Experiment) { $Experiment = $settings.DefaultExperiment }
    if (-not $settings.Experiments.ContainsKey($Experiment)) { throw "Unknown experiment: $Experiment" }
    $arguments += @('--rollback', $settings.Experiments[$Experiment].Task)
}
Push-Location $MimicProjectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Viewer selection failed. The current selection was preserved.' }
} finally { Pop-Location }

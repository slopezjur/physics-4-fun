param(
    [Parameter(Mandatory = $true)][string] $Out,
    [string] $Config = "$PSScriptRoot/perturb-pilot-config.ps1"
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$settings = Read-MimicConfig $Config
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'mujoco_rig[./\\]mimic[./\\](train_|run_|evaluate_|measure_|benchmark_)'
}
if ($active) { throw "Another Mimic experiment is active: $($active.ProcessId -join ', ')" }
$arguments = @('-B', '-u', '-m', 'mujoco_rig.mimic.run_perturb_pilot',
    '--out', $destination, '--protocol', (Require-MimicPath $settings.Protocol),
    '--mimickit', (Require-MimicPath $settings.MimicKit))
Push-Location $MimicProjectRoot
try {
    & (Require-MimicPath $settings.Python) @arguments
    if ($LASTEXITCODE -ne 0) { throw "Perturb pilot exited with code $LASTEXITCODE; inspect preserved failure.json and console logs." }
} finally { Pop-Location }

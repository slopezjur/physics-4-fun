param([Parameter(Mandatory = $true)][string] $Out)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'mujoco_rig[./\\]mimic[./\\](train_|run_|evaluate_|measure_|benchmark_)'
}
if ($active) { throw "Another Mimic experiment is active: $($active.ProcessId -join ', ')" }
$arguments = @('-B', '-u', '-m', 'mujoco_rig.mimic.evaluate_stand_ball_boundary',
    '--out', $destination,
    '--protocol', (Require-MimicPath 'mujoco_rig/mimic/stand_ball_boundary.json'),
    '--mimickit', (Require-MimicPath 'D:/Proyectos/Juegos/Tools/MimicKit'))
Push-Location $MimicProjectRoot
try {
    & (Require-MimicPath 'mujoco_rig/mimic/.venv/Scripts/python.exe') @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ball assessment failed ($LASTEXITCODE); inspect preserved output." }
} finally { Pop-Location }

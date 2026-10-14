param([Parameter(Mandatory = $true)][string] $Out)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$destination = Resolve-MimicPath $Out
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'mujoco_rig[./\\]mimic[./\\](train_|run_|evaluate_|admit_|measure_|benchmark_)'
}
if ($active) { throw "Another Mimic experiment is active: $($active.ProcessId -join ', ')" }
$arguments = @('-B', '-u', '-m', 'mujoco_rig.mimic.admit_stand_ball_gpu',
    '--out', $destination,
    '--prior', (Require-MimicPath 'logs/mimickit-training/stand-ball-boundary-20261006-02'),
    '--mimickit', (Require-MimicPath 'D:/Proyectos/Juegos/Tools/MimicKit'))
Push-Location $MimicProjectRoot
try {
    & (Require-MimicPath 'mujoco_rig/mimic/.venv/Scripts/python.exe') @arguments
    if ($LASTEXITCODE -ne 0) { throw "GPU ball admission failed ($LASTEXITCODE); inspect preserved output." }
} finally { Pop-Location }

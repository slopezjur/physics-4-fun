# The gate that makes GPU training safe. Run it after ANY change to the rig, the MJCF, the
# environments, or the mujoco / mujoco_warp versions.
#
#     .\parity.ps1
#
# Two independent checks, because there are two independent ways this can go wrong:
#
#   1. ENGINE parity  (parity_gpu.py)  - does mujoco_warp on the GPU integrate the same physics as
#      MuJoCo's C engine on the CPU? It cannot be identical: warp is float32, the C engine float64.
#      What matters is the SIZE of the gap, and whether it is the engine at all - so the script
#      also runs two CPU trajectories started 1 nanometre apart. Measured, that control pair does
#      not diverge at all over 2.5 s, which means the GPU departure is real and not chaos.
#
#      It also asserts the constraint buffers have headroom. Overflow is reported only as a line on
#      stderr from inside a CUDA kernel and otherwise just silently drops constraints - raising
#      njmax from the default took 2.5 s divergence from 21 degrees to 0.14.
#
#   2. TASK parity  (test_env_parity.py) - do perturb_env.py and perturb_env_warp.py define the
#      same task? They are two implementations of one environment, and when they drift the failure
#      is silent and expensive: training optimises one reward while scoring reports another, and
#      the result looks like a transfer problem rather than a bug.
#
# Neither check licenses scoring on the GPU. Scoring stays on CPU MuJoCo, which is what Godot runs.
. "$PSScriptRoot/config.ps1"

$failed = @()

Write-Host ""
Write-Host "=== 1/2  engine parity: mujoco_warp (float32) vs MuJoCo C (float64) ===" -ForegroundColor Cyan
try {
    Invoke-Mujoco -Script "parity_gpu.py"
} catch {
    $failed += "engine parity"
    Write-Host "[parity] engine check FAILED: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 2/2  task parity: perturb_env.py vs perturb_env_warp.py ===" -ForegroundColor Cyan
try {
    Invoke-Mujoco -Script "test_env_parity.py"
} catch {
    $failed += "task parity"
    Write-Host "[parity] task check FAILED: $_" -ForegroundColor Red
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "[parity] both checks passed - the GPU backend is safe to train on." -ForegroundColor Green
} else {
    Write-Host "[parity] FAILED: $($failed -join ', ')." -ForegroundColor Red
    Write-Host "         Do not trust GPU training until this passes. Train with -Backend cpu" -ForegroundColor Red
    Write-Host "         in the meantime; it is 26x slower and correct." -ForegroundColor Red
    exit 1
}

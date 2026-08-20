# Same as train.ps1 but renders ONE instance in a window so you can watch it live.
#
# Camera: right-drag to look, WASD + Q/E to move, Shift to boost.
#
# Two things worth knowing:
#  - The visible instance is essentially FREE on this machine: measured 2,235 steps/s with
#    --viz against 2,144 without, in the same run. Earlier runs suggested a large penalty;
#    those predate the action-space fix and had other load on the machine, and the note
#    warning about a slowdown was wrong.
#  - the visible instance runs at $Speedup like the rest, so motion looks very fast. Lower
#    $Speedup in config.ps1 (2-4) if you actually want to watch the movement.
. "$PSScriptRoot/config.ps1"

& "$PSScriptRoot/export.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Training with 1 visible instance..." -ForegroundColor Cyan
& $Python $TrainScript `
    --env_path=$BuildExe `
    --n_parallel=$NParallel `
    --speedup=$Speedup `
    --timesteps=$Timesteps `
    --experiment_dir=$ExperimentDir `
    --experiment_name=$ExperimentName `
    --save_every_seconds=$SaveEverySeconds `
    --max_seconds=$MaxSeconds `
    --viz

# Full training run: export first, then train headless.
#
# Checkpoints (.zip + .onnx together, step-stamped) land in rl/runs/<name>_N/ every
# $SaveEverySeconds. Nothing is ever overwritten, so any earlier session stays recoverable.
# Ctrl+C stops cleanly and still writes a final checkpoint.
. "$PSScriptRoot/config.ps1"

& "$PSScriptRoot/export.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Training: $NParallel procs, speedup x$Speedup, $Timesteps steps" -ForegroundColor Cyan
& $Python $TrainScript `
    --env_path=$BuildExe `
    --n_parallel=$NParallel `
    --speedup=$Speedup `
    --timesteps=$Timesteps `
    --experiment_dir=$ExperimentDir `
    --experiment_name=$ExperimentName `
    --save_every_seconds=$SaveEverySeconds `
    --max_seconds=$MaxSeconds


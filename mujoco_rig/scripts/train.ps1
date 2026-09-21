# Perturb fine-tuning uses explicit seeds and CPU-evaluated chunks of at most five minutes.
#
#     .\train.ps1 -InitFrom <validated-checkpoint> -Minutes 15
#
# A long budget increases the number of chunks. Failed candidates are retained on disk
# for diagnosis but never become the next seed. Export is a separate promotion step.
# Walk and intentional -FromScratch experiments use the direct trainer.
param(
    # 'perturb' or 'walk'. Overrides $Task in config.ps1 for this command only.
    [ValidateSet('perturb', 'walk')]
    [string] $Task,
    # 'warp' (GPU) or 'cpu'. Overrides $Backend for this command only.
    [ValidateSet('warp', 'cpu')]
    [string] $Backend,
    # Overrides $MaxMinutes for this command only, e.g. .\train.ps1 -Minutes 5
    [double] $Minutes = -1,
    [int] $Envs = 0,
    # Names the run directory under logs/mujoco. Give variants distinct names: eval.ps1 picks the
    # NEWEST run, so an unnamed smoke test silently becomes the thing you score.
    [string] $Name = "",
    [int] $Seed = 0,
    # Continue from an explicit checkpoint; compatible optimiser/warm-up state travels with it.
    [string] $InitFrom = "",
    # Explicit research escape hatch: zero torque on this plant does not stand.
    [switch] $FromScratch
)
# Capture BEFORE dot-sourcing - config.ps1 defines $Backend, $Envs and $MaxMinutes itself, so
# reading the parameters afterwards returns the config values and every override is ignored.
$TaskOverride    = $Task
$BackendOverride = $Backend
$EnvsOverride    = $Envs
$MinutesOverride = $Minutes
. "$PSScriptRoot/config.ps1"

# Default the run name to the task BEFORE $cmd is built - it is passed by value.
if (-not $Name) { $Name = $Task }

if ($FromScratch -and $InitFrom) { throw 'Choose -InitFrom or -FromScratch, not both.' }
if ($Task -eq 'perturb' -and -not $FromScratch) {
    if (-not $InitFrom) {
        throw 'Perturb requires -InitFrom <validated checkpoint>. No checkpoint is selected by recency.'
    }
    if ($MaxMinutes -le 0) { throw 'Perturb requires a positive -Minutes training budget.' }
    # Resume the checkpoint's difficulty; overnight.py evaluates whether it should advance.
    # Long budgets are divided into evaluated chunks; rejected results never become seeds.
    $sessionCount = [int][Math]::Ceiling($MaxMinutes / 5.0)
    $sessionMinutes = $MaxMinutes / $sessionCount
    $chainArgs = @('--task', 'perturb', '--backend', $Backend, '--seed', $InitFrom,
                   '--sessions', $sessionCount, '--minutes', $sessionMinutes,
                   '--envs', $Envs, '--steps', $Steps, '--seconds', $Seconds, '--training_seed', $Seed,
                   '--entropy_coef', $EntropyCoef,
                   '--speed_end', $SpeedEnd, '--speed_step', $SpeedStep,
                   '--ball_every', $BallEvery[0], $BallEvery[1], '--tag', $Name,
                   '--abort_at', 2, '--no_promote',
                   '--extra', "--desired_kl $DesiredKl --init_std $InitStd --epochs $Epochs")
    Push-Location $ProjectRoot
    try {
        & $Python -u (Join-Path $PSScriptRoot 'overnight.py') @chainArgs
        if ($LASTEXITCODE -ne 0) { throw "Perturb training chain exited with code $LASTEXITCODE." }
    } finally { Pop-Location }
    return
}

$cmd = @('--task', $Task,
         '--backend', $Backend,
         '--num_envs', $Envs,
         '--steps', $Steps,
         '--seconds', $Seconds,
         '--seed', $Seed,
         '--run_name', $Name,
         '--max_minutes', $MaxMinutes,
         '--init_std', $InitStd,
         '--entropy_coef', $EntropyCoef,
         '--desired_kl', $DesiredKl,
         '--epochs', $Epochs,
         '--ball_every', $BallEvery[0], $BallEvery[1],
         '--speed_start', $SpeedStart,
         '--speed_end', $SpeedEnd,
         '--speed_step', $SpeedStep,
         '--promote_at', $PromoteAt,
         '--stage_min_episodes', $StageMinEpisodes)
if ($InitFrom)         { $cmd += @('--init_from', $InitFrom) }
if ($FromScratch)      { $cmd += '--from_scratch' }
if ($Iterations -gt 0) { $cmd += @('--iterations', $Iterations) } else { $cmd += @('--iterations', 1000000) }

$budget = if ($MaxMinutes -gt 0) { "$MaxMinutes min" } else { "no time cap" }
$batch = $Envs * $Steps
Write-Host "[train] $Task on $Backend : $Envs envs x $Steps steps = $batch samples/update, $budget" -ForegroundColor Cyan
Write-Host "[train] curriculum $SpeedStart -> $SpeedEnd m/s, promote at $PromoteAt after $StageMinEpisodes episodes" -ForegroundColor DarkGray

Invoke-Mujoco -Script "train.py" -Arguments $cmd

Write-Host ""
Write-Host "[train] done. Score it on the SHIPPING engine before believing anything:" -ForegroundColor Cyan
Write-Host "        .\eval.ps1" -ForegroundColor Cyan

# Trains the perturbation policy on the MuJoCo plant using every setting from config.ps1.
#
#     .\train.ps1                       # GPU, 30 minutes, settings from config.ps1
#     .\train.ps1 -Minutes 90
#     .\train.ps1 -Backend cpu -Envs 96 # the CPU control, for an apples-to-apples comparison
#
# Stops at whichever comes first, $MaxMinutes or $Iterations, always on a checkpoint boundary.
# Ctrl+C also leaves the last checkpoint intact.
#
# The policy starts AS the do-nothing controller: the actor's output layer is zeroed, so an
# untrained network commands the rest pose, which already stands 40 s at 100% upright. A randomly
# initialised actor throws that away on step one - measured, it scored 27.9% upright in a QUIET
# room where doing nothing scores 100%.
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
    # Continue from a checkpoint (weights only). Use it to run successive sessions that accumulate
    # instead of each starting from the rest pose.
    [string] $InitFrom = ""
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
if ($Iterations -gt 0) { $cmd += @('--iterations', $Iterations) } else { $cmd += @('--iterations', 1000000) }

$budget = if ($MaxMinutes -gt 0) { "$MaxMinutes min" } else { "no time cap" }
$batch = $Envs * $Steps
Write-Host "[train] $Task on $Backend : $Envs envs x $Steps steps = $batch samples/update, $budget" -ForegroundColor Cyan
Write-Host "[train] curriculum $SpeedStart -> $SpeedEnd m/s, promote at $PromoteAt after $StageMinEpisodes episodes" -ForegroundColor DarkGray

Invoke-Mujoco -Script "train.py" -Arguments $cmd

Write-Host ""
Write-Host "[train] done. Score it on the SHIPPING engine before believing anything:" -ForegroundColor Cyan
Write-Host "        .\eval.ps1" -ForegroundColor Cyan

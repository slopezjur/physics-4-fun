# Scores $Task's newest policy against its own criteria and prints the table.
#
# Deliberately not "mean reward". Each task has a way of looking good while being wrong: stand can
# survive in a crouch, walk can slide with both feet planted, run can be a fast walk with no
# flight phase, and any of them can degenerate to bang-bang control while every quality metric
# still reads well. The evaluators measure those directly - see "Judging a checkpoint" in the
# README for what to read in each table.
param(
    # Overrides $Task in config.ps1 for this command only. Omit to use the configured task.
    [ValidateSet('stand', 'walk', 'perturb', 'run')]
    [string] $Task
)
$TaskOverride = $Task
. "$PSScriptRoot/config.ps1"

$run = Get-IsaacRun -Experiment $Experiment -Minimum $MinIterations -Pinned $RunName
if (-not $run) {
    throw "No trained run for '$Task'. Train it first with .\train.ps1, or set `$RunName in config.ps1."
}
$ckpt = Get-IsaacCheckpoint -RunDir $run.FullName

$cmd = @("$PSScriptRoot/$Evaluator", '--checkpoint', $ckpt, '--num_envs', $EvalEnvs)
# Only the gait evaluator takes a commanded speed; stand and perturbation have no command.
if ($Evaluator -eq 'evaluate_walk.py') { $cmd += @('--speed', $EvalSpeed) }

Write-Host "[eval] $Task : $($run.Name), $EvalEnvs envs" -ForegroundColor Cyan
Invoke-Isaac -Arguments $cmd

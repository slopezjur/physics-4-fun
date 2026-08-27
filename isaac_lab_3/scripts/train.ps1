# Trains $Task under Newton/XPBD using every setting from config.ps1.
#
# Stops at whichever comes first, $MaxMinutes or $Iterations, always on a checkpoint boundary so
# the run stays resumable. Ctrl+C also leaves the last checkpoint intact.
#
#     .\train.ps1 -Task perturb -Minutes 15 -InitFrom stand
#
# Perturb, Walk and Run all inherit Stand's observation and action layout, so -InitFrom seeds them
# from a Stand checkpoint - weights only, no optimiser state, no iteration count.
param(
    # Overrides $Task in config.ps1 for this command only.
    [ValidateSet('stand', 'perturb', 'walk', 'run')]
    [string] $Task,
    # Overrides $ExperimentName - which logs/rsl_rl subtree this run WRITES into. Use it to keep a
    # variant off a chained run's checkpoint path; night.py resumes from the newest checkpoint in
    # an experiment tree, so two runs sharing a name silently adopt each other's weights.
    [string] $Experiment,
    # Seed the networks from another task's newest checkpoint. Takes a task name ('stand') or a
    # path to a model_*.pt. Mutually exclusive with $ResumeFrom, which train.py enforces.
    [string] $InitFrom,
    # Overrides $MaxMinutes for this command only, e.g. .\train.ps1 -Minutes 5
    [double] $Minutes = -1,
    [int] $Envs = 0,
    # Override env-cfg fields for this run, e.g. -Set balance_max_torque=75,balance_reaction=True.
    # Recorded in the run's params/env.yaml, unlike the P4F_* environment variables.
    [string[]] $Set = @(),
    # XPBD solver iterations for this run only. **Part of the trained dynamics** - a policy trained
    # at 8 is driving a different body from one trained at 2, and the two are not interchangeable.
    # 0 keeps $SolverIterations from config.ps1.
    [int] $Solver = 0
)
# Capture BEFORE dot-sourcing - config.ps1 defines $Envs and $MaxMinutes itself, so reading the
# parameters afterwards returns the config values and every override is silently ignored.
$EnvsOverride       = $Envs
$MinutesOverride    = $Minutes
$TaskOverride       = $Task
$ExperimentOverride = $Experiment
. "$PSScriptRoot/config.ps1"

if ($MinutesOverride -ge 0) { $MaxMinutes = $MinutesOverride }
if ($EnvsOverride -gt 0)    { $Envs = $EnvsOverride }

# A task name resolves to that task's newest checkpoint; anything else is taken as a path. Resolved
# here rather than in train.py because the run-selection rules ($MinIterations, $ExperimentName)
# live in config.ps1 and should mean the same thing for a seed as for playback.
if ($InitFrom -and -not (Test-Path -LiteralPath $InitFrom)) {
    if (-not $TaskMap.Contains($InitFrom)) {
        throw "-InitFrom '$InitFrom' is neither a checkpoint path nor a known task ($(($TaskMap.Keys) -join ', '))."
    }
    $seedExperiment = $TaskMap[$InitFrom].Experiment
    $seedRun = Get-Isaac3Run -Experiment $seedExperiment -Minimum $MinIterations
    if (-not $seedRun) { throw "-InitFrom '$InitFrom': nothing trained under $seedExperiment." }
    $InitFrom = Get-Isaac3Checkpoint -RunDir $seedRun.FullName
    Write-Host "[train] seeding from $seedExperiment/$($seedRun.Name)" -ForegroundColor DarkGray
}

if ($Solver -gt 0) { $SolverIterations = $Solver }
$env:P4F_XPBD_ITERATIONS = "$SolverIterations"

$cmd = @("$PSScriptRoot/train.py", '--task', $TaskId, '--num_envs', $Envs,
         '--iterations', $Iterations, '--max_minutes', $MaxMinutes)
if ($RunSuffix)       { $cmd += @('--run_name', $RunSuffix) }
if ($ResumeFrom)      { $cmd += @('--resume', $ResumeFrom) }
if ($InitFrom)        { $cmd += @('--init_from', $InitFrom) }
if ($ExperimentName)  { $cmd += @('--experiment', $ExperimentName) }
foreach ($pair in $Set) { $cmd += @('--set', $pair) }

$budget = if ($MaxMinutes -gt 0) { "$MaxMinutes min" } else { "no time cap" }
Write-Host "[train] $Task -> $Experiment : $Envs envs, $budget, XPBD iterations $SolverIterations" -ForegroundColor Cyan

Invoke-Isaac3 -Arguments $cmd

# Headless training run for $Task, at $Envs environments.
#
# Checkpoints land in isaac_lab/logs/rsl_rl/<experiment>/<timestamp>/ every 50 iterations, and
# nothing is ever overwritten, so an earlier session stays recoverable. Ctrl+C stops cleanly.
#
# Set $BootstrapFrom in config.ps1 to start from another task's weights - that is what makes walk
# and perturbation cheap, since both inherit a policy that can already stand.
param(
    # Overrides $Task in config.ps1 for this command only. Omit to use the configured task.
    [ValidateSet('stand', 'walk', 'perturb', 'run')]
    [string] $Task
)
$TaskOverride = $Task
. "$PSScriptRoot/config.ps1"

$cmd = @("$PSScriptRoot/train.py", '--task', $TaskId, '--headless', '--num_envs', $Envs)
if ($Iterations -gt 0) { $cmd += @('--max_iterations', $Iterations) }
if ($MaxMinutes -gt 0) { $cmd += @('--max_seconds', ($MaxMinutes * 60)) }

if ($BootstrapFrom) {
    $parentSpec = @{ 'stand' = 'p4f_stand'; 'walk' = 'p4f_walk'; 'perturb' = 'p4f_perturb'; 'run' = 'p4f_run' }
    $parentExp = $parentSpec[$BootstrapFrom]
    if (-not $parentExp) { throw "Unknown `$BootstrapFrom '$BootstrapFrom'." }

    $parentRun = Get-IsaacRun -Experiment $parentExp -Minimum $MinIterations
    if (-not $parentRun) {
        throw "`$BootstrapFrom is '$BootstrapFrom' but it has no trained run. Train that first, or clear `$BootstrapFrom."
    }
    $parentCkpt = Get-IsaacCheckpoint -RunDir $parentRun.FullName
    Write-Host "[train] Bootstrapping from '$BootstrapFrom': $($parentRun.Name)" -ForegroundColor DarkCyan
    $cmd += @('--init_from', $parentCkpt)
}

if ($Iterations -gt 0) {
    $iterLabel = "$Iterations iterations"
} else {
    $iterLabel = "the task's own max_iterations"
}
if ($MaxMinutes -gt 0) { $iterLabel += ", stopping after $MaxMinutes min" }
Write-Host "[train] $Task : $Envs envs, $iterLabel" -ForegroundColor Cyan

Invoke-Isaac -Arguments $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& "$PSScriptRoot/summary.ps1"

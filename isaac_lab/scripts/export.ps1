# Exports $Task's newest policy to ONNX and promotes it to isaac_lab/exported/<task>_policy.onnx.
#
# The promotion is the point. play.py always writes "policy.onnx" next to the checkpoint, so
# without a task-specific name every export sits in a different directory under the same filename
# and you have to remember which was which.
#
# The graph is obs[1,143] -> actions[1,36] with the observation normalizer folded in, so Godot
# feeds raw observations and must not normalize them itself. See obs_action_contract.md.
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

Write-Host "[export] $Task : $($run.Name) -> $(Split-Path $ckpt -Leaf)" -ForegroundColor Cyan
Invoke-Isaac -Arguments @("$PSScriptRoot/play.py", '--task', $TaskId, '--headless',
                          '--num_envs', 16, '--checkpoint', $ckpt, '--export_only')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$src = Join-Path $run.FullName 'exported'
if (-not (Test-Path -LiteralPath (Join-Path $src 'policy.onnx'))) {
    throw "play.py reported success but wrote no policy.onnx in $src."
}
if (-not (Test-Path -LiteralPath $Exported)) { New-Item -ItemType Directory -Path $Exported | Out-Null }

Copy-Item (Join-Path $src 'policy.onnx') (Join-Path $Exported "$($Task)_policy.onnx") -Force
Copy-Item (Join-Path $src 'policy.pt')   (Join-Path $Exported "$($Task)_policy.pt")   -Force
Write-Host "[export] promoted to exported\$($Task)_policy.onnx" -ForegroundColor Green

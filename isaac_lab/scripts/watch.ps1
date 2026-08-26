# Opens the Isaac Sim viewport running $Task's newest policy - the arena.
#
# This never exits on its own: play.py loops on `while simulation_app.is_running()`. Close the
# window when you are done. A stray one holds several GB of VRAM and quietly starves anything
# training - that cost 35% throughput for 90 minutes once, with nothing in either log to say why.
#
# The FIRST windowed launch takes several minutes building the renderer's shader cache and looks
# frozen while it does. Every launch after that is under a minute.
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

$cmd = @("$PSScriptRoot/play.py", '--task', $TaskId, '--num_envs', $WatchEnvs, '--load_run', $run.Name)
if ($KitArgs) { $cmd += $KitArgs }

Write-Host "[watch] $Task : $($run.Name), $WatchEnvs envs" -ForegroundColor Cyan
if (-not $ForceD3D12) {
    Write-Host "[watch] `$ForceD3D12 is off - Isaac Sim will use Vulkan, which crashes on driver 610.88." -ForegroundColor Yellow
}

Invoke-Isaac -Arguments $cmd

# Opens a window running a policy on CPU MuJoCo - the engine that scores it AND the one Godot
# drives through P/Invoke.
#
#     .\watch.ps1                        # the brain the Godot scenes are loading
#     .\watch.ps1 -Latest                # the newest checkpoint instead
#     .\watch.ps1 -ZeroAction            # the UNPOWERED body, the baseline every result is read against
#     .\watch.ps1 -Task walk -Command 0.6,0,0
#     .\watch.ps1 -Reload                # follow a training session; reloads as checkpoints appear
#     .\watch.ps1 logs\mujoco\<run>\model_300.pt
#
# **The default is what SHIPS, not what is newest.** It resolves the checkpoint from
# balance_policy.contract.json's `source_checkpoint`, so the window shows the same brain
# MujocoStand.tscn and MujocoPerturb.tscn load. Watching one policy while shipping another is not a
# display bug - it cost the Isaac track a wrong conclusion, where the viewer opened
# stand/night13 at action_scale 0.4 while Godot ran stand_assist/night06 at 0.15. `-Latest` is the
# opt-in for "what did I just make".
#
# Keys in the window, matching MujocoDummy so the two surfaces behave alike:
#   R reset   B fire a ball now   Space shove the pelvis   P pause
param(
    [Parameter(Position = 0)][string] $Checkpoint = "",
    # 'perturb' or 'walk'. Overrides $Task in config.ps1 for this command only.
    [ValidateSet('perturb', 'walk')]
    [string] $Task,
    # Open the NEWEST checkpoint of this task instead of the one the Godot scenes load.
    [switch] $Latest,
    # Watch the UNTRAINED body - no policy at all. On the torque plant this crumples, and that is
    # the reference every scored result is measured against.
    [switch] $ZeroAction,
    # Ball speed in m/s. Default: the difficulty the checkpoint actually reached, which travels
    # with the weights - watching a policy at a difficulty it never trained at shows nothing.
    [double] $BallSpeed = 0,
    # Seconds between shots. Default: the training cadence from config.ps1.
    [double[]] $BallEvery = @(),
    # Walk only: forward m/s, lateral m/s, turn rad/s.
    [double[]] $Command = @(0.6, 0.0, 0.0),
    # Reload the checkpoint when it changes on disk, to follow a training run.
    [switch] $Reload
)
# Capture BEFORE dot-sourcing: config.ps1 defines $Task and $BallEvery itself, so reading the
# parameters afterwards returns the config values and every override is silently ignored. eval.ps1
# and train.ps1 carry the same guard for the same reason.
$TaskOverride      = $Task
$BallEveryOverride = $BallEvery
. "$PSScriptRoot/config.ps1"

# One value means a fixed cadence. Launched through `powershell -File`, `-BallEvery 3,3` can
# arrive as a single element rather than two, so accepting one is not just a convenience.
if ($BallEveryOverride.Count -ge 2)     { $BallEvery = @($BallEveryOverride[0], $BallEveryOverride[1]) }
elseif ($BallEveryOverride.Count -eq 1) { $BallEvery = @($BallEveryOverride[0], $BallEveryOverride[0]) }

# **The task is NOT taken from config.ps1.** $Task there is what you are TRAINING; a checkpoint
# records what it was trained for, and watch.py reads it. Passing config's value opened the balance
# brain on the walk task - no ball gun, wrong question, and nothing on screen said so. -Task
# overrides, and -ZeroAction has no checkpoint to ask so it needs the explicit value.
$cmd = @('--ball_every', $BallEvery[0], $BallEvery[1])
if ($TaskOverride) { $cmd += @('--task', $TaskOverride) }
if ($BallSpeed -gt 0) { $cmd += @('--ball_speed', $BallSpeed) }
if ($TaskOverride -eq 'walk') { $cmd += @('--command', $Command[0], $Command[1], $Command[2]) }

if ($ZeroAction) {
    Write-Host "[watch] the unpowered body on CPU MuJoCo - no policy at all" -ForegroundColor Cyan
    $zero = $cmd
    if (-not $TaskOverride) { $zero += @('--task', 'perturb') }
    Invoke-Mujoco -Script 'watch.py' -Arguments ($zero + '--zero_action')
    return
}

if (-not $Checkpoint) {
    $contract = Join-Path $ProjectRoot 'mujoco_rig/balance_policy.contract.json'
    if (-not $Latest -and (Test-Path -LiteralPath $contract)) {
        $shipped = (Get-Content -LiteralPath $contract -Raw | ConvertFrom-Json).source_checkpoint
        if ($shipped -and (Test-Path -LiteralPath (Join-Path $ProjectRoot $shipped))) {
            $Checkpoint = Join-Path $ProjectRoot $shipped
            Write-Host "[watch] the brain the Godot scenes load ($shipped)" -ForegroundColor DarkGray
        }
        else {
            Write-Host "[watch] $shipped is named by the contract but missing; using the newest run." -ForegroundColor Yellow
        }
    }
}

if (-not $Checkpoint) {
    $dir = Get-MujocoRun -Minimum $MinIterations -Pinned $RunName
    if (-not $dir) {
        Write-Host "[watch] nothing trained yet under $LogRoot (need >= $MinIterations checkpoints)." -ForegroundColor Yellow
        Write-Host "        Run .\train.ps1 first, or .\watch.ps1 -ZeroAction to see the plant." -ForegroundColor Yellow
        return
    }
    $Checkpoint = Get-MujocoCheckpoint -RunDir $dir.FullName
    Write-Host "[watch] run $($dir.Name)" -ForegroundColor DarkGray
}

$cmd += @('--checkpoint', $Checkpoint)
if ($Reload) { $cmd += '--reload' }

Write-Host "[watch] $(Split-Path $Checkpoint -Leaf) on CPU MuJoCo - close the window to exit" -ForegroundColor Cyan
Invoke-Mujoco -Script 'watch.py' -Arguments $cmd

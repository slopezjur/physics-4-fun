# Scores a trained policy on CPU MuJoCo - the engine Godot actually drives through P/Invoke.
#
#     .\eval.ps1                    # newest run, its last checkpoint
#     .\eval.ps1 -Run 2026-09-08_23-06-28_gpu_smoke
#     .\eval.ps1 -Checkpoint logs\mujoco\<run>\model_500.pt
#     .\eval.ps1 -BaselineOnly      # what the plant does with NO policy
#
# **This never runs on the GPU engine, by design.** mujoco_warp is float32 and MuJoCo's C engine is
# float64; parity.ps1 measures the divergence. Scoring a policy on the engine that trained it is
# how the Isaac track kept producing checkpoints that scored 100% and then fell over in Godot.
#
# The number to beat, measured with no policy at all over 40 s:
#   Measured on the REAL-COMPLIANCE plant (Godot's effective gains), zero action, 40 s:
#     quiet room    6.4% upright, 0/8 survived, falls at 2.5 s
#     under fire    6.4% upright - identical, because it collapses before the first ball lands
#
#   That is the point of the compliant plant: the body CANNOT stand by itself, so any uprightness
#   in a scored run is the policy's legs and nothing else. The old references (100% quiet, 29.1%
#   under fire) were measured on a body 2.5-6x stiffer than the character and are void.
#
# Read the quiet row first. An all-zero action is a stiff PD hold on the rest pose and it stands
# indefinitely, so a policy below 100% there has destroyed a working controller rather than built
# one - which is exactly what the first two training runs did.
param(
    # 'perturb' or 'walk'. Overrides $Task in config.ps1 for this command only.
    [ValidateSet('perturb', 'walk')]
    [string] $Task,
    # Pin a run directory under logs/mujoco. Default: newest with >= $MinIterations checkpoints.
    [string] $Run = "",
    # Score one specific checkpoint instead of the newest in the run.
    [string] $Checkpoint = "",
    [int] $Envs = 0,
    [double] $Seconds = 0,
    # Score the unaided plant instead of a policy - the reference every result is read against.
    [switch] $BaselineOnly
)
# Capture BEFORE dot-sourcing. config.ps1 defines $Envs and $Seconds itself (they are the TRAINING
# scale), so reading these parameters afterwards returns the config values and every override is
# silently ignored - `-Envs 4` became 8,192 and tried to allocate 8,192 MjData on the heap.
$TaskOverride = $Task
$EnvsArg      = $Envs
$SecondsArg   = $Seconds
. "$PSScriptRoot/config.ps1"

if ($EnvsArg -gt 0)    { $EvalEnvs = $EnvsArg }
if ($SecondsArg -gt 0) { $EvalSeconds = $SecondsArg }

# The walk scorer pins each manoeuvre itself and has no projectile, so it takes no ball options.
$Scorer = if ($Task -eq "walk") { "eval_walk.py" } else { "eval.py" }
$cmd = @('--num_envs', $EvalEnvs, '--seconds', $EvalSeconds)
if ($Task -ne "walk") { $cmd += @('--ball_every', $BallEvery[0], $BallEvery[1]) }

if ($BaselineOnly) {
    Write-Host "[eval] $Task : unaided plant on CPU MuJoCo, $EvalEnvs envs x $EvalSeconds s" -ForegroundColor Cyan
    Invoke-Mujoco -Script $Scorer -Arguments ($cmd + '--baseline_only')
    return
}

if (-not $Checkpoint) {
    $pinned = if ($Run) { $Run } else { $RunName }
    $dir = Get-MujocoRun -Minimum $MinIterations -Pinned $pinned
    if (-not $dir) {
        Write-Host "[eval] nothing trained yet under $LogRoot (need >= $MinIterations checkpoints)." -ForegroundColor Yellow
        Write-Host "       Run .\train.ps1 first, or .\eval.ps1 -BaselineOnly to see the reference." -ForegroundColor Yellow
        return
    }
    $Checkpoint = Get-MujocoCheckpoint -RunDir $dir.FullName
    Write-Host "[eval] run $($dir.Name)" -ForegroundColor DarkGray
}

Write-Host "[eval] $(Split-Path $Checkpoint -Leaf) on CPU MuJoCo, $EvalEnvs envs x $EvalSeconds s" -ForegroundColor Cyan
Write-Host "[eval] reference: doing nothing scores 6.4% upright and falls at 2.5 s" -ForegroundColor DarkGray

# The ball speed the checkpoint actually reached travels with the weights, so eval.py defaults to
# it. A policy scored at a difficulty it never trained at is not a measurement of that policy.
Invoke-Mujoco -Script $Scorer -Arguments ($cmd + @('--checkpoint', $Checkpoint))

# Continues training from an existing checkpoint, chosen from a menu instead of typed as a path.
#
# Ported from the Jolt track's rl/scripts/resume.ps1, which exists because checkpoint paths are long
# and near-identical, and picking the wrong one by eye is exactly the kind of mistake that costs a
# training session before anyone notices. The mechanism that makes it work is not the menu - it is
# that each run is described from ITS OWN params/env.yaml and params/agent.yaml rather than guessed
# from the directory name.
#
# WHERE THE LOGS GO is decided by $ExperimentName / -Experiment, NOT by the checkpoint you pick.
# The picked checkpoint decides what the networks START from; the experiment decides which lineage
# the new run is recorded as. They are independent on purpose - that is how you fork a variant off a
# good checkpoint without contaminating the tree it came from.
#
# Usage:
#   .\resume.ps1                          # menu, newest run of every compatible lineage
#   .\resume.ps1 -Task perturb            # menu, ranked for perturb
#   .\resume.ps1 -PerRun 3                # show the top 3 checkpoints of each run, not just one
#   .\resume.ps1 -All                     # every run of every lineage, not one per lineage
#   .\resume.ps1 logs\rsl_rl\...\model_6272.pt    # explicit path, no menu
param(
    [Parameter(Position = 0)][string] $Checkpoint,
    # Which task the NEW run trains. Decides which checkpoints are marked compatible.
    [ValidateSet('stand', 'perturb', 'walk', 'run')]
    [string] $Task,
    # Which logs/rsl_rl subtree the NEW run writes into. Not where the checkpoint comes from.
    [string] $Experiment,
    [double] $Minutes = -1,
    [int] $Envs = 0,
    # Checkpoints listed per run, highest iteration first.
    [int] $PerRun = 1,
    # List every run in each lineage rather than only its newest.
    [switch] $All,
    # After training, measure the new checkpoint against the one currently in Godot and offer to
    # promote it. Opt-in on purpose - see the block at the end of this file.
    [switch] $Promote,
    # XPBD solver iterations for this run only. **Part of the trained dynamics.** Continuing a
    # lineage trained at 8 with config.ps1 still saying 2 would silently change the body underneath
    # the policy - the same class of bug as action_scale reverting on resume. 0 keeps the config.
    [int] $Solver = 0,
    # Override env-cfg fields, e.g. -Set balance_reaction=True,balance_max_torque=300.
    #
    # NEEDED because the carry-forward below only restores action_scale and action_rate_limit from
    # the source run. balance_reaction and balance_max_torque are equally part of the plant, and a
    # resume that drops them trains a DIFFERENT body than the checkpoint came from - silently. The
    # proper fix is to widen the carry-forward to every trained condition, as play.py already does;
    # until then pass them here explicitly.
    [string[]] $Set = @()
)
# Captured BEFORE dot-sourcing: config.ps1 defines $Envs, $MaxMinutes, $Task and $Experiment itself,
# so reading the parameters afterwards returns the config values and every override is lost.
$EnvsOverride       = $Envs
$MinutesOverride    = $Minutes
$TaskOverride       = $Task
$ExperimentOverride = $Experiment
. "$PSScriptRoot/config.ps1"

if ($MinutesOverride -ge 0) { $MaxMinutes = $MinutesOverride }
if ($EnvsOverride -gt 0)    { $Envs = $EnvsOverride }

# ---------------------------------------------------------------- selection

if (-not $Checkpoint) {
    $picked = Select-Isaac3Checkpoint -TaskName $Task -Interactive
    if (-not $picked) { exit 0 }
    $Checkpoint  = $picked.File
    $sameTask    = $picked.SameTask
    $sourceScale = $picked.ActionScale
    $sourceLimit = $picked.RateLimit
} else {
    # An explicit path still gets described, so the mode below is chosen on what the checkpoint IS
    # rather than on an assumption about why it was passed.
    $info = Get-Isaac3RunInfo -RunDir (Split-Path -Parent $Checkpoint)
    $sameTask    = ($info.Task -eq $Task)
    $sourceScale = $info.ActionScale
    $sourceLimit = $info.RateLimit
    Write-Host "[resume] checkpoint trained [$($info.Task)], action_scale $($info.ActionScale), std $($info.StdType)" -ForegroundColor DarkGray
}

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    Write-Host "Checkpoint not found: $Checkpoint" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- mode

# Two modes, and picking the wrong one is quiet rather than fatal, so it is decided here instead of
# being left to a flag someone has to remember.
#
#   same task -> --resume    weights AND optimizer AND iteration count. A genuine continuation:
#                            restarting instead re-pays the whole Adam warm-up, not just the steps.
#   new task  -> --init_from weights only. The optimizer state belongs to a different reward scale,
#                            and carried across it shrinks the effective learning rate for exactly
#                            as long as it takes to look like a plateau rather than a bug. It also
#                            re-inflates the exploration noise, which a converged checkpoint has
#                            collapsed (measured 0.14 against a configured 0.4).
if ($sameTask) {
    $modeArgs = @('--resume', $Checkpoint)
    $mode = "resume (weights + optimizer + iteration)"
} else {
    $modeArgs = @('--init_from', $Checkpoint)
    $mode = "init_from (weights only, optimizer dropped, exploration noise reset)"
}

if ($Solver -gt 0) { $SolverIterations = $Solver }
$env:P4F_XPBD_ITERATIONS = "$SolverIterations"

$cmd = @("$PSScriptRoot/train.py", '--task', $TaskId, '--num_envs', $Envs,
         '--iterations', $Iterations, '--max_minutes', $MaxMinutes) + $modeArgs
if ($RunSuffix)      { $cmd += @('--run_name', $RunSuffix) }
if ($ExperimentName) { $cmd += @('--experiment', $ExperimentName) }

# **Carry the source run's action_scale and action_rate_limit forward, or the continuation trains a
# DIFFERENT controller than the checkpoint it came from.**
#
# Both come from the task registry defaults otherwise, and the default is not what a tuned lineage
# used: `action_scale` reads `P4F_ACTION_SCALE` with a fallback of 0.4, so a lineage trained at 0.15
# through that environment variable silently reverts to 0.4 on resume - every action rescaled 2.7x,
# no error, and the run looks normal until the policy is measured in Godot.
#
# This is the exact failure the `scale` column in the menu warns about, and this script caused it
# once before the guard existed. `-1` is train.py's "keep the task default" sentinel, so a run whose
# params could not be read falls back rather than asserting a wrong number.
#
# **Parsed against InvariantCulture, and passed on as the ORIGINAL STRING.** Both halves matter on a
# non-English locale. `[double]::TryParse("0.15")` under es-ES reads the dot as a thousands
# separator and yields 15 - a hundredfold error that TryParse reports as SUCCESS. Handing the parsed
# double back to the command line then re-formats it as "0,15", which Python's float() rejects. So
# the number is parsed only to validate it, and the untouched YAML text is what gets passed.
function Test-InvariantNumber {
    param([AllowNull()][string] $Text, [ref] $Value)
    return [double]::TryParse($Text, [Globalization.NumberStyles]::Float,
                              [cultureinfo]::InvariantCulture, $Value)
}

$scale = 0.0; $limit = -1.0
if ((Test-InvariantNumber $sourceScale ([ref] $scale)) -and $scale -gt 0.0) {
    $cmd += @('--action_scale', $sourceScale)
}
if ((Test-InvariantNumber $sourceLimit ([ref] $limit)) -and $limit -ge 0.0) {
    $cmd += @('--action_rate_limit', $sourceLimit)
}
foreach ($pair in $Set) { $cmd += @('--set', $pair) }

$budget = if ($MaxMinutes -gt 0) { "$MaxMinutes min" } else { "no time cap" }
Write-Host ""
Write-Host "[resume] from  $Checkpoint" -ForegroundColor Cyan
Write-Host "[resume] mode  $mode" -ForegroundColor Cyan
Write-Host "[resume] into  $Experiment : $Task, $Envs envs, $budget, XPBD iterations $SolverIterations" -ForegroundColor Cyan
Write-Host "[resume] carry action_scale $sourceScale, action_rate_limit $sourceLimit (from the source run)" -ForegroundColor Cyan

Invoke-Isaac3 -Arguments $cmd

# ---------------------------------------------------------------- promote

# **Opt-in, and measured rather than assumed.** Promoting whatever just finished would be wrong on
# this project: `stand_assist/night06` at 3,087 iterations beats `night10` at 6,272 in Godot, so
# "newer" has already meant "worse" more than once, and an unconditional promote would have
# overwritten the working brain with the worse one without a word.
#
# So this measures both in Godot - highest action authority that still stands, and the push impulse
# survived there - prints them side by side, and asks. compare_godot.py restores `Models/` whatever
# happens, so declining costs nothing but the runtime.
if ($Promote) {
    $run = Get-Isaac3Run -Experiment $Experiment -Minimum 1 -Pinned ""
    if (-not $run) {
        Write-Host "[promote] training wrote no checkpoint to $Experiment - nothing to promote." -ForegroundColor Yellow
        exit 1
    }
    $fresh = Get-Isaac3Checkpoint -RunDir $run.FullName
    $brain = Get-Isaac3Brain $Task
    $current = Get-Isaac3Promoted -TaskName $Task

    Write-Host ""
    Write-Host "[promote] comparing in GODOT - the Isaac score does not decide this." -ForegroundColor Cyan
    $compare = @("$PSScriptRoot/compare_godot.py", '--new', $fresh, '--task', $TaskId, '--brain', $brain)
    if ($current) { $compare += @('--current', $current) }
    Invoke-Isaac3 -Arguments $compare
    $verdict = $LASTEXITCODE   # 0 better, 1 equal, 2 worse - see compare_godot.py

    if (-not $current) {
        Write-Host "[promote] nothing is promoted for '$brain' yet, so there is no baseline to beat." -ForegroundColor Yellow
    }

    $answer = Read-Host "  Promote the new checkpoint into Models/$($brain)_policy.onnx? (y/N)"
    if ($answer -ne "y") {
        Write-Host "[promote] left Models/ unchanged." -ForegroundColor DarkGray
        exit 0
    }

    Invoke-Isaac3 -Arguments @("$PSScriptRoot/export.py", '--task', $TaskId,
                               '--checkpoint', $fresh, '--promote')
    Write-Host "[promote] Models/$($brain)_policy.onnx now holds $($run.Name)/$(Split-Path -Leaf $fresh)" -ForegroundColor Green
}

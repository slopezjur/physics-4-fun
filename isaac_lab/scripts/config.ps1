# ============================================================================
#  EDIT THIS FILE to change how Isaac Lab runs. Every script reads these values.
# ============================================================================

# --- Paths -------------------------------------------------------------------
# The Isaac Lab Python. Deliberately resolved rather than assumed: a bare `python` on this machine
# is a standalone 3.14 that Isaac Sim cannot load at all, and the only symptom is
# "ModuleNotFoundError: No module named 'isaaclab'", which says nothing about the real cause.
#
# Resolution order:
#   1. $env:ISAAC_PYTHON  - set once per machine:
#        setx ISAAC_PYTHON "D:\Programas\anaconda3\envs\env_isaaclab\python.exe"
#      then open a NEW terminal (setx only affects processes started afterwards).
#   2. the conda env this project was set up with
#   3. a scan of the usual conda roots for an env named env_isaaclab
function Resolve-IsaacPython {
    if ($env:ISAAC_PYTHON) {
        if (Test-Path $env:ISAAC_PYTHON) { return (Resolve-Path $env:ISAAC_PYTHON).Path }
        throw "ISAAC_PYTHON is set to '$env:ISAAC_PYTHON' but no file exists there."
    }

    $known = 'D:\Programas\anaconda3\envs\env_isaaclab\python.exe'
    if (Test-Path $known) { return $known }

    foreach ($root in @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3",
                        'C:\ProgramData\anaconda3', 'D:\Programas\anaconda3')) {
        $candidate = Join-Path $root 'envs\env_isaaclab\python.exe'
        if (Test-Path $candidate) { return $candidate }
    }

    throw @"
Could not locate the Isaac Lab Python.

This project needs the conda environment with isaacsim 5.1 and isaaclab installed - the system
Python cannot run Isaac Sim. Point ISAAC_PYTHON at it and re-run:

    setx ISAAC_PYTHON "D:\path\to\anaconda3\envs\env_isaaclab\python.exe"

then open a NEW terminal.
"@
}

$Python = Resolve-IsaacPython

# --- Which task to run -------------------------------------------------------
#   "stand"   - start upright, learn to stay upright. The parent of everything else.
#   "walk"    - follow a velocity command. Bootstrap it FROM stand.
#   "perturb" - stand while something shoves you. Same reward as stand, so bootstrap FROM stand.
#   "run"     - fast command with a flight phase. Bootstrap it FROM walk.
#
# Every script reads this. Change this one line to switch tasks.
#
# Scripts also accept -Task to override it for one command without editing this file, e.g.
#   .\watch.ps1 -Task walk
$Task = "perturb"

# --- Training scale ----------------------------------------------------------
# One knob, unlike the Godot track: there are no processes to spawn, because every environment is
# a slice of one GPU tensor in one process.
#
# MEASURED on this machine (RTX 4080 SUPER 16 GB, 32 GB system RAM), on the PERTURBATION task,
# 12 iterations per row:
#
#    num_envs    steps/s    peak RAM    peak VRAM
#       2,048     52,587     19.1 GB      6.2 GB
#       4,096     78,179     20.7 GB      7.5 GB
#       8,192     86,088     23.6 GB      9.8 GB   <- the plateau, and the default
#      16,384     85,360     29.0 GB     14.5 GB   <- 1% SLOWER for 5.4 GB more RAM
#
# Throughput peaks at 8,192 and then goes backwards. Doubling past it buys nothing and spends the
# headroom that keeps a long run from dying on an allocation.
#
# RAM binds before VRAM, which is not obvious. Fitting the rows: RAM is about 18.7 GB fixed plus
# 0.69 MB per env, VRAM about 5 GB fixed plus 0.58 MB per env. Per-env cost is similar; the fixed
# cost is not - Isaac Sim's process plus the desktop takes 18.7 GB before a single environment
# exists, so system memory hits its ceiling first. At 16,384 the machine sat at 2.6 GB free with
# 31.3 GB committed, i.e. no headroom for anything else at all.
#
# This is PER TASK. Stand measured 136,000 steps/s at 16,384 because it applies a wrench to one
# body and has no contact bookkeeping; Perturbation applies one across all 46 and draws markers,
# and is roughly 35% slower everywhere. Re-measure before assuming a number transfers between
# tasks - a table from the wrong task is what set this to 16,384 in the first place.
#
# Any integer works, not just powers of two: rsl_rl splits the rollout with integer division, so
# num_envs * num_steps_per_env must divide by num_mini_batches, and 24 / 4 already does for every
# env count. 12,288 or 10,000 are equally valid - they just sit on the same plateau.
$Envs = 8192

# 0 = use the task's own max_iterations (see p4f_isaac/tasks/<task>/agents/rsl_rl_ppo_cfg.py).
$Iterations = 0

# Wall-clock cap in minutes. Training stops at whichever comes first, this or $Iterations.
#
# Use this when you want a run of a known DURATION. An iteration is roughly 1.0-1.5 s at 4,096
# envs, but that moves with the environment count, the task, and anything else using the GPU - so
# "thirty minutes" is not reliably expressible as an iteration count.
#
#   0  = no time limit
#   30 = half an hour
#  120 = two hours
#
# The run stops on a checkpoint boundary rather than being killed, so it always ends resumable.
$MaxMinutes = 30

# Bootstrap this run's weights from another task's newest checkpoint, e.g. "stand".
# Empty = train from scratch.
#
# Only valid while the two tasks share an observation and action layout - all four here do, which
# is what obs_action_contract.md's reserved command slots exist to guarantee. Weights only: the
# optimiser state belongs to the run that produced it, and carrying an adaptive-KL optimiser across
# a change of reward function makes a bootstrap worse than a cold start.
$BootstrapFrom = "perturb"

# --- Watching (the arena) ----------------------------------------------------
# Environments to render. The training default would draw an unusable grid.
$WatchEnvs = 4

# Force the Windows-default D3D12 renderer instead of Vulkan.
#
# LEAVE THIS ON unless you have changed graphics driver. Isaac Lab's apps/isaaclab.python.kit sets
# `vulkan = true`, overriding the Windows default, and Isaac Sim 5.1 crashes on the Vulkan path
# with this driver (610.88) - an access violation on the first rendered frame, no Python traceback.
# NVIDIA have acknowledged it as a Vulkan regression on newer drivers. Headless never renders, so
# training was unaffected and this only ever mattered the first time something was watched.
$ForceD3D12 = $true

# --- Evaluation --------------------------------------------------------------
# Environments to score against. More is a tighter measurement and nothing else.
$EvalEnvs = 256

# Commanded forward speed for walk / run evaluation, m/s. Ignored by stand and perturb.
$EvalSpeed = 0.6

# --- Run selection -----------------------------------------------------------
# Scripts pick the newest run for $Task that reached at least this many iterations.
#
# "Newest run" alone is wrong, for the reason rl/scripts/summary.ps1 already records: a throwaway
# smoke run finishing later than the real one makes every script silently use it, and the numbers
# look plausible. A three-iteration test leaves model_2.pt behind and wins on timestamp. 50 is
# rsl_rl's save_interval, so anything that trained at all clears it.
$MinIterations = 50

# Pin a specific run directory instead of taking the newest, e.g. "2026-08-26_10-25-46_limbs".
# Empty = newest that clears $MinIterations.
$RunName = ""

# ============================================================================
#  Derived - normally no need to edit below here.
# ============================================================================
$IsaacRoot = (Resolve-Path "$PSScriptRoot/..").Path
$ProjectRoot = (Resolve-Path "$PSScriptRoot/../..").Path

# A script may set $TaskOverride from its own -Task parameter before dot-sourcing this file.
if ($TaskOverride) { $Task = $TaskOverride }

if ($Task -eq "stand") {
    $TaskId = "P4F-Dummy-Stand-Direct-v0"; $Experiment = "p4f_stand";   $Evaluator = "evaluate_stand.py"
} elseif ($Task -eq "walk") {
    $TaskId = "P4F-Dummy-Walk-Direct-v0";  $Experiment = "p4f_walk";    $Evaluator = "evaluate_walk.py"
} elseif ($Task -eq "perturb") {
    $TaskId = "P4F-Dummy-Perturb-Direct-v0"; $Experiment = "p4f_perturb"; $Evaluator = "evaluate_perturb.py"
} elseif ($Task -eq "run") {
    # Scored with the walk evaluator on purpose: it measures gait - steps/s, air time, and the
    # flight fraction that separates a run from a fast walk - which is exactly the question.
    $TaskId = "P4F-Dummy-Run-Direct-v0";   $Experiment = "p4f_run";     $Evaluator = "evaluate_walk.py"
} else {
    throw "Unknown `$Task '$Task'. Use 'stand', 'walk', 'perturb' or 'run'."
}

if ($Envs -lt 1)          { throw "`$Envs must be at least 1 (got $Envs)." }
if ($WatchEnvs -lt 1)     { throw "`$WatchEnvs must be at least 1 (got $WatchEnvs)." }
# Bootstrapping a task from ITSELF is legitimate and deliberately allowed: it continues from that
# task's newest checkpoint under changed settings - a harder curriculum, a retuned reward - which is
# the cheapest way to raise difficulty without discarding what the policy already learned. It is
# still weights-only, so the optimiser does not carry across the change.

$LogRoot   = Join-Path $IsaacRoot "logs\rsl_rl"
$Exported  = Join-Path $IsaacRoot "exported"
$KitArgs   = if ($ForceD3D12) { '--kit_args=--/app/vulkan=false' } else { $null }

# Newest run for an experiment that actually trained. Returns $null rather than throwing so
# callers can report "not trained yet" in their own words.
function Get-IsaacRun {
    param(
        [Parameter(Mandatory = $true)][string] $Experiment,
        [int] $Minimum = 50,
        [string] $Pinned = ""
    )
    $root = Join-Path $LogRoot $Experiment
    if (-not (Test-Path -LiteralPath $root)) { return $null }

    if ($Pinned) {
        $dir = Join-Path $root $Pinned
        if (-not (Test-Path -LiteralPath $dir)) { throw "No run '$Pinned' under $root." }
        return (Get-Item -LiteralPath $dir)
    }

    foreach ($dir in (Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime -Descending)) {
        $ckpts = Get-ChildItem -LiteralPath $dir.FullName -Filter 'model_*.pt' -ErrorAction SilentlyContinue
        if (-not $ckpts) { continue }
        $best = ($ckpts | ForEach-Object { [int]($_.BaseName -replace 'model_', '') } |
                 Measure-Object -Maximum).Maximum
        if ($best -ge $Minimum) { return $dir }
    }
    return $null
}

# Highest-numbered checkpoint in a run - not the newest file, because exported/ is written after
# the last checkpoint and would win on timestamp.
function Get-IsaacCheckpoint {
    param([Parameter(Mandatory = $true)][string] $RunDir)
    $ckpt = Get-ChildItem -LiteralPath $RunDir -Filter 'model_*.pt' |
            Sort-Object { [int]($_.BaseName -replace 'model_', '') } -Descending |
            Select-Object -First 1
    if (-not $ckpt) { throw "No checkpoints in $RunDir." }
    return $ckpt.FullName
}

# Run a Python entry point from the isaac_lab root, not from scripts/.
#
# This is load-bearing. Isaac Lab's train.py and play.py build their log root as the RELATIVE path
# "logs/rsl_rl/<experiment>", so the working directory decides where checkpoints are written and
# looked for. Invoked from scripts/, play.py went looking in isaac_lab/scripts/logs/... and failed
# with a bare FileNotFoundError naming a path nobody had ever created.
function Invoke-Isaac {
    param([Parameter(Mandatory = $true)][string[]] $Arguments)
    Push-Location $IsaacRoot
    try {
        & $Python @Arguments
    } finally {
        Pop-Location
    }
}

Write-Host "[config] Task '$Task' -> $TaskId (experiment '$Experiment')." -ForegroundColor DarkGray

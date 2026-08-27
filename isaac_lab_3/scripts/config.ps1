# ============================================================================
#  EDIT THIS FILE to change how the Newton/XPBD track runs. Every script reads it.
#
#  Deliberately the same shape as isaac_lab/scripts/config.ps1, so the workflow carries over.
#  What differs is forced by the backend, and is called out where it appears:
#    - a different interpreter (env_isaaclab3, Python 3.12)
#    - no Kit, so no $ForceD3D12 / --kit_args; $Viewer replaces them
#    - $SolverIterations, which is part of the trained dynamics
# ============================================================================

# --- Paths -------------------------------------------------------------------
# The Isaac Lab 3 Python. Resolved rather than assumed, for the same reason the 2.3.2 config does
# it: a bare `python` on this machine is a standalone 3.14, and the only symptom is
# "ModuleNotFoundError: No module named 'isaaclab'", which says nothing about the real cause.
#
# NOTE this is a DIFFERENT environment from the 2.3.2 track's `env_isaaclab` (Python 3.11, Isaac
# Sim 5.1). Both must keep working; do not point one at the other.
function Resolve-Isaac3Python {
    if ($env:ISAAC3_PYTHON) {
        if (Test-Path $env:ISAAC3_PYTHON) { return (Resolve-Path $env:ISAAC3_PYTHON).Path }
        throw "ISAAC3_PYTHON is set to '$env:ISAAC3_PYTHON' but no file exists there."
    }

    $known = 'D:\Programas\anaconda3\envs\env_isaaclab3\python.exe'
    if (Test-Path $known) { return $known }

    foreach ($root in @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3",
                        'C:\ProgramData\anaconda3', 'D:\Programas\anaconda3')) {
        $candidate = Join-Path $root 'envs\env_isaaclab3\python.exe'
        if (Test-Path $candidate) { return $candidate }
    }

    throw @"
Could not locate the Isaac Lab 3 Python.

This track needs the conda environment with isaaclab 6.x, isaaclab_newton and newton installed.
It is NOT the same environment as the 2.3.2 track. Point ISAAC3_PYTHON at it and re-run:

    setx ISAAC3_PYTHON "D:\path\to\anaconda3\envs\env_isaaclab3\python.exe"

then open a NEW terminal.
"@
}

$Python = Resolve-Isaac3Python

# --- Which task to run -------------------------------------------------------
#   "stand"   - start upright, learn to stay upright. The parent of everything else.
#   "perturb" - Stand, with something shoving back. Inherits StandEnv, so a Stand checkpoint is a
#               valid bootstrap: same observation and action layout.
#   "walk"    - Stand plus a velocity command. Deletes five of Stand's reward terms, adds four.
#   "run"     - Walk at a speed the rig cannot reach by walking, so the gait has to change.
#
# All four are registered by p4f_newton/tasks/__init__.py. See $TaskMap below for the ids.
#
# Scripts also accept -Task to override this for one command without editing the file.
$Task = "stand"

# --- Training scale ----------------------------------------------------------
# MEASURED on this machine (RTX 4080 SUPER 16 GB, 32 GB system), Stand under XPBD at 2 solver
# iterations, via `scripts/benchmark.py --envs ... --iterations 25`. Whole sweep in one session on
# an IDLE machine - 10.8 GB RAM and 0.9 GB VRAM baseline before the first row.
#
#   num_envs    steps/s   per env   iters/min   proc RAM    VRAM
#       1024     61,804        60       150.9      1.9 GB   1.6 GB
#       2048    125,381        61       153.1      2.1 GB   1.9 GB   <- last linear point
#       4096    215,165        53       131.3      2.3 GB   2.5 GB
#       8192    361,757        44       110.4      2.8 GB   3.5 GB
#      12288    454,420        37        92.5      3.3 GB   4.6 GB
#      16384    512,821        31        78.3      3.8 GB   5.6 GB   <- what the night chain used
#      24576    566,127        23        57.6      4.8 GB   7.8 GB
#      32768    622,179        19        47.5      6.0 GB   9.9 GB
#
# `iters/min` is DERIVED: steps/s divided by (num_envs * num_steps_per_env), with the rollout at 24.
# It is a collection rate, not an end-to-end training rate - the night chain observed ~29 iters/min
# at 16384 against the 78 here, because a real iteration also pays the PPO update and the
# checkpoint write. Use the column for comparing rows, not for predicting wall-clock.
#
# **MEASURE ON AN IDLE MACHINE.** An earlier sweep of the same rows, taken with background work
# running (14.9 GB RAM idle instead of 10.8), read ~50% LOW at every single count - 334,282 at
# 16384 against 512,821 here. That is far larger than any difference between adjacent rows, so a
# contaminated table does not merely shift: it will point at the wrong env count entirely. Check
# the idle baseline the benchmark prints before trusting a row.
#
# **Scaling is linear to 2048 and degrades steadily after.** Per-env throughput holds at 60-61k
# through 2048, then falls: 53k / 44k / 37k / 31k / 23k / 19k. Past 2048 each added environment
# buys less than the one before it, and by 32768 an environment is worth under a third of what it
# is worth at 1024.
#
# **More envs is a genuine trade-off, not a free win, and this table cannot settle it.** 32768 has
# the best throughput in the table by a wide margin - 622k steps/s against 513k at 16384 - and the
# worst update rate, 47.5/min against 78.3. Which matters depends on something nothing here
# measures: a bigger batch gives each PPO update a lower-variance gradient, so fewer-but-better
# updates may beat more-but-noisier ones, or may not. The only configuration with a DEMONSTRATED
# result on this project is 16384, which is what the night chain that produced the working brain
# ran at - that is history, not evidence of optimality.
#
# If you want to raise it, the check is a learning one: same task, same wall-clock, two env counts,
# compare where the reward curve gets to. Throughput alone will always favour the largest count
# that fits.
#
# **Trust `proc RAM`; `sys RAM` and `VRAM` include the desktop.** Process RAM is reproducible across
# sessions at the same env count (3.8 GB at 16384 in both the clean and contaminated sweeps) while
# the system totals moved by 4-7 GB with background load. The 2.3.2 track's ceiling was system RAM,
# and an env count that fits at startup can still die on an allocation hours into an unattended
# run - 32768 commits 6.0 GB of process RAM and 9.9 GB of VRAM, so leave headroom.
#
# For comparison the 2.3.2 PhysX track measured 110,000 at 4,096 on the URDF rig and about 204,000
# at 8,192 on the D6 rig. XPBD on the D6 rig is roughly 2x both (215k and 362k), which is the
# opposite of what an iterative position-based solver is usually assumed to cost - though those
# PhysX figures came from different solver settings and were not re-taken idle.
#
# **Every row above was taken at $SolverIterations = 2, with balance_reaction OFF.** Both matter.
# Solver iterations are per-tick constraint work, so 8 costs materially more than 2 and this whole
# table would need re-taking before it could advise an env count for an 8-iteration run - which the
# honest-physics line (reaction ON at full 300 N.m) requires to be stable at all. Do not carry these
# numbers across that boundary.
#
# Conditions in full, so a future row can be compared like for like:
#   task Stand, XPBD iterations 2, action_scale 0.4 (task default), balance_assist 0.0,
#   balance_reaction off, num_steps_per_env 24, episode length ~44 steps, machine idle.
#
# Re-measure before assuming this transfers to another task: the 2.3.2 config records that a table
# taken from the wrong task is exactly how its own env count got set wrong once.
$Envs = 24576

# 0 = use the task's own max_iterations (p4f_newton/tasks/<task>/agents/rsl_rl_ppo_cfg.py).
$Iterations = 0

# Wall-clock cap in minutes. Training stops at whichever comes first, this or $Iterations.
#
# At 4,096 envs and ~222k steps/s, 15 minutes is roughly 200M steps / 2,000 iterations - which is
# the same scale as the 2.3.2 Stand run that reached 100% strict success in 39 minutes. So a short
# run here is a real result, not a smoke test.
#
# The run stops on a checkpoint boundary rather than being killed, so it always ends resumable.
$MaxMinutes = 1

# Suffix appended to the run directory, e.g. "2026-08-27_00-15-02_baseline". Empty = timestamp only.
$RunSuffix = ""

# Continue from another run's newest checkpoint. Empty = train from scratch.
# Weights and optimiser state both, unlike the 2.3.2 track's cross-task bootstrap.
$ResumeFrom = ""

# --- Physics -----------------------------------------------------------------
# XPBD solver iterations. **This is part of the dynamics a policy is trained against**, exactly as
# Jolt's position/velocity steps are on the Godot side - see docs/RL-SESSION-INVARIANTS.md
# invalidator #3, which is about precisely this parameter in the other engine.
#
# Measured on the zero-action hold: 2 / 8 / 16 hold the joints roughly 25x tighter as they rise and
# the body degrades far more gracefully - and the rest pose is unstable at every one of them, 0%
# standing at 8 s. Same shape as the Jolt 2/10 -> 16/30 measurement, same conclusion.
#
# Change this only at a retrain boundary, never between training and playback. A policy replayed
# against different iterations is driving a body it never saw.
$SolverIterations = 2

# --- Watching ----------------------------------------------------------------
# Environments to render. The training default would draw an unusable grid.
$WatchEnvs = 4

# Which viewer backend to open.
#
#   "newton" - a native OpenGL window (pyglet + imgui). The default.
#   "viser"  - serves a viewer in the browser; useful over a remote session.
#   "none"   - headless, for scripted playback that only needs numbers.
#
# There is no Kit option and no $ForceD3D12 here. That whole class of problem is gone: the 2.3.2
# track had to pass `--kit_args=--/app/vulkan=false` because Isaac Lab forced Vulkan on and Isaac
# Sim 5.1 crashed on it with this driver. This track never starts Kit, so it never renders through
# Vulkan and there is no shader cache to build on first launch either.
$Viewer = "newton"

# --- Run selection -----------------------------------------------------------
# Scripts pick the newest run for $Task that reached at least this many iterations.
#
# "Newest run" alone is wrong: a throwaway smoke run finishing later than the real one makes every
# script silently use it, and the numbers look plausible. A three-iteration test leaves model_2.pt
# behind and wins on timestamp. 50 is rsl_rl's save_interval, so anything that trained clears it.
$MinIterations = 50

# Pin a specific run directory instead of taking the newest. Empty = newest that clears the bar.
$RunName = ""

# Which logs/rsl_rl subtree to read and write. Empty = the task's default from $TaskMap.
#
# **Needed because the default is usually NOT where the good checkpoints are.** `train.py
# --experiment` exists so a variant can be kept off a chained run's checkpoint path - night.py
# resumes from the newest checkpoint in its experiment tree, so two runs sharing a name silently
# adopt each other's weights. The consequence is that Stand's best work lives in
# p4f_newton_stand_assist / _trim / _muscle / _v3, and a script pointed at the bare
# "p4f_newton_stand" tree finds an old run and reports it as the newest.
#
# Get-Isaac3Experiments lists what exists for the current task, and the "not trained yet" errors
# name them, so this does not have to be guessed.
#
# Scripts also accept -Experiment to override it for one command.
$ExperimentName = ""

# ============================================================================
#  Derived - normally no need to edit below here.
# ============================================================================
$Isaac3Root  = (Resolve-Path "$PSScriptRoot/..").Path
$ProjectRoot = (Resolve-Path "$PSScriptRoot/../..").Path

# A script may set $TaskOverride / $ExperimentOverride from its own parameters before dot-sourcing.
if ($TaskOverride)       { $Task = $TaskOverride }
if ($ExperimentOverride) { $ExperimentName = $ExperimentOverride }

# The single place a task name becomes a gym id and a log subtree. Keep it in step with
# p4f_newton/tasks/*/__init__.py (the id) and */agents/rsl_rl_ppo_cfg.py (the experiment_name) -
# a mismatch here does not fail loudly, it trains the right task into the wrong log tree.
$TaskMap = [ordered] @{
    stand   = @{ Id = "P4F-Dummy-Stand-Newton-v0";   Experiment = "p4f_newton_stand"   }
    perturb = @{ Id = "P4F-Dummy-Perturb-Newton-v0"; Experiment = "p4f_newton_perturb" }
    walk    = @{ Id = "P4F-Dummy-Walk-Newton-v0";    Experiment = "p4f_newton_walk"    }
    run     = @{ Id = "P4F-Dummy-Run-Newton-v0";     Experiment = "p4f_newton_run"     }
}

if (-not $TaskMap.Contains($Task)) {
    throw "Unknown `$Task '$Task'. Known tasks: $(($TaskMap.Keys) -join ', ')."
}
$TaskId     = $TaskMap[$Task].Id
$Experiment = if ($ExperimentName) { $ExperimentName } else { $TaskMap[$Task].Experiment }

if ($Envs -lt 1)      { throw "`$Envs must be at least 1 (got $Envs)." }
if ($WatchEnvs -lt 1) { throw "`$WatchEnvs must be at least 1 (got $WatchEnvs)." }

$LogRoot  = Join-Path $Isaac3Root "logs\rsl_rl"
$Exported = Join-Path $Isaac3Root "exported"

# Every experiment subtree that plausibly belongs to a task, newest-trained first: the task's own
# name plus any "<name>_suffix" variant. Exists so a "nothing trained" error can name the trees that
# DO have checkpoints - the variants are where the work actually is, and the default tree being
# empty says nothing about whether the task has ever been trained.
function Get-Isaac3Experiments {
    param([Parameter(Mandatory = $true)][string] $Task)

    $base = $TaskMap[$Task].Experiment
    if (-not (Test-Path -LiteralPath $LogRoot)) { return @() }

    Get-ChildItem -LiteralPath $LogRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $base -or $_.Name -like "${base}_*" } |
        ForEach-Object {
            $run = Get-Isaac3Run -Experiment $_.Name -Minimum $MinIterations
            if ($run) { [pscustomobject] @{ Name = $_.Name; Run = $run.Name; When = $run.LastWriteTime } }
        } |
        Sort-Object When -Descending
}

# Newest run for an experiment that actually trained. Returns $null rather than throwing so callers
# can report "not trained yet" in their own words.
function Get-Isaac3Run {
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
function Get-Isaac3Checkpoint {
    param([Parameter(Mandatory = $true)][string] $RunDir)
    $ckpt = Get-ChildItem -LiteralPath $RunDir -Filter 'model_*.pt' |
            Sort-Object { [int]($_.BaseName -replace 'model_', '') } -Descending |
            Select-Object -First 1
    if (-not $ckpt) { throw "No checkpoints in $RunDir." }
    return $ckpt.FullName
}

# Resolve $Task / $Experiment to a run directory, or throw an error that says what IS available.
#
# Shared rather than written per script because the failure it reports is the confusing one: the
# default experiment tree for a task is routinely empty while several variants of it are full, so a
# bare "no trained run" sends you off to retrain something that already exists.
function Resolve-Isaac3Run {
    $run = Get-Isaac3Run -Experiment $Experiment -Minimum $MinIterations -Pinned $RunName
    if ($run) { return $run }

    $available = Get-Isaac3Experiments -Task $Task
    $message = "No run for '$Task' in experiment '$Experiment' with at least $MinIterations iterations."
    if ($available) {
        $message += "`n`nThese HAVE trained runs - pass -Experiment <name>, or set `$ExperimentName in config.ps1:`n"
        foreach ($item in $available) {
            $message += "`n    {0,-32} newest: {1}" -f $item.Name, $item.Run
        }
    } else {
        $message += " Nothing under $LogRoot matches this task at all - train it with .\train.ps1."
    }
    throw $message
}

# Run a Python entry point from the isaac_lab_3 root, not from scripts/.
#
# Load-bearing: train.py builds its log root as the relative path "logs/rsl_rl/<experiment>", so
# the working directory decides where checkpoints are written and looked for. Invoked from
# scripts/, playback went looking in isaac_lab_3/scripts/logs/... and failed with a bare
# FileNotFoundError naming a path nobody had ever created.
function Invoke-Isaac3 {
    param([Parameter(Mandatory = $true)][string[]] $Arguments)
    Push-Location $Isaac3Root
    try {
        & $Python @Arguments
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------- run metadata

# One scalar out of a machine-written YAML file, by regex rather than a parser.
#
# Windows PowerShell 5.1 has no YAML support and these files are written by `dump_yaml`, not by
# hand, so the layout is stable. -TopLevel anchors at column 0, which matters: `action_scale` is a
# root key, while `std_type` is nested under the distribution config.
function Read-YamlScalar {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string] $Text,
        [Parameter(Mandatory = $true)][string] $Key,
        [switch] $TopLevel
    )
    # `\r` in the trailing class is load-bearing. These files reach memory with CRLF line endings,
    # and .NET's multiline `$` matches only before the `\n` - so a class of just [ \t] cannot cross
    # the `\r` and every lookup silently returns null. It fails as "no such key", not as an error.
    $indent = if ($TopLevel) { "" } else { "\s*" }
    if ($Text -match "(?m)^$indent$([regex]::Escape($Key)):[ \t]*(\S+)[ \t\r]*$") { return $matches[1] }
    return $null
}

# What a run actually WAS, read from its own params/ rather than inferred from where it sits.
#
# There is no task-id field in env.yaml, so the task is identified by marker keys that only one
# task's config contributes. Order is load-bearing: Run inherits Walk's config and therefore carries
# `cmd_lin_vel_x` too, so Run has to be tested first or every Run run reports as Walk.
function Get-Isaac3RunInfo {
    param([Parameter(Mandatory = $true)][string] $RunDir)

    $info = [pscustomobject]@{
        Task = "?"; Brain = "?"; ActionScale = "?"; RateLimit = "?"; StdType = "?"
        ObsDim = "?"; ActDim = "?"; NumEnvs = 0; StepsPerEnv = 0
    }

    $envYaml = Join-Path $RunDir "params\env.yaml"
    if (Test-Path -LiteralPath $envYaml) {
        $text = Get-Content -LiteralPath $envYaml -Raw
        if     ($text -match '(?m)^rew_double_support:') { $info.Task = "run"     }
        elseif ($text -match '(?m)^cmd_lin_vel_x:')      { $info.Task = "walk"    }
        elseif ($text -match '(?m)^push_impulse_range:') { $info.Task = "perturb" }
        else                                             { $info.Task = "stand"   }

        foreach ($pair in @(@("ActionScale", "action_scale"), @("RateLimit", "action_rate_limit"),
                            @("ObsDim", "observation_space"), @("ActDim", "action_space"))) {
            $value = Read-YamlScalar -Text $text -Key $pair[1] -TopLevel
            if ($value) { $info.($pair[0]) = $value }
        }
        # Nested under `scene:`, so no -TopLevel. This is the env count the run ACTUALLY used,
        # which is not necessarily $Envs in config.ps1 today.
        $value = Read-YamlScalar -Text $text -Key "num_envs"
        if ($value) { $info.NumEnvs = [int64] $value }
    }

    $agentYaml = Join-Path $RunDir "params\agent.yaml"
    if (Test-Path -LiteralPath $agentYaml) {
        $text = Get-Content -LiteralPath $agentYaml -Raw
        $value = Read-YamlScalar -Text $text -Key "std_type"
        if ($value) { $info.StdType = $value }
        $value = Read-YamlScalar -Text $text -Key "num_steps_per_env" -TopLevel
        if ($value) { $info.StepsPerEnv = [int64] $value }
    }

    $info.Brain = Get-Isaac3Brain $info.Task
    return $info
}

# Environment steps behind a checkpoint: one PPO iteration collects a rollout from every env in
# parallel, so `iter * num_envs * num_steps_per_env`.
#
# **This column exists because `iter` alone is not comparable between runs.** An iteration at 16384
# envs is worth twice one at 8196, and nothing about the number says so - two lineages trained at
# different env counts look rankable side by side when they are not. Same failure class as the
# `scale` column: a figure that reads as comparable and is not. Steps is the unit the Jolt track
# always showed, because SB3 step-stamps its checkpoint filenames.
function Get-Isaac3Steps {
    param([int64] $Iteration, [Parameter(Mandatory = $true)] $Info)
    if ($Info.NumEnvs -le 0 -or $Info.StepsPerEnv -le 0) { return -1 }
    return $Iteration * $Info.NumEnvs * $Info.StepsPerEnv
}

# Compact, because the numbers reach billions and a menu column cannot carry 5,485,363,200.
#
# Formatted against InvariantCulture deliberately. `-f` uses the CURRENT culture, so on a Spanish
# locale "2.47B" renders as "2,47B" - which in a column of numbers reads as a thousands separator
# and makes 2.47 billion look like 247.
function Format-Isaac3Steps {
    param([int64] $Steps)
    $invariant = [cultureinfo]::InvariantCulture
    if ($Steps -lt 0)   { return "?" }
    if ($Steps -ge 1e9) { return [string]::Format($invariant, "{0:0.00}B", $Steps / 1e9) }
    if ($Steps -ge 1e6) { return [string]::Format($invariant, "{0:0}M",    $Steps / 1e6) }
    return [string]::Format($invariant, "{0:N0}", $Steps)
}

# The checkpoint currently exported into the Godot project for this task, or $null.
#
# **This is the row that matters and the menu could not previously point at it.** Godot loads a
# FILE - Models/<task>_policy.onnx - and never sees a run directory, so "which brain is in Godot"
# is not answerable from the log tree at all. `export.py --promote` writes the contract beside it
# recording its own `source_checkpoint`, and that is the only link back.
#
# Worth having because the newest checkpoint is routinely NOT the promoted one: `select_by_godot.py`
# ranks by how a policy behaves in Godot rather than by its Isaac score, and it has already chosen a
# 3,087-iteration checkpoint over a 6,272-iteration one from the same chain. A menu that highlights
# only "newest" actively points away from the thing known to work.
function Get-Isaac3Promoted {
    param([Parameter(Mandatory = $true)][string] $TaskName)

    # Keyed by BRAIN, not task: Stand and Perturb export the same artifact, so asking "what is in
    # Godot for perturb" and "for stand" must reach the same file.
    $contract = Join-Path $ProjectRoot "Models\$(Get-Isaac3Brain $TaskName)_policy.contract.json"
    if (-not (Test-Path -LiteralPath $contract)) { return $null }
    try {
        $source = (Get-Content -LiteralPath $contract -Raw | ConvertFrom-Json).source_checkpoint
    } catch {
        return $null   # A malformed contract should cost the marker, not the menu.
    }
    if (-not $source) { return $null }
    return $source
}

# Which BRAIN a task trains. Checkpoints are freely interchangeable inside one of these and moving
# between them is a deliberate act.
#
# Stand and Perturb share Stand's reward dictionary verbatim - Perturb changes nothing but adds a
# disturbance - so a checkpoint moves between them with no reward discontinuity at all. Walk deletes
# five of Stand's terms and adds four, and Run inherits that, so the two families score different
# things. The Jolt track measured what ignoring this costs: standing/all went 0.478 -> 0.170 after
# 33M steps of walk training on a stand policy.
#
# **The return value is also the exported artifact's stem** - `balance_policy.onnx`,
# `locomotion_policy.onnx` - so one map decides both what is compatible and what the file is called.
# **Mirrored by `BRAIN` in scripts/export.py.** Change one and change the other.
function Get-Isaac3Brain {
    param([string] $TaskName)
    if ($TaskName -eq "walk" -or $TaskName -eq "run")      { return "locomotion" }
    if ($TaskName -eq "stand" -or $TaskName -eq "perturb") { return "balance" }
    return "?"
}

# ---------------------------------------------------------------- candidates

function Get-Isaac3Candidates {
    param([Parameter(Mandatory = $true)][string] $WantTask)

    $wantBrain = Get-Isaac3Brain $WantTask
    $promoted = Get-Isaac3Promoted -TaskName $WantTask
    $found = @()

    if (-not (Test-Path -LiteralPath $LogRoot)) { return $found }

    foreach ($tree in (Get-ChildItem -LiteralPath $LogRoot -Directory -ErrorAction SilentlyContinue)) {
        # One entry per LINEAGE by default. A tree is a lineage - that is what $ExperimentName is
        # for - and its newest run is almost always the one worth continuing. Listing every run of
        # every tree buries the good checkpoints: Stand alone has ten trees on this machine.
        $runs = Get-ChildItem -LiteralPath $tree.FullName -Directory -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending
        if (-not $All) { $runs = $runs | Select-Object -First 1 }

        foreach ($run in $runs) {
            $models = Get-ChildItem -LiteralPath $run.FullName -Filter 'model_*.pt' -ErrorAction SilentlyContinue |
                      Sort-Object { [int]($_.BaseName -replace 'model_', '') } -Descending |
                      Select-Object -First ([Math]::Max(1, $PerRun))
            if (-not $models) { continue }

            $info = Get-Isaac3RunInfo -RunDir $run.FullName
            foreach ($model in $models) {
                $iteration = [int]($model.BaseName -replace 'model_', '')
                if ($iteration -lt $MinIterations) { continue }
                $found += [pscustomobject]@{
                    Experiment  = $tree.Name
                    Run         = $run.Name
                    File        = $model.FullName
                    Iteration   = $iteration
                    Steps       = (Get-Isaac3Steps -Iteration $iteration -Info $info)
                    NumEnvs     = $info.NumEnvs
                    When        = $run.LastWriteTime
                    Task        = $info.Task
                    Brain       = $info.Brain
                    ActionScale = $info.ActionScale
                    RateLimit   = $info.RateLimit
                    StdType     = $info.StdType
                    SameBrain   = ($info.Brain -eq $wantBrain)
                    SameTask    = ($info.Task -eq $WantTask)
                    # Compared as normalised full paths: the contract records an absolute path,
                    # while a menu entry is built from wherever $LogRoot resolved.
                    Promoted    = ($promoted -and
                                   ([System.IO.Path]::GetFullPath($promoted) -ieq
                                    [System.IO.Path]::GetFullPath($model.FullName)))
                }
            }
        }
    }

    # **Force the promoted checkpoint into the list.** The walk above keeps one run per lineage and
    # $PerRun checkpoints of it, and the promoted brain routinely satisfies neither: the current
    # Stand one is model_3087 of night06, four runs back in a lineage whose newest is night10. It
    # was therefore absent from the menu entirely - not merely unlabelled - which is the exact
    # failure this marker exists to prevent.
    if ($promoted -and -not ($found | Where-Object Promoted) -and (Test-Path -LiteralPath $promoted)) {
        $model = Get-Item -LiteralPath $promoted
        $run   = $model.Directory
        $info  = Get-Isaac3RunInfo -RunDir $run.FullName
        $iteration = [int]($model.BaseName -replace 'model_', '')
        $found += [pscustomobject]@{
            Experiment  = $run.Parent.Name
            Run         = $run.Name
            File        = $model.FullName
            Iteration   = $iteration
            Steps       = (Get-Isaac3Steps -Iteration $iteration -Info $info)
            NumEnvs     = $info.NumEnvs
            When        = $run.LastWriteTime
            Task        = $info.Task
            Brain       = $info.Brain
            ActionScale = $info.ActionScale
            RateLimit   = $info.RateLimit
            StdType     = $info.StdType
            SameBrain   = ($info.Brain -eq $wantBrain)
            SameTask    = ($info.Task -eq $WantTask)
            Promoted    = $true
        }
    }

    # **The promoted checkpoint sorts first**, so `Enter = 1` continues the brain that is actually
    # running in Godot rather than whichever run finished most recently. Newest is a fact about the
    # filesystem; promoted is a fact about what was measured to work, and only one of those is a
    # sensible default.
    #
    # Then compatible, then newest. Cross-brain options are LISTED, not filtered out: bootstrapping
    # across brains is sometimes deliberate - the Jolt track trained walk from a stand policy on
    # purpose - so hiding them would break a workflow that works.
    return @($found | Sort-Object @{Expression = "Promoted";  Descending = $true},
                                  @{Expression = "SameBrain"; Descending = $true},
                                  @{Expression = "SameTask";  Descending = $true},
                                  @{Expression = "When";      Descending = $true})
}


# ---------------------------------------------------------------- picker

# Present the candidate list and return the chosen one, or $null if the user quits.
#
# **Shared by watch.ps1 and resume.ps1 so the two cannot disagree about "the current brain".** They
# did: watch resolved through Get-Isaac3Run, which searches only the base $Experiment tree, so for
# Stand it opened stand/night13 (action_scale 0.4) while the policy actually running in Godot was
# stand_assist/night06 (0.15). Watching one brain and shipping another is not a display bug.
#
# -Interactive:$false returns row 1 - the promoted checkpoint - WITHOUT rendering or prompting.
# That is watch.ps1's default, and it must bypass Read-Host outright rather than relying on the
# Enter-defaults-to-1 path: Read-Host against a null stdin does not throw, it returns empty and
# takes the default, which is how an unattended 8196-env training run got started once.
function Select-Isaac3Checkpoint {
    param(
        [Parameter(Mandatory = $true)][string] $TaskName,
        [string] $Title = "Resume from which brain?",
        [switch] $Interactive,
        # Non-interactive: take the newest checkpoint of THIS task instead of the promoted one.
        #
        # The two defensible defaults disagree exactly when it matters. Defaulting to the promoted
        # brain answers "what is shipping"; straight after a training run the question is "what did
        # I just make", and promoted-first ordering buries the new run at row 2 where it looks like
        # the training produced nothing at all.
        [switch] $PreferNewest
    )

    $options = Get-Isaac3Candidates -WantTask $TaskName
    if (-not $options) {
        throw "No checkpoints of at least $MinIterations iterations under $LogRoot. Train one with .	rain.ps1."
    }
    if (-not $Interactive) {
        if ($PreferNewest) {
            $newest = $options | Where-Object SameTask | Sort-Object When -Descending | Select-Object -First 1
            if ($newest) { return $newest }
        }
        return $options[0]
    }

    Write-Host ""
    Write-Host "  $Title          task: $TaskName   logs -> $Experiment" -ForegroundColor Cyan
    Write-Host "  ------------------------------------------------------------------------------------------------------"
    # Alignment in a .NET composite format is a SIGNED INTEGER - {2,6} right, {2,-6} left. There is
    # no {2,>6}: it throws FormatError at render time, which killed the header row while every data
    # row below carried on printing. Leading width is 4, matching the data rows' "{0,2}) ".
    Write-Host ("    {0,-26} {1,-22} {2,6} {3,7} {4,6}  {5,-8} {6,6} {7,-7}" -f `
                "lineage", "run", "iter", "steps", "envs", "task", "scale", "std") -ForegroundColor DarkGray
    # Max timestamp among SAME-TASK rows, taken BEFORE the loop - the list is sorted promoted-first,
    # so "the first compatible entry" is no longer the newest one.
    #
    # Same task rather than same brain: Stand and Perturb share a brain, so on a Stand menu the
    # newest Stand-brain run is frequently a PERTURB one, and labelling that "newest" while the user
    # is training Stand points at the wrong row for the second time.
    $newestWhen = ($options | Where-Object SameTask | Sort-Object When -Descending |
                   Select-Object -First 1).When

    for ($i = 0; $i -lt $options.Count; $i++) {
        $o = $options[$i]
        $label = "{0,2}) {1,-26} {2,-22} {3,6} {4,7} {5,6}  {6,-8} {7,6} {8,-7}" -f `
                 ($i + 1), ($o.Experiment -replace '^p4f_newton_', ''),
                 ($o.Run -replace '^\d{4}-\d{2}-\d{2}_', ''), $o.Iteration,
                 (Format-Isaac3Steps $o.Steps), $o.NumEnvs,
                 $o.Task, $o.ActionScale, $o.StdType
        # `newest` is only worth flagging when it is NOT the promoted row - otherwise the menu
        # decorates the same line twice and buries the one label that means something.
        $isNewest = ($newestWhen -and $o.When -eq $newestWhen -and $o.SameTask -and -not $o.Promoted)
        if ($o.Promoted) {
            Write-Host "$label  <- IN GODOT NOW" -ForegroundColor Green
        } elseif (-not $o.SameBrain) {
            Write-Host "$label  different brain" -ForegroundColor DarkYellow
        } elseif ($isNewest) {
            Write-Host "$label  <- newest" -ForegroundColor Cyan
        } else {
            Write-Host $label
        }
    }
    Write-Host "  ------------------------------------------------------------------------------------------------------"
    # `scale` is here because it is the incompatibility that does NOT announce itself. A policy
    # trained at action_scale 0.15 and driven under a config saying 0.4 is a different controller,
    # and nothing errors - the same silent class as the rate-limit bug that scored 25.5% with its
    # own limit applied and 0.0% without. `std` only ever fails loudly.
    Write-Host "  iter = PPO updates; steps = iter x envs x 24 rollout, the only figure comparable ACROSS rows." -ForegroundColor DarkGray
    Write-Host "  scale = action_scale it trained with; a change here silently rescales every action." -ForegroundColor DarkGray

    $choice = Read-Host "  Select 1-$($options.Count) (Enter = 1, q = quit)"
    if ($choice -eq "q") { return $null }
    if (-not $choice) { $choice = "1" }

    $index = 0
    if (-not [int]::TryParse($choice, [ref]$index) -or $index -lt 1 -or $index -gt $options.Count) {
        throw "Not a valid selection: '$choice'"
    }
    $picked = $options[$index - 1]

    if (-not $picked.SameBrain) {
        Write-Host ""
        Write-Host "  WARNING: that checkpoint trained [$($picked.Task)] ($($picked.Brain)) and you are" -ForegroundColor Yellow
        Write-Host "  working on [$TaskName] ($(Get-Isaac3Brain $TaskName)). Those families score different" -ForegroundColor Yellow
        Write-Host "  things - Walk deletes five of Stand's reward terms and adds four. The Jolt track" -ForegroundColor Yellow
        Write-Host "  measured standing/all going 0.478 -> 0.170 after 33M steps of walk training on a" -ForegroundColor Yellow
        Write-Host "  stand policy. Deliberate bootstrapping is fine; an accident is not." -ForegroundColor Yellow
        Write-Host ""
        $confirm = Read-Host "  Continue anyway? (y/N)"
        if ($confirm -ne "y") { return $null }
    }
    return $picked
}


$pinned = if ($ExperimentName) { " [pinned]" } else { "" }
Write-Host "[config] Task '$Task' -> $TaskId (experiment '$Experiment'$pinned), XPBD iterations $SolverIterations." -ForegroundColor DarkGray

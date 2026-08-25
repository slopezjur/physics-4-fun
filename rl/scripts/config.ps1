# ============================================================================
#  EDIT THIS FILE to change how training runs. Every script reads these values.
# ============================================================================

# --- Paths -------------------------------------------------------------------
# Godot executable. Deliberately NOT hardcoded: this file is committed, and an absolute path from
# one machine is meaningless to anyone else cloning the repo.
#
# Resolution order:
#   1. $env:GODOT_EXE  - set once per machine, the reliable answer:
#        setx GODOT_EXE "C:\path\to\Godot_v4.7.1-stable_mono_win64_console.exe"
#      then open a NEW terminal (setx only affects processes started afterwards).
#   2. godot / godot-mono on PATH
#   3. a shallow scan of fixed drives for a Godot*mono*console.exe
#
# Use the *_console.exe* build on Windows. The plain executable detaches from the terminal, so
# export output and errors vanish - which is exactly the output export.ps1 needs to check.
function Resolve-GodotExe {
    if ($env:GODOT_EXE) {
        if (Test-Path $env:GODOT_EXE) { return (Resolve-Path $env:GODOT_EXE).Path }
        throw "GODOT_EXE is set to '$env:GODOT_EXE' but no file exists there."
    }

    foreach ($name in @('godot_console', 'godot-mono', 'godot')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }

    # Depth 1 keeps this to two levels of directory listing per drive - fast, and enough to find
    # both <drive>\Godot_.../ and the common <drive>\<tools folder>\Godot_.../ layout.
    foreach ($drive in [System.IO.DriveInfo]::GetDrives()) {
        if ($drive.DriveType -ne 'Fixed' -or -not $drive.IsReady) { continue }
        $hit = Get-ChildItem -Path $drive.RootDirectory.FullName -Directory -Depth 1 `
                             -Filter 'Godot*' -ErrorAction SilentlyContinue |
               ForEach-Object {
                   Get-ChildItem -Path $_.FullName -Filter 'Godot*console.exe' -ErrorAction SilentlyContinue
               } |
               Sort-Object Name -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }

    throw @"
Could not locate the Godot executable.

This project needs the .NET/Mono build of Godot 4.7.1 or newer - godot_rl_agents' in-engine ONNX
inference does not work on the standard build. Point GODOT_EXE at it and re-run:

    setx GODOT_EXE "C:\path\to\Godot_v4.7.1-stable_mono_win64_console.exe"

then open a NEW terminal.
"@
}

$GodotExe    = Resolve-GodotExe

# --- Which task to train -----------------------------------------------------
#   "stand"        - start upright, learn to stay upright. Carries the reverse-curriculum
#                    machinery, which is why start poses spread slightly below vertical.
#   "perturbation" - a ball gun fires at the dummy on an interval set in the agent scene; learn
#                    to keep balance.
#   "getup"        - start prone, learn to stand up. Owns the reverse curriculum: the floor
#                    starts at 0.99 (almost upright, where a stand policy already succeeds)
#                    and walks back toward flat. RESUME IT FROM A STAND CHECKPOINT.
#   "upright"      - MIXED: stand + getup + perturbation sampled per episode, one brain. This is
#                    the one to train for the Upright group; the three single-task scenes exist
#                    for focused work and for measuring one skill in isolation.
#   "walk"         - start standing, walk forward in a straight line. No perturbation.
# Picks the export preset, the exported binary, and (via the preset's feature tag)
# which scene the build boots. Change this one line to switch tasks.
$Task = "perturbation"

# --- Training scale ----------------------------------------------------------
# Three knobs multiply into one sample rate, and only two of them are worth turning.
#
#   $NParallel - Godot PROCESSES. Each is a full engine: its own socket, render server and
#                startup cost. Scales sample DIVERSITY across seeds.
#   $Dummies   - bodies inside each process, spawned by RagdollSpawner from the scene's
#                *Agent.tscn. Adds simulation without adding engine overhead.
#   $Speedup   - a physics-tick multiplier REQUEST, not an achieved rate.
#
# MEASURED on this 7800X3D (8 cores / 16 threads), headless - full table in
# docs/RL-DESIGN-NOTES.md:
#
#    procs x dummies   speedup   steps/s
#         32 x 1           8       2,162      <- the old ceiling, before $Dummies existed
#         40 x 1           8       2,258
#          8 x 64          1       2,977
#         16 x 16          1       2,988
#         16 x 64          1       4,311      <- the knee
#         16 x 128         1       4,335      <- +0.6% for double the bodies
#         32 x 32          1       4,031
#
# Two results worth reading off that table. First, dummies beat processes: 16 x 64 doubles the
# best process-only figure, because per-process overhead was the ceiling all along. Second,
# $Speedup stops mattering once dummies carry the parallelism - 16 x 16 measured 2,988 at speedup
# 1, 3,104 at 4 and 3,045 at 16, i.e. flat. The CPU is already saturated by simulation, so asking
# for faster ticks cannot produce them. Leave it at 1 unless running --viz, where it sets how fast
# the visible window plays. Simulation is unchanged either way - delta stays 1/120 s.
#
# 16 x 64 x 1 is what stand_v28 trained on for 117M steps at a measured 3,819 steps/s.
$NParallel = 16
$Dummies   = 64
$Speedup   = 1

if ($Dummies -lt 1) {
    throw "`$Dummies must be at least 1 (got $Dummies)."
}
if ($NParallel -lt 1) {
    throw "`$NParallel must be at least 1 (got $NParallel)."
}

# 1M steps is roughly 9 minutes at the sweet spot.
# Pure-RL get-up realistically needs 10-100M.
$Timesteps   = 800000000

# Wall-clock cap in seconds. Training stops at whichever comes first, this or $Timesteps.
# Use this when you want a run of a known DURATION - throughput is not predictable enough to
# express "15 minutes" as a step count, especially with a visible instance.
# 0 = no time limit.
# 300 = 5 minutes.
# 900 = 15 minutes.
# 7200 = 2 hours.
$MaxSeconds = 7200

# --- Run identity ------------------------------------------------------------
# LEAVE EMPTY. The name is derived below as "<task>_v<next unused>" by scanning
# rl/runs, so it can never disagree with $Task and can never go stale.
#
# Hand-maintaining this caused two separate failures in one day. First, $Task was
# switched to a new experiment while the name still read "perturbation_v3", so
# three runs wrote into one directory and mixed two lineages. Second, the name
# was left on an old version, so a resume silently appended to a finished run.
# Both are invisible until you read the event files and wonder why the numbers
# do not line up.
#
# Set it explicitly ONLY to deliberately continue an existing lineage in place -
# for example to append more steps to perturbation_v8 rather than starting v9.
# $ExperimentName = "stand_v28"

# --- Checkpointing -----------------------------------------------------------
# Wall-clock seconds between saves. A crash costs at most this much work.
# Costs ~2.7% throughput (the .onnx export takes ~1.6s). 0 disables.
$SaveEverySeconds = 300

# ============================================================================
#  Derived - normally no need to edit below here.
# ============================================================================
$ProjectPath = (Resolve-Path "$PSScriptRoot/../..").Path

if ($Task -eq "perturbation") {
    $ExportPreset = "Windows Perturbation"
    $BuildName    = "RagdollPerturbationTraining.exe"
    $ScenePath    = "res://Scenes/RL/Upright/RagdollPerturbationTraining.tscn"
} elseif ($Task -eq "getup") {
    # Must match export_presets.cfg EXACTLY. It read "Windows Get Up" against a preset actually
    # named "Windows GetUp", so export.ps1's $FeatureTag lookup returned $null and the getup build
    # would have shipped with no feature tag - booting whatever run/main_scene defaults to rather
    # than the get-up scene, with nothing in the log to say so.
    $ExportPreset = "Windows GetUp"
    $BuildName    = "RagdollGetUpTraining.exe"
    $ScenePath    = "res://Scenes/RL/Upright/RagdollGetUpTraining.tscn"
} elseif ($Task -eq "upright") {
    $ExportPreset = "Windows Upright"
    $BuildName    = "RagdollUprightTraining.exe"
    $ScenePath    = "res://Scenes/RL/Upright/RagdollUprightTraining.tscn"
} elseif ($Task -eq "stand") {
    $ExportPreset = "Windows Stand"
    $BuildName    = "RagdollStandTraining.exe"
    $ScenePath    = "res://Scenes/RL/Upright/RagdollStandTraining.tscn"
} elseif ($Task -eq "walk") {
    $ExportPreset = "Windows Walk"
    $BuildName    = "RagdollWalkTraining.exe"
    $ScenePath    = "res://Scenes/RL/Locomotion/RagdollWalkTraining.tscn"
} else {
    throw "Unknown `$Task '$Task'. Use 'stand', 'getup', 'upright', 'perturbation' or 'walk'."
}

# There is deliberately no "stand_multiple" / "walk_multiple" task. Those existed when running N
# bodies meant a SEPARATE scene with N agent subtrees copied into the file by hand - forty blocks
# of ~16 lines each, regenerated by a script whenever the count changed. $Dummies replaced that:
# every *Training.tscn now carries a RagdollSpawner and takes its count from --dummies=N, so the
# body count is a number rather than a scene.
# Absolute on purpose: every script (training AND tensorboard) must agree on one location.
$ExperimentDir = "$ProjectPath/rl/runs"

# Next unused experiment version for a task, by scanning the run directories.
#
# Matches "<task>_v<N>" with SB3's optional "_<run id>" suffix, so perturbation_v7_0
# yields 7 and the next name is perturbation_v8. A task with no runs yet starts at v1.
function Get-NextExperimentName {
    param(
        [Parameter(Mandatory = $true)][string] $Task,
        [Parameter(Mandatory = $true)][string] $RunsDir
    )

    $highest = 0
    if (Test-Path -LiteralPath $RunsDir) {
        $pattern = "^{0}_v(\d+)(_\d+)?$" -f [regex]::Escape($Task)
        foreach ($dir in (Get-ChildItem -LiteralPath $RunsDir -Directory -ErrorAction SilentlyContinue)) {
            if ($dir.Name -match $pattern) {
                $version = [int] $Matches[1]
                if ($version -gt $highest) { $highest = $version }
            }
        }
    }

    return ("{0}_v{1}" -f $Task, ($highest + 1))
}

if ([string]::IsNullOrWhiteSpace($ExperimentName)) {
    $ExperimentName = Get-NextExperimentName -Task $Task -RunsDir $ExperimentDir
    Write-Host "[config] Task '$Task' -> experiment '$ExperimentName' (next unused)." -ForegroundColor DarkGray
} else {
    Write-Host "[config] Task '$Task' -> experiment '$ExperimentName' (pinned in config.ps1)." -ForegroundColor Yellow
}
$Python      = "$ProjectPath/rl/.venv/Scripts/python.exe"
$TrainScript = "$ProjectPath/rl/train.py"
$BuildExe = $GodotExe

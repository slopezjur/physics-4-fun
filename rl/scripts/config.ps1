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
#   "perturbation" - a ball gun fires at the dummy every 3 s; learn to keep balance.
#   "getup"        - start prone, learn to stand up. Owns the reverse curriculum: the floor
#                    starts at 0.99 (almost upright, where a stand policy already succeeds)
#                    and walks back toward flat. RESUME IT FROM A STAND CHECKPOINT.
#   "upright"      - MIXED: stand + getup + perturbation sampled per episode, one brain. This is
#                    the one to train for the Upright group; the three single-task scenes exist
#                    for focused work and for measuring one skill in isolation.
#   "walk"         - start standing, walk forward in a straight line. No perturbation.
# Picks the export preset, the exported binary, and (via the preset's feature tag)
# which scene the build boots. Change this one line to switch tasks.
$Task        = "perturbation"

# --- Training scale ----------------------------------------------------------
# MEASURED on this 7800X3D (8 cores / 16 threads), headless, 180s per point:
#
#    procs   steps/s   per-proc   gain over previous
#       16      1394       87.1        -
#       24      1840       76.6     +32.0%
#       32      2162       67.6     +17.5%
#       40      2258       56.5      +4.5%   <- 25% more processes for 4.5% more throughput
#
# 32 sits at the knee. Going to 40 buys 96 steps/s and costs 8 processes' worth of CPU, which is
# the difference between being able to use the machine while it trains and not.
#
# $Speedup is a REQUEST, not an achieved rate: RagdollRLBridge sets physics ticks to
# $Speedup * 120 per process, and the CPU delivers well under that. It only matters when the
# target falls BELOW what the CPU could otherwise do, at which point it throttles:
#
#              target steps/s = $Speedup * 15 * $NParallel
#    n=32:  speedup 4 -> 1,920  THROTTLES (ceiling is 2,162)
#           speedup 8 -> 3,840  safe
#    n=40:  speedup 4 -> 2,400  barely safe (ceiling 2,258)
#           speedup 8 -> 4,800  safe
#
# 8 is the smallest safe value at n=32, and smaller is better for --viz because the window runs
# at $Speedup. Verified: 40 procs gave 2,258 steps/s at speedup 8 vs ~2,200 at speedup 16, so
# dropping from 16 costs nothing. Simulation is unchanged either way - delta stays 1/120 s.
$NParallel   = 32
$Speedup     = 8

# 1M steps is roughly 9 minutes at the sweet spot.
# Pure-RL get-up realistically needs 10-100M.
$Timesteps   = 80000000

# Wall-clock cap in seconds. Training stops at whichever comes first, this or $Timesteps.
# Use this when you want a run of a known DURATION - throughput is not predictable enough to
# express "15 minutes" as a step count, especially with a visible instance.
# 0 = no time limit.
# 300 = 5 minutes.
# 900 = 15 minutes.
# 7200 = 2 hours.
$MaxSeconds  = 3600

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
$ExperimentName = ""

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
} elseif ($Task -eq "stand") {
    $ExportPreset = "Windows Stand"
    $BuildName    = "RagdollStandTraining.exe"
} elseif ($Task -eq "getup") {
    $ExportPreset = "Windows GetUp"
    $BuildName    = "RagdollGetUpTraining.exe"
} elseif ($Task -eq "upright") {
    $ExportPreset = "Windows Upright"
    $BuildName    = "RagdollUprightTraining.exe"
} elseif ($Task -eq "walk") {
    $ExportPreset = "Windows Walk"
    $BuildName    = "RagdollWalkTraining.exe"
} else {
    throw "Unknown `$Task '$Task'. Use 'stand', 'getup', 'upright', 'perturbation' or 'walk'."
}
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
$BuildExe    = "$ProjectPath/build/$BuildName"

# Continues training from an existing checkpoint instead of starting over.
#
# Restores the policy AND the Adam optimizer state and keeps the step count, so training genuinely
# continues rather than warm-starting from weights alone. Restarting instead of resuming re-pays
# the whole optimizer warm-up, not just the elapsed steps.
#
# WHERE THE LOGS GO is decided by $ExperimentName from config.ps1, NOT by the checkpoint you pick.
# This script never inspects the checkpoint's directory - it passes $ExperimentName straight to
# train.py, which hands it to SB3 as tb_log_name.
#
# SB3 then resolves that name with reset_num_timesteps = False (configure_logger, "Continue training
# in the same directory"), which appends to the LATEST existing directory carrying that name, or
# creates <name>_0 if none exists. So:
#
#   $ExperimentName derived (the default)  -> a NEW directory, e.g. resuming stand_v28 writes
#                                             stand_v29_0. Two runs, two TensorBoard series.
#   $ExperimentName pinned to an existing  -> APPENDS to that directory, one continuous series.
#     lineage, e.g. "stand_v29"
#
# Neither is safer for your DATA: checkpoints are files, and an earlier one stays restorable however
# later runs are named. The step count carries across either way, so on TensorBoard's global-step
# axis two runs appear as adjacent segments rather than a broken line. What pinning buys is that the
# version number keeps meaning "one experiment" instead of "one session"; what it costs is the trap
# config.ps1 documents - change the reward later, forget to unpin, and two incomparable lineages
# land in one curve. Prefer the derived default unless you are deliberately extending a lineage
# whose configuration has not changed.
#
# New checkpoints are step-stamped past the restored count, so nothing existing is overwritten.
#
# Usage:
#   ./resume.ps1                     # pick from a menu of recent checkpoints
#   ./resume.ps1 -Viz                # same, with one instance rendered in a window
#   ./resume.ps1 ../runs/getup_v4_2/final_003608200.zip     # explicit path, no menu
param(
    [Parameter(Position = 0)][string]$Checkpoint,
    [switch]$Viz
)
. "$PSScriptRoot/config.ps1"

# --- Checkpoint selection -----------------------------------------------------------------------
# Presented as a menu rather than requiring a hand-typed path: the filenames carry nine-digit step
# counts, and picking the right one by eye across several runs is exactly the kind of thing that
# leads to silently resuming the wrong session.
# Which BRAIN a task trains. Checkpoints are only interchangeable inside one of these: the four
# upright tasks share BodyStateObservation, UprightProgressReward and UprightTermination, while
# walk has its own reward/termination and an objective that actively conflicts (standing FAILS
# above 0.6 m/s; walking is REWARDED for reaching 0.4 m/s). Measured cost of ignoring that:
# 33M steps of walk training took standing/all from 0.478 to 0.170.
function Get-Brain([string]$TaskName) {
    if ($TaskName -eq "walk") { return "Walk" }
    return "GetUp"
}

# What a run actually was, read from its own manifest rather than guessed from the directory name.
# environment.task_kind gives the brain; cli_args.env_path gives the specific task, because four
# tasks share task_kind=GetUp and the enum alone cannot tell stand from getup from upright.
function Get-RunInfo([string]$RunDir) {
    $info = [PSCustomObject]@{ Task = "?"; Brain = "?" }
    $manifest = Join-Path $RunDir "manifest.json"
    if (-not (Test-Path $manifest)) { return $info }
    try {
        $m = Get-Content $manifest -Raw | ConvertFrom-Json
        if ($m.environment -and $m.environment.task_kind) { $info.Brain = $m.environment.task_kind }
        if ($m.cli_args -and $m.cli_args.env_path) {
            $exe = Split-Path $m.cli_args.env_path -Leaf
            if ($exe -match "^Ragdoll(.+)Training\.exe$") { $info.Task = $matches[1].ToLower() }
        }
    } catch { }
    return $info
}

if (-not $Checkpoint) {
    $wantBrain = Get-Brain $Task

    # Six rather than three: there are five tasks now, and a three-run window can hide every
    # checkpoint of the task you are actually training behind unrelated recent runs.
    $runs = Get-ChildItem -Path $ExperimentDir -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 6

    if (-not $runs) {
        Write-Host "No runs found in $ExperimentDir - nothing to resume." -ForegroundColor Red
        exit 1
    }

    $options = @()
    foreach ($run in $runs) {
        $meta = Get-RunInfo $run.FullName
        $ckpts = Get-ChildItem -Path $run.FullName -Filter *.zip -ErrorAction SilentlyContinue |
                 ForEach-Object {
                     $steps = 0
                     if ($_.Name -match "_(\d+)\.zip$") { $steps = [int64]$matches[1] }
                     [PSCustomObject]@{
                         Run       = $run.Name
                         File      = $_.FullName
                         Name      = $_.Name
                         Steps     = $steps
                         When      = $_.LastWriteTime
                         Task      = $meta.Task
                         Brain     = $meta.Brain
                         SameBrain = ($meta.Brain -eq $wantBrain)
                     }
                 } |
                 Sort-Object Steps -Descending |
                 Group-Object Steps |
                 ForEach-Object {
                     # One entry per step count. A periodic checkpoint and the end-of-run final
                     # land on the same step whenever the session length is a multiple of the save
                     # interval, and they hold the same policy - listing both just makes the menu
                     # ask a question with two identical answers. Prefer final_, then interrupted_,
                     # then checkpoint_, so the surviving name says the most about the run.
                     $_.Group |
                         Sort-Object @{Expression = {
                             if ($_.Name -like 'final_*')       { 0 }
                             elseif ($_.Name -like 'interrupted_*') { 1 }
                             else                               { 2 }
                         }} |
                         Select-Object -First 1
                 } |
                 Sort-Object Steps -Descending |
                 Select-Object -First 2
        $options += $ckpts
    }

    if (-not $options) {
        Write-Host "No .zip checkpoints found under $ExperimentDir." -ForegroundColor Red
        exit 1
    }

    # Compatible checkpoints first. Cross-brain options are still LISTED, not filtered out:
    # bootstrapping across brains is sometimes deliberate - walk was trained from a stand policy
    # on purpose - so hiding them would break a workflow that works.
    $options = @($options | Sort-Object @{Expression = "SameBrain"; Descending = $true},
                                        @{Expression = "When"; Descending = $true})

    Write-Host ""
    Write-Host "  Recent checkpoints                    building: $Task ($ExportPreset)" -ForegroundColor Cyan
    Write-Host "  ----------------------------------------------------------------------------------"
    for ($i = 0; $i -lt $options.Count; $i++) {
        $o = $options[$i]
        $label = "{0,2}) {1,-16} {2,-26} {3,12} steps  [{4}]" -f `
                 ($i + 1), $o.Run, $o.Name, $o.Steps.ToString("N0"), $o.Task
        if (-not $o.SameBrain) {
            Write-Host "$label  different brain" -ForegroundColor DarkYellow
        } elseif ($i -eq 0) {
            Write-Host "$label  <- newest compatible" -ForegroundColor Green
        } else {
            Write-Host $label
        }
    }
    Write-Host "  ----------------------------------------------------------------------------------"

    $choice = Read-Host "  Select 1-$($options.Count) (Enter = 1, q = quit)"
    if ($choice -eq "q") { exit 0 }
    if (-not $choice) { $choice = "1" }

    $index = 0
    if (-not [int]::TryParse($choice, [ref]$index) -or $index -lt 1 -or $index -gt $options.Count) {
        Write-Host "Not a valid selection: '$choice'" -ForegroundColor Red
        exit 1
    }
    $picked = $options[$index - 1]
    $Checkpoint = $picked.File

    if (-not $picked.SameBrain) {
        Write-Host ""
        Write-Host "  WARNING: that checkpoint trained [$($picked.Task)] ($($picked.Brain)) and you are" -ForegroundColor Yellow
        Write-Host "  building [$Task] ($wantBrain). Those brains have conflicting objectives, and" -ForegroundColor Yellow
        Write-Host "  training one on the other degrades it: standing/all went 0.478 -> 0.170 after" -ForegroundColor Yellow
        Write-Host "  33M steps of walk training. Deliberate bootstrapping is fine; an accident is not." -ForegroundColor Yellow
        Write-Host ""
        $confirm = Read-Host "  Continue anyway? (y/N)"
        if ($confirm -ne "y") { exit 0 }
    }
}

if (-not (Test-Path $Checkpoint)) {
    Write-Host "Checkpoint not found: $Checkpoint" -ForegroundColor Red
    exit 1
}


Write-Host ""
Write-Host "Resuming from $Checkpoint" -ForegroundColor Cyan
if ($Viz) {
    Write-Host "  one instance visible - right-drag to look, WASD + Q/E to move, Shift to boost" -ForegroundColor DarkGray
    Write-Host "  it renders at speedup x$Speedup, so motion looks fast" -ForegroundColor DarkGray
}

if ($BuildExe -eq $GodotExe) { Write-Host 'Using Godot Editor directly.' -ForegroundColor Yellow; dotnet build "$ProjectPath" }

$vizArg = @()
if ($Viz) { $vizArg = @('--viz') }

& $Python $TrainScript `
    --env_path=$BuildExe `
    --n_parallel=$NParallel `
    --speedup=$Speedup `
    --dummies=$Dummies `
    --timesteps=$Timesteps `
    --experiment_dir=$ExperimentDir `
    --experiment_name=$ExperimentName `
    --save_every_seconds=$SaveEverySeconds `
    --max_seconds=$MaxSeconds `
    --scene_path="$ScenePath" `
    --restore=$Checkpoint `
    @vizArg









& "$PSScriptRoot/summary.ps1" -NamePrefix $ExperimentName


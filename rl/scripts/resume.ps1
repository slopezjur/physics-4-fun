# Continues training from an existing checkpoint instead of starting over.
#
# Restores the policy AND the Adam optimizer state and keeps the step count, so training genuinely
# continues rather than warm-starting from weights alone. Restarting instead of resuming re-pays
# the whole optimizer warm-up, not just the elapsed steps.
#
# SB3 continues INTO the checkpoint's own run directory rather than creating a new one
# (configure_logger: "Continue training in the same directory" when reset_num_timesteps is False),
# so the TensorBoard curve stays one continuous line. New checkpoints are step-stamped past the
# restored count, so nothing existing is overwritten.
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
if (-not $Checkpoint) {
    $runs = Get-ChildItem -Path $ExperimentDir -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 3

    if (-not $runs) {
        Write-Host "No runs found in $ExperimentDir - nothing to resume." -ForegroundColor Red
        exit 1
    }

    $options = @()
    foreach ($run in $runs) {
        $ckpts = Get-ChildItem -Path $run.FullName -Filter *.zip -ErrorAction SilentlyContinue |
                 ForEach-Object {
                     $steps = 0
                     if ($_.Name -match '_(\d+)\.zip$') { $steps = [int64]$matches[1] }
                     [PSCustomObject]@{
                         Run   = $run.Name
                         File  = $_.FullName
                         Name  = $_.Name
                         Steps = $steps
                         When  = $_.LastWriteTime
                     }
                 } |
                 Sort-Object Steps -Descending |
                 Select-Object -First 3
        $options += $ckpts
    }

    if (-not $options) {
        Write-Host "No .zip checkpoints found under $ExperimentDir." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "  Recent checkpoints (newest runs first)" -ForegroundColor Cyan
    Write-Host "  ---------------------------------------------------------------------"
    for ($i = 0; $i -lt $options.Count; $i++) {
        $o = $options[$i]
        $label = "{0,2}) {1,-16} {2,-28} {3,12} steps  {4}" -f `
                 ($i + 1), $o.Run, $o.Name, $o.Steps.ToString("N0"), $o.When.ToString("MM-dd HH:mm")
        if ($i -eq 0) {
            Write-Host "$label   <- newest" -ForegroundColor Green
        } else {
            Write-Host $label
        }
    }
    Write-Host "  ---------------------------------------------------------------------"

    $choice = Read-Host "  Select 1-$($options.Count) (Enter = 1, q = quit)"
    if ($choice -eq 'q') { exit 0 }
    if (-not $choice) { $choice = '1' }

    $index = 0
    if (-not [int]::TryParse($choice, [ref]$index) -or $index -lt 1 -or $index -gt $options.Count) {
        Write-Host "Not a valid selection: '$choice'" -ForegroundColor Red
        exit 1
    }
    $Checkpoint = $options[$index - 1].File
}

if (-not (Test-Path $Checkpoint)) {
    Write-Host "Checkpoint not found: $Checkpoint" -ForegroundColor Red
    exit 1
}

& "$PSScriptRoot/export.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Resuming from $Checkpoint" -ForegroundColor Cyan
if ($Viz) {
    Write-Host "  one instance visible - right-drag to look, WASD + Q/E to move, Shift to boost" -ForegroundColor DarkGray
    Write-Host "  it renders at speedup x$Speedup, so motion looks fast" -ForegroundColor DarkGray
}

$vizArg = @()
if ($Viz) { $vizArg = @('--viz') }

& $Python $TrainScript `
    --env_path=$BuildExe `
    --n_parallel=$NParallel `
    --speedup=$Speedup `
    --timesteps=$Timesteps `
    --experiment_dir=$ExperimentDir `
    --experiment_name=$ExperimentName `
    --save_every_seconds=$SaveEverySeconds `
    --max_seconds=$MaxSeconds `
    --restore=$Checkpoint `
    @vizArg

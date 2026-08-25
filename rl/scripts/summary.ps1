# Prints the closing summary for a finished run, read from its manifest.json.
#
# Called at the end of train.ps1 and resume.ps1. Exits silently (0) when there is nothing to
# report: this is a convenience epilogue, and a training run that succeeded must not be reported as
# failed because its summary could not be rendered.
param(
    # The run directory to summarise. Normally omitted - the caller passes -NamePrefix instead.
    [string] $RunDir,

    # Experiment name to resolve $RunDir from, matching "<prefix>" or SB3's "<prefix>_<n>".
    #
    # Prefer this over letting the script pick the newest directory in rl/runs. "Newest" is wrong
    # whenever a second run is training concurrently, or a throwaway smoke run finished later than
    # the run being summarised - and a summary that silently describes a DIFFERENT experiment is
    # worse than no summary, because the numbers look plausible.
    [string] $NamePrefix
)

$ExperimentDir = "$PSScriptRoot/../runs"

if (-not $RunDir) {
    $candidates = Get-ChildItem -Path $ExperimentDir -Directory -ErrorAction SilentlyContinue
    if ($NamePrefix) {
        # Anchored, and tolerant of SB3's numeric suffix only - so "stand_v2" cannot match
        # "stand_v28", the same trap PolicyAutoLoader.BrainRunPrefixes documents.
        $pattern = "^$([regex]::Escape($NamePrefix))(_\d+)?$"
        $candidates = $candidates | Where-Object { $_.Name -match $pattern }
    }
    $latest = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { $RunDir = $latest.FullName }
}

if (-not $RunDir -or -not (Test-Path "$RunDir/manifest.json")) { exit 0 }

try {
    $manifest = Get-Content "$RunDir/manifest.json" -Raw -ErrorAction Stop | ConvertFrom-Json
} catch {
    Write-Host "[summary] Could not read $RunDir/manifest.json - skipping." -ForegroundColor DarkYellow
    exit 0
}

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " TRAINING SESSION SUMMARY" -ForegroundColor Cyan
Write-Host " Run: $(Split-Path $RunDir -Leaf)" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

if ($manifest.cli_args) {
    $scene = (Split-Path "$($manifest.cli_args.scene_path)" -Leaf) -replace '\.tscn$', ''
    Write-Host " Scene           : $scene" -ForegroundColor DarkGray
    # Bodies, not processes: with $Dummies the two differ by up to 64x, and reporting only
    # n_parallel makes two runs of wildly different sample rates look identical.
    $dummies = if ($null -ne $manifest.cli_args.dummies) { $manifest.cli_args.dummies } else { 1 }
    $bodies = [int]$manifest.cli_args.n_parallel * [int]$dummies
    Write-Host " Hardware Load   : $($manifest.cli_args.n_parallel) procs x $dummies dummies = $bodies bodies (speedup x$($manifest.cli_args.speedup))" -ForegroundColor DarkGray
    Write-Host "-----------------------------------------------------" -ForegroundColor DarkGray
}

Write-Host " SPS (Steps/Sec) : " -NoNewline; Write-Host "$($manifest.steps_per_second)" -ForegroundColor Green
Write-Host " Steps this run  : $($manifest.steps_this_session)"
Write-Host " Total Steps     : $($manifest.timesteps_done)"
Write-Host " Time Elapsed    : $([math]::Round($manifest.elapsed_seconds, 1))s"
Write-Host ""

if ($manifest.latest_metrics) {
    Write-Host " Latest Dummy Skills:" -ForegroundColor Cyan

    # ep_rew_mean is absent on a run that ended before the first rollout completed. Casting $null
    # to [double] yields 0 and prints a confident "0.000" for "not measured yet".
    $rewRaw = $manifest.latest_metrics.ep_rew_mean
    if ($null -ne $rewRaw) {
        $rew = [double]$rewRaw
        $colour = if ($rew -gt 0) { 'Green' } else { 'Yellow' }
        Write-Host " Mean Reward     : " -NoNewline
        Write-Host $rew.ToString("0.000") -ForegroundColor $colour
    } else {
        Write-Host " Mean Reward     : (no completed rollout)" -ForegroundColor DarkYellow
    }

    Write-Host " Episode Length  : $($manifest.latest_metrics.ep_len_mean)"

    if ($null -ne $manifest.environment.curriculum_pose_t) {
        Write-Host " Curriculum Pose : $([math]::Round($manifest.environment.curriculum_pose_t, 3))"
    }
}
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

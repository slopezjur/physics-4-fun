# Opens a viewer running the policy that is CURRENTLY IN GODOT - the arena.
#
# Unlike the 2.3.2 track's watch.ps1 this needs no Kit, no D3D12 override and no shader-cache warm
# up: $Viewer selects a native OpenGL window ("newton") or a browser viewer ("viser"). Close the
# window to exit.
#
#     .\watch.ps1                        # the brain exported to Models/, no prompt
#     .\watch.ps1 -Pick                  # choose from the same menu resume.ps1 uses
#     .\watch.ps1 -Task perturb          # Stand, with shots landing every 3 s from t=2 s
#     .\watch.ps1 logs\rsl_rl\...\model_3087.pt      # explicit path
#
# **It used to open a different brain from the one Godot was running**, and silently. Resolution
# went through Get-Isaac3Run, which only ever searches the base $Experiment tree - so for Stand it
# picked stand/night13 at action_scale 0.4 while Godot was running stand_assist/night06 at 0.15.
# Watching one policy and shipping another is not a display bug, so selection is now the shared
# Select-Isaac3Checkpoint that resume.ps1 uses, and the two cannot drift apart again.
#
# R restarts the episode without leaving the viewer, which is the only practical way to see a
# perturbation recovery more than once.
param(
    [Parameter(Position = 0)][string] $Checkpoint,
    # Overrides $Task in config.ps1 for this command only.
    [ValidateSet('stand', 'perturb', 'walk', 'run')]
    [string] $Task,
    # Overrides $ExperimentName - which logs/rsl_rl subtree to read. The task's default tree is
    # often not where its best checkpoints are; an error names the trees that do have runs.
    [string] $Experiment,
    # Overrides $WatchEnvs for this command only.
    [int] $Envs = 0,
    # Show the checkpoint menu instead of taking the one that is in Godot.
    [switch] $Pick,
    # Open the NEWEST checkpoint of this task instead of the one promoted to Godot. What you want
    # straight after training: the default answers "what is shipping", not "what did I just make".
    [switch] $Latest,
    # Widen the menu, same meaning as in resume.ps1.
    #
    # **Without these the picker cannot reach most checkpoints.** The menu lists one run per
    # lineage and one checkpoint per run, so anything but the newest run's final model is
    # unreachable - `perturb_honest2/night11/model_7350` was invisible while `night13/model_9310`
    # was the only offer, even though 7350 measured better in Godot.
    [switch] $All,
    [int] $PerRun = 1,
    # Watch the UNTRAINED body instead of a checkpoint - the zero-action baseline. This is the
    # comparison every result on this track is measured against, so it is worth being able to see.
    [switch] $ZeroAction
)
# Capture the parameters BEFORE dot-sourcing. config.ps1 defines $Envs itself (the TRAINING env
# count), so reading $Envs after the dot-source returns 4096, not what was passed - `-Envs 4`
# silently rendered 4096 bodies. Parameters and config share one scope here; only a distinct name
# survives. $Experiment is the same trap in the other direction: config DERIVES it, so the
# parameter has to be handed over under a different name or the derivation overwrites it.
$EnvsOverride       = $Envs
$TaskOverride       = $Task
$ExperimentOverride = $Experiment
. "$PSScriptRoot/config.ps1"

if ($EnvsOverride -gt 0) { $WatchEnvs = $EnvsOverride }
$env:P4F_XPBD_ITERATIONS = "$SolverIterations"

$cmd = @("$PSScriptRoot/play.py", '--task', $TaskId, '--num_envs', $WatchEnvs, '--viewer', $Viewer)

if ($ZeroAction) {
    Write-Host "[watch] $Task : ZERO-ACTION baseline, $WatchEnvs envs, viewer '$Viewer'" -ForegroundColor Cyan
    $cmd += '--zero_action'
} else {
    if (-not $Checkpoint) {
        # -Interactive only when asked. Without -Pick this neither renders nor prompts, so
        # watch.ps1 stays usable unattended - and cannot be answered by a null stdin returning
        # empty and silently taking row 1, which is how a stray training run got launched once.
        $picked = Select-Isaac3Checkpoint -TaskName $Task -Title "Watch which brain?" `
                                          -Interactive:$Pick -PreferNewest:$Latest `
                                          -All:$All -PerRun $PerRun
        if (-not $picked) { exit 0 }
        $Checkpoint = $picked.File
        $where = "{0}/{1}  model_{2}  scale {3}{4}" -f `
                 ($picked.Experiment -replace '^p4f_newton_', ''), $picked.Run,
                 $picked.Iteration, $picked.ActionScale,
                 $(if ($picked.Promoted) { "  <- the brain in Godot" } else { "" })

        # **Say when a newer checkpoint exists but is not the one opening.** Defaulting to the
        # promoted brain is right for "show me what is shipping" and silently wrong right after a
        # training run - the new checkpoint sorts to row 2 and the run looks like it produced
        # nothing. Nothing is hidden that the user is not told about.
        $newest = Get-Isaac3Candidates -WantTask $Task -All:$All -PerRun $PerRun |
                  Where-Object SameTask | Sort-Object When -Descending | Select-Object -First 1
        if ($newest -and $newest.File -ne $picked.File) {
            Write-Host ("[watch] NOTE a newer {0} checkpoint exists: {1}/{2} model_{3}" -f `
                        $Task, ($newest.Experiment -replace '^p4f_newton_', ''),
                        $newest.Run, $newest.Iteration) -ForegroundColor Yellow
            Write-Host "[watch]      opening the promoted one. Use -Latest for the new one, or -Pick to choose." -ForegroundColor Yellow
        }
    } else {
        $info  = Get-Isaac3RunInfo -RunDir (Split-Path -Parent $Checkpoint)
        $where = "{0}  scale {1}" -f (Split-Path -Leaf $Checkpoint), $info.ActionScale
    }

    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        Write-Host "Checkpoint not found: $Checkpoint" -ForegroundColor Red
        exit 1
    }

    Write-Host "[watch] $Task : $where" -ForegroundColor Cyan
    Write-Host "[watch] $WatchEnvs envs, viewer '$Viewer'" -ForegroundColor Cyan
    $cmd += @('--checkpoint', $Checkpoint)
}

Invoke-Isaac3 -Arguments $cmd

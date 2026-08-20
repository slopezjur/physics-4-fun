# Resume training with ONE instance rendered in a window so you can watch it live.
#
# Thin wrapper over resume.ps1 so the checkpoint picker, the export step and the argument list
# exist in exactly one place - the headless and windowed paths cannot drift apart.
#
# Camera: right-drag to look, WASD + Q/E to move, Shift to boost.
#
# Two things worth knowing, same as train-viz.ps1:
#  - The visible instance is essentially FREE on this machine: measured 2,235 steps/s with
#    --viz against 2,144 without, in the same run. Earlier runs suggested a large penalty;
#    those predate the action-space fix and had other load on the machine, and the note
#    warning about a slowdown was wrong.
#  - the visible instance runs at $Speedup like the rest, so motion looks very fast. Lower
#    $Speedup in config.ps1 (2-4) if you actually want to watch the movement.
#
# Usage:
#   ./resume-viz.ps1                                          # pick from the menu
#   ./resume-viz.ps1 ../runs/getup_v4_2/final_003608200.zip   # explicit path
param(
    [Parameter(Position = 0)][string]$Checkpoint
)

& "$PSScriptRoot/resume.ps1" -Checkpoint $Checkpoint -Viz
exit $LASTEXITCODE

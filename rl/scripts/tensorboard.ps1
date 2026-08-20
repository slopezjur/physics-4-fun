# Opens the training curves at http://localhost:6008
#
# Port 6008, not the usual 6006 - the Godot editor's remote debugger already listens on 6006.
#
# What to watch:
#   rollout/ep_rew_mean - is it learning at all?
#   rollout/ep_len_mean - 121 means every episode runs the full 8s. Dropping below that means
#                         episodes end early, i.e. reaching Standing (good) or Inverted (bad).
#
# Run this in its OWN terminal, alongside training - it blocks until you Ctrl+C it.
. "$PSScriptRoot/config.ps1"

Write-Host "TensorBoard -> http://localhost:6008  (logdir: $ExperimentDir)" -ForegroundColor Cyan
Write-Host "Leave this window open; Ctrl+C to stop." -ForegroundColor DarkGray
& "$ProjectPath/rl/.venv/Scripts/tensorboard.exe" --logdir "$ExperimentDir" --port 6008

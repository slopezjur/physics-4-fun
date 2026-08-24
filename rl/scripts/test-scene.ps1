# Opens the inspection scene: dummy + arena + free-look camera, no training, no server.
#
# This scene never opens a socket (its Sync node is in HUMAN mode), so it is safe to run at any
# time - including while a training session is going.
. "$PSScriptRoot/config.ps1"

Write-Host "Opening RagdollStandArena (test scene)..." -ForegroundColor Cyan
& $GodotExe --path $ProjectPath "res://Scenes/RL/Upright/RagdollStandArena.tscn"

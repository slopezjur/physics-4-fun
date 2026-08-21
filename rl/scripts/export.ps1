# Rebuilds the standalone binary that training runs against.
#
# Training NEVER uses the open editor - it launches this exported build. So any C# or scene change
# must be re-exported first, or you will silently train stale code.
#
# This script deliberately does NOT touch project.godot.
#
# It used to: a Godot export builds whatever run/main_scene points at, and the editor wants the
# inspection Arena while the export needs the Training scene, so the script swapped the setting and
# restored it in a finally block. That round-tripped project.godot through PowerShell text encoding
# twice per export, and Set-Content -Encoding utf8 on Windows PowerShell 5.1 writes UTF-8 WITH a
# BOM. The next run then read those BOM bytes back using the ANSI codepage, turning them into the
# literal text "Ã¯Â»Â¿" glued onto config_version - which Godot then preserved as a quoted key on
# its next save, corrupting the project file and making the Project Manager report "unknown version
# of Godot".
#
# The replacement is a feature-tagged project setting. project.godot declares both:
#
#     run/main_scene="res://Scenes/RL/Upright/RagdollPerturbationArena.tscn"
#     run/main_scene.stand="res://Scenes/RL/Upright/RagdollStandTraining.tscn"
#
# and export_presets.cfg sets custom_features="training". Godot resolves "<setting>.<feature>"
# against the active feature tags, so the editor boots the Arena and the exported build boots the
# Training scene - with no file rewriting, and therefore no encoding round-trip to get wrong.
. "$PSScriptRoot/config.ps1"

Write-Host "Exporting '$ExportPreset' -> $BuildExe" -ForegroundColor Cyan
$FeatureTag = @{ "Windows Stand" = "stand"; "Windows GetUp" = "getup"; "Windows Upright" = "upright"; "Windows Perturbation" = "perturbation"; "Windows Walk" = "walk" }[$ExportPreset]
Write-Host "  (main scene comes from run/main_scene.$FeatureTag via the '$FeatureTag' feature tag)" -ForegroundColor DarkGray

& $GodotExe --headless --path $ProjectPath --export-release $ExportPreset $BuildExe
if ($LASTEXITCODE -ne 0) {
    Write-Host "Export FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}

Write-Host "Export complete." -ForegroundColor Green

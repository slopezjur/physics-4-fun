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
#     run/main_scene="res://Scenes/RL/Jolt/Upright/RagdollPerturbationArena.tscn"
#     run/main_scene.stand="res://Scenes/RL/Jolt/Upright/RagdollStandTraining.tscn"
#
# and export_presets.cfg sets custom_features="training". Godot resolves "<setting>.<feature>"
# against the active feature tags, so the editor boots the Arena and the exported build boots the
# Training scene - with no file rewriting, and therefore no encoding round-trip to get wrong.
. "$PSScriptRoot/config.ps1"

Write-Host "Exporting '$ExportPreset' -> $BuildExe" -ForegroundColor Cyan
$FeatureTag = @{ "Windows Stand" = "stand"; "Windows GetUp" = "getup"; "Windows Upright" = "upright"; "Windows Perturbation" = "perturbation"; "Windows Walk" = "walk" }[$ExportPreset]
Write-Host "  (main scene comes from run/main_scene.$FeatureTag via the '$FeatureTag' feature tag)" -ForegroundColor DarkGray

Write-Host 'Building ExportRelease assembly...' -ForegroundColor Cyan
# Bypassing Godot 4 feature tag bug in headless console wrappers by temporarily replacing the default main_scene
$ProjectGodot = "$ProjectPath/project.godot"
$FeatureOverride = "run/main_scene.$FeatureTag"
$TargetSceneLine = (Get-Content $ProjectGodot | Select-String $FeatureOverride).Line
if ($TargetSceneLine) {
    $TargetScene = $TargetSceneLine.Split("=")[1]
    $GodotContent = Get-Content $ProjectGodot
    $GodotContent = $GodotContent -replace '^run/main_scene=.*', "run/main_scene=$TargetScene"
    Set-Content -Path $ProjectGodot -Value $GodotContent
    Write-Host "Temporarily set default main_scene to $TargetScene" -ForegroundColor Yellow
}

dotnet build "$ProjectPath" -c ExportRelease
if ($LASTEXITCODE -ne 0) { 
    git checkout -- $ProjectGodot
    exit 1 
}

$PckPath = $BuildExe.Replace(".exe", ".pck")
Start-Process -FilePath $GodotExe -ArgumentList "--headless --path `"$ProjectPath`" --export-debug `"$ExportPreset`" `"$PckPath`"" -Wait -NoNewWindow
$ExportExitCode = $LASTEXITCODE

# Restore project.godot
git checkout -- $ProjectGodot

if ($ExportExitCode -ne 0) {
    Write-Host "Export FAILED (exit $ExportExitCode)" -ForegroundColor Red
    exit 1
}

Write-Host "Export complete." -ForegroundColor Green


# Fix Godot 4 C# export bug on Windows
$BuildNameWithoutExe = $BuildName.Replace('.exe', '')
$ExpectedDataFolder = "$ProjectPath/build/data_$($BuildNameWithoutExe)_windows_x86_64"
$ProjectDataFolder = "$ProjectPath/build/data_Physics4Fun_windows_x86_64"

if (Test-Path $ProjectDataFolder) {
    if (Test-Path $ExpectedDataFolder) {
        Remove-Item -Recurse -Force $ExpectedDataFolder
    }
    Rename-Item $ProjectDataFolder $ExpectedDataFolder
}

# Regenerates the dummy from the Godot scene: ActiveRagdoll.tscn -> URDF -> USD.
#
# Run this after ANY change to the ragdoll scene. The Godot scene is the single source of truth for
# masses, joint limits and PD gains - nothing on the Isaac side is authored by hand - so a scene
# edit that is not followed by this leaves training on a stale dummy, silently.
#
# convert_asset.py verifies the imported articulation against the rig contract (46 bodies, 45
# joints, 80.60 kg) rather than trusting that a URDF which parses is one that imported correctly.
. "$PSScriptRoot/config.ps1"

Write-Host "[convert] Godot scene -> URDF" -ForegroundColor Cyan
Invoke-Isaac -Arguments @((Join-Path $ProjectRoot 'tools\tscn_to_urdf.py'))
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[convert] URDF -> USD" -ForegroundColor Cyan
Invoke-Isaac -Arguments @("$PSScriptRoot/convert_asset.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[convert] done - retrain to pick up the new rig." -ForegroundColor Green

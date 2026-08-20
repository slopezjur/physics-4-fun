# Promotes a trained policy so the engine can run it (ONNX inference mode).
#
# Training artifacts live in rl/, which is hidden from Godot by rl/.gdignore - otherwise every
# checkpoint .onnx would be packed into the exported build and it would grow with every run.
# Models/ IS scanned, so a policy has to be copied there to be usable in-engine.
#
# Usage:
#   ./promote.ps1 ../runs/getup_v1_1/final_001234567.onnx
#
# Then set the Sync node's control_mode to 2 (Onnx Inference) and press Play - it runs with no
# Python process at all.
param(
    [Parameter(Mandatory = $true)][string]$OnnxFile
)
. "$PSScriptRoot/config.ps1"

if (-not (Test-Path $OnnxFile)) {
    Write-Host "Not found: $OnnxFile" -ForegroundColor Red
    exit 1
}

$Target = "$ProjectPath/Models/policy.onnx"
Copy-Item $OnnxFile $Target -Force
Write-Host "Promoted -> $Target" -ForegroundColor Green
Write-Host "Re-export (./export.ps1) before it appears in a standalone build." -ForegroundColor Yellow

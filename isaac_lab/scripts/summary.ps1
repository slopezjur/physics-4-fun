# Prints what is trained and what is exported.
#
# Called at the end of train.ps1, and useful on its own. Reports quietly rather than throwing when
# something is missing: this is a convenience epilogue, and a training run that succeeded must not
# look failed because its summary could not be rendered.
. "$PSScriptRoot/config.ps1"

Write-Host ""
Write-Host "Trained brains:" -ForegroundColor Cyan
foreach ($pair in @(@('stand', 'p4f_stand'), @('walk', 'p4f_walk'),
                    @('perturb', 'p4f_perturb'), @('run', 'p4f_run'))) {
    $run = Get-IsaacRun -Experiment $pair[1] -Minimum $MinIterations
    if ($run) {
        $ckpt = Split-Path (Get-IsaacCheckpoint -RunDir $run.FullName) -Leaf
        Write-Host ("  {0,-9} {1}  ({2})" -f $pair[0], $run.Name, $ckpt)
    } else {
        Write-Host ("  {0,-9} (not trained)" -f $pair[0]) -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Exported policies:" -ForegroundColor Cyan
$any = $false
if (Test-Path -LiteralPath $Exported) {
    foreach ($f in (Get-ChildItem -LiteralPath $Exported -Filter '*.onnx' -ErrorAction SilentlyContinue)) {
        Write-Host ("  {0,-24} {1,6:N0} KB   {2}" -f $f.Name, ($f.Length / 1KB), $f.LastWriteTime)
        $any = $true
    }
}
if (-not $any) { Write-Host "  (none)" -ForegroundColor DarkGray }
Write-Host ""

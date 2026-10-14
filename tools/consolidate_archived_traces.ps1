$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$root = (Resolve-Path -LiteralPath (Join-Path $workspace 'logs/mimickit-training')).Path
$reportRoot = Join-Path $workspace 'logs/storage-cleanup-20261007'
$manifest = Get-Content -LiteralPath (Join-Path $reportRoot 'duplicate-traces.json') -Raw | ConvertFrom-Json
if ($manifest.root -ne $root) { throw 'Manifest root mismatch' }
function Assert-Trace([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($full) -ne '.npz') { throw "Invalid archive path: $full" }
    $item = Get-Item -LiteralPath $full
    while ($item.FullName -ne $root) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $full" }
        $item = if ($item -is [IO.FileInfo]) { $item.Directory } else { $item.Parent }
    }
    return $full
}
$done = [Collections.Generic.List[object]]::new()
try {
    foreach ($row in $manifest.copies) {
        $path = Assert-Trace $row.path
        $canonical = Assert-Trace $row.canonical
        if ($path -eq $canonical) { throw 'Cannot consolidate a file with itself' }
        foreach ($file in @($path,$canonical)) {
            if ((Get-Item -LiteralPath $file).Length -ne $row.bytes -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $row.sha256) { throw "Changed trace: $file" }
        }
        $backup = [IO.Path]::GetFullPath($path + '.cleanup-backup')
        if (-not $backup.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $backup)) { throw "Unsafe backup: $backup" }
        Move-Item -LiteralPath $path -Destination $backup
        try {
            New-Item -ItemType HardLink -Path $path -Target $canonical | Out-Null
            if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $row.sha256) { throw "Link verification failed: $path" }
        } catch {
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
            Move-Item -LiteralPath $backup -Destination $path
            throw
        }
        Remove-Item -LiteralPath $backup
        $done.Add($row)
    }
} finally {
    [pscustomobject]@{ schema='immutable_trace_consolidation_v1'; linked_files=$done.Count; redundant_bytes=($done|Measure-Object bytes -Sum).Sum; content_hashes_verified=$true; note='Archived NPZ files are immutable. Identical contents now share NTFS storage; all original paths remain. Replace files atomically if ever regenerating an archive.'; copies=$done } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportRoot 'consolidated.json') -Encoding utf8
}
[pscustomobject]@{ ConsolidatedFiles=$done.Count; RedundantGiB=[math]::Round(($done|Measure-Object bytes -Sum).Sum/1GB,3); AllHashesVerified=$true } | ConvertTo-Json

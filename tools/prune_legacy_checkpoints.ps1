param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$reportRoot = Join-Path $root 'logs/storage-cleanup-20261007'
$manifestPath = Join-Path $reportRoot 'legacy-checkpoints.json'
$legacy = @('isaac_lab_3/logs', 'isaac_lab/logs', 'logs/rsl_rl', 'logs/mujoco', 'rl/runs', 'rl/runs_archive')
$allowed = @($legacy | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $root $_)) + [IO.Path]::DirectorySeparatorChar })
function Assert-Target([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not ($full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) -or
        -not ($allowed | Where-Object { $full.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })) {
        throw "Outside approved legacy roots: $full"
    }
    $current = Get-Item -LiteralPath $full
    while ($current.FullName -ne $root) {
        if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($current.FullName)" }
        $current = if ($current -is [IO.FileInfo]) { $current.Directory } else { $current.Parent }
    }
    return $full
}
if (-not $Apply) {
    if (Test-Path -LiteralPath $manifestPath) { throw 'Inventory already exists; inspect it before applying.' }
    $tracked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    & git -C $root ls-files | ForEach-Object { [void]$tracked.Add([IO.Path]::GetFullPath((Join-Path $root $_))) }
    $referenced = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    # Conservatively protect every explicitly named checkpoint, even in a different run.
    $textFiles = & rg --files --hidden -g '*.md' -g '*.json' -g '*.yaml' -g '*.yml' -g '*.ps1' -g '*.py' -g '*.tscn' -g '*.cs' -g '!**/.venv/**' -g '!**/.git/**' -g '!logs/**' -g '!isaac_lab/logs/**' -g '!isaac_lab_3/logs/**' $root
    $textFiles += @($allowed | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File | Where-Object { $_.Extension -in '.json','.yaml','.yml','.md' } | ForEach-Object FullName })
    foreach ($path in $textFiles) {
        foreach ($match in [regex]::Matches([IO.File]::ReadAllText($path), '(?i)(?:model_\d+\.pt|checkpoint_\d+\.(?:pt|zip))')) { [void]$referenced.Add($match.Value) }
    }
    $remove = [Collections.Generic.List[object]]::new()
    $keep = [Collections.Generic.List[object]]::new()
    foreach ($directory in $allowed) {
        $files = Get-ChildItem -LiteralPath $directory -Recurse -File | Where-Object { $_.Name -match '^(model|checkpoint)_(\d+)\.(pt|zip)$' }
        foreach ($group in ($files | Group-Object DirectoryName)) {
            $ordered = @($group.Group | Sort-Object { [long]([regex]::Match($_.Name, '\d+').Value) })
            $retain = @($ordered | Select-Object -First 1) + @($ordered | Select-Object -Last 2)
            foreach ($file in $ordered) {
                $reason = if ($tracked.Contains($file.FullName)) {'tracked'} elseif ($referenced.Contains($file.Name)) {'explicitly referenced'} elseif ($file.FullName -in $retain.FullName) {'first or latest two'} else {$null}
                $row = [pscustomobject]@{ path=(Assert-Target $file.FullName); bytes=$file.Length; modified_utc=$file.LastWriteTimeUtc.ToString('o'); reason=$reason }
                if ($reason) { $keep.Add($row) } else { $row.reason='unreferenced intermediate legacy checkpoint'; $remove.Add($row) }
            }
        }
    }
    New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
    [pscustomobject]@{ schema='legacy_checkpoint_prune_v1'; root=$root; created_utc=[DateTime]::UtcNow.ToString('o'); remove=$remove; retain=$keep } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    [pscustomobject]@{ DeleteFiles=$remove.Count; ReclaimGiB=[math]::Round(($remove|Measure-Object bytes -Sum).Sum/1GB,3); RetainFiles=$keep.Count; Manifest=$manifestPath } | ConvertTo-Json
    exit
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.root -ne $root) { throw 'Manifest root mismatch' }
if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' }) { throw 'Python is running; finish active work before pruning.' }
# Verify every exact file before any deletion; never delete a directory or wildcard.
foreach ($row in $manifest.remove) {
    $file = Get-Item -LiteralPath (Assert-Target $row.path)
    if ($file.Length -ne $row.bytes -or $file.LastWriteTimeUtc.Ticks -ne ([datetime]$row.modified_utc).ToUniversalTime().Ticks) { throw "Changed since inventory: $($row.path)" }
}
$deleted = [Collections.Generic.List[object]]::new()
try {
    foreach ($row in $manifest.remove) {
        Remove-Item -LiteralPath (Assert-Target $row.path)
        $deleted.Add($row)
    }
} finally {
    [pscustomobject]@{ deleted_files=$deleted.Count; deleted_bytes=($deleted|Measure-Object bytes -Sum).Sum; completed_utc=[DateTime]::UtcNow.ToString('o'); files=$deleted } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportRoot 'deleted.json') -Encoding utf8
}
foreach ($row in $manifest.retain) { if (-not (Test-Path -LiteralPath $row.path -PathType Leaf)) { throw "Retained checkpoint missing: $($row.path)" } }
[pscustomobject]@{ DeletedFiles=$deleted.Count; ReclaimedGiB=[math]::Round(($deleted|Measure-Object bytes -Sum).Sum/1GB,3); RetainedCheckpointsVerified=$manifest.retain.Count } | ConvertTo-Json

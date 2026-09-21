$MimicProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
function Resolve-MimicPath([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'An explicit nonempty path is required.' }
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $MimicProjectRoot $Path))
}
function Read-MimicConfig([string] $Path) {
    $value = & { param($ConfigFile) . $ConfigFile; $Mimic } (Resolve-MimicPath $Path)
    if ($value -isnot [hashtable]) { throw 'Config must define the $Mimic hashtable.' }
    return $value
}
function Require-MimicPath([string] $Path) {
    $resolved = Resolve-MimicPath $Path
    if (-not (Test-Path -LiteralPath $resolved)) { throw "Missing: $resolved" }
    return $resolved
}
function Format-MimicNumber($Value) { return ([double]$Value).ToString('G', [Globalization.CultureInfo]::InvariantCulture) }

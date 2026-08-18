$files = @(
    'Debug/telemetry_dump_20260819_001216/telemetry_dump_20260819_001216.csv',
    'Debug/telemetry_dump_20260819_001227/telemetry_dump_20260819_001227.csv',
    'Debug/telemetry_dump_20260819_001238/telemetry_dump_20260819_001238.csv'
)

foreach ($file in $files) {
    Write-Host "
=== Analyzing $file ==="
    $lines = Get-Content $file
    $header = $lines[0].Split(',')
    
    $timeIdx = $header.IndexOf('Time')
    $stateIdx = $header.IndexOf('State')
    $speedIdx = $header.IndexOf('PelvisSpeed')
    $kneeLIdx = $header.IndexOf('KneeAngleL')
    $kneeRIdx = $header.IndexOf('KneeAngleR')
    
    $lastState = ""
    foreach ($line in $lines | Select-Object -Skip 1) {
        $cols = $line.Split(',')
        $state = $cols[$stateIdx]
        if ($state -ne $lastState) {
            Write-Host "Time: $($cols[$timeIdx]) - State changed to $state (Speed: $($cols[$speedIdx]), KneeL: $($cols[$kneeLIdx]))"
            $lastState = $state
        }
    }
}

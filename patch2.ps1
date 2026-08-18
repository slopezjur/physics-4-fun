$content = Get-Content -Raw -Path 'Scenes/ActiveRagdoll.tscn'

function Update-Limits($node, $xl, $xu, $yl, $yu, $zl, $zu) {
    if ($script:content -match "(?s)(\[node name=`"$node`".*?\](?:(?!\[node).)*)") {
        $block = $matches[1]
        
        function Repl($axis, $low, $up) {
            if ($low -ne $null -and $up -ne $null) {
                if ($script:block -match "angular_limit_$axis/upper_angle =") {
                    $script:block = $script:block -replace "angular_limit_$axis/upper_angle = [^\n]+", "angular_limit_$axis/upper_angle = $up"
                } else {
                    $script:block = $script:block -replace "angular_limit_$axis/enabled = true\n", "angular_limit_$axis/enabled = true`n  angular_limit_$axis/upper_angle = $up`n"
                }
                if ($script:block -match "angular_limit_$axis/lower_angle =") {
                    $script:block = $script:block -replace "angular_limit_$axis/lower_angle = [^\n]+", "angular_limit_$axis/lower_angle = $low"
                } else {
                    $script:block = $script:block -replace "angular_limit_$axis/upper_angle = $up\n", "angular_limit_$axis/upper_angle = $up`n  angular_limit_$axis/lower_angle = $low`n"
                }
            }
        }
        
        $script:block = $block
        Repl 'x' $xl $xu
        Repl 'y' $yl $yu
        Repl 'z' $zl $zu
        
        $script:content = $script:content.Replace($block, $script:block)
    }
}

Update-Limits 'Joint_Pelvis_Spine' -0.4 1.0 $null $null $null $null
Update-Limits 'Joint_Chest_Head' -0.6 1.0 -1.0 1.0 -1.0 1.0
Update-Limits 'Joint_Chest_UpperArm_L' -1.0 3.0 -1.5 1.5 -0.5 2.5
Update-Limits 'Joint_Chest_UpperArm_R' -1.0 3.0 -1.5 1.5 -2.5 0.5
Update-Limits 'Joint_UpperArm_Forearm_L' -0.1 2.6 $null $null $null $null
Update-Limits 'Joint_UpperArm_Forearm_R' -0.1 2.6 $null $null $null $null
Update-Limits 'Joint_Pelvis_Thigh_L' -0.5 2.1 -0.5 0.5 -0.2 0.8
Update-Limits 'Joint_Pelvis_Thigh_R' -0.5 2.1 -0.5 0.5 -0.8 0.2
Update-Limits 'Joint_Thigh_Shin_L' -2.6 0.1 $null $null $null $null
Update-Limits 'Joint_Thigh_Shin_R' -2.6 0.1 $null $null $null $null
Update-Limits 'Joint_Shin_Foot_L' -0.8 0.6 $null $null $null $null
Update-Limits 'Joint_Shin_Foot_R' -0.8 0.6 $null $null $null $null

Set-Content -Path 'Scenes/ActiveRagdoll.tscn' -Value $script:content
Write-Host "Limits Patched successfully!"

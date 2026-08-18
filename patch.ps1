$content = Get-Content -Raw -Path 'Scenes/ActiveRagdoll.tscn'

function Update-Limits($node, $xl, $xu, $yl, $yu, $zl, $zu) {
    global $content
    if ($content -match "(?s)(\[node name=`"$node`".*?\](?:(?!\[node).)*)") {
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
        
        $content = $content.Replace($block, $script:block)
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

# Add ApparentInertiaMultiplier
foreach ($arm in @('UpperArm_L', 'UpperArm_R', 'Forearm_L', 'Forearm_R')) {
    if ($content -match "(?s)(\[node name=`"$arm`".*?\](?:(?!\[node).)*)") {
        $block = $matches[1]
        if ($block -notmatch "ApparentInertiaMultiplier") {
            $newBlock = $block -replace "MaxTorque =", "ApparentInertiaMultiplier = 5.0`n  MaxTorque ="
            $content = $content.Replace($block, $newBlock)
        }
    }
}

# Add Hands
if ($content -notmatch "Hand_L") {
    $shapePattern = '\[sub_resource type="BoxShape3D" id="Shape_Foot"\]\r?\nsize = Vector3\(0\.12, 0\.08, 0\.22\)'
    $meshPattern = '\[sub_resource type="BoxMesh" id="Mesh_Foot"\]\r?\nmaterial = SubResource\("Mat_Feet"\)\r?\nsize = Vector3\(0\.12, 0\.08, 0\.22\)'
    
    $handShape = "`n`n[sub_resource type=`"BoxShape3D`" id=`"Shape_Hand`"]`nsize = Vector3(0.08, 0.08, 0.12)"
    $handMesh = "`n`n[sub_resource type=`"BoxMesh`" id=`"Mesh_Hand`"]`nmaterial = SubResource(`"Mat_Body`")`nsize = Vector3(0.08, 0.08, 0.12)"
    
    if ($content -match $shapePattern) {
        $content = $content.Replace($matches[0], $matches[0] + $handShape)
    }
    if ($content -match $meshPattern) {
        $content = $content.Replace($matches[0], $matches[0] + $handMesh)
    }
    
    $handsNodes = "
[node name=`"Joint_Forearm_Hand_L`" type=`"Generic6DOFJoint3D`" parent=`".`"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0.36, 0.68, 0)
node_a = NodePath(`"../Forearm_L`")
node_b = NodePath(`"../Hand_L`")
angular_limit_x/enabled = true
angular_limit_x/upper_angle = 1.0
angular_limit_x/lower_angle = -1.0
angular_limit_y/enabled = true
angular_limit_y/upper_angle = 0.5
angular_limit_y/lower_angle = -0.5
angular_limit_z/enabled = true
angular_limit_z/upper_angle = 0.5
angular_limit_z/lower_angle = -0.5

[node name=`"Hand_L`" type=`"RigidBody3D`" parent=`".`" node_paths=PackedStringArray(`"ParentBone`")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0.36, 0.62, 0)
collision_layer = 2
collision_mask = 1
mass = 0.5
script = ExtResource(`"2_bone`")
BoneName = `"Hand_L`"
ParentBone = NodePath(`"../Forearm_L`")
ProportionalGain = 60.0
DerivativeGain = 6.0
ApparentInertiaMultiplier = 5.0
MaxTorque = 50.0

[node name=`"CollisionShape3D`" type=`"CollisionShape3D`" parent=`"Hand_L`"]
shape = SubResource(`"Shape_Hand`")

[node name=`"MeshInstance3D`" type=`"MeshInstance3D`" parent=`"Hand_L`"]
mesh = SubResource(`"Mesh_Hand`")

[node name=`"Joint_Forearm_Hand_R`" type=`"Generic6DOFJoint3D`" parent=`".`"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -0.36, 0.68, 0)
node_a = NodePath(`"../Forearm_R`")
node_b = NodePath(`"../Hand_R`")
angular_limit_x/enabled = true
angular_limit_x/upper_angle = 1.0
angular_limit_x/lower_angle = -1.0
angular_limit_y/enabled = true
angular_limit_y/upper_angle = 0.5
angular_limit_y/lower_angle = -0.5
angular_limit_z/enabled = true
angular_limit_z/upper_angle = 0.5
angular_limit_z/lower_angle = -0.5

[node name=`"Hand_R`" type=`"RigidBody3D`" parent=`".`" node_paths=PackedStringArray(`"ParentBone`")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -0.36, 0.62, 0)
collision_layer = 2
collision_mask = 1
mass = 0.5
script = ExtResource(`"2_bone`")
BoneName = `"Hand_R`"
ParentBone = NodePath(`"../Forearm_R`")
ProportionalGain = 60.0
DerivativeGain = 6.0
ApparentInertiaMultiplier = 5.0
MaxTorque = 50.0

[node name=`"CollisionShape3D`" type=`"CollisionShape3D`" parent=`"Hand_R`"]
shape = SubResource(`"Shape_Hand`")

[node name=`"MeshInstance3D`" type=`"MeshInstance3D`" parent=`"Hand_R`"]
mesh = SubResource(`"Mesh_Hand`")
"
    $content = $content + "`n" + $handsNodes
}

Set-Content -Path 'Scenes/ActiveRagdoll.tscn' -Value $content
Write-Host "Patched successfully!"

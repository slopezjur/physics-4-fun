# Required integration seam for godot_rl_agents: AIController3D is a GDScript base class, and
# Godot does not support a C# script extending a GDScript one, so this file has to exist and be
# GDScript. It is deliberately kept to almost nothing - every method just forwards into the C#
# RagdollRLBridge (Source/RL/RagdollRLBridge.cs), which is where all the real logic lives, matching
# the rest of this project's C#-first convention. Do not add behavior here; add it to the bridge.
extends "res://addons/godot_rl_agents/controller/ai_controller_3d.gd"

## Set in the editor to the sibling RagdollRLBridge node.
@export var bridge: Node


func get_obs() -> Dictionary:
	return {"obs": bridge.ComputeObservations()}


func get_reward() -> float:
	return bridge.DrainReward()


func get_action_space() -> Dictionary:
	return {
		"bone_offsets": {"size": bridge.ActionSize, "action_type": "continuous"},
	}


func set_action(action) -> void:
	bridge.ApplyAction(action["bone_offsets"])


## Per-transition side channel. Carries the environment-side provenance on the first few calls
## (recorded into the run manifest) and the per-term reward decomposition on each terminal
## transition (logged to TensorBoard). Empty on every other step.
func get_info() -> Dictionary:
	return bridge.GetStepInfo()


func get_done() -> bool:
	return bridge.IsDone()


func set_done_false() -> void:
	bridge.ClearDone()


## The base class's reset() only clears its own n_steps/needs_reset bookkeeping - the actual
## physical reset is driven by RagdollRLBridge polling needs_reset itself (see
## RagdollRLBridge.CheckAndConsumeResetRequest), which calls back into this method afterward purely
## to clear that bookkeeping. Nothing else needs to happen here.
func reset() -> void:
	super()

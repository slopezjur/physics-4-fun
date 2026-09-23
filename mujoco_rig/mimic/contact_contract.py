"""Contact observation semantics are part of the policy contract, not its width."""
LEGACY = "legacy"
SUPPORT_V2 = "support_v2"
SAMPLING = "last_physics_solve; reset_forward; preserve_on_launch"
SUPPORT_BODIES = [["Foot_L", "Toe_L"], ["Foot_R", "Toe_R"]]


def mode_from_contract(contract):
    schema = contract.get("observation_contract")
    if schema in ("mimic_stand_v1", "mimic_stand_target_pd_v1"):
        if "ground_contact_sampling" in contract or "support_bodies" in contract:
            raise ValueError("Legacy observation contract cannot override contact semantics")
        return LEGACY
    if schema == "mimic_stand_target_pd_v2" and contract.get("ground_contact_sampling") == SAMPLING \
            and contract.get("support_bodies") == SUPPORT_BODIES:
        return SUPPORT_V2
    raise ValueError("Unsupported contact observation contract")

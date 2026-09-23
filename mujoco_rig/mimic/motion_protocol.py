"""Fixed tracking criteria for the isolated forward/back motion experiments."""
from .motion_contacts import TOE_REFERENCE_SCHEMA
GATES = dict(episodes=8, seconds=3., max_root_error_m=.15, max_foot_error_m=.1,
             max_joint_rmse_rad=.25, min_each_foot_lift_m=.03)
REFERENCE_SCHEMA = "mimic_step_reference_v5"


def validate_motion_reference(metadata):
    if metadata.get("schema") not in (REFERENCE_SCHEMA, TOE_REFERENCE_SCHEMA) or metadata.get("passed") is not True \
            or any(metadata.get("checks", {}).get(key) is not True
                   for key in ("ground_support", "stance_contact", "force_consistency", "root_wrench_consistency",
                               "actuation_feasibility")):
        raise ValueError("Motion training requires v5 ground-support, stance-contact, force-consistency, root-wrench and actuation-feasibility validation")
    if metadata["schema"] == TOE_REFERENCE_SCHEMA and metadata.get("checks", {}).get("sole_contact") is not True:
        raise ValueError("v6 motion training also requires explicit sole-contact validation")
    if metadata.get("force_fit", {}).get("self_clearance_m", 0.) > 0 \
            and metadata.get("checks", {}).get("self_clearance") is not True:
        raise ValueError("Motion training requires the declared self-clearance validation")


def tracking_checks(metrics, cases):
    return dict(survival=metrics["successes"] == GATES["episodes"],
                root_tracking=metrics["mean_root_tracking_error_m"] < GATES["max_root_error_m"],
                foot_tracking=all(c["mean_foot_error_m"] < GATES["max_foot_error_m"] for c in cases),
                joint_tracking=all(c["joint_rmse_rad"] < GATES["max_joint_rmse_rad"] for c in cases),
                foot_lift=all(min(c["foot_lift_range_m"]) >= GATES["min_each_foot_lift_m"] for c in cases))

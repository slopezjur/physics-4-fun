"""Check physical equivalence before accepting ordered Newton/MuJoCo mappings."""
import mujoco
import numpy as np


def collision_pairs(model):
    """Effective broad-phase eligibility, accounting for same/parent body filters."""
    excluded = set(map(int, model.exclude_signature))
    pairs = set()
    filter_parent = not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    for a in range(model.ngeom):
        for b in range(a + 1, model.ngeom):
            ba, bb = sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b])))
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            if wa == wb or ((ba << 16) + bb) in excluded:
                continue
            if filter_parent and wa and wb:
                pa = model.body_weldid[model.body_parentid[wa]]
                pb = model.body_weldid[model.body_parentid[wb]]
                if pa == wb or pb == wa:
                    continue
            if ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                    (model.geom_contype[b] & model.geom_conaffinity[a])):
                pairs.add((a, b))
    return pairs


def model_checks(source, imported):
    checks = {}
    for field in ("nq", "nv", "nu", "nbody", "ngeom", "njnt"):
        checks[field] = getattr(source, field) == getattr(imported, field)
    if not all(checks.values()):
        return checks
    # Ordered topology must match before any same-index mapping is accepted.
    for field in ("body_parentid", "body_jntadr", "body_jntnum", "body_pos", "body_quat",
                  "body_ipos", "body_iquat", "body_mass", "body_inertia", "jnt_type", "jnt_bodyid",
                  "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_stiffness", "jnt_solref",
                  "jnt_solimp", "dof_damping", "dof_armature", "dof_frictionloss", "qpos0", "qpos_spring",
                  "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_friction", "geom_condim",
                  "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "actuator_trnid",
                  "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_dynprm",
                  "actuator_gaintype", "actuator_biastype", "actuator_dyntype", "actuator_ctrlrange",
                  "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited"):
        checks[field] = bool(np.allclose(getattr(source, field), getattr(imported, field), atol=2e-6, rtol=2e-5))
    # Plane size is visual; unused sphere/capsule size components differ in Newton.
    dimensions = {int(mujoco.mjtGeom.mjGEOM_PLANE): 0, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
                  int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2, int(mujoco.mjtGeom.mjGEOM_CYLINDER): 2}
    checks["collision_dimensions"] = all(np.allclose(source.geom_size[i, :dimensions.get(int(t), 3)],
                                                     imported.geom_size[i, :dimensions.get(int(t), 3)],
                                                     atol=2e-6) for i, t in enumerate(source.geom_type))
    checks["collision_eligibility"] = collision_pairs(source) == collision_pairs(imported)
    for field in ("timestep", "gravity", "integrator", "solver", "cone", "jacobian", "iterations",
                  "ls_iterations", "impratio", "tolerance", "ls_tolerance", "disableflags", "enableflags"):
        checks["option_" + field] = bool(np.allclose(getattr(source.opt, field), getattr(imported.opt, field),
                                                   atol=1e-10, rtol=1e-5))
    return checks


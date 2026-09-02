"""The plant a checkpoint was trained against, restored from the checkpoint's own run.

**One list, used by both playback and resume, because getting it wrong is silent.** A policy is only
meaningful against the body it learned on. Every one of these fields reverts to a task default if
nobody restores it, the run then looks completely normal, and the mistake only surfaces as a policy
that mysteriously underperforms.

That has now happened four times on this project:

* `action_scale` reverted 0.15 -> 0.4 on resume, rescaling every action by 2.7x.
* `balance_assist` reverted 1.0 -> 0.0 in playback, putting a checkpoint that holds 75% standing at
  0% and on the floor in two seconds - which read exactly like a broken brain.
* `balance_reaction` reverted True -> False on resume, quietly training against a free torque.
* Solver iterations reverted 8 -> 2, which is a different solver, not a different setting.

The fix is structural rather than a flag to remember: read what the run recorded and put it back.
"""

from __future__ import annotations

import pathlib

# Fields that change what the policy IS, as opposed to how a run is presented. Dotted names walk
# into nested config objects. Anything here read from the task registry instead of the checkpoint's
# own run makes the new run a different experiment from the one it claims to continue.
TRAINED_CONDITIONS = (
    "action_scale",
    "action_rate_limit",
    "obs_joint_vel_clip",
    # **Whether the policy can see joint velocity at all.** A policy trained with slice [55:100]
    # masked and scored with it live is being fed 45 floats it has never seen, and reads as broken
    # rather than as masked - measured at 46.9% for a checkpoint whose training was healthy. This
    # is the same restore-the-plant trap as action_scale and balance_assist, on a new field.
    "obs_joint_vel_enabled",
    "obs_joint_vel_min_range",
    "obs_joint_vel_mask_narrow",
    "obs_joint_vel_narrow_noise",
    "balance_assist",
    "balance_gain",
    "balance_damping",
    "balance_max_torque",
    "balance_reaction",
    "enforce_effort_limit",
    # **Part of the dynamics, not a performance knob.** XPBD iterations decide how much constraint
    # work each tick gets, and the honest-physics line only trains at all at 8 - at 2 the balance
    # reaction cannot be absorbed and the run diverges. A resume that dropped back to 2 would be
    # continuing a lineage on a body it was never trained on.
    "sim.physics.solver_cfg.iterations",
)


def _dig(source, dotted: str):
    """Read a dotted path out of nested dicts (the YAML) or objects (the cfg). None if absent."""
    node = source
    for part in dotted.split("."):
        if isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            if not hasattr(node, part):
                return None
            node = getattr(node, part)
    return node


def _plant(target, dotted: str, value) -> bool:
    """Write a dotted path into the cfg. False if the path does not exist rather than creating it."""
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        if not hasattr(node, part):
            return False
        node = getattr(node, part)
    if not hasattr(node, parts[-1]):
        return False
    setattr(node, parts[-1], value)
    return True


def restore(env_cfg, checkpoint: str, label: str = "run") -> None:
    """Re-apply the checkpoint's own `params/env.yaml` over the task defaults.

    Silent about fields that already match, loud about the ones it changed - the changes are what
    someone reading the log needs, and listing nine unchanged fields every run trains people to
    skip the line.
    """
    run = pathlib.Path(checkpoint).resolve().parent
    env_yaml = run / "params" / "env.yaml"
    if not env_yaml.is_file():
        print(f"[{label}] WARNING no {env_yaml}; using TASK DEFAULTS, which may not be the plant "
              "this checkpoint was trained on.")
        return

    import yaml

    with open(env_yaml, encoding="utf-8") as fh:
        trained = yaml.unsafe_load(fh) or {}

    changed = []
    for key in TRAINED_CONDITIONS:
        now = _dig(trained, key)
        if now is None:
            continue
        was = _dig(env_cfg, key)
        if was == now:
            continue
        if _plant(env_cfg, key, now):
            changed.append(f"{key} {was} -> {now}")
        else:
            # Loud on purpose. A path that silently fails to apply is indistinguishable from one
            # that was already correct, and that is precisely how the wrong plant gets trained -
            # this very list shipped with `sim.solver_cfg.iterations`, guessed from the YAML's
            # indentation, which does not exist on the cfg (it is `sim.physics.solver_cfg`).
            print(f"[{label}] WARNING recorded '{key}' has no matching field on the cfg; NOT restored.")

    if changed:
        print(f"[{label}] restored the trained conditions: " + ", ".join(changed))


def apply_overrides(env_cfg, pairs, label: str = "run") -> None:
    """Apply `--set key=value` on top of whatever the trained conditions established.

    Values are parsed against the EXISTING field's type, so `balance_reaction=True` sets a bool and
    `action_scale=0.2` a float. An unknown key raises rather than being ignored: a mis-set condition
    that reads as "no effect" is how a measurement quietly becomes fiction.
    """
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got '{pair}'")
        key, _, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()

        current = _dig(env_cfg, key)
        if current is None and not hasattr(env_cfg, key.split(".")[0]):
            raise SystemExit(f"--set '{key}' is not a field of this task's env cfg.")

        if isinstance(current, bool):
            value = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw

        if not _plant(env_cfg, key, value):
            raise SystemExit(f"--set '{key}' is not a field of this task's env cfg.")
        print(f"[{label}] override {key} {current} -> {value}")

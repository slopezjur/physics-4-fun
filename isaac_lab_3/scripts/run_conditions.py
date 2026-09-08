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
    # Changes what "tracking" MEANS, so a checkpoint scored under the other shape is being judged
    # on a different objective than it trained on.
    "drive_overspeed_sigma",
    # Changes how hard the drive tracks its target, i.e. the actuator itself.
    "target_damping",
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
    # **Part of the plant, not a performance knob.** Godot's gravity feed-forward carries
    # 45-68% of the body's holding torque; a checkpoint trained with it and scored without is
    # being run on an actuator with half the authority it learned against.
    "gravity_feedforward",
    "balance_gain",
    "balance_damping",
    "balance_max_torque",
    "balance_reaction",
    # Part of the plant the policy trained against: a checkpoint trained on a spread of
    # command gains and scored on a single one is being measured on a body it did not learn.
    "action_scale_range",
    "enforce_effort_limit",
    "hill_max_shortening_velocity",
    # **The integrator IS the plant, and these were missing.** Measured 2026-09-05: Isaac ran at
    # `sim.dt` 1/120 while Godot runs at 240 Hz, and the foot penetrates the floor 14.55 mm median
    # at 120 Hz against 3.30 mm at 240. A checkpoint trained at 1/240 and scored without these
    # replays on a floor four times softer than the one it learned on - which produced a "HOP"
    # verdict on a policy that had never been run at its own timestep. `decimation` travels with
    # `dt` because the two together set the POLICY rate, which the frozen contract fixes at 60 Hz.
    "sim.dt",
    "decimation",
    # Changes what the policy is allowed to READ, so a checkpoint trained with the flags stuck and
    # scored with them live is being judged on a different task.
    "obs_contact_stuck_prob",
    "action_latency_steps",
    # Per-joint plant randomisation: same restore trap as the whole-body ranges. A policy
    # trained against a SPREAD of per-joint responses and scored against one fixed response is
    # being judged on a plant it never saw.
    "per_joint_action_scale_range",
    "per_joint_latency_steps",
    "reset_pitch_noise",
    "reset_ang_vel_noise",
    # How much torque the Hill law leaves the actuator. Raw velocity holds Isaac at a
    # median 58% of ceiling through a gait; Godot, filtering at 0.15, reports 0.97-1.00.
    "hill_velocity_filter",
    # **Part of the dynamics, not a performance knob.** XPBD iterations decide how much constraint
    # work each tick gets, and the honest-physics line only trains at all at 8 - at 2 the balance
    # reaction cannot be absorbed and the run diverges. A resume that dropped back to 2 would be
    # continuing a lineage on a body it was never trained on.
    "sim.physics.solver_cfg.iterations",
)


#: Where the per-joint actuator gains live, in the YAML and in the cfg object. Tried in order.
_STIFFNESS_PATHS = ("robot.actuators.all.stiffness", "scene.robot.actuators.all.stiffness")


def _check_plant(env_cfg, trained: dict, label: str) -> None:
    """Warn when a checkpoint was trained on different actuator gains than are live now.

    **The plant is NOT part of `TRAINED_CONDITIONS` and cannot be, because it does not live on
    `env_cfg`.** The gains come from `p4f_newton/assets.py`, which is module-level and global, so
    `restore()` has nothing to put back - every checkpoint is scored on whatever plant the working
    tree currently defines.

    That became a live hazard on 2026-09-03, when the actuator gains were rescaled per-joint onto
    Godot's measured Stable-PD plant (spine 600 -> 121.45, foot 1200 -> 76.93). Every checkpoint
    from before that change trained on gains ~6.5x stiffer body-wide. Scoring one now produces a
    number that looks comparable with its own history and is not - the same class of silent
    invalidation that has repeatedly cost this project a night's conclusions.

    A warning rather than a refusal: scoring an old checkpoint on the new plant is exactly what you
    want when the question is "does the old brain survive the new plant", and only wrong when the
    number is filed alongside its pre-change scores.
    """
    for path in _STIFFNESS_PATHS:
        was = _dig(trained, path)
        now = _dig(env_cfg, path)
        if isinstance(was, dict) and isinstance(now, dict):
            break
    else:
        return

    worst_joint, worst_ratio = None, 1.0
    for joint, trained_value in was.items():
        live_value = now.get(joint)
        if not isinstance(live_value, (int, float)) or not isinstance(trained_value, (int, float)):
            continue
        if trained_value <= 0.0 or live_value <= 0.0:
            continue
        ratio = max(trained_value / live_value, live_value / trained_value)
        if ratio > worst_ratio:
            worst_joint, worst_ratio = joint, ratio

    # 1% absorbs YAML float round-tripping without hiding a real rescale.
    if worst_joint is None or worst_ratio < 1.01:
        return

    print(
        f"[{label}] WARNING actuator PLANT CHANGED since this checkpoint was trained: "
        f"{worst_joint} stiffness {was[worst_joint]:g} -> {now[worst_joint]:g} "
        f"({worst_ratio:.2f}x, the largest of {len(was)} joints). The gains live in "
        "p4f_newton/assets.py, not on env_cfg, so restore() cannot put them back. This score is "
        "NOT comparable with scores this checkpoint earned before the change."
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

    _check_plant(env_cfg, trained, label)


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

        # **Tuples fall through to the string branch and detonate at the first reset.** Measured
        # 2026-09-05: `--set action_scale_range="(0.7,1.4)"` stored the literal STRING, which then
        # unpacked into nine characters inside `_reset_idx` - "too many values to unpack" fifteen
        # minutes into a run, after the GPU had already booted 24576 environments.
        if isinstance(current, tuple):
            parts = [p for p in raw.strip().strip("()[]").split(",") if p.strip()]
            value = tuple(float(p) for p in parts)
            if len(value) != len(current):
                raise SystemExit(
                    f"--set '{key}' expects {len(current)} numbers, got {len(value)} from '{raw}'")
        elif isinstance(current, bool):
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

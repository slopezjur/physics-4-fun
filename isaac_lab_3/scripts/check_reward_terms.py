"""Report which reward terms each task actually ends up with, and fail if a regulariser vanished.

**This exists because the bug it catches has already shipped twice.** The 2.3.2 `WalkEnv` overrode
`_get_rewards` wholesale and silently lost `action_clip` - the barrier that stops PPO's Gaussian mean
drifting outside the clip range - and every quality metric reported the run as healthy, because
losing a penalty makes the numbers look BETTER. A subclass that edits an inherited reward dictionary
by key is safer than one that rebuilds it, but it is still one typo away from the same failure.

Static, not a simulation: it reads the source, so it costs nothing and can run before a long
unattended chain rather than after it.

    python isaac_lab_3/scripts/check_reward_terms.py

Exits non-zero if any task is missing a term from REQUIRED.
"""

from __future__ import annotations

import pathlib
import re
import sys

TASKS = pathlib.Path(__file__).resolve().parent.parent / "p4f_newton" / "tasks"

# Terms every task needs whatever its objective is. `action_clip` is the one with history, but the
# others are the same class of quiet loss: dropping a cost never looks like a regression.
REQUIRED = ("action_clip", "action_rate", "joint_vel", "effort", "termination")


def base_terms(source: str) -> list[str]:
    """Keys from the literal dictionary `StandEnv._reward_terms` returns."""
    start = source.find('        return {\n            "alive"')
    if start < 0:
        return []
    block = source[start : source.index("\n        }", start)]
    return re.findall(r'"(\w+)":', block)


# Which task each env class inherits from, read from its `class X(YEnv)` line. Resolving the chain
# matters: Run inherits WALK, which deletes five of Stand's terms and adds four of its own, so
# composing Run's edits directly onto Stand's list reports terms Run does not actually have.
PARENT = {"walk": "stand", "perturb": "stand", "run": "walk"}


def resolve(name: str, sources: dict[str, str], base: list[str]) -> list[str]:
    """Terms a task ends up with, composing every edit along its inheritance chain."""
    if name == "stand":
        return list(base)

    parent = PARENT.get(name)
    terms = resolve(parent, sources, base) if parent else list(base)

    src = sources[name]
    deleted = re.findall(r'del terms\["(\w+)"\]', src)
    added = re.findall(r'terms\["(\w+)"\]\s*=', src)
    terms = [k for k in terms if k not in deleted]
    terms += [a for a in added if a not in terms]
    return terms


def main() -> int:
    stand_src = (TASKS / "stand" / "stand_env.py").read_text(encoding="utf-8")
    inherited = base_terms(stand_src)
    if not inherited:
        print("could not parse StandEnv._reward_terms - has its shape changed?")
        return 1

    sources = {f.parent.name: f.read_text(encoding="utf-8") for f in TASKS.glob("*/*_env.py")}

    failures = 0
    for name in sorted(sources):

        terms = resolve(name, sources, inherited)

        missing = [k for k in REQUIRED if k not in terms]
        status = "ok" if not missing else f"MISSING {', '.join(missing)}"
        failures += bool(missing)
        print(f"  {name:<6} {len(terms):>2} terms  {status}")
        print(f"         {', '.join(sorted(terms))}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

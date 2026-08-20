# Vendored third-party code

Everything in this directory is **not** part of Physics4Fun. It is a copy of the Godot editor
plugin from the `godot_rl_agents` project, checked into this repository so that a fresh clone can
open and run the RL scenes without a separate plugin install.

| | |
|---|---|
| Project | godot_rl_agents |
| Copyright | (c) 2021 Edward Beeching |
| License | MIT — see [LICENSE](LICENSE) |
| Upstream | https://github.com/edbeeching/godot_rl_agents |
| Paired Python package | `godot-rl==0.8.2` (pinned in [rl/requirements.txt](../../rl/requirements.txt)) |

`plugin.cfg` reports `version="0.1"`, which is the plugin's own internal string and does not track
upstream releases — the pinned pip version above is the reliable identifier for which generation of
the toolkit this project is built against. The GDScript side here and the Python side installed
from PyPI speak the same socket protocol and have to be upgraded together.

## What this provides

The `Sync` node and `AIController3D` that `Scenes/RL/*.tscn` instantiate, and the counterpart to the
`godot_rl` package that `rl/train.py` imports (`from godot_rl.core.godot_env import GodotEnv`).
`Source/RL/RagdollRLBridge.cs` is *our* code and bridges this plugin's obs/action/reward/done
contract to the ragdoll; see `docs/RL-DESIGN-NOTES.md` for where that seam sits.

## If you update it

Replace the directory wholesale from upstream, keep `LICENSE` and this file, and bump
`godot-rl` in `rl/requirements.txt` to the matching release. Two known behaviours in this
generation are worked around on our side rather than patched here, so that this copy stays a clean
upstream snapshot — both are documented in `docs/RL-DESIGN-NOTES.md`:

- the truncation flag is computed and then discarded, handled by `TruncationBootstrapWrapper` in
  `rl/train.py`;
- a done step returns the terminal observation where SB3 expects the reset one.

Re-check both after any upgrade.

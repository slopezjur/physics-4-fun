# Frozen ankle response comparison - 2026-10-09

Question: do the learned ankle heads' post-impact commands contribute to the
150 N X recovery scores, beyond the stance they establish before impact?

Use frozen heads 921009 and 931009 from `learned-ankle-20261009-01`.
For each head, evaluate 16 worlds at each of the two saved onsets (1.0 and
1.7 seconds): eight reset poses, positive and negative X, six force intervals,
six-second episodes. This adds 64 cases. The normal branches already exist.

Execute the normal learned head through the first force interval. Starting at
the next control decision, issue the scalar latent commands recorded at matching
frames in that same head's quiet rollout. Preserve tanh, slew, target limits and
the two-interval delay. The frozen original source controller still computes its
actions from live observations throughout. This is a diagnostic intervention;
the hit clock and replay are not a proposed deployable controller.

Require exact normal/replay trace prefixes before the intervention, identical
physical states for the two queued intervals after the switch, exact replayed
latents, matching quiet additions while active, unchanged force windows and
independently reconstructed foot geometry and recovery outcomes. Cache each
saved NPZ once during verification. Record case-level rescues and losses, not
only aggregate differences. Seal inputs and outputs, and preserve existing code.

The conclusion applies only to the ankle head's response in these 64 cases.
The source controller remains reactive; removing the head's reaction does not
remove all feedback. Six-second settling is not natural recovery or stepping.
No PPO, physical-limit changes, model export, or Arena promotion in this step.

# Bounded feasible-correction PPO pilot - 2026-10-08

Freeze the retained queue-masked seed-451028 proposal and f832bab8 Stand anchor. Train a separate correction policy through the verified feasible-command placement. Initial correction is exactly zero. Before fitting, require nominal qpos/issued parity against the archived 540-frame reference and sixteen 30-second quiet cases identical to the anchor.

Policy inputs are the existing 421 values normalized by the proposal's frozen statistics, plus previous addition/0.1, nominal slew-limited addition/0.1 and normalized episode frame: 488 values. Include the full delayed queue in this new policy. Actor and critic each use two 128-unit ELU layers. Actor output is 33 corrections in radians; its final linear layer starts at zero. Critic's final layer also starts at zero. Shared initial actor seed 521027; independent sampling/optimizer seeds 521028 and 531028, each from identical actor weights and fresh optimizers.

Each run: 32 collections of 64 frames across 12 active worlds (two instances of poses 0-5), 24,576 transitions. Entire poses 6 and 7 are excluded from actor and critic losses. Gaussian std 0.005 rad, actor rate 3e-6, critic rate 3e-4, three epochs, minibatch 256, PPO clip 0.2, gamma 0.995, GAE lambda 0.98, sampled approximate-KL stop 0.01 and full-batch analytic KL diagnostics. Store the unprojected sampled correction and its Gaussian log probability. Quiet frames are excluded from actor loss. No correlation or action-repeat changes.

Use broader motor/COM training reward scales 0.10 rad/0.04 m with prior weights and failure/success treatment. Preserve all physical bounds, +/-0.02-rad addition slew, queue delay, quiet bypass and original full-skill gates. Log component rewards and whether sampled corrections survive projection. Initial/final actors and final critics only.

Evaluate both final deterministic policies on eight nominal cases and sixteen 30-second quiet cases per seed using the actual correction adapter. Recompute physical gates from traces independently. Varied timing is conditional on nonzero complete-skill success with at least 4/8 supported holds and 4/8 returns. Zero-correction reference and earlier PPO studies are context, not a matched comparison of identical trainable architectures. No claim that this pilot isolates architecture from action placement.

Exactly two bounded fits. No Arena update, export, C# compilation, teacher fitting, spring/torque changes or longer training. Report physical outcomes even if projection transmission improves without skill success.

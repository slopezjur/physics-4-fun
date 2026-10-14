# Bounded arms-clear rollout learning

Start from the retained queue-masked seed-451028 learner (4/8 supported holds, 4/8 returned Stand, 0/8 full skills). Keep the f832bab8 Stand anchor, normalization, quiet bypass, target bounds, slew limit, two-command delay and zero passive springs. No teacher queries or supervised target loss.

Before training, require the new adapter's deterministic mean policy to exactly reproduce the retained nominal eight-pose qpos and issued-command traces across the full episode. Do not train if parity fails.

Two independent fresh optimizer/sampling seeds, 501028 and 511028. Native MuJoCo only, 12 active worlds (two instances each of training poses 0-5); poses 6 and 7 remain inactive and excluded from every loss. Per run: 32 collection/update cycles, 64 control frames per collection, 768 transitions per batch, 24,576 transitions total. PPO has three epochs, minibatch 256, actor learning rate 0.00001, critic learning rate 0.0003, fixed Gaussian exploration std 0.005 rad per output, ratio clip 0.2, discount 0.995, GAE lambda 0.98 and actor approximate-KL early-stop 0.01. Save initial/final weights only.

Sample unbounded raw goal-relative actions and store their unmodified Gaussian log-probabilities; the physical adapter applies existing clipping/fade/slew. Quiet bypass frames contribute to critic training but not actor loss. The actor retains its queue mask. The critic additionally observes the full queue, previous addition and episode time. Physical failures and the finite nine-second episode boundary are terminal; GAE must not cross resets.

Dense reward uses measured motor tracking (0.45), COM tracking (0.20), foot-position retention (0.10) and supported posture (0.25). Tracking/COM/feet use exponentials at their existing gate scales. Failed physical steps receive -2. A completed full hold-and-return episode receives +10. Hold and return gates are exactly the frozen evaluation gates; no success credit for survival alone. The reward is a learning signal, not admission.

Evaluate both final deterministic learners on all eight nominal cases and 30-second quiet retention. No teacher, exploration noise or optimizer runs during evaluation. Preserve the full tracking gate. Only a learner that improves full-skill success without losing the existing supported-hold and return counts warrants varied-timing evaluation. No transfer, hit training, Arena change, export or C# build in this pilot.

# First-collection PPO gradient audit - completed 2026-10-09

Decision: the first PPO update is mechanically correct and is not uniformly
dominated by a constant offset. Do not change branch weighting, actor learning
rate or reward on this evidence. Locate the later training behavior before
another candidate configuration. Arena remains unchanged.

Protocol: [RECOVERY_PPO_GRADIENT_PLAN.md](RECOVERY_PPO_GRADIENT_PLAN.md).

## Exact reconstruction

Both seeds' first collections were reconstructed from saved evidence: 64
intervals by 12 training worlds, 768 samples per seed, 1,536 total. The separate
reset RNG reproduced each ordered reset record exactly. An independent check
assigned episodes using each world's own record sequence. Neither first
collection contained a terminal/reset event after the initial assignments;
these results therefore do not exercise asynchronous reset reconstruction.

The global RNG was restored by recreating the initial critic and drawing all
64 full 16-world Gaussian action batches. Every saved training action matched
exactly. Replaying the original three epochs / nine minibatches reproduced:

- Every recorded first-update KL value, without a rollback.
- All saved post-update actor conditional means, bit for bit.
- All saved post-update critic values, bit for bit.
- The minibatch permutations, using the reconstructed RNG state.

This replay used existing data and ephemeral models. It performed 18 actor and
18 critic optimizer steps across the two seeds to reproduce historical updates;
it collected no new transitions, ran no physical simulations and saved no policy
checkpoints. This is a diagnostic replay, not two newly trained candidates.

## Branch credit and gradients

Seed 921009 began with 4 quiet, 5 +X and 3 -X worlds. They contributed 256,
320 and 192 samples. Mean normalized advantages were +0.508, -0.076 and -0.550.
Their weighted ascent-gradient norms were 0.0734, 0.0618 and 0.0542. All three
branch gradients were positively aligned with one another; quiet/+X cosine
was 0.658, quiet/-X 0.880, and +X/-X 0.396. The summed gradient retained 88.0%
of the sum of branch norms. Quiet credit was not canceling the pushed gradients.

Seed 931009 began with 2 quiet, 7 +X and 3 -X worlds: 128, 448 and 192 samples.
Mean normalized advantages were +0.636, +0.069 and -0.584. Weighted gradient
norms were 0.0166, 0.0885 and 0.0527. Quiet/-X gradients opposed one another
(cosine -0.835), while +X/-X alignment was weakly negative (-0.102). The summed
gradient retained 62.6% of the sum of branch norms. The largest contribution
came from +X, not quiet samples.

These norms use each branch's actual contribution to the whole-batch mean,
not independently normalized branch objectives. A negative branch-average
advantage does not mean all of its actions are discouraged, nor does the sign
of the output-bias gradient determine the mean change in every state.

At the zero-initialized final layer, earlier hidden-layer gradients are exactly
zero for the initial score-function calculation. Independent analytic gradients
matched autograd within 1.82e-8. Hidden features still contribute through the
last-layer weights. After later minibatches, hidden layers can also change.

## Actual first-update mean changes

For seed 921009, the mean conditional change was +0.000134 latent units, with
RMS 0.001471. Its constant component accounted for only 0.83% of squared change
over the collected states. Mean changes by branch were +0.000746 quiet,
-0.000748 +X and +0.000786 -X.

For seed 931009, the mean change was -0.000504, with RMS 0.000685. Its constant
component accounted for 53.99% of squared change. Branch means were -0.000134
quiet, -0.000809 +X and -0.000036 -X.

This corrects the stronger suggestion that PPO immediately learns only a shared
ankle offset. One seed's first update varied substantially across observed
states. State variation is not proof of causal impact response: poses, random
action memory and branch membership also differ. These are conditional latent
means, not physical ankle angles or paired quiet/pushed recovery outcomes.

The gradient's infinitesimal plain-gradient response is also distinct from the
actual nine Adam updates. Their outputs are recorded separately in the evidence.

## Weighting sensitivity

A saved-data calculation gave each branch equal total gradient weight without
changing actions, advantages or features. Its gradient cosine to the original
was 0.996 and 0.966 for the two seeds. Subtracting the within-branch advantage
mean instead gave 0.989 and 0.904. Neither calculation constitutes an optimizer
run or a physical recovery result; neither isolates a sufficient explanation
for the later failure. In particular, equal weighting alone is not supported
as a demonstrated fix.

Both first collections used the initial zero critic. Their behavior cannot
establish whether later learned value estimates distort credit. Only collections
1 and 32 were retained in the preceding experiment; the middle sequence needed
to locate a transition in behavior is missing.

## Verification and evidence

The independent audit passed both seeds and all 1,536 samples, checking 16 input
hashes and four output hashes, branch labels, reset memory, exact saved optimizer
outputs, permutations, analytic gradient decomposition and constant/variable
change identities. The later weighting sensitivity records its own source and
input hashes. Original evidence remains unchanged.

The reconstruction and gradient calculation took 1.53 seconds after imports.
Core evidence is about 0.8 MiB; no model checkpoints or physical traces were
added. No implementation repair or physical rerun was needed.

Evidence: `logs/mimickit-training/recovery-ppo-gradients-20261009-01/`.
See audit.json, results.json, weighting-sensitivity.json and the compact per-seed
arrays. These findings do not supersede rejection of the half-noise candidates.

## Next priority

Reproduce the existing two half-noise training streams with compact diagnostics
at all 32 collection updates. Keep seeds, initial weights, samples, reset logic,
objective and optimizer settings fixed. Require identical historical final
actor and critic weights, first/last batch evidence, reset records and numerical
training history before interpreting added measurements. Exclude only wall-clock
fields from equality checks. This is instrumentation of the existing result,
not a new hyperparameter attempt or a longer budget.

Record explicit quiet/+X/-X labels and episode ages during collection; branch
credit and actor-gradient contributions; actor changes on a fixed set of saved
training-pose observations; and critic/GAE statistics. Diagnostic computations
must not consume random draws or change optimizer state. Save compact metrics
and labels, without intermediate policy checkpoint series.

Locate when branch-specific behavior shifts or useful response remains weak,
then choose one targeted change based on that evidence. The previous evaluation
suite need not be repeated if exact final-weight and stream parity passes.
This would repeat only the original bounded training streams, approximately
two minutes per seed on the current setup, with no Arena change or new promotion.


## Successor completed

Both full historical training streams were reproduced with per-update diagnostics.
See [RECOVERY_UPDATE_TRACE_PROGRESS.md](RECOVERY_UPDATE_TRACE_PROGRESS.md)
for exact parity, the early/late exposure findings and the next actor-weighting experiment.

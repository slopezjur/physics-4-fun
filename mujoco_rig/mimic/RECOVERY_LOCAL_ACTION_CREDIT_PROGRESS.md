# Paired local action credit - completed 2026-10-09

Decision: local pushed-action credit points in the useful direction throughout
this diagnostic. The next investigation is how actor updates convert that
credit into conditional response. These results do not support a pushed-credit
failure as the immediate explanation for unsuccessful recovery learning.

Protocol: [RECOVERY_LOCAL_ACTION_CREDIT_PLAN.md](RECOVERY_LOCAL_ACTION_CREDIT_PLAN.md).

## Experiment and verification

Both weighted actors and their own final critics remained frozen. At training
onsets 44/74 and poses 0-5, identical source-derived starts were paired with
first innovations of minus/plus HALF_SIGMA. Later Gaussian draws were shared
between the two arms, while each actor reacted to its own evolving body state
and latent memory. Quiet poses were counted once, without signed duplicates.

All 64 rollouts / 576 continuations / 288 paired comparisons passed independent
verification. There were 384 pushed and 192 quiet continuations. The analysis
also retained 72 actor/start averages over four future-noise sequences each;
these repeated sequences are not independent physical starts.

The audit verified historical starts, native restoration, equal paired initial
observations/means/values, the two inherited physical intervals, Gaussian draws,
actor/critic predictions, motor limits/delay, memory, geometry, outcomes and
independent GAE/return arithmetic. It checked 608 input and 140 output hashes,
including 582 shared historical inputs. Maximum independent GAE discrepancy
was 1.03e-4 under the established absolute-plus-relative float32 tolerance;
first-state reward/value decomposition error was 9.17e-5. No verifier repair
or repeated physical run was needed.

## Pushed action rankings

For each actor, all 96 pushed pairs agreed across first-state GAE, same-window
lambda-weighted reward, observed 64-interval return, bootstrapped 64-interval
return and full measured episode return:

- +X pushes preferred the negative first innovation in 48/48 pairs per actor.
- -X pushes preferred the positive first innovation in 48/48 pairs per actor.
- All 24 pushed start conditions per actor retained the same preference across
  all four future-noise sequences. All 48 pushed actor/start averages agreed.
- These comparisons remained strict under the predeclared 1e-3 numerical sign
  band. This band is not a statistical significance threshold.

The first action changed settled recovery in 16 pushed pairs for seed 921009
and 24 for seed 931009. GAE and full return preferred the recovering arm in all
40 cases. This tests local paired action credit, rather than comparing unrelated
complete trajectories or larger scripted pulses.

For completeness, negative/positive first-action arms recovered 22/96 versus
24/96 pushed cases for seed 921009, and 29/96 versus 25/96 for seed 931009.
Pooling force directions hides the opposing preferred actions. These are
diagnostic interventions on frozen policies, not new learned-policy scores.

## Quiet cases and limits

Quiet GAE and full-return rankings disagreed in 12/48 and 10/48 individual pairs,
and 5/12 and 6/12 four-noise actor/start averages. GAE reversed the same-window
lambda-reward ranking in 10/48 and 13/48 pairs, or five averaged starts for each
actor. These remain separate measurements of different reward/value targets.

Seed 921009 retained quiet Stand in all 48/48 cases in each arm. Seed 931009
retained 46/48 with the negative innovation and 47/48 with the positive one.
In the one changed quiet outcome, GAE did not favor the recovering arm; its
same-window reward component also disagreed, so this does not isolate critic
error. Four future-noise sequences do not establish calibrated expected values.

The pushed results do not reconstruct historical PPO advantages or show that
the original optimizer received this balanced paired dataset. They establish
useful local credit under the tested frozen states and continuations. Whether
that information becomes a reliable feedback policy is still unresolved.

No corrective steps occurred. No optimizer ran and no policy checkpoint was
created. Both weighted candidates remain rejected; Arena is unchanged.

## Cost and evidence

Physical rollouts took 229.30 seconds after setup. Evidence occupies about
147.5 MiB, including compressed physical traces and first-window features.
Analysis rules and tolerances were frozen before inspecting outcomes.

Evidence: `logs/mimickit-training/recovery-local-action-credit-20261009-01/`.
See audit.json, audit-pairs.json, paired-credit.json, results.json,
historical-inputs.json, completed.json and per-continuation traces/reports.

## Next priority: controlled offline actor updates

Use the saved first-decision data, 288 samples / 144 pairs per actor, without
new physics. Start scratch copies from each frozen actor and freeze its critic.
Compare ordinary whole-batch-normalized GAE with pair-centered GAE divided by
the same original standard deviation. Keep saved observations, actions and old
likelihoods fixed. Pair centering is a diagnostic alternative, not an admitted
training change.

First verify the analytic invariant: equal/opposite innovations at identical
observations have opposite score gradients. Subtracting the shared pair mean,
while retaining the same denominator, must leave the initial full-batch actor
gradient unchanged. Any apparent improvement in that initial gradient would
signal a computation or normalization confound.

Decompose quiet/+X/-X contributions and state-output sensitivities. Then compare
the two scratch objectives with identical fresh Adam state, actor learning rate
1e-4, fixed minibatch permutations, three epochs, gradient clipping and the
existing KL guard. With 288 diagnostic samples and minibatch size 256, each
epoch has a 256-sample and a 32-sample minibatch: at most six attempted updates
per actor/objective arm, or 24 maximum across the four arms. The KL guard may
stop an arm early.
This is explicitly not a replay of the unavailable historical optimizer state
or the original 768-sample collections.

Measure per-minibatch departure from the full gradient, policy-mean movement
toward the independently preferred action at each pushed start, quiet drift and
the pushed-minus-quiet response at matching pose/timing. The question is whether
useful opposing credit produces conditional behavior or mostly shared movement,
and whether separating paired samples introduces avoidable update variance.
Retain only compact diagnostic evidence, without model checkpoint series,
physical evaluation, reward changes or Arena replacement in this next step.

## Successor completed

All four offline actor-update arms and their independent replay are complete.
See [RECOVERY_LOCAL_ACTOR_UPDATES_PROGRESS.md](RECOVERY_LOCAL_ACTOR_UPDATES_PROGRESS.md).
The initial gradient invariant passed. Pair centering reduced initial
minibatch imbalance but increased quiet movement alongside the useful relative
push response. Shared movement exists in the full gradient before minibatching
or Adam. Next: independently verify a quiet-preserving gradient projection and
its small finite effect before a new physical learning comparison.

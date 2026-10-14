# Fresh learning with expanded quiet coverage

Date: 2026-10-09. Approved successor to RECOVERY_QUIET_COVERAGE_PROGRESS.md.
Run seeds 921009 and 931009 from the original zero head, 24,576 collected
transitions each, comparing 48 quiet constraints with the sealed lower-rate
12-constraint runs and zero-head physical evaluations.

## Single controlled change

Use the exact audited bank from recovery-quiet-coverage-20261009-01/bank.npz:
poses 0-5, training onsets 40/44/48/54/66/74/82/90, zero action memory and
zero initial target. Original12 anchors at44/74 remain included unchanged.
Reserved poses6/7 and evaluation onsets60/102 stay outside the bank. Reuse
the frozen bank throughout learning; do not regenerate it from learned states.

Keep actor Adam1e-4, critic Adam3e-4, initial actor seed920009, fixed noise,
reward, observations, recurrent command memory, action limits, source policy,
training reset distribution, sample weighting, advantage normalization,
64-step collections over12 active worlds, three epochs and256-sample
minibatches unchanged. Each seed has32 collections and288 nominal actor slots.
Critic updates continue normally after any rejected actor attempt. Preserve
the exact first collection, seeded critic and all minibatch permutations
against the lower-rate control; later physical collections may diverge.

Project each actual clipped-Adam displacement with float64 direct Jacobian
SVD and cutoff1e-6, apply in float32, then apply exactly one local finite
correction to the pre-step quiet outputs. Retain the original zero-target
bound1e-5, collection KL bound0.01, actor/optimizer rollback on rejection and
stop-actor-for-that-collection behavior. No renormalization, line search,
retry, changed acceptance objective or rate adjustment. Use the feasibility
study's generalized independent QR/reduced-SVD gates at48 rows; primary
projection/correction arithmetic is unchanged. A failed numerical gate stops
the run and is preserved; do not relax it in place.

Use successor sources where the original learner/audit assumes12 anchors.
Reuse the existing physical runner through scoped dependency replacements;
never change the sealed implementation or previous evidence.

## Lightweight response tracking

At initialization and after each of32 collections, evaluate the same36
training probe inputs (12 quiet,24 pushed) and the48 quiet anchors. Save
small numeric arrays, actor hashes and summaries, without intermediate
policy checkpoints. Track raw and paired response, amplitude, historical
reference classification, all48 quiet error and input identity. Probe
inference must consume no randomness and must not change actor parameters,
optimizer state, environment state or acceptance decisions.

The older direction reference is diagnostic only: +X negative, -X positive,
with1e-7 sign band. Both original12 and expanded48 first updates opposed it;
both final lower-rate controls later aligned24/24. Do not train toward the
labels, tune parameters to them or require every intermediate update to
agree. Record the trajectory and judge physical outcomes.

## Evaluation and verification

Use the complete established720-case,45-trace suite: baseline matched-start
checks plus both candidates' deterministic continuous and matched-start
force groups, core timings, sampled core and quiet standing. Verify sampled
noise reconstruction, matched prefixes, measured footstep events and case
accounting through the existing independent physical/objective audit.
Compare each candidate with the same-seed sealed12-anchor control and the
zero-head baseline. No retraining of controls is necessary.

Independently verify source and input hashes, the exact48-state bank, fresh
initial parameters, original first samples/critic/permutations, the exact
first48 candidate from feasibility, first-collection actor/Adam/critic replay,
all compact projection/correction/rollback records and final outputs. Rebuild
boundary probe summaries from their recorded arrays and independently verify
network predictions at the initial and final boundaries.
Intermediate summaries remain observations tied to the actor hash chain;
they do not justify storing full intermediate policies.

After audit passes, produce paired physical gains/losses, quiet retention,
acceptance/rank/norm/KL accounting, first/last collection objective changes
and command generalization on the existing seen/reserved evaluation windows.
Include first decisions and first30 post-hit intervals; keep live masks and
all cases. Distinguish common-input probes from own-trajectory commands.

Snapshot this plan and current local Python sources before execution. Hash
the frozen parent, sidecar, physics/reference files, protocol demonstrations,
consumed historical outputs and external normalization implementation. Check
modern shared-source identity against the12-anchor and feasibility runs.
Require prior audits and completed-manifest identity for consumed older
baseline artifacts without imposing current-source equality on unrelated
old diagnostic scripts. Retain initial/final policy and final critic only,
first/last collections, compact logs and physical evidence. Preserve failed
runs; repairs require successor evidence and unchanged failed sources.

Decision: require reproduced physical gains with quiet retention before any
promotion. Improved constraint fitting, surrogate, direction labels or
accepted-update count alone is insufficient. Reserved poses/timings remain
diagnostic. No rate sweep, longer continuation or automatic Arena promotion.

Evidence: logs/mimickit-training/recovery-coverage-learning-20261009-01.

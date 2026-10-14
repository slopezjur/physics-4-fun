# Controlled offline paired-credit actor updates

Approved successor to RECOVERY_LOCAL_ACTION_CREDIT_PROGRESS.md. Use existing
checked continuations only; no new physics, critic fitting, policy checkpoint
series, production configuration or Arena changes.

## Fixed design

- Frozen weighted actors 921009/931009, each with 288 first-decision samples,
  144 negative/positive action pairs and 36 unique starting observations.
- Onsets 44/74, six training poses, quiet/+X/-X, four shared future-noise seeds.
  Actions, old means, own-critic GAE and full measured returns come from the
  audited recovery-local-action-credit-20261009-01 traces and reports.
- Ordinary advantage: subtract whole-dataset mean and divide by its unbiased
  standard deviation. Centered advantage: subtract each two-action pair mean,
  divide by the SAME original standard deviation. All samples are age zero;
  the previous normalized early-age weighting is therefore uniformly one.
- Verify initial full and branch ascent gradients agree within absolute 2e-6,
  relative 2e-4, and full relative norm difference below 2e-4. Stored float32
  actions are only approximately symmetric; preserve them without adjustment.
- Record unique-state Jacobian Gram matrix, branch gradients and their output
  sensitivities. Evaluate both objectives on all six fixed minibatches at the
  unchanged initial actor. Check complete-pair baseline cancellation and the
  residual contribution of unmatched rows.
- Three fixed permutations from CPU generator seed 961011, shared across both
  actors and objectives. Each epoch has 256 + 32 samples. Fresh Adam per arm,
  learning rate 1e-4, default betas/epsilon, PPO ratio clipping [0.8, 1.2],
  gradient norm clipping 0.5 and whole-dataset KL limit 0.01. Reject and restore
  actor plus optimizer on a KL violation, then stop that arm. At most six
  attempted updates per arm, 24 across all four arms.

## Measurements and interpretation

For every attempted update, compare its raw minibatch ascent gradient against
the full gradient at the SAME current parameters and objective. Preserve loss,
ratios, clipping, pair membership, attempted parameter movement before any
rollback and acceptance decision. Saved mean histories record accepted states.
Report 256-row and 32-row batches separately. Sample-weighted initial minibatch
gradients must reconstruct the full gradient; two sequential Adam updates need
not equal a single full-batch update.

On 36 unique observations per actor, measure latent-mean changes, quiet drift,
constant-change energy, movement toward the independently preferred pushed
action, and pushed-minus-quiet changes at matching pose/onset. A 1e-7 latent
movement band only guards numerical sign counts; it is not a practical-effect
threshold. Preserve magnitudes. These quantities are not ankle radians or
closed-loop recovery measurements.

Pair centering preserves the initial full gradient. After parameters change,
likelihood ratios and PPO clipping can break that cancellation; later arm
differences cannot be assigned solely to baseline sampling variance. The four
noise repeats are not independent starts. This 288-row dataset and fresh Adam
state do not reconstruct historical 768-row PPO batches or optimizer history.
The deliberately paired +/- sigma actions define an empirical interventional
surrogate; its minibatch variability is not an unbiased estimate of historical
PPO gradient variance.

Independent verification reconstructs inputs, formulas, gradients and all
attempted scratch updates. Preserve source/input hashes and compact evidence.
Decide the next bounded experiment from the observed response pattern; do not
promote a policy based on this offline diagnostic.

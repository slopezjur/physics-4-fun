# Instrumented replay of half-noise PPO streams

Approved successor to RECOVERY_PPO_GRADIENT_PROGRESS.md, 2026-10-09.

Reproduce seeds 921009 and 931009 for their original 24,576 transitions each,
using identical initial weights, reset RNG, Gaussian samples, minibatch ordering,
reward and optimizer settings. Do not introduce a new candidate configuration.

Add diagnostics to all 32 collections: explicit force branch and episode age
before each transition, branch and early/late whole-batch actor gradients, reward
and value contributions to GAE, critic target fitting, and learned-drive changes
separated from latent memory. Use autograd.grad without altering live gradients
or consuming RNG. Keep full-network gradient summaries after initialization.

At every update, probe identical saved zero-correction quiet/pushed observations
from training poses, onset frames 44/74 and first 30 continuation intervals.
These fixed observations are diagnostics only; do not add them to training.
Their latent memory is zero. Probe contrasts are not new physical recovery tests.

Require exact historical final actor/critic tensors, all numerical history rows,
all reset records and accounting, and every first/last saved batch array. Original
middle batches were not saved, so do not claim direct equality against missing
historical arrays. Save labels, statistics and probe outputs without another
checkpoint series or duplicated full physics traces. If parity passes, reuse
the already audited evaluations; no Arena or production configuration changes.

Interpret value-target error as bootstrapped-target fitting, not independent
critic calibration. Locate persistent weak response or changes across updates
before selecting one targeted follow-up.

# Recovery-start learning comparison - 2026-10-09

Train two fresh scalar ankle heads from the original frozen Stand source's
continuations, rather than allowing the optimizer to change the pre-impact stance.
Use the previous initial actor seed 920009 and fitting seeds 921009/931009 for a
matched-budget comparison with `learned-ankle-20261009-01`. Reusing seed numbers
does not resume either prior head: both begin at the identical zero-output weights.

Keep the 427-input architecture, conditional Gaussian exploration, physical
limits, direct-recovery reward and PPO implementation unchanged. Each seed gets
32 collections of 64 intervals over 12 training worlds (poses 0-5): 24,576
optimizer transitions, three epochs and 256-sample minibatches. Keep initial and
final actors and final critic only. No intermediate checkpoint selection.

For each reset choose a training onset frame uniformly from 40-90 excluding
55-65, signed 150 N X with equal sign probability and 25% quiet episodes, as in
the preceding pilot. A separate frozen-source environment reconstructs the
prefix with zero ankle correction. Transfer each selected world immediately
after its first force interval, including full native state, delayed commands,
cached contacts and observation/controller memory. Five force intervals remain.
The reset retains the original absolute episode clock and six-second endpoint.
Only subsequent transitions enter PPO. Episode length therefore varies with
source onset, as implied by preserving the original endpoint.

Generate starts on demand, using one scratch environment and a temporary full
snapshot. A native MjData reserves about 15 MiB of arena space: caching every
timing/branch/world would reserve tens of GiB. No large persistent bank or disk
snapshot cache is needed. Record reset choices, prefix simulation work, exact
feature transfer and collected-transition counts. Exclude poses 6/7 from fitting.

Before fitting, require zero-output initial weights to match the preceding pilot;
check mixed asynchronous resets and preservation of unselected worlds. Reproduce
the archived source using restored matched starts at both core evaluation timings
and a quiet start. The validated snapshot mechanism remains unchanged.

For each final head evaluate normal continuously active execution using the
existing 128 deterministic pushes, 32 sampled core pushes and 16 sixty-second
quiet cases. Also evaluate matched source prefixes followed by restored recovery
starts across the same 128 deterministic pushes and 32 six-second quiet cases.
The actor never sees force labels or hit clocks. Matched-prefix takeover is an
evaluation/training intervention only, not an admissible deployed trigger.

Three initial parity rollouts plus 21 per candidate give 45 traces / 720 cases.
Independently audit saved force/control/history/geometry/outcome evidence, fit
budgets and reset schedules. Report core signs, reserved poses, broader-direction
retention, quiet retention and audited stepping for both execution modes.

Admission requires useful core recovery in both seeds without reserved-pose or
quiet regression, and adequate broader-direction retention in normal continuous
execution. Gains only under a privileged takeover cannot qualify. No automatic
export or Arena replacement. If this bounded comparison fails, inspect the
saved response and objective evidence before further training.

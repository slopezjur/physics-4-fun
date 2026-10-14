# Bounded fresh recovery learning with finite quiet correction

Approved successor to RECOVERY_QUIET_ANCHOR_TARGETS_PROGRESS.md. Train fresh
seeds 921009 and 931009 from the same original zero head initialized with
seed 920009, for 24,576 continuation transitions each. Retain the complete
age-weighted recipe, half noise, memory, reset distribution, full episodes,
reward, advantage normalization, critic objective, learning rates, three
epochs, 256-sample minibatches and 288 nominal actor slots per seed.

Keep the projection of the actual clipped-gradient Adam displacement using
the pre-step quiet Jacobian. On EVERY finite projected candidate, form its
current quiet Jacobian and apply ONE minimum-norm Newton correction toward
that update's actual pre-step quiet outputs. Use relative SVD cutoff 1e-6,
float64 algebra, and explicit float32 displacement addition for both stages.
Do not retry, line-search, renormalize, compensate with extra updates or add
an objective-improvement acceptance rule.

Evaluate final corrected candidates against the ORIGINAL ZERO anchor target:
maximum absolute deviation <=1e-5 and collection-old mean KL <=0.01. Do not
reject an intermediate anchor violation before correction. Reject nonfinite
candidates. Retain tentative Adam moments only on acceptance; otherwise
restore both actor and Adam and skip remaining actor slots in that collection.
Critic updates continue as before.

Independently gate projection and correction using Gram eigensystem checks.
Record both stages separately, including pre/intermediate/final anchor outputs,
geometry, correction size, local residual, cumulative zero drift, KL, hashes,
acceptance/rejection reasons and skipped slots. Keep compact diagnostics,
first/last complete batches and initial/final actor plus final critic only.
No intermediate checkpoint series.

Require exact first pre-learning collection, normalized advantages, actor
weights and initial/post-collection critic parity with historical controls.
Verify all 32 minibatch permutations. Updated actors and later trajectories
are expected to differ. Independently replay the first collection's actor,
optimizer and critic work; audit later compact accounting and source hashes.

Run the established 45-trace / 720-case physical suite: continuous and matched
execution, reserved poses/timings, signed 150/300 N X/Y pushes, sampled core
pushes, sixty-second quiet standing and measured footsteps. Compare individual
outcomes with projected-only constrained, weighted and zero controls. Analyze
early issued ankle commands and pushed-minus-quiet differences separately.

Fresh correction from initialization differs from saved ninth-candidate
diagnostics. More accepted updates and changed directions are coupled; update
acceptance is not a recovery result. Require reproducible physical gains and
quiet retention before promotion. No automatic Arena replacement or longer
continuation follows. Preserve all earlier sealed sources and evidence.

Evidence: logs/mimickit-training/recovery-retracted-learning-20261009-01.

# Steps 21-30: on-policy credit check and conditional critic warm-up

Frozen execution comparisons have not produced a complete skill. Expected
projected commands improve the untrained head to 6/8 holds and 5/8 returns;
this is not learning. Common-state physical interventions change reward sign
in 54/175 comparisons when extending one second to two seconds. Frozen
one-step critic ranking agrees with two-second physical changes in 51.25%
overall and 31.25% of held-out comparisons. These comparisons are off-policy;
they do not alone establish that the critic mislearned its own policy.

21. Collect full frozen noisy-policy episodes with features. Zero head: seeds
    611001-611006. Final heads 521028 and 531028: seeds 611001-611002 each.
    Compare each final critic with realized discounted returns under its own
    actor, and a fresh critic with zero-head returns. Episodes are Monte Carlo
    samples; frames are not independent observations.
22. Fit two fresh critics to zero-head full-episode discounted returns. Use only
    poses 0-5 of seeds 611001-611004, all alive frames including quiet. Actor,
    proposal and physics remain frozen. Seeds 611028 and 621028, Adam 3e-4,
    four passes, batches 256, gradient norm limit 1. Retain initial/final critics.
23. Validate both critics on unused zero-head seeds 611005-611006, separately
    for poses 0-5 and held-out poses 6-7. Admit the PPO comparison only if both
    reduce realized-return MSE by at least 20% in both groups relative to fresh
    critics. This gate checks coarse calibration, not action ranking.
24. If admitted, run 24,576-transition correction PPO with warm critic 611028.
25. Replicate with 621028 and evaluate both under changed timing. All PPO
    settings and execution semantics remain the prior pilot's.
26. Verify final warm actors on sixteen quiet 30-second cases each against
    exact anchor qpos/issued commands.
27. Compare motor/COM/foot errors and support failures against shared initial
    and existing rejected actors. Arms-clear motion is not corrective stepping.
28. Run the matched fresh-critic control for 611028 and evaluate physically.
29. Replicate the control for 621028 and evaluate changed timing for controls.
30. Final admission: new seeds 612001-612002 with noisy execution for all four
    actors, earlier deterministic results, and exact quiet retention for controls.

Four bounded PPO fits maximum: 98,304 transitions total. Shared warm-up
rollouts and extra critic fitting are additional resources reported explicitly;
this is not an equal-total-compute efficiency claim. Initial actors are the
same zero-output head. Matched controls/warm runs share sampling seeds and
critic initialization RNG consumption; only initial critic weights differ.
Fresh Adam for every PPO actor and critic. No reward, distribution, rollout
length, springs, limits or actor learning-rate changes.

If step 23 fails, do not fit actors: use the remaining comparisons to diagnose
out-of-sample value errors and terminal/phase targets, documenting the branch
before execution. No automatic extension of fitting to achieve a pass.

Report mean and stochastic outcomes separately. Even a pass establishes only
arms-clear hold/return, not learned single-foot support or 10 kg impact recovery.
Arena promotion requires full nominal and changed-timing skills plus quiet
retention; partial counts are insufficient.

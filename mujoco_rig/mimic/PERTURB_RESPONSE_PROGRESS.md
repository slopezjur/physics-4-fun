# Recovery response block progress - 2026-10-08

Status: 20/20 substantive steps complete. No successful Arena replacement.

Prospective scope: PERTURB_RESPONSE_PLAN.md. Learning protocol:
PERTURB_RESPONSE_LEARNING.md. These sealed plans are not edited after execution.

1. Complete: parent pose 7 departs at 6.817212 s and falls before the second hit.
   The short-flight single hit recovers; the clear-distance hit does not.
2. Complete: no actor or joint-target clipping up to departure; max raw action
   magnitude .553. Only Toe_R_rx has brief torque saturation (5.69% of substeps
   in the contact-to-departure interval), not general actuator saturation.
3. Complete: complete saved PD equations and exact two-frame command queue
   verified. Action changes >.01 on the first frame after contact (16.668 ms).
   Matched no-hit comparison confirms the same timing; precontact action
   difference is at most 1.2e-7. This is not merely normal quiet-action drift.
4. Complete: both feet stay loaded and move only millimeters while COM advances.
   At departure COM x=.208 m exceeds the forward foot/toe box bound (~.160 m).
   Geometry bounds describe the trajectory; they are not a force support polygon.
5. Complete: additive stationary imitation and 80/20 task/AMP reward retained;
   quiet-action rehearsal remains weight 100. No hard gate forces feet stationary.
   PPO clipping was <.1% in prior runs. Delayed credit is a hypothesis to test.
6. Complete: prior mixed curriculum gave pose 7 45 short-distance and 43
   clear-distance throws; all poses/conditions were represented. Missing exposure
   is not an explanation by itself.
7. Complete: frozen horizontal-velocity input gains .75 and 1.25, no physics edits.
8. Complete: normal-gain matched control reproduces 7/8 at 5/9 and 2/6 s.
9. Complete: 96 raw diagnostic cases, 80 admitted. Gain .75 at 2/6 s failed
   backend agreement; all 16 cases excluded. Preserved the evidence and stopped
   treating its raw count as a confirmed regression.
10. Complete: gain 1.25 stays 7/8 at both schedules, native and GPU. No probe
    improves recovery or qualifies for Arena; no new runtime heuristic introduced.
The sealed execution plan includes administrative gates. The substantive-step
ledger below does not count protocol writing, audits, export, or documentation
as experiments. It refines that sequence within the same bounds; no evaluation
threshold changes. Steps 1-10 above are the completed diagnostic comparisons.

11. Complete: seed 401028, matched TD-lambda .98/.995 learning comparison.
    Both completed 100 updates / 409,600 transitions, about 121 s training each;
    independent start, exposure and sample accounting audit passed.
12. Complete: first-seed repeated-hit recovery is 6/8 in both arms, both
    backends admitted. Both regress from parent 7/8; no treatment benefit.
13. Complete: easier single hits retain 8/8 in both directions and both backends.
14. Complete: stronger single hits retain 8/8 in both directions and backends.
15. Complete: 30-second quiet Stand remains 8/8 in both arms/backends. Neither
    arm qualifies; independent recomputation of all 192 physical cases passed.
16. Complete: seed 411028, both arms 100 updates / 409,600 transitions; matched
    starts and full coverage pass. Total block training: 400 updates, 1,638,400
    transitions, 485.114 measured training seconds. No further learning runs.
17. Complete: second seed control 7/8 versus long_return 8/8 repeated recovery,
    admitted on both backends. First seed was 6/8 versus 6/8: improved candidate,
    but not a replicated recipe improvement across both optimizer seeds.
18. Complete: both second-seed arms retain every single-hit condition and
    30-second quiet Stand at 8/8 on both backends. All 192 outcomes independently
    recomputed. Only second-seed long_return qualifies for final gates.
19. Complete: untrained 4/8-second timing rejects the candidate. Parent 7/8;
    second-seed long_return 6/8, both native/GPU admitted. No successor selected.
20. Complete: no recovered lifted-step signatures in any of the five native
    repeated-hit actor groups. Largest foot displacement in successful trials:
    parent 2.196 mm; first control/treatment 2.084/2.166 mm; second control/
    treatment 2.472/2.330 mm. The improvement is standing balance, not a learned
    corrective step. Signature required unloading (<5 N), foot/toe clearance
    >15 mm, then loaded placement displaced >=30 mm and successful recovery.
    This descriptive diagnostic does not alter recovery gates or score naturalness.

Conditional 120-second candidate retention, Arena eligibility and independent
audits remain required delivery gates outside the count of twenty comparisons.
The candidate failed timing, so its conditional 120-second assessment was not
run. Arena keeps f832bab8; no export, restart, C# build or controller change.

Evidence folders under logs/mimickit-training:
- perturb-response-audit-20261008-01
- perturb-response-probe-20261008-01
- perturb-response-probe-review-20261008-01

Arena remains on f832bab8. No C# build or changes to authored physics.

## Closed result

- 400 PPO updates, 1,638,400 transitions, 485.114 measured training seconds.
- 512 new physical evaluation cases: 496 admitted and 16 explicitly excluded
  diagnostic cases. All persisted outcomes independently recomputed.
- One candidate (53da704a, second optimizer seed, TD-lambda .995) reaches 8/8
  at the primary 5/9-second schedule but falls to 6/8 at untrained 4/8-second
  timing, below the retained parent's 7/8. Rejected without export.
- No candidate demonstrates corrective stepping. Successful repeated trials
  move feet only 2-3 mm. Heavy random Arena hits remain outside validated scope.
- About 1.075 GB of compact new evidence; initial/final checkpoints only,
  no periodic copies, no old evidence deleted. Full control equations were
  checked before recorded substep-data compaction; outcome data remain auditable.
- Godot process 10144 remains responsive on the previous f832bab8 bundle.

Final audit: logs/mimickit-training/perturb-response-block-review-20261008-01/report.json.
Training/evaluation runs: perturb-response-learning/evaluation-20261008-01 and
-02. Final timing: perturb-response-final-20261008-01. Foot motion:
perturb-response-foot-audit-20261008-01. Review folders record all recomputations.

Next priority: design the learned recovery task around timing robustness and
actual foot placement, with the existing quiet Stand and final recovery gates
retained. This block does not justify larger action limits, altered delays or
passive springs, nor another longer run of this TD-lambda comparison. Scripted
diagnostic motion remains separate from a learned policy and cannot be presented
as an updated recovery brain.

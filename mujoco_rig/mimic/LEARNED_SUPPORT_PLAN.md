# Learned support block, 2026-10-08

Goal: acquire a neural lift/hold/replant prerequisite for learned foot placement and hit recovery. No claim of natural stepping follows from lifting in place. The Arena parent f832bab8 remains selected.

This successor explicitly permits complete successful trajectories from the previous physical teacher as experimental supervised data. The previous all-16 export gate remains failed and unchanged. Exclude right-foot poses 2, 3, 4; evaluate them nevertheless. Hold out entire poses 6 and 7, both feet, from training and normalization. Reconstructed observations use float32 saved positions and summed foot loads, so they are approximate observations, not the original exact tensors. Issued joint targets are stored labels; no future physical state enters the learner observation.

Freeze four learners: source-relative and goal-relative target prediction, each seeds 421028 and 431028. Each starts with fresh neural weights and Adam state, 4,000 minibatch updates of 512 examples, learning rate 0.001, two 256-unit ELU layers. Same data, sample ordering per seed, original joints/torques, passive springs zero, two-command delay, 1-radian residual envelope. Anchor is current f832bab8. Training labels reconstruct stored issued targets relative to that anchor, rather than treating the old teacher's actor as the current actor.

Twenty substantive experiments, in order:

1. Train source-relative seed 421028; whole-pose held-out prediction.
2. Train goal-relative seed 421028; matched prediction comparison.
3. Train source-relative seed 431028; independent replication.
4. Train goal-relative seed 431028; independent replication.
5-8. Closed-loop ordinary-start lift/hold/replant/return, all sixteen pose/foot cases, each learner.
9-12. Same full coverage with command onset at 3 seconds.
13-16. Same full coverage with command onset at 9 seconds.
17-20. Thirty-second quiet Stand with each learned adapter enabled and the original pose commanded.

Data reconstruction, code, source sealing and bookkeeping do not count as steps. A step is complete only when its actual training or physical evaluation finishes. Goal timing is commanded by the existing measured-support phase scheduler; joint targets come from the neural learner, with no physical optimizer, scripted joint actions, external support or live pose writes. These onset experiments are command-timing experiments, not hit-timing recovery tests.

All-case entry gate: at least one continuous second of valid support with whole swing-foot clearance >=15 mm, swing load <5 N, stance load >=30 N, stance drift <=20 mm, physical survival and final two continuous seconds of original Stand. Quiet gate: all sixteen worlds survive and retain Stand. Report trained poses, held-out poses and excluded-teacher cases separately. No GPU parity or Arena export is warranted if native entry fails. Actual foot placement and impacts remain the next gated work, not completed by this block.

Only initial/final neural weights and compact physical traces are retained. All evidence and source hashes are recorded under learned-support-20261008-01.

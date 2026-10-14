# Adaptive block: complete arm mask and matched task-relative representation

Steps 1-13 found valid teacher label semantics and no widespread normalized-input
clipping on the inherited controller's active states. The physical teacher again
passed all eight nominal and eight varied cases. On teacher training states the
inherited proposal's active RMSE was 0.005 rad, versus 0.022 on held-out states.
On inherited rollout states, same-state teacher disagreement grew during ramp:
left-arm RMSE 0.015 rad at frame 90 and 0.264 at frame 150. Later both teacher
and learner request targets far from the final goal. A late target offset alone
does not establish an action-label or constant-offset defect.

The first arm masks selected six UpperArm motors and missed two Forearm motors
because of case-sensitive name matching. Preserve those runs as upper-arm-only
ablations. Step 14 completes the intended full eight-motor comparisons using an
explicit UpperArm/Forearm name set: direct arm goals with anchor other joints,
static arm feedforward with anchor other joints, and zero arm residuals while
retaining the inherited non-arm proposal. Nominal and changed timing for each.

Step 15 fits fresh seed-711028 heads using inherited versus task-relative inputs.
Same 421-128ELU-128ELU-33 architecture, identical initial weights within the pair,
zero final layer, inherited per-motor output scaling, teacher samples, optimizer,
4,000 Adam updates at 1e-4, and 512 examples per update. No correction PPO head.
The proposal directly predicts raw target minus motor goal through unchanged
physical projection. Both start as zero residuals; do not continue the old head.

Control inputs: inherited normalized 421 values, with delayed queue columns
95:229 masked to zero. Task-relative inputs: 116 values padded to 421 zeros:
root goal error/0.05 m (3), orientation tangent/normal error/0.1 (6), root linear
velocity/0.5 and angular velocity/2 (6), motor goal error/0.1 rad (33), motor
velocity/0.5 rad/s (33), motor goal angles (33), and two existing scaled loads.
All features clamp to +/-20. Physical scaling is fixed, not fitted on holdouts.
This changes the input representation while retaining network size and budgets;
it is a hypothesis about feedback generalization, not proof from low offline loss.

Use only archived training poses 0-5 for loss. Entire poses 6/7 remain evaluation
only. Include quiet training samples equally in both arms of the comparison.
Step 16 evaluates nominal complete skills. Step 17 evaluates changed timing.
Step 18 repeats the task-relative fit with seed 721028 from a fresh head and
optimizer and evaluates both timings. Step 19 checks exact quiet retention of
all three heads. Three supervised fits here, reserving up to three additional
bounded fits for steps 20-30 if teacher feedback on learner states supports them.

The direct-goal and teacher controllers are scripted references. A fitted head
is learned task tracking over a frozen learned Stand anchor, not learned impact
selection or corrective stepping. Arena remains unchanged in this investigation.

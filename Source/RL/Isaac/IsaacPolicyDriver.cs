using System.Collections.Generic;
using Godot;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Physics4Fun.Ragdoll;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// Runs an Isaac Lab policy on one ragdoll, in-engine, with no training and nothing that resets it.
/// The Isaac-side equivalent of pressing F5 on an arena scene.
///
/// <para><b>Why this bypasses godot_rl_agents entirely.</b> That addon's
/// <c>GodotONNX.ONNXInference</c> is written against the Stable-Baselines3 export convention: it
/// requires an input named <c>state_ins</c>, reads outputs named <c>output</c> and
/// <c>state_outs</c>, and its <c>Initialize</c> does <c>session.OutputMetadata["output"]</c>. An
/// rsl_rl graph is <c>obs[1,143] -> actions[1,36]</c> with none of those names, so it throws at
/// load rather than misbehaving. Rather than bend the Isaac export to fit an addon convention
/// nothing else here needs, this owns an <see cref="InferenceSession"/> directly.</para>
///
/// <para>Bypassing <c>Sync</c> also removes <c>action_repeat</c> from the picture. That property is
/// scene-wide, and the Godot-native track needs it at 8 (15 Hz) while an Isaac policy needs 2
/// (60 Hz) - one scene cannot serve both. Counting physics ticks here means the two tracks coexist
/// without either being retuned.</para>
///
/// <para>The observation normaliser is baked into the graph (<c>actor_obs_normalization=True</c>),
/// so raw observations go in and nothing here may normalise them. The output is the distribution
/// MEAN, not a sample: deterministic at inference, which is what you want in-engine.</para>
/// </summary>
public partial class IsaacPolicyDriver : Node
{
    /// <summary>
    /// Physics ticks per policy step, derived so the policy always runs at its trained 60 Hz.
    ///
    /// <para>Was a const 2, correct only while physics ran at 120 Hz. The frozen Isaac contract
    /// fixes the POLICY rate at 60 Hz; it says nothing about the physics rate, and the physics rate
    /// is a lever worth pulling. Godot's Stable PD divides both gains by
    /// <c>1 + Kd*dt/I + Kp*dt^2/I</c>, so a smaller dt collapses the denominator toward 1 and moves
    /// Godot's EFFECTIVE gains up toward the authored values Isaac trains against. Measured at
    /// 120 Hz the body applies 15% of its authored gain; the damping term contributes 4.11 of the
    /// 6.45 denominator and falls linearly with dt.</para>
    ///
    /// <para>Deriving it means changing <c>physics_ticks_per_second</c> alone keeps the contract
    /// intact instead of silently retraining the policy rate along with it.</para>
    /// </summary>
    public static int PhysicsTicksPerPolicyStep => Mathf.Max(1, Engine.PhysicsTicksPerSecond / 60);

    /// <summary>The body this drives. Required.</summary>
    [Export] public HumanoidRagdoll? Ragdoll { get; set; }

    /// <summary>
    /// Policy to run, e.g. <c>res://isaac_lab/exported/stand_policy.onnx</c>. Which brain this is
    /// - stand, walk, perturb or run - is decided entirely by this path; they share one
    /// observation and action contract, which is what makes the "experts" split cheap.
    /// </summary>
    [Export(PropertyHint.File, "*.onnx")] public string PolicyPath { get; set; } = string.Empty;

    /// <summary>Rig contract location. The default is right unless the assets moved.</summary>
    [Export] public string RigContractPath { get; set; } = IsaacRigContract.DefaultPath;

    /// <summary>
    /// The policy's own contract, written beside the .onnx by <c>isaac_lab_3/scripts/export.py</c>.
    /// Empty for an Isaac Lab 2.3.2 / PhysX policy, which carries no such file.
    ///
    /// <para><b>Required for any policy trained under Isaac Lab 3 / Newton.</b> It supplies
    /// `newton_dof_order`, which differs from the rig's own `physx_dof_order` in 42 of 45 slots.
    /// Without it, 90 of the 143 observation floats reach the network permuted - and because every
    /// joint angle is near zero at the rest pose, nothing numeric can detect it. The dummy just
    /// flails.</para>
    /// </summary>
    [Export(PropertyHint.File, "*.json")] public string PolicyContractPath { get; set; } = string.Empty;

    /// <summary>
    /// Derive the four contact flags from bone height instead of real contact. Set for Newton
    /// policies; see <see cref="IsaacObservation.UseHeightContacts"/>.
    /// </summary>
    [Export] public bool HeightContacts { get; set; }

    /// <summary>
    /// Velocity command (vx, vy, yaw_rate) written into observation slice [140:143]. Zero for
    /// Stand and Perturbation, which were trained with it held at zero throughout. Walk and Run
    /// read it, so an arena for either wants a non-zero forward component.
    /// </summary>
    [Export] public Vector3 Command { get; set; } = Vector3.Zero;

    /// <summary>
    /// EMA smoothing applied to the joint velocity feeding each bone's Hill force-velocity law.
    /// 1 leaves it raw (Godot's default). ~0.15 filters solver chatter out of it.
    ///
    /// <para>Preferred over <see cref="DisableHillLimit"/>: it keeps the biomechanics and removes
    /// only the artifact. See <see cref="ActiveBone.HillVelocityFilterAlpha"/>.</para>
    /// </summary>
    [Export] public float HillVelocityFilter { get; set; } = 1.0f;

    /// <summary>
    /// Scales every bone's gravity feed-forward. 1 keeps Godot's biomechanics (the default and the
    /// shipping behaviour); 0 removes the feed-forward entirely.
    ///
    /// <para><b>A MEASUREMENT INSTRUMENT, NOT A FIX.</b> The standing rule on this project is that
    /// transfer gets fixed Isaac -> Godot - model Godot's actuator on the Isaac side - and never by
    /// switching Godot's biomechanics off until the body matches Isaac's plain PD. That drift
    /// already happened once, on 2026-08-27, one defensible step at a time
    /// (<c>DisableHillLimit</c> -> <c>DisableLoadCompensation</c> -> <c>JointMotorDrive</c>), and
    /// those knobs were removed for that reason. This one exists to ANSWER A QUESTION, not to ship
    /// at 0.</para>
    ///
    /// <para>The question: Godot's feed-forward supplies 45-68% of the body's holding torque
    /// (measured as <c>ffShare</c>), and Isaac's `ImplicitActuatorCfg` has none. Once Isaac's gains
    /// were matched down onto Godot's Stable-PD plant on 2026-09-03, a policy trained without the
    /// feed-forward must learn to supply that torque itself through its position targets - and then
    /// receives Godot's feed-forward ON TOP of its own, roughly double what the pose needs. Setting
    /// this to 0 for one run says whether that is what topples the body. If it is, the fix is to
    /// implement the feed-forward in Isaac, not to leave this at 0.</para>
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float LoadCompensation { get; set; } = 1.0f;

    /// <summary>
    /// Restrict Godot's balance layer to the part Isaac actually models: pelvis stabilisation.
    ///
    /// <para><b>`BalanceAssist` does far more than its name suggests.</b> It puts the ragdoll in
    /// `RagdollState.Balanced` and wakes the WHOLE balance layer at full strength - dynamic
    /// stepping, weight transfer, ankle ground-reaction, hip strategy, arm reflexes and gaze - each
    /// of which writes <see cref="ActiveBone.FeedForwardTargetOffset"/>. Those offsets compose into
    /// every joint target alongside the policy's own
    /// (<c>TargetLocalRotation * feedForward * PolicyTargetOffset</c>).</para>
    ///
    /// <para>Isaac models exactly ONE of them: `StandEnv._apply_balance_assist` is a pelvis attitude
    /// PD applied as a body torque, ported from `PelvisStabilizationModule`. So a policy trained in
    /// Isaac has never seen the other five, and in Godot it is composed with a procedural balance
    /// controller it knows nothing about.</para>
    ///
    /// <para>That asymmetry is regime-dependent in exactly the way the transfer results are. For
    /// STANDING the extra modules help - keeping the body upright is their whole purpose, so a
    /// standing policy is assisted by them. For WALKING they fight it: stepping and weight transfer
    /// are trying to hold a stationary balanced stance while the policy is trying to translate. That
    /// is a candidate explanation for a brain that stands at authority 0.03 and falls at 0.05.</para>
    ///
    /// <para>True leaves Godot's full layer running (the shipping behaviour). False zeroes the
    /// joint-level modules and keeps pelvis stabilisation, which is the configuration Isaac trains
    /// against.</para>
    /// </summary>
    [Export] public bool BalanceJointModules { get; set; } = true;

    /// <summary>
    /// Send the pelvis stabiliser's reaction into the thighs, as Isaac does, instead of the feet.
    /// See <see cref="Physics4Fun.Ragdoll.Modules.PelvisStabilizationModule.ReactIntoThighs"/>.
    /// </summary>
    [Export] public bool PelvisReactIntoThighs { get; set; }

    /// <summary>
    /// Low-pass alpha for the joint velocities written into observation slice [55:100]. 1.0 is off.
    ///
    /// <para><b>The filter existed and was wired to nothing.</b> `IsaacObservation.JointVelocityFilter`
    /// was implemented and documented - including the measurement that motivates it - but no arena,
    /// driver or scene ever set it, so it sat at its 1.0 default and every deployment fed the policy
    /// raw solver velocity.</para>
    ///
    /// <para>Measured with the observation clip lifted, while the dummy stood still at pelvis 0.83:
    /// joint velocities of 8, 18, 20 and 35 rad/s on Spine.x and Shin_L.y. Isaac's maximum across 64
    /// environments in the same posture is 7.9. The clip at 15 hid roughly half of it. So 45 of 143
    /// observation floats reported thrashing to a policy trained on a body that does not thrash.</para>
    ///
    /// <para><b>MEASURED, AND IT MAKES TRANSFER WORSE. Leave it at 1.0.</b> With the promoted
    /// balance brain on the Stand check at authority 0.10: filter 1.00 STANDS, 0.50 falls, 0.30
    /// falls, 0.15 falls - and at 0.15 the brain needed the authority dropped to 0.05 to stand at
    /// all. The reasoning above is sound and the chatter is real; the remedy is not. An EMA at
    /// alpha 0.15 is a ~0.11 s time constant, and the lag it adds to a balance-critical signal
    /// costs more than the noise it removes.</para>
    ///
    /// <para>Kept wired rather than deleted because it was previously implemented, documented and
    /// reachable from nothing, which is how it survived long enough to look like an answer. The
    /// knob is now settable and OFF by default, and this paragraph is the reason to leave it
    /// there.</para>
    /// </summary>
    [Export] public float JointVelocityFilter { get; set; } = 1.0f;

    /// <summary>
    /// Feed observation slice [55:100] to the policy at all. **Must match the task config's
    /// `obs_joint_vel_enabled` for the checkpoint being run.** See
    /// <see cref="IsaacObservation.JointVelocityEnabled"/>.
    /// </summary>
    [Export] public bool JointVelocityEnabled { get; set; } = true;

    /// <summary>
    /// Measure joint velocity by differencing the joint ANGLE the policy is shown, instead of from
    /// body angular velocities. See <see cref="IsaacObservation.JointVelocityFromDifference"/>.
    /// </summary>
    [Export] public bool JointVelocityFromDifference { get; set; } = true;

    /// <summary>
    /// Mask the joint-velocity observation on joints narrower than this, in radians. 0 is off.
    /// **Must match `obs_joint_vel_min_range` for the checkpoint being run.**
    /// </summary>
    [Export] public float JointVelocityMinRange { get; set; }

    /// <summary>
    /// Seconds after which to dump the FULL 45-DOF joint pose once, as a JSON array. 0 disables it.
    ///
    /// <para><b>Why this exists.</b> While Godot's balance layer holds a successful stand - head
    /// 1.533, rock steady - its joints sit about 0.82 rad away from the rest pose. Isaac's
    /// `default_joint_pos` is all zeros, and the policy commands offsets from there bounded by
    /// `action_scale * span`, which for a +/-0.5 rad axis is 0.2 rad. **Godot's standing stance is
    /// therefore outside the policy's reachable action space**, and "zero action" means a pose that
    /// stands in Isaac and falls over in Godot.</para>
    ///
    /// <para>Dumping the stance lets it become Isaac's operating point, so action zero means the
    /// same body configuration in both engines.</para>
    /// </summary>
    [Export] public float DumpPoseAfterSeconds { get; set; }

    /// <summary>Logs the first step's observation and action. Cheap, and the fastest way to spot a mapping error.</summary>
    [Export] public bool LogFirstStep { get; set; } = true;

    /// <summary>
    /// Seconds between per-slice observation diagnostics. Zero disables them.
    ///
    /// Reports the largest magnitude in each slice, which is what localises an out-of-distribution
    /// input. The policy's observation normaliser is baked into the graph, so a slice whose values
    /// are far outside what training saw is amplified into a saturated action - and the action
    /// magnitude alone cannot say WHICH slice did it.
    /// </summary>
    [Export] public float DiagnosticInterval { get; set; } = 0.5f;

    /// <summary>
    /// Absolute path for a per-policy-step CSV of every DOF's angle and rate. Empty disables it.
    ///
    /// <para>Observation slices [10:55] and [55:100] are the same 45 DOFs in the same order, so
    /// writing both lets an angle be checked against its own reported rate offline. That is the one
    /// test that separates "Godot's body really is chattering" from "Godot's velocity channel is
    /// measuring the wrong quantity", and the two have completely different fixes.</para>
    /// </summary>
    [Export] public string DofTracePath { get; set; } = "";

    /// <summary>
    /// Uniform noise added to every action for the first <see cref="SpawnNoiseSeconds"/>, breaking
    /// the dummy's perfect left/right symmetry at spawn. 0 disables it.
    ///
    /// <para><b>Isaac resets with `reset_joint_noise = 0.1` and Godot spawns at the exact rest
    /// pose.</b> That asymmetry is not cosmetic for a gait: a memoryless policy on a perfectly
    /// symmetric body issues near-symmetric commands, both legs answer together, the contact flags
    /// never separate, and the policy never sees the single-support signal that drives its swing
    /// phase. Measured in Godot, `model_23600` holds both contact flags at 1 for 15 s straight while
    /// the same checkpoint runs 78.5% single support in Isaac.</para>
    ///
    /// <para>Applied to the action rather than to joint positions because Godot's bodies are
    /// physics-driven and cannot simply be posed; the effect on the first few ticks is the same.</para>
    /// </summary>
    /// <summary>
    /// Multiplier applied to every controlled bone's <see cref="ActiveBone.MaxTorque"/> at setup.
    /// 1.0 leaves the authored ceilings alone.
    ///
    /// <para><b>For testing whether Godot's actuator SATURATES.</b> Measured 2026-09-05 open-loop,
    /// Godot delivers 1.65x its commanded hip angle at a quarter command and only 1.21x at full,
    /// while Isaac delivers a flat ~2.6x at every amplitude. A falling ratio is what a torque
    /// ceiling looks like: the bigger the commanded deflection, the more the request exceeds the
    /// clamp and the further the delivered angle falls behind. If raising this flattens the
    /// 1.65 -> 1.21 falloff, the ceiling is the mechanism and the fix belongs in Isaac's
    /// `_effort_limited`, not in rescaling actions.</para>
    /// </summary>
    [Export] public float EffortScale { get; set; } = 1.0f;

    /// <summary>
    /// Multiplier on every controlled bone's <see cref="ActiveBone.MaxShorteningVelocity"/>. A large
    /// value effectively disables the Hill force-velocity derating.
    ///
    /// <para>The Hill law scales torque down as a joint shortens faster, which is a sublinear
    /// response by construction and matches the measured shape: Godot delivers 1.65x its commanded
    /// hip angle at a quarter command and 1.21x at full, while Isaac is flat at ~2.6x. Raising vmax
    /// removes the derating; if the falloff flattens, the Hill law is the mechanism.</para>
    ///
    /// <para><b>There is no `DisableHillLimit` export</b>, despite what some comments in this file
    /// still say - it is a stale reference, and `--set DisableHillLimit=true` silently does nothing.</para>
    /// </summary>
    [Export] public float HillVmaxScale { get; set; } = 1.0f;

    /// <summary>
    /// Multiplier on every controlled bone's <see cref="ActiveBone.DerivativeGain"/> at setup.
    /// 1.0 leaves the authored damping alone.
    ///
    /// <para><b>For the growing vertical oscillation that ends every Godot run.</b> Measured
    /// 2026-09-05 at authority 0.15 on a policy that walks in Isaac: pelvis height oscillates at
    /// about 3 Hz with GROWING amplitude (+/-0.015 m at t=0.5 s, +/-0.04 m at t=1.8 s) until both
    /// feet leave the ground together at t~1.7 s and the body collapses at t~2.3 s. That is an
    /// under-damped mode being pumped, not a balance failure - the same compliant-leg pogo recorded
    /// earlier on this project.</para>
    ///
    /// <para>The legs act as springs carrying body weight, and Godot's Stable-PD divides `Kd` by the
    /// same denominator it divides `Kp` by, so raising damping alone is not reachable from the
    /// authored gains. If scaling this flattens the height oscillation, the vertical mode is the
    /// mechanism and the fix belongs in the leg damping rather than in the policy.</para>
    /// </summary>
    [Export] public float DampingScale { get; set; } = 1.0f;

    /// <summary>
    /// Damping multiplier applied to the leg chain (Thigh/Shin/Foot) whose foot is CURRENTLY in
    /// contact, on top of <see cref="DampingScale"/>. 1.0 leaves the stance leg alone.
    ///
    /// <para><b>The reason a single body-wide damping scalar cannot work.</b> Measured 2026-09-05,
    /// the same `Kd` is pulled in opposite directions by the two phases of a gait. At
    /// `DampingScale = 1` the support legs are under-damped, so pelvis height oscillates at ~3 Hz
    /// with growing amplitude until both feet leave the ground and the body collapses. At
    /// `DampingScale = 3-4` the bounce is gone - height std falls 0.0845 -> 0.0047 - but the SWING
    /// leg is now too slow to lift a foot, so `footZ` sits at its resting 0.040 for seconds on end
    /// and the dummy shuffles instead of stepping. Stability and stepping want opposite damping on
    /// the same joints.</para>
    ///
    /// <para>Splitting by contact resolves it: the stance leg carries the body and wants damping,
    /// the swing leg has to move fast and does not. The contact test is the same foot-height rule
    /// the observation uses (<see cref="IsaacObservation.ContactHeight"/>), so the split agrees with
    /// what the policy is told.</para>
    /// </summary>
    [Export] public float StanceDampingScale { get; set; } = 1.0f;

    /// <summary>
    /// Damping multiplier for the leg chain whose foot is AIRBORNE, on top of
    /// <see cref="DampingScale"/>. See <see cref="StanceDampingScale"/>; values below 1.0 free the
    /// swing leg to move faster than the authored gains allow.
    /// </summary>
    [Export] public float SwingDampingScale { get; set; } = 1.0f;



    /// <summary>
    /// How the stance/swing split decides which leg is which.
    /// 0 = foot in CONTACT is stance (the height threshold the observation uses).
    /// 1 = the LOWER foot is stance, with <see cref="StanceGateMargin"/> of hysteresis.
    ///
    /// <para><b>Mode 0 cannot bootstrap.</b> Measured 2026-09-05: no foot ever leaves the ground in
    /// Godot, so both legs read as stance forever and the swing branch never runs. Mode 1 is
    /// asymmetric even in double support - exactly one leg is always the lower one - so the split
    /// engages from the first tick and a swing does not have to exist before it can be detected.</para>
    /// </summary>
    [Export(PropertyHint.Range, "0,1,1")] public int StanceGateMode { get; set; }

    /// <summary>
    /// Hysteresis (m) on the mode-1 height comparison. The stance foot keeps the role until the
    /// other foot is lower by this much, so two feet resting level do not flip the assignment every
    /// tick.
    /// </summary>
    [Export] public float StanceGateMargin { get; set; } = 0.004f;






    /// <summary>
    /// Drive the joints with Jolt's ANGULAR SPRING - a position constraint resolved inside the
    /// solver - instead of `ActiveBone`'s explicit torque. The explicit path is silenced by setting
    /// `MuscleStrength` to zero on every controlled bone.
    ///
    /// <para><b>This is the structural analogue of what Isaac does, and the explicit path may not
    /// be able to reach it.</b> Measured 2026-09-06 with the network removed from both loops and the
    /// SAME recorded target trajectory: Isaac's stride is 0.253 m and Godot's is 0.012-0.032 m, and
    /// every Godot configuration either stands still or falls. Making the explicit controller
    /// stronger does not help - compensating the Stable-PD gain reduction so the joint receives the
    /// authored gains (which Isaac is stable at) makes Godot fall in all 27 cells swept.</para>
    ///
    /// <para>That points at WHEN the correction enters the solve rather than how large it is. XPBD
    /// projects the target as a positional constraint inside the solve and is unconditionally
    /// stable; `ActiveBone` computes a torque and applies it as an external force after the solve,
    /// which is the formulation that destabilises as gain rises. Jolt's angular spring is solved as
    /// a constraint, so it should hold high impedance without the explicit path's instability.</para>
    ///
    /// <para><b>Deliberately no biomechanics on this path yet.</b> Hill force-velocity, the effort
    /// clamp and gravity feed-forward all live in the torque path. Porting them at the same time
    /// would leave three unknowns moving at once; establish first whether a pure positional
    /// constraint reproduces the gait, then add them back one at a time.</para>
    /// </summary>
    [Export] public bool UseAngularSpring { get; set; }

    /// <summary>Multiplier on the angular springs' stiffness, over the rig contract value.</summary>
    [Export] public float AngularSpringStiffnessScale { get; set; } = 1.0f;

    /// <summary>
    /// Multiplier on the angular springs' damping, over the rig contract value.
    ///
    /// <para>The rig's `Kd/Kp` ratio is 0.02-0.03 (e.g. 36/1800), while the scene's own authored
    /// angular-spring defaults are 350/35, a ratio of 0.1 - so feeding the rig's damping straight in
    /// leaves the constraint markedly under-damped relative to what the joint was tuned for.</para>
    /// </summary>
    [Export] public float AngularSpringDampingScale { get; set; } = 1.0f;







    /// <summary>
    /// Hold the ARM chain (UpperArm / Forearm / Hand, both sides) at its rest pose instead of
    /// applying the policy's offset.
    ///
    /// <para><b>The arms are the largest out-of-distribution input the policy receives in Godot,
    /// and they carry none of the gait.</b> Measured 2026-09-06 over 143 channels, the fraction of
    /// samples outside the 1st-99th percentile range Isaac trained on:</para>
    ///
    /// <code>
    /// pos_UpperArm_R.z   99.6%   godot mean +0.111   isaac [-0.205, -0.017]   opposite sign
    /// pos_Forearm_R.x    98.3%   godot mean +0.340   isaac [-0.102,  0.110]
    /// pos_Forearm_L.x    98.3%   godot mean +0.515   isaac [-0.010,  0.216]
    /// </code>
    ///
    /// <para>Godot's arms are soft (kp 100-120) and sag under gravity where XPBD's positional drive
    /// holds them, so the elbow sits three to five times more flexed than anything the policy ever
    /// saw. Holding them at rest puts those channels back near zero, inside Isaac's range, and stops
    /// the arms perturbing the torso. `LoadCompensation` was tried first and is not reliable: it
    /// moved `Forearm_L.x` from 0.372 into range at authority 0.15 and further OUT of range (0.495)
    /// at 0.125.</para>
    /// </summary>
    [Export] public bool LockArmsAtRest { get; set; }

    [Export] public float SpawnActionNoise { get; set; }

    /// <summary>Seconds over which <see cref="SpawnActionNoise"/> is applied, from the first step.</summary>
    [Export] public float SpawnNoiseSeconds { get; set; } = 0.5f;

    /// <summary>
    /// CSV of pre-scaling actions to apply INSTEAD of running the policy, one row per policy step
    /// with columns `act0..act35`. Empty runs the policy normally.
    ///
    /// <para><b>The only test that compares the two plants without the closed loop in the way.</b>
    /// Godot and Isaac running the same policy diverge for two reasons at once - the bodies respond
    /// differently, and the policy then sees different observations and commands something else.
    /// Driving both engines from one recorded action sequence removes the second, so any remaining
    /// difference in the joint trajectories is the mechanism and nothing else.</para>
    ///
    /// <para>Rows are consumed one per policy step. When the file runs out the driver holds the
    /// last row rather than reverting to inference, which would silently mix the two regimes.</para>
    /// </summary>
    [Export] public string ReplayActionsPath { get; set; } = "";

    /// <summary>
    /// When the replay rows run out, hand control to the POLICY instead of holding the last row.
    ///
    /// <para><b>Separates "cannot walk in Godot" from "cannot start walking in Godot".</b> The
    /// policy is memoryless, so its gait phase lives entirely in the observation; measured in
    /// Godot it parks at a constant posture with both contact flags stuck at 1, never sees the
    /// single-support signal, and therefore never enters its swing phase - a fixed point of the
    /// closed loop. Scripting a couple of steps and then handing over tests whether the gait
    /// SUSTAINS once the loop has been pushed into it. If it does, a kick-start is a real fix; if
    /// it does not, the policy genuinely cannot carry a gait on this body.</para>
    /// </summary>
    [Export] public bool ReplayHandoff { get; set; }

    /// <summary>
    /// Seconds spent holding EVERY bone at its rest pose before inference starts.
    ///
    /// This exists to reproduce Isaac's initial condition rather than to be cautious. An Isaac
    /// episode begins from `joint_pos = 0` on all 45 DOF with the drives already tracking that
    /// target, so the policy's first observation is a body genuinely at rest. Godot's ragdoll
    /// arrives from its procedural controller in a posed stance - measured at 0.94 rad on the worst
    /// joint against Isaac's 0.10 - and commanding rest does not teleport it there; the PD has to
    /// drive it, which takes time the first observation does not give it.
    ///
    /// Without this the policy's first action came out at 1.9, nearly twice its clamp, from a body
    /// that was standing perfectly well.
    /// </summary>
    [Export] public float WarmupSeconds { get; set; } = 1.0f;

    /// <summary>
    /// Ignore the policy and command all-zero actions, i.e. hold the rest pose.
    ///
    /// A control, not a feature. Zero action is the rest pose by construction, and in Isaac that
    /// pose is statically stable - the feet carry 790 N against an 80.6 kg body, and a zero-action
    /// policy survives 168 steps. If Godot's body collapses under the same command, then the gap is
    /// in the body and its actuators rather than in the observation or action mapping, and no
    /// amount of fixing the wiring will make a trained policy transfer.
    /// </summary>
    [Export] public bool ZeroActionBaseline { get; set; }

    /// <summary>
    /// Balance-controller strength during the RL state, 0 to 1. 0 is Godot's default (off).
    ///
    /// <para><b>This supplies the one thing measurement says is missing.</b> Godot's dummy stands at
    /// essentially the REST POSE - dumping the stance while its balance layer holds a steady 1.533 m
    /// head shows every actuated joint within the policy's reach (foot 0.11 rad against a 0.24
    /// reach, upper arm 0.10 against 0.20); only the passive HAND joints sit far out, and they
    /// dangle. So the pose was never the problem.</para>
    ///
    /// <para>What differs is that Godot reaches that pose with an external stabilising wrench -
    /// pelvis stabilisation, ankle and hip strategies - which `StateBalanceStrengthMap` sets to
    /// **0.0** for `ReinforcementLearning`. Isaac has no equivalent and does not need one, because
    /// its body is passively far more stable: under zero action Isaac's joints deviate 0.12 rad and
    /// its centre of mass drifts 0.05 m, against Godot's 1.01 rad and 0.30 m. The policy therefore
    /// learned to stand on a body that mostly holds itself, and in Godot it is asked to replace a
    /// balance controller it never knew existed.</para>
    ///
    /// <para>Unlike <see cref="AssistMode"/> this keeps the ragdoll in the RL state, so the
    /// procedural layer still does NOT write joint targets - the policy owns the pose, and the
    /// balance layer owns root stabilisation. That division is a real architecture, not a
    /// workaround.</para>
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float BalanceAssist { get; set; }

    /// <summary>
    /// Overrides the action scale the policy was trained with. 0 uses the contract's value.
    ///
    /// <para>A diagnostic dial, not a fix: it tells you how much of the policy's commanded
    /// deflection a body can absorb before the stand breaks. With balance assist on, zero action
    /// stands indefinitely and the trained policy falls in four seconds, so somewhere between them
    /// is the authority at which this policy stops being actively harmful. Driving a policy at a
    /// scale it was not trained at is a mismatch by definition - the honest fix is retraining at
    /// whatever scale this identifies.</para>
    /// </summary>
    [Export] public float ActionScaleOverride { get; set; }

    [Export] public bool AssistMode { get; set; }

    /// <summary>
    /// How much of the policy's commanded offset is applied in <see cref="AssistMode"/>, 0 to 1.
    /// 1 is the full offset; 0 is Godot's balance layer alone.
    ///
    /// <para><b>Why this needs a dial at all.</b> Godot's balance layer holds the dummy indefinitely
    /// on its own - measured at head 1.533 and pelvis 0.816, dead steady for 20 s. The Isaac policy
    /// was trained as the ONLY controller, in a simulator whose body is passively far more stable
    /// (under zero action Isaac's joints deviate 0.12 rad and its centre of mass drifts 0.05 m,
    /// against Godot's 1.01 rad and 0.30 m). Dropped in at full authority it does not assist that
    /// layer, it fights it - and the pair together fall faster than either alone.</para>
    ///
    /// <para>At reduced authority the policy becomes a corrective term on a body that is already
    /// balanced, which is a legitimate architecture and an honest one to report: the brain is
    /// modulating a stable stance rather than producing it.</para>
    /// </summary>
    [Export(PropertyHint.Range, "0,1,0.05")] public float AssistAuthority { get; set; } = 1.0f;

    /// <summary>
    /// Drive the joints the way ISAAC does - per-DOF PD in joint space - instead of the way Godot
    /// does, and disable <see cref="ActiveBone"/>'s own actuator on the bones this owns.
    ///
    /// <para><b>The last structural difference between the engines.</b> Everything else now matches:
    /// same 16-body mechanism with 15 D6 joints, geometry to a centimetre, identical gains on every
    /// bone, joint ordering exact, ONNX inference to 1.5e-07, and a frame map that is a proper
    /// rotation. What does not match is the control law. `ActiveBone` runs Stable PD on the QUATERNION
    /// error of a bone's orientation, with a gravity feed-forward, an integral term, an SPD velocity
    /// factor and a hard torque clamp; Isaac runs a plain per-DOF PD folded into the articulation
    /// solve.</para>
    ///
    /// <para>The evidence that this is what matters: Godot's body collapses in under two seconds
    /// under ZERO action - commanding nothing but the rest pose - while Isaac's holds 84% standing
    /// for eight seconds under the identical command. That is a property of the actuator, not of any
    /// policy, observation or joint ordering.</para>
    ///
    /// <para>Check this with <see cref="ZeroActionBaseline"/> before involving the policy at all. If
    /// joint-space PD cannot hold the rest pose either, the gap is deeper than the control law and
    /// nothing downstream is worth testing.</para>
    /// </summary>
    [Export] public bool JointSpacePd { get; set; }

    private InferenceSession? _session;
    private IsaacRigContract? _rig;
    private IsaacObservation? _observation;
    private IsaacActionSpace? _actions;
    private ActiveBone?[] _controlledBones = System.Array.Empty<ActiveBone?>();

    /// <summary>Leg chains by side, resolved once: index 0 = left, 1 = right.</summary>
    private readonly ActiveBone?[][] _legChains = { new ActiveBone?[3], new ActiveBone?[3] };

    /// <summary>Foot bones used for the stance test, index-matched to <see cref="_legChains"/>.</summary>
    private readonly ActiveBone?[] _feet = new ActiveBone?[2];

    /// <summary>
    /// Authored derivative gain per leg bone, captured AFTER <see cref="DampingScale"/> is applied
    /// so the phase split multiplies the base once per tick instead of compounding.
    /// </summary>
    private readonly float[][] _legBaseKd = { new float[3], new float[3] };

    /// <summary>Which side currently holds the stance role under <see cref="StanceGateMode"/> 1.</summary>
    private int _stanceSide;

    /// <summary>
    /// Smoothed per-side damping scale, so the stance/swing switch is not a step. **Seeded to the
    /// STANCE value in `ResolveLegChains`, not to 1.0**: starting at 1.0 and ramping up leaves the
    /// body under-damped for the first tens of milliseconds, which is long enough for the vertical
    /// pogo to start - measured 2026-09-06, that alone turned a 100%-upright run into a fall.
    /// </summary>
    private readonly float[] _blendedScale = { 1.0f, 1.0f };
    private Quaternion[] _offsets = System.Array.Empty<Quaternion>();
    private bool _dumpedPose;
    private float _elapsed;
    private string _inputName = "obs";
    private int _tick;
    private bool _ready;
    private bool _loggedFirstStep;
    private float _sinceDiagnostic;
    private float _warmupRemaining;
    /// <summary>Reporting only; see <see cref="IsaacDriverDiagnostics"/>. Null until _Ready finishes.</summary>
    private IsaacDriverDiagnostics? _diagnostics;

    private float _trackingError;
    private float _appliedTorque;

    /// <summary>Slice bounds from obs_action_contract.md §3, for the per-slice diagnostic.</summary>

    public override void _Ready()
    {
        if (Ragdoll == null || !IsInstanceValid(Ragdoll))
        {
            GD.PushError("[IsaacPolicyDriver] No Ragdoll assigned; this node will do nothing.");
            return;
        }

        try
        {
            _rig = IsaacRigContract.Load(
                RigContractPath,
                IsaacRigContract.LoadDofOrderOverride(PolicyContractPath));
        }
        catch (System.Exception e)
        {
            GD.PushError($"[IsaacPolicyDriver] Could not load the rig contract: {e.Message}");
            return;
        }

        _observation = new IsaacObservation(_rig, Ragdoll)
        {
            UseHeightContacts = HeightContacts,
            JointVelocityClip = IsaacRigContract.LoadJointVelocityClip(PolicyContractPath),
            JointVelocityFilter = JointVelocityFilter,
            JointVelocityEnabled = JointVelocityEnabled,
            JointVelocityFromDifference = JointVelocityFromDifference,
            PolicyRate = 1.0f / (float)GetPhysicsProcessDeltaTime() / PhysicsTicksPerPolicyStep,
            JointVelocityMinRange = JointVelocityMinRange,
        };
        if (!JointVelocityEnabled)
        {
            GD.Print("[IsaacPolicyDriver] joint velocity channel MASKED - obs[55:100] is zero, "
                     + "matching a policy trained with obs_joint_vel_enabled=False");
        }
        if (JointVelocityFilter < 1.0f)
        {
            GD.Print($"[IsaacPolicyDriver] joint velocity filter {JointVelocityFilter:F2} on the "
                     + "observation - the policy reads the limb, not the solver");
        }
        _actions = new IsaacActionSpace(_rig)
        {
            ActionRateLimit = IsaacRigContract.LoadActionRateLimit(PolicyContractPath),
            ActionScale = ActionScaleOverride > 0.0f
                ? ActionScaleOverride
                : (IsaacRigContract.LoadActionScale(PolicyContractPath) is var scale && scale > 0.0f
                    ? scale
                    : IsaacActionSpace.DefaultActionScale),
        };

        ResolveControlledBones();

        if (!LoadPolicy())
        {
            return;
        }

        // After ResolveControlledBones, so the bone array it holds is the final one.
        _diagnostics = new IsaacDriverDiagnostics(
            Ragdoll!, _rig!, _actions!, _controlledBones, JointSpacePd);

        // After _actions exists: the loader sizes each row from the action space.
        LoadReplay();

        if (!Mathf.IsEqualApprox(EffortScale, 1.0f))
        {
            int scaled = 0;
            foreach (ActiveBone? bone in _controlledBones)
            {
                if (bone != null && GodotObject.IsInstanceValid(bone))
                {
                    bone.MaxTorque *= EffortScale;
                    scaled++;
                }
            }

            GD.Print($"[IsaacPolicyDriver] EffortScale {EffortScale:F2} applied to {scaled} bones");
        }

        if (!Mathf.IsEqualApprox(DampingScale, 1.0f))
        {
            int scaled = 0;
            foreach (ActiveBone? bone in _controlledBones)
            {
                if (bone != null && GodotObject.IsInstanceValid(bone))
                {
                    bone.DerivativeGain *= DampingScale;
                    scaled++;
                }
            }

            GD.Print($"[IsaacPolicyDriver] DampingScale {DampingScale:F2} applied to {scaled} bones");
        }

        if (UseAngularSpring)
        {
            ConfigureAngularSprings();
        }

        ResolveLegChains();

        if (!Mathf.IsEqualApprox(HillVmaxScale, 1.0f))
        {
            int scaled = 0;
            foreach (ActiveBone? bone in _controlledBones)
            {
                if (bone != null && GodotObject.IsInstanceValid(bone))
                {
                    bone.MaxShorteningVelocity *= HillVmaxScale;
                    scaled++;
                }
            }

            GD.Print($"[IsaacPolicyDriver] HillVmaxScale {HillVmaxScale:F1} applied to {scaled} bones");
        }

        // The body must be in RL state or its own procedural controller keeps driving the bones and
        // fights every command this issues.
        //
        // Order matters. `UpdateBoneMuscleStiffness` reads BOTH of these flags - it softens to
        // ReinforcementLearningIdleStiffness while no policy is active, and applies
        // ArmLoadBearingGain only once one is - so they have to be set before anything triggers a
        // refresh, or the bones keep the previous state's gains.
        if (!AssistMode)
        {
            Ragdoll.CurrentState = RagdollState.ReinforcementLearning;
            Ragdoll.ReinforcementLearningPolicyActive = true;

            if (BalanceAssist > 0.0f && Ragdoll.Balance != null && IsInstanceValid(Ragdoll.Balance))
            {
                // **Balanced, not ReinforcementLearning.** Raising the RL entry in
                // `StateBalanceStrengthMap` is not enough on its own: every balance strategy
                // self-gates to Balanced/Stumbling, so in the RL state they stay off whatever the
                // strength says - measured as a trace identical to no assist at all, at both 1.0
                // and 0.6. Staying in Balanced wakes them, and `SuppressProceduralPose` is what
                // stops the procedural layer stamping its own joint targets over the policy's.
                Ragdoll.CurrentState = RagdollState.Balanced;
                Ragdoll.SuppressProceduralPose = true;
                Ragdoll.Balance.StateBalanceStrengthMap[(int)RagdollState.Balanced] =
                    Mathf.Clamp(BalanceAssist, 0.0f, 1.0f);
                Ragdoll.Balance.PelvisReactIntoThighs = PelvisReactIntoThighs;

                if (!BalanceJointModules)
                {
                    // Everything that writes a per-joint FeedForwardTargetOffset, off. Pelvis
                    // stabilisation stays, because that is the one module Isaac reproduces.
                    Ragdoll.Balance.EnableDynamicStepping = false;
                    Ragdoll.Balance.EnableArmReflexes = false;
                    Ragdoll.Balance.EnableVestibularGaze = false;
                    Ragdoll.Balance.AnklePitchGain = 0.0f;
                    Ragdoll.Balance.AnklePitchDamping = 0.0f;
                    Ragdoll.Balance.AnkleRollGain = 0.0f;
                    Ragdoll.Balance.AnkleRollDamping = 0.0f;
                    GD.Print("[IsaacPolicyDriver] joint-level balance modules OFF - pelvis "
                             + "stabilisation only, matching what Isaac trains against");
                }
                GD.Print($"[IsaacPolicyDriver] balance assist {BalanceAssist:F2} - policy owns the "
                         + "pose, Godot's balance layer owns root stabilisation");
            }
        }
        else
        {
            // Stay in the procedural state so UpdateBoneTargetRotations keeps writing a balanced
            // base pose every tick. Run after the ragdoll has written it, so the offset composes
            // on top rather than being overwritten the same frame.
            ProcessPriority = 100;
            ProcessPhysicsPriority = 100;
        }

        // NOT TeleportToStanding(), though it is tempting: it would also refresh the muscle
        // stiffness, which is otherwise private and only reachable through a state transition. But
        // it was measured to make things worse - the teleported pose is not statically stable in
        // Godot once the balance controller is off, so the body is already tilted 32 degrees by the
        // time warmup ends, against 0.5 degrees when left where the procedural controller had it.
        // Refreshing the gains is not worth starting the policy from a toppling body.
        //
        // (StartReinforcementLearning() is likewise wrong here - it drops the body to prone.)
        NeutraliseUncommandedBones();
        if (HillVelocityFilter < 1.0f)
        {
            int filtered = 0;
            foreach (ActiveBone bone in Ragdoll!.GetBones())
            {
                if (!IsInstanceValid(bone))
                {
                    continue;
                }
                bone.HillVelocityFilterAlpha = HillVelocityFilter;
                filtered++;
            }
            GD.Print($"[IsaacPolicyDriver] Hill velocity filter {HillVelocityFilter:F2} on "
                     + $"{filtered} bone(s) - the force-velocity law sees the limb, not the solver");
        }

        if (LoadCompensation < 1.0f)
        {
            int scaled = 0;
            foreach (ActiveBone bone in Ragdoll!.GetBones())
            {
                if (!IsInstanceValid(bone))
                {
                    continue;
                }
                bone.LoadCompensationScale = LoadCompensation;
                scaled++;
            }
            GD.Print($"[IsaacPolicyDriver] DIAGNOSTIC gravity feed-forward scaled to "
                     + $"{LoadCompensation:F2} on {scaled} bone(s) - this is an experiment, not a "
                     + "shipping configuration; the fix belongs on the Isaac side");
        }
        if (JointSpacePd)
        {
            SilenceGodotActuators();
        }
        _warmupRemaining = WarmupSeconds;

        _ready = true;
        // The physics rate, decimation and resource root are printed EXPLICITLY, not just the
        // derived policy rate. "60 Hz" alone is identical under 120/2 and 480/8, so it cannot
        // confirm that a change to `physics_ticks_per_second` actually reached the engine - which is
        // exactly the ambiguity that made a 480 Hz run read as byte-identical to 120 Hz.
        //
        // `root` is the one that matters most. An EMPTY root means Godot is running from a packed
        // project rather than this directory, so every project setting comes from the pack and edits
        // to project.godot do nothing at all. That is not hypothetical: an export written into the
        // Godot INSTALL folder on 2026-08-26 left a `.pck` beside - and name-matching -
        // `Godot_v4.7.1-stable_mono_win64_console.exe`, which is the binary `godot_check.py`
        // defaults to. Godot auto-mounts a pack whose basename matches the executable, so that
        // binary booted as a self-contained game and silently ignored `--path` for settings.
        // Scenes and models still came from disk, which is what made it so hard to see.
        GD.Print($"[IsaacPolicyDriver] {System.IO.Path.GetFileName(PolicyPath)} -> "
                 + $"{_observation.Size} obs / {_actions.Size} actions at "
                 + $"{Engine.PhysicsTicksPerSecond / PhysicsTicksPerPolicyStep} Hz "
                 + $"(physics {Engine.PhysicsTicksPerSecond} Hz / decimation {PhysicsTicksPerPolicyStep}, "
                 + $"root {ProjectSettings.GlobalizePath("res://")})");
    }

    private bool LoadPolicy()
    {
        if (string.IsNullOrWhiteSpace(PolicyPath))
        {
            GD.PushError("[IsaacPolicyDriver] PolicyPath is empty. Point it at an exported .onnx.");
            return false;
        }

        using Godot.FileAccess? file = Godot.FileAccess.Open(PolicyPath, Godot.FileAccess.ModeFlags.Read);
        if (file == null)
        {
            GD.PushError($"[IsaacPolicyDriver] Cannot open '{PolicyPath}' "
                         + $"(Godot error {Godot.FileAccess.GetOpenError()}). Export a policy first with "
                         + "isaac_lab/scripts/export.ps1.");
            return false;
        }

        try
        {
            // Read through FileAccess rather than handing OnnxRuntime the path: `res://` is not a
            // filesystem path once the project is exported into a .pck.
            _session = new InferenceSession(file.GetBuffer((long)file.GetLength()));
        }
        catch (System.Exception e)
        {
            GD.PushError($"[IsaacPolicyDriver] Failed to load '{PolicyPath}': {e.Message}");
            return false;
        }

        return ValidateSignature();
    }

    /// <summary>
    /// Checks the graph against the contract before a single step runs. A width mismatch would
    /// otherwise surface as an OnnxRuntime exception once per physics tick, or - worse, if the
    /// widths happen to line up - as a body that merely moves wrongly.
    /// </summary>
    private bool ValidateSignature()
    {
        IReadOnlyDictionary<string, NodeMetadata> inputs = _session!.InputMetadata;
        if (inputs.Count != 1)
        {
            GD.PushError($"[IsaacPolicyDriver] Expected exactly one graph input, found {inputs.Count}: "
                         + string.Join(", ", inputs.Keys));
            return false;
        }

        foreach (KeyValuePair<string, NodeMetadata> input in inputs)
        {
            _inputName = input.Key;
            int width = input.Value.Dimensions[^1];
            if (width != _observation!.Size)
            {
                GD.PushError($"[IsaacPolicyDriver] Graph expects {width} observations, the contract "
                             + $"builds {_observation.Size}. The rig and the policy disagree - "
                             + "re-export, or re-run convert.ps1.");
                return false;
            }
        }

        foreach (KeyValuePair<string, NodeMetadata> output in _session.OutputMetadata)
        {
            int width = output.Value.Dimensions[^1];
            if (width != _actions!.Size)
            {
                GD.PushError($"[IsaacPolicyDriver] Graph emits {width} actions, the contract expects "
                             + $"{_actions.Size}.");
                return false;
            }
            break;
        }

        return true;
    }

    /// <summary>
    /// Commands every bone the policy does NOT actuate back to its rest pose, and holds it there.
    ///
    /// <para>Head, Hand_L and Hand_R - 9 of the 45 DOF - are passive in both engines. On the Isaac
    /// side that means a PD drive tracking a constant zero target, so they sit at rest for the
    /// entire episode and the policy only ever saw them there. On the Godot side nothing owns them
    /// once the procedural controller is switched off: they keep whatever
    /// <see cref="ActiveBone.TargetLocalRotation"/> that controller last wrote, which is a posed
    /// stance, not rest. Measured before this: both hands held 0.98 rad (56 degrees) off rest
    /// indefinitely, against 0.02-0.06 in Isaac's reference.</para>
    ///
    /// <para>That is six observation slots pinned a full radian away from anything training
    /// contained, feeding a normaliser that turns the discrepancy into a saturated action. The
    /// actuated bones need no such treatment - the policy overwrites them every step.</para>
    /// </summary>
    /// <summary>
    /// True when this bone belongs to the leg that is currently SWINGING, using the same stance
    /// resolution the damping split uses.
    /// </summary>
    /// <summary>Arm chain membership for <see cref="LockArmsAtRest"/>.</summary>
    private static bool IsArmBone(string name) =>
        name.StartsWith("UpperArm") || name.StartsWith("Forearm") || name.StartsWith("Hand");

    private void NeutraliseUncommandedBones()
    {
        var actuated = new HashSet<string>(_rig!.ActuatedBoneNames());
        int reset = 0;

        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            if (actuated.Contains(bone.BoneName) || !IsInstanceValid(bone))
            {
                continue;
            }
            bone.TargetLocalRotation = bone.GetRestLocalRotation();
            reset++;
        }

        GD.Print($"[IsaacPolicyDriver] held {reset} uncommanded bone(s) at rest, matching Isaac's "
                 + "zero-target PD on the passive DOF");
    }

    /// <summary>
    /// Holds every bone at rest during warmup, actuated ones included. Reproduces the state an
    /// Isaac episode starts from: all 45 DOF at zero with the drives already tracking it.
    /// </summary>
    private void HoldAllAtRest()
    {
        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            if (IsInstanceValid(bone))
            {
                bone.TargetLocalRotation = bone.GetRestLocalRotation();
            }
        }
    }

    /// <summary>Index into the action vector for a controlled bone, or -1.</summary>
    private int IndexOfControlledBone(ActiveBone bone)
    {
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            if (ReferenceEquals(_controlledBones[i], bone))
            {
                return i * IsaacRigContract.AxesPerBone;
            }
        }
        return -1;
    }

    private static char AxisName(int axis) => axis switch { 0 => 'x', 1 => 'y', _ => 'z' };

    /// <summary>Fallback motor force for a bone with no contract entry, N.m.</summary>
    private const float DefaultMotorForce = 100.0f;

    private void SilenceGodotActuators()
    {
        int silenced = 0;
        foreach (ActiveBone? bone in _controlledBones)
        {
            if (bone != null && IsInstanceValid(bone))
            {
                bone.MuscleStrength = 0.0f;
                silenced++;
            }
        }
        GD.Print($"[IsaacPolicyDriver] joint-space PD: silenced {silenced} Godot actuator(s)");
    }

    private void ApplyJointSpaceTorque(float delta)
    {
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone == null || !IsInstanceValid(bone))
            {
                continue;
            }

            ActiveBone? parent = bone.ParentBone;
            if (parent == null || !IsInstanceValid(parent))
            {
                continue;
            }

            Vector3 current = IsaacObservation.DeviationFromRest(bone);
            Vector3 target = _actions!.TargetEuler[i];

            // Rate about the joint's axes: the child's angular velocity relative to its parent,
            // resolved in the parent's frame - which is the frame the angles are measured in.
            Vector3 rate = parent.GlobalTransform.Basis.Inverse()
                           * (bone.AngularVelocity - parent.AngularVelocity);

            var torque = Vector3.Zero;
            int baseIndex = i * IsaacRigContract.AxesPerBone;
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                IsaacRigContract.JointSpec spec = _rig!.ActuatedJoints[baseIndex + axis];
                float error = Component(target, spec.GodotAxis) - Component(current, spec.GodotAxis);

                // Inertia about this joint's own axis, not the bone's smallest principal axis: a
                // flexion axis is usually transverse to a limb's slender axis, and using the
                // minimum throws away authority by up to 5x at the knee. Guarded well BELOW any
                // real inertia - clamping it UP to a fictitious value voids the SPD stability
                // guarantee and turns the D term into a per-tick velocity amplifier.
                float inertia = Mathf.Max(
                    MinEffectiveInertia, bone.GetEffectiveInertiaAboutAxis(AxisVector(spec.GodotAxis)));
                float denominator = 1.0f
                                    + (spec.Damping * delta / inertia)
                                    + (spec.Stiffness * delta * delta / inertia);

                float t = (spec.Stiffness * error - spec.Damping * Component(rate, spec.GodotAxis))
                          / denominator;
                SetComponent(ref torque, spec.GodotAxis, Mathf.Clamp(t, -spec.Effort, spec.Effort));
            }

            Vector3 world = parent.GlobalTransform.Basis * torque;
            bone.ApplyTorque(world);
            parent.ApplyTorque(-world);

            // Tracking error, accumulated for the diagnostic. This is the number that separates the
            // two remaining explanations: if the joints SIT at their commanded angles and the body
            // still falls, the controller is fine and something about the body is wrong. If they
            // cannot reach them under full torque, the constraint solver is sagging under load -
            // which is a property of Godot's solver that no Isaac actuator model could reproduce.
            _trackingError = Mathf.Max(_trackingError, (target - current).Length());
            _appliedTorque = Mathf.Max(_appliedTorque, torque.Length());
        }
    }

    /// <summary>
    /// Divide-by-zero guard for the SPD denominator. Must stay below every real body inertia - see
    /// <see cref="ApplyJointSpaceTorque"/>.
    /// </summary>
    private const float MinEffectiveInertia = 1e-5f;

    private static Vector3 AxisVector(char axis) => axis switch
    {
        'x' => Vector3.Right,
        'y' => Vector3.Up,
        _ => Vector3.Back,
    };

    private static float Component(in Vector3 v, char axis) => axis switch
    {
        'x' => v.X,
        'y' => v.Y,
        'z' => v.Z,
        _ => 0.0f,
    };

    private static void SetComponent(ref Vector3 v, char axis, float value)
    {
        switch (axis)
        {
            case 'x': v.X = value; break;
            case 'y': v.Y = value; break;
            case 'z': v.Z = value; break;
        }
    }

    private void ResolveControlledBones()
    {
        string[] boneNames = _rig!.ActuatedBoneNames();
        _controlledBones = new ActiveBone?[boneNames.Length];
        _offsets = new Quaternion[boneNames.Length];

        var byName = new Dictionary<string, ActiveBone>();
        foreach (ActiveBone bone in Ragdoll!.GetBones())
        {
            byName[bone.BoneName] = bone;
        }

        for (int i = 0; i < boneNames.Length; i++)
        {
            _offsets[i] = Quaternion.Identity;
            if (byName.TryGetValue(boneNames[i], out ActiveBone? bone))
            {
                _controlledBones[i] = bone;
            }
            else
            {
                GD.PushError($"[IsaacPolicyDriver] Rig contract actuates '{boneNames[i]}', "
                             + "which this ragdoll does not have.");
            }
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!_ready)
        {
            return;
        }

        if (_warmupRemaining > 0.0f)
        {
            _warmupRemaining -= (float)delta;
            if (!AssistMode)
            {
                HoldAllAtRest();
            }
            if (_warmupRemaining <= 0.0f)
            {
                GD.Print("[IsaacPolicyDriver] warmup complete, policy engaging");
            }
            return;
        }

        // Decimation, matching `decimation = 2` on the Isaac side. The commanded targets persist
        // between policy steps, so the PD drives keep tracking on the tick in between - which is
        // exactly what the trainer does.
        if (_tick++ % PhysicsTicksPerPolicyStep == 0)
        {
            Step((float)delta);
        }

        // The PD itself runs at the FULL physics rate, on every tick including the ones between
        // policy steps. That is what Isaac's drives do - the target is held while the actuator keeps
        // integrating - and running it only on policy ticks would halve the control rate.
        if (UseAngularSpring)
        {
            ApplyAngularSpringTargets();

        }

        // Before the PD reads the gains: the split is a property of the CURRENT contact state, and
        // applying it after the torque is computed would use last tick's phase.
        ApplyPhaseDamping();

        if (JointSpacePd)
        {
            ApplyJointSpaceTorque((float)delta);
        }

        // At the FULL physics rate, like the PD above: the target persists between policy steps and
        // the motor keeps closing on it, which is what Isaac's drives do between decimated steps.
    }

    /// <summary>
    /// Switch every controlled joint over to Jolt's angular spring and silence the explicit torque
    /// path. See <see cref="UseAngularSpring"/>.
    /// </summary>
    private void ConfigureAngularSprings()
    {
        int configured = 0;
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone == null || !IsInstanceValid(bone) || bone.Joint == null)
            {
                continue;
            }

            Generic6DofJoint3D joint = bone.Joint;
            int baseIndex = i * IsaacRigContract.AxesPerBone;
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                IsaacRigContract.JointSpec spec = _rig!.ActuatedJoints[baseIndex + axis];
                SetSpring(joint, spec.GodotAxis,
                    spec.Stiffness * AngularSpringStiffnessScale,
                    spec.Damping * AngularSpringDampingScale);
            }

            // The explicit actuator must go quiet or the two drives fight: `ActiveBone.UpdateBone`
            // returns early at this threshold, so no torque is applied at all.
            bone.MuscleStrength = 0.0f;
            configured++;
        }

        GD.Print($"[IsaacPolicyDriver] angular-spring drive on {configured} joints; explicit torque "
                 + "path silenced (MuscleStrength 0)");
    }

    private static void SetSpring(Generic6DofJoint3D joint, char axis, float stiffness, float damping)
    {
        switch (axis)
        {
            case 'x':
                joint.SetFlagX(Generic6DofJoint3D.Flag.EnableAngularSpring, true);
                joint.SetParamX(Generic6DofJoint3D.Param.AngularSpringStiffness, stiffness);
                joint.SetParamX(Generic6DofJoint3D.Param.AngularSpringDamping, damping);
                break;
            case 'y':
                joint.SetFlagY(Generic6DofJoint3D.Flag.EnableAngularSpring, true);
                joint.SetParamY(Generic6DofJoint3D.Param.AngularSpringStiffness, stiffness);
                joint.SetParamY(Generic6DofJoint3D.Param.AngularSpringDamping, damping);
                break;
            default:
                joint.SetFlagZ(Generic6DofJoint3D.Flag.EnableAngularSpring, true);
                joint.SetParamZ(Generic6DofJoint3D.Param.AngularSpringStiffness, stiffness);
                joint.SetParamZ(Generic6DofJoint3D.Param.AngularSpringDamping, damping);
                break;
        }
    }

    /// <summary>Write the commanded angles to the springs' equilibrium points.</summary>
    private void ApplyAngularSpringTargets()
    {
        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone == null || !IsInstanceValid(bone) || bone.Joint == null)
            {
                continue;
            }

            Generic6DofJoint3D joint = bone.Joint;
            Vector3 target = _actions!.TargetEuler[i];
            int baseIndex = i * IsaacRigContract.AxesPerBone;
            for (int axis = 0; axis < IsaacRigContract.AxesPerBone; axis++)
            {
                IsaacRigContract.JointSpec spec = _rig!.ActuatedJoints[baseIndex + axis];

                // **Negated: Jolt's angular constraint frame is MIRRORED relative to the angles this
                // rig reports**, the same inversion already recorded for the joint limits (the scene
                // stores `[-upper, -lower]` so Jolt enforces the anatomical range). Measured
                // 2026-09-06 before this negation: a commanded hip of +0.315 rad settled at -0.319,
                // i.e. the body was driven backwards, which is why the first angular-spring run
                // fell at every authority.
                float value = -Component(target, spec.GodotAxis);
                switch (spec.GodotAxis)
                {
                    case 'x':
                        joint.SetParamX(Generic6DofJoint3D.Param.AngularSpringEquilibriumPoint, value);
                        break;
                    case 'y':
                        joint.SetParamY(Generic6DofJoint3D.Param.AngularSpringEquilibriumPoint, value);
                        break;
                    default:
                        joint.SetParamZ(Generic6DofJoint3D.Param.AngularSpringEquilibriumPoint, value);
                        break;
                }
            }
        }
    }

    /// <summary>Resolve the two leg chains and capture their authored damping.</summary>
    private void ResolveLegChains()
    {
        string[][] names =
        {
            new[] { "Thigh_L", "Shin_L", "Foot_L" },
            new[] { "Thigh_R", "Shin_R", "Foot_R" },
        };

        for (int side = 0; side < 2; side++)
        {
            for (int j = 0; j < names[side].Length; j++)
            {
                ActiveBone? found = null;
                foreach (ActiveBone? bone in _controlledBones)
                {
                    if (bone != null && GodotObject.IsInstanceValid(bone) && bone.BoneName == names[side][j])
                    {
                        found = bone;
                        break;
                    }
                }

                _legChains[side][j] = found;
                _legBaseKd[side][j] = found?.DerivativeGain ?? 0.0f;
                _blendedScale[side] = StanceDampingScale;
                if (names[side][j].StartsWith("Foot"))
                {
                    _feet[side] = found;
                }
            }
        }
    }

    /// <summary>
    /// Split leg damping by gait phase. See <see cref="StanceDampingScale"/> for why one scalar
    /// cannot serve both phases.
    /// </summary>
    private void ApplyPhaseDamping()
    {
        if (Mathf.IsEqualApprox(StanceDampingScale, 1.0f) && Mathf.IsEqualApprox(SwingDampingScale, 1.0f))
        {
            return;
        }

        // Mode 1 resolves the stance side ONCE from the two foot heights, so exactly one leg is
        // stance even when both feet are on the floor.
        if (StanceGateMode == 1)
        {
            ActiveBone? l = _feet[0];
            ActiveBone? r = _feet[1];
            if (l != null && r != null && GodotObject.IsInstanceValid(l) && GodotObject.IsInstanceValid(r))
            {
                float zl = l.GlobalPosition.Y;
                float zr = r.GlobalPosition.Y;
                if (zl < zr - StanceGateMargin)
                {
                    _stanceSide = 0;
                }
                else if (zr < zl - StanceGateMargin)
                {
                    _stanceSide = 1;
                }
            }
        }

        for (int side = 0; side < 2; side++)
        {
            ActiveBone? foot = _feet[side];
            bool grounded = StanceGateMode == 1
                ? side == _stanceSide
                : foot != null && GodotObject.IsInstanceValid(foot)
                    && foot.GlobalPosition.Y < IsaacObservation.ContactHeight;
            float scale = grounded ? StanceDampingScale : SwingDampingScale;

            _blendedScale[side] = scale;

            for (int j = 0; j < _legChains[side].Length; j++)
            {
                ActiveBone? bone = _legChains[side][j];
                if (bone != null && GodotObject.IsInstanceValid(bone))
                {
                    bone.DerivativeGain = _legBaseKd[side][j] * _blendedScale[side];
                }
            }
        }
    }

    private void Step(float delta)
    {
        _elapsed += delta * PhysicsTicksPerPolicyStep;
        _observation!.SetCommand(Command);

        var context = new RlContext(
            Ragdoll!,
            Ragdoll.Pelvis!,
            Ragdoll.Balance!,
            _controlledBones,
            0.0f,
            0.0f,
            delta);

        float[] obs = _observation.Build(context);
        float[] actions = ZeroActionBaseline ? new float[_actions!.Size] : Infer(obs);
        if (SpawnActionNoise > 0.0f && _elapsed < SpawnNoiseSeconds)
        {
            for (int i = 0; i < actions.Length; i++)
            {
                actions[i] += (float)GD.RandRange(-SpawnActionNoise, SpawnActionNoise);
            }
        }

        if (_replay != null)
        {
            if (_replayRow < _replay.Length)
            {
                actions = _replay[_replayRow];
            }
            else if (ReplayHandoff)
            {
                if (_replayRow == _replay.Length)
                {
                    GD.Print($"[IsaacPolicyDriver] replay exhausted at t={_elapsed:F2}s - "
                             + "POLICY now has control");
                }
            }
            else
            {
                actions = _replay[_replay.Length - 1];
            }

            _replayRow++;
        }

        if (actions.Length == 0)
        {
            return;
        }

        // **Phase lead is applied to the ACTION, before decoding**, so every drive path inherits it -
        // the explicit torque path, the angular spring and the joint-space PD all derive their
        // targets from this one decode. Applying it further downstream reached only one of them:
        // measured 2026-09-06, a lead wired into the spring path alone left the explicit path
        // byte-identical across lead 0/1/2/3, which reads as "no effect" rather than "not applied".

        _actions!.Decode(actions, _offsets);

        // The clamped action - not the raw one - is what the next observation must carry, because
        // that is what the environment recorded during training.
        _observation.RecordAction(_actions.ClampedAction);

        for (int i = 0; i < _controlledBones.Length; i++)
        {
            ActiveBone? bone = _controlledBones[i];
            if (bone != null && IsInstanceValid(bone))
            {
                if (AssistMode)
                {
                    // FeedForwardTargetOffset, not TargetLocalRotation. The procedural layer
                    // rewrites TargetLocalRotation every tick and ActiveBone consumes it in the
                    // same step, so an offset written from here lands after it has already been
                    // used and is gone by the next one - measured as output identical to three
                    // decimals with the policy ignored, i.e. completely inert.
                    //
                    // This property is the seam built for exactly this: ActiveBone composes it as
                    // `parentGlobalRot * TargetLocalRotation * _smoothedFeedForwardOffset`, and
                    // nothing in the procedural pipeline writes it.
                    // Slerp from identity: identity is "leave the balance layer's pose alone",
                    // so this scales the policy's correction rather than blending two poses.
                    bone.PolicyTargetOffset = AssistAuthority >= 1.0f
                        ? _offsets[i]
                        : Quaternion.Identity.Slerp(_offsets[i], Mathf.Clamp(AssistAuthority, 0.0f, 1.0f))
                            .Normalized();
                }
                                else if (LockArmsAtRest && IsArmBone(bone.BoneName))
                {
                    // Rest pose, not the policy's offset: see `LockArmsAtRest`.
                    bone.TargetLocalRotation = bone.GetRestLocalRotation();
                }
                else
                {
                    bone.TargetLocalRotation = (bone.GetRestLocalRotation() * _offsets[i]).Normalized();
                }
            }
        }

        if (DumpPoseAfterSeconds > 0.0f && !_dumpedPose && _elapsed >= DumpPoseAfterSeconds)
        {
            _dumpedPose = true;
            var pose = new System.Text.StringBuilder("[");
            for (int i = 0; i < _rig!.DofOrder.Count; i++)
            {
                pose.Append(i > 0 ? ", " : "").Append(obs[10 + i].ToString(
                    "0.0000", System.Globalization.CultureInfo.InvariantCulture));
            }
            GD.Print($"[IsaacPolicyDriver] STANCE (newton_dof_order, rad from rest) {pose.Append(']')}");
        }

        if (LogFirstStep && !_loggedFirstStep)
        {
            _loggedFirstStep = true;
            _diagnostics?.LogFirstStep(obs, actions, _trackingError, _appliedTorque);
        }

        if (!string.IsNullOrEmpty(DofTracePath))
        {
            _diagnostics?.TraceDofs(obs, _elapsed, DofTracePath);
        }

        if (DiagnosticInterval > 0.0f)
        {
            _sinceDiagnostic += delta * PhysicsTicksPerPolicyStep;
            if (_sinceDiagnostic >= DiagnosticInterval)
            {
                _sinceDiagnostic = 0.0f;
                _diagnostics?.LogSlices(obs, actions, _trackingError, _appliedTorque);
            }
        }
    }

    private float[] Infer(float[] obs)
    {
        var tensor = new DenseTensor<float>(obs, new[] { 1, obs.Length });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        try
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session!.Run(inputs);
            foreach (DisposableNamedOnnxValue result in results)
            {
                return result.AsEnumerable<float>().ToArray();
            }
        }
        catch (OnnxRuntimeException e)
        {
            GD.PushError($"[IsaacPolicyDriver] Inference failed: {e.Message}");
            _ready = false;
        }

        return System.Array.Empty<float>();
    }

    /// <summary>
    /// Prints the slices with sharp, independently-known rest-pose values. This is the cheapest
    /// instrument for the failure this whole path is exposed to: a permuted joint order or a
    /// flipped roll sign produces a plausible-looking vector and a body that merely moves wrongly.
    /// Gravity should read (0,0,-1) upright, pelvis height 0.82, joint positions ~0 at rest.
    /// </summary>
    public override void _ExitTree()
    {
        _diagnostics?.CloseTrace();
        _session?.Dispose();
        _session = null;
    }

    /// <summary>Recorded actions for <see cref="ReplayActionsPath"/>, or null when inferring.</summary>
    private float[][]? _replay;
    private int _replayRow;

    /// <summary>Reads the replay CSV, taking the `act0..actN` columns by NAME, not by position.</summary>
    private void LoadReplay()
    {
        if (string.IsNullOrEmpty(ReplayActionsPath) || _actions == null)
        {
            return;
        }

        if (!System.IO.File.Exists(ReplayActionsPath))
        {
            GD.PrintErr($"[IsaacPolicyDriver] replay file not found: {ReplayActionsPath}");
            return;
        }

        string[] lines = System.IO.File.ReadAllLines(ReplayActionsPath);
        if (lines.Length < 2)
        {
            GD.PrintErr("[IsaacPolicyDriver] replay file has no rows.");
            return;
        }

        string[] header = lines[0].Split(',');
        var columns = new int[_actions.Size];
        for (int a = 0; a < _actions.Size; a++)
        {
            columns[a] = System.Array.IndexOf(header, $"act{a}");
            if (columns[a] < 0)
            {
                GD.PrintErr($"[IsaacPolicyDriver] replay file has no column 'act{a}'.");
                return;
            }
        }

        var rows = new System.Collections.Generic.List<float[]>(lines.Length - 1);
        for (int i = 1; i < lines.Length; i++)
        {
            string[] cells = lines[i].Split(',');
            var row = new float[_actions.Size];
            for (int a = 0; a < _actions.Size; a++)
            {
                float.TryParse(cells[columns[a]], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out row[a]);
            }

            rows.Add(row);
        }

        _replay = rows.ToArray();
        GD.Print($"[IsaacPolicyDriver] REPLAY {_replay.Length} recorded action rows - the policy is "
                 + "NOT running; this measures the plant, not the controller.");
    }
}

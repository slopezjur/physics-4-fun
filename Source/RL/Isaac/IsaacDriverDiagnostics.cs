using Godot;
using Physics4Fun.Ragdoll;

namespace Physics4Fun.RL.Isaac;

/// <summary>
/// Everything <see cref="IsaacPolicyDriver"/> prints about itself, and nothing that drives the body.
///
/// <para><b>Split out because it was 29% of the driver and shares none of its purpose.</b> The
/// driver's job is to turn observations into torques; this turns the same state into strings. They
/// changed for entirely different reasons - a diagnostic gets added whenever a transfer question
/// comes up, the control path only at a retrain boundary - and mixing them meant every reporting
/// tweak edited the class that decides what the dummy does.</para>
///
/// <para>Strictly read-only. It holds the collaborators it reports on and mutates none of them, so
/// no diagnostic can change a result while measuring it - which on this project matters, because
/// nearly every conclusion here rests on a printed number.</para>
/// </summary>
internal sealed class IsaacDriverDiagnostics
{
    /// <summary>Observation layout, for reporting a per-slice peak next to the action it produced.</summary>
        private static readonly (string Name, int From, int To)[] Slices =
        {
            ("gravity", 0, 3), ("linVel", 3, 6), ("angVel", 6, 9), ("height", 9, 10),
            ("jointPos", 10, 55), ("jointVel", 55, 100), ("contacts", 100, 104),
            ("prevAct", 104, 140), ("command", 140, 143),
        };

    private readonly HumanoidRagdoll _ragdoll;
    private readonly IsaacRigContract _rig;
    private readonly IsaacActionSpace _actions;
    private readonly bool _jointSpacePd;

    /// <summary>
    /// The bones the driver commands. Held by reference rather than re-read per call: the driver
    /// assigns the array once, in <c>ResolveControlledBones</c> during <c>_Ready</c>, and this is
    /// constructed after that - so the reference cannot go stale while diagnostics are running.
    /// </summary>
    private readonly ActiveBone?[] _bones;

    internal IsaacDriverDiagnostics(
        HumanoidRagdoll ragdoll,
        IsaacRigContract rig,
        IsaacActionSpace actions,
        ActiveBone?[] bones,
        bool jointSpacePd)
    {
        _ragdoll = ragdoll;
        _rig = rig;
        _actions = actions;
        _bones = bones;
        _jointSpacePd = jointSpacePd;
    }

        internal void LogFirstStep(float[] obs, float[] actions, float trackingError, float appliedTorque)
        {
            float maxJoint = 0.0f;
            for (int i = 10; i < 55 && i < obs.Length; i++)
            {
                maxJoint = Mathf.Max(maxJoint, Mathf.Abs(obs[i]));
            }
    
            float maxAction = 0.0f;
            foreach (float a in actions)
            {
                maxAction = Mathf.Max(maxAction, Mathf.Abs(a));
            }
    
            GD.Print($"[IsaacPolicyDriver] first step: gravity=({obs[0]:F3},{obs[1]:F3},{obs[2]:F3}) "
                     + $"pelvisHeight={obs[9]:F3} maxJointPos={maxJoint:F3} maxAction={maxAction:F3} "
                     + $"worstJoint={WorstDof(obs, 10, 55)}");
        }
    
        /// <summary>
        /// Where the actuators' torque budget is actually going, across the controlled bones.
        ///
        /// <para>Sweeping the torque budget in Isaac locates Godot at an effective
        /// <c>effort_scale</c> of 0.4-0.5 - it delivers roughly HALF the authority its
        /// <see cref="ActiveBone.MaxTorque"/> numbers promise, even though those numbers are identical
        /// to the rig contract's <c>effort</c> values. This reports the candidates for the missing
        /// half, so the answer is measured rather than reasoned about:</para>
        ///
        /// <list type="bullet">
        /// <item><description><c>demand</c> - PD plus feed-forward, as a fraction of the bone's
        /// ceiling. Above 1.0 means the actuator is being asked for more than it can ever give.</description></item>
        /// <item><description><c>deliver</c> - what survived every clamp, same units. The gap between
        /// this and <c>demand</c> IS the missing authority.</description></item>
        /// <item><description><c>fvScale</c> - the Hill force-velocity derating. It falls as a joint
        /// moves fast, so a chattering body loses torque exactly when it needs it most.</description></item>
        /// </list>
        /// </summary>
        private string TorqueBudgetReport(float trackingError, float appliedTorque)
        {
            float demand = 0.0f;
            float deliver = 0.0f;
            float worstFv = 1.0f;
            int counted = 0;
    
            foreach (ActiveBone? bone in _bones)
            {
                if (bone == null || !GodotObject.IsInstanceValid(bone))
                {
                    continue;
                }
    
                float ceiling = bone.MaxTorque * bone.MuscleStrength;
                if (ceiling <= 0.0f)
                {
                    continue;
                }
    
                demand = Mathf.Max(demand,
                    (bone.LastPdTorque + bone.LastLoadCompensationTorque).Length() / ceiling);
                deliver = Mathf.Max(deliver, bone.LastAppliedTorque.Length() / ceiling);
                worstFv = Mathf.Min(worstFv, bone.LastForceVelocityScale);
                counted++;
            }
    
            return counted == 0
                ? string.Empty
                : $" demand={demand:F2} deliver={deliver:F2} fvScale={worstFv:F2}"
                  + $" kEff={EffectiveGainFraction():F2}" + StiffnessReport() + TrackingReport(trackingError, appliedTorque) + BalanceReport();
        }
    
        /// <summary>
        /// What fraction of its authored proportional gain each joint actually applies, averaged.
        ///
        /// <para><b>This is the suspected home of the missing authority.</b> `ActiveBone` drives through
        /// `PidController3D`, which uses the Tan-Liu-Turk SPD form and divides BOTH gains by
        /// <c>1 + kd*dt/I + kp*dt^2/I</c>. For the knee - kp=1800, kd=36, dt=1/120 - a limb inertia near
        /// 0.1 kg m^2 gives a denominator around 5.25, so the effective stiffness is under a fifth of
        /// the authored value. Isaac's XPBD applies the drive inside the solve at the full gain, with no
        /// such division.</para>
        ///
        /// <para>It also explains a number that looked reassuring: <c>demand</c> reads only ~0.18 of the
        /// ceiling, which seemed to say the actuators were not even working hard. They are not - the
        /// denominator divided the request down before the ceiling ever came into it.</para>
        ///
        /// <para>Near 1.0 means Godot is applying what the rig contract says. Well below it means the
        /// policy is driving a much softer joint than the one it trained against, and that no amount of
        /// raising <see cref="ActiveBone.MaxTorque"/> will help, because the request never reaches the
        /// ceiling.</para>
        /// </summary>
        private float EffectiveGainFraction()
        {
            float total = 0.0f;
            int counted = 0;
            float dt = 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);
    
            foreach (ActiveBone? bone in _bones)
            {
                if (bone == null || !GodotObject.IsInstanceValid(bone) || bone.ProportionalGain <= 0.0f)
                {
                    continue;
                }
    
                float inertia = Mathf.Max(1e-5f, bone.LastEffectiveInertia);
                float denominator = 1.0f
                                    + (bone.DerivativeGain * dt / inertia)
                                    + (bone.ProportionalGain * dt * dt / inertia);
                total += 1.0f / denominator;
                counted++;
            }
    
            return counted == 0 ? 1.0f : total / counted;
        }
    
        /// <summary>
        /// Horizontal offset of the whole-body centre of mass from the midpoint between the feet, and
        /// the feet's height above the floor.
        ///
        /// <para><b>The balance question, which every joint-level diagnostic misses.</b> Zero-action
        /// traces show the two engines failing in different ways: Isaac SAGS - drops 8 cm, joints stay
        /// near rest, angular velocity near zero, holds for a second - while Godot TIPS, holding its
        /// height while angular velocity grows monotonically from the first sample. A topple means the
        /// centre of mass is leaving the support polygon, which is upstream of anything the actuator
        /// does.</para>
        ///
        /// <para>An offset well inside the foot span is a body that can sag but not fall over; one
        /// outside it is falling over regardless of how well the joints track.</para>
        /// </summary>
        private string BalanceReport()
        {
            if (!GodotObject.IsInstanceValid(_ragdoll))
            {
                return string.Empty;
            }
    
            var com = Vector3.Zero;
            float mass = 0.0f;
            foreach (ActiveBone bone in _ragdoll.GetBones())
            {
                if (!GodotObject.IsInstanceValid(bone))
                {
                    continue;
                }
                com += bone.GlobalPosition * bone.Mass;
                mass += bone.Mass;
            }
    
            if (mass <= 0.0f)
            {
                return string.Empty;
            }
            com /= mass;
    
            ActiveBone? left = _ragdoll.FindBone("Foot_L");
            ActiveBone? right = _ragdoll.FindBone("Foot_R");
            if (left == null || right == null || !GodotObject.IsInstanceValid(left) || !GodotObject.IsInstanceValid(right))
            {
                return string.Empty;
            }
    
            Vector3 mid = (left.GlobalPosition + right.GlobalPosition) * 0.5f;
            float offset = new Vector2(com.X - mid.X, com.Z - mid.Z).Length();
            float footHeight = Mathf.Min(left.GlobalPosition.Y, right.GlobalPosition.Y);
            float strength = 0.0f;
            int bones = 0;
            foreach (ActiveBone bone in _ragdoll.GetBones())
            {
                if (GodotObject.IsInstanceValid(bone))
                {
                    strength += bone.MuscleStrength;
                    bones++;
                }
            }
    
            return $" comOff={offset:F3}m feet={footHeight:F3}m mass={mass:F1}kg"
                   + $" muscle={(bones > 0 ? strength / bones : 0.0f):F2}";
        }
    
        /// <summary>
        /// How far each joint sits from the angle the policy actually commanded, radians.
        ///
        /// <para>The question every other diagnostic dances around: <b>does the body ever adopt the pose
        /// the policy asked for?</b> Torque can be unclamped and gains can be whatever they are, but if
        /// the joints never reach their targets then the policy's intent is not reaching the body at
        /// all, and no amount of matching the actuator model will help.</para>
        ///
        /// <para>Reported as mean and worst across the controlled bones. Isaac's drives hold their
        /// targets closely once settled - joint velocity decays to 0.18 rad/s - so a large error here is
        /// a difference in kind, not degree.</para>
        /// </summary>
        private string TrackingReport(float trackingError, float appliedTorque)
        {
            if (_actions == null)
            {
                return string.Empty;
            }
    
            float total = 0.0f;
            float worst = 0.0f;
            int counted = 0;
    
            for (int i = 0; i < _bones.Length && i < _actions.TargetEuler.Length; i++)
            {
                ActiveBone? bone = _bones[i];
                if (bone == null || !GodotObject.IsInstanceValid(bone) || bone.ParentBone == null)
                {
                    continue;
                }
    
                float error = (_actions.TargetEuler[i] - IsaacObservation.DeviationFromRest(bone)).Length();
                total += error;
                worst = Mathf.Max(worst, error);
                counted++;
            }
    
            return counted == 0 ? string.Empty : $" trackErr={total / counted:F2}/{worst:F2}rad";
        }
    
        /// <summary>
        /// Absolute effective stiffness per bone against the gain the rig contract authored, and the
        /// hard ceiling <c>I/dt^2</c> that no gain can exceed.
        ///
        /// <para>Reported because the FRACTION alone misleads once you try to compensate: raising
        /// <c>kp</c> also raises the SPD denominator, so the fraction falls while the absolute value
        /// barely moves. The ceiling is the number that matters - as <c>kp</c> tends to infinity the
        /// effective gain tends to <c>I/dt^2</c>, so a limb light enough at 120 Hz simply cannot be
        /// driven as stiffly as the contract asks, by any gain.</para>
        /// </summary>
        private string StiffnessReport()
        {
            float dt = 1.0f / Mathf.Max(1, Engine.PhysicsTicksPerSecond);
            float authored = 0.0f;
            float effective = 0.0f;
            float ceiling = 0.0f;
            int counted = 0;
    
            foreach (ActiveBone? bone in _bones)
            {
                if (bone == null || !GodotObject.IsInstanceValid(bone) || bone.ProportionalGain <= 0.0f)
                {
                    continue;
                }
    
                float inertia = Mathf.Max(1e-5f, bone.LastEffectiveInertia);
                float denominator = 1.0f
                                    + (bone.DerivativeGain * dt / inertia)
                                    + (bone.ProportionalGain * dt * dt / inertia);
                authored += bone.ProportionalGain;
                effective += bone.ProportionalGain / denominator;
                ceiling += inertia / (dt * dt);
                counted++;
            }
    
            if (counted == 0)
            {
                return string.Empty;
            }
            return $" kp={authored / counted:F0}->{effective / counted:F0} ceiling={ceiling / counted:F0}";
        }
    
        /// <summary>
        /// Name and value of the largest-magnitude entry in an observation slice, as
        /// <c>Name:value</c>. The DOF order is the policy's own, so the index maps straight onto the
        /// contract's joint list and the answer is directly comparable with Isaac.
        /// </summary>
        private string WorstDof(float[] obs, int from, int to)
        {
            int worst = -1;
            float peak = -1.0f;
            for (int i = from; i < to && i < obs.Length; i++)
            {
                float magnitude = Mathf.Abs(obs[i]);
                if (magnitude > peak)
                {
                    peak = magnitude;
                    worst = i - from;
                }
            }
    
            if (worst < 0 || _rig == null || worst >= _rig.DofOrder.Count)
            {
                return "n/a";
            }
    
            IsaacRigContract.JointSpec spec = _rig.DofOrder[worst];
            return $"{spec.Bone}.{spec.GodotAxis}:{peak:F1}";
        }
    
        internal void LogSlices(float[] obs, float[] actions, float trackingError, float appliedTorque)
        {
            var line = new System.Text.StringBuilder("[IsaacPolicyDriver] ");
            foreach ((string name, int from, int to) in Slices)
            {
                float peak = 0.0f;
                for (int i = from; i < to && i < obs.Length; i++)
                {
                    peak = Mathf.Max(peak, Mathf.Abs(obs[i]));
                }
                line.Append($"{name}={peak:F2} ");
            }
    
            float maxAction = 0.0f;
            foreach (float a in actions)
            {
                maxAction = Mathf.Max(maxAction, Mathf.Abs(a));
            }
            line.Append($"| maxAction={maxAction:F2}");
    
            // Name the worst joint-velocity DOF, not just its magnitude.
            //
            // The slice peak alone cannot distinguish "the whole body is moving" from "one light distal
            // bone is chattering", and those need opposite fixes. Measured against Isaac running the
            // same policy at the same instant - Godot 33.90 rad/s against Isaac's 2.95, on a body still
            // standing at 0.81 m - the difference has to be localised before it can be explained.
            line.Append($" worstDof={WorstDof(obs, 55, 100)}");
            line.Append(TorqueBudgetReport(trackingError, appliedTorque));
            if (_jointSpacePd)
            {
                line.Append($" trackErr={trackingError:F3}rad torque={appliedTorque:F0}Nm");
                trackingError = 0.0f;
                appliedTorque = 0.0f;
            }
            GD.Print(line.ToString());
        }
}

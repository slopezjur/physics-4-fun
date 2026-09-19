using System;
using System.Runtime.InteropServices;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// A loaded MuJoCo model and its state, exposing body poses already converted into Godot's frame.
/// </summary>
/// <remarks>
/// <para>Phase 0 established that Godot's Jolt ragdoll cannot hold single-leg support in ~100
/// configurations, while MuJoCo holds it for 17.5 s and walks the same rig 4.8 m in 40 s at 100%
/// uprightness. This is the bridge that lets the MuJoCo body drive what Godot renders.</para>
/// <para>The rig is generated from the Godot scene by <c>mujoco_rig/build_mjcf.py</c>, so there is no
/// second hand-maintained skeleton to drift.</para>
/// </remarks>
internal sealed class MjBridge : IDisposable, IMjPolicyPlant
{
    private IntPtr _model;
    private IntPtr _data;

    /// <summary>Bodies in the model, including the world body at index 0.</summary>
    internal int BodyCount { get; }

    /// <summary>Actuators in the model.</summary>
    internal int ActuatorCount { get; }

    /// <summary>Loads a model, or throws carrying MuJoCo's own compiler error.</summary>
    internal MjBridge(string xmlPath)
    {
        string? version = Marshal.PtrToStringAnsi(MjInterop.mj_versionString());
        if (IntPtr.Size != 8 || version != MjLayout.Version)
            throw new InvalidOperationException("MuJoCo ABI mismatch: loaded " + version
                + "; offsets require 64-bit MuJoCo " + MjLayout.Version + ".");
        try
        {
            var error = new byte[1024];
            _model = MjInterop.mj_loadXML(xmlPath, IntPtr.Zero, error, error.Length);
            if (_model == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "MuJoCo failed to load " + xmlPath + ": "
                    + System.Text.Encoding.ASCII.GetString(error).TrimEnd('\0'));
            }

            BodyCount = (int)Marshal.ReadInt64(_model, MjLayout.ModelNbody);
            ActuatorCount = (int)Marshal.ReadInt64(_model, MjLayout.ModelNu);
            long nq = Marshal.ReadInt64(_model, MjLayout.ModelNq);
            long nv = Marshal.ReadInt64(_model, MjLayout.ModelNv);

            // **The offsets are generated, so verify them rather than trust them.** Hand-computed offsets
            // on this struct already returned plausible wrong values once - nq=52, nv=0, nu=51 against the
            // true 52/51/36 - and a misread pointer would silently corrupt every body pose.
            if (BodyCount <= 0 || BodyCount > 4096 || ActuatorCount < 0 || nq <= 0 || nv <= 0)
            {
                throw new InvalidOperationException(
                    "MuJoCo struct offsets look wrong (nq=" + nq + " nv=" + nv + " nu=" + ActuatorCount
                    + " nbody=" + BodyCount + "). Regenerate them with mujoco_rig/gen_offsets.py against "
                    + "the installed MuJoCo; MjLayout was built for " + MjLayout.Version + ".");
            }

            _data = MjInterop.mj_makeData(_model);
            if (_data == IntPtr.Zero) throw new InvalidOperationException("MuJoCo could not allocate data.");
            MjInterop.mj_forward(_model, _data);
        }
        catch { Dispose(); throw; }
    }

    /// <summary>Advances the simulation by one MuJoCo timestep.</summary>
    internal void Step() { EnsureAlive(); MjInterop.mj_step(_model, _data); }

    /// <summary>Index of a named body, or -1.</summary>
    public int BodyId(string name) { EnsureAlive(); return MjInterop.mj_name2id(_model, MjInterop.ObjBody, name); }

    /// <summary>Index of a named actuator, or -1.</summary>
    public int ActuatorId(string name) { EnsureAlive(); return MjInterop.mj_name2id(_model, MjInterop.ObjActuator, name); }

    /// <summary>Writes one actuator's position target.</summary>
    public void SetControl(int actuator, double value)
    {
        EnsureAlive();
        if (actuator < 0 || actuator >= ActuatorCount)
            throw new ArgumentOutOfRangeException(nameof(actuator));

        IntPtr ctrl = Marshal.ReadIntPtr(_data, MjLayout.DataCtrl);
        Marshal.WriteInt64(ctrl, actuator * sizeof(double), BitConverter.DoubleToInt64Bits(value));
    }

    /// <summary>Clears every actuator target.</summary>
    internal void ClearControls()
    {
        EnsureAlive();
        IntPtr ctrl = Marshal.ReadIntPtr(_data, MjLayout.DataCtrl);
        for (int i = 0; i < ActuatorCount; i++)
        {
            Marshal.WriteInt64(ctrl, i * sizeof(double), 0L);
        }
    }

    /// <summary>
    /// A body's pose, converted from MuJoCo's Z-up frame into Godot's Y-up frame.
    /// </summary>
    /// <remarks>
    /// The basis change is the one the rest of this project already uses, <c>mj = (-gz, -gx, gy)</c>,
    /// so a rotation maps as <c>R_godot = M^T R_mj M</c> and a position inverts directly to
    /// <c>godot = (-my, mz, -mx)</c>. Getting this wrong is silent: the body renders in a plausible
    /// pose that drifts from the simulation.
    /// </remarks>
    public Transform3D BodyTransform(int body)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr xpos = Marshal.ReadIntPtr(_data, MjLayout.DataXpos);
        IntPtr xquat = Marshal.ReadIntPtr(_data, MjLayout.DataXquat);

        double mx = ReadDouble(xpos, (body * 3) + 0);
        double my = ReadDouble(xpos, (body * 3) + 1);
        double mz = ReadDouble(xpos, (body * 3) + 2);

        // MuJoCo stores quaternions as w, x, y, z.
        double qw = ReadDouble(xquat, (body * 4) + 0);
        double qx = ReadDouble(xquat, (body * 4) + 1);
        double qy = ReadDouble(xquat, (body * 4) + 2);
        double qz = ReadDouble(xquat, (body * 4) + 3);

        var mjBasis = new Basis(new Quaternion((float)qx, (float)qy, (float)qz, (float)qw));
        var origin = new Vector3((float)-my, (float)mz, (float)-mx);
        return new Transform3D(MjToGodot(mjBasis), origin);
    }

    /// <summary>
    /// A body's angular velocity in Godot's frame, from <c>mjData.cvel</c>.
    /// </summary>
    /// <remarks>
    /// MuJoCo's <c>cvel</c> is a 6D spatial velocity per body with the ANGULAR part first, then the
    /// linear part. Reading the halves the wrong way round yields a plausible vector, so the order is
    /// stated here rather than inferred at the call site.
    /// </remarks>
    public Vector3 BodyAngularVelocity(int body)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr cvel = Marshal.ReadIntPtr(_data, MjLayout.DataCvel);
        double wx = ReadDouble(cvel, (body * 6) + 0);
        double wy = ReadDouble(cvel, (body * 6) + 1);
        double wz = ReadDouble(cvel, (body * 6) + 2);
        return MjToGodot(new Vector3((float)wx, (float)wy, (float)wz));
    }

    /// <summary>One body's mass, from <c>mjModel.body_mass</c>.</summary>
    internal double BodyMass(int body)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr mass = Marshal.ReadIntPtr(_model, MjLayout.ModelBodyMass);
        return ReadDouble(mass, body);
    }

    /// <summary>A body's linear velocity in Godot's frame, from <c>mjData.cvel</c>.</summary>
    /// <remarks>The ANGULAR part comes first in <c>cvel</c>, so the linear part starts at index 3.</remarks>
    public Vector3 BodySpatialLinearVelocity(int body)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr cvel = Marshal.ReadIntPtr(_data, MjLayout.DataCvel);
        return MjToGodot(new Vector3(
            (float)ReadDouble(cvel, (body * 6) + 3),
            (float)ReadDouble(cvel, (body * 6) + 4),
            (float)ReadDouble(cvel, (body * 6) + 5)));
    }

    /// <summary>
    /// Applies an external torque to a body, in Godot's frame.
    /// </summary>
    /// <remarks>
    /// <c>mjData.xfrc_applied</c> is 6 per body with the FORCE first and the torque second, and it
    /// persists across steps, so it is overwritten every step by the caller rather than accumulated.
    /// </remarks>
    internal void SetBodyTorque(int body, Vector3 torqueGodot)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr xfrc = Marshal.ReadIntPtr(_data, MjLayout.DataXfrcApplied);
        Vector3 t = GodotToMj(torqueGodot);
        Marshal.WriteInt64(xfrc, ((body * 6) + 3) * sizeof(double), BitConverter.DoubleToInt64Bits(t.X));
        Marshal.WriteInt64(xfrc, ((body * 6) + 4) * sizeof(double), BitConverter.DoubleToInt64Bits(t.Y));
        Marshal.WriteInt64(xfrc, ((body * 6) + 5) * sizeof(double), BitConverter.DoubleToInt64Bits(t.Z));
    }

    /// <summary>
    /// Returns the whole simulation to the model's authored rest pose.
    /// </summary>
    /// <remarks>
    /// <para><c>mj_resetData</c> restores <c>qpos</c> to <c>qpos0</c> and zeroes every velocity,
    /// accumulated force and contact. The <c>mj_forward</c> afterwards is not optional: without it
    /// the derived arrays this bridge reads - <c>xpos</c>, <c>xmat</c>, <c>cvel</c> - still hold the
    /// pose from before the reset, so the first frame rendered after a reset would be the old one.
    /// </para>
    /// <para>The <c>rest</c> keyframe is used when the model has one, because <c>qpos0</c> is NOT a
    /// valid standing pose: the arms hang inside the legs there, 5.5 cm of forearm inside the
    /// thigh, which costs 331 N.m per shoulder to hold and starts the body in contact with itself.
    /// The training environments reset to the same keyframe, so Godot and the trainer agree on
    /// where an episode begins.</para>
    /// </remarks>
    internal void ResetData()
    {
        EnsureAlive();
        int key = MjInterop.mj_name2id(_model, MjInterop.ObjKey, "rest");
        if (key >= 0)
        {
            MjInterop.mj_resetDataKeyframe(_model, _data, key);
        }
        else
        {
            MjInterop.mj_resetData(_model, _data);
        }

        MjInterop.mj_forward(_model, _data);
    }

    /// <summary>Index of a named joint, or -1.</summary>
    public int JointId(string name) { EnsureAlive(); return MjInterop.mj_name2id(_model, MjInterop.ObjJoint, name); }

    /// <summary>
    /// A hinge joint's angle in radians, straight from <c>mjData.qpos</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>No frame conversion, deliberately.</b> A hinge angle is a scalar about an axis that
    /// <c>build_mjcf.py</c> already mapped into MuJoCo's frame, so the number here is exactly what a
    /// policy trained on this model saw. Rotating it "into Godot's frame" would be meaningless and
    /// would silently corrupt every observation.</para>
    /// <para><c>qpos</c> is indexed by <c>jnt_qposadr</c>, not by joint id: the free joint ahead of
    /// these occupies seven slots, so the two indices differ by six from the root onwards.</para>
    /// </remarks>
    public double JointPosition(int joint)
    {
        EnsureAlive();
        IntPtr adr = Marshal.ReadIntPtr(_model, MjLayout.ModelJntQposadr);
        int slot = Marshal.ReadInt32(adr, joint * sizeof(int));
        IntPtr qpos = Marshal.ReadIntPtr(_data, MjLayout.DataQpos);
        return ReadDouble(qpos, slot);
    }

    /// <summary>A hinge joint's angular rate in rad/s, from <c>mjData.qvel</c>.</summary>
    /// <remarks>
    /// Indexed by <c>jnt_dofadr</c>, which differs from <c>jnt_qposadr</c> because a free joint takes
    /// seven qpos slots but only six DOFs. Using one address for both is a silent one-slot shift.
    /// </remarks>
    public double JointVelocity(int joint)
    {
        EnsureAlive();
        IntPtr adr = Marshal.ReadIntPtr(_model, MjLayout.ModelJntDofadr);
        int slot = Marshal.ReadInt32(adr, joint * sizeof(int));
        IntPtr qvel = Marshal.ReadIntPtr(_data, MjLayout.DataQvel);
        return ReadDouble(qvel, slot);
    }

    // NOTE: joint LIMITS are deliberately not read from the model here. The policy's action is a
    // fraction of a joint's span, and the runtime must use exactly the span the checkpoint trained
    // with - so those come from the generated <c>*.contract.json</c> beside the ONNX file, written
    // by export_onnx.py from the training environment itself. A second copy read from the model
    // would agree until the day the model is regenerated and the checkpoint is not.

    /// <summary>
    /// Places a free joint's body: position, identity orientation, and linear velocity.
    /// </summary>
    /// <remarks>
    /// <c>jnt_qposadr</c> and <c>jnt_dofadr</c> are read from the model rather than assumed. Treating
    /// the projectile as "the last joint" happens to be true today and would break silently the first
    /// time the rig gains a body.
    /// </remarks>
    internal void SetFreeJoint(int joint, Vector3 positionGodot, Vector3 velocityGodot)
    {
        EnsureAlive();
        if (joint < 0)
        {
            return;
        }

        IntPtr qadrPtr = Marshal.ReadIntPtr(_model, MjLayout.ModelJntQposadr);
        IntPtr vadrPtr = Marshal.ReadIntPtr(_model, MjLayout.ModelJntDofadr);
        int qadr = Marshal.ReadInt32(qadrPtr, joint * sizeof(int));
        int vadr = Marshal.ReadInt32(vadrPtr, joint * sizeof(int));

        IntPtr qpos = Marshal.ReadIntPtr(_data, MjLayout.DataQpos);
        IntPtr qvel = Marshal.ReadIntPtr(_data, MjLayout.DataQvel);

        Vector3 p = GodotToMj(positionGodot);
        Vector3 v = GodotToMj(velocityGodot);
        WriteDouble(qpos, qadr + 0, p.X);
        WriteDouble(qpos, qadr + 1, p.Y);
        WriteDouble(qpos, qadr + 2, p.Z);
        WriteDouble(qpos, qadr + 3, 1.0);
        WriteDouble(qpos, qadr + 4, 0.0);
        WriteDouble(qpos, qadr + 5, 0.0);
        WriteDouble(qpos, qadr + 6, 0.0);
        for (int i = 0; i < 6; i++)
        {
            WriteDouble(qvel, vadr + i, i < 3 ? (i == 0 ? v.X : i == 1 ? v.Y : v.Z) : 0.0);
        }
    }

    /// <summary>A free joint's linear velocity, in Godot's frame.</summary>
    internal Vector3 FreeJointVelocity(int joint)
    {
        EnsureAlive();
        EnsureAlive();
        IntPtr vadrPtr = Marshal.ReadIntPtr(_model, MjLayout.ModelJntDofadr);
        int vadr = Marshal.ReadInt32(vadrPtr, joint * sizeof(int));
        IntPtr qvel = Marshal.ReadIntPtr(_data, MjLayout.DataQvel);
        return MjToGodot(new Vector3(
            (float)ReadDouble(qvel, vadr + 0),
            (float)ReadDouble(qvel, vadr + 1),
            (float)ReadDouble(qvel, vadr + 2)));
    }

    private static void WriteDouble(IntPtr array, int index, double value) =>
        Marshal.WriteInt64(array, index * sizeof(double), BitConverter.DoubleToInt64Bits(value));

    /// <summary>
    /// Sum of <c>mjModel.body_mass</c> over the BODY, excluding the projectile: 80.6 kg for this rig.
    /// </summary>
    /// <remarks>
    /// Counting the 3 kg ball made the startup diagnostic read 83.6 kg, which does not match the rig
    /// the rest of the project quotes and would quietly undermine the mass check it exists to be.
    /// </remarks>
    internal double TotalMass()
    {
        EnsureAlive();
        IntPtr mass = Marshal.ReadIntPtr(_model, MjLayout.ModelBodyMass);
        double total = 0.0;
        for (int b = 0; b < BodyCount; b++)
        {
            if (b == ExcludeFromCom)
            {
                continue;
            }

            total += ReadDouble(mass, b);
        }

        return total;
    }

    /// <summary>Body excluded from centre-of-mass sums, e.g. the projectile. -1 for none.</summary>
    internal int ExcludeFromCom { get; set; } = -1;

    /// <summary>
    /// Whole-body centre of mass, in Godot's frame.
    /// </summary>
    /// <remarks>
    /// Weighted over <c>mjData.xipos</c> (each body's CoM) by <c>mjModel.body_mass</c>. The projectile
    /// is excluded via <see cref="ExcludeFromCom"/>: it is a 3 kg free body, and including it drags
    /// the centre of mass the balance term regulates - while parked it moved the reported COM to
    /// z = 53.
    /// </remarks>
    internal Vector3 CenterOfMass()
    {
        EnsureAlive();
        IntPtr xipos = Marshal.ReadIntPtr(_data, MjLayout.DataXipos);
        IntPtr mass = Marshal.ReadIntPtr(_model, MjLayout.ModelBodyMass);
        double total = 0.0, x = 0.0, y = 0.0, z = 0.0;
        for (int b = 0; b < BodyCount; b++)
        {
            if (b == ExcludeFromCom)
            {
                continue;
            }

            double mb = ReadDouble(mass, b);
            total += mb;
            x += mb * ReadDouble(xipos, (b * 3) + 0);
            y += mb * ReadDouble(xipos, (b * 3) + 1);
            z += mb * ReadDouble(xipos, (b * 3) + 2);
        }

        if (total <= 0.0)
        {
            return Vector3.Zero;
        }

        return MjToGodot(new Vector3((float)(x / total), (float)(y / total), (float)(z / total)));
    }

    /// <summary>Whole-body centre-of-mass velocity, in Godot's frame.</summary>
    /// <remarks>
    /// Shifts each spatial velocity from the subtree COM to its body's COM before weighting it.
    /// Raw cvel values are retained only for the policy observation, matching its training contract.
    /// </remarks>
    internal Vector3 CenterOfMassVelocity()
    {
        EnsureAlive();
        IntPtr mass = Marshal.ReadIntPtr(_model, MjLayout.ModelBodyMass);
        double total = 0.0, x = 0.0, y = 0.0, z = 0.0;
        for (int b = 0; b < BodyCount; b++)
        {
            if (b == ExcludeFromCom)
            {
                continue;
            }

            double mb = ReadDouble(mass, b);
            total += mb;
            Vector3 velocity = LinearVelocityAt(b, MjLayout.DataXipos);
            x += mb * velocity.X;
            y += mb * velocity.Y;
            z += mb * velocity.Z;
        }

        if (total <= 0.0)
        {
            return Vector3.Zero;
        }

        return new Vector3((float)(x / total), (float)(y / total), (float)(z / total));
    }

    /// <summary>Velocity at a body's origin, rather than cvel's subtree COM reference point.</summary>
    internal Vector3 BodyOriginVelocity(int body) => LinearVelocityAt(body, MjLayout.DataXpos);

    private Vector3 LinearVelocityAt(int body, int positionOffset)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr roots = Marshal.ReadIntPtr(_model, MjLayout.ModelBodyRootid);
        int root = Marshal.ReadInt32(roots, body * sizeof(int));
        IntPtr com = Marshal.ReadIntPtr(_data, MjLayout.DataSubtreeCom);
        IntPtr position = Marshal.ReadIntPtr(_data, positionOffset);
        var arm = new Vector3(
            (float)(ReadDouble(position, body * 3) - ReadDouble(com, root * 3)),
            (float)(ReadDouble(position, body * 3 + 1) - ReadDouble(com, root * 3 + 1)),
            (float)(ReadDouble(position, body * 3 + 2) - ReadDouble(com, root * 3 + 2)));
        return BodySpatialLinearVelocity(body) + BodyAngularVelocity(body).Cross(MjToGodot(arm));
    }

    /// <summary>Applies an external force to a body, in Godot's frame. Used for perturbation tests.</summary>
    internal void SetBodyForce(int body, Vector3 forceGodot)
    {
        EnsureAlive();
        if (body < 0 || body >= BodyCount) throw new ArgumentOutOfRangeException(nameof(body));
        IntPtr xfrc = Marshal.ReadIntPtr(_data, MjLayout.DataXfrcApplied);
        Vector3 f = GodotToMj(forceGodot);
        Marshal.WriteInt64(xfrc, ((body * 6) + 0) * sizeof(double), BitConverter.DoubleToInt64Bits(f.X));
        Marshal.WriteInt64(xfrc, ((body * 6) + 1) * sizeof(double), BitConverter.DoubleToInt64Bits(f.Y));
        Marshal.WriteInt64(xfrc, ((body * 6) + 2) * sizeof(double), BitConverter.DoubleToInt64Bits(f.Z));
    }

    /// <summary>Godot vector into MuJoCo's frame: <c>mj = (-gz, -gx, gy)</c>.</summary>
    internal static Vector3 GodotToMj(Vector3 g) => new(-g.Z, -g.X, g.Y);

    /// <summary>MuJoCo vector into Godot's frame, the inverse of <see cref="GodotToMj"/>.</summary>
    private static Vector3 MjToGodot(Vector3 v) => new(-v.Y, v.Z, -v.X);

    /// <summary>Change of basis from MuJoCo's frame into Godot's.</summary>
    private static Basis MjToGodot(Basis mj)
    {
        // M maps godot -> mj, i.e. mj = M * godot. Columns of M are the images of Godot's axes.
        var m = new Basis(
            new Vector3(0.0f, -1.0f, 0.0f),
            new Vector3(0.0f, 0.0f, 1.0f),
            new Vector3(-1.0f, 0.0f, 0.0f));
        return m.Transposed() * mj * m;
    }

    private static double ReadDouble(IntPtr array, int index) =>
        BitConverter.Int64BitsToDouble(Marshal.ReadInt64(array, index * sizeof(double)));

    private void EnsureAlive() => ObjectDisposedException.ThrowIf(_model == IntPtr.Zero || _data == IntPtr.Zero, this);

    public void Dispose()
    {
        if (_data != IntPtr.Zero)
        {
            MjInterop.mj_deleteData(_data);
            _data = IntPtr.Zero;
        }

        if (_model != IntPtr.Zero)
        {
            MjInterop.mj_deleteModel(_model);
            _model = IntPtr.Zero;
        }
    }
}

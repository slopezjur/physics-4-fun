using System;
using System.Runtime.InteropServices;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Foot and toe normal loads against the floor, in newtons. Borrows native handles.</summary>
internal sealed class MjFootLoadSensor
{
    private readonly IntPtr _model, _data, _geomBodies;
    private readonly int _floor, _left, _leftToe, _right, _rightToe;
    private readonly double[] _force = new double[6];

    internal MjFootLoadSensor(IntPtr model, IntPtr data)
    {
        _model = model;
        _data = data;
        _geomBodies = Marshal.ReadIntPtr(model, MjLayout.ModelGeomBodyid);
        _floor = RequiredId(MjInterop.ObjGeom, "floor");
        _left = RequiredId(MjInterop.ObjBody, "Foot_L"); _leftToe = RequiredId(MjInterop.ObjBody, "Toe_L");
        _right = RequiredId(MjInterop.ObjBody, "Foot_R"); _rightToe = RequiredId(MjInterop.ObjBody, "Toe_R");
    }

    private int RequiredId(int type, string name)
    {
        int id = MjInterop.mj_name2id(_model, type, name);
        return id >= 0 ? id : throw new InvalidOperationException("Foot load sensor requires '" + name + "'.");
    }

    internal Vector2 Read()
    {
        double left = 0, right = 0;
        // Contacts live in the arena: reacquire the pointer after each physics step.
        IntPtr contacts = Marshal.ReadIntPtr(_data, MjLayout.DataContact);
        int count = Marshal.ReadInt32(_data, MjLayout.DataNcon);
        for (int i = 0; i < count; i++)
        {
            int offset = i * MjLayout.ContactStride + MjLayout.ContactGeom;
            int first = Marshal.ReadInt32(contacts, offset);
            int second = Marshal.ReadInt32(contacts, offset + sizeof(int));
            int other = first == _floor ? second : second == _floor ? first : -1;
            if (other < 0) continue;
            int body = Marshal.ReadInt32(_geomBodies, other * sizeof(int));
            bool isLeft = body == _left || body == _leftToe;
            if (!isLeft && body != _right && body != _rightToe) continue;
            MjInterop.mj_contactForce(_model, _data, i, _force);
            double normal = Math.Max(0, _force[0]);
            if (isLeft) left += normal; else right += normal;
        }
        return new Vector2((float)left, (float)right);
    }
}

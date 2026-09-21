using System;
using System.IO;
using System.Linq;
using System.Xml;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Character-only policy state inside an optional projectile world.</summary>
internal sealed class MjMimicWorld
{
    private readonly MjBridge _bridge;
    private readonly double[]? _fullQ, _fullV;
    internal string CharacterPath { get; }
    internal int CharacterBodyCount { get; }

    internal MjMimicWorld(MjBridge bridge, string modelPath)
    {
        _bridge = bridge;
        bool ball = bridge.BodyId("ball") >= 0;
        CharacterPath = ball ? Path.Combine(Path.GetDirectoryName(modelPath)!, "dummy.xml") : modelPath;
        CharacterBodyCount = bridge.BodyCount - (ball ? 1 : 0);
        if (!ball) return;
        var world = new XmlDocument(); world.Load(modelPath);
        var character = new XmlDocument(); character.Load(CharacterPath);
        var projectile = world.SelectSingleNode("/mujoco/worldbody/body[@name='ball']")
            ?? throw new InvalidOperationException("Missing projectile definition.");
        projectile.ParentNode!.RemoveChild(projectile);
        foreach (var doc in new[] { world, character })
        {
            doc.DocumentElement!.RemoveAttribute("model");
            var key = doc.SelectSingleNode("/mujoco/keyframe");
            key?.ParentNode!.RemoveChild(key);
            foreach (XmlNode comment in doc.SelectNodes("//comment()")!.Cast<XmlNode>().ToArray())
                comment.ParentNode!.RemoveChild(comment);
        }
        if (world.DocumentElement!.OuterXml != character.DocumentElement!.OuterXml
            || bridge.JointId("ball_free") != 40 || bridge.BodyId("ball") != 19)
            throw new InvalidOperationException("Ball world must append a projectile without changing the character.");
        _fullQ = new double[53]; _fullV = new double[51];
        bridge.ReadNativeState(_fullQ, _fullV); // Validate the complete native layout.
    }

    internal void Read(double[] q, double[] v)
    {
        if (_fullQ == null) { _bridge.ReadNativeState(q, v); return; }
        _bridge.ReadNativeState(_fullQ, _fullV!);
        Array.Copy(_fullQ, q, q.Length); Array.Copy(_fullV!, v, v.Length);
    }

    internal void Reset(double[] q, double[] v)
    {
        if (_fullQ == null) { _bridge.ResetNativeState(q, v); return; }
        Array.Clear(_fullQ); Array.Clear(_fullV!);
        Array.Copy(q, _fullQ, q.Length); Array.Copy(v, _fullV!, v.Length);
        _fullQ[46] = _fullQ[47] = 40; _fullQ[48] = 2; _fullQ[49] = 1;
        _bridge.ResetNativeState(_fullQ, _fullV!);
    }
}
